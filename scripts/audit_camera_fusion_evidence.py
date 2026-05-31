from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit camera vehicle fusion evidence CSV")
    parser.add_argument("--csv-path", required=True, help="Path to camera_evidence_events.csv")
    parser.add_argument("--summary-path", help="Optional path to sequence summary.json")
    parser.add_argument("--output-path", help="Optional JSON audit output path")
    parser.add_argument("--require-inside-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allowed-classes", nargs="+", default=["car"])
    parser.add_argument("--max-bbox-height-ratio", type=float, default=0.85)
    parser.add_argument("--max-bbox-area-ratio", type=float, default=0.45)
    parser.add_argument("--fail-on-multi-frame-evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fail-on-multi-direct-detection", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    rows = read_rows(csv_path)
    summary = read_summary(Path(args.summary_path)) if args.summary_path else {}
    audit = build_audit(rows, summary, args)

    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print(json.dumps(audit, indent=2))
    if audit["failures"]:
        raise SystemExit(1)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    return list(csv.DictReader(text.splitlines()))


def read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_audit(rows: list[dict[str, str]], summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    match_type_counts = Counter(row.get("match_type", "") for row in rows)
    class_counts = Counter(row.get("class_name", "") for row in rows)
    camera_counts = Counter(row.get("camera", "") for row in rows)
    held_counts = Counter(row.get("is_held_camera_evidence", "") for row in rows)
    multi_frame_evidence = find_multi_frame_evidence(rows)
    multi_direct_detections = find_multi_direct_detections(rows)
    bbox_stats = numeric_stats(
        rows,
        [
            "bbox_width_ratio",
            "bbox_height_ratio",
            "bbox_area_ratio",
            "bbox_bottom_y_ratio",
            "bbox_near_score",
        ],
    )

    if args.require_inside_only and set(match_type_counts) - {"inside"}:
        failures.append(f"non-inside match types found: {dict(match_type_counts)}")

    disallowed_classes = set(class_counts) - set(args.allowed_classes)
    if disallowed_classes:
        failures.append(f"disallowed classes found: {sorted(disallowed_classes)}")

    if args.fail_on_multi_frame_evidence and multi_frame_evidence:
        failures.append(f"frames with multiple evidence slots: {len(multi_frame_evidence)}")

    if args.fail_on_multi_direct_detection and multi_direct_detections:
        failures.append(f"direct camera detections assigned to multiple slots: {len(multi_direct_detections)}")

    max_height = bbox_stats.get("bbox_height_ratio", {}).get("max")
    if max_height is not None and max_height > args.max_bbox_height_ratio:
        failures.append(f"bbox_height_ratio max {max_height:.4f} > {args.max_bbox_height_ratio:.4f}")

    max_area = bbox_stats.get("bbox_area_ratio", {}).get("max")
    if max_area is not None and max_area > args.max_bbox_area_ratio:
        failures.append(f"bbox_area_ratio max {max_area:.4f} > {args.max_bbox_area_ratio:.4f}")

    summary_counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
    if summary_counts:
        expected_rows = int(summary_counts.get("slots_with_camera_evidence", -1))
        if expected_rows >= 0 and expected_rows != len(rows):
            failures.append(f"CSV rows {len(rows)} != summary slots_with_camera_evidence {expected_rows}")

    return {
        "csv_path": str(args.csv_path),
        "summary_path": args.summary_path,
        "rows": len(rows),
        "summary_counts": summary_counts,
        "match_type_counts": dict(sorted(match_type_counts.items())),
        "class_counts": dict(sorted(class_counts.items())),
        "camera_counts": dict(sorted(camera_counts.items())),
        "held_counts": dict(sorted(held_counts.items())),
        "multi_frame_evidence_count": len(multi_frame_evidence),
        "multi_direct_detection_count": len(multi_direct_detections),
        "multi_frame_evidence_examples": multi_frame_evidence[:10],
        "multi_direct_detection_examples": multi_direct_detections[:10],
        "bbox_stats": bbox_stats,
        "failures": failures,
    }


def find_multi_frame_evidence(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_frame[row.get("frame_idx", "")].append(row)
    examples = []
    for frame_idx, frame_rows in sorted(by_frame.items(), key=lambda item: int(item[0]) if item[0].isdigit() else -1):
        if len(frame_rows) <= 1:
            continue
        examples.append(
            {
                "frame_idx": frame_idx,
                "evidence": [
                    {
                        "slot_id": row.get("slot_id"),
                        "camera": row.get("camera"),
                        "detection_idx": row.get("detection_idx"),
                        "is_held_camera_evidence": row.get("is_held_camera_evidence"),
                    }
                    for row in frame_rows
                ],
            }
        )
    return examples


def find_multi_direct_detections(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_detection: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("is_held_camera_evidence") != "False":
            continue
        key = (row.get("frame_idx", ""), row.get("camera", ""), row.get("detection_idx", ""))
        by_detection[key].append(row)
    examples = []
    for (frame_idx, camera, detection_idx), detection_rows in sorted(
        by_detection.items(),
        key=lambda item: int(item[0][0]) if item[0][0].isdigit() else -1,
    ):
        if len(detection_rows) <= 1:
            continue
        examples.append(
            {
                "frame_idx": frame_idx,
                "camera": camera,
                "detection_idx": detection_idx,
                "slot_ids": [row.get("slot_id") for row in detection_rows],
            }
        )
    return examples


def numeric_stats(rows: list[dict[str, str]], fields: list[str]) -> dict[str, dict[str, float | int | None]]:
    stats = {}
    for field in fields:
        values = []
        for row in rows:
            raw = row.get(field, "")
            if raw in {"", None}:
                continue
            values.append(float(raw))
        stats[field] = {
            "count": len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
        }
    return stats


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
