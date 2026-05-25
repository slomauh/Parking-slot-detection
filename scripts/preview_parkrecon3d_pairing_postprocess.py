from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sweep_parkrecon3d_pairing_postprocess import (  # noqa: E402
    IMAGE_EXTENSIONS,
    SweepVariant,
    crpsd_slots_to_polygons,
    inference_slots_relaxed,
    load_crpsd_modules,
    match_polygons,
    postprocess_inferred_slots,
    raw_slots_to_polygons,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render ParkRecon3D prepared-pairing previews with relaxed postprocess")
    parser.add_argument("--dataset-dir", default="outputs/parkrecon3d_bev_crpsd_format")
    parser.add_argument("--external-repo-path", default="external/CRPS-D")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_pairing_postprocess_preview")
    parser.add_argument("--split", default="test")
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--angle-scale", type=float, default=1.0)
    parser.add_argument("--suppress-dot", type=float, default=1.01)
    parser.add_argument("--allow-any-pairing", action="store_true")
    parser.add_argument("--smart-suppress-distance", type=float, default=-1.0)
    parser.add_argument("--cross-prune-overlap", type=float, default=0.0)
    parser.add_argument("--cross-prune-min-crossings", type=int, default=2)
    parser.add_argument("--max-point-degree", type=int, default=0)
    parser.add_argument("--contact-sheet-limit", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = load_crpsd_modules(Path(args.external_repo_path))
    variant = SweepVariant(
        name="preview",
        distance_scale=args.distance_scale,
        angle_scale=args.angle_scale,
        suppress_dot=args.suppress_dot,
        allow_any_pairing=args.allow_any_pairing,
        smart_suppress_distance=args.smart_suppress_distance,
        cross_prune_overlap=args.cross_prune_overlap,
        cross_prune_min_crossings=args.cross_prune_min_crossings,
        max_point_degree=args.max_point_degree,
    )
    output_dir = Path(args.output_dir)
    preview_dir = output_dir / args.split / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    records, counts = render_split(Path(args.dataset_dir), args.split, preview_dir, modules, variant, args)
    summary = build_summary(args, variant, counts, records, preview_dir)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    make_contact_sheet(preview_dir, output_dir / f"{args.split}_contact_sheet.jpg", args.contact_sheet_limit)
    print(json.dumps(summary, indent=2))


def render_split(
    dataset_dir: Path,
    split: str,
    preview_dir: Path,
    modules: dict[str, Any],
    variant: SweepVariant,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    image_dir = dataset_dir / "raw" / split / "img"
    raw_label_dir = dataset_dir / "raw" / split / "slot_label"
    prepared_dir = dataset_dir / "prepared" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    if not raw_label_dir.exists():
        raise FileNotFoundError(f"Raw label dir not found: {raw_label_dir}")
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Prepared label dir not found: {prepared_dir}")

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    records = []
    counts = Counter()
    for image_path in tqdm(image_paths, desc=f"Rendering {split} previews"):
        raw_label_path = raw_label_dir / f"{image_path.stem}.json"
        prepared_label_path = prepared_dir / f"{image_path.stem}.json"
        if not raw_label_path.exists() or not prepared_label_path.exists():
            counts["missing_labels"] += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            counts["unreadable_images"] += 1
            continue

        raw_label = json.loads(raw_label_path.read_text(encoding="utf-8"))
        prepared_label = json.loads(prepared_label_path.read_text(encoding="utf-8"))
        gt_slots = raw_slots_to_polygons(raw_label)
        marking_points = [modules["MarkingPoint"](*mark) for mark in prepared_label]
        inferred_raw_slots = inference_slots_relaxed(marking_points, modules["config"], variant)
        inferred_raw_slots, inferred_slots = postprocess_inferred_slots(
            image, marking_points, inferred_raw_slots, modules["config"], variant
        )
        matches = match_polygons(gt_slots, inferred_slots, args.match_iou)
        matched_gt_ids = {match["gt_idx"] for match in matches}
        matched_pred_ids = {match["pred_idx"] for match in matches}
        false_negatives = len(gt_slots) - len(matched_gt_ids)
        false_positives = len(inferred_slots) - len(matched_pred_ids)

        counts["images"] += 1
        counts["gt_slots"] += len(gt_slots)
        counts["inferred_slots"] += len(inferred_slots)
        counts["matched_slots"] += len(matches)
        counts["false_negative_slots"] += false_negatives
        counts["false_positive_slots"] += false_positives
        if false_negatives or false_positives:
            counts["images_with_errors"] += 1

        preview = draw_preview(image, image_path.stem, gt_slots, inferred_slots, matched_gt_ids, matched_pred_ids)
        cv2.imwrite(str(preview_dir / image_path.name), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        records.append(
            {
                "image": str(image_path),
                "preview": str(preview_dir / image_path.name),
                "gt_slots": len(gt_slots),
                "inferred_slots": len(inferred_slots),
                "matched_slots": len(matches),
                "false_negative_slots": false_negatives,
                "false_positive_slots": false_positives,
                "matches": matches,
            }
        )
    return records, counts


def build_summary(
    args: argparse.Namespace,
    variant: SweepVariant,
    counts: Counter[str],
    records: list[dict[str, Any]],
    preview_dir: Path,
) -> dict[str, Any]:
    matched = counts["matched_slots"]
    gt_slots = counts["gt_slots"]
    inferred_slots = counts["inferred_slots"]
    precision = matched / max(1, inferred_slots)
    recall = matched / max(1, gt_slots)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    worst = sorted(
        records,
        key=lambda row: (row["false_negative_slots"] + row["false_positive_slots"], row["false_negative_slots"]),
        reverse=True,
    )[:20]
    return {
        "dataset_dir": args.dataset_dir,
        "split": args.split,
        "preview_dir": str(preview_dir),
        "variant": {
            "distance_scale": variant.distance_scale,
            "angle_scale": variant.angle_scale,
            "suppress_dot": variant.suppress_dot,
            "allow_any_pairing": variant.allow_any_pairing,
            "smart_suppress_distance": variant.smart_suppress_distance,
            "cross_prune_overlap": variant.cross_prune_overlap,
            "cross_prune_min_crossings": variant.cross_prune_min_crossings,
            "max_point_degree": variant.max_point_degree,
        },
        "counts": dict(counts),
        "metrics": {"recall": recall, "precision": precision, "f1": f1},
        "worst_examples": worst,
    }


def draw_preview(
    image: np.ndarray,
    stem: str,
    gt_slots,
    inferred_slots,
    matched_gt_ids: set[int],
    matched_pred_ids: set[int],
) -> np.ndarray:
    output = image.copy()
    for idx, points in enumerate(gt_slots):
        color = (0, 180, 0) if idx in matched_gt_ids else (0, 0, 255)
        draw_polygon(output, points, color, f"GT{idx + 1}")
    for idx, points in enumerate(inferred_slots):
        color = (255, 120, 0) if idx in matched_pred_ids else (0, 220, 255)
        draw_polygon(output, points, color, f"P{idx + 1}", y_offset=15)
    label = (
        f"{stem} | green GT matched | red GT missed | blue pred matched | yellow FP | "
        f"GT={len(gt_slots)} P={len(inferred_slots)} M={len(matched_gt_ids)}"
    )
    cv2.rectangle(output, (0, 0), (output.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(output, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def draw_polygon(image, points, color, label, y_offset: int = 0) -> None:
    polygon = np.array(points, dtype=np.int32)
    cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2)
    x, y = polygon[0]
    cv2.putText(
        image,
        label,
        (int(x), max(16, int(y) - 5 + y_offset)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


def make_contact_sheet(input_dir: Path, output_path: Path, limit: int) -> None:
    image_paths = sorted(input_dir.glob("*.jpg"))[:limit]
    if not image_paths:
        return
    columns = 5
    tile_size = (320, 320)
    tiles = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        tile = cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            image_path.stem,
            (6, tile.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        return
    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank.copy())
    rows = [cv2.hconcat(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    cv2.imwrite(str(output_path), cv2.vconcat(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 95])


if __name__ == "__main__":
    main()
