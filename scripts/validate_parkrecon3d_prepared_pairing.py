from __future__ import annotations

import argparse
import json
import sys
import types
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate whether CRPS-D postprocessing can reconstruct raw slots from prepared GT marks"
    )
    parser.add_argument("--dataset-dir", default="outputs/parkrecon3d_bev_crpsd_format")
    parser.add_argument("--external-repo-path", default="external/CRPS-D")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_prepared_pairing_validation")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preview-limit", type=int, default=120)
    parser.add_argument("--preview-errors-only", action="store_true")
    parser.add_argument("--contact-sheet-limit", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    crpsd_modules = load_crpsd_modules(Path(args.external_repo_path))
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "external_repo_path": args.external_repo_path,
        "match_iou": args.match_iou,
        "splits": {},
    }
    all_records = {}
    for split in args.splits:
        split_summary, records = validate_split(dataset_dir, output_dir, split, crpsd_modules, args)
        summary["splits"][split] = split_summary
        all_records[split] = records

    summary["total"] = combine_split_summaries(summary["splits"])
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "records.json").write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def validate_split(
    dataset_dir: Path,
    output_dir: Path,
    split: str,
    crpsd_modules: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_dir = dataset_dir / "raw" / split / "img"
    raw_label_dir = dataset_dir / "raw" / split / "slot_label"
    prepared_dir = dataset_dir / "prepared" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    if not raw_label_dir.exists():
        raise FileNotFoundError(f"Raw label dir not found: {raw_label_dir}")
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Prepared label dir not found: {prepared_dir}")

    preview_dir = output_dir / split / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    counts: Counter[str] = Counter()
    records = []
    preview_written = 0

    for image_path in tqdm(image_paths, desc=f"Validating {split} prepared pairing"):
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
        marking_points = [crpsd_modules["MarkingPoint"](*mark) for mark in prepared_label]
        inferred_raw_slots = crpsd_modules["inference_slots"](marking_points) if marking_points else []
        inferred_slots = crpsd_slots_to_polygons(image, marking_points, inferred_raw_slots, crpsd_modules["config"])
        matches = match_polygons(gt_slots, inferred_slots, args.match_iou)

        matched_gt_ids = {match["gt_idx"] for match in matches}
        matched_pred_ids = {match["pred_idx"] for match in matches}
        false_negatives = len(gt_slots) - len(matched_gt_ids)
        false_positives = len(inferred_slots) - len(matched_pred_ids)

        counts["images"] += 1
        counts["gt_slots"] += len(gt_slots)
        counts["prepared_marks"] += len(prepared_label)
        counts["inferred_slots"] += len(inferred_slots)
        counts["matched_slots"] += len(matches)
        counts["false_negative_slots"] += false_negatives
        counts["false_positive_slots"] += false_positives
        if false_negatives or false_positives:
            counts["images_with_pairing_errors"] += 1

        record = {
            "image": str(image_path),
            "raw_label": str(raw_label_path),
            "prepared_label": str(prepared_label_path),
            "gt_slots": len(gt_slots),
            "prepared_marks": len(prepared_label),
            "inferred_slots": len(inferred_slots),
            "matched_slots": len(matches),
            "false_negative_slots": false_negatives,
            "false_positive_slots": false_positives,
            "matches": matches,
        }
        records.append(record)

        should_preview = preview_written < args.preview_limit
        if args.preview_errors_only:
            should_preview = should_preview and (false_negatives > 0 or false_positives > 0)
        if should_preview:
            preview = draw_validation_preview(image, image_path.stem, gt_slots, inferred_slots, matched_gt_ids, matched_pred_ids)
            cv2.imwrite(str(preview_dir / image_path.name), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            preview_written += 1

    make_contact_sheet(preview_dir, output_dir / split / "contact_sheet.jpg", args.contact_sheet_limit)
    return build_split_summary(counts, preview_dir), records


def load_crpsd_modules(external_repo_path: Path) -> dict[str, Any]:
    external_repo_path = external_repo_path.resolve()
    if not external_repo_path.exists():
        raise FileNotFoundError(f"CRPS-D repo not found: {external_repo_path}")
    if str(external_repo_path) not in sys.path:
        sys.path.insert(0, str(external_repo_path))

    install_visdom_stub()
    import config as crpsd_config
    from data.struct import MarkingPoint
    from inference import inference_slots

    return {
        "config": crpsd_config,
        "MarkingPoint": MarkingPoint,
        "inference_slots": inference_slots,
    }


def install_visdom_stub() -> None:
    if "visdom" in sys.modules:
        return

    visdom = types.ModuleType("visdom")

    class Visdom:
        def __init__(self, *args, **kwargs) -> None:
            pass

    visdom.Visdom = Visdom
    sys.modules["visdom"] = visdom


def raw_slots_to_polygons(raw_label: dict[str, Any]) -> list[list[tuple[float, float]]]:
    marks = raw_label.get("marks", [])
    slots = raw_label.get("slots", [])
    polygons = []
    for raw_slot in slots:
        if not isinstance(raw_slot, list) or len(raw_slot) < 2:
            continue
        try:
            mark_a_idx = int(raw_slot[0])
            mark_b_idx = int(raw_slot[1])
        except (TypeError, ValueError):
            continue
        if mark_a_idx < 1 or mark_b_idx < 1 or mark_a_idx > len(marks) or mark_b_idx > len(marks):
            continue
        mark_a = marks[mark_a_idx - 1]
        mark_b = marks[mark_b_idx - 1]
        if not isinstance(mark_a, list) or not isinstance(mark_b, list) or len(mark_a) < 4 or len(mark_b) < 4:
            continue
        polygons.append(
            [
                (float(mark_a[0]), float(mark_a[1])),
                (float(mark_b[0]), float(mark_b[1])),
                (float(mark_b[2]), float(mark_b[3])),
                (float(mark_a[2]), float(mark_a[3])),
            ]
        )
    return polygons


def crpsd_slots_to_polygons(image, marking_points, raw_slots, crpsd_config) -> list[list[tuple[float, float]]]:
    image_size = max(image.shape[:2])
    polygons = []
    for raw_slot in raw_slots:
        point_a = marking_points[raw_slot[0]]
        point_b = marking_points[raw_slot[1]]
        p0_x = image_size * point_a.x - 0.5
        p0_y = image_size * point_a.y - 0.5
        p1_x = image_size * point_b.x - 0.5
        p1_y = image_size * point_b.y - 0.5

        if point_a.type < 0.5:
            distance = (point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2
            if distance <= crpsd_config.VSLOT_MAX_DIST * crpsd_config.SQUARED_RATIO:
                separating_length = crpsd_config.LONG_SEPARATOR_LENGTH * crpsd_config.RATIO
            else:
                separating_length = crpsd_config.SHORT_SEPARATOR_LENGTH * crpsd_config.RATIO
        else:
            separating_length = crpsd_config.SLANT_SEPARATOR_LENGTH * crpsd_config.RATIO

        cos_val = np.cos(raw_slot[2])
        sin_val = np.sin(raw_slot[2])
        p2_x = p0_x + image_size * separating_length * cos_val
        p2_y = p0_y + image_size * separating_length * sin_val
        p3_x = p1_x + image_size * separating_length * cos_val
        p3_y = p1_y + image_size * separating_length * sin_val
        polygons.append([(p0_x, p0_y), (p1_x, p1_y), (p3_x, p3_y), (p2_x, p2_y)])
    return polygons


def match_polygons(gt_slots, pred_slots, min_iou: float) -> list[dict[str, Any]]:
    candidates = []
    for gt_idx, gt_points in enumerate(gt_slots):
        gt_polygon = make_polygon(gt_points)
        if gt_polygon is None:
            continue
        for pred_idx, pred_points in enumerate(pred_slots):
            pred_polygon = make_polygon(pred_points)
            if pred_polygon is None:
                continue
            iou = polygon_iou(gt_polygon, pred_polygon)
            if iou >= min_iou:
                candidates.append((iou, gt_idx, pred_idx))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_gt = set()
    used_pred = set()
    matches = []
    for iou, gt_idx, pred_idx in candidates:
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append({"gt_idx": gt_idx, "pred_idx": pred_idx, "iou": iou})
    return matches


def make_polygon(points) -> ShapelyPolygon | None:
    polygon = ShapelyPolygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def polygon_iou(polygon_a: ShapelyPolygon, polygon_b: ShapelyPolygon) -> float:
    intersection = polygon_a.intersection(polygon_b).area
    union = polygon_a.union(polygon_b).area
    return float(intersection / union) if union > 0 else 0.0


def build_split_summary(counts: Counter[str], preview_dir: Path) -> dict[str, Any]:
    matched = counts["matched_slots"]
    gt_slots = counts["gt_slots"]
    inferred_slots = counts["inferred_slots"]
    images = counts["images"]
    return {
        "counts": dict(counts),
        "metrics": {
            "pairing_recall": matched / max(1, gt_slots),
            "pairing_precision": matched / max(1, inferred_slots),
            "images_with_pairing_errors_ratio": counts["images_with_pairing_errors"] / max(1, images),
        },
        "preview_dir": str(preview_dir),
    }


def combine_split_summaries(split_summaries: dict[str, Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for split_summary in split_summaries.values():
        counts.update(split_summary["counts"])
    return build_split_summary(counts, Path(""))


def draw_validation_preview(
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
        draw_polygon(output, points, color, f"INF{idx + 1}", y_offset=14)
    label = (
        f"{stem} | raw GT={len(gt_slots)} | inferred from prepared={len(inferred_slots)} | "
        f"matched={len(matched_gt_ids)}"
    )
    cv2.rectangle(output, (0, 0), (output.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(output, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
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
    tile_size = (256, 256)
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
