from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import types
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ParkRecon3D BEV to CRPS-D-like training/eval datasets")
    parser.add_argument(
        "--dataset-root",
        action="append",
        dest="dataset_root",
        help="ParkRecon3D part root. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        help="ParkRecon3D part roots. Alternative to repeated --dataset-root.",
    )
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_bev_crpsd_format")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--gap-size", type=int, default=30, help="Frames dropped between train and test for chronological split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--prepared-strategy", choices=("simple", "optimized"), default="simple")
    parser.add_argument("--external-repo-path", default="external/CRPS-D")
    parser.add_argument("--pairing-match-iou", type=float, default=0.10)
    parser.add_argument("--optimize-passes", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_roots = resolve_dataset_roots(args)
    output_dir = Path(args.output_dir)
    crpsd_modules = load_crpsd_modules(Path(args.external_repo_path))

    pairs, duplicate_count = collect_pairs(dataset_roots)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise FileNotFoundError(f"No image/label pairs found in {dataset_roots}")

    train_pairs, val_pairs, gap_pairs = split_pairs(pairs, args.val_ratio, args.split_strategy, args.gap_size, args.seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_raw = output_dir / "raw" / "train"
    val_raw = output_dir / "raw" / "test"
    train_prepared = output_dir / "prepared" / "train"
    val_prepared = output_dir / "prepared" / "test"
    for directory in (
        train_raw / "img",
        train_raw / "slot_label",
        val_raw / "img",
        val_raw / "slot_label",
        train_prepared,
        val_prepared,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    train_stats = convert_split(train_pairs, train_raw, train_prepared, args, crpsd_modules)
    val_stats = convert_split(val_pairs, val_raw, val_prepared, args, crpsd_modules)

    summary = {
        "dataset_roots": [str(path) for path in dataset_roots],
        "output_dir": str(output_dir),
        "image_size": args.image_size,
        "val_ratio": args.val_ratio,
        "split_strategy": args.split_strategy,
        "gap_size": args.gap_size,
        "seed": args.seed,
        "prepared_strategy": args.prepared_strategy,
        "pairing_match_iou": args.pairing_match_iou,
        "optimize_passes": args.optimize_passes,
        "total_pairs": len(pairs),
        "duplicate_pairs_dropped": duplicate_count,
        "dropped_gap_pairs": len(gap_pairs),
        "gap_range": stem_range(gap_pairs),
        "train_range": stem_range(train_pairs),
        "test_range": stem_range(val_pairs),
        "train": train_stats,
        "test": val_stats,
        "raw_format": {
            "train_images": str(train_raw / "img"),
            "train_labels": str(train_raw / "slot_label"),
            "test_images": str(val_raw / "img"),
            "test_labels": str(val_raw / "slot_label"),
        },
        "prepared_format": {
            "train": str(train_prepared),
            "test": str(val_prepared),
            "note": "Each .json is a list of generalized marking points expected by external/CRPS-D/data/dataset.py.",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def resolve_dataset_roots(args: argparse.Namespace) -> list[Path]:
    raw_roots = []
    list_roots = getattr(args, "dataset_roots", None)
    repeated_roots = getattr(args, "dataset_root", None)
    if list_roots:
        raw_roots.extend(list_roots)
    if repeated_roots:
        raw_roots.extend(repeated_roots)
    if not raw_roots:
        raw_roots = ["/home/slomauh/Downloads/data1"]

    resolved = []
    seen = set()
    for root in raw_roots:
        path = Path(root).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


def collect_pairs(dataset_roots: list[Path]) -> tuple[list[tuple[Path, Path]], int]:
    pairs_by_stem: dict[str, tuple[Path, Path]] = {}
    duplicate_count = 0
    for dataset_root in dataset_roots:
        image_dir = dataset_root / "BEV" / "Data" / "Image"
        label_dir = dataset_root / "BEV" / "Data" / "label"
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.json"
            if not label_path.exists():
                continue
            if image_path.stem in pairs_by_stem:
                duplicate_count += 1
                continue
            pairs_by_stem[image_path.stem] = (image_path, label_path)
    pairs = [pairs_by_stem[stem] for stem in sorted(pairs_by_stem, key=int)]
    return pairs, duplicate_count


def split_pairs(
    pairs: list[tuple[Path, Path]],
    val_ratio: float,
    split_strategy: str,
    gap_size: int,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    val_count = int(round(len(pairs) * val_ratio))
    if val_count <= 0:
        return pairs, [], []

    if split_strategy == "random":
        rng = random.Random(seed)
        shuffled_pairs = pairs.copy()
        rng.shuffle(shuffled_pairs)
        return shuffled_pairs[val_count:], shuffled_pairs[:val_count], []

    split_start = max(0, len(pairs) - val_count)
    gap_start = max(0, split_start - max(0, gap_size))
    train_pairs = pairs[:gap_start]
    gap_pairs = pairs[gap_start:split_start]
    val_pairs = pairs[split_start:]
    return train_pairs, val_pairs, gap_pairs


def stem_range(pairs: list[tuple[Path, Path]]) -> dict[str, str | None]:
    if not pairs:
        return {"first": None, "last": None}
    return {"first": pairs[0][0].stem, "last": pairs[-1][0].stem}


def convert_split(
    pairs: list[tuple[Path, Path]],
    raw_split_dir: Path,
    prepared_split_dir: Path,
    args: argparse.Namespace,
    crpsd_modules: dict | None,
) -> dict:
    stats = {
        "images": 0,
        "slots": 0,
        "marks": 0,
        "skipped_images": 0,
        "skipped_marks": 0,
        "inferred_slots": 0,
        "matched_slots": 0,
        "false_negative_slots": 0,
        "false_positive_slots": 0,
        "failed_pairing_frames": 0,
    }

    for image_path, label_path in tqdm(pairs, desc=f"Converting {raw_split_dir.name}"):
        image = cv2.imread(str(image_path))
        if image is None:
            stats["skipped_images"] += 1
            continue

        label = json.loads(label_path.read_text(encoding="utf-8"))
        resized_image, scale_x, scale_y = resize_to_square(image, args.image_size)
        converted_label = convert_raw_label(label, scale_x, scale_y, args.image_size)
        generalized_marks, pairing_stats = build_prepared_marks(converted_label, args, crpsd_modules)

        stats["images"] += 1
        stats["slots"] += len(converted_label["slots"])
        stats["marks"] += len(generalized_marks)
        stats["skipped_marks"] += pairing_stats["skipped_marks"]
        stats["inferred_slots"] += pairing_stats["inferred_slots"]
        stats["matched_slots"] += pairing_stats["matched_slots"]
        stats["false_negative_slots"] += pairing_stats["false_negative_slots"]
        stats["false_positive_slots"] += pairing_stats["false_positive_slots"]
        stats["failed_pairing_frames"] += int(pairing_stats["false_negative_slots"] > 0 or pairing_stats["false_positive_slots"] > 0)

        raw_image_path = raw_split_dir / "img" / image_path.name
        raw_label_path = raw_split_dir / "slot_label" / f"{image_path.stem}.json"
        prepared_image_path = prepared_split_dir / image_path.name
        prepared_label_path = prepared_split_dir / f"{image_path.stem}.json"

        cv2.imwrite(str(raw_image_path), resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        cv2.imwrite(str(prepared_image_path), resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        raw_label_path.write_text(json.dumps(converted_label), encoding="utf-8")
        prepared_label_path.write_text(json.dumps(generalized_marks), encoding="utf-8")

    stats["prepared_pairing_recall"] = stats["matched_slots"] / max(1, stats["slots"])
    stats["prepared_pairing_precision"] = stats["matched_slots"] / max(1, stats["inferred_slots"])
    return stats


def resize_to_square(image, image_size: int):
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized_image, image_size / width, image_size / height


def convert_raw_label(label: dict, scale_x: float, scale_y: float, image_size: int) -> dict:
    marks = []
    old_to_new_index: dict[int, int] = {}
    for old_index, mark in enumerate(label.get("marks", []), start=1):
        if not isinstance(mark, list) or len(mark) < 5:
            continue
        x0 = clamp(float(mark[0]) * scale_x, 0.0, image_size - 1.0)
        y0 = clamp(float(mark[1]) * scale_y, 0.0, image_size - 1.0)
        x1 = clamp(float(mark[2]) * scale_x, 0.0, image_size - 1.0)
        y1 = clamp(float(mark[3]) * scale_y, 0.0, image_size - 1.0)
        mark_type = int(round(float(mark[4])))
        marks.append([x0, y0, x1, y1, mark_type])
        old_to_new_index[old_index] = len(marks)

    slots = []
    for slot in label.get("slots", []):
        if not isinstance(slot, list) or len(slot) < 4:
            continue
        mark_a = old_to_new_index.get(int(slot[0]))
        mark_b = old_to_new_index.get(int(slot[1]))
        if mark_a is None or mark_b is None:
            continue
        slots.append([mark_a, mark_b, int(round(float(slot[2]))), float(slot[3])])

    return {"marks": marks, "slots": slots}


def build_prepared_marks(
    converted_label: dict,
    args: argparse.Namespace,
    crpsd_modules: dict | None,
) -> tuple[list[list[float]], dict[str, int]]:
    if args.prepared_strategy == "optimized":
        if crpsd_modules is None:
            raise ValueError("optimized prepared strategy requires CRPS-D modules")
        return optimize_generalized_marks(converted_label, args.image_size, args, crpsd_modules)
    generalized_marks, skipped_marks = generalize_marks_simple(converted_label["marks"], args.image_size)
    pairing_stats = score_generalized_marks(converted_label, generalized_marks, args.image_size, args, crpsd_modules)
    pairing_stats["skipped_marks"] = skipped_marks
    return generalized_marks, pairing_stats


def generalize_marks_simple(marks: list[list[float]], image_size: int) -> tuple[list[list[float]], int]:
    generalized_marks = []
    skipped = 0
    for mark in marks:
        if len(mark) < 5:
            skipped += 1
            continue
        x0, y0, x1, y1, mark_type = mark
        if not (0.0 <= x0 < image_size and 0.0 <= y0 < image_size):
            skipped += 1
            continue

        direction0 = math.atan2(y1 - y0, x1 - x0)
        direction1 = normalize_angle(direction0 + math.pi / 2)
        xval = x0 / image_size
        yval = y0 / image_size
        shape = 0.0
        generalized_marks.append([xval, yval, direction0, direction1, shape, float(mark_type)])

    return generalized_marks, skipped


def optimize_generalized_marks(
    converted_label: dict,
    image_size: int,
    args: argparse.Namespace,
    crpsd_modules: dict,
) -> tuple[list[list[float]], dict[str, int]]:
    marks = converted_label["marks"]
    simple_marks, skipped_marks = generalize_marks_simple(marks, image_size)
    if not simple_marks:
        pairing_stats = score_generalized_marks(converted_label, simple_marks, image_size, args, crpsd_modules)
        pairing_stats["skipped_marks"] = skipped_marks
        return simple_marks, pairing_stats

    candidates_by_mark = build_mark_candidates(converted_label, image_size)
    current = [candidates[0] for candidates in candidates_by_mark]
    best_score, best_stats = score_prepared_tuple(converted_label, current, image_size, args, crpsd_modules)

    for _ in range(max(1, args.optimize_passes)):
        changed = False
        for mark_idx, candidates in enumerate(candidates_by_mark):
            mark_best = current[mark_idx]
            mark_best_score = best_score
            mark_best_stats = best_stats
            for candidate in candidates:
                if candidate == current[mark_idx]:
                    continue
                trial = current.copy()
                trial[mark_idx] = candidate
                trial_score, trial_stats = score_prepared_tuple(converted_label, trial, image_size, args, crpsd_modules)
                if trial_score > mark_best_score:
                    mark_best = candidate
                    mark_best_score = trial_score
                    mark_best_stats = trial_stats
            if mark_best != current[mark_idx]:
                current[mark_idx] = mark_best
                best_score = mark_best_score
                best_stats = mark_best_stats
                changed = True
        if not changed:
            break

    best_stats["skipped_marks"] = skipped_marks
    return [list(candidate) for candidate in current], best_stats


def build_mark_candidates(converted_label: dict, image_size: int) -> list[list[tuple[float, float, float, float, float, float]]]:
    marks = converted_label["marks"]
    slots = converted_label["slots"]
    incident_by_mark: dict[int, list[tuple[int, list]]] = {idx: [] for idx in range(1, len(marks) + 1)}
    for slot in slots:
        if not isinstance(slot, list) or len(slot) < 2:
            continue
        mark_a_idx = int(slot[0])
        mark_b_idx = int(slot[1])
        if mark_a_idx in incident_by_mark:
            incident_by_mark[mark_a_idx].append((mark_b_idx, slot))
        if mark_b_idx in incident_by_mark:
            incident_by_mark[mark_b_idx].append((mark_a_idx, slot))

    candidates_by_mark = []
    for mark_idx, mark in enumerate(marks, start=1):
        if len(mark) < 5:
            candidates_by_mark.append([])
            continue
        x0, y0, x1, y1, raw_mark_type = mark
        xval = x0 / image_size
        yval = y0 / image_size
        own_direction = math.atan2(y1 - y0, x1 - x0)
        directions = [
            own_direction,
            normalize_angle(own_direction + math.pi),
            normalize_angle(own_direction + math.pi / 2),
            normalize_angle(own_direction - math.pi / 2),
        ]

        for neighbor_idx, _ in incident_by_mark.get(mark_idx, []):
            if 1 <= neighbor_idx <= len(marks):
                neighbor = marks[neighbor_idx - 1]
                bridge = math.atan2(float(neighbor[1]) - y0, float(neighbor[0]) - x0)
                directions.extend(
                    [
                        bridge,
                        normalize_angle(bridge + math.pi),
                        normalize_angle(bridge + math.pi / 2),
                        normalize_angle(bridge - math.pi / 2),
                    ]
                )

        normalized_directions = unique_angles(directions)
        candidates = []
        for direction0 in normalized_directions:
            secondary_directions = unique_angles(
                [
                    normalize_angle(direction0 + math.pi / 2),
                    normalize_angle(direction0 - math.pi / 2),
                    normalize_angle(direction0 + math.pi),
                    own_direction,
                    normalize_angle(own_direction + math.pi / 2),
                ]
            )
            for direction1 in secondary_directions:
                candidates.append((xval, yval, direction0, direction1, 0.0, 0.0))
                candidates.append((xval, yval, direction0, direction1, 1.0, 0.0))
                candidates.append((xval, yval, direction0, direction1, 0.0, 1.0))
                candidates.append((xval, yval, direction0, direction1, 1.0, 1.0))

        simple_type = 0.0 if float(raw_mark_type) < 0.5 else 1.0
        simple = (xval, yval, own_direction, normalize_angle(own_direction + math.pi / 2), 0.0, simple_type)
        forced_vertical = (xval, yval, own_direction, normalize_angle(own_direction + math.pi / 2), 0.0, 0.0)
        candidates = [forced_vertical, simple] + candidates
        candidates_by_mark.append(unique_candidates(candidates))

    return candidates_by_mark


def score_prepared_tuple(
    converted_label: dict,
    prepared_marks: list[tuple[float, float, float, float, float, float]],
    image_size: int,
    args: argparse.Namespace,
    crpsd_modules: dict,
) -> tuple[tuple[float, float, float, float], dict[str, int]]:
    stats = score_generalized_marks(converted_label, [list(mark) for mark in prepared_marks], image_size, args, crpsd_modules)
    avg_iou = stats.get("matched_iou_sum", 0.0) / max(1, stats["matched_slots"])
    score = (
        float(stats["matched_slots"]),
        -float(stats["false_negative_slots"]),
        -float(stats["false_positive_slots"]),
        avg_iou,
    )
    return score, stats


def score_generalized_marks(
    converted_label: dict,
    generalized_marks: list[list[float]],
    image_size: int,
    args: argparse.Namespace,
    crpsd_modules: dict | None,
) -> dict[str, int]:
    if crpsd_modules is None:
        return {
            "inferred_slots": 0,
            "matched_slots": 0,
            "false_negative_slots": len(converted_label["slots"]),
            "false_positive_slots": 0,
            "matched_iou_sum": 0.0,
        }

    marking_points = [crpsd_modules["MarkingPoint"](*mark) for mark in generalized_marks]
    inferred_raw_slots = crpsd_modules["inference_slots"](marking_points) if marking_points else []
    gt_polygons = raw_slots_to_polygons(converted_label)
    inferred_polygons = crpsd_slots_to_polygons(marking_points, inferred_raw_slots, image_size, crpsd_modules["config"])
    matches = match_polygons(gt_polygons, inferred_polygons, args.pairing_match_iou)
    matched_gt = {match["gt_idx"] for match in matches}
    matched_pred = {match["pred_idx"] for match in matches}
    return {
        "inferred_slots": len(inferred_polygons),
        "matched_slots": len(matches),
        "false_negative_slots": len(gt_polygons) - len(matched_gt),
        "false_positive_slots": len(inferred_polygons) - len(matched_pred),
        "matched_iou_sum": sum(match["iou"] for match in matches),
    }


def raw_slots_to_polygons(converted_label: dict) -> list[list[tuple[float, float]]]:
    marks = converted_label.get("marks", [])
    slots = converted_label.get("slots", [])
    polygons = []
    for slot in slots:
        if not isinstance(slot, list) or len(slot) < 2:
            continue
        mark_a_idx = int(slot[0])
        mark_b_idx = int(slot[1])
        if mark_a_idx < 1 or mark_b_idx < 1 or mark_a_idx > len(marks) or mark_b_idx > len(marks):
            continue
        mark_a = marks[mark_a_idx - 1]
        mark_b = marks[mark_b_idx - 1]
        polygons.append(
            [
                (float(mark_a[0]), float(mark_a[1])),
                (float(mark_b[0]), float(mark_b[1])),
                (float(mark_b[2]), float(mark_b[3])),
                (float(mark_a[2]), float(mark_a[3])),
            ]
        )
    return polygons


def crpsd_slots_to_polygons(marking_points, raw_slots, image_size: int, crpsd_config) -> list[list[tuple[float, float]]]:
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

        cos_val = math.cos(raw_slot[2])
        sin_val = math.sin(raw_slot[2])
        p2_x = p0_x + image_size * separating_length * cos_val
        p2_y = p0_y + image_size * separating_length * sin_val
        p3_x = p1_x + image_size * separating_length * cos_val
        p3_y = p1_y + image_size * separating_length * sin_val
        polygons.append([(p0_x, p0_y), (p1_x, p1_y), (p3_x, p3_y), (p2_x, p2_y)])
    return polygons


def match_polygons(gt_slots, pred_slots, min_iou: float) -> list[dict]:
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


def unique_angles(angles: list[float], tolerance: float = 1e-3) -> list[float]:
    unique = []
    for angle in angles:
        normalized = normalize_angle(angle)
        if all(abs(normalize_angle(normalized - existing)) > tolerance for existing in unique):
            unique.append(normalized)
    return unique


def unique_candidates(candidates: list[tuple[float, float, float, float, float, float]]) -> list[tuple[float, float, float, float, float, float]]:
    unique = []
    seen = set()
    for candidate in candidates:
        key = tuple(round(value, 4) for value in candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def load_crpsd_modules(external_repo_path: Path) -> dict:
    external_repo_path = external_repo_path.resolve()
    if not external_repo_path.exists():
        raise FileNotFoundError(f"CRPS-D repo not found: {external_repo_path}")
    install_visdom_stub()
    if str(external_repo_path) not in sys.path:
        sys.path.insert(0, str(external_repo_path))

    import config as crpsd_config
    from data.struct import MarkingPoint
    from inference import inference_slots

    return {"config": crpsd_config, "MarkingPoint": MarkingPoint, "inference_slots": inference_slots}


def install_visdom_stub() -> None:
    if "visdom" in sys.modules:
        return

    visdom = types.ModuleType("visdom")

    class Visdom:
        def __init__(self, *args, **kwargs) -> None:
            pass

    visdom.Visdom = Visdom
    sys.modules["visdom"] = visdom


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


if __name__ == "__main__":
    main()
