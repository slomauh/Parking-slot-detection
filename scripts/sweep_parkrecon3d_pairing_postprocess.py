from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import types
from collections import Counter, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class SweepVariant:
    name: str
    distance_scale: float = 1.0
    angle_scale: float = 1.0
    suppress_dot: float = 0.8
    allow_any_pairing: bool = False
    smart_suppress_distance: float = -1.0
    cross_prune_overlap: float = 0.0
    cross_prune_min_crossings: int = 2
    max_point_degree: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep relaxed CRPS-D slot pairing postprocess on ParkRecon3D prepared labels"
    )
    parser.add_argument("--dataset-dir", default="outputs/parkrecon3d_bev_crpsd_format")
    parser.add_argument("--external-repo-path", default="external/CRPS-D")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_pairing_postprocess_sweep")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modules = load_crpsd_modules(Path(args.external_repo_path))
    variants = build_variants()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for variant in tqdm(variants, desc="Sweeping postprocess variants"):
        counts = Counter()
        for split in args.splits:
            counts.update(evaluate_split(dataset_dir, split, modules, variant, args))
        results.append(build_result_row(variant, counts))

    results_by_recall = sorted(results, key=lambda row: (row["recall"], row["precision"], row["f1"]), reverse=True)
    results_by_f1 = sorted(results, key=lambda row: (row["f1"], row["recall"], row["precision"]), reverse=True)
    write_csv(output_dir / "sweep_results.csv", results_by_f1)
    summary = {
        "dataset_dir": str(dataset_dir),
        "match_iou": args.match_iou,
        "splits": args.splits,
        "limit": args.limit,
        "best_by_f1": results_by_f1[0] if results_by_f1 else None,
        "best_by_recall": results_by_recall[0] if results_by_recall else None,
        "top_by_f1": results_by_f1[: args.top_k],
        "top_by_recall": results_by_recall[: args.top_k],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_variants() -> list[SweepVariant]:
    variants = [
        SweepVariant("crpsd_default"),
        SweepVariant("disable_third_point", suppress_dot=1.01),
    ]
    for distance_scale in (1.0, 1.15, 1.35, 1.6, 2.0):
        for angle_scale in (1.0, 1.25, 1.5, 2.0):
            for suppress_dot in (0.8, 0.95, 1.01):
                variants.append(
                    SweepVariant(
                        name=f"d{distance_scale:g}_a{angle_scale:g}_s{suppress_dot:g}",
                        distance_scale=distance_scale,
                        angle_scale=angle_scale,
                        suppress_dot=suppress_dot,
                    )
                )
                variants.append(
                    SweepVariant(
                        name=f"d{distance_scale:g}_a{angle_scale:g}_s{suppress_dot:g}_any",
                        distance_scale=distance_scale,
                        angle_scale=angle_scale,
                        suppress_dot=suppress_dot,
                        allow_any_pairing=True,
                    )
                )
    for smart_distance in (0.004, 0.0075, 0.0125, 0.02):
        variants.append(
            SweepVariant(
                name=f"smart_suppress_{smart_distance:g}",
                smart_suppress_distance=smart_distance,
            )
        )
        for cross_overlap in (0.08, 0.15, 0.25):
            for min_crossings in (2, 3):
                variants.append(
                    SweepVariant(
                        name=f"smart_suppress_{smart_distance:g}_cross{cross_overlap:g}_n{min_crossings}",
                        smart_suppress_distance=smart_distance,
                        cross_prune_overlap=cross_overlap,
                        cross_prune_min_crossings=min_crossings,
                    )
                )
    for cross_overlap in (0.08, 0.15, 0.25):
        for min_crossings in (2, 3):
            variants.append(
                SweepVariant(
                    name=f"disable_third_point_cross{cross_overlap:g}_n{min_crossings}",
                    suppress_dot=1.01,
                    cross_prune_overlap=cross_overlap,
                    cross_prune_min_crossings=min_crossings,
                )
            )
    for max_degree in (2, 3):
        variants.append(
            SweepVariant(
                name=f"disable_third_point_degree{max_degree}",
                suppress_dot=1.01,
                max_point_degree=max_degree,
            )
        )
        variants.append(
            SweepVariant(
                name=f"smart_suppress_0.004_degree{max_degree}",
                smart_suppress_distance=0.004,
                max_point_degree=max_degree,
            )
        )
        variants.append(
            SweepVariant(
                name=f"smart_suppress_0.0075_degree{max_degree}",
                smart_suppress_distance=0.0075,
                max_point_degree=max_degree,
            )
        )
    unique = {}
    for variant in variants:
        unique[
            (
                variant.distance_scale,
                variant.angle_scale,
                variant.suppress_dot,
                variant.allow_any_pairing,
                variant.smart_suppress_distance,
                variant.cross_prune_overlap,
                variant.cross_prune_min_crossings,
                variant.max_point_degree,
            )
        ] = variant
    return list(unique.values())


def evaluate_split(
    dataset_dir: Path,
    split: str,
    modules: dict[str, Any],
    variant: SweepVariant,
    args: argparse.Namespace,
) -> Counter[str]:
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

    counts = Counter()
    for image_path in image_paths:
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
        counts["images"] += 1
        counts["gt_slots"] += len(gt_slots)
        counts["prepared_marks"] += len(prepared_label)
        counts["inferred_slots"] += len(inferred_slots)
        counts["matched_slots"] += len(matches)
        counts["false_negative_slots"] += len(gt_slots) - len(matched_gt_ids)
        counts["false_positive_slots"] += len(inferred_slots) - len(matched_pred_ids)
    return counts


def inference_slots_relaxed(marking_points, config, variant: SweepVariant) -> list[tuple[int, int, float]]:
    slots = []
    for i in range(len(marking_points) - 1):
        for j in range(i + 1, len(marking_points)):
            point_i = marking_points[i]
            point_j = marking_points[j]
            distance = calc_point_square_dist(point_i, point_j)
            if not distance_allowed(point_i, point_j, distance, config, variant):
                continue
            if pass_through_third_point(marking_points, i, j, variant):
                continue

            pair_results = []
            if variant.allow_any_pairing:
                pair_results.append(pair_marking_points_vertical(point_i, point_j, config, variant))
                pair_results.append(pair_marking_points_slant(point_i, point_j, config, variant))
            elif point_i.type < 0.5:
                pair_results.append(pair_marking_points_vertical(point_i, point_j, config, variant))
            else:
                pair_results.append(pair_marking_points_slant(point_i, point_j, config, variant))

            for result, angle in pair_results:
                if result == 1:
                    slots.append((i, j, angle))
                    break
                if result == -1:
                    slots.append((j, i, angle))
                    break
    return slots


def distance_allowed(point_i, point_j, distance: float, config, variant: SweepVariant) -> bool:
    scale = variant.distance_scale
    use_slant = (point_i.type < 0.5 < point_j.type) or (point_j.type < 0.5 < point_i.type)
    use_vertical = False
    if use_slant and distance > scaled_max(config.SLANT_MAX_DIST, scale, config):
        use_vertical = True
        use_slant = False

    if variant.allow_any_pairing:
        return (
            scaled_min(config.VSLOT_MIN_DIST, scale, config) <= distance <= scaled_max(config.VSLOT_MAX_DIST, scale, config)
            or scaled_min(config.HSLOT_MIN_DIST, scale, config) <= distance <= scaled_max(config.HSLOT_MAX_DIST, scale, config)
            or scaled_min(config.SLANT_MIN_DIST, scale, config) <= distance <= scaled_max(config.SLANT_MAX_DIST, scale, config)
        )
    if point_i.type < 0.5:
        return (
            scaled_min(config.VSLOT_MIN_DIST, scale, config) <= distance <= scaled_max(config.VSLOT_MAX_DIST, scale, config)
            or scaled_min(config.HSLOT_MIN_DIST, scale, config) <= distance <= scaled_max(config.HSLOT_MAX_DIST, scale, config)
            or use_vertical
        )
    return scaled_min(config.SLANT_MIN_DIST, scale, config) <= distance <= scaled_max(config.SLANT_MAX_DIST, scale, config) or use_vertical


def scaled_min(value: float, scale: float, config) -> float:
    return (value / scale) * config.SQUARED_RATIO


def scaled_max(value: float, scale: float, config) -> float:
    return (value * scale) * config.SQUARED_RATIO


def pass_through_third_point(marking_points, i: int, j: int, variant: SweepVariant) -> bool:
    if variant.smart_suppress_distance >= 0:
        return pass_through_third_point_smart(marking_points, i, j, variant.smart_suppress_distance)
    if variant.suppress_dot > 1.0:
        return False
    x_1 = marking_points[i].x
    y_1 = marking_points[i].y
    x_2 = marking_points[j].x
    y_2 = marking_points[j].y
    for point_idx, point in enumerate(marking_points):
        if point_idx == i or point_idx == j:
            continue
        if point.type > 0.5 and marking_points[i].type < 0.5:
            continue
        if point.type < 0.5 and marking_points[i].type > 0.5:
            continue
        vec1 = np.array([point.x - x_1, point.y - y_1])
        vec2 = np.array([x_2 - point.x, y_2 - point.y])
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            continue
        if np.dot(vec1 / norm1, vec2 / norm2) > variant.suppress_dot:
            return True
    return False


def pass_through_third_point_smart(marking_points, i: int, j: int, max_line_distance: float) -> bool:
    point_i = marking_points[i]
    point_j = marking_points[j]
    start = np.array([point_i.x, point_i.y], dtype=np.float32)
    end = np.array([point_j.x, point_j.y], dtype=np.float32)
    segment = end - start
    length = float(np.linalg.norm(segment))
    if length == 0:
        return False
    for point_idx, point in enumerate(marking_points):
        if point_idx == i or point_idx == j:
            continue
        if point.type > 0.5 and point_i.type < 0.5:
            continue
        if point.type < 0.5 and point_i.type > 0.5:
            continue
        candidate = np.array([point.x, point.y], dtype=np.float32)
        projection = float(np.dot(candidate - start, segment) / (length * length))
        if projection <= 0.08 or projection >= 0.92:
            continue
        closest = start + projection * segment
        distance_to_line = float(np.linalg.norm(candidate - closest))
        if distance_to_line <= max_line_distance:
            return True
    return False


def postprocess_inferred_slots(image, marking_points, raw_slots, config, variant: SweepVariant):
    polygons = crpsd_slots_to_polygons(image, marking_points, raw_slots, config)
    if variant.cross_prune_overlap > 0 and len(polygons) >= 3:
        remove_ids = find_crossing_slot_ids(
            raw_slots,
            polygons,
            min_overlap=variant.cross_prune_overlap,
            min_crossings=variant.cross_prune_min_crossings,
        )
        if remove_ids:
            raw_slots = [slot for idx, slot in enumerate(raw_slots) if idx not in remove_ids]
            polygons = [polygon for idx, polygon in enumerate(polygons) if idx not in remove_ids]
    if variant.max_point_degree > 0 and len(raw_slots) >= 3:
        raw_slots, polygons = prune_by_point_degree(raw_slots, polygons, variant.max_point_degree)
    return raw_slots, polygons


def prune_by_point_degree(raw_slots, polygons, max_degree: int):
    order = sorted(range(len(raw_slots)), key=lambda idx: bridge_length(polygons[idx]))
    degrees: Counter[int] = Counter()
    keep_ids = []
    for idx in order:
        point_a = int(raw_slots[idx][0])
        point_b = int(raw_slots[idx][1])
        if degrees[point_a] >= max_degree or degrees[point_b] >= max_degree:
            continue
        degrees[point_a] += 1
        degrees[point_b] += 1
        keep_ids.append(idx)
    keep_ids = set(keep_ids)
    return [slot for idx, slot in enumerate(raw_slots) if idx in keep_ids], [
        polygon for idx, polygon in enumerate(polygons) if idx in keep_ids
    ]


def bridge_length(points) -> float:
    p0 = np.array(points[0], dtype=np.float32)
    p1 = np.array(points[1], dtype=np.float32)
    return float(np.linalg.norm(p1 - p0))


def find_crossing_slot_ids(raw_slots, polygons, min_overlap: float, min_crossings: int) -> set[int]:
    shapely_polygons = [make_polygon(points) for points in polygons]
    angles = [slot_bridge_angle(points) for points in polygons]
    bridge_segments = [(np.array(points[0], dtype=np.float32), np.array(points[1], dtype=np.float32)) for points in polygons]
    remove_ids: set[int] = set()
    for idx, polygon in enumerate(shapely_polygons):
        if polygon is None:
            continue
        crossing_ids = []
        for other_idx, other_polygon in enumerate(shapely_polygons):
            if idx == other_idx or other_polygon is None:
                continue
            if direction_diff(angles[idx], angles[other_idx]) < math.radians(55):
                continue
            polygon_cross = False
            min_area = min(float(polygon.area), float(other_polygon.area))
            if min_area > 0:
                overlap = float(polygon.intersection(other_polygon).area) / min_area
                polygon_cross = overlap >= min_overlap
            segment_cross = segments_cross_or_close(
                bridge_segments[idx][0],
                bridge_segments[idx][1],
                bridge_segments[other_idx][0],
                bridge_segments[other_idx][1],
                max_distance=8.0,
            )
            if polygon_cross or segment_cross:
                crossing_ids.append(other_idx)
        if len(crossing_ids) >= min_crossings:
            remove_ids.add(idx)
    return remove_ids


def segments_cross_or_close(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray, max_distance: float) -> bool:
    if segments_intersect(a1, a2, b1, b2):
        return True
    return segment_distance(a1, a2, b1, b2) <= max_distance


def segments_intersect(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> bool:
    def orient(p, q, r) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a1, a2, b1)
    o2 = orient(a1, a2, b2)
    o3 = orient(b1, b2, a1)
    o4 = orient(b1, b2, a2)
    return o1 * o2 < 0 and o3 * o4 < 0


def segment_distance(a1: np.ndarray, a2: np.ndarray, b1: np.ndarray, b2: np.ndarray) -> float:
    return min(
        point_segment_distance(a1, b1, b2),
        point_segment_distance(a2, b1, b2),
        point_segment_distance(b1, a1, a2),
        point_segment_distance(b2, a1, a2),
    )


def point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_squared = float(np.dot(segment, segment))
    if length_squared == 0:
        return float(np.linalg.norm(point - start))
    projection = max(0.0, min(1.0, float(np.dot(point - start, segment) / length_squared)))
    closest = start + projection * segment
    return float(np.linalg.norm(point - closest))


def slot_bridge_angle(points) -> float:
    p0 = points[0]
    p1 = points[1]
    return math.atan2(p1[1] - p0[1], p1[0] - p0[0])


def pair_marking_points_vertical(point_a, point_b, config, variant: SweepVariant) -> tuple[int, float]:
    vector_ab = np.array([point_b.x - point_a.x, point_b.y - point_a.y])
    norm = np.linalg.norm(vector_ab)
    if norm == 0:
        return 0, 0.0
    vector_ab = vector_ab / norm
    point_shape_a = determine_point_shape_vertical(point_a, vector_ab, config, variant)
    point_shape_b = determine_point_shape_vertical(point_b, -vector_ab, config, variant)
    if point_shape_a == 0 or point_shape_b == 0:
        return 0, 0.0
    if point_shape_a == 3 and point_shape_b == 3:
        return 0, 0.0
    if point_shape_a > 3 and point_shape_b > 3:
        return 0, 0.0
    if point_shape_a < 3 and point_shape_b < 3:
        return 0, 0.0
    vec_direct_up = math.atan2(-vector_ab[0], vector_ab[1])
    vec_direct_down = math.atan2(vector_ab[0], -vector_ab[1])
    if point_shape_a != 3:
        if point_shape_a > 3:
            return 1, vec_direct_up
        if point_shape_a < 3:
            return -1, vec_direct_down
    if point_shape_b < 3:
        return 1, vec_direct_up
    if point_shape_b > 3:
        return -1, vec_direct_down
    return 0, 0.0


def pair_marking_points_slant(point_a, point_b, config, variant: SweepVariant) -> tuple[int, float]:
    vector_ab = np.array([point_b.x - point_a.x, point_b.y - point_a.y])
    norm = np.linalg.norm(vector_ab)
    if norm == 0:
        return 0, 0.0
    vector_ab = vector_ab / norm
    point_shape_a = determine_point_shape_slant(point_a, vector_ab, config, variant)
    point_shape_b = determine_point_shape_slant(point_b, -vector_ab, config, variant)
    if point_shape_a == 0 or point_shape_b == 0:
        return 0, 0.0
    if point_shape_a > 3 and point_shape_b > 3:
        return 0, 0.0
    if point_shape_a < 3 and point_shape_b < 3:
        return 0, 0.0
    point_angle_a = calc_slant_angle(point_shape_a, point_a, vector_ab)
    point_angle_b = calc_slant_angle(point_shape_b, point_b, -vector_ab)
    if abs(point_angle_a - point_angle_b) < config.SEPARATOR_ANGLE_DIFF * variant.angle_scale:
        if point_shape_a > 3:
            return 1, (point_angle_a + point_angle_b) / 2
        if point_shape_a < 3:
            return -1, (point_angle_a + point_angle_b) / 2
    return 0, 0.0


def determine_point_shape_vertical(point, vector, config, variant: SweepVariant) -> int:
    vec_direct = math.atan2(vector[1], vector[0])
    vec_direct_up = math.atan2(-vector[0], vector[1])
    vec_direct_down = math.atan2(vector[0], -vector[1])
    bridge_thresh = config.BRIDGE_ANGLE_DIFF * variant.angle_scale
    separator_thresh = config.SEPARATOR_ANGLE_DIFF * variant.angle_scale
    if point.shape < 0.5:
        if direction_diff(vec_direct, point.direction0) < bridge_thresh:
            return 3
        if direction_diff(vec_direct_up, point.direction0) < separator_thresh:
            return 4
        if direction_diff(vec_direct_down, point.direction0) < separator_thresh:
            return 2
    else:
        if direction_diff(vec_direct, point.direction0) < bridge_thresh:
            return 1
        if direction_diff(vec_direct_up, point.direction0) < separator_thresh:
            return 5
    return 0


def determine_point_shape_slant(point, vector, config, variant: SweepVariant) -> int:
    vec_direct = math.atan2(vector[1], vector[0])
    if point.shape < 0.5:
        if -math.pi < point.direction0 - vec_direct < 0 or point.direction0 - vec_direct > math.pi:
            return 4
        return 2
    if direction_diff(vec_direct, point.direction0) < config.BRIDGE_ANGLE_DIFF * variant.angle_scale:
        return 1
    if direction_diff(vec_direct, point.direction1) < config.BRIDGE_ANGLE_DIFF * variant.angle_scale:
        return 5
    return 0


def calc_slant_angle(point_shape: int, point, vector) -> float:
    if point_shape in {1, 5}:
        vec_direct = math.atan2(vector[1], vector[0])
        if direction_diff(vec_direct, point.direction0) < direction_diff(vec_direct, point.direction1):
            return point.direction1
        return point.direction0
    return point.direction0


def calc_point_square_dist(point_a, point_b) -> float:
    return (point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2


def direction_diff(direction_a: float, direction_b: float) -> float:
    diff = abs(direction_a - direction_b)
    return diff if diff < math.pi else 2 * math.pi - diff


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


def crpsd_slots_to_polygons(image, marking_points, raw_slots, config) -> list[list[tuple[float, float]]]:
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
            distance = calc_point_square_dist(point_a, point_b)
            if distance <= config.VSLOT_MAX_DIST * config.SQUARED_RATIO:
                separating_length = config.LONG_SEPARATOR_LENGTH * config.RATIO
            else:
                separating_length = config.SHORT_SEPARATOR_LENGTH * config.RATIO
        else:
            separating_length = config.SLANT_SEPARATOR_LENGTH * config.RATIO

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


def build_result_row(variant: SweepVariant, counts: Counter[str]) -> dict[str, Any]:
    matched = counts["matched_slots"]
    gt_slots = counts["gt_slots"]
    inferred_slots = counts["inferred_slots"]
    precision = matched / max(1, inferred_slots)
    recall = matched / max(1, gt_slots)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "name": variant.name,
        "distance_scale": variant.distance_scale,
        "angle_scale": variant.angle_scale,
        "suppress_dot": variant.suppress_dot,
        "allow_any_pairing": variant.allow_any_pairing,
        "smart_suppress_distance": variant.smart_suppress_distance,
        "cross_prune_overlap": variant.cross_prune_overlap,
        "cross_prune_min_crossings": variant.cross_prune_min_crossings,
        "max_point_degree": variant.max_point_degree,
        "images": counts["images"],
        "gt_slots": gt_slots,
        "inferred_slots": inferred_slots,
        "matched_slots": matched,
        "false_negative_slots": counts["false_negative_slots"],
        "false_positive_slots": counts["false_positive_slots"],
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_crpsd_modules(external_repo_path: Path) -> dict[str, Any]:
    external_repo_path = external_repo_path.resolve()
    if not external_repo_path.exists():
        raise FileNotFoundError(f"CRPS-D repo not found: {external_repo_path}")
    if str(external_repo_path) not in sys.path:
        sys.path.insert(0, str(external_repo_path))

    install_visdom_stub()
    import config as crpsd_config

    try:
        from data.struct import MarkingPoint
    except Exception:
        MarkingPoint = namedtuple("MarkingPoint", ["x", "y", "direction0", "direction1", "shape", "type"])

    return {"config": crpsd_config, "MarkingPoint": MarkingPoint}


def install_visdom_stub() -> None:
    if "visdom" in sys.modules:
        return

    visdom = types.ModuleType("visdom")

    class Visdom:
        def __init__(self, *args, **kwargs) -> None:
            pass

    visdom.Visdom = Visdom
    sys.modules["visdom"] = visdom


if __name__ == "__main__":
    main()
