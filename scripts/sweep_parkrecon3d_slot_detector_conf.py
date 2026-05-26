from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_parkrecon3d_bev import build_metrics, detect_slots, load_parkrecon3d_slots, match_slots
from src.detection.parking_slot_detector import ParkingSlotDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep slot confidence and pairing strategy on ParkRecon3D BEV test split")
    parser.add_argument("--image-dir", default="outputs/parkrecon3d_bev_crpsd_format/raw/test/img")
    parser.add_argument("--label-dir", default="outputs/parkrecon3d_bev_crpsd_format/raw/test/slot_label")
    parser.add_argument("--slot-model-path", default="models/slot_detector/parkrecon3d_slot_detector_finetuned.pth")
    parser.add_argument("--slot-external-repo-path", default="external/CRPS-D")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-input-size", type=int, default=512)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30])
    parser.add_argument("--strategies", nargs="+", choices=["crpsd", "relaxed"], default=["crpsd", "relaxed"])
    parser.add_argument("--postprocess-modes", nargs="+", choices=["standard", "row_consensus"], default=["standard"])
    parser.add_argument("--relaxed-max-point-degree", type=int, default=2)
    parser.add_argument("--min-slot-scores", nargs="+", type=float, default=[0.0])
    parser.add_argument("--slot-nms-ious", nargs="+", type=float, default=[0.0])
    parser.add_argument("--slot-center-nms-distances", nargs="+", type=float, default=[0.0])
    parser.add_argument("--slot-angle-nms-thresholds", nargs="+", type=float, default=[20.0])
    parser.add_argument("--slot-centerline-overlap-thresholds", nargs="+", type=float, default=[0.0])
    parser.add_argument("--enable-geometry-filter", action="store_true")
    parser.add_argument("--min-slot-area-ratio", type=float, default=0.0)
    parser.add_argument("--max-slot-area-ratio", type=float, default=1.0)
    parser.add_argument("--max-slot-aspect-ratio", type=float, default=0.0)
    parser.add_argument("--max-out-of-frame-ratio", type=float, default=1.0)
    parser.add_argument("--orientation-filter-modes", nargs="+", choices=["off", "on"], default=["off"])
    parser.add_argument("--orientation-neighbor-radii", nargs="+", type=float, default=[130.0])
    parser.add_argument("--orientation-angle-thresholds", nargs="+", type=float, default=[60.0])
    parser.add_argument("--orientation-min-neighbors", type=int, default=2)
    parser.add_argument("--orientation-score-margins", nargs="+", type=float, default=[1.05])
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_slot_detector_conf_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    image_paths = sorted(image_dir.glob("*.jpg"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    thresholds = sorted(set(args.thresholds))
    min_slot_scores = sorted(set(args.min_slot_scores))
    slot_nms_ious = sorted(set(args.slot_nms_ious))
    slot_center_nms_distances = sorted(set(args.slot_center_nms_distances))
    slot_angle_nms_thresholds = sorted(set(args.slot_angle_nms_thresholds))
    slot_centerline_overlap_thresholds = sorted(set(args.slot_centerline_overlap_thresholds))
    orientation_neighbor_radii = sorted(set(args.orientation_neighbor_radii))
    orientation_angle_thresholds = sorted(set(args.orientation_angle_thresholds))
    orientation_score_margins = sorted(set(args.orientation_score_margins))
    min_threshold = min(thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    detector = ParkingSlotDetector(
        {
            "backend": "crpsd",
            "model_path": args.slot_model_path,
            "external_repo_path": args.slot_external_repo_path,
            "device": args.device,
            "conf_threshold": min_threshold,
            "depth_factor": 32,
            "slot_pairing_strategy": "crpsd",
        }
    )
    detector._load_crpsd_model()
    assert detector._crpsd is not None
    assert detector.model is not None

    variants = build_variants(
        args.strategies,
        args.postprocess_modes,
        thresholds,
        min_slot_scores,
        slot_nms_ious,
        slot_center_nms_distances,
        slot_angle_nms_thresholds,
        slot_centerline_overlap_thresholds,
        args.orientation_filter_modes,
        orientation_neighbor_radii,
        orientation_angle_thresholds,
        args.orientation_min_neighbors,
        orientation_score_margins,
    )
    counters = {variant["key"]: Counter() for variant in variants}

    records = []
    for image_path in tqdm(image_paths, desc="Sweeping slot detector conf"):
        label_path = label_dir / f"{image_path.stem}.json"
        if not label_path.exists():
            continue
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        gt_slots = load_parkrecon3d_slots(label_path)
        detector_frame, scale_x, scale_y = prepare_detector_frame(frame, args.detector_input_size)

        pred_points = detector._crpsd["detect_marking_points"](
            detector.model,
            detector_frame,
            min_threshold,
            detector._crpsd["device"],
        )
        image_record: dict[str, Any] = {"image": str(image_path), "gt_slots": len(gt_slots), "variants": {}}
        for variant in variants:
            apply_variant(detector, variant, args, strategy_max_point_degree=args.relaxed_max_point_degree)
            filtered_points = [item for item in pred_points if float(item[0]) >= variant["slot_conf"]]
            pred_slots = slots_from_pred_points(detector, detector_frame, filtered_points)
            pred_slots = scale_slots(pred_slots, scale_x, scale_y)
            matches = match_slots(gt_slots, pred_slots, args.match_iou)
            key = variant["key"]
            update_counts(counters[key], gt_slots, pred_slots, matches)
            image_record["variants"][key] = {
                "pred_slots": len(pred_slots),
                "matched_slots": len(matches),
                "false_negative_slots": len(gt_slots) - len(matches),
                "false_positive_slots": len(pred_slots) - len(matches),
            }
        records.append(image_record)

    rows = []
    for variant in variants:
        key = variant["key"]
        counts = counters[key]
        metrics = build_metrics(counts)
        precision = metrics["slot_precision"]
        recall = metrics["slot_recall"]
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append(
            {
                "strategy": variant["strategy"],
                "postprocess_mode": variant["postprocess_mode"],
                "slot_conf": variant["slot_conf"],
                "min_slot_score": variant["min_slot_score"],
                "slot_nms_iou": variant["slot_nms_iou"],
                "slot_center_nms_distance": variant["slot_center_nms_distance"],
                "slot_angle_nms_threshold": variant["slot_angle_nms_threshold"],
                "slot_centerline_overlap_threshold": variant["slot_centerline_overlap_threshold"],
                "orientation_filter": variant["orientation_filter"],
                "orientation_neighbor_radius": variant["orientation_neighbor_radius"],
                "orientation_angle_threshold": variant["orientation_angle_threshold"],
                "orientation_score_margin": variant["orientation_score_margin"],
                "images": counts["images"],
                "gt_slots": counts["gt_slots"],
                "pred_slots": counts["pred_slots"],
                "matched_slots": counts["matched_slots"],
                "false_negative_slots": counts["false_negative_slots"],
                "false_positive_slots": counts["false_positive_slots"],
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }
            )
    rows.sort(key=lambda row: (row["f1"], row["recall"], row["precision"]), reverse=True)

    summary = {
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "slot_model_path": args.slot_model_path,
        "match_iou": args.match_iou,
        "limit": args.limit,
        "thresholds": thresholds,
        "strategies": args.strategies,
        "postprocess_modes": args.postprocess_modes,
        "relaxed_max_point_degree": args.relaxed_max_point_degree,
        "min_slot_scores": min_slot_scores,
        "slot_nms_ious": slot_nms_ious,
        "slot_center_nms_distances": slot_center_nms_distances,
        "slot_angle_nms_thresholds": slot_angle_nms_thresholds,
        "slot_centerline_overlap_thresholds": slot_centerline_overlap_thresholds,
        "geometry_filter_enabled": args.enable_geometry_filter,
        "min_slot_area_ratio": args.min_slot_area_ratio,
        "max_slot_area_ratio": args.max_slot_area_ratio,
        "max_slot_aspect_ratio": args.max_slot_aspect_ratio,
        "max_out_of_frame_ratio": args.max_out_of_frame_ratio,
        "orientation_filter_modes": args.orientation_filter_modes,
        "orientation_neighbor_radii": orientation_neighbor_radii,
        "orientation_angle_thresholds": orientation_angle_thresholds,
        "orientation_min_neighbors": args.orientation_min_neighbors,
        "orientation_score_margins": orientation_score_margins,
        "best_by_f1": rows[0] if rows else None,
        "best_by_recall_with_precision_70": first_row(rows, min_precision=0.70),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    write_csv(output_dir / "sweep_results.csv", rows)
    print(json.dumps(summary, indent=2))


def prepare_detector_frame(frame, input_size: int | None):
    if input_size is None:
        return frame, 1.0, 1.0
    original_height, original_width = frame.shape[:2]
    detector_frame = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_AREA)
    return detector_frame, original_width / input_size, original_height / input_size


def slots_from_pred_points(detector: ParkingSlotDetector, detector_frame, pred_points) -> list:
    return detector._convert_pred_points_to_slots(detector_frame, pred_points)


def scale_slots(slots, scale_x: float, scale_y: float) -> list:
    if scale_x == 1.0 and scale_y == 1.0:
        return slots
    for slot in slots:
        slot.points = [(float(x) * scale_x, float(y) * scale_y) for x, y in slot.points]
    return slots


def update_counts(counts: Counter, gt_slots: list, pred_slots: list, matches: list) -> None:
    counts["images"] += 1
    counts["gt_slots"] += len(gt_slots)
    counts["pred_slots"] += len(pred_slots)
    counts["matched_slots"] += len(matches)
    counts["false_negative_slots"] += len(gt_slots) - len(matches)
    counts["false_positive_slots"] += len(pred_slots) - len(matches)


def build_variants(
    strategies: list[str],
    postprocess_modes: list[str],
    thresholds: list[float],
    min_slot_scores: list[float],
    slot_nms_ious: list[float],
    slot_center_nms_distances: list[float],
    slot_angle_nms_thresholds: list[float],
    slot_centerline_overlap_thresholds: list[float],
    orientation_filter_modes: list[str],
    orientation_neighbor_radii: list[float],
    orientation_angle_thresholds: list[float],
    orientation_min_neighbors: int,
    orientation_score_margins: list[float],
) -> list[dict[str, Any]]:
    variants = []
    for strategy in strategies:
        for postprocess_mode in postprocess_modes:
            center_distances = slot_center_nms_distances if postprocess_mode == "row_consensus" else [0.0]
            angle_nms_thresholds = slot_angle_nms_thresholds if postprocess_mode == "row_consensus" else [slot_angle_nms_thresholds[0]]
            centerline_thresholds = (
                slot_centerline_overlap_thresholds if postprocess_mode == "row_consensus" else [0.0]
            )
            for threshold in thresholds:
                for min_slot_score in min_slot_scores:
                    for slot_nms_iou in slot_nms_ious:
                        for center_distance in center_distances:
                            for angle_nms_threshold in angle_nms_thresholds:
                                for centerline_threshold in centerline_thresholds:
                                    for orientation_mode in orientation_filter_modes:
                                        radii = (
                                            orientation_neighbor_radii
                                            if orientation_mode == "on"
                                            else [orientation_neighbor_radii[0]]
                                        )
                                        angle_thresholds = (
                                            orientation_angle_thresholds
                                            if orientation_mode == "on"
                                            else [orientation_angle_thresholds[0]]
                                        )
                                        score_margins = (
                                            orientation_score_margins
                                            if orientation_mode == "on"
                                            else [orientation_score_margins[0]]
                                        )
                                        for neighbor_radius in radii:
                                            for angle_threshold in angle_thresholds:
                                                for score_margin in score_margins:
                                                    variant = {
                                                        "strategy": strategy,
                                                        "postprocess_mode": postprocess_mode,
                                                        "slot_conf": threshold,
                                                        "min_slot_score": min_slot_score,
                                                        "slot_nms_iou": slot_nms_iou,
                                                        "slot_center_nms_distance": center_distance,
                                                        "slot_angle_nms_threshold": angle_nms_threshold,
                                                        "slot_centerline_overlap_threshold": centerline_threshold,
                                                        "orientation_filter": orientation_mode == "on",
                                                        "orientation_neighbor_radius": neighbor_radius,
                                                        "orientation_angle_threshold": angle_threshold,
                                                        "orientation_min_neighbors": orientation_min_neighbors,
                                                        "orientation_score_margin": score_margin,
                                                    }
                                                    variant["key"] = variant_key(variant)
                                                    variants.append(variant)
    return variants


def apply_variant(
    detector: ParkingSlotDetector,
    variant: dict[str, Any],
    args: argparse.Namespace,
    strategy_max_point_degree: int,
) -> None:
    detector.slot_pairing_strategy = variant["strategy"]
    detector.slot_postprocess_mode = variant["postprocess_mode"]
    detector.max_point_degree = strategy_max_point_degree if variant["strategy"] == "relaxed" else 0
    detector.min_slot_score = variant["min_slot_score"]
    detector.slot_nms_iou = variant["slot_nms_iou"]
    detector.slot_center_nms_distance = variant["slot_center_nms_distance"]
    detector.slot_angle_nms_threshold = variant["slot_angle_nms_threshold"]
    detector.slot_centerline_overlap_threshold = variant["slot_centerline_overlap_threshold"]
    detector.geometry_filter_enabled = args.enable_geometry_filter
    detector.min_slot_area_ratio = args.min_slot_area_ratio
    detector.max_slot_area_ratio = args.max_slot_area_ratio
    detector.max_slot_aspect_ratio = args.max_slot_aspect_ratio
    detector.max_out_of_frame_ratio = args.max_out_of_frame_ratio
    detector.orientation_filter_enabled = variant["orientation_filter"]
    detector.orientation_neighbor_radius = variant["orientation_neighbor_radius"]
    detector.orientation_min_neighbors = variant["orientation_min_neighbors"]
    detector.orientation_angle_threshold = variant["orientation_angle_threshold"]
    detector.orientation_score_margin = variant["orientation_score_margin"]


def variant_key(variant: dict[str, Any]) -> str:
    orientation = "orient_on" if variant["orientation_filter"] else "orient_off"
    return (
        f"{variant['strategy']}_{variant['postprocess_mode']}_conf_{variant['slot_conf']:.3f}"
        f"_score_{variant['min_slot_score']:.3f}_nms_{variant['slot_nms_iou']:.2f}"
        f"_cnms_{variant['slot_center_nms_distance']:.0f}_anms_{variant['slot_angle_nms_threshold']:.0f}"
        f"_clo_{variant['slot_centerline_overlap_threshold']:.2f}"
        f"_{orientation}_r_{variant['orientation_neighbor_radius']:.0f}_a_{variant['orientation_angle_threshold']:.0f}"
        f"_m_{variant['orientation_score_margin']:.2f}"
    )


def first_row(rows: list[dict[str, Any]], min_precision: float) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["precision"] >= min_precision]
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row["recall"], row["f1"], row["precision"]), reverse=True)
    return candidates[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
