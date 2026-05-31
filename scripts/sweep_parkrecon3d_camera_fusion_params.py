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

from scripts.evaluate_parkrecon3d_bev import load_parkrecon3d_slots
from scripts.evaluate_parkrecon3d_camera_vehicle_fusion import (  # noqa: E402
    CAMERA_DIRS,
    filter_near_projected_points,
    load_camera_params,
    project_detection_to_bev,
    projected_distance_m,
    select_label_paths,
)
from src.detection.parking_slot_detector import ParkingSlotDetector  # noqa: E402
from src.detection.vehicle_detector import VehicleDetector  # noqa: E402
from src.occupancy.camera_vehicle_fusion import match_projected_vehicle_points_to_slots  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep ParkRecon3D camera-vehicle -> BEV-slot fusion thresholds")
    parser.add_argument("--dataset-root", default="/home/slomauh/Documents/parkrecon3d_dataset/data3")
    parser.add_argument("--stitch-json", default="external/parkrecon3d_calibration/stitch.json")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_camera_fusion_param_sweep")
    parser.add_argument("--timestamps-from-label-dir", default="outputs/parkrecon3d_bev_crpsd_format/raw/test/slot_label")
    parser.add_argument("--timestamps", nargs="+")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--sample-strategy", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-camera-detections", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=0)
    parser.add_argument("--cameras", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vehicle-model-path", default="models/vehicle/yolo11n.pt")
    parser.add_argument("--vehicle-conf", type=float, default=0.35)
    parser.add_argument("--vehicle-imgsz", type=int, default=960)
    parser.add_argument("--vehicle-classes", nargs="+", default=["car", "truck", "bus", "motorcycle"])
    parser.add_argument("--near-min-bottom-y-ratio", type=float, default=0.35)
    parser.add_argument("--near-min-height-ratio", type=float, default=0.08)
    parser.add_argument("--near-min-area-ratio", type=float, default=0.010)
    parser.add_argument("--near-min-score", type=float, default=0.70)
    parser.add_argument("--max-vehicles-per-camera", type=int, default=1)
    parser.add_argument("--slot-source", choices=("detector", "labels"), default="detector")
    parser.add_argument("--slot-backend", choices=("crpsd", "yolo_obb"), default="yolo_obb")
    parser.add_argument("--slot-model-path", default="models/slot_detector/best_yolo_parkrecon.pt")
    parser.add_argument("--slot-external-repo-path", default="external/CRPS-D")
    parser.add_argument("--slot-conf", type=float, default=0.55)
    parser.add_argument("--detector-input-size", type=int, default=1024)
    parser.add_argument("--match-distances-px", nargs="+", type=float, default=[10.0, 20.0, 35.0, 50.0])
    parser.add_argument("--max-projected-distances-m", nargs="+", type=float, default=[3.0, 5.0, 7.0, 0.0])
    parser.add_argument("--min-projected-points", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--min-camera-evidence-scores", nargs="+", type=float, default=[0.0, 0.20, 0.35])
    parser.add_argument("--min-camera-distance-qualities", nargs="+", type=float, default=[0.0, 0.25, 0.30])
    parser.add_argument("--conservative-max-match-distance-px", type=float, default=75.0)
    parser.add_argument("--bev-cx", type=float, default=676.5)
    parser.add_argument("--bev-cy", type=float, default=815.5)
    parser.add_argument("--bev-meter-per-pixel", type=float, default=0.0105)
    parser.add_argument("--swap-xy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--camera-origin-sign", choices=("negative_t", "positive_t"), default="negative_t")
    parser.add_argument("--rotation-mode", choices=("r", "rt"), default="r")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = collect_frame_data(args)
    rows = run_sweep(frames, args)
    rows.sort(key=lambda row: (-row["slots_with_camera_evidence"], row["projected_points"], row["max_projected_distance_m"]))

    write_csv(output_dir / "sweep_results.csv", rows)
    summary = {
        "output_dir": str(output_dir),
        "frames": len(frames),
        "base_counts": summarize_frames(frames),
        "best_by_evidence": rows[0] if rows else None,
        "recommended_conservative": choose_conservative(rows, args.conservative_max_match_distance_px),
        "rows": rows[:20],
        "note": "No occupancy GT is used. Pick thresholds by visual QA: enough evidence, but few projected points.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def collect_frame_data(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset_root = Path(args.dataset_root)
    camera_params = load_camera_params(Path(args.stitch_json))
    label_paths = select_label_paths(dataset_root, args)
    if not label_paths:
        raise FileNotFoundError(f"No labels selected under {dataset_root}")

    vehicle_detector = VehicleDetector(
        {
            "backend": "yolo",
            "enabled": True,
            "model_path": args.vehicle_model_path,
            "device": args.device,
            "classes": args.vehicle_classes,
            "conf_threshold": args.vehicle_conf,
            "imgsz": args.vehicle_imgsz,
            "min_bottom_y_ratio": args.near_min_bottom_y_ratio,
            "min_bbox_height_ratio": args.near_min_height_ratio,
            "min_bbox_area_ratio": args.near_min_area_ratio,
            "min_near_score": args.near_min_score,
            "max_detections": args.max_vehicles_per_camera,
            "near_filter_classes": args.vehicle_classes,
        }
    )
    slot_detector = None
    if args.slot_source == "detector":
        slot_detector = ParkingSlotDetector(
            {
                "backend": args.slot_backend,
                "model_path": args.slot_model_path,
                "external_repo_path": args.slot_external_repo_path,
                "device": args.device,
                "conf_threshold": args.slot_conf,
                "imgsz": args.detector_input_size,
            }
        )

    frames = []
    for label_path in tqdm(label_paths, desc="Collect camera fusion candidates"):
        timestamp = label_path.stem
        bev_path = dataset_root / "BEV" / "Data" / "Image" / f"{timestamp}.jpg"
        bev_frame = cv2.imread(str(bev_path))
        if bev_frame is None:
            continue
        slots = slot_detector.detect(bev_frame) if slot_detector is not None else load_parkrecon3d_slots(label_path)
        projected_points = []
        accepted_detections = 0
        for camera_id in args.cameras:
            image_path = dataset_root / CAMERA_DIRS.get(camera_id, f"Camera{camera_id}") / "Data" / "Image" / f"{timestamp}.jpg"
            image = cv2.imread(str(image_path))
            if image is None or camera_id not in camera_params:
                continue
            for detection_idx, detection in enumerate(vehicle_detector.detect(image)):
                accepted_detections += 1
                points = [point for point in project_detection_to_bev(detection, camera_params[camera_id], args) if point]
                for point in points:
                    projected_points.append(
                        {
                            "camera": CAMERA_DIRS.get(camera_id, f"Camera{camera_id}"),
                            "detection_idx": detection_idx,
                            "point": point,
                            "distance_m": projected_distance_m(point, args),
                            "confidence": detection.confidence,
                            "class_name": detection.class_name,
                        }
                    )
        frames.append(
            {
                "timestamp": timestamp,
                "slots": slots,
                "projected_points": projected_points,
                "accepted_detections": accepted_detections,
            }
        )
    return frames


def run_sweep(frames: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for max_distance_m in args.max_projected_distances_m:
        for match_distance_px in args.match_distances_px:
            for min_points in args.min_projected_points:
                for min_evidence_score in args.min_camera_evidence_scores:
                    for min_distance_quality in args.min_camera_distance_qualities:
                        counts: Counter[str] = Counter()
                        for frame in frames:
                            projected_points = [
                                point
                                for point in frame["projected_points"]
                                if max_distance_m <= 0 or float(point["distance_m"]) <= max_distance_m
                            ]
                            evidence = match_projected_vehicle_points_to_slots(
                                frame["slots"],
                                projected_points,
                                match_distance_px,
                                min_points_per_detection=min_points,
                                min_evidence_score=min_evidence_score,
                                min_distance_quality=min_distance_quality,
                            )
                            counts["frames"] += 1
                            counts["slots"] += len(frame["slots"])
                            counts["accepted_detections"] += int(frame["accepted_detections"])
                            counts["projected_points"] += len(projected_points)
                            counts["slots_with_camera_evidence"] += len(evidence)
                            if evidence:
                                counts["frames_with_camera_evidence"] += 1
                        rows.append(
                            {
                                "max_projected_distance_m": max_distance_m,
                                "match_distance_px": match_distance_px,
                                "min_projected_points": min_points,
                                "min_camera_evidence_score": min_evidence_score,
                                "min_camera_distance_quality": min_distance_quality,
                                "frames": counts["frames"],
                                "accepted_detections": counts["accepted_detections"],
                                "projected_points": counts["projected_points"],
                                "slots": counts["slots"],
                                "slots_with_camera_evidence": counts["slots_with_camera_evidence"],
                                "frames_with_camera_evidence": counts["frames_with_camera_evidence"],
                            }
                        )
    return rows


def summarize_frames(frames: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for frame in frames:
        counts["frames"] += 1
        counts["slots"] += len(frame["slots"])
        counts["accepted_detections"] += int(frame["accepted_detections"])
        counts["projected_points_unfiltered"] += len(frame["projected_points"])
    return dict(counts)


def choose_conservative(rows: list[dict[str, Any]], max_match_distance_px: float) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["slots_with_camera_evidence"] > 0]
    if not candidates:
        return None
    if max_match_distance_px > 0:
        capped_candidates = [row for row in candidates if float(row["match_distance_px"]) <= max_match_distance_px]
        if capped_candidates:
            candidates = capped_candidates
    best_evidence = max(row["slots_with_camera_evidence"] for row in candidates)
    quality_candidates = [
        row
        for row in candidates
        if float(row.get("min_camera_distance_quality", 0.0)) >= 0.25
        and row["slots_with_camera_evidence"] >= max(1, int(best_evidence * 0.5))
    ]
    if quality_candidates:
        candidates = quality_candidates
    candidates.sort(
        key=lambda row: (
            float(row.get("min_camera_distance_quality", 0.0)),
            row["slots_with_camera_evidence"] / max(1, row["projected_points"]),
            -row["projected_points"],
        ),
        reverse=True,
    )
    return candidates[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
