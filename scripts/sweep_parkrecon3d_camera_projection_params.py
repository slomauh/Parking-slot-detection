from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_parkrecon3d_bev import load_parkrecon3d_slots  # noqa: E402
from scripts.evaluate_parkrecon3d_camera_vehicle_fusion import (  # noqa: E402
    CAMERA_DIRS,
    camera_pixel_to_bev,
    filter_near_projected_points,
    load_camera_params,
    projected_distance_m,
    select_label_paths,
)
from src.detection.parking_slot_detector import ParkingSlotDetector  # noqa: E402
from src.detection.schemas import Detection  # noqa: E402
from src.detection.vehicle_detector import VehicleDetector  # noqa: E402
from src.occupancy.camera_vehicle_fusion import match_projected_vehicle_points_to_slots  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep ParkRecon3D camera->BEV projection calibration parameters")
    parser.add_argument("--dataset-root", default="/home/slomauh/Documents/parkrecon3d_dataset/data3")
    parser.add_argument("--stitch-json", default="external/parkrecon3d_calibration/stitch.json")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_camera_projection_param_sweep")
    parser.add_argument("--timestamps-from-label-dir", default="outputs/parkrecon3d_bev_crpsd_format/raw/test/slot_label")
    parser.add_argument("--timestamps", nargs="+")
    parser.add_argument("--limit", type=int, default=120)
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
    parser.add_argument("--match-distance-px", type=float, default=75.0)
    parser.add_argument("--max-projected-distance-m", type=float, default=5.0)
    parser.add_argument("--min-projected-points-per-evidence", type=int, default=1)
    parser.add_argument("--min-camera-evidence-score", type=float, default=0.05)
    parser.add_argument("--min-camera-distance-quality", type=float, default=0.0)
    parser.add_argument("--bev-cx-values", nargs="+", type=float, default=[616.5, 646.5, 676.5, 706.5, 736.5])
    parser.add_argument("--bev-cy-values", nargs="+", type=float, default=[755.5, 785.5, 815.5, 845.5, 875.5])
    parser.add_argument("--bev-meter-per-pixel-values", nargs="+", type=float, default=[0.0095, 0.0105, 0.0115])
    parser.add_argument("--rotation-modes", nargs="+", choices=("r", "rt"), default=["rt"])
    parser.add_argument("--camera-origin-signs", nargs="+", choices=("negative_t", "positive_t"), default=["negative_t"])
    parser.add_argument("--swap-xy-values", nargs="+", choices=("true", "false"), default=["true"])
    parser.add_argument("--flip-x-values", nargs="+", choices=("true", "false"), default=["false"])
    parser.add_argument("--flip-y-values", nargs="+", choices=("true", "false"), default=["false"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_params = load_camera_params(Path(args.stitch_json))
    frames = collect_frames(args, camera_params)
    rows = sweep_projection(frames, camera_params, args)
    rows.sort(key=lambda row: projection_rank(row), reverse=True)

    write_csv(output_dir / "sweep_results.csv", rows)
    summary = {
        "output_dir": str(output_dir),
        "frames": len(frames),
        "base_counts": summarize_frames(frames),
        "best_by_inside": rows[0] if rows else None,
        "rows": rows[:30],
        "note": "No vehicle occupancy GT is used. Prefer variants with more inside_slot_evidence and fewer only-nearby matches.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def collect_frames(args: argparse.Namespace, camera_params: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    dataset_root = Path(args.dataset_root)
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
    for label_path in tqdm(label_paths, desc="Collect camera detections"):
        timestamp = label_path.stem
        bev_path = dataset_root / "BEV" / "Data" / "Image" / f"{timestamp}.jpg"
        bev_frame = cv2.imread(str(bev_path))
        if bev_frame is None:
            continue

        slots = slot_detector.detect(bev_frame) if slot_detector is not None else load_parkrecon3d_slots(label_path)
        detections = []
        for camera_id in args.cameras:
            if camera_id not in camera_params:
                continue
            camera_name = CAMERA_DIRS.get(camera_id, f"Camera{camera_id}")
            image_path = dataset_root / camera_name / "Data" / "Image" / f"{timestamp}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            for detection_idx, detection in enumerate(vehicle_detector.detect(image)):
                detections.append(
                    {
                        "camera_id": camera_id,
                        "camera": camera_name,
                        "detection_idx": detection_idx,
                        "detection": detection,
                    }
                )
        frames.append({"timestamp": timestamp, "slots": slots, "detections": detections})
    return frames


def sweep_projection(
    frames: list[dict[str, Any]],
    camera_params: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows = []
    for rotation_mode in args.rotation_modes:
        for camera_origin_sign in args.camera_origin_signs:
            for swap_xy in bool_values(args.swap_xy_values):
                for flip_x in bool_values(args.flip_x_values):
                    for flip_y in bool_values(args.flip_y_values):
                        for bev_cx in args.bev_cx_values:
                            for bev_cy in args.bev_cy_values:
                                for meter_per_pixel in args.bev_meter_per_pixel_values:
                                    projection_args = projection_namespace(
                                        args,
                                        bev_cx,
                                        bev_cy,
                                        meter_per_pixel,
                                        swap_xy,
                                        flip_x,
                                        flip_y,
                                        camera_origin_sign,
                                        rotation_mode,
                                    )
                                    rows.append(evaluate_projection(frames, camera_params, projection_args))
    return rows


def evaluate_projection(
    frames: list[dict[str, Any]],
    camera_params: dict[int, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for frame in frames:
        projected_points = []
        for detection_row in frame["detections"]:
            detection: Detection = detection_row["detection"]
            camera_param = camera_params[int(detection_row["camera_id"])]
            points = project_detection_bottom_samples(detection, camera_param, args)
            valid_points = filter_near_projected_points(points, args)
            for point in valid_points:
                projected_points.append(
                    {
                        "camera": detection_row["camera"],
                        "detection_idx": detection_row["detection_idx"],
                        "point": point,
                        "distance_m": projected_distance_m(point, args),
                        "confidence": detection.confidence,
                        "class_name": detection.class_name,
                    }
                )

        evidence = match_projected_vehicle_points_to_slots(
            frame["slots"],
            projected_points,
            args.match_distance_px,
            min_points_per_detection=args.min_projected_points_per_evidence,
            min_evidence_score=args.min_camera_evidence_score,
            min_distance_quality=args.min_camera_distance_quality,
        )
        counts["frames"] += 1
        counts["slots"] += len(frame["slots"])
        counts["camera_detections"] += len(frame["detections"])
        counts["projected_points"] += len(projected_points)
        counts["slots_with_camera_evidence"] += len(evidence)
        inside = sum(1 for item in evidence.values() if item.get("inside_slot_polygon"))
        counts["inside_slot_evidence"] += inside
        counts["nearby_slot_evidence"] += len(evidence) - inside
        if evidence:
            counts["frames_with_camera_evidence"] += 1
        if inside:
            counts["frames_with_inside_evidence"] += 1

    return {
        "bev_cx": args.bev_cx,
        "bev_cy": args.bev_cy,
        "bev_meter_per_pixel": args.bev_meter_per_pixel,
        "swap_xy": args.swap_xy,
        "flip_x": args.flip_x,
        "flip_y": args.flip_y,
        "camera_origin_sign": args.camera_origin_sign,
        "rotation_mode": args.rotation_mode,
        "match_distance_px": args.match_distance_px,
        "max_projected_distance_m": args.max_projected_distance_m,
        "min_camera_evidence_score": args.min_camera_evidence_score,
        "min_camera_distance_quality": args.min_camera_distance_quality,
        **dict(counts),
    }


def project_detection_bottom_samples(
    detection: Detection,
    camera_param: dict[str, Any],
    args: argparse.Namespace,
) -> list[tuple[float, float] | None]:
    x1, y1, x2, y2 = detection.bbox
    sample_pixels = [
        ((x1 + x2) / 2.0, y2),
        (x1 + (x2 - x1) * 0.25, y2),
        (x1 + (x2 - x1) * 0.75, y2),
        ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.85),
    ]
    return [camera_pixel_to_bev(pixel, camera_param, args) for pixel in sample_pixels]


def projection_namespace(
    args: argparse.Namespace,
    bev_cx: float,
    bev_cy: float,
    meter_per_pixel: float,
    swap_xy: bool,
    flip_x: bool,
    flip_y: bool,
    camera_origin_sign: str,
    rotation_mode: str,
) -> argparse.Namespace:
    return SimpleNamespace(
        bev_cx=bev_cx,
        bev_cy=bev_cy,
        bev_meter_per_pixel=meter_per_pixel,
        swap_xy=swap_xy,
        flip_x=flip_x,
        flip_y=flip_y,
        camera_origin_sign=camera_origin_sign,
        rotation_mode=rotation_mode,
        max_projected_distance_m=args.max_projected_distance_m,
        match_distance_px=args.match_distance_px,
        min_projected_points_per_evidence=args.min_projected_points_per_evidence,
        min_camera_evidence_score=args.min_camera_evidence_score,
        min_camera_distance_quality=args.min_camera_distance_quality,
    )


def bool_values(values: list[str]) -> list[bool]:
    return [value.lower() == "true" for value in values]


def projection_rank(row: dict[str, Any]) -> tuple[float, float, float, float]:
    projected_points = max(1, int(row["projected_points"]))
    evidence = int(row["slots_with_camera_evidence"])
    inside = int(row["inside_slot_evidence"])
    nearby = int(row["nearby_slot_evidence"])
    return (
        inside,
        inside / max(1, evidence),
        evidence / projected_points,
        -nearby,
    )


def summarize_frames(frames: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for frame in frames:
        counts["frames"] += 1
        counts["slots"] += len(frame["slots"])
        counts["camera_detections"] += len(frame["detections"])
    return dict(counts)


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
