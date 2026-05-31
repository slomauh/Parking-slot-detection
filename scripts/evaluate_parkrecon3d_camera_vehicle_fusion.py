from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_parkrecon3d_bev import load_parkrecon3d_slots
from src.detection.parking_slot_detector import ParkingSlotDetector
from src.detection.schemas import Detection, ParkingSlot
from src.detection.vehicle_detector import VehicleDetector
from src.occupancy.camera_vehicle_fusion import (
    fuse_classifier_and_camera_vehicle,
    match_projected_vehicle_points_to_slots,
)
from src.occupancy.classifier import EfficientNetOccupancyClassifier
from src.utils.config import load_config


CAMERA_DIRS = {
    0: "Camera0",
    1: "Camera1",
    2: "Camera2",
    3: "Camera3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project ParkRecon3D camera vehicle detections to BEV slots for occupancy QA"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-root", default="/home/slomauh/Documents/parkrecon3d_dataset/data1")
    parser.add_argument("--stitch-json", default="external/parkrecon3d_calibration/stitch.json")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_camera_vehicle_projection_smoke50")
    parser.add_argument("--cameras", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--timestamps", nargs="+", help="Optional exact frame stems to process")
    parser.add_argument(
        "--timestamps-from-label-dir",
        help="Optional label directory whose filenames define the timestamp split to process.",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--sample-strategy", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-camera-detections",
        type=int,
        default=0,
        help="Accept only frames where camera YOLO finds at least this many vehicles across selected cameras.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional cap on candidate frames scanned before applying min-camera-detections. 0 means no cap.",
    )
    parser.add_argument("--model-path", default="models/vehicle/yolo11n.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--classes", nargs="+", default=["car", "truck", "motorcycle"])
    parser.add_argument(
        "--near-vehicles-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter out far camera vehicles before projection, using bbox position and size.",
    )
    parser.add_argument(
        "--near-filter-mode",
        choices=("all", "any"),
        default="all",
        help="With --near-vehicles-only, require all near thresholds or any of them.",
    )
    parser.add_argument(
        "--max-vehicles-per-camera",
        type=int,
        default=0,
        help="Keep only top-K nearest-looking vehicle detections per camera. 0 disables this cap.",
    )
    parser.add_argument(
        "--near-min-bottom-y-ratio",
        type=float,
        default=0.35,
        help="Keep vehicles whose bbox bottom is at least this image-height ratio.",
    )
    parser.add_argument(
        "--near-min-height-ratio",
        type=float,
        default=0.08,
        help="Keep vehicles whose bbox height is at least this image-height ratio.",
    )
    parser.add_argument(
        "--near-min-area-ratio",
        type=float,
        default=0.010,
        help="Keep vehicles whose bbox area is at least this image-area ratio.",
    )
    parser.add_argument(
        "--near-max-height-ratio",
        type=float,
        default=1.0,
        help="Drop implausibly large vehicle boxes whose bbox height exceeds this image-height ratio.",
    )
    parser.add_argument(
        "--near-max-area-ratio",
        type=float,
        default=1.0,
        help="Drop implausibly large vehicle boxes whose bbox area exceeds this image-area ratio.",
    )
    parser.add_argument(
        "--near-min-score",
        type=float,
        default=0.70,
        help="Keep vehicles whose combined near score passes this threshold.",
    )
    parser.add_argument("--match-distance-px", type=float, default=75.0)
    parser.add_argument(
        "--max-projected-distance-m",
        type=float,
        default=5.0,
        help="Drop camera detections whose projected ground points are farther than this from the BEV origin. 0 disables.",
    )
    parser.add_argument(
        "--draw-rejected-vehicles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw filtered-out far vehicles in camera previews for debugging.",
    )
    parser.add_argument(
        "--min-projected-points-per-evidence",
        type=int,
        default=1,
        help="Require this many projected bbox-bottom samples from one detection to match the same slot.",
    )
    parser.add_argument(
        "--min-camera-evidence-score",
        type=float,
        default=0.05,
        help="Drop weak camera->slot matches after distance-weighted confidence scoring.",
    )
    parser.add_argument(
        "--min-camera-distance-quality",
        type=float,
        default=0.30,
        help="Drop camera->slot matches whose projected point is too close to the match-radius edge.",
    )
    parser.add_argument(
        "--require-inside-slot-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require projected vehicle evidence to land inside a detected slot polygon.",
    )
    parser.add_argument(
        "--slot-source",
        choices=("detector", "labels"),
        default="detector",
        help="Use detected BEV slots for the real pipeline, or GT labels for projection calibration QA.",
    )
    parser.add_argument("--slot-backend", choices=("crpsd", "yolo_obb"), default="yolo_obb")
    parser.add_argument("--slot-model-path", default="models/slot_detector/best_yolo_parkrecon.pt")
    parser.add_argument("--slot-external-repo-path", default="external/CRPS-D")
    parser.add_argument("--slot-conf", type=float, default=0.55)
    parser.add_argument("--detector-input-size", type=int, default=1024)
    parser.add_argument("--occupancy-model-path", default="models/occupancy/efficientnet_b0_crpsd.pt")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--occupancy-threshold", type=float, default=0.50)
    parser.add_argument("--preview-limit", type=int, default=50)
    parser.add_argument("--contact-sheet-limit", type=int, default=60)
    parser.add_argument("--bev-cx", type=float, default=586.5)
    parser.add_argument("--bev-cy", type=float, default=725.5)
    parser.add_argument("--bev-meter-per-pixel", type=float, default=0.0085)
    parser.add_argument("--swap-xy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--camera-origin-sign",
        choices=("negative_t", "positive_t"),
        default="negative_t",
        help="Use -t or +t as camera origin when intersecting camera rays with the ground plane.",
    )
    parser.add_argument(
        "--rotation-mode",
        choices=("r", "rt"),
        default="rt",
        help="Use R or R.T to map camera rays to the BEV/vehicle coordinate frame.",
    )
    args = parser.parse_args()
    args._cli_options = cli_option_names(sys.argv[1:])
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_config_defaults(args, config)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "preview"
    camera_preview_dir = output_dir / "camera_preview"
    bev_preview_dir = output_dir / "bev_preview"
    for directory in (preview_dir, camera_preview_dir, bev_preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    camera_params = load_camera_params(Path(args.stitch_json))
    label_paths = select_label_paths(dataset_root, args)
    if not label_paths:
        raise FileNotFoundError(f"No ParkRecon3D labels selected under {dataset_root}")

    detector = VehicleDetector(
        {
            "backend": "yolo",
            "enabled": True,
            "model_path": args.model_path,
            "device": args.device,
            "classes": args.classes,
            "conf_threshold": args.conf,
            "imgsz": args.imgsz,
            "min_bottom_y_ratio": args.near_min_bottom_y_ratio if args.near_vehicles_only else 0.0,
            "min_bbox_height_ratio": args.near_min_height_ratio if args.near_vehicles_only else 0.0,
            "min_bbox_area_ratio": args.near_min_area_ratio if args.near_vehicles_only else 0.0,
            "max_bbox_height_ratio": args.near_max_height_ratio if args.near_vehicles_only else 1.0,
            "max_bbox_area_ratio": args.near_max_area_ratio if args.near_vehicles_only else 1.0,
            "min_near_score": args.near_min_score if args.near_vehicles_only else 0.0,
            "max_detections": args.max_vehicles_per_camera,
            "near_filter_classes": args.classes,
        }
    )
    classifier = None
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
    if not args.skip_classifier:
        classifier = EfficientNetOccupancyClassifier(
            {
                "model_path": args.occupancy_model_path,
                "device": args.device,
                "occupied_threshold": args.occupancy_threshold,
                "use_pretrained_backbone": False,
            }
        )

    records = []
    counts: Counter[str] = Counter()
    camera_detection_counts: Counter[str] = Counter()

    for frame_idx, label_path in enumerate(tqdm(label_paths, desc="Camera vehicle fusion QA")):
        if args.max_candidates and counts["candidate_images"] >= args.max_candidates:
            break
        if args.limit and counts["images"] >= args.limit:
            break
        counts["candidate_images"] += 1
        timestamp = label_path.stem
        bev_path = dataset_root / "BEV" / "Data" / "Image" / f"{timestamp}.jpg"
        bev_frame = cv2.imread(str(bev_path))
        if bev_frame is None:
            records.append({"timestamp": timestamp, "error": f"Could not read BEV image: {bev_path}"})
            counts["missing_bev"] += 1
            continue

        if slot_detector is not None:
            slots = slot_detector.detect(bev_frame)
        else:
            slots = load_parkrecon3d_slots(label_path)
        camera_records = []
        all_projected_points = []
        frame_camera_detection_counts: Counter[str] = Counter()

        for camera_id in args.cameras:
            camera_name = CAMERA_DIRS.get(camera_id, f"Camera{camera_id}")
            image_path = dataset_root / camera_name / "Data" / "Image" / f"{timestamp}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                camera_records.append({"camera": camera_name, "image": str(image_path), "error": "missing image"})
                counts[f"{camera_name}_missing"] += 1
                continue
            if camera_id not in camera_params:
                camera_records.append({"camera": camera_name, "image": str(image_path), "error": "missing calibration"})
                counts[f"{camera_name}_missing_calibration"] += 1
                continue

            raw_detections = detector.detect(image)
            detections, rejected_detections = filter_near_vehicle_detections(image, raw_detections, args)
            counts["raw_camera_detections"] += len(raw_detections)
            accepted_detections = []
            projected_detections = []
            for detection in detections:
                projected_points = project_detection_to_bev(detection, camera_params[camera_id], args)
                valid_points = filter_near_projected_points(projected_points, args)
                if not valid_points:
                    rejected_detections.append(detection)
                    continue
                det_idx = len(accepted_detections)
                accepted_detections.append(detection)
                bbox_features = vehicle_near_features(detection, image.shape[1], image.shape[0])
                if valid_points:
                    counts["projected_detections"] += 1
                all_projected_points.extend(
                    {
                        "camera": camera_name,
                        "detection_idx": det_idx,
                        "point": point,
                        "confidence": detection.confidence,
                        "class_name": detection.class_name,
                        "bbox": detection.bbox,
                        "bbox_features": bbox_features,
                    }
                    for point in valid_points
                )
                projected_detections.append(
                    {
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "bbox": detection.bbox,
                        "bbox_features": bbox_features,
                        "projected_points": valid_points,
                    }
                )

            counts["filtered_far_camera_detections"] += len(rejected_detections)
            counts["filtered_far_projected_detections"] += len(detections) - len(accepted_detections)
            frame_camera_detection_counts[camera_name] += len(accepted_detections)

            if counts["images"] < args.preview_limit:
                camera_preview = draw_camera_preview(
                    image,
                    accepted_detections,
                    rejected_detections,
                    draw_rejected=args.draw_rejected_vehicles,
                )
                cv2.imwrite(
                    str(camera_preview_dir / f"{timestamp}_{camera_name}.jpg"),
                    camera_preview,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )

            camera_records.append(
                {
                    "camera": camera_name,
                    "image": str(image_path),
                    "detections": projected_detections,
                    "rejected_far_detections": [
                        {
                            "class_name": detection.class_name,
                            "confidence": detection.confidence,
                            "bbox": detection.bbox,
                        }
                        for detection in rejected_detections
                    ],
                }
            )

        frame_camera_detection_count = sum(len(record.get("detections", [])) for record in camera_records)
        if frame_camera_detection_count < args.min_camera_detections:
            counts["skipped_low_camera_detections"] += 1
            continue

        classifier_predictions = classifier.predict(bev_frame, slots) if classifier is not None else {}
        slot_evidence = match_projected_vehicle_points_to_slots(
            slots,
            all_projected_points,
            args.match_distance_px,
            min_points_per_detection=args.min_projected_points_per_evidence,
            min_evidence_score=args.min_camera_evidence_score,
            min_distance_quality=args.min_camera_distance_quality,
            require_inside_slot=args.require_inside_slot_match,
        )
        fused_slots = fuse_classifier_and_camera_vehicle(slots, classifier_predictions, slot_evidence)
        counts["images"] += 1
        counts["slots"] += len(slots)
        counts["camera_detections"] += sum(len(record.get("detections", [])) for record in camera_records)
        camera_detection_counts.update(frame_camera_detection_counts)
        counts["projected_points"] += len(all_projected_points)
        counts["slots_with_camera_evidence"] += len(slot_evidence)
        counts["fused_occupied_slots"] += sum(1 for slot in fused_slots if slot["fused_status"] == "occupied")

        if counts["images"] <= args.preview_limit:
            bev_preview = draw_bev_preview(bev_frame, slots, all_projected_points, slot_evidence, fused_slots)
            cv2.imwrite(
                str(bev_preview_dir / f"{timestamp}_bev.jpg"),
                bev_preview,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            contact = build_frame_contact_sheet(timestamp, bev_preview, camera_preview_dir, args.cameras)
            cv2.imwrite(str(preview_dir / f"{timestamp}_contact.jpg"), contact, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        records.append(
            {
                "timestamp": timestamp,
                "bev_image": str(bev_path),
                "slot_source": args.slot_source,
                "camera_records": camera_records,
                "projected_points": all_projected_points,
                "slot_evidence": list(slot_evidence.values()),
                "slots": fused_slots,
            }
        )

    summary = {
        "config": args.config,
        "dataset_root": str(dataset_root),
        "stitch_json": args.stitch_json,
        "model_path": args.model_path,
        "device": args.device,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "limit": args.limit,
        "sample_strategy": args.sample_strategy,
        "seed": args.seed,
        "min_camera_detections": args.min_camera_detections,
        "max_candidates": args.max_candidates,
        "classes": args.classes,
        "slot_source": args.slot_source,
        "slot_detector": {
            "backend": args.slot_backend,
            "model_path": args.slot_model_path,
            "external_repo_path": args.slot_external_repo_path,
            "conf": args.slot_conf,
            "imgsz": args.detector_input_size,
        },
        "near_vehicle_filter": {
            "enabled": args.near_vehicles_only,
            "near_min_bottom_y_ratio": args.near_min_bottom_y_ratio,
            "near_min_height_ratio": args.near_min_height_ratio,
            "near_min_area_ratio": args.near_min_area_ratio,
            "near_max_height_ratio": args.near_max_height_ratio,
            "near_max_area_ratio": args.near_max_area_ratio,
            "near_min_score": args.near_min_score,
            "near_filter_mode": args.near_filter_mode,
            "max_vehicles_per_camera": args.max_vehicles_per_camera,
        },
        "cameras": [CAMERA_DIRS.get(camera_id, f"Camera{camera_id}") for camera_id in args.cameras],
        "match_distance_px": args.match_distance_px,
        "max_projected_distance_m": args.max_projected_distance_m,
        "min_projected_points_per_evidence": args.min_projected_points_per_evidence,
        "min_camera_evidence_score": args.min_camera_evidence_score,
        "min_camera_distance_quality": args.min_camera_distance_quality,
        "require_inside_slot_match": args.require_inside_slot_match,
        "draw_rejected_vehicles": args.draw_rejected_vehicles,
        "bev_projection": {
            "bev_cx": args.bev_cx,
            "bev_cy": args.bev_cy,
            "bev_meter_per_pixel": args.bev_meter_per_pixel,
            "swap_xy": args.swap_xy,
            "flip_x": args.flip_x,
            "flip_y": args.flip_y,
            "camera_origin_sign": args.camera_origin_sign,
            "rotation_mode": args.rotation_mode,
        },
        "counts": dict(counts),
        "camera_detection_counts": dict(camera_detection_counts),
        "preview_dir": str(preview_dir),
        "camera_preview_dir": str(camera_preview_dir),
        "bev_preview_dir": str(bev_preview_dir),
        "contact_sheet": str(output_dir / "contact_sheet.jpg"),
        "note": "No ParkRecon3D vehicle/occupancy GT is available here; camera detections are visual QA evidence, while fused_status follows the EfficientNet classifier by default.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    make_contact_sheet(preview_dir, output_dir / "contact_sheet.jpg", args.contact_sheet_limit, (360, 520))
    print(json.dumps(summary, indent=2))


def cli_option_names(argv: list[str]) -> set[str]:
    names = set()
    for item in argv:
        if not item.startswith("--"):
            continue
        option = item.split("=", 1)[0][2:]
        names.add(option)
    return names


def apply_config_defaults(args: argparse.Namespace, config: dict[str, Any]) -> None:
    supplied = set(getattr(args, "_cli_options", set()))
    occupancy_config = config.get("occupancy", {})
    fusion_config = occupancy_config.get("camera_vehicle_fusion", {})
    vehicle_config = config.get("vehicle_detector", {})
    slot_config = config.get("parking_slot_detector", {})
    classifier_config = occupancy_config.get("classifier", {})

    apply_if_not_supplied(args, supplied, "model_path", fusion_config.get("vehicle_model_path", vehicle_config.get("model_path")))
    apply_if_not_supplied(args, supplied, "conf", fusion_config.get("vehicle_conf", vehicle_config.get("conf_threshold")))
    apply_if_not_supplied(args, supplied, "imgsz", fusion_config.get("vehicle_imgsz", vehicle_config.get("imgsz")))
    apply_if_not_supplied(args, supplied, "classes", fusion_config.get("vehicle_classes", occupancy_config.get("vehicle_classes")))
    apply_if_not_supplied(args, supplied, "slot_backend", slot_config.get("backend"))
    apply_if_not_supplied(args, supplied, "slot_model_path", slot_config.get("model_path"))
    apply_if_not_supplied(args, supplied, "slot_external_repo_path", slot_config.get("external_repo_path"))
    apply_if_not_supplied(args, supplied, "slot_conf", slot_config.get("conf_threshold"))
    apply_if_not_supplied(args, supplied, "detector_input_size", slot_config.get("imgsz"))
    apply_if_not_supplied(args, supplied, "occupancy_model_path", classifier_config.get("model_path"))
    apply_if_not_supplied(args, supplied, "occupancy_threshold", classifier_config.get("occupied_threshold"))

    for name in (
        "match_distance_px",
        "max_projected_distance_m",
        "min_projected_points_per_evidence",
        "min_camera_evidence_score",
        "min_camera_distance_quality",
        "require_inside_slot_match",
        "max_vehicles_per_camera",
        "near_vehicles_only",
        "near_filter_mode",
        "near_min_bottom_y_ratio",
        "near_min_height_ratio",
        "near_min_area_ratio",
        "near_max_height_ratio",
        "near_max_area_ratio",
        "near_min_score",
        "bev_cx",
        "bev_cy",
        "bev_meter_per_pixel",
        "swap_xy",
        "flip_x",
        "flip_y",
        "camera_origin_sign",
        "rotation_mode",
    ):
        apply_if_not_supplied(args, supplied, name, fusion_config.get(name))


def apply_if_not_supplied(args: argparse.Namespace, supplied: set[str], name: str, value: Any) -> None:
    if value is None:
        return
    flag_name = name.replace("_", "-")
    if flag_name in supplied or f"no-{flag_name}" in supplied:
        return
    setattr(args, name, value)


def select_label_paths(dataset_root: Path, args: argparse.Namespace) -> list[Path]:
    label_dir = dataset_root / "BEV" / "Data" / "label"
    label_paths = sorted(label_dir.glob("*.json"), key=lambda path: int(path.stem))
    if args.timestamps:
        timestamps = set(args.timestamps)
        label_paths = [path for path in label_paths if path.stem in timestamps]
    elif args.timestamps_from_label_dir:
        split_label_dir = Path(args.timestamps_from_label_dir)
        timestamps = {path.stem for path in split_label_dir.glob("*.json")}
        label_paths = [path for path in label_paths if path.stem in timestamps]

    if args.sample_strategy == "random":
        rng = random.Random(args.seed)
        rng.shuffle(label_paths)

    if args.limit and args.min_camera_detections <= 0:
        return label_paths[: args.limit]
    return label_paths


def filter_near_vehicle_detections(
    image: np.ndarray,
    detections: list[Detection],
    args: argparse.Namespace,
) -> tuple[list[Detection], list[Detection]]:
    height, width = image.shape[:2]
    scored_detections = [
        (
            detection,
            vehicle_near_features(detection, width, height),
        )
        for detection in detections
    ]

    kept = []
    rejected = []
    for detection, features in scored_detections:
        checks = (
            features["bottom_y_ratio"] >= args.near_min_bottom_y_ratio,
            features["height_ratio"] >= args.near_min_height_ratio,
            features["area_ratio"] >= args.near_min_area_ratio,
            features["height_ratio"] <= args.near_max_height_ratio,
            features["area_ratio"] <= args.near_max_area_ratio,
            features["near_score"] >= args.near_min_score,
        )
        is_near = not args.near_vehicles_only or (all(checks) if args.near_filter_mode == "all" else any(checks))
        if is_near:
            kept.append(detection)
        else:
            rejected.append(detection)

    if args.max_vehicles_per_camera > 0 and len(kept) > args.max_vehicles_per_camera:
        scores = {id(detection): vehicle_near_features(detection, width, height)["near_score"] for detection in kept}
        ranked = sorted(kept, key=lambda detection: scores[id(detection)], reverse=True)
        kept = ranked[: args.max_vehicles_per_camera]
        rejected.extend(ranked[args.max_vehicles_per_camera :])
    return kept, rejected


def vehicle_near_features(detection: Detection, width: int, height: int) -> dict[str, float]:
    x1, y1, x2, y2 = detection.bbox
    bbox_width = max(0.0, x2 - x1)
    bbox_height = max(0.0, y2 - y1)
    width_ratio = bbox_width / max(1, width)
    bottom_y_ratio = y2 / max(1, height)
    height_ratio = bbox_height / max(1, height)
    area_ratio = (bbox_width * bbox_height) / max(1, width * height)
    # Larger/lower boxes are better proxies for nearby vehicles in fisheye camera views.
    near_score = bottom_y_ratio + 2.0 * height_ratio + 8.0 * area_ratio
    return {
        "width_ratio": float(width_ratio),
        "bottom_y_ratio": float(bottom_y_ratio),
        "height_ratio": float(height_ratio),
        "area_ratio": float(area_ratio),
        "near_score": float(near_score),
    }


def load_camera_params(path: Path) -> dict[int, dict[str, np.ndarray]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    params = {}
    for row in data["camParams"]:
        camera_id = int(row["camId"])
        params[camera_id] = {
            "K": np.array(row["oriIntrinsics"], dtype=np.float64).reshape(3, 3),
            "D": np.array(row["Distortion"], dtype=np.float64).reshape(4, 1),
            "R": np.array(row["Extrinsics"], dtype=np.float64).reshape(3, 4)[:, :3],
            "t": np.array(row["Extrinsics"], dtype=np.float64).reshape(3, 4)[:, 3],
        }
    return params


def project_detection_to_bev(detection: Detection, camera_param: dict[str, np.ndarray], args: argparse.Namespace):
    x1, y1, x2, y2 = detection.bbox
    sample_pixels = [
        ((x1 + x2) / 2.0, y2),
        (x1 + (x2 - x1) * 0.25, y2),
        (x1 + (x2 - x1) * 0.75, y2),
        ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.85),
    ]
    return [camera_pixel_to_bev(pixel, camera_param, args) for pixel in sample_pixels]


def filter_near_projected_points(
    points: list[tuple[float, float] | None],
    args: argparse.Namespace,
) -> list[tuple[float, float]]:
    valid_points = [point for point in points if point is not None]
    max_distance = float(getattr(args, "max_projected_distance_m", 0.0))
    if max_distance <= 0:
        return valid_points
    return [point for point in valid_points if projected_distance_m(point, args) <= max_distance]


def projected_distance_m(point: tuple[float, float], args: argparse.Namespace) -> float:
    dx = (float(point[0]) - float(args.bev_cx)) * float(args.bev_meter_per_pixel)
    dy = (float(args.bev_cy) - float(point[1])) * float(args.bev_meter_per_pixel)
    return float(np.hypot(dx, dy))


def camera_pixel_to_bev(
    pixel: tuple[float, float],
    camera_param: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[float, float] | None:
    point = np.array([[[pixel[0], pixel[1]]]], dtype=np.float64)
    undistorted = cv2.fisheye.undistortPoints(point, camera_param["K"], camera_param["D"])
    x_norm, y_norm = undistorted.reshape(2)
    ray_camera = np.array([x_norm, y_norm, 1.0], dtype=np.float64)

    rotation = camera_param["R"] if args.rotation_mode == "r" else camera_param["R"].T
    ray_vehicle = rotation @ ray_camera
    origin_vehicle = -camera_param["t"] if args.camera_origin_sign == "negative_t" else camera_param["t"]
    if abs(float(ray_vehicle[2])) < 1e-9:
        return None
    scale = -float(origin_vehicle[2]) / float(ray_vehicle[2])
    if scale <= 0:
        return None
    vehicle_point = origin_vehicle + scale * ray_vehicle
    return vehicle_to_bev_pixel(float(vehicle_point[0]), float(vehicle_point[1]), args)


def vehicle_to_bev_pixel(x: float, y: float, args: argparse.Namespace) -> tuple[float, float]:
    if args.swap_xy:
        x, y = y, x
    if args.flip_x:
        x = -x
    if args.flip_y:
        y = -y
    u = x / args.bev_meter_per_pixel + args.bev_cx
    v = args.bev_cy - y / args.bev_meter_per_pixel
    return (float(u), float(v))


def draw_camera_preview(
    image: np.ndarray,
    detections: list[Detection],
    rejected_detections: list[Detection] | None = None,
    draw_rejected: bool = False,
) -> np.ndarray:
    output = image.copy()
    if draw_rejected:
        for detection in rejected_detections or []:
            x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
            label = f"ignored {detection.class_name} {detection.confidence:.2f}"
            cv2.rectangle(output, (x1, y1), (x2, y2), (120, 120, 120), 1)
            cv2.putText(output, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)
    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox]
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 190, 255), 2)
        cv2.circle(output, (int((x1 + x2) / 2), y2), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(output, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 190, 255), 2)
    return output


def draw_bev_preview(
    bev_frame: np.ndarray,
    slots: list[ParkingSlot],
    projected_points: list[dict[str, Any]],
    slot_evidence: dict[int, dict[str, Any]],
    fused_slots: list[dict[str, Any]],
) -> np.ndarray:
    output = bev_frame.copy()
    fused_by_slot = {slot["slot_id"]: slot for slot in fused_slots}
    for slot in slots:
        fused = fused_by_slot.get(slot.slot_id, {})
        if slot.slot_id in slot_evidence:
            color = (0, 0, 255)
        elif fused.get("classifier_status") == "occupied":
            color = (0, 140, 255)
        elif fused.get("classifier_status") == "free":
            color = (0, 180, 0)
        else:
            color = (160, 160, 160)
        label = f"S{slot.slot_id}:{fused.get('fused_status', 'unknown')}"
        draw_polygon(output, slot.points, color, label)

    for projected in projected_points:
        point = projected["point"]
        x, y = int(round(point[0])), int(round(point[1]))
        color = camera_color(str(projected["camera"]))
        cv2.circle(output, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            f"{projected['camera']} {projected['confidence']:.2f}",
            (x + 6, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


def draw_polygon(image: np.ndarray, points, color, label: str) -> None:
    polygon = np.array(points, dtype=np.int32)
    cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2)
    x, y = polygon[0]
    cv2.putText(image, label, (int(x), max(16, int(y) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)


def camera_color(camera_name: str) -> tuple[int, int, int]:
    if camera_name == "Camera0":
        return (255, 80, 80)
    if camera_name == "Camera1":
        return (80, 255, 255)
    if camera_name == "Camera2":
        return (255, 80, 255)
    return (255, 255, 255)


def build_frame_contact_sheet(timestamp: str, bev_preview: np.ndarray, camera_preview_dir: Path, camera_ids: list[int]) -> np.ndarray:
    tile_width, tile_height = 360, 260
    tiles = [resize_with_title(bev_preview, (tile_width, tile_height), f"{timestamp} BEV")]
    for camera_id in camera_ids:
        camera_name = CAMERA_DIRS.get(camera_id, f"Camera{camera_id}")
        image_path = camera_preview_dir / f"{timestamp}_{camera_name}.jpg"
        image = cv2.imread(str(image_path))
        if image is None:
            image = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
        tiles.append(resize_with_title(image, (tile_width, tile_height), camera_name))
    while len(tiles) < 4:
        tiles.append(np.zeros((tile_height, tile_width, 3), dtype=np.uint8))
    return cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:4])])


def resize_with_title(image: np.ndarray, size: tuple[int, int], title: str) -> np.ndarray:
    tile = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(tile, title, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def make_contact_sheet(input_dir: Path, output_path: Path, limit: int, tile_size: tuple[int, int]) -> None:
    image_paths = sorted(input_dir.glob("*.jpg"))[:limit]
    if not image_paths:
        return
    columns = 3
    tiles = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        tiles.append(cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA))
    if not tiles:
        return
    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank.copy())
    rows = [cv2.hconcat(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    cv2.imwrite(str(output_path), cv2.vconcat(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 95])


if __name__ == "__main__":
    main()
