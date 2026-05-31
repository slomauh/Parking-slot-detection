from __future__ import annotations

import argparse
import csv
import json
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

from scripts.evaluate_parkrecon3d_camera_vehicle_fusion import (  # noqa: E402
    CAMERA_DIRS,
    build_frame_contact_sheet,
    draw_camera_preview,
    filter_near_projected_points,
    load_camera_params,
    project_detection_to_bev,
    filter_near_vehicle_detections,
    vehicle_near_features,
)
from scripts.run_parkrecon3d_bev_sequence import (  # noqa: E402
    SlotTrackManager,
    detect_slots,
    draw_frame_header,
    hold_low_confidence_decisions,
    make_video_writer,
    record_events,
    state_to_record,
)
from src.detection.parking_slot_detector import ParkingSlotDetector  # noqa: E402
from src.detection.vehicle_detector import VehicleDetector  # noqa: E402
from src.occupancy.camera_vehicle_fusion import match_projected_vehicle_points_to_slots  # noqa: E402
from src.occupancy.estimator import OccupancyEstimator  # noqa: E402
from src.occupancy.state_manager import TemporalStateManager  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.visualization.draw import Visualizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ParkRecon3D BEV + perimeter-camera vehicle fusion sequence")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-root", default="/home/slomauh/Documents/parkrecon3d_dataset/data3")
    parser.add_argument("--timestamps-from-label-dir", default="outputs/parkrecon3d_bev_crpsd_format/raw/test/slot_label")
    parser.add_argument("--timestamps", nargs="+", help="Optional exact timestamp stems to process in the given order")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_multicamera_fusion_sequence")
    parser.add_argument("--stitch-json", default="external/parkrecon3d_calibration/stitch.json")
    parser.add_argument("--cameras", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--slot-every-n", type=int, default=3)
    parser.add_argument("--vehicle-every-n", type=int, default=3)
    parser.add_argument("--preview-every-n", type=int, default=20)
    parser.add_argument("--save-evidence-previews", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detector-input-size", type=int)
    parser.add_argument("--vehicle-model-path", default="models/vehicle/yolo11n.pt")
    parser.add_argument("--vehicle-conf", type=float, default=0.35)
    parser.add_argument("--vehicle-imgsz", type=int, default=960)
    parser.add_argument("--vehicle-classes", nargs="+", default=["car", "truck", "motorcycle"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--match-distance-px", type=float, default=75.0)
    parser.add_argument("--max-projected-distance-m", type=float, default=5.0)
    parser.add_argument("--min-projected-points-per-evidence", type=int, default=1)
    parser.add_argument("--min-camera-evidence-score", type=float, default=0.05)
    parser.add_argument("--min-camera-distance-quality", type=float, default=0.30)
    parser.add_argument("--require-inside-slot-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-evidence-hold-frames", type=int, default=6)
    parser.add_argument("--camera-evidence-decay", type=float, default=0.85)
    parser.add_argument("--min-held-camera-evidence-score", type=float, default=0.03)
    parser.add_argument("--draw-rejected-vehicles", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-vehicles-per-camera", type=int, default=1)
    parser.add_argument("--near-vehicles-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--near-filter-mode", choices=("all", "any"), default="all")
    parser.add_argument("--near-min-bottom-y-ratio", type=float, default=0.35)
    parser.add_argument("--near-min-height-ratio", type=float, default=0.08)
    parser.add_argument("--near-min-area-ratio", type=float, default=0.010)
    parser.add_argument("--near-max-height-ratio", type=float, default=1.0)
    parser.add_argument("--near-max-area-ratio", type=float, default=1.0)
    parser.add_argument("--near-min-score", type=float, default=0.70)
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
    )
    parser.add_argument("--rotation-mode", choices=("r", "rt"), default="rt")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--no-hold-low-confidence", action="store_true")
    parser.add_argument("--slot-track-iou", type=float, default=0.10)
    parser.add_argument("--slot-track-max-center-distance", type=float, default=45.0)
    parser.add_argument("--slot-track-max-missing", type=int, default=8)
    args = parser.parse_args()
    args._cli_options = cli_option_names(sys.argv[1:])
    return args


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_config_defaults(args, config)
    dataset_root = Path(args.dataset_root)
    timestamps = args.timestamps if args.timestamps else load_timestamps(Path(args.timestamps_from_label_dir), args.limit)
    if not timestamps:
        raise FileNotFoundError(f"No timestamps found in {args.timestamps_from_label_dir}")

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "preview_frames"
    evidence_preview_dir = output_dir / "evidence_frames"
    camera_preview_dir = output_dir / "camera_preview"
    for directory in (output_dir, preview_dir, evidence_preview_dir, camera_preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    first_frame = read_bev_frame(dataset_root, timestamps[0])
    height, width = first_frame.shape[:2]
    writer = make_video_writer(output_dir / "video.mp4", args.fps, (width, height))

    camera_params = load_camera_params(Path(args.stitch_json))
    slot_detector = ParkingSlotDetector(config.get("parking_slot_detector", {}))
    vehicle_detector = VehicleDetector(
        {
            "backend": "yolo",
            "enabled": True,
            "model_path": args.vehicle_model_path,
            "device": args.device,
            "classes": args.vehicle_classes,
            "conf_threshold": args.vehicle_conf,
            "imgsz": args.vehicle_imgsz,
            "min_bottom_y_ratio": args.near_min_bottom_y_ratio if args.near_vehicles_only else 0.0,
            "min_bbox_height_ratio": args.near_min_height_ratio if args.near_vehicles_only else 0.0,
            "min_bbox_area_ratio": args.near_min_area_ratio if args.near_vehicles_only else 0.0,
            "max_bbox_height_ratio": args.near_max_height_ratio if args.near_vehicles_only else 1.0,
            "max_bbox_area_ratio": args.near_max_area_ratio if args.near_vehicles_only else 1.0,
            "min_near_score": args.near_min_score if args.near_vehicles_only else 0.0,
            "max_detections": args.max_vehicles_per_camera,
            "near_filter_classes": args.vehicle_classes,
        }
    )
    occupancy_estimator = OccupancyEstimator(config.get("occupancy", {}))
    state_manager = TemporalStateManager(config.get("occupancy", {}))
    visualizer = Visualizer({"draw_slots": True, "draw_detections": False, "draw_tracks": False})
    slot_tracker = SlotTrackManager(
        min_iou=args.slot_track_iou,
        max_center_distance=args.slot_track_max_center_distance,
        max_missing_frames=args.slot_track_max_missing,
    )

    timeline: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    evidence_events: list[dict[str, Any]] = []
    previous_statuses: dict[int, str] = {}
    counts: Counter[str] = Counter()
    last_slots = []
    last_camera_evidence: dict[int, dict[str, Any]] = {}
    camera_evidence_memory: dict[int, dict[str, Any]] = {}
    last_projected_points: list[dict[str, Any]] = []
    last_camera_detections: list[dict[str, Any]] = []

    try:
        for frame_idx, timestamp in enumerate(tqdm(timestamps, desc="ParkRecon3D multicamera fusion")):
            frame = read_bev_frame(dataset_root, timestamp)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            should_detect_slots = frame_idx % max(1, args.slot_every_n) == 0
            should_detect_vehicles = frame_idx % max(1, args.vehicle_every_n) == 0
            should_write_camera_preview = should_save_preview(frame_idx, args.preview_every_n) or bool(
                args.save_evidence_previews
            )
            if should_detect_slots:
                raw_slots = detect_slots(slot_detector, frame, args.detector_input_size)
                last_slots = slot_tracker.update(frame_idx, raw_slots)
                counts["slot_detection_runs"] += 1
            else:
                last_slots = slot_tracker.active_slots()

            if should_detect_vehicles:
                last_projected_points, last_camera_detections = collect_camera_vehicle_projection(
                    dataset_root,
                    timestamp,
                    args,
                    camera_params,
                    vehicle_detector,
                    camera_preview_dir if should_write_camera_preview else None,
                )
                current_camera_evidence = match_projected_vehicle_points_to_slots(
                    last_slots,
                    last_projected_points,
                    args.match_distance_px,
                    min_points_per_detection=args.min_projected_points_per_evidence,
                    min_evidence_score=args.min_camera_evidence_score,
                    min_distance_quality=args.min_camera_distance_quality,
                    require_inside_slot=args.require_inside_slot_match,
                )
                last_camera_evidence = update_camera_evidence_memory(
                    camera_evidence_memory,
                    current_camera_evidence,
                    frame_idx,
                    args,
                )
                counts["vehicle_detection_runs"] += 1
            else:
                last_camera_evidence = update_camera_evidence_memory(
                    camera_evidence_memory,
                    {},
                    frame_idx,
                    args,
                )

            raw_decisions = occupancy_estimator.estimate(
                last_slots,
                [],
                frame,
                camera_vehicle_evidence=last_camera_evidence,
            )
            decisions = hold_low_confidence_decisions(
                raw_decisions,
                state_manager.states,
                args.low_confidence_threshold,
                enabled=not args.no_hold_low_confidence,
            )
            states = state_manager.update(frame_idx, last_slots, decisions)
            record_events(events, frame_idx, Path(f"{timestamp}.jpg"), states, previous_statuses)

            rendered = visualizer.draw(frame, [], [], states)
            draw_projected_points(rendered, last_projected_points)
            draw_camera_evidence(rendered, last_camera_evidence)
            draw_frame_header(rendered, frame_idx, f"{timestamp}.jpg", len(last_slots), should_detect_slots)
            draw_fusion_header(rendered, len(last_projected_points), len(last_camera_evidence), should_detect_vehicles)
            writer.write(rendered)

            if should_save_preview(frame_idx, args.preview_every_n):
                cv2.imwrite(str(preview_dir / f"{frame_idx:06d}_{timestamp}.jpg"), rendered)
                if should_detect_vehicles:
                    contact = build_frame_contact_sheet(timestamp, rendered, camera_preview_dir, args.cameras)
                    cv2.imwrite(str(preview_dir / f"{frame_idx:06d}_{timestamp}_contact.jpg"), contact)

            if args.save_evidence_previews and last_camera_evidence:
                evidence_prefix = f"{frame_idx:06d}_{timestamp}"
                evidence_contact_saved = False
                cv2.imwrite(str(evidence_preview_dir / f"{evidence_prefix}.jpg"), rendered)
                if should_detect_vehicles:
                    contact = build_frame_contact_sheet(timestamp, rendered, camera_preview_dir, args.cameras)
                    cv2.imwrite(str(evidence_preview_dir / f"{evidence_prefix}_contact.jpg"), contact)
                    evidence_contact_saved = True
                counts["evidence_preview_frames"] += 1
            elif last_camera_evidence:
                evidence_prefix = f"{frame_idx:06d}_{timestamp}"
                evidence_contact_saved = False

            if last_camera_evidence:
                evidence_events.extend(
                    evidence_to_records(
                        frame_idx,
                        timestamp,
                        last_camera_evidence,
                        evidence_prefix if args.save_evidence_previews else "",
                        evidence_contact_saved if args.save_evidence_previews else False,
                    )
                )

            counts["frames"] += 1
            counts["slots"] += len(last_slots)
            counts["projected_points"] += len(last_projected_points)
            counts["slots_with_camera_evidence"] += len(last_camera_evidence)
            counts["direct_camera_evidence"] += sum(
                1 for evidence in last_camera_evidence.values() if not evidence.get("is_held_camera_evidence", False)
            )
            counts["held_camera_evidence"] += sum(
                1 for evidence in last_camera_evidence.values() if evidence.get("is_held_camera_evidence", False)
            )
            counts["camera_detections"] += len(last_camera_detections)
            timeline.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "slot_detection_run": should_detect_slots,
                    "vehicle_detection_run": should_detect_vehicles,
                    "projected_points": last_projected_points,
                    "camera_evidence": list(last_camera_evidence.values()),
                    "slots": [state_to_record(state) for state in states],
                }
            )
    finally:
        writer.release()

    summary = {
        "dataset_root": str(dataset_root),
        "timestamps_from_label_dir": args.timestamps_from_label_dir,
        "output_dir": str(output_dir),
        "video": str(output_dir / "video.mp4"),
        "frames": counts["frames"],
        "slot_every_n": args.slot_every_n,
        "vehicle_every_n": args.vehicle_every_n,
        "vehicle_model_path": args.vehicle_model_path,
        "vehicle_conf": args.vehicle_conf,
        "vehicle_imgsz": args.vehicle_imgsz,
        "vehicle_classes": args.vehicle_classes,
        "match_distance_px": args.match_distance_px,
        "max_projected_distance_m": args.max_projected_distance_m,
        "min_projected_points_per_evidence": args.min_projected_points_per_evidence,
        "min_camera_evidence_score": args.min_camera_evidence_score,
        "min_camera_distance_quality": args.min_camera_distance_quality,
        "require_inside_slot_match": args.require_inside_slot_match,
        "camera_evidence_hold_frames": args.camera_evidence_hold_frames,
        "camera_evidence_decay": args.camera_evidence_decay,
        "min_held_camera_evidence_score": args.min_held_camera_evidence_score,
        "draw_rejected_vehicles": args.draw_rejected_vehicles,
        "save_evidence_previews": args.save_evidence_previews,
        "max_vehicles_per_camera": args.max_vehicles_per_camera,
        "near_vehicle_filter": {
            "enabled": args.near_vehicles_only,
            "near_filter_mode": args.near_filter_mode,
            "near_min_bottom_y_ratio": args.near_min_bottom_y_ratio,
            "near_min_height_ratio": args.near_min_height_ratio,
            "near_min_area_ratio": args.near_min_area_ratio,
            "near_max_height_ratio": args.near_max_height_ratio,
            "near_max_area_ratio": args.near_max_area_ratio,
            "near_min_score": args.near_min_score,
            "max_vehicles_per_camera": args.max_vehicles_per_camera,
        },
        "counts": dict(counts),
        "evidence_preview_dir": str(evidence_preview_dir),
        "evidence_events_csv": str(output_dir / "camera_evidence_events.csv"),
        "evidence_events_jsonl": str(output_dir / "camera_evidence_events.jsonl"),
        "event_count": len(events),
        "note": "Camera vehicle detections are diagnostic evidence only in the default config; occupancy status is decided by EfficientNet.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "timeline.json").write_text(json.dumps({"frames": timeline}, indent=2), encoding="utf-8")
    (output_dir / "events.json").write_text(json.dumps({"events": events}, indent=2), encoding="utf-8")
    write_evidence_events(output_dir, evidence_events)
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

    apply_if_not_supplied(args, supplied, "vehicle_model_path", fusion_config.get("vehicle_model_path", vehicle_config.get("model_path")))
    apply_if_not_supplied(args, supplied, "vehicle_conf", fusion_config.get("vehicle_conf", vehicle_config.get("conf_threshold")))
    apply_if_not_supplied(args, supplied, "vehicle_imgsz", fusion_config.get("vehicle_imgsz", vehicle_config.get("imgsz")))
    apply_if_not_supplied(args, supplied, "vehicle_classes", fusion_config.get("vehicle_classes", occupancy_config.get("vehicle_classes")))
    apply_if_not_supplied(args, supplied, "detector_input_size", slot_config.get("imgsz"))

    for name in (
        "match_distance_px",
        "max_projected_distance_m",
        "min_projected_points_per_evidence",
        "min_camera_evidence_score",
        "min_camera_distance_quality",
        "require_inside_slot_match",
        "camera_evidence_hold_frames",
        "camera_evidence_decay",
        "min_held_camera_evidence_score",
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


def evidence_to_records(
    frame_idx: int,
    timestamp: str,
    camera_evidence: dict[int, dict[str, Any]],
    evidence_prefix: str,
    evidence_contact_saved: bool,
) -> list[dict[str, Any]]:
    records = []
    for slot_id, evidence in camera_evidence.items():
        point = evidence.get("point", [None, None])
        bbox = evidence.get("bbox") or [None, None, None, None]
        bbox_features = evidence.get("bbox_features", {}) or {}
        records.append(
            {
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "slot_id": slot_id,
                "camera": evidence.get("camera"),
                "class_name": evidence.get("class_name"),
                "detection_idx": evidence.get("detection_idx"),
                "bbox_x1": bbox[0] if len(bbox) > 0 else None,
                "bbox_y1": bbox[1] if len(bbox) > 1 else None,
                "bbox_x2": bbox[2] if len(bbox) > 2 else None,
                "bbox_y2": bbox[3] if len(bbox) > 3 else None,
                "bbox_width_ratio": bbox_features.get("width_ratio"),
                "bbox_height_ratio": bbox_features.get("height_ratio"),
                "bbox_area_ratio": bbox_features.get("area_ratio"),
                "bbox_bottom_y_ratio": bbox_features.get("bottom_y_ratio"),
                "bbox_near_score": bbox_features.get("near_score"),
                "evidence_score": evidence.get("evidence_score", evidence.get("confidence")),
                "detector_confidence": evidence.get("detector_confidence"),
                "distance_px": evidence.get("distance_px"),
                "distance_quality": evidence.get("distance_quality"),
                "inside_slot_polygon": evidence.get("inside_slot_polygon", False),
                "match_type": evidence.get("match_type"),
                "matched_projected_point_count": evidence.get("matched_projected_point_count"),
                "held_frames": evidence.get("held_frames", 0),
                "is_held_camera_evidence": evidence.get("is_held_camera_evidence", False),
                "source": evidence.get("source"),
                "point_x": point[0] if len(point) > 0 else None,
                "point_y": point[1] if len(point) > 1 else None,
                "preview_image": f"evidence_frames/{evidence_prefix}.jpg" if evidence_prefix else "",
                "contact_image": (
                    f"evidence_frames/{evidence_prefix}_contact.jpg"
                    if evidence_prefix and evidence_contact_saved
                    else ""
                ),
            }
        )
    return records


def write_evidence_events(output_dir: Path, records: list[dict[str, Any]]) -> None:
    csv_path = output_dir / "camera_evidence_events.csv"
    jsonl_path = output_dir / "camera_evidence_events.jsonl"
    if not records:
        csv_path.write_text("", encoding="utf-8")
        jsonl_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0])
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def update_camera_evidence_memory(
    memory: dict[int, dict[str, Any]],
    current_evidence: dict[int, dict[str, Any]],
    frame_idx: int,
    args: argparse.Namespace,
) -> dict[int, dict[str, Any]]:
    for current_slot_id, evidence in current_evidence.items():
        camera = evidence.get("camera")
        detection_idx = evidence.get("detection_idx")
        for memory_slot_id in list(memory):
            if memory_slot_id == current_slot_id:
                continue
            memory_row = memory[memory_slot_id]
            if memory_row.get("camera") == camera and memory_row.get("detection_idx") == detection_idx:
                del memory[memory_slot_id]

    for slot_id, evidence in current_evidence.items():
        row = dict(evidence)
        row["first_seen_frame"] = memory.get(slot_id, {}).get("first_seen_frame", frame_idx)
        row["last_seen_frame"] = frame_idx
        row["held_frames"] = 0
        row["is_held_camera_evidence"] = False
        memory[slot_id] = row

    active: dict[int, dict[str, Any]] = {}
    for slot_id in list(memory):
        row = memory[slot_id]
        age = frame_idx - int(row.get("last_seen_frame", frame_idx))
        if age > args.camera_evidence_hold_frames:
            del memory[slot_id]
            continue

        evidence_score = float(row.get("evidence_score", row.get("confidence", 0.0)))
        if age > 0:
            evidence_score *= float(args.camera_evidence_decay) ** age
        if evidence_score < float(args.min_held_camera_evidence_score):
            if age > 0:
                del memory[slot_id]
            continue

        active_row = dict(row)
        active_row["confidence"] = evidence_score
        active_row["evidence_score"] = evidence_score
        active_row["held_frames"] = age
        active_row["is_held_camera_evidence"] = age > 0
        if age > 0 and "held_camera_vehicle" not in str(active_row.get("source", "")):
            active_row["source"] = "held_camera_vehicle"
        active[slot_id] = active_row
    return active


def load_timestamps(label_dir: Path, limit: int | None) -> list[str]:
    timestamps = sorted((path.stem for path in label_dir.glob("*.json")), key=int)
    return timestamps[:limit] if limit is not None else timestamps


def read_bev_frame(dataset_root: Path, timestamp: str) -> np.ndarray:
    path = dataset_root / "BEV" / "Data" / "Image" / f"{timestamp}.jpg"
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"Could not read BEV image: {path}")
    return frame


def collect_camera_vehicle_projection(
    dataset_root: Path,
    timestamp: str,
    args: argparse.Namespace,
    camera_params: dict[int, dict[str, np.ndarray]],
    vehicle_detector: VehicleDetector,
    camera_preview_dir: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    projected_points = []
    camera_detections = []
    for camera_id in args.cameras:
        camera_name = CAMERA_DIRS.get(camera_id, f"Camera{camera_id}")
        image_path = dataset_root / camera_name / "Data" / "Image" / f"{timestamp}.jpg"
        image = cv2.imread(str(image_path))
        if image is None or camera_id not in camera_params:
            continue

        raw_detections = vehicle_detector.detect(image)
        detections, rejected_detections = filter_near_vehicle_detections(image, raw_detections, args)
        accepted_detections = []
        for detection in detections:
            points = filter_near_projected_points(
                project_detection_to_bev(detection, camera_params[camera_id], args),
                args,
            )
            if not points:
                rejected_detections.append(detection)
                continue
            detection_idx = len(accepted_detections)
            accepted_detections.append(detection)
            bbox_features = vehicle_near_features(detection, image.shape[1], image.shape[0])
            camera_detections.append(
                {
                    "camera": camera_name,
                    "detection_idx": detection_idx,
                    "class_name": detection.class_name,
                    "confidence": detection.confidence,
                    "bbox": detection.bbox,
                    "bbox_features": bbox_features,
                }
            )
            projected_points.extend(
                {
                    "camera": camera_name,
                    "detection_idx": detection_idx,
                    "point": point,
                    "confidence": detection.confidence,
                    "class_name": detection.class_name,
                    "bbox": detection.bbox,
                    "bbox_features": bbox_features,
                }
                for point in points
            )
        if camera_preview_dir is not None:
            camera_preview = draw_camera_preview(
                image,
                accepted_detections,
                rejected_detections,
                draw_rejected=args.draw_rejected_vehicles,
            )
            cv2.imwrite(str(camera_preview_dir / f"{timestamp}_{camera_name}.jpg"), camera_preview)
    return projected_points, camera_detections


def should_save_preview(frame_idx: int, preview_every_n: int) -> bool:
    return preview_every_n > 0 and frame_idx % preview_every_n == 0


def draw_projected_points(frame: np.ndarray, projected_points: list[dict[str, Any]]) -> None:
    for projected in projected_points:
        x, y = int(round(projected["point"][0])), int(round(projected["point"][1]))
        color = (255, 255, 0)
        cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{projected['camera']} {projected['confidence']:.2f}",
            (x + 6, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_camera_evidence(frame: np.ndarray, camera_evidence: dict[int, dict[str, Any]]) -> None:
    for slot_id, evidence in camera_evidence.items():
        if "point" not in evidence:
            continue
        x, y = int(round(evidence["point"][0])), int(round(evidence["point"][1]))
        is_held = bool(evidence.get("is_held_camera_evidence", False))
        color = (255, 0, 255) if is_held else (0, 255, 255)
        cv2.drawMarker(
            frame,
            (x, y),
            color,
            markerType=cv2.MARKER_DIAMOND,
            markerSize=18,
            thickness=2,
            line_type=cv2.LINE_AA,
        )
        label = f"cam->{slot_id} {float(evidence.get('evidence_score', evidence.get('confidence', 0.0))):.2f}"
        if is_held:
            label += f" h{int(evidence.get('held_frames', 0))}"
        cv2.putText(
            frame,
            label,
            (x + 8, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_fusion_header(
    frame: np.ndarray,
    projected_points_count: int,
    camera_evidence_count: int,
    detected_vehicles: bool,
) -> None:
    label = (
        f"camera vehicle {'on' if detected_vehicles else 'reuse'} | "
        f"projected {projected_points_count} | evidence slots {camera_evidence_count}"
    )
    cv2.rectangle(frame, (0, 28), (frame.shape[1], 56), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
