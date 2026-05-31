from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.parking_slot_detector import ParkingSlotDetector
from src.detection.schemas import OccupancyDecision, ParkingSlot, SlotState
from src.occupancy.estimator import OccupancyEstimator
from src.occupancy.state_manager import TemporalStateManager
from src.utils.config import load_config
from src.visualization.draw import Visualizer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class SlotTrack:
    track_id: int
    slot: ParkingSlot
    missed_frames: int = 0
    last_seen_frame: int = 0


class SlotTrackManager:
    """Keeps detector-produced slots stable across a chronological frame sequence."""

    def __init__(self, min_iou: float, max_center_distance: float, max_missing_frames: int) -> None:
        self.min_iou = min_iou
        self.max_center_distance = max_center_distance
        self.max_missing_frames = max_missing_frames
        self.next_track_id = 1
        self.tracks: dict[int, SlotTrack] = {}

    def update(self, frame_idx: int, detections: list[ParkingSlot]) -> list[ParkingSlot]:
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        candidates = []

        for track_id, track in self.tracks.items():
            for detection_idx, detection in enumerate(detections):
                iou = polygon_iou(track.slot.points, detection.points)
                center_distance = polygon_center_distance(track.slot.points, detection.points)
                if iou >= self.min_iou or center_distance <= self.max_center_distance:
                    score = iou + max(0.0, 1.0 - center_distance / max(1.0, self.max_center_distance))
                    candidates.append((score, iou, center_distance, track_id, detection_idx))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, _, _, track_id, detection_idx in candidates:
            if track_id in matched_tracks or detection_idx in matched_detections:
                continue
            detection = detections[detection_idx]
            self.tracks[track_id].slot = clone_slot(detection, track_id)
            self.tracks[track_id].missed_frames = 0
            self.tracks[track_id].last_seen_frame = frame_idx
            matched_tracks.add(track_id)
            matched_detections.add(detection_idx)

        for detection_idx, detection in enumerate(detections):
            if detection_idx in matched_detections:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = SlotTrack(
                track_id=track_id,
                slot=clone_slot(detection, track_id),
                missed_frames=0,
                last_seen_frame=frame_idx,
            )

        for track_id in list(self.tracks):
            if track_id not in matched_tracks and self.tracks[track_id].last_seen_frame != frame_idx:
                self.tracks[track_id].missed_frames += 1
            if self.tracks[track_id].missed_frames > self.max_missing_frames:
                del self.tracks[track_id]

        return self.active_slots()

    def active_slots(self) -> list[ParkingSlot]:
        active_tracks = sorted(self.tracks.values(), key=lambda track: track.track_id)
        return [track.slot for track in active_tracks if track.missed_frames <= self.max_missing_frames]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detector + occupancy as a video-like ParkRecon3D BEV sequence")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--image-dir", default="outputs/parkrecon3d_bev_crpsd_format/raw/test/img")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_bev_sequence")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--slot-every-n", type=int, default=3)
    parser.add_argument("--preview-every-n", type=int, default=25)
    parser.add_argument("--detector-input-size", type=int)
    parser.add_argument("--slot-track-iou", type=float, default=0.10)
    parser.add_argument("--slot-track-max-center-distance", type=float, default=45.0)
    parser.add_argument("--slot-track-max-missing", type=int, default=8)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--no-hold-low-confidence", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    image_paths = list_image_paths(Path(args.image_dir))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "preview_frames"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    first_frame = read_frame(image_paths[0])
    height, width = first_frame.shape[:2]
    writer = make_video_writer(output_dir / "video.mp4", args.fps, (width, height))

    slot_detector = ParkingSlotDetector(config.get("parking_slot_detector", {}))
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
    previous_statuses: dict[int, str] = {}
    summary: dict[str, Any] = {
        "image_dir": str(args.image_dir),
        "output_dir": str(output_dir),
        "fps": args.fps,
        "slot_every_n": args.slot_every_n,
        "low_confidence_threshold": args.low_confidence_threshold,
        "hold_low_confidence": not args.no_hold_low_confidence,
        "frames": 0,
        "detections_run": 0,
        "max_active_slots": 0,
        "status_counts": {},
        "event_count": 0,
    }

    last_slots: list[ParkingSlot] = []
    try:
        for frame_idx, image_path in enumerate(tqdm(image_paths, desc="Processing BEV sequence")):
            frame = read_frame(image_path)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            should_detect_slots = frame_idx % max(1, args.slot_every_n) == 0
            if should_detect_slots:
                raw_slots = detect_slots(slot_detector, frame, args.detector_input_size)
                last_slots = slot_tracker.update(frame_idx, raw_slots)
                summary["detections_run"] += 1
            else:
                last_slots = slot_tracker.active_slots()

            raw_decisions = occupancy_estimator.estimate(last_slots, [], frame)
            decisions = hold_low_confidence_decisions(
                raw_decisions,
                state_manager.states,
                args.low_confidence_threshold,
                enabled=not args.no_hold_low_confidence,
            )
            states = state_manager.update(frame_idx, last_slots, decisions)
            record_events(events, frame_idx, image_path, states, previous_statuses)

            rendered = visualizer.draw(frame, [], [], states)
            draw_frame_header(rendered, frame_idx, image_path.name, len(last_slots), should_detect_slots)
            writer.write(rendered)
            if args.preview_every_n > 0 and frame_idx % args.preview_every_n == 0:
                cv2.imwrite(str(preview_dir / f"{frame_idx:06d}_{image_path.name}"), rendered)

            timeline.append(
                {
                    "frame_idx": frame_idx,
                    "image": str(image_path),
                    "slot_detection_run": should_detect_slots,
                    "slots": [state_to_record(state) for state in states],
                }
            )
            summary["frames"] += 1
            summary["max_active_slots"] = max(summary["max_active_slots"], len(last_slots))
    finally:
        writer.release()

    summary["status_counts"] = count_final_statuses(timeline)
    summary["event_count"] = len(events)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "timeline.json").write_text(json.dumps({"frames": timeline}, indent=2), encoding="utf-8")
    (output_dir / "events.json").write_text(json.dumps({"events": events}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def list_image_paths(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_frame(image_path: Path) -> np.ndarray:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return frame


def make_video_writer(output_path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer: {output_path}")
    return writer


def detect_slots(slot_detector: ParkingSlotDetector, frame: np.ndarray, input_size: int | None) -> list[ParkingSlot]:
    if input_size is None:
        return slot_detector.detect(frame)

    original_height, original_width = frame.shape[:2]
    detector_frame = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_AREA)
    detected_slots = slot_detector.detect(detector_frame)
    scale_x = original_width / input_size
    scale_y = original_height / input_size
    return [
        ParkingSlot(
            slot_id=slot.slot_id,
            points=[(float(x) * scale_x, float(y) * scale_y) for x, y in slot.points],
            confidence=slot.confidence,
            type=slot.type,
            occupancy_label=slot.occupancy_label,
        )
        for slot in detected_slots
    ]


def hold_low_confidence_decisions(
    decisions: dict[int, OccupancyDecision],
    previous_states: dict[int, SlotState],
    low_confidence_threshold: float,
    enabled: bool,
) -> dict[int, OccupancyDecision]:
    if not enabled:
        return decisions

    stabilized = {}
    for slot_id, decision in decisions.items():
        previous_status = previous_states.get(slot_id).status if slot_id in previous_states else "unknown"
        is_camera_vehicle_evidence = "camera_vehicle" in decision.source
        if (
            not is_camera_vehicle_evidence
            and decision.confidence < low_confidence_threshold
            and previous_status in {"free", "occupied"}
        ):
            stabilized[slot_id] = OccupancyDecision(
                slot_id=decision.slot_id,
                status=previous_status,
                confidence=decision.confidence,
                source=f"{decision.source}_held_low_confidence",
                assigned_track_id=decision.assigned_track_id,
            )
        else:
            stabilized[slot_id] = decision
    return stabilized


def record_events(
    events: list[dict[str, Any]],
    frame_idx: int,
    image_path: Path,
    states: list[SlotState],
    previous_statuses: dict[int, str],
) -> None:
    for state in states:
        previous_status = previous_statuses.get(state.slot_id)
        if previous_status is not None and previous_status != state.status:
            events.append(
                {
                    "frame_idx": frame_idx,
                    "image": str(image_path),
                    "slot_id": state.slot_id,
                    "from": previous_status,
                    "to": state.status,
                    "confidence": state.confidence,
                    "source": state.source,
                }
            )
        previous_statuses[state.slot_id] = state.status


def state_to_record(state: SlotState) -> dict[str, Any]:
    return {
        "slot_id": state.slot_id,
        "status": state.status,
        "confidence": state.confidence,
        "source": state.source,
        "occupied_counter": state.occupied_counter,
        "free_counter": state.free_counter,
        "points": [[float(x), float(y)] for x, y in state.slot.points] if state.slot else None,
    }


def count_final_statuses(timeline: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for frame in timeline:
        for slot in frame["slots"]:
            status = slot["status"]
            counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def clone_slot(slot: ParkingSlot, slot_id: int) -> ParkingSlot:
    return ParkingSlot(
        slot_id=slot_id,
        points=[(float(x), float(y)) for x, y in slot.points],
        confidence=slot.confidence,
        type=slot.type,
        occupancy_label=slot.occupancy_label,
    )


def polygon_iou(points_a, points_b) -> float:
    polygon_a = make_polygon(points_a)
    polygon_b = make_polygon(points_b)
    if polygon_a is None or polygon_b is None:
        return 0.0
    intersection = polygon_a.intersection(polygon_b).area
    union = polygon_a.union(polygon_b).area
    return float(intersection / union) if union > 0 else 0.0


def make_polygon(points):
    polygon = ShapelyPolygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def polygon_center_distance(points_a, points_b) -> float:
    center_a = np.array(points_a, dtype=np.float32).mean(axis=0)
    center_b = np.array(points_b, dtype=np.float32).mean(axis=0)
    return float(np.linalg.norm(center_a - center_b))


def draw_frame_header(frame: np.ndarray, frame_idx: int, image_name: str, slots_count: int, detected: bool) -> None:
    label = f"frame {frame_idx} | {image_name} | slots {slots_count} | detector {'on' if detected else 'reuse'}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(frame, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


if __name__ == "__main__":
    main()
