from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from src.detection.parking_slot_detector import ParkingSlotDetector
from src.detection.vehicle_detector import VehicleDetector
from src.occupancy.estimator import OccupancyEstimator
from src.occupancy.release_predictor import ReleasePredictor
from src.occupancy.state_manager import TemporalStateManager
from src.tracking.tracker import Tracker
from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.video.reader import VideoReader, source_exists
from src.video.writer import VideoWriter
from src.visualization.draw import Visualizer

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parking slot detection MVP pipeline")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config)

    video_config = config.get("video", {})
    runtime_config = config.get("runtime", {})
    source = video_config.get("source", "data/samples/demo.mp4")
    if not source_exists(source):
        create_demo_video(source)
        LOGGER.info("Created demo video at %s", source)

    reader = VideoReader(source)
    writer = VideoWriter(
        output_path=video_config.get("output_path", "outputs/videos/demo_result.mp4"),
        fps=reader.fps,
        frame_size=reader.frame_size,
    )

    vehicle_detector = VehicleDetector(config.get("vehicle_detector", {}))
    slot_detector = ParkingSlotDetector(config.get("parking_slot_detector", {}))
    tracker = Tracker(config.get("tracker", {}))
    occupancy_estimator = OccupancyEstimator(config.get("occupancy", {}))
    release_predictor = ReleasePredictor(config.get("release_prediction", {}))
    state_manager = TemporalStateManager(config.get("occupancy", {}))
    visualizer = Visualizer(config.get("visualization", {}))

    max_frames = runtime_config.get("max_frames")
    detect_every_n = int(runtime_config.get("detect_every_n_frames", 3))
    slot_every_n = int(runtime_config.get("parking_slot_every_n_frames", 5))

    last_detections = []
    last_slots = []
    json_frames: list[dict[str, Any]] = []

    try:
        for frame_idx, frame in tqdm(reader.frames(), desc="Processing frames"):
            if max_frames is not None and frame_idx >= int(max_frames):
                break

            if frame_idx % detect_every_n == 0:
                last_detections = vehicle_detector.detect(frame)
            if frame_idx % slot_every_n == 0:
                last_slots = slot_detector.detect(frame)

            tracks = tracker.update(last_detections)
            decisions = occupancy_estimator.estimate(last_slots, tracks, frame)
            states = state_manager.update(frame_idx, last_slots, decisions)
            release_predictions = release_predictor.update(frame_idx, frame, last_slots, tracks)
            apply_release_predictions(states, release_predictions, release_predictor.release_probability_threshold)

            rendered = visualizer.draw(frame, last_detections, tracks, states)
            writer.write(rendered)

            json_frames.append(
                {
                    "frame_idx": frame_idx,
                    "slots": [
                        {
                            "slot_id": state.slot_id,
                            "status": state.status,
                            "assigned_track_id": state.assigned_track_id,
                            "confidence": state.confidence,
                            "source": state.source,
                            "release_probability": state.release_probability,
                            "release_features": state.release_features,
                            "occupied_counter": state.occupied_counter,
                            "free_counter": state.free_counter,
                            "points": state.slot.points if state.slot else None,
                        }
                        for state in states
                    ],
                    "tracks": [
                        {
                            "track_id": track.track_id,
                            "class_name": track.class_name,
                            "bbox": track.bbox,
                            "center": track.center,
                            "speed_px": track.speed_px,
                            "age": track.age,
                            "missed_frames": track.missed_frames,
                        }
                        for track in tracks
                    ],
                }
            )
    finally:
        reader.release()
        writer.release()

    if bool(video_config.get("save_json", True)):
        json_path = Path(video_config.get("json_path", "outputs/json/demo_result.json"))
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w", encoding="utf-8") as file:
            json.dump({"frames": json_frames}, file, ensure_ascii=False, indent=2)
        LOGGER.info("Saved JSON output to %s", json_path)

    LOGGER.info("Pipeline finished. Processed %d frames.", len(json_frames))


def apply_release_predictions(states, release_predictions, threshold: float) -> None:
    for state in states:
        prediction = release_predictions.get(state.slot_id)
        if prediction is None:
            state.release_probability = None
            state.release_features = None
            continue

        state.release_probability = prediction.release_probability
        state.release_features = prediction.features
        if (
            prediction.release_probability >= threshold
            and prediction.is_vehicle_occupying
            and state.status in {"occupied", "potentially_occupied"}
        ):
            state.status = "soon_free"
            state.source = f"{state.source}+release_prediction"


def create_demo_video(path: str, frame_size: tuple[int, int] = (960, 540), frames: int = 60) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not create demo video: {path}")

    width, height = frame_size
    for frame_idx in range(frames):
        frame = np.full((height, width, 3), (38, 42, 46), dtype=np.uint8)
        cv2.rectangle(frame, (0, int(height * 0.58)), (width, height), (50, 55, 60), -1)
        for x in range(120, width, 170):
            cv2.line(frame, (x, int(height * 0.66)), (x + 100, height - 35), (120, 120, 120), 2)

        car_x = 80 + frame_idx * 7
        car_y = int(height * 0.70)
        cv2.rectangle(frame, (car_x, car_y), (car_x + 110, car_y + 58), (30, 110, 220), -1)
        cv2.circle(frame, (car_x + 25, car_y + 58), 11, (15, 15, 15), -1)
        cv2.circle(frame, (car_x + 85, car_y + 58), 11, (15, 15, 15), -1)
        writer.write(frame)

    writer.release()


if __name__ == "__main__":
    main()
