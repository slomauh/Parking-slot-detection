from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.detection.schemas import ParkingSlot, Track
from src.geometry.iou import slot_bbox_coverage
from src.geometry.point_in_polygon import point_in_polygon
from src.tracking.motion import bbox_center, euclidean_distance


@dataclass(slots=True)
class ReleasePrediction:
    slot_id: int
    release_probability: float
    occupied_overlap: float
    occupying_track_id: int | None
    pedestrian_nearby: bool
    pedestrian_distance_px: float | None
    brake_lights_on: bool
    brake_light_score: float
    motion_state: str
    mean_speed_px: float
    is_vehicle_occupying: bool
    is_passing_vehicle: bool
    features: dict[str, Any]


class ReleasePredictor:
    """Heuristic probability that an occupied parking slot may soon be released."""

    def __init__(self, config: dict) -> None:
        self.enabled = bool(config.get("enabled", False))
        self.vehicle_classes = set(config.get("vehicle_classes", ["car", "truck", "bus", "motorcycle"]))
        self.pedestrian_classes = set(config.get("pedestrian_classes", ["person"]))
        self.slot_overlap_threshold = float(config.get("slot_vehicle_overlap_threshold", 0.20))
        self.pedestrian_near_distance_px = float(config.get("pedestrian_near_distance_px", 90.0))
        self.brake_light_red_ratio_threshold = float(config.get("brake_light_red_ratio_threshold", 0.015))
        self.parked_speed_threshold_px = float(config.get("parked_speed_threshold_px", 2.0))
        self.moving_speed_threshold_px = float(config.get("moving_speed_threshold_px", 5.0))
        self.motion_window = int(config.get("motion_window", 5))
        self.release_probability_threshold = float(config.get("release_probability_threshold", 0.65))
        self.base_score = float(config.get("base_score", 0.05))
        self.pedestrian_weight = float(config.get("pedestrian_weight", 0.30))
        self.brake_light_weight = float(config.get("brake_light_weight", 0.35))
        self.motion_weight = float(config.get("motion_weight", 0.20))
        self.overlap_weight = float(config.get("overlap_weight", 0.10))
        self.track_centers: dict[int, deque[tuple[float, float]]] = defaultdict(
            lambda: deque(maxlen=max(2, self.motion_window))
        )

    def update(
        self,
        frame_idx: int,
        frame: np.ndarray,
        slots: list[ParkingSlot],
        tracks: list[Track],
    ) -> dict[int, ReleasePrediction]:
        if not self.enabled:
            return {}

        self._update_track_history(tracks)
        vehicle_tracks = [track for track in tracks if track.class_name in self.vehicle_classes]
        pedestrian_tracks = [track for track in tracks if track.class_name in self.pedestrian_classes]

        predictions: dict[int, ReleasePrediction] = {}
        for slot in slots:
            vehicle, overlap = self._best_vehicle_for_slot(slot, vehicle_tracks)
            if vehicle is None:
                predictions[slot.slot_id] = self._empty_prediction(slot.slot_id)
                continue

            mean_speed = self._mean_track_speed(vehicle)
            motion_state = self._motion_state(mean_speed)
            is_passing_vehicle = overlap < self.slot_overlap_threshold or mean_speed >= self.moving_speed_threshold_px
            is_vehicle_occupying = overlap >= self.slot_overlap_threshold and not is_passing_vehicle
            pedestrian_nearby, pedestrian_distance = self._pedestrian_near_vehicle(vehicle, pedestrian_tracks)
            brake_score = brake_light_score(frame, vehicle.bbox)
            brake_on = brake_score >= self.brake_light_red_ratio_threshold

            release_probability = self._score_release_probability(
                overlap=overlap,
                is_vehicle_occupying=is_vehicle_occupying,
                pedestrian_nearby=pedestrian_nearby,
                brake_lights_on=brake_on,
                mean_speed=mean_speed,
            )
            features = {
                "frame_idx": frame_idx,
                "slot_vehicle_overlap": overlap,
                "slot_vehicle_overlap_threshold": self.slot_overlap_threshold,
                "pedestrian_nearby": pedestrian_nearby,
                "pedestrian_distance_px": pedestrian_distance,
                "brake_lights_on": brake_on,
                "brake_light_score": brake_score,
                "brake_light_red_ratio_threshold": self.brake_light_red_ratio_threshold,
                "motion_state": motion_state,
                "mean_speed_px": mean_speed,
                "parked_speed_threshold_px": self.parked_speed_threshold_px,
                "moving_speed_threshold_px": self.moving_speed_threshold_px,
                "is_vehicle_occupying": is_vehicle_occupying,
                "is_passing_vehicle": is_passing_vehicle,
            }
            predictions[slot.slot_id] = ReleasePrediction(
                slot_id=slot.slot_id,
                release_probability=release_probability,
                occupied_overlap=overlap,
                occupying_track_id=vehicle.track_id,
                pedestrian_nearby=pedestrian_nearby,
                pedestrian_distance_px=pedestrian_distance,
                brake_lights_on=brake_on,
                brake_light_score=brake_score,
                motion_state=motion_state,
                mean_speed_px=mean_speed,
                is_vehicle_occupying=is_vehicle_occupying,
                is_passing_vehicle=is_passing_vehicle,
                features=features,
            )
        return predictions

    def _update_track_history(self, tracks: list[Track]) -> None:
        active_track_ids = {track.track_id for track in tracks}
        for track in tracks:
            self.track_centers[track.track_id].append(track.center)
        for track_id in list(self.track_centers):
            if track_id not in active_track_ids and len(self.track_centers[track_id]) == 0:
                del self.track_centers[track_id]

    def _best_vehicle_for_slot(self, slot: ParkingSlot, vehicle_tracks: list[Track]) -> tuple[Track | None, float]:
        best_track = None
        best_overlap = 0.0
        for track in vehicle_tracks:
            overlap = float(slot_bbox_coverage(slot.points, track.bbox))
            bottom_center = ((track.bbox[0] + track.bbox[2]) / 2.0, track.bbox[3])
            if point_in_polygon(bottom_center, slot.points):
                overlap = max(overlap, self.slot_overlap_threshold)
            if overlap > best_overlap:
                best_track = track
                best_overlap = overlap
        return best_track, best_overlap

    def _mean_track_speed(self, track: Track) -> float:
        centers = list(self.track_centers.get(track.track_id, []))
        if len(centers) < 2:
            return float(track.speed_px)
        distances = [euclidean_distance(previous, current) for previous, current in zip(centers, centers[1:])]
        return float(sum(distances) / max(1, len(distances)))

    def _motion_state(self, mean_speed: float) -> str:
        if mean_speed <= self.parked_speed_threshold_px:
            return "stationary"
        if mean_speed >= self.moving_speed_threshold_px:
            return "moving"
        return "slow_movement"

    def _pedestrian_near_vehicle(
        self,
        vehicle: Track,
        pedestrian_tracks: list[Track],
    ) -> tuple[bool, float | None]:
        if not pedestrian_tracks:
            return False, None
        vehicle_center = bbox_center(vehicle.bbox)
        distances = [euclidean_distance(vehicle_center, bbox_center(track.bbox)) for track in pedestrian_tracks]
        min_distance = min(distances)
        return min_distance <= self.pedestrian_near_distance_px, float(min_distance)

    def _score_release_probability(
        self,
        overlap: float,
        is_vehicle_occupying: bool,
        pedestrian_nearby: bool,
        brake_lights_on: bool,
        mean_speed: float,
    ) -> float:
        if not is_vehicle_occupying:
            return 0.0

        score = self.base_score
        score += self.overlap_weight * min(1.0, overlap / max(self.slot_overlap_threshold, 1e-6))
        if pedestrian_nearby:
            score += self.pedestrian_weight
        if brake_lights_on:
            score += self.brake_light_weight
        if mean_speed > self.parked_speed_threshold_px:
            motion_ratio = min(
                1.0,
                (mean_speed - self.parked_speed_threshold_px)
                / max(self.moving_speed_threshold_px - self.parked_speed_threshold_px, 1e-6),
            )
            score += self.motion_weight * motion_ratio
        return float(max(0.0, min(1.0, score)))

    @staticmethod
    def _empty_prediction(slot_id: int) -> ReleasePrediction:
        features = {
            "slot_vehicle_overlap": 0.0,
            "pedestrian_nearby": False,
            "brake_lights_on": False,
            "brake_light_score": 0.0,
            "motion_state": "no_vehicle",
            "mean_speed_px": 0.0,
            "is_vehicle_occupying": False,
            "is_passing_vehicle": False,
        }
        return ReleasePrediction(
            slot_id=slot_id,
            release_probability=0.0,
            occupied_overlap=0.0,
            occupying_track_id=None,
            pedestrian_nearby=False,
            pedestrian_distance_px=None,
            brake_lights_on=False,
            brake_light_score=0.0,
            motion_state="no_vehicle",
            mean_speed_px=0.0,
            is_vehicle_occupying=False,
            is_passing_vehicle=False,
            features=features,
        )


def brake_light_score(frame: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    rear_region = crop[crop.shape[0] // 2 :, :]
    if rear_region.size == 0:
        return 0.0

    hsv = cv2.cvtColor(rear_region, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 80, 120], dtype=np.uint8)
    upper_red_1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([165, 80, 120], dtype=np.uint8)
    upper_red_2 = np.array([180, 255, 255], dtype=np.uint8)
    red_mask = cv2.inRange(hsv, lower_red_1, upper_red_1) | cv2.inRange(hsv, lower_red_2, upper_red_2)
    return float(np.count_nonzero(red_mask) / max(1, red_mask.size))
