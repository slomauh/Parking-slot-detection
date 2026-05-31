from __future__ import annotations

import numpy as np

from src.detection.schemas import OccupancyDecision, ParkingSlot, Track
from src.geometry.iou import slot_bbox_coverage
from src.geometry.point_in_polygon import point_in_polygon
from src.occupancy.classifier import EfficientNetOccupancyClassifier


class OccupancyEstimator:
    def __init__(self, config: dict) -> None:
        self.backend = str(config.get("backend", "geometry")).lower()
        self.coverage_threshold = float(config.get("slot_coverage_threshold", 0.20))
        self.speed_threshold_px = float(config.get("speed_threshold_px", 2.0))
        self.vehicle_classes = set(config.get("vehicle_classes", ["car", "truck", "bus", "motorcycle"]))
        camera_fusion_config = config.get("camera_vehicle_fusion", {})
        self.camera_vehicle_fusion_enabled = bool(camera_fusion_config.get("enabled", False))
        self.classifier = None
        if self.backend == "classifier":
            self.classifier = EfficientNetOccupancyClassifier(config.get("classifier", {}))
        elif self.backend != "geometry":
            raise ValueError(f"Unsupported occupancy backend: {self.backend}")

    def estimate(
        self,
        slots: list[ParkingSlot],
        tracks: list[Track],
        frame: np.ndarray | None = None,
        camera_vehicle_evidence: dict[int, dict] | None = None,
    ) -> dict[int, OccupancyDecision]:
        if self.backend == "classifier":
            if frame is None:
                raise ValueError("Occupancy classifier backend requires current frame")
            assert self.classifier is not None
            predictions = self.classifier.predict(frame, slots)
            decisions = {
                slot.slot_id: OccupancyDecision(
                    slot_id=slot.slot_id,
                    status=predictions.get(slot.slot_id, ("unknown", 0.0))[0],
                    confidence=predictions.get(slot.slot_id, ("unknown", 0.0))[1],
                    source="classifier",
                    assigned_track_id=None,
                )
                for slot in slots
            }
            return self._apply_camera_vehicle_evidence(decisions, camera_vehicle_evidence)

        decisions = self._estimate_geometry(slots, tracks)
        return self._apply_camera_vehicle_evidence(decisions, camera_vehicle_evidence)

    def _estimate_geometry(self, slots: list[ParkingSlot], tracks: list[Track]) -> dict[int, OccupancyDecision]:
        decisions: dict[int, OccupancyDecision] = {}
        for slot in slots:
            assigned_track_id = None
            best_score = 0.0
            for track in tracks:
                if track.class_name not in self.vehicle_classes:
                    continue
                coverage = slot_bbox_coverage(slot.points, track.bbox)
                bottom_center = ((track.bbox[0] + track.bbox[2]) / 2.0, track.bbox[3])
                is_in_slot = point_in_polygon(bottom_center, slot.points)
                is_slow = track.speed_px <= self.speed_threshold_px
                if is_slow and (coverage >= self.coverage_threshold or is_in_slot):
                    assigned_track_id = track.track_id
                    best_score = max(float(coverage), 1.0 if is_in_slot else 0.0)
                    break
            decisions[slot.slot_id] = OccupancyDecision(
                slot_id=slot.slot_id,
                status="occupied" if assigned_track_id is not None else "free",
                confidence=best_score if assigned_track_id is not None else 1.0,
                source="geometry",
                assigned_track_id=assigned_track_id,
            )
        return decisions

    def _apply_camera_vehicle_evidence(
        self,
        decisions: dict[int, OccupancyDecision],
        camera_vehicle_evidence: dict[int, dict] | None,
    ) -> dict[int, OccupancyDecision]:
        if not self.camera_vehicle_fusion_enabled or not camera_vehicle_evidence:
            return decisions

        for slot_id, evidence in camera_vehicle_evidence.items():
            existing = decisions.get(slot_id)
            camera_confidence = float(evidence.get("confidence", 0.0))
            if existing is None:
                decisions[slot_id] = OccupancyDecision(
                    slot_id=slot_id,
                    status="occupied",
                    confidence=camera_confidence,
                    source="camera_vehicle",
                    assigned_track_id=None,
                )
                continue

            if existing.status == "occupied":
                existing.confidence = max(float(existing.confidence), camera_confidence)
                existing.source = f"{existing.source}+camera_vehicle"
            else:
                existing.status = "occupied"
                existing.confidence = camera_confidence
                existing.source = "camera_vehicle"
                existing.assigned_track_id = None
        return decisions
