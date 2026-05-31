from __future__ import annotations

import unittest

from src.detection.schemas import ParkingSlot
from src.occupancy.camera_vehicle_fusion import fuse_classifier_and_camera_vehicle, match_projected_vehicle_points_to_slots
from src.occupancy.estimator import OccupancyEstimator


def slot(slot_id: int, x1: float, y1: float, x2: float, y2: float) -> ParkingSlot:
    return ParkingSlot(
        slot_id=slot_id,
        points=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
        confidence=1.0,
    )


def projected(camera: str, detection_idx: int, point: tuple[float, float], confidence: float = 0.8) -> dict:
    return {
        "camera": camera,
        "detection_idx": detection_idx,
        "point": point,
        "confidence": confidence,
        "class_name": "car",
        "bbox": (0.0, 0.0, 10.0, 10.0),
        "bbox_features": {"area_ratio": 0.02},
    }


class CameraVehicleFusionTest(unittest.TestCase):
    def test_one_detection_can_match_only_one_slot(self) -> None:
        slots = [
            slot(1, 0, 0, 10, 10),
            slot(2, 20, 0, 30, 10),
        ]
        points = [
            projected("Camera0", 0, (5, 5), 0.9),
            projected("Camera0", 0, (25, 5), 0.9),
        ]

        evidence = match_projected_vehicle_points_to_slots(
            slots,
            points,
            match_distance_px=20.0,
            min_points_per_detection=1,
            require_inside_slot=True,
        )

        self.assertEqual(len(evidence), 1)
        self.assertIn(next(iter(evidence)), {1, 2})

    def test_require_inside_slot_rejects_nearby_point(self) -> None:
        slots = [slot(1, 0, 0, 10, 10)]
        points = [projected("Camera0", 0, (14, 5), 0.9)]

        evidence = match_projected_vehicle_points_to_slots(
            slots,
            points,
            match_distance_px=5.0,
            min_points_per_detection=1,
            require_inside_slot=True,
        )

        self.assertEqual(evidence, {})

    def test_nearby_point_can_match_when_inside_not_required(self) -> None:
        slots = [slot(1, 0, 0, 10, 10)]
        points = [projected("Camera0", 0, (14, 5), 0.9)]

        evidence = match_projected_vehicle_points_to_slots(
            slots,
            points,
            match_distance_px=5.0,
            min_points_per_detection=1,
            require_inside_slot=False,
        )

        self.assertEqual(set(evidence), {1})
        self.assertEqual(evidence[1]["match_type"], "nearby")

    def test_preserves_bbox_diagnostics_in_evidence(self) -> None:
        slots = [slot(1, 0, 0, 10, 10)]
        points = [projected("Camera2", 3, (5, 5), 0.7)]

        evidence = match_projected_vehicle_points_to_slots(
            slots,
            points,
            match_distance_px=5.0,
            min_points_per_detection=1,
            require_inside_slot=True,
        )

        self.assertEqual(evidence[1]["bbox"], (0.0, 0.0, 10.0, 10.0))
        self.assertEqual(evidence[1]["bbox_features"]["area_ratio"], 0.02)

    def test_camera_evidence_does_not_override_classifier_by_default(self) -> None:
        slots = [slot(1, 0, 0, 10, 10)]
        classifier_predictions = {1: ("free", 0.8)}
        slot_evidence = {1: {"confidence": 0.9, "source": "camera_vehicle"}}

        records = fuse_classifier_and_camera_vehicle(slots, classifier_predictions, slot_evidence)

        self.assertEqual(records[0]["fused_status"], "free")
        self.assertEqual(records[0]["source"], "classifier+camera_vehicle_evidence")
        self.assertEqual(records[0]["vehicle_projected_status"], "occupied")

    def test_camera_evidence_override_is_explicit(self) -> None:
        slots = [slot(1, 0, 0, 10, 10)]
        classifier_predictions = {1: ("unknown", 0.0)}
        slot_evidence = {1: {"confidence": 0.9, "source": "camera_vehicle"}}

        records = fuse_classifier_and_camera_vehicle(
            slots,
            classifier_predictions,
            slot_evidence,
            camera_overrides_classifier=True,
        )

        self.assertEqual(records[0]["fused_status"], "occupied")
        self.assertEqual(records[0]["source"], "camera_vehicle")

    def test_estimator_default_config_can_ignore_camera_evidence(self) -> None:
        estimator = OccupancyEstimator({"backend": "geometry", "camera_vehicle_fusion": {"enabled": False}})
        decisions = estimator.estimate(
            [slot(1, 0, 0, 10, 10)],
            [],
            camera_vehicle_evidence={1: {"confidence": 0.9, "source": "camera_vehicle"}},
        )

        self.assertEqual(decisions[1].status, "free")
        self.assertEqual(decisions[1].source, "geometry")


if __name__ == "__main__":
    unittest.main()
