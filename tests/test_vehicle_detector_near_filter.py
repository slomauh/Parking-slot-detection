from __future__ import annotations

import unittest

from src.detection.schemas import Detection
from src.detection.vehicle_detector import VehicleDetector


class VehicleDetectorNearFilterTest(unittest.TestCase):
    def test_keeps_reasonable_near_vehicle(self) -> None:
        detector = VehicleDetector(
            {
                "backend": "mock",
                "classes": ["car"],
                "near_filter_classes": ["car"],
                "min_bottom_y_ratio": 0.35,
                "min_bbox_height_ratio": 0.08,
                "min_bbox_area_ratio": 0.01,
                "max_bbox_height_ratio": 0.85,
                "max_bbox_area_ratio": 0.45,
                "min_near_score": 0.70,
            }
        )
        detections = [Detection(class_name="car", bbox=(800, 290, 1010, 420), confidence=0.80)]

        kept = detector._filter_and_rank_detections(detections, width=1280, height=960)

        self.assertEqual(kept, detections)

    def test_rejects_implausibly_large_fisheye_box(self) -> None:
        detector = VehicleDetector(
            {
                "backend": "mock",
                "classes": ["car"],
                "near_filter_classes": ["car"],
                "min_bottom_y_ratio": 0.35,
                "min_bbox_height_ratio": 0.08,
                "min_bbox_area_ratio": 0.01,
                "max_bbox_height_ratio": 0.85,
                "max_bbox_area_ratio": 0.45,
                "min_near_score": 0.70,
            }
        )
        detections = [Detection(class_name="car", bbox=(0, 0, 1280, 960), confidence=0.80)]

        kept = detector._filter_and_rank_detections(detections, width=1280, height=960)

        self.assertEqual(kept, [])

    def test_keeps_highest_near_score_when_capped(self) -> None:
        detector = VehicleDetector(
            {
                "backend": "mock",
                "classes": ["car"],
                "near_filter_classes": ["car"],
                "min_bottom_y_ratio": 0.30,
                "min_bbox_height_ratio": 0.05,
                "min_bbox_area_ratio": 0.005,
                "max_bbox_height_ratio": 0.85,
                "max_bbox_area_ratio": 0.45,
                "min_near_score": 0.50,
                "max_detections": 1,
            }
        )
        far = Detection(class_name="car", bbox=(900, 200, 1040, 310), confidence=0.90)
        near = Detection(class_name="car", bbox=(760, 300, 1040, 470), confidence=0.70)

        kept = detector._filter_and_rank_detections([far, near], width=1280, height=960)

        self.assertEqual(kept, [near])


if __name__ == "__main__":
    unittest.main()
