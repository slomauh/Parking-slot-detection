from __future__ import annotations

from src.detection.schemas import ParkingSlot, Track
from src.geometry.iou import slot_bbox_coverage
from src.geometry.point_in_polygon import point_in_polygon


class OccupancyEstimator:
    def __init__(self, config: dict) -> None:
        self.coverage_threshold = float(config.get("slot_coverage_threshold", 0.20))
        self.speed_threshold_px = float(config.get("speed_threshold_px", 2.0))

    def estimate(self, slots: list[ParkingSlot], tracks: list[Track]) -> dict[int, int | None]:
        assignments: dict[int, int | None] = {}
        for slot in slots:
            assigned_track_id = None
            for track in tracks:
                coverage = slot_bbox_coverage(slot.points, track.bbox)
                bottom_center = ((track.bbox[0] + track.bbox[2]) / 2.0, track.bbox[3])
                is_in_slot = point_in_polygon(bottom_center, slot.points)
                is_slow = track.speed_px <= self.speed_threshold_px
                if is_slow and (coverage >= self.coverage_threshold or is_in_slot):
                    assigned_track_id = track.track_id
                    break
            assignments[slot.slot_id] = assigned_track_id
        return assignments
