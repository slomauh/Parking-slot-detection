from __future__ import annotations

from src.detection.schemas import Detection, Track
from src.tracking.motion import bbox_center, euclidean_distance


class Tracker:
    """Simple bbox-center tracker placeholder for Stage A."""

    def __init__(self, config: dict) -> None:
        self.max_missed_frames = int(config.get("max_missed_frames", 15))
        self.next_track_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        updated: dict[int, Track] = {}
        unmatched_track_ids = set(self.tracks)

        for detection in detections:
            center = bbox_center(detection.bbox)
            track_id = self._match_track(center, unmatched_track_ids)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                speed = 0.0
                age = 1
            else:
                previous = self.tracks[track_id]
                speed = euclidean_distance(previous.center, center)
                age = previous.age + 1
                unmatched_track_ids.discard(track_id)

            updated[track_id] = Track(
                track_id=track_id,
                class_name=detection.class_name,
                bbox=detection.bbox,
                center=center,
                speed_px=speed,
                age=age,
                missed_frames=0,
            )

        for track_id in unmatched_track_ids:
            previous = self.tracks[track_id]
            if previous.missed_frames + 1 <= self.max_missed_frames:
                previous.missed_frames += 1
                updated[track_id] = previous

        self.tracks = updated
        return list(updated.values())

    def _match_track(self, center: tuple[float, float], candidates: set[int]) -> int | None:
        best_track_id = None
        best_distance = 60.0
        for track_id in candidates:
            distance = euclidean_distance(self.tracks[track_id].center, center)
            if distance < best_distance:
                best_distance = distance
                best_track_id = track_id
        return best_track_id
