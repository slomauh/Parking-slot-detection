from __future__ import annotations

import cv2
import numpy as np

from src.detection.schemas import Detection, SlotState, Track


class Visualizer:
    def __init__(self, config: dict) -> None:
        self.draw_slots = bool(config.get("draw_slots", True))
        self.draw_detections = bool(config.get("draw_detections", True))
        self.draw_tracks = bool(config.get("draw_tracks", True))

    def draw(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        tracks: list[Track],
        states: list[SlotState],
    ) -> np.ndarray:
        output = frame.copy()
        if self.draw_slots:
            self._draw_slots(output, states)
        if self.draw_detections:
            self._draw_detections(output, detections)
        if self.draw_tracks:
            self._draw_tracks(output, tracks)
        return output

    def _draw_slots(self, frame: np.ndarray, states: list[SlotState]) -> None:
        colors = {
            "free": (0, 180, 0),
            "occupied": (0, 0, 220),
            "potentially_occupied": (0, 190, 220),
            "soon_free": (0, 220, 220),
            "unknown": (180, 180, 180),
        }
        for state in states:
            if state.slot is None:
                continue
            points = np.array(state.slot.points, dtype=np.int32)
            color = colors.get(state.status, colors["unknown"])
            cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
            x, y = points[0]
            cv2.putText(
                frame,
                f"S{state.slot_id}: {state.status}",
                (int(x), int(y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

    def _draw_detections(self, frame: np.ndarray, detections: list[Detection]) -> None:
        for detection in detections:
            x1, y1, x2, y2 = [int(value) for value in detection.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
            cv2.putText(
                frame,
                f"{detection.class_name} {detection.confidence:.2f}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 160, 0),
                1,
                cv2.LINE_AA,
            )

    def _draw_tracks(self, frame: np.ndarray, tracks: list[Track]) -> None:
        for track in tracks:
            x, y = [int(value) for value in track.center]
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
            cv2.putText(
                frame,
                f"ID {track.track_id}",
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
