from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class VideoReader:
    def __init__(self, source: str | int) -> None:
        self.source = source
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise FileNotFoundError(f"Could not open video source: {source}")

    @property
    def fps(self) -> float:
        return self.capture.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def frame_size(self) -> tuple[int, int]:
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        frame_idx = 0
        while True:
            ok, frame = self.capture.read()
            if not ok:
                break
            yield frame_idx, frame
            frame_idx += 1

    def release(self) -> None:
        self.capture.release()


def source_exists(source: str | int) -> bool:
    if isinstance(source, int):
        return True
    return Path(source).exists()
