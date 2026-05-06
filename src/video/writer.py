from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int]) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output_path}")

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()
