from __future__ import annotations

import logging

import numpy as np

from src.detection.schemas import Detection

LOGGER = logging.getLogger(__name__)


class VehicleDetector:
    """Vehicle detector adapter.

    The default mock backend keeps local smoke tests lightweight. The YOLO
    backend imports ultralytics lazily, so Kaggle/Colab can run real inference
    without making local development depend on torch.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.backend = str(config.get("backend", "mock")).lower()
        self.enabled = bool(config.get("enabled", True))
        self.model_path = str(config.get("model_path", "yolo11n.pt"))
        self.device = str(config.get("device", "cpu"))
        self.conf_threshold = float(config.get("conf_threshold", 0.35))
        self.imgsz = int(config.get("imgsz", 640))
        self.allowed_classes = set(config.get("classes", []))
        self.model = None

        if self.backend not in {"mock", "yolo"}:
            raise ValueError(f"Unsupported vehicle detector backend: {self.backend}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if not self.enabled:
            return []
        if self.backend == "mock":
            return self._detect_mock(frame)
        return self._detect_yolo(frame)

    def _detect_mock(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        bbox = (
            width * 0.18,
            height * 0.70,
            width * 0.34,
            height * 0.82,
        )
        return [Detection(class_name="car", bbox=bbox, confidence=0.90)]

    def _detect_yolo(self, frame: np.ndarray) -> list[Detection]:
        if self.model is None:
            self.model = self._load_yolo_model()

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = str(names[class_id])
            if self.allowed_classes and class_name not in self.allowed_classes:
                continue

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            confidence = float(box.conf.item())
            detections.append(
                Detection(
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                )
            )

        return detections

    def _load_yolo_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "vehicle_detector.backend is 'yolo', but ultralytics is not installed. "
                "Install Kaggle/Colab dependencies from requirements.txt or switch backend to 'mock'."
            ) from exc

        LOGGER.info("Loading YOLO vehicle model: %s", self.model_path)
        return YOLO(self.model_path)
