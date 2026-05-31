from __future__ import annotations

import logging

import numpy as np
import torch

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
        self.min_bottom_y_ratio = float(config.get("min_bottom_y_ratio", 0.0))
        self.min_bbox_height_ratio = float(config.get("min_bbox_height_ratio", 0.0))
        self.min_bbox_area_ratio = float(config.get("min_bbox_area_ratio", 0.0))
        self.max_bbox_height_ratio = float(config.get("max_bbox_height_ratio", 1.0))
        self.max_bbox_area_ratio = float(config.get("max_bbox_area_ratio", 1.0))
        self.min_near_score = float(config.get("min_near_score", 0.0))
        self.max_detections = int(config.get("max_detections", 0))
        self.near_filter_classes = set(config.get("near_filter_classes", []))
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

        device_name = self.device
        if device_name != "cpu" and not torch.cuda.is_available():
            device_name = "cpu"

        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=device_name,
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

        return self._filter_and_rank_detections(detections, frame.shape[1], frame.shape[0])

    def _filter_and_rank_detections(
        self,
        detections: list[Detection],
        width: int,
        height: int,
    ) -> list[Detection]:
        if not detections:
            return []

        filtered = []
        passthrough = []
        for detection in detections:
            should_apply_near_filter = not self.near_filter_classes or detection.class_name in self.near_filter_classes
            if not should_apply_near_filter:
                passthrough.append(detection)
                continue

            features = self._bbox_near_features(detection, width, height)
            if features["bottom_y_ratio"] < self.min_bottom_y_ratio:
                continue
            if features["height_ratio"] < self.min_bbox_height_ratio:
                continue
            if features["area_ratio"] < self.min_bbox_area_ratio:
                continue
            if features["height_ratio"] > self.max_bbox_height_ratio:
                continue
            if features["area_ratio"] > self.max_bbox_area_ratio:
                continue
            if features["near_score"] < self.min_near_score:
                continue
            filtered.append((detection, features["near_score"]))

        filtered.sort(key=lambda item: item[1], reverse=True)
        ranked = [detection for detection, _ in filtered]
        if self.max_detections > 0:
            ranked = ranked[: self.max_detections]
        return ranked + passthrough

    @staticmethod
    def _bbox_near_features(detection: Detection, width: int, height: int) -> dict[str, float]:
        x1, y1, x2, y2 = detection.bbox
        bbox_width = max(0.0, x2 - x1)
        bbox_height = max(0.0, y2 - y1)
        width_ratio = bbox_width / max(1, width)
        bottom_y_ratio = y2 / max(1, height)
        height_ratio = bbox_height / max(1, height)
        area_ratio = (bbox_width * bbox_height) / max(1, width * height)
        near_score = bottom_y_ratio + 2.0 * height_ratio + 8.0 * area_ratio
        return {
            "width_ratio": float(width_ratio),
            "bottom_y_ratio": float(bottom_y_ratio),
            "height_ratio": float(height_ratio),
            "area_ratio": float(area_ratio),
            "near_score": float(near_score),
        }

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
