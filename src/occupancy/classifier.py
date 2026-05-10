from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models

from src.detection.schemas import ParkingSlot, SlotStatus


class EfficientNetOccupancyClassifier:
    """EfficientNet-B0 binary classifier for parking-slot crops."""

    def __init__(self, config: dict) -> None:
        self.model_path = str(config.get("model_path", "models/occupancy/efficientnet_b0_crpsd.pt"))
        self.device_name = str(config.get("device", "cpu"))
        self.crop_size = int(config.get("crop_size", 224))
        self.occupied_threshold = float(config.get("occupied_threshold", 0.50))
        self.use_pretrained_backbone = bool(config.get("use_pretrained_backbone", False))
        self.model: nn.Module | None = None
        self.device = torch.device(self.device_name if torch.cuda.is_available() or self.device_name == "cpu" else "cpu")

    def predict(self, frame: np.ndarray, slots: list[ParkingSlot]) -> dict[int, tuple[SlotStatus, float]]:
        if not slots:
            return {}
        self._load_model()
        assert self.model is not None

        crops = [self._preprocess_crop(frame, slot) for slot in slots]
        batch = torch.stack(crops).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch)
            probabilities = torch.softmax(logits, dim=1)
            occupied_probs = probabilities[:, 1].detach().cpu().numpy()

        decisions: dict[int, tuple[SlotStatus, float]] = {}
        for slot, occupied_prob in zip(slots, occupied_probs):
            prob = float(occupied_prob)
            status: SlotStatus = "occupied" if prob >= self.occupied_threshold else "free"
            confidence = prob if status == "occupied" else 1.0 - prob
            decisions[slot.slot_id] = (status, confidence)
        return decisions

    def _load_model(self) -> None:
        if self.model is not None:
            return
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Occupancy classifier weights not found: {self.model_path}. "
                "Train EfficientNet-B0 first or switch occupancy.backend to 'geometry'."
            )

        model = build_efficientnet_b0(num_classes=2, pretrained=self.use_pretrained_backbone)
        checkpoint = torch.load(self.model_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        self.model = model

    def _preprocess_crop(self, frame: np.ndarray, slot: ParkingSlot) -> torch.Tensor:
        crop = crop_slot_bbox(frame, slot, self.crop_size)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        crop = (crop - mean) / std
        return torch.from_numpy(crop.transpose(2, 0, 1)).float()


def build_efficientnet_b0(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def crop_slot_bbox(frame: np.ndarray, slot: ParkingSlot, crop_size: int = 224) -> np.ndarray:
    points = np.array(slot.points, dtype=np.float32)
    x1 = max(0, int(np.floor(points[:, 0].min())))
    y1 = max(0, int(np.floor(points[:, 1].min())))
    x2 = min(frame.shape[1], int(np.ceil(points[:, 0].max())))
    y2 = min(frame.shape[0], int(np.ceil(points[:, 1].max())))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)
