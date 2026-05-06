from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import torch

from src.detection.schemas import ParkingSlot


class ParkingSlotDetector:
    """Parking slot detector adapter.

    Backends:
    - mock: deterministic synthetic slots for smoke tests.
    - crpsd: pretrained SS-PSD/CRPS-D PyTorch checkpoint from zzh362/CRPS-D.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.backend = str(config.get("backend", "mock")).lower()
        self.model_path = str(config.get("model_path", "models/parking_slot/pretrained.pt"))
        self.device_name = str(config.get("device", "cpu"))
        self.conf_threshold = float(config.get("conf_threshold", 0.30))
        self.depth_factor = int(config.get("depth_factor", 32))
        self.external_repo_path = Path(config.get("external_repo_path", "external/CRPS-D")).resolve()
        self.model = None
        self._crpsd = None

        if self.backend not in {"mock", "crpsd"}:
            raise ValueError(f"Unsupported parking slot detector backend: {self.backend}")

    def detect(self, frame: np.ndarray) -> list[ParkingSlot]:
        if self.backend == "crpsd":
            return self._detect_crpsd(frame)
        return self._detect_mock(frame)

    def _detect_mock(self, frame: np.ndarray) -> list[ParkingSlot]:
        height, width = frame.shape[:2]
        slot_width = width * 0.18
        slot_height = height * 0.16
        y1 = height * 0.68
        slots: list[ParkingSlot] = []
        for idx, x1 in enumerate((width * 0.18, width * 0.42, width * 0.66), start=1):
            points = [
                (x1, y1),
                (x1 + slot_width, y1),
                (x1 + slot_width, y1 + slot_height),
                (x1, y1 + slot_height),
            ]
            slots.append(ParkingSlot(slot_id=idx, points=points, confidence=0.5, type="mock"))
        return slots

    def _detect_crpsd(self, frame: np.ndarray) -> list[ParkingSlot]:
        self._load_crpsd_model()
        modules = self._crpsd
        assert modules is not None
        assert self.model is not None

        with torch.inference_mode():
            pred_points = modules["detect_marking_points"](
                self.model,
                frame,
                self.conf_threshold,
                modules["device"],
            )
        if not pred_points:
            return []

        marking_points = list(list(zip(*pred_points))[1])
        raw_slots = modules["inference_slots"](marking_points)
        return self._convert_crpsd_slots(frame, marking_points, raw_slots)

    def _load_crpsd_model(self) -> None:
        if self.model is not None:
            return
        if not self.external_repo_path.exists():
            raise FileNotFoundError(
                f"CRPS-D repo not found: {self.external_repo_path}. "
                "Clone https://github.com/zzh362/CRPS-D into external/CRPS-D."
            )
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"CRPS-D detector weights not found: {self.model_path}")

        self._install_visdom_stub()
        if str(self.external_repo_path) not in sys.path:
            sys.path.insert(0, str(self.external_repo_path))

        import config as crpsd_config
        from inference import detect_marking_points, inference_slots
        from model import TeacherDetector

        device = torch.device(self.device_name if torch.cuda.is_available() or self.device_name == "cpu" else "cpu")
        model = TeacherDetector(3, self.depth_factor, crpsd_config.NUM_FEATURE_MAP_CHANNEL).to(device)
        state = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        self.model = model
        self._crpsd = {
            "config": crpsd_config,
            "detect_marking_points": detect_marking_points,
            "inference_slots": inference_slots,
            "device": device,
        }

    def _convert_crpsd_slots(self, frame: np.ndarray, marking_points: list, raw_slots: list) -> list[ParkingSlot]:
        if self._crpsd is None:
            return []
        crpsd_config = self._crpsd["config"]
        image_size = max(frame.shape[:2])
        slots: list[ParkingSlot] = []

        for slot_id, raw_slot in enumerate(raw_slots, start=1):
            point_a = marking_points[raw_slot[0]]
            point_b = marking_points[raw_slot[1]]
            p0_x = image_size * point_a.x - 0.5
            p0_y = image_size * point_a.y - 0.5
            p1_x = image_size * point_b.x - 0.5
            p1_y = image_size * point_b.y - 0.5

            if point_a.type < 0.5:
                distance = self._crpsd_calc_point_square_dist(point_a, point_b)
                if distance <= crpsd_config.VSLOT_MAX_DIST * crpsd_config.SQUARED_RATIO:
                    separating_length = crpsd_config.LONG_SEPARATOR_LENGTH * crpsd_config.RATIO
                else:
                    separating_length = crpsd_config.SHORT_SEPARATOR_LENGTH * crpsd_config.RATIO
                slot_type = "perpendicular"
            else:
                separating_length = crpsd_config.SLANT_SEPARATOR_LENGTH * crpsd_config.RATIO
                slot_type = "slanted"

            cos_val = math.cos(raw_slot[2])
            sin_val = math.sin(raw_slot[2])
            p2_x = p0_x + image_size * separating_length * cos_val
            p2_y = p0_y + image_size * separating_length * sin_val
            p3_x = p1_x + image_size * separating_length * cos_val
            p3_y = p1_y + image_size * separating_length * sin_val

            slots.append(
                ParkingSlot(
                    slot_id=slot_id,
                    points=[(p0_x, p0_y), (p1_x, p1_y), (p3_x, p3_y), (p2_x, p2_y)],
                    confidence=1.0,
                    type=slot_type,
                )
            )

        return slots

    @staticmethod
    def _crpsd_calc_point_square_dist(point_a, point_b) -> float:
        return (point_a.x - point_b.x) ** 2 + (point_a.y - point_b.y) ** 2

    @staticmethod
    def _install_visdom_stub() -> None:
        if "visdom" in sys.modules:
            return

        visdom = types.ModuleType("visdom")

        class Visdom:
            def __init__(self, *args, **kwargs) -> None:
                pass

        visdom.Visdom = Visdom
        sys.modules["visdom"] = visdom
