from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import numpy as np
import torch
from shapely.geometry import Polygon as ShapelyPolygon

from src.detection.schemas import ParkingSlot


class ParkingSlotDetector:
    """Parking slot detector adapter.

    Backends:
    - mock: deterministic synthetic slots for smoke tests.
    - crpsd: pretrained SS-PSD/CRPS-D PyTorch checkpoint from zzh362/CRPS-D.
    - yolo_obb: Ultralytics oriented bounding-box detector for direct slot polygons.
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.backend = str(config.get("backend", "mock")).lower()
        self.model_path = str(config.get("model_path", "models/parking_slot/pretrained.pt"))
        self.device_name = str(config.get("device", "cpu"))
        self.conf_threshold = float(config.get("conf_threshold", 0.30))
        self.imgsz = int(config.get("imgsz", 512))
        self.depth_factor = int(config.get("depth_factor", 32))
        self.external_repo_path = Path(config.get("external_repo_path", "external/CRPS-D")).resolve()
        self.slot_pairing_strategy = str(config.get("slot_pairing_strategy", "crpsd")).lower()
        self.slot_postprocess_mode = str(config.get("slot_postprocess_mode", "standard")).lower()
        self.pairing_distance_scale = float(config.get("pairing_distance_scale", 1.0))
        self.max_point_degree = int(config.get("max_point_degree", 0))
        self.min_slot_score = float(config.get("min_slot_score", 0.0))
        self.slot_nms_iou = float(config.get("slot_nms_iou", 0.0))
        self.slot_center_nms_distance = float(config.get("slot_center_nms_distance", 0.0))
        self.slot_angle_nms_threshold = float(config.get("slot_angle_nms_threshold", 20.0))
        self.slot_centerline_overlap_threshold = float(config.get("slot_centerline_overlap_threshold", 0.0))
        self.geometry_filter_enabled = bool(config.get("geometry_filter_enabled", False))
        self.min_slot_area_ratio = float(config.get("min_slot_area_ratio", 0.0))
        self.max_slot_area_ratio = float(config.get("max_slot_area_ratio", 1.0))
        self.max_slot_aspect_ratio = float(config.get("max_slot_aspect_ratio", 0.0))
        self.max_out_of_frame_ratio = float(config.get("max_out_of_frame_ratio", 1.0))
        self.orientation_filter_enabled = bool(config.get("orientation_filter_enabled", False))
        self.orientation_neighbor_radius = float(config.get("orientation_neighbor_radius", 130.0))
        self.orientation_min_neighbors = int(config.get("orientation_min_neighbors", 2))
        self.orientation_angle_threshold = float(config.get("orientation_angle_threshold", 60.0))
        self.orientation_score_margin = float(config.get("orientation_score_margin", 1.05))
        self.model = None
        self._crpsd = None
        self.last_suppressed_slots: list[dict] = []

        if self.backend not in {"mock", "crpsd", "yolo_obb"}:
            raise ValueError(f"Unsupported parking slot detector backend: {self.backend}")
        if self.backend == "crpsd" and self.slot_pairing_strategy not in {"crpsd", "relaxed"}:
            raise ValueError(f"Unsupported slot pairing strategy: {self.slot_pairing_strategy}")
        if self.backend == "crpsd" and self.slot_postprocess_mode not in {"standard", "row_consensus"}:
            raise ValueError(f"Unsupported slot postprocess mode: {self.slot_postprocess_mode}")

    def detect(self, frame: np.ndarray) -> list[ParkingSlot]:
        if self.backend == "crpsd":
            return self._detect_crpsd(frame)
        if self.backend == "yolo_obb":
            return self._detect_yolo_obb(frame)
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

        return self._convert_pred_points_to_slots(frame, pred_points)

    def _detect_yolo_obb(self, frame: np.ndarray) -> list[ParkingSlot]:
        self._load_yolo_obb_model()
        assert self.model is not None

        device_name = self.device_name
        if device_name != "cpu" and not torch.cuda.is_available():
            device_name = "cpu"
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=device_name,
            verbose=False,
        )
        if not results:
            self.last_suppressed_slots = []
            return []
        result = results[0]
        if getattr(result, "obb", None) is None or result.obb is None:
            self.last_suppressed_slots = []
            return []

        polygons = result.obb.xyxyxyxy.detach().cpu().numpy()
        confidences = result.obb.conf.detach().cpu().numpy() if result.obb.conf is not None else np.ones(len(polygons))
        slots = []
        for idx, (polygon, confidence) in enumerate(zip(polygons, confidences), start=1):
            points = [(float(x), float(y)) for x, y in polygon]
            slots.append(ParkingSlot(slot_id=idx, points=points, confidence=float(confidence), type="yolo_obb"))
        return self._postprocess_slots(frame, slots)

    def _load_yolo_obb_model(self) -> None:
        if self.model is not None:
            return
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"YOLO-OBB detector weights not found: {self.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "parking_slot_detector.backend is 'yolo_obb', but ultralytics is not installed. "
                "Install dependencies from requirements.txt or switch backend to 'crpsd'."
            ) from exc
        self.model = YOLO(self.model_path)

    def _convert_pred_points_to_slots(self, frame: np.ndarray, pred_points: list) -> list[ParkingSlot]:
        if not pred_points:
            return []

        marking_points = list(list(zip(*pred_points))[1])
        point_confidences = [float(item[0]) for item in pred_points]
        if self.slot_pairing_strategy == "relaxed":
            raw_slots = self._infer_crpsd_slots_relaxed(marking_points)
        else:
            assert self._crpsd is not None
            raw_slots = self._crpsd["inference_slots"](marking_points)
        raw_slots = self._postprocess_raw_slots(marking_points, raw_slots, point_confidences)
        slots = self._convert_crpsd_slots(frame, marking_points, raw_slots, point_confidences)
        return self._postprocess_slots(frame, slots)

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
        from data.process import pair_marking_points, pair_marking_points_slant, pair_marking_points_vertical
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
            "pair_marking_points": pair_marking_points,
            "pair_marking_points_slant": pair_marking_points_slant,
            "pair_marking_points_vertical": pair_marking_points_vertical,
            "device": device,
        }

    def _infer_crpsd_slots_relaxed(self, marking_points: list) -> list[tuple[int, int, float]]:
        """CRPS-D slot pairing without pass-through-third-point suppression."""
        if self._crpsd is None:
            return []
        crpsd_config = self._crpsd["config"]
        slots: list[tuple[int, int, float]] = []
        for i in range(len(marking_points) - 1):
            for j in range(i + 1, len(marking_points)):
                point_i = marking_points[i]
                point_j = marking_points[j]
                distance = self._crpsd_calc_point_square_dist(point_i, point_j)
                use_slant = False
                use_vertical = False

                if point_i.type < 0.5 < point_j.type or point_j.type < 0.5 < point_i.type:
                    use_slant = True
                if use_slant and distance > self._scaled_max_dist(crpsd_config.SLANT_MAX_DIST):
                    use_vertical = True
                    use_slant = False

                if point_i.type < 0.5:
                    if not (
                        self._scaled_min_dist(crpsd_config.VSLOT_MIN_DIST)
                        <= distance
                        <= self._scaled_max_dist(crpsd_config.VSLOT_MAX_DIST)
                        or self._scaled_min_dist(crpsd_config.HSLOT_MIN_DIST)
                        <= distance
                        <= self._scaled_max_dist(crpsd_config.HSLOT_MAX_DIST)
                        or use_vertical
                    ):
                        continue
                elif not (
                    self._scaled_min_dist(crpsd_config.SLANT_MIN_DIST)
                    <= distance
                    <= self._scaled_max_dist(crpsd_config.SLANT_MAX_DIST)
                    or use_vertical
                ):
                    continue

                result = self._crpsd["pair_marking_points"](point_i, point_j)
                if use_slant:
                    result = self._crpsd["pair_marking_points_slant"](point_i, point_j)
                if use_vertical:
                    result = self._crpsd["pair_marking_points_vertical"](point_i, point_j)

                if result[0] == 1:
                    slots.append((i, j, result[1]))
                elif result[0] == -1:
                    slots.append((j, i, result[1]))
        return slots

    def _postprocess_raw_slots(
        self,
        marking_points: list,
        raw_slots: list,
        point_confidences: list[float] | None = None,
    ) -> list:
        if self.max_point_degree <= 0 or len(raw_slots) < 3:
            return raw_slots

        if self.slot_postprocess_mode == "row_consensus":
            ordered_slots = sorted(
                raw_slots,
                key=lambda raw_slot: (
                    self._raw_slot_score(point_confidences, raw_slot),
                    -self._raw_slot_bridge_length(marking_points, raw_slot),
                ),
                reverse=True,
            )
        else:
            ordered_slots = sorted(raw_slots, key=lambda raw_slot: self._raw_slot_bridge_length(marking_points, raw_slot))
        point_degrees: dict[int, int] = {}
        kept_slots = []
        for raw_slot in ordered_slots:
            point_a_idx = int(raw_slot[0])
            point_b_idx = int(raw_slot[1])
            if point_degrees.get(point_a_idx, 0) >= self.max_point_degree:
                continue
            if point_degrees.get(point_b_idx, 0) >= self.max_point_degree:
                continue
            point_degrees[point_a_idx] = point_degrees.get(point_a_idx, 0) + 1
            point_degrees[point_b_idx] = point_degrees.get(point_b_idx, 0) + 1
            kept_slots.append(raw_slot)
        return kept_slots

    def _raw_slot_bridge_length(self, marking_points: list, raw_slot: tuple[int, int, float]) -> float:
        point_a = marking_points[raw_slot[0]]
        point_b = marking_points[raw_slot[1]]
        return self._crpsd_calc_point_square_dist(point_a, point_b)

    def _scaled_min_dist(self, value: float) -> float:
        if self._crpsd is None:
            return value
        return (value / self.pairing_distance_scale) * self._crpsd["config"].SQUARED_RATIO

    def _scaled_max_dist(self, value: float) -> float:
        if self._crpsd is None:
            return value
        return (value * self.pairing_distance_scale) * self._crpsd["config"].SQUARED_RATIO

    def _convert_crpsd_slots(
        self,
        frame: np.ndarray,
        marking_points: list,
        raw_slots: list,
        point_confidences: list[float] | None = None,
    ) -> list[ParkingSlot]:
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
                    confidence=self._raw_slot_score(point_confidences, raw_slot),
                    type=slot_type,
                )
            )

        return slots

    def _raw_slot_score(self, point_confidences: list[float] | None, raw_slot: tuple[int, int, float]) -> float:
        if point_confidences is None:
            return 1.0
        point_a_idx = int(raw_slot[0])
        point_b_idx = int(raw_slot[1])
        if point_a_idx >= len(point_confidences) or point_b_idx >= len(point_confidences):
            return 1.0
        return float(math.sqrt(max(0.0, point_confidences[point_a_idx]) * max(0.0, point_confidences[point_b_idx])))

    def _postprocess_slots(self, frame: np.ndarray, slots: list[ParkingSlot]) -> list[ParkingSlot]:
        self.last_suppressed_slots = []
        if not slots:
            return []

        kept = [slot for slot in slots if slot.confidence >= self.min_slot_score]
        if self.geometry_filter_enabled:
            kept = [slot for slot in kept if self._slot_geometry_is_valid(frame, slot)]
        if self.slot_postprocess_mode == "row_consensus":
            kept = self._slot_center_angle_nms(kept)
        if self.slot_nms_iou > 0:
            kept = self._slot_polygon_nms(kept, self.slot_nms_iou)
        if self.orientation_filter_enabled:
            kept = self._filter_orientation_outliers(kept)

        return [
            ParkingSlot(
                slot_id=slot_id,
                points=slot.points,
                confidence=slot.confidence,
                type=slot.type,
                occupancy_label=slot.occupancy_label,
            )
            for slot_id, slot in enumerate(kept, start=1)
        ]

    def _slot_geometry_is_valid(self, frame: np.ndarray, slot: ParkingSlot) -> bool:
        polygon = self._make_polygon(slot.points)
        if polygon is None:
            return False

        height, width = frame.shape[:2]
        image_area = max(1.0, float(width * height))
        area_ratio = float(polygon.area) / image_area
        if area_ratio < self.min_slot_area_ratio or area_ratio > self.max_slot_area_ratio:
            return False

        if self.max_slot_aspect_ratio > 0:
            points = np.array(slot.points, dtype=np.float32)
            side_lengths = [
                float(np.linalg.norm(points[(idx + 1) % len(points)] - points[idx]))
                for idx in range(len(points))
            ]
            nonzero_lengths = [length for length in side_lengths if length > 1e-6]
            if not nonzero_lengths:
                return False
            aspect_ratio = max(nonzero_lengths) / max(1e-6, min(nonzero_lengths))
            if aspect_ratio > self.max_slot_aspect_ratio:
                return False

        frame_polygon = ShapelyPolygon([(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)])
        inside_area = polygon.intersection(frame_polygon).area
        out_of_frame_ratio = 1.0 - float(inside_area / max(1e-6, polygon.area))
        return out_of_frame_ratio <= self.max_out_of_frame_ratio

    def _slot_polygon_nms(self, slots: list[ParkingSlot], iou_threshold: float) -> list[ParkingSlot]:
        ordered_slots = sorted(slots, key=lambda slot: slot.confidence, reverse=True)
        kept: list[ParkingSlot] = []
        kept_polygons = []
        for slot in ordered_slots:
            polygon = self._make_polygon(slot.points)
            if polygon is None:
                continue
            suppressing_iou = max((self._polygon_iou(polygon, kept_polygon) for kept_polygon in kept_polygons), default=0.0)
            if suppressing_iou >= iou_threshold:
                self._record_suppressed_slot(slot, "duplicate_polygon_iou", suppressing_iou)
                continue
            kept.append(slot)
            kept_polygons.append(polygon)
        return kept

    def _slot_center_angle_nms(self, slots: list[ParkingSlot]) -> list[ParkingSlot]:
        if not slots or self.slot_center_nms_distance <= 0:
            return slots

        ordered_slots = sorted(slots, key=lambda slot: slot.confidence, reverse=True)
        kept: list[ParkingSlot] = []
        kept_centers: list[np.ndarray] = []
        kept_angles: list[float] = []
        kept_slots: list[ParkingSlot] = []
        kept_polygons = []
        for slot in ordered_slots:
            center = self._slot_center(slot)
            angle = self._slot_bridge_angle(slot)
            polygon = self._make_polygon(slot.points)
            if polygon is None:
                self._record_suppressed_slot(slot, "invalid_polygon", 0.0)
                continue

            duplicate_score = 0.0
            is_duplicate = False
            for kept_center, kept_angle, kept_slot, kept_polygon in zip(
                kept_centers,
                kept_angles,
                kept_slots,
                kept_polygons,
            ):
                center_distance = float(np.linalg.norm(center - kept_center))
                angle_diff = self._axial_angle_diff_degrees(angle, kept_angle)
                iou = self._polygon_iou(polygon, kept_polygon)
                centerline_overlap = self._centerline_overlap_ratio(slot.points, kept_slot.points)
                duplicate_score = max(duplicate_score, iou, centerline_overlap)
                if center_distance <= self.slot_center_nms_distance and angle_diff <= self.slot_angle_nms_threshold:
                    is_duplicate = True
                    break
                if self.slot_centerline_overlap_threshold > 0 and centerline_overlap >= self.slot_centerline_overlap_threshold:
                    if angle_diff <= self.slot_angle_nms_threshold:
                        is_duplicate = True
                        break

            if is_duplicate:
                self._record_suppressed_slot(slot, "duplicate_center_angle", duplicate_score)
                continue
            kept.append(slot)
            kept_centers.append(center)
            kept_angles.append(angle)
            kept_slots.append(slot)
            kept_polygons.append(polygon)
        return kept

    def _filter_orientation_outliers(self, slots: list[ParkingSlot]) -> list[ParkingSlot]:
        if len(slots) < self.orientation_min_neighbors + 1:
            return slots

        centers = [self._slot_center(slot) for slot in slots]
        bridge_angles = [self._slot_bridge_angle(slot) for slot in slots]
        suppressed = set()
        for idx, slot in enumerate(slots):
            neighbors = []
            for other_idx, other_slot in enumerate(slots):
                if idx == other_idx:
                    continue
                if float(np.linalg.norm(centers[idx] - centers[other_idx])) <= self.orientation_neighbor_radius:
                    neighbors.append(other_idx)
            if len(neighbors) < self.orientation_min_neighbors:
                continue

            angle_diffs = [self._axial_angle_diff_degrees(bridge_angles[idx], bridge_angles[other_idx]) for other_idx in neighbors]
            median_diff = float(np.median(angle_diffs))
            neighbor_confidence = float(np.median([slots[other_idx].confidence for other_idx in neighbors]))
            intersects_neighbors = self._slot_intersects_any(slot, [slots[other_idx] for other_idx in neighbors])
            if (
                median_diff >= self.orientation_angle_threshold
                and slot.confidence <= neighbor_confidence * self.orientation_score_margin
                and (self.slot_postprocess_mode != "row_consensus" or intersects_neighbors)
            ):
                suppressed.add(idx)
                self._record_suppressed_slot(slot, "transverse_orientation_outlier", median_diff)

        return [slot for idx, slot in enumerate(slots) if idx not in suppressed]

    def _slot_intersects_any(self, slot: ParkingSlot, other_slots: list[ParkingSlot]) -> bool:
        polygon = self._make_polygon(slot.points)
        if polygon is None:
            return False
        for other_slot in other_slots:
            other_polygon = self._make_polygon(other_slot.points)
            if other_polygon is None:
                continue
            if polygon.intersects(other_polygon) or self._polygon_iou(polygon, other_polygon) > 0:
                return True
        return False

    def _record_suppressed_slot(self, slot: ParkingSlot, reason: str, score: float) -> None:
        self.last_suppressed_slots.append(
            {
                "reason": reason,
                "score": float(score),
                "confidence": float(slot.confidence),
                "points": [[float(x), float(y)] for x, y in slot.points],
            }
        )

    def _centerline_overlap_ratio(self, points_a, points_b) -> float:
        center_a = np.mean(np.array(points_a, dtype=np.float32), axis=0)
        center_b = np.mean(np.array(points_b, dtype=np.float32), axis=0)
        length_a = self._slot_bridge_pixel_length(points_a)
        length_b = self._slot_bridge_pixel_length(points_b)
        if length_a <= 1e-6 or length_b <= 1e-6:
            return 0.0
        center_distance = float(np.linalg.norm(center_a - center_b))
        return max(0.0, 1.0 - center_distance / max(length_a, length_b))

    @staticmethod
    def _slot_bridge_pixel_length(points) -> float:
        array = np.array(points, dtype=np.float32)
        return float(np.linalg.norm(array[1] - array[0]))

    @staticmethod
    def _slot_center(slot: ParkingSlot) -> np.ndarray:
        return np.mean(np.array(slot.points, dtype=np.float32), axis=0)

    @staticmethod
    def _slot_bridge_angle(slot: ParkingSlot) -> float:
        points = np.array(slot.points, dtype=np.float32)
        vector = points[1] - points[0]
        return float(math.atan2(vector[1], vector[0]))

    @staticmethod
    def _axial_angle_diff_degrees(angle_a: float, angle_b: float) -> float:
        diff = abs((angle_a - angle_b + math.pi) % (2 * math.pi) - math.pi)
        diff = min(diff, math.pi - diff)
        return math.degrees(diff)

    @staticmethod
    def _make_polygon(points) -> ShapelyPolygon | None:
        polygon = ShapelyPolygon(points)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 0:
            return None
        return polygon

    @staticmethod
    def _polygon_iou(polygon_a: ShapelyPolygon, polygon_b: ShapelyPolygon) -> float:
        union = polygon_a.union(polygon_b).area
        if union <= 0:
            return 0.0
        return float(polygon_a.intersection(polygon_b).area / union)

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
