from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon

from src.detection.schemas import ParkingSlot


def match_projected_vehicle_points_to_slots(
    slots: list[ParkingSlot],
    projected_points: list[dict[str, Any]],
    match_distance_px: float,
    min_points_per_detection: int = 2,
    min_evidence_score: float = 0.0,
    min_distance_quality: float = 0.0,
    require_inside_slot: bool = False,
) -> dict[int, dict[str, Any]]:
    """Assign projected camera vehicle points to parking-slot polygons.

    A camera detection is positive occupancy evidence only when the projected
    ground contact point lands inside a slot polygon or close to its boundary.
    If one point can match several slots, the nearest slot center wins.
    """

    slot_polygons = []
    for slot in slots:
        polygon = make_valid_polygon(slot.points)
        if polygon is not None:
            slot_polygons.append((slot, polygon))

    evidence_candidates: dict[tuple[str, int, int], list[tuple[float, bool, ParkingSlot, dict[str, Any]]]] = {}
    for projected in projected_points:
        point_xy = projected["point"]
        point = ShapelyPoint(point_xy)
        candidates = []
        for slot, polygon in slot_polygons:
            inside_polygon = polygon.contains(point) or polygon.touches(point)
            distance = 0.0 if inside_polygon else float(polygon.distance(point))
            if distance <= match_distance_px:
                candidates.append((distance, inside_polygon, slot))
        if not candidates:
            continue

        candidates.sort(key=lambda item: (item[0], polygon_center_distance(item[2].points, point_xy)))
        distance, inside_polygon, slot = candidates[0]
        key = (str(projected["camera"]), int(projected["detection_idx"]), int(slot.slot_id))
        evidence_candidates.setdefault(key, []).append((distance, inside_polygon, slot, projected))

    detection_candidates: dict[tuple[str, int], list[dict[str, Any]]] = {}
    min_points = max(1, int(min_points_per_detection))
    for (camera, detection_idx, _), matches in evidence_candidates.items():
        if len(matches) < min_points:
            continue

        matches.sort(key=lambda item: item[0])
        distance, inside_polygon, slot, projected = matches[0]
        if require_inside_slot and not any(match[1] for match in matches):
            continue
        distance_quality = max(0.0, 1.0 - float(distance) / max(float(match_distance_px), 1e-6))
        if distance_quality < min_distance_quality:
            continue
        point_count_quality = min(1.0, len(matches) / max(1, min_points))
        evidence_score = float(projected["confidence"]) * distance_quality * point_count_quality
        if evidence_score < min_evidence_score:
            continue

        matched_points = [match[3]["point"] for match in matches]
        inside_match_count = sum(1 for match in matches if match[1])
        detection_key = (camera, detection_idx)
        detection_candidates.setdefault(detection_key, []).append(
            {
                "slot": slot,
                "projected": projected,
                "distance": distance,
                "distance_quality": distance_quality,
                "evidence_score": evidence_score,
                "inside_polygon": inside_polygon,
                "inside_match_count": inside_match_count,
                "matched_points": matched_points,
            }
        )

    evidence_by_slot: dict[int, dict[str, Any]] = {}
    for candidates in detection_candidates.values():
        candidates.sort(
            key=lambda item: (
                int(item["inside_match_count"]),
                float(item["evidence_score"]),
                len(item["matched_points"]),
                -float(item["distance"]),
            ),
            reverse=True,
        )
        best_candidate = candidates[0]
        slot = best_candidate["slot"]
        projected = best_candidate["projected"]
        point_xy = projected["point"]
        evidence_score = float(best_candidate["evidence_score"])
        existing = evidence_by_slot.get(slot.slot_id)
        if existing is None or evidence_score > float(existing.get("evidence_score", 0.0)):
            evidence_by_slot[slot.slot_id] = {
                "slot_id": slot.slot_id,
                "status": "occupied",
                "confidence": evidence_score,
                "detector_confidence": projected["confidence"],
                "evidence_score": evidence_score,
                "distance_quality": best_candidate["distance_quality"],
                "camera": projected["camera"],
                "class_name": projected["class_name"],
                "detection_idx": projected["detection_idx"],
                "bbox": projected.get("bbox"),
                "bbox_features": projected.get("bbox_features", {}),
                "point": point_xy,
                "distance_px": best_candidate["distance"],
                "inside_slot_polygon": best_candidate["inside_polygon"],
                "match_type": "inside" if best_candidate["inside_polygon"] else "nearby",
                "matched_projected_points": best_candidate["matched_points"],
                "matched_projected_point_count": len(best_candidate["matched_points"]),
                "source": "camera_vehicle",
            }
    return evidence_by_slot


def fuse_classifier_and_camera_vehicle(
    slots: list[ParkingSlot],
    classifier_predictions: dict[int, tuple[str, float]],
    slot_evidence: dict[int, dict[str, Any]],
    camera_overrides_classifier: bool = False,
) -> list[dict[str, Any]]:
    """Attach camera vehicle evidence to crop-classifier occupancy decisions.

    By default the EfficientNet crop classifier remains authoritative for
    `fused_status`. Camera evidence is reported separately as a diagnostic
    positive signal. The old behavior can still be enabled explicitly with
    `camera_overrides_classifier=True`.
    """

    records = []
    for slot in slots:
        classifier_status, classifier_confidence = classifier_predictions.get(slot.slot_id, ("unknown", 0.0))
        camera_evidence = slot_evidence.get(slot.slot_id)
        camera_status = "occupied" if camera_evidence else "unknown"
        camera_confidence = float(camera_evidence["confidence"]) if camera_evidence else 0.0

        if classifier_status in {"occupied", "free"}:
            fused_status = classifier_status
            fused_confidence = float(classifier_confidence)
            source = "classifier+camera_vehicle_evidence" if camera_evidence else "classifier"
        elif camera_overrides_classifier and camera_status == "occupied":
            fused_status = "occupied"
            fused_confidence = camera_confidence
            source = "camera_vehicle"
        else:
            fused_status = "unknown"
            fused_confidence = 0.0
            source = "unknown"

        records.append(
            {
                "slot_id": slot.slot_id,
                "points": slot.points,
                "classifier_status": classifier_status,
                "classifier_confidence": classifier_confidence,
                "vehicle_projected_status": camera_status,
                "vehicle_projected_confidence": camera_confidence,
                "fused_status": fused_status,
                "fused_confidence": fused_confidence,
                "source": source,
                "camera_evidence": camera_evidence,
            }
        )
    return records


def polygon_center_distance(points: list[tuple[float, float]], point_xy: tuple[float, float]) -> float:
    point_array = np.array(points, dtype=np.float32)
    center = point_array.mean(axis=0)
    return float(np.linalg.norm(center - np.array(point_xy, dtype=np.float32)))


def make_valid_polygon(points: list[tuple[float, float]]) -> ShapelyPolygon | None:
    if len(points) < 3:
        return None
    polygon = ShapelyPolygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or not polygon.is_valid:
        return None
    return polygon
