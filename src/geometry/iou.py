from __future__ import annotations

from src.detection.schemas import BBox, Polygon
from src.geometry.polygons import polygon_area


def slot_bbox_coverage(slot_polygon: Polygon, bbox: BBox) -> float:
    """Approximate slot coverage by intersecting bounding rectangles."""
    slot_area = polygon_area(slot_polygon)
    if slot_area <= 0:
        return 0.0

    sx1 = min(point[0] for point in slot_polygon)
    sy1 = min(point[1] for point in slot_polygon)
    sx2 = max(point[0] for point in slot_polygon)
    sy2 = max(point[1] for point in slot_polygon)
    bx1, by1, bx2, by2 = bbox

    ix1 = max(sx1, bx1)
    iy1 = max(sy1, by1)
    ix2 = min(sx2, bx2)
    iy2 = min(sy2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    return ((ix2 - ix1) * (iy2 - iy1)) / slot_area
