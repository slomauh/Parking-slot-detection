from __future__ import annotations

from src.detection.schemas import Point, Polygon


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        xi, yi = current
        xj, yj = previous
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        if intersects:
            inside = not inside
        j = i
    return inside
