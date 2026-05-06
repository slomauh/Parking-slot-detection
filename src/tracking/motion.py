from __future__ import annotations

from math import hypot

from src.detection.schemas import BBox, Point


def bbox_center(bbox: BBox) -> Point:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def euclidean_distance(a: Point, b: Point) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])
