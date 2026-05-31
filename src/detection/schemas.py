from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


BBox = tuple[float, float, float, float]
Point = tuple[float, float]
Polygon = list[Point]
SlotStatus = Literal["free", "occupied", "potentially_occupied", "soon_free", "unknown"]


@dataclass(slots=True)
class Detection:
    class_name: str
    bbox: BBox
    confidence: float


@dataclass(slots=True)
class ParkingSlot:
    slot_id: int
    points: Polygon
    confidence: float = 1.0
    type: str = "unknown"
    occupancy_label: SlotStatus | None = None


@dataclass(slots=True)
class Track:
    track_id: int
    class_name: str
    bbox: BBox
    center: Point
    speed_px: float = 0.0
    age: int = 1
    missed_frames: int = 0


@dataclass(slots=True)
class SlotState:
    slot_id: int
    status: SlotStatus = "unknown"
    occupied_counter: int = 0
    free_counter: int = 0
    last_seen_frame: int = 0
    assigned_track_id: int | None = None
    confidence: float | None = None
    source: str = "unknown"
    release_probability: float | None = None
    release_features: dict | None = None
    slot: ParkingSlot | None = field(default=None, repr=False)


@dataclass(slots=True)
class OccupancyDecision:
    slot_id: int
    status: SlotStatus
    confidence: float
    source: str
    assigned_track_id: int | None = None
