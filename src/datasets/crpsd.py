from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.detection.schemas import ParkingSlot, SlotStatus


@dataclass(slots=True)
class CRPSDSlotLabel:
    slot_id: int
    mark_a_idx: int
    mark_b_idx: int
    slot_type_id: float
    angle_deg: float
    occupancy_raw: float
    slot: ParkingSlot


def image_path_for_label(label_path: str | Path, dataset_root: str | Path) -> Path:
    label_path = Path(label_path)
    dataset_root = Path(dataset_root)
    split = label_path.parent.parent.name
    return dataset_root / split / "img" / f"{label_path.stem}.jpg"


def label_path_for_image(image_path: str | Path, dataset_root: str | Path) -> Path:
    image_path = Path(image_path)
    dataset_root = Path(dataset_root)
    split = image_path.parent.parent.name
    return dataset_root / split / "slot_label" / f"{image_path.stem}.json"


def load_crpsd_slot_labels(label_path: str | Path) -> list[CRPSDSlotLabel]:
    label_path = Path(label_path)
    data = json.loads(label_path.read_text(encoding="utf-8"))
    marks = data.get("marks", [])
    raw_slots = data.get("slots", [])
    labels: list[CRPSDSlotLabel] = []

    for slot_idx, raw_slot in enumerate(raw_slots, start=1):
        if not isinstance(raw_slot, list) or len(raw_slot) < 5:
            continue

        mark_a_idx = int(raw_slot[0])
        mark_b_idx = int(raw_slot[1])
        if mark_a_idx < 1 or mark_b_idx < 1:
            continue
        if mark_a_idx > len(marks) or mark_b_idx > len(marks):
            continue

        mark_a = marks[mark_a_idx - 1]
        mark_b = marks[mark_b_idx - 1]
        if len(mark_a) < 4 or len(mark_b) < 4:
            continue

        occupancy_raw = float(raw_slot[4])
        occupancy_label = occupancy_status_from_raw(occupancy_raw)
        slot_type_id = float(raw_slot[2])
        slot_type = slot_type_name(slot_type_id)

        points = [
            (float(mark_a[0]), float(mark_a[1])),
            (float(mark_b[0]), float(mark_b[1])),
            (float(mark_b[2]), float(mark_b[3])),
            (float(mark_a[2]), float(mark_a[3])),
        ]
        slot = ParkingSlot(
            slot_id=slot_idx,
            points=points,
            confidence=1.0,
            type=slot_type,
            occupancy_label=occupancy_label,
        )
        labels.append(
            CRPSDSlotLabel(
                slot_id=slot_idx,
                mark_a_idx=mark_a_idx,
                mark_b_idx=mark_b_idx,
                slot_type_id=slot_type_id,
                angle_deg=float(raw_slot[3]),
                occupancy_raw=occupancy_raw,
                slot=slot,
            )
        )

    return labels


def occupancy_status_from_raw(value: float) -> SlotStatus:
    return "occupied" if value >= 0.5 else "free"


def slot_type_name(value: float) -> str:
    if int(value) == 1:
        return "perpendicular"
    if int(value) == 2:
        return "slanted"
    return "unknown"
