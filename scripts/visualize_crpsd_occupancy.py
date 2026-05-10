from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.crpsd import image_path_for_label, load_crpsd_slot_labels
from src.detection.schemas import ParkingSlot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CRPS-D occupancy labels")
    parser.add_argument("--dataset-root", default="/home/slomauh/CRPS-D/CRPS-D")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/crpsd_occupancy_labels")
    parser.add_argument("--export-crops", action="store_true")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--preview-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    label_dir = dataset_root / args.split / "slot_label"
    label_paths = sorted(label_dir.glob("*.json"))
    if not label_paths:
        raise FileNotFoundError(f"No json labels found in {label_dir}")

    rng = random.Random(args.seed)
    selected = label_paths[:]
    rng.shuffle(selected)
    selected = selected[: args.limit]

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "preview"
    crop_dir = output_dir / "crops"
    preview_dir.mkdir(parents=True, exist_ok=True)
    if args.export_crops:
        (crop_dir / "free").mkdir(parents=True, exist_ok=True)
        (crop_dir / "occupied").mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    image_count = 0
    slot_count = 0
    missing_images = []

    for label_path in selected:
        image_path = image_path_for_label(label_path, dataset_root)
        image = cv2.imread(str(image_path))
        if image is None:
            missing_images.append(str(image_path))
            continue

        labels = load_crpsd_slot_labels(label_path)
        slots = [label.slot for label in labels]
        if not slots:
            continue

        image_count += 1
        slot_count += len(slots)
        counts.update(slot.occupancy_label or "unknown" for slot in slots)

        if image_count <= args.preview_limit:
            cv2.imwrite(str(preview_dir / image_path.name), draw_slots(image, slots))

        if args.export_crops:
            for slot in slots:
                status = slot.occupancy_label or "unknown"
                if status not in {"free", "occupied"}:
                    continue
                crop = crop_slot(image, slot, args.crop_size)
                crop_name = f"{image_path.stem}_slot_{slot.slot_id:02d}.jpg"
                cv2.imwrite(str(crop_dir / status / crop_name), crop)

    summary = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "sample_size": len(selected),
        "images_written": image_count,
        "slot_count": slot_count,
        "label_counts": dict(counts),
        "missing_images": missing_images,
        "preview_dir": str(preview_dir),
        "crop_dir": str(crop_dir) if args.export_crops else None,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def draw_slots(image, slots: list[ParkingSlot]):
    output = image.copy()
    colors = {
        "free": (0, 190, 0),
        "occupied": (0, 0, 230),
        "unknown": (180, 180, 180),
    }
    for slot in slots:
        status = slot.occupancy_label or "unknown"
        color = colors.get(status, colors["unknown"])
        points = np.array(slot.points, dtype=np.int32)
        cv2.polylines(output, [points], isClosed=True, color=color, thickness=2)
        x, y = points[0]
        cv2.putText(
            output,
            f"S{slot.slot_id} {status}",
            (int(x), max(16, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def crop_slot(image, slot: ParkingSlot, crop_size: int):
    points = np.array(slot.points, dtype=np.float32)
    x1 = max(0, int(np.floor(points[:, 0].min())))
    y1 = max(0, int(np.floor(points[:, 1].min())))
    x2 = min(image.shape[1], int(np.ceil(points[:, 0].max())))
    y2 = min(image.shape[0], int(np.ceil(points[:, 1].max())))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
