from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.parking_slot_detector import ParkingSlotDetector
from src.detection.schemas import ParkingSlot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test CRPS-D parking slot detector")
    parser.add_argument("--dataset-root", default="/home/slomauh/CRPS-D/CRPS-D")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="/home/slomauh/pretrain_model/pretrain_model/1:2.pth")
    parser.add_argument("--external-repo-path", default="external/CRPS-D")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--save-preview", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/crpsd_slot_test")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.dataset_root) / args.split / "img"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No jpg images found in {image_dir}")

    rng = random.Random(args.seed)
    selected_paths = image_paths[:]
    rng.shuffle(selected_paths)
    selected_paths = selected_paths[: args.limit]

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "preview"
    positive_preview_dir = output_dir / "positive_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    positive_preview_dir.mkdir(parents=True, exist_ok=True)

    detector = ParkingSlotDetector(
        {
            "backend": "crpsd",
            "model_path": args.model_path,
            "external_repo_path": args.external_repo_path,
            "device": args.device,
            "conf_threshold": args.conf,
            "depth_factor": 32,
        }
    )

    started_at = perf_counter()
    records = []
    images_with_slots = 0
    total_slots = 0

    for index, image_path in enumerate(selected_paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            records.append({"image": str(image_path), "error": "cv2.imread returned None"})
            continue

        slots = detector.detect(frame)
        total_slots += len(slots)
        if slots:
            images_with_slots += 1

        if index < args.save_preview:
            cv2.imwrite(str(preview_dir / image_path.name), draw_slots(frame, slots))
        if slots and images_with_slots <= args.save_preview:
            cv2.imwrite(str(positive_preview_dir / image_path.name), draw_slots(frame, slots))

        records.append(
            {
                "image": str(image_path),
                "slots": [
                    {
                        "slot_id": slot.slot_id,
                        "points": [[float(x), float(y)] for x, y in slot.points],
                        "confidence": float(slot.confidence),
                        "type": slot.type,
                    }
                    for slot in slots
                ],
            }
        )

    elapsed_sec = perf_counter() - started_at
    summary = {
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "image_dir": str(image_dir),
        "model_path": args.model_path,
        "device": args.device,
        "conf_threshold": args.conf,
        "sample_size": len(selected_paths),
        "images_with_slots": images_with_slots,
        "slot_detection_rate": images_with_slots / max(1, len(selected_paths)),
        "total_slots": total_slots,
        "avg_slots_per_image": total_slots / max(1, len(selected_paths)),
        "elapsed_sec": elapsed_sec,
        "sec_per_image": elapsed_sec / max(1, len(selected_paths)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "detections.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved previews to {preview_dir}")
    print(f"Saved positive previews to {positive_preview_dir}")


def draw_slots(frame, slots: list[ParkingSlot]):
    output = frame.copy()
    for slot in slots:
        points = [(int(round(x)), int(round(y))) for x, y in slot.points]
        for start, end in zip(points, points[1:] + points[:1]):
            cv2.line(output, start, end, (0, 220, 0), 2)
        x, y = points[0]
        cv2.putText(
            output,
            f"S{slot.slot_id}",
            (x, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )
    return output


if __name__ == "__main__":
    main()
