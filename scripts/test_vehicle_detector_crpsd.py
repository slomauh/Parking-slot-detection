from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.schemas import Detection
from src.detection.vehicle_detector import VehicleDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test vehicle detector on CRPS-D images")
    parser.add_argument("--dataset-root", default="/home/slomauh/CRPS-D/CRPS-D")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="models/vehicle/yolo11n.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save-preview", type=int, default=20)
    parser.add_argument("--save-positive-preview", type=int, default=50)
    parser.add_argument("--output-dir", default="outputs/crpsd_vehicle_test")
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

    detector = VehicleDetector(
        {
            "backend": "yolo",
            "enabled": True,
            "model_path": args.model_path,
            "device": args.device,
            "classes": ["car", "truck", "bus", "motorcycle", "person"],
            "conf_threshold": args.conf,
            "imgsz": args.imgsz,
        }
    )

    started_at = perf_counter()
    records: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    images_with_detections = 0

    for index, image_path in enumerate(selected_paths):
        frame = cv2.imread(str(image_path))
        if frame is None:
            records.append({"image": str(image_path), "error": "cv2.imread returned None"})
            continue

        detections = detector.detect(frame)
        if detections:
            images_with_detections += 1
        class_counts.update(detection.class_name for detection in detections)

        if index < args.save_preview:
            preview = draw_detections(frame, detections)
            cv2.imwrite(str(preview_dir / image_path.name), preview)
        if detections and images_with_detections <= args.save_positive_preview:
            preview = draw_detections(frame, detections)
            cv2.imwrite(str(positive_preview_dir / image_path.name), preview)

        records.append(
            {
                "image": str(image_path),
                "detections": [
                    {
                        "class_name": detection.class_name,
                        "bbox": detection.bbox,
                        "confidence": detection.confidence,
                    }
                    for detection in detections
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
        "imgsz": args.imgsz,
        "sample_size": len(selected_paths),
        "images_with_detections": images_with_detections,
        "detection_rate": images_with_detections / max(1, len(selected_paths)),
        "total_detections": sum(class_counts.values()),
        "class_counts": dict(class_counts),
        "elapsed_sec": elapsed_sec,
        "sec_per_image": elapsed_sec / max(1, len(selected_paths)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "detections.json").write_text(json.dumps(records, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved previews to {preview_dir}")
    print(f"Saved positive previews to {positive_preview_dir}")


def draw_detections(frame, detections: list[Detection]):
    output = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(value) for value in detection.bbox]
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 180, 255), 2)
        cv2.putText(
            output,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    return output


if __name__ == "__main__":
    main()
