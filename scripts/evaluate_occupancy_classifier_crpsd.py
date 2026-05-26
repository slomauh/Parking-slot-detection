from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.crpsd import image_path_for_label, load_crpsd_slot_labels
from src.occupancy.classifier import EfficientNetOccupancyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate occupancy classifier on CRPS-D full images")
    parser.add_argument("--dataset-root", default="/home/slomauh/CRPS-D/CRPS-D")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", default="models/occupancy/efficientnet_b0_crpsd.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--output-dir", default="outputs/crpsd_occupancy_eval")
    parser.add_argument("--preview-limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    label_dir = dataset_root / args.split / "slot_label"
    label_paths = sorted(label_dir.glob("*.json"))
    if not label_paths:
        raise FileNotFoundError(f"No labels found in {label_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(label_paths)
    if args.limit is not None:
        label_paths = label_paths[: args.limit]

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "preview"
    error_dir = output_dir / "errors"
    preview_dir.mkdir(parents=True, exist_ok=True)
    error_dir.mkdir(parents=True, exist_ok=True)

    classifier = EfficientNetOccupancyClassifier(
        {
            "model_path": args.model_path,
            "device": args.device,
            "crop_size": args.crop_size,
            "occupied_threshold": args.threshold,
            "use_pretrained_backbone": False,
        }
    )

    records = []
    counts: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    preview_written = 0
    error_preview_written = 0

    for label_path in tqdm(label_paths, desc="Evaluating full images"):
        image_path = image_path_for_label(label_path, dataset_root)
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        labels = load_crpsd_slot_labels(label_path)
        slots = [label.slot for label in labels]
        if not slots:
            continue

        predictions = classifier.predict(frame, slots)
        image_records = []
        has_error = False

        for slot in slots:
            gt = slot.occupancy_label or "unknown"
            pred, confidence = predictions.get(slot.slot_id, ("unknown", 0.0))
            counts[f"gt_{gt}"] += 1
            counts[f"pred_{pred}"] += 1
            confusion[(gt, pred)] += 1
            if gt != pred:
                has_error = True
            image_records.append(
                {
                    "slot_id": slot.slot_id,
                    "gt": gt,
                    "pred": pred,
                    "confidence": confidence,
                    "points": [[float(x), float(y)] for x, y in slot.points],
                }
            )

        records.append({"image": str(image_path), "slots": image_records})

        if preview_written < args.preview_limit:
            cv2.imwrite(str(preview_dir / image_path.name), draw_slots(frame, slots, predictions))
            preview_written += 1
        if has_error and error_preview_written < args.preview_limit:
            cv2.imwrite(str(error_dir / image_path.name), draw_slots(frame, slots, predictions))
            error_preview_written += 1

    metrics = build_metrics(confusion)
    summary = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "model_path": args.model_path,
        "device": args.device,
        "threshold": args.threshold,
        "images_evaluated": len(records),
        "slots_evaluated": sum(len(record["slots"]) for record in records),
        "counts": dict(counts),
        "confusion": {f"{gt}->{pred}": count for (gt, pred), count in confusion.items()},
        "metrics": metrics,
        "preview_dir": str(preview_dir),
        "error_dir": str(error_dir),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_metrics(confusion: Counter[tuple[str, str]]) -> dict[str, float]:
    tp = confusion[("occupied", "occupied")]
    tn = confusion[("free", "free")]
    fp = confusion[("free", "occupied")]
    fn = confusion[("occupied", "free")]
    total = tp + tn + fp + fn
    return {
        "accuracy": (tp + tn) / max(1, total),
        "occupied_precision": tp / max(1, tp + fp),
        "occupied_recall": tp / max(1, tp + fn),
        "occupied_f1": (2 * tp) / max(1, 2 * tp + fp + fn),
        "free_precision": tn / max(1, tn + fn),
        "free_recall": tn / max(1, tn + fp),
        "free_f1": (2 * tn) / max(1, 2 * tn + fp + fn),
    }


def draw_slots(frame, slots, predictions):
    output = frame.copy()
    for slot in slots:
        gt = slot.occupancy_label or "unknown"
        pred, confidence = predictions.get(slot.slot_id, ("unknown", 0.0))
        ok = gt == pred
        color = (0, 180, 0) if ok else (0, 0, 230)
        points = np.array(slot.points, dtype=np.int32)
        cv2.polylines(output, [points], isClosed=True, color=color, thickness=2)
        x, y = points[0]
        cv2.putText(
            output,
            f"S{slot.slot_id} gt:{gt} pred:{pred} {confidence:.2f}",
            (int(x), max(16, int(y) - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return output


if __name__ == "__main__":
    main()
