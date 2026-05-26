from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.crpsd import image_path_for_label, load_crpsd_slot_labels
from src.detection.parking_slot_detector import ParkingSlotDetector
from src.detection.schemas import ParkingSlot
from src.occupancy.classifier import EfficientNetOccupancyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate slot detector + occupancy classifier on CRPS-D full images")
    parser.add_argument("--dataset-root", default="/home/slomauh/CRPS-D/CRPS-D")
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slot-model-path", default="/home/slomauh/pretrain_model/pretrain_model/1:2.pth")
    parser.add_argument("--slot-external-repo-path", default="external/CRPS-D")
    parser.add_argument("--slot-conf", type=float, default=0.40)
    parser.add_argument("--occupancy-model-path", default="models/occupancy/efficientnet_b0_crpsd.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--occupancy-threshold", type=float, default=0.50)
    parser.add_argument("--match-iou", type=float, default=0.20)
    parser.add_argument("--output-dir", default="outputs/crpsd_full_pipeline_eval")
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

    slot_detector = ParkingSlotDetector(
        {
            "backend": "crpsd",
            "model_path": args.slot_model_path,
            "external_repo_path": args.slot_external_repo_path,
            "device": args.device,
            "conf_threshold": args.slot_conf,
            "depth_factor": 32,
        }
    )
    classifier = EfficientNetOccupancyClassifier(
        {
            "model_path": args.occupancy_model_path,
            "device": args.device,
            "crop_size": 224,
            "occupied_threshold": args.occupancy_threshold,
            "use_pretrained_backbone": False,
        }
    )

    records = []
    counts: Counter[str] = Counter()
    confusion: Counter[tuple[str, str]] = Counter()
    preview_written = 0
    error_written = 0

    for label_path in tqdm(label_paths, desc="Evaluating full pipeline"):
        image_path = image_path_for_label(label_path, dataset_root)
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        gt_slots = [label.slot for label in load_crpsd_slot_labels(label_path)]
        if not gt_slots:
            continue

        pred_slots = slot_detector.detect(frame)
        occupancy_predictions = classifier.predict(frame, pred_slots)
        matches = match_slots(gt_slots, pred_slots, args.match_iou)

        matched_gt_ids = {match["gt_slot"].slot_id for match in matches}
        matched_pred_ids = {match["pred_slot"].slot_id for match in matches}
        false_negative_slots = [slot for slot in gt_slots if slot.slot_id not in matched_gt_ids]
        false_positive_slots = [slot for slot in pred_slots if slot.slot_id not in matched_pred_ids]

        counts["images"] += 1
        counts["gt_slots"] += len(gt_slots)
        counts["pred_slots"] += len(pred_slots)
        counts["matched_slots"] += len(matches)
        counts["false_negative_slots"] += len(false_negative_slots)
        counts["false_positive_slots"] += len(false_positive_slots)

        image_matches = []
        has_error = bool(false_negative_slots or false_positive_slots)
        for match in matches:
            gt_slot = match["gt_slot"]
            pred_slot = match["pred_slot"]
            gt_status = gt_slot.occupancy_label or "unknown"
            pred_status, confidence = occupancy_predictions.get(pred_slot.slot_id, ("unknown", 0.0))
            confusion[(gt_status, pred_status)] += 1
            if gt_status != pred_status:
                has_error = True
            image_matches.append(
                {
                    "gt_slot_id": gt_slot.slot_id,
                    "pred_slot_id": pred_slot.slot_id,
                    "iou": match["iou"],
                    "gt": gt_status,
                    "pred": pred_status,
                    "confidence": confidence,
                    "gt_points": [[float(x), float(y)] for x, y in gt_slot.points],
                    "pred_points": [[float(x), float(y)] for x, y in pred_slot.points],
                }
            )

        records.append(
            {
                "image": str(image_path),
                "matches": image_matches,
                "false_negative_slot_ids": [slot.slot_id for slot in false_negative_slots],
                "false_positive_slot_ids": [slot.slot_id for slot in false_positive_slots],
            }
        )

        if preview_written < args.preview_limit:
            cv2.imwrite(str(preview_dir / image_path.name), draw_result(frame, gt_slots, pred_slots, matches, occupancy_predictions))
            preview_written += 1
        if has_error and error_written < args.preview_limit:
            cv2.imwrite(str(error_dir / image_path.name), draw_result(frame, gt_slots, pred_slots, matches, occupancy_predictions))
            error_written += 1

    metrics = build_metrics(counts, confusion)
    summary = {
        "dataset_root": str(dataset_root),
        "split": args.split,
        "slot_model_path": args.slot_model_path,
        "slot_conf": args.slot_conf,
        "occupancy_model_path": args.occupancy_model_path,
        "occupancy_threshold": args.occupancy_threshold,
        "match_iou": args.match_iou,
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


def match_slots(gt_slots: list[ParkingSlot], pred_slots: list[ParkingSlot], min_iou: float) -> list[dict]:
    candidates = []
    for gt_slot in gt_slots:
        for pred_slot in pred_slots:
            iou = polygon_iou(gt_slot.points, pred_slot.points)
            if iou >= min_iou:
                candidates.append((iou, gt_slot, pred_slot))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_gt = set()
    used_pred = set()
    matches = []
    for iou, gt_slot, pred_slot in candidates:
        if gt_slot.slot_id in used_gt or pred_slot.slot_id in used_pred:
            continue
        used_gt.add(gt_slot.slot_id)
        used_pred.add(pred_slot.slot_id)
        matches.append({"iou": iou, "gt_slot": gt_slot, "pred_slot": pred_slot})
    return matches


def polygon_iou(points_a, points_b) -> float:
    polygon_a = make_polygon(points_a)
    polygon_b = make_polygon(points_b)
    if polygon_a is None or polygon_b is None:
        return 0.0
    intersection = polygon_a.intersection(polygon_b).area
    union = polygon_a.union(polygon_b).area
    return float(intersection / union) if union > 0 else 0.0


def make_polygon(points):
    polygon = ShapelyPolygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        return None
    return polygon


def build_metrics(counts: Counter[str], confusion: Counter[tuple[str, str]]) -> dict[str, float]:
    tp = confusion[("occupied", "occupied")]
    tn = confusion[("free", "free")]
    fp = confusion[("free", "occupied")]
    fn = confusion[("occupied", "free")]
    matched = counts["matched_slots"]
    gt_slots = counts["gt_slots"]
    pred_slots = counts["pred_slots"]
    return {
        "slot_recall": matched / max(1, gt_slots),
        "slot_precision": matched / max(1, pred_slots),
        "matched_occupancy_accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "end_to_end_slot_status_accuracy_over_gt": (tp + tn) / max(1, gt_slots),
        "occupied_precision": tp / max(1, tp + fp),
        "occupied_recall_on_matched": tp / max(1, tp + fn),
        "occupied_f1_on_matched": (2 * tp) / max(1, 2 * tp + fp + fn),
        "free_precision": tn / max(1, tn + fn),
        "free_recall_on_matched": tn / max(1, tn + fp),
        "free_f1_on_matched": (2 * tn) / max(1, 2 * tn + fp + fn),
    }


def draw_result(frame, gt_slots, pred_slots, matches, occupancy_predictions):
    output = frame.copy()
    matched_gt_ids = {match["gt_slot"].slot_id for match in matches}
    matched_pred_ids = {match["pred_slot"].slot_id for match in matches}

    for gt_slot in gt_slots:
        color = (255, 160, 0) if gt_slot.slot_id in matched_gt_ids else (255, 0, 255)
        draw_polygon(output, gt_slot.points, color, f"GT{gt_slot.slot_id}:{gt_slot.occupancy_label}")

    for pred_slot in pred_slots:
        pred_status, confidence = occupancy_predictions.get(pred_slot.slot_id, ("unknown", 0.0))
        color = (0, 180, 0) if pred_slot.slot_id in matched_pred_ids else (0, 220, 220)
        draw_polygon(output, pred_slot.points, color, f"P{pred_slot.slot_id}:{pred_status} {confidence:.2f}", y_offset=14)

    return output


def draw_polygon(image, points, color, label, y_offset: int = 0) -> None:
    polygon = np.array(points, dtype=np.int32)
    cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2)
    x, y = polygon[0]
    cv2.putText(
        image,
        label,
        (int(x), max(16, int(y) - 6 + y_offset)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
