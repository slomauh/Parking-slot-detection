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

from src.detection.parking_slot_detector import ParkingSlotDetector
from src.detection.schemas import ParkingSlot
from src.occupancy.classifier import EfficientNetOccupancyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate current pipeline on ParkRecon3D BEV images")
    parser.add_argument("--dataset-root", default="/home/slomauh/Downloads/data1")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slot-model-path", default="/home/slomauh/pretrain_model/pretrain_model/1:2.pth")
    parser.add_argument("--slot-external-repo-path", default="external/CRPS-D")
    parser.add_argument("--slot-conf", type=float, default=0.40)
    parser.add_argument("--detector-input-size", type=int, help="Resize BEV frame to NxN before slot detection")
    parser.add_argument("--occupancy-model-path", default="models/occupancy/efficientnet_b0_crpsd.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--occupancy-threshold", type=float, default=0.50)
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_bev_pipeline_test")
    parser.add_argument("--preview-limit", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    image_dir = dataset_root / "BEV" / "Data" / "Image"
    label_dir = dataset_root / "BEV" / "Data" / "label"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No BEV images found in {image_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(image_paths)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

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
    pred_status_counts: Counter[str] = Counter()
    preview_written = 0

    for image_path in tqdm(image_paths, desc="Evaluating ParkRecon3D BEV"):
        label_path = label_dir / f"{image_path.stem}.json"
        if not label_path.exists():
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        gt_slots = load_parkrecon3d_slots(label_path)
        pred_slots = detect_slots(slot_detector, frame, args.detector_input_size)
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
        for match in matches:
            pred_slot = match["pred_slot"]
            pred_status, confidence = occupancy_predictions.get(pred_slot.slot_id, ("unknown", 0.0))
            pred_status_counts[pred_status] += 1
            image_matches.append(
                {
                    "gt_slot_id": match["gt_slot"].slot_id,
                    "pred_slot_id": pred_slot.slot_id,
                    "iou": match["iou"],
                    "pred_status": pred_status,
                    "pred_status_confidence": confidence,
                    "gt_points": [[float(x), float(y)] for x, y in match["gt_slot"].points],
                    "pred_points": [[float(x), float(y)] for x, y in pred_slot.points],
                }
            )

        for pred_slot in false_positive_slots:
            pred_status, _ = occupancy_predictions.get(pred_slot.slot_id, ("unknown", 0.0))
            pred_status_counts[pred_status] += 1

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

    metrics = build_metrics(counts)
    summary = {
        "dataset_root": str(dataset_root),
        "image_dir": str(image_dir),
        "slot_model_path": args.slot_model_path,
        "slot_conf": args.slot_conf,
        "occupancy_model_path": args.occupancy_model_path,
        "occupancy_threshold": args.occupancy_threshold,
        "match_iou": args.match_iou,
        "counts": dict(counts),
        "pred_status_counts": dict(pred_status_counts),
        "metrics": metrics,
        "preview_dir": str(preview_dir),
        "note": "ParkRecon3D BEV labels contain slot geometry here, but no occupancy ground truth.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def load_parkrecon3d_slots(label_path: Path) -> list[ParkingSlot]:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    marks = data.get("marks", [])
    raw_slots = data.get("slots", [])
    slots: list[ParkingSlot] = []

    for slot_idx, raw_slot in enumerate(raw_slots, start=1):
        if not isinstance(raw_slot, list) or len(raw_slot) < 4:
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

        points = [
            (float(mark_a[0]), float(mark_a[1])),
            (float(mark_b[0]), float(mark_b[1])),
            (float(mark_b[2]), float(mark_b[3])),
            (float(mark_a[2]), float(mark_a[3])),
        ]
        slots.append(
            ParkingSlot(
                slot_id=slot_idx,
                points=points,
                confidence=1.0,
                type=slot_type_name(float(raw_slot[2])),
            )
        )

    return slots


def detect_slots(slot_detector: ParkingSlotDetector, frame: np.ndarray, input_size: int | None) -> list[ParkingSlot]:
    if input_size is None:
        return slot_detector.detect(frame)

    original_height, original_width = frame.shape[:2]
    detector_frame = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_AREA)
    detected_slots = slot_detector.detect(detector_frame)
    scale_x = original_width / input_size
    scale_y = original_height / input_size

    scaled_slots = []
    for slot in detected_slots:
        scaled_slots.append(
            ParkingSlot(
                slot_id=slot.slot_id,
                points=[(float(x) * scale_x, float(y) * scale_y) for x, y in slot.points],
                confidence=slot.confidence,
                type=slot.type,
                occupancy_label=slot.occupancy_label,
            )
        )
    return scaled_slots


def slot_type_name(value: float) -> str:
    if int(value) == 1:
        return "perpendicular"
    if int(value) == 2:
        return "slanted"
    return "unknown"


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


def build_metrics(counts: Counter[str]) -> dict[str, float]:
    matched = counts["matched_slots"]
    gt_slots = counts["gt_slots"]
    pred_slots = counts["pred_slots"]
    return {
        "slot_recall": matched / max(1, gt_slots),
        "slot_precision": matched / max(1, pred_slots),
    }


def draw_result(frame, gt_slots, pred_slots, matches, occupancy_predictions):
    output = frame.copy()
    matched_gt_ids = {match["gt_slot"].slot_id for match in matches}
    matched_pred_ids = {match["pred_slot"].slot_id for match in matches}

    for gt_slot in gt_slots:
        color = (255, 160, 0) if gt_slot.slot_id in matched_gt_ids else (255, 0, 255)
        draw_polygon(output, gt_slot.points, color, f"GT{gt_slot.slot_id}")

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
