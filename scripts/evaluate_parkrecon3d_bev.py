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
from src.occupancy.classifier import EfficientNetOccupancyClassifier, crop_slot_bbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate current pipeline on ParkRecon3D BEV images")
    parser.add_argument("--dataset-root", default="/home/slomauh/Downloads/data1")
    parser.add_argument("--image-dir", help="Optional direct directory with BEV images")
    parser.add_argument("--label-dir", help="Optional direct directory with ParkRecon3D slot labels")
    parser.add_argument("--limit", type=int, help="Optional number of images to evaluate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--slot-backend", choices=["crpsd", "yolo_obb"], default="crpsd")
    parser.add_argument("--slot-model-path", default="/home/slomauh/pretrain_model/pretrain_model/1:2.pth")
    parser.add_argument("--slot-external-repo-path", default="external/CRPS-D")
    parser.add_argument("--slot-conf", type=float, default=0.40)
    parser.add_argument("--slot-pairing-strategy", choices=["crpsd", "relaxed"], default="crpsd")
    parser.add_argument("--slot-postprocess-mode", choices=["standard", "row_consensus"], default="standard")
    parser.add_argument("--slot-pairing-distance-scale", type=float, default=1.0)
    parser.add_argument("--slot-max-point-degree", type=int, default=0)
    parser.add_argument("--slot-min-score", type=float, default=0.0)
    parser.add_argument("--slot-nms-iou", type=float, default=0.0)
    parser.add_argument("--slot-center-nms-distance", type=float, default=0.0)
    parser.add_argument("--slot-angle-nms-threshold", type=float, default=20.0)
    parser.add_argument("--slot-centerline-overlap-threshold", type=float, default=0.0)
    parser.add_argument("--slot-geometry-filter", action="store_true")
    parser.add_argument("--slot-min-area-ratio", type=float, default=0.0)
    parser.add_argument("--slot-max-area-ratio", type=float, default=1.0)
    parser.add_argument("--slot-max-aspect-ratio", type=float, default=0.0)
    parser.add_argument("--slot-max-out-of-frame-ratio", type=float, default=1.0)
    parser.add_argument("--slot-orientation-filter", action="store_true")
    parser.add_argument("--slot-orientation-neighbor-radius", type=float, default=130.0)
    parser.add_argument("--slot-orientation-min-neighbors", type=int, default=2)
    parser.add_argument("--slot-orientation-angle-threshold", type=float, default=60.0)
    parser.add_argument("--slot-orientation-score-margin", type=float, default=1.05)
    parser.add_argument("--detector-input-size", type=int, help="Resize BEV frame to NxN before slot detection")
    parser.add_argument("--occupancy-model-path", default="models/occupancy/efficientnet_b0_crpsd.pt")
    parser.add_argument("--skip-occupancy", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--occupancy-threshold", type=float, default=0.50)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_bev_pipeline_test")
    parser.add_argument("--preview-limit", type=int, default=30)
    parser.add_argument("--qa-crop-size", type=int, default=224)
    parser.add_argument("--contact-sheet-limit", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    image_dir = Path(args.image_dir) if args.image_dir else dataset_root / "BEV" / "Data" / "Image"
    label_dir = Path(args.label_dir) if args.label_dir else dataset_root / "BEV" / "Data" / "label"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No BEV images found in {image_dir}")

    rng = random.Random(args.seed)
    rng.shuffle(image_paths)
    if args.limit is not None:
        image_paths = image_paths[: args.limit]

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "preview"
    suppressed_dirs = {
        "duplicates": output_dir / "suppressed" / "duplicates",
        "transverse": output_dir / "suppressed" / "transverse",
        "other": output_dir / "suppressed" / "other",
    }
    preview_dir.mkdir(parents=True, exist_ok=True)
    for suppressed_dir in suppressed_dirs.values():
        suppressed_dir.mkdir(parents=True, exist_ok=True)
    crop_dirs = {
        "free": output_dir / "crops" / "free",
        "occupied": output_dir / "crops" / "occupied",
        "low_confidence": output_dir / "crops" / "low_confidence",
    }
    for crop_dir in crop_dirs.values():
        crop_dir.mkdir(parents=True, exist_ok=True)

    slot_detector = ParkingSlotDetector(
        {
            "backend": args.slot_backend,
            "model_path": args.slot_model_path,
            "external_repo_path": args.slot_external_repo_path,
            "device": args.device,
            "conf_threshold": args.slot_conf,
            "depth_factor": 32,
            "imgsz": args.detector_input_size or 512,
            "slot_pairing_strategy": args.slot_pairing_strategy,
            "slot_postprocess_mode": args.slot_postprocess_mode,
            "pairing_distance_scale": args.slot_pairing_distance_scale,
            "max_point_degree": args.slot_max_point_degree,
            "min_slot_score": args.slot_min_score,
            "slot_nms_iou": args.slot_nms_iou,
            "slot_center_nms_distance": args.slot_center_nms_distance,
            "slot_angle_nms_threshold": args.slot_angle_nms_threshold,
            "slot_centerline_overlap_threshold": args.slot_centerline_overlap_threshold,
            "geometry_filter_enabled": args.slot_geometry_filter,
            "min_slot_area_ratio": args.slot_min_area_ratio,
            "max_slot_area_ratio": args.slot_max_area_ratio,
            "max_slot_aspect_ratio": args.slot_max_aspect_ratio,
            "max_out_of_frame_ratio": args.slot_max_out_of_frame_ratio,
            "orientation_filter_enabled": args.slot_orientation_filter,
            "orientation_neighbor_radius": args.slot_orientation_neighbor_radius,
            "orientation_min_neighbors": args.slot_orientation_min_neighbors,
            "orientation_angle_threshold": args.slot_orientation_angle_threshold,
            "orientation_score_margin": args.slot_orientation_score_margin,
        }
    )
    classifier = None
    if not args.skip_occupancy:
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
    crop_bucket_counts: Counter[str] = Counter()
    confidence_bins: Counter[str] = Counter()
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
        suppressed_slots = getattr(slot_detector, "last_suppressed_slots", [])
        occupancy_predictions = classifier.predict(frame, pred_slots) if classifier is not None else {}
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

        predicted_slot_records = []
        for pred_slot in pred_slots:
            pred_status, confidence = occupancy_predictions.get(pred_slot.slot_id, ("unknown", 0.0))
            bucket = occupancy_bucket(pred_status, confidence, args.low_confidence_threshold)
            crop_path = None
            if classifier is not None:
                confidence_bins[confidence_bin(confidence)] += 1
                crop_bucket_counts[bucket] += 1
                crop_path = save_qa_crop(
                    frame,
                    pred_slot,
                    image_path.stem,
                    pred_status,
                    confidence,
                    bucket,
                    crop_dirs[bucket],
                    args.qa_crop_size,
                    pred_slot.slot_id in matched_pred_ids,
                )
            predicted_slot_records.append(
                {
                    "pred_slot_id": pred_slot.slot_id,
                    "pred_status": pred_status,
                    "pred_status_confidence": confidence,
                    "qa_bucket": bucket,
                    "crop_path": str(crop_path) if crop_path is not None else None,
                    "matched": pred_slot.slot_id in matched_pred_ids,
                    "points": [[float(x), float(y)] for x, y in pred_slot.points],
                }
            )

        records.append(
            {
                "image": str(image_path),
                "matches": image_matches,
                "predicted_slots": predicted_slot_records,
                "suppressed_slots": suppressed_slots,
                "false_negative_slot_ids": [slot.slot_id for slot in false_negative_slots],
                "false_positive_slot_ids": [slot.slot_id for slot in false_positive_slots],
            }
        )

        rendered_preview = None
        if preview_written < args.preview_limit:
            rendered_preview = draw_result(frame, gt_slots, pred_slots, matches, occupancy_predictions, suppressed_slots)
            cv2.imwrite(str(preview_dir / image_path.name), rendered_preview)
            preview_written += 1
        if suppressed_slots:
            if rendered_preview is None:
                rendered_preview = draw_result(frame, gt_slots, pred_slots, matches, occupancy_predictions, suppressed_slots)
            write_suppressed_preview(rendered_preview, image_path.name, suppressed_slots, suppressed_dirs)

    metrics = build_metrics(counts)
    summary = {
        "dataset_root": str(dataset_root),
        "image_dir": str(image_dir),
        "slot_backend": args.slot_backend,
        "slot_model_path": args.slot_model_path,
        "slot_conf": args.slot_conf,
        "slot_pairing_strategy": args.slot_pairing_strategy,
        "slot_postprocess_mode": args.slot_postprocess_mode,
        "slot_pairing_distance_scale": args.slot_pairing_distance_scale,
        "slot_max_point_degree": args.slot_max_point_degree,
        "slot_min_score": args.slot_min_score,
        "slot_nms_iou": args.slot_nms_iou,
        "slot_center_nms_distance": args.slot_center_nms_distance,
        "slot_angle_nms_threshold": args.slot_angle_nms_threshold,
        "slot_centerline_overlap_threshold": args.slot_centerline_overlap_threshold,
        "slot_geometry_filter": args.slot_geometry_filter,
        "slot_min_area_ratio": args.slot_min_area_ratio,
        "slot_max_area_ratio": args.slot_max_area_ratio,
        "slot_max_aspect_ratio": args.slot_max_aspect_ratio,
        "slot_max_out_of_frame_ratio": args.slot_max_out_of_frame_ratio,
        "slot_orientation_filter": args.slot_orientation_filter,
        "slot_orientation_neighbor_radius": args.slot_orientation_neighbor_radius,
        "slot_orientation_min_neighbors": args.slot_orientation_min_neighbors,
        "slot_orientation_angle_threshold": args.slot_orientation_angle_threshold,
        "slot_orientation_score_margin": args.slot_orientation_score_margin,
        "occupancy_model_path": args.occupancy_model_path,
        "skip_occupancy": args.skip_occupancy,
        "occupancy_threshold": args.occupancy_threshold,
        "low_confidence_threshold": args.low_confidence_threshold,
        "match_iou": args.match_iou,
        "counts": dict(counts),
        "pred_status_counts": dict(pred_status_counts),
        "crop_bucket_counts": dict(crop_bucket_counts),
        "confidence_bins": dict(sorted(confidence_bins.items())),
        "metrics": metrics,
        "preview_dir": str(preview_dir),
        "crop_dirs": {name: str(path) for name, path in crop_dirs.items()},
        "suppressed_counts": dict(count_suppressed_reasons(records)),
        "suppressed_dirs": {name: str(path) for name, path in suppressed_dirs.items()},
        "contact_sheets": {
            "preview": str(output_dir / "contact_sheet.jpg"),
            "free": str(output_dir / "crops_contact_sheet_free.jpg"),
            "occupied": str(output_dir / "crops_contact_sheet_occupied.jpg"),
            "low_confidence": str(output_dir / "crops_contact_sheet_low_confidence.jpg"),
        },
        "note": "ParkRecon3D BEV labels contain slot geometry here, but no occupancy ground truth.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "qa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    make_contact_sheet(preview_dir, output_dir / "contact_sheet.jpg", args.contact_sheet_limit, tile_size=(192, 231))
    make_contact_sheet(
        crop_dirs["free"],
        output_dir / "crops_contact_sheet_free.jpg",
        args.contact_sheet_limit,
        tile_size=(160, 160),
    )
    make_contact_sheet(
        crop_dirs["occupied"],
        output_dir / "crops_contact_sheet_occupied.jpg",
        args.contact_sheet_limit,
        tile_size=(160, 160),
    )
    make_contact_sheet(
        crop_dirs["low_confidence"],
        output_dir / "crops_contact_sheet_low_confidence.jpg",
        args.contact_sheet_limit,
        tile_size=(160, 160),
    )
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
    if getattr(slot_detector, "last_suppressed_slots", None):
        slot_detector.last_suppressed_slots = scale_suppressed_slots(
            slot_detector.last_suppressed_slots,
            scale_x,
            scale_y,
        )

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


def scale_suppressed_slots(suppressed_slots: list[dict], scale_x: float, scale_y: float) -> list[dict]:
    scaled = []
    for suppressed_slot in suppressed_slots:
        record = dict(suppressed_slot)
        record["points"] = [
            [float(x) * scale_x, float(y) * scale_y]
            for x, y in suppressed_slot.get("points", [])
        ]
        scaled.append(record)
    return scaled


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
    recall = matched / max(1, gt_slots)
    precision = matched / max(1, pred_slots)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "slot_recall": recall,
        "slot_precision": precision,
        "slot_f1": f1,
    }


def occupancy_bucket(status: str, confidence: float, low_confidence_threshold: float) -> str:
    if confidence < low_confidence_threshold:
        return "low_confidence"
    if status == "occupied":
        return "occupied"
    return "free"


def confidence_bin(confidence: float) -> str:
    lower = int(max(0.0, min(0.999, confidence)) * 10) * 10
    upper = lower + 10
    return f"{lower:02d}-{upper:02d}"


def save_qa_crop(
    frame,
    slot: ParkingSlot,
    image_stem: str,
    status: str,
    confidence: float,
    bucket: str,
    output_dir: Path,
    crop_size: int,
    matched: bool,
) -> Path:
    crop = crop_slot_bbox(frame, slot, crop_size)
    filename = (
        f"{image_stem}_slot{slot.slot_id:03d}_{status}_{confidence:.3f}_"
        f"{'matched' if matched else 'fp'}.jpg"
    )
    crop_path = output_dir / filename
    label = f"{bucket} {status} {confidence:.2f}"
    cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, 22), (0, 0, 0), -1)
    cv2.putText(crop, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return crop_path


def make_contact_sheet(input_dir: Path, output_path: Path, limit: int, tile_size: tuple[int, int]) -> None:
    image_paths = sorted(input_dir.glob("*.jpg"))[:limit]
    if not image_paths:
        return

    columns = 6
    tiles = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        tile = cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            image_path.stem[:24],
            (4, tile.shape[0] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)

    if not tiles:
        return

    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank.copy())
    rows = [cv2.hconcat(tiles[index : index + columns]) for index in range(0, len(tiles), columns)]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.vconcat(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def write_suppressed_preview(image, filename: str, suppressed_slots: list[dict], output_dirs: dict[str, Path]) -> None:
    reasons = {str(slot.get("reason", "")) for slot in suppressed_slots}
    if any("transverse" in reason for reason in reasons):
        cv2.imwrite(str(output_dirs["transverse"] / filename), image)
    if any("duplicate" in reason for reason in reasons):
        cv2.imwrite(str(output_dirs["duplicates"] / filename), image)
    if not any("transverse" in reason or "duplicate" in reason for reason in reasons):
        cv2.imwrite(str(output_dirs["other"] / filename), image)


def count_suppressed_reasons(records: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for suppressed_slot in record.get("suppressed_slots", []):
            counts[str(suppressed_slot.get("reason", "unknown"))] += 1
    return counts


def draw_result(frame, gt_slots, pred_slots, matches, occupancy_predictions, suppressed_slots: list[dict] | None = None):
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

    for suppressed_idx, suppressed_slot in enumerate(suppressed_slots or [], start=1):
        reason = str(suppressed_slot.get("reason", "suppressed"))
        points = [(float(x), float(y)) for x, y in suppressed_slot.get("points", [])]
        if len(points) == 4:
            color = (0, 0, 255) if "transverse" in reason else (80, 80, 255)
            draw_polygon(output, points, color, f"S{suppressed_label(reason, suppressed_idx)}", y_offset=28)

    return output


def suppressed_label(reason: str, idx: int) -> str:
    if "transverse" in reason:
        return f"{idx}:cross"
    if "duplicate" in reason:
        return f"{idx}:dup"
    return str(idx)


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
