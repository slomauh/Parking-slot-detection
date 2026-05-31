from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detection.parking_slot_detector import ParkingSlotDetector
from src.detection.schemas import ParkingSlot
from src.occupancy.classifier import EfficientNetOccupancyClassifier, crop_slot_bbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ParkRecon3D slot crops for manual occupancy labeling")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset-root", default="/home/slomauh/Documents/parkrecon3d_dataset/data3")
    parser.add_argument("--image-dir", help="Optional direct BEV image directory")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_occupancy_manual_review")
    parser.add_argument("--limit-frames", type=int, help="Limit number of BEV frames to scan")
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every Nth frame after sorting")
    parser.add_argument("--max-crops", type=int, default=1000)
    parser.add_argument(
        "--sample-strategy",
        choices=("uncertain", "balanced_pred", "random", "chronological"),
        default="uncertain",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", help="Override detector/classifier device")
    parser.add_argument("--slot-model-path", help="Override parking slot detector weights")
    parser.add_argument("--slot-conf", type=float, help="Override parking slot detector confidence")
    parser.add_argument("--detector-input-size", type=int, help="Override parking slot detector image size")
    parser.add_argument("--occupancy-model-path", help="Override EfficientNet occupancy weights")
    parser.add_argument("--occupancy-threshold", type=float, help="Override occupied probability threshold")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--context-size", type=int, default=640)
    parser.add_argument("--contact-sheet-limit", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config))
    rng = random.Random(args.seed)

    image_dir = Path(args.image_dir) if args.image_dir else Path(args.dataset_root) / "BEV" / "Data" / "Image"
    image_paths = sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        raise FileNotFoundError(f"No BEV images found in {image_dir}")
    image_paths = image_paths[:: max(1, args.frame_stride)]
    if args.limit_frames is not None:
        image_paths = image_paths[: args.limit_frames]

    output_dir = Path(args.output_dir)
    crop_dir = output_dir / "unlabeled"
    context_dir = output_dir / "context"
    sheet_dir = output_dir / "sheets"
    for directory in (crop_dir, context_dir, sheet_dir):
        directory.mkdir(parents=True, exist_ok=True)
    make_label_dirs(output_dir)

    slot_detector = build_slot_detector(config, args)
    classifier = None if args.skip_classifier else build_classifier(config, args)

    candidates = collect_candidates(image_paths, slot_detector, classifier, args)
    selected = select_candidates(candidates, args, rng)
    write_review_files(selected, crop_dir, context_dir, sheet_dir, args)
    write_metadata(output_dir, image_dir, candidates, selected, args)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data


def make_label_dirs(output_dir: Path) -> None:
    for name in ("free", "occupied", "skip"):
        directory = output_dir / "labeled" / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / ".gitkeep").touch()


def build_slot_detector(config: dict[str, Any], args: argparse.Namespace) -> ParkingSlotDetector:
    detector_config = dict(config.get("parking_slot_detector", {}))
    if args.device:
        detector_config["device"] = args.device
    if args.slot_model_path:
        detector_config["model_path"] = args.slot_model_path
    if args.slot_conf is not None:
        detector_config["conf_threshold"] = args.slot_conf
    if args.detector_input_size is not None:
        detector_config["imgsz"] = args.detector_input_size
    return ParkingSlotDetector(detector_config)


def build_classifier(config: dict[str, Any], args: argparse.Namespace) -> EfficientNetOccupancyClassifier:
    classifier_config = dict(config.get("occupancy", {}).get("classifier", {}))
    if args.device:
        classifier_config["device"] = args.device
    if args.occupancy_model_path:
        classifier_config["model_path"] = args.occupancy_model_path
    if args.occupancy_threshold is not None:
        classifier_config["occupied_threshold"] = args.occupancy_threshold
    classifier_config["crop_size"] = args.crop_size
    return EfficientNetOccupancyClassifier(classifier_config)


def collect_candidates(
    image_paths: list[Path],
    slot_detector: ParkingSlotDetector,
    classifier: EfficientNetOccupancyClassifier | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for frame_idx, image_path in enumerate(tqdm(image_paths, desc="Collect occupancy review crops")):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue

        slots = slot_detector.detect(frame)
        predictions = classifier.predict(frame, slots) if classifier is not None else {}
        for slot in slots:
            status, confidence = predictions.get(slot.slot_id, ("unknown", 0.0))
            occupied_prob = occupied_probability(status, confidence)
            candidates.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": image_path.stem,
                    "image_path": str(image_path),
                    "slot_id": slot.slot_id,
                    "slot_confidence": float(slot.confidence),
                    "points": [[float(x), float(y)] for x, y in slot.points],
                    "classifier_status": status,
                    "classifier_confidence": float(confidence),
                    "occupied_probability": occupied_prob,
                    "uncertainty": abs(occupied_prob - 0.5) if occupied_prob is not None else 1.0,
                }
            )
    return candidates


def occupied_probability(status: str, confidence: float) -> float | None:
    if status == "occupied":
        return float(confidence)
    if status == "free":
        return 1.0 - float(confidence)
    return None


def select_candidates(candidates: list[dict[str, Any]], args: argparse.Namespace, rng: random.Random) -> list[dict[str, Any]]:
    if args.max_crops <= 0 or len(candidates) <= args.max_crops:
        selected = list(candidates)
    elif args.sample_strategy == "random":
        selected = rng.sample(candidates, args.max_crops)
    elif args.sample_strategy == "chronological":
        selected = candidates[: args.max_crops]
    elif args.sample_strategy == "balanced_pred":
        selected = balanced_by_predicted_status(candidates, args.max_crops, rng)
    else:
        selected = sorted(candidates, key=lambda item: (float(item["uncertainty"]), item["frame_idx"]))[: args.max_crops]

    selected.sort(key=lambda item: (item["frame_idx"], item["slot_id"]))
    for review_id, item in enumerate(selected, start=1):
        item["review_id"] = review_id
        item["review_name"] = f"{review_id:05d}_{item['timestamp']}_slot{int(item['slot_id']):03d}"
    return selected


def balanced_by_predicted_status(candidates: list[dict[str, Any]], max_crops: int, rng: random.Random) -> list[dict[str, Any]]:
    groups = {
        "free": [item for item in candidates if item["classifier_status"] == "free"],
        "occupied": [item for item in candidates if item["classifier_status"] == "occupied"],
        "unknown": [item for item in candidates if item["classifier_status"] not in {"free", "occupied"}],
    }
    for group in groups.values():
        group.sort(key=lambda item: (float(item["uncertainty"]), item["frame_idx"]))

    target_per_known_class = max_crops // 2
    selected = groups["free"][:target_per_known_class] + groups["occupied"][:target_per_known_class]
    remaining = [item for item in candidates if item not in selected]
    remaining.sort(key=lambda item: (float(item["uncertainty"]), item["frame_idx"]))
    selected.extend(remaining[: max(0, max_crops - len(selected))])
    if len(selected) > max_crops:
        selected = rng.sample(selected, max_crops)
    return selected


def write_review_files(
    selected: list[dict[str, Any]],
    crop_dir: Path,
    context_dir: Path,
    sheet_dir: Path,
    args: argparse.Namespace,
) -> None:
    sheet_tiles = []
    for item in tqdm(selected, desc="Write review files"):
        frame = cv2.imread(item["image_path"])
        if frame is None:
            item["write_error"] = f"Could not read image: {item['image_path']}"
            continue
        slot = record_to_slot(item)
        crop = crop_slot_bbox(frame, slot, args.crop_size)
        context = draw_context(frame, slot, item, args.context_size)

        crop_path = crop_dir / f"{item['review_name']}.jpg"
        context_path = context_dir / f"{item['review_name']}.jpg"
        cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(str(context_path), context, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        item["crop_path"] = str(crop_path)
        item["context_path"] = str(context_path)

        if len(sheet_tiles) < args.contact_sheet_limit:
            sheet_tiles.append(make_sheet_tile(crop, item))

    if sheet_tiles:
        sheet = build_contact_sheet(sheet_tiles, columns=5)
        cv2.imwrite(str(sheet_dir / "review_contact_sheet.jpg"), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def record_to_slot(item: dict[str, Any]) -> ParkingSlot:
    return ParkingSlot(
        slot_id=int(item["slot_id"]),
        points=[(float(x), float(y)) for x, y in item["points"]],
        confidence=float(item["slot_confidence"]),
        type="review",
    )


def draw_context(frame: np.ndarray, slot: ParkingSlot, item: dict[str, Any], context_size: int) -> np.ndarray:
    output = frame.copy()
    points = np.array(slot.points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(output, [points], isClosed=True, color=(0, 255, 255), thickness=3)
    for idx, (x, y) in enumerate(slot.points, start=1):
        cv2.circle(output, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.putText(output, str(idx), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    label = (
        f"{item['review_name']} pred={item['classifier_status']} "
        f"conf={float(item['classifier_confidence']):.2f} slot={float(item['slot_confidence']):.2f}"
    )
    cv2.rectangle(output, (8, 8), (min(output.shape[1] - 1, 8 + 10 * len(label)), 38), (0, 0, 0), -1)
    cv2.putText(output, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return resize_to_width(output, context_size)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or image.shape[1] == width:
        return image
    scale = width / image.shape[1]
    height = max(1, int(round(image.shape[0] * scale)))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def make_sheet_tile(crop: np.ndarray, item: dict[str, Any]) -> np.ndarray:
    tile = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_AREA)
    canvas = np.full((200, 160, 3), 255, dtype=np.uint8)
    canvas[:160] = tile
    label = f"{item['review_id']:05d} {item['classifier_status']} {float(item['classifier_confidence']):.2f}"
    cv2.putText(canvas, label[:22], (4, 184), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    return canvas


def build_contact_sheet(tiles: list[np.ndarray], columns: int) -> np.ndarray:
    columns = max(1, columns)
    rows = int(np.ceil(len(tiles) / columns))
    tile_h, tile_w = tiles[0].shape[:2]
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        row = idx // columns
        col = idx % columns
        sheet[row * tile_h : (row + 1) * tile_h, col * tile_w : (col + 1) * tile_w] = tile
    return sheet


def write_metadata(output_dir: Path, image_dir: Path, candidates: list[dict[str, Any]], selected: list[dict[str, Any]], args: argparse.Namespace) -> None:
    counts = Counter(item["classifier_status"] for item in selected)
    summary = {
        "image_dir": str(image_dir),
        "output_dir": str(output_dir),
        "frames_scanned": len({item["image_path"] for item in candidates}),
        "candidate_crops": len(candidates),
        "selected_crops": len(selected),
        "sample_strategy": args.sample_strategy,
        "classifier_status_counts": dict(counts),
        "instructions": [
            "Open context/ when the crop is ambiguous.",
            "Move each crop from unlabeled/ to labeled/free, labeled/occupied, or labeled/skip.",
            "Do not label uncertain or unreadable crops as free/occupied; put them into skip.",
            "Keep train/test split by frame later; do not randomly split individual crops.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({"records": selected}, indent=2), encoding="utf-8")
    write_manifest_csv(output_dir / "manifest.csv", selected)
    write_readme(output_dir / "README_LABELING.md")
    print(json.dumps(summary, indent=2))


def write_manifest_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "review_id",
        "review_name",
        "timestamp",
        "frame_idx",
        "slot_id",
        "crop_path",
        "context_path",
        "classifier_status",
        "classifier_confidence",
        "occupied_probability",
        "slot_confidence",
        "image_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record.get(name, "") for name in fieldnames})


def write_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# ParkRecon3D Occupancy Manual Labeling",
                "",
                "Разметка нужна для дообучения EfficientNet occupancy classifier под ParkRecon3D.",
                "",
                "Порядок работы:",
                "",
                "1. Открывай изображения из `unlabeled/`.",
                "2. Если по crop непонятно, смотри соответствующий файл в `context/`.",
                "3. Перемещай crop в одну из папок:",
                "   - `labeled/free/`",
                "   - `labeled/occupied/`",
                "   - `labeled/skip/`",
                "4. Спорные, обрезанные и нечитаемые случаи лучше класть в `skip/`.",
                "",
                "Важно: потом split нужно делать по `timestamp`, а не случайно по crop-файлам.",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
