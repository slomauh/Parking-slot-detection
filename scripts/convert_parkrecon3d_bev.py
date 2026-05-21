from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ParkRecon3D BEV to CRPS-D-like training/eval datasets")
    parser.add_argument(
        "--dataset-root",
        action="append",
        dest="dataset_root",
        help="ParkRecon3D part root. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        help="ParkRecon3D part roots. Alternative to repeated --dataset-root.",
    )
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_bev_crpsd_format")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--gap-size", type=int, default=30, help="Frames dropped between train and test for chronological split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_roots = resolve_dataset_roots(args)
    output_dir = Path(args.output_dir)

    pairs, duplicate_count = collect_pairs(dataset_roots)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise FileNotFoundError(f"No image/label pairs found in {dataset_roots}")

    train_pairs, val_pairs, gap_pairs = split_pairs(pairs, args.val_ratio, args.split_strategy, args.gap_size, args.seed)

    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_raw = output_dir / "raw" / "train"
    val_raw = output_dir / "raw" / "test"
    train_prepared = output_dir / "prepared" / "train"
    val_prepared = output_dir / "prepared" / "test"
    for directory in (
        train_raw / "img",
        train_raw / "slot_label",
        val_raw / "img",
        val_raw / "slot_label",
        train_prepared,
        val_prepared,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    train_stats = convert_split(train_pairs, train_raw, train_prepared, args.image_size, args.jpeg_quality)
    val_stats = convert_split(val_pairs, val_raw, val_prepared, args.image_size, args.jpeg_quality)

    summary = {
        "dataset_roots": [str(path) for path in dataset_roots],
        "output_dir": str(output_dir),
        "image_size": args.image_size,
        "val_ratio": args.val_ratio,
        "split_strategy": args.split_strategy,
        "gap_size": args.gap_size,
        "seed": args.seed,
        "total_pairs": len(pairs),
        "duplicate_pairs_dropped": duplicate_count,
        "dropped_gap_pairs": len(gap_pairs),
        "gap_range": stem_range(gap_pairs),
        "train_range": stem_range(train_pairs),
        "test_range": stem_range(val_pairs),
        "train": train_stats,
        "test": val_stats,
        "raw_format": {
            "train_images": str(train_raw / "img"),
            "train_labels": str(train_raw / "slot_label"),
            "test_images": str(val_raw / "img"),
            "test_labels": str(val_raw / "slot_label"),
        },
        "prepared_format": {
            "train": str(train_prepared),
            "test": str(val_prepared),
            "note": "Each .json is a list of generalized marking points expected by external/CRPS-D/data/dataset.py.",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def resolve_dataset_roots(args: argparse.Namespace) -> list[Path]:
    raw_roots = []
    list_roots = getattr(args, "dataset_roots", None)
    repeated_roots = getattr(args, "dataset_root", None)
    if list_roots:
        raw_roots.extend(list_roots)
    if repeated_roots:
        raw_roots.extend(repeated_roots)
    if not raw_roots:
        raw_roots = ["/home/slomauh/Downloads/data1"]

    resolved = []
    seen = set()
    for root in raw_roots:
        path = Path(root).expanduser().resolve()
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return resolved


def collect_pairs(dataset_roots: list[Path]) -> tuple[list[tuple[Path, Path]], int]:
    pairs_by_stem: dict[str, tuple[Path, Path]] = {}
    duplicate_count = 0
    for dataset_root in dataset_roots:
        image_dir = dataset_root / "BEV" / "Data" / "Image"
        label_dir = dataset_root / "BEV" / "Data" / "label"
        for image_path in sorted(image_dir.glob("*.jpg")):
            label_path = label_dir / f"{image_path.stem}.json"
            if not label_path.exists():
                continue
            if image_path.stem in pairs_by_stem:
                duplicate_count += 1
                continue
            pairs_by_stem[image_path.stem] = (image_path, label_path)
    pairs = [pairs_by_stem[stem] for stem in sorted(pairs_by_stem, key=int)]
    return pairs, duplicate_count


def split_pairs(
    pairs: list[tuple[Path, Path]],
    val_ratio: float,
    split_strategy: str,
    gap_size: int,
    seed: int,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    val_count = int(round(len(pairs) * val_ratio))
    if val_count <= 0:
        return pairs, [], []

    if split_strategy == "random":
        rng = random.Random(seed)
        shuffled_pairs = pairs.copy()
        rng.shuffle(shuffled_pairs)
        return shuffled_pairs[val_count:], shuffled_pairs[:val_count], []

    split_start = max(0, len(pairs) - val_count)
    gap_start = max(0, split_start - max(0, gap_size))
    train_pairs = pairs[:gap_start]
    gap_pairs = pairs[gap_start:split_start]
    val_pairs = pairs[split_start:]
    return train_pairs, val_pairs, gap_pairs


def stem_range(pairs: list[tuple[Path, Path]]) -> dict[str, str | None]:
    if not pairs:
        return {"first": None, "last": None}
    return {"first": pairs[0][0].stem, "last": pairs[-1][0].stem}


def convert_split(
    pairs: list[tuple[Path, Path]],
    raw_split_dir: Path,
    prepared_split_dir: Path,
    image_size: int,
    jpeg_quality: int,
) -> dict:
    stats = {
        "images": 0,
        "slots": 0,
        "marks": 0,
        "skipped_images": 0,
        "skipped_marks": 0,
    }

    for image_path, label_path in tqdm(pairs, desc=f"Converting {raw_split_dir.name}"):
        image = cv2.imread(str(image_path))
        if image is None:
            stats["skipped_images"] += 1
            continue

        label = json.loads(label_path.read_text(encoding="utf-8"))
        resized_image, scale_x, scale_y = resize_to_square(image, image_size)
        converted_label = convert_raw_label(label, scale_x, scale_y, image_size)
        generalized_marks, skipped_marks = generalize_marks(converted_label["marks"], image_size)

        stats["images"] += 1
        stats["slots"] += len(converted_label["slots"])
        stats["marks"] += len(generalized_marks)
        stats["skipped_marks"] += skipped_marks

        raw_image_path = raw_split_dir / "img" / image_path.name
        raw_label_path = raw_split_dir / "slot_label" / f"{image_path.stem}.json"
        prepared_image_path = prepared_split_dir / image_path.name
        prepared_label_path = prepared_split_dir / f"{image_path.stem}.json"

        cv2.imwrite(str(raw_image_path), resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        cv2.imwrite(str(prepared_image_path), resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        raw_label_path.write_text(json.dumps(converted_label), encoding="utf-8")
        prepared_label_path.write_text(json.dumps(generalized_marks), encoding="utf-8")

    return stats


def resize_to_square(image, image_size: int):
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return resized_image, image_size / width, image_size / height


def convert_raw_label(label: dict, scale_x: float, scale_y: float, image_size: int) -> dict:
    marks = []
    old_to_new_index: dict[int, int] = {}
    for old_index, mark in enumerate(label.get("marks", []), start=1):
        if not isinstance(mark, list) or len(mark) < 5:
            continue
        x0 = clamp(float(mark[0]) * scale_x, 0.0, image_size - 1.0)
        y0 = clamp(float(mark[1]) * scale_y, 0.0, image_size - 1.0)
        x1 = clamp(float(mark[2]) * scale_x, 0.0, image_size - 1.0)
        y1 = clamp(float(mark[3]) * scale_y, 0.0, image_size - 1.0)
        mark_type = int(round(float(mark[4])))
        marks.append([x0, y0, x1, y1, mark_type])
        old_to_new_index[old_index] = len(marks)

    slots = []
    for slot in label.get("slots", []):
        if not isinstance(slot, list) or len(slot) < 4:
            continue
        mark_a = old_to_new_index.get(int(slot[0]))
        mark_b = old_to_new_index.get(int(slot[1]))
        if mark_a is None or mark_b is None:
            continue
        slots.append([mark_a, mark_b, int(round(float(slot[2]))), float(slot[3])])

    return {"marks": marks, "slots": slots}


def generalize_marks(marks: list[list[float]], image_size: int) -> tuple[list[list[float]], int]:
    generalized_marks = []
    skipped = 0
    for mark in marks:
        if len(mark) < 5:
            skipped += 1
            continue
        x0, y0, x1, y1, mark_type = mark
        if not (0.0 <= x0 < image_size and 0.0 <= y0 < image_size):
            skipped += 1
            continue

        direction0 = math.atan2(y1 - y0, x1 - x0)
        direction1 = normalize_angle(direction0 + math.pi / 2)
        xval = x0 / image_size
        yval = y0 / image_size
        shape = 0.0
        generalized_marks.append([xval, yval, direction0, direction1, shape, float(mark_type)])

    return generalized_marks, skipped


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle <= -math.pi:
        angle += 2 * math.pi
    return angle


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


if __name__ == "__main__":
    main()
