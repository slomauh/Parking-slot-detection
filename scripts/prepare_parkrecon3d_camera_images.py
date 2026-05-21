from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ParkRecon3D camera image-only train/test splits")
    parser.add_argument("--dataset-roots", nargs="+", required=True)
    parser.add_argument("--cameras", nargs="+", default=["Camera0", "Camera1", "Camera2"])
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_camera_images")
    parser.add_argument("--image-size", type=int, help="Optional square resize size")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--split-strategy", choices=("chronological", "random"), default="chronological")
    parser.add_argument("--gap-size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_roots = [Path(root).expanduser().resolve() for root in args.dataset_roots]
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_roots": [str(root) for root in dataset_roots],
        "cameras": args.cameras,
        "output_dir": str(output_dir),
        "image_size": args.image_size,
        "val_ratio": args.val_ratio,
        "split_strategy": args.split_strategy,
        "gap_size": args.gap_size,
        "seed": args.seed,
        "note": "Camera folders do not contain slot labels here. These are image-only splits, not supervised slot-detector training data.",
        "splits": {},
    }

    for camera in args.cameras:
        pairs, duplicate_count = collect_images(dataset_roots, camera)
        if not pairs:
            raise FileNotFoundError(f"No images found for {camera} in {dataset_roots}")
        train_pairs, test_pairs, gap_pairs = split_pairs(pairs, args.val_ratio, args.split_strategy, args.gap_size, args.seed)
        camera_dir = output_dir / camera
        train_stats = write_split(train_pairs, camera_dir / "train" / "img", args.image_size, args.jpeg_quality, f"{camera} train")
        test_stats = write_split(test_pairs, camera_dir / "test" / "img", args.image_size, args.jpeg_quality, f"{camera} test")
        summary["splits"][camera] = {
            "total_pairs": len(pairs),
            "duplicate_pairs_dropped": duplicate_count,
            "dropped_gap_pairs": len(gap_pairs),
            "train_range": stem_range(train_pairs),
            "test_range": stem_range(test_pairs),
            "gap_range": stem_range(gap_pairs),
            "train": train_stats,
            "test": test_stats,
        }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def collect_images(dataset_roots: list[Path], camera: str) -> tuple[list[Path], int]:
    images_by_stem: dict[str, Path] = {}
    duplicate_count = 0
    for root in dataset_roots:
        image_dir = root / camera / "Data" / "Image"
        for image_path in sorted(image_dir.glob("*.jpg")):
            if image_path.stem in images_by_stem:
                duplicate_count += 1
                continue
            images_by_stem[image_path.stem] = image_path
    return [images_by_stem[stem] for stem in sorted(images_by_stem, key=int)], duplicate_count


def split_pairs(
    pairs: list[Path],
    val_ratio: float,
    split_strategy: str,
    gap_size: int,
    seed: int,
) -> tuple[list[Path], list[Path], list[Path]]:
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
    test_pairs = pairs[split_start:]
    return train_pairs, test_pairs, gap_pairs


def write_split(paths: list[Path], output_dir: Path, image_size: int | None, jpeg_quality: int, desc: str) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "skipped_images": 0}
    for path in tqdm(paths, desc=desc):
        image = cv2.imread(str(path))
        if image is None:
            stats["skipped_images"] += 1
            continue
        if image_size is not None:
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output_dir / path.name), image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        stats["images"] += 1
    return stats


def stem_range(paths: list[Path]) -> dict[str, str | None]:
    if not paths:
        return {"first": None, "last": None}
    return {"first": paths[0].stem, "last": paths[-1].stem}


if __name__ == "__main__":
    main()
