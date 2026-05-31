from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


CLASSES = ("free", "occupied")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build leakage-safe ParkRecon3D occupancy train/val split")
    parser.add_argument("--review-dir", default="outputs/parkrecon3d_occupancy_manual_review")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_occupancy_finetune_dataset")
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--gap-timestamps", type=int, default=5, help="Unique timestamp gap between train and val")
    parser.add_argument("--min-val-per-class", type=int, default=1)
    parser.add_argument("--copy-mode", choices=("copy", "symlink"), default="copy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_dir = Path(args.review_dir)
    output_dir = Path(args.output_dir)
    records = collect_labeled_records(review_dir)
    if not records:
        raise FileNotFoundError(f"No labeled crops found in {review_dir}/labeled/free|occupied")

    split = chronological_split(records, args.val_ratio, args.gap_timestamps, args.min_val_per_class)
    write_split(split, output_dir, args.copy_mode)
    write_metadata(records, split, review_dir, output_dir, args)


def collect_labeled_records(review_dir: Path) -> list[dict]:
    records = []
    for class_name in CLASSES:
        class_dir = review_dir / "labeled" / class_name
        for path in sorted(class_dir.glob("*.jpg")):
            timestamp = parse_timestamp(path.name)
            if timestamp is None:
                continue
            records.append(
                {
                    "path": path,
                    "filename": path.name,
                    "class_name": class_name,
                    "timestamp": timestamp,
                    "review_id": path.name.split("_", 1)[0],
                }
            )
    records.sort(key=lambda item: (int(item["timestamp"]), item["class_name"], item["filename"]))
    return records


def parse_timestamp(filename: str) -> str | None:
    parts = filename.split("_")
    if len(parts) < 3:
        return None
    timestamp = parts[1]
    return timestamp if timestamp.isdigit() else None


def chronological_split(records: list[dict], val_ratio: float, gap_timestamps: int, min_val_per_class: int) -> dict[str, list[dict]]:
    timestamps = sorted({record["timestamp"] for record in records}, key=int)
    if len(timestamps) < 3:
        raise ValueError("Need at least 3 unique timestamps for chronological train/val split")

    val_timestamp_count = max(1, int(round(len(timestamps) * val_ratio)))
    val_start_idx = max(1, len(timestamps) - val_timestamp_count)
    gap_start_idx = max(0, val_start_idx - max(0, gap_timestamps))
    train_timestamps = set(timestamps[:gap_start_idx])
    gap_timestamps_set = set(timestamps[gap_start_idx:val_start_idx])
    val_timestamps = set(timestamps[val_start_idx:])

    split = {
        "train": [record for record in records if record["timestamp"] in train_timestamps],
        "val": [record for record in records if record["timestamp"] in val_timestamps],
        "gap": [record for record in records if record["timestamp"] in gap_timestamps_set],
    }

    val_counts = Counter(record["class_name"] for record in split["val"])
    train_counts = Counter(record["class_name"] for record in split["train"])
    for class_name in CLASSES:
        if val_counts[class_name] < min_val_per_class:
            raise ValueError(
                f"Validation split has only {val_counts[class_name]} '{class_name}' samples. "
                "Label more later-frame examples or lower --min-val-per-class."
            )
        if train_counts[class_name] < 1:
            raise ValueError(f"Train split has no '{class_name}' samples")
    return split


def write_split(split: dict[str, list[dict]], output_dir: Path, copy_mode: str) -> None:
    for split_name in ("train", "val"):
        for class_name in CLASSES:
            target_dir = output_dir / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for old_file in target_dir.glob("*.jpg"):
                old_file.unlink()

    for split_name in ("train", "val"):
        for record in split[split_name]:
            target = output_dir / split_name / record["class_name"] / record["filename"]
            if copy_mode == "symlink":
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(record["path"].resolve())
            else:
                shutil.copy2(record["path"], target)


def write_metadata(
    records: list[dict],
    split: dict[str, list[dict]],
    review_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    summary = {
        "review_dir": str(review_dir),
        "output_dir": str(output_dir),
        "total_labeled": len(records),
        "unique_timestamps": len({record["timestamp"] for record in records}),
        "val_ratio": args.val_ratio,
        "gap_timestamps": args.gap_timestamps,
        "copy_mode": args.copy_mode,
        "splits": {
            split_name: {
                "samples": len(split_records),
                "unique_timestamps": len({record["timestamp"] for record in split_records}),
                "class_counts": dict(Counter(record["class_name"] for record in split_records)),
            }
            for split_name, split_records in split.items()
        },
        "note": "Train/val are split by timestamp; gap records are intentionally not used for training.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "split_manifest.csv", split)
    print(json.dumps(summary, indent=2))


def write_csv(path: Path, split: dict[str, list[dict]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["split", "class_name", "timestamp", "filename", "path"])
        writer.writeheader()
        for split_name, records in split.items():
            for record in records:
                writer.writerow(
                    {
                        "split": split_name,
                        "class_name": record["class_name"],
                        "timestamp": record["timestamp"],
                        "filename": record["filename"],
                        "path": str(record["path"]),
                    }
                )


if __name__ == "__main__":
    main()
