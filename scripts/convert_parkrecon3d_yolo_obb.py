from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml
from shapely.geometry import Polygon as ShapelyPolygon
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ParkRecon3D BEV raw labels to Ultralytics YOLO-OBB format")
    parser.add_argument("--raw-dir", default="outputs/parkrecon3d_bev_crpsd_format/raw")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_yolo_obb")
    parser.add_argument("--class-name", default="parking_slot")
    parser.add_argument("--preview-limit", type=int, default=50)
    parser.add_argument("--jpeg-quality", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    stats = {
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "class_name": args.class_name,
        "splits": {},
    }
    for split_name, yolo_split in (("train", "train"), ("test", "val")):
        split_stats = convert_split(raw_dir / split_name, output_dir, yolo_split, args)
        stats["splits"][yolo_split] = split_stats

    data_yaml = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": {0: args.class_name},
    }
    (output_dir / "parkrecon3d_obb.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def convert_split(raw_split_dir: Path, output_dir: Path, yolo_split: str, args: argparse.Namespace) -> dict:
    image_dir = raw_split_dir / "img"
    label_dir = raw_split_dir / "slot_label"
    output_image_dir = output_dir / "images" / yolo_split
    output_label_dir = output_dir / "labels" / yolo_split
    preview_dir = output_dir / "preview" / yolo_split
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    counts: Counter[str] = Counter()
    image_paths = sorted(image_dir.glob("*.jpg"), key=lambda path: int(path.stem))
    for image_path in tqdm(image_paths, desc=f"Converting {yolo_split}"):
        label_path = label_dir / f"{image_path.stem}.json"
        if not label_path.exists():
            counts["missing_labels"] += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            counts["unreadable_images"] += 1
            continue
        height, width = image.shape[:2]

        polygons = load_slot_polygons(label_path)
        yolo_rows = []
        valid_polygons = []
        for polygon in polygons:
            clipped = clip_polygon(polygon, width, height)
            if not valid_polygon(clipped):
                counts["invalid_slots"] += 1
                continue
            normalized = normalize_polygon(clipped, width, height)
            yolo_rows.append("0 " + " ".join(f"{value:.8f}" for point in normalized for value in point))
            valid_polygons.append(clipped)

        shutil.copy2(image_path, output_image_dir / image_path.name)
        (output_label_dir / f"{image_path.stem}.txt").write_text("\n".join(yolo_rows) + ("\n" if yolo_rows else ""), encoding="utf-8")

        counts["images"] += 1
        counts["slots"] += len(valid_polygons)
        if len(valid_polygons) != len(polygons):
            counts["images_with_invalid_slots"] += 1

        if counts["preview_images"] < args.preview_limit:
            preview = draw_preview(image, valid_polygons)
            cv2.imwrite(
                str(preview_dir / image_path.name),
                preview,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
            )
            counts["preview_images"] += 1

    return dict(counts)


def load_slot_polygons(label_path: Path) -> list[list[tuple[float, float]]]:
    data = json.loads(label_path.read_text(encoding="utf-8"))
    marks = data.get("marks", [])
    polygons = []
    for slot in data.get("slots", []):
        if not isinstance(slot, list) or len(slot) < 2:
            continue
        mark_a_idx = int(slot[0])
        mark_b_idx = int(slot[1])
        if mark_a_idx < 1 or mark_b_idx < 1 or mark_a_idx > len(marks) or mark_b_idx > len(marks):
            continue
        mark_a = marks[mark_a_idx - 1]
        mark_b = marks[mark_b_idx - 1]
        if len(mark_a) < 4 or len(mark_b) < 4:
            continue
        polygons.append(
            [
                (float(mark_a[0]), float(mark_a[1])),
                (float(mark_b[0]), float(mark_b[1])),
                (float(mark_b[2]), float(mark_b[3])),
                (float(mark_a[2]), float(mark_a[3])),
            ]
        )
    return polygons


def clip_polygon(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[float, float]]:
    return [
        (
            min(max(float(x), 0.0), float(width - 1)),
            min(max(float(y), 0.0), float(height - 1)),
        )
        for x, y in points
    ]


def normalize_polygon(points: list[tuple[float, float]], width: int, height: int) -> list[tuple[float, float]]:
    return [(float(x) / width, float(y) / height) for x, y in points]


def valid_polygon(points: list[tuple[float, float]]) -> bool:
    polygon = ShapelyPolygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return bool(not polygon.is_empty and polygon.area > 1.0)


def draw_preview(image, polygons: list[list[tuple[float, float]]]):
    output = image.copy()
    for idx, points in enumerate(polygons, start=1):
        polygon = np.array(points, dtype=np.int32)
        cv2.polylines(output, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = polygon[0]
        cv2.putText(
            output,
            f"slot {idx}",
            (int(x), max(16, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return output


if __name__ == "__main__":
    main()
