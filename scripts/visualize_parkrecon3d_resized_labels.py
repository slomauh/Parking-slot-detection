from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm



IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize resized ParkRecon3D raw labels and prepared CRPS-D labels")
    parser.add_argument("--dataset-dir", default="outputs/parkrecon3d_bev_crpsd_format")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_resized_label_preview")
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--every-n", type=int, default=25)
    parser.add_argument("--contact-sheet-limit", type=int, default=60)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "splits": {},
    }

    for split in args.splits:
        split_summary = visualize_split(dataset_dir, output_dir, split, args)
        summary["splits"][split] = split_summary

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def visualize_split(dataset_dir: Path, output_dir: Path, split: str, args: argparse.Namespace) -> dict:
    image_dir = dataset_dir / "raw" / split / "img"
    raw_label_dir = dataset_dir / "raw" / split / "slot_label"
    prepared_dir = dataset_dir / "prepared" / split
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")
    if not raw_label_dir.exists():
        raise FileNotFoundError(f"Raw label dir not found: {raw_label_dir}")
    if not prepared_dir.exists():
        raise FileNotFoundError(f"Prepared label dir not found: {prepared_dir}")

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    selected_paths = image_paths[args.start_index :: max(1, args.every_n)]
    if args.limit is not None:
        selected_paths = selected_paths[: args.limit]

    preview_dir = output_dir / split / "preview"
    raw_dir = output_dir / split / "raw_resized"
    prepared_overlay_dir = output_dir / split / "prepared_overlay"
    side_by_side_dir = output_dir / split / "side_by_side"
    for directory in (preview_dir, raw_dir, prepared_overlay_dir, side_by_side_dir):
        directory.mkdir(parents=True, exist_ok=True)

    counts = {
        "available_images": len(image_paths),
        "selected_images": len(selected_paths),
        "written": 0,
        "missing_raw_labels": 0,
        "missing_prepared_labels": 0,
        "unreadable_images": 0,
        "raw_slots": 0,
        "raw_marks": 0,
        "prepared_marks": 0,
    }

    for image_path in tqdm(selected_paths, desc=f"Visualizing {split} resized labels"):
        raw_label_path = raw_label_dir / f"{image_path.stem}.json"
        prepared_label_path = prepared_dir / f"{image_path.stem}.json"
        if not raw_label_path.exists():
            counts["missing_raw_labels"] += 1
            continue
        if not prepared_label_path.exists():
            counts["missing_prepared_labels"] += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            counts["unreadable_images"] += 1
            continue

        raw_label = json.loads(raw_label_path.read_text(encoding="utf-8"))
        prepared_label = json.loads(prepared_label_path.read_text(encoding="utf-8"))
        raw_preview = draw_raw_resized_label(image, raw_label, image_path.stem)
        prepared_preview = draw_prepared_label(image, prepared_label, image_path.stem)
        side_by_side = cv2.hconcat([raw_preview, prepared_preview])

        cv2.imwrite(str(raw_dir / image_path.name), raw_preview, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
        cv2.imwrite(
            str(prepared_overlay_dir / image_path.name),
            prepared_preview,
            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
        )
        cv2.imwrite(
            str(side_by_side_dir / image_path.name),
            side_by_side,
            [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality],
        )
        cv2.imwrite(str(preview_dir / image_path.name), side_by_side, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])

        counts["written"] += 1
        counts["raw_slots"] += len(raw_label.get("slots", []))
        counts["raw_marks"] += len(raw_label.get("marks", []))
        counts["prepared_marks"] += len(prepared_label)

    make_contact_sheet(side_by_side_dir, output_dir / split / "contact_sheet.jpg", args.contact_sheet_limit)
    return counts


def draw_raw_resized_label(image: np.ndarray, label: dict, stem: str) -> np.ndarray:
    output = image.copy()
    marks = label.get("marks", [])
    slots = label.get("slots", [])

    for slot_idx, raw_slot in enumerate(slots, start=1):
        points = slot_points_from_raw_slot(marks, raw_slot)
        if points is None:
            continue
        polygon = np.array(points, dtype=np.int32)
        cv2.polylines(output, [polygon], isClosed=True, color=(0, 220, 0), thickness=2)
        x, y = polygon[0]
        cv2.putText(
            output,
            f"S{slot_idx}/t{safe_slot_type(raw_slot)}",
            (int(x), max(16, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )

    for mark_idx, mark in enumerate(marks, start=1):
        if not isinstance(mark, list) or len(mark) < 5:
            continue
        x0, y0, x1, y1, mark_type = mark[:5]
        p0 = (int(round(x0)), int(round(y0)))
        p1 = (int(round(x1)), int(round(y1)))
        color = mark_color(int(round(float(mark_type))))
        cv2.line(output, p0, p1, color, 2, cv2.LINE_AA)
        cv2.circle(output, p0, 3, color, -1, cv2.LINE_AA)
        cv2.putText(output, str(mark_idx), (p0[0] + 3, p0[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    draw_header(output, f"RAW resized GT | {stem} | slots={len(slots)} marks={len(marks)}")
    return output


def draw_prepared_label(image: np.ndarray, label: list, stem: str) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]

    for mark_idx, mark in enumerate(label, start=1):
        if not isinstance(mark, list) or len(mark) < 6:
            continue
        x_norm, y_norm, direction0, direction1, shape, mark_type = mark[:6]
        x = float(x_norm) * width
        y = float(y_norm) * height
        p0 = (int(round(x)), int(round(y)))
        color = mark_color(int(round(float(mark_type))))
        draw_direction(output, p0, float(direction0), 34, color, 2)
        draw_direction(output, p0, float(direction1), 24, (0, 180, 255), 1)
        cv2.circle(output, p0, 4, color, -1, cv2.LINE_AA)
        cv2.putText(
            output,
            f"{mark_idx}/t{int(round(float(mark_type)))}",
            (p0[0] + 4, p0[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    draw_header(output, f"PREPARED CRPS-D marks | {stem} | marks={len(label)}")
    draw_legend(output)
    return output


def slot_points_from_raw_slot(marks: list, raw_slot: list) -> list[tuple[float, float]] | None:
    if not isinstance(raw_slot, list) or len(raw_slot) < 2:
        return None
    try:
        mark_a_idx = int(raw_slot[0])
        mark_b_idx = int(raw_slot[1])
    except (TypeError, ValueError):
        return None
    if mark_a_idx < 1 or mark_b_idx < 1 or mark_a_idx > len(marks) or mark_b_idx > len(marks):
        return None
    mark_a = marks[mark_a_idx - 1]
    mark_b = marks[mark_b_idx - 1]
    if not isinstance(mark_a, list) or not isinstance(mark_b, list) or len(mark_a) < 4 or len(mark_b) < 4:
        return None
    return [
        (float(mark_a[0]), float(mark_a[1])),
        (float(mark_b[0]), float(mark_b[1])),
        (float(mark_b[2]), float(mark_b[3])),
        (float(mark_a[2]), float(mark_a[3])),
    ]


def safe_slot_type(raw_slot: list) -> str:
    if isinstance(raw_slot, list) and len(raw_slot) >= 3:
        return str(raw_slot[2])
    return "?"


def mark_color(mark_type: int) -> tuple[int, int, int]:
    colors = {
        0: (255, 120, 0),
        1: (0, 220, 255),
        2: (255, 0, 220),
        3: (180, 255, 0),
    }
    return colors.get(mark_type, (255, 255, 255))


def draw_direction(
    image: np.ndarray,
    origin: tuple[int, int],
    angle: float,
    length: int,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    end = (
        int(round(origin[0] + math.cos(angle) * length)),
        int(round(origin[1] + math.sin(angle) * length)),
    )
    cv2.arrowedLine(image, origin, end, color, thickness, cv2.LINE_AA, tipLength=0.28)


def draw_header(image: np.ndarray, text: str) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def draw_legend(image: np.ndarray) -> None:
    cv2.putText(
        image,
        "long arrow=direction0, short orange=direction1",
        (8, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def make_contact_sheet(input_dir: Path, output_path: Path, limit: int) -> None:
    image_paths = sorted(input_dir.glob("*.jpg"))[:limit]
    if not image_paths:
        return

    columns = 3
    tile_size = (512, 256)
    tiles = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        tile = cv2.resize(image, tile_size, interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            image_path.stem,
            (6, tile.shape[0] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
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
    cv2.imwrite(str(output_path), cv2.vconcat(rows), [int(cv2.IMWRITE_JPEG_QUALITY), 95])


if __name__ == "__main__":
    main()
