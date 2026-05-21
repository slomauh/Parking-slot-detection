from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


CAMERA_DIRS = {
    0: "Camera0",
    1: "Camera1",
    2: "Camera2",
    3: "Camera3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project ParkRecon3D BEV slot labels onto camera images")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--stitch-json", default="external/parkrecon3d_calibration/stitch.json")
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_projection_debug")
    parser.add_argument("--cameras", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--timestamps", nargs="+", help="Optional exact frame stems to project")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--mode", choices=("vehicle_to_camera", "camera_to_vehicle"), default="vehicle_to_camera")
    parser.add_argument("--bev-width", type=int, default=1354)
    parser.add_argument("--bev-height", type=int, default=1632)
    parser.add_argument("--bev-cx", type=float, default=676.5)
    parser.add_argument("--bev-cy", type=float, default=815.5)
    parser.add_argument("--bev-meter-per-pixel", type=float, default=0.0105)
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument("--swap-xy", action="store_true")
    parser.add_argument("--draw-all-marks", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_params = load_camera_params(Path(args.stitch_json))
    label_paths = sorted((dataset_root / "BEV" / "Data" / "label").glob("*.json"), key=lambda path: int(path.stem))
    if args.timestamps:
        timestamp_set = set(args.timestamps)
        label_paths = [path for path in label_paths if path.stem in timestamp_set]
    elif args.limit:
        label_paths = label_paths[: args.limit]

    summary = []
    for label_path in label_paths:
        label = json.loads(label_path.read_text(encoding="utf-8"))
        mark_segments = collect_mark_segments(label, args.draw_all_marks)
        vehicle_points = bev_segments_to_vehicle_points(mark_segments, args)

        for camera_id in args.cameras:
            camera_dir = CAMERA_DIRS[camera_id]
            image_path = dataset_root / camera_dir / "Data" / "Image" / f"{label_path.stem}.jpg"
            image = cv2.imread(str(image_path))
            if image is None:
                continue

            projected_segments, stats = project_segments(vehicle_points, camera_params[camera_id], args.mode, image.shape)
            draw_segments(image, projected_segments)
            out_path = output_dir / f"{label_path.stem}_{camera_dir}_{args.mode}.jpg"
            cv2.imwrite(str(out_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            summary.append(
                {
                    "timestamp": label_path.stem,
                    "camera": camera_dir,
                    "mode": args.mode,
                    "input_segments": len(mark_segments),
                    "visible_segments": stats["visible_segments"],
                    "projected_points": stats["projected_points"],
                    "output": str(out_path),
                }
            )

    (output_dir / f"summary_{args.mode}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def load_camera_params(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    params = {}
    for row in data["camParams"]:
        camera_id = int(row["camId"])
        params[camera_id] = {
            "K": np.array(row["oriIntrinsics"], dtype=np.float64).reshape(3, 3),
            "D": np.array(row["Distortion"], dtype=np.float64).reshape(4, 1),
            "R": np.array(row["Extrinsics"], dtype=np.float64).reshape(3, 4)[:, :3],
            "t": np.array(row["Extrinsics"], dtype=np.float64).reshape(3, 4)[:, 3:4],
        }
    return params


def collect_mark_segments(label: dict, draw_all_marks: bool) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    marks = label.get("marks", [])
    if draw_all_marks:
        return [((float(mark[0]), float(mark[1])), (float(mark[2]), float(mark[3]))) for mark in marks if len(mark) >= 4]

    selected_mark_ids = set()
    for slot in label.get("slots", []):
        if len(slot) >= 2:
            selected_mark_ids.add(int(slot[0]) - 1)
            selected_mark_ids.add(int(slot[1]) - 1)

    segments = []
    for mark_index in sorted(selected_mark_ids):
        if mark_index < 0 or mark_index >= len(marks):
            continue
        mark = marks[mark_index]
        if len(mark) >= 4:
            segments.append(((float(mark[0]), float(mark[1])), (float(mark[2]), float(mark[3]))))
    return segments


def bev_segments_to_vehicle_points(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    args: argparse.Namespace,
) -> np.ndarray:
    points = []
    for start, end in segments:
        for u, v in (start, end):
            x = (u - args.bev_cx) * args.bev_meter_per_pixel
            y = (args.bev_cy - v) * args.bev_meter_per_pixel
            if args.flip_x:
                x = -x
            if args.flip_y:
                y = -y
            if args.swap_xy:
                x, y = y, x
            points.append([x, y, 0.0])
    return np.array(points, dtype=np.float64).reshape(-1, 1, 3)


def project_segments(
    vehicle_points: np.ndarray,
    camera_param: dict,
    mode: str,
    image_shape: tuple[int, int, int],
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], dict]:
    if len(vehicle_points) == 0:
        return [], {"visible_segments": 0, "projected_points": 0}

    K = camera_param["K"]
    D = camera_param["D"]
    R = camera_param["R"]
    t = camera_param["t"]

    if mode == "vehicle_to_camera":
        rvec, _ = cv2.Rodrigues(np.ascontiguousarray(R))
        image_points, _ = cv2.fisheye.projectPoints(
            np.ascontiguousarray(vehicle_points),
            np.ascontiguousarray(rvec),
            np.ascontiguousarray(t),
            K,
            D,
        )
    else:
        flat_points = vehicle_points.reshape(-1, 3).T
        camera_points = (R.T @ (flat_points - t)).T.reshape(-1, 1, 3)
        image_points, _ = cv2.fisheye.projectPoints(
            camera_points,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            K,
            D,
        )

    image_points = image_points.reshape(-1, 2)
    height, width = image_shape[:2]
    segments = []
    visible_segments = 0
    for index in range(0, len(image_points), 2):
        p0 = image_points[index]
        p1 = image_points[index + 1]
        if not np.all(np.isfinite(p0)) or not np.all(np.isfinite(p1)):
            continue
        p0_tuple = (int(round(p0[0])), int(round(p0[1])))
        p1_tuple = (int(round(p1[0])), int(round(p1[1])))
        segments.append((p0_tuple, p1_tuple))
        if point_in_image(p0_tuple, width, height) or point_in_image(p1_tuple, width, height):
            visible_segments += 1

    return segments, {"visible_segments": visible_segments, "projected_points": len(image_points)}


def point_in_image(point: tuple[int, int], width: int, height: int) -> bool:
    x, y = point
    return 0 <= x < width and 0 <= y < height


def draw_segments(image: np.ndarray, segments: list[tuple[tuple[int, int], tuple[int, int]]]) -> None:
    height, width = image.shape[:2]
    for start, end in segments:
        if not line_maybe_visible(start, end, width, height):
            continue
        cv2.line(image, start, end, (0, 255, 255), 3, lineType=cv2.LINE_AA)
        cv2.circle(image, start, 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(image, end, 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)


def line_maybe_visible(start: tuple[int, int], end: tuple[int, int], width: int, height: int) -> bool:
    xs = [start[0], end[0]]
    ys = [start[1], end[1]]
    if max(xs) < 0 or min(xs) >= width:
        return False
    if max(ys) < 0 or min(ys) >= height:
        return False
    return math.hypot(end[0] - start[0], end[1] - start[1]) < max(width, height) * 3


if __name__ == "__main__":
    main()
