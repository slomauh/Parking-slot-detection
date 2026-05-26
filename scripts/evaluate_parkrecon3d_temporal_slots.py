from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_parkrecon3d_bev import build_metrics, detect_slots, draw_result, load_parkrecon3d_slots, match_slots
from src.detection.parking_slot_detector import ParkingSlotDetector
from src.detection.schemas import ParkingSlot


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SLOT_TEMPORAL_METADATA: dict[int, dict[str, Any]] = {}


@dataclass
class TemporalTrack:
    track_id: int
    slot: ParkingSlot
    history: deque[bool]
    confidence_sum: float
    total_hits: int = 1
    track_age: int = 1
    missed_frames: int = 0
    last_seen_frame: int = 0
    last_confidence: float = 0.0
    consecutive_hits: int = 1
    consecutive_misses: int = 0
    filled_gap_count: int = 0
    last_match_score: float = 0.0
    was_filled_gap: bool = False

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / max(1, self.total_hits)


@dataclass
class TemporalVariant:
    mode: str
    window: int
    min_hits: int
    max_missing: int
    track_iou: float
    max_center_distance: float
    min_track_score: float = 0.0
    min_track_age: int = 1
    min_consecutive_hits: int = 1
    fill_gap_min_confidence: float = 0.0
    decay_on_miss: float = 1.0
    max_gap_fill: int = 0
    key: str = field(init=False)

    def __post_init__(self) -> None:
        base = (
            f"{self.mode}_w{self.window}_h{self.min_hits}_miss{self.max_missing}"
            f"_iou{self.track_iou:.2f}_dist{self.max_center_distance:.0f}"
        )
        if self.mode == "quality_fill_gap":
            base += (
                f"_score{self.min_track_score:.2f}_age{self.min_track_age}"
                f"_ch{self.min_consecutive_hits}_fgconf{self.fill_gap_min_confidence:.2f}"
                f"_decay{self.decay_on_miss:.2f}_gap{self.max_gap_fill}"
            )
        self.key = base


class TemporalSlotSmoother:
    def __init__(self, variant: TemporalVariant) -> None:
        self.variant = variant
        self.next_track_id = 1
        self.tracks: dict[int, TemporalTrack] = {}

    def update(self, frame_idx: int, detections: list[ParkingSlot]) -> list[ParkingSlot]:
        if self.variant.mode == "none":
            return [clone_slot(slot, slot.slot_id) for slot in detections]

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        candidates = []
        for track_id, track in self.tracks.items():
            for detection_idx, detection in enumerate(detections):
                iou = polygon_iou(track.slot.points, detection.points)
                center_distance = polygon_center_distance(track.slot.points, detection.points)
                if iou >= self.variant.track_iou or center_distance <= self.variant.max_center_distance:
                    score = self.match_score(iou, center_distance, detection.confidence)
                    candidates.append((score, iou, center_distance, track_id, detection_idx))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for score, _, _, track_id, detection_idx in candidates:
            if track_id in matched_tracks or detection_idx in matched_detections:
                continue
            detection = detections[detection_idx]
            track = self.tracks[track_id]
            track.slot = clone_slot(detection, track_id)
            track.history.append(True)
            track.total_hits += 1
            track.track_age += 1
            track.confidence_sum += float(detection.confidence)
            track.missed_frames = 0
            track.consecutive_hits += 1
            track.consecutive_misses = 0
            track.filled_gap_count = 0
            track.last_confidence = float(detection.confidence)
            track.last_match_score = score
            track.was_filled_gap = False
            track.last_seen_frame = frame_idx
            matched_tracks.add(track_id)
            matched_detections.add(detection_idx)

        for detection_idx, detection in enumerate(detections):
            if detection_idx in matched_detections:
                continue
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = TemporalTrack(
                track_id=track_id,
                slot=clone_slot(detection, track_id),
                history=deque([True], maxlen=self.variant.window),
                confidence_sum=float(detection.confidence),
                total_hits=1,
                track_age=1,
                missed_frames=0,
                last_seen_frame=frame_idx,
                last_confidence=float(detection.confidence),
                last_match_score=float(detection.confidence),
            )
            matched_tracks.add(track_id)

        for track_id in list(self.tracks):
            track = self.tracks[track_id]
            if track_id not in matched_tracks:
                track.history.append(False)
                track.track_age += 1
                track.missed_frames += 1
                track.consecutive_hits = 0
                track.consecutive_misses += 1
                track.last_match_score *= self.variant.decay_on_miss
                track.was_filled_gap = True
            if track.missed_frames > self.variant.max_missing:
                del self.tracks[track_id]

        return self.active_slots()

    def match_score(self, iou: float, center_distance: float, confidence: float) -> float:
        center_score = max(0.0, 1.0 - center_distance / max(1.0, self.variant.max_center_distance))
        return (0.45 * iou) + (0.35 * center_score) + (0.20 * float(confidence))

    def active_slots(self) -> list[ParkingSlot]:
        slots = []
        for track in sorted(self.tracks.values(), key=lambda item: item.track_id):
            recent_hits = sum(track.history)
            confirmed = recent_hits >= self.variant.min_hits
            if not confirmed:
                continue
            if self.variant.mode == "confirm" and track.missed_frames == 0:
                slots.append(track_to_slot(track, filled_gap=False))
            elif self.variant.mode == "fill_gap" and track.missed_frames <= self.variant.max_missing:
                filled_gap = track.missed_frames > 0
                if filled_gap:
                    track.filled_gap_count += 1
                slots.append(track_to_slot(track, filled_gap=filled_gap))
            elif self.variant.mode == "quality_fill_gap" and self.is_quality_track(track):
                filled_gap = track.missed_frames > 0
                if filled_gap:
                    if track.filled_gap_count >= self.variant.max_gap_fill:
                        continue
                    track.filled_gap_count += 1
                slots.append(track_to_slot(track, filled_gap=filled_gap))
        return slots

    def is_quality_track(self, track: TemporalTrack) -> bool:
        track_score = temporal_track_score(track)
        if track.track_age < self.variant.min_track_age:
            return False
        if track_score < self.variant.min_track_score:
            return False
        if track.missed_frames == 0:
            return track.consecutive_hits >= self.variant.min_consecutive_hits or sum(track.history) >= self.variant.min_hits
        if track.missed_frames > self.variant.max_missing:
            return False
        if track.last_confidence < self.variant.fill_gap_min_confidence:
            return False
        return track.total_hits >= self.variant.min_hits and track.consecutive_misses <= self.variant.max_gap_fill


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate temporal slot smoothing on chronological ParkRecon3D BEV frames")
    parser.add_argument("--image-dir", default="outputs/parkrecon3d_bev_crpsd_format/raw/test/img")
    parser.add_argument("--label-dir", default="outputs/parkrecon3d_bev_crpsd_format/raw/test/slot_label")
    parser.add_argument("--slot-model-path", default="models/slot_detector/parkrecon3d_slot_detector_finetuned.pth")
    parser.add_argument("--slot-external-repo-path", default="external/CRPS-D")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--detector-input-size", type=int, default=512)
    parser.add_argument("--slot-conf", type=float, default=0.30)
    parser.add_argument("--slot-min-score", type=float, default=0.35)
    parser.add_argument("--slot-pairing-strategy", choices=["crpsd", "relaxed"], default="relaxed")
    parser.add_argument("--slot-max-point-degree", type=int, default=2)
    parser.add_argument("--match-iou", type=float, default=0.10)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["none", "confirm", "fill_gap", "quality_fill_gap"],
        default=["none", "confirm", "fill_gap"],
    )
    parser.add_argument("--windows", nargs="+", type=int, default=[5])
    parser.add_argument("--min-hits", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--max-missing", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--track-ious", nargs="+", type=float, default=[0.10])
    parser.add_argument("--max-center-distances", nargs="+", type=float, default=[45.0, 70.0])
    parser.add_argument("--min-track-scores", nargs="+", type=float, default=[0.50])
    parser.add_argument("--min-track-ages", nargs="+", type=int, default=[2])
    parser.add_argument("--min-consecutive-hits", nargs="+", type=int, default=[2])
    parser.add_argument("--fill-gap-min-confidences", nargs="+", type=float, default=[0.45, 0.55])
    parser.add_argument("--decay-on-misses", nargs="+", type=float, default=[0.85])
    parser.add_argument("--max-gap-fills", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--output-dir", default="outputs/parkrecon3d_temporal_slot_sweep")
    parser.add_argument("--preview-limit", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = list_image_paths(Path(args.image_dir), args.limit)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    detector = ParkingSlotDetector(
        {
            "backend": "crpsd",
            "model_path": args.slot_model_path,
            "external_repo_path": args.slot_external_repo_path,
            "device": args.device,
            "conf_threshold": args.slot_conf,
            "depth_factor": 32,
            "slot_pairing_strategy": args.slot_pairing_strategy,
            "max_point_degree": args.slot_max_point_degree,
            "min_slot_score": args.slot_min_score,
        }
    )

    frames = load_frames_and_detections(args, image_paths, detector)
    variants = build_variants(args)
    results = evaluate_variants(frames, variants, args.match_iou)
    rows = build_rows(results, variants)
    rows.sort(key=lambda row: (row["f1"], row["recall"], row["precision"]), reverse=True)

    baseline_key = baseline_variant_key(args)
    best_key = rows[0]["key"] if rows else baseline_key
    baseline_row = next((row for row in rows if row["key"] == baseline_key), None)
    add_baseline_deltas(rows, baseline_row)
    write_previews(frames, results, baseline_key, best_key, preview_dir, args.preview_limit, args.match_iou)

    summary = {
        "image_dir": args.image_dir,
        "label_dir": args.label_dir,
        "slot_model_path": args.slot_model_path,
        "slot_conf": args.slot_conf,
        "slot_min_score": args.slot_min_score,
        "slot_pairing_strategy": args.slot_pairing_strategy,
        "slot_max_point_degree": args.slot_max_point_degree,
        "match_iou": args.match_iou,
        "limit": args.limit,
        "baseline_key": baseline_key,
        "baseline": baseline_row,
        "best_by_f1": rows[0] if rows else None,
        "best_by_recall_with_precision_70": first_row(rows, min_precision=0.70),
        "rows": rows,
        "preview_dir": str(preview_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "sweep_results.csv", rows)
    write_records(output_dir / "records.json", frames, results, baseline_key, best_key)
    print(json.dumps(summary, indent=2))


def load_frames_and_detections(args: argparse.Namespace, image_paths: list[Path], detector: ParkingSlotDetector) -> list[dict[str, Any]]:
    frames = []
    label_dir = Path(args.label_dir)
    for frame_idx, image_path in enumerate(tqdm(image_paths, desc="Detecting single-frame slots")):
        label_path = label_dir / f"{image_path.stem}.json"
        if not label_path.exists():
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        gt_slots = load_parkrecon3d_slots(label_path)
        raw_slots = detect_slots(detector, image, args.detector_input_size)
        frames.append({"frame_idx": frame_idx, "image": str(image_path), "gt_slots": gt_slots, "raw_slots": raw_slots})
    return frames


def evaluate_variants(
    frames: list[dict[str, Any]],
    variants: list[TemporalVariant],
    match_iou: float,
) -> dict[str, dict[str, Any]]:
    smoothers = {variant.key: TemporalSlotSmoother(variant) for variant in variants}
    results = {
        variant.key: {"variant": variant, "counts": Counter(), "frames": []}
        for variant in variants
    }
    for frame in tqdm(frames, desc="Evaluating temporal variants"):
        for variant in variants:
            pred_slots = smoothers[variant.key].update(frame["frame_idx"], frame["raw_slots"])
            matches = match_slots(frame["gt_slots"], pred_slots, match_iou)
            counts = results[variant.key]["counts"]
            update_counts(counts, frame["gt_slots"], pred_slots, matches)
            results[variant.key]["frames"].append(
                {
                    "frame_idx": frame["frame_idx"],
                    "image": frame["image"],
                    "gt_slots": len(frame["gt_slots"]),
                    "pred_slots": len(pred_slots),
                    "matched_slots": len(matches),
                    "false_negative_slots": len(frame["gt_slots"]) - len(matches),
                    "false_positive_slots": len(pred_slots) - len(matches),
                    "slots": [slot_to_record(slot) for slot in pred_slots],
                }
            )
    return results


def build_variants(args: argparse.Namespace) -> list[TemporalVariant]:
    variants = []
    for mode in args.modes:
        if mode == "none":
            variants.append(TemporalVariant(mode="none", window=1, min_hits=1, max_missing=0, track_iou=0.0, max_center_distance=0.0))
            continue
        for window in sorted(set(args.windows)):
            for min_hits in sorted(set(args.min_hits)):
                if min_hits > window:
                    continue
                for max_missing in sorted(set(args.max_missing)):
                    for track_iou in sorted(set(args.track_ious)):
                        for max_center_distance in sorted(set(args.max_center_distances)):
                            if mode == "quality_fill_gap":
                                for min_track_score in sorted(set(args.min_track_scores)):
                                    for min_track_age in sorted(set(args.min_track_ages)):
                                        for min_consecutive_hits in sorted(set(args.min_consecutive_hits)):
                                            for fill_gap_min_confidence in sorted(set(args.fill_gap_min_confidences)):
                                                for decay_on_miss in sorted(set(args.decay_on_misses)):
                                                    for max_gap_fill in sorted(set(args.max_gap_fills)):
                                                        variants.append(
                                                            TemporalVariant(
                                                                mode=mode,
                                                                window=window,
                                                                min_hits=min_hits,
                                                                max_missing=max_missing,
                                                                track_iou=track_iou,
                                                                max_center_distance=max_center_distance,
                                                                min_track_score=min_track_score,
                                                                min_track_age=min_track_age,
                                                                min_consecutive_hits=min_consecutive_hits,
                                                                fill_gap_min_confidence=fill_gap_min_confidence,
                                                                decay_on_miss=decay_on_miss,
                                                                max_gap_fill=max_gap_fill,
                                                            )
                                                        )
                                continue
                            variants.append(
                                TemporalVariant(
                                    mode=mode,
                                    window=window,
                                    min_hits=min_hits,
                                    max_missing=max_missing,
                                    track_iou=track_iou,
                                    max_center_distance=max_center_distance,
                                )
                            )
    return variants


def build_rows(results: dict[str, dict[str, Any]], variants: list[TemporalVariant]) -> list[dict[str, Any]]:
    rows = []
    for variant in variants:
        counts = results[variant.key]["counts"]
        metrics = build_metrics(counts)
        precision = metrics["slot_precision"]
        recall = metrics["slot_recall"]
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        rows.append(
            {
                "key": variant.key,
                "mode": variant.mode,
                "window": variant.window,
                "min_hits": variant.min_hits,
                "max_missing": variant.max_missing,
                "track_iou": variant.track_iou,
                "max_center_distance": variant.max_center_distance,
                "min_track_score": variant.min_track_score,
                "min_track_age": variant.min_track_age,
                "min_consecutive_hits": variant.min_consecutive_hits,
                "fill_gap_min_confidence": variant.fill_gap_min_confidence,
                "decay_on_miss": variant.decay_on_miss,
                "max_gap_fill": variant.max_gap_fill,
                "images": counts["images"],
                "gt_slots": counts["gt_slots"],
                "pred_slots": counts["pred_slots"],
                "matched_slots": counts["matched_slots"],
                "false_negative_slots": counts["false_negative_slots"],
                "false_positive_slots": counts["false_positive_slots"],
                "recall": recall,
                "precision": precision,
                "f1": f1,
            }
        )
    return rows


def write_previews(
    frames: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    baseline_key: str,
    best_key: str,
    preview_dir: Path,
    limit: int,
    match_iou: float,
) -> None:
    if limit <= 0 or baseline_key not in results or best_key not in results:
        return
    improved_dir = preview_dir / "improved"
    worsened_dir = preview_dir / "worsened"
    improved_dir.mkdir(parents=True, exist_ok=True)
    worsened_dir.mkdir(parents=True, exist_ok=True)
    baseline_frames = {row["frame_idx"]: row for row in results[baseline_key]["frames"]}
    best_frames = {row["frame_idx"]: row for row in results[best_key]["frames"]}
    ranked = []
    for frame in frames:
        frame_idx = frame["frame_idx"]
        baseline_errors = baseline_frames[frame_idx]["false_negative_slots"] + baseline_frames[frame_idx]["false_positive_slots"]
        best_errors = best_frames[frame_idx]["false_negative_slots"] + best_frames[frame_idx]["false_positive_slots"]
        ranked.append((baseline_errors - best_errors, frame_idx, baseline_errors, best_errors))

    improved = sorted([item for item in ranked if item[0] > 0], key=lambda item: (item[0], item[2]), reverse=True)
    worsened = sorted([item for item in ranked if item[0] < 0], key=lambda item: (item[0], -item[3]))
    frame_by_idx = {frame["frame_idx"]: frame for frame in frames}
    render_preview_group(improved[: limit // 2], improved_dir, frame_by_idx, baseline_frames, best_frames, best_key, match_iou)
    render_preview_group(worsened[: max(0, limit - limit // 2)], worsened_dir, frame_by_idx, baseline_frames, best_frames, best_key, match_iou)


def render_preview_group(
    rows: list[tuple[int, int, int, int]],
    output_dir: Path,
    frame_by_idx: dict[int, dict[str, Any]],
    baseline_frames: dict[int, dict[str, Any]],
    best_frames: dict[int, dict[str, Any]],
    best_key: str,
    match_iou: float,
) -> None:
    for delta, frame_idx, baseline_errors, best_errors in rows:
        frame = frame_by_idx[frame_idx]
        image = cv2.imread(frame["image"])
        if image is None:
            continue
        best_slots = [record_to_slot(record) for record in best_frames[frame_idx]["slots"]]
        matches = match_slots(frame["gt_slots"], best_slots, match_iou)
        preview = draw_result(image, frame["gt_slots"], best_slots, matches, {})
        label = f"{Path(frame['image']).name} | best={best_key} | delta_errors={delta} base={baseline_errors} best={best_errors}"
        cv2.rectangle(preview, (0, 0), (preview.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(preview, label[:150], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(output_dir / f"{frame_idx:06d}_{Path(frame['image']).name}"), preview, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def write_records(
    path: Path,
    frames: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    baseline_key: str,
    best_key: str,
) -> None:
    baseline_frames = {row["frame_idx"]: row for row in results[baseline_key]["frames"]}
    best_frames = {row["frame_idx"]: row for row in results[best_key]["frames"]}
    records = []
    for frame in frames:
        frame_idx = frame["frame_idx"]
        records.append(
            {
                "frame_idx": frame_idx,
                "image": frame["image"],
                "gt_slots": len(frame["gt_slots"]),
                "baseline": compact_frame_record(baseline_frames[frame_idx]),
                "best": compact_frame_record(best_frames[frame_idx]),
            }
        )
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def compact_frame_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "pred_slots": record["pred_slots"],
        "matched_slots": record["matched_slots"],
        "false_negative_slots": record["false_negative_slots"],
        "false_positive_slots": record["false_positive_slots"],
        "slots": record["slots"],
    }


def list_image_paths(image_dir: Path, limit: int | None) -> list[Path]:
    paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:limit] if limit is not None else paths


def update_counts(counts: Counter[str], gt_slots: list[ParkingSlot], pred_slots: list[ParkingSlot], matches: list[dict]) -> None:
    counts["images"] += 1
    counts["gt_slots"] += len(gt_slots)
    counts["pred_slots"] += len(pred_slots)
    counts["matched_slots"] += len(matches)
    counts["false_negative_slots"] += len(gt_slots) - len(matches)
    counts["false_positive_slots"] += len(pred_slots) - len(matches)


def baseline_variant_key(args: argparse.Namespace) -> str:
    return TemporalVariant(mode="none", window=1, min_hits=1, max_missing=0, track_iou=0.0, max_center_distance=0.0).key


def first_row(rows: list[dict[str, Any]], min_precision: float) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["precision"] >= min_precision]
    if not candidates:
        return None
    candidates.sort(key=lambda row: (row["recall"], row["f1"], row["precision"]), reverse=True)
    return candidates[0]


def add_baseline_deltas(rows: list[dict[str, Any]], baseline_row: dict[str, Any] | None) -> None:
    if baseline_row is None:
        return
    for row in rows:
        row["delta_precision"] = row["precision"] - baseline_row["precision"]
        row["delta_recall"] = row["recall"] - baseline_row["recall"]
        row["delta_f1"] = row["f1"] - baseline_row["f1"]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def clone_slot(slot: ParkingSlot, slot_id: int) -> ParkingSlot:
    return ParkingSlot(
        slot_id=slot_id,
        points=[(float(x), float(y)) for x, y in slot.points],
        confidence=slot.confidence,
        type=slot.type,
        occupancy_label=slot.occupancy_label,
    )


def track_to_slot(track: TemporalTrack, filled_gap: bool) -> ParkingSlot:
    slot = clone_slot(track.slot, track.track_id)
    if filled_gap:
        slot.confidence = float(slot.confidence) * (track.last_match_score if track.last_match_score > 0 else 1.0)
    SLOT_TEMPORAL_METADATA[id(slot)] = {
        "track_id": track.track_id,
        "track_score": temporal_track_score(track),
        "is_filled_gap": filled_gap,
    }
    return slot


def temporal_track_score(track: TemporalTrack) -> float:
    hit_ratio = sum(track.history) / max(1, len(track.history))
    age_score = min(1.0, track.track_age / 5.0)
    miss_penalty = 1.0 / (1.0 + track.missed_frames)
    return float((0.50 * track.mean_confidence + 0.30 * hit_ratio + 0.20 * age_score) * miss_penalty)


def slot_to_record(slot: ParkingSlot) -> dict[str, Any]:
    metadata = SLOT_TEMPORAL_METADATA.get(id(slot), {})
    return {
        "slot_id": slot.slot_id,
        "points": [[float(x), float(y)] for x, y in slot.points],
        "confidence": slot.confidence,
        "type": slot.type,
        "track_id": metadata.get("track_id"),
        "track_score": metadata.get("track_score"),
        "is_filled_gap": bool(metadata.get("is_filled_gap", False)),
    }


def record_to_slot(record: dict[str, Any]) -> ParkingSlot:
    return ParkingSlot(
        slot_id=int(record["slot_id"]),
        points=[(float(x), float(y)) for x, y in record["points"]],
        confidence=float(record["confidence"]),
        type=str(record["type"]),
    )


def polygon_iou(points_a, points_b) -> float:
    from scripts.evaluate_parkrecon3d_bev import polygon_iou as eval_polygon_iou

    return eval_polygon_iou(points_a, points_b)


def polygon_center_distance(points_a, points_b) -> float:
    center_a = np.array(points_a, dtype=np.float32).mean(axis=0)
    center_b = np.array(points_b, dtype=np.float32).mean(axis=0)
    return float(np.linalg.norm(center_a - center_b))


if __name__ == "__main__":
    main()
