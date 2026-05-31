from __future__ import annotations

import argparse
import unittest

from scripts.run_parkrecon3d_multicamera_fusion_sequence import update_camera_evidence_memory


def args() -> argparse.Namespace:
    return argparse.Namespace(
        camera_evidence_hold_frames=2,
        camera_evidence_decay=0.5,
        min_held_camera_evidence_score=0.1,
    )


def evidence(slot_id: int, camera: str = "Camera0", detection_idx: int = 0, score: float = 0.8) -> dict:
    return {
        "slot_id": slot_id,
        "camera": camera,
        "detection_idx": detection_idx,
        "evidence_score": score,
        "confidence": score,
        "source": "camera_vehicle",
    }


class CameraEvidenceMemoryTest(unittest.TestCase):
    def test_same_detection_moving_to_new_slot_removes_old_memory(self) -> None:
        memory: dict[int, dict] = {}
        active = update_camera_evidence_memory(memory, {1: evidence(1)}, frame_idx=10, args=args())
        self.assertEqual(set(active), {1})

        active = update_camera_evidence_memory(memory, {2: evidence(2)}, frame_idx=11, args=args())

        self.assertEqual(set(active), {2})
        self.assertEqual(set(memory), {2})
        self.assertFalse(active[2]["is_held_camera_evidence"])

    def test_missing_evidence_is_held_with_decay(self) -> None:
        memory: dict[int, dict] = {}
        update_camera_evidence_memory(memory, {1: evidence(1, score=0.8)}, frame_idx=10, args=args())

        active = update_camera_evidence_memory(memory, {}, frame_idx=11, args=args())

        self.assertEqual(set(active), {1})
        self.assertTrue(active[1]["is_held_camera_evidence"])
        self.assertEqual(active[1]["held_frames"], 1)
        self.assertAlmostEqual(active[1]["evidence_score"], 0.4)
        self.assertEqual(active[1]["source"], "held_camera_vehicle")

    def test_held_evidence_expires_after_hold_window(self) -> None:
        memory: dict[int, dict] = {}
        update_camera_evidence_memory(memory, {1: evidence(1, score=0.8)}, frame_idx=10, args=args())

        active = update_camera_evidence_memory(memory, {}, frame_idx=13, args=args())

        self.assertEqual(active, {})
        self.assertEqual(memory, {})


if __name__ == "__main__":
    unittest.main()
