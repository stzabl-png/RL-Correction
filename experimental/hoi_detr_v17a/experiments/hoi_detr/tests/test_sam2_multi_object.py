"""Contract tests for shared-state, multi-object SAM2 propagation."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.hoi_detr.sam2_multi_object import (
    ObjectMaskSeed,
    object_seed_from_box,
    propagate_multi_object_logits,
)


class _FakePredictor:
    def __init__(self):
        self.non_overlap_masks = False
        self.non_overlap_masks_for_mem_enc = False
        self.added = []
        self.reset = False
        self.propagation_calls = []

    def init_state(self, *, video_path):
        return {"video_path": video_path}

    def add_new_mask(self, state, *, frame_idx, obj_id, mask):
        self.added.append((state, frame_idx, obj_id, np.asarray(mask).copy()))

    def add_new_points_or_box(self, state, *, frame_idx, obj_id, box, clear_old_points):
        del state, frame_idx, box, clear_old_points
        logits = np.full((1, 1, 4, 4), 2.0, dtype=np.float32)
        return 0, [obj_id], logits

    def propagate_in_video(self, state, *, start_frame_idx, reverse, max_frame_num_to_track=None):
        del state
        self.propagation_calls.append((start_frame_idx, reverse, max_frame_num_to_track))
        frame_indices = range(5, -1, -1) if reverse else range(6)
        direction_offset = 100.0 if reverse else 0.0
        for frame_idx in frame_indices:
            logits = np.stack(
                [
                    np.full((4, 4), direction_offset + 10.0 + frame_idx, dtype=np.float32),
                    np.full((4, 4), direction_offset + 20.0 + frame_idx, dtype=np.float32),
                ]
            )[:, None]
            yield frame_idx, [1, 2], logits

    def reset_state(self, state):
        del state
        self.reset = True


class MultiObjectPropagationTests(unittest.TestCase):
    def test_uses_one_state_unique_ids_and_bidirectional_logits(self):
        predictor = _FakePredictor()
        seeds = [
            ObjectMaskSeed("lid_0", 2, np.ones((4, 4), dtype=bool)),
            ObjectMaskSeed("cup_0", 4, np.ones((4, 4), dtype=bool)),
        ]
        result = propagate_multi_object_logits(
            predictor,
            video_path="clip.mp4",
            seeds=seeds,
        )

        self.assertEqual(result.sam_object_ids, {"lid_0": 1, "cup_0": 2})
        self.assertEqual(len({id(item[0]) for item in predictor.added}), 1)
        self.assertEqual([item[2] for item in predictor.added], [1, 2])
        self.assertTrue(predictor.non_overlap_masks)
        self.assertTrue(predictor.non_overlap_masks_for_mem_enc)
        self.assertTrue(predictor.reset)

        self.assertEqual(float(result.logits_by_frame[1]["lid_0"][0, 0]), 111.0)
        self.assertEqual(float(result.logits_by_frame[2]["lid_0"][0, 0]), 12.0)
        self.assertEqual(float(result.logits_by_frame[3]["cup_0"][0, 0]), 123.0)
        self.assertEqual(float(result.logits_by_frame[4]["cup_0"][0, 0]), 24.0)

    def test_accepts_multiple_conditioning_frames_for_one_persistent_id(self):
        predictor = _FakePredictor()
        result = propagate_multi_object_logits(
            predictor,
            video_path="clip.mp4",
            seeds=[
                ObjectMaskSeed("lid_0", 1, np.ones((4, 4), dtype=bool)),
                ObjectMaskSeed("lid_0", 3, np.ones((4, 4), dtype=bool)),
                ObjectMaskSeed("cup_0", 3, np.ones((4, 4), dtype=bool)),
            ],
        )
        self.assertEqual(result.sam_object_ids, {"lid_0": 1, "cup_0": 2})
        self.assertEqual(result.seed_frames, {"lid_0": 1, "cup_0": 3})
        self.assertEqual(result.conditioning_frames, {"lid_0": (1, 3), "cup_0": (3,)})
        self.assertEqual([item[2] for item in predictor.added], [1, 2, 1])

    def test_limits_propagation_to_requested_frame_range(self):
        predictor = _FakePredictor()
        result = propagate_multi_object_logits(
            predictor,
            video_path="clip.mp4",
            seeds=[
                ObjectMaskSeed("lid_0", 2, np.ones((4, 4), dtype=bool)),
                ObjectMaskSeed("cup_0", 4, np.ones((4, 4), dtype=bool)),
            ],
            frame_start=1,
            frame_end=5,
        )
        self.assertEqual(predictor.propagation_calls, [(2, False, 4), (4, True, 4)])
        self.assertEqual(sorted(result.logits_by_frame), [1, 2, 3, 4, 5])
        with self.assertRaisesRegex(ValueError, "outside propagation range"):
            propagate_multi_object_logits(
                _FakePredictor(),
                video_path="clip.mp4",
                seeds=[ObjectMaskSeed("lid_0", 0, np.ones((4, 4), dtype=bool))],
                frame_start=1,
                frame_end=5,
            )

    def test_rejects_empty_duplicate_and_shape_mismatched_seeds(self):
        predictor = _FakePredictor()
        with self.assertRaises(ValueError):
            propagate_multi_object_logits(predictor, video_path="clip.mp4", seeds=[])
        with self.assertRaises(ValueError):
            propagate_multi_object_logits(
                predictor,
                video_path="clip.mp4",
                seeds=[
                    ObjectMaskSeed("same", 0, np.ones((4, 4))),
                    ObjectMaskSeed("same", 0, np.ones((4, 4))),
                ],
            )

    def test_creates_nonempty_seed_from_valid_box(self):
        predictor = _FakePredictor()
        seed, logits = object_seed_from_box(
            predictor,
            video_path="clip.mp4",
            object_id="lid_0",
            frame_idx=3,
            box_xyxy=[1, 2, 3, 4],
        )
        self.assertEqual(seed.object_id, "lid_0")
        self.assertEqual(seed.frame_idx, 3)
        self.assertTrue(seed.mask.all())
        self.assertEqual(logits.shape, (4, 4))
        self.assertTrue(predictor.reset)
        with self.assertRaises(ValueError):
            propagate_multi_object_logits(
                predictor,
                video_path="clip.mp4",
                seeds=[
                    ObjectMaskSeed("lid", 0, np.ones((4, 4))),
                    ObjectMaskSeed("cup", 1, np.ones((5, 4))),
                ],
            )


if __name__ == "__main__":
    unittest.main()
