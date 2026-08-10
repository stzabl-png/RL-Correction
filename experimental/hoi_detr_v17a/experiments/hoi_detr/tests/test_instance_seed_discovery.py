import unittest
from unittest.mock import patch

import numpy as np

from experiments.hoi_detr.run_instance_seed_discovery import (
    _box_prompt_candidate_pool,
    _candidate_pool,
    _distinct_hand_box_prompt_decision,
    _linked_object_boxes,
    _linked_object_prompts,
    _resolve_direct_box_prompt_ownership,
    rank_interaction_seed_frames,
)
from experiments.hoi_detr.instance_association import InstanceCandidate


class InstanceSeedDiscoveryTests(unittest.TestCase):
    def test_linked_boxes_keep_only_hand_linked_visual_object_boxes(self):
        frame = {
            "detections": [
                {"detection_id": "hand", "class_name": "hand", "box_xyxy": [0, 0, 5, 5]},
                {"detection_id": "first", "class_name": "firstobject", "box_xyxy": [5, 5, 10, 10]},
                {"detection_id": "second", "class_name": "secondobject", "box_xyxy": [10, 10, 15, 15]},
            ],
            "links": {"hf": [{"source_detection_id": "hand", "target_detection_id": "first"}], "fs": [{"source_detection_id": "first", "target_detection_id": "second"}]},
        }

        boxes = _linked_object_boxes(frame)

        self.assertEqual(boxes, [[5.0, 5.0, 10.0, 10.0]])

        self.assertEqual(
            _linked_object_prompts(frame),
            [
                {
                    "source_detection_id": "first",
                    "box_xyxy": [5.0, 5.0, 10.0, 10.0],
                    "linked_hand_ids": ["hand"],
                }
            ],
        )

    def test_distinct_hands_with_disjoint_masks_bypass_motion_test(self):
        first_mask = np.zeros((10, 10), dtype=bool)
        first_mask[1:4, 1:4] = True
        second_mask = np.zeros((10, 10), dtype=bool)
        second_mask[6:9, 6:9] = True
        candidates = [
            InstanceCandidate("box_prompt_candidate_000", first_mask, 0.9, "box"),
            InstanceCandidate("box_prompt_candidate_001", second_mask, 0.8, "box"),
        ]
        prompts = {
            "box_prompt_candidate_000": {
                "source_detection_id": "object_left",
                "linked_hand_ids": ["hand_left"],
            },
            "box_prompt_candidate_001": {
                "source_detection_id": "object_right",
                "linked_hand_ids": ["hand_right"],
            },
        }

        decision = _distinct_hand_box_prompt_decision(
            candidates,
            prompt_by_candidate_id=prompts,
            max_candidate_overlap_fraction=0.05,
        )

        self.assertEqual(decision["status"], "multiple_components")
        self.assertEqual(
            decision["selected_candidate_ids"],
            ["box_prompt_candidate_000", "box_prompt_candidate_001"],
        )

    def test_same_hand_candidates_still_require_motion_test(self):
        first_mask = np.zeros((10, 10), dtype=bool)
        first_mask[1:4, 1:4] = True
        second_mask = np.zeros((10, 10), dtype=bool)
        second_mask[6:9, 6:9] = True
        candidates = [
            InstanceCandidate("box_prompt_candidate_000", first_mask, 0.9, "box"),
            InstanceCandidate("box_prompt_candidate_001", second_mask, 0.8, "box"),
        ]
        prompts = {
            "box_prompt_candidate_000": {
                "source_detection_id": "object_a",
                "linked_hand_ids": ["same_hand"],
            },
            "box_prompt_candidate_001": {
                "source_detection_id": "object_b",
                "linked_hand_ids": ["same_hand"],
            },
        }

        self.assertIsNone(
            _distinct_hand_box_prompt_decision(
                candidates,
                prompt_by_candidate_id=prompts,
                max_candidate_overlap_fraction=0.05,
            )
        )

    def test_overlapping_distinct_hand_candidates_still_require_motion_test(self):
        first_mask = np.zeros((10, 10), dtype=bool)
        first_mask[1:6, 1:6] = True
        second_mask = np.zeros((10, 10), dtype=bool)
        second_mask[2:7, 2:7] = True
        candidates = [
            InstanceCandidate("box_prompt_candidate_000", first_mask, 0.9, "box"),
            InstanceCandidate("box_prompt_candidate_001", second_mask, 0.8, "box"),
        ]
        prompts = {
            "box_prompt_candidate_000": {
                "source_detection_id": "object_left",
                "linked_hand_ids": ["hand_left"],
            },
            "box_prompt_candidate_001": {
                "source_detection_id": "object_right",
                "linked_hand_ids": ["hand_right"],
            },
        }

        self.assertIsNone(
            _distinct_hand_box_prompt_decision(
                candidates,
                prompt_by_candidate_id=prompts,
                max_candidate_overlap_fraction=0.05,
            )
        )

    def test_low_overlap_direct_hand_seeds_are_made_exactly_disjoint(self):
        first_mask = np.zeros((12, 12), dtype=bool)
        first_mask[1:6, 1:6] = True
        second_mask = np.zeros((12, 12), dtype=bool)
        second_mask[5:10, 5:10] = True
        candidates = [
            InstanceCandidate("box_prompt_candidate_000", first_mask, 0.9, "box"),
            InstanceCandidate("box_prompt_candidate_001", second_mask, 0.8, "box"),
        ]
        prompts = {
            "box_prompt_candidate_000": {
                "source_detection_id": "object_left",
                "linked_hand_ids": ["hand_left"],
            },
            "box_prompt_candidate_001": {
                "source_detection_id": "object_right",
                "linked_hand_ids": ["hand_right"],
            },
        }
        decision = _distinct_hand_box_prompt_decision(
            candidates,
            prompt_by_candidate_id=prompts,
            max_candidate_overlap_fraction=0.05,
        )
        first_logits = np.where(first_mask, 1.0, -1.0).astype(np.float32)
        second_logits = np.where(second_mask, 1.0, -1.0).astype(np.float32)
        first_logits[5, 5] = 2.0

        resolved = _resolve_direct_box_prompt_ownership(
            candidates,
            component_decision=decision,
            logits_by_candidate_id={
                "box_prompt_candidate_000": first_logits,
                "box_prompt_candidate_001": second_logits,
            },
        )

        self.assertFalse(np.any(resolved[0].mask & resolved[1].mask))
        self.assertTrue(resolved[0].mask[5, 5])
        self.assertEqual(
            decision["ownership_resolution"]["overlap_pixels_after"], 0
        )

    def test_candidate_pool_keeps_only_candidates_in_visual_interaction_roi(self):
        roi = np.zeros((10, 10), dtype=bool)
        roi[2:8, 2:8] = True
        inside = np.zeros((10, 10), dtype=np.uint8)
        inside[2:6, 2:6] = 255
        outside = np.zeros((10, 10), dtype=np.uint8)
        outside[0:2, 0:2] = 255
        import tempfile
        from pathlib import Path
        import cv2

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inside_path = root / "inside.png"
            outside_path = root / "outside.png"
            cv2.imwrite(str(inside_path), inside)
            cv2.imwrite(str(outside_path), outside)
            candidates = _candidate_pool(
                {
                    "candidates": [
                        {"candidate_idx": 0, "mask": str(inside_path), "predicted_iou": 0.9, "stability_score": 0.9},
                        {"candidate_idx": 1, "mask": str(outside_path), "predicted_iou": 0.9, "stability_score": 0.9},
                    ]
                },
                interaction_roi=roi,
                max_seed_candidates=4,
                min_candidate_roi_fraction=0.15,
            )

        self.assertEqual([candidate.candidate_id for candidate in candidates], ["candidate_000"])

    def test_box_prompt_primary_keeps_a_mask_not_the_detector_box(self):
        roi = np.zeros((10, 10), dtype=bool)
        roi[2:8, 2:8] = True
        prompted_mask = np.zeros((10, 10), dtype=bool)
        prompted_mask[3:7, 3:7] = True

        class Seed:
            mask = prompted_mask

        with patch(
            "experiments.hoi_detr.run_instance_seed_discovery.object_seed_from_box",
            return_value=(Seed(), np.zeros((10, 10), dtype=np.float32)),
        ):
            candidates = _box_prompt_candidate_pool(
                object(),
                video_path=__import__("pathlib").Path("video.mp4"),
                source_frame=2,
                boxes=[[2, 2, 8, 8]],
                interaction_roi=roi,
                max_seed_candidates=2,
                min_candidate_roi_fraction=0.15,
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source, "sam2_hand_linked_box_prompt")
        self.assertTrue(np.array_equal(candidates[0].mask, prompted_mask))

    def test_seed_ranking_uses_interaction_start_and_end(self):
        def frame(frame_idx, hand_box, object_box):
            return {
                "frame_idx": frame_idx,
                "detections": [
                    {"detection_id": "hand", "class_name": "hand", "box_xyxy": hand_box},
                    {"detection_id": "object", "class_name": "firstobject", "box_xyxy": object_box},
                ],
                "links": {"hf": [{"source_detection_id": "hand", "target_detection_id": "object"}]},
            }

        frames = [
            frame(0, [0, 0, 8, 8], [0, 0, 10, 10]),
            frame(1, [0, 0, 2, 2], [0, 0, 10, 10]),
            frame(2, [0, 0, 4, 4], [0, 0, 10, 10]),
        ]

        selected = rank_interaction_seed_frames(
            {"frames": frames},
            interaction_start_frame=0,
            interaction_end_frame=2,
            max_seed_frames=3,
        )

        self.assertEqual(selected, [0, 2])

        self.assertEqual(
            rank_interaction_seed_frames(
                {"frames": frames},
                interaction_start_frame=0,
                interaction_end_frame=2,
                max_seed_frames=1,
            ),
            [0],
        )

    def test_seed_ranking_uses_last_linked_frame_before_public_end(self):
        frames = [
            {
                "frame_idx": 0,
                "detections": [
                    {"detection_id": "hand", "class_name": "hand", "box_xyxy": [0, 0, 2, 2]},
                    {"detection_id": "object", "class_name": "firstobject", "box_xyxy": [0, 0, 10, 10]},
                ],
                "links": {"hf": [{"source_detection_id": "hand", "target_detection_id": "object"}]},
            },
            {
                "frame_idx": 1,
                "detections": [
                    {"detection_id": "hand", "class_name": "hand", "box_xyxy": [0, 0, 2, 2]},
                    {"detection_id": "object", "class_name": "firstobject", "box_xyxy": [0, 0, 10, 10]},
                ],
                "links": {"hf": [{"source_detection_id": "hand", "target_detection_id": "object"}]},
            },
            {"frame_idx": 2, "detections": [], "links": {"hf": []}},
        ]

        self.assertEqual(
            rank_interaction_seed_frames(
                {"frames": frames},
                interaction_start_frame=0,
                interaction_end_frame=2,
                max_seed_frames=2,
            ),
            [0, 1],
        )


if __name__ == "__main__":
    unittest.main()
