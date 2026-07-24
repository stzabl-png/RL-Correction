import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.hoi_detr.run_persistent_mask_sequence import (
    _apply_conditioning_ownership,
    _apply_expansion_decompositions,
    _build_expansion_known_tracks,
    _expansion_decomposition_groups,
    _instance_identity_conflicts,
    _load_trusted_interaction_boxes,
    _motion_compensated_pair_metrics,
)
from experiments.hoi_detr.sam2_multi_object import ObjectMaskSeed


class ConditioningOwnershipTests(unittest.TestCase):
    def test_conditioning_mask_wins_over_propagated_prediction(self):
        condition = np.zeros((4, 4), dtype=bool)
        condition[1:3, 1:3] = True
        propagated = np.ones((4, 4), dtype=bool)
        resolved = _apply_conditioning_ownership(
            {"part": np.zeros((4, 4), dtype=bool), "support": propagated},
            {"part": condition},
        )
        np.testing.assert_array_equal(resolved["part"], condition)
        self.assertFalse(np.any(resolved["support"] & condition))

    def test_overlapping_conditions_are_rejected(self):
        mask = np.ones((2, 2), dtype=bool)
        with self.assertRaisesRegex(ValueError, "conditioning masks overlap"):
            _apply_conditioning_ownership(
                {"part": mask, "support": mask},
                {"part": mask, "support": mask},
            )


class InstanceIdentityConflictTests(unittest.TestCase):
    def test_detects_related_tracks_collapsing_onto_one_instance(self):
        part = np.zeros((6, 6), dtype=np.float32)
        support = np.zeros((6, 6), dtype=np.float32)
        part[1:5, 1:5] = 1.0
        support[2:5, 1:5] = 1.0

        conflicts = _instance_identity_conflicts(
            {"part": part, "support": support},
            [("part", "support")],
            max_overlap_fraction=0.5,
        )

        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["first_object_id"], "part")
        self.assertEqual(conflicts[0]["second_object_id"], "support")
        self.assertEqual(conflicts[0]["overlap_fraction_of_smaller_instance"], 1.0)

    def test_allows_small_visible_boundary_overlap(self):
        part = np.zeros((6, 6), dtype=np.float32)
        support = np.zeros((6, 6), dtype=np.float32)
        part[1:3, 1:5] = 1.0
        support[2:6, 1:5] = 1.0

        conflicts = _instance_identity_conflicts(
            {"part": part, "support": support},
            [("part", "support")],
            max_overlap_fraction=0.5,
        )

        self.assertEqual(conflicts, [])


class ExpansionFrameDecompositionTests(unittest.TestCase):
    def test_trusted_box_loader_reuses_only_a_short_accepted_hf_gap(self):
        frames = [
            {
                "frame_idx": 0,
                "status": "accepted",
                "selection_source": "hf_link",
                "box_xyxy": [1, 2, 8, 9],
            },
            {
                "frame_idx": 1,
                "status": "rejected_background_spike",
                "selection_source": "hf_link",
                "box_xyxy": [0, 0, 100, 100],
            },
            {"frame_idx": 2, "status": "missing"},
            {"frame_idx": 3, "status": "missing"},
            {"frame_idx": 4, "status": "missing"},
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "boxes.json"
            path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
            trusted = _load_trusted_interaction_boxes(
                path,
                frame_start=0,
                frame_end=4,
                max_fallback_gap_frames=3,
            )

        self.assertEqual(trusted[0]["source"], "accepted_hf_link")
        self.assertEqual(trusted[1]["box_xyxy"], [1.0, 2.0, 8.0, 9.0])
        self.assertEqual(trusted[3]["source"], "short_gap_last_accepted_hf_link")
        self.assertNotIn(4, trusted)

    def test_trusted_box_loader_uses_union_of_all_accepted_linked_candidates(self):
        frames = [
            {
                "frame_idx": 0,
                "status": "accepted",
                "selection_source": "hf_link",
                "box_xyxy": [10, 10, 20, 20],
                "hand_linked_object_candidates": [
                    {
                        "status": "accepted",
                        "source_detection_id": "small",
                        "box_xyxy": [10, 10, 20, 20],
                    },
                    {
                        "status": "accepted",
                        "source_detection_id": "large",
                        "box_xyxy": [8, 9, 28, 40],
                    },
                    {
                        "status": "rejected_background_spike",
                        "source_detection_id": "bad",
                        "box_xyxy": [0, 0, 100, 100],
                    },
                ],
            }
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "boxes.json"
            path.write_text(json.dumps({"frames": frames}), encoding="utf-8")
            trusted = _load_trusted_interaction_boxes(
                path,
                frame_start=0,
                frame_end=0,
            )
        self.assertEqual(trusted[0]["box_xyxy"], [8.0, 9.0, 28.0, 40.0])
        self.assertEqual(trusted[0]["source"], "accepted_hf_link_candidate_union")
        self.assertEqual(trusted[0]["source_detection_ids"], ["small", "large"])

    def test_composite_prediction_becomes_disjoint_known_and_new_masks(self):
        shape = (12, 12)
        known = np.zeros(shape, dtype=bool)
        known[2:4, 4:8] = True
        body = np.zeros(shape, dtype=bool)
        body[4:10, 3:9] = True
        composite = known | body
        known_logits = np.full(shape, -3.0, dtype=np.float32)
        known_logits[known] = 6.0
        known_logits[body] = 0.5
        new_logits = np.where(composite, 4.0, -3.0).astype(np.float32)
        envelope_logits = new_logits.copy()
        envelope_logits[0:2, 0:2] = 5.0
        registry = {
            "object_order": ["instance_0001", "instance_0002"],
            "confirmed_hoi_box_expansions": [
                {
                    "event": {"activation_frame": 1, "seed_frame": 1},
                    "known_object_ids": ["instance_0001"],
                    "known_component_area_references": {
                        "instance_0001": int(known.sum())
                    },
                    "new_object_id": "instance_0002",
                    "interaction_envelope": {
                        "object_id": "__interaction_envelope_0000"
                    },
                }
            ],
        }
        groups = _expansion_decomposition_groups(
            registry,
            object_ids=["instance_0001", "instance_0002"],
            frame_start=0,
            frame_end=2,
        )
        tracked = _build_expansion_known_tracks(
            groups,
            seeds=[
                ObjectMaskSeed("instance_0001", 1, known),
                ObjectMaskSeed("instance_0002", 1, body),
            ],
            logits_by_frame={
                1: {"instance_0001": known_logits, "instance_0002": new_logits},
                2: {"instance_0001": known_logits, "instance_0002": new_logits},
            },
            frame_end=2,
            min_area_retention=0.5,
            max_area_growth=1.0,
        )

        adjusted, audit = _apply_expansion_decompositions(
            2,
            {"instance_0001": known_logits, "instance_0002": new_logits},
            {
                "instance_0001": known_logits,
                "instance_0002": new_logits,
                "__interaction_envelope_0000": envelope_logits,
            },
            tracked,
            frame_shape=shape,
            trusted_interaction_boxes={
                2: {
                    "box_xyxy": [2, 1, 10, 11],
                    "source_frame": 2,
                    "source": "accepted_hf_link",
                }
            },
            interaction_box_margin_fraction=0.0,
        )
        known_result = adjusted["instance_0001"] > 0
        new_result = adjusted["instance_0002"] > 0

        np.testing.assert_array_equal(known_result, known)
        np.testing.assert_array_equal(new_result, body)
        self.assertFalse(np.any(known_result & new_result))
        self.assertEqual(audit[0]["status"], "success")

    def test_unreliable_known_track_suppresses_the_whole_group(self):
        shape = (8, 8)
        known = np.zeros(shape, dtype=bool)
        known[1:3, 2:6] = True
        logits = np.full(shape, -2.0, dtype=np.float32)
        group = {
            "group_idx": 0,
            "activation_frame": 1,
            "seed_frame": 2,
            "known_object_ids": ["instance_0001"],
            "new_object_id": "instance_0002",
            "interaction_envelope_object_id": "instance_0002",
            "known_component_area_references": {"instance_0001": 8},
            "locked_known_masks_by_frame": {2: {"instance_0001": known}},
            "lock_metrics_by_frame": {},
            "track_failures": [{"frame_idx": 3, "status": "lost"}],
        }

        adjusted, audit = _apply_expansion_decompositions(
            3,
            {"instance_0001": logits, "instance_0002": logits},
            {"instance_0001": logits, "instance_0002": logits},
            [group],
            frame_shape=shape,
        )

        self.assertFalse(np.any(adjusted["instance_0001"] > 0))
        self.assertFalse(np.any(adjusted["instance_0002"] > 0))
        self.assertEqual(
            audit[0]["status"], "temporarily_occluded_empty_visible_masks"
        )
        self.assertEqual(audit[0]["identity_state"], "frozen_last_reliable")

    def test_partial_visibility_ends_locked_track_without_identity_guess(self):
        shape = (20, 20)
        known = np.zeros(shape, dtype=bool)
        known[4:14, 5:15] = True
        partial = np.zeros(shape, dtype=bool)
        partial[7:12, 7:13] = True
        seed_logits = np.where(known, 4.0, -3.0).astype(np.float32)
        partial_logits = np.where(partial, 4.0, -3.0).astype(np.float32)
        group = {
            "group_idx": 0,
            "activation_frame": 2,
            "seed_frame": 2,
            "known_object_ids": ["instance_0001"],
            "new_object_id": "instance_0002",
            "interaction_envelope_object_id": "instance_0002",
            "known_component_area_references": {"instance_0001": 100},
        }
        tracked = _build_expansion_known_tracks(
            [group],
            seeds=[ObjectMaskSeed("instance_0001", 2, known)],
            logits_by_frame={
                2: {"instance_0001": seed_logits},
                3: {"instance_0001": partial_logits},
            },
            frame_end=3,
            min_area_retention=0.5,
            max_area_growth=1.35,
        )[0]

        self.assertNotIn(3, tracked["locked_known_masks_by_frame"])
        self.assertEqual(
            tracked["track_failures"][0]["status"],
            "temporarily_occluded",
        )
        self.assertEqual(
            tracked["track_failures"][0]["cause"],
            "failed_known_component_lost_at_expansion",
        )
        self.assertEqual(
            tracked["frozen_intervals"],
            [
                {
                    "start_frame": 3,
                    "end_frame": 3,
                    "frame_count": 1,
                    "visible_masks": "empty",
                    "identity_state": "frozen_last_reliable",
                }
            ],
        )

    def test_later_conditioning_anchor_restores_same_id_without_partial_update(self):
        shape = (20, 20)
        known = np.zeros(shape, dtype=bool)
        known[4:14, 5:15] = True
        partial = np.zeros(shape, dtype=bool)
        partial[7:12, 7:13] = True
        seed_logits = np.where(known, 4.0, -3.0).astype(np.float32)
        partial_logits = np.where(partial, 4.0, -3.0).astype(np.float32)
        group = {
            "group_idx": 0,
            "activation_frame": 2,
            "seed_frame": 2,
            "known_object_ids": ["instance_0001"],
            "new_object_id": "instance_0002",
            "interaction_envelope_object_id": "__interaction_envelope_0000",
            "known_component_area_references": {"instance_0001": 100},
        }
        tracked = _build_expansion_known_tracks(
            [group],
            seeds=[
                ObjectMaskSeed("instance_0001", 2, known),
                ObjectMaskSeed("instance_0001", 4, known),
            ],
            logits_by_frame={
                2: {"instance_0001": seed_logits},
                3: {"instance_0001": partial_logits},
                4: {"instance_0001": seed_logits},
                5: {"instance_0001": seed_logits},
            },
            frame_end=5,
            min_area_retention=0.5,
            max_area_growth=1.35,
        )[0]

        self.assertNotIn(3, tracked["locked_known_masks_by_frame"])
        np.testing.assert_array_equal(
            tracked["locked_known_masks_by_frame"][4]["instance_0001"], known
        )
        np.testing.assert_array_equal(
            tracked["locked_known_masks_by_frame"][5]["instance_0001"], known
        )
        self.assertEqual(tracked["conditioning_anchor_frames"], [2, 4])

    def test_motion_compensation_accepts_a_small_connected_translation(self):
        previous = np.zeros((40, 40), dtype=bool)
        previous[10:20, 10:20] = True
        current = np.zeros((40, 40), dtype=bool)
        current[10:20, 12:22] = True
        previous_gray = previous.astype(np.uint8) * 255
        current_gray = current.astype(np.uint8) * 255

        metrics = _motion_compensated_pair_metrics(
            previous, current, previous_gray, current_gray
        )

        self.assertGreaterEqual(metrics["flow_warp_continuity"], 0.5)
        self.assertGreaterEqual(metrics["cycle_iou"], 0.6)
        self.assertLessEqual(metrics["compensated_centroid_step_diagonals"], 0.5)

if __name__ == "__main__":
    unittest.main()
