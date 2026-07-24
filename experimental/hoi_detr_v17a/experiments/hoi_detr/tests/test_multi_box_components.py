"""Tests for category-free all-box component discovery."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.hoi_detr.multi_box_components import (
    classify_candidate_masks,
    confirm_new_component_track,
    refine_known_masks_from_candidates,
    select_component_anchor_proposals,
)


def _rect(y1, y2, x1, x2):
    mask = np.zeros((100, 100), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


class MultiBoxComponentTests(unittest.TestCase):
    def test_small_candidate_refines_known_part_before_large_mask_subtraction(self):
        known = _rect(10, 35, 10, 35)
        drifted = known | _rect(35, 55, 10, 35)
        small = _rect(11, 36, 11, 36)
        large = small | _rect(36, 75, 11, 36)
        refined, assignments = refine_known_masks_from_candidates(
            {"instance_0001": drifted},
            [
                {"candidate_id": "small", "mask": small, "quality_score": 0.9},
                {"candidate_id": "large", "mask": large, "quality_score": 0.9},
            ],
            area_references={"instance_0001": int(np.count_nonzero(known))},
        )
        self.assertEqual(assignments["instance_0001"]["candidate_id"], "small")
        self.assertTrue(np.array_equal(refined["instance_0001"], small))
        classified = classify_candidate_masks(
            [
                {"candidate_id": "small", "mask": small, "quality_score": 0.9},
                {"candidate_id": "large", "mask": large, "quality_score": 0.9},
            ],
            existing_masks=refined,
            min_area_pixels=100,
        )
        by_id = {item["candidate_id"]: item for item in classified}
        self.assertEqual(by_id["small"]["status"], "explained_by_existing_ids")
        self.assertEqual(by_id["large"]["status"], "new_component_proposal")
        self.assertEqual(by_id["large"]["proposal_type"], "composite_residual")

    def test_composite_candidate_registers_only_unexplained_residual(self):
        known = _rect(10, 35, 10, 35)
        body = _rect(35, 75, 10, 35)
        results = classify_candidate_masks(
            [
                {
                    "candidate_id": "large",
                    "mask": known | body,
                    "quality_score": 0.95,
                }
            ],
            existing_masks={"instance_0001": known},
            min_area_pixels=100,
        )
        self.assertEqual(results[0]["status"], "new_component_proposal")
        self.assertEqual(results[0]["proposal_type"], "composite_residual")
        self.assertTrue(np.array_equal(results[0]["mask"], body))

    def test_direct_distinct_candidate_can_become_new_component(self):
        known = _rect(10, 30, 10, 30)
        other = _rect(50, 80, 50, 80)
        results = classify_candidate_masks(
            [{"candidate_id": "other", "mask": other, "quality_score": 0.9}],
            existing_masks={"instance_0001": known},
            min_area_pixels=100,
        )
        self.assertEqual(results[0]["proposal_type"], "direct_disjoint_candidate")

    def test_repeated_existing_candidate_does_not_create_id(self):
        known = _rect(10, 35, 10, 35)
        results = classify_candidate_masks(
            [{"candidate_id": "same", "mask": known, "quality_score": 0.9}],
            existing_masks={"instance_0001": known},
            min_area_pixels=100,
        )
        self.assertEqual(results[0]["status"], "explained_by_existing_ids")

    def test_three_consistent_frames_confirm_but_single_frame_does_not(self):
        proposals = []
        for frame_idx in range(3):
            mask = _rect(40, 70, 20 + frame_idx, 50 + frame_idx)
            proposals.append(
                {
                    "status": "new_component_proposal",
                    "frame_idx": frame_idx,
                    "candidate_id": f"c{frame_idx}",
                    "proposal_area": int(np.count_nonzero(mask)),
                    "proposal_score": 0.9,
                    "quality_score": 0.9,
                    "mask": mask,
                }
            )
        confirmed = confirm_new_component_track(proposals, min_consecutive_frames=3)
        self.assertEqual(confirmed["status"], "success")
        self.assertEqual(confirmed["track_frames"], [0, 1, 2])
        failed = confirm_new_component_track(proposals[:1], min_consecutive_frames=3)
        self.assertEqual(
            failed["status"], "failed_no_temporally_confirmed_new_component"
        )

    def test_anchor_selection_bounds_gap_and_preserves_seed(self):
        proposals = [
            {
                "status": "new_component_proposal",
                "frame_idx": frame_idx,
                "proposal_score": 0.8 + 0.01 * (frame_idx % 3),
            }
            for frame_idx in range(10, 31)
        ]
        anchors = select_component_anchor_proposals(
            proposals,
            track_frames=list(range(10, 31)),
            seed_frame=17,
            max_gap_frames=6,
        )
        frames = [item["frame_idx"] for item in anchors]
        self.assertIn(17, frames)
        self.assertEqual(frames[0], 10)
        self.assertEqual(frames[-1], 30)
        self.assertTrue(all(second - first <= 6 for first, second in zip(frames, frames[1:])))


if __name__ == "__main__":
    unittest.main()
