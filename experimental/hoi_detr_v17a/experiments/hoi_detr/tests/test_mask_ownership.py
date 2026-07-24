"""Tests for persistent object ownership and composite-mask subtraction."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.hoi_detr.mask_ownership import (
    composite_residual_mask,
    resolve_relation_mask_ownership,
    resolve_visible_mask_ownership,
)


class MaskOwnershipTests(unittest.TestCase):
    def test_visible_masks_are_disjoint_and_highest_logit_wins_normally(self):
        lid = np.full((7, 7), -2.0, dtype=np.float32)
        cup = np.full((7, 7), -2.0, dtype=np.float32)
        lid[2:5, 2:5] = 1.0
        cup[3:7, 1:6] = 2.0
        result = resolve_visible_mask_ownership({"lid_0": lid, "cup_0": cup})
        self.assertGreater(result.overlap_pixels_before, 0)
        self.assertEqual(result.overlap_pixels_after, 0)
        self.assertTrue(result.masks["cup_0"][3, 3])
        self.assertFalse(result.masks["lid_0"][3, 3])

    def test_confirmed_object_core_cannot_be_swallowed_by_new_composite(self):
        lid = np.full((9, 9), -2.0, dtype=np.float32)
        composite = np.full((9, 9), -2.0, dtype=np.float32)
        lid[2:7, 2:7] = 1.0
        composite[1:9, 1:8] = 4.0
        result = resolve_visible_mask_ownership(
            {"lid_0": lid, "tentative_1": composite},
            protected_object_ids={"lid_0"},
            protected_core_erosion_pixels=1,
        )
        self.assertTrue(result.masks["lid_0"][4, 4])
        self.assertFalse(result.masks["tentative_1"][4, 4])
        self.assertTrue(result.masks["tentative_1"][1, 1])
        self.assertEqual(result.overlap_pixels_after, 0)

    def test_relation_part_occludes_support_without_checkerboard_competition(self):
        part = np.full((6, 6), -1.0, dtype=np.float32)
        support = np.full((6, 6), -1.0, dtype=np.float32)
        part[1:4, 1:5] = 0.5
        support[2:6, 0:6] = 5.0
        result = resolve_relation_mask_ownership(
            {"part": part, "support": support},
            part_support_pairs=[("part", "support")],
        )
        self.assertTrue(result.masks["part"][2, 2])
        self.assertFalse(result.masks["support"][2, 2])
        self.assertTrue(result.masks["support"][5, 2])
        self.assertEqual(result.overlap_pixels_after, 0)

    def test_composite_residual_removes_existing_lid_and_keeps_cup_region(self):
        composite = np.zeros((20, 20), dtype=bool)
        composite[3:18, 5:15] = True
        lid = np.zeros_like(composite)
        lid[3:8, 6:14] = True
        residual, diagnostics = composite_residual_mask(
            composite,
            {"lid_0": lid},
            min_component_area=10,
        )
        self.assertFalse(np.any(residual & lid))
        self.assertTrue(residual[12, 10])
        self.assertEqual(diagnostics["kept_components"], 1)
        self.assertEqual(diagnostics["kept_residual_area"], int(composite.sum() - lid.sum()))

    def test_small_residual_components_are_not_promoted(self):
        composite = np.zeros((10, 10), dtype=bool)
        composite[1, 1] = True
        residual, diagnostics = composite_residual_mask(
            composite,
            {},
            min_component_area=2,
        )
        self.assertFalse(residual.any())
        self.assertEqual(diagnostics["kept_components"], 0)

    def test_shape_and_unknown_protected_id_errors_are_explicit(self):
        logits = np.ones((5, 5), dtype=np.float32)
        with self.assertRaises(KeyError):
            resolve_visible_mask_ownership({"lid_0": logits}, protected_object_ids={"cup_0"})
        with self.assertRaises(ValueError):
            composite_residual_mask(np.ones((5, 5)), {"lid_0": np.ones((4, 5))})


if __name__ == "__main__":
    unittest.main()
