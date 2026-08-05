"""Dependency-light tests for the bottle screw specification."""
from __future__ import annotations

import unittest

import numpy as np

from tasks.recon_kailang.bottle_reconstruction.screw_joint import (
    ScrewSpec,
    assembled_cap_pose,
    helical_travel,
)


class ScrewJointTest(unittest.TestCase):
    def test_pco1810_ratio_and_limits(self):
        spec = ScrewSpec()
        self.assertAlmostEqual(spec.travel_m, 0.00636, places=9)
        self.assertAlmostEqual(spec.angle_limit_deg, 720.0, places=9)
        self.assertAlmostEqual(spec.ratio_deg_per_m, 360.0 / 0.00318)

    def test_one_positive_turn_rises_by_one_pitch(self):
        spec = ScrewSpec()
        self.assertAlmostEqual(
            float(helical_travel(2.0 * np.pi, spec)), spec.pitch_m, places=9
        )

    def test_assembled_pose_follows_body_axis(self):
        # 90 degrees about +Y maps local +Z to world +X.
        q = np.array([np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0])
        pose = assembled_cap_pose(np.r_[1.0, 2.0, 3.0, q], ScrewSpec())
        np.testing.assert_allclose(pose[:3], [1.180, 2.0, 3.0], atol=1e-9)
        np.testing.assert_allclose(pose[3:], q, atol=1e-9)

    def test_invalid_direction_rejected(self):
        with self.assertRaises(ValueError):
            ScrewSpec(direction=0)

    def test_invalid_velocity_limit_rejected(self):
        with self.assertRaises(ValueError):
            ScrewSpec(max_angular_velocity_rad_s=0.0)


if __name__ == "__main__":
    unittest.main()
