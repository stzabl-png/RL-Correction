import unittest

import numpy as np

from ocir.isaac.simulate_grasp_traj import govern_contact_targets


class ContactTargetGovernorTest(unittest.TestCase):
    def test_preserves_reachable_targets(self) -> None:
        actual = np.array([0.1, -0.2, 0.3])
        desired = np.array([0.12, -0.18, 0.29])

        governed, limited_count = govern_contact_targets(desired, actual, 0.03)

        np.testing.assert_allclose(governed, desired)
        self.assertEqual(limited_count, 0)

    def test_limits_each_joint_in_both_directions(self) -> None:
        actual = np.array([0.1, -0.2, 0.3, 0.4])
        desired = np.array([0.5, -0.6, 0.31, 0.2])

        governed, limited_count = govern_contact_targets(desired, actual, 0.03)

        np.testing.assert_allclose(governed, np.array([0.13, -0.23, 0.31, 0.37]))
        self.assertEqual(limited_count, 3)

    def test_rejects_nonpositive_lead(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            govern_contact_targets(np.zeros(2), np.zeros(2), 0.0)


if __name__ == "__main__":
    unittest.main()
