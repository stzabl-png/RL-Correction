import unittest

import numpy as np

from experiments.hoi_detr.component_decision import ComponentMotionEvidence
from experiments.hoi_detr.instance_association import InstanceCandidate
from experiments.hoi_detr.instance_hypothesis import select_initial_instance_hypothesis


def _mask(shape, y_slice, x_slice):
    mask = np.zeros(shape, dtype=bool)
    mask[y_slice, x_slice] = True
    return mask


class InstanceHypothesisTests(unittest.TestCase):
    def test_selects_multiple_components_only_with_motion_evidence(self):
        roi = _mask((30, 30), slice(5, 25), slice(5, 25))
        first = _mask((30, 30), slice(5, 15), slice(5, 15))
        second = _mask((30, 30), slice(15, 25), slice(15, 25))
        evidence = ComponentMotionEvidence("a", "b", (0, 1, 2), 0.1, 3, True)

        decision = select_initial_instance_hypothesis(
            [InstanceCandidate("a", first, 0.95), InstanceCandidate("b", second, 0.95)],
            interaction_roi=roi,
            pair_evidence=[evidence],
            min_multi_component_score_gain=0.01,
        )

        self.assertEqual(decision["status"], "multiple_components")
        self.assertEqual(set(decision["selected_candidate_ids"]), {"a", "b"})

    def test_fixed_subregions_fall_back_to_one_object(self):
        roi = _mask((30, 30), slice(5, 25), slice(5, 25))
        first = _mask((30, 30), slice(5, 15), slice(5, 15))
        second = _mask((30, 30), slice(15, 25), slice(15, 25))
        evidence = ComponentMotionEvidence("a", "b", (0, 1, 2), 0.0, 3, False)

        decision = select_initial_instance_hypothesis(
            [InstanceCandidate("a", first, 0.95), InstanceCandidate("b", second, 0.95)],
            interaction_roi=roi,
            pair_evidence=[evidence],
        )

        self.assertEqual(decision["status"], "single_object")
        self.assertEqual(len(decision["selected_candidate_ids"]), 1)


if __name__ == "__main__":
    unittest.main()
