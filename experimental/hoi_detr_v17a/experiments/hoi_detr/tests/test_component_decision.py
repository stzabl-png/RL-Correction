import unittest

import numpy as np

from experiments.hoi_detr.component_decision import (
    component_motion_evidence,
    decide_component_hypothesis,
)


def _mask(x: int, y: int) -> np.ndarray:
    value = np.zeros((100, 100), dtype=bool)
    value[y : y + 10, x : x + 10] = True
    return value


class ComponentDecisionTests(unittest.TestCase):
    def test_relative_motion_supports_multiple_components(self):
        evidence = component_motion_evidence(
            "candidate_a",
            "candidate_b",
            {
                0: {"candidate_a": _mask(10, 10), "candidate_b": _mask(50, 10)},
                1: {"candidate_a": _mask(25, 10), "candidate_b": _mask(50, 10)},
                2: {"candidate_a": _mask(35, 10), "candidate_b": _mask(50, 10)},
            },
            frame_shape=(100, 100),
        )

        decision = decide_component_hypothesis(["candidate_a", "candidate_b"], [evidence])

        self.assertTrue(evidence.reliable)
        self.assertEqual(decision["status"], "multiple_components")

    def test_locked_subregions_remain_one_object(self):
        evidence = component_motion_evidence(
            "candidate_a",
            "candidate_b",
            {
                0: {"candidate_a": _mask(10, 10), "candidate_b": _mask(50, 10)},
                1: {"candidate_a": _mask(20, 15), "candidate_b": _mask(60, 15)},
                2: {"candidate_a": _mask(30, 20), "candidate_b": _mask(70, 20)},
            },
            frame_shape=(100, 100),
        )

        decision = decide_component_hypothesis(["candidate_a", "candidate_b"], [evidence])

        self.assertFalse(evidence.reliable)
        self.assertEqual(decision["status"], "single_object")
