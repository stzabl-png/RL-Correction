import unittest

import numpy as np

from experiments.hoi_detr.instance_association import (
    InstanceCandidate,
    InstanceTrack,
    associate_tracks_to_candidates,
    derive_residual_instance_candidates,
)


def _mask(shape, ys, xs):
    result = np.zeros(shape, dtype=bool)
    result[ys, xs] = True
    return result


class InstanceAssociationTests(unittest.TestCase):
    def test_nested_proposals_produce_a_disjoint_residual_candidate(self):
        whole = _mask((12, 12), slice(2, 10), slice(2, 10))
        child = _mask((12, 12), slice(2, 5), slice(2, 10))

        residuals = derive_residual_instance_candidates(
            [
                InstanceCandidate("whole", whole, 0.95),
                InstanceCandidate("child", child, 0.95),
            ],
            min_residual_area=10,
        )

        self.assertEqual(len(residuals), 1)
        self.assertFalse(np.any(residuals[0].mask & child))
        self.assertEqual(np.count_nonzero(residuals[0].mask), 40)

    def test_association_assigns_disjoint_candidates_without_semantic_roles(self):
        first = _mask((20, 20), slice(2, 7), slice(2, 7))
        second = _mask((20, 20), slice(11, 17), slice(11, 17))
        result = associate_tracks_to_candidates(
            [
                InstanceTrack("object_0001", first, first),
                InstanceTrack("object_0002", second, second),
            ],
            [
                InstanceCandidate("candidate_a", first, 0.95),
                InstanceCandidate("candidate_b", second, 0.95),
            ],
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["assignment"],
            {"object_0001": "candidate_a", "object_0002": "candidate_b"},
        )

    def test_overlapping_candidate_assignment_fails_instead_of_mixing_ids(self):
        first = _mask((20, 20), slice(2, 8), slice(2, 8))
        second = _mask((20, 20), slice(10, 16), slice(10, 16))
        collapsed = _mask((20, 20), slice(2, 14), slice(2, 14))
        result = associate_tracks_to_candidates(
            [
                InstanceTrack("object_0001", first, collapsed),
                InstanceTrack("object_0002", second, collapsed),
            ],
            [
                InstanceCandidate("candidate_a", collapsed, 0.95),
                InstanceCandidate("candidate_b", collapsed, 0.94),
            ],
        )

        self.assertEqual(result["status"], "failed_no_mutually_exclusive_assignment")


if __name__ == "__main__":
    unittest.main()
