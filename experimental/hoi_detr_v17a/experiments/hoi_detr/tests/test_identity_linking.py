import unittest

import numpy as np

from experiments.hoi_detr.identity_linking import (
    EpisodeInstanceObservation,
    GlobalInstanceObservation,
    link_episode_instances,
)


def _mask(x: int, y: int) -> np.ndarray:
    value = np.zeros((50, 50), dtype=bool)
    value[y : y + 10, x : x + 10] = True
    return value


class IdentityLinkingTests(unittest.TestCase):
    def test_reuses_global_ids_by_visual_match_not_local_name(self):
        result = link_episode_instances(
            [
                GlobalInstanceObservation("object_0001", _mask(5, 5)),
                GlobalInstanceObservation("object_0002", _mask(30, 30)),
            ],
            [
                EpisodeInstanceObservation("instance_b", _mask(31, 30)),
                EpisodeInstanceObservation("instance_a", _mask(6, 5)),
            ],
            min_score_margin=0.001,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["local_to_existing_global"],
            {"instance_a": "object_0001", "instance_b": "object_0002"},
        )

    def test_new_visual_instance_is_left_unmatched(self):
        result = link_episode_instances(
            [GlobalInstanceObservation("object_0001", _mask(5, 5))],
            [EpisodeInstanceObservation("instance_new", _mask(30, 30))],
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["unmatched_local_object_ids"], ["instance_new"])


if __name__ == "__main__":
    unittest.main()
