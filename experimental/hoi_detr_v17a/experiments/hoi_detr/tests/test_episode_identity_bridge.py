import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.hoi_detr.episode_identity_bridge import BridgeSource, bridge_hidden_gap


class EpisodeIdentityBridgeTests(unittest.TestCase):
    def test_bridges_only_to_the_hidden_gap_target_frame(self):
        source_mask = np.zeros((8, 8), dtype=bool)
        source_mask[1:3, 1:3] = True
        target_mask = np.zeros((8, 8), dtype=np.float32)
        target_mask[3:5, 3:5] = 1.0

        class Result:
            logits_by_frame = {12: {"object_0001": target_mask}}

        with patch(
            "experiments.hoi_detr.episode_identity_bridge.propagate_independent_object_logits",
            return_value=Result(),
        ) as propagate:
            observations = bridge_hidden_gap(
                object(),
                video_path=Path("video.mp4"),
                sources=[BridgeSource("object_0001", 9, source_mask)],
                target_frame=12,
                max_gap_frames=4,
            )

        self.assertEqual(observations[0].global_object_id, "object_0001")
        self.assertTrue(np.array_equal(observations[0].mask, target_mask > 0.0))
        self.assertEqual(propagate.call_args.kwargs["frame_start"], 9)
        self.assertEqual(propagate.call_args.kwargs["frame_end"], 12)

    def test_rejects_a_long_hidden_gap(self):
        mask = np.ones((4, 4), dtype=bool)

        with self.assertRaisesRegex(ValueError, "exceeds"):
            bridge_hidden_gap(
                object(),
                video_path=Path("video.mp4"),
                sources=[BridgeSource("object_0001", 1, mask)],
                target_frame=10,
                max_gap_frames=4,
            )


if __name__ == "__main__":
    unittest.main()
