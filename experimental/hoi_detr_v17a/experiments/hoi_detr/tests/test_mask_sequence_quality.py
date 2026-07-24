import unittest

import numpy as np

from experiments.hoi_detr.mask_sequence_quality import gate_mask_sequence


def _square(x, y, size=20, shape=(100, 100)):
    mask = np.zeros(shape, dtype=bool)
    mask[y : y + size, x : x + size] = True
    return mask


class MaskSequenceQualityTests(unittest.TestCase):
    def test_transient_background_spike_is_rejected_without_poisoning_history(self):
        masks = {
            0: _square(10, 10),
            1: _square(12, 10),
            2: np.ones((100, 100), dtype=bool),
            3: _square(14, 10),
        }
        decisions = gate_mask_sequence(
            masks,
            object_id="part_0",
            seed_frame=0,
            frame_shape=(100, 100),
        )
        self.assertEqual(decisions[2]["status"], "rejected_temporal_outlier")
        self.assertIn("covers_too_much_of_frame", decisions[2]["reasons"])
        self.assertEqual(decisions[3]["status"], "accepted")

    def test_bidirectional_history_starts_from_seed(self):
        masks = {
            0: _square(8, 10),
            1: _square(10, 10),
            2: _square(12, 10),
            3: _square(14, 10),
            4: _square(16, 10),
        }
        decisions = gate_mask_sequence(
            masks,
            object_id="part_0",
            seed_frame=2,
            frame_shape=(100, 100),
        )
        self.assertTrue(all(item["status"] == "accepted" for item in decisions.values()))

    def test_fragmented_mask_is_rejected(self):
        fragmented = _square(10, 10)
        fragmented[50:62, 50:62] = True
        decisions = gate_mask_sequence(
            {0: _square(10, 10), 1: fragmented},
            object_id="part_0",
            seed_frame=0,
            frame_shape=(100, 100),
        )
        self.assertIn("fragmented_mask", decisions[1]["reasons"])

    def test_missing_seed_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "seed frame"):
            gate_mask_sequence(
                {0: _square(10, 10)},
                object_id="part_0",
                seed_frame=1,
                frame_shape=(100, 100),
            )


if __name__ == "__main__":
    unittest.main()
