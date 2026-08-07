from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tasks.pour.dexonomy_adapter import convert_candidate


class DexonomyAdapterTest(unittest.TestCase):
    def test_side_verified_candidate_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.npy"
            info = root / "simplified.json"
            output = root / "prior.npz"
            row = np.r_[np.array([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]), np.arange(22)]
            candidate = {
                "hand_name": "sharpa_wave_left",
                "grasp_qpos": row[None],
                "squeeze_qpos": row[None],
                "pregrasp_qpos": np.stack((row, row)),
                "ho_c": {
                    "pos": np.array([[0.0, 0.1, 0.0], [0.0, -0.1, 0.0]]),
                    "normal": np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
                },
            }
            np.save(source, candidate)
            info.write_text(
                json.dumps({"canonical_from_input_rot_wxyz": [1.0, 0.0, 0.0, 0.0]}),
                encoding="utf-8",
            )

            convert_candidate(source, info, output, side="left")
            with np.load(output, allow_pickle=False) as prior:
                self.assertEqual(prior["grasp"].shape, (29,))
                self.assertEqual(prior["pregrasp"].shape, (2, 29))
                np.testing.assert_allclose(prior["contact_centroid"], np.zeros(3))
                self.assertEqual(
                    prior["hand_side"].tobytes().decode("utf-8"), "left"
                )

    def test_rejects_wrong_hand_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "candidate.npy"
            info = root / "simplified.json"
            row = np.r_[np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.zeros(22)]
            np.save(
                source,
                {
                    "hand_name": "sharpa_wave",
                    "grasp_qpos": row[None],
                    "squeeze_qpos": row[None],
                    "pregrasp_qpos": row[None],
                    "ho_c": {"pos": np.ones((1, 3)), "normal": np.ones((1, 3))},
                },
            )
            info.write_text(
                json.dumps({"canonical_from_input_rot_wxyz": [1.0, 0.0, 0.0, 0.0]}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected 'sharpa_wave_left'"):
                convert_candidate(source, info, root / "prior.npz", side="left")


if __name__ == "__main__":
    unittest.main()
