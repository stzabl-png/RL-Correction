import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from experiments.hoi_detr.run_instance_video_segmentation import _sequence_endpoint_observations


class InstanceVideoSegmentationTests(unittest.TestCase):
    def test_endpoint_observations_keep_first_and_last_accepted_frame_indices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_mask = root / "first.png"
            last_mask = root / "last.png"
            cv2.imwrite(str(first_mask), np.full((4, 4), 255, dtype=np.uint8))
            cv2.imwrite(str(last_mask), np.full((4, 4), 255, dtype=np.uint8))
            manifest = {
                "object_ids": ["instance_0001"],
                "frames": [
                    {
                        "frame_idx": 4,
                        "objects": {
                            "instance_0001": {"status": "accepted", "mask": str(first_mask)}
                        },
                    },
                    {
                        "frame_idx": 5,
                        "objects": {
                            "instance_0001": {"status": "rejected", "mask": None}
                        },
                    },
                    {
                        "frame_idx": 6,
                        "objects": {
                            "instance_0001": {"status": "accepted", "mask": str(last_mask)}
                        },
                    },
                ],
            }

            first = _sequence_endpoint_observations(manifest, at_start=True)
            last = _sequence_endpoint_observations(manifest, at_start=False)

        self.assertEqual(first[0][1], 4)
        self.assertEqual(last[0][1], 6)
