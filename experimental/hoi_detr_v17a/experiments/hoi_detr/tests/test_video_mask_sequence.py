import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from experiments.hoi_detr.video_mask_sequence import combine_mask_sequences


def _write_mask(path: Path, x1: int, x2: int) -> str:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[4:12, x1:x2] = 255
    cv2.imwrite(str(path), mask)
    return str(path)


def _manifest(mask_paths, *, seed_frame):
    frames = []
    for frame_idx, (mask_path, conditioned) in enumerate(mask_paths):
        frames.append(
            {
                "frame_idx": frame_idx,
                "objects": {
                    "part_0": {
                        "status": "accepted",
                        "reasons": [],
                        "metrics": {"object_id": "part_0"},
                        "conditioning_frame": conditioned,
                        "raw_mask": mask_path,
                        "mask": mask_path,
                    }
                },
            }
        )
    return {
        "status": "ready",
        "frame_shape": [20, 20],
        "frame_window": [0, len(frames) - 1],
        "source_registry": "registry.json",
        "object_ids": ["part_0"],
        "objects": {
            "part_0": {
                "seed_frame": seed_frame,
                "conditioning_frames": [seed_frame],
            }
        },
        "frames": frames,
    }


class VideoMaskSequenceTests(unittest.TestCase):
    def test_conditioning_mask_wins_over_conflicting_propagation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = [
                (_write_mask(root / "a0.png", 1, 5), True),
                (_write_mask(root / "a1.png", 5, 11), False),
            ]
            second = [
                (_write_mask(root / "b0.png", 14, 18), False),
                (_write_mask(root / "b1.png", 7, 13), True),
            ]
            registry = {
                "status": "ready",
                "video": "video.mp4",
                "cycles": [
                    {
                        "cycle_idx": 0,
                        "object_ids": ["part_0"],
                        "interaction_onset_frame": 0,
                    },
                    {
                        "cycle_idx": 1,
                        "object_ids": ["part_1"],
                        "interaction_onset_frame": 1,
                    },
                ],
            }
            combined = combine_mask_sequences(
                registry,
                [_manifest(first, seed_frame=0), _manifest(second, seed_frame=1)],
            )
            frame_one = combined["frames"][1]["objects"]
            self.assertEqual(frame_one["part_0"]["status"], "rejected_cross_object_overlap")
            self.assertEqual(frame_one["part_1"]["status"], "accepted")
            self.assertEqual(combined["object_ids"], ["part_0", "part_1"])
            self.assertEqual(combined["status"], "ready")
            self.assertNotIn("part_1", combined["frames"][0]["objects"])

    def test_two_propagated_masks_are_both_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = [
                (_write_mask(root / "a0.png", 1, 5), True),
                (_write_mask(root / "a1.png", 5, 11), False),
            ]
            second = [
                (_write_mask(root / "b0.png", 14, 18), True),
                (_write_mask(root / "b1.png", 7, 13), False),
            ]
            registry = {
                "status": "ready",
                "video": "video.mp4",
                "cycles": [
                    {"cycle_idx": 0, "object_ids": ["part_0"]},
                    {"cycle_idx": 1, "object_ids": ["part_1"]},
                ],
            }
            combined = combine_mask_sequences(
                registry,
                [_manifest(first, seed_frame=0), _manifest(second, seed_frame=0)],
            )
            frame_one = combined["frames"][1]["objects"]
            self.assertTrue(
                all(
                    entry["status"] == "rejected_cross_object_overlap"
                    for entry in frame_one.values()
                )
            )

    def test_reuses_one_global_id_in_disjoint_interaction_windows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = [
                (_write_mask(root / f"mask_{index}.png", 2, 7), index == 0)
                for index in range(4)
            ]
            registry = {
                "status": "ready",
                "video": "video.mp4",
                "objects": {
                    "object_0001": {
                        "cycle_observations": [
                            {
                                "cycle_idx": 0,
                                "source_object_id": "part_0",
                                "activation_frame": 0,
                            },
                            {
                                "cycle_idx": 1,
                                "source_object_id": "part_0",
                                "activation_frame": 2,
                            },
                        ]
                    }
                },
                "cycles": [
                    {
                        "cycle_idx": 0,
                        "object_ids": ["object_0001"],
                        "source_object_id_map": {"part_0": "object_0001"},
                        "interaction_onset_frame": 0,
                        "interaction_end_frame": 1,
                    },
                    {
                        "cycle_idx": 1,
                        "object_ids": ["object_0001"],
                        "source_object_id_map": {"part_0": "object_0001"},
                        "interaction_onset_frame": 2,
                        "interaction_end_frame": 3,
                    },
                ],
            }
            first = _manifest(paths, seed_frame=0)
            second = _manifest(paths, seed_frame=2)

            combined = combine_mask_sequences(
                registry,
                [first, second],
            )

            self.assertEqual(combined["object_ids"], ["object_0001"])
            self.assertEqual(len(combined["objects"]["object_0001"]["interaction_spans"]), 2)
            self.assertEqual(
                [list(frame["objects"]) for frame in combined["frames"]],
                [["object_0001"], ["object_0001"], ["object_0001"], ["object_0001"]],
            )


if __name__ == "__main__":
    unittest.main()
