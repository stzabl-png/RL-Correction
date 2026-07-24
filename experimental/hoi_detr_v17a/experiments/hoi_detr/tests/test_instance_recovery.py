import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from experiments.hoi_detr.run_instance_recovery import run


def _write_mask(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    return str(path.resolve())


class InstanceRecoveryTests(unittest.TestCase):
    def test_reconditions_all_ids_from_mutually_exclusive_candidates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = np.zeros((20, 20), dtype=bool)
            second = np.zeros((20, 20), dtype=bool)
            first[2:7, 2:7] = True
            second[12:18, 12:18] = True
            registry = {
                "status": "ready",
                "objects": {
                    "instance_0001": {
                        "mask": _write_mask(root / "anchor_a.png", first),
                        "conditioning_masks": [{"frame_idx": 0, "mask": _write_mask(root / "anchor_a2.png", first)}],
                    },
                    "instance_0002": {
                        "mask": _write_mask(root / "anchor_b.png", second),
                        "conditioning_masks": [{"frame_idx": 0, "mask": _write_mask(root / "anchor_b2.png", second)}],
                    },
                },
            }
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps(registry))
            unresolved = root / "unresolved"
            _write_mask(unresolved / "instance_0001.png", first)
            _write_mask(unresolved / "instance_0002.png", second)
            summary = {
                "status": "success",
                "candidates": [
                    {"candidate_idx": 0, "mask": _write_mask(root / "candidate_a.png", first), "predicted_iou": 0.95, "stability_score": 0.95},
                    {"candidate_idx": 1, "mask": _write_mask(root / "candidate_b.png", second), "predicted_iou": 0.95, "stability_score": 0.95},
                ],
            }
            candidate_summary = root / "amg.json"
            candidate_summary.write_text(json.dumps(summary))

            result = run(
                type("Args", (), {
                    "video": root / "video.mp4",
                    "source_registry": registry_path,
                    "candidate_summary": candidate_summary,
                    "unresolved_mask_dir": unresolved,
                    "recovery_frame": 5,
                    "output_dir": root / "recovery",
                    "min_child_inside_parent": 0.9,
                    "min_residual_area": 1,
                    "max_cross_instance_overlap_fraction": 0.05,
                    "min_assignment_margin": 0.05,
                })()
            )

            repaired = json.loads(Path(result["reconstruction_registry"]).read_text())
            self.assertEqual(result["status"], "success")
            self.assertEqual(len(repaired["objects"]["instance_0001"]["conditioning_masks"]), 2)
            self.assertEqual(len(repaired["objects"]["instance_0002"]["conditioning_masks"]), 2)


if __name__ == "__main__":
    unittest.main()
