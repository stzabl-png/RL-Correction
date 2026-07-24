import tempfile
import unittest
from pathlib import Path

from experiments.hoi_detr.segmentation_report import (
    render_segmentation_report,
    summarize_instances_by_episode,
    write_segmentation_report,
)


class SegmentationReportTests(unittest.TestCase):
    def test_reports_episode_instance_counts_and_output_mask_duration(self):
        manifest = {
            "interaction_keyframes": [
                {
                    "cycle_idx": 0,
                    "frame_idx": 10,
                    "start_frame": 10,
                    "end_frame": 12,
                    "object_ids": ["object_0001", "object_0002"],
                },
                {
                    "cycle_idx": 1,
                    "frame_idx": 20,
                    "start_frame": 20,
                    "end_frame": 21,
                    "object_ids": ["object_0001"],
                },
            ],
            "frames": [
                {
                    "frame_idx": 10,
                    "objects": {
                        "object_0001": {"mask": "a.png"},
                        "object_0002": {"mask": None},
                    },
                },
                {
                    "frame_idx": 11,
                    "objects": {
                        "object_0001": {"mask": "b.png"},
                        "object_0002": {"mask": "c.png"},
                    },
                },
                {
                    "frame_idx": 12,
                    "objects": {
                        "object_0001": {"mask": None},
                        "object_0002": {"mask": "d.png"},
                    },
                },
                {
                    "frame_idx": 20,
                    "objects": {"object_0001": {"mask": "e.png"}},
                },
                {
                    "frame_idx": 21,
                    "objects": {"object_0001": {"mask": "f.png"}},
                },
            ],
        }

        summary = summarize_instances_by_episode(manifest)

        self.assertEqual(summary["episode_count"], 2)
        self.assertEqual(summary["episodes"][0]["instance_count"], 2)
        self.assertEqual(
            summary["episodes"][0]["instances"],
            [
                {"instance_id": "object_0001", "duration_frames": 2},
                {"instance_id": "object_0002", "duration_frames": 2},
            ],
        )
        self.assertEqual(
            summary["episodes"][1]["instances"],
            [{"instance_id": "object_0001", "duration_frames": 2}],
        )
        report = render_segmentation_report(summary)
        self.assertIn("Total episodes: 2", report)
        self.assertIn("| episode_00 | 2 | object_0002 | 2 |", report)
        self.assertNotIn("accepted", report.lower())
        self.assertNotIn("status", report.lower())

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "segmentation_report.md"
            write_segmentation_report(manifest, output_path)
            self.assertEqual(output_path.read_text(encoding="utf-8"), report)


if __name__ == "__main__":
    unittest.main()
