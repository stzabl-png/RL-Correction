"""Contract tests for point-prompt, multi-object SAM2 propagation."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SAM2_OBJECT_DIR = Path(__file__).resolve().parents[1]
RECON_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RECON_ROOT))
sys.path.insert(0, str(SAM2_OBJECT_DIR))

import run_sequence as run_sequence_module  # noqa: E402
import sam2_object_common as common  # noqa: E402


def _object(object_id: str, frame_idx: int, point: tuple[float, float]) -> dict:
    return {
        "object_id": object_id,
        "frame_idx": frame_idx,
        "points": [list(point)],
        "labels": [1],
        "locked": True,
    }


class LabelPromptContractTest(unittest.TestCase):
    def test_multi_object_prompts_keep_independent_frames_and_points(self):
        prompt = common.LabelPrompt.from_json(
            {
                "schema_version": "sam2_object_prompt_v2",
                "objects": [
                    _object("dustpan", 55, (620.0, 430.0)),
                    _object("broom", 61, (810.0, 390.0)),
                ],
            }
        )
        self.assertEqual(prompt.object_ids(), ["dustpan", "broom"])
        self.assertEqual(prompt.objects[0].frame_idx, 55)
        self.assertEqual(prompt.objects[1].frame_idx, 61)
        self.assertEqual(prompt.objects[1].points, [(810.0, 390.0)])

    def test_legacy_single_object_prompt_remains_supported(self):
        prompt = common.LabelPrompt.from_json(
            {"frame_idx": 3, "points": [[12, 18]], "labels": [1]}
        )
        self.assertEqual(prompt.object_ids(), ["object_0"])

    def test_rejects_invalid_point_contracts(self):
        invalid_objects = [
            {"frame_idx": -1, "points": [[1, 2]], "labels": [1]},
            {"frame_idx": 0, "points": [], "labels": []},
            {"frame_idx": 0, "points": [[1, 2]], "labels": []},
            {"frame_idx": 0, "points": [[1, 2]], "labels": [2]},
            {"frame_idx": 0, "points": [[1, 2]], "labels": [0]},
            {"frame_idx": 0, "points": [[float("nan"), 2]], "labels": [1]},
        ]
        for payload in invalid_objects:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                common.ObjectPrompt.from_json(payload)

    def test_rejects_duplicate_ids_and_out_of_video_coordinates(self):
        with self.assertRaises(ValueError):
            common.LabelPrompt.from_json(
                {"objects": [_object("same", 0, (1, 1)), _object("same", 1, (2, 2))]}
            )

        prompt = common.LabelPrompt.from_json(
            {"objects": [_object("object_0", 5, (100.0, 10.0))]}
        )
        with self.assertRaises(ValueError):
            common.validate_label_prompt_for_video(
                prompt, num_frames=5, height=50, width=100
            )


class PropagationLifecycleTest(unittest.TestCase):
    def test_multi_object_run_clears_stale_outputs_and_keeps_prompt(self):
        with tempfile.TemporaryDirectory() as tmp:
            step_dir = Path(tmp)
            prompt = common.LabelPrompt.from_json(
                {
                    "objects": [
                        _object("cup", 3, (40, 50)),
                        _object("bottle", 4, (140, 55)),
                    ]
                }
            )
            common.save_label_prompt(step_dir, prompt)
            (step_dir / "frame_plan.json").write_text("{}", encoding="utf-8")
            stale = step_dir / "video_segmentation" / "masks" / "stale.png"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"old")
            stale_vis = step_dir / "vis" / "old.mp4"
            stale_vis.parent.mkdir(parents=True)
            stale_vis.write_bytes(b"old")

            def fake_segment(_predictor, **kwargs):
                return {
                    "num_frames": 8,
                    "detected_frames": 8,
                    "prompt_frame_idx": kwargs["frame_idx"],
                    "backend": "sam2",
                    "object_id": kwargs["object_id"],
                    "mask_filename": f"{kwargs['object_id']}.png",
                }

            with (
                mock.patch.object(common, "count_video_frames", return_value=8),
                mock.patch.object(common, "read_frame_size", return_value=(100, 200)),
                mock.patch.object(common, "build_object_predictor", return_value=object()),
                mock.patch.object(common, "segment_object_on_video", side_effect=fake_segment) as segment,
                mock.patch.object(common, "discard_sam2_object_nonessential"),
            ):
                stats = common.run_object_masks(
                    video_path=Path("video.mp4"),
                    step_dir=step_dir,
                    video_id="sample",
                    gpu_id=0,
                    visualize=False,
                )

            self.assertEqual(segment.call_count, 2)
            self.assertEqual(stats["object_ids"], ["cup", "bottle"])
            self.assertEqual([row["frame_idx"] for row in stats["objects"]], [3, 4])
            self.assertFalse(stale.exists())
            self.assertFalse(stale_vis.exists())
            self.assertTrue(common.label_prompt_path(step_dir).is_file())
            self.assertTrue((step_dir / "frame_plan.json").is_file())

    def test_failed_force_run_invalidates_old_completion_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            step_dir = Path(tmp)
            common.save_label_prompt(
                step_dir,
                common.LabelPrompt.from_json({"objects": [_object("object_0", 0, (1, 1))]}),
            )
            marker = step_dir / "sam2_object_complete.json"
            marker.write_text(json.dumps({"status": "complete"}), encoding="utf-8")

            with (
                mock.patch.object(run_sequence_module, "interim_step_dir", return_value=step_dir),
                mock.patch.object(run_sequence_module, "is_step_complete", return_value=True),
                mock.patch.object(
                    run_sequence_module,
                    "run_object_masks",
                    side_effect=RuntimeError("simulated failure"),
                ),
                self.assertRaisesRegex(RuntimeError, "simulated failure"),
            ):
                run_sequence_module.main(
                    [
                        "--dataset",
                        "test",
                        "--video-id",
                        "sample",
                        "--video",
                        "video.mp4",
                        "--force",
                    ]
                )
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
