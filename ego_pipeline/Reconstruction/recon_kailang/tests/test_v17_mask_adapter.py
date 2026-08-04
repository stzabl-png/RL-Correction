from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


RECONSTRUCTION_ROOT = Path(__file__).resolve().parents[2]
KAILANG_ROOT = RECONSTRUCTION_ROOT / "recon_kailang"
PIPELINE_ROOT = RECONSTRUCTION_ROOT / "recon_pipeline"
SAM2_OBJECT_ROOT = PIPELINE_ROOT / "sam2_object"
for path in (KAILANG_ROOT, PIPELINE_ROOT, SAM2_OBJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v17_mask_adapter import import_v17a_masks as adapter  # noqa: E402


class ImportV17AMasksTest(unittest.TestCase):
    def test_imports_selected_track_and_zero_fills_other_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shape = (8, 10)
            source_mask = np.zeros(shape, dtype=np.uint8)
            source_mask[2:6, 3:8] = 255
            mask_path = root / "source.png"
            self.assertTrue(cv2.imwrite(str(mask_path), source_mask))

            manifest = {
                "schema_version": "persistent_mask_sequence_v1",
                "status": "ready",
                "frame_shape": list(shape),
                "object_ids": ["object_2"],
                "frames": [
                    {"frame_idx": 0, "objects": {}},
                    {
                        "frame_idx": 1,
                        "objects": {
                            "object_2": {
                                "status": "accepted",
                                "mask": str(mask_path),
                            }
                        },
                    },
                    {
                        "frame_idx": 2,
                        "objects": {
                            "object_2": {
                                "status": "rejected",
                                "mask": str(mask_path),
                            }
                        },
                    },
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            video_path = root / "input.mp4"
            video_path.touch()
            step_dir = root / "sam2_object"

            with mock.patch.object(adapter, "_video_geometry", return_value=(3, *shape)):
                metadata = adapter.import_v17a_sequence(
                    video_path=video_path,
                    manifest_path=manifest_path,
                    step_dir=step_dir,
                    dataset="egodex",
                    video_id="test__basic_pick_place__1",
                    source_object_id="object_2",
                    reconstruction_frame=1,
                    object_name="phone",
                )

            self.assertEqual(metadata["detected_frames"], 1)
            self.assertEqual(metadata["reconstruction_frame"], 1)
            masks_root = step_dir / "video_segmentation" / "masks"
            imported = []
            for frame_idx in range(3):
                path = (
                    masks_root
                    / f"frame_{frame_idx:06d}_masks"
                    / adapter.object_mask_filename(adapter.OBJECT_MASK_ID)
                )
                imported.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
            self.assertEqual(int(np.count_nonzero(imported[0])), 0)
            np.testing.assert_array_equal(imported[1], source_mask)
            self.assertEqual(int(np.count_nonzero(imported[2])), 0)

            prompt = json.loads(
                (step_dir / "label_prompt.json").read_text(encoding="utf-8")
            )
            point = prompt["objects"][0]["points"][0]
            x, y = (int(round(value)) for value in point)
            self.assertGreater(source_mask[y, x], 0)
            completion = json.loads(
                (step_dir / "sam2_object_complete.json").read_text(encoding="utf-8")
            )
            self.assertEqual(completion["status"], "complete")
            self.assertEqual(completion["backend"], "v17a_manifest_import")

    def test_rejects_unready_manifest(self):
        manifest = {
            "schema_version": "persistent_mask_sequence_v1",
            "status": "failed",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not ready"):
                adapter._load_manifest(path)


if __name__ == "__main__":
    unittest.main()
