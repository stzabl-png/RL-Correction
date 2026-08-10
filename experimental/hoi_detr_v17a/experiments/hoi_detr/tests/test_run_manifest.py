from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from ..run_manifest import finish_manifest, start_manifest


class RunManifestTest(unittest.TestCase):
    def test_manifest_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "input.mp4"
            detections = root / "detections.json"
            checkpoint = root / "sam2.pt"
            video.write_bytes(b"video")
            detections.write_text("{}\n", encoding="utf-8")
            checkpoint.write_bytes(b"weights")
            output = root / "run"
            output.mkdir()
            args = argparse.Namespace(
                video=video,
                detections=detections,
                checkpoint=checkpoint,
                output_dir=output,
                gpu=0,
            )

            running = start_manifest(output, args=args, command=["python", "-m", "v17a"])
            self.assertTrue(running.is_file())
            (output / "summary.json").write_text(
                '{"status": "success"}\n', encoding="utf-8"
            )
            final = finish_manifest(output, {"status": "success"})

            self.assertIsNotNone(final)
            payload = json.loads(final.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["validation"]["pipeline_status"], "success")
            self.assertIn("sha256", payload["inputs"]["video"])
            self.assertFalse(running.exists())


if __name__ == "__main__":
    unittest.main()
