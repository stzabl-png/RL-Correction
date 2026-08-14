"""Schema compatibility tests for the HOI-DETR review renderer."""

from __future__ import annotations

import unittest

from experiments.hoi_detr.visualize import _box, _link_indices, _links


class VisualizeSchemaTest(unittest.TestCase):
    def test_upstream_schema(self):
        record = {"hf": [{"a": 0, "b": 1, "prob": 0.9}], "fs": []}
        self.assertEqual(_box({"box": [1, 2, 3, 4]}), [1, 2, 3, 4])
        self.assertEqual(_links(record, "hf"), record["hf"])
        self.assertEqual(_link_indices(record["hf"][0]), (0, 1))

    def test_normalized_schema(self):
        link = {
            "source_detection_index": 2,
            "target_detection_index": 4,
            "prob": 0.8,
        }
        record = {"links": {"hf": [link], "fs": []}}
        self.assertEqual(_box({"box_xyxy": [5, 6, 7, 8]}), [5, 6, 7, 8])
        self.assertEqual(_links(record, "hf"), [link])
        self.assertEqual(_link_indices(link), (2, 4))

    def test_missing_box_is_rejected(self):
        with self.assertRaises(ValueError):
            _box({"score": 0.9})


if __name__ == "__main__":
    unittest.main()
