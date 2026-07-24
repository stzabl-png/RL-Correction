import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.hoi_detr.instance_association import InstanceCandidate
from experiments.hoi_detr.instance_registry import write_instance_registry


class InstanceRegistryTests(unittest.TestCase):
    def test_writes_relation_free_component_registry(self):
        first = np.zeros((12, 12), dtype=bool)
        second = np.zeros((12, 12), dtype=bool)
        first[1:4, 1:4] = True
        second[7:10, 7:10] = True
        with tempfile.TemporaryDirectory() as temp_dir:
            result = write_instance_registry(
                [InstanceCandidate("candidate_a", first), InstanceCandidate("candidate_b", second)],
                source_frame=12,
                interaction_start_frame=10,
                interaction_end_frame=30,
                component_decision={
                    "status": "multiple_components",
                    "selected_candidate_ids": ["candidate_a", "candidate_b"],
                },
                output_dir=Path(temp_dir) / "registry",
            )

            registry = __import__("json").loads(Path(result["reconstruction_registry"]).read_text())

        self.assertEqual(result["object_ids"], ["instance_0001", "instance_0002"])
        self.assertNotIn("relations", registry)
        self.assertEqual(registry["objects"]["instance_0001"]["activation_frame"], 10)

    def test_rejects_overlapping_component_masks(self):
        first = np.ones((4, 4), dtype=bool)
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "overlap"):
                write_instance_registry(
                    [InstanceCandidate("candidate_a", first), InstanceCandidate("candidate_b", first)],
                    source_frame=1,
                    interaction_start_frame=1,
                    interaction_end_frame=2,
                    component_decision={
                        "selected_candidate_ids": ["candidate_a", "candidate_b"]
                    },
                    output_dir=Path(temp_dir) / "registry",
                )


if __name__ == "__main__":
    unittest.main()
