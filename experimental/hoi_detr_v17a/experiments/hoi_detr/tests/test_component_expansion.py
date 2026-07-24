import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from experiments.hoi_detr.component_expansion import (
    append_expansion_component,
    lock_known_components_from_logits,
    select_unexplained_component,
)
from experiments.hoi_detr.instance_association import InstanceCandidate


class ComponentExpansionTests(unittest.TestCase):
    def test_known_component_lock_caps_a_composite_expansion(self):
        reference = np.zeros((20, 20), dtype=bool)
        reference[4:8, 7:13] = True
        logits = np.full((20, 20), -4.0, dtype=np.float32)
        logits[4:8, 7:13] = 6.0
        logits[8:17, 5:15] = 0.5

        result = lock_known_components_from_logits(
            {"instance_0001": reference},
            {"instance_0001": logits},
            max_area_growth=1.25,
        )

        self.assertEqual(result["status"], "success")
        locked = result["masks"]["instance_0001"]
        self.assertTrue(result["metrics"]["instance_0001"]["area_cap_applied"])
        self.assertTrue(np.all(locked <= (logits > 0)))
        self.assertLessEqual(int(locked.sum()), int(round(reference.sum() * 1.25)))
        self.assertGreaterEqual(int((locked & reference).sum()), int(reference.sum() * 0.9))

    def test_known_component_lock_rejects_a_lost_track(self):
        reference = np.zeros((20, 20), dtype=bool)
        reference[4:10, 6:14] = True
        logits = np.full((20, 20), -2.0, dtype=np.float32)
        logits[5:7, 7:9] = 1.0

        result = lock_known_components_from_logits(
            {"instance_0001": reference},
            {"instance_0001": logits},
        )

        self.assertEqual(result["status"], "failed_known_component_lost_at_expansion")

    def test_known_component_lock_keeps_nearby_occlusion_fragments(self):
        reference = np.zeros((30, 30), dtype=bool)
        reference[8:14, 7:23] = True
        logits = np.full((30, 30), -3.0, dtype=np.float32)
        logits[8:14, 7:13] = 5.0
        logits[8:14, 17:23] = 4.5
        logits[14:25, 5:25] = 0.4

        result = lock_known_components_from_logits(
            {"instance_0001": reference},
            {"instance_0001": logits},
            min_area_retention=0.5,
            max_area_growth=1.25,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["metrics"]["instance_0001"]["retained_component_count"],
            2,
        )
        self.assertGreaterEqual(
            int(result["masks"]["instance_0001"].sum()),
            int(reference.sum() * 0.5),
        )

    def test_known_component_lock_uses_fixed_area_reference_across_frames(self):
        spatial_reference = np.zeros((20, 20), dtype=bool)
        spatial_reference[5:10, 7:12] = True
        logits = np.full((20, 20), -3.0, dtype=np.float32)
        logits[5:10, 7:12] = 5.0
        logits[10:18, 5:15] = 0.5

        result = lock_known_components_from_logits(
            {"instance_0001": spatial_reference},
            {"instance_0001": logits},
            area_reference_pixels={"instance_0001": 16},
            min_area_retention=0.5,
            max_area_growth=1.25,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["metrics"]["instance_0001"]["area_reference_pixels"], 16
        )
        self.assertLessEqual(int(result["masks"]["instance_0001"].sum()), 20)

    def test_uses_composite_residual_without_creating_a_whole_object_id(self):
        lid = np.zeros((20, 20), dtype=bool)
        lid[4:7, 7:13] = True
        composite = np.zeros((20, 20), dtype=bool)
        composite[4:16, 6:14] = True

        selection = select_unexplained_component(
            [InstanceCandidate("whole_candidate", composite, 0.95)],
            existing_masks={"instance_0001": lid},
            interaction_box_xyxy=[5, 3, 15, 17],
            min_area_pixels=10,
        )

        self.assertEqual(selection["status"], "success")
        self.assertEqual(selection["selected"]["proposal_type"], "composite_residual")
        self.assertFalse(np.any(selection["selected"]["mask"] & lid))
        self.assertEqual(int(selection["selected"]["mask"].sum()), int(composite.sum() - lid.sum()))

    def test_rejects_a_candidate_that_only_repeats_the_known_component(self):
        lid = np.zeros((20, 20), dtype=bool)
        lid[4:7, 7:13] = True

        selection = select_unexplained_component(
            [InstanceCandidate("repeat", lid, 0.95)],
            existing_masks={"instance_0001": lid},
            interaction_box_xyxy=[5, 3, 15, 17],
            min_area_pixels=10,
        )

        self.assertEqual(selection["status"], "failed_no_unexplained_visible_component")

    def test_new_component_activates_at_expansion_not_original_interaction_start(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first_mask = root / "first.png"
            cv2.imwrite(str(first_mask), np.full((12, 12), 255, dtype=np.uint8))
            source = root / "registry.json"
            source.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "objects": {
                            "instance_0001": {
                                "activation_frame": 2,
                                "frame_idx": 2,
                                "mask": str(first_mask),
                                "conditioning_masks": [],
                            }
                        },
                        "object_order": ["instance_0001"],
                    }
                ),
                encoding="utf-8",
            )
            existing = np.zeros((12, 12), dtype=bool)
            existing[2:5, 3:9] = True
            new = np.zeros((12, 12), dtype=bool)
            new[5:10, 3:9] = True
            later_existing = np.zeros((12, 12), dtype=bool)
            later_existing[2:5, 4:10] = True
            later_new = np.zeros((12, 12), dtype=bool)
            later_new[5:10, 4:10] = True
            result = append_expansion_component(
                source,
                existing_masks={"instance_0001": existing},
                event={"activation_frame":8, "seed_frame":10, "box_xyxy":[2, 1, 10, 11]},
                selection={"status":"success","selected":{"mask":new,"candidate_id":"whole","candidate_source":"sam2","candidate_quality_score":0.9,"proposal_type":"composite_residual"}},
                output_dir=root / "expanded",
                known_component_area_references={"instance_0001": 24},
                interaction_envelope_mask=existing | new,
                conditioning_anchors=[
                    {
                        "frame_idx": 12,
                        "existing_masks": {"instance_0001": later_existing},
                        "new_mask": later_new,
                        "candidate_id": "later_whole",
                        "candidate_source": "sam2",
                        "candidate_quality_score": 0.92,
                        "proposal_type": "composite_residual",
                    }
                ],
            )
            registry = json.loads(Path(result["reconstruction_registry"]).read_text(encoding="utf-8"))

        self.assertEqual(registry["objects"]["instance_0002"]["activation_frame"], 8)
        self.assertEqual(registry["objects"]["instance_0002"]["frame_idx"], 10)
        expansion = registry["confirmed_hoi_box_expansions"][0]
        self.assertEqual(expansion["known_object_ids"], ["instance_0001"])
        self.assertEqual(
            expansion["known_component_area_references"],
            {"instance_0001": 24},
        )
        self.assertEqual(
            expansion["interaction_envelope"]["object_id"],
            "__interaction_envelope_0000",
        )
        self.assertNotIn("__interaction_envelope_0000", registry["objects"])
        self.assertNotIn("__interaction_envelope_0000", registry["object_order"])
        self.assertEqual(expansion["conditioning_anchor_frames"], [10, 12])
        self.assertEqual(
            [
                item["frame_idx"]
                for item in registry["objects"]["instance_0002"]["conditioning_masks"]
            ],
            [10, 12],
        )


if __name__ == "__main__":
    unittest.main()
