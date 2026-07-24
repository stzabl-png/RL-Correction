import unittest

from experiments.hoi_detr.video_registry import combine_cycle_registries


def _registry(frame):
    return {
        "status": "ready",
        "objects": {
            "lid_0": {"frame_idx": frame, "mask": f"lid_{frame}.png"},
            "cup_0": {"frame_idx": frame, "mask": f"cup_{frame}.png"},
        },
        "relations": [
            {
                "kind": "supports",
                "support_object_id": "cup_0",
                "part_object_id": "lid_0",
                "evidence_frame_idx": frame - 5,
            }
        ],
    }


class VideoRegistryTests(unittest.TestCase):
    def test_relation_free_cycle_registries_emit_no_relation_field(self):
        registry = {
            "status": "ready",
            "objects": {"instance_0001": {"frame_idx": 4, "mask": "mask.png"}},
            "object_order": ["instance_0001"],
        }

        combined = combine_cycle_registries([registry], video="video.mp4")

        self.assertNotIn("relations", combined)

    def test_assigns_unique_ids_and_rewrites_relations(self):
        metadata = [
            {
                "interaction_onset_frame": 19,
                "interaction_rising_edge_frame": 16,
                "interaction_keyframe_offset_frames": 3,
                "interaction_onset_source": "stable_hand_firstobject_link_rising_edge",
                "interaction_onset_confidence": 0.9,
                "interaction_onset_evidence_frames": [19, 20, 21],
                "interaction_event_frame": 47,
                "interaction_end_frame": 83,
                "interaction_raw_end_frame": 80,
                "interaction_end_source": "stable_hand_object_link_falling_edge_plus_delay",
                "interaction_end_truncated": False,
            },
            {
                "interaction_onset_frame": 84,
                "interaction_rising_edge_frame": 81,
                "interaction_keyframe_offset_frames": 3,
                "interaction_onset_source": "stable_hand_firstobject_link_rising_edge",
                "interaction_onset_confidence": 0.8,
                "interaction_onset_evidence_frames": [84, 85, 86],
                "interaction_event_frame": 108,
                "interaction_end_frame": 141,
                "interaction_raw_end_frame": 139,
                "interaction_end_source": "stable_hand_object_link_falling_edge_plus_delay",
                "interaction_end_truncated": False,
            },
        ]
        combined = combine_cycle_registries(
            [_registry(85), _registry(141)],
            video="1.mp4",
            cycle_metadata=metadata,
        )
        self.assertEqual(
            set(combined["objects"]),
            {"object_0001", "object_0002", "object_0003", "object_0004"},
        )
        self.assertEqual(combined["relations"][1]["support_object_id"], "object_0004")
        self.assertEqual(combined["relations"][1]["part_object_id"], "object_0003")
        self.assertEqual(combined["objects"]["object_0003"]["source_object_id"], "lid_0")
        self.assertEqual(combined["interaction_keyframes"][1]["frame_idx"], 84)
        self.assertEqual(combined["interaction_keyframes"][1]["end_frame"], 141)
        self.assertEqual(combined["interaction_keyframes"][1]["rising_edge_frame"], 81)
        self.assertEqual(combined["interaction_keyframes"][1]["offset_frames"], 3)
        self.assertEqual(
            combined["interaction_keyframes"][1]["object_ids"],
            ["object_0003", "object_0004"],
        )

    def test_incomplete_cycle_prevents_final_registry(self):
        failed = _registry(85)
        failed["status"] = "failed_seed_quality"
        with self.assertRaisesRegex(ValueError, "not ready"):
            combine_cycle_registries([failed], video="1.mp4")

    def test_empty_object_set_is_rejected(self):
        malformed = _registry(85)
        del malformed["objects"]["lid_0"]
        del malformed["objects"]["cup_0"]
        with self.assertRaisesRegex(ValueError, "at least one"):
            combine_cycle_registries([malformed], video="1.mp4")

    def test_arbitrary_object_count_and_relation_kind_are_supported(self):
        registry = {
            "status": "ready",
            "event_type": "generic_visual_interaction",
            "objects": {
                "moving_piece": {"frame_idx": 10, "mask": "moving.png"},
                "fixed_piece": {"frame_idx": 10, "mask": "fixed.png"},
                "tool": {"frame_idx": 10, "mask": "tool.png"},
            },
            "relations": [
                {
                    "kind": "separates_from",
                    "moving_object_id": "moving_piece",
                    "target_object_id": "fixed_piece",
                    "context_object_ids": ["tool"],
                    "evidence_frame_idx": 12,
                }
            ],
        }
        combined = combine_cycle_registries([registry], video="generic.mp4")
        self.assertEqual(
            list(combined["objects"]),
            ["object_0001", "object_0002", "object_0003"],
        )
        relation = combined["relations"][0]
        self.assertEqual(relation["kind"], "separates_from")
        self.assertEqual(relation["moving_object_id"], "object_0001")
        self.assertEqual(relation["target_object_id"], "object_0002")
        self.assertEqual(relation["context_object_ids"], ["object_0003"])
        self.assertEqual(combined["cycles"][0]["event_type"], "generic_visual_interaction")

    def test_reuses_explicit_visual_identity_across_cycles(self):
        combined = combine_cycle_registries(
            [_registry(20), _registry(80)],
            video="generic.mp4",
            cycle_object_id_maps=[
                {"lid_0": "object_0001", "cup_0": "object_0002"},
                {"lid_0": "object_0001", "cup_0": "object_0002"},
            ],
        )

        self.assertEqual(set(combined["objects"]), {"object_0001", "object_0002"})
        self.assertEqual(len(combined["objects"]["object_0001"]["cycle_observations"]), 2)
        self.assertEqual(
            combined["cycles"][1]["source_object_id_map"],
            {"lid_0": "object_0001", "cup_0": "object_0002"},
        )
        self.assertEqual(combined["relations"][1]["part_object_id"], "object_0001")


if __name__ == "__main__":
    unittest.main()
