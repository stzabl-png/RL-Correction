import unittest

from experiments.hoi_detr.hoi_box_events import (
    HOIBoxExpansionConfig,
    detect_confirmed_box_expansions,
)


def _frame(index, box=None):
    detections = []
    links = []
    if box is not None:
        detections = [
            {"detection_id": f"h{index}", "class_name": "hand", "box_xyxy": [0, 0, 10, 10]},
            {"detection_id": f"o{index}", "class_name": "firstobject", "box_xyxy": box},
        ]
        links = [{"source_detection_id": f"h{index}", "target_detection_id": f"o{index}"}]
    return {"frame_idx": index, "detections": detections, "links": {"hf": links}}


def _detections(boxes):
    return {"video": {"height": 100, "width": 100}, "frames": [_frame(i, box) for i, box in enumerate(boxes)]}


class HOIBoxExpansionEventTests(unittest.TestCase):
    def test_confirms_a_sustained_containing_expansion(self):
        result = detect_confirmed_box_expansions(
            _detections([[40, 40, 50, 50], [40, 40, 50, 50], [35, 35, 55, 55], [35, 35, 55, 55], [35, 35, 55, 55]]),
            interaction_start_frame=0,
            interaction_end_frame=4,
            initial_box=(40.0, 40.0, 50.0, 50.0),
            config=HOIBoxExpansionConfig(min_consecutive_frames=3),
        )

        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(result["events"][0]["activation_frame"], 2)
        self.assertEqual(result["events"][0]["seed_frame"], 4)

    def test_rejects_a_one_frame_box_spike(self):
        result = detect_confirmed_box_expansions(
            _detections([[40, 40, 50, 50], [30, 30, 60, 60], [40, 40, 50, 50], [40, 40, 50, 50]]),
            interaction_start_frame=0,
            interaction_end_frame=3,
            initial_box=(40.0, 40.0, 50.0, 50.0),
            config=HOIBoxExpansionConfig(min_consecutive_frames=2),
        )

        self.assertEqual(result["events"], [])

    def test_rejects_a_scene_sized_box_even_when_it_contains_the_old_box(self):
        result = detect_confirmed_box_expansions(
            _detections([[40, 40, 50, 50], [0, 0, 100, 100], [0, 0, 100, 100]]),
            interaction_start_frame=0,
            interaction_end_frame=2,
            initial_box=(40.0, 40.0, 50.0, 50.0),
            config=HOIBoxExpansionConfig(min_consecutive_frames=2),
        )

        self.assertEqual(result["events"], [])
        self.assertIn(
            "rejected_background_sized_expansion",
            [item["status"] for item in result["audit"]],
        )


if __name__ == "__main__":
    unittest.main()
