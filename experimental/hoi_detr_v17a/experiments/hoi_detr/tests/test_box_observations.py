"""Tests for temporal first-object box selection and background-spike gating."""

from __future__ import annotations

import unittest

from experiments.hoi_detr.box_observations import BoxGateConfig, build_box_observations


def _detection(frame_idx, detection_index, class_name, box, score=0.9):
    class_id = {"hand": 0, "firstobject": 1, "secondobject": 2}[class_name]
    return {
        "detection_id": f"f{frame_idx:06d}_d{detection_index:04d}",
        "detection_index": detection_index,
        "box_xyxy": list(box),
        "score": score,
        "class_id": class_id,
        "class_name": class_name,
    }


def _frame(frame_idx, object_boxes, *, linked_index=0):
    detections = [_detection(frame_idx, 0, "hand", [40, 45, 60, 75])]
    detections.extend(
        _detection(frame_idx, index + 1, "firstobject", box, score=0.8 + index * 0.05)
        for index, box in enumerate(object_boxes)
    )
    links = []
    if object_boxes and linked_index is not None:
        links.append(
            {
                "kind": "hf",
                "source_detection_id": detections[0]["detection_id"],
                "target_detection_id": detections[linked_index + 1]["detection_id"],
                "prob": 0.9,
            }
        )
    return {
        "frame_idx": frame_idx,
        "processed": True,
        "detections": detections,
        "links": {"hf": links, "fs": []},
    }


def _frame_with_all_linked(frame_idx, object_boxes):
    frame = _frame(frame_idx, object_boxes, linked_index=None)
    hand_id = frame["detections"][0]["detection_id"]
    frame["links"]["hf"] = [
        {
            "kind": "hf",
            "source_detection_id": hand_id,
            "target_detection_id": detection["detection_id"],
            "prob": 0.9 - 0.05 * index,
        }
        for index, detection in enumerate(frame["detections"][1:])
    ]
    return frame


def _document(frames):
    return {
        "dataset": "test",
        "video_id": "clip",
        "video": {"width": 100, "height": 100, "num_frames": len(frames)},
        "frames": frames,
    }


class BoxObservationTests(unittest.TestCase):
    def test_prefers_hand_linked_object_over_higher_score_unlinked_object(self):
        frame = _frame(0, [[10, 10, 20, 20], [70, 70, 90, 90]], linked_index=0)
        observations = build_box_observations(_document([frame]))
        selected = observations["frames"][0]
        self.assertEqual(selected["selection_source"], "hf_link")
        self.assertEqual(selected["source_detection_id"], "f000000_d0001")
        self.assertEqual(selected["hand_link_probability"], 0.9)

    def test_rejects_screen_spanning_bootstrap_box_without_poisoning_history(self):
        frames = [
            _frame(0, [[1, 40, 99, 90]]),
            _frame(1, [[40, 40, 55, 60]]),
        ]
        observations = build_box_observations(_document(frames))
        self.assertEqual(observations["frames"][0]["status"], "rejected_background_spike")
        self.assertEqual(observations["frames"][1]["status"], "accepted")
        self.assertEqual(observations["frames"][1]["accepted_history_size"], 0)

    def test_rejects_area_spike_against_stable_history_and_keeps_following_box(self):
        frames = [
            _frame(0, [[40, 40, 55, 60]]),
            _frame(1, [[41, 40, 56, 60]]),
            _frame(2, [[42, 40, 57, 60]]),
            _frame(3, [[5, 15, 95, 90]]),
            _frame(4, [[43, 40, 58, 60]]),
        ]
        observations = build_box_observations(_document(frames))
        self.assertEqual(observations["frames"][3]["status"], "rejected_background_spike")
        self.assertIn("area_spike_vs_history", observations["frames"][3]["reason"])
        self.assertEqual(observations["frames"][4]["status"], "accepted")
        self.assertEqual(
            observations["summary"]["rejected_background_spike_frames"],
            [3],
        )

    def test_retains_every_valid_hand_linked_box_in_one_frame(self):
        frame = _frame_with_all_linked(
            0,
            [[40, 40, 55, 55], [35, 35, 65, 75]],
        )
        observations = build_box_observations(_document([frame]))
        candidates = observations["frames"][0]["hand_linked_object_candidates"]
        self.assertEqual(len(candidates), 2)
        self.assertEqual([item["status"] for item in candidates], ["accepted", "accepted"])
        self.assertEqual(
            {item["source_detection_id"] for item in candidates},
            {"f000000_d0001", "f000000_d0002"},
        )

    def test_rejects_bad_linked_box_without_hiding_valid_linked_box(self):
        frame = _frame_with_all_linked(
            0,
            [[40, 40, 55, 55], [1, 35, 99, 95]],
        )
        observations = build_box_observations(_document([frame]))
        candidates = observations["frames"][0]["hand_linked_object_candidates"]
        status_by_id = {
            item["source_detection_id"]: item["status"] for item in candidates
        }
        self.assertEqual(status_by_id["f000000_d0001"], "accepted")
        self.assertEqual(
            status_by_id["f000000_d0002"], "rejected_background_spike"
        )
        self.assertEqual(observations["frames"][0]["status"], "accepted")

    def test_missing_frames_are_explicit_and_do_not_reset_history(self):
        frames = [
            _frame(0, [[40, 40, 55, 60]]),
            _frame(1, []),
            _frame(2, [[41, 40, 56, 60]]),
        ]
        observations = build_box_observations(_document(frames))
        self.assertEqual([frame["status"] for frame in observations["frames"]], ["accepted", "missing", "accepted"])
        self.assertEqual(observations["frames"][2]["accepted_history_size"], 1)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            build_box_observations(
                _document([_frame(0, [[40, 40, 55, 60]])]),
                config=BoxGateConfig(history_size=2, min_history=3),
            )


if __name__ == "__main__":
    unittest.main()
