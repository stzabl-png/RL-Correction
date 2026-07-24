import unittest

from experiments.hoi_detr.interaction_episodes import detect_interaction_episodes


def _frames(frame_count, positive_frames):
    return [
        {
            "frame_idx": frame_idx,
            "status": "accepted",
            "selection_source": (
                "hf_link" if frame_idx in positive_frames else "unlinked_firstobject"
            ),
            "hand_link_probability": 0.9 if frame_idx in positive_frames else None,
        }
        for frame_idx in range(frame_count)
    ]


class InteractionEpisodeTests(unittest.TestCase):
    def test_returns_one_start_and_end_keyframe_per_stable_run(self):
        result = detect_interaction_episodes(
            _frames(30, set(range(2, 9)) | set(range(12, 18))), fps=30.0
        )
        self.assertEqual([(item["start_frame"], item["end_frame"]) for item in result], [(5, 11), (15, 20)])
        self.assertNotIn("evidence_frames", result[0])
        self.assertFalse(result[0]["end_truncated"])

    def test_allows_one_missing_link_without_splitting_episode(self):
        result = detect_interaction_episodes(
            _frames(20, {2, 3, 5, 6, 7, 8}), fps=30.0
        )
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0]["raw_start_frame"], result[0]["raw_end_frame"]), (2, 8))

    def test_ignores_unconfirmed_flicker(self):
        result = detect_interaction_episodes(_frames(20, {1, 8, 9}), fps=30.0)
        self.assertEqual(result, [])

    def test_marks_final_episode_truncated_at_video_end(self):
        result = detect_interaction_episodes(
            _frames(20, set(range(15, 20))), fps=30.0
        )
        self.assertEqual(result[0]["end_frame"], 19)
        self.assertTrue(result[0]["end_truncated"])
        self.assertEqual(result[0]["end_source"], "video_end_truncated")


if __name__ == "__main__":
    unittest.main()
