import unittest

import numpy as np

from experiments.hoi_detr.visual_instance_memory import (
    decide_memory_identity,
    describe_masked_instance,
)


def _image_and_mask(color, x=2):
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:] = (30, 30, 30)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:13, x : x + 8] = True
    image[mask] = color
    return image, mask


class VisualInstanceMemoryTests(unittest.TestCase):
    def test_reuses_a_visually_matching_instance_after_translation(self):
        first_image, first_mask = _image_and_mask((0, 220, 240), x=2)
        second_image, second_mask = _image_and_mask((0, 220, 240), x=9)
        first = describe_masked_instance("object_0001", first_image, first_mask)
        current = describe_masked_instance("local", second_image, second_mask)

        decision = decide_memory_identity({"object_0001": [first]}, current)

        self.assertEqual(decision["status"], "reuse_visual_memory")
        self.assertEqual(decision["object_id"], "object_0001")

    def test_creates_new_id_for_visibly_distinct_instance(self):
        yellow_image, yellow_mask = _image_and_mask((0, 220, 240))
        white_image, white_mask = _image_and_mask((230, 230, 230))
        yellow = describe_masked_instance("object_0001", yellow_image, yellow_mask)
        white = describe_masked_instance("local", white_image, white_mask)

        decision = decide_memory_identity({"object_0001": [yellow]}, white)

        self.assertEqual(decision["status"], "new_visual_instance")

    def test_rejects_similar_ambiguous_memory_matches(self):
        image, mask = _image_and_mask((0, 220, 240))
        first = describe_masked_instance("object_0001", image, mask)
        second = describe_masked_instance("object_0002", image, mask)
        current = describe_masked_instance("local", image, mask)

        decision = decide_memory_identity(
            {"object_0001": [first], "object_0002": [second]}, current
        )

        self.assertEqual(decision["status"], "failed_ambiguous_visual_memory")


if __name__ == "__main__":
    unittest.main()
