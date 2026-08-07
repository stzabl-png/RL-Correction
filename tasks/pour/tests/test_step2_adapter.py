from __future__ import annotations

import unittest

import numpy as np

from tasks.pour.reference import HAND_JOINT_SUFFIXES, PourReference
from tasks.pour.stage_reconstruction import _matrix_to_pose
from tasks.pour.step2_adapter import estimate_contact_region, fit_step2_world_to_arkit


class Step2AdapterTest(unittest.TestCase):
    @staticmethod
    def _reference(camera_matrix: np.ndarray) -> PourReference:
        length = len(camera_matrix)
        joints = np.zeros((length, len(HAND_JOINT_SUFFIXES), 3), dtype=np.float32)
        for suffix in ("ThumbTip", "IndexFingerTip", "MiddleFingerTip", "RingFingerTip", "LittleFingerTip"):
            joints[:, HAND_JOINT_SUFFIXES.index(suffix)] = [0.01, 0.0, 0.0]
        return PourReference(
            demo_id="11",
            fps=20.0,
            camera_intrinsic=np.eye(3),
            camera_pose=_matrix_to_pose(camera_matrix),
            left_wrist=np.tile([0, 0, 0, 1, 0, 0, 0], (length, 1)),
            right_wrist=np.tile([0, 0, 0, 1, 0, 0, 0], (length, 1)),
            left_joints=joints,
            right_joints=joints.copy(),
            video_phase=np.full(length, 3),
            right_tilt_rad=np.zeros(length),
            hand_confidence=np.ones((length, 2)),
            source_frame=np.arange(length),
            bottle_pose=np.full((length, 7), np.nan),
            cup_pose=np.full((length, 7), np.nan),
            object_confidence=np.zeros((length, 2)),
            video_contact_left=np.empty((0, 3)),
            video_contact_right=np.empty((0, 3)),
            video_contact_confidence=np.zeros(2),
        )

    def test_camera_alignment_recovers_one_world_transform(self):
        length = 8
        step2 = np.repeat(np.eye(4)[None], length, axis=0)
        step2[:, 0, 3] = np.linspace(0.0, 0.35, length)
        angle = np.deg2rad(25.0)
        world = np.eye(4)
        world[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        world[:3, 3] = [1.0, -0.2, 0.4]
        camera_axis = np.diag([1.0, -1.0, -1.0, 1.0])
        arkit = world @ step2 @ camera_axis
        reference = self._reference(arkit)

        fitted, report = fit_step2_world_to_arkit(step2, reference)
        np.testing.assert_allclose(fitted, world, atol=1.0e-6)
        self.assertLess(report["translation_p95_m"], 1.0e-6)
        self.assertLess(report["rotation_p95_deg"], 1.0e-5)

    def test_contact_projection_uses_object_local_mesh(self):
        matrix = np.repeat(np.eye(4)[None], 4, axis=0)
        reference = self._reference(matrix)
        pose = _matrix_to_pose(matrix)
        mesh = np.array(
            [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
        )
        points, confidence, diagnostics = estimate_contact_region(
            reference,
            side="left",
            object_pose=pose,
            mesh_vertices=mesh,
            distance_threshold_m=0.005,
        )
        np.testing.assert_allclose(points[0], [0.01, 0.0, 0.0])
        self.assertGreater(confidence, 0.99)
        self.assertEqual(diagnostics["raw_contact_observations"], 20)


if __name__ == "__main__":
    unittest.main()
