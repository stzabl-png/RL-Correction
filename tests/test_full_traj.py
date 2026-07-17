import tempfile
import unittest
from pathlib import Path

import numpy as np

from ocir.full_traj.controller import ControllerConfig, PathFollowingController
from ocir.full_traj.generate_full_traj import build_full_trajectory
from ocir.full_traj.metrics import ordered_pose_path_metrics
from ocir.full_traj.reference import FullTrajectoryReference, resample_synchronized_poses
from ocir.full_traj.se3 import pose_error, rotate_pose_about_point, rotation_from_vector, rotation_vector
from ocir.grasp_traj.trajectory_schema import SEGMENT_CARRY, GraspTrajectory


def _pose(position=(0.0, 0.0, 0.0), rotation_vector=(0.0, 0.0, 0.0)) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation_from_vector(np.asarray(rotation_vector, dtype=float))
    out[:3, 3] = np.asarray(position, dtype=float)
    return out


class SE3Test(unittest.TestCase):
    def test_pose_error_is_decoupled_in_world_frame(self) -> None:
        actual = _pose((1.0, 2.0, 3.0), (0.0, 0.0, 0.2))
        target = _pose((1.1, 1.8, 3.3), (0.0, 0.0, 0.5))

        translation, rotation = pose_error(target, actual)

        np.testing.assert_allclose(translation, [0.1, -0.2, 0.3], atol=1e-9)
        np.testing.assert_allclose(rotation, [0.0, 0.0, 0.3], atol=1e-9)

    def test_rotate_pose_about_point_keeps_pivot_fixed(self) -> None:
        pose = _pose((0.1, 0.0, 0.2), (0.0, 0.0, 0.3))
        pivot = np.array([0.1, 0.0, 0.1])
        rotation = np.array([0.0, 0.0, np.pi / 2.0])

        out = rotate_pose_about_point(pose, rotation, pivot)

        # A point at the pivot must not move; the pose position swings around it.
        np.testing.assert_allclose(out[:3, 3], [0.1, 0.0, 0.2], atol=1e-9)
        np.testing.assert_allclose(
            rotation_vector(out[:3, :3] @ pose[:3, :3].T), rotation, atol=1e-9
        )
        offset_pose = _pose((0.2, 0.0, 0.1))
        swung = rotate_pose_about_point(offset_pose, rotation, pivot)
        np.testing.assert_allclose(swung[:3, 3], [0.1, 0.1, 0.1], atol=1e-9)

    def test_resampling_limits_translation_and_rotation_steps(self) -> None:
        objects = np.stack([_pose(), _pose((0.021, 0.0, 0.0), (0.0, 0.0, np.deg2rad(11.0)))])
        wrists = objects.copy()

        out_obj, out_wrist, frames = resample_synchronized_poses(
            objects,
            wrists,
            np.array([10.0, 11.0]),
            max_translation_step_m=0.005,
            max_rotation_step_rad=np.deg2rad(3.0),
        )

        self.assertEqual(out_obj.shape, out_wrist.shape)
        self.assertEqual(out_obj.shape[0], 6)
        self.assertAlmostEqual(frames[0], 10.0)
        self.assertAlmostEqual(frames[-1], 11.0)
        self.assertLessEqual(np.linalg.norm(np.diff(out_obj[:, :3, 3], axis=0), axis=1).max(), 0.005 + 1e-12)

    def test_ordered_metrics_ignore_timing_but_preserve_path_order(self) -> None:
        reference = np.stack([_pose((0.00, 0.0, 0.0)), _pose((0.01, 0.0, 0.0)), _pose((0.02, 0.0, 0.0))])
        actual = np.stack(
            [_pose((0.00, 0.0, 0.0)), _pose((0.00, 0.0, 0.0)), _pose((0.01, 0.0, 0.0)), _pose((0.02, 0.0, 0.0))]
        )

        metrics = ordered_pose_path_metrics(
            actual,
            reference,
            position_tolerance_m=0.001,
            orientation_tolerance_rad=np.deg2rad(1.0),
        )

        self.assertAlmostEqual(metrics["translation_rmse_m"], 0.0)
        self.assertAlmostEqual(metrics["orientation_rmse_rad"], 0.0)
        self.assertAlmostEqual(metrics["joint_pose_path_coverage_fraction"], 1.0)


class ControllerTest(unittest.TestCase):
    def _controller(self, **config_overrides) -> tuple[PathFollowingController, np.ndarray]:
        objects = np.stack([_pose((0.00, 0.0, 0.0)), _pose((0.01, 0.0, 0.0)), _pose((0.02, 0.0, 0.0))])
        grasp = _pose((0.0, 0.0, 0.1))
        wrists = objects @ grasp
        defaults = dict(
            catchup_seconds=0.1,
            waypoint_timeout_seconds=0.1,
            total_timeout_scale=20.0,
            max_linear_speed_mps=10.0,
            max_angular_speed_radps=10.0,
            max_linear_accel_mps2=1000.0,
            max_angular_accel_radps2=1000.0,
        )
        defaults.update(config_overrides)
        controller = PathFollowingController(objects, wrists, source_fps=10.0, config=ControllerConfig(**defaults))
        return controller, grasp

    def test_projection_progress_follows_the_object_along_the_path(self) -> None:
        controller, grasp = self._controller()
        start_object = _pose((0.0, 0.0, 0.0))
        controller.start(start_object, start_object @ grasp)

        first = controller.step(start_object, start_object @ grasp, dt=0.01)
        self.assertEqual(first.progress_index, 0)
        # The carrot leads the projection so the hand keeps moving.
        self.assertGreater(first.path_index, 0)
        self.assertFalse(first.done)

        moved_object = _pose((0.01, 0.0, 0.0))
        second = controller.step(moved_object, moved_object @ grasp, dt=0.01)
        self.assertEqual(second.progress_index, 1)

    def test_projection_jumps_multiple_points_in_one_step(self) -> None:
        controller, grasp = self._controller()
        start_object = _pose((0.0, 0.0, 0.0))
        controller.start(start_object, start_object @ grasp)

        end_object = _pose((0.02, 0.0, 0.0))
        result = controller.step(end_object, end_object @ grasp, dt=0.01)

        self.assertEqual(result.progress_index, 2)
        self.assertTrue(controller.path_completed)
        self.assertTrue(result.done)
        self.assertTrue(controller.final_pose_within_tolerance)

    def test_hold_alignment_tracks_the_relative_path(self) -> None:
        controller, grasp = self._controller()
        offset = np.array([0.5, -0.2, 0.3])
        start_object = _pose(tuple(offset))
        controller.start(start_object, start_object @ grasp)

        end_object = _pose(tuple(offset + [0.02, 0.0, 0.0]))
        result = controller.step(end_object, end_object @ grasp, dt=0.01)

        # Progress and completion are judged against the aligned path, not
        # the absolute reference half a metre away.
        self.assertEqual(result.progress_index, 2)
        self.assertTrue(result.done)
        np.testing.assert_allclose(
            controller.aligned_object_reference[0][:3, 3], offset, atol=1e-9
        )

    def test_stalled_object_escapes_via_timeout_not_projection(self) -> None:
        controller, grasp = self._controller(
            waypoint_timeout_seconds=0.02, hold_alignment=False, catchup_seconds=0.0
        )
        actual_object = _pose((0.2, 0.0, 0.0))
        actual_hand = actual_object @ grasp
        controller.start(actual_object, actual_hand)

        controller.step(actual_object, actual_hand, dt=0.01)
        controller.step(actual_object, actual_hand, dt=0.01)

        # 0.2m off-path is outside the projection acceptance radius, so the
        # only progress is the stall-escape creep.
        self.assertEqual(controller.path_index, 1)
        self.assertEqual(controller.timed_out_indices, [1])

    def test_lost_grasp_is_reported_but_controller_continues(self) -> None:
        controller, grasp = self._controller(lost_grasp_confirm_steps=2)
        actual_object = _pose()
        actual_hand = actual_object @ grasp
        controller.start(actual_object, actual_hand)
        slipped_hand = actual_hand.copy()
        slipped_hand[0, 3] += 0.2

        controller.step(actual_object, slipped_hand, dt=0.01)
        result = controller.step(actual_object, slipped_hand, dt=0.01)

        self.assertTrue(result.lost_grasp)
        self.assertFalse(result.done)
        self.assertEqual(len(controller.history), 2)


class GenerationTest(unittest.TestCase):
    def test_preserves_prefix_and_holds_final_grasp_fingers(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sequence_dir = root / "sequence"
            source_dir = root / "source"
            sequence_dir.mkdir()
            source_dir.mkdir()

            demo_t = 4
            object_pose = np.tile(np.eye(4, dtype=np.float32), (demo_t, 1, 1))
            object_pose[:, 0, 3] = np.arange(demo_t) * 0.01
            joints = np.zeros((demo_t, 21, 3), dtype=np.float32)
            joints[:, 5] = [0.03, 0.02, 0.0]
            joints[:, 9] = [0.04, 0.00, 0.0]
            joints[:, 13] = [0.03, -0.02, 0.0]
            np.savez_compressed(
                sequence_dir / "human_demo.npz",
                frame_ids=np.arange(demo_t, dtype=np.int32),
                mano_side="right",
                betas=np.zeros(10, dtype=np.float32),
                pose_m_camera=np.zeros((demo_t, 51), dtype=np.float32),
                object_pose_camera=object_pose,
                hand_joints_object=joints,
                hand_vertices_object=np.zeros((demo_t, 778, 3), dtype=np.float32),
                vertex_part_ids=np.zeros(778, dtype=np.int8),
                valid_mask=np.ones(demo_t, dtype=bool),
            )

            t = 5
            finger = np.arange(t * 22, dtype=np.float64).reshape(t, 22) / 100.0
            source = GraspTrajectory(
                hand_pos_camera=np.zeros((t, 3)),
                hand_quat_camera=np.tile([1.0, 0.0, 0.0, 0.0], (t, 1)),
                finger_targets=finger,
                object_pos_camera=np.zeros((t, 3)),
                object_quat_camera=np.tile([1.0, 0.0, 0.0, 0.0], (t, 1)),
                segment=np.array([0, 1, 2, 3, SEGMENT_CARRY], dtype=np.int8),
                dt=1.0 / 30.0,
                joint_order=tuple(f"joint_{i}" for i in range(22)),
                grasp_json="grasp.json",
                sequence_dir=str(sequence_dir),
                switch_frame_index=0,
                grasp_root_tf=np.eye(4).tolist(),
                extra_metadata={"grasp_frame_index": 1},
            )
            source.save(source_dir)

            trajectory, reference = build_full_trajectory(source_dir)
            carry_start = reference.carry_start_step

            for name in (
                "hand_pos_camera",
                "hand_quat_camera",
                "finger_targets",
                "object_pos_camera",
                "object_quat_camera",
                "segment",
            ):
                np.testing.assert_array_equal(getattr(trajectory, name)[:carry_start], getattr(source, name)[:carry_start])
            carry_targets = trajectory.finger_targets[trajectory.segment == SEGMENT_CARRY]
            np.testing.assert_array_equal(
                carry_targets,
                np.tile(source.finger_targets[carry_start - 1], (carry_targets.shape[0], 1)),
            )
            self.assertEqual(int(trajectory.segment[carry_start - 1]), 3)
            self.assertEqual(trajectory.extra_metadata["full_traj"]["legacy_segment_3_semantics"], "stationary_grasp_hold_only")

            reference.save(root / "reference")
            loaded = FullTrajectoryReference.load(root / "reference")
            np.testing.assert_array_equal(loaded.fixed_finger_targets, reference.fixed_finger_targets)


if __name__ == "__main__":
    unittest.main()
