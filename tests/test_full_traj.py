import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ocir.full_traj.generate_full_traj import build_full_trajectory
from ocir.full_traj.metrics import ordered_pose_path_metrics, paired_pose_path_metrics
from ocir.full_traj.reference import (
    FullTrajectoryReference,
    resample_synchronized_poses,
    retime_synchronized_poses,
)
from ocir.full_traj.se3 import interpolate_transform, rotation_angle, rotation_from_vector
from ocir.full_traj.simulate_full_traj import _add_open_loop_report, _validate_open_loop_trajectory
from ocir.grasp_traj.trajectory_schema import SEGMENT_CARRY, GraspTrajectory, pos_quat_to_matrix


def _pose(position=(0.0, 0.0, 0.0), rotation_vector=(0.0, 0.0, 0.0)) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rotation_from_vector(np.asarray(rotation_vector, dtype=float))
    out[:3, 3] = np.asarray(position, dtype=float)
    return out


class ReferenceTest(unittest.TestCase):
    def test_resampling_only_fills_missing_video_frames(self) -> None:
        objects = np.stack([_pose(), _pose((0.02, 0.0, 0.0), (0.0, 0.0, 0.2))])
        wrists = np.stack([_pose((0.0, 0.1, 0.0)), _pose((0.02, 0.1, 0.0), (0.0, 0.0, 0.2))])

        out_obj, out_wrist, frames = resample_synchronized_poses(
            objects, wrists, np.array([10.0, 12.0])
        )

        np.testing.assert_array_equal(frames, [10.0, 11.0, 12.0])
        np.testing.assert_allclose(out_obj[1], interpolate_transform(objects[0], objects[1], 0.5))
        np.testing.assert_allclose(out_wrist[1], interpolate_transform(wrists[0], wrists[1], 0.5))

    def test_consecutive_frames_are_not_geometry_densified(self) -> None:
        objects = np.stack([_pose(), _pose((0.20, 0.0, 0.0), (0.0, 0.0, 1.0))])
        wrists = objects.copy()

        out_obj, out_wrist, frames = resample_synchronized_poses(
            objects, wrists, np.array([3.0, 4.0])
        )

        np.testing.assert_array_equal(out_obj, objects)
        np.testing.assert_array_equal(out_wrist, wrists)
        np.testing.assert_array_equal(frames, [3.0, 4.0])


class RetimingTest(unittest.TestCase):
    def _path(self, n=7, step=0.02):
        objects = np.stack([_pose((i * step, 0.0, 0.0), (0.0, 0.0, 0.05 * i)) for i in range(n)])
        wrists = np.stack([_pose((i * step, 0.1, 0.0), (0.0, 0.0, 0.05 * i)) for i in range(n)])
        frames = np.arange(10.0, 10.0 + n)
        return objects, wrists, frames

    def test_identity_when_unscaled(self) -> None:
        objects, wrists, frames = self._path()
        out_obj, out_wrist, out_frames = retime_synchronized_poses(
            objects, wrists, frames, dt=1.0 / 30.0, time_scale=1.0, ease_in_seconds=0.0, ease_out_seconds=0.0
        )
        np.testing.assert_array_equal(out_obj, objects)
        np.testing.assert_array_equal(out_wrist, wrists)
        np.testing.assert_array_equal(out_frames, frames)

    def test_scaling_doubles_duration_and_preserves_endpoints(self) -> None:
        objects, wrists, frames = self._path()
        out_obj, out_wrist, out_frames = retime_synchronized_poses(
            objects, wrists, frames, dt=1.0 / 30.0, time_scale=2.0, ease_in_seconds=0.0, ease_out_seconds=0.0
        )
        self.assertEqual(out_obj.shape[0], 2 * (objects.shape[0] - 1) + 1)
        np.testing.assert_allclose(out_obj[0], objects[0], atol=1e-12)
        np.testing.assert_allclose(out_obj[-1], objects[-1], atol=1e-12)
        np.testing.assert_allclose(out_wrist[-1], wrists[-1], atol=1e-12)
        self.assertAlmostEqual(out_frames[0], frames[0])
        self.assertAlmostEqual(out_frames[-1], frames[-1])
        # Uniform warp: every second output row lands on a source row.
        np.testing.assert_allclose(out_obj[::2], objects, atol=1e-9)
        # Both paths share one clock.
        self.assertTrue(np.all(np.diff(out_frames) > 0.0))

    def test_ease_in_starts_from_rest(self) -> None:
        objects, wrists, frames = self._path(n=31, step=0.01)
        dt = 1.0 / 30.0
        out_obj, _, out_frames = retime_synchronized_poses(
            objects, wrists, frames, dt=dt, time_scale=2.0, ease_in_seconds=0.3, ease_out_seconds=0.3
        )
        steps = np.linalg.norm(np.diff(out_obj[:, :3, 3], axis=0), axis=1)
        # First and last steps are far below the plateau speed; the geometric
        # endpoints are untouched.
        self.assertLess(steps[0], 0.2 * steps.max())
        self.assertLess(steps[-1], 0.2 * steps.max())
        np.testing.assert_allclose(out_obj[0], objects[0], atol=1e-12)
        np.testing.assert_allclose(out_obj[-1], objects[-1], atol=1e-12)
        self.assertTrue(np.all(np.diff(out_frames) >= 0.0))


class MetricsTest(unittest.TestCase):
    def test_paired_metrics_use_open_loop_timestamps(self) -> None:
        reference = np.stack([_pose(), _pose((0.01, 0.0, 0.0))])
        actual = np.stack([_pose(), _pose((0.02, 0.0, 0.0))])

        metrics = paired_pose_path_metrics(actual, reference)

        self.assertAlmostEqual(metrics["translation_rmse_m"], np.sqrt(0.0001 / 2.0))
        self.assertAlmostEqual(metrics["translation_final_m"], 0.01)
        self.assertAlmostEqual(metrics["orientation_rmse_rad"], 0.0)

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


class GenerationTest(unittest.TestCase):
    def _build_fixture(self, root: Path) -> tuple[Path, GraspTrajectory]:
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
        source_object_pos = np.tile([0.3, -0.1, 0.2], (t, 1))
        source = GraspTrajectory(
            hand_pos_camera=np.tile([0.4, 0.2, 0.3], (t, 1)),
            hand_quat_camera=np.tile([1.0, 0.0, 0.0, 0.0], (t, 1)),
            finger_targets=finger,
            object_pos_camera=source_object_pos,
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
        return source_dir, source

    def test_open_loop_retarget_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir, source = self._build_fixture(root)
            # Raw video clock, wrist replay: retiming is covered by
            # RetimingTest, the object-anchored mode by its own test below.
            trajectory, reference = build_full_trajectory(
                source_dir, time_scale=1.0, ease_in_seconds=0.0, ease_out_seconds=0.0,
                retarget_mode="wrist",
            )
            carry_start = reference.carry_start_step

            for name in (
                "hand_pos_camera",
                "hand_quat_camera",
                "finger_targets",
                "object_pos_camera",
                "object_quat_camera",
                "segment",
            ):
                np.testing.assert_array_equal(
                    getattr(trajectory, name)[:carry_start], getattr(source, name)[:carry_start]
                )

            carry_targets = trajectory.finger_targets[trajectory.segment == SEGMENT_CARRY]
            np.testing.assert_array_equal(
                carry_targets,
                np.tile(source.finger_targets[carry_start - 1], (carry_targets.shape[0], 1)),
            )
            np.testing.assert_allclose(
                trajectory.hand_pos_camera[carry_start], source.hand_pos_camera[carry_start - 1], atol=1e-10
            )
            np.testing.assert_allclose(
                trajectory.hand_quat_camera[carry_start], source.hand_quat_camera[carry_start - 1], atol=1e-10
            )
            np.testing.assert_allclose(
                trajectory.object_pos_camera[carry_start], source.object_pos_camera[carry_start - 1], atol=1e-10
            )
            np.testing.assert_allclose(
                trajectory.object_quat_camera[carry_start], source.object_quat_camera[carry_start - 1], atol=1e-10
            )

            commanded_wrist = pos_quat_to_matrix(
                trajectory.hand_pos_camera[carry_start:], trajectory.hand_quat_camera[carry_start:]
            )
            mano_wrist = reference.mano_wrist_poses_camera
            np.testing.assert_allclose(
                np.linalg.inv(commanded_wrist[0]) @ commanded_wrist,
                np.linalg.inv(mano_wrist[0]) @ mano_wrist,
                atol=1e-9,
            )
            commanded_object = pos_quat_to_matrix(
                trajectory.object_pos_camera[carry_start:], trajectory.object_quat_camera[carry_start:]
            )
            demo_object = reference.object_poses_camera
            np.testing.assert_allclose(
                np.linalg.inv(commanded_object[0]) @ commanded_object,
                np.linalg.inv(demo_object[0]) @ demo_object,
                atol=1e-9,
            )
            self.assertEqual(reference.num_path_points, 3)
            self.assertAlmostEqual(reference.metadata["source_duration_seconds"], 2.0 / 30.0)
            self.assertAlmostEqual(reference.metadata["generated_duration_seconds"], 2.0 / 30.0)
            self.assertEqual(trajectory.extra_metadata["full_traj"]["control_mode"], "open_loop")

            out_dir = root / "full"
            trajectory.save(out_dir)
            reference.save(out_dir)
            loaded_trajectory, loaded_reference = _validate_open_loop_trajectory(out_dir)
            self.assertEqual(loaded_trajectory.num_steps, trajectory.num_steps)
            np.testing.assert_array_equal(loaded_reference.fixed_finger_targets, reference.fixed_finger_targets)

    def test_retimed_generation_keeps_contracts_and_scales_duration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir, source = self._build_fixture(root)
            trajectory, reference = build_full_trajectory(source_dir)  # default 3x + eases
            carry_start = reference.carry_start_step

            # Continuity at carry entry survives retiming.
            np.testing.assert_allclose(
                trajectory.hand_pos_camera[carry_start], source.hand_pos_camera[carry_start - 1], atol=1e-10
            )
            np.testing.assert_allclose(
                trajectory.object_pos_camera[carry_start], source.object_pos_camera[carry_start - 1], atol=1e-10
            )
            # Duration tripled: 2 source intervals -> 6 output intervals.
            n_carry = int((trajectory.segment == SEGMENT_CARRY).sum())
            self.assertEqual(n_carry, 7)
            self.assertAlmostEqual(reference.metadata["generated_duration_seconds"], 6.0 / 30.0)
            self.assertAlmostEqual(reference.metadata["source_duration_seconds"], 2.0 / 30.0)
            self.assertEqual(reference.metadata["time_scale"], 3.0)
            # Default object-anchored mode: the wrist keeps a constant rigid
            # relation to the commanded object path (no in-hand drift).
            commanded_wrist = pos_quat_to_matrix(
                trajectory.hand_pos_camera[carry_start:], trajectory.hand_quat_camera[carry_start:]
            )
            commanded_object = pos_quat_to_matrix(
                trajectory.object_pos_camera[carry_start:], trajectory.object_quat_camera[carry_start:]
            )
            relations = np.linalg.inv(commanded_object) @ commanded_wrist
            np.testing.assert_allclose(relations, np.tile(relations[0], (relations.shape[0], 1, 1)), atol=1e-9)
            self.assertEqual(trajectory.extra_metadata["full_traj"]["retarget_mode"], "object")
            # Ease-in: the first carry step moves far less than a uniform step.
            steps = np.linalg.norm(np.diff(trajectory.hand_pos_camera[carry_start:], axis=0), axis=1)
            self.assertLess(steps[0], 0.5 * steps.max())
            out_dir = root / "full"
            trajectory.save(out_dir)
            reference.save(out_dir)
            _validate_open_loop_trajectory(out_dir)

    def test_reference_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir, _ = self._build_fixture(root)
            _, reference = build_full_trajectory(source_dir)
            reference.save(root / "reference")
            loaded = FullTrajectoryReference.load(root / "reference")
            np.testing.assert_array_equal(loaded.fixed_finger_targets, reference.fixed_finger_targets)
            np.testing.assert_array_equal(loaded.source_frame_index, reference.source_frame_index)

    def test_open_loop_report_uses_object_track_only_for_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source_dir, _ = self._build_fixture(root)
            trajectory, reference = build_full_trajectory(source_dir)
            out_dir = root / "sim"
            out_dir.mkdir()

            np.savez_compressed(
                out_dir / "object_track.npz",
                position_world=trajectory.object_pos_camera,
                orientation_world_wxyz=trajectory.object_quat_camera,
                reference_position_world=trajectory.object_pos_camera,
                reference_orientation_world_wxyz=trajectory.object_quat_camera,
                segment=trajectory.segment,
            )
            np.savez_compressed(
                out_dir / "finger_track.npz",
                desired_position_rad=trajectory.finger_targets,
            )
            args = SimpleNamespace(
                out_dir=out_dir,
                path_position_tolerance=0.008,
                path_orientation_tolerance_deg=5.0,
            )

            report = _add_open_loop_report({}, args, trajectory, reference)

            self.assertEqual(report["task"], "full_traj_simulation")
            metrics = report["full_traj_open_loop"]
            self.assertFalse(metrics["object_pose_used_for_control"])
            self.assertTrue(metrics["finger_targets_constant"])
            self.assertAlmostEqual(metrics["time_aligned_path_metrics"]["translation_rmse_m"], 0.0)
            self.assertAlmostEqual(metrics["ordered_path_metrics"]["orientation_rmse_rad"], 0.0)
            self.assertTrue((out_dir / "report.json").is_file())


class RotationTest(unittest.TestCase):
    def test_rotation_angle(self) -> None:
        self.assertAlmostEqual(rotation_angle(rotation_from_vector([0.0, 0.0, 0.4])), 0.4)


if __name__ == "__main__":
    unittest.main()
