import tempfile
import unittest
from pathlib import Path

import numpy as np

from ocir.grasp_robustness.analyze_rotation_gauntlet import analyze
from ocir.grasp_robustness.rotation_gauntlet import GauntletConfig, build_rotation_gauntlet
from ocir.grasp_traj.trajectory_schema import SEGMENT_CARRY, GraspTrajectory


def _vertical_lift_source(prefix_steps=4, carry_steps=30, dt=1.0 / 30.0) -> GraspTrajectory:
    t = prefix_steps + carry_steps
    hand_pos = np.tile([0.4, 0.2, 0.3], (t, 1))
    # Vertical lift straight along camera -y (arbitrary; the builder recovers it).
    up = np.array([0.0, -1.0, 0.0])
    for i in range(carry_steps):
        hand_pos[prefix_steps + i] = hand_pos[prefix_steps - 1] + up * 0.15 * (i + 1) / carry_steps
    segment = np.concatenate([
        np.arange(prefix_steps).clip(0, 3).astype(np.int8),
        np.full(carry_steps, SEGMENT_CARRY, dtype=np.int8),
    ])
    return GraspTrajectory(
        hand_pos_camera=hand_pos,
        hand_quat_camera=np.tile([1.0, 0.0, 0.0, 0.0], (t, 1)),
        finger_targets=np.tile(np.linspace(0.0, 2.1, 22), (t, 1)),
        object_pos_camera=np.tile([0.35, 0.25, 0.28], (t, 1)),
        object_quat_camera=np.tile([1.0, 0.0, 0.0, 0.0], (t, 1)),
        segment=segment,
        dt=dt,
        joint_order=tuple(f"joint_{i}" for i in range(22)),
        grasp_json="grasp.json",
        sequence_dir="/tmp/sequence",
        switch_frame_index=0,
        grasp_root_tf=np.eye(4).tolist(),
        extra_metadata={"grasp_frame_index": 1},
    )


class GauntletBuildTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _vertical_lift_source()
        self.config = GauntletConfig(
            lift_height_m=0.1, lift_seconds=0.5, angles_deg=(30.0,),
            peak_angular_speed_deg_s=60.0, hold_seconds=0.1, axes=("yaw", "pitch"),
        )
        self.trajectory, self.meta = build_rotation_gauntlet(self.source, self.config)
        self.carry_start = self.meta["carry_start_step"]

    def test_prefix_and_fingers(self) -> None:
        cs = self.carry_start
        np.testing.assert_array_equal(self.trajectory.hand_pos_camera[:cs], self.source.hand_pos_camera[:cs])
        carry = self.trajectory.segment == SEGMENT_CARRY
        np.testing.assert_array_equal(
            self.trajectory.finger_targets[carry],
            np.tile(self.source.finger_targets[cs - 1], (int(carry.sum()), 1)),
        )

    def test_object_reference_is_pivot_invariant_during_rotation(self) -> None:
        lifted_obj = None
        for phase in self.meta["phases"]:
            if phase["name"] == "post_lift_hold":
                lifted_obj = self.trajectory.object_pos_camera[phase["start"]]
            if phase.get("kind") == "rotate":
                seg = self.trajectory.object_pos_camera[phase["start"]:phase["end"]]
                np.testing.assert_allclose(seg, np.tile(lifted_obj, (seg.shape[0], 1)), atol=1e-9)

    def test_rotation_legs_return_to_base_and_respect_speed_cap(self) -> None:
        base_quat = None
        for phase in self.meta["phases"]:
            if phase["name"] == "post_lift_hold":
                base_quat = self.trajectory.hand_quat_camera[phase["start"]]
        last_return = [p for p in self.meta["phases"] if p["name"].endswith("_return")][-1]
        end_quat = self.trajectory.hand_quat_camera[last_return["end"] - 1]
        self.assertGreater(abs(float(np.dot(base_quat, end_quat))), 1.0 - 1e-9)

        dt = float(self.trajectory.dt)
        cap = np.radians(self.config.peak_angular_speed_deg_s) * dt * 1.05
        quats = self.trajectory.hand_quat_camera[self.carry_start:]
        dots = np.abs(np.sum(quats[1:] * quats[:-1], axis=1)).clip(-1.0, 1.0)
        self.assertLessEqual(float((2.0 * np.arccos(dots)).max()), cap)

    def test_lift_height_along_recovered_up(self) -> None:
        lift = [p for p in self.meta["phases"] if p["name"] == "lift"][0]
        start = self.trajectory.hand_pos_camera[self.carry_start - 1]
        end = self.trajectory.hand_pos_camera[lift["end"] - 1]
        np.testing.assert_allclose(end - start, np.array([0.0, -0.1, 0.0]), atol=1e-9)

    def test_rejects_non_lift_source(self) -> None:
        bad = _vertical_lift_source()
        flat = bad.hand_pos_camera.copy()
        flat[bad.segment == SEGMENT_CARRY] = flat[3]
        bad = GraspTrajectory(**{**bad.__dict__, "hand_pos_camera": flat})
        with self.assertRaises(ValueError):
            build_rotation_gauntlet(bad, self.config)


class GauntletAnalyzeTest(unittest.TestCase):
    def _write_sim(self, root: Path, trajectory: GraspTrajectory, meta: dict, *, break_phase: str | None) -> Path:
        trajectory.save(root)
        sim = root / "isaac_sim"
        sim.mkdir()
        actual_pos = trajectory.object_pos_camera.copy()
        actual_quat = trajectory.object_quat_camera.copy()
        if break_phase is not None:
            phase = [p for p in meta["phases"] if p["name"] == break_phase][0]
            actual_pos[phase["start"]:] += np.array([0.0, 0.0, -0.08])
        np.savez_compressed(
            sim / "object_track.npz",
            position_world=actual_pos,
            orientation_world_wxyz=actual_quat,
            reference_position_world=trajectory.object_pos_camera,
            reference_orientation_world_wxyz=trajectory.object_quat_camera,
            segment=trajectory.segment,
            carry_mask=trajectory.segment == SEGMENT_CARRY,
        )
        return sim

    def test_pass_and_fail_classification(self) -> None:
        source = _vertical_lift_source()
        config = GauntletConfig(lift_height_m=0.1, lift_seconds=0.5, angles_deg=(30.0,),
                                peak_angular_speed_deg_s=60.0, hold_seconds=0.1, axes=("yaw",))
        trajectory, meta = build_rotation_gauntlet(source, config)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "ok"
            root.mkdir(parents=True)
            sim = self._write_sim(root, trajectory, meta, break_phase=None)
            report = analyze(root, sim, pos_threshold_m=0.03, rot_threshold_deg=20.0)
            self.assertTrue(report["passed"])
            self.assertIsNone(report["first_failing_phase"])

            broken = Path(temp) / "broken"
            broken.mkdir(parents=True)
            sim = self._write_sim(broken, trajectory, meta, break_phase="yaw-30")
            report = analyze(broken, sim, pos_threshold_m=0.03, rot_threshold_deg=20.0)
            self.assertFalse(report["passed"])
            self.assertEqual(report["first_failing_phase"], "yaw-30")
            self.assertTrue((sim / "rotation_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
