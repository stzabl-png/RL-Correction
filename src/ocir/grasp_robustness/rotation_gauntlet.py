"""Build a wrist-rotation stress-test trajectory from a vertical-lift carry.

The source ``grasp_traj`` result proves that the grasp survives a plain
vertical lift.  The gauntlet keeps everything up to and including that lift's
starting pose, lifts the object with the same proven easing, then rotates the
wrist about the held object's center through +/- legs on several axes.  The
object's reference position never moves during rotation, so any recorded
deviation is grasp slip, attributable to a specific rotation leg.

All math is camera-frame NumPy; the standard ``grasp_traj_simulation`` task
replays the result unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ocir.grasp_traj.trajectory_schema import (
    SEGMENT_CARRY,
    GraspTrajectory,
    matrix_to_pos_quat,
    pos_quat_to_matrix,
)


@dataclass(frozen=True)
class GauntletConfig:
    lift_height_m: float = 0.15
    lift_seconds: float = 3.0
    angles_deg: tuple[float, ...] = (45.0,)
    peak_angular_speed_deg_s: float = 45.0
    hold_seconds: float = 0.4
    axes: tuple[str, ...] = ("yaw", "pitch", "roll")

    def validate(self) -> None:
        if self.lift_height_m <= 0.0 or self.lift_seconds <= 0.0:
            raise ValueError("lift height and duration must be positive")
        if not self.angles_deg or any(a <= 0.0 or a > 180.0 for a in self.angles_deg):
            raise ValueError("angles_deg must be in (0, 180]")
        if self.peak_angular_speed_deg_s <= 0.0:
            raise ValueError("peak_angular_speed_deg_s must be positive")
        if self.hold_seconds < 0.0:
            raise ValueError("hold_seconds must be nonnegative")
        bad = [a for a in self.axes if a not in ("yaw", "pitch", "roll")]
        if bad or not self.axes:
            raise ValueError(f"axes must be a nonempty subset of yaw/pitch/roll, got {self.axes}")


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    s, c = math.sin(angle), math.cos(angle)
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + s * skew + (1.0 - c) * (skew @ skew)


def _rotate_about_pivot(pose: np.ndarray, axis: np.ndarray, angle: float, pivot: np.ndarray) -> np.ndarray:
    rot = _axis_angle_matrix(axis, angle)
    out = pose.copy()
    out[:3, :3] = rot @ pose[:3, :3]
    out[:3, 3] = pivot + rot @ (pose[:3, 3] - pivot)
    return out


def _ease(alpha: float) -> float:
    return 0.5 * (1.0 - math.cos(math.pi * float(np.clip(alpha, 0.0, 1.0))))


def _world_basis_from_lift(lift_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal (up, horizontal1, horizontal2) from the camera-frame lift.

    The source carry translates the wrist straight up in the Isaac world, so
    its camera-frame displacement IS world-up expressed in the camera frame.
    The absolute heading of the horizontal axes is arbitrary, which is fine
    for a robustness sweep.
    """

    up = np.asarray(lift_vector, dtype=np.float64)
    norm = float(np.linalg.norm(up))
    if norm < 0.03:
        raise ValueError(
            "source carry displaces the wrist by less than 3cm; the gauntlet requires a "
            "vertical-lift grasp_traj result (not a retargeted full_traj one) to recover world-up"
        )
    up = up / norm
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, up))) > 0.9:
        seed = np.array([0.0, 1.0, 0.0])
    h1 = seed - np.dot(seed, up) * up
    h1 /= float(np.linalg.norm(h1))
    h2 = np.cross(up, h1)
    return up, h1, h2


@dataclass
class _Timeline:
    dt: float
    wrist: list[np.ndarray] = field(default_factory=list)
    obj: list[np.ndarray] = field(default_factory=list)
    phases: list[dict] = field(default_factory=list)

    def add_phase(self, name: str, wrist_poses: list[np.ndarray], obj_poses: list[np.ndarray], **info) -> None:
        start = len(self.wrist)
        self.wrist.extend(wrist_poses)
        self.obj.extend(obj_poses)
        self.phases.append({"name": name, "start": start, "end": len(self.wrist), **info})


def build_rotation_gauntlet(
    source: GraspTrajectory, config: GauntletConfig | None = None
) -> tuple[GraspTrajectory, dict]:
    config = config or GauntletConfig()
    config.validate()
    carry_rows = np.flatnonzero(source.segment == SEGMENT_CARRY)
    if carry_rows.size == 0:
        raise ValueError("source trajectory has no carry segment")
    carry_start = int(carry_rows[0])
    if carry_start == 0:
        raise ValueError("carry cannot be the first frame; a completed grasp prefix is required")
    if np.any(source.segment[carry_start:] != SEGMENT_CARRY):
        raise ValueError("carry segment must be the final contiguous trajectory segment")
    dt = float(source.dt)

    wrist0 = pos_quat_to_matrix(source.hand_pos_camera[carry_start - 1], source.hand_quat_camera[carry_start - 1])
    obj0 = pos_quat_to_matrix(source.object_pos_camera[carry_start - 1], source.object_quat_camera[carry_start - 1])
    lift_vector = source.hand_pos_camera[len(source.segment) - 1] - source.hand_pos_camera[carry_start]
    up, h1, h2 = _world_basis_from_lift(lift_vector)
    axis_vectors = {"yaw": up, "pitch": h1, "roll": h2}

    timeline = _Timeline(dt=dt)

    def hold(name: str, wrist: np.ndarray, obj: np.ndarray) -> None:
        n = max(int(round(config.hold_seconds / dt)), 1)
        timeline.add_phase(name, [wrist.copy()] * n, [obj.copy()] * n, kind="hold")

    # Lift with the proven easing.
    n_lift = max(int(round(config.lift_seconds / dt)), 1)
    lift_w, lift_o = [], []
    for k in range(1, n_lift + 1):
        s = _ease(k / n_lift) * config.lift_height_m
        w = wrist0.copy(); w[:3, 3] = wrist0[:3, 3] + up * s
        o = obj0.copy(); o[:3, 3] = obj0[:3, 3] + up * s
        lift_w.append(w); lift_o.append(o)
    timeline.add_phase("lift", lift_w, lift_o, kind="lift", height_m=float(config.lift_height_m))
    wrist_base, obj_base = lift_w[-1], lift_o[-1]
    pivot = obj_base[:3, 3].copy()
    hold("post_lift_hold", wrist_base, obj_base)

    peak = math.radians(config.peak_angular_speed_deg_s)
    for axis_name in config.axes:
        axis = axis_vectors[axis_name]
        for angle_deg in config.angles_deg:
            amp = math.radians(float(angle_deg))
            for leg_name, a0, a1 in (
                (f"{axis_name}+{angle_deg:g}", 0.0, amp),
                (f"{axis_name}+{angle_deg:g}_return", amp, 0.0),
                (f"{axis_name}-{angle_deg:g}", 0.0, -amp),
                (f"{axis_name}-{angle_deg:g}_return", -amp, 0.0),
            ):
                # Cosine-eased leg whose PEAK angular speed equals the cap.
                duration = abs(a1 - a0) * (math.pi / 2.0) / peak
                n = max(int(math.ceil(duration / dt)), 1)
                leg_w, leg_o = [], []
                for k in range(1, n + 1):
                    theta = a0 + (a1 - a0) * _ease(k / n)
                    leg_w.append(_rotate_about_pivot(wrist_base, axis, theta, pivot))
                    leg_o.append(_rotate_about_pivot(obj_base, axis, theta, pivot))
                timeline.add_phase(
                    leg_name, leg_w, leg_o,
                    kind="rotate", axis=axis_name, angle_deg=float(angle_deg),
                    axis_camera=[float(v) for v in axis],
                )
                hold(f"{leg_name}_hold", leg_w[-1], leg_o[-1])
    hold("final_hold", wrist_base, obj_base)

    n_carry = len(timeline.wrist)
    wrist_pos, wrist_quat = matrix_to_pos_quat(np.stack(timeline.wrist))
    obj_pos, obj_quat = matrix_to_pos_quat(np.stack(timeline.obj))
    fixed_fingers = np.asarray(source.finger_targets[carry_start - 1], dtype=np.float64)
    prefix = slice(0, carry_start)
    phases = [
        {**p, "start": p["start"] + carry_start, "end": p["end"] + carry_start} for p in timeline.phases
    ]
    gauntlet_meta = {
        "carry_start_step": carry_start,
        "lift_height_m": float(config.lift_height_m),
        "lift_seconds": float(config.lift_seconds),
        "angles_deg": [float(a) for a in config.angles_deg],
        "peak_angular_speed_deg_s": float(config.peak_angular_speed_deg_s),
        "hold_seconds": float(config.hold_seconds),
        "axes": list(config.axes),
        "up_camera": [float(v) for v in up],
        "pivot_rule": "object center after lift; the object reference never translates during rotation",
        "num_gauntlet_frames": int(n_carry),
        "duration_seconds": float(n_carry * dt),
        "phases": phases,
    }
    new_extra = dict(source.extra_metadata)
    new_extra["rotation_gauntlet"] = gauntlet_meta
    trajectory = GraspTrajectory(
        hand_pos_camera=np.concatenate([source.hand_pos_camera[prefix], wrist_pos], axis=0),
        hand_quat_camera=np.concatenate([source.hand_quat_camera[prefix], wrist_quat], axis=0),
        finger_targets=np.concatenate(
            [source.finger_targets[prefix], np.tile(fixed_fingers[None], (n_carry, 1))], axis=0
        ),
        object_pos_camera=np.concatenate([source.object_pos_camera[prefix], obj_pos], axis=0),
        object_quat_camera=np.concatenate([source.object_quat_camera[prefix], obj_quat], axis=0),
        segment=np.concatenate(
            [source.segment[prefix], np.full(n_carry, SEGMENT_CARRY, dtype=np.int8)], axis=0
        ),
        dt=source.dt,
        joint_order=source.joint_order,
        grasp_json=source.grasp_json,
        sequence_dir=source.sequence_dir,
        switch_frame_index=source.switch_frame_index,
        grasp_root_tf=source.grasp_root_tf,
        clearance_report=source.clearance_report,
        extra_metadata=new_extra,
    )
    return trajectory, gauntlet_meta
