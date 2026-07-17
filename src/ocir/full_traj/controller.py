"""Adaptive SE(3) path following for wrist-only object transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocir.full_traj.se3 import (
    apply_world_correction,
    clamp_norm,
    interpolate_transform,
    pose_error,
    rotate_pose_about_point,
    rotation_angle,
    rotation_vector,
    smoothstep,
)


@dataclass(frozen=True)
class ControllerConfig:
    control_hz: float = 60.0
    hold_alignment: bool = True
    catchup_seconds: float = 1.0
    position_tolerance_m: float = 0.008
    orientation_tolerance_rad: float = np.deg2rad(5.0)
    lookahead_translation_m: float = 0.02
    lookahead_rotation_rad: float = np.deg2rad(10.0)
    projection_window_points: int = 40
    projection_rotation_scale_m_per_rad: float = 0.05
    projection_acceptance_translation_m: float = 0.05
    waypoint_timeout_seconds: float = 1.0
    total_timeout_scale: float = 3.0
    kp_translation: float = 0.6
    ki_translation: float = 0.15
    kp_rotation: float = 0.6
    ki_rotation: float = 0.15
    max_integral_translation_m_s: float = 0.20
    max_integral_rotation_rad_s: float = np.deg2rad(45.0)
    max_correction_translation_m: float = 0.05
    max_correction_rotation_rad: float = np.deg2rad(20.0)
    max_linear_speed_mps: float = 0.25
    max_angular_speed_radps: float = np.deg2rad(90.0)
    max_linear_accel_mps2: float = 1.0
    max_angular_accel_radps2: float = np.deg2rad(360.0)
    lost_grasp_translation_m: float = 0.05
    lost_grasp_rotation_rad: float = np.deg2rad(30.0)
    lost_grasp_confirm_steps: int = 5

    def validate(self) -> None:
        positive = {
            "control_hz": self.control_hz,
            "position_tolerance_m": self.position_tolerance_m,
            "orientation_tolerance_rad": self.orientation_tolerance_rad,
            "lookahead_translation_m": self.lookahead_translation_m,
            "lookahead_rotation_rad": self.lookahead_rotation_rad,
            "projection_rotation_scale_m_per_rad": self.projection_rotation_scale_m_per_rad,
            "projection_acceptance_translation_m": self.projection_acceptance_translation_m,
            "waypoint_timeout_seconds": self.waypoint_timeout_seconds,
            "total_timeout_scale": self.total_timeout_scale,
            "max_linear_speed_mps": self.max_linear_speed_mps,
            "max_angular_speed_radps": self.max_angular_speed_radps,
            "max_linear_accel_mps2": self.max_linear_accel_mps2,
            "max_angular_accel_radps2": self.max_angular_accel_radps2,
            "max_integral_translation_m_s": self.max_integral_translation_m_s,
            "max_integral_rotation_rad_s": self.max_integral_rotation_rad_s,
            "max_correction_translation_m": self.max_correction_translation_m,
            "max_correction_rotation_rad": self.max_correction_rotation_rad,
            "lost_grasp_translation_m": self.lost_grasp_translation_m,
            "lost_grasp_rotation_rad": self.lost_grasp_rotation_rad,
        }
        bad = [name for name, value in positive.items() if float(value) <= 0.0]
        if bad:
            raise ValueError(f"controller values must be positive: {bad}")
        nonnegative = {
            "catchup_seconds": self.catchup_seconds,
            "kp_translation": self.kp_translation,
            "ki_translation": self.ki_translation,
            "kp_rotation": self.kp_rotation,
            "ki_rotation": self.ki_rotation,
        }
        bad = [name for name, value in nonnegative.items() if float(value) < 0.0]
        if bad:
            raise ValueError(f"controller values must be nonnegative: {bad}")
        if int(self.projection_window_points) <= 0:
            raise ValueError("projection_window_points must be positive")
        if int(self.lost_grasp_confirm_steps) <= 0:
            raise ValueError("lost_grasp_confirm_steps must be positive")


@dataclass(frozen=True)
class ControllerStep:
    hand_command: np.ndarray
    nominal_hand: np.ndarray
    object_target: np.ndarray
    path_index: int
    progress_index: int
    translation_error_m: float
    orientation_error_rad: float
    cross_track_translation_m: float
    cross_track_rotation_rad: float
    correction_translation_m: float
    correction_rotation_rad: float
    correction_translation_saturated: bool
    correction_rotation_saturated: bool
    velocity_saturated: bool
    acceleration_saturated: bool
    lost_grasp: bool
    done: bool


class PathFollowingController:
    """Monotonic pure-pursuit object-path follower.

    Progress along the reference is measured by projecting the actual object
    pose onto the path (monotonic, bounded by an acceptance radius), and the
    wrist is steered toward a lookahead carrot ahead of that projection.  The
    controller replays MANO's changing wrist/object relation on top of the
    actual synthesized grasp relation measured at carry entry.  P feedback
    acts on the carrot error, I feedback on the cross-track error, and the
    rotation correction pivots about the object's own position so it does not
    inject translation error through the hand-object lever arm.  Finger
    control is intentionally outside this class.
    """

    def __init__(
        self,
        object_reference_world: np.ndarray,
        mano_wrist_reference_world: np.ndarray,
        *,
        source_fps: float,
        source_duration_seconds: float | None = None,
        config: ControllerConfig | None = None,
    ) -> None:
        self.object_reference = np.asarray(object_reference_world, dtype=np.float64)
        self.mano_wrist_reference = np.asarray(mano_wrist_reference_world, dtype=np.float64)
        if self.object_reference.shape != self.mano_wrist_reference.shape:
            raise ValueError("object and MANO wrist references must have identical shape")
        if self.object_reference.ndim != 3 or self.object_reference.shape[1:] != (4, 4):
            raise ValueError("reference paths must have shape (N,4,4)")
        if self.object_reference.shape[0] < 2:
            raise ValueError("path follower needs at least two reference poses")
        self.source_fps = float(source_fps)
        self.config = config or ControllerConfig()
        self.config.validate()
        inferred_duration = (self.object_reference.shape[0] - 1) / self.source_fps
        self.source_duration_seconds = max(
            float(source_duration_seconds) if source_duration_seconds is not None else inferred_duration,
            1.0 / self.source_fps,
        )
        self.max_duration_seconds = self.config.total_timeout_scale * self.source_duration_seconds
        self.raw_hand_in_object = np.linalg.inv(self.object_reference) @ self.mano_wrist_reference
        # Per-segment path arc lengths used by the lookahead walk; alignment
        # is rigid, so raw reference spacing is valid for the aligned path.
        deltas = np.linalg.inv(self.object_reference[:-1]) @ self.object_reference[1:]
        self.segment_translation = np.linalg.norm(
            self.object_reference[1:, :3, 3] - self.object_reference[:-1, :3, 3], axis=1
        )
        self.segment_rotation = np.asarray([rotation_angle(d[:3, :3]) for d in deltas], dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        self.started = False
        self.path_index = 0
        self.elapsed_seconds = 0.0
        self.waypoint_elapsed_seconds = 0.0
        self.integral_translation = np.zeros(3, dtype=np.float64)
        self.integral_rotation = np.zeros(3, dtype=np.float64)
        self.last_command: np.ndarray | None = None
        self.last_linear_velocity = np.zeros(3, dtype=np.float64)
        self.last_angular_velocity = np.zeros(3, dtype=np.float64)
        self.initial_object_offset = np.eye(4, dtype=np.float64)
        self.actual_grasp_relation = np.eye(4, dtype=np.float64)
        self.reference_grasp_relation = self.raw_hand_in_object[0].copy()
        self.timed_out_indices: list[int] = []
        self.lost_grasp = False
        self.first_lost_step: int | None = None
        self.recovered_after_loss = False
        self._lost_count = 0
        self._step_count = 0
        self.done = False
        self.path_completed = False
        self.final_pose_within_tolerance = False
        self.history: list[dict] = []

    def start(self, actual_object_world: np.ndarray, actual_hand_world: np.ndarray) -> None:
        self.reset()
        actual_object_world = np.asarray(actual_object_world, dtype=np.float64)
        actual_hand_world = np.asarray(actual_hand_world, dtype=np.float64)
        self.initial_object_offset = actual_object_world @ np.linalg.inv(self.object_reference[0])
        self.actual_grasp_relation = np.linalg.inv(actual_object_world) @ actual_hand_world
        self.last_command = actual_hand_world.copy()
        self.started = True

    def _alignment(self) -> np.ndarray:
        if self.config.hold_alignment:
            return self.initial_object_offset
        if self.config.catchup_seconds <= 0.0:
            return np.eye(4, dtype=np.float64)
        alpha = smoothstep(self.elapsed_seconds / self.config.catchup_seconds)
        return interpolate_transform(self.initial_object_offset, np.eye(4), alpha)

    @property
    def aligned_object_reference(self) -> np.ndarray:
        """Reference path in the frame the controller actually tracks."""

        return self._alignment()[None] @ self.object_reference

    def _advance_projection(self, actual_object_world: np.ndarray, alignment: np.ndarray) -> None:
        cfg = self.config
        end = min(self.path_index + int(cfg.projection_window_points), self.object_reference.shape[0] - 1)
        best_index = None
        best_distance = np.inf
        for index in range(self.path_index, end + 1):
            aligned = alignment @ self.object_reference[index]
            translation = float(np.linalg.norm(aligned[:3, 3] - actual_object_world[:3, 3]))
            if translation > cfg.projection_acceptance_translation_m:
                continue
            angle = rotation_angle(aligned[:3, :3] @ actual_object_world[:3, :3].T)
            distance = translation + cfg.projection_rotation_scale_m_per_rad * angle
            # Prefer the later index on ties so duplicated (stationary demo)
            # path points are consumed immediately instead of one per step.
            if distance <= best_distance + 1e-9:
                best_distance = distance
                best_index = index
        if best_index is not None and best_index > self.path_index:
            self.path_index = best_index
            self.waypoint_elapsed_seconds = 0.0
        elif (
            self.waypoint_elapsed_seconds >= cfg.waypoint_timeout_seconds
            and self.path_index < self.object_reference.shape[0] - 1
        ):
            # Stall escape: the object is not making progress (off the path
            # beyond the acceptance radius or pinned); creep forward so a
            # single unreachable region cannot consume the entire run.
            self.path_index += 1
            self.timed_out_indices.append(self.path_index)
            self.waypoint_elapsed_seconds = 0.0

    def _carrot_index(self, progress: int) -> int:
        cfg = self.config
        last = self.object_reference.shape[0] - 1
        if progress >= last:
            return last
        index = progress
        accumulated_translation = 0.0
        accumulated_rotation = 0.0
        while index < last:
            accumulated_translation += float(self.segment_translation[index])
            accumulated_rotation += float(self.segment_rotation[index])
            index += 1
            if (
                accumulated_translation >= cfg.lookahead_translation_m
                or accumulated_rotation >= cfg.lookahead_rotation_rad
            ):
                break
        return index

    def _target_poses(self, index: int, alignment: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        object_target = alignment @ self.object_reference[index]
        relative_change = np.linalg.inv(self.reference_grasp_relation) @ self.raw_hand_in_object[index]
        expected_grasp_relation = self.actual_grasp_relation @ relative_change
        nominal_hand = object_target @ expected_grasp_relation
        return object_target, nominal_hand, expected_grasp_relation

    def _slew_limit(self, desired: np.ndarray, dt: float) -> tuple[np.ndarray, bool, bool]:
        assert self.last_command is not None
        translation_delta = desired[:3, 3] - self.last_command[:3, 3]
        rotation_delta = desired[:3, :3] @ self.last_command[:3, :3].T
        linear_velocity = translation_delta / dt
        angular_velocity = rotation_vector(rotation_delta) / dt
        linear_velocity, linear_sat = clamp_norm(linear_velocity, self.config.max_linear_speed_mps)
        angular_velocity, angular_sat = clamp_norm(angular_velocity, self.config.max_angular_speed_radps)

        dv, linear_accel_sat = clamp_norm(
            linear_velocity - self.last_linear_velocity,
            self.config.max_linear_accel_mps2 * dt,
        )
        dw, angular_accel_sat = clamp_norm(
            angular_velocity - self.last_angular_velocity,
            self.config.max_angular_accel_radps2 * dt,
        )
        linear_velocity = self.last_linear_velocity + dv
        angular_velocity = self.last_angular_velocity + dw
        self.last_linear_velocity = linear_velocity
        self.last_angular_velocity = angular_velocity

        out = apply_world_correction(self.last_command, linear_velocity * dt, angular_velocity * dt)
        return out, bool(linear_sat or angular_sat), bool(linear_accel_sat or angular_accel_sat)

    def step(
        self,
        actual_object_world: np.ndarray,
        actual_hand_world: np.ndarray,
        *,
        dt: float | None = None,
    ) -> ControllerStep:
        if not self.started:
            raise RuntimeError("call start() at the first carry update")
        if self.done:
            raise RuntimeError("controller has already finished")
        dt = float(dt if dt is not None else 1.0 / self.config.control_hz)
        self.elapsed_seconds += dt
        self.waypoint_elapsed_seconds += dt
        self._step_count += 1
        actual_object_world = np.asarray(actual_object_world, dtype=np.float64)
        actual_hand_world = np.asarray(actual_hand_world, dtype=np.float64)

        alignment = self._alignment()
        self._advance_projection(actual_object_world, alignment)
        progress = self.path_index
        carrot = self._carrot_index(progress)
        object_target, nominal_hand, expected_relation = self._target_poses(carrot, alignment)
        trans_error, rot_error = pose_error(object_target, actual_object_world)
        trans_norm = float(np.linalg.norm(trans_error))
        rot_norm = float(np.linalg.norm(rot_error))

        # Cross-track error against the projected point: the deviation from
        # the path itself, excluding the standing lookahead offset, so the
        # integrator removes persistent bias without winding up along-path.
        projected_target = alignment @ self.object_reference[progress]
        cross_trans_error, cross_rot_error = pose_error(projected_target, actual_object_world)
        cross_trans_norm = float(np.linalg.norm(cross_trans_error))
        cross_rot_norm = float(np.linalg.norm(cross_rot_error))

        candidate_integral_t, _ = clamp_norm(
            self.integral_translation + cross_trans_error * dt,
            self.config.max_integral_translation_m_s,
        )
        candidate_integral_r, _ = clamp_norm(
            self.integral_rotation + cross_rot_error * dt,
            self.config.max_integral_rotation_rad_s,
        )
        correction_t_raw = self.config.kp_translation * trans_error + self.config.ki_translation * candidate_integral_t
        correction_r_raw = self.config.kp_rotation * rot_error + self.config.ki_rotation * candidate_integral_r
        correction_t, correction_t_sat = clamp_norm(correction_t_raw, self.config.max_correction_translation_m)
        correction_r, correction_r_sat = clamp_norm(correction_r_raw, self.config.max_correction_rotation_rad)
        # Conditional integration: retain a candidate only on axes whose
        # combined correction did not hit its total correction bound.
        if not correction_t_sat:
            self.integral_translation = candidate_integral_t
        if not correction_r_sat:
            self.integral_rotation = candidate_integral_r

        # The rotation correction pivots about the object's actual position:
        # rotating the wrist about its own origin would swing the object
        # through the hand-object lever arm and corrupt the translation loop.
        desired_hand = rotate_pose_about_point(nominal_hand, correction_r, actual_object_world[:3, 3])
        desired_hand[:3, 3] += correction_t
        hand_command, velocity_sat, acceleration_sat = self._slew_limit(desired_hand, dt)
        self.last_command = hand_command

        last_index = self.object_reference.shape[0] - 1
        if progress >= last_index:
            self.path_completed = True
            within_final_tolerance = (
                cross_trans_norm <= self.config.position_tolerance_m
                and cross_rot_norm <= self.config.orientation_tolerance_rad
            )
            if within_final_tolerance:
                self.final_pose_within_tolerance = True
            if within_final_tolerance or self.waypoint_elapsed_seconds >= self.config.waypoint_timeout_seconds:
                self.done = True
        if self.elapsed_seconds >= self.max_duration_seconds:
            self.done = True

        actual_relation = np.linalg.inv(actual_object_world) @ actual_hand_world
        slip_delta = np.linalg.inv(expected_relation) @ actual_relation
        slip_translation = float(np.linalg.norm(slip_delta[:3, 3]))
        slip_rotation = rotation_angle(slip_delta[:3, :3])
        excessive_slip = (
            slip_translation > self.config.lost_grasp_translation_m
            or slip_rotation > self.config.lost_grasp_rotation_rad
        )
        self._lost_count = self._lost_count + 1 if excessive_slip else 0
        if not self.lost_grasp and self._lost_count >= int(self.config.lost_grasp_confirm_steps):
            self.lost_grasp = True
            self.first_lost_step = self._step_count
        elif self.lost_grasp and not excessive_slip:
            self.recovered_after_loss = True

        item = {
            "path_index": int(carrot),
            "progress_index": int(progress),
            "translation_error_m": trans_norm,
            "orientation_error_rad": rot_norm,
            "cross_track_translation_m": cross_trans_norm,
            "cross_track_rotation_rad": cross_rot_norm,
            "correction_translation_m": float(np.linalg.norm(correction_t)),
            "correction_rotation_rad": float(np.linalg.norm(correction_r)),
            "correction_translation_saturated": bool(correction_t_sat),
            "correction_rotation_saturated": bool(correction_r_sat),
            "velocity_saturated": velocity_sat,
            "acceleration_saturated": acceleration_sat,
            "slip_translation_m": slip_translation,
            "slip_rotation_rad": slip_rotation,
        }
        self.history.append(item)
        return ControllerStep(
            hand_command=hand_command.copy(),
            nominal_hand=nominal_hand.copy(),
            object_target=object_target.copy(),
            path_index=carrot,
            progress_index=progress,
            translation_error_m=trans_norm,
            orientation_error_rad=rot_norm,
            cross_track_translation_m=cross_trans_norm,
            cross_track_rotation_rad=cross_rot_norm,
            correction_translation_m=item["correction_translation_m"],
            correction_rotation_rad=item["correction_rotation_rad"],
            correction_translation_saturated=bool(correction_t_sat),
            correction_rotation_saturated=bool(correction_r_sat),
            velocity_saturated=velocity_sat,
            acceleration_saturated=acceleration_sat,
            lost_grasp=self.lost_grasp,
            done=self.done,
        )

    def report(self) -> dict:
        def values(key: str) -> list[float]:
            return [float(item[key]) for item in self.history]

        trans = np.asarray(values("translation_error_m"), dtype=np.float64)
        rot = np.asarray(values("orientation_error_rad"), dtype=np.float64)
        cross_trans = np.asarray(values("cross_track_translation_m"), dtype=np.float64)
        cross_rot = np.asarray(values("cross_track_rotation_rad"), dtype=np.float64)
        return {
            "path_completed": bool(self.path_completed),
            "final_pose_within_tolerance": bool(self.final_pose_within_tolerance),
            "path_fraction": float(self.path_index / max(self.object_reference.shape[0] - 1, 1)),
            "final_path_index": int(self.path_index),
            "num_path_points": int(self.object_reference.shape[0]),
            "elapsed_seconds": float(self.elapsed_seconds),
            "source_duration_seconds": float(self.source_duration_seconds),
            "duration_ratio": float(self.elapsed_seconds / self.source_duration_seconds),
            "hold_alignment": bool(self.config.hold_alignment),
            "alignment_translation_m": float(np.linalg.norm(self.initial_object_offset[:3, 3])),
            "alignment_rotation_rad": rotation_angle(self.initial_object_offset[:3, :3]),
            "timed_out_path_indices": list(self.timed_out_indices),
            "translation_rmse_m": float(np.sqrt(np.mean(trans**2))) if trans.size else None,
            "translation_max_m": float(trans.max()) if trans.size else None,
            "translation_final_m": float(trans[-1]) if trans.size else None,
            "orientation_rmse_rad": float(np.sqrt(np.mean(rot**2))) if rot.size else None,
            "orientation_max_rad": float(rot.max()) if rot.size else None,
            "orientation_final_rad": float(rot[-1]) if rot.size else None,
            "cross_track_translation_rmse_m": float(np.sqrt(np.mean(cross_trans**2))) if cross_trans.size else None,
            "cross_track_translation_max_m": float(cross_trans.max()) if cross_trans.size else None,
            "cross_track_translation_final_m": float(cross_trans[-1]) if cross_trans.size else None,
            "cross_track_rotation_rmse_rad": float(np.sqrt(np.mean(cross_rot**2))) if cross_rot.size else None,
            "cross_track_rotation_max_rad": float(cross_rot.max()) if cross_rot.size else None,
            "cross_track_rotation_final_rad": float(cross_rot[-1]) if cross_rot.size else None,
            "translation_coverage_fraction": float(np.mean(cross_trans <= self.config.position_tolerance_m)) if cross_trans.size else 0.0,
            "orientation_coverage_fraction": float(np.mean(cross_rot <= self.config.orientation_tolerance_rad)) if cross_rot.size else 0.0,
            "joint_pose_coverage_fraction": float(np.mean(
                (cross_trans <= self.config.position_tolerance_m) & (cross_rot <= self.config.orientation_tolerance_rad)
            )) if cross_trans.size else 0.0,
            "translation_correction_saturation_count": int(sum(item["correction_translation_saturated"] for item in self.history)),
            "rotation_correction_saturation_count": int(sum(item["correction_rotation_saturated"] for item in self.history)),
            "velocity_saturation_count": int(sum(item["velocity_saturated"] for item in self.history)),
            "acceleration_saturation_count": int(sum(item["acceleration_saturated"] for item in self.history)),
            "lost_grasp": bool(self.lost_grasp),
            "first_lost_control_step": self.first_lost_step,
            "recovered_after_loss": bool(self.recovered_after_loss),
        }
