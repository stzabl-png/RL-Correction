#!/usr/bin/env python3
"""Build a MANO/object full-trajectory reference from a grasp trajectory.

The input trajectory is treated as authoritative before its first carry
frame.  Generation copies that prefix unchanged, replaces the carry with the
recorded object/MANO path, and holds the final grasp finger targets exactly.
Closed-loop correction happens later inside ``simulate_full_traj``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

import numpy as np
import yaml

from ocir.full_traj.reference import FullTrajectoryReference, poses_to_components, resample_synchronized_poses
from ocir.grasp_synthesis.anchored_bodex.demo_analysis import wrist_frame_from_keypoints
from ocir.grasp_synthesis.anchored_bodex.demo_data import HumanDemo
from ocir.grasp_synthesis.assets import DEFAULT_SHARPA_WAVE_RIGHT_CONFIG, load_sharpa_wave_right
from ocir.grasp_traj.trajectory_schema import (
    SEGMENT_CARRY,
    GraspTrajectory,
    matrix_to_pos_quat,
    quat_wxyz_to_matrix,
)


def _load_wrist_calibration(asset_config: str | Path | None) -> tuple[np.ndarray, Path]:
    asset = load_sharpa_wave_right(asset_config)
    # This is the MANO-wrist -> Sharpa-base calibration produced by
    # calibrate_mano_sharpa.py.  It is distinct from BODex's joint-pose
    # transfer YAML (asset.bodex["hand_pose_transfer"]).
    path = asset.root / "grasp_synthesis/bodex/mano_transfer.yml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(data["wrist"]["r"], dtype=np.float64).reshape(3, 3)
    pose[:3, 3] = np.asarray(data["wrist"]["t"], dtype=np.float64).reshape(3)
    return pose, path


def build_full_trajectory(
    grasp_traj_dir: str | Path,
    *,
    asset_config: str | Path | None = None,
    max_translation_step_m: float = 0.005,
    max_rotation_step_rad: float = np.deg2rad(3.0),
) -> tuple[GraspTrajectory, FullTrajectoryReference]:
    grasp_traj_dir = Path(grasp_traj_dir)
    source = GraspTrajectory.load(grasp_traj_dir)
    carry_rows = np.flatnonzero(source.segment == SEGMENT_CARRY)
    if carry_rows.size == 0:
        raise ValueError(f"{grasp_traj_dir}: trajectory has no carry segment")
    carry_start = int(carry_rows[0])
    if carry_start == 0:
        raise ValueError("carry cannot be the first trajectory frame; a completed grasp prefix is required")
    if np.any(source.segment[carry_start:] != SEGMENT_CARRY):
        raise ValueError("carry segment must be the final contiguous trajectory segment")

    demo = HumanDemo.from_sequence_dir(source.sequence_dir)
    if demo.mano_side.lower() != "right":
        raise ValueError(f"full trajectory currently requires the right-hand Sharpa demo, got {demo.mano_side!r}")
    extra = source.extra_metadata
    start_frame = int(extra.get("grasp_frame_index", extra.get("carry_start_frame", 0)))
    frames = demo.valid_indices[demo.valid_indices >= start_frame]
    if frames.size < 2:
        raise ValueError(f"{source.sequence_dir}: fewer than two valid MANO/object frames at or after {start_frame}")

    wrist_calib, calib_path = _load_wrist_calibration(asset_config)
    wrist_object = np.stack(
        [wrist_frame_from_keypoints(demo.hand_joints_object[int(frame)]) @ wrist_calib for frame in frames],
        axis=0,
    )
    object_camera = np.asarray(demo.object_pose_camera[frames], dtype=np.float64)
    wrist_camera = object_camera @ wrist_object
    object_camera, wrist_camera, source_frame = resample_synchronized_poses(
        object_camera,
        wrist_camera,
        frames.astype(np.float64),
        max_translation_step_m=max_translation_step_m,
        max_rotation_step_rad=max_rotation_step_rad,
    )
    object_pos, object_quat = poses_to_components(object_camera)
    wrist_pos, wrist_quat = poses_to_components(wrist_camera)
    fixed_fingers = np.asarray(source.finger_targets[carry_start - 1], dtype=np.float64).copy()

    # The compatible trajectory contains a continuous nominal wrist path for
    # inspection/open-loop A/B.  Runtime closed-loop playback recomputes the
    # nominal hand from the measured carry-entry grasp relation.
    source_hand_last = np.eye(4, dtype=np.float64)
    source_hand_last[:3, :3] = quat_wxyz_to_matrix(source.hand_quat_camera[carry_start - 1])
    source_hand_last[:3, 3] = source.hand_pos_camera[carry_start - 1]
    wrist_alignment = source_hand_last @ np.linalg.inv(wrist_camera[0])
    nominal_wrist_camera = wrist_alignment[None] @ wrist_camera
    nominal_pos, nominal_quat = matrix_to_pos_quat(nominal_wrist_camera)

    n_carry = object_camera.shape[0]
    prefix = slice(0, carry_start)
    new_extra = dict(source.extra_metadata)
    new_extra["full_traj"] = {
        "source_trajectory_dir": str(grasp_traj_dir),
        "carry_start_step": carry_start,
        "source_demo_start_frame": start_frame,
        "num_raw_demo_frames": int(frames.size),
        "num_path_points": int(n_carry),
        "mano_transfer": str(calib_path),
        "max_translation_step_m": float(max_translation_step_m),
        "max_rotation_step_rad": float(max_rotation_step_rad),
        "finger_policy": "fixed_final_grasp_target",
        "legacy_segment_3_semantics": "stationary_grasp_hold_only",
    }
    trajectory = GraspTrajectory(
        hand_pos_camera=np.concatenate([source.hand_pos_camera[prefix], nominal_pos], axis=0),
        hand_quat_camera=np.concatenate([source.hand_quat_camera[prefix], nominal_quat], axis=0),
        finger_targets=np.concatenate(
            [source.finger_targets[prefix], np.tile(fixed_fingers[None], (n_carry, 1))], axis=0
        ),
        object_pos_camera=np.concatenate([source.object_pos_camera[prefix], object_pos], axis=0),
        object_quat_camera=np.concatenate([source.object_quat_camera[prefix], object_quat], axis=0),
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
    raw_duration = float((frames[-1] - frames[0]) / source.fps)
    reference = FullTrajectoryReference(
        object_pos_camera=object_pos,
        object_quat_camera=object_quat,
        mano_wrist_pos_camera=wrist_pos,
        mano_wrist_quat_camera=wrist_quat,
        source_frame_index=source_frame,
        fixed_finger_targets=fixed_fingers,
        carry_start_step=carry_start,
        source_fps=float(source.fps),
        source_trajectory_dir=str(grasp_traj_dir),
        sequence_dir=source.sequence_dir,
        metadata={
            "source_duration_seconds": raw_duration,
            "raw_start_frame": int(frames[0]),
            "raw_end_frame": int(frames[-1]),
            "joint_order": list(source.joint_order),
            "mano_transfer": str(calib_path),
        },
    )
    return trajectory, reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-traj-dir", type=Path, required=True, help="Existing trajectory.npz/json with a completed grasp and carry segment.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--asset-config", type=Path, default=DEFAULT_SHARPA_WAVE_RIGHT_CONFIG)
    parser.add_argument("--max-translation-step", type=float, default=0.005, help="Maximum translation between resampled path points (m).")
    parser.add_argument("--max-rotation-step-deg", type=float, default=3.0, help="Maximum rotation between resampled path points (degrees).")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        print(f"OCIR_FULL_TRAJ refusing non-empty --out-dir {args.out_dir}; pass --overwrite", file=sys.stderr)
        return 2
    try:
        trajectory, reference = build_full_trajectory(
            args.grasp_traj_dir,
            asset_config=args.asset_config,
            max_translation_step_m=float(args.max_translation_step),
            max_rotation_step_rad=np.deg2rad(float(args.max_rotation_step_deg)),
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        trajectory.save(args.out_dir)
        reference.save(args.out_dir)
    except Exception as exc:
        print(f"OCIR_FULL_TRAJ generation failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    print(
        f"OCIR_FULL_TRAJ wrote {args.out_dir} "
        f"(carry_start={reference.carry_start_step}, path_points={reference.num_path_points}, "
        f"fixed_finger_targets={reference.fixed_finger_targets.size})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
