#!/usr/bin/env python3
"""Build a MANO/object full-trajectory reference from a grasp trajectory.

The input trajectory is treated as authoritative before its first carry
frame.  Generation copies that prefix unchanged, replaces the carry with the
recorded object/MANO path, and holds the final grasp finger targets exactly.
The resulting carry is replayed open-loop by ``simulate_full_traj``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

import numpy as np
import yaml

from ocir.full_traj.reference import (
    FullTrajectoryReference,
    poses_to_components,
    resample_synchronized_poses,
    retime_synchronized_poses,
)
from ocir.grasp_synthesis.anchored_bodex.demo_analysis import wrist_frame_from_keypoints
from ocir.grasp_synthesis.anchored_bodex.demo_data import HumanDemo
from ocir.grasp_synthesis.assets import DEFAULT_SHARPA_WAVE_RIGHT_CONFIG, load_sharpa_wave_right
from ocir.grasp_traj.trajectory_schema import (
    SEGMENT_CARRY,
    GraspTrajectory,
    matrix_to_pos_quat,
    pos_quat_to_matrix,
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
    time_scale: float = 3.0,
    ease_in_seconds: float = 0.3,
    ease_out_seconds: float = 0.3,
    retarget_mode: str = "object",
) -> tuple[GraspTrajectory, FullTrajectoryReference]:
    if retarget_mode not in ("wrist", "object"):
        raise ValueError(f"retarget_mode must be 'wrist' or 'object', got {retarget_mode!r}")
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
    )
    # The raw human carry is too dynamic for the fixed-target PD grasp: it
    # starts at full speed (the clip is cut at the grasp frame) and sustains
    # 2-3x the wrist speed of the proven vertical lift.  Slow it down and
    # ramp the speed at both ends; the geometric path is unchanged.
    object_camera, wrist_camera, source_frame = retime_synchronized_poses(
        object_camera,
        wrist_camera,
        source_frame,
        dt=float(source.dt),
        time_scale=float(time_scale),
        ease_in_seconds=float(ease_in_seconds),
        ease_out_seconds=float(ease_out_seconds),
    )
    wrist_pos, wrist_quat = poses_to_components(wrist_camera)
    fixed_fingers = np.asarray(source.finger_targets[carry_start - 1], dtype=np.float64).copy()

    source_hand_last = np.eye(4, dtype=np.float64)
    source_hand_last[:3, :3] = quat_wxyz_to_matrix(source.hand_quat_camera[carry_start - 1])
    source_hand_last[:3, 3] = source.hand_pos_camera[carry_start - 1]

    # Align the demonstrated object path to the nominal object pose at the
    # end of the grasp prefix.  The object path is reference-only in friction
    # simulation, but this removes the artificial carry-boundary jump and
    # makes tracking metrics meaningful.
    source_object_last = pos_quat_to_matrix(
        source.object_pos_camera[carry_start - 1],
        source.object_quat_camera[carry_start - 1],
    )
    object_alignment = source_object_last @ np.linalg.inv(object_camera[0])
    aligned_object_camera = object_alignment[None] @ object_camera
    object_pos, object_quat = poses_to_components(aligned_object_camera)

    if retarget_mode == "object":
        # Drive the wrist so the OBJECT replays its demonstrated path through
        # the synthesized grasp relation, held rigid.  The human's wrist
        # motion relative to the object (in-hand adjustment, MANO wrist
        # noise) is deliberately discarded: with fixed finger targets it is
        # pure commanded slip that pries the grasp open (measured 30+mm /
        # ~10deg over a DexYCB carry).
        grasp_relation = np.linalg.inv(source_object_last) @ source_hand_last
        retargeted_wrist_camera = aligned_object_camera @ grasp_relation
        retarget_rule = "aligned_demo_object_t @ inverse(source_object_0) @ robot_wrist_0"
    else:
        # Transfer the demonstrated MANO wrist motion onto the synthesized
        # wrist at carry entry.  Left alignment preserves every MANO wrist
        # displacement: inv(robot_wrist[0]) @ robot_wrist[k] ==
        # inv(mano_wrist[0]) @ mano_wrist[k].
        wrist_alignment = source_hand_last @ np.linalg.inv(wrist_camera[0])
        retargeted_wrist_camera = wrist_alignment[None] @ wrist_camera
        retarget_rule = "robot_wrist_0 @ inverse(mano_wrist_0) @ mano_wrist_t"
    retargeted_pos, retargeted_quat = matrix_to_pos_quat(retargeted_wrist_camera)

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
        "control_mode": "open_loop",
        "retarget_mode": retarget_mode,
        "retarget_rule": retarget_rule,
        "object_reference_rule": "source_object_0 @ inverse(demo_object_0) @ demo_object_t",
        "timing_policy": "source_video_frames_retimed",
        "time_scale": float(time_scale),
        "ease_in_seconds": float(ease_in_seconds),
        "ease_out_seconds": float(ease_out_seconds),
        "object_alignment_camera": object_alignment.tolist(),
        "finger_policy": "fixed_final_grasp_target",
        "legacy_segment_3_semantics": "stationary_grasp_hold_only",
    }
    trajectory = GraspTrajectory(
        hand_pos_camera=np.concatenate([source.hand_pos_camera[prefix], retargeted_pos], axis=0),
        hand_quat_camera=np.concatenate([source.hand_quat_camera[prefix], retargeted_quat], axis=0),
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
            "generated_duration_seconds": float((n_carry - 1) * source.dt),
            "raw_start_frame": int(frames[0]),
            "raw_end_frame": int(frames[-1]),
            "raw_valid_frame_count": int(frames.size),
            "generated_frame_count": int(n_carry),
            "joint_order": list(source.joint_order),
            "mano_transfer": str(calib_path),
            "control_mode": "open_loop",
            "retarget_mode": retarget_mode,
            "time_scale": float(time_scale),
            "ease_in_seconds": float(ease_in_seconds),
            "ease_out_seconds": float(ease_out_seconds),
        },
    )
    return trajectory, reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-traj-dir", type=Path, required=True, help="Existing trajectory.npz/json with a completed grasp and carry segment.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--asset-config", type=Path, default=DEFAULT_SHARPA_WAVE_RIGHT_CONFIG)
    parser.add_argument("--retarget-mode", choices=["object", "wrist"], default="object", help="object: drive the wrist so the demo OBJECT path replays through the rigid grasp relation (discards the human's in-hand wrist adjustments, which are pure commanded slip under fixed fingers). wrist: replay the MANO wrist motion verbatim.")
    parser.add_argument("--time-scale", type=float, default=3.0, help="Carry duration multiple of the source video clock. The demo carry is typically 2-3x too fast for the fixed-target PD grasp; 1.0 replays the raw video timing.")
    parser.add_argument("--ease-in", type=float, default=0.3, help="Seconds over which the carry ramps from rest to path speed (removes the velocity step at carry entry).")
    parser.add_argument("--ease-out", type=float, default=0.3, help="Seconds over which the carry ramps back to rest at the end.")
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
            time_scale=float(args.time_scale),
            ease_in_seconds=float(args.ease_in),
            ease_out_seconds=float(args.ease_out),
            retarget_mode=str(args.retarget_mode),
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
