"""Static reconstructed object -> Step4 ``DataUnit`` placement.

This reference builder is intentionally object-only.  It uses the reconstructed hand
trajectory solely as a spatial marker for the first interaction frame; it does
not replace the robot reference trajectory or move either robot arm.

Placement contract:

* select the earliest ``phase_left/right == 1`` frame;
* align the selected wrist through the existing replay -> table transform;
* put the rotated mesh AABB centre on that wrist XY;
* preserve the FoundationPose quaternion (apart from frame conversion and
  normalization);
* translate only Z until the rotated mesh bottom is ``obj_gap`` above the table.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl_rebuild.correction import frames as F
from rl_rebuild.correction.load_replay import load as load_replay
from rl_rebuild.correction.schema import DataUnit, ObjectSemantics


@dataclass(frozen=True)
class StaticPlacement:
    """Auditable result of one static-object placement."""

    pose: np.ndarray
    hand: str
    source_frame: int
    aligned_frame: int
    wrist_xy: np.ndarray
    center_xy: np.ndarray
    center_error_m: float
    bottom_gap_m: float
    table_margin_m: float
    reach_min_m: float | None = None


def first_interaction(data, hand: str | None = None) -> tuple[str, int]:
    """Return the interacting hand and earliest contact frame.

    A simultaneous left/right start is ambiguous for object placement.  Such a
    clip must provide ``hand=`` explicitly instead of silently preferring one
    side.
    """

    if hand not in (None, "left", "right"):
        raise ValueError(f"hand must be left/right/None, got {hand!r}")

    starts: dict[str, int] = {}
    for side in ("left", "right"):
        key = f"phase_{side}"
        if key not in data.files:
            continue
        frames = np.flatnonzero(np.asarray(data[key]).astype(np.int8) == 1)
        if len(frames):
            starts[side] = int(frames[0])

    if hand is not None:
        if hand not in starts:
            raise ValueError(f"phase_{hand} has no interaction frame")
        return hand, starts[hand]
    if not starts:
        raise ValueError("replay has no phase_left/right == 1 interaction frame")

    first = min(starts.values())
    sides = [side for side, frame in starts.items() if frame == first]
    if len(sides) != 1:
        raise ValueError(
            f"left/right interactions both start at frame {first}; set hand explicitly"
        )
    return sides[0], first


def compute_static_placement(
    replay,
    mesh_path: str,
    aligned_joints: np.ndarray,
    aligned_length: int,
    *,
    hand: str | None = None,
    table_height: float = 0.85,
    obj_gap: float = 0.002,
    table_half: float = 0.6,
    scene_rot: str = "identity",
    quat_order: str = "wxyz",
    reach_origins: np.ndarray | None = None,
    arm_reach: float = 0.755,
) -> StaticPlacement:
    """Compute and validate the initial pose for a static reconstructed mesh."""

    side, source_frame = first_interaction(replay, hand)
    source_length = len(replay["obj_pose"])
    if source_length < 1 or aligned_length < 1:
        raise ValueError("empty replay/aligned trajectory")
    aligned_frame = 0 if source_length == 1 else int(round(
        source_frame * (aligned_length - 1) / (source_length - 1)
    ))

    joints = np.asarray(aligned_joints, dtype=np.float64)
    if joints.shape != (aligned_length, 21, 3):
        raise ValueError(
            f"aligned_joints must be ({aligned_length},21,3), got {joints.shape}"
        )
    wrist = joints[aligned_frame, 0]
    if not np.isfinite(wrist).all():
        raise ValueError(f"non-finite {side} wrist at aligned frame {aligned_frame}")

    q_fp = np.asarray(replay["obj_pose"][source_frame, 3:7], dtype=np.float64)
    if quat_order == "xyzw":
        q_fp = F.quat_xyzw_to_wxyz(q_fp)
    elif quat_order != "wxyz":
        raise ValueError(f"unsupported quat_order={quat_order!r}")
    q_scene = F.rotmat_to_quat(F.scene_rotation(scene_rot))
    q_fp = F.quat_mul(q_scene, q_fp)
    norm = float(np.linalg.norm(q_fp))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"invalid FoundationPose quaternion at frame {source_frame}")
    q_fp = q_fp / norm

    vertices = F.load_obj_verts(mesh_path)
    if len(vertices) == 0 or not np.isfinite(vertices).all():
        raise ValueError(f"mesh has no finite vertices: {mesh_path}")
    center_local = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    q_many = np.broadcast_to(q_fp, (len(vertices), 4))
    rotated = F.rot_apply(q_many, vertices)
    center_rotated = F.rot_apply(q_fp[None], center_local[None])[0]

    origin = np.array([
        wrist[0] - center_rotated[0],
        wrist[1] - center_rotated[1],
        table_height + obj_gap - rotated[:, 2].min(),
    ])
    world_vertices = rotated + origin
    center_world = center_rotated + origin
    center_error = float(np.linalg.norm(center_world[:2] - wrist[:2]))
    bottom_gap = float(world_vertices[:, 2].min() - table_height)

    xy_min = world_vertices[:, :2].min(axis=0)
    xy_max = world_vertices[:, :2].max(axis=0)
    table_margin = float(min(
        xy_min[0] + table_half,
        table_half - xy_max[0],
        xy_min[1] + table_half,
        table_half - xy_max[1],
    ))

    if center_error > 1e-3:
        raise ValueError(f"object centre/wrist XY error {center_error:.6f}m > 1mm")
    if abs(bottom_gap - obj_gap) > 1e-3:
        raise ValueError(
            f"mesh bottom gap {bottom_gap:.6f}m differs from requested {obj_gap:.6f}m"
        )
    if table_margin < 0.0:
        raise ValueError(
            f"placed mesh exceeds table by {-table_margin:.4f}m (table half={table_half:.3f}m)"
        )

    reach_min = None
    if reach_origins is not None:
        anchors = np.asarray(reach_origins, dtype=np.float64).reshape(-1, 3)
        if not len(anchors) or not np.isfinite(anchors).all():
            raise ValueError("reach_origins must contain finite XYZ anchors")
        reach_min = float(np.linalg.norm(anchors - center_world[None], axis=1).min())
        if reach_min > arm_reach:
            raise ValueError(
                f"object centre is {reach_min:.3f}m from nearest arm anchor; "
                f"limit is {arm_reach:.3f}m"
            )

    pose = np.concatenate([origin, q_fp]).astype(np.float32)
    return StaticPlacement(
        pose=pose,
        hand=side,
        source_frame=source_frame,
        aligned_frame=aligned_frame,
        wrist_xy=wrist[:2].astype(np.float32),
        center_xy=center_world[:2].astype(np.float32),
        center_error_m=center_error,
        bottom_gap_m=bottom_gap,
        table_margin_m=table_margin,
        reach_min_m=reach_min,
    )


def load_static_reconstruction(
    npz_path: str,
    mesh_path: str,
    *,
    usd_path: str = "",
    clip_id: str = "",
    hand: str | None = None,
    table_height: float = 0.85,
    obj_gap: float = 0.002,
    table_half: float = 0.6,
    scene_rot: str = "identity",
    quat_order: str = "wxyz",
    target_hz: float | None = None,
    semantics: ObjectSemantics | None = None,
    verbose: bool = False,
    return_placement: bool = False,
) -> DataUnit | tuple[DataUnit, StaticPlacement]:
    """Load replay data and replace only its initial object placement."""

    with np.load(npz_path, allow_pickle=True) as replay:
        side, _ = first_interaction(replay, hand)
        data_unit = load_replay(
            npz_path,
            mesh_path,
            usd_path=usd_path,
            clip_id=clip_id,
            hand=side,
            table_height=table_height,
            obj_gap=obj_gap,
            scene_rot=scene_rot,
            quat_order=quat_order,
            target_hz=target_hz,
            semantics=semantics,
            verbose=verbose,
            initial_pose_mode="preserve",
        )
        placement = compute_static_placement(
            replay,
            mesh_path,
            data_unit.ref.mano_joints,
            data_unit.ref.L,
            hand=side,
            table_height=table_height,
            obj_gap=obj_gap,
            table_half=table_half,
            scene_rot=scene_rot,
            quat_order=quat_order,
        )
    data_unit.object_init_pose = placement.pose
    # The generic replay loader estimates interaction from hand/mesh distance.
    # Static placement is defined by phase_left/right instead, so expose that
    # same authoritative start frame to every downstream consumer.
    interaction_end = max(
        placement.aligned_frame, int(data_unit.ref.interaction_seg[1])
    )
    data_unit.ref.interaction_seg = (placement.aligned_frame, interaction_end)

    if verbose:
        print(
            f"[static] hand={placement.hand} contact={placement.source_frame} "
            f"aligned={placement.aligned_frame} centre_xy="
            f"{placement.center_xy.round(4).tolist()} gap="
            f"{placement.bottom_gap_m * 1000:.1f}mm table_margin="
            f"{placement.table_margin_m * 100:.1f}cm"
        )
    if return_placement:
        return data_unit, placement
    return data_unit
