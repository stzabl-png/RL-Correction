"""Gravity alignment helpers for final world-frame export."""

from __future__ import annotations

from pathlib import Path

import numpy as np

GRAVITY_SCHEMA_VERSION = "vipe_gravity_v2"
LEGACY_GEOCALIB_UP_AS_GRAVITY_SCHEMA = "vipe_gravity_v1"
GRAVITY_DIRNAME = "gravity"
GRAVITY_FRAME = "gravity_z_up_world"
VIPE_FRAME = "vipe_world"


def vipe_gravity_path(vipe_dir: Path, video_id: str) -> Path:
    return vipe_dir / GRAVITY_DIRNAME / f"{video_id}.npz"


def _normalize(vec: np.ndarray) -> np.ndarray:
    out = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(out))
    if norm <= 1e-9:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return out / norm


def rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a rotation matrix that maps source direction to target direction."""
    a = _normalize(source)
    b = _normalize(target)
    cross = np.cross(a, b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return np.eye(3, dtype=np.float64)
    if dot < -1.0 + 1e-9:
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(a, helper))) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize(np.cross(a, helper))
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / float(np.dot(cross, cross)))


def gravity_alignment_transform(gravity_world: np.ndarray) -> np.ndarray:
    """Return T_zup_from_world so +Z is opposite the gravity direction."""
    gravity = _normalize(gravity_world)
    up = -gravity
    rot = rotation_from_vectors(up, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    return out


def gravity_camera_forward_alignment_transform(
    gravity_world: np.ndarray,
    reference_c2w_world: np.ndarray,
    *,
    min_horizontal_norm: float = 1e-6,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """Return z-up T_from_world with world +X set by the reference camera +Z axis.

    The gravity direction fixes only roll/pitch. This composes that alignment
    with a yaw around final +Z so the reference camera focal axis, projected into
    the gravity-aligned XY plane, becomes final world +X.
    """
    t_zup_from_world = gravity_alignment_transform(gravity_world)
    r_zup_from_world = t_zup_from_world[:3, :3]
    c2w = np.asarray(reference_c2w_world, dtype=np.float64)
    if c2w.shape != (4, 4):
        raise ValueError(f"reference_c2w_world must have shape (4, 4); got {c2w.shape}")

    camera_forward_world = _normalize(c2w[:3, 2])
    camera_forward_zup = r_zup_from_world @ camera_forward_world
    camera_forward_xy = camera_forward_zup.copy()
    camera_forward_xy[2] = 0.0
    horizontal_norm = float(np.linalg.norm(camera_forward_xy))
    if horizontal_norm <= min_horizontal_norm:
        raise ValueError(
            "Reference camera focal axis is too close to gravity to define world +X "
            f"(horizontal norm {horizontal_norm:.3e})."
        )

    target_x_pre_yaw = camera_forward_xy / horizontal_norm
    yaw_angle = float(np.arctan2(target_x_pre_yaw[1], target_x_pre_yaw[0]))
    cos_yaw = float(np.cos(-yaw_angle))
    sin_yaw = float(np.sin(-yaw_angle))
    r_yaw = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r_yaw @ r_zup_from_world
    r_world_from_out = out[:3, :3].T
    metadata = {
        "camera_forward_vipe_world": camera_forward_world,
        "camera_forward_gravity_z_up_pre_yaw": camera_forward_zup,
        "camera_forward_xy_pre_yaw": target_x_pre_yaw,
        "camera_forward_xy_norm": horizontal_norm,
        "yaw_angle_rad": yaw_angle,
        "world_x_direction_vipe_world": r_world_from_out @ np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "world_y_direction_vipe_world": r_world_from_out @ np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "world_z_direction_vipe_world": r_world_from_out @ np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    return out, metadata


def load_vipe_gravity(vipe_dir: Path, video_id: str) -> dict[str, np.ndarray | str]:
    path = vipe_gravity_path(vipe_dir, video_id)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing ViPE gravity artifact: {path}. Re-run the vipe step with the gravity-enabled wrapper."
        )
    with np.load(path, allow_pickle=False) as data:
        required = {"schema_version", "gravity_world", "up_world", "sample_frame_indices", "gravity_cam"}
        missing = sorted(required - set(data.files))
        if missing:
            raise RuntimeError(f"ViPE gravity artifact missing keys: {missing}")
        schema = str(data["schema_version"].item())
        if schema not in {GRAVITY_SCHEMA_VERSION, LEGACY_GEOCALIB_UP_AS_GRAVITY_SCHEMA}:
            raise RuntimeError(
                f"Unsupported ViPE gravity schema {schema}; expected {GRAVITY_SCHEMA_VERSION}"
            )
        gravity_world = _normalize(data["gravity_world"])
        up_world = _normalize(data["up_world"])
        gravity_cam = np.asarray(data["gravity_cam"], dtype=np.float64)
        up_cam = np.asarray(data["up_cam"], dtype=np.float64) if "up_cam" in data.files else None
        if schema == LEGACY_GEOCALIB_UP_AS_GRAVITY_SCHEMA:
            # GeoCalib's Gravity.vec3d projects to the image up-field. Older
            # artifacts stored it as gravity_world; reinterpret those files so
            # fused outputs get +Z upward instead of toward the ground.
            up_world = gravity_world
            gravity_world = -up_world
            up_cam = gravity_cam
            gravity_cam = -gravity_cam
        return {
            "schema_version": schema,
            "gravity_world": gravity_world,
            "up_world": up_world,
            "sample_frame_indices": np.asarray(data["sample_frame_indices"], dtype=np.int32),
            "gravity_cam": gravity_cam,
            "up_cam": up_cam if up_cam is not None else -gravity_cam,
            "gravity_uncertainty": np.asarray(data["gravity_uncertainty"], dtype=np.float64)
            if "gravity_uncertainty" in data.files
            else np.zeros((0,), dtype=np.float64),
            "source": str(data["source"].item()) if "source" in data.files else "vipe_geocalib",
        }
