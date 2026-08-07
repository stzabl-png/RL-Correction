"""Explicit right/left conversion used by the fallback Dexonomy path.

This adapter is not a success shortcut.  It only produces a left-hand
candidate for the *same object*.  The candidate must still pass the normal
Gate-1 IK check and Gate-2 Isaac physical screening before the scene manifest
may be marked ready.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


_REFLECT_Y = np.diag([1.0, -1.0, 1.0])


def swap_side_name(name: str, target_side: str) -> str:
    if target_side not in {"left", "right"}:
        raise ValueError(f"target_side must be left/right, got {target_side}")
    for prefix in ("left_", "right_"):
        if name.startswith(prefix):
            return target_side + name[len(prefix) - 1 :]
    for prefix in ("L_", "R_"):
        if name.startswith(prefix):
            return ("L_" if target_side == "left" else "R_") + name[2:]
    raise ValueError(f"joint/body name has no recognized side prefix: {name}")


def mirrored_joint_order(right_names: list[str] | tuple[str, ...]) -> list[str]:
    return [swap_side_name(name, "left") for name in right_names]


def quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[..., 0, 1] = 2.0 * (x * y - w * z)
    matrix[..., 0, 2] = 2.0 * (x * z + w * y)
    matrix[..., 1, 0] = 2.0 * (x * y + w * z)
    matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[..., 1, 2] = 2.0 * (y * z - w * x)
    matrix[..., 2, 0] = 2.0 * (x * z - w * y)
    matrix[..., 2, 1] = 2.0 * (y * z + w * x)
    matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    from tasks.pour.reference import rotation_matrix_to_wxyz

    return rotation_matrix_to_wxyz(matrix).astype(np.float64)


def mirror_position_y(position: np.ndarray) -> np.ndarray:
    return np.asarray(position, dtype=np.float64) @ _REFLECT_Y


def mirror_quaternion_y(quaternion: np.ndarray) -> np.ndarray:
    rotation = quat_wxyz_to_matrix(quaternion)
    mirrored = _REFLECT_Y @ rotation @ _REFLECT_Y
    return matrix_to_quat_wxyz(mirrored)


def mirror_pose29(row: np.ndarray) -> np.ndarray:
    value = np.asarray(row, dtype=np.float64)
    if value.shape[-1] != 29:
        raise ValueError(f"Dexonomy pose must end in 29 values, got {value.shape}")
    result = value.copy()
    result[..., :3] = mirror_position_y(value[..., :3])
    result[..., 3:7] = mirror_quaternion_y(value[..., 3:7])
    # Sharpa left/right joints share the same flexion sign convention.
    return result


def mirror_prior_for_left(source: str | Path, destination: str | Path) -> None:
    """Create an unapproved left candidate from a right-hand prior NPZ."""

    with np.load(source, allow_pickle=False) as data:
        required = {
            "grasp",
            "squeeze",
            "pregrasp",
            "contact_pos",
            "contact_normal",
            "canon_rot",
            "hand_side",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"prior missing fields: {sorted(missing)}")
        payload = {name: data[name].copy() for name in data.files}
    source_side = np.asarray(payload["hand_side"], dtype=np.uint8).tobytes().decode(
        "utf-8"
    )
    if source_side != "right":
        raise ValueError(f"mirror fallback expects a right prior, got {source_side!r}")
    payload["grasp"] = mirror_pose29(payload["grasp"])
    payload["squeeze"] = mirror_pose29(payload["squeeze"])
    payload["pregrasp"] = mirror_pose29(payload["pregrasp"])
    payload["contact_pos"] = mirror_position_y(payload["contact_pos"])
    payload["contact_normal"] = mirror_position_y(payload["contact_normal"])
    payload["contact_centroid"] = payload["contact_pos"].mean(axis=0)
    payload["canon_rot"] = mirror_quaternion_y(payload["canon_rot"])
    payload["hand_side"] = np.frombuffer(b"left", dtype=np.uint8)
    payload["mirror_requires_physical_screen"] = np.array(1, dtype=np.int64)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)
