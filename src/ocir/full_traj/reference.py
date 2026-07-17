"""Serialization and construction helpers for full-trajectory references."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from ocir.full_traj.se3 import batch_pos_quat_to_matrix, interpolate_transform, rotation_angle
from ocir.grasp_traj.trajectory_schema import matrix_to_pos_quat, pos_quat_to_matrix


REFERENCE_NPZ = "full_traj_reference.npz"
REFERENCE_JSON = "full_traj_reference.json"


def resample_synchronized_poses(
    object_poses: np.ndarray,
    wrist_poses: np.ndarray,
    source_frame_indices: np.ndarray,
    *,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Densify synchronized poses without changing the demonstrated path."""

    object_poses = np.asarray(object_poses, dtype=np.float64)
    wrist_poses = np.asarray(wrist_poses, dtype=np.float64)
    source_frame_indices = np.asarray(source_frame_indices, dtype=np.float64)
    if object_poses.shape != wrist_poses.shape or object_poses.ndim != 3 or object_poses.shape[1:] != (4, 4):
        raise ValueError("object_poses and wrist_poses must both have shape (N,4,4)")
    if object_poses.shape[0] != source_frame_indices.shape[0]:
        raise ValueError("source_frame_indices length must match the pose paths")
    if object_poses.shape[0] == 0:
        raise ValueError("full-trajectory reference cannot be empty")

    out_object = [object_poses[0]]
    out_wrist = [wrist_poses[0]]
    out_source = [source_frame_indices[0]]
    for i in range(object_poses.shape[0] - 1):
        obj_delta = np.linalg.inv(object_poses[i]) @ object_poses[i + 1]
        wrist_delta = np.linalg.inv(wrist_poses[i]) @ wrist_poses[i + 1]
        distance = max(
            float(np.linalg.norm(object_poses[i + 1, :3, 3] - object_poses[i, :3, 3])),
            float(np.linalg.norm(wrist_poses[i + 1, :3, 3] - wrist_poses[i, :3, 3])),
        )
        angle = max(rotation_angle(obj_delta[:3, :3]), rotation_angle(wrist_delta[:3, :3]))
        n_trans = int(np.ceil(distance / max_translation_step_m)) if max_translation_step_m > 0.0 else 1
        n_rot = int(np.ceil(angle / max_rotation_step_rad)) if max_rotation_step_rad > 0.0 else 1
        subdivisions = max(1, n_trans, n_rot)
        for step in range(1, subdivisions + 1):
            alpha = step / float(subdivisions)
            out_object.append(interpolate_transform(object_poses[i], object_poses[i + 1], alpha))
            out_wrist.append(interpolate_transform(wrist_poses[i], wrist_poses[i + 1], alpha))
            out_source.append(source_frame_indices[i] * (1.0 - alpha) + source_frame_indices[i + 1] * alpha)
    return np.asarray(out_object), np.asarray(out_wrist), np.asarray(out_source)


@dataclass(frozen=True)
class FullTrajectoryReference:
    object_pos_camera: np.ndarray
    object_quat_camera: np.ndarray
    mano_wrist_pos_camera: np.ndarray
    mano_wrist_quat_camera: np.ndarray
    source_frame_index: np.ndarray
    fixed_finger_targets: np.ndarray
    carry_start_step: int
    source_fps: float
    source_trajectory_dir: str
    sequence_dir: str
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = int(self.object_pos_camera.shape[0])
        expected = {
            "object_pos_camera": (n, 3),
            "object_quat_camera": (n, 4),
            "mano_wrist_pos_camera": (n, 3),
            "mano_wrist_quat_camera": (n, 4),
            "source_frame_index": (n,),
        }
        for name, shape in expected.items():
            if tuple(np.asarray(getattr(self, name)).shape) != shape:
                raise ValueError(f"{name} has shape {np.asarray(getattr(self, name)).shape}, expected {shape}")
        if n < 2:
            raise ValueError("full-trajectory reference needs at least two poses")
        if np.asarray(self.fixed_finger_targets).ndim != 1:
            raise ValueError("fixed_finger_targets must be a 1-D joint vector")

    @property
    def num_path_points(self) -> int:
        return int(self.object_pos_camera.shape[0])

    @property
    def object_poses_camera(self) -> np.ndarray:
        return batch_pos_quat_to_matrix(self.object_pos_camera, self.object_quat_camera)

    @property
    def mano_wrist_poses_camera(self) -> np.ndarray:
        return batch_pos_quat_to_matrix(self.mano_wrist_pos_camera, self.mano_wrist_quat_camera)

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / REFERENCE_NPZ,
            object_pos_camera=np.asarray(self.object_pos_camera, dtype=np.float64),
            object_quat_camera=np.asarray(self.object_quat_camera, dtype=np.float64),
            mano_wrist_pos_camera=np.asarray(self.mano_wrist_pos_camera, dtype=np.float64),
            mano_wrist_quat_camera=np.asarray(self.mano_wrist_quat_camera, dtype=np.float64),
            source_frame_index=np.asarray(self.source_frame_index, dtype=np.float64),
            fixed_finger_targets=np.asarray(self.fixed_finger_targets, dtype=np.float64),
        )
        payload = {
            "carry_start_step": int(self.carry_start_step),
            "source_fps": float(self.source_fps),
            "source_trajectory_dir": self.source_trajectory_dir,
            "sequence_dir": self.sequence_dir,
            "num_path_points": self.num_path_points,
            "metadata": self.metadata,
        }
        (out_dir / REFERENCE_JSON).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "FullTrajectoryReference":
        directory = Path(directory)
        with np.load(directory / REFERENCE_NPZ, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
        payload = json.loads((directory / REFERENCE_JSON).read_text(encoding="utf-8"))
        return cls(
            object_pos_camera=arrays["object_pos_camera"],
            object_quat_camera=arrays["object_quat_camera"],
            mano_wrist_pos_camera=arrays["mano_wrist_pos_camera"],
            mano_wrist_quat_camera=arrays["mano_wrist_quat_camera"],
            source_frame_index=arrays["source_frame_index"],
            fixed_finger_targets=arrays["fixed_finger_targets"],
            carry_start_step=int(payload["carry_start_step"]),
            source_fps=float(payload["source_fps"]),
            source_trajectory_dir=str(payload["source_trajectory_dir"]),
            sequence_dir=str(payload["sequence_dir"]),
            metadata=payload.get("metadata", {}),
        )


def poses_to_components(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return matrix_to_pos_quat(np.asarray(poses, dtype=np.float64))


def components_to_poses(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    return pos_quat_to_matrix(pos, quat)
