"""The Stage A / Stage B contract: a per-step grasp trajectory.

Pure NumPy + stdlib only -- this module is imported by both the grasp-synthesis
conda env (Stage A, ``generator.py``) and the isaacsim conda env (Stage B,
``ocir.isaac.simulate_grasp_traj``). It must never import torch or pxr.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

SEGMENT_RETARGET = 0
SEGMENT_APPROACH = 1
SEGMENT_CLOSE = 2
SEGMENT_SQUEEZE = 3
SEGMENT_CARRY = 4
SEGMENT_NAMES = ("retarget", "approach", "close", "squeeze", "carry")

_ARRAY_KEYS = (
    "hand_pos_camera",
    "hand_quat_camera",
    "finger_targets",
    "object_pos_camera",
    "object_quat_camera",
    "segment",
)


def matrix_to_pos_quat(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(...,4,4) homogeneous transform -> (pos (...,3), quat_wxyz (...,4))."""

    matrix = np.asarray(matrix, dtype=np.float64)
    pos = matrix[..., :3, 3]
    rot = matrix[..., :3, :3]
    quat = matrix_to_quat_wxyz(rot)
    return pos, quat


def pos_quat_to_matrix(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """(...,3), (...,4) wxyz -> (...,4,4) homogeneous transform."""

    pos = np.asarray(pos, dtype=np.float64)
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    rot = quat_wxyz_to_matrix(quat_wxyz)
    batch_shape = pos.shape[:-1]
    out = np.zeros(batch_shape + (4, 4), dtype=np.float64)
    out[..., :3, :3] = rot
    out[..., :3, 3] = pos
    out[..., 3, 3] = 1.0
    return out


def compose_pos_quat(
    pos_a: np.ndarray, quat_a: np.ndarray, pos_b: np.ndarray, quat_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compose (pos_a, quat_a) after (pos_b, quat_b): result = A @ B."""

    mat_a = pos_quat_to_matrix(pos_a, quat_a)
    mat_b = pos_quat_to_matrix(pos_b, quat_b)
    return matrix_to_pos_quat(mat_a @ mat_b)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12, None)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    out = np.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return out.reshape(quat.shape[:-1] + (3, 3))


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Standard Shepperd/branchless-max-trace-term rotation-matrix -> wxyz quaternion."""

    m = matrix
    batch_shape = m.shape[:-2]
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    q_abs_sq = np.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        axis=-1,
    )
    q_abs = np.sqrt(np.clip(q_abs_sq, 0.0, None))

    quat_candidates = np.stack(
        [
            np.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], axis=-1),
            np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], axis=-1),
        ],
        axis=-2,
    )
    denom = 2.0 * np.clip(q_abs[..., None], 0.1, None)
    quat_candidates = quat_candidates / denom

    best = np.argmax(q_abs, axis=-1)
    flat_candidates = quat_candidates.reshape((-1, 4, 4))
    flat_best = best.reshape(-1)
    flat_out = flat_candidates[np.arange(flat_candidates.shape[0]), flat_best]
    out = flat_out.reshape(batch_shape + (4,))
    out = out / np.clip(np.linalg.norm(out, axis=-1, keepdims=True), 1e-12, None)
    # Standardize sign: real part >= 0.
    sign = np.where(out[..., 0:1] < 0, -1.0, 1.0)
    return out * sign


@dataclass(frozen=True)
class GraspTrajectory:
    hand_pos_camera: np.ndarray        # (T,3) float64, camera frame
    hand_quat_camera: np.ndarray       # (T,4) float64 wxyz, camera frame
    finger_targets: np.ndarray         # (T,22) float64 radians, full joint_order
    object_pos_camera: np.ndarray      # (T,3) float64, reference/recorded object pose
    object_quat_camera: np.ndarray     # (T,4) float64 wxyz
    segment: np.ndarray                # (T,) int8, one of SEGMENT_*
    dt: float
    joint_order: tuple[str, ...]
    grasp_json: str
    sequence_dir: str
    switch_frame_index: int
    grasp_root_tf: list                # (4,4) nested list, hand<-object rigid transform at grasp
    clearance_report: dict = field(default_factory=dict)
    extra_metadata: dict = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return int(self.hand_pos_camera.shape[0])

    @property
    def fps(self) -> float:
        return 1.0 / float(self.dt)

    def __post_init__(self) -> None:
        t = self.num_steps
        expected = {
            "hand_pos_camera": (t, 3),
            "hand_quat_camera": (t, 4),
            "finger_targets": (t, len(self.joint_order)),
            "object_pos_camera": (t, 3),
            "object_quat_camera": (t, 4),
            "segment": (t,),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(self, name).shape)
            if actual != shape:
                raise ValueError(f"GraspTrajectory.{name} has shape {actual}, expected {shape}")

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "trajectory.npz",
            hand_pos_camera=np.asarray(self.hand_pos_camera, dtype=np.float64),
            hand_quat_camera=np.asarray(self.hand_quat_camera, dtype=np.float64),
            finger_targets=np.asarray(self.finger_targets, dtype=np.float64),
            object_pos_camera=np.asarray(self.object_pos_camera, dtype=np.float64),
            object_quat_camera=np.asarray(self.object_quat_camera, dtype=np.float64),
            segment=np.asarray(self.segment, dtype=np.int8),
        )
        meta = {
            "dt": float(self.dt),
            "joint_order": list(self.joint_order),
            "grasp_json": str(self.grasp_json),
            "sequence_dir": str(self.sequence_dir),
            "switch_frame_index": int(self.switch_frame_index),
            "grasp_root_tf": self.grasp_root_tf,
            "clearance_report": self.clearance_report,
            "extra_metadata": self.extra_metadata,
            "num_steps": self.num_steps,
            "segment_names": SEGMENT_NAMES,
        }
        (out_dir / "trajectory.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, out_dir: str | Path) -> "GraspTrajectory":
        out_dir = Path(out_dir)
        with np.load(out_dir / "trajectory.npz") as data:
            arrays = {key: data[key] for key in _ARRAY_KEYS}
        meta = json.loads((out_dir / "trajectory.json").read_text(encoding="utf-8"))
        return cls(
            hand_pos_camera=arrays["hand_pos_camera"],
            hand_quat_camera=arrays["hand_quat_camera"],
            finger_targets=arrays["finger_targets"],
            object_pos_camera=arrays["object_pos_camera"],
            object_quat_camera=arrays["object_quat_camera"],
            segment=arrays["segment"],
            dt=float(meta["dt"]),
            joint_order=tuple(meta["joint_order"]),
            grasp_json=str(meta["grasp_json"]),
            sequence_dir=str(meta["sequence_dir"]),
            switch_frame_index=int(meta["switch_frame_index"]),
            grasp_root_tf=meta["grasp_root_tf"],
            clearance_report=meta.get("clearance_report", {}),
            extra_metadata=meta.get("extra_metadata", {}),
        )
