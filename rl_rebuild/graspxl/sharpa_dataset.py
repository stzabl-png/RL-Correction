# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utilities for deterministic SHARPA object/hand reset annotations.

This module intentionally has no Isaac Sim imports. It can be used from the
scenario config while building USD spawn configs and from task reset code while
building tensor root states.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


DEFAULT_SHARPA_DATASET_ROOT = Path(
    os.environ.get(
        "SHARPA_DATASET_ROOT",
        "/home/magics/mt_dir/Grasp_dataset/sharpa_tabletop/sharpa_extracted",
    )
)
DEFAULT_SHARPA_OBJECT_ID = "0e57cbe8ad3b4cea99d817a7b216a899"


@lru_cache(maxsize=64)
def sample_object_surface_points(obj_path: str, num_points: int, seed: int = 0) -> np.ndarray:
    """Deterministically sample `num_points` surface points (area-weighted) from a GraspXL object mesh,
    in the object's LOCAL frame (the GraspXL meshes are centered at the origin, matching the rigid-body
    frame used by object_pos/object_rot). Used to build the STAGE-2/3 object point cloud for mesh objects
    instead of the analytic cylinder. Returns (num_points, 3) float32."""
    import trimesh
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    rng = np.random.RandomState(seed)
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    except TypeError:  # older trimesh without the seed kwarg
        np.random.seed(seed)
        pts, _ = trimesh.sample.sample_surface(mesh, num_points)
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape[0] < num_points:  # pad by resampling vertices if the mesh was tiny/degenerate
        extra = mesh.vertices[rng.randint(0, len(mesh.vertices), num_points - pts.shape[0])]
        pts = np.concatenate([pts, np.asarray(extra, dtype=np.float32)], axis=0)
    return pts[:num_points]


SHARPA_USD_JOINT_NAMES = [
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
]


GRASPXL_SHARPA_FINGER_JOINT_NAMES = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]


@dataclass(frozen=True)
class PoseWxyz:
    position: list[float]
    quat_wxyz: list[float]

    @property
    def quat_xyzw(self) -> list[float]:
        w, x, y, z = self.quat_wxyz
        return [x, y, z, w]


@dataclass(frozen=True)
class SharpaGrasp:
    dataset_root: str
    object_id: str
    object_dir: str
    object_usd_path: str
    object_obj_path: str
    pose_json_path: str
    pose_npz_path: str | None
    pose_index: int
    scale: str | None
    sequence_name: str | None
    sequence_index: int | None
    frame_idx: int | None
    num_frames: int | None
    object_pose_world: PoseWxyz
    right_hand_world: PoseWxyz
    right_hand_object_relative: PoseWxyz | None
    usd_joint_names: list[str]
    joint_positions_rad: list[float]
    joint_positions_deg: list[float]
    trajectory_object_positions: list[list[float]] | None
    trajectory_object_quats_wxyz: list[list[float]] | None
    trajectory_wrist_positions: list[list[float]] | None
    trajectory_wrist_quats_wxyz: list[list[float]] | None
    trajectory_joint_positions_rad: list[list[float]] | None
    trajectory_frame_indices: list[int] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read from dict-like, OmegaConf-like, or attribute-style config objects."""
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(cfg, key, default)


def _as_float_list(values: Iterable[Any], expected_len: int | None, field_name: str) -> list[float]:
    out = [float(v) for v in values]
    if expected_len is not None and len(out) != expected_len:
        raise ValueError(f"{field_name} expected length {expected_len}, got {len(out)}")
    return out


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _pose_from_payload(payload: dict[str, Any], pos_key: str, quat_key: str) -> PoseWxyz:
    return PoseWxyz(
        position=_as_float_list(payload[pos_key], 3, pos_key),
        quat_wxyz=_as_float_list(payload[quat_key], 4, quat_key),
    )


def _normalize_dataset_root(dataset_root: str | Path) -> Path:
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return root


def object_dir_from_id(dataset_root: str | Path, object_id: str) -> Path:
    root = _normalize_dataset_root(dataset_root)
    object_dir = root / object_id
    if not object_dir.is_dir():
        raise FileNotFoundError(f"Object directory does not exist: {object_dir}")
    return object_dir


def load_pose_records(pose_json_path: str | Path) -> list[dict[str, Any]]:
    pose_json_path = Path(pose_json_path).expanduser().resolve()
    if not pose_json_path.is_file():
        raise FileNotFoundError(f"Pose JSON does not exist: {pose_json_path}")

    with pose_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = [data]
    else:
        raise TypeError(f"Unsupported pose JSON root type: {type(data).__name__}")

    if not records:
        raise ValueError(f"No pose records found in {pose_json_path}")
    return records


def _reorder_named_values(
    values: list[float],
    source_names: list[str],
    target_names: list[str],
) -> list[float]:
    if len(values) != len(source_names):
        raise ValueError(f"Got {len(values)} values for {len(source_names)} source names")
    by_name = dict(zip(source_names, values))
    missing = [name for name in target_names if name not in by_name]
    if missing:
        raise ValueError(f"Missing joint values for USD joints: {missing}")
    return [float(by_name[name]) for name in target_names]


def _joint_payload_from_record(record: dict[str, Any]) -> tuple[list[str], list[float], list[float]]:
    """Return USD-ordered joint names, radians, and degrees."""
    joint_payload = record.get("right_hand_joints", {})
    usd_payload = joint_payload.get("usd")

    if usd_payload is not None:
        joint_names = list(usd_payload["joint_names"])
        if "positions_rad" in usd_payload:
            joint_rad = _as_float_list(
                usd_payload["positions_rad"], len(joint_names), "positions_rad"
            )
        elif "positions_deg" in usd_payload:
            joint_rad = [
                math.radians(v)
                for v in _as_float_list(
                    usd_payload["positions_deg"], len(joint_names), "positions_deg"
                )
            ]
        else:
            raise KeyError("right_hand_joints.usd must contain positions_rad or positions_deg")

        if "positions_deg" in usd_payload:
            joint_deg = _as_float_list(
                usd_payload["positions_deg"], len(joint_names), "positions_deg"
            )
        else:
            joint_deg = [math.degrees(v) for v in joint_rad]
        return joint_names, joint_rad, joint_deg

    source = record.get("right_hand_world") or record.get("right_hand_object_relative")
    if source is None:
        raise KeyError("Record has no right_hand_joints.usd or hand joint fallback payload")

    source_names = list(source.get("finger_joint_names", GRASPXL_SHARPA_FINGER_JOINT_NAMES))
    source_rad = _as_float_list(source["finger_joints_22"], len(source_names), "finger_joints_22")
    joint_rad = _reorder_named_values(source_rad, source_names, SHARPA_USD_JOINT_NAMES)
    joint_deg = [math.degrees(v) for v in joint_rad]
    return list(SHARPA_USD_JOINT_NAMES), joint_rad, joint_deg


def _load_npz_metadata(pose_npz_path: Path) -> dict[str, Any]:
    meta_path = pose_npz_path.with_name(f"{pose_npz_path.stem}_meta.json")
    if not meta_path.is_file():
        return {}
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _select_sequence_frame_ids(sequence_index: np.ndarray | None, pose_index: int, record_count: int) -> tuple[int | None, np.ndarray]:
    if sequence_index is None:
        if pose_index not in (0, -1):
            raise IndexError(f"pose_index {pose_index} out of range for single-sequence NPZ")
        return None, np.arange(record_count, dtype=np.int64)

    sequence_index = np.asarray(sequence_index).reshape(-1)
    if sequence_index.shape[0] != record_count:
        raise ValueError(
            f"sequence_index has {sequence_index.shape[0]} rows, expected {record_count}"
        )
    sequence_ids = sorted(int(v) for v in np.unique(sequence_index))
    if pose_index < 0:
        raise IndexError(f"pose_index must be non-negative, got {pose_index}")
    if pose_index < len(sequence_ids):
        sequence_id = sequence_ids[pose_index]
    elif pose_index in sequence_ids:
        sequence_id = pose_index
    else:
        raise IndexError(
            f"pose_index {pose_index} out of range for NPZ sequences; "
            f"valid sequence slots are [0, {len(sequence_ids) - 1}]"
        )
    frame_ids = np.flatnonzero(sequence_index == sequence_id).astype(np.int64)
    if frame_ids.size == 0:
        raise ValueError(f"No frames found for sequence_id {sequence_id}")
    return sequence_id, frame_ids


def _npz_array(data: np.lib.npyio.NpzFile, key: str, expected_last_dim: int | None = None) -> np.ndarray:
    if key not in data.files:
        raise KeyError(f"NPZ missing required key: {key}")
    arr = np.asarray(data[key])
    if expected_last_dim is not None and (arr.ndim < 1 or arr.shape[-1] != expected_last_dim):
        raise ValueError(f"{key} expected last dim {expected_last_dim}, got shape {arr.shape}")
    return arr


def _load_sharpa_grasp_from_npz(
    root: Path, object_id: str, object_dir: Path, pose_npz_path: Path, pose_index: int
) -> SharpaGrasp:
    object_usd_path = object_dir / "object.usd"
    object_obj_path = object_dir / "object.obj"
    if not object_usd_path.is_file():
        raise FileNotFoundError(f"Object USD does not exist: {object_usd_path}")
    if not object_obj_path.is_file():
        raise FileNotFoundError(f"Object OBJ does not exist: {object_obj_path}")

    meta = _load_npz_metadata(pose_npz_path)
    record_object_id = meta.get("object_id")
    if record_object_id is not None and record_object_id != object_id:
        raise ValueError(f"Requested object_id {object_id}, but NPZ metadata says {record_object_id}")

    with np.load(pose_npz_path, allow_pickle=False) as data:
        object_positions = _npz_array(data, "object_position", 3).astype(np.float32)
        record_count = int(object_positions.shape[0])
        sequence_index = np.asarray(data["sequence_index"]) if "sequence_index" in data.files else None
        sequence_id, frame_ids = _select_sequence_frame_ids(sequence_index, pose_index, record_count)
        if "frame_idx" in data.files:
            frame_idx_all = np.asarray(data["frame_idx"]).reshape(-1)
            frame_ids = frame_ids[np.argsort(frame_idx_all[frame_ids], kind="stable")]
            frame_indices = [int(v) for v in frame_idx_all[frame_ids].tolist()]
        else:
            frame_indices = [int(v) for v in range(len(frame_ids))]

        object_positions = object_positions[frame_ids]
        object_quats = _npz_array(data, "object_quat_wxyz", 4).astype(np.float32)[frame_ids]
        wrist_positions = _npz_array(data, "wrist_position", 3).astype(np.float32)[frame_ids]
        wrist_quats = _npz_array(data, "wrist_quat_wxyz", 4).astype(np.float32)[frame_ids]
        relative_wrist_positions = _npz_array(data, "relative_wrist_position", 3).astype(np.float32)[frame_ids]
        relative_wrist_quats = _npz_array(data, "relative_wrist_quat_wxyz", 4).astype(np.float32)[frame_ids]
        source_pose = _npz_array(data, "source_pose_28", None).astype(np.float32)[frame_ids]
        if source_pose.shape[-1] >= 28:
            finger_positions = source_pose[:, 6:28]
        elif source_pose.shape[-1] == 22:
            finger_positions = source_pose
        else:
            raise ValueError(f"source_pose_28 expected 28 or 22 columns, got {source_pose.shape}")

        joint_meta = meta.get("joint_names", {}) if isinstance(meta, dict) else {}
        source_joint_names = list(joint_meta.get("graspxl_finger_22", GRASPXL_SHARPA_FINGER_JOINT_NAMES))
        if len(source_joint_names) != finger_positions.shape[-1]:
            raise ValueError(
                f"NPZ source joint names has {len(source_joint_names)} names, "
                f"but trajectory has {finger_positions.shape[-1]} joints"
            )
        by_name = {name: idx for idx, name in enumerate(source_joint_names)}
        missing = [name for name in SHARPA_USD_JOINT_NAMES if name not in by_name]
        if missing:
            raise ValueError(f"NPZ missing USD joint values for: {missing}")
        reorder_ids = np.asarray([by_name[name] for name in SHARPA_USD_JOINT_NAMES], dtype=np.int64)
        joint_traj = finger_positions[:, reorder_ids].astype(np.float32)

        sequence_names = data["sequence_names"] if "sequence_names" in data.files else None
        if sequence_names is not None and sequence_id is not None and 0 <= sequence_id < len(sequence_names):
            sequence_name = str(sequence_names[sequence_id])
        elif sequence_names is not None and pose_index < len(sequence_names):
            sequence_name = str(sequence_names[pose_index])
        else:
            sequence_name = None
        if "num_frames" in data.files:
            num_frames = int(np.asarray(data["num_frames"]).reshape(-1)[frame_ids[-1]])
        else:
            num_frames = int(len(frame_ids))

    final_idx = -1
    object_pose = PoseWxyz(
        position=_as_float_list(object_positions[final_idx].tolist(), 3, "object_position"),
        quat_wxyz=_as_float_list(object_quats[final_idx].tolist(), 4, "object_quat_wxyz"),
    )
    hand_pose = PoseWxyz(
        position=_as_float_list(wrist_positions[final_idx].tolist(), 3, "wrist_position"),
        quat_wxyz=_as_float_list(wrist_quats[final_idx].tolist(), 4, "wrist_quat_wxyz"),
    )
    rel_pose = PoseWxyz(
        position=_as_float_list(relative_wrist_positions[final_idx].tolist(), 3, "relative_wrist_position"),
        quat_wxyz=_as_float_list(relative_wrist_quats[final_idx].tolist(), 4, "relative_wrist_quat_wxyz"),
    )
    joint_rad = _as_float_list(joint_traj[final_idx].tolist(), len(SHARPA_USD_JOINT_NAMES), "joint_positions_rad")
    joint_deg = [math.degrees(v) for v in joint_rad]

    return SharpaGrasp(
        dataset_root=str(root),
        object_id=object_id,
        object_dir=str(object_dir),
        object_usd_path=str(object_usd_path),
        object_obj_path=str(object_obj_path),
        pose_json_path=str(pose_npz_path),
        pose_npz_path=str(pose_npz_path),
        pose_index=pose_index,
        scale=meta.get("scale"),
        sequence_name=sequence_name,
        sequence_index=sequence_id,
        frame_idx=frame_indices[final_idx],
        num_frames=num_frames,
        object_pose_world=object_pose,
        right_hand_world=hand_pose,
        right_hand_object_relative=rel_pose,
        usd_joint_names=list(SHARPA_USD_JOINT_NAMES),
        joint_positions_rad=joint_rad,
        joint_positions_deg=joint_deg,
        trajectory_object_positions=object_positions.astype(float).tolist(),
        trajectory_object_quats_wxyz=object_quats.astype(float).tolist(),
        trajectory_wrist_positions=wrist_positions.astype(float).tolist(),
        trajectory_wrist_quats_wxyz=wrist_quats.astype(float).tolist(),
        trajectory_joint_positions_rad=joint_traj.astype(float).tolist(),
        trajectory_frame_indices=frame_indices,
    )


@lru_cache(maxsize=4096)
def _load_sharpa_grasp_cached(dataset_root: str, object_id: str, pose_index: int) -> SharpaGrasp:
    root = _normalize_dataset_root(dataset_root)
    object_dir = object_dir_from_id(root, object_id)
    pose_npz_path = object_dir / "graspxl_final_poses.npz"
    if pose_npz_path.is_file():
        return _load_sharpa_grasp_from_npz(root, object_id, object_dir, pose_npz_path, pose_index)

    pose_json_path = object_dir / "graspxl_final_poses.json"
    object_usd_path = object_dir / "object.usd"
    object_obj_path = object_dir / "object.obj"

    if not object_usd_path.is_file():
        raise FileNotFoundError(f"Object USD does not exist: {object_usd_path}")
    if not object_obj_path.is_file():
        raise FileNotFoundError(f"Object OBJ does not exist: {object_obj_path}")

    records = load_pose_records(pose_json_path)
    if pose_index < 0 or pose_index >= len(records):
        raise IndexError(
            f"pose_index {pose_index} out of range for {pose_json_path}; "
            f"valid range is [0, {len(records) - 1}]"
        )

    record = records[pose_index]
    record_object_id = record.get("object_id")
    if record_object_id is not None and record_object_id != object_id:
        raise ValueError(f"Requested object_id {object_id}, but pose record says {record_object_id}")

    object_pose = _pose_from_payload(record["object_pose_world"], "position", "quat_wxyz")
    hand_pose = _pose_from_payload(
        record["right_hand_world"], "wrist_position", "wrist_quat_wxyz"
    )

    rel_payload = record.get("right_hand_object_relative")
    rel_pose = None
    if rel_payload is not None:
        rel_pose = _pose_from_payload(rel_payload, "wrist_position", "wrist_quat_wxyz")

    joint_names, joint_rad, joint_deg = _joint_payload_from_record(record)

    return SharpaGrasp(
        dataset_root=str(root),
        object_id=object_id,
        object_dir=str(object_dir),
        object_usd_path=str(object_usd_path),
        object_obj_path=str(object_obj_path),
        pose_json_path=str(pose_json_path),
        pose_npz_path=None,
        pose_index=pose_index,
        scale=record.get("scale"),
        sequence_name=record.get("sequence_name"),
        sequence_index=None,
        frame_idx=record.get("frame_idx"),
        num_frames=record.get("num_frames"),
        object_pose_world=object_pose,
        right_hand_world=hand_pose,
        right_hand_object_relative=rel_pose,
        usd_joint_names=joint_names,
        joint_positions_rad=joint_rad,
        joint_positions_deg=joint_deg,
        trajectory_object_positions=None,
        trajectory_object_quats_wxyz=None,
        trajectory_wrist_positions=None,
        trajectory_wrist_quats_wxyz=None,
        trajectory_joint_positions_rad=None,
        trajectory_frame_indices=None,
    )


def load_sharpa_grasp(
    dataset_root: str | Path = DEFAULT_SHARPA_DATASET_ROOT,
    object_id: str = DEFAULT_SHARPA_OBJECT_ID,
    pose_index: int = 0,
) -> SharpaGrasp:
    """Load one corrected SHARPA grasp record."""
    root = str(_normalize_dataset_root(dataset_root))
    return _load_sharpa_grasp_cached(root, str(object_id), int(pose_index))


def list_object_ids(dataset_root: str | Path = DEFAULT_SHARPA_DATASET_ROOT) -> list[str]:
    root = _normalize_dataset_root(dataset_root)
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir())


def get_sharpa_reset_config(task_or_env_config: Any) -> Any:
    task_cfg = cfg_get(task_or_env_config, "task", task_or_env_config)
    return cfg_get(task_cfg, "sharpa_reset", None)


def is_sharpa_reset_enabled(task_or_env_config: Any) -> bool:
    return bool(cfg_get(get_sharpa_reset_config(task_or_env_config), "enable", False))


def get_sharpa_object_name(task_or_env_config: Any) -> str:
    cfg = get_sharpa_reset_config(task_or_env_config)
    return str(cfg_get(cfg, "object_name", "front_object"))


def resolve_sharpa_grasps(sharpa_cfg: Any, num_envs: int) -> list[SharpaGrasp]:
    """Resolve deterministic per-env SHARPA grasp records.

    Object ids and pose indices are cycled by env id. For evaluation and
    visualization, ``random_object_count`` can select a reproducible subset from
    the configured object pool before cycling.
    """
    if not bool(cfg_get(sharpa_cfg, "enable", False)):
        return []

    dataset_root = cfg_get(sharpa_cfg, "dataset_root", DEFAULT_SHARPA_DATASET_ROOT)
    object_ids = _as_list(cfg_get(sharpa_cfg, "object_ids", None))
    object_id = cfg_get(sharpa_cfg, "object_id", DEFAULT_SHARPA_OBJECT_ID)
    if bool(cfg_get(sharpa_cfg, "sample_all_objects", False)) and not object_ids:
        object_ids = list_object_ids(dataset_root)
    if not object_ids:
        object_ids = [object_id]

    random_object_count = int(cfg_get(sharpa_cfg, "random_object_count", 0))
    if random_object_count > 0 and len(object_ids) > random_object_count:
        object_sample_seed = int(cfg_get(sharpa_cfg, "object_sample_seed", 0))
        rng = np.random.default_rng(object_sample_seed)
        selected_ids = sorted(
            rng.choice(len(object_ids), size=random_object_count, replace=False).tolist()
        )
        object_ids = [object_ids[idx] for idx in selected_ids]

    pose_indices = [int(v) for v in _as_list(cfg_get(sharpa_cfg, "pose_indices", None))]
    pose_index = int(cfg_get(sharpa_cfg, "pose_index", 0))
    if not pose_indices:
        pose_indices = [pose_index]

    if num_envs <= 0:
        return []

    grasps = []
    for env_idx in range(num_envs):
        grasp = load_sharpa_grasp(
            dataset_root=dataset_root,
            object_id=str(object_ids[env_idx % len(object_ids)]),
            pose_index=pose_indices[env_idx % len(pose_indices)],
        )
        grasps.append(grasp)
    return grasps
