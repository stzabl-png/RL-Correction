"""Small MANO loader and NumPy LBS implementation for DexYCB labels."""

from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
import sys
import types

import numpy as np
import yaml


@dataclass(frozen=True)
class ManoModel:
    side: str
    v_template: np.ndarray
    shapedirs: np.ndarray
    posedirs: np.ndarray
    weights: np.ndarray
    faces: np.ndarray
    j_regressor: np.ndarray
    kintree_table: np.ndarray
    hands_components: np.ndarray
    hands_mean: np.ndarray


def _install_chumpy_pickle_shim() -> None:
    """Install minimal classes required to unpickle official MANO py2 files."""

    if "chumpy" in sys.modules and "chumpy.reordering" in sys.modules:
        return

    class FakeCh:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def __setstate__(self, state) -> None:
            if isinstance(state, dict):
                self.__dict__.update(state)
            else:
                self.__dict__["state"] = state

        @property
        def r(self) -> np.ndarray:
            for key in ("x", "_x", "v", "_v", "a"):
                if key in self.__dict__:
                    value = self.__dict__[key]
                    return np.asarray(value.r if hasattr(value, "r") else value)
            return np.asarray(self.__dict__.get("state", []))

    class FakeSelect(FakeCh):
        @property
        def r(self) -> np.ndarray:
            base = np.asarray(self.a.r if hasattr(self.a, "r") else self.a).reshape(-1)
            selected = base[np.asarray(self.idxs, dtype=int)]
            return selected.reshape(tuple(self.preferred_shape))

    chumpy = types.ModuleType("chumpy")
    ch = types.ModuleType("chumpy.ch")
    reordering = types.ModuleType("chumpy.reordering")
    ch.Ch = FakeCh
    reordering.Select = FakeSelect
    chumpy.Ch = FakeCh
    chumpy.ch = ch
    chumpy.reordering = reordering
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = ch
    sys.modules["chumpy.reordering"] = reordering


def load_mano_model(path: str | Path, side: str) -> ManoModel:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"missing MANO model file: {path}")
    _install_chumpy_pickle_shim()
    with path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    return ManoModel(
        side=side,
        v_template=np.asarray(data["v_template"], dtype=float),
        shapedirs=np.asarray(data["shapedirs"].r if hasattr(data["shapedirs"], "r") else data["shapedirs"], dtype=float)[:, :, :10],
        posedirs=np.asarray(data["posedirs"], dtype=float),
        weights=np.asarray(data["weights"], dtype=float),
        faces=np.asarray(data["f"], dtype=np.int64),
        j_regressor=np.asarray(data["J_regressor"].toarray(), dtype=float),
        kintree_table=np.asarray(data["kintree_table"], dtype=np.int64),
        hands_components=np.asarray(data["hands_components"], dtype=float),
        hands_mean=np.asarray(data["hands_mean"], dtype=float),
    )


def load_mano_betas(path: str | Path) -> np.ndarray:
    data = yaml.safe_load(Path(path).expanduser().read_text(encoding="utf-8"))
    return np.asarray(data["betas"], dtype=float)[:10]


def rodrigues(axis_angles: np.ndarray) -> np.ndarray:
    axis_angles = np.asarray(axis_angles, dtype=float).reshape(-1, 3)
    out = np.tile(np.eye(3), (axis_angles.shape[0], 1, 1))
    theta = np.linalg.norm(axis_angles, axis=1)
    valid = theta > 1e-8
    if np.any(valid):
        k = axis_angles[valid] / theta[valid, None]
        K = np.zeros((k.shape[0], 3, 3), dtype=float)
        K[:, 0, 1] = -k[:, 2]
        K[:, 0, 2] = k[:, 1]
        K[:, 1, 0] = k[:, 2]
        K[:, 1, 2] = -k[:, 0]
        K[:, 2, 0] = -k[:, 1]
        K[:, 2, 1] = k[:, 0]
        I = np.eye(3)[None, :, :]
        t = theta[valid]
        out[valid] = I + np.sin(t)[:, None, None] * K + (1.0 - np.cos(t))[:, None, None] * (K @ K)
    return out


def mano_lbs(model: ManoModel, pose_axis_angle: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose_axis_angle = np.asarray(pose_axis_angle, dtype=float).reshape(16, 3)
    betas = np.asarray(betas, dtype=float)[:10]

    v_shaped = model.v_template + np.tensordot(model.shapedirs, betas, axes=([2], [0]))
    joints = model.j_regressor @ v_shaped
    rotations = rodrigues(pose_axis_angle)

    pose_feature = (rotations[1:] - np.eye(3)).reshape(-1)
    v_posed = v_shaped + np.tensordot(model.posedirs, pose_feature, axes=([2], [0]))

    parents = []
    for joint_idx in range(16):
        if joint_idx == 0:
            parents.append(-1)
            continue
        parent_id = model.kintree_table[0, joint_idx]
        parents.append(int(np.where(model.kintree_table[1] == parent_id)[0][0]))

    transforms = np.zeros((16, 4, 4), dtype=float)
    for joint_idx, parent_idx in enumerate(parents):
        local = np.eye(4)
        local[:3, :3] = rotations[joint_idx]
        local[:3, 3] = joints[joint_idx] if parent_idx < 0 else joints[joint_idx] - joints[parent_idx]
        transforms[joint_idx] = local if parent_idx < 0 else transforms[parent_idx] @ local

    rest = transforms.copy()
    joints_h = np.concatenate([joints, np.zeros((16, 1), dtype=float)], axis=1)
    for joint_idx in range(16):
        rest[joint_idx] -= np.outer(transforms[joint_idx] @ joints_h[joint_idx], np.array([0.0, 0.0, 0.0, 1.0]))

    vertex_transforms = np.tensordot(model.weights, rest, axes=([1], [0]))
    v_h = np.concatenate([v_posed, np.ones((v_posed.shape[0], 1), dtype=float)], axis=1)
    vertices = np.einsum("vij,vj->vi", vertex_transforms, v_h)[:, :3]
    transformed_joints = transforms[:, :3, 3]
    return vertices, transformed_joints


def pose_m_to_vertices_and_joints(model: ManoModel, pose_m: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose_m = np.asarray(pose_m, dtype=float).reshape(-1)
    if pose_m.shape[0] != 51:
        raise ValueError(f"DexYCB pose_m must have 51 values; got shape {pose_m.shape}")
    pose_axis_angle = np.concatenate(
        [
            pose_m[:3],
            model.hands_mean + pose_m[3:48] @ model.hands_components[:45],
        ]
    )
    vertices, joints16 = mano_lbs(model, pose_axis_angle, betas)
    if model.side == "left":
        tip_indices = [745, 317, 445, 556, 673]
    else:
        tip_indices = [745, 317, 444, 556, 673]
    joints21 = np.concatenate([joints16, vertices[tip_indices]], axis=0)
    joints21 = joints21[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
    translation = pose_m[48:51]
    return vertices + translation, joints21 + translation
