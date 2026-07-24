"""BODex-faithful surface-normal-facing grasp seed generator.

Original BODex avoids naive random-orientation seeding
(``third_party/BODex/src/curobo/util/sample_grasp.py::HeurGraspSeedGenerator``)
by:

1. Sampling candidate points + outward normals on the object's (inflated,
   optionally convex-hulled) mesh surface.
2. Building a wrist rotation whose local +X axis (the "palm/approach" axis)
   points from the wrist toward the sampled surface point (i.e. the
   ``-normal`` direction), via a 6D-rotation Gram-Schmidt construction.
3. Starting every seed from a fixed, pre-shaped "cupped hand" joint
   configuration (``seeder_cfg.q``) instead of random joint angles.
4. Applying small jitter (``jitter_dist``/``jitter_angle``) around this base
   pose: a free ~180 degree roll around the approach axis, a small +-15
   degree tilt, and a 0-3cm standoff pullback + lateral offset.

This module ports steps 1-4 faithfully for OCIR's single-object, no-arm
Sharpa Wave setup (BODex's optional IK-for-arm and load-from-disk seed paths
are out of scope here, as is ``seeder_cfg.palm_down_bias``, which is dead
config in this codepath -- it is not referenced anywhere in BODex's own
``sample_grasp.py`` and is a vestige of a different, door/handle-grasping
pipeline). It does not import anything from ``third_party/BODex``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import torch
import trimesh
import yaml

from curobo._src.types.device_cfg import DeviceCfg

from ocir.grasp_synthesis.bodex_curobo_v2.grasp_cost import HaltonGenerator
from ocir.grasp_synthesis.bodex_curobo_v2.grasp_energy import normalize_vector


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == "X":
        flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"axis must be X, Y, or Z, got {axis!r}")
    return torch.stack(flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    if euler_angles.shape[-1] != 3 or len(convention) != 3:
        raise ValueError("euler_angles must be (..., 3) and convention must have 3 letters")
    matrices = [_axis_angle_rotation(c, e) for c, e in zip(convention, torch.unbind(euler_angles, -1))]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def compute_rotation_matrix_from_ortho6d(poses: torch.Tensor) -> torch.Tensor:
    x_raw = poses[..., 0:3]
    y_raw = poses[..., 3:6]
    x = normalize_vector(x_raw)
    y = normalize_vector(y_raw - (y_raw * x).sum(dim=-1, keepdim=True) * x)
    z = normalize_vector(torch.cross(x, y, dim=-1))
    return torch.stack((x, y, z), -1)


def init_r_from_axis(axis_palm_raw: torch.Tensor, device_cfg: DeviceCfg) -> torch.Tensor:
    """Build a wrist rotation whose local +X axis equals ``axis_palm_raw``."""

    axis_palm = normalize_vector(axis_palm_raw)
    shape = [1] * (axis_palm.ndim - 1) + [3]
    base_t1 = device_cfg.to_device([0.0, 1.0, 0.0]).view(shape)
    base_t2 = device_cfg.to_device([0.0, 0.0, 1.0]).view(shape)
    proj_xy = (base_t1 * axis_palm).sum(dim=-1, keepdim=True).abs()
    axis_thumb = torch.where(proj_xy > 0.99, base_t2, base_t1)
    r6d = torch.cat([axis_palm, axis_thumb], dim=-1)
    return compute_rotation_matrix_from_ortho6d(r6d)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive = x > 0
    ret = torch.where(positive, torch.sqrt(torch.clamp(x, min=0)), ret)
    return ret


def _standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (..., 3, 3) -> quaternion (..., 4), real part first."""

    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"invalid rotation matrix shape {matrix.shape}")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(
        torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )
    flr = torch.tensor(0.1, device=q_abs.device, dtype=q_abs.dtype)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[
        torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return _standardize_quaternion(out)


def sample_surface_points_and_normals(
    mesh: trimesh.Trimesh, num: int, inflate: float, convex_hull: bool, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Port of BODex's ``Mesh.get_samples_on_surface`` (without the collision-
    free-filtering radius column, which needs a multi-obstacle scene this
    single-object port does not have)."""

    m = mesh.copy()
    if inflate:
        m.vertices = m.vertices + inflate * m.vertex_normals
    sample_mesh = m.convex_hull if convex_hull else m
    points, face_index = trimesh.sample.sample_surface_even(sample_mesh, num, seed=seed)
    points = np.asarray(points)
    face_index = np.asarray(face_index)
    if points.shape[0] < num:
        repeat = math.ceil(num / max(points.shape[0], 1))
        points = np.tile(points, (repeat, 1))[:num]
        face_index = np.tile(face_index, repeat)[:num]
    normals = np.asarray(sample_mesh.face_normals)[face_index]
    return points, normals


def load_hand_pose_transfer(path: Path, device_cfg: DeviceCfg) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    transfer = {}
    for link, entry in data.items():
        r = device_cfg.to_device(np.asarray(entry["r"], dtype=np.float32))
        t = device_cfg.to_device(np.asarray(entry["t"], dtype=np.float32))
        transfer[link] = (r, t)
    return transfer


@dataclass
class HeurGraspSeedGenerator:
    """Surface-normal-facing grasp seed generator (BODex's ``HeurGraspSeedGenerator``,
    simplified for a single object, floating hand, no arm/IK)."""

    device_cfg: DeviceCfg
    base_link: str
    joint_order: list[str]
    seeder_cfg: dict
    cspace_joint_names: list[str]
    transfer_rot: torch.Tensor
    transfer_trans: torch.Tensor
    index_seed: int = 1312
    jitter_seed: int = 1312

    base_t: torch.Tensor = field(init=False, repr=False)
    base_r: torch.Tensor = field(init=False, repr=False)
    base_q: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        q_values = self.seeder_cfg["q"]
        if len(q_values) != len(self.cspace_joint_names):
            raise ValueError(
                f"seeder_cfg.q has {len(q_values)} values but robot config cspace has "
                f"{len(self.cspace_joint_names)} joints"
            )
        q_by_name = dict(zip(self.cspace_joint_names, q_values))
        missing = [name for name in self.joint_order if name not in q_by_name]
        if missing:
            raise KeyError(f"seeder_cfg.q has no value for joints {missing}")
        self.base_q = self.device_cfg.to_device([q_by_name[name] for name in self.joint_order]).view(1, -1)

        jitter_dist = self.seeder_cfg["jitter_dist"]
        jitter_angle = self.seeder_cfg["jitter_angle"]
        jitter_low = list(jitter_dist[0]) + [a / 180.0 * math.pi for a in jitter_angle[0]]
        jitter_up = list(jitter_dist[1]) + [a / 180.0 * math.pi for a in jitter_angle[1]]
        self._jitter_gen = HaltonGenerator(
            6, self.device_cfg, up_bounds=jitter_up, low_bounds=jitter_low, seed=self.jitter_seed
        )
        self._index_gen: HaltonGenerator | None = None

    def reset(self, surface_points: torch.Tensor, surface_normals: torch.Tensor) -> None:
        """(Re)build the pool of candidate base poses from object-surface samples."""

        axis_palm = -surface_normals  # inward-facing approach direction
        base_r = init_r_from_axis(axis_palm, self.device_cfg)
        base_t = surface_points
        # Apply the robot-specific hand-pose transfer (identity for Sharpa,
        # but computed generally in case the asset config ever changes).
        base_r = base_r @ self.transfer_rot.view(1, 3, 3)
        base_t = base_t + (base_r @ self.transfer_trans.view(1, 3, 1)).squeeze(-1)
        self.base_r = base_r
        self.base_t = base_t
        n = base_t.shape[0]
        self._index_gen = HaltonGenerator(
            1, self.device_cfg, up_bounds=[float(n)], low_bounds=[0.0], seed=self.index_seed
        )

    def get_samples(self, num_samples: int) -> torch.Tensor:
        """Sample ``num_samples`` full action vectors [pos(3), quat_wxyz(4), joints]."""

        if self._index_gen is None:
            raise RuntimeError("HeurGraspSeedGenerator.reset() must be called before get_samples()")
        n = self.base_t.shape[0]
        idx = self._index_gen.get_samples(num_samples, bounded=True).view(-1).long().clamp(min=0, max=n - 1)
        base_t = self.base_t[idx]
        base_r = self.base_r[idx]

        jitter = self._jitter_gen.get_samples(num_samples, bounded=True)
        rand_dist = jitter[:, :3]
        rand_angle = jitter[:, 3:]
        jitter_rot = euler_angles_to_matrix(torch.flip(rand_angle, dims=[-1]), "ZYX")
        final_rot = base_r @ jitter_rot
        final_trans = base_t - (final_rot @ rand_dist.unsqueeze(-1)).squeeze(-1)
        final_quat = matrix_to_quaternion(final_rot)

        base_q = self.base_q.expand(num_samples, -1)
        return torch.cat([final_trans, final_quat, base_q], dim=-1)
