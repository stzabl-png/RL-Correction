"""All-collision-sphere clearance checking against the object SDF.

Shared by the synthesis and trajectory pipelines: ``anchored_bodex`` uses it
at synthesis time to compute the four-stage grasp poses (retreating the raw
grasp out of penetration), and ``grasp_traj`` uses it for switch-frame
selection and transit-path validation. Torch/CUDA (grasp-synthesis conda
env).

Reuses ``bodex_curobo_v2``'s own contact-query machinery
(``SingleObjectContactWorld.get_sphere_contact_pdn``), whose forward pass
(read directly, ``contact_world.py::SphereContactPdnFunction.forward``) only
consumes ``centers``/``radii`` (arbitrary leading batch dims) and the object's
Warp mesh SDF query -- it never reads ``contact_query_buffer`` or
``env_query_idx`` at all, so a clearance-only query needs neither a populated
``ContactBuffer`` nor per-env indices; both are passed as ``None``.
``perturb`` is only used for its ``shape[-2]`` (debug-buffer sizing), so a
minimal ``(1, 1, 3)`` placeholder suffices.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg

from ocir.grasp_synthesis.assets import SharpaWaveAsset
from ocir.grasp_synthesis.bodex_curobo_v2.contact_world import SingleObjectContactWorld
from ocir.grasp_synthesis.bodex_curobo_v2.solver import _load_yaml


def load_all_collision_spheres(collision_config: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """All collision spheres from the curobo sphere yaml (not just the 11
    BODex contact points): ``(link_names (N,), local_centers (N,3), radii (N,))``."""

    spheres_map = collision_config.get("collision_spheres")
    if not isinstance(spheres_map, dict):
        spheres_map = collision_config.get("geometry", {}).get("collision_spheres", {}).get("spheres", {})
    link_names: list[str] = []
    centers: list[list[float]] = []
    radii: list[float] = []
    for link, entries in spheres_map.items():
        for entry in entries:
            link_names.append(link)
            centers.append(list(entry["center"]))
            radii.append(float(entry["radius"]))
    return link_names, np.asarray(centers, dtype=np.float64), np.asarray(radii, dtype=np.float64)


class ClearanceChecker:
    """FKs all 37 hand collision spheres and queries their signed distance to
    the object mesh, for a batch of full (pos, quat, joints) actions."""

    def __init__(self, asset: SharpaWaveAsset, device_cfg: DeviceCfg):
        self.asset = asset
        self.device_cfg = device_cfg

        collision_config = _load_yaml(asset.collision_spheres_path)
        self.sphere_link_names, local_centers, radii = load_all_collision_spheres(collision_config)
        if not self.sphere_link_names:
            raise ValueError(f"no collision spheres found in {asset.collision_spheres_path}")

        tool_frames = list(dict.fromkeys(self.sphere_link_names))
        robot_cfg = RobotCfg.from_basic(
            urdf_path=str(asset.urdf_path),
            base_link=asset.config["base_link"],
            tool_frames=tool_frames,
            device_cfg=device_cfg,
        )
        self.kinematics = Kinematics(robot_cfg.kinematics)
        self.joint_names = tuple(self.kinematics.joint_names)
        self.full_joint_order = tuple(asset.config["joint_order"])
        missing = [name for name in self.joint_names if name not in self.full_joint_order]
        if missing:
            raise ValueError(f"kinematics joints not present in asset joint_order: {missing}")
        self._joint_select_idx = [self.full_joint_order.index(name) for name in self.joint_names]

        self.local_centers = device_cfg.to_device(local_centers.astype(np.float32))  # (N,3)
        self.radii = device_cfg.to_device(radii.astype(np.float32))  # (N,)

    def _select_active_joints(self, joints_full: torch.Tensor) -> torch.Tensor:
        """(..., len(full_joint_order)) -> (..., len(self.joint_names))."""

        idx = torch.as_tensor(self._joint_select_idx, device=joints_full.device, dtype=torch.long)
        return joints_full.index_select(-1, idx)

    def sphere_world_positions(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: (F, 7 + len(full_joint_order)) -- [pos, quat_wxyz, joints],
        object-canonical frame. Returns (F, N, 3) sphere centers in the same
        object-canonical frame."""

        from curobo._src.geom.transform import torch_quaternion_to_matrix

        root_pos = actions[:, :3]
        root_quat = actions[:, 3:7]
        root_quat = root_quat / root_quat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        joints_full = actions[:, 7:]
        q = self._select_active_joints(joints_full)

        kin_state = self.kinematics.compute_kinematics(JointState.from_position(q, joint_names=list(self.joint_names)))
        f = actions.shape[0]
        n = len(self.sphere_link_names)
        link_pos = torch.empty((f, n, 3), device=actions.device, dtype=actions.dtype)
        link_quat = torch.empty((f, n, 4), device=actions.device, dtype=actions.dtype)
        for link in set(self.sphere_link_names):
            pose = kin_state.tool_poses.get_link_pose(link)
            idx = [i for i, name in enumerate(self.sphere_link_names) if name == link]
            for i in idx:
                link_pos[:, i, :] = pose.position
                link_quat[:, i, :] = pose.quaternion

        link_rot = torch_quaternion_to_matrix(link_quat)  # (F,N,3,3)
        local = self.local_centers.view(1, n, 3, 1).expand(f, n, 3, 1)
        base_frame_centers = (link_rot @ local).squeeze(-1) + link_pos  # (F,N,3), base_link frame

        root_rot = torch_quaternion_to_matrix(root_quat)  # (F,3,3)
        world_centers = (root_rot.unsqueeze(1) @ base_frame_centers.unsqueeze(-1)).squeeze(-1) + root_pos.unsqueeze(1)
        return world_centers

    def compute_sphere_clearances(self, actions: torch.Tensor, world: SingleObjectContactWorld) -> torch.Tensor:
        """actions: (F, 7+J). Returns (F, N) signed distance (sphere surface to
        object surface) for every one of the N hand spheres -- positive =
        clear, negative = penetrating. ``compute_clearances`` is the
        min-over-spheres reduction of this; callers that need a per-finger or
        per-sphere reduction (e.g. the pregrasp opening) use this directly."""

        centers = self.sphere_world_positions(actions)  # (F,N,3)
        f, n = centers.shape[:2]
        radii = self.radii.view(1, n).expand(f, n)
        contact_robot_sphere = torch.cat([centers, radii.unsqueeze(-1)], dim=-1)  # (F,N,4)
        perturb = torch.zeros((1, 1, 3), device=actions.device, dtype=actions.dtype)
        _, distance, _, _, _ = world.get_sphere_contact_pdn(
            contact_robot_sphere, None, perturb, env_query_idx=None
        )
        return distance

    def compute_clearances(self, actions: torch.Tensor, world: SingleObjectContactWorld) -> torch.Tensor:
        """actions: (F, 7+J). Returns (F,) min signed distance (sphere surface
        to object surface) over all 37 spheres per frame -- positive = clear,
        negative = penetrating."""

        return self.compute_sphere_clearances(actions, world).min(dim=-1).values

    def min_clearance_m(self, action: torch.Tensor | np.ndarray, world: SingleObjectContactWorld) -> float:
        action_t = action if isinstance(action, torch.Tensor) else self.device_cfg.to_device(np.asarray(action, dtype=np.float32))
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0)
        return float(self.compute_clearances(action_t, world)[0].item())


def build_contact_world(object_mesh_path: Path, urdf_path: Path, device_cfg: DeviceCfg) -> SingleObjectContactWorld:
    """A ``SingleObjectContactWorld`` for clearance-only use: no contact link
    names needed (only ``get_sphere_contact_pdn``'s object-mesh SDF query is
    used, never the mesh-mesh robot-hull path)."""

    return SingleObjectContactWorld(
        object_mesh_path=object_mesh_path,
        robot_urdf_path=urdf_path,
        contact_link_names=(),
        device_cfg=device_cfg,
    )
