"""Step4 environment extension for a bottle body plus a rigid cap.

The existing correction environment remains single-primary-object.  This
isolated subclass keeps the bottle body as that primary object and adds the cap
as a second rigid body.  The screw clips use a GPU-batched analytic PCO-1810
helical constraint; this task does not introduce a reward.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from rl_rebuild.correction import clips, frames as F
from rl_rebuild.correction.load_replay import load as load_replay
from rl_rebuild.correction.recon_kailang.static_reconstruction import (
    StaticPlacement,
    compute_static_placement,
)
from rl_rebuild.correction.env.dexmate_env import DexmateCorrectionEnv
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg
from tasks.recon_kailang.bottle_reconstruction.screw_joint import (
    ScrewSpec,
    assembled_cap_pose,
    author_screw_metadata,
)


class BottleReconstructionEnv(DexmateCorrectionEnv):
    """Add a separately simulated or helically assembled bottle cap."""

    cfg: DexmateCorrectionEnvCfg

    def __init__(self, cfg: DexmateCorrectionEnvCfg, render_mode=None, **kwargs):
        entry = clips.clip_entry(cfg.clip_name)
        secondary = entry.get("secondary")
        if not secondary:
            raise ValueError(f"{cfg.clip_name!r} has no secondary bottle asset")
        self.cap_entry = secondary
        self.screw_spec = ScrewSpec.from_mapping(secondary.get("assembly"))
        body_vertices = F.load_obj_verts(entry["mesh"])
        cap_vertices = F.load_obj_verts(secondary["mesh"])
        self.body_top_offset_m = float(body_vertices[:, 2].max())
        self.free_collision_radius_m = float(
            np.linalg.norm(body_vertices[:, :2], axis=1).max()
            + np.linalg.norm(cap_vertices[:, :2], axis=1).max()
        )

        # Align the right-hand trajectory with the same primary-body scene
        # gauge used by the left-hand DataUnit.  Only cap geometry is used for
        # the final centre/bottom placement calculation.
        self.cap_ref = load_replay(
            entry["npz"], entry["mesh"], usd_path=secondary["usd"],
            clip_id=f"{cfg.clip_name}:cap", hand=secondary["hand"],
            table_height=cfg.table_top_z, obj_gap=0.002,
            target_hz=cfg.target_hz, semantics=secondary["semantics"],
            initial_pose_mode="preserve",
        )
        if self.screw_spec is None or self.screw_spec.mode == "capture":
            with np.load(entry["npz"], allow_pickle=True) as replay:
                self.cap_placement: StaticPlacement = compute_static_placement(
                    replay,
                    secondary["mesh"],
                    self.cap_ref.ref.mano_joints,
                    self.cap_ref.ref.L,
                    hand=secondary["hand"],
                    placement_frame=secondary["placement_frame"],
                    table_height=cfg.table_top_z,
                    obj_gap=0.002,
                    table_half=min(cfg.table_size[0], cfg.table_size[1]) / 2.0,
                )
            self.cap_init_pose_np = self.cap_placement.pose.copy()
        else:
            # During scene construction the primary object still uses its cfg
            # placeholder.  Keeping all three bodies in the closed relative
            # pose prevents a startup impulse before the first reset.
            body_placeholder = np.concatenate([
                np.asarray(cfg.object_cfg.init_state.pos, dtype=np.float64),
                np.asarray(cfg.object_cfg.init_state.rot, dtype=np.float64),
            ])
            self.cap_init_pose_np = assembled_cap_pose(
                body_placeholder, self.screw_spec
            )
        if self.screw_spec is not None:
            cfg.rsi_prob = 0.0
        super().__init__(cfg, render_mode, **kwargs)
        if self.screw_spec is not None:
            if self.screw_spec.mode == "preengaged":
                body_pose = np.concatenate([
                    self.obj_init_pos.detach().cpu().numpy(),
                    self.obj_init_quat.detach().cpu().numpy(),
                ])
                self.cap_init_pose_np = assembled_cap_pose(body_pose, self.screw_spec)
            self.screw_angle = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )
            self.screw_engaged = torch.full(
                (self.num_envs,), self.screw_spec.mode == "preengaged",
                dtype=torch.bool, device=self.device,
            )
            self.screw_has_depth = self.screw_engaged.clone()

    def _setup_scene(self):
        super()._setup_scene()

        clips.ensure_mesh_usd(
            self.cap_entry["mesh"], self.cap_entry["usd"],
            self.cap_entry["semantics"],
        )
        pose = self.cap_init_pose_np
        cap_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cap",
            spawn=sim_utils.UsdFileCfg(usd_path=self.cap_entry["usd"]),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=tuple(float(value) for value in pose[:3]),
                rot=tuple(float(value) for value in pose[3:7]),
            ),
        )
        # Environments already exist at this point. IsaacLab's parent-path
        # regex spawner creates one Cap under every existing env namespace.
        self.cap = RigidObject(cap_cfg)
        self.scene.rigid_objects["cap"] = self.cap

        self.screw_metadata_paths: list[dict[str, str]] = []
        if self.screw_spec is not None:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            for object_path in sim_utils.find_matching_prim_paths(
                    "/World/envs/env_.*/Object"):
                env_path = str(object_path).rsplit("/", 1)[0]
                self.screw_metadata_paths.append(
                    author_screw_metadata(stage, env_path, self.screw_spec)
                )

        self.scene.filter_collisions()

        material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self.cap_entry["semantics"].friction),
            dynamic_friction=float(self.cap_entry["semantics"].friction),
            restitution=0.0,
            friction_combine_mode="average",
            restitution_combine_mode="multiply",
        )
        material_path = "/World/Materials/BottleCap"
        material.func(material_path, material)
        count = 0
        for prim_path in sim_utils.find_matching_prim_paths(
                "/World/envs/env_.*/Cap"):
            sim_utils.bind_physics_material(prim_path, material_path)
            count += 1
        print(
            f"[bottle] cap rigid bodies={count} "
            f"mass={self.cap_entry['semantics'].mass_kg}kg "
            f"friction={self.cap_entry['semantics'].friction}"
        )
        if self.screw_spec is not None:
            print(
                f"[bottle] screw mechanisms={len(self.screw_metadata_paths)} "
                f"pitch={self.screw_spec.pitch_m * 1000:.2f}mm "
                f"turns={self.screw_spec.turns:.1f} "
                f"travel={self.screw_spec.travel_m * 1000:.2f}mm"
            )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        ids = self.cap._ALL_INDICES if env_ids is None else env_ids
        super()._reset_idx(ids)
        origins = self.scene.env_origins[ids]
        pose = torch.as_tensor(
            self.cap_init_pose_np, dtype=torch.float32, device=self.device
        ).unsqueeze(0).repeat(len(ids), 1)
        pose[:, :3] += origins
        velocity = torch.zeros(len(ids), 6, dtype=torch.float32, device=self.device)
        self.cap.write_root_pose_to_sim(pose, ids)
        self.cap.write_root_velocity_to_sim(velocity, ids)
        if self.screw_spec is not None:
            preengaged = self.screw_spec.mode == "preengaged"
            self.screw_angle[ids] = 0.0
            self.screw_engaged[ids] = preengaged
            self.screw_has_depth[ids] = preengaged

    def apply_screw_constraint(
            self, extra_cap_torque_local=None, *, integrate_angle: bool = True):
        """Capture and enforce the helical relation with GPU tensor writes.

        ``integrate_angle=False`` performs the final projection after the last
        physics substep without advancing the screw coordinate a second time.
        IsaacLab exposes no dedicated post-substep callback, so this keeps the
        state consumed by rewards and observations exactly on the helix.
        """

        if self.screw_spec is None:
            return
        body_q = self.object.data.root_quat_w
        cap_q = self.cap.data.root_quat_w
        axis_local = torch.zeros(self.num_envs, 3, device=self.device)
        axis_local[:, 2] = 1.0
        axis_w = quat_apply(body_q, axis_local)

        relative_pos = self.cap.data.root_pos_w - self.object.data.root_pos_w
        absolute_axis_offset = (relative_pos * axis_w).sum(dim=1)
        axial = absolute_axis_offset - self.screw_spec.closed_offset_m
        radial_vector = relative_pos - absolute_axis_offset[:, None] * axis_w
        radial = radial_vector.norm(dim=1)
        cap_axis_w = quat_apply(cap_q, axis_local)
        axis_cos = (cap_axis_w * axis_w).sum(dim=1).clamp(-1.0, 1.0)
        relative_q = quat_mul(quat_conjugate(body_q), cap_q)
        relative_q = relative_q * torch.where(
            relative_q[:, :1] < 0.0, -1.0, 1.0
        )
        yaw = 2.0 * torch.atan2(relative_q[:, 3], relative_q[:, 0])

        if self.screw_spec.mode == "capture":
            alignment_valid = (
                (radial <= self.screw_spec.capture_radial_m)
                & (axis_cos >= np.cos(np.deg2rad(self.screw_spec.capture_tilt_deg)))
                & (yaw.abs() <= np.deg2rad(self.screw_spec.capture_yaw_deg))
            )

            # Body/cap triangle contact is filtered because it fights the
            # analytic helix after engagement.  While the cap is still free,
            # this GPU barrier prevents a wrongly aligned cap from passing
            # downward through the bottle envelope.  Correctly aligned caps
            # may enter the neck corridor and reach the capture window.
            blocked = (
                ~self.screw_engaged
                & ~alignment_valid
                & (radial <= self.free_collision_radius_m)
                & (absolute_axis_offset < self.body_top_offset_m)
            )
            if blocked.any():
                corrected_pos = self.cap.data.root_pos_w + (
                    self.body_top_offset_m - absolute_axis_offset
                )[:, None] * axis_w
                relative_lin = (
                    self.cap.data.root_lin_vel_w
                    - self.object.data.root_lin_vel_w
                )
                inward_speed = (relative_lin * axis_w).sum(dim=1).clamp(max=0.0)
                corrected_lin = (
                    self.cap.data.root_lin_vel_w
                    - inward_speed[:, None] * axis_w
                )
                blocked_pose = torch.cat([corrected_pos, cap_q], dim=1)
                blocked_velocity = torch.cat([
                    corrected_lin, self.cap.data.root_ang_vel_w
                ], dim=1)
                blocked_ids = blocked.nonzero(as_tuple=False).squeeze(1)
                self.cap.write_root_pose_to_sim(
                    blocked_pose[blocked_ids], blocked_ids
                )
                self.cap.write_root_velocity_to_sim(
                    blocked_velocity[blocked_ids], blocked_ids
                )

            capture = (
                ~self.screw_engaged
                & alignment_valid
                & ((axial - self.screw_spec.travel_m).abs()
                   <= self.screw_spec.capture_axial_m)
            )
            self.screw_engaged |= capture
            self.screw_angle[capture] = 2.0 * np.pi * self.screw_spec.turns
            self.screw_has_depth[capture] = False

        relative_ang = self.cap.data.root_ang_vel_w - self.object.data.root_ang_vel_w
        angular_velocity = (relative_ang * axis_w).sum(dim=1).clamp(
            -self.screw_spec.max_angular_velocity_rad_s,
            self.screw_spec.max_angular_velocity_rad_s,
        )
        active = self.screw_engaged
        angle_step = (
            angular_velocity * float(self.cfg.sim.dt)
            if integrate_angle else torch.zeros_like(angular_velocity)
        )
        proposed = self.screw_angle + angle_step
        max_angle = 2.0 * np.pi * self.screw_spec.turns
        self.screw_angle[active] = proposed[active].clamp(0.0, max_angle)
        self.screw_has_depth |= active & (self.screw_angle < max_angle - 0.25 * np.pi)

        detach = (
            active & self.screw_has_depth
            & (proposed >= max_angle) & (angular_velocity > 0.0)
        )
        self.screw_engaged[detach] = False
        active = self.screw_engaged
        if active.any():
            lead = self.screw_spec.direction * self.screw_spec.pitch_m / (2.0 * np.pi)
            angle = self.screw_angle
            target_offset = self.screw_spec.closed_offset_m + lead * angle
            target_pos = self.object.data.root_pos_w + target_offset[:, None] * axis_w
            half = 0.5 * angle
            twist = torch.zeros(self.num_envs, 4, device=self.device)
            twist[:, 0] = torch.cos(half)
            twist[:, 3] = torch.sin(half)
            target_q = quat_mul(body_q, twist)
            pose = torch.cat([target_pos, target_q], dim=1)

            outward = ((self.screw_angle >= max_angle) & (angular_velocity > 0.0))
            inward = ((self.screw_angle <= 0.0) & (angular_velocity < 0.0))
            allowed_omega = torch.where(outward | inward, 0.0, angular_velocity)
            cap_lin = self.object.data.root_lin_vel_w + (
                lead * allowed_omega
            )[:, None] * axis_w
            cap_ang = self.object.data.root_ang_vel_w + allowed_omega[:, None] * axis_w
            velocity = torch.cat([cap_lin, cap_ang], dim=1)
            ids = active.nonzero(as_tuple=False).squeeze(1)
            self.cap.write_root_pose_to_sim(pose[ids], ids)
            self.cap.write_root_velocity_to_sim(velocity[ids], ids)

        if extra_cap_torque_local is not None:
            zeros = torch.zeros_like(extra_cap_torque_local)
            self.cap.set_external_force_and_torque(
                zeros.unsqueeze(1), extra_cap_torque_local.unsqueeze(1),
                body_ids=[0], is_global=False,
            )

    def _apply_action(self):
        super()._apply_action()
        self.apply_screw_constraint()

    def _get_dones(self):
        # DirectRLEnv calls this immediately after the final physics substep
        # and before rewards/observations.  Remove the one-substep integration
        # lag without advancing the internal screw angle again.
        self.apply_screw_constraint(integrate_angle=False)
        return super()._get_dones()
