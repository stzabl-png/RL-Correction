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
            if self.screw_spec.breakaway_torque_nm is not None:
                # U40 真实螺纹副的状态 (惯量缓冲在首次约束调用时惰性初始化,
                # 那时 physx view 才可用).
                N, dev = self.num_envs, self.device
                self.screw_omega = torch.zeros(N, device=dev)
                self.screw_tau_ema = torch.zeros(N, device=dev)
                self.screw_locked = torch.ones(N, dtype=torch.bool, device=dev)
                self._screw_capw_vec = torch.zeros(N, 3, device=dev)
                self._screw_unlock_dwell = torch.zeros(N, device=dev)
                self._bottle_react_f = torch.zeros(N, 1, 3, device=dev)
                self._bottle_react_t = torch.zeros(N, 1, 3, device=dev)
                self._cap_heavy_state = None

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
            if self.screw_spec.breakaway_torque_nm is not None:
                self.screw_omega[ids] = 0.0
                self.screw_tau_ema[ids] = 0.0
                self.screw_locked[ids] = True
                self._screw_capw_vec[ids] = 0.0
                self._screw_unlock_dwell[ids] = 0.0

    def apply_screw_constraint(
            self, extra_cap_torque_local=None, *, integrate_angle: bool = True,
            drive_mask=None, omega_damping=None):
        """Capture and enforce the helical relation with GPU tensor writes.

        ``integrate_angle=False`` performs the final projection after the last
        physics substep without advancing the screw coordinate a second time.
        IsaacLab exposes no dedicated post-substep callback, so this keeps the
        state consumed by rewards and observations exactly on the helix.

        ``drive_mask`` (N,) bool **or float in [0,1]**: thread-stiction
        abstraction for training.  It is applied multiplicatively, so a float
        gain grades the drive speed (see ``screw_triad_drive``: rate scales
        with how many of thumb/index/middle actually touch the cap).
        tasks.  Where False, the cap's relative angular velocity is treated as
        zero — it neither advances the screw coordinate nor survives the
        constraint's velocity write.  A real PCO thread is self-locking under
        static friction; without this, one glancing sub-threshold touch spins
        the frictionless analytic coordinate through both turns for free
        (measured: 41% spurious "release" at near-random policy).  ``None``
        keeps the original free-coordinate behaviour for the physics audits.
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
        if self.screw_spec.breakaway_torque_nm is not None:
            # U40 真实螺纹副: drive_mask / omega_damping 是假摩擦替身, 在此
            # 模式下全部退役, 由静锁 + 库仑 + 粘滞取代 (见 LEDGER U40).
            angular_velocity = self._thread_friction_step(
                relative_ang, axis_w, integrate_angle)
        else:
            angular_velocity = (relative_ang * axis_w).sum(dim=1).clamp(
                -self.screw_spec.max_angular_velocity_rad_s,
                self.screw_spec.max_angular_velocity_rad_s,
            )
            if drive_mask is not None:
                angular_velocity = angular_velocity * drive_mask.float()
            if omega_damping is not None:
                # 螺纹粘滞摩擦抽象 (训练任务用): 相对角速度每子步衰减, 轻弹的惯性
                # 立刻消散, 只有持续接触驱动才能维持转动. None = 审计原行为.
                angular_velocity = angular_velocity * float(omega_damping)
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

        if self.screw_spec.breakaway_torque_nm is not None:
            self._thread_friction_finish(axis_w, angular_velocity, max_angle)

        if extra_cap_torque_local is not None:
            zeros = torch.zeros_like(extra_cap_torque_local)
            self.cap.set_external_force_and_torque(
                zeros.unsqueeze(1), extra_cap_torque_local.unsqueeze(1),
                body_ids=[0], is_global=False,
            )

    def _thread_friction_step(self, relative_ang, axis_w, integrate_angle):
        """U40 真实螺纹副: 重惯量螺旋自由度上的静锁/库仑/粘滞.

        咬合期盖绕螺轴惯量被调成 ``inertia_eff_kgm2`` (见 finish 的惯量
        切换), 指尖→盖的可传扭矩因此受 PhysX 摩擦锥真实限制 (≤ μ·N·r,
        捏得紧才传得多, 打滑/粘着由求解器裁决). 本函数只负责螺纹阻力:

        - 静锁: 上一子步写入 ω 后, 物理子步里接触实际注入的角冲量给出
          净轴向力矩估计 τ = I_eff·Δω/dt (EMA 抗单子步碰撞尖峰);
          |τ| 超过 breakaway 才解锁 —— 轻拍/轻擦永远不解锁.
        - 动阶段: 物理携带 ω, 每子步扣除 (τ_k + b·|ω|)·dt/I_eff 的阻力;
          停止驱动 → ω 衰减, |ω| 低于 lock_omega_eps 且 τ 低于阈值时回锁
          (换把重捏 = 重新破静摩擦, 与真螺纹一致).
        """
        spec = self.screw_spec
        if not integrate_angle:
            return self.screw_omega          # 子步末投影调用: 不重复推进状态
        dt = float(self.cfg.sim.dt)
        # U40b/U40c: 力矩估计只看**盖侧**且按**矢量**差分 —— 盖的角速度矢量
        # 实测 − 上一子步实际写给盖的矢量, 再投影到当前螺轴. 只有真实作用在
        # 盖上的接触冲量能改这个差分:
        # - 相对 Δω (U40 原版) 会把瓶身加速误读成扭矩 (4.7M 尸检: 0 接触
        #   61-76 mN·m, 100% 白解锁);
        # - 轴向标量差分 (U40b) 会把螺轴变向漏进来 —— 左手持瓶晃动时轴每
        #   子步变向, 同一矢量投影到新轴差出 I_eff·Ω⊥² (实测 68 mN·m).
        # 矢量差分下无接触时严格为零 (盖轴对称, 无扭矩则角速度矢量保持).
        #
        # U44 (2026-08-31, 用户质询"手没摩擦盖为何转"坐实): 以上只修了**力矩
        # 估计**, **转动积分**当年原样用着有毒的 `relative_ang` (盖ω−瓶ω), 且
        # 真值门管不到它. 盖是自由刚体、约束靠每子步"写回"事后施加, 子步内不
        # 随瓶加速 => 读到的相对 ω 混入 −Δ瓶ω, 被逐步累加成螺纹转速.
        # **量级更正 (U44b)**: 瓶抖 ±0.5 rad/s 是每**控制步**(12 子步)的幅度,
        # 摊到子步约 ±0.04 rad/s, 与接触注入的 dw 同量级而非压倒性 —— 本条是
        # 真实泄漏但不是"盖自己猛转"的主因 (主因 = 重惯量飞轮下的戳转策略,
        # 见 LEDGER U44b), 修前修后行为相近, 已有成绩不因本条失效.
        # 修法: 转动与力矩同源 —— 都用"接触注入的角速度增量" dw:
        #   dw = (盖ω − 上一子步写入矢量)·当前轴     (瓶的加速自动抵消, 因为
        #        盖同样没跟随它; 写入矢量已含写时瓶ω)
        #   ω_phys = 上一子步写入的螺纹转速 + dw
        # 于是 τ = I_eff·dw/dt 与 ω 严格同一口径, 同上接触真值门.
        cap_tau = 10.0 * spec.breakaway_torque_nm
        dw_max = cap_tau * dt / spec.inertia_eff_kgm2
        dw = ((self.cap.data.root_ang_vel_w - self._screw_capw_vec)
              * axis_w).sum(dim=1).clamp(-dw_max, dw_max)
        # U40d 真值门: 指尖对盖零接触 = 物理上没有外力矩, 增量强制归零.
        # 封死一切残余/未知的数值泄漏通道. 探针的外加扭矩审计置
        # _thread_tau_contact_gate=False 走旁路.
        if (getattr(self, "_thread_tau_contact_gate", True)
                and hasattr(self, "_cap_contacts")):
            dw = dw * (self._cap_contacts().sum(dim=1) > 0).float()
        tau_in = spec.inertia_eff_kgm2 * dw / dt
        omega_phys = self.screw_omega + dw
        alpha = dt / max(spec.torque_ema_s, dt)
        self.screw_tau_ema += alpha * (tau_in - self.screw_tau_ema)
        # 解锁 = EMA 持续超阈 unlock_dwell_s (连续子步计数), 冲击自动清零.
        above = self.screw_tau_ema.abs() > spec.breakaway_torque_nm
        self._screw_unlock_dwell = (self._screw_unlock_dwell + dt) * above.float()
        unlock = self.screw_locked & (
            self._screw_unlock_dwell >= spec.unlock_dwell_s)
        self.screw_locked = self.screw_locked & ~unlock
        drag = (spec.kinetic_torque_nm + spec.viscous_nms
                * omega_phys.abs()) * dt / spec.inertia_eff_kgm2
        omega = torch.where(
            self.screw_locked, torch.zeros_like(omega_phys),
            torch.sign(omega_phys) * (omega_phys.abs() - drag).clamp(min=0.0))
        relock = (~self.screw_locked
                  & (omega.abs() < spec.lock_omega_eps)
                  & (self.screw_tau_ema.abs() < spec.breakaway_torque_nm))
        self.screw_locked = self.screw_locked | relock
        omega = torch.where(self.screw_locked, torch.zeros_like(omega), omega)
        self.screw_omega = omega.clamp(-spec.max_angular_velocity_rad_s,
                                       spec.max_angular_velocity_rad_s)
        return self.screw_omega

    def _thread_friction_finish(self, axis_w, angular_velocity, max_angle):
        """写回后的收尾: 记录实写 ω / 咬合边界切惯量 / 反作用扭矩回瓶身."""
        spec = self.screw_spec
        outward = (self.screw_angle >= max_angle) & (angular_velocity > 0.0)
        inward = (self.screw_angle <= 0.0) & (angular_velocity < 0.0)
        written = torch.where(
            self.screw_engaged & ~(outward | inward), angular_velocity,
            torch.zeros_like(angular_velocity))
        self.screw_omega = written
        # U40c: 记录实际写给盖的角速度**矢量** (= 瓶 ω 矢量 + DOF ω·轴).
        # 脱扣 env 没有写入, 记当前实测矢量使 Δ≈0 (估计器对它们本就不使用).
        written_vec = (self.object.data.root_ang_vel_w
                       + written[:, None] * axis_w)
        self._screw_capw_vec = torch.where(
            self.screw_engaged[:, None], written_vec,
            self.cap.data.root_ang_vel_w)

        # 咬合 ↔ 脱扣边界: 盖绕螺轴惯量在 I_eff 与实物之间切换 (脱扣后的
        # 自由盖必须还原实物惯量, 否则抓放手感全错).
        if self._cap_heavy_state is None:
            orig = self.cap.root_physx_view.get_inertias().clone()
            heavy = orig.clone()
            # U40d: 三主轴全部加重 (球形惯量). 只加重 Izz 时 Izz/Ixx ~ 1e4,
            # 瓶晃引入横向 ω 后欧拉陀螺项把角速度矢量搅出真实变化, 矢量差分
            # 也读出幻影扭矩 (第三通道). 球形惯量下无扭矩 → ω 矢量严格守恒;
            # 横向转动自由度本就被螺旋投影每子步覆写, 无副作用.
            heavy[:, 0] = spec.inertia_eff_kgm2
            heavy[:, 4] = spec.inertia_eff_kgm2
            heavy[:, 8] = spec.inertia_eff_kgm2
            self._cap_inertia_orig = orig
            self._cap_inertia_heavy = heavy
            self._cap_heavy_state = torch.zeros(
                orig.shape[0], dtype=torch.bool, device=orig.device)
        want = self.screw_engaged.to(self._cap_heavy_state.device)
        flip = want != self._cap_heavy_state
        if flip.any():
            data = torch.where(want.unsqueeze(1),
                               self._cap_inertia_heavy, self._cap_inertia_orig)
            self.cap.root_physx_view.set_inertias(
                data, flip.nonzero(as_tuple=False).squeeze(1))
            self._cap_heavy_state = want

        if spec.react_on_bottle and not getattr(self.cfg, "clamp_body", False):
            # 螺纹反作用扭矩回瓶身: 锁定 = 指尖扭矩经锁死螺纹透传 (≤breakaway),
            # 转动 = 库仑 + 粘滞阻力矩的反作用. 左手持瓶必须抗住这份扭.
            # clamp_body 环境 (v1/探针) 跳过: 瓶身每控制步才被钉一次, 反作用
            # 会在子步间把轻瓶旋起来, 污染相对角速度测量 (探针 B 段实测 1.6×).
            tau = torch.where(
                self.screw_locked,
                self.screw_tau_ema.clamp(-spec.breakaway_torque_nm,
                                         spec.breakaway_torque_nm),
                torch.sign(self.screw_omega)
                * (spec.kinetic_torque_nm
                   + spec.viscous_nms * self.screw_omega.abs()))
            tau = torch.where(self.screw_engaged, tau, torch.zeros_like(tau))
            self._bottle_react_t[:, 0, :] = tau.unsqueeze(1) * axis_w
            self.object.set_external_force_and_torque(
                self._bottle_react_f, self._bottle_react_t,
                body_ids=[0], is_global=True)

    def _apply_action(self):
        super()._apply_action()
        self.apply_screw_constraint()

    def _get_dones(self):
        # DirectRLEnv calls this immediately after the final physics substep
        # and before rewards/observations.  Remove the one-substep integration
        # lag without advancing the internal screw angle again.
        self.apply_screw_constraint(integrate_angle=False)
        return super()._get_dones()
