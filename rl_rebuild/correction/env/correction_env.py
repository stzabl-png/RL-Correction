"""SharpaCorrectionEnv — 参考条件化残差修正环境 (M0: 骨架 + 零残差回放).

动作 (28): [腕Δpos(3), 腕Δ轴角(3), 手指Δq(22)] ∈ [-1,1], 按 Cfg 残差界缩放,
叠加到参考帧上. 腕 = wrench-PD 推浮动根刚体, 手指 = 隐式位置 PD.
M1 再接 obs/reward/dones 的正式实现 (当前为占位).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import (axis_angle_from_quat, quat_apply,
                                 quat_conjugate, quat_mul)

from rl_rebuild.correction import clips
from rl_rebuild.correction import frames as F
from rl_rebuild.correction.env.correction_env_cfg import SharpaCorrectionEnvCfg
from rl_rebuild.correction.env.reward import RewardWeights, compute_reward


def _quat_from_aa(v: torch.Tensor) -> torch.Tensor:
    """轴角向量 (N,3) -> 四元数 wxyz (N,4), 零向量安全."""
    angle = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    axis = v / angle
    half = 0.5 * angle
    return torch.cat([torch.cos(half), torch.sin(half) * axis], dim=-1)


class SharpaCorrectionEnv(DirectRLEnv):
    cfg: SharpaCorrectionEnvCfg

    def __init__(self, cfg: SharpaCorrectionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- 参考数据 -> device 张量 (clips 注册表分派 bi_v2ap / ocir loader) ----
        du = clips.load_data_unit(cfg)
        self.du = du
        r = du.ref
        dev = self.device
        to = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)

        # SDK 序 -> 本 env 的 USD 关节序 (按名字, 一次性)
        assert r.human_finger is not None and r.finger_names is not None, \
            "缺 ref_qpos.npz, 先跑 export_qpos.py"
        assert len(self.hand.joint_names) == 22, f"手关节数 {len(self.hand.joint_names)} != 22"
        perm = [r.finger_names.index(n) for n in self.hand.joint_names]
        self.ref_finger = to(r.human_finger[:, perm])            # (L,22) USD 序
        self.ref_wrist_pos = to(r.track_wrist[:, :3])            # (L,3)
        self.ref_wrist_quat = to(r.track_wrist[:, 3:7])          # (L,4)
        self.ref_obj_pos = to(r.track_object[:, :3])
        self.ref_obj_quat = to(r.track_object[:, 3:7])
        # 物体初始位姿: stable-pose 投影版 (物理稳定), 与参考轨迹解耦
        self.obj_init_pos = to(du.object_init_pose[:3])
        self.obj_init_quat = to(du.object_init_pose[3:7])
        # 物体几何: mesh 表面 FPS 点云 (物体规范系, 观测时转腕系) — 两 variant 共用
        self.obj_points = (to(F.sample_surface_points(du.mesh_path, cfg.n_obj_points))
                           if cfg.n_obj_points > 0 else None)

        self.L = r.L
        self.t0 = int(r.valid_seg[0])                            # M0: 固定从 valid 段起点开始
        # ---- 抓取窗口 (grasp_only 模式): PreGrasp -> GraspPose 合拢完成 ----
        # 起 = interaction_seg[0] = cuRobo close 段起点 = PreGrasp 位姿
        # 止 = grasp.grasp_phase_frame = squeeze 末帧 = 抓握建立完成
        self.grasp_start = int(r.interaction_seg[0])
        self.grasp_end = int(r.grasp.grasp_phase_frame) if r.grasp is not None else self.L - 1
        self.grasp_end = max(self.grasp_end, self.grasp_start + 1)
        if cfg.grasp_only:
            # 时间轴: [静置 | RL抓取窗口 | 硬编码抬升 | hold]
            self.n_grasp = self.grasp_end - self.grasp_start + 1
            self.lift_step0 = cfg.settle_steps + self.n_grasp          # 抬升起始步
            self.hold_step0 = self.lift_step0 + cfg.lift_steps         # hold 起始步
            self.ep_total = self.hold_step0 + cfg.hold_steps
            self.rw_lift_target_override = cfg.lift_height
            print(f"[grasp_only] 抓取窗口 帧{self.grasp_start}->{self.grasp_end} ({self.n_grasp}步) | "
                  f"抬升 {cfg.lift_height*100:.0f}cm/{cfg.lift_steps}步 | hold {cfg.hold_steps}步 | "
                  f"episode {self.ep_total} 步")
        else:
            # 时间轴: [静置前奏 | 参考回放 | hold]
            self.ep_total = cfg.settle_steps + (self.L - self.t0) + cfg.hold_steps
            self.rw_lift_target_override = None

        # 残差缩放 (28,)
        self.res_scale = to(np.concatenate([
            np.full(3, cfg.wrist_pos_residual_max),
            np.full(3, cfg.wrist_rot_residual_max),
            np.full(22, cfg.finger_residual_max)]))

        # 关节软限位
        limits = self.hand.root_physx_view.get_dof_limits().to(dev)
        self.dof_lower, self.dof_upper = limits[..., 0], limits[..., 1]

        masses = self.hand.root_physx_view.get_masses()[0]
        print(f"[debug] hand bodies: {list(zip(self.hand.body_names, masses.tolist()))}")
        print(f"[debug] hand 总质量 {masses.sum():.3f} kg")

        # 摩擦: 手与物体的表面材质由 _setup_scene 绑定的 SuperGrip 统一设定
        # (3.0/3.0/multiply, 复刻 grasp 验证器); 此处不再用 physx view 覆盖.

        # 目标缓存 (每控制步更新, 每物理子步使用)
        self.wrist_tgt_pos = torch.zeros(self.num_envs, 3, device=dev)
        self.wrist_tgt_quat = torch.zeros(self.num_envs, 4, device=dev)
        self.finger_tgt = torch.zeros(self.num_envs, 22, device=dev)

        # ---- reward 相关状态 ----
        self.rw = RewardWeights()                       # 权重集中处; 退火=训练循环改这里
        _lift = clips.clip_entry(cfg.clip_name).get("lift_target")
        if _lift:                                       # 按 clip 覆盖成功/满分抬升高度
            self.rw.lift_target = float(_lift)
        if self.rw_lift_target_override is not None:    # grasp_only: 门槛=硬编码抬升高度
            self.rw.lift_target = self.rw_lift_target_override
            # 抬升期物体本就该离开参考位, 再拿"贴着桌上参考轨迹"打分是反向激励
            self.rw.lam_traj = 0.0
        # ---- affordance (小/扁物体指尖抓取): 逐点热图(物体系) + 奖励开关 ----
        self.aff_points = self.aff_heat = None
        _aff = clips.clip_entry(cfg.clip_name).get("affordance")
        if _aff:
            _a = np.load(_aff, allow_pickle=True)
            self.aff_points = to(_a["points_raw"])          # (P,3) 物体规范系
            self.aff_heat = to(_a["heatmap"])               # (P,)
            self.rw.lam_afford = cfg.lam_afford             # 开 affordance 奖励
            # 逼指尖抓: cage/scoop(无指尖)不再拿 0.3 抬升折扣分
            self.rw.loose_grip_factor = cfg.loose_grip_afford
            print(f"[affordance] {self.aff_points.shape[0]} 点 heatmap∈[{self.aff_heat.min():.2f},"
                  f"{self.aff_heat.max():.2f}] | lam_afford={self.rw.lam_afford} "
                  f"loose_grip={self.rw.loose_grip_factor}")
        # ---- 数据先验 (消融): GraspPose 手指目标(USD序) + cuRobo 预抓取路点 + 抓握段起点 ----
        self.grasp_finger = None                        # (22,) 抓握段手指模仿目标 (纯抓姿)
        g = du.ref.grasp
        if cfg.use_grasp_prior and g is not None and g.finger_names is not None:
            pg = [g.finger_names.index(n) for n in self.hand.joint_names]
            self.grasp_finger = to(g.finger_q[pg])
        self.curobo_wp = None                           # (3,) cuRobo 接近末路点位置
        if cfg.use_curobo_guide and du.ref.curobo_pregrasp is not None:
            self.curobo_wp = to(du.ref.curobo_pregrasp[:3])
            self.rw.lam_curobo = float(cfg.lam_curobo_init)
        # 交互段起帧 (参考帧坐标): GraspPose 手指切换 / cuRobo 引导 相位判断用 (RSI 兼容)
        self.inter0 = int(du.ref.interaction_seg[0])
        self.inter_step = cfg.settle_steps + (self.inter0 - self.t0)
        # cuRobo 引导时间窗结束步: active 段前 curobo_guide_frac 比例 (初始阶段指引, 与交互段解耦
        # —— pp0 无远距 reach, interaction_seg 从帧0起, 故用时间窗而非"接触前")
        self.early_end_step = cfg.settle_steps + int(round(
            cfg.curobo_guide_frac * (self.L - self.t0)))
        self.obj_rest_z = float(du.object_init_pose[2])  # 静置中心高度 (抬升的零点)
        self.tip_ids = [self.hand.body_names.index(n) for n in cfg.fingertip_bodies]
        # 远端指节 4x 权重 (D-Grasp 指尖加权的 qpos 版)
        fw = [4.0 if ("DIP" in n or n.endswith("_IP")) else 1.0 for n in self.hand.joint_names]
        self.finger_weight = to(fw)
        N = self.num_envs
        self.actions_buf = torch.zeros(N, 28, device=dev)
        self.prev_actions = torch.zeros(N, 28, device=dev)
        self.prev_palm_dist = torch.zeros(N, device=dev)   # 势差分 shaping 状态
        self.lift_hw = torch.zeros(N, device=dev)          # 抬升高水位状态
        self.wrench_norm = torch.zeros(N, device=dev)
        self.hold_ok = torch.zeros(N, dtype=torch.long, device=dev)  # hold 段达标步数
        # RSI: 每 env 参考起始帧 (默认 t0); freeze_ctr: 剩余冻结步 (物体钉在参考位)
        self.rsi_start = torch.full((N,), self.t0, dtype=torch.long, device=dev)
        self.freeze_ctr = torch.zeros(N, dtype=torch.long, device=dev)
        self.was_rsi = torch.zeros(N, dtype=torch.bool, device=dev)   # 本回合是否 RSI 起步
        # 参考物体线速度 (帧间差分*fps): 冻结释放时给物体, 避免速度跳变
        _dop = np.diff(r.track_object[:, :3], axis=0, append=r.track_object[-1:, :3])
        self.ref_obj_vel = to(_dop * float(du.fps))                  # (L,3)
        self._ep_sums: dict[str, torch.Tensor] = {}        # 逐项 reward 的 episode 累计
        # critic 特权观测的常量部分 (M2 域随机化后改为每 env 采样值)
        self.obj_mass = self.object.root_physx_view.get_masses().view(N, 1).to(dev)
        self.obj_fric = torch.full((N, 1), du.semantics.friction, device=dev)
        # 本体历史 (HORA 式, stage-2 ProprioAdapt 的输入; PPO 阶段只维护不消费)
        self.proprio_hist = torch.zeros(N, cfg.prop_hist_len, 44, device=dev)

    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        entry = clips.clip_entry(self.cfg.clip_name)
        # ocir 源: 首次把 object.obj 转成物理烘焙 USD (缓存); bi_v2ap 源: usd 已就位
        clips.ensure_object_usd(self.cfg.clip_name)
        self.object = RigidObject(self.cfg.object_cfg)
        if entry["runtime_object_physics"]:
            # bi_v2ap 的 object.usd 是纯视觉网格, 与在线 replay 相同: 运行时贴刚体+碰撞
            import omni.usd
            from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
            stage = omni.usd.get_context().get_stage()
            root = stage.GetPrimAtPath("/World/envs/env_0/Object")
            UsdPhysics.RigidBodyAPI.Apply(root)
            UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(entry["semantics"].mass_kg))
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
            rb.CreateSleepThresholdAttr(0.005)
            rb.CreateStabilizationThresholdAttr(0.0025)
            n_mesh = 0
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdGeom.Mesh):
                    UsdPhysics.CollisionAPI.Apply(prim)
                    UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
                    pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                    pc.CreateContactOffsetAttr(0.002)
                    pc.CreateRestOffsetAttr(0.0)
                    n_mesh += 1
            print(f"[setup] object 刚体化: {n_mesh} mesh, mass={entry['semantics'].mass_kg}kg")
        # 桌子: 静态碰撞体, 桌面 = table_top_z
        sx, sy, sz = self.cfg.table_size
        table_cfg = sim_utils.CuboidCfg(
            size=(sx, sy, sz),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5, dynamic_friction=0.5),  # 物↔桌=3.0×0.5=1.5 (验证器)
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 0.2)),
        )
        table_cfg.func("/World/envs/env_.*/Table", table_cfg,
                       translation=(0.0, 0.0, self.cfg.table_top_z - sz / 2))
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()
        # ---- SuperGrip 摩擦材质: 复刻 grasp 验证器 (isaac_sim/report.json) ----
        # static=dynamic=3.0, combine=multiply => 手↔物 3.0×3.0=9.0, 物↔桌 3.0×0.5=1.5.
        # UsdFileCfg 不支持 physics_material, 故 clone 后显式绑定到每个 env 的
        # Object 与 Robot (bind 带 apply_nested, 自动覆盖全部 collider 子孙; 与验证器
        # 的 strongerThanDescendants 绑定同构). 手↔桌由 filter_collisions 过滤.
        # SuperGrip 只给 5 指尖 elastomer + 物体; 手身(掌/背/指节)用低摩擦.
        # 教训: 整手 9.0 会让手掌一碰物体就刚性粘住拖飞(参考手差几cm没抓上时尤甚).
        # 指尖↔物=3.0×3.0=9.0 (真抓握强握); 手身↔物=0.2×3.0=0.6 (掌蹭不拖飞).
        grip = sim_utils.RigidBodyMaterialCfg(
            static_friction=3.0, dynamic_friction=3.0, restitution=0.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply")
        grip.func("/World/Materials/SuperGrip", grip)
        low = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.2, dynamic_friction=0.2, restitution=0.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply")
        low.func("/World/Materials/LowGrip", low)
        n_grip = n_low = 0
        for p in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Object"):
            sim_utils.bind_physics_material(p, "/World/Materials/SuperGrip"); n_grip += 1
        for p in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Robot"):
            sim_utils.bind_physics_material(p, "/World/Materials/LowGrip",
                                            stronger_than_descendants=False); n_low += 1
        for name in self.cfg.fingertip_bodies:                       # 指尖覆盖回 SuperGrip
            for p in sim_utils.find_matching_prim_paths(f"/World/envs/env_.*/Robot/{name}"):
                sim_utils.bind_physics_material(p, "/World/Materials/SuperGrip"); n_grip += 1
        print(f"[setup] 指尖SuperGrip×{n_grip} 手身LowGrip×{n_low} (指尖↔物=9.0, 掌↔物=0.6)")
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # 指尖接触传感器 (冠军 env 同款模式)
        self._contact_sensors = []
        for i, scfg in enumerate(self.cfg.contact_sensors):
            s = ContactSensor(scfg)
            self._contact_sensors.append(s)
            self.scene.sensors[f"tip_contact_{i}"] = s
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    def _ref_t(self) -> torch.Tensor:
        """当前每 env 的参考帧下标. 从各自 rsi_start 起(冻结窗内夹在起点), hold 夹在末帧."""
        played = (self.episode_length_buf - self.cfg.settle_steps).clamp(min=0)
        cap = self.grasp_end if self.cfg.grasp_only else self.L - 1
        return (self.rsi_start + played).clamp(max=cap)

    def _lift_prog(self) -> torch.Tensor:
        """硬编码抬升进度 (N,) ∈ [0,1]: 抓取窗口内=0, 抬升段线性升到1, hold 段保持1."""
        if not self.cfg.grasp_only:
            return torch.zeros(self.num_envs, device=self.device)
        return ((self.episode_length_buf - self.lift_step0).float()
                / max(self.cfg.lift_steps, 1)).clamp(0.0, 1.0)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions_buf)
        self.actions_buf.copy_(actions.clamp(-1.0, 1.0))               # 归一化动作 (正则用)
        t = self._ref_t()
        # 冻结窗 (kinematic-freeze): 把物体钉在当前参考位姿 (RSI 抬升态=手里, t0=桌上),
        # 手不施加残差, 让手指抓握/物体沉降先稳定. 逐步 decrement, 到 0 释放.
        frozen = self.freeze_ctr > 0
        if frozen.any():
            idx = torch.nonzero(frozen, as_tuple=False).squeeze(-1)
            pose = torch.cat([self.ref_obj_pos[t[idx]] + self.scene.env_origins[idx],
                              self.ref_obj_quat[t[idx]]], dim=1)
            vel = torch.zeros(len(idx), 6, device=self.device)
            vel[:, :3] = self.ref_obj_vel[t[idx]]                      # 给参考速度, 释放不跳变
            self.object.write_root_pose_to_sim(pose, idx)
            self.object.write_root_velocity_to_sim(vel, idx)
            self.freeze_ctr[idx] -= 1
        # 冻结窗内动作不生效 (否则未训练策略会在计分前把物体打飞)
        gate = (~frozen).float().unsqueeze(1)
        a = self.actions_buf * gate * self.res_scale                   # (N,28)
        t = self._ref_t()
        self.wrist_tgt_pos = self.ref_wrist_pos[t] + self.scene.env_origins + a[:, 0:3]
        # 硬编码抬升: 抓取窗口结束后腕目标垂直匀速上升 lift_height.
        # _ref_t 已在 grasp_end 钳住 -> ref_wrist_pos[t] 恒为抓取末姿态, 这里只叠 z 斜坡.
        if self.cfg.grasp_only:
            self.wrist_tgt_pos[:, 2] += self._lift_prog() * self.cfg.lift_height
        self.wrist_tgt_quat = quat_mul(_quat_from_aa(a[:, 3:6]), self.ref_wrist_quat[t])
        self.finger_tgt = (self.ref_finger[t] + a[:, 6:28]).clamp(
            self.dof_lower, self.dof_upper)

    def _apply_action(self) -> None:
        # 手指: 隐式 PD 位置目标
        self.hand.set_joint_position_target(self.finger_tgt)
        # 腕: wrench-PD, 每个物理子步用当前根状态重算 (世界系)
        root_pos = self.hand.data.root_pos_w
        root_quat = self.hand.data.root_quat_w
        lin_vel = self.hand.data.root_lin_vel_w
        ang_vel = self.hand.data.root_ang_vel_w
        pos_err = self.wrist_tgt_pos - root_pos
        q_err = quat_mul(self.wrist_tgt_quat, quat_conjugate(root_quat))
        q_err = q_err * torch.sign(q_err[:, 0:1])   # 双覆盖归正: 保证走短路径 (w>=0)
        rot_err = axis_angle_from_quat(q_err)
        force = self.cfg.wrist_kp_pos * pos_err - self.cfg.wrist_kd_pos * lin_vel
        torque = self.cfg.wrist_kp_rot * rot_err - self.cfg.wrist_kd_rot * ang_vel
        # 安全钳制, 防数值爆炸 (~10 倍手重的推力上限)
        force = force.clamp(-200.0, 200.0)
        torque = torque.clamp(-5.0, 5.0)
        # 世界系 -> 根 link 体坐标系: 旧 API 的 is_global 对力矩不可靠,
        # 统一走文档保证的 local 路径
        q_inv = quat_conjugate(root_quat)
        self.hand.set_external_force_and_torque(
            quat_apply(q_inv, force).unsqueeze(1),
            quat_apply(q_inv, torque).unsqueeze(1),
            body_ids=[0], is_global=False)
        # wrench 用量 (归一到 0~1 量级), 供 reward 正则. copy_ 保持持久 buffer,
        # 避免 inference_mode 下重新绑定产生推理张量 (reset 时就地写会报错)
        self.wrench_norm.copy_(0.5 * (force.norm(dim=1) / 200.0 + torque.norm(dim=1) / 5.0))

    @staticmethod
    def _qsign(q: torch.Tensor) -> torch.Tensor:
        """四元数半球归一 (w>=0), 防 q/-q 双覆盖在观测里跳变."""
        return q * torch.sign(q[:, 0:1] + 1e-12)

    def _get_observations(self) -> dict:
        """actor 144 维 (真机可得) + critic 151 维 (追加仿真特权).
        设计纪律: 无 clip 身份信息; 物体量在腕系/相对量表达; 速度 ×0.2 缩放."""
        cfg, origins = self.cfg, self.scene.env_origins
        t = self._ref_t()
        t5 = (t + cfg.lookahead).clamp(max=self.L - 1)
        vs = cfg.obs_vel_scale

        wrist_pos = self.hand.data.root_pos_w - origins
        wrist_quat = self._qsign(self.hand.data.root_quat_w)
        q_inv = quat_conjugate(wrist_quat)
        obj_pos = self.object.data.root_pos_w - origins
        obj_quat = self.object.data.root_quat_w

        # 物体表面点云 (腕系): 规范系点按当前物体位姿摆好, 再转进腕系, 展平.
        # 给策略物体形状感知 (接触点推理 + 多物体泛化地基). 展平入 MLP (点序固定).
        pc_wrist = None
        if self.obj_points is not None:
            P = self.obj_points.shape[0]
            qp = obj_quat[:, None, :].expand(-1, P, -1).reshape(-1, 4)
            pc = self.obj_points[None].expand(self.num_envs, -1, -1).reshape(-1, 3)
            pw = quat_apply(qp, pc) + obj_pos[:, None, :].expand(-1, P, -1).reshape(-1, 3)
            qi = q_inv[:, None, :].expand(-1, P, -1).reshape(-1, 4)
            wp = wrist_pos[:, None, :].expand(-1, P, -1).reshape(-1, 3)
            pc_wrist = quat_apply(qi, pw - wp).reshape(self.num_envs, P, 3)
        # enable_pointcloud: 点云走独立的 PointNet 分支 (置换不变), 不进展平 obs.
        # 否则退回旧行为 (展平 3N 追加到 obs 末尾) —— 需 observation_space 含 3N.
        if pc_wrist is not None and not cfg.enable_pointcloud:
            obj_geom = pc_wrist.reshape(self.num_envs, -1)
        else:
            obj_geom = torch.zeros(self.num_envs, 0, device=self.device)

        # 参考误差通道 (策略直接看到"该修多少")
        wrist_pos_err = self.ref_wrist_pos[t] - wrist_pos
        q_err = quat_mul(self.ref_wrist_quat[t], quat_conjugate(wrist_quat))
        wrist_rot_err = axis_angle_from_quat(self._qsign(q_err))
        # 相位: 进度标量 + [静置|回放|hold] one-hot (状态性质)
        in_settle = (self.episode_length_buf < cfg.settle_steps).float()
        if cfg.grasp_only:
            # 相位 = [静置 | RL抓取窗口 | 抬升+hold]; progress 用整回合进度 (单调, 简单)
            progress = (self.episode_length_buf.float() / max(self.ep_total, 1)
                        ).clamp(max=1.0).unsqueeze(1)
            in_lift = (self.episode_length_buf >= self.lift_step0).float()
            phase = torch.stack([in_settle, 1.0 - in_settle - in_lift, in_lift], dim=1)
        else:
            played = (self.episode_length_buf - cfg.settle_steps).clamp(min=0).float()
            progress = (played / max(self.L - self.t0, 1)).clamp(max=1.0).unsqueeze(1)
            in_hold = (self.episode_length_buf >= cfg.settle_steps + self.L - self.t0).float()
            phase = torch.stack([in_settle, 1.0 - in_settle - in_hold, in_hold], dim=1)

        obs = torch.cat([
            self.hand.data.joint_pos,                                   # 22
            self.hand.data.joint_vel * vs,                              # 22
            wrist_pos,                                                  # 3
            wrist_quat,                                                 # 4
            self.hand.data.root_lin_vel_w,                              # 3
            self.hand.data.root_ang_vel_w * vs,                         # 3
            quat_apply(q_inv, obj_pos - wrist_pos),                     # 3  物体位置(腕系)
            self._qsign(quat_mul(q_inv, obj_quat)),                     # 4  物体姿态(相对腕)
            self.object.data.root_lin_vel_w,                            # 3
            self.object.data.root_ang_vel_w * vs,                       # 3
            self.ref_finger[t],                                         # 22 参考手指
            wrist_pos_err,                                              # 3
            wrist_rot_err,                                              # 3
            self.ref_obj_pos[t] - obj_pos,                              # 3  物体参考误差
            self.ref_wrist_pos[t5] - self.ref_wrist_pos[t],             # 3  前瞻: 腕将去哪
            self.ref_obj_pos[t5] - self.ref_obj_pos[t],                 # 3  前瞻: 物体该去哪
            progress,                                                   # 1
            phase,                                                      # 3
            self._tip_contacts(),                                       # 5
            self.actions_buf,                                           # 28
            obj_geom,                                                   # 3P 物体几何(腕系)
        ], dim=1).clamp(-cfg.clip_obs, cfg.clip_obs).nan_to_num(0.0)     # NaN 防护: clamp 不吃 NaN

        # 特权通道 (HORA priv_info): 物体质量/摩擦真值 + 指尖接触力模长.
        # v3net: critic 吃 (obs+priv 原始), actor 只拿 priv 的 8 维嵌入 z
        # (stage-2 ProprioAdapt 会用本体历史学着估计 z, 消除部署依赖)
        tip_f = torch.cat([s.data.force_matrix_w.view(self.num_envs, 1, 3)
                           for s in self._contact_sensors], dim=1).norm(dim=-1)  # (N,5)
        priv = torch.cat([self.obj_mass, self.obj_fric, tip_f], dim=1
                         ).clamp(-cfg.clip_obs, cfg.clip_obs).nan_to_num(0.0)
        # 本体历史滚动更新 (就地写, 不重绑 buffer — inference_mode 安全)
        self.proprio_hist[:, :-1].copy_(self.proprio_hist[:, 1:].clone())
        self.proprio_hist[:, -1] = torch.cat(
            [self.hand.data.joint_pos, self.hand.data.joint_vel * vs], dim=1)
        out = {"policy": obs, "priv_info": priv, "proprio_hist": self.proprio_hist}
        if cfg.enable_pointcloud and pc_wrist is not None:
            out["pointcloud"] = pc_wrist.clamp(-cfg.clip_obs, cfg.clip_obs).nan_to_num(0.0)
        return out

    # ---- reward 输入采集 ----
    def _palm_pos(self) -> torch.Tensor:
        """掌心位置 (N,3, 含 env_origin): 腕根 + 手系 +z(指向) 9cm 固定偏移.
        不用指尖质心 — 手指开合会移动质心, 污染 approach 的距离信号."""
        offset = torch.tensor([0.0, 0.0, 0.09], device=self.device).expand(self.num_envs, 3)
        return self.hand.data.root_pos_w + quat_apply(self.hand.data.root_quat_w, offset)

    def _tip_contacts(self) -> torch.Tensor:
        """(N,5) 指尖-物体接触力是否超阈 (传感器已 filter 到 Object)."""
        f = torch.cat([s.data.force_matrix_w.view(self.num_envs, 1, 3)
                       for s in self._contact_sensors], dim=1)          # (N,5,3)
        return (f.norm(dim=-1) > self.cfg.contact_force_thresh).float()

    def _tip_affordance(self) -> torch.Tensor | None:
        """(N,5) 每个指尖最近 affordance 点的 heatmap 值 [0,1]. 无 affordance 返回 None.
        指尖世界位 -> 物体规范系(用当前物体位姿) -> 最近点 heatmap. 逼指尖落到接触区/环."""
        if self.aff_points is None:
            return None
        tips = self.hand.data.body_pos_w[:, self.tip_ids]                # (N,5,3) 世界
        op = self.object.data.root_pos_w                                # (N,3)
        oq = self.object.data.root_quat_w                               # (N,4)
        rel = tips - op[:, None]                                        # (N,5,3)
        qi = quat_conjugate(oq)[:, None].expand(-1, 5, -1).reshape(-1, 4)
        tip_obj = quat_apply(qi, rel.reshape(-1, 3)).reshape(self.num_envs, 5, 3)
        d = torch.cdist(tip_obj, self.aff_points[None].expand(self.num_envs, -1, -1))  # (N,5,P)
        return self.aff_heat[d.argmin(dim=-1)]                          # (N,5)

    def _get_rewards(self) -> torch.Tensor:
        t = self._ref_t()
        origins = self.scene.env_origins
        obj_pos = self.object.data.root_pos_w - origins
        palm_pos = self._palm_pos() - origins
        contacts = self._tip_contacts()
        active = self.episode_length_buf >= self.cfg.settle_steps

        # 刚转入 active 的 env: 用当前距离初始化势差分状态 (避免一步假 approach 分)
        newly = active & (self.episode_length_buf == self.cfg.settle_steps)
        if newly.any():
            self.prev_palm_dist[newly] = (palm_pos - obj_pos)[newly].norm(dim=1)

        # 手指模仿目标: 抓握段(参考帧≥交互起)换成 GraspPose 纯抓姿, 否则重建手指 (RSI 兼容)
        ref_fin = self.ref_finger[t]
        if self.grasp_finger is not None:
            in_grasp = (t >= self.inter0).unsqueeze(1)
            ref_fin = torch.where(in_grasp, self.grasp_finger.unsqueeze(0), ref_fin)
        # cuRobo 预抓取路点引导: 仅接近段 (参考帧<交互起)
        curobo_wp = in_approach = None
        if self.curobo_wp is not None:
            curobo_wp = self.curobo_wp.expand(self.num_envs, 3)
            in_approach = active & (t < self.inter0)

        # grasp_only: 模仿项的腕参考要含硬编码抬升, 否则"提上去"这件事本身会被 imit 扣分
        ref_wrist = self.ref_wrist_pos[t]
        if self.cfg.grasp_only:
            ref_wrist = ref_wrist.clone()
            ref_wrist[:, 2] = ref_wrist[:, 2] + self._lift_prog() * self.cfg.lift_height
        total, terms = compute_reward(
            obj_pos=obj_pos,
            obj_linvel=self.object.data.root_lin_vel_w,
            obj_rest_z=self.obj_rest_z,
            palm_pos=palm_pos,
            contacts=contacts,
            finger_q=self.hand.data.joint_pos,
            wrist_pos=self.hand.data.root_pos_w - origins,
            ref_obj_pos=self.ref_obj_pos[t],
            ref_finger_q=ref_fin,
            ref_wrist_pos=ref_wrist,
            action=self.actions_buf,
            prev_action=self.prev_actions,
            wrench_norm=self.wrench_norm,
            active=active,
            prev_palm_dist=self.prev_palm_dist,
            lift_hw=self.lift_hw,
            w=self.rw,
            finger_weight=self.finger_weight,
            curobo_wp=curobo_wp,
            in_approach=in_approach,
            tip_afford=self._tip_affordance(),
        )
        # 逐项 episode 累计 (reward hacking 账本)
        for k, v in terms.items():
            if k not in self._ep_sums:
                self._ep_sums[k] = torch.zeros(self.num_envs, device=self.device)
            self._ep_sums[k] += v
        # 成功判定素材: hold 段内 (参考帧到末尾, 抬≥目标 且 ≥2 指接触) 计数 (RSI 兼容)
        if self.cfg.grasp_only:
            in_hold = self.episode_length_buf >= self.hold_step0   # 硬编码抬升完成之后才判
        else:
            in_hold = self._ref_t() >= (self.L - 1)
        _lift_thr = (self.cfg.success_lift_height if self.cfg.grasp_only
                     else self.rw.lift_target)
        lifted = (obj_pos[:, 2] - self.obj_rest_z) >= _lift_thr
        held = contacts.sum(dim=1) >= self.rw.min_contacts
        self.hold_ok += (in_hold & lifted & held).long()
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._ref_t()
        origins = self.scene.env_origins
        obj_pos = self.object.data.root_pos_w - origins
        wrist_pos = self.hand.data.root_pos_w - origins
        fell = obj_pos[:, 2] < self.cfg.table_top_z - self.cfg.term_obj_fall
        # 发散只看水平面: 垂直超前抬升不是发散 (pp0_anchor 教训: 参考物体 carry 段才慢慢
        # 升, 策略提前举到位反被 3D 距离判死 — "因成功而终止"). 下坠由 fell 兜底.
        obj_div = (obj_pos - self.ref_obj_pos[t])[:, :2].norm(dim=1) > self.cfg.term_obj_div
        wrist_div = (wrist_pos - self.ref_wrist_pos[t]).norm(dim=1) > self.cfg.term_wrist_div
        terminated = fell | obj_div | wrist_div
        truncated = self.episode_length_buf >= self.ep_total - 1
        # 终止原因计数 (日志)
        self._last_fell, self._last_obj_div, self._last_wrist_div = fell, obj_div, wrist_div
        return terminated, truncated

    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # episode 长度必须在 super() 清零 episode_length_buf 之前取
        ep_len = self.episode_length_buf[env_ids].clamp(min=1).float()
        super()._reset_idx(env_ids)
        n = len(env_ids)
        origins = self.scene.env_origins[env_ids]

        # ---- RSI: 逐 env 采样参考起始帧 (偏重接触+抬升段), 置满冻结窗 ----
        if self.cfg.grasp_only:
            # 恒从 PreGrasp 起步; 物体留在桌面稳定位 (use_rsi 全 False 走 obj_init_pose 分支)
            starts = torch.full((n,), self.grasp_start, dtype=torch.long, device=self.device)
            use_rsi = torch.zeros(n, dtype=torch.bool, device=self.device)
        else:
            inter0 = max(int(self.du.ref.interaction_seg[0]), self.t0)
            use_rsi = torch.rand(n, device=self.device) < self.cfg.rsi_prob
            rsi = torch.randint(inter0, self.L, (n,), device=self.device)
            starts = torch.where(use_rsi, rsi,
                                 torch.full((n,), self.t0, dtype=torch.long, device=self.device))
        self.rsi_start[env_ids] = starts
        self.freeze_ctr[env_ids] = self.cfg.settle_steps

        # 手根/手指 -> 参考 starts 帧 (逐 env)
        root = torch.zeros(n, 13, device=self.device)
        root[:, 0:3] = self.ref_wrist_pos[starts] + origins
        root[:, 3:7] = self.ref_wrist_quat[starts]
        self.hand.write_root_state_to_sim(root, env_ids)
        dof = self.ref_finger[starts].clone()
        self.hand.write_joint_state_to_sim(dof, torch.zeros_like(dof), env_ids=env_ids)
        self.hand.set_joint_position_target(dof, env_ids=env_ids)
        # 物体: RSI env 用参考物体位姿 (抬升态=在手里, 靠冻结窗撑住); t0 env 用贴底稳定初始位
        um = use_rsi.unsqueeze(1)
        obj_p = torch.where(um, self.ref_obj_pos[starts], self.obj_init_pos.expand(n, -1))
        obj_q = torch.where(um, self.ref_obj_quat[starts], self.obj_init_quat.expand(n, -1))
        obj = torch.zeros(n, 13, device=self.device)
        obj[:, 0:3] = obj_p + origins
        obj[:, 3:7] = obj_q
        self.object.write_root_pose_to_sim(obj[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj[:, 7:], env_ids)
        # 目标缓存同步到起始帧 (reset 后第一个物理步就有合法目标)
        self.wrist_tgt_pos[env_ids] = self.ref_wrist_pos[starts] + origins
        self.wrist_tgt_quat[env_ids] = self.ref_wrist_quat[starts]
        self.finger_tgt[env_ids] = self.ref_finger[starts]

        # ---- episode 结算日志 ----
        success = (self.hold_ok[env_ids].float()
                   >= self.cfg.success_hold_frac * self.cfg.hold_steps).float()
        log = {"success_rate": success.mean().item()}
        # 从头(t0)成功率单独统计 —— 总成功率被 RSI 抬高, t0 才是真实"从头抓"能力
        _t0m = ~self.was_rsi[env_ids]
        if _t0m.any():
            log["success_rate_t0"] = success[_t0m].mean().item()
        if self.was_rsi[env_ids].any():
            log["success_rate_rsi"] = success[self.was_rsi[env_ids]].mean().item()
        for k, v in self._ep_sums.items():
            log[f"ep_rew/{k}"] = (v[env_ids] / ep_len).mean().item()
            v[env_ids] = 0.0
        if hasattr(self, "_last_fell"):
            log["term/fell"] = self._last_fell[env_ids].float().mean().item()
            log["term/obj_div"] = self._last_obj_div[env_ids].float().mean().item()
            log["term/wrist_div"] = self._last_wrist_div[env_ids].float().mean().item()
        # 这套 PPO 只转发 extras 顶层标量到 TensorBoard (ppo.py:398), 必须平铺
        self.extras.update(log)
        self.extras["log"] = log

        # ---- reward 跨步状态清零 ----
        self.was_rsi[env_ids] = use_rsi                  # 更新(在上面 t0/rsi 成功率统计之后)
        self.proprio_hist[env_ids] = 0.0
        self.hold_ok[env_ids] = 0
        self.lift_hw[env_ids] = 0.0
        self.actions_buf[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.wrench_norm[env_ids] = 0.0
        # 势差分状态: 用参考起始腕位到物体位的距离近似 (首个 active 步会精确重置)
        self.prev_palm_dist[env_ids] = (self.ref_wrist_pos[starts] - obj_p).norm(dim=1)
