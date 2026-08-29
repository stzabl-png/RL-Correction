"""SharpaCorrectionEnv — 参考条件化残差修正环境 (M0: 骨架 + 零残差回放).

动作 (28): [腕Δpos(3), 腕Δ轴角(3), 手指Δq(22)] ∈ [-1,1], 按 Cfg 残差界缩放,
叠加到参考帧上. 腕 = wrench-PD 推浮动根刚体, 手指 = 隐式位置 PD.
M1 再接 obs/reward/dones 的正式实现 (当前为占位).
"""
from __future__ import annotations

import os

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
from rl_rebuild.correction.env.diagnostics import EpisodeDiagnostics
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
        # 手指/手臂在 self.hand 关节表里的下标. 飞手 = 22 个关节全是手指、没有臂;
        # DexMate 子类会覆盖成"整机 67 关节里挑出 22 手指 + 7 臂".
        self._resolve_joint_ids()

        # ---- 参考数据 -> device 张量 (clips 注册表按 source 分派 loader) ----
        du = clips.load_data_unit(cfg)
        self.du = du
        r = du.ref
        dev = self.device
        to = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)

        # SDK 序 -> 本 env 的 USD 关节序 (按名字, 一次性)
        assert r.human_finger is not None and r.finger_names is not None, \
            "缺 ref_qpos.npz, 先跑 export_qpos.py"
        assert len(self.hand_joint_names) == 22, f"手关节数 {len(self.hand_joint_names)} != 22"
        perm = [r.finger_names.index(n) for n in self.hand_joint_names]
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
        # approach / 接触触发合拢 (env 变量覆盖, 默认沿用旧行为)
        import os as _os
        self.approach_steps = int(_os.environ.get("GRASP_APPROACH", cfg.approach_steps))
        self.approach_steps = int(np.clip(self.approach_steps, 0, max(self.grasp_start - 1, 0)))
        self.contact_close = _os.environ.get("GRASP_CONTACT_CLOSE",
                                             "1" if cfg.contact_close else "0") == "1"
        self.start_jitter = float(_os.environ.get("GRASP_START_JITTER", cfg.start_jitter))
        if cfg.grasp_only:
            # 时间轴: [静置 | approach | RL抓取窗口 | 硬编码抬升 | hold]
            self.n_grasp = self.grasp_end - self.grasp_start + 1
            # approach 段的参考帧 = grasp_start-approach_steps .. grasp_start, 播放长度并入
            self.lift_step0 = cfg.settle_steps + self.approach_steps + self.n_grasp
            self.hold_step0 = self.lift_step0 + cfg.lift_steps         # hold 起始步
            self.ep_total = self.hold_step0 + cfg.hold_steps
            self.rw_lift_target_override = cfg.lift_height
            print(f"[grasp_only] approach {self.approach_steps}步 | 抓取窗口 帧{self.grasp_start}->{self.grasp_end}"
                  f" ({self.n_grasp}步) | 抬升 {cfg.lift_height*100:.0f}cm/{cfg.lift_steps}步 | "
                  f"hold {cfg.hold_steps}步 | episode {self.ep_total} 步 | "
                  f"合拢={'接触触发' if self.contact_close else '按帧'}")
        else:
            # 时间轴: [静置前奏 | 参考回放 | hold]
            self.ep_total = cfg.settle_steps + (self.L - self.t0) + cfg.hold_steps
            self.rw_lift_target_override = None

        # 残差缩放 (飞手 28 = 腕pos3+腕rot3+指22; DexMate 子类覆盖成 29 = 臂7+指22)
        self.res_scale = self._build_res_scale(to)

        # 手指关节软限位 (只取手指那 22 个, 与 finger_tgt 同序)
        limits = self.hand.root_physx_view.get_dof_limits().to(dev)
        self.dof_lower = limits[..., 0][:, self.hand_jids]
        self.dof_upper = limits[..., 1][:, self.hand_jids]

        masses = self.hand.root_physx_view.get_masses()[0]
        print(f"[debug] hand bodies: {list(zip(self.hand.body_names, masses.tolist()))}")
        print(f"[debug] hand 总质量 {masses.sum():.3f} kg")

        # 摩擦: _setup_scene 按 clip 来源绑定. static reconstruction 使用语义摩擦;
        # 其余已验证 clip 沿用 SuperGrip. 此处不再用 physx view 覆盖.

        # 目标缓存 (每控制步更新, 每物理子步使用)
        self.wrist_tgt_pos = torch.zeros(self.num_envs, 3, device=dev)
        self.wrist_tgt_quat = torch.zeros(self.num_envs, 4, device=dev)
        self.finger_tgt = torch.zeros(self.num_envs, 22, device=dev)

        # ---- reward 相关状态 ----
        self.rw = RewardWeights()                       # 权重集中处; 退火=训练循环改这里
        # "跟随"类奖励的处置 (无 GraspPose 时关掉手指模仿 / 接触后的逐帧物体跟随)
        self.rw.imit_finger_w = float(cfg.imit_finger_w)
        self.rw.traj_after_contact = float(cfg.traj_after_contact)
        self.rw.lam_goal = float(cfg.lam_goal)
        self.rw.sigma_goal = float(cfg.sigma_goal)
        self.rw.w_grip = float(cfg.w_grip)
        self.rw.grip_deadband = float(cfg.grip_deadband)
        # 终点势只对"物体要被搬到某处"的完整轨迹任务有意义; grasp_only 的目标是抬起来
        self.use_goal = cfg.lam_goal > 0.0 and not cfg.grasp_only
        if cfg.lam_goal > 0.0 and cfg.grasp_only:
            print("[reward] grasp_only 模式忽略终点势 (目标是抬升, 不是放置)")
        if cfg.imit_finger_w == 0.0 and not cfg.use_grasp_prior:
            print("[reward] 手指模仿通道已关 (无 GraspPose, 参考手指不可信);"
                  " 抓法由 contact/afford/lift 三项决定")
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
            pg = [g.finger_names.index(n) for n in self.hand_joint_names]
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
        fw = [4.0 if ("DIP" in n or n.endswith("_IP")) else 1.0 for n in self.hand_joint_names]
        self.finger_weight = to(fw)
        N = self.num_envs
        self.actions_buf = torch.zeros(N, cfg.action_space, device=dev)
        self.prev_actions = torch.zeros(N, cfg.action_space, device=dev)
        # 累积手指残差 (accumulate_finger 时): 跨步状态, 必须进观测 + reset 清零
        self.finger_res = torch.zeros(N, 22, device=dev)
        if cfg.accumulate_finger:
            print(f"[finger] 累积残差: rate ±{np.degrees(cfg.finger_rate_max):.1f}°/步"
                  f" | total ±{np.degrees(cfg.finger_total_max):.1f}°"
                  f" | decay {cfg.finger_res_decay}")
        self.prev_palm_dist = torch.zeros(N, device=dev)   # 势差分 shaping 状态
        self.lift_hw = torch.zeros(N, device=dev)          # 抬升高水位状态
        self.wrench_norm = torch.zeros(N, device=dev)
        self.hold_ok = torch.zeros(N, dtype=torch.long, device=dev)  # hold 段达标步数
        # 接触触发合拢: 张开/合拢构型(USD序, 从 ref_finger 斜坡两端取) + 逐env合拢进度
        self.open_pose = to(self.ref_finger[0].cpu().numpy())        # 帧0 = 张开
        self.closed_pose = to(self.ref_finger[self.grasp_end].cpu().numpy())  # grasp_end = 合拢到位
        self.close_prog = torch.zeros(N, device=dev)                 # 合拢进度 ∈[0,1]
        self.close_started = torch.zeros(N, dtype=torch.bool, device=dev)
        self.wrist_jitter = torch.zeros(N, 3, device=dev)            # 每回合起点腕位随机偏移(鲁棒性)
        # RSI: 每 env 参考起始帧 (默认 t0); freeze_ctr: 剩余冻结步 (物体钉在参考位)
        self.rsi_start = torch.full((N,), self.t0, dtype=torch.long, device=dev)
        self.freeze_ctr = torch.zeros(N, dtype=torch.long, device=dev)
        self.was_rsi = torch.zeros(N, dtype=torch.bool, device=dev)   # 本回合是否 RSI 起步
        # 参考物体线速度 (帧间差分*fps): 冻结释放时给物体, 避免速度跳变
        _dop = np.diff(r.track_object[:, :3], axis=0, append=r.track_object[-1:, :3])
        self.ref_obj_vel = to(_dop * float(du.fps))                  # (L,3)
        self._ep_sums: dict[str, torch.Tensor] = {}        # 逐项 reward 的 episode 累计
        # 诊断仪表 (滚动窗口, 可信分母). 每条量都对应一条**可证伪的设计假设**,
        # 对照表见 docs/DESIGN_LOOP.md —— 曲线怎么长对应哪个设计错了, 事先写死.
        self.diag = EpisodeDiagnostics(N, dev, window=cfg.diag_window)
        self.obj_p0 = torch.zeros(N, 3, device=dev)        # 回合起始物体位 (扰动量基线)
        self.grasped_once = torch.zeros(N, dtype=torch.bool, device=dev)
        # ---- 终点势的跨步状态 ----
        self.prev_goal_dist = torch.zeros(N, device=dev)   # 上一步物体到终点距离
        self.contact_run = torch.zeros(N, dtype=torch.long, device=dev)  # 连续接触步数
        self.goal_gate = torch.zeros(N, dtype=torch.bool, device=dev)    # latch: 稳定抓取成立
        self.grip_rel0 = torch.zeros(N, 3, device=dev)     # 门开时物体在掌心系的位置 (测抓取稳定)
        # critic 特权观测的常量部分 (M2 域随机化后改为每 env 采样值)
        self.obj_mass = self.object.root_physx_view.get_masses().view(N, 1).to(dev)
        self.obj_fric = torch.full((N, 1), du.semantics.friction, device=dev)
        # 本体历史 (HORA 式, stage-2 ProprioAdapt 的输入; PPO 阶段只维护不消费)
        # 宽度 = 2×受控关节数 (飞手 2×22=44; DexMate 2×(7+22)=58)
        self.prop_width = 2 * (len(self.hand_jids) + len(self.arm_jids))
        self.proprio_hist = torch.zeros(N, cfg.prop_hist_len, self.prop_width, device=dev)

        # ---- DexMate 摆到 cfg 的默认姿态 ----
        # 坑: ArticulationCfg.InitialStateCfg(joint_pos=...) **只填 data.default_joint_pos
        # 这个数据字段**, 既不写物理状态、也不设驱动目标 (target 默认全 0).
        # 不显式下发的话, PD 会把所有关节死死拉回零位 —— 表现就是"机器人还是初始姿态".
        # 必须两件事都做: write_joint_state_to_sim(瞬间到位) + set_joint_position_target(保持住).
        if getattr(self, "dexmate", None) is not None:
            _q = self.dexmate.data.default_joint_pos.clone()
            self.dexmate.write_joint_state_to_sim(_q, torch.zeros_like(_q))
            self.dexmate.set_joint_position_target(_q)
            self.dexmate.write_data_to_sim()
            _nz = int((_q.abs() > 1e-6).sum().item())
            print(f"[setup] DexMate 默认姿态已下发 ({_nz} 个非零关节)")
            # ⚠ 必须步进+update 才能读到真实 body 位姿.
            # write_joint_state_to_sim 只写进物理引擎, IsaacLab 的 data.body_pos_w 是
            # 带缓存的, 不刷新就还是旧值 —— 实测差 58.7cm (ZED 标记跑到头外面去了),
            # 而且这个旧值"既不是零位也不是终值", 很容易被误判成"已生效".
            self.sim.step(render=False)
            self.dexmate.update(self.sim.get_physics_dt())
            _zl = self.dexmate.data.body_pos_w[0, list(self.dexmate.body_names).index(
                "zed_left_camera")] if "zed_left_camera" in self.dexmate.body_names else None
            _bnl = list(self.dexmate.body_names)
            _bpl = self.dexmate.data.body_pos_w[0]
            print("[setup] 刷新后 (全部来自 Articulation, 同一来源):")
            for _k in ("vega_1p_head_l2", "vega_1p_head_l3", "zed_left_camera",
                       "zed_right_camera", "vega_1p_torso_l3"):
                if _k in _bnl:
                    print(f"[setup]   {_k:<20} {[round(v, 4) for v in _bpl[_bnl.index(_k)].tolist()]}")
            if "vega_1p_head_l3" in _bnl and "zed_left_camera" in _bnl:
                _d = (_bpl[_bnl.index("zed_left_camera")]
                      - _bpl[_bnl.index("vega_1p_head_l3")])
                print(f"[setup]   ZED 相对 head_l3 偏移 {[round(v,4) for v in _d.tolist()]} "
                      f"模长 {float(_d.norm())*100:.2f}cm (URDF 挂载定义是 6.11cm)")

        # ---- 双手轨迹摆放预览 (查看器专用) ----
        if getattr(cfg, "show_human_traj", False) and self.dexmate is not None:
            self._place_bimanual()

    # ---- 机器人抽象层 -------------------------------------------------
    # reward / 观测 / 终止 里的共享代码只通过这几个访问器读机器人状态, 不直接摸
    # self.hand.data.*. 飞手的"腕"是浮动根刚体, DexMate 的"腕"是 7 关节链的末端 body ——
    # 两者物理意义不同, 但对上层是同一组量, 所以能共用同一套 reward/终止逻辑.
    def _resolve_joint_ids(self):
        """哪些关节是手指、哪些是手臂 (下标进 self.hand 的关节表)."""
        self.hand_joint_names = list(self.hand.joint_names)
        self.hand_jids = list(range(self.hand.num_joints))
        self.arm_jids: list[int] = []                      # 飞手没有臂

    @property
    def finger_bound(self) -> float:
        """res_scale 的手指段用哪个界: 累积模式下动作是**每步增量**(rate),
        非累积模式下动作直接就是残差本身 (旧的 finger_residual_max)."""
        cfg = self.cfg
        return cfg.finger_rate_max if cfg.accumulate_finger else cfg.finger_residual_max

    def _accum_finger(self, fin_res: torch.Tensor) -> torch.Tensor:
        """手指残差累积: 吃本步增量 (已乘 rate 界、已过 dyn_res 缩放), 返回叠加到参考上的残差.

        非累积模式直接透传 —— 调用方不用分支.
        ⚠ dyn_res 只缩放**增量**, 不缩放累积上限: 若上限也跟着距离缩, 手一旦被物体
          推开一点, 已经攒下的握力就会被强行泄掉 (握不住 -> 更远 -> 泄更多, 正反馈).
        """
        cfg = self.cfg
        if not cfg.accumulate_finger:
            return fin_res
        self.finger_res = (self.finger_res * cfg.finger_res_decay + fin_res).clamp(
            -cfg.finger_total_max, cfg.finger_total_max)
        return self.finger_res

    def _build_res_scale(self, to):
        cfg = self.cfg
        return to(np.concatenate([
            np.full(3, cfg.wrist_pos_residual_max),
            np.full(3, cfg.wrist_rot_residual_max),
            np.full(22, self.finger_bound)]))

    @property
    def wrist_pos_w(self) -> torch.Tensor:
        return self.hand.data.root_pos_w

    @property
    def wrist_quat_w(self) -> torch.Tensor:
        return self.hand.data.root_quat_w

    @property
    def wrist_linvel_w(self) -> torch.Tensor:
        return self.hand.data.root_lin_vel_w

    @property
    def wrist_angvel_w(self) -> torch.Tensor:
        return self.hand.data.root_ang_vel_w

    @property
    def finger_q(self) -> torch.Tensor:
        return self.hand.data.joint_pos[:, self.hand_jids]

    @property
    def finger_qd(self) -> torch.Tensor:
        return self.hand.data.joint_vel[:, self.hand_jids]

    @property
    def tip_pos_w(self) -> torch.Tensor:
        return self.hand.data.body_pos_w[:, self.tip_ids]

    @property
    def effort_norm(self) -> torch.Tensor:
        """省力正则的输入 (0~1 量级). 飞手 = 凭空 wrench 的用量;
        DexMate 覆盖成关节力矩用量 (电机真的要出多少力, 物理意义更实)."""
        return self.wrench_norm

    def _reset_robot(self, env_ids, starts, origins):
        """把机器人摆到参考 starts 帧. 飞手 = 直接写浮动根位姿 + 手指 qpos."""
        n = len(env_ids)
        root = torch.zeros(n, 13, device=self.device)
        root[:, 0:3] = self.ref_wrist_pos[starts] + origins + self.wrist_jitter[env_ids]
        root[:, 3:7] = self.ref_wrist_quat[starts]
        self.hand.write_root_state_to_sim(root, env_ids)
        dof = self.ref_finger[starts].clone()
        self.hand.write_joint_state_to_sim(dof, torch.zeros_like(dof), env_ids=env_ids)
        self.hand.set_joint_position_target(dof, env_ids=env_ids)
        # 目标缓存同步到起始帧 (reset 后第一个物理步就有合法目标)
        self.wrist_tgt_pos[env_ids] = self.ref_wrist_pos[starts] + origins
        self.wrist_tgt_quat[env_ids] = self.ref_wrist_quat[starts]
        self.finger_tgt[env_ids] = self.ref_finger[starts]

    # ------------------------------------------------------------------
    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        entry = clips.clip_entry(self.cfg.clip_name)
        # ocir 源: 首次把 object.obj 转成物理烘焙 USD (缓存); replay_grasp 源: usd 已就位
        clips.ensure_object_usd(self.cfg.clip_name)
        self.object = RigidObject(self.cfg.object_cfg)
        if entry["runtime_object_physics"]:
            # replay_grasp 的 object.usd 是纯视觉网格, 与在线 replay 相同: 运行时贴刚体+碰撞
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
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        )
        table_cfg.func("/World/envs/env_.*/Table", table_cfg,
                       translation=(0.0, 0.0, self.cfg.table_top_z - sz / 2))
        # DexMate(Vega) 视觉参考: 挂 /World/Dexmate (env 命名空间之外 -> 不被克隆),
        # 不注册进 scene -> IsaacLab 不控制它; USD 自带 fixed root joint 所以不会倒.
        self.dexmate = None
        self._bimanual_res = None
        if getattr(self.cfg, "show_dexmate", False):
            import os as _os                       # 本模块顶层没有 import os
            if _os.path.exists(self.cfg.dexmate_usd):
                # ⚠ 必须注册成 IsaacLab Articulation, 不能只 spawn 成 USD prim.
                # GPU 仿真开着 PxSceneFlag::eENABLE_DIRECT_GPU_API, 这时 PhysX **禁止**
                # 通过 USD 的 DriveAPI.targetPosition 设驱动目标 (会报
                # "illegal to call this method if eENABLE_DIRECT_GPU_API is enabled").
                # 唯一合法路径是 Articulation 的张量 API (set_joint_position_target).
                from isaaclab.actuators import ImplicitActuatorCfg
                from isaaclab.assets import ArticulationCfg
                _dm = ArticulationCfg(
                    prim_path="/World/Dexmate",       # env 命名空间之外 -> 不被克隆
                    spawn=sim_utils.UsdFileCfg(usd_path=self.cfg.dexmate_usd),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=tuple(self.cfg.dexmate_pos), rot=tuple(self.cfg.dexmate_quat),
                        # 度->弧度 (prismatic 保持米); 这样重建后姿态不会丢
                        joint_pos={n: (float(v) if "prismatic" in n else float(np.deg2rad(v)))
                                   for n, v in (getattr(self.cfg, "dexmate_joints", None) or {}).items()}),
                    # stiffness/damping=None -> 沿用 USD 里已标定的增益 (实测 2503/150)
                    actuators={"all": ImplicitActuatorCfg(
                        joint_names_expr=[".*"], stiffness=None, damping=None)},
                )
                self.dexmate = Articulation(_dm)
                self.scene.articulations["dexmate"] = self.dexmate   # 让 scene 每步替它 write/update
                print(f"[setup] DexMate Articulation @ {self.cfg.dexmate_pos} 朝 +X")
            else:
                print(f"[setup] ⚠ 找不到 DexMate USD: {self.cfg.dexmate_usd} (跳过)")
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()
        # ---- 摩擦材质 ----
        # 已验证 OCIR/replay clip 继续复刻 SuperGrip; static reconstruction 必须
        # 使用 clip semantics 的物体摩擦, 不能被统一的 3.0 覆盖.
        # 对现有验证 clip: static=dynamic=3.0, combine=multiply =>
        # 手↔物 3.0×3.0=9.0, 物↔桌 3.0×0.5=1.5.
        # UsdFileCfg 不支持 physics_material, 故 clone 后显式绑定到每个 env 的
        # Object 与 Robot (bind 带 apply_nested, 自动覆盖全部 collider 子孙; 与验证器
        # 的 strongerThanDescendants 绑定同构). 手↔桌由 filter_collisions 过滤.
        # SuperGrip 只给 5 指尖 elastomer + 物体; 手身(掌/背/指节)用低摩擦.
        # 教训: 整手 9.0 会让手掌一碰物体就刚性粘住拖飞(参考手差几cm没抓上时尤甚).
        # 指尖↔物=3.0×3.0=9.0 (真抓握强握); 手身↔物=0.2×3.0=0.6 (掌蹭不拖飞).
        # POUR_PAD_FRIC: 指垫(及绑同材质的主体物)摩擦覆写, 默认 3.0 保持原行为。
        # 用途: 难度消融(调大=更好抓)。2026-08-29 加, 不设变量时零影响。
        _grip_mu = float(os.environ.get("POUR_PAD_FRIC", "3.0"))
        grip = sim_utils.RigidBodyMaterialCfg(
            static_friction=_grip_mu, dynamic_friction=_grip_mu, restitution=0.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply")
        grip.func("/World/Materials/SuperGrip", grip)
        low = sim_utils.RigidBodyMaterialCfg(
            static_friction=0.2, dynamic_friction=0.2, restitution=0.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply")
        low.func("/World/Materials/LowGrip", low)
        object_material = "/World/Materials/SuperGrip"
        if entry["source"] == "static_reconstruction":
            object_mu = float(entry["semantics"].friction)
            clip_object = sim_utils.RigidBodyMaterialCfg(
                static_friction=object_mu, dynamic_friction=object_mu,
                restitution=0.0, friction_combine_mode="average",
                restitution_combine_mode="multiply")
            object_material = "/World/Materials/ClipObject"
            clip_object.func(object_material, clip_object)
        n_object = n_grip = n_low = 0
        for p in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Object"):
            sim_utils.bind_physics_material(p, object_material); n_object += 1
        for p in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Robot"):
            sim_utils.bind_physics_material(p, "/World/Materials/LowGrip",
                                            stronger_than_descendants=False); n_low += 1
        for name in self.cfg.fingertip_bodies:                       # 指尖覆盖回 SuperGrip
            for p in sim_utils.find_matching_prim_paths(f"/World/envs/env_.*/Robot/{name}"):
                sim_utils.bind_physics_material(p, "/World/Materials/SuperGrip"); n_grip += 1
        # 双臂 B 侧指垫 (2026-08-20 摩擦孪生案: fingertip_bodies 只含交互侧,
        # 另一只手的垫从建仓起是 LowGrip 0.2 —— 与 tip_ids 串台同根, 这里补绑)
        for name in getattr(self.cfg, "extra_supergrip_bodies", []) or []:
            for p in sim_utils.find_matching_prim_paths(f"/World/envs/env_.*/Robot/{name}"):
                sim_utils.bind_physics_material(p, "/World/Materials/SuperGrip"); n_grip += 1
        print(f"[setup] 物体材质={object_material}×{n_object} "
              f"指尖SuperGrip×{n_grip} 手身LowGrip×{n_low}")
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # ★ 多物体: 第 2 个起。每个都注册进 scene, 于是 clone_environments 会复制,
        #   GUI 里也看得到。它们**不参与**接触传感器/奖励 —— 本步只做摆放。
        # 指尖接触传感器 (冠军 env 同款模式)
        self._contact_sensors = []
        for i, scfg in enumerate(self.cfg.contact_sensors):
            s = ContactSensor(scfg)
            self._contact_sensors.append(s)
            self.scene.sensors[f"tip_contact_{i}"] = s
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_curve(prim_path, pts, rgb, width):
        """用 UsdGeom.BasisCurves 画一条细线 (纯视觉, 无碰撞/无物理)."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            stage.RemovePrim(prim_path)
        c = UsdGeom.BasisCurves.Define(stage, prim_path)
        c.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
        c.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*[float(v) for v in p]) for p in pts]))
        c.CreateCurveVertexCountsAttr().Set(Vt.IntArray([len(pts)]))
        c.CreateWidthsAttr().Set(Vt.FloatArray([float(width)] * len(pts)))
        c.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        c.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))

    @staticmethod
    def _draw_sphere(prim_path, center, rgb, radius=0.015):
        """标记球 (纯视觉)."""
        import omni.usd
        from pxr import Gf, UsdGeom, Vt
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(prim_path).IsValid():
            stage.RemovePrim(prim_path)
        sp = UsdGeom.Sphere.Define(stage, prim_path)
        sp.CreateRadiusAttr().Set(float(radius))
        sp.CreateDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*rgb)]))
        UsdGeom.Xformable(sp).AddTranslateOp().Set(
            Gf.Vec3d(*[float(v) for v in center]))

    def _place_bimanual(self):
        """按 bimanual_align 的通用摆放: 挪物体 + 画轨迹 + 跑质量门."""
        import numpy as _np
        from rl_rebuild.correction import bimanual_align as BA
        bn = list(self.dexmate.body_names)
        bp = self.dexmate.data.body_pos_w[0]
        try:
            rh = {"left": bp[bn.index("left_hand_C_MC")].tolist(),
                  "right": bp[bn.index("right_hand_C_MC")].tolist()}
        except ValueError:
            print("[bimanual] ⚠ 找不到 left/right_hand_C_MC, 跳过"); return
        # 机器人"头"锚定点 (cfg.head_anchor 选)
        _ha = getattr(self.cfg, "head_anchor", "vega_1p_head_l3")
        rhead = None
        try:
            if _ha == "zed_mid":
                rhead = ((bp[bn.index("zed_left_camera")]
                          + bp[bn.index("zed_right_camera")]) / 2.0).tolist()
            elif _ha in bn:
                rhead = bp[bn.index(_ha)].tolist()
        except ValueError:
            pass
        if rhead is None:
            print(f"[bimanual] ⚠ 找不到锚定 link '{_ha}', 退回锚在交互手")
        else:
            print(f"[bimanual] 头锚定点 = {_ha} {[round(v,4) for v in rhead]}")
        e = clips.clip_entry(self.cfg.clip_name)
        res = BA.compute_placement(e["npz"], e["mesh"], rh, robot_head=rhead,
                                   table_top_z=self.cfg.table_top_z, obj_gap=0.002,
                                   grasp_gap=self.cfg.grasp_gap)
        prim, gf, it = res["primary"], res["grasp_frame"], res["inter"]
        print(f"[bimanual] 交互手={it['hands']} 主手={prim} 抓取帧={gf} 释放帧={it['release_frame']}")
        print(f"[bimanual] yaw={_np.degrees(res['yaw']):+.1f}° (级{res['yaw_tier']}) "
              f"锚定={res['anchor_mode']} 平移={res['shift'].round(4).tolist()}")
        # 轨迹细线: 交互手实色(会被跟踪), 非交互手灰色(仅参考, 机器人对应臂保持默认姿态)
        for h in ("left", "right"):
            col = ((0.95, 0.25, 0.15) if h == "right" else (0.15, 0.85, 0.25)) \
                if h in it["hands"] else (0.45, 0.45, 0.45)
            self._draw_curve(f"/World/HumanTraj_{h[0].upper()}", res["joints"][h][:, 0],
                             col, self.cfg.traj_width)
        # 红球 = 交互手掌心@抓取帧 (物体 XY 就取自这里)
        self._draw_sphere("/World/Mark_Palm", res["palm"][prim][gf], (1.0, 0.05, 0.05), 0.018)
        self._draw_sphere("/World/Mark_Wrist", res["joints"][prim][gf, 0], (1.0, 0.6, 0.0), 0.012)
        # 相机(人头)位姿: 蓝球标位置 + 蓝线标视线方向
        if res.get("cam_pos") is not None:
            cp, cf = res["cam_pos"], res["cam_fwd"]
            # 相机位姿仍用于 头->头 XY 锚定, 只是不再画任何可视标记
            print(f"[bimanual] 相机(人头) @帧{gf} {cp[gf].round(3).tolist()} "
                  f"(全程位移仅 {_np.linalg.norm(cp.max(0)-cp.min(0))*100:.1f}cm, 仅用于锚定)")
        # 物体: ref_obj_pos/quat 也必须覆盖, 否则冻结窗每步把它钉回旧 re-anchor 位
        org = self.scene.env_origins[0].cpu().numpy()
        op = torch.tensor(res["obj_pos"] - org, dtype=torch.float32, device=self.device)
        oq = torch.tensor(res["obj_quat"], dtype=torch.float32, device=self.device)
        self.obj_init_pos, self.obj_init_quat = op, oq
        self.ref_obj_pos[:] = op
        self.ref_obj_quat[:] = oq
        self.ref_obj_vel[:] = 0.0
        pose = torch.cat([op + self.scene.env_origins[0], oq]).unsqueeze(0)
        self.object.write_root_pose_to_sim(pose.repeat(self.num_envs, 1))
        print(f"[bimanual] 物体 -> {res['obj_pos'].round(4).tolist()} | "
              f"🔴红球=掌心@帧{gf} 距物体 "
              f"{_np.linalg.norm(res['palm'][prim][gf] - res['obj_pos'])*100:.2f}cm")
        # 质量门 —— 肩位必须从 Articulation **实测**取.
        # 之前用 URDF 零位偏移 (-0.246,0,0.428) 是错的: 默认姿态 torso_j1=45° j2=90°
        # 把躯干折起来了, 实际肩心 z=1.300 而不是 0.428, 差 87cm, 导致 reach 被严重误判.
        _shl = f"vega_1p_{'R' if prim == 'right' else 'L'}_arm_l1"
        sh = bp[bn.index(_shl)].tolist() if _shl in bn else None
        ok, m = BA.placement_report(res, table_top_z=self.cfg.table_top_z, shoulder=sh,
                                    grasp_gap=self.cfg.grasp_gap)
        # reach 明细: 分段看哪一段超, 以及物体本身可不可达
        if sh is not None:
            _sh = _np.asarray(sh, float)
            _pm = res["palm"][prim]
            _a, _b = it["per_hand"][prim]
            print(f"[reach] 肩心(实测) {[round(v,3) for v in _sh.tolist()]}  臂展上限 0.755 m")
            print(f"[reach] 物体离肩 {_np.linalg.norm(res['obj_pos']-_sh)*100:.1f} cm "
                  f"-> {'✅可达' if _np.linalg.norm(res['obj_pos']-_sh)<=0.755 else '❌够不到'}")
            for _n, _lo, _hi in (("抓取帧", _a, _a + 1), ("接触段", _a, _b + 1),
                                 ("训练实际用到(到grasp_end)", 0, min(_b + 1, 57)),
                                 ("全程", 0, len(_pm))):
                _d = _np.linalg.norm(_pm[_lo:_hi] - _sh, axis=1)
                print(f"[reach]   {_n:<26} max {_d.max()*100:6.1f} cm  "
                      f"超限帧 {_np.sum(_d>0.755)}/{_hi-_lo}  "
                      f"{'✅' if _d.max()<=0.755 else '❌'}")
        self._bimanual_res = res          # 供查看器做 IK 跟踪
        print(f"[质量门] {'✅ 全过' if ok else '❌ 有项未过'}")
        for k, v in m.items():
            print(f"[质量门]   {'OK ' if v['ok'] else 'FAIL'} {k:<22} {v['v']:.4f} {v.get('note','')}")

    def set_dexmate_joints(self, deg_map: dict) -> list:
        """给 DexMate 摆姿势 —— 走 Articulation 张量 API (GPU 仿真下唯一合法路径).

        单位: revolute=**度**(内部转弧度), prismatic=**米**.
        目标写进去后由 scene 每步 write_data_to_sim 推给 PhysX, 会保持住.
        """
        if self.dexmate is None or not deg_map:
            return []
        names = list(self.dexmate.joint_names)
        tgt = self.dexmate.data.joint_pos_target.clone() \
            if hasattr(self.dexmate.data, "joint_pos_target") \
            else self.dexmate.data.joint_pos.clone()
        ok, bad = [], []
        for n, v in deg_map.items():
            if n not in names:
                bad.append(n)
                continue
            i = names.index(n)
            # dummy_base 的两个平移关节用米, 其余角关节用度
            tgt[:, i] = float(v) if "prismatic" in n else float(np.deg2rad(v))
            ok.append(f"{n}={v}")
        if ok:
            self.dexmate.set_joint_position_target(tgt)
            self.dexmate.write_data_to_sim()
            # 读回实测角, 用来确认目标真的传到 PhysX 并驱动了关节 (不是只写了个数)
            cur = {}
            for n in deg_map:
                if n not in names:
                    continue
                raw = float(self.dexmate.data.joint_pos[0, names.index(n)].item())
                cur[n] = round(raw, 3) if "prismatic" in n else round(float(np.rad2deg(raw)), 1)
            print(f"[dexmate] 目标={ok}  设定瞬间实测={cur}", flush=True)
        if bad:
            print(f"[dexmate] ⚠ 无此关节: {bad}")
        return ok

    def set_dexmate_base(self, pos=None, yaw_deg=None) -> bool:
        """整机位姿 —— 同理必须走 write_root_pose_to_sim, 不能改 USD xform."""
        if self.dexmate is None:
            return False
        root = self.dexmate.data.root_state_w[:, :7].clone()
        if pos is not None:
            root[:, :3] = torch.tensor([float(v) for v in pos], device=self.device)
        if yaw_deg is not None:
            h = float(np.deg2rad(yaw_deg)) * 0.5
            root[:, 3:7] = torch.tensor([np.cos(h), 0.0, 0.0, np.sin(h)],
                                        device=self.device, dtype=root.dtype)   # wxyz
        self.dexmate.write_root_pose_to_sim(root)
        print(f"[dexmate] 底座目标 pos={pos} yaw={yaw_deg} | 写入前实测 pos="
              f"{[round(v, 3) for v in self.dexmate.data.root_state_w[0, :3].tolist()]}", flush=True)
        return True

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
        wrist_pos_res = a[:, 0:3]
        if self.approach_steps > 0:                    # 接近段(还没到PreGrasp)腕残差放大
            in_approach = (t < self.grasp_start).float().unsqueeze(1)
            wrist_pos_res = wrist_pos_res * (1 + in_approach * (self.cfg.approach_res_scale - 1))
        self.wrist_tgt_pos = self.ref_wrist_pos[t] + self.scene.env_origins + wrist_pos_res + self.wrist_jitter
        # 硬编码抬升: 抓取窗口结束后腕目标垂直匀速上升 lift_height.
        # _ref_t 已在 grasp_end 钳住 -> ref_wrist_pos[t] 恒为抓取末姿态, 这里只叠 z 斜坡.
        if self.cfg.grasp_only:
            self.wrist_tgt_pos[:, 2] += self._lift_prog() * self.cfg.lift_height
        self.wrist_tgt_quat = quat_mul(_quat_from_aa(a[:, 3:6]), self.ref_wrist_quat[t])
        finger_ref = self._contact_close_ref() if self.contact_close else self.ref_finger[t]
        self.finger_tgt = (finger_ref + self._accum_finger(a[:, 6:28])
                           ).clamp(self.dof_lower, self.dof_upper)

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
            # 累积残差状态: **必须给**, 否则策略看不见自己已经积了多少 -> 非马尔可夫,
            # 只能靠 finger_q 猜, 而恰恰在接触时(手指被物体挡住)两者差最大.
            # 归一到 [-1,1] 与其他通道同量级.
            self.finger_res / self.cfg.finger_total_max if cfg.accumulate_finger
            else torch.zeros(self.num_envs, 0, device=self.device),     # 22 (累积模式)
            # 终点势的门控状态: 奖励依赖它, 策略就必须看得见 —— 否则 reward 取决于
            # 一个不可观测的 latch, 信用分配是瞎的
            self.goal_gate.float().unsqueeze(1) if self.use_goal
            else torch.zeros(self.num_envs, 0, device=self.device),     # 1 (终点势时)
            self.actions_buf,                                           # 28
            obj_geom,                                                   # 3P 物体几何(腕系)
        ], dim=1).clamp(-cfg.clip_obs, cfg.clip_obs).nan_to_num(0.0)     # NaN 防护: clamp 不吃 NaN
        self._check_obs_dim(obs)

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

    def _check_obs_dim(self, obs: torch.Tensor) -> None:
        """观测宽度对不上 cfg.observation_space 就当场报错.

        不查的话要等到网络前向才炸, 报错里只有两个数字 —— 而 dyn_res / accumulate_finger
        这类开关各自带一段**条件拼接**的通道, 开关一改就容易忘同步 observation_space.
        """
        n = int(obs.shape[1])
        if n != self.cfg.observation_space:
            raise RuntimeError(
                f"观测宽度 {n} != cfg.observation_space {self.cfg.observation_space}"
                f" (accumulate_finger={self.cfg.accumulate_finger} 带 22 维,"
                f" dyn_res={self.cfg.dyn_res} 带 1 维) —— 改开关要同步改 observation_space")

    # ---- reward 输入采集 ----
    def _palm_pos(self) -> torch.Tensor:
        """掌心位置 (N,3, 含 env_origin): 腕根 + 手系 +z(指向) 9cm 固定偏移.
        不用指尖质心 — 手指开合会移动质心, 污染 approach 的距离信号."""
        offset = torch.tensor([0.0, 0.0, 0.09], device=self.device).expand(self.num_envs, 3)
        return self.wrist_pos_w + quat_apply(self.wrist_quat_w, offset)

    def _tip_contacts(self) -> torch.Tensor:
        """(N,5) 指尖-物体接触力是否超阈 (传感器已 filter 到 Object)."""
        f = torch.cat([s.data.force_matrix_w.view(self.num_envs, 1, 3)
                       for s in self._contact_sensors], dim=1)          # (N,5,3)
        return (f.norm(dim=-1) > self.cfg.contact_force_thresh).float()

    def _contact_close_ref(self) -> torch.Tensor:
        """接触触发合拢参考(N,22): 指尖离物体表面近于阈值 -> 开始合拢, close_speed_steps 内合到位.
        触发后即使手离开也继续合(close_started 锁存), 保证合拢不回退."""
        tips = self.tip_pos_w                                        # (N,5,3) 世界
        op = self.object.data.root_pos_w                            # (N,3)
        if self.obj_points is not None:
            P = self.obj_points.shape[0]
            oq = self.object.data.root_quat_w
            pw = quat_apply(oq[:, None].expand(-1, P, -1).reshape(-1, 4),
                            self.obj_points[None].expand(self.num_envs, -1, -1).reshape(-1, 3)
                            ).reshape(self.num_envs, P, 3) + op[:, None]
            d = torch.cdist(tips, pw).amin(dim=(1, 2))              # (N,) 最近指尖-表面
        else:
            d = (tips - op[:, None]).norm(dim=-1).min(dim=1).values
        self.close_started |= d < self.cfg.close_trigger_dist
        self.close_prog = (self.close_prog + self.close_started.float()
                           / max(self.cfg.close_speed_steps, 1)).clamp(0.0, 1.0)
        u = self.close_prog.unsqueeze(1)
        return (1 - u) * self.open_pose + u * self.closed_pose

    # ---- 动态残差 (按 掌心->物体表面 距离连续缩放) ---------------------
    def _obj_surface_points(self) -> torch.Tensor:
        """(N,P,3) 物体表面采样点的世界位置. 无点云时退化成 (N,1,3) 的物体原点."""
        op = self.object.data.root_pos_w
        if self.obj_points is None:
            return op[:, None]
        P = self.obj_points.shape[0]
        oq = self.object.data.root_quat_w
        return quat_apply(oq[:, None].expand(-1, P, -1).reshape(-1, 4),
                          self.obj_points[None].expand(self.num_envs, -1, -1).reshape(-1, 3)
                          ).reshape(self.num_envs, P, 3) + op[:, None]

    def _palm_obj_dist(self) -> torch.Tensor:
        """(N,) **手上任一接触点(掌心 + 5 指尖)到物体表面**的最小距离.

        为什么不只用掌心: 抓取窗内腕是被冻结的(ref builder 的设计, 见 dexmate_env),
        掌心到物体的距离按构造几乎不变(实测全程只在 3.5~14cm 间晃), 拿它当"抓得多近"
        的信号是没有分辨力的 —— 真正在变的是**手指在合拢**.
        取 6 个点的最小值, 接近段由掌心/整手主导, 抓取段由指尖主导, 两段都有分辨力.

        用表面而不是物体中心: 后者对大物体会系统性偏大, 同一阈值在不同尺寸物体上含义
        就不一样, 没法跨物体泛化.
        ⚠ "表面"是 n_obj_points 个 FPS 采样点, 是真实表面距离的上界;
        64 点在 8cm 量级的物体上误差几毫米, 够用.
        """
        pts = torch.cat([self._palm_pos()[:, None], self.tip_pos_w], dim=1)   # (N,6,3)
        return torch.cdist(pts, self._obj_surface_points()).amin(dim=(1, 2))

    def _dyn_res_scale(self):
        """按手-物距离算残差缩放 -> (臂/腕缩放, 手指缩放, 归一化距离 u∈[0,1]).

        为什么要这个: 人手轨迹只是**弱参考**, 作用是把手引到物体附近、省掉盲目探索.
          远处 (u→1): 没有精细约束, 臂放开去找物体; 手指基本不用动
          近处 (u→0): 毫米级容忍度, 臂收紧免得把物体捅飞; 手指反过来放开, 去试怎么握住
        所以**臂和手指的缩放方向是相反的**。

        比现在按帧号切 (approach_res_scale) 好在三点:
          ① 距离是物理量, 帧号只是它的代理 —— 策略学会提前到位后帧号就失效了
          ② 连续, 没有 grasp_start 那个不连续跳变点 (值函数不好拟合)
          ③ 手指也跟着调, 而不是只调臂
        """
        cfg = self.cfg
        d = self._palm_obj_dist()
        u = ((d - cfg.dyn_d_near) / max(cfg.dyn_d_far - cfg.dyn_d_near, 1e-6)).clamp(0.0, 1.0)
        s_arm = cfg.dyn_arm_near + (cfg.dyn_arm_far - cfg.dyn_arm_near) * u
        s_fin = cfg.dyn_fin_near + (cfg.dyn_fin_far - cfg.dyn_fin_near) * u
        return s_arm, s_fin, u

    def _tip_affordance(self) -> torch.Tensor | None:
        """(N,5) 每个指尖最近 affordance 点的 heatmap 值 [0,1]. 无 affordance 返回 None.
        指尖世界位 -> 物体规范系(用当前物体位姿) -> 最近点 heatmap. 逼指尖落到接触区/环."""
        if self.aff_points is None:
            return None
        tips = self.tip_pos_w                                            # (N,5,3) 世界
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

        tip_afford = self._tip_affordance()
        # ---- 终点势的门控: 连续 goal_gate_steps 步 ≥2 指接触 -> latch 打开, 保持整回合 ----
        goal_pos = None
        if self.use_goal:
            n_c = contacts.sum(dim=1)
            self.contact_run = torch.where(n_c >= self.rw.min_contacts,
                                           self.contact_run + 1,
                                           torch.zeros_like(self.contact_run))
            newly_open = (~self.goal_gate) & (self.contact_run >= self.cfg.goal_gate_steps)
            if newly_open.any():
                # 门开瞬间记下物体在掌心系的位置 —— 之后的漂移就是"抓得稳不稳"
                self.grip_rel0[newly_open] = self._obj_in_palm(obj_pos)[newly_open]
            self.goal_gate |= self.contact_run >= self.cfg.goal_gate_steps
            goal_pos = self.ref_obj_pos[self.L - 1].expand(self.num_envs, 3)
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
            finger_q=self.finger_q,
            wrist_pos=self.wrist_pos_w - origins,
            ref_obj_pos=self.ref_obj_pos[t],
            ref_finger_q=ref_fin,
            ref_wrist_pos=ref_wrist,
            action=self.actions_buf,
            prev_action=self.prev_actions,
            wrench_norm=self.effort_norm,
            active=active,
            prev_palm_dist=self.prev_palm_dist,
            lift_hw=self.lift_hw,
            w=self.rw,
            finger_weight=self.finger_weight,
            curobo_wp=curobo_wp,
            in_approach=in_approach,
            tip_afford=tip_afford,
            goal_pos=goal_pos,
            goal_gate=self.goal_gate if self.use_goal else None,
            prev_goal_dist=self.prev_goal_dist,
            grip_drift=self._grip_drift(obj_pos) if self.use_goal else None,
        )
        self._update_diag(active, contacts, obj_pos, tip_afford, terms)
        # 逐项 episode 累计 (reward hacking 账本)
        for k, v in terms.items():
            if k not in self._ep_sums:
                self._ep_sums[k] = torch.zeros(self.num_envs, device=self.device)
            self._ep_sums[k] += v
        self.hold_ok += self._success_step(obj_pos, contacts).long()
        return total

    def _obj_in_palm(self, obj_pos) -> torch.Tensor:
        """物体位置换算到掌心系 (N,3). 抓稳了这个量应该基本不动 —— 手物相对位姿漂移
        就是"稳定抓取"的直接度量, 比"接触了几根手指"更贴近我们真正要的东西."""
        rel = obj_pos - (self._palm_pos() - self.scene.env_origins)
        return quat_apply(quat_conjugate(self.wrist_quat_w), rel)

    def _grip_drift(self, obj_pos) -> torch.Tensor:
        """(N,) 门开以来物体在掌心系里漂了多远 (m). 抓稳了应该几乎不动."""
        return (self._obj_in_palm(obj_pos) - self.grip_rel0).norm(dim=1).nan_to_num(0.0)

    def _update_diag(self, active, contacts, obj_pos, tip_afford, terms) -> None:
        """采集诊断量. **每一条都对应 docs/DESIGN_LOOP.md 里一条可证伪的设计假设** ——
        加新指标前先想清楚"它长成什么样能说明我哪个设计错了", 想不出来就别加.
        """
        cfg = self.cfg
        d = self.diag
        d.tick(active)
        n_contact = contacts.sum(dim=1)
        # [假设 A] 手指够得到物体 (总权限 finger_total_max 足够)
        d.add("contact2_frac", (n_contact >= 2).float(), active)
        d.add("contact_any_frac", (n_contact >= 1).float(), active)
        # [假设 B] rate/total 拆分是对的: 每步够温柔, 总量够到位
        if cfg.accumulate_finger:
            fr = self.finger_res.abs()
            d.add("fres_used_deg", torch.rad2deg(fr.mean(dim=1)), active)
            d.add("fres_sat_frac",
                  (fr.amax(dim=1) >= 0.95 * cfg.finger_total_max).float(), active)
        fa = self.actions_buf[:, -22:].abs().mean(dim=1)      # 手指段动作用量 ∈[0,1]
        d.add("frate_util", fa, active)
        # [假设 C] 给手指 69° 权限不会把物体在抓住之前碰跑
        # ⚠ 必须钳制 + 去 NaN: 物理发散的回合物体会跑到几百米外, 而这是个"回合内取 max、
        #   窗口内取 mean"的量 —— 一个发散回合就能把 256 回合的均值顶到 5 万 cm,
        #   读出来完全是垃圾 (2026-07-27 accum run 实测 51593cm). 超界的单独计数,
        #   "发散了几成回合"本来就该是独立的一条信息, 不该混进"推开多远".
        pre = active & ~self.grasped_once
        disp = ((obj_pos - self.obj_p0).norm(dim=1) * 100.0).nan_to_num(1e4)
        d.add("obj_disturb_cm", disp.clamp(max=100.0), pre, mode="max")
        d.add("diverge_frac", (disp > 100.0).float(), active)
        self.grasped_once |= n_contact >= self.rw.min_contacts
        # [假设 D] 抓住之后握得住 (抬得起来)
        lift_cm = ((obj_pos[:, 2] - self.obj_rest_z) * 100.0).nan_to_num(1e4)
        d.add("lift_peak_cm", lift_cm.clamp(-100.0, 100.0), active, mode="max")
        # [假设 G] 成功判据本身: 完整轨迹任务的成功 = 物体最终落在人手结束位 place_tol 内.
        # 单独测它 —— reward 的 task 核测的是"抬多高", 与这个判据**不是同一件事**.
        if not cfg.grasp_only:
            goal = self.ref_obj_pos[self.L - 1]
            dg = (((obj_pos - goal).norm(dim=1)) * 100.0).nan_to_num(1e4).clamp(max=200.0)
            d.add("place_err_cm", dg, active & (self._ref_t() >= self.L - 1))
            d.add("goal_d_cm", dg, active)                 # 全程到终点的距离
        # [假设 H] 终点势的门确实会打开 —— 门若从没开, "有没有用"根本无从谈起
        if self.use_goal:
            d.add("goal_gate_frac", self.goal_gate.float(), active)
            # 抓取稳定性: 门开之后物体在掌心系里漂了多远 (抓稳了应该基本不动)
            drift = ((self._obj_in_palm(obj_pos) - self.grip_rel0).norm(dim=1) * 100.0
                     ).nan_to_num(1e4).clamp(max=100.0)
            d.add("grip_drift_cm", drift, active & self.goal_gate)
        # [假设 I] 策略不是靠"打"物体完成任务 (录像里看到的快速摆手)
        spd = self.object.data.root_lin_vel_w.norm(dim=1).nan_to_num(1e4)
        d.add("obj_speed_max", spd.clamp(max=20.0), active, mode="max")
        d.add("fast_frac", (spd > self.rw.spike_vel).float(), active)
        # [假设 J] 策略的残差出得了力. 力矩顶满时残差**静默失效**, 信用分配被污染 ——
        # 开自碰撞会大幅推高饱和度 (零动作实测 0.24 -> 0.96), 必须盯着.
        tq = getattr(self, "arm_torque_norm", None)
        if tq is not None:
            d.add("torque_sat_frac", (tq >= 0.95).float().mean(dim=1), active)
            d.add("torque_mean", tq.mean(dim=1).nan_to_num(0.0).clamp(max=2.0), active)
        # [假设 E] 没有 GraspPose 时, affordance 能顶替它指出"该抓哪儿"
        if tip_afford is not None:
            touching = active & (n_contact >= 1)
            d.add("afford_hit", (contacts * tip_afford).sum(dim=1), touching)
        # [假设 F] 奖励配比: 引导项不该盖过任务核. 这里重算一份逐项均值 ——
        # _reset_idx 里那份 ep_rew/* 走的是 extras 单点路径, 分母同样不可信.
        # ⚠ 必须按 term_clamp 钳 —— compute_reward 返回的 terms 是**未钳制**的原始值,
        #   而策略实际拿到的是 clamp(min=term_clamp) 之后的. 不钳的话某一项发散到 -40
        #   (实测 rew_approach), 而它对策略的真实影响上限是 -10, E1 那条"引导项压过任务核"
        #   的判读就会被一个根本没生效的数字触发.
        for k, v in terms.items():
            d.add(f"rew_{k}", v.clamp(min=self.rw.term_clamp))
        self.extras.update(d.publish())

    def _success_step(self, obj_pos, contacts) -> torch.Tensor:
        """(N,) 本步是否计入成功. 子类可覆盖成别的任务判据.

        默认(飞手 / grasp_only): hold 段内 抬够高 且 ≥min_contacts 指接触.
        """
        if self.cfg.grasp_only:
            in_hold = self.episode_length_buf >= self.hold_step0   # 硬编码抬升完成之后才判
        else:
            in_hold = self._ref_t() >= (self.L - 1)
        _lift_thr = (self.cfg.success_lift_height if self.cfg.grasp_only
                     else self.rw.lift_target)
        lifted = (obj_pos[:, 2] - self.obj_rest_z) >= _lift_thr
        held = contacts.sum(dim=1) >= self.rw.min_contacts
        return in_hold & lifted & held

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        t = self._ref_t()
        origins = self.scene.env_origins
        obj_pos = self.object.data.root_pos_w - origins
        wrist_pos = self.wrist_pos_w - origins
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
            # approach>0: 从 grasp_start-approach_steps 起步 (沿接近路径够到 PreGrasp), 再抓
            starts = torch.full((n,), self.grasp_start - self.approach_steps,
                                dtype=torch.long, device=self.device)
            use_rsi = torch.zeros(n, dtype=torch.bool, device=self.device)
        else:
            inter0 = max(int(self.du.ref.interaction_seg[0]), self.t0)
            use_rsi = torch.rand(n, device=self.device) < self.cfg.rsi_prob
            rsi = torch.randint(inter0, self.L, (n,), device=self.device)
            starts = torch.where(use_rsi, rsi,
                                 torch.full((n,), self.t0, dtype=torch.long, device=self.device))
        self.rsi_start[env_ids] = starts
        self.freeze_ctr[env_ids] = self.cfg.settle_steps
        self.close_prog[env_ids] = 0.0                  # 接触触发合拢进度清零
        self.close_started[env_ids] = False
        # 起点随机化(鲁棒性): 每回合采样腕位偏移, 全程叠加 -> 每 env 从不同起点抓
        if self.start_jitter > 0:
            self.wrist_jitter[env_ids] = (torch.rand(n, 3, device=self.device) * 2 - 1) * self.start_jitter

        self._reset_robot(env_ids, starts, origins)      # 子类覆盖: 关节空间摆位
        # 物体: RSI env 用参考物体位姿 (抬升态=在手里, 靠冻结窗撑住); t0 env 用贴底稳定初始位
        um = use_rsi.unsqueeze(1)
        obj_p = torch.where(um, self.ref_obj_pos[starts], self.obj_init_pos.expand(n, -1))
        obj_q = torch.where(um, self.ref_obj_quat[starts], self.obj_init_quat.expand(n, -1))
        obj = torch.zeros(n, 13, device=self.device)
        obj[:, 0:3] = obj_p + origins
        obj[:, 3:7] = obj_q
        self.object.write_root_pose_to_sim(obj[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(obj[:, 7:], env_ids)

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
        if hasattr(self, "_last_stuck"):      # DexMate 特有: 臂持续追不上目标 (撞桌/自碰撞/力矩不够)
            log["term/stuck"] = self._last_stuck[env_ids].float().mean().item()
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
        self.finger_res[env_ids] = 0.0                   # 累积残差: 跨步状态, 必须清
        self.wrench_norm[env_ids] = 0.0
        # 诊断: 先结算本回合 (必须在清零之前), 再重置本回合基线
        self.diag.finish(env_ids)
        self.obj_p0[env_ids] = obj_p
        self.grasped_once[env_ids] = False
        # 终点势跨步状态: prev 必须初始化成"当前距离", 否则第一步白送一整段距离的横财
        self.contact_run[env_ids] = 0
        self.goal_gate[env_ids] = False
        self.grip_rel0[env_ids] = 0.0
        self.prev_goal_dist[env_ids] = (obj_p - self.ref_obj_pos[self.L - 1]).norm(dim=1)
        # 势差分状态: 用参考起始腕位到物体位的距离近似 (首个 active 步会精确重置)
        self.prev_palm_dist[env_ids] = (self.ref_wrist_pos[starts] - obj_p).norm(dim=1)
