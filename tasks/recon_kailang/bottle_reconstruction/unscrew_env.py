"""扭开瓶盖 RL 任务 v1 — 在 BottleReconstructionEnv (assembled 螺旋场景) 上加任务层.

设计台账见同目录 `LEDGER_unscrew.md` (假设 + 证伪信号, 训练前写好).

## 结构 (对照 tasks/pregrasp 的分层惯例)

底层继承链已提供并验证过的东西, 这里**只加不改**:
  - BottleReconstructionEnv: 双刚体场景 + GPU 解析螺旋 (screw_angle/engaged/has_depth),
    盖子由指尖接触施加的**物理转动**驱动, 满 2 圈 detach 释放.
  - DexmateCorrectionEnv:  29 维残差动作 (臂7+指22, 累积 + rate/total 拆分),
    object_only 模式 = 参考钉在常量位姿, 管道(tube)圈住参考腕位.

本类做四件事:
  1. **把右臂参考换成"帽正上方的 IK 预抓姿"** —— object_only 默认参考是站姿,
     残差界(毫米级/步)结构上走不到帽; 换参考后零动作 = 悬停在帽上方,
     管道自动圈住帽周围 5cm 工作空间. (对照 pregrasp v2 "起点=PreGrasp" 的同一设计)
  2. 帽接触传感器 ×5 (基类的 5 个 filter 到瓶身, 拧盖要的是**帽**的接触).
  3. 奖励: 势差分为主 (拧动角 Δθ 是任务核), 接触/接近小额引导, 成功 +20 终止.
  4. 观测 +10: 帽位置(腕系)3 + 拧动进度1 + engaged1 + 帽接触5.

## 已声明的限制 (v1)

  - 无域随机化: 场景确定 (瓶位/摆角/质量固定), 只验可行性, 不验鲁棒.
  - 手指是 22 维原始残差, 没有合拢模板 —— pregrasp 的教训是随机噪声难凑出协调
    抓握 (DESIGN_LOOP §2.1); 若 cap_contact2_frac 长期为 0, 下一步是加合拢斜坡
    (证伪信号 U2, 见台账).
  - replay 时钟 = 15Hz (用户 2026-08-25 裁定, 信任 replay_world.npz 的 fps 元数据).
    本任务 object_only 不回放人手轨迹, 该决定当前只影响参考重采样, 但记录在案:
    未来接人手拧盖参考时按 15Hz 算速度/时长.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate

from rl_rebuild.correction import clips, frames as F
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R
from tasks.recon_kailang.bottle_reconstruction.env import BottleReconstructionEnv


@configclass
class UnscrewTaskCfg(DexmateCorrectionEnvCfg):
    # ---- 动作/观测 ----
    action_space = 29
    # 基础 (enable_pointcloud=False, 表面点并入 obs, 走 pregrasp 验证过的路径):
    #   本体58 + 腕13 + 物体13 + 参考42 + 相位4 + 瓶接触5 + 力矩7 + dyn_res 1
    #   + 动作29 + 累积残差22 + 点云 3×64=192  = 386
    # 任务层 +10: 帽位置(腕系)3 + 拧动进度1 + engaged 1 + 帽接触5
    # ⚠ 依赖 RL_ACC_FINGER 默认开 / lam_goal=0 (下面显式关) —— env init 有断言.
    observation_space = 396
    enable_pointcloud = False         # 点云并入 obs (无 PointNet 分支, 冠军 run 同款)
    lam_goal = 0.0                    # 终点势没有意义: 瓶身本来就不该动
    settle_steps = 5                  # 场景静态稳定 (实测 27/27), 不需要长静置
    rsi_prob = 0.0
    grasp_only = False
    hand_side = "right"               # 会被 clip 的 robot_hand 确认 (而不是覆盖)

    # ---- 预抓姿 (IK 目标) ----
    # 腕高 = 帽顶 + cap_clearance + hand_drop. 手指向下时**指尖垂在腕下方 20.3cm**
    # (2026-08-25 pk7 实测, 默认张开手型), 只按掌心 9cm 定高会把指尖插进瓶颈 9cm,
    # 每次 reset 都是去穿透爆炸 (实测瓶身被顶起 2.7cm/倾 30° -> knock 终止循环).
    cap_clearance_m = 0.015           # 张开手最低点(指尖)离帽顶的净空
    hand_drop_m = 0.203               # 腕 -> 最低指尖的垂距 (指尖朝下姿态, 实测)
    yaw_candidates = 12               # 绕竖轴扫多少个候选朝向, 取 IK 误差最小

    # ---- 奖励 ----
    # 比例设计 (证伪信号 U4): 任务核总量 = k_screw·4π ≈ 37.7 + 成功 20;
    # 引导总量 = 接近势 ~0.5 + 接触年金 ≤ 0.05·~180 ≈ 9 —— 引导 < 任务核/2.
    # v1 "左手扶瓶"抽象: 每控制步把瓶身写回初始位姿 (基类 settle 冻结同一手法).
    # 依据: 瓶的侧推倾覆临界力只有 ~0.94N (mg·r_base=0.169Nm / 帽高杠杆 0.18m),
    # 指尖一碰必倒 —— Unscrew1 实测 knock_frac=100%. 台账 U5 算错了对象:
    # 桌面摩擦 19× 余量是抗**扭**, 不是抗**倾覆**. 源视频里左手全程持瓶,
    # 数据一开始就给了答案. 左手主动协同留作 v2 课程.
    clamp_body = True
    k_screw = 3.0                     # 每 rad 拧动的势差分系数
    k_approach = 10.0                 # 指尖->帽 距离势差分 (每步钳 ±2cm -> ±0.2)
    w_cap_contact = 0.05              # ≥2 指尖接触帽的每步小额
    w_disturb = 5.0                   # 瓶身位移棘轮罚 (只罚新增位移)
    success_bonus = 20.0              # detach (拧满 2 圈释放) 一次性
    term_body_xy = 0.10               # m; 瓶身水平位移超过它终止 (打飞)
    term_body_tilt_deg = 30.0         # 瓶身倾倒终止
    # ---- U34 (2026-08-27 用户裁定 "三根手指慢慢拧, 需要找准位置") ----
    # 原 drive_mask = 任意 1 指接触即**全速**拧 -> 单根拇指戳一下就能拧完
    # 全程 (pk22: 触盖 99.8% 是拇指, 食/中在 8-10cm 外), 制度上不需要三指.
    # 改为拧转速率 ∝ 拇/食/中三指里实际接触的根数 (1 指 1/3 速, 3 指全速).
    # 关键: **无悬崖** —— 单指仍拧得动, 只是慢; 每多一根指 r_screw 连续变多,
    # "先松开拇指才能转腕"的收入山谷被填平 (加档治不好的原因, 见 pk22).
    screw_triad_drive = False         # True: 速率按 triad 接触指数分级
    screw_drive_floor = 1.0 / 3.0     # 1 根 triad 指时的速率下限


class UnscrewTaskEnv(BottleReconstructionEnv):
    """右手拧开预合拢的 PCO-1810 瓶盖."""

    cfg: UnscrewTaskCfg

    def __init__(self, cfg: UnscrewTaskCfg, render_mode=None, **kwargs):
        entry = clips.clip_entry(cfg.clip_name)
        assembly = (entry.get("secondary") or {}).get("assembly") or {}
        if assembly.get("mode") != "preengaged":
            raise ValueError(
                f"{cfg.clip_name!r} 不是 preengaged 螺旋 clip, 扭盖任务起点必须是合拢帽")
        if entry.get("robot_hand") != "right":
            raise ValueError(f"{cfg.clip_name!r} 需要 robot_hand='right' (任务手)")
        if entry.get("place_mode") != "object_only":
            raise ValueError("扭盖任务依赖 object_only 参考契约 (见 loader robot_hand 注释)")
        super().__init__(cfg, render_mode, **kwargs)
        assert self.cfg.hand_side == "right", "robot_hand 未生效"
        assert self.cfg.accumulate_finger, "任务假设累积手指残差开启 (RL_ACC_FINGER)"
        assert not self.use_goal, "lam_goal 必须为 0 (obs 记账依赖)"
        if self.screw_spec is None:
            raise ValueError("screw_spec 缺失")
        self._max_angle = 2.0 * np.pi * float(self.screw_spec.turns)

        N, dev = self.num_envs, self.device
        # 夹持写回用的常量缓冲 (瓶身初始位姿 + 零速度)
        self._body_hold_pose = torch.cat([
            self.obj_init_pos.expand(N, 3) + self.scene.env_origins,
            self.obj_init_quat.expand(N, 4)], dim=1).contiguous()
        self._body_hold_vel = torch.zeros(N, 6, device=dev)
        self.prev_screw = torch.zeros(N, device=dev)
        self.prev_tip_cap = torch.zeros(N, device=dev)
        self.prev_body_xy = torch.zeros(N, device=dev)
        self.released_latch = torch.zeros(N, dtype=torch.bool, device=dev)
        self._released_step = torch.zeros(N, dtype=torch.bool, device=dev)
        self._knock = torch.zeros(N, dtype=torch.bool, device=dev)
        self._defer_obs_check = False

        self._build_cap_pregrasp()

    # ---- 场景: 追加帽接触传感器 --------------------------------------
    def _setup_scene(self):
        super()._setup_scene()
        self._cap_sensors = []
        for i, name in enumerate(self.cfg.fingertip_bodies):
            scfg = ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{name}",
                history_length=1,
                filter_prim_paths_expr=["/World/envs/env_.*/Cap"],
            )
            s = ContactSensor(scfg)
            self._cap_sensors.append(s)
            self.scene.sensors[f"cap_contact_{i}"] = s

    def _cap_contacts(self) -> torch.Tensor:
        """(N,5) 指尖-帽接触力是否超阈."""
        f = torch.cat([s.data.force_matrix_w.view(self.num_envs, 1, 3)
                       for s in self._cap_sensors], dim=1)
        return (f.norm(dim=-1) > self.cfg.contact_force_thresh).float()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        super()._pre_physics_step(actions)
        if self.cfg.clamp_body:
            # "左手扶瓶": 瓶身整回合钉在初始位姿 (settle 冻结的手法, 只是不限时段).
            self.object.write_root_pose_to_sim(self._body_hold_pose)
            self.object.write_root_velocity_to_sim(self._body_hold_vel)

    def apply_screw_constraint(self, extra_cap_torque_local=None, *,
                               integrate_angle: bool = True, drive_mask=None):
        # 螺纹静摩擦抽象: 只有指尖真接触帽时才允许螺旋坐标转动.
        # 没有它, 轻擦一下的自由惯性能免费转完 2 圈 (首训 Unscrew0 实测:
        # release=41.5% 而 cap_contact_any 仅 2.7% —— 全是空转, 见台账 U6/U7).
        if drive_mask is None:
            capc = self._cap_contacts()
            if self.cfg.screw_triad_drive:
                # U34: 连续增益 (env.py 里 drive_mask 是**乘性**因子, 传浮点即可).
                # 只认拇/食/中 (fingertip_bodies[:3]); 环/小扒盖驱动不了螺纹.
                n_tri = capc[:, :3].sum(dim=1)
                drive_mask = torch.where(
                    n_tri > 0,
                    (n_tri / 3.0).clamp(min=self.cfg.screw_drive_floor, max=1.0),
                    torch.zeros_like(n_tri))
            else:
                drive_mask = capc.sum(dim=1) >= 1
        super().apply_screw_constraint(
            extra_cap_torque_local, integrate_angle=integrate_angle,
            drive_mask=drive_mask)

    # ---- 预抓姿: 把 object_only 的站姿参考换成帽上方的 IK 解 ----------
    def _build_cap_pregrasp(self):
        cfg = self.cfg
        cap_p = np.asarray(self.cap_init_pose_np[:3], dtype=float)   # env 局部系
        cap_verts = F.load_obj_verts(self.cap_entry["mesh"])
        cap_top = cap_p[2] + float(cap_verts[:, 2].max())            # 帽直立, 局部 z∈[0,h]
        # 手系 +z 从腕指向指尖; 顶抓 => 手 z = -世界 z, 指尖下垂环绕帽周,
        # 卷指即可裹帽, 且 j7 滚转轴恰为竖直 (= 拧盖轴).
        wrist_tgt = np.array([cap_p[0], cap_p[1],
                              cap_top + cfg.cap_clearance_m + cfg.hand_drop_m])

        # object_only 路径不解 IK, 也不设 _anchor_T —— 这里按 ref_builder 分支的
        # 同一配方现量: 锚在**实测 arm_center** 上, 不假设躯干停在配置角度.
        bn = list(self.hand.body_names)
        org = self.scene.env_origins[0].cpu().numpy()
        ac = bn.index("arm_center")
        anchor_T = np.eye(4)
        anchor_T[:3, :3] = quat_to_R(self.hand.data.body_quat_w[0, ac].cpu().numpy())
        anchor_T[:3, 3] = self.hand.data.body_pos_w[0, ac].cpu().numpy() - org
        self._anchor_T = anchor_T
        ik = ArmIK("right", anchor_link="arm_center", anchor_T=anchor_T)

        j7_mid = 0.5 * (ik.lower[6] + ik.upper[6])
        best = None
        for psi in np.linspace(0.0, 2.0 * np.pi, cfg.yaw_candidates, endpoint=False):
            x = np.array([np.cos(psi), np.sin(psi), 0.0])
            z = np.array([0.0, 0.0, -1.0])
            R = np.column_stack([x, np.cross(z, x), z])
            sol = ik.solve(wrist_tgt, R, q0=ik.q_default)
            # 位置优先; j7 居中加小分 (留出双向滚转余量 —— 拧盖的主自由度)
            score = (sol["pos_err"] + 0.05 * sol["rot_err"]
                     + 0.02 * abs(float(sol["q"][6]) - j7_mid)
                     + (0.0 if sol["ok"] else 10.0))
            if best is None or score < best[0]:
                best = (score, sol, R, psi)
        _, sol, R_best, psi = best
        if sol["pos_err"] > 0.02:
            raise ValueError(
                f"帽预抓姿 IK 不可达: pos_err={sol['pos_err']*100:.1f}cm "
                f"(帽位 {cap_p.round(3).tolist()}) —— 检查摆放/reach")

        dev = self.device
        q = torch.tensor(sol["q"], dtype=torch.float32, device=dev)
        self.q_ref[:] = q                                            # (L,7) 常量参考
        self.ref_wrist_pos[:] = torch.tensor(wrist_tgt, dtype=torch.float32, device=dev)
        self.ref_wrist_quat[:] = torch.tensor(
            F.rotmat_to_quat(R_best), dtype=torch.float32, device=dev)
        # open/closed 保持默认手型 (object_only 已设), 手指全权交给累积残差
        j7_deg = float(np.degrees(sol["q"][6]))
        j7_lo, j7_hi = np.degrees(ik.lower[6]), np.degrees(ik.upper[6])
        print(f"[unscrew] 预抓姿: 腕 {np.round(wrist_tgt, 3).tolist()} yaw={np.degrees(psi):.0f}° "
              f"| IK err {sol['pos_err']*100:.2f}cm/{np.degrees(sol['rot_err']):.1f}° "
              f"| j7={j7_deg:.0f}° (限位 [{j7_lo:.0f},{j7_hi:.0f}]) "
              f"| 帽顶 z={cap_top:.3f} 指尖净空 {cfg.cap_clearance_m*100:.1f}cm")

    # ---- 终止: 成功(释放) / 打翻瓶身 ---------------------------------
    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        released = self.screw_has_depth & ~self.screw_engaged
        self._released_step = released & ~self.released_latch
        self.released_latch |= released

        origins = self.scene.env_origins
        obj_pos = self.object.data.root_pos_w - origins
        body_xy = (obj_pos - self.obj_init_pos)[:, :2].norm(dim=1)
        up = quat_apply(self.object.data.root_quat_w,
                        torch.tensor([0.0, 0.0, 1.0], device=self.device
                                     ).expand(self.num_envs, 3))
        tilted = up[:, 2] < float(np.cos(np.radians(self.cfg.term_body_tilt_deg)))
        self._knock = (body_xy > self.cfg.term_body_xy) | tilted
        return terminated | self._knock | self.released_latch, truncated

    # ---- 奖励 --------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        origins = self.scene.env_origins
        active = self.episode_length_buf >= cfg.settle_steps

        cap_pos = self.cap.data.root_pos_w - origins
        tips = self.tip_pos_w - origins[:, None]
        d_cap = (tips - cap_pos[:, None]).norm(dim=-1).min(dim=1).values
        capc = self._cap_contacts()
        n_cap = capc.sum(dim=1)
        obj_pos = self.object.data.root_pos_w - origins
        body_xy = (obj_pos - self.obj_init_pos)[:, :2].norm(dim=1)

        # 刚转入 active 的 env: 用当前值初始化势差分状态 (避免一步假分)
        newly = active & (self.episode_length_buf == cfg.settle_steps)
        if newly.any():
            self.prev_tip_cap[newly] = d_cap[newly]
            self.prev_screw[newly] = self.screw_angle[newly]
            self.prev_body_xy[newly] = body_xy[newly]

        r_app = cfg.k_approach * (self.prev_tip_cap - d_cap).clamp(-0.02, 0.02)
        self.prev_tip_cap.copy_(d_cap)
        dtheta = self.screw_angle - self.prev_screw
        self.prev_screw.copy_(self.screw_angle)
        r_screw = cfg.k_screw * dtheta
        r_contact = cfg.w_cap_contact * (n_cap >= 2).float()
        # 棘轮罚: 只罚**新增**的瓶身位移 (总额有界 = w_disturb × term_body_xy)
        r_disturb = -cfg.w_disturb * (body_xy - self.prev_body_xy).clamp(min=0.0)
        self.prev_body_xy = torch.maximum(self.prev_body_xy, body_xy)
        r_succ = cfg.success_bonus * self._released_step.float()

        total = (r_app + r_screw + r_contact + r_disturb) * active.float() + r_succ

        # ---- 诊断 (窗口=完整回合, 见 DESIGN_LOOP §1) ----
        d = self.diag
        d.tick(active)
        d.add("cap_contact2_frac", (n_cap >= 2).float(), mask=active)
        d.add("cap_contact_any_frac", (n_cap >= 1).float(), mask=active)
        d.add("screw_deg", torch.rad2deg(self.screw_angle), mode="max")
        d.add("release", self.released_latch.float(), mode="max")
        d.add("tip_cap_cm", d_cap * 100.0, mask=active)
        d.add("body_disp_cm", (body_xy * 100.0).clamp(max=100.0), mode="max")
        d.add("knock_frac", self._knock.float(), mode="max")
        d.add("rew_screw", r_screw, mask=active)
        d.add("rew_approach", r_app, mask=active)
        d.add("rew_contact", r_contact, mask=active)
        d.add("rew_disturb", r_disturb, mask=active)
        self.extras.update(d.publish())
        return total

    # ---- 观测: 基类 386 + 任务 10 ------------------------------------
    def _get_observations(self) -> dict:
        self._defer_obs_check = True
        try:
            out = super()._get_observations()
        finally:
            self._defer_obs_check = False
        origins = self.scene.env_origins
        wrist_pos = self.wrist_pos_w - origins
        q_inv = quat_conjugate(self._qsign(self.wrist_quat_w))
        cap_pos = self.cap.data.root_pos_w - origins
        extra = torch.cat([
            quat_apply(q_inv, cap_pos - wrist_pos),                    # 3 帽位置(腕系)
            (self.screw_angle / self._max_angle).unsqueeze(1),         # 1 拧动进度
            self.screw_engaged.float().unsqueeze(1),                   # 1
            self._cap_contacts(),                                      # 5
        ], dim=1).clamp(-self.cfg.clip_obs, self.cfg.clip_obs).nan_to_num(0.0)
        obs = torch.cat([out["policy"], extra], dim=1)
        self._check_obs_dim(obs)
        out["policy"] = obs
        return out

    def _check_obs_dim(self, obs: torch.Tensor) -> None:
        if self._defer_obs_check:
            return                       # 基类拼完是 386, 任务层追加后才是 396
        super()._check_obs_dim(obs)

    # ---- 复位 --------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        ids = self.cap._ALL_INDICES if env_ids is None else env_ids
        super()._reset_idx(ids)
        self.prev_screw[ids] = 0.0
        self.prev_tip_cap[ids] = 0.0
        self.prev_body_xy[ids] = 0.0
        self.released_latch[ids] = False
        self._released_step[ids] = False
        self._knock[ids] = False
