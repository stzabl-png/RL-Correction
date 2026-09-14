"""Clean/3 Stage-2 交互段 env —— 最简六项配方 (台账 §5.7; 用户 2026-09-08 批准, 三项默认: 认证不带接触条件 / 全转角 / 先单 seed).

回合 (20Hz):
  行 0..release      : 手在母带 v1 第 0 行 grasp 位, 两物体钉在名义位姿; 指 grasp→squeeze 斜坡 K_CLOSE 行后常量; 臂前馈第 0 行 (+下垂补偿)。
  放手 (release_row)  : 锁存两物体在各自掌系的位姿 (rel_p0/rel_q0)。
  认证                : 连续 S2_CERT_STEPS 步 两物体 相对锁存 位移<S2_CERT_POS ∧ 全转角<S2_CERT_ROT ⇒ +B_CERT, 时钟 k 从 0 开始走。
                        放手后 S2_CERT_BUDGET 行内未认证 ⇒ 截断 (不另罚)。
  交互 (k)            : 臂前馈播放 ref_arm[k] + 残差; 海绵在**盘规范系**的位姿与 psc_ref[k] 的差在门内 (法向 GATE_N / 面内 GATE_XY) ⇒ k+1, adv +R_ADV;
                        超出该行置信档皮筋 ⇒ leash 罚 (只在认证后)。第一版不做在线重锚。
  A4 (L3 progress_batch 判据, 不借其奖励): 覆盖≥0.45 ∧ 行程≥0.70m ∧ 盘倾角<15° ∧ 盘偏离第 k 行参考<3cm ∧ 双物在手 ⇒ +B_SUCC 一次 (锁存, 不终止)。
  结束 = 时钟走完 (截断) | 死线 (终止, B_DIE): 任一物体相对锁存 >S2_DIE_POS/S2_DIE_ROT | 盘倾角 >30° | 盘偏离参考 >10cm | 物体落桌 | 手掌/指垫低于桌面。
奖励 = adv + leash + 认证 + 成功 + 死亡 + 动作罚。无接触奖/窗内奖/渐进罚/交叉罚 (交叉改残差硬钳, cross_frac 只诊断)。
观测 = 掌系, 无放手倒计时 (见 _get_observations)。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_L3 = os.path.abspath(os.path.join(_HERE, "..", "A_Design", "L3_Learning"))
sys.path.insert(0, _L3)
import clean_env as CE  # noqa: E402
import hold_env as HE  # noqa: E402
import task_config as TC  # noqa: E402
from clean_env import SIDES, CleanHoldEnv  # noqa: E402
from hold_env import _qangle  # noqa: E402
from progress_batch import CleanGeometry, CleanProgressBatch, CleanSignals  # noqa: E402  (Clean3 L3 判据单一来源)

ACT_DIM = 58
PRIV_DIM = 22
_NLOOK = len(TC.S2_LOOKAHEAD)
# 观测维: 臂 q/qd 28 + 指 q/qd 88 + 累积残差 58 + 每手 [rp3 rq4 dvec6 v_rel3 w_rel3 g3]=22×2 + 盘系海绵 [psc3 gap1 contact1 err3 look 3×N] + 时钟 [k/T, gate_ok, tier3, certified, pinned] 7
#      + 垫力 10 + 触发 10 + 前馈指姿 44 + 上步动作 58
OBS_DIM = 28 + 88 + 58 + 44 + (5 + 3 + 3 * _NLOOK) + 7 + 20 + 44 + 58
T_ROWS_REF = None   # 由母带定 (424)


def build_cfg(num_envs=1, reference=None):
    reference = reference or TC.REFERENCE
    cfg = HE.build_cfg(num_envs, reference=reference)
    z = np.load(reference, allow_pickle=True)
    t_ref = int(z["right_q"].shape[0])
    cfg.action_space = ACT_DIM
    cfg.observation_space = OBS_DIM
    t_ep = int(TC.RELEASE_MAX) + int(TC.S2_CERT_BUDGET) + int(round(TC.S2_CLOCK_SLACK * t_ref))
    cfg.episode_length_s = float((t_ep + 2) / TC.CONTROL_HZ)
    cfg.clean_s2_t_ep = t_ep
    return cfg


class CleanTaskEnv(CleanHoldEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        N, dev, to = self.num_envs, self.device, self.to
        self.T_EP = int(getattr(cfg, "clean_s2_t_ep", TC.T_EP))
        self.T_REF = int(self.ref_arm.shape[0])
        # ---- 残差界: 臂用 Pour 标定形状 (cfg.arm_residual_max × arm_step_scale), 累积上限 S2_ARM_DEV; 指沿 Stage-1 ----
        arm_max = getattr(cfg, "arm_residual_max", None)
        if arm_max is not None and np.asarray(arm_max).size == 7:
            arm_step = to(np.asarray(arm_max, np.float64) * float(getattr(cfg, "arm_step_scale", 0.25)))
            arm_step = torch.cat([arm_step, arm_step])
            src = "cfg.arm_residual_max×arm_step_scale"
        else:
            arm_step = to([TC.ARM_STEP] * 14); src = "均一 ARM_STEP (缺标定!)"
        self.step_sz = torch.cat([arm_step, to([TC.FIN_STEP] * 44)])
        self.dev_hi = to([TC.S2_ARM_DEV] * 14 + [TC.FIN_DEV] * 44)
        # 握形合法性: 指定关节残差硬钳 (无罚项)
        from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER as _GJO
        clamped = []
        for s, names in TC.S2_CLAMP_JOINTS.items():
            for jn in names:
                j = [n.replace("right_", f"{s}_") for n in _GJO].index(jn)
                col = 14 + (0 if s == "right" else 22) + j          # 动作布局: 臂14 + R指22 + L指22
                self.step_sz[col] = 0.0; self.dev_hi[col] = 0.0; clamped.append(jn)
        print(f"[CleanTaskEnv] 臂每步界 {src}: R={np.round(arm_step[:7].cpu().numpy(), 4).tolist()} 累积≤{TC.S2_ARM_DEV}; 硬钳 {clamped}")
        # ---- 母带派生表: 海绵在盘规范系的参考位姿 psc_ref (T,3), 档位 tier_row (T,), 盘参考位置 ref_pos[0] ----
        # 判据几何按母带走 (盘剖面/海绵面偏移); 老母带缺键则退回 take3 默认 (逐位不变)
        self.geo = CleanGeometry.from_reference(self._z)
        print(f"[CleanTaskEnv] 判据几何: 盘顶面剖面 y∈[{min(self.geo.prof_y)*100:.2f},{max(self.geo.prof_y)*100:.2f}]cm "
              f"({len(self.geo.prof_r)} 档) | 海绵擦盘面偏移 {self.geo.sponge_face_offset*100:.2f}cm")
        S_ref = CleanSignals(self.T_REF, dev, self.q_ci[0], self.geo)
        sig_ref = S_ref(self.ref_pos[0], self.ref_quat[0], self.ref_pos[1], self.ref_quat[1])
        self.psc_ref = sig_ref["sponge_in_plate"].clone()                    # (T,3)
        conf = np.minimum(np.asarray(self._z["confidence_0"], np.float64), np.asarray(self._z["confidence_1"], np.float64))
        tier = np.where(conf >= TC.S2_TIER_HI, 2, np.where(conf >= TC.S2_TIER_LO, 1, 0))
        # ★A2 消融: 拍平置信度分档 —— 每行一律黄档。皮筋随之常量 0.05, 观测里的档位独热也变常量
        #   (obs 读的是 tier_row[k], 拍平这一处就同时管住奖励与观测两条路)。
        self._conf_flat = bool(TC.S2_CONF_FLAT)
        if self._conf_flat:
            tier = np.ones_like(tier)
            print("[CleanTaskEnv] ★CLEAN_S2_CONF_FLAT=1: 置信度分档拍平为全黄档 (皮筋常量 "
                  f"{TC.S2_LEASH_XY[1]}, 观测档位独热恒定)")
        self.tier_row = torch.tensor(tier, dtype=torch.long, device=dev)
        self.leash_xy_row = to([TC.S2_LEASH_XY[int(t)] for t in tier])
        print(f"[CleanTaskEnv] 母带 {self.T_REF} 行: 档位 绿{int((tier == 2).sum())}/黄{int((tier == 1).sum())}/红{int((tier == 0).sum())}; "
              f"psc_ref z∈[{self.psc_ref[:, 2].min()*100:.2f},{self.psc_ref[:, 2].max()*100:.2f}]cm; T_EP={self.T_EP}")
        # ★Base 消融: 人手指姿指引 —— 只进指列, 增量焊在母带 squeeze 姿上, 只在认证后生效。
        self.hand_ref = bool(TC.S2_HAND_REF)
        self.hand_f_delta = None
        if self.hand_ref:
            assert "human_right_f" in self._z, (
                "CLEAN_S2_HAND_REF=1 需要母带含 human_right_f/human_left_f —— "
                "用 build_reference.py 重造一条 (默认就写, 除非加了 --no_hand_ref)")
            _d = np.concatenate([np.asarray(self._z["human_right_f"], np.float64) - np.asarray(self._z["right_f"], np.float64),
                                 np.asarray(self._z["human_left_f"], np.float64) - np.asarray(self._z["left_f"], np.float64)], 1)
            self.hand_f_delta = to(_d)                                   # (T,44)
            print(f"[CleanTaskEnv] ★CLEAN_S2_HAND_REF=1 (W={TC.S2_HAND_W}): 人手指姿增量 "
                  f"中位={np.degrees(np.abs(_d)).mean():.2f}° 最大={np.degrees(np.abs(_d)).max():.2f}°, "
                  f"首行={np.degrees(np.abs(_d[0])).max():.3f}° (应 0); 只进指列, 认证后生效")
        # ---- 判据 (L3) ----
        self.S = CleanSignals(N, dev, self.q_ci[0], self.geo)
        self.PBj = CleanProgressBatch(N, dev, self.S, self.geo)
        # ---- 逐 env 状态 ----
        self.k = torch.zeros(N, dtype=torch.long, device=dev)
        self.latched = torch.zeros(N, dtype=torch.bool, device=dev)
        self.certified = torch.zeros(N, dtype=torch.bool, device=dev)
        self.died = torch.zeros(N, dtype=torch.bool, device=dev)
        self.succ = torch.zeros(N, dtype=torch.bool, device=dev)
        self.clock_done = torch.zeros(N, dtype=torch.bool, device=dev)
        self.rel_p0 = {s: torch.zeros(N, 3, device=dev) for s in SIDES}
        self.rel_q0 = {s: torch.zeros(N, 4, device=dev) for s in SIDES}
        self.rel_max = torch.zeros(N, 4, device=dev)                         # [dp_L, dp_R, dr_L, dr_R] 回合内最大 (锁存后)
        self.die_kind = torch.zeros(N, dtype=torch.long, device=dev)         # 1 rel 2 tilt 3 plate_dev 4 drop 5 table
        self.gate_ok = torch.zeros(N, dtype=torch.bool, device=dev)
        self._ep = {kk: torch.zeros(N, device=dev) for kk in
                    ("r_adv", "r_leash", "r_bonus", "r_act", "r_soft", "n_gate", "n_ia", "contact", "cross_any", "cert_timeout", "n_push")}
        self._acc = {}
        # ---- 候选旗 ----
        self.clock_contact = bool(TC.S2_CLOCK_NEEDS_CONTACT)
        self.reanchor = bool(TC.S2_REANCHOR)
        self.soft_rel = bool(TC.S2_SOFT_REL)
        self.gate_hold = bool(TC.S2_GATE_HOLD)
        self.push = bool(TC.S2_PUSH)
        self.push_left = torch.zeros(N, 2, dtype=torch.long, device=dev)          # 剩余推力行数 [盘, 海绵]
        self.push_f = torch.zeros(N, 2, 3, device=dev)                            # 世界系力 (N)
        self._mass = {"left": 0.3, "right": 0.05}
        try:
            self._mass = {s: float(self.art[s].root_physx_view.get_masses()[0, 0]) for s in SIDES}
        except Exception:
            pass
        self.dq_re = torch.zeros(N, 7, device=dev)           # 右臂重锚修正 (rad, 积分)
        self._re_dx = torch.zeros(N, device=dev)             # 诊断: |e| m (盘面系海绵位置误差)
        self._jac_bid = self.hand_bid["right"] - (1 if self.hand.is_fixed_base else 0)
        self._arm_dofs_r = self.map_ids_t[:7]
        p_oh, q_oh = self.T_oh["right"]
        self.p_oh_r, self.q_oh_r = p_oh.clone(), q_oh.clone()
        print(f"[CleanTaskEnv] 候选旗: clock_needs_contact={self.clock_contact} reanchor={self.reanchor} soft_rel={self.soft_rel} (W={TC.S2_SOFT_W}) "
              f"gate_hold={self.gate_hold} push={self.push} (p={TC.S2_PUSH_P}, {TC.S2_PUSH_STEPS}行, {TC.S2_PUSH_F}×mg, m={ {k: round(v, 3) for k, v in self._mass.items()} }) "
              f"(fixed_base={self.hand.is_fixed_base}, jac_bid={self._jac_bid}, dq_max={TC.S2_REANCHOR_DQ_MAX})")
        self._low_ids = torch.tensor([self.hand_bid[s] for s in SIDES] + self.pad_ids["left"] + self.pad_ids["right"], device=dev)
        # 上一回合摘要 (在 _book 里写, 自动重置后仍可读 —— eval/record 用)
        self.last_ep = {kk: torch.zeros(N, device=dev) for kk in
                        ("cert", "success", "clock_frac", "coverage", "travel_cm", "relp_max_plate_cm", "relp_max_sponge_cm",
                         "relrot_max_plate_deg", "relrot_max_sponge_deg", "die_kind", "cross_frac", "ep_len", "release_row", "cert_timeout")}
        print(f"[CleanTaskEnv] N={N} obs={OBS_DIM} act={ACT_DIM} T_EP={self.T_EP} release={self.release_row_cur} "
              f"cert={TC.S2_CERT_POS*100:.1f}cm/{TC.S2_CERT_ROT_DEG:.0f}°×{TC.S2_CERT_STEPS} die={TC.S2_DIE_POS*100:.0f}cm/{TC.S2_DIE_ROT_DEG:.0f}° "
              f"gate n/xy={TC.S2_GATE_N*100:.0f}/{TC.S2_GATE_XY*100:.0f}cm")

    # ---------------------------------------------------------------- 工具 ----
    def _rel_pose(self, side):
        hp, hq = self._hand_pose(side); op, oq = self._obj_pose(side)
        return quat_apply(quat_conjugate(hq), op - hp), quat_mul(quat_conjugate(hq), oq), hq

    def _update_reanchor(self, sig):
        """海绵臂前馈重锚 (闭环, 负反馈): e = 海绵在盘面系的位置 − psc_ref[k]; 世界系 Δx = −KP·R_plate·e (带死区);
        Δq = DLS(J_wrist)·Δx 累加进 dq_re (钳 DQ_MAX)。只在认证后生效; 门/皮筋目标不动, 只动前馈。"""
        N, dev = self.num_envs, self.device
        e = sig["sponge_in_plate"] - self.psc_ref[self.k]                             # (N,3) 盘规范系
        e = torch.where(e.abs() < TC.S2_REANCHOR_DEADBAND, torch.zeros_like(e), e)
        # 抗饱和 (A3 冒烟: 接触后继续下压把海绵在手里顶转 14.6° 致死): 已接触时法向误差只保留 "压过头" 分量 (e_z<0 ⇒ 抬), 不再往下压
        e_z = torch.where(sig["contact"] & (e[:, 2] > 0), torch.zeros_like(e[:, 2]), e[:, 2])
        e = torch.cat([e[:, :2], e_z.unsqueeze(1)], 1)
        pp, pq = self._obj_pose("left")
        pqc = quat_mul(pq, quat_conjugate(self.q_ci[0].expand(N, 4)))                  # 盘规范系 → 世界
        dx_pos = -TC.S2_REANCHOR_KP * quat_apply(pqc, e)
        self._re_dx = torch.linalg.vector_norm(e, dim=1)
        active = self.certified & ~self.died
        if not active.any():
            return
        dx = torch.cat([dx_pos, torch.zeros(N, 3, device=dev)], 1)
        jac = self.hand.root_physx_view.get_jacobians()                                  # (N, links, 6, dofs)
        J = jac[:, self._jac_bid][:, :, self._arm_dofs_r]                               # (N,6,7)
        lam2 = TC.S2_REANCHOR_DAMP ** 2
        JJt = J @ J.transpose(1, 2) + lam2 * torch.eye(6, device=dev).unsqueeze(0)
        dqa = (J.transpose(1, 2) @ torch.linalg.solve(JJt, dx.unsqueeze(2))).squeeze(2).nan_to_num(0.0)
        self.dq_re = (self.dq_re + dqa * active.float().unsqueeze(1)).clamp(-TC.S2_REANCHOR_DQ_MAX, TC.S2_REANCHOR_DQ_MAX)

    # ---------------------------------------------------------------- RL 面 ----
    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)                       # 摆位/钉住/课程抽 release_row/Stage-1 标志/_book
        self.k[env_ids] = 0
        self.latched[env_ids] = False; self.certified[env_ids] = False; self.died[env_ids] = False
        self.succ[env_ids] = False; self.clock_done[env_ids] = False; self.gate_ok[env_ids] = False
        self.rel_max[env_ids] = 0.0; self.die_kind[env_ids] = 0
        for s in SIDES:
            self.rel_p0[s][env_ids] = 0.0; self.rel_q0[s][env_ids] = 0.0
        self.PBj.reset(env_ids, plate_ref_pos=self.ref_pos[0][0].expand(len(env_ids), 3))
        self.dq_re[env_ids] = 0.0; self._re_dx[env_ids] = 0.0
        self.push_left[env_ids] = 0; self.push_f[env_ids] = 0.0
        for kk in self._ep:
            self._ep[kk][env_ids] = 0

    def _ff_fingers(self, row):
        """指前馈 = Stage-1 的 grasp→squeeze 斜坡, 认证后再叠人手指姿增量 (Base 消融)。
        前馈与观测同走这一个出口, 不会出现"喂进去的和看到的不是一回事"。"""
        f = super()._ff_fingers(row)
        if self.hand_ref and self.hand_f_delta is not None and hasattr(self, "k"):
            f = f + TC.S2_HAND_W * self.hand_f_delta[self.k] * self.certified.float().unsqueeze(1)
        return f

    def _pre_physics_step(self, actions):
        self.last_act = actions.clamp(-1.0, 1.0)
        self.cum_res = torch.maximum(torch.minimum(self.cum_res + self.last_act * self.step_sz, self.dev_hi), -self.dev_hi)
        row = self.episode_length_buf
        arm = self.ref_arm[self.k] + self.arm_sag + self.cum_res[:, :14]          # 认证前 k=0 ⇒ 第 0 行
        if self.reanchor:
            pp, pq = self._obj_pose("left"); sp, sq = self._obj_pose("right")
            self._update_reanchor(self.S(pp, pq, sp, sq))
            arm = arm.clone(); arm[:, :7] = arm[:, :7] + self.dq_re
        fin = self._ff_fingers(row) + self.cum_res[:, 14:]
        self.q_cmd[:, self.act_ids] = torch.cat([arm, fin], 1)
        self._pinned_ids = torch.nonzero(row < self.release_row).flatten()
        if self.push:                             # 认证后随机推力: 每步以 S2_PUSH_P 起一次, 持续 S2_PUSH_STEPS 行
            N, dev = self.num_envs, self.device
            self.push_left = (self.push_left - 1).clamp(min=0)
            for j, s in enumerate(SIDES):
                start = self.certified & ~self.died & (self.push_left[:, j] == 0) & (torch.rand(N, device=dev) < TC.S2_PUSH_P)
                if start.any():
                    d = torch.randn(N, 3, device=dev); d = d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)
                    mag = (TC.S2_PUSH_F[0] + (TC.S2_PUSH_F[1] - TC.S2_PUSH_F[0]) * torch.rand(N, device=dev)) * self._mass[s] * 9.81
                    self.push_f[start, j] = (d * mag.unsqueeze(1))[start]
                    self.push_left[start, j] = int(TC.S2_PUSH_STEPS)
                    self._ep["n_push"][start] += 1

    def _apply_action(self):
        super()._apply_action()                   # 关节目标 + 钉住
        if self.push:                             # 外力写入 (每物理子步; 无推力时写零)
            for j, s in enumerate(SIDES):
                on = (self.push_left[:, j] > 0).float().unsqueeze(1)
                self.art[s].set_external_force_and_torque((self.push_f[:, j] * on).unsqueeze(1),
                                                          torch.zeros(self.num_envs, 1, 3, device=self.device))

    def _get_dones(self):
        N, dev = self.num_envs, self.device
        row = self.episode_length_buf                     # 已 +1
        released = row > self.release_row
        # ---- 掌系相对位姿 + 放手锁存 ----
        newrel = released & ~self.latched
        dp, dr, rel = {}, {}, {}
        for s in SIDES:
            rp, rq, hq = self._rel_pose(s)
            if newrel.any():
                self.rel_p0[s][newrel] = rp[newrel]; self.rel_q0[s][newrel] = rq[newrel]
            rel[s] = (rp, rq, hq)
        self.latched |= newrel
        for s in SIDES:
            rp, rq, _ = rel[s]
            dp[s] = torch.linalg.vector_norm(rp - self.rel_p0[s], dim=1) * self.latched.float()
            dr[s] = _qangle(rq, self.rel_q0[s]) * self.latched.float()
        self.rel_max = torch.maximum(self.rel_max, torch.stack([dp["left"], dp["right"], dr["left"], dr["right"]], 1))
        within = torch.stack([(dp[s] < TC.S2_CERT_POS) & (dr[s] < np.radians(TC.S2_CERT_ROT_DEG)) for s in SIDES], 1).all(1)
        # ---- 认证 ----
        cert_now = released & ~self.certified & within
        self.cert_run = torch.where(cert_now, self.cert_run + 1, torch.zeros_like(self.cert_run))
        new_cert = (self.cert_run >= TC.S2_CERT_STEPS) & ~self.certified & released
        self.certified |= new_cert
        cert_timeout = released & ~self.certified & (row > self.release_row + TC.S2_CERT_BUDGET)
        # ---- 盘规范系信号 + 时钟 ----
        pp, pq = self._obj_pose("left"); sp, sq = self._obj_pose("right")
        sig = self.S(pp, pq, sp, sq)
        k = self.k
        err = sig["sponge_in_plate"] - self.psc_ref[k]
        e_n, e_xy = err[:, 2].abs(), torch.linalg.vector_norm(err[:, :2], dim=1)
        self.gate_ok = self.certified & (e_n < TC.S2_GATE_N) & (e_xy < TC.S2_GATE_XY)
        can = self.gate_ok & (k < self.T_REF - 1) & ~self.died
        if self.clock_contact:
            can = can & sig["contact"]
        if self.gate_hold:                        # 推进还要求握持不变形 (两物体在认证窗内)
            can = can & within
        self.k = torch.where(can, k + 1, k)
        r_adv = TC.S2_R_ADV * can.float()
        lp = self.leash_xy_row[k]
        r_leash = -(((e_xy - lp).clamp(min=0) / lp) + ((e_n - TC.S2_LEASH_N).clamp(min=0) / TC.S2_LEASH_N)).clamp(max=3.0) \
            * self.certified.float()
        # ---- A4 判据 (L3), 盘参考 = 第 k 行 ----
        z_low = torch.stack([self._obj_pose(s)[0][:, 2] < self.cfg.table_top_z + 0.03 for s in SIDES], 1).any(1)
        self.PBj.plate_ref = self.ref_pos[0][self.k]
        out = self.PBj.step(sig, pp, z_low | self.died)
        new_succ = out["new_gate"][:, 3] & self.certified & ~self.died
        self.succ |= new_succ
        # ---- 死线 ----
        die_rel = self.latched & ((torch.stack([dp[s] for s in SIDES], 1) > TC.S2_DIE_POS).any(1)
                                  | (torch.stack([dr[s] for s in SIDES], 1) > np.radians(TC.S2_DIE_ROT_DEG)).any(1))
        die_tilt = released & (sig["plate_tilt"] > np.radians(TC.S2_PLATE_TILT_DIE_DEG))
        die_pdev = self.certified & (torch.linalg.vector_norm(pp - self.ref_pos[0][self.k], dim=1) > TC.S2_PLATE_DEV_DIE)
        die_drop = released & z_low
        die_table = (self.hand.data.body_pos_w[:, self._low_ids, 2] < self.cfg.table_top_z - 0.005).any(1)
        kinds = [die_rel, die_tilt, die_pdev, die_drop, die_table]
        new_die = torch.zeros(N, dtype=torch.bool, device=dev)
        for i, dk in enumerate(kinds):
            nd = dk & ~self.died & ~new_die
            self.die_kind[nd] = i + 1; new_die |= nd
        self.died |= new_die
        self.clock_done = self.certified & (self.k >= self.T_REF - 1)
        timeout = row >= self.T_EP
        terminated = self.died.clone()
        truncated = (self.clock_done | cert_timeout | timeout) & ~terminated
        # ---- 奖励 (五项 + 动作罚) ----
        r_bonus = TC.S2_B_CERT * new_cert.float() + TC.S2_B_SUCC * new_succ.float() + TC.S2_B_DIE * new_die.float()
        r_act = -TC.S2_W_ACT * (self.last_act ** 2).mean(1)
        r_soft = torch.zeros(N, device=dev)
        if self.soft_rel:      # ★H-C3: 相对位姿软罚 (认证后), 1cm/5° 起渐进到死线 3cm/20° 满值
            terms = []
            for s in SIDES:
                # 跨度用 S2_SOFT_SPAN_* (恒 3cm/20°), 不跟死线走 —— 放宽死线时罚的斜率保持原样
                tp = (dp[s] - TC.S2_CERT_POS) / max(TC.S2_SOFT_SPAN_POS - TC.S2_CERT_POS, 1e-6)
                tr = (dr[s] - np.radians(TC.S2_CERT_ROT_DEG)) / max(np.radians(TC.S2_SOFT_SPAN_ROT_DEG - TC.S2_CERT_ROT_DEG), 1e-6)
                terms.append(torch.maximum(tp, tr).clamp(0.0, 1.0))
            r_soft = -TC.S2_SOFT_W * torch.stack(terms, 1).mean(1) * self.certified.float() * (~self.died).float()
        reward = r_adv + r_leash + r_bonus + r_act + r_soft
        _, cross_any = self._cross_penalty()
        for kk, v in (("r_adv", r_adv), ("r_leash", r_leash), ("r_bonus", r_bonus), ("r_act", r_act), ("r_soft", r_soft),
                      ("n_gate", self.gate_ok.float()), ("n_ia", self.certified.float()),
                      ("contact", (sig["contact"] & self.certified).float()), ("cross_any", cross_any.float()),
                      ("cert_timeout", cert_timeout.float())):
            self._ep[kk] += v
        F = {s: self._pad_force_mat(s) for s in SIDES}
        self._tick = dict(reward=reward, released=released, within=within, dp=dp, dr=dr, F=F, rel=rel, sig=sig, out=out,
                          e_n=e_n, e_xy=e_xy, new_cert=new_cert, new_succ=new_succ, new_die=new_die, cert_timeout=cert_timeout,
                          timeout=timeout, success=self.succ.clone(), plate_ok=sig["plate_tilt"] < np.radians(15.0),
                          sponge_ok=sig["contact"],
                          # 逐步账目快照 (record_task.py 落盘用; 只读, 不影响行为)
                          r_adv=r_adv, r_leash=r_leash, r_bonus=r_bonus, r_act=r_act, r_soft=r_soft, can=can, gate_ok=self.gate_ok.clone(),
                          k=self.k.clone(), row=row.clone(), release_row=self.release_row.clone(), certified=self.certified.clone(),
                          died=self.died.clone(), die_kind=self.die_kind.clone(), cross_any=cross_any, dq_re=self.dq_re.clone())
        return terminated, truncated

    def _get_rewards(self):
        return self._tick["reward"]

    def _get_observations(self):
        N, dev = self.num_envs, self.device
        row = self.episode_length_buf
        q = self.hand.data.joint_pos; qd = self.hand.data.joint_vel
        fin_ids = self.act_ids[14:]
        blocks = [q[:, self.map_ids_t], qd[:, self.map_ids_t] * 0.1, q[:, fin_ids], qd[:, fin_ids] * 0.1, self.cum_res / self.dev_hi]
        g_w = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(N, 3)
        devs, forces = [], []
        for s in SIDES:
            rp, rq, hq = self._rel_pose(s)
            dq = quat_mul(quat_conjugate(self.rel_q0[s]), rq)
            dvec = torch.cat([rp - self.rel_p0[s], 2.0 * dq[:, 1:] * torch.sign(dq[:, :1])], 1) * self.latched.float().unsqueeze(1)
            hv = self.hand.data.body_lin_vel_w[:, self.hand_bid[s]]; hw = self.hand.data.body_ang_vel_w[:, self.hand_bid[s]]
            v_rel = quat_apply(quat_conjugate(hq), self.art[s].data.root_lin_vel_w - hv)
            w_rel = quat_apply(quat_conjugate(hq), self.art[s].data.root_ang_vel_w - hw) * 0.1
            g_h = quat_apply(quat_conjugate(hq), g_w)
            blocks += [rp, rq, dvec, v_rel, w_rel, g_h]
            devs.append(dvec)
            forces.append((self._pad_force_mat(s) / 10.0).clamp(max=3.0))
        # 盘规范系海绵位姿 + 参考前瞻
        pp, pq = self._obj_pose("left"); sp, sq = self._obj_pose("right")
        sig = self.S(pp, pq, sp, sq)
        psc = sig["sponge_in_plate"]
        looks = [self.psc_ref[(self.k + int(d)).clamp(max=self.T_REF - 1)] for d in TC.S2_LOOKAHEAD]
        blocks += [psc, (sig["gap_min"] * 100).clamp(-3, 3).unsqueeze(1), sig["contact"].float().unsqueeze(1),
                   (psc - looks[0]) * 10.0] + [(lk - psc) * 10.0 for lk in looks]
        tier1h = torch.nn.functional.one_hot(self.tier_row[self.k], 3).float()
        pinned = (row < self.release_row).float().unsqueeze(1)
        blocks += [(self.k.float() / max(self.T_REF - 1, 1)).unsqueeze(1), self.gate_ok.float().unsqueeze(1), tier1h,
                   self.certified.float().unsqueeze(1), pinned]
        F = torch.cat(forces, 1)
        blocks += [F, (F > TC.PAD_FTH / 10.0).float(), self._ff_fingers(row), self.last_act]
        obs = torch.cat(blocks, 1)
        assert obs.shape[1] == OBS_DIM, (obs.shape, OBS_DIM)
        obs = obs.float().clamp(-10, 10).nan_to_num(0.0)
        priv = torch.cat(devs + [F], 1)
        assert priv.shape[1] == PRIV_DIM, priv.shape
        return {"policy": obs, "priv_info": priv.float().clamp(-10, 10).nan_to_num(0.0)}

    # ---------------------------------------------------------------- 台账 ----
    def _book(self, env_ids):
        t = self._tick
        rows = self.episode_length_buf[env_ids].float().clamp(min=1)
        n_ia = self._ep["n_ia"][env_ids].clamp(min=1)
        cov = self.PBj.coverage()[env_ids]
        add = {
            "sr/cert": self.certified[env_ids].float(),
            "sr/success": (self.succ[env_ids] & ~self.died[env_ids]).float(),
            "sr/clock_done": self.clock_done[env_ids].float(),
            "prog/clock_frac": self.k[env_ids].float() / max(self.T_REF - 1, 1),
            "prog/gate_frac": self._ep["n_gate"][env_ids] / n_ia,
            "term/die": self.died[env_ids].float(),
            "term/die_rel": (self.die_kind[env_ids] == 1).float(), "term/die_tilt": (self.die_kind[env_ids] == 2).float(),
            "term/die_plate_dev": (self.die_kind[env_ids] == 3).float(), "term/die_drop": (self.die_kind[env_ids] == 4).float(),
            "term/die_table": (self.die_kind[env_ids] == 5).float(),
            "term/cert_timeout": (self._ep["cert_timeout"][env_ids] > 0).float(),
            "hold/relp_max_plate_cm": self.rel_max[env_ids, 0] * 100, "hold/relp_max_sponge_cm": self.rel_max[env_ids, 1] * 100,
            "hold/relrot_max_plate_deg": torch.rad2deg(self.rel_max[env_ids, 2]), "hold/relrot_max_sponge_deg": torch.rad2deg(self.rel_max[env_ids, 3]),
            "task/coverage": cov, "task/travel_cm": self.PBj.travel[env_ids] * 100,
            "task/contact_frac": self._ep["contact"][env_ids] / n_ia,
            "shape/cross_frac": self._ep["cross_any"][env_ids] / rows,
            "reanchor/dx_cm": self._re_dx[env_ids] * 100, "reanchor/dq_deg": torch.rad2deg(self.dq_re[env_ids].abs().amax(1)),
            "push/n_per_ep": self._ep["n_push"][env_ids],
            "ep_rew/adv": self._ep["r_adv"][env_ids], "ep_rew/leash": self._ep["r_leash"][env_ids],
            "ep_rew/bonus": self._ep["r_bonus"][env_ids], "ep_rew/action": self._ep["r_act"][env_ids], "ep_rew/soft": self._ep["r_soft"][env_ids],
            "curr/release_row": self.release_row[env_ids].float(), "ep/len": rows,
        }
        for kk, v in add.items():
            s, n = self._acc.get(kk, (0.0, 0))
            self._acc[kk] = (s + float(v.sum()), n + int(v.numel()))
        le = self.last_ep
        le["cert"][env_ids] = add["sr/cert"]; le["success"][env_ids] = add["sr/success"]; le["clock_frac"][env_ids] = add["prog/clock_frac"]
        le["coverage"][env_ids] = cov; le["travel_cm"][env_ids] = add["task/travel_cm"]
        le["relp_max_plate_cm"][env_ids] = add["hold/relp_max_plate_cm"]; le["relp_max_sponge_cm"][env_ids] = add["hold/relp_max_sponge_cm"]
        le["relrot_max_plate_deg"][env_ids] = add["hold/relrot_max_plate_deg"]; le["relrot_max_sponge_deg"][env_ids] = add["hold/relrot_max_sponge_deg"]
        le["die_kind"][env_ids] = self.die_kind[env_ids].float(); le["cross_frac"][env_ids] = add["shape/cross_frac"]
        le["ep_len"][env_ids] = rows; le["release_row"][env_ids] = add["curr/release_row"]; le["cert_timeout"][env_ids] = add["term/cert_timeout"]
