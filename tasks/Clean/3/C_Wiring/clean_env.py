"""Clean/3 Stage-1 抓稳段 env (台账 §5.6; 用户 2026-09-07 三点裁定).

回合 (20Hz, T_EP=110 行固定):
  行 0        : 臂在 grasp 位 (母带 v1 第 0 行 IK), 指在 grasp_qpos; 两物体在名义世界位姿 (母带 0 行), **钉住**。
  行 <release : 钉住 —— 每物理子步把物体写回名义位姿 (零速度); 碰撞全开, 指合拢的接触冲量照常产生 (指力建立)。
  行 ≥release : 放手, 物体只靠指力。
  指前馈: grasp → squeeze 斜坡 K_CLOSE 行, 之后 squeeze 常量; 臂前馈: 第 0 行常量 (+一次性 PD 下垂补偿)。
动作 58 = [R臂7, L臂7, R指22, L指22] 有界累积残差 (Sweep2 体制)。
判据: 放手后连续 CERT_STEPS 步 两物体世界位姿相对钉住位姿 <CERT_POS/CERT_ROT ∧ 接触条件 ⇒ G0 认证 (锁存);
      掉落 (>DROP_POS/DROP_ROT 或落桌) ⇒ 终止; 成功 = 回合末 (放手后 ≥HOLD_ROWS) 认证过 ∧ 仍在阈内 ∧ 未掉。
课程: release_row_cur 由训练入口按成功率 EMA 单向退火 RELEASE_MAX → RELEASE_MIN。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import hold_env as HE  # noqa: E402
import task_config as TC  # noqa: E402
from hold_env import HoldProbeEnv, _qangle  # noqa: E402

ACT_DIM = 58
OBS_DIM = 367
PRIV_DIM = 22
SIDES = ("left", "right")          # 0=盘(左) 1=海绵(右)


def build_cfg(num_envs=1, reference=None):
    reference = reference or TC.REFERENCE
    cfg = HE.build_cfg(num_envs, reference=reference)
    cfg.action_space = ACT_DIM
    cfg.observation_space = OBS_DIM
    cfg.episode_length_s = float((TC.T_EP + 2) / TC.CONTROL_HZ)
    return cfg


class CleanHoldEnv(HoldProbeEnv):
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        N, dev = self.num_envs, self.device
        to = self.to
        assert self.n_sub == int(round(240 / TC.CONTROL_HZ)), self.n_sub
        # 动作布局: 臂 14 (map_ids: R1..7, L1..7) + 指 44 (R 22 GENERIC 序, L 22)
        self.act_ids = torch.tensor(self.map_ids + self.fid["right"] + self.fid["left"], dtype=torch.long, device=dev)
        self.step_sz = to([TC.ARM_STEP] * 14 + [TC.FIN_STEP] * 44)
        self.dev_hi = to([TC.ARM_DEV] * 14 + [TC.FIN_DEV] * 44)
        self.cum_res = torch.zeros(N, ACT_DIM, device=dev)
        self.last_act = torch.zeros(N, ACT_DIM, device=dev)
        self.q_cmd = self.hand.data.default_joint_pos.clone()
        self.nominal = {s: (self.ref_pos[self.oi[s]][0].clone(), self.ref_quat[self.oi[s]][0].clone()) for s in SIDES}
        # 前馈 squeeze 姿去交叉: 指定关节的合拢剂量按 SQUEEZE_JOINT_BETA 缩放 (task_config 有 FK 实测依据)
        from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER as _GJO
        for s in SIDES:
            for jn, b in TC.SQUEEZE_JOINT_BETA.get(s, {}).items():
                j = [n.replace("right_", f"{s}_") for n in _GJO].index(jn)
                self.f_squeeze[s][j] = self.f_grasp[s][j] + float(b) * (self.f_squeeze[s][j] - self.f_grasp[s][j])
                print(f"[CleanHoldEnv] squeeze 剂量覆写 {jn}: β={b}")
        # 续跑用: 课程已退火到底的 run 重新起跑时不该被打回 RELEASE_MAX 再退火一遍
        # (2026-09-10 Denso 迁 msc; 两条 take3 都已到 release=10, sr/cert≈0.99)。
        # 不传 = 原行为 (RELEASE_MAX), 进 world.json 的 release_row_start 可查。
        self.release_row_cur = int(os.environ.get("CLEAN_RELEASE_START") or TC.RELEASE_MAX)
        self.release_row = torch.full((N,), int(TC.RELEASE_MAX), dtype=torch.long, device=dev)
        self.cert = torch.zeros(N, dtype=torch.bool, device=dev)
        self.cert_run = torch.zeros(N, dtype=torch.long, device=dev)
        self.dropped = torch.zeros(N, dtype=torch.bool, device=dev)
        self.drop_side = torch.zeros(N, 2, dtype=torch.bool, device=dev)
        self.pad_ids = {s: [list(self.hand.body_names).index(f"{s}_{f}_elastomer")
                            for f in ("thumb", "index", "middle", "ring", "pinky")] for s in SIDES}
        self._tick = None
        self._ep = {k: torch.zeros(N, device=dev) for k in ("r_contact", "r_hold", "r_bonus", "r_act", "c_plate", "c_sponge")}
        self._acc = {}
        self._n_ep = 0
        self.arm_sag = self._calibrate_sag()
        # 手指交叉/重叠: 相邻指尖对 (食-中, 中-无名, 无名-小), 掌系侧向 = hand_C_MC 局部 y; 顺序符号取自 grasp 姿实测
        self.tip_ids = {s: torch.tensor(self.pad_ids[s][1:], dtype=torch.long, device=dev) for s in SIDES}   # index..pinky
        self.cross_sign = {}
        for s in SIDES:
            g, _ = self._finger_gaps(s, signed=None)
            self.cross_sign[s] = torch.sign(g[0]).clamp(min=-1).detach()      # (3,), 用 env0 的 grasp 姿
            self.cross_sign[s][self.cross_sign[s] == 0] = 1.0
        self._ep["r_cross"] = torch.zeros(N, device=dev); self._ep["cross_any"] = torch.zeros(N, device=dev)
        g0 = {s: (self._finger_gaps(s)[0][0] * 100).cpu().numpy().round(2).tolist() for s in SIDES}
        d0 = {s: (self._finger_gaps(s)[1][0] * 100).cpu().numpy().round(2).tolist() for s in SIDES}
        print(f"[CleanHoldEnv] grasp 姿相邻指尖 侧向间距(cm)={g0} 3D距离(cm)={d0} (罚阈 {TC.CROSS_GAP_MIN*100:.1f}/{TC.OVERLAP_MIN*100:.1f}cm)")
        print(f"[CleanHoldEnv] N={N} obs={OBS_DIM} act={ACT_DIM} T_EP={TC.T_EP} release={self.release_row_cur} "
              f"sag(deg)={np.round(np.degrees(self.arm_sag.cpu().numpy()), 2).tolist()}")

    # ---------------------------------------------------------------- 工具 ----
    def _ff_fingers(self, row):
        """指前馈 (N,44): grasp→squeeze 斜坡 K_CLOSE 行, 之后 squeeze。"""
        a = (row.float() / float(TC.K_CLOSE)).clamp(0.0, 1.0).unsqueeze(1)
        out = []
        for s in ("right", "left"):
            out.append((1 - a) * self.f_grasp[s] + a * self.f_squeeze[s])
        return torch.cat(out, 1)

    def _pin(self, env_ids=None):
        """钉住: 两物体写回名义世界位姿 (零速度)。"""
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if len(ids) == 0:
            return
        org = self._org()[ids]
        for s in SIDES:
            p, q = self.nominal[s]
            pose = torch.cat([p.expand(len(ids), 3) + org, q.expand(len(ids), 4)], 1)
            self.art[s].write_root_pose_to_sim(pose, env_ids=ids)
            self.art[s].write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)

    def _pad_force_mat(self, side):
        """(N,5) 各垫接触力 N。"""
        sens = self._all_sensors[:5] if side == "left" else self._all_sensors[5:10]
        cols = []
        for s_ in sens:
            fw = s_.data.net_forces_w                       # (N, B, 3), B=1
            cols.append(torch.linalg.vector_norm(fw.reshape(self.num_envs, -1, 3), dim=2).amax(dim=1))
        return torch.stack(cols, 1)

    def _dev(self, side):
        op, oq = self._obj_pose(side); p, q = self.nominal[side]
        return torch.linalg.vector_norm(op - p, dim=1), _qangle(oq, q.expand_as(oq))

    def _finger_gaps(self, side, signed=True):
        """相邻指尖对的 (掌系侧向有符号间距 (N,3), 3D 距离 (N,3))。signed=None 时返回原始差 (用于定名义符号)。"""
        b = self.hand_bid[side]
        hp = self.hand.data.body_pos_w[:, b]; hq = self.hand.data.body_quat_w[:, b]
        tips = self.hand.data.body_pos_w[:, self.tip_ids[side]]                        # (N,4,3) index..pinky
        rel = quat_apply(quat_conjugate(hq)[:, None, :].expand(-1, 4, -1).reshape(-1, 4),
                         (tips - hp[:, None, :]).reshape(-1, 3)).reshape(self.num_envs, 4, 3)
        dy = rel[:, :-1, 1] - rel[:, 1:, 1]                                           # (N,3) 相邻侧向差
        d3 = torch.linalg.vector_norm(tips[:, :-1] - tips[:, 1:], dim=2)              # (N,3)
        if signed is None:
            return dy, d3
        return dy * self.cross_sign[side][None, :], d3

    def _cross_penalty(self):
        """交叉 (侧向间距 < CROSS_GAP_MIN, 翻转为负) + 重叠 (3D 距离 < OVERLAP_MIN); 双手 6 对取均值 ∈ [0,2]."""
        terms = []
        for s in SIDES:
            g, d = self._finger_gaps(s)
            terms.append(((TC.CROSS_GAP_MIN - g) / TC.CROSS_GAP_MIN).clamp(0.0, 1.0) * 1.0
                         + (1.0 + (-g / TC.CROSS_GAP_MIN).clamp(0.0, 1.0)) * (g < 0).float() * 0.0)   # 翻转已含在第一项 (g<0 ⇒ 1)
            terms.append(((TC.OVERLAP_MIN - d) / TC.OVERLAP_MIN).clamp(0.0, 1.0))
        t = torch.cat(terms, 1)                                                        # (N,12)
        return t.mean(1) * 2.0, (t > 0).any(1)

    def _calibrate_sag(self, steps=8, rounds=3, refine=4):
        """一次性 PD 下垂补偿 (Sweep2 同款): 钉住物体、指在 grasp, 臂目标微调直到实测=参考。"""
        N = self.num_envs
        qfull = self.hand.data.default_joint_pos.clone()
        qfull[:, self.map_ids_t] = self.ref_arm[0]
        for s in SIDES:
            qfull[:, self.fid[s]] = self.f_grasp[s]
        self.hand.write_joint_state_to_sim(qfull, torch.zeros_like(qfull))
        for _ in range(rounds):
            self._pin(); self._phys(qfull, steps)
        ref0 = self.ref_arm[0].expand(N, -1)
        for _ in range(refine):
            err = ref0 - self.hand.data.joint_pos[:, self.map_ids_t]
            qfull[:, self.map_ids_t] += err.clamp(-0.02, 0.02)
            self._pin(); self._phys(qfull, steps)
        sag = (qfull[:, self.map_ids_t] - ref0).mean(dim=0)
        return sag

    # ---------------------------------------------------------------- RL 面 ----
    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if self._tick is not None:
            self._book(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)
        n = len(env_ids)
        qfull = self.hand.data.default_joint_pos[env_ids].clone()
        qfull[:, self.map_ids_t] = self.ref_arm[0]
        for s in SIDES:
            qfull[:, self.fid[s]] = self.f_grasp[s]
        self.hand.write_joint_state_to_sim(qfull, torch.zeros_like(qfull), env_ids=env_ids)
        qcmd = qfull.clone(); qcmd[:, self.map_ids_t] += self.arm_sag
        self.hand.set_joint_position_target(qcmd, env_ids=env_ids)
        self.q_cmd[env_ids] = qcmd
        self._pin(env_ids)
        self.cum_res[env_ids] = 0; self.last_act[env_ids] = 0
        self.cert[env_ids] = False; self.cert_run[env_ids] = 0
        self.dropped[env_ids] = False; self.drop_side[env_ids] = False
        lo = max(int(TC.RELEASE_MIN), self.release_row_cur - int(TC.RELEASE_JITTER))
        self.release_row[env_ids] = torch.randint(lo, self.release_row_cur + 1, (n,), device=self.device)
        for k in self._ep:
            self._ep[k][env_ids] = 0

    def _pre_physics_step(self, actions):
        self.last_act = actions.clamp(-1.0, 1.0)
        self.cum_res = torch.maximum(torch.minimum(self.cum_res + self.last_act * self.step_sz, self.dev_hi), -self.dev_hi)
        row = self.episode_length_buf
        arm = self.ref_arm[0].expand(self.num_envs, -1) + self.arm_sag + self.cum_res[:, :14]
        fin = self._ff_fingers(row) + self.cum_res[:, 14:]
        self.q_cmd[:, self.act_ids] = torch.cat([arm, fin], 1)
        self._pinned_ids = torch.nonzero(row < self.release_row).flatten()

    def _apply_action(self):
        self.hand.set_joint_position_target(self.q_cmd)
        self._pin(self._pinned_ids)             # 钉住期每物理子步写回 (碰撞全开)

    def _get_dones(self):
        N, dev = self.num_envs, self.device
        row = self.episode_length_buf                     # 已 +1
        released = row > self.release_row
        F = {s: self._pad_force_mat(s) for s in SIDES}
        on = {s: F[s] > TC.PAD_FTH for s in SIDES}
        plate_ok = on["left"][:, 0] & (on["left"][:, 1:].sum(1) >= TC.PLATE_SUPPORT_MIN)
        sponge_ok = on["right"].sum(1) >= TC.SPONGE_PADS_MIN
        c_plate = on["left"][:, 0].float() + on["left"][:, 1:].sum(1).clamp(max=3).float() / 3.0     # [0,2]
        c_sponge = on["right"].sum(1).float() / 5.0                                                  # [0,1]
        dp, dr = {}, {}
        for s in SIDES:
            dp[s], dr[s] = self._dev(s)
        within = torch.stack([(dp[s] < TC.CERT_POS) & (dr[s] < np.radians(TC.CERT_ROT_DEG)) for s in SIDES], 1).all(1)
        z_low = torch.stack([self._obj_pose(s)[0][:, 2] < self.cfg.table_top_z + 0.03 for s in SIDES], 1)
        drop_now = torch.stack([(dp[s] > TC.DROP_POS) | (dr[s] > np.radians(TC.DROP_ROT_DEG)) for s in SIDES], 1) | z_low
        drop_now = drop_now & released.unsqueeze(1)
        new_drop = drop_now.any(1) & ~self.dropped
        self.drop_side |= drop_now; self.dropped |= drop_now.any(1)
        cert_now = released & within & plate_ok & sponge_ok
        self.cert_run = torch.where(cert_now, self.cert_run + 1, torch.zeros_like(self.cert_run))
        new_cert = (self.cert_run >= TC.CERT_STEPS) & ~self.cert
        self.cert |= new_cert
        timeout = row >= int(TC.T_EP)
        success = timeout & self.cert & within & ~self.dropped
        # ---- 奖励 ----
        r_contact = TC.W_CONTACT * (c_plate + c_sponge)
        hold_pen = -(TC.W_HOLD_POS * 0.5 * (dp["left"] + dp["right"]) / TC.CERT_POS
                     + TC.W_HOLD_ROT * 0.5 * (dr["left"] + dr["right"]) / np.radians(TC.CERT_ROT_DEG)).clamp(min=-3.0)
        r_hold = torch.where(released, hold_pen + TC.W_WITHIN * within.float(), torch.zeros_like(hold_pen))
        r_bonus = TC.B_CERT * new_cert.float() + TC.B_SUCCESS * success.float() + TC.B_DROP * new_drop.float()
        r_act = -TC.W_ACT * (self.last_act ** 2).mean(1)
        cross, cross_any = self._cross_penalty()
        r_cross = -TC.W_CROSS * cross
        reward = r_contact + r_hold + r_bonus + r_act + r_cross
        for k, v in (("r_contact", r_contact), ("r_hold", r_hold), ("r_bonus", r_bonus), ("r_act", r_act),
                     ("r_cross", r_cross), ("cross_any", cross_any.float()), ("c_plate", c_plate), ("c_sponge", c_sponge)):
            self._ep[k] += v
        terminated = self.dropped.clone()
        self._tick = dict(reward=reward, released=released, within=within, dp=dp, dr=dr, F=F,
                          success=success, timeout=timeout & ~terminated, plate_ok=plate_ok, sponge_ok=sponge_ok,
                          # 逐步账目快照 (record_clean.py 落盘用; 只读, 不影响行为)
                          r_contact=r_contact, r_hold=r_hold, r_bonus=r_bonus, r_act=r_act, r_cross=r_cross,
                          cross=cross, cross_any=cross_any, c_plate=c_plate, c_sponge=c_sponge,
                          new_cert=new_cert, new_drop=new_drop, cert=self.cert.clone(), cert_run=self.cert_run.clone(),
                          dropped=self.dropped.clone(), drop_side=self.drop_side.clone(),
                          row=row.clone(), release_row=self.release_row.clone())
        return terminated, timeout & ~terminated

    def _get_rewards(self):
        return self._tick["reward"]

    def _get_observations(self):
        N, dev = self.num_envs, self.device
        row = self.episode_length_buf
        q = self.hand.data.joint_pos; qd = self.hand.data.joint_vel
        arm_q, arm_qd = q[:, self.map_ids_t], qd[:, self.map_ids_t]
        fin_ids = self.act_ids[14:]
        fin_q, fin_qd = q[:, fin_ids], qd[:, fin_ids]
        blocks = [arm_q, arm_qd * 0.1, fin_q, fin_qd * 0.1, self.cum_res / self.dev_hi]
        devs, forces = [], []
        for s in SIDES:
            hp, hq = self._hand_pose(s); op, oq = self._obj_pose(s)
            rp = quat_apply(quat_conjugate(hq), op - hp); rq = quat_mul(quat_conjugate(hq), oq)
            p0, q0 = self.nominal[s]
            dq = quat_mul(quat_conjugate(q0.expand_as(oq)), oq)
            dvec = torch.cat([op - p0, 2.0 * dq[:, 1:] * torch.sign(dq[:, :1])], 1)          # 位置差 + 小角轴角
            blocks += [hp, hq, op, oq, rp, rq, dvec,
                       self.art[s].data.root_lin_vel_w, self.art[s].data.root_ang_vel_w * 0.1]
            devs.append(dvec)
            forces.append((self._pad_force_mat(s) / 10.0).clamp(max=3.0))
        F = torch.cat(forces, 1)
        blocks += [F, (F > TC.PAD_FTH / 10.0).float()]
        pinned = (row < self.release_row).float().unsqueeze(1)
        t_to = ((self.release_row - row).float() / 60.0).clamp(-2, 2).unsqueeze(1)
        t_since = ((row - self.release_row).float() / 60.0).clamp(-2, 2).unsqueeze(1)
        blocks += [pinned, t_to, t_since, self.cert.float().unsqueeze(1),
                   (self.cert_run.float() / TC.CERT_STEPS).clamp(max=1).unsqueeze(1),
                   self._ff_fingers(row), self.last_act]
        obs = torch.cat(blocks, 1)
        assert obs.shape[1] == OBS_DIM, obs.shape
        obs = obs.float().clamp(-10, 10).nan_to_num(0.0)
        priv = torch.cat(devs + [F], 1)
        assert priv.shape[1] == PRIV_DIM, priv.shape
        return {"policy": obs, "priv_info": priv.float().clamp(-10, 10).nan_to_num(0.0)}

    # ---------------------------------------------------------------- 台账 ----
    def _book(self, env_ids):
        t = self._tick
        rows = self.episode_length_buf[env_ids].float().clamp(min=1)
        add = {
            "sr/success": t["success"][env_ids].float(),
            "sr/cert": self.cert[env_ids].float(),
            "term/drop": self.dropped[env_ids].float(),
            "term/drop_plate": self.drop_side[env_ids, 0].float(),
            "term/drop_sponge": self.drop_side[env_ids, 1].float(),
            "hold/dev_pos_plate_cm": t["dp"]["left"][env_ids] * 100,
            "hold/dev_pos_sponge_cm": t["dp"]["right"][env_ids] * 100,
            "hold/dev_rot_plate_deg": torch.rad2deg(t["dr"]["left"][env_ids]),
            "hold/dev_rot_sponge_deg": torch.rad2deg(t["dr"]["right"][env_ids]),
            "hold/within_end": t["within"][env_ids].float(),
            "contact/plate_score": self._ep["c_plate"][env_ids] / rows,
            "contact/sponge_score": self._ep["c_sponge"][env_ids] / rows,
            "ep_rew/contact": self._ep["r_contact"][env_ids], "ep_rew/hold": self._ep["r_hold"][env_ids],
            "ep_rew/bonus": self._ep["r_bonus"][env_ids], "ep_rew/action": self._ep["r_act"][env_ids],
            "ep_rew/cross": self._ep["r_cross"][env_ids], "shape/cross_frac": self._ep["cross_any"][env_ids] / rows,
            "curr/release_row": self.release_row[env_ids].float(),
            "ep/len": self.episode_length_buf[env_ids].float(),
        }
        for k, v in add.items():
            s, n = self._acc.get(k, (0.0, 0))
            self._acc[k] = (s + float(v.sum()), n + int(v.numel()))

    def pop_rates(self):
        out = {k: (s / n if n else float("nan")) for k, (s, n) in self._acc.items()}
        out["curr/release_row_cur"] = float(self.release_row_cur)
        self._acc = {}
        return out
