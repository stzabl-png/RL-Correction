"""UnscrewProgress 批量版 (torch, N envs) —— 训练接线用。

与 progress.py 同一套定稿数值/判据 (标量版为规格真源, 本文件是它的向量化镜像;
改判据必须两处同步并重跑 selftest_progress_batch.py 的一致性断言)。

[TASK] 与 Pour 批量版的差异与标量版一一对应: G1 只判左手; G3=拧开释放
(env 喂 screw_released); placed=母带末行目标 (瓶3cm/15° 盖5cm/30°);
tmix=双物体短板; HCF_R/L=人手置信度形状门控权重列。
TB 记账: pop_rates() 吐 sr/gate1..4(+t0 口径), prog/clock_frac, 认证针, 分母针。
"""
from __future__ import annotations

import json

import numpy as np
import torch

from progress import (LEASH_POS, LEASH_ROT, GATE_POS, GATE_ROT, RED_GATE_POS,
                      MS_REWARD, PLACED_POS, PLACED_ROT, PLACED_HOLD,
                      M4_ARM, M4_HOLD, M4_DIST_POS, M4_DIST_ROT, W_OBJ, W_HAND,
                      W_HCONF, _tier,
                      G1_HOLD, CERT_RAMP, CERT_HOLD, CERT_RET, CERT_RISE,
                      CERT_SLIP, CERT_WAIT, CERT_TRIES, WAGE, WAGE_CAP,
                      D1_DROP, D2_TILT, D3_DEV, TABLE_Z, UP_LOCAL)

# 观测/接线兼容别名 (框架 task_env 引用这些名字)
M2_HOLD = 1                      # G3=释放是锁存事件, 无 hold (进度条观测用)
M3_HOLD = PLACED_HOLD


def _q2R(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1)], -2)


def _tilt(quat, up):
    R = _q2R(quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-9))
    v = torch.einsum("nij,j->ni", R, up)
    return torch.acos((v[:, 2] / v.norm(dim=1).clamp(min=1e-9)).clamp(-1, 1))


def _qang(a, b):
    d = (a * b).sum(-1).abs().clamp(max=1.0)
    return 2 * torch.acos(d)


class UnscrewProgressBatch:
    def __init__(self, npz_path, num_envs, device, kcap=None, no_hand_ref=False):
        z = np.load(npz_path, allow_pickle=True)
        rows = np.where(np.asarray(z["source"]) == 1)[0]
        self.N_ROW = len(rows)
        self.Ne, self.dev = int(num_envs), device
        f32 = lambda a: torch.tensor(np.asarray(a, np.float64)[rows],
                                     dtype=torch.float32, device=device)
        self.ref_obj = {oi: torch.cat([f32(z[f"obj_pos_{oi}"]),
                                       f32(z[f"obj_quat_{oi}"])], dim=1)
                        for oi in (0, 1)}
        self.stance = {s: torch.tensor(np.asarray(z[f"{s}_q"], np.float64)[-1],
                                       dtype=torch.float32, device=device)
                       for s in ("right", "left")}
        def tiers(key):
            return torch.tensor([(_tier(v) if _tier(v) is not None else 2)
                                 for v in np.asarray(z[key])[rows]],
                                dtype=torch.long, device=device)
        self.tp = {oi: tiers(f"conf_pos_{oi}") for oi in (0, 1)}
        self.tr = {oi: tiers(f"conf_rot_{oi}") for oi in (0, 1)}
        # [TASK] 主档 = 双物体短板 (标量版 tmix 同口径)
        self.tmix = torch.minimum(torch.minimum(self.tp[0], self.tr[0]),
                                  torch.minimum(self.tp[1], self.tr[1]))
        # [TASK] 人手置信度形状门控权重列 (列缺席=全绿)
        def hcf(key):
            if key not in z:
                return torch.ones(self.N_ROW, device=device)
            hw = [W_HCONF[(_tier(v) if _tier(v) is not None else 2)]
                  for v in np.asarray(z[key], np.float64)[rows]]
            return torch.tensor(hw, dtype=torch.float32, device=device)
        self.HCF_R, self.HCF_L = hcf("hand_conf_fin_r"), hcf("hand_conf_fin_l")
        self.LP = torch.tensor([LEASH_POS[0], LEASH_POS[1], LEASH_POS[2]],
                               device=device)
        self.LR = torch.tensor([float(LEASH_ROT[0] or 1e9),
                                LEASH_ROT[1], LEASH_ROT[2]], device=device)
        self.rot_ban = torch.tensor([True, False, False], device=device)
        self.no_hand_ref = bool(no_hand_ref)
        if self.no_hand_ref:
            self.WO = torch.ones(3, device=device)
            self.WH = torch.zeros(3, device=device)
        else:
            self.WO = torch.tensor([W_OBJ[0], W_OBJ[1], W_OBJ[2]], device=device)
            self.WH = torch.tensor([W_HAND[0], W_HAND[1], W_HAND[2]],
                                   device=device)
        self.rest = {oi: self.ref_obj[oi][0].clone() for oi in (0, 1)}
        self.end = {oi: self.ref_obj[oi][-1].clone() for oi in (0, 1)}
        self.up = {oi: torch.tensor(UP_LOCAL[oi], dtype=torch.float32,
                                    device=device) for oi in (0, 1)}
        self.end_tilt = {oi: _tilt(self.end[oi][3:7].unsqueeze(0),
                                   self.up[oi])[0] for oi in (0, 1)}
        self.kcap = int(kcap) if kcap is not None else None
        self.k_sep = self.N_ROW - 1
        try:
            meta = json.loads(str(z["meta"]))
            self.k_sep = int(np.clip(meta["windows"]["k_sep"], 1, self.N_ROW - 1))
        except Exception:
            pass
        # 状态
        self.k = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.g1 = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.g2 = torch.zeros_like(self.g1)
        self.g3 = torch.zeros_like(self.g1)
        self.g4 = torch.zeros_like(self.g1)
        self.placed = torch.zeros_like(self.g1)
        self.released = torch.zeros_like(self.g1)
        self.g1_run = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.wage_paid = torch.zeros(num_envs, device=device)
        self.born_t0 = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.cert_phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cert_t = torch.zeros_like(self.cert_phase)
        self.cert_try = torch.zeros_like(self.cert_phase)
        self.cert_wait = torch.zeros_like(self.cert_phase)
        self.cert_pending = torch.zeros_like(self.g1)
        self.cert_z0 = torch.zeros(num_envs, 2, device=device)
        self.cert_rel0 = torch.zeros(num_envs, 3, device=device)   # 左腕-瓶
        self.m2_run = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.m3_run = torch.zeros_like(self.m2_run)
        self.m4_run = torch.zeros_like(self.m2_run)
        self.m3_snap = {oi: torch.zeros(num_envs, 7, device=device) for oi in (0, 1)}
        self.lb = {oi: torch.zeros(num_envs, 3, device=device) for oi in (0, 1)}
        self.lb_set = torch.zeros_like(self.g1)
        self.pre1 = torch.zeros_like(self.g1)
        self.pre2 = torch.zeros_like(self.g1)
        self.pre3 = torch.zeros_like(self.g1)
        self.done = torch.zeros_like(self.g1)
        self._acc = {"ep": 0, "g1": 0, "g2": 0, "g3": 0, "g4": 0,
                     "clock": 0.0, "catt": 0, "cpass": 0,
                     "ep_t0": 0, "g1_t0": 0, "g2_t0": 0, "g3_t0": 0, "g4_t0": 0}

    def reset_idx(self, env_ids):
        n = len(env_ids)
        if n:
            self._acc["ep"] += n
            self._acc["g1"] += int((self.g1 & ~self.pre1)[env_ids].sum())
            self._acc["g2"] += int((self.g2 & ~self.pre2)[env_ids].sum())
            self._acc["g3"] += int((self.g3 & ~self.pre3)[env_ids].sum())
            self._acc["g4"] += int(self.g4[env_ids].sum())
            self._acc["clock"] += float(self.k[env_ids].float().sum()) \
                / max(self.N_ROW - 1, 1)
            t0m = self.born_t0[env_ids]
            self._acc["ep_t0"] += int(t0m.sum())
            for gk, gt in (("g1_t0", self.g1), ("g2_t0", self.g2),
                           ("g3_t0", self.g3), ("g4_t0", self.g4)):
                self._acc[gk] += int((gt[env_ids] & t0m).sum())
        for t_ in (self.g1, self.g2, self.g3, self.g4, self.placed, self.done,
                   self.released, self.pre1, self.pre2, self.pre3, self.lb_set,
                   self.cert_pending):
            t_[env_ids] = False
        for oi in (0, 1):
            self.lb[oi][env_ids] = 0.0
        self.cert_z0[env_ids] = 0.0
        self.cert_rel0[env_ids] = 0.0
        for t_ in (self.k, self.g1_run, self.m2_run, self.m3_run, self.m4_run,
                   self.cert_phase, self.cert_t, self.cert_try, self.cert_wait):
            t_[env_ids] = 0
        self.wage_paid[env_ids] = 0.0
        self.born_t0[env_ids] = True

    def enter(self, env_ids, rows, g1, g2, g3, placed):
        self.reset_idx(env_ids)
        self.k[env_ids] = rows
        self.g1[env_ids] = g1
        self.g2[env_ids] = g2
        self.g3[env_ids] = g3
        self.released[env_ids] = g3          # G3 预置 = 已释放
        self.placed[env_ids] = placed
        self.pre1[env_ids] = g1
        self.pre2[env_ids] = g2
        self.pre3[env_ids] = g3
        self.born_t0[env_ids] = ~(g1 | g2 | g3 | placed)
        for oi in (0, 1):
            self.m3_snap[oi][env_ids] = self.end[oi].unsqueeze(0) \
                .expand(len(env_ids), 7)

    def pop_rates(self):
        # ★分母为 0 发 NaN 不发 0 (L5-17); 分母本身也发出去
        _ep, _ep0 = self._acc["ep"], self._acc["ep_t0"]
        _at = self._acc["catt"]
        nan = float("nan")
        _r = (lambda num, den: (num / den) if den > 0 else nan)
        out = {"sr/gate1": _r(self._acc["g1"], _ep),
               "sr/gate2": _r(self._acc["g2"], _ep),
               "sr/gate3": _r(self._acc["g3"], _ep),
               "sr/gate4": _r(self._acc["g4"], _ep),
               "sr_t0/gate1": _r(self._acc["g1_t0"], _ep0),
               "sr_t0/gate2": _r(self._acc["g2_t0"], _ep0),
               "sr_t0/gate3": _r(self._acc["g3_t0"], _ep0),
               "sr_t0/gate4": _r(self._acc["g4_t0"], _ep0),
               "prog/ep_t0_frac": _r(self._acc["ep_t0"], _ep),
               "prog/clock_frac": _r(self._acc["clock"], _ep),
               "sr/cert_pass": _r(self._acc["cpass"], _at),
               "prog/cert_att": _r(self._acc["catt"], _ep),
               "n/ep_done": float(_ep),
               "n/ep_done_t0": float(_ep0),
               "n/cert_attempts": float(_at)}
        self._acc = {"ep": 0, "g1": 0, "g2": 0, "g3": 0, "g4": 0,
                     "clock": 0.0, "catt": 0, "cpass": 0,
                     "ep_t0": 0, "g1_t0": 0, "g2_t0": 0, "g3_t0": 0, "g4_t0": 0}
        return out

    def step(self, obj0, obj1, armq_r, armq_l, pads3, wrist_r, wrist_l,
             screw_released=None, run_mask=None):
        """全部 (N,·) 张量; obj0=瓶 obj1=盖; pads3=左手>=3/5垫;
        screw_released (N,)bool = env 螺旋 detach 锁存; run_mask=交互行管辖。"""
        k = self.k.clamp(max=self.N_ROW - 1)
        tier = self.tmix[k]
        w_obj = self.WO[tier]
        active = ~self.done if run_mask is None else (~self.done) & run_mask
        if screw_released is not None:
            self.released |= screw_released & active
        # 药A: G1 起捕获持握基线
        cap = self.g1 & (~self.lb_set) & active
        if cap.any():
            for oi, act in ((0, obj0), (1, obj1)):
                self.lb[oi][cap] = (act[:, :3] - self.ref_obj[oi][k][:, :3])[cap]
            self.lb_set = self.lb_set | cap
        # ---- 皮筋 ----
        leash = torch.zeros(self.Ne, device=self.dev)
        for oi, act in ((0, obj0), (1, obj1)):
            ref = self.ref_obj[oi][k]
            dvec = act[:, :3] - ref[:, :3] \
                - self.lb[oi] * self.lb_set.float().unsqueeze(1)
            dp = dvec.norm(dim=1)
            lp = self.LP[self.tp[oi][k]]
            leash = leash - ((dp - lp).clamp(min=0) / lp)
            trot = self.tr[oi][k]
            dr = _qang(act[:, 3:7] / act[:, 3:7].norm(dim=1, keepdim=True)
                       .clamp(min=1e-9), ref[:, 3:7])
            lr = self.LR[trot]
            pen = ((dr - lr).clamp(min=0) / lr)
            leash = leash - torch.where(self.rot_ban[trot],
                                        torch.zeros_like(pen), pen)
        earning = (self.k < self.N_ROW - 1)
        leash = leash.clamp(min=-3.0) * active.float() * earning.float()
        # ---- 时钟门 ----
        ok_obj = torch.ones(self.Ne, dtype=torch.bool, device=self.dev)
        gp = torch.where(tier > 0,
                         torch.full_like(w_obj, GATE_POS),
                         torch.full_like(w_obj, RED_GATE_POS))
        for oi, act in ((0, obj0), (1, obj1)):
            ref = self.ref_obj[oi][k]
            ok_obj &= ((act[:, :3] - ref[:, :3]).norm(dim=1) <= gp)
            rot_ok = (_qang(act[:, 3:7] / act[:, 3:7].norm(dim=1, keepdim=True)
                            .clamp(min=1e-9), ref[:, 3:7]) <= GATE_ROT)
            ok_obj &= torch.where(tier > 0, rot_ok, torch.ones_like(rot_ok))
        ok = ok_obj & active
        _cap = self.N_ROW - 1 if self.kcap is None else min(self.kcap,
                                                            self.N_ROW - 1)
        can = ok & (self.k < _cap) & self.g2
        self.k = self.k + can.long()
        adv = can.float()
        ms_r = torch.zeros(self.Ne, device=self.dev)
        # ---- G1: [TASK] 左手 ----
        g1_gate = (~self.g1) & active
        self.g1_run = torch.where(g1_gate & pads3, self.g1_run + 1,
                                  torch.zeros_like(self.g1_run))
        new1 = g1_gate & (self.g1_run >= G1_HOLD)
        self.g1 |= new1
        ms_r += new1.float() * MS_REWARD[1]
        # ---- 站位维持费 ----
        w_ok = pads3 & (~self.g2) & active & (self.wage_paid < WAGE_CAP)
        wage = torch.minimum(torch.full_like(self.wage_paid, WAGE),
                             (WAGE_CAP - self.wage_paid).clamp(min=0.0)) \
            * w_ok.float()
        self.wage_paid = self.wage_paid + wage
        # ---- G2 认证机 ([TASK] 滑移只判左腕-瓶) ----
        self.cert_wait = (self.cert_wait - 1).clamp(min=0)
        start = (self.g1 & (~self.g2) & (self.cert_phase == 0)
                 & (self.cert_wait == 0) & (self.cert_try < CERT_TRIES) & active)
        if start.any():
            self.cert_phase[start] = 1
            self.cert_t[start] = 0
            self.cert_z0[start, 0] = obj0[start, 2]
            self.cert_z0[start, 1] = obj1[start, 2]
            self.cert_rel0[start] = (wrist_l - obj0[:, :3])[start]
        in_cert = (self.cert_phase > 0) & (~start) & active
        self.cert_t = torch.where(in_cert, self.cert_t + 1, self.cert_t)
        to_hold = in_cert & (self.cert_phase == 1) & (self.cert_t >= CERT_RAMP)
        self.cert_phase[to_hold] = 2
        self.cert_t[to_hold] = 0
        judge = in_cert & (self.cert_phase == 2) & (self.cert_t >= CERT_HOLD)
        if judge.any():
            rel_l = ((wrist_l - obj0[:, :3]) - self.cert_rel0).norm(dim=1)
            ok5 = ((obj0[:, 2] - self.cert_z0[:, 0] >= CERT_RISE)
                   & (obj1[:, 2] - self.cert_z0[:, 1] >= CERT_RISE)
                   & (rel_l < CERT_SLIP) & pads3)
            self.cert_pending |= judge & ok5
            failj = judge & (~ok5)
            self.cert_try[failj] += 1
            self.cert_wait[failj] = CERT_WAIT
            self._acc["catt"] += int(judge.sum())
            self._acc["cpass"] += int((judge & ok5).sum())
            self.cert_phase[judge] = 3
            self.cert_t[judge] = 0
        fin = in_cert & (self.cert_phase == 3) & (self.cert_t >= CERT_RET)
        if fin.any():
            new2 = fin & self.cert_pending
            self.g2 |= new2
            ms_r += new2.float() * MS_REWARD[2]
            self.cert_pending[fin] = False
            self.cert_phase[fin] = 0
            self.cert_t[fin] = 0
        # ==== [TASK] G3 = 拧开释放 ====
        new3 = self.g2 & (~self.g3) & self.released & active
        self.g3 |= new3
        ms_r += new3.float() * MS_REWARD[3]
        # ---- placed: [TASK] 母带末行目标, 瓶3cm/15° 盖5cm/30° ----
        ok3 = torch.ones_like(ok_obj)
        for oi, act in ((0, obj0), (1, obj1)):
            ok3 &= ((act[:, :3] - self.end[oi][:3]).norm(dim=1)
                    <= PLACED_POS[oi])
            dt = (_tilt(act[:, 3:7], self.up[oi]) - self.end_tilt[oi]).abs()
            ok3 &= (dt <= PLACED_ROT[oi])
        gatep = self.g3 & (~self.placed) & active
        self.m3_run = torch.where(gatep & ok3, self.m3_run + 1,
                                  torch.zeros_like(self.m3_run))
        newp = gatep & (self.m3_run >= PLACED_HOLD)
        self.placed |= newp
        for oi, act in ((0, obj0), (1, obj1)):
            self.m3_snap[oi][newp] = act[newp]
        # ---- G4 = Success ----
        ok4 = ((armq_r - self.stance["right"]).abs().max(dim=1).values <= M4_ARM) \
            & ((armq_l - self.stance["left"]).abs().max(dim=1).values <= M4_ARM)
        for oi, act in ((0, obj0), (1, obj1)):
            sn = self.m3_snap[oi]
            ok4 &= ((act[:, :3] - sn[:, :3]).norm(dim=1) <= M4_DIST_POS)
            dt = (_tilt(act[:, 3:7], self.up[oi])
                  - _tilt(sn[:, 3:7], self.up[oi])).abs()
            ok4 &= (dt <= M4_DIST_ROT)
        gate4 = self.placed & (~self.g4) & active
        self.m4_run = torch.where(gate4 & ok4, self.m4_run + 1,
                                  torch.zeros_like(self.m4_run))
        new4 = gate4 & (self.m4_run >= M4_HOLD)
        self.g4 |= new4
        self.done |= new4
        ms_r += new4.float() * MS_REWARD[4]
        # ---- 死线 (D2pre 只判瓶; D2/D8 倾角=相对末行目标) ----
        fail = torch.zeros(self.Ne, dtype=torch.bool, device=self.dev)
        fail |= (~self.g2) & (_tilt(obj0[:, 3:7], self.up[0]) > np.radians(60))
        for oi, act in ((0, obj0), (1, obj1)):
            fail |= (act[:, 2] < TABLE_Z - D1_DROP)
            ref = self.ref_obj[oi][k]
            fail |= (~self.placed) & ((act[:, :3] - ref[:, :3]).norm(dim=1)
                                      > D3_DEV)
            sn = self.m3_snap[oi]
            dt_sn = (_tilt(act[:, 3:7], self.up[oi])
                     - _tilt(sn[:, 3:7], self.up[oi])).abs()
            fail |= self.placed & (((act[:, :3] - sn[:, :3]).norm(dim=1)
                                    > M4_DIST_POS) | (dt_sn > M4_DIST_ROT))
            dt_end = (_tilt(act[:, 3:7], self.up[oi]) - self.end_tilt[oi]).abs()
            fail |= self.placed & (dt_end > D2_TILT)
        fail &= active
        self.done |= fail
        return {"adv": adv, "leash": leash, "ms": ms_r, "wage": wage,
                "w_obj": w_obj, "w_hand": self.WH[tier],
                "w_hconf_r": self.HCF_R[k], "w_hconf_l": self.HCF_L[k],
                "clock": self.k.clone(), "done": self.done.clone(),
                "tier": tier, "fail": fail,
                "cert_phase": self.cert_phase.clone(),
                "cert_t": self.cert_t.clone()}
