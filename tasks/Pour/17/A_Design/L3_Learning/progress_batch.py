"""PourProgress 批量版 (torch, N envs) —— 训练接线用。

与 progress.py 同一套定稿数值/判据 (标量版为规格真源, 本文件是它的向量化镜像;
改判据必须两处同步并重跑 selftest_progress_batch.py 的一致性断言)。

新增: 逐关率 TB 记账 —— reset_idx() 时按回合累计 M1/M2/M3 达成数,
pop_rates() 吐 {"sr/gate1","sr/gate2","sr/gate3","prog/clock_frac"} 并清零。
"""
from __future__ import annotations

import numpy as np
import torch

from progress import (LEASH_POS, LEASH_ROT, GATE_POS, GATE_ROT, MS_REWARD,
                      M2_TILT, M2_HOLD, M3_POS, M3_ROT, M3_HOLD,
                      M4_ARM, M4_HOLD, M4_DIST_POS, M4_DIST_ROT, W_OBJ, _tier,
                      D1_DROP, D2_TILT, D3_DEV, TABLE_Z)


def _q2R(q):
    """(N,4) wxyz -> (N,3,3)"""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], -1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], -1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], -1)], -2)


def _tilt(quat, up):
    """长轴倾角 (yaw 豁免口径, 2026-08-28 拍板): 局部 up 轴映到世界后与竖直夹角."""
    R = _q2R(quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-9))
    v = torch.einsum("nij,j->ni", R, up)
    return torch.acos((v[:, 2] / v.norm(dim=1).clamp(min=1e-9)).clamp(-1, 1))


def _qang(a, b):
    d = (a * b).sum(-1).abs().clamp(max=1.0)
    return 2 * torch.acos(d)


class PourProgressBatch:
    def __init__(self, npz_path, num_envs, device,
                 mouth_local_bot, mouth_local_cup,
                 up_local_bot=(0.0, 1.0, 0.0), mouth_gate=0.12,
                 leash_rot_tilt=False, kcap=None):
        z = np.load(npz_path, allow_pickle=True)
        rows = np.where(np.asarray(z["source"]) == 1)[0]
        self.N_ROW = len(rows)
        self.Ne, self.dev = int(num_envs), device
        f32 = lambda a: torch.tensor(np.asarray(a, np.float64)[rows],
                                     dtype=torch.float32, device=device)
        self.ref_obj = {oi: torch.cat([f32(z[f"obj_pos_{oi}"]),
                                       f32(z[f"obj_quat_{oi}"])], dim=1)
                        for oi in (0, 1)}
        self.ref_arm = {s: f32(z[f"{s}_q"]) for s in ("right", "left")}
        self.stance = {s: torch.tensor(np.asarray(z[f"{s}_q"], np.float64)[-1],
                                       dtype=torch.float32, device=device)
                       for s in ("right", "left")}
        def tiers(key):
            return torch.tensor([( _tier(v) if _tier(v) is not None else 2)
                                 for v in np.asarray(z[key])[rows]],
                                dtype=torch.long, device=device)
        self.tp = {oi: tiers(f"conf_pos_{oi}") for oi in (0, 1)}
        self.tr = {oi: tiers(f"conf_rot_{oi}") for oi in (0, 1)}
        self.tmix = torch.minimum(self.tp[1], self.tr[1])
        self.LP = torch.tensor([LEASH_POS[0], LEASH_POS[1], LEASH_POS[2]],
                               device=device)
        self.LR = torch.tensor([float(LEASH_ROT[0] or 1e9),
                                LEASH_ROT[1], LEASH_ROT[2]], device=device)
        self.rot_ban = torch.tensor([True, False, False], device=device)  # 红档rot禁
        self.WO = torch.tensor([W_OBJ[0], W_OBJ[1], W_OBJ[2]], device=device)
        self.rest = {oi: self.ref_obj[oi][0].clone() for oi in (0, 1)}
        self.mb = torch.tensor(mouth_local_bot, dtype=torch.float32, device=device)
        self.mc = torch.tensor(mouth_local_cup, dtype=torch.float32, device=device)
        self.up = torch.tensor(up_local_bot, dtype=torch.float32, device=device)
        self.mgate = float(mouth_gate)
        self.leash_rot_tilt = bool(leash_rot_tilt)
        self.kcap = int(kcap) if kcap is not None else None
        # 参考逐行倾角预计算 (E线口径用)
        upc = torch.tensor([0.0, 1.0, 0.0], device=device)
        self.ref_tilt = {oi: _tilt(self.ref_obj[oi][:, 3:7],
                                   self.up if oi == 1 else upc)
                         for oi in (0, 1)}
        # 状态
        self.k = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.ms1 = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.ms2 = torch.zeros_like(self.ms1)
        self.ms3 = torch.zeros_like(self.ms1)
        self.ms4 = torch.zeros_like(self.ms1)
        self.m2_run = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.m3_run = torch.zeros_like(self.m2_run)
        self.m4_run = torch.zeros_like(self.m2_run)
        self.m3_snap = {oi: torch.zeros(num_envs, 7, device=device) for oi in (0, 1)}
        # 药A 持握换基基线 (M1 后捕获)
        self.lb = {oi: torch.zeros(num_envs, 3, device=device) for oi in (0, 1)}
        self.lb_set = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # RSI 预置标记: 结账只认挣来的关 (预置计入达成会灌水 sr/gate*, 而 p_t0/相B
        # 两个单旋钮都吃这个数 —— 2026-08-28 训练冒烟实测 gate1 被灌到 1.0)
        self.pre1 = torch.zeros_like(self.ms1)
        self.pre2 = torch.zeros_like(self.ms1)
        self.pre3 = torch.zeros_like(self.ms1)
        self.done = torch.zeros_like(self.ms1)
        # TB 记账
        self._acc = {"ep": 0, "g1": 0, "g2": 0, "g3": 0, "g4": 0, "clock": 0.0}

    def reset_idx(self, env_ids):
        n = len(env_ids)
        if n:
            self._acc["ep"] += n
            self._acc["g1"] += int((self.ms1 & ~self.pre1)[env_ids].sum())
            self._acc["g2"] += int((self.ms2 & ~self.pre2)[env_ids].sum())
            self._acc["g3"] += int((self.ms3 & ~self.pre3)[env_ids].sum())
            self._acc["g4"] += int(self.ms4[env_ids].sum())
            self._acc["clock"] += float(self.k[env_ids].float().sum()) \
                / max(self.N_ROW - 1, 1)
        for t_ in (self.ms1, self.ms2, self.ms3, self.ms4, self.done,
                   self.pre1, self.pre2, self.pre3, self.lb_set):
            t_[env_ids] = False
        for oi in (0, 1):
            self.lb[oi][env_ids] = 0.0
        for t_ in (self.k, self.m2_run, self.m3_run, self.m4_run):
            t_[env_ids] = 0

    def enter(self, env_ids, rows, ms1, ms2, ms3):
        """RSI 批量进入: rows/ms* 均为对应 env_ids 的张量."""
        self.reset_idx(env_ids)
        self.k[env_ids] = rows
        self.ms1[env_ids] = ms1
        self.ms2[env_ids] = ms2
        self.ms3[env_ids] = ms3
        self.pre1[env_ids] = ms1
        self.pre2[env_ids] = ms2
        self.pre3[env_ids] = ms3
        for oi in (0, 1):
            self.m3_snap[oi][env_ids] = self.rest[oi].unsqueeze(0)                 .expand(len(env_ids), 7)

    def pop_rates(self):
        ep = max(self._acc["ep"], 1)
        out = {"sr/gate1": self._acc["g1"] / ep, "sr/gate2": self._acc["g2"] / ep,
               "sr/gate3": self._acc["g3"] / ep, "sr/gate4": self._acc["g4"] / ep,
               "prog/clock_frac": self._acc["clock"] / ep}
        self._acc = {"ep": 0, "g1": 0, "g2": 0, "g3": 0, "g4": 0, "clock": 0.0}
        return out

    def step(self, obj0, obj1, armq_r, armq_l, cand_ok, run_mask=None):
        """全部 (N,·) 张量。返回 dict of (N,) 张量。
        run_mask: (N,) bool, False 的 env 本步不受本机管辖(机器段行) —— 状态冻结、
        账目输出为零。接线参数, 不改判据语义(机器段不归本模块管辖, 见模块头)。"""
        k = self.k.clamp(max=self.N_ROW - 1)
        tier = self.tmix[k]
        w_obj = self.WO[tier]
        active = ~self.done if run_mask is None else (~self.done) & run_mask
        # 药A: M1 后首个受管步捕获持握基线
        cap = self.ms1 & (~self.lb_set) & active
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
            if self.leash_rot_tilt:
                upl = self.up if oi == 1 else torch.tensor(
                    [0.0, 1.0, 0.0], device=self.dev)
                dr = (_tilt(act[:, 3:7], upl) - self.ref_tilt[oi][k]).abs()
            else:
                dr = _qang(act[:, 3:7] / act[:, 3:7].norm(dim=1, keepdim=True)
                           .clamp(min=1e-9), ref[:, 3:7])
            lr = self.LR[trot]
            pen = ((dr - lr).clamp(min=0) / lr)
            leash = leash - torch.where(self.rot_ban[trot],
                                        torch.zeros_like(pen), pen)
        # 时钟耗尽停判皮筋 (与标量版同口径: 病尾冻结参考不再当判据)
        earning = (self.k < self.N_ROW - 1)
        leash = leash.clamp(min=-3.0) * active.float() * earning.float()
        # ---- 时钟门 ----
        ok_obj = torch.ones(self.Ne, dtype=torch.bool, device=self.dev)
        for oi, act in ((0, obj0), (1, obj1)):
            ref = self.ref_obj[oi][k]
            ok_obj &= ((act[:, :3] - ref[:, :3]).norm(dim=1) <= GATE_POS)
            ok_obj &= (_qang(act[:, 3:7] / act[:, 3:7].norm(dim=1, keepdim=True)
                             .clamp(min=1e-9), ref[:, 3:7]) <= GATE_ROT)
        ok_hand = torch.ones_like(ok_obj)
        for s, act in (("right", armq_r), ("left", armq_l)):
            ok_hand &= ((act - self.ref_arm[s][k]).abs().max(dim=1).values
                        <= np.radians(20))
        ok = torch.where(tier > 0, ok_obj, ok_hand) & active
        # 自主抓稳阶段: M1 前时钟不走 (与标量版同拍)
        _cap = self.N_ROW - 1 if self.kcap is None else min(self.kcap, self.N_ROW - 1)
        can = ok & (self.k < _cap) & self.ms1
        self.k = self.k + can.long()
        adv = can.float()
        # ---- 里程碑 ----
        ms_r = torch.zeros(self.Ne, device=self.dev)
        new1 = (~self.ms1) & cand_ok & active
        self.ms1 |= new1
        ms_r += new1.float() * MS_REWARD[1]
        # M2
        R1 = _q2R(obj1[:, 3:7])
        v = torch.einsum("nij,j->ni", R1, self.up)
        tilt = torch.acos((v[:, 2] / v.norm(dim=1).clamp(min=1e-9))
                          .clamp(-1, 1))
        mb_w = obj1[:, :3] + torch.einsum("nij,j->ni", R1, self.mb)
        R0 = _q2R(obj0[:, 3:7])
        mc_w = obj0[:, :3] + torch.einsum("nij,j->ni", R0, self.mc)
        m2_now = (tilt >= M2_TILT) & ((mb_w - mc_w).norm(dim=1) <= self.mgate)
        gate2 = self.ms1 & (~self.ms2) & active
        self.m2_run = torch.where(gate2 & m2_now, self.m2_run + 1,
                                  torch.zeros_like(self.m2_run))
        new2 = gate2 & (self.m2_run >= M2_HOLD)
        self.ms2 |= new2
        ms_r += new2.float() * MS_REWARD[2]
        # M3 (双物体 AND)
        ok3 = torch.ones_like(ok_obj)
        for oi, act in ((0, obj0), (1, obj1)):
            ok3 &= ((act[:, :3] - self.rest[oi][:3]).norm(dim=1) <= M3_POS)
            up3 = self.up if oi == 1 else torch.tensor([0.0, 1.0, 0.0],
                                                       device=self.dev)
            ok3 &= (_tilt(act[:, 3:7], up3) <= M3_ROT)   # 倾角口径, yaw豁免
        gate3 = self.ms2 & (~self.ms3) & active
        self.m3_run = torch.where(gate3 & ok3, self.m3_run + 1,
                                  torch.zeros_like(self.m3_run))
        new3 = gate3 & (self.m3_run >= M3_HOLD)
        self.ms3 |= new3
        ms_r += new3.float() * MS_REWARD[3]          # 放回≠功成: 不终局
        for oi, act in ((0, obj0), (1, obj1)):
            self.m3_snap[oi][new3] = act[new3]
        # ---- M4 = Success: 撤退归位 + 物体不被碰倒碰歪 (相对M3快照) ----
        ok4 = ((armq_r - self.stance["right"]).abs().max(dim=1).values <= M4_ARM)             & ((armq_l - self.stance["left"]).abs().max(dim=1).values <= M4_ARM)
        for oi, act in ((0, obj0), (1, obj1)):
            sn = self.m3_snap[oi]
            ok4 &= ((act[:, :3] - sn[:, :3]).norm(dim=1) <= M4_DIST_POS)
            up4 = self.up if oi == 1 else torch.tensor([0.0, 1.0, 0.0],
                                                       device=self.dev)
            ok4 &= (_tilt(act[:, 3:7], up4) <= M4_DIST_ROT)  # 倾角口径
        gate4 = self.ms3 & (~self.ms4) & active
        self.m4_run = torch.where(gate4 & ok4, self.m4_run + 1,
                                  torch.zeros_like(self.m4_run))
        new4 = gate4 & (self.m4_run >= M4_HOLD)
        self.ms4 |= new4
        self.done |= new4
        ms_r += new4.float() * MS_REWARD[4]
        # ---- 死线 D1/D2/D3/D8 (+D2交互段补丁: M1前倾角>60°=倒伏即终) ----
        fail = torch.zeros(self.Ne, dtype=torch.bool, device=self.dev)
        for oi, act in ((0, obj0), (1, obj1)):
            up_ = self.up if oi == 1 else torch.tensor([0.0, 1.0, 0.0],
                                                       device=self.dev)
            R_ = _q2R(act[:, 3:7])
            v_ = torch.einsum("nij,j->ni", R_, up_)
            tl = torch.acos((v_[:, 2] / v_.norm(dim=1).clamp(min=1e-9))
                            .clamp(-1, 1))
            fail |= (~self.ms1) & (tl > np.radians(60))
        for oi, act in ((0, obj0), (1, obj1)):
            up_ = self.up if oi == 1 else torch.tensor([0.0, 1.0, 0.0],
                                                       device=self.dev)
            fail |= (act[:, 2] < TABLE_Z - D1_DROP)
            ref = self.ref_obj[oi][k]
            fail |= (~self.ms3) & ((act[:, :3] - ref[:, :3]).norm(dim=1) > D3_DEV)
            sn = self.m3_snap[oi]
            tl = _tilt(act[:, 3:7], up_)
            fail |= self.ms3 & (((act[:, :3] - sn[:, :3]).norm(dim=1) > M4_DIST_POS)
                                | (tl > M4_DIST_ROT))
            fail |= self.ms3 & (tl > D2_TILT)
        fail &= active
        self.done |= fail
        return {"adv": adv, "leash": leash, "ms": ms_r, "w_obj": w_obj,
                "clock": self.k.clone(), "done": self.done.clone(),
                "tier": tier, "fail": fail}
