"""PourProgress 批量版 (torch, N envs) —— 训练接线用。

与 progress.py 同一套定稿数值/判据 (标量版为规格真源, 本文件是它的向量化镜像;
改判据必须两处同步并重跑 selftest_progress_batch.py 的一致性断言)。

L5-1 换代: 四Gate阶段机 + 5mm认证; 渐进RSI; P-OBJ/P-HYB 变体 (no_hand_ref)。
TB 记账: pop_rates() 吐 sr/gate1..4, prog/clock_frac, sr/cert_pass, prog/cert_att。
"""
from __future__ import annotations

import numpy as np
import torch

from progress import (LEASH_POS, LEASH_ROT, GATE_POS, GATE_ROT, RED_GATE_POS,
                      MS_REWARD, M2_TILT, M2_HOLD, M3_POS, M3_ROT, M3_HOLD,
                      M4_ARM, M4_HOLD, M4_DIST_POS, M4_DIST_ROT, W_OBJ, W_HAND,
                      _tier,
                      G1_HOLD, CERT_RAMP, CERT_HOLD, CERT_RET, CERT_RISE,
                      CERT_SLIP, CERT_WAIT, CERT_TRIES, WAGE, WAGE_CAP,
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
    """长轴倾角 (yaw 豁免口径): 局部 up 轴映到世界后与竖直夹角."""
    R = _q2R(quat / quat.norm(dim=1, keepdim=True).clamp(min=1e-9))
    v = torch.einsum("nij,j->ni", R, up)
    return torch.acos((v[:, 2] / v.norm(dim=1).clamp(min=1e-9)).clamp(-1, 1))


def _qang(a, b):
    d = (a * b).sum(-1).abs().clamp(max=1.0)
    return 2 * torch.acos(d)


def collide_flags(net_arm, net_pad, fil_pad, fm_objobj, fm_d6, thr=1.0):
    """禁碰判断(纯张量逻辑, 与 Isaac 无关, 便于自检)。

    约定: **只有手垫与自己要操作的物体可以接触**, 其余一切接触都是禁碰。
      net_arm  (N,Ka,3) 臂节/掌根的**净**接触力 —— 这些体不该碰任何东西
      net_pad  (N,Kp)   手垫净接触力模长(每垫取最大)
      fil_pad  (N,Kp)   手垫对**自己物体**的接触力模长
      fm_objobj(N,Ko,3) 瓶对杯的接触力
      fm_d6    (N,Kd,3) 右侧体对左侧体的接触力
    返回 dict of (N,) bool。
    ★判据: 垫的"净力 - 对自物体的力 > thr" ⟹ 还碰到了别的东西。这是模长差,
      不是矢量分解 —— 同向叠加时会低估, 故属**保守**(漏报而非误报)。
    """
    import torch as _t
    z = lambda x: _t.nan_to_num(x, nan=0.0)          # noqa: E731
    N = net_pad.shape[0] if net_pad is not None and net_pad.numel() else (
        net_arm.shape[0] if net_arm is not None else 0)
    out = {}
    out["arm"] = ((z(net_arm).norm(dim=-1) > thr).any(dim=1)
                  if net_arm is not None and net_arm.numel()
                  else _t.zeros(N, dtype=_t.bool, device=net_pad.device))
    out["pad"] = (((z(net_pad) - z(fil_pad)) > thr).any(dim=1)
                  if net_pad is not None and net_pad.numel()
                  else _t.zeros(N, dtype=_t.bool))
    for k, fm in (("objobj", fm_objobj), ("d6", fm_d6)):
        out[k] = ((z(fm).norm(dim=-1) > thr).any(dim=1)
                  if fm is not None and fm.numel()
                  else _t.zeros(N, dtype=_t.bool, device=out["arm"].device))
    return out


class PourProgressBatch:
    def __init__(self, npz_path, num_envs, device,
                 mouth_local_bot, mouth_local_cup,
                 up_local_bot=(0.0, 1.0, 0.0), mouth_gate=0.12,
                 leash_rot_tilt=False, kcap=None, no_hand_ref=False):
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
        self.no_hand_ref = bool(no_hand_ref)
        if self.no_hand_ref:
            self.WO = torch.ones(3, device=device)          # P-OBJ: w_obj 恒 1
            self.WH = torch.zeros(3, device=device)
        else:
            self.WO = torch.tensor([W_OBJ[0], W_OBJ[1], W_OBJ[2]], device=device)
            self.WH = torch.tensor([W_HAND[0], W_HAND[1], W_HAND[2]],
                                   device=device)
        self.rest = {oi: self.ref_obj[oi][0].clone() for oi in (0, 1)}
        self.mb = torch.tensor(mouth_local_bot, dtype=torch.float32, device=device)
        self.mc = torch.tensor(mouth_local_cup, dtype=torch.float32, device=device)
        self.up = torch.tensor(up_local_bot, dtype=torch.float32, device=device)
        self.upc = torch.tensor([0.0, 1.0, 0.0], device=device)
        self.mgate = float(mouth_gate)
        self.leash_rot_tilt = bool(leash_rot_tilt)
        self.kcap = int(kcap) if kcap is not None else None
        self.ref_tilt = {oi: _tilt(self.ref_obj[oi][:, 3:7],
                                   self.up if oi == 1 else self.upc)
                         for oi in (0, 1)}
        # 状态
        self.k = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.g1 = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.g2 = torch.zeros_like(self.g1)
        self.g3 = torch.zeros_like(self.g1)
        self.g4 = torch.zeros_like(self.g1)
        self.placed = torch.zeros_like(self.g1)
        self.g1_run = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.wage_paid = torch.zeros(num_envs, device=device)      # 药④ 逐回合工资账
        self.born_t0 = torch.ones(num_envs, dtype=torch.bool, device=device)
        # 认证机
        self.cert_phase = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.cert_t = torch.zeros_like(self.cert_phase)
        self.cert_try = torch.zeros_like(self.cert_phase)
        self.cert_wait = torch.zeros_like(self.cert_phase)
        self.cert_pending = torch.zeros_like(self.g1)
        self.cert_z0 = torch.zeros(num_envs, 2, device=device)
        self.cert_rel0 = {s: torch.zeros(num_envs, 3, device=device)
                          for s in ("right", "left")}
        self.m2_run = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.m3_run = torch.zeros_like(self.m2_run)
        self.m4_run = torch.zeros_like(self.m2_run)
        self.m3_snap = {oi: torch.zeros(num_envs, 7, device=device) for oi in (0, 1)}
        # 药A 持握换基基线 (G2 后捕获)
        self.lb = {oi: torch.zeros(num_envs, 3, device=device) for oi in (0, 1)}
        self.lb_set = torch.zeros_like(self.g1)
        # RSI 预置标记: 结账只认挣来的关
        self.pre1 = torch.zeros_like(self.g1)
        self.pre2 = torch.zeros_like(self.g1)
        self.pre3 = torch.zeros_like(self.g1)
        self.done = torch.zeros_like(self.g1)
        # TB 记账
        self._acc = {"ep": 0, "g1": 0, "g2": 0, "g3": 0, "g4": 0,
                     "clock": 0.0, "catt": 0, "cpass": 0,
                     "ep_t0": 0, "g1_t0": 0, "g2_t0": 0, "g3_t0": 0, "g4_t0": 0,
                     "cf_rise_bot": 0, "cf_rise_cup": 0, "cf_slip_r": 0,
                     "cf_slip_l": 0, "cf_pads": 0, "term_any": 0,
                     "term_D2pre": 0, "term_D1_drop": 0, "term_D3_dev": 0, "term_D8_disturb": 0, "term_D2_tilt": 0}

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
            # 药②: t0 出生口径 (预置出生不进分母, 消课程稀释偏差)
            t0m = self.born_t0[env_ids]
            self._acc["ep_t0"] += int(t0m.sum())
            for gk, gt in (("g1_t0", self.g1), ("g2_t0", self.g2),
                           ("g3_t0", self.g3), ("g4_t0", self.g4)):
                self._acc[gk] += int((gt[env_ids] & t0m).sum())
        for t_ in (self.g1, self.g2, self.g3, self.g4, self.placed, self.done,
                   self.pre1, self.pre2, self.pre3, self.lb_set,
                   self.cert_pending):
            t_[env_ids] = False
        for oi in (0, 1):
            self.lb[oi][env_ids] = 0.0
        self.cert_z0[env_ids] = 0.0
        for s in ("right", "left"):
            self.cert_rel0[s][env_ids] = 0.0
        for t_ in (self.k, self.g1_run, self.m2_run, self.m3_run, self.m4_run,
                   self.cert_phase, self.cert_t, self.cert_try, self.cert_wait):
            t_[env_ids] = 0
        self.wage_paid[env_ids] = 0.0
        self.born_t0[env_ids] = True

    def enter(self, env_ids, rows, g1, g2, g3, placed):
        """RSI 批量进入: rows/g* 均为对应 env_ids 的张量."""
        self.reset_idx(env_ids)
        self.k[env_ids] = rows
        self.g1[env_ids] = g1
        self.g2[env_ids] = g2
        self.g3[env_ids] = g3
        self.placed[env_ids] = placed
        self.pre1[env_ids] = g1
        self.pre2[env_ids] = g2
        self.pre3[env_ids] = g3
        self.born_t0[env_ids] = ~(g1 | g2 | g3 | placed)
        for oi in (0, 1):
            self.m3_snap[oi][env_ids] = self.rest[oi].unsqueeze(0) \
                .expand(len(env_ids), 7)

    def pop_rates(self):
        # ★L5-17: 分母为 0 时必须发 NaN, 不得发 0.0 ——
        # 原写法 max(ep,1) 把"本窗没有任何回合结算"渲染成"成功率 0%", 二者在图上
        # 一模一样。实测被这个坑过一次: EASY 线开局 20 窗全 0, 我一度判成"全部失败",
        # 真相是回合太长(903步)还没有一个结算。同族: 缺数据被写成一个正常的值。
        # 同时把分母本身作为指标发出去 —— 只有计数能让"没发生的事"可见。
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
               "n/ep_done": float(_ep),          # ★分母本身: 判读前先看它
               "n/ep_done_t0": float(_ep0),
               "n/cert_attempts": float(_at)}
        # F: 认证失败分项占比 (分母=认证尝试数; 空分母同样发 NaN, 不发 0.0)
        for _k in ("rise_bot", "rise_cup", "slip_r", "slip_l", "pads"):
            out["cert_fail/" + _k] = _r(self._acc["cf_" + _k], _at)
        # L5-23 死因分项 (分母=本窗判据侧死亡数; 空分母同样发 NaN)
        _tn = self._acc["term_any"]
        for _k in ("D2pre", "D1_drop", "D3_dev", "D8_disturb", "D2_tilt"):
            out["term/" + _k] = _r(self._acc["term_" + _k], _tn)
        out["n/term_judge"] = float(_tn)
        self._acc = {"ep": 0, "g1": 0, "g2": 0, "g3": 0, "g4": 0,
                     "clock": 0.0, "catt": 0, "cpass": 0,
                     "ep_t0": 0, "g1_t0": 0, "g2_t0": 0, "g3_t0": 0, "g4_t0": 0,
                     "cf_rise_bot": 0, "cf_rise_cup": 0, "cf_slip_r": 0,
                     "cf_slip_l": 0, "cf_pads": 0, "term_any": 0,
                     "term_D2pre": 0, "term_D1_drop": 0, "term_D3_dev": 0, "term_D8_disturb": 0, "term_D2_tilt": 0}
        return out

    def step(self, obj0, obj1, armq_r, armq_l, pads3, wrist_r, wrist_l,
             run_mask=None):
        """全部 (N,·) 张量; pads3 (N,)bool 双手各>=3/5垫; wrist_r/l (N,3)。
        run_mask: False 的 env 本步不受本机管辖(机器段行)。"""
        k = self.k.clamp(max=self.N_ROW - 1)
        tier = self.tmix[k]
        w_obj = self.WO[tier]
        active = ~self.done if run_mask is None else (~self.done) & run_mask
        # 药A: G2 后首个受管步捕获持握基线
        cap = self.g1 & (~self.lb_set) & active     # 药⑤: G1 起换基
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
                upl = self.up if oi == 1 else self.upc
                dr = (_tilt(act[:, 3:7], upl) - self.ref_tilt[oi][k]).abs()
            else:
                dr = _qang(act[:, 3:7] / act[:, 3:7].norm(dim=1, keepdim=True)
                           .clamp(min=1e-9), ref[:, 3:7])
            lr = self.LR[trot]
            pen = ((dr - lr).clamp(min=0) / lr)
            leash = leash - torch.where(self.rot_ban[trot],
                                        torch.zeros_like(pen), pen)
        earning = (self.k < self.N_ROW - 1)
        leash = leash.clamp(min=-3.0) * active.float() * earning.float()
        # ---- 时钟门 (L5-6: 双变体统一; 红档=宽松物门) ----
        ok_obj = torch.ones(self.Ne, dtype=torch.bool, device=self.dev)
        gp = torch.where(tier > 0,
                         torch.full_like(w_obj, GATE_POS),
                         torch.full_like(w_obj, RED_GATE_POS))
        for oi, act in ((0, obj0), (1, obj1)):
            ref = self.ref_obj[oi][k]
            ok_obj &= ((act[:, :3] - ref[:, :3]).norm(dim=1) <= gp)
            rot_ok = (_qang(act[:, 3:7] / act[:, 3:7].norm(dim=1, keepdim=True)
                            .clamp(min=1e-9), ref[:, 3:7]) <= GATE_ROT)
            ok_obj &= torch.where(tier > 0, rot_ok,
                                  torch.ones_like(rot_ok))   # 红档 rot 禁入
        ok = ok_obj & active
        # 时钟: G2 前不走
        _cap = self.N_ROW - 1 if self.kcap is None else min(self.kcap, self.N_ROW - 1)
        can = ok & (self.k < _cap) & self.g2
        self.k = self.k + can.long()
        adv = can.float()
        ms_r = torch.zeros(self.Ne, device=self.dev)
        # ---- G1 ----
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
        # ---- G2 认证机 ----
        self.cert_wait = (self.cert_wait - 1).clamp(min=0)
        start = (self.g1 & (~self.g2) & (self.cert_phase == 0)
                 & (self.cert_wait == 0) & (self.cert_try < CERT_TRIES) & active)
        if start.any():
            self.cert_phase[start] = 1
            self.cert_t[start] = 0
            self.cert_z0[start, 0] = obj0[start, 2]
            self.cert_z0[start, 1] = obj1[start, 2]
            self.cert_rel0["right"][start] = (wrist_r - obj1[:, :3])[start]
            self.cert_rel0["left"][start] = (wrist_l - obj0[:, :3])[start]
        in_cert = (self.cert_phase > 0) & (~start) & active
        self.cert_t = torch.where(in_cert, self.cert_t + 1, self.cert_t)
        to_hold = in_cert & (self.cert_phase == 1) & (self.cert_t >= CERT_RAMP)
        self.cert_phase[to_hold] = 2
        self.cert_t[to_hold] = 0
        judge = in_cert & (self.cert_phase == 2) & (self.cert_t >= CERT_HOLD)
        if judge.any():
            rel_r = ((wrist_r - obj1[:, :3]) - self.cert_rel0["right"]).norm(dim=1)
            rel_l = ((wrist_l - obj0[:, :3]) - self.cert_rel0["left"]).norm(dim=1)
            # F (L5-21): 与标量版逐项对齐 —— 分项失败计数
            _cj = (("rise_bot", obj1[:, 2] - self.cert_z0[:, 1] >= CERT_RISE),
                   ("rise_cup", obj0[:, 2] - self.cert_z0[:, 0] >= CERT_RISE),
                   ("slip_r", rel_r < CERT_SLIP),
                   ("slip_l", rel_l < CERT_SLIP),
                   ("pads", pads3))
            ok5 = _cj[0][1] & _cj[1][1] & _cj[2][1] & _cj[3][1] & _cj[4][1]
            for _k, _okm in _cj:
                self._acc["cf_" + _k] += int((judge & (~_okm)).sum())
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
        # ---- G3 倒水完成 ----
        R1 = _q2R(obj1[:, 3:7])
        v = torch.einsum("nij,j->ni", R1, self.up)
        tilt = torch.acos((v[:, 2] / v.norm(dim=1).clamp(min=1e-9))
                          .clamp(-1, 1))
        mb_w = obj1[:, :3] + torch.einsum("nij,j->ni", R1, self.mb)
        R0 = _q2R(obj0[:, 3:7])
        mc_w = obj0[:, :3] + torch.einsum("nij,j->ni", R0, self.mc)
        m2_now = (tilt >= M2_TILT) & ((mb_w - mc_w).norm(dim=1) <= self.mgate)
        gate3 = self.g2 & (~self.g3) & active
        self.m2_run = torch.where(gate3 & m2_now, self.m2_run + 1,
                                  torch.zeros_like(self.m2_run))
        new3 = gate3 & (self.m2_run >= M2_HOLD)
        self.g3 |= new3
        ms_r += new3.float() * MS_REWARD[3]
        # ---- placed (不付奖) ----
        ok3 = torch.ones_like(ok_obj)
        for oi, act in ((0, obj0), (1, obj1)):
            ok3 &= ((act[:, :3] - self.rest[oi][:3]).norm(dim=1) <= M3_POS)
            up3 = self.up if oi == 1 else self.upc
            ok3 &= (_tilt(act[:, 3:7], up3) <= M3_ROT)   # 倾角口径, yaw豁免
        gatep = self.g3 & (~self.placed) & active
        self.m3_run = torch.where(gatep & ok3, self.m3_run + 1,
                                  torch.zeros_like(self.m3_run))
        newp = gatep & (self.m3_run >= M3_HOLD)
        self.placed |= newp
        for oi, act in ((0, obj0), (1, obj1)):
            self.m3_snap[oi][newp] = act[newp]
        # ---- G4 = Success ----
        ok4 = ((armq_r - self.stance["right"]).abs().max(dim=1).values <= M4_ARM) \
            & ((armq_l - self.stance["left"]).abs().max(dim=1).values <= M4_ARM)
        for oi, act in ((0, obj0), (1, obj1)):
            sn = self.m3_snap[oi]
            ok4 &= ((act[:, :3] - sn[:, :3]).norm(dim=1) <= M4_DIST_POS)
            up4 = self.up if oi == 1 else self.upc
            ok4 &= (_tilt(act[:, 3:7], up4) <= M4_DIST_ROT)  # 倾角口径
        gate4 = self.placed & (~self.g4) & active
        self.m4_run = torch.where(gate4 & ok4, self.m4_run + 1,
                                  torch.zeros_like(self.m4_run))
        new4 = gate4 & (self.m4_run >= M4_HOLD)
        self.g4 |= new4
        self.done |= new4
        ms_r += new4.float() * MS_REWARD[4]
        # ---- 死线 D1/D2/D3/D8 (D2pre: G2前倾>60°) ----
        # ★L5-23 死因分项: 原来五种死因 OR 进一个 fail, 死了却说不出为什么死。
        # G4|G3 只有 0.10~0.31, 但"卡在放回还是撤退"无法回答 —— 因为 D8(撤退期扰动)
        # 和 D1/D3(放回前) 混在同一个布尔里。分项后二者可分。
        _cz = lambda: torch.zeros(self.Ne, dtype=torch.bool, device=self.dev)
        _fc = {"D2pre": _cz(), "D1_drop": _cz(), "D3_dev": _cz(),
               "D8_disturb": _cz(), "D2_tilt": _cz()}
        for oi, act in ((0, obj0), (1, obj1)):
            up_ = self.up if oi == 1 else self.upc
            tl = _tilt(act[:, 3:7], up_)
            _fc["D2pre"] |= (~self.g2) & (tl > np.radians(60))
        for oi, act in ((0, obj0), (1, obj1)):
            up_ = self.up if oi == 1 else self.upc
            _fc["D1_drop"] |= (act[:, 2] < TABLE_Z - D1_DROP)
            ref = self.ref_obj[oi][k]
            _fc["D3_dev"] |= (~self.placed) & (
                (act[:, :3] - ref[:, :3]).norm(dim=1) > D3_DEV)
            sn = self.m3_snap[oi]
            tl = _tilt(act[:, 3:7], up_)
            _fc["D8_disturb"] |= self.placed & (
                ((act[:, :3] - sn[:, :3]).norm(dim=1) > M4_DIST_POS)
                | (tl > M4_DIST_ROT))
            _fc["D2_tilt"] |= self.placed & (tl > D2_TILT)
        fail = _cz()
        for _k, _v in _fc.items():
            fail |= _v
        fail &= active
        # 记账: fail 即刻置 done, 故 (因 & active) 恰好在终止那一步命中一次;
        # 一次死亡可同时命中多因, 各自计数(占比之和可 >1, 判读时按此理解)。
        for _k, _v in _fc.items():
            self._acc["term_" + _k] += int((_v & active).sum())
        self._acc["term_any"] += int(fail.sum())
        self.done |= fail
        return {"adv": adv, "leash": leash, "ms": ms_r, "wage": wage,
                "w_obj": w_obj, "w_hand": self.WH[tier], "clock": self.k.clone(), "done": self.done.clone(),
                "tier": tier, "fail": fail,
                "cert_phase": self.cert_phase.clone(),
                "cert_t": self.cert_t.clone()}
