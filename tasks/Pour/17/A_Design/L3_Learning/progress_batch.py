"""PourProgress 批量版 (torch, N envs) —— 训练接线用。

与 progress.py 同一套定稿数值/判据 (标量版为规格真源, 本文件是它的向量化镜像;
改判据必须两处同步并重跑 selftest_progress_batch.py 的一致性断言)。

L5-1 换代: 四Gate阶段机 + 5mm认证; 渐进RSI; P-OBJ/P-HYB 变体 (no_hand_ref)。
TB 记账: pop_rates() 吐 sr/gate1..4, prog/clock_frac, sr/cert_pass, prog/cert_att。
"""
from __future__ import annotations

import os
import numpy as np
import torch

from progress import (M2_STRICT_HORIZ, M2_STRICT_DZ_LO, M2_STRICT_DZ_HI,
                      PLACE_SHAPE_K, PLACE_SHAPE_D0, PLACE_SHAPE_T0,
                      LEASH_POS, LEASH_ROT, GATE_POS, GATE_ROT, RED_GATE_POS,
                      MS_REWARD, M2_TILT, M2_HOLD, M3_POS, M3_ROT, M3_HOLD,
                      M4_ARM, M4_HOLD, M4_DIST_POS, M4_DIST_ROT, W_OBJ, W_HAND,
                      _tier,
                      G1_HOLD, CERT_RAMP, CERT_HOLD, CERT_RET, CERT_RISE,
                      CERT_SLIP, CERT_WAIT, CERT_TRIES, WAGE, WAGE_CAP,
                      HOLD_POSE, HOLD_V2, HOLD_MODE, HOLD_TILT_TOL, HOLD_DRIFT_TOL,
                      HOLD_PEN_TILT0, HOLD_PEN_TILT1, HOLD_PEN_DRIFT0, HOLD_PEN_DRIFT1,
                      HOLD_K_TILT, HOLD_K_DRIFT,
                      MIN_RECIPE, MIN_CERT_POS, MIN_CERT_ROT, MIN_REL_POS, MIN_REL_ROT,
                      PLACE_TERMINAL, PLACE_REWARD,
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
                 leash_rot_tilt=False, kcap=None, no_hand_ref=False,
                 goal_only=False):
        # ★L5-37 多母带: npz_path 可逗号分隔多条 (格式一致, 只差物体静置位)。ref_obj 堆成
        #   (N_TAPE, oi, N_ROW, 7); 每 env 挂 tape_id, judge 时按 tape gather。置信度/mouth/up 取首条
        #   (同 recon, 平移不变)。单条时逐位与旧行为一致。
        _paths = [p.strip() for p in npz_path.split(",") if p.strip()]
        self.N_TAPE = len(_paths)
        _zs = [np.load(p, allow_pickle=True) for p in _paths]
        z = _zs[0]
        rows = np.where(np.asarray(z["source"]) == 1)[0]
        self.N_ROW = len(rows)
        self.Ne, self.dev = int(num_envs), device
        f32 = lambda a: torch.tensor(np.asarray(a, np.float64)[rows],
                                     dtype=torch.float32, device=device)
        # ref_obj_stack: (N_TAPE, 2, N_ROW, 7)
        _ro = np.stack([np.stack([np.concatenate([
            np.asarray(_z[f"obj_pos_{oi}"], np.float64)[rows],
            np.asarray(_z[f"obj_quat_{oi}"], np.float64)[rows]], axis=1) for oi in (0, 1)])
            for _z in _zs])
        self.ref_obj_stack = torch.tensor(_ro, dtype=torch.float32, device=device)
        self.ref_obj = {oi: self.ref_obj_stack[0, oi] for oi in (0, 1)}   # 单母带口径视图
        self.tape_id = torch.zeros(int(num_envs), dtype=torch.long, device=device)
        self.stance = {s: torch.tensor(np.asarray(z[f"{s}_q"], np.float64)[-1],
                                       dtype=torch.float32, device=device)
                       for s in ("right", "left")}
        def tiers_np(key):
            return np.array([(_tier(v) if _tier(v) is not None else 2)
                             for v in np.asarray(z[key])[rows]], np.int64)
        _tp = {oi: tiers_np(f"conf_pos_{oi}") for oi in (0, 1)}
        _tr = {oi: tiers_np(f"conf_rot_{oi}") for oi in (0, 1)}
        # ★L5-31 消融旗 POUR_CONF_FLAT: 把逐帧置信度**整条链路**拍平到黄档(tier=1)。
        #   全部消费者都通过 tp/tr/tmix 索引, 所以改这一处, 下游六项自动跟着变:
        #     ① W_OBJ/W_HAND 0.5/0.5   ② 皮筋位置 恒5cm   ③ 皮筋朝向 恒30°且红档禁判失效
        #     ④ 时钟门 恒5cm(不再走红档8cm宽门)  ⑤ 残差界 恒0.08  ⑥ regime 观测变常量
        #   ★第⑥项才是本质: 策略从"知道这一行可不可信"变成"完全不知道"。
        #   ★不对称提示 (写进台账 L5-31): OBJ 变体走 no_hand_ref 分支, W_OBJ 本就恒 1.0
        #     ⟹ base_o vs flat_o 只差 ②③④⑤⑥, 不差 ①。置信度有两个作用:
        #     (a) 在物体与人手间分权 —— 只有带手时存在; (b) 调容差/门/残差界/告知策略 ——
        #     两种都有。判读 2×2 的交互项时必须记住这个不对称, 别当成发现。
        self.conf_flat = os.environ.get("POUR_CONF_FLAT") == "1"
        if self.conf_flat:
            for _d in (_tp, _tr):
                for oi in (0, 1):
                    _d[oi][:] = 1
            print("[PB] ★POUR_CONF_FLAT=1: 逐帧置信度已拍平到黄档 —— "
                  "权重/皮筋/时钟门/残差界/regime观测 全部恒定", flush=True)
        # ★L5-36 通用设计 G-B (接触起始黄窗) + 对照变换 (打乱/反向): 拍平 → 打乱/反向 → 下限。
        #   共享 helper rl_rebuild/correction/tier_floor.py, 标量版 progress.py 走同一函数。
        #   tier_info 由 world_fingerprint 记入 switches (CRITICAL)。
        from rl_rebuild.correction import tier_floor as _TF
        from progress import CERT_RISE as _CERT_RISE
        _tp, _tr, _tm, self.tier_info = _TF.transform(
            _tp, _tr, np.asarray(z["obj_pos_1"], np.float64)[rows], _CERT_RISE, tag="PB")
        self.tp = {oi: torch.tensor(_tp[oi], dtype=torch.long, device=device) for oi in (0, 1)}
        self.tr = {oi: torch.tensor(_tr[oi], dtype=torch.long, device=device) for oi in (0, 1)}
        self.tmix = torch.tensor(_tm, dtype=torch.long, device=device)
        self.LP = torch.tensor([LEASH_POS[0], LEASH_POS[1], LEASH_POS[2]],
                               device=device)
        self.LR = torch.tensor([float(LEASH_ROT[0] or 1e9),
                                LEASH_ROT[1], LEASH_ROT[2]], device=device)
        # ★2026-09-04 Group A: POUR_LEASH_FLAT=Y(米) 把位置皮筋**统一**成 Y(不分档)。
        _leash_flat = os.environ.get("POUR_LEASH_FLAT")
        if _leash_flat:
            self.LP = torch.full((3,), float(_leash_flat), device=device)
            print(f"[PB] ★POUR_LEASH_FLAT={_leash_flat}: 位置皮筋统一={float(_leash_flat)}m",
                  flush=True)
        self.rot_ban = torch.tensor([True, False, False], device=device)  # 红档rot禁
        self.no_hand_ref = bool(no_hand_ref)
        # ★L5-32 `goal` 臂 (只有目标, 没有轨迹): 去掉**一切读参考轨迹形状**的
        #   奖励与死线, 并**冻结时钟** —— 时钟决定 `row = IA0 + k`, 即前馈取母带
        #   哪一行; 不冻它, "没有轨迹"就是假的(策略仍在按母带的时间表拿前馈)。
        #   保留的: ms(里程碑, 绝对量) · wage(抓握维持费, 不读参考) ·
        #           pen/pen_slope/r_reflex/pen6/bonus/lift(接触与物理, 绝对量)
        #   去掉的: adv(时钟推进奖) · leash(离参考罚) · D3_dev(离参考死线)
        #           + env 侧 r_shape(人手形状指引, 读人手参考)
        self.goal_only = bool(goal_only)
        # ★R (2026-09-04): 关系表征 —— leash+时钟门 从"世界绝对物体位姿"换成"瓶口↔杯口关系向量"
        #   [ρ 水平口距, Δz 高差, θ 瓶倾角]。方案C: 视频=软趋势先验, 单边奖励(允许超过视频值
        #   直到物理 success), 不硬追 r_ref 的精确值。默认关(=绝对轨迹现状)。
        self.rel_ref = os.environ.get("POUR_REL_REF") == "1"
        self.REL_TOL_HZ = float(os.environ.get("POUR_REL_TOL_HZ", "0.03"))
        self.REL_TOL_DZ = float(os.environ.get("POUR_REL_TOL_DZ", "0.03"))
        self.REL_TOL_TH = float(os.environ.get("POUR_REL_TOL_TH", "0.15"))
        self.REL_GATE_HZ = float(os.environ.get("POUR_REL_GATE_HZ", "0.05"))
        self.REL_GATE_TH = float(os.environ.get("POUR_REL_GATE_TH", "0.30"))
        if self.rel_ref:
            print(f"[PB] ★POUR_REL_REF=1: 关系表征 leash+时钟门(方案C单边) "
                  f"tol ρ{self.REL_TOL_HZ}/Δz{self.REL_TOL_DZ}/θ{self.REL_TOL_TH}rad "
                  f"gate ρ{self.REL_GATE_HZ}/θ{self.REL_GATE_TH}rad", flush=True)
        if self.no_hand_ref:
            self.WO = torch.ones(3, device=device)          # P-OBJ: w_obj 恒 1
            self.WH = torch.zeros(3, device=device)
        else:
            self.WO = torch.tensor([W_OBJ[0], W_OBJ[1], W_OBJ[2]], device=device)
            self.WH = torch.tensor([W_HAND[0], W_HAND[1], W_HAND[2]],
                                   device=device)
        # rest_stack: (N_TAPE, 2, 7) 每条母带的静置位; rest 单母带口径视图。
        self.rest_stack = self.ref_obj_stack[:, :, 0].clone()   # (N_TAPE,2,7)
        self.rest = {oi: self.rest_stack[0, oi] for oi in (0, 1)}
        # ★2026-09-03 B 课程: 逐带 success/ep 累加 (t0 出生口径), 供 train_pour 算 ALP + 监控遗忘。
        self._acc_tsucc = torch.zeros(self.N_TAPE, device=device)
        self._acc_tep = torch.zeros(self.N_TAPE, device=device)
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
        # ★L5-36.3 抓稳期姿态守恒: 认证起点的倾角/xy, 与斜坡+保持期内的最大偏离 (判决用最大值, 不只看终态)
        self.hold_pose = bool(HOLD_POSE)
        self.cert_tilt0 = torch.zeros(num_envs, 2, device=device)
        self.cert_xy0 = torch.zeros(num_envs, 2, 2, device=device)
        self.cert_tilt_dev = torch.zeros(num_envs, 2, device=device)
        self.cert_drift = torch.zeros(num_envs, 2, device=device)
        # ★POUR_MIN: 斜坡+保持期内 手物相对位移最大值 [杯(左), 瓶(右)] (m)
        self.min_recipe = bool(MIN_RECIPE)
        self.cert_relmax = torch.zeros(num_envs, 2, device=device)
        self.hold_v2 = bool(HOLD_V2)
        if self.hold_pose:
            print(f"[PB] ★POUR_HOLD_POSE={HOLD_MODE}: 认证加 倾角≤{np.degrees(HOLD_TILT_TOL):.0f}° / 漂移≤{HOLD_DRIFT_TOL*100:.0f}cm "
                  f"({'v2: 相对参考静置的绝对倾角/漂移' if self.hold_v2 else 'v1: 认证内偏离'}); "
                  f"渐进罚 倾角{np.degrees(HOLD_PEN_TILT0):.0f}°→{np.degrees(HOLD_PEN_TILT1):.0f}° K={HOLD_K_TILT}, "
                  f"漂移{HOLD_PEN_DRIFT0*1000:.0f}mm→{HOLD_PEN_DRIFT1*100:.0f}cm K={HOLD_K_DRIFT} "
                  f"(窗口 {'缝1合拢起→G2, env 侧算' if self.hold_v2 else 'G1→G2'})", flush=True)
        self.m2_run = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.m2l_run = torch.zeros_like(self.m2_run)   # ★旧松判据影子(诊断)
        # ★L5-33 placed 绝对整形 (旗控, 关旗时行为与基线逐位一致):
        #   earn-only 棘轮 —— 只为"新高"付钱, 来回晃不挣钱(复用相B赏钱的已验证机制)。
        #   由来: AB2 终表 placed 仅 base_o 0.061, 其余全 0; placed 此前一分钱不付,
        #   唯一支付路径是 G4 的 +15 且还要求双臂回站姿 —— 极稀疏长链。
        self.place_shape = os.environ.get("POUR_PLACE_SHAPE") == "1"
        self.place_pot = torch.zeros(num_envs, device=device)
        # ★L5-36.7 修真空 (旗 POUR_PLACE_BOTH=1, 用户 2026-09-02 批准): place 势拆瓶/杯两份独立棘轮。
        #   探针实锤: 原 φ 只读瓶 ("杯在正常回合里不动"的假设已过时 —— 策略把杯举到半空接水),
        #   杯的放回零梯度 ⟹ s54 型 "瓶回杯不回"。双棘轮各 K/2, 总封顶不变 (=PLACE_SHAPE_K)。
        self.place_both = os.environ.get("POUR_PLACE_BOTH") == "1"
        self.place_pot2 = torch.zeros(num_envs, 2, device=device)
        if self.place_both:
            print(f"[PB] ★POUR_PLACE_BOTH=1: place 势拆 瓶/杯 双棘轮, 各 K={PLACE_SHAPE_K/2:.1f}, "
                  f"总封顶 {PLACE_SHAPE_K:.1f} 不变", flush=True)
        self.place_terminal = bool(PLACE_TERMINAL)
        if self.place_terminal:
            print(f"[PB] ★POUR_PLACE_TERMINAL=1: placed(双物≤3cm/≤5°, 保持{M3_HOLD}步)=成功终止 + 付 {PLACE_REWARD:.0f} 终局奖; "
                  f"撤离G4退役(诊断影子); RSI丢撤离出生点。placed倾角判据={np.degrees(M3_ROT):.0f}°", flush=True)
        # ★2026-09-03 ③ 抓握保持到 placed (旗 POUR_GRIP_HOLD=1): g3 达成、placed 未认证
        #   期间脱握(pads3=False)每步罚 K。动机: D3_dev(>35cm)只有松手才可能 —— 残差界下
        #   抓着的物体离参考不过几 cm。逼"抓着放稳"而非"倒完撒手→物体甩飞→D3_dev 死"。
        self.grip_hold = os.environ.get("POUR_GRIP_HOLD") == "1"
        self.grip_hold_k = float(os.environ.get("POUR_GRIP_HOLD_K", "0.5"))
        if self.grip_hold:
            print(f"[PB] ★POUR_GRIP_HOLD=1: 放置期(g3达成→placed)脱握每步罚 {self.grip_hold_k:.2f} "
                  f"—— 掐掉'倒完撒手→甩飞→D3_dev'链", flush=True)
        # ★2026-09-03 ① D3_dev 宽限期 (旗 POUR_PLACE_GRACE=N>0): g3 达成后 N 步内不判偏离死,
        #   给策略把物体收回窗的时间。g3_age = g3 后连续步数 (非 g3 时清零)。
        self.place_grace = int(os.environ.get("POUR_PLACE_GRACE", "0"))
        self.g3_age = torch.zeros(num_envs, dtype=torch.long, device=device)
        if self.place_grace > 0:
            print(f"[PB] ★POUR_PLACE_GRACE={self.place_grace}: g3达成后 {self.place_grace} 步内不判 D3_dev 偏离死",
                  flush=True)
        # ★2026-09-03 ④ placed 容差课程 (旗 POUR_PLACE_CURR=1): M3_ROT/M3_POS 松→紧退火,
        #   由 train_pour 按 EMA(placed) 单向棘轮收紧。digest 仍按最终(紧)M3_ROT 算, eval 用最终值。
        self.place_curr = os.environ.get("POUR_PLACE_CURR") == "1"
        self._pc_rot0, self._pc_pos0 = np.radians(15.0), 0.05    # 起始松容差
        self._pc_lmax = 4
        self.pc_level = 0
        if self.place_curr:
            self.m3_rot_cur, self.m3_pos_cur = self._pc_rot0, self._pc_pos0
            print(f"[PB] ★POUR_PLACE_CURR=1: placed容差课程 "
                  f"{np.degrees(self._pc_rot0):.0f}°/{self._pc_pos0*100:.0f}cm → "
                  f"{np.degrees(M3_ROT):.0f}°/{M3_POS*100:.0f}cm ({self._pc_lmax}档, EMA(placed)棘轮)", flush=True)
        else:
            self.m3_rot_cur, self.m3_pos_cur = M3_ROT, M3_POS
        self.g3_loose = torch.zeros(num_envs, dtype=torch.bool, device=device)
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
                     "clock": 0.0, "catt": 0, "cpass": 0, "cdone": 0, "cdone_t0": 0, "cge90": 0, "succ": 0, "succ_t0": 0,
                     "g3l": 0, "g3l_t0": 0, "plc": 0, "plc_t0": 0,
                     "ep_t0": 0, "g1_t0": 0, "g2_t0": 0, "g3_t0": 0, "g4_t0": 0,
                     "cf_rise_bot": 0, "cf_rise_cup": 0, "cf_slip_r": 0,
                     "cf_slip_l": 0, "cf_pads": 0, "cf_tilt": 0, "cf_drift": 0, "term_any": 0,
                     "cf_relp_r": 0, "cf_relp_l": 0, "cf_tilt_r": 0, "cf_tilt_l": 0,
                     "term_D2pre": 0, "term_D1_drop": 0, "term_D3_dev": 0, "term_D8_disturb": 0, "term_D2_tilt": 0}

    def reset_idx(self, env_ids):
        n = len(env_ids)
        if n:
            self._acc["ep"] += n
            self._acc["g1"] += int((self.g1 & ~self.pre1)[env_ids].sum())
            self._acc["g2"] += int((self.g2 & ~self.pre2)[env_ids].sum())
            self._acc["g3"] += int((self.g3 & ~self.pre3)[env_ids].sum())
            self._acc["g4"] += int(self.g4[env_ids].sum())
            _gl = self.g3_loose[env_ids] & (~self.pre3[env_ids])
            self._acc["g3l"] += int(_gl.sum())
            self._acc["clock"] += float(self.k[env_ids].float().sum()) \
                / max(self.N_ROW - 1, 1)
            # ★ 纯记账: "物体跟着参考走完全程"的逐回合通过率。
            #   clock_frac 是**平均走了多远**, 回答不了"有多少比例走完了" ——
            #   两者在图上长得一样, 但 0.768 的平均可以由"全都走 77%"或
            #   "77% 走完+23% 原地"产生, 判读完全不同。
            _cdone = (self.k[env_ids] >= self.N_ROW - 1)
            self._acc["cdone"] += int(_cdone.sum())
            self._acc["cdone_t0"] += int((_cdone & self.born_t0[env_ids]).sum())
            # ★躲门哨兵 (L5-31): 主判据换成 clock_done 后**不改终止条件** ——
            #   走完时钟后回合继续跑撤退段, 在那里死掉要扣 -10。理论上策略可以学会
            #   "不走完以躲开后面的死亡"(台账: 门后更差 ⟹ 躲门)。
            #   判读: ge90 ≈ cdone 正常; **ge90 ≫ cdone = 大量回合停在末尾几行不走完**。
            _ge90 = (self.k[env_ids].float() >= 0.90 * (self.N_ROW - 1))
            self._acc["cge90"] += int(_ge90.sum())
            # ★L5-32 主判据 = **G3_pour ∧ placed** (用户裁定 2026-08-31)。
            #   两项都是**纯绝对量**: 只读瓶位姿/杯位姿/rest(=母带第0帧), 一个字
            #   都不读参考轨迹的形状。这是换掉 clock_done 的根本理由 ——
            #   clock 的门是**逐行按置信档取的**: `gp = where(tier>0, 5cm, 8cm)`,
            #   且 `rot_ok` 在红档**根本不判**。base 有红档、flat 没有, straight/goal
            #   又是另一套 ⟹ 六条臂用的不是同一把尺, 跨臂比较从根上不成立。
            #   (旧铁证仍然成立且已归档: P17v6_OBJ_s14 9M 步 G3 0.73/clock 0.19,
            #    28M 步 G3 0.49/clock 0.95 —— 越训越会走轨迹, 越训越不倒水。)
            #   rest 已核实在 v3 与 v3noconf 两条母带上**逐位相同**:
            #     瓶 (-0.1279,+0.0011,+0.9568) · 杯 (-0.1401,+0.1646,+0.9360)
            #   ⟹ 新判据跨母带、跨臂同尺, 六条臂第一次可以直接放一起比。
            #   可达性已核 (防 L5-27"判据比参考自身还严"重演):
            #     G3_pour 母带连续满足 v3=42行 / v3noconf=89行 (需 M2_HOLD=25) ✓
            #     placed  母带末态精确等于第0帧 ✓
            #   clock_done 降级为诊断量, 仍在 sr/clock_done 报。
            _plc = self.placed[env_ids]
            self._acc["plc"] += int(_plc.sum())
            self._acc["plc_t0"] += int((_plc & self.born_t0[env_ids]).sum())
            _succ = self.g3[env_ids] & _plc
            self._acc["succ"] += int(_succ.sum())
            self._acc["succ_t0"] += int((_succ & self.born_t0[env_ids]).sum())
            # ★B: 逐带 t0 进度分累加 (scatter by tape_id) —— ALP 课程信号 + 遗忘监控。
            #   ★修(2026-09-04): 原用稀疏 success (placed) 作信号, 早期全 0 => ALP 无信号
            #   => 权重退地板=均匀=空转。改用密集进度分 (g1+g2+g3+placed)/4: 每回合按走到
            #   哪关给 0~1, 从第0步就有区分度, ALP 才排得出近→远。
            if self.N_TAPE > 1:
                _b0 = self.born_t0[env_ids]
                _tidb = self.tape_id[env_ids][_b0]
                if _tidb.numel():
                    _score = (self.g1[env_ids].float() + self.g2[env_ids].float()
                              + self.g3[env_ids].float()
                              + self.placed[env_ids].float()) / 4.0
                    self._acc_tep.scatter_add_(0, _tidb, torch.ones_like(_tidb, dtype=torch.float))
                    self._acc_tsucc.scatter_add_(0, _tidb, _score[_b0])
            self._acc["g3l_t0"] += int((_gl & self.born_t0[env_ids]).sum())
            # 药②: t0 出生口径 (预置出生不进分母, 消课程稀释偏差)
            t0m = self.born_t0[env_ids]
            self._acc["ep_t0"] += int(t0m.sum())
            for gk, gt in (("g1_t0", self.g1), ("g2_t0", self.g2),
                           ("g3_t0", self.g3), ("g4_t0", self.g4)):
                self._acc[gk] += int((gt[env_ids] & t0m).sum())
        for t_ in (self.g1, self.g2, self.g3, self.g4, self.placed, self.done,
                   self.pre1, self.pre2, self.pre3, self.lb_set, self.g3_loose,
                   self.cert_pending):
            t_[env_ids] = False
        for oi in (0, 1):
            self.lb[oi][env_ids] = 0.0
        self.cert_z0[env_ids] = 0.0
        self.cert_relmax[env_ids] = 0.0
        for s in ("right", "left"):
            self.cert_rel0[s][env_ids] = 0.0
        self.place_pot[env_ids] = 0.0
        self.place_pot2[env_ids] = 0.0
        for t_ in (self.k, self.g1_run, self.m2_run, self.m2l_run, self.m3_run, self.m4_run,
                   self.cert_phase, self.cert_t, self.cert_try, self.cert_wait, self.g3_age):
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
        self.g3_loose[env_ids] = g3       # 预置出生点: 与 g3 同步, 免得口径不齐
        self.placed[env_ids] = placed
        self.pre1[env_ids] = g1
        self.pre2[env_ids] = g2
        self.pre3[env_ids] = g3
        self.born_t0[env_ids] = ~(g1 | g2 | g3 | placed)
        for oi in (0, 1):
            # ★L5-37 多母带: 撤退快照 = 该 env 母带的静置位
            self.m3_snap[oi][env_ids] = (self.rest_stack[self.tape_id[env_ids], oi]
                                         if self.N_TAPE > 1
                                         else self.rest[oi].unsqueeze(0).expand(len(env_ids), 7))

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
               "sr/clock_done": _r(self._acc["cdone"], _ep),
               "sr_t0/clock_done": _r(self._acc["cdone_t0"], _ep0),
               "sr/clock_ge90": _r(self._acc["cge90"], _ep),
               # ★主判据
               "sr/success": _r(self._acc["succ"], _ep),
               "sr_t0/success": _r(self._acc["succ_t0"], _ep0),
               # ★严格几何(诊断口径, 不进 criteria.digest)
               "sr/placed": _r(self._acc["plc"], _ep),
               "sr_t0/placed": _r(self._acc["plc_t0"], _ep0),
               "sr/g3_loose": _r(self._acc["g3l"], _ep),
               "sr_t0/g3_loose": _r(self._acc["g3l_t0"], _ep0),
               "sr/cert_pass": _r(self._acc["cpass"], _at),
               "prog/cert_att": _r(self._acc["catt"], _ep),
               "n/ep_done": float(_ep),          # ★分母本身: 判读前先看它
               "n/ep_done_t0": float(_ep0),
               "n/cert_attempts": float(_at)}
        # F: 认证失败分项占比 (分母=认证尝试数; 空分母同样发 NaN, 不发 0.0)
        for _k in ("rise_bot", "rise_cup", "slip_r", "slip_l", "pads", "tilt", "drift",
                   "relp_r", "relp_l", "tilt_r", "tilt_l"):
            out["cert_fail/" + _k] = _r(self._acc["cf_" + _k], _at)
        # L5-23 死因分项 (分母=本窗判据侧死亡数; 空分母同样发 NaN)
        _tn = self._acc["term_any"]
        for _k in ("D2pre", "D1_drop", "D3_dev", "D8_disturb", "D2_tilt"):
            out["term/" + _k] = _r(self._acc["term_" + _k], _tn)
        out["n/term_judge"] = float(_tn)
        self._acc = {"ep": 0, "g1": 0, "g2": 0, "g3": 0, "g4": 0,
                     "clock": 0.0, "catt": 0, "cpass": 0, "cdone": 0, "cdone_t0": 0, "cge90": 0, "succ": 0, "succ_t0": 0,
                     "g3l": 0, "g3l_t0": 0, "plc": 0, "plc_t0": 0,
                     "ep_t0": 0, "g1_t0": 0, "g2_t0": 0, "g3_t0": 0, "g4_t0": 0,
                     "cf_rise_bot": 0, "cf_rise_cup": 0, "cf_slip_r": 0,
                     "cf_slip_l": 0, "cf_pads": 0, "cf_tilt": 0, "cf_drift": 0, "term_any": 0,
                     "cf_relp_r": 0, "cf_relp_l": 0, "cf_tilt_r": 0, "cf_tilt_l": 0,
                     "term_D2pre": 0, "term_D1_drop": 0, "term_D3_dev": 0, "term_D8_disturb": 0, "term_D2_tilt": 0}
        return out

    def pop_tape(self):
        """★B: 逐带 t0 success 率 + 本窗回合数, 用完清零。train_pour 拿去算 ALP。"""
        ep = self._acc_tep.clamp(min=1.0)
        rate = (self._acc_tsucc / ep).detach().cpu().tolist()
        n = self._acc_tep.detach().cpu().tolist()
        self._acc_tsucc.zero_()
        self._acc_tep.zero_()
        return rate, n

    def tighten_place_tol(self):
        """★④ 单向棘轮收紧 placed 容差一档 (train_pour 按 EMA(placed) 调用)。返回是否变化。"""
        if not self.place_curr or self.pc_level >= self._pc_lmax:
            return False
        self.pc_level += 1
        a = self.pc_level / self._pc_lmax
        self.m3_rot_cur = self._pc_rot0 + (M3_ROT - self._pc_rot0) * a
        self.m3_pos_cur = self._pc_pos0 + (M3_POS - self._pc_pos0) * a
        return True

    def _rel_feat(self, o0, o1):
        """关系特征 (Ne,): ρ=瓶口↔杯口水平距, Δz=瓶口−杯口高差, θ=瓶倾角(rad)。
        与 G3 判据同公式(见 mouth 段), 供 rel_ref 的 leash/时钟门复用。"""
        R1 = _q2R(o1[:, 3:7])
        R0 = _q2R(o0[:, 3:7])
        mb = o1[:, :3] + torch.einsum("nij,j->ni", R1, self.mb)
        mc = o0[:, :3] + torch.einsum("nij,j->ni", R0, self.mc)
        d = mb - mc
        v = torch.einsum("nij,j->ni", R1, self.up)
        tilt = torch.acos((v[:, 2] / v.norm(dim=1).clamp(min=1e-9)).clamp(-1, 1))
        return d[:, :2].norm(dim=1), d[:, 2], tilt

    def step(self, obj0, obj1, armq_r, armq_l, pads3, wrist_r, wrist_l,
             run_mask=None):
        """全部 (N,·) 张量; pads3 (N,)bool 双手各>=3/5垫; wrist_r/l (N,3)。
        run_mask: False 的 env 本步不受本机管辖(机器段行)。"""
        k = self.k.clamp(max=self.N_ROW - 1)

        def _ro(oi, kk):
            """每 env 参考物体行 (Ne,7): 单母带广播 ref_obj[oi][kk]; 多母带按 tape_id gather。"""
            if self.N_TAPE == 1:
                return self.ref_obj[oi][kk]
            return self.ref_obj_stack[self.tape_id, oi, kk]

        def _rst(oi):
            """每 env 静置位 (Ne,7): 单母带广播; 多母带按 tape_id。"""
            if self.N_TAPE == 1:
                return self.rest[oi].unsqueeze(0).expand(self.Ne, 7)
            return self.rest_stack[self.tape_id, oi]

        tier = self.tmix[k]
        w_obj = self.WO[tier]
        active = ~self.done if run_mask is None else (~self.done) & run_mask
        # 药A: G2 后首个受管步捕获持握基线
        cap = self.g1 & (~self.lb_set) & active     # 药⑤: G1 起换基
        if cap.any():
            for oi, act in ((0, obj0), (1, obj1)):
                self.lb[oi][cap] = (act[:, :3] - _ro(oi, k)[:, :3])[cap]
            self.lb_set = self.lb_set | cap
        # ---- 皮筋 ----
        earning = (self.k < self.N_ROW - 1)
        self._rel_ok = None      # 关系时钟门 (rel_ref 时在此算, 门段复用)
        if self.rel_ref:
            # ★R 关系皮筋(方案C 单边): 只罚"向倒水方向不够", 允许/鼓励超过视频值。
            #   ρ 只罚"不够近" · θ 只罚"落后倾倒趋势" · Δz 双边小带。
            hz_s, dz_s, th_s = self._rel_feat(obj0, obj1)
            hz_r, dz_r, th_r = self._rel_feat(_ro(0, k), _ro(1, k))
            pen_hz = ((hz_s - hz_r) - self.REL_TOL_HZ).clamp(min=0) / self.REL_TOL_HZ
            pen_dz = ((dz_s - dz_r).abs() - self.REL_TOL_DZ).clamp(min=0) / self.REL_TOL_DZ
            pen_th = ((th_r - th_s) - self.REL_TOL_TH).clamp(min=0) / self.REL_TOL_TH
            leash = -(pen_hz + pen_dz + pen_th)
            self._rel_ok = ((hz_s <= hz_r + self.REL_GATE_HZ)
                            & (th_s >= th_r - self.REL_GATE_TH))
        else:
            leash = torch.zeros(self.Ne, device=self.dev)
            for oi, act in ((0, obj0), (1, obj1)):
                ref = _ro(oi, k)
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
        leash = leash.clamp(min=-3.0) * active.float() * earning.float()
        if self.goal_only:
            leash = torch.zeros_like(leash)
        # ---- 时钟门 (L5-6: 双变体统一; 红档=宽松物门) ----
        if self.rel_ref:
            ok_obj = self._rel_ok          # ★R 关系时钟门(单边, 见皮筋段)
        else:
            ok_obj = torch.ones(self.Ne, dtype=torch.bool, device=self.dev)
            gp = torch.where(tier > 0,
                             torch.full_like(w_obj, GATE_POS),
                             torch.full_like(w_obj, RED_GATE_POS))
            for oi, act in ((0, obj0), (1, obj1)):
                ref = _ro(oi, k)
                ok_obj &= ((act[:, :3] - ref[:, :3]).norm(dim=1) <= gp)
                rot_ok = (_qang(act[:, 3:7] / act[:, 3:7].norm(dim=1, keepdim=True)
                                .clamp(min=1e-9), ref[:, 3:7]) <= GATE_ROT)
                ok_obj &= torch.where(tier > 0, rot_ok,
                                      torch.ones_like(rot_ok))   # 红档 rot 禁入
        ok = ok_obj & active
        # 时钟: G2 前不走
        _cap = self.N_ROW - 1 if self.kcap is None else min(self.kcap, self.N_ROW - 1)
        can = ok & (self.k < _cap) & self.g2
        if self.goal_only:
            can = torch.zeros_like(can)      # ★时钟冻死: row 恒为 IA0(抓握姿势)
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
            self.cert_relmax[start] = 0.0
            if self.hold_pose or self.min_recipe:
                for oi, act in ((0, obj0), (1, obj1)):
                    up_ = self.up if oi == 1 else self.upc
                    if self.hold_v2 and not self.min_recipe:
                        # v2: 基准 = 参考静置 (交互首行) 的倾角与 xy —— 量"绝对"倾斜/漂移, 合拢期推歪的也算
                        self.cert_tilt0[start, oi] = self.ref_tilt[oi][0]
                        self.cert_xy0[start, oi] = _rst(oi)[start, :2]
                    else:
                        self.cert_tilt0[start, oi] = _tilt(act[:, 3:7], up_)[start]
                        self.cert_xy0[start, oi] = act[start, :2]
                self.cert_tilt_dev[start] = 0.0
                self.cert_drift[start] = 0.0
        in_cert = (self.cert_phase > 0) & (~start) & active
        self.cert_t = torch.where(in_cert, self.cert_t + 1, self.cert_t)
        if self.hold_pose or self.min_recipe:
            # 斜坡+保持期 (相 1/2) 逐步追踪最大倾角偏离与 xy 漂移 —— 判决看最大值, 瞬时歪一下再回正也算
            _trk = in_cert & (self.cert_phase <= 2)
            if self.min_recipe:     # ★POUR_MIN: 相对位移最大值 (腕-物向量相对认证起点)
                _rr = ((wrist_r - obj1[:, :3]) - self.cert_rel0["right"]).norm(dim=1)
                _rl = ((wrist_l - obj0[:, :3]) - self.cert_rel0["left"]).norm(dim=1)
                self.cert_relmax[:, 1] = torch.where(_trk, torch.maximum(self.cert_relmax[:, 1], _rr), self.cert_relmax[:, 1])
                self.cert_relmax[:, 0] = torch.where(_trk, torch.maximum(self.cert_relmax[:, 0], _rl), self.cert_relmax[:, 0])
            for oi, act in ((0, obj0), (1, obj1)):
                up_ = self.up if oi == 1 else self.upc
                _dv = (_tilt(act[:, 3:7], up_) - self.cert_tilt0[:, oi]).abs()
                _dr = (act[:, :2] - self.cert_xy0[:, oi]).norm(dim=1)
                self.cert_tilt_dev[:, oi] = torch.where(_trk, torch.maximum(self.cert_tilt_dev[:, oi], _dv),
                                                        self.cert_tilt_dev[:, oi])
                self.cert_drift[:, oi] = torch.where(_trk, torch.maximum(self.cert_drift[:, oi], _dr),
                                                     self.cert_drift[:, oi])
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
            if self.hold_pose:      # ★L5-36.3 A: 两物体在斜坡+保持期内 倾角偏离≤10° 且 xy 漂移≤1cm
                _cj = _cj + (("tilt", (self.cert_tilt_dev <= HOLD_TILT_TOL).all(dim=1)),
                             ("drift", (self.cert_drift <= HOLD_DRIFT_TOL).all(dim=1)))
            if self.min_recipe:     # ★POUR_MIN: 判决只看 相对位移 ≤1cm ∧ 倾斜偏离 ≤5° (两物体), 抬升由 +15mm 认证行 + 1cm 容差隐含
                _cj = (("relp_r", self.cert_relmax[:, 1] <= MIN_CERT_POS),
                       ("relp_l", self.cert_relmax[:, 0] <= MIN_CERT_POS),
                       ("tilt_r", self.cert_tilt_dev[:, 1] <= MIN_CERT_ROT),
                       ("tilt_l", self.cert_tilt_dev[:, 0] <= MIN_CERT_ROT))
            ok5 = _cj[0][1]
            for _k, _okm in _cj[1:]:
                ok5 = ok5 & _okm
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
        # ★L5-32 (用户裁定 2026-08-31): G3 = **倒水几何**, 不再是 3D 欧氏距。
        #   旧口径 `‖瓶口−杯口‖ <= mgate(12cm)` **完全不分上下** —— 瓶口在杯口
        #   正上方 10cm / 正下方 10cm / 正左边 10cm, 判据眼里一模一样。
        #   实测 (probe_pour_geometry, 11M 步 ckpt):
        #     base_oh  dz>0 仅 56.4%, 倾角 P10=90.2° 卡在门槛上  ⟸ 擦边过关
        #     flat_oh  dz>0   79.8%   母带 100% · 人类原始重建 84.8%
        #   而 **G3 是付奖的 (MS_REWARD[3]=10.0)** ⟹ 旧判据不只是记账松, 它在
        #   真金白银地奖励"把瓶子横杵在杯子旁边"。这就是必须换判据的理由。
        _d = mb_w - mc_w
        _hz = _d[:, :2].norm(dim=1)
        _dz = _d[:, 2]
        m2_now = ((tilt >= M2_TILT) & (_hz <= M2_STRICT_HORIZ)
                  & (_dz >= M2_STRICT_DZ_LO) & (_dz <= M2_STRICT_DZ_HI))
        gate3 = self.g2 & (~self.g3) & active
        self.m2_run = torch.where(gate3 & m2_now, self.m2_run + 1,
                                  torch.zeros_like(self.m2_run))
        new3 = gate3 & (self.m2_run >= M2_HOLD)
        # ★旧松判据**降级为诊断影子** sr/g3_loose (不付奖, 不进主判据)。
        #   保留它是为了让新旧口径在**同一条 run** 上可比: g3_loose − gate3
        #   就是"旧口径里有多少是横杵刷出来的"。没有它, 所有历史数字都失去参照。
        m2l_now = (tilt >= M2_TILT) & (_d.norm(dim=1) <= self.mgate)
        _gl = self.g2 & (~self.g3_loose) & active
        self.m2l_run = torch.where(_gl & m2l_now, self.m2l_run + 1,
                                   torch.zeros_like(self.m2l_run))
        self.g3_loose |= _gl & (self.m2l_run >= M2_HOLD)
        self.g3 |= new3
        ms_r += new3.float() * MS_REWARD[3]
        # ---- placed (不付奖) ----
        ok3 = torch.ones_like(ok_obj)
        for oi, act in ((0, obj0), (1, obj1)):
            ok3 &= ((act[:, :3] - _rst(oi)[:, :3]).norm(dim=1) <= self.m3_pos_cur)  # ★④课程容差
            up3 = self.up if oi == 1 else self.upc
            ok3 &= (_tilt(act[:, 3:7], up3) <= self.m3_rot_cur)   # ★④课程容差; 倾角口径, yaw豁免
        gatep = self.g3 & (~self.placed) & active
        self.m3_run = torch.where(gatep & ok3, self.m3_run + 1,
                                  torch.zeros_like(self.m3_run))
        newp = gatep & (self.m3_run >= M3_HOLD)
        self.placed |= newp
        # ★L5-36.9: placed=成功终点 —— 保持 15 步即付终局奖 + 终止回合 (不做撤离)。
        if PLACE_TERMINAL:
            ms_r += newp.float() * PLACE_REWARD
            self.done |= newp
        # ---- ★L5-33 placed 绝对整形 (earn-only 棘轮) ----
        #   φ 只读瓶位姿与 rest —— 与 placed 判据同源的**绝对量**, 不读参考轨迹。
        #   只按瓶算(杯在正常回合里不动; placed 判据本身仍判双物)。
        #   全程支付上限 = PLACE_SHAPE_K × (1−φ@G3达成) ≤ 6.0。
        r_place = torch.zeros(self.Ne, device=self.dev)
        if self.place_shape and self.place_both:
            # ★L5-36.7: 瓶/杯各一份 φ 与棘轮 (播种逻辑同款, 逐物体独立), 各付 K/2 —— 杯的放回从此有梯度。
            _gp = self.g3 & (~self.placed) & active
            for _oi, (_o, _up) in enumerate(((obj0, self.upc), (obj1, self.up))):
                _d = (_o[:, :3] - _rst(_oi)[:, :3]).norm(dim=1)
                _tl = _tilt(_o[:, 3:7], _up)
                _phi = (0.5 * (1.0 - (_d / PLACE_SHAPE_D0).clamp(0, 1))
                        + 0.5 * (1.0 - (_tl / PLACE_SHAPE_T0).clamp(0, 1)))
                _pot = self.place_pot2[:, _oi]
                _seed = _gp & (_pot <= 0)
                _pot = torch.where(_seed, _phi, _pot)
                r_place = r_place + (PLACE_SHAPE_K / 2) * (_phi - _pot).clamp(min=0) \
                    * (_gp & ~_seed).float()
                self.place_pot2[:, _oi] = torch.where(_gp, torch.maximum(_pot, _phi), _pot)
        elif self.place_shape:
            _db = (obj1[:, :3] - _rst(1)[:, :3]).norm(dim=1)
            _tl = _tilt(obj1[:, 3:7], self.up)
            _phi = (0.5 * (1.0 - (_db / PLACE_SHAPE_D0).clamp(0, 1))
                    + 0.5 * (1.0 - (_tl / PLACE_SHAPE_T0).clamp(0, 1)))
            _gp = self.g3 & (~self.placed) & active
            # ★首个受管步只**播种**不付钱: 否则 G3 达成那一刻会白送 K×φ(约1.8) ——
            #   那份 φ 是达成 G3 前就有的, 不是"往回放"的进步。g3 预置出生同理。
            _seed = _gp & (self.place_pot <= 0)
            self.place_pot = torch.where(_seed, _phi, self.place_pot)
            r_place = PLACE_SHAPE_K * (_phi - self.place_pot).clamp(min=0) \
                * (_gp & ~_seed).float()
            self.place_pot = torch.where(_gp,
                                         torch.maximum(self.place_pot, _phi),
                                         self.place_pot)
        # ★③ 抓握保持罚 (2026-09-03): 放置期(g3达成、未placed)脱握每步扣 grip_hold_k,
        #   折进 ms 通道(weight=1.0)。self.placed 已含本步 newp, ~placed 排除刚认证的(它拿终局奖)。
        if self.grip_hold:
            _php = self.g3 & (~self.placed) & active
            ms_r = ms_r - self.grip_hold_k * (_php & (~pads3)).float()
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
        # ★L5-36.9: PLACE_TERMINAL 时撤离 G4 退役 —— 仍记 g4 (诊断影子), 但不终止不付奖
        #   (回合已在 placed 终止)。旗关时保持原行为 (G4 终止 + 付 MS_REWARD[4])。
        if not PLACE_TERMINAL:
            self.done |= new4
            ms_r += new4.float() * MS_REWARD[4]
        # ★① g3_age 递增 (g3 后连续步数; 非 g3 清零) —— D3_dev 宽限期用
        self.g3_age = torch.where(self.g3, self.g3_age + 1,
                                  torch.zeros_like(self.g3_age))
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
            ref = _ro(oi, k)
            if not self.goal_only:      # ★D3 读参考轨迹, goal 臂必须关掉
                # ★① 宽限期: g3 达成后 place_grace 步内豁免 D3_dev (给收回窗时间)
                _grace = self.g3 & (self.g3_age <= self.place_grace) \
                    if self.place_grace > 0 else torch.zeros_like(self.g3)
                _fc["D3_dev"] |= (~self.placed) & (~_grace) & (
                    (act[:, :3] - ref[:, :3]).norm(dim=1) > D3_DEV)
            sn = self.m3_snap[oi]
            tl = _tilt(act[:, 3:7], up_)
            _fc["D8_disturb"] |= self.placed & (
                ((act[:, :3] - sn[:, :3]).norm(dim=1) > M4_DIST_POS)
                | (tl > M4_DIST_ROT))
            _fc["D2_tilt"] |= self.placed & (tl > D2_TILT)
        # ---- ★L5-36.3 B: 抓稳期 (G1 后 G2 前) 姿态守恒渐进罚: 绝对倾角 + 离静置 xy 漂移, 每步 −K×(0..1) ----
        hold_pen = torch.zeros(self.Ne, device=self.dev)
        if self.hold_pose and not self.hold_v2:      # v2 的 B 在 env 侧算 (窗口含缝 1 合拢期)
            _grasp = (self.g1 & (~self.g2) & active).float()
            for oi, act in ((0, obj0), (1, obj1)):
                up_ = self.up if oi == 1 else self.upc
                _pt = ((_tilt(act[:, 3:7], up_) - HOLD_PEN_TILT0)
                       / (HOLD_PEN_TILT1 - HOLD_PEN_TILT0)).clamp(0, 1)
                _pd = (((act[:, :2] - _rst(oi)[:, :2]).norm(dim=1) - HOLD_PEN_DRIFT0)
                       / (HOLD_PEN_DRIFT1 - HOLD_PEN_DRIFT0)).clamp(0, 1)
                hold_pen = hold_pen - (HOLD_K_TILT * _pt + HOLD_K_DRIFT * _pd) * _grasp
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
                "place": r_place, "hold_pen": hold_pen,
                "w_obj": w_obj, "w_hand": self.WH[tier], "clock": self.k.clone(), "done": self.done.clone(),
                "tier": tier, "fail": fail,
                "cert_phase": self.cert_phase.clone(),
                "cert_t": self.cert_t.clone()}
