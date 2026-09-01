"""Unscrew Success Tracker (标量规格; 框架身=Pour17 v5, [TASK] 块已换核)。

任务: 瓶立桌上 → 左手抓稳(G1)+提升认证(G2) → 时钟播交互行(左手带瓶按谱转平,
右手进场) → 拧开释放(G3) → 盖放到母带终点、瓶放回(placed) → 撤退归位(G4=Success)。

与 Pour17 的差异 (逐条有据):
  G1  只判**左手** >=3/5 垫 —— 演示时序: 右手在瓶被拿起转平**之后**才进场碰盖
      (clip32: 瓶 onset f22, 盖 onset f32), 双手同判会让 G1 永不点火。
      右手的引导交给 env 侧贴实奖金 (框架 C 线, 分侧) + 拧转势。
  G3  = 拧开释放 (env 的 screw_assembly detach, 拧满 SCREW_TURNS 即脱开) ——
      螺旋角是仿真物理量, 由 env 喂进来 (step 的 screw_released 参数)。
      重建的盖转角不可作真值 (螺轴对称, 视觉不可观 —— 数据集 README)。
  placed = 双物体到**母带末交互行位姿** (不是静置位: 盖起点在瓶上、终点在桌上,
      rest≠end)。瓶 3cm/15°, 盖 5cm/30° (盖小且放置点由人示范定义, 松一档)。
      **且护送未失败** (T2-3, 2026-08-30 用户裁定): 成功 = 右手拿着盖放到
      桌上; 盖无右垫接触地快速穿过放下带顶 (过带单事件) 即 escort_fail
      粘滞, placed 永不立 —— 盖自己脱落/被甩到目标邻域不算成功。
  置信度主档 tmix = min(瓶, 盖) 的 min(pos,rot) 档 —— 双物体任务里短板决定
      参考可信度 (README 排名同口径)。
  人手置信度 (本任务新增): hand_conf_fin_r/l 列 → HCF 权重 (绿1/黄0.5/红0),
      门控 env 侧形状指引奖 (低置信的人手行不该教手型)。

改判据必须双版同步 (progress_batch.py) + 自检家族重跑。
消费母带: reference_v1/v2.npz (obj_0=瓶 obj_1=盖; source=1 为交互行)。
"""
from __future__ import annotations

import json

import numpy as np

TIER_HI, TIER_LO = 70.0, 40.0
W_OBJ = {2: 1.0, 1: 0.5, 0: 0.2}    # L5-6: 高置信=纯物轨(人手0)
W_HAND = {2: 0.0, 1: 0.5, 0: 0.8}   # 人手形状指引权重: 绿0/黄五五开/红主导
W_HCONF = {2: 1.0, 1: 0.5, 0: 0.0}  # [TASK] 人手自身置信度门 (乘在形状奖上)
LEASH_POS = {2: 0.03, 1: 0.05, 0: 0.08}
LEASH_ROT = {2: np.radians(15), 1: np.radians(30), 0: None}   # 红档 rot 禁判
GATE_POS, GATE_ROT = 0.05, np.radians(45)
RED_GATE_POS = 0.08
MS_REWARD = {1: 5.0, 2: 8.0, 3: 10.0, 4: 15.0}   # G1/G2/G3释放/G4(=Success终局)
# ---- [TASK] placed 判据 (目标=母带末交互行, 双物体 AND) ----
PLACED_POS = {0: 0.03, 1: 0.05}          # 瓶 3cm / 盖 5cm
PLACED_ROT = {0: np.radians(15), 1: np.radians(30)}
PLACED_HOLD = 15
# ---- [TASK] placed 护送判据 (2026-08-30 用户裁定, 台账 T2-3): 成功必须是
# **右手拿着盖放到桌上**, 盖自己脱落/被甩到目标邻域不算。机制 = 过带单事件:
# 释放后任何落桌的盖必恰好穿过放下带顶 (桌面+ESCORT_BAND) 一次; 穿带那步若
# [单步下降>ESCORT_FALL 且 无右垫接触] => escort_fail 粘滞, placed 永不立。
# 受控放下: 要么带着接触穿带, 要么慢放 (<ESCORT_FALL/步 = 16cm/s @20Hz)。
# 自由落体哪怕从带顶上方 2cm 起掉, 穿带步降幅也 >3cm => 必抓; 母带释放高度
# 仅 0.92m (人转平低位拧开), 高度无关的单事件判据不受此影响。----
ESCORT_BAND = 0.03
ESCORT_FALL = 0.008
# U41 (老方法线同一条用户裁定的第二、三个必要条件, 2026-08-30 移植):
# 光"没被甩下去"还不够 —— 还要**真的拿过**且**放得住**:
#   ② 持盖运输: 释放后右垫 >=CARRY_PADS 触盖累计 >=PLACE_CARRY 步 (0.5s@20Hz);
#   ③ 受控下放: 释放后单步降幅峰值 <PLACE_FALL (=0.8m/s@20Hz); 自由落体从
#      瓶口高度落桌 ~1.7-2.2m/s 一票否决, 3cm 内的轻放不受影响。
CARRY_PADS = 2
PLACE_CARRY = 10
PLACE_FALL = 0.04
M4_ARM, M4_HOLD = np.radians(10), 15     # G4 双臂贴站姿逐关节<10°, hold15
M4_DIST_POS, M4_DIST_ROT = 0.05, np.radians(30)   # 撤退期物体相对 placed 快照
# ---- G1/G2 认证 (框架 L5-1 拍板; G1 只判左手, 见模块 docstring) ----
G1_HOLD = 10
CERT_RAMP, CERT_HOLD, CERT_RET = 8, 5, 8
CERT_RISE = 0.005             # 双物 z 升 >=5mm (认证行=腕参考+15mm)
CERT_SLIP = 0.008             # 左腕-瓶相对位移 <8mm
CERT_WAIT, CERT_TRIES = 20, 3
WAGE = 0.05
WAGE_CAP = 3.0
# ---- 死线 ----
D1_DROP = 0.05
D2_PRE_TILT = np.radians(60)  # G2 前瓶倒伏
D2_TILT = np.radians(30)      # placed 后撤退段
D3_DEV = 0.35
D4_SLIP = 0.05               # 交互期腕-物相对滑移
D5_BELOW_TABLE = 0.005       # 手部低于桌面 5mm
TABLE_Z = 0.87
PAD_FTH = 0.5
PADS_MIN = 3
SCREW_CONTACT_FTH = 0.1
SCREW_TRIAD = (0, 1, 2)      # 右手拇/食/中
UP_LOCAL = {0: np.array([0.0, 0.0, 1.0]), 1: np.array([0.0, 0.0, 1.0])}  # [TASK]

# Checkpoint compatibility contract: only success/failure semantics belong in
# this digest. Reward weights and accounting fields deliberately stay out.
# schema=3: 2026-08-30 加 placed 护送判据 (ESCORT_*), placed 语义收紧。
CRITERIA_SCHEMA = 4


def criteria_items():
    """Return normalized constants that determine task success or failure."""
    def _n(value):
        if isinstance(value, dict):
            return {str(key): _n(value[key]) for key in sorted(value, key=str)}
        if isinstance(value, np.ndarray):
            return [_n(item) for item in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [_n(item) for item in value]
        if value is None:
            return None
        try:
            return round(float(value), 9)
        except Exception:
            return str(value)

    return {key: _n(value) for key, value in {
        "GATE_POS": GATE_POS,
        "TIER_HI": TIER_HI,
        "TIER_LO": TIER_LO,
        "GATE_ROT": GATE_ROT,
        "RED_GATE_POS": RED_GATE_POS,
        "LEASH_POS": LEASH_POS,
        "LEASH_ROT": LEASH_ROT,
        "G1_HOLD": G1_HOLD,
        "CERT_RAMP": CERT_RAMP,
        "CERT_HOLD": CERT_HOLD,
        "CERT_RET": CERT_RET,
        "CERT_RISE": CERT_RISE,
        "CERT_SLIP": CERT_SLIP,
        "CERT_WAIT": CERT_WAIT,
        "CERT_TRIES": CERT_TRIES,
        "PLACED_POS": PLACED_POS,
        "PLACED_ROT": PLACED_ROT,
        "PLACED_HOLD": PLACED_HOLD,
        "ESCORT_BAND": ESCORT_BAND,
        "ESCORT_FALL": ESCORT_FALL,
        "CARRY_PADS": CARRY_PADS,
        "PLACE_CARRY": PLACE_CARRY,
        "PLACE_FALL": PLACE_FALL,
        "M4_ARM": M4_ARM,
        "M4_HOLD": M4_HOLD,
        "M4_DIST_POS": M4_DIST_POS,
        "M4_DIST_ROT": M4_DIST_ROT,
        "D1_DROP": D1_DROP,
        "D2_PRE_TILT": D2_PRE_TILT,
        "D2_TILT": D2_TILT,
        "D3_DEV": D3_DEV,
        "D4_SLIP": D4_SLIP,
        "D5_BELOW_TABLE": D5_BELOW_TABLE,
        "PAD_FTH": PAD_FTH,
        "PADS_MIN": PADS_MIN,
        "SCREW_CONTACT_FTH": SCREW_CONTACT_FTH,
        "SCREW_TRIAD": SCREW_TRIAD,
        "UP_LOCAL": UP_LOCAL,
        "TABLE_Z": TABLE_Z,
    }.items()}


def criteria_digest():
    import hashlib
    blob = json.dumps(criteria_items(), sort_keys=True, ensure_ascii=False)
    return CRITERIA_SCHEMA, hashlib.md5(blob.encode()).hexdigest()[:16]


def _tier(v):
    if np.isnan(v):
        return None                    # 机器行: conf 结构性缺席
    return 2 if v >= TIER_HI else (1 if v >= TIER_LO else 0)


def _qang(a, b):
    return 2 * np.arccos(min(1.0, abs(float(np.dot(a, b)))))


def _axis_tilt(q, up_local):
    w, x, y, z = q / np.linalg.norm(q)
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    v = R @ up_local
    return float(np.arccos(np.clip(v[2] / max(np.linalg.norm(v), 1e-9), -1, 1)))


class UnscrewProgress:
    """时钟=交互行索引, 只进不退。每步喂实测+螺旋状态, 吐奖励账目."""

    def __init__(self, npz_path, kcap=None, no_hand_ref=False):
        z = np.load(npz_path, allow_pickle=True)
        rows = np.where(np.asarray(z["source"]) == 1)[0]
        self.stance = {s: np.asarray(z[f"{s}_q"], np.float64)[-1]
                       for s in ("right", "left")}    # 母带末行 = InitialPose
        self.r0, self.r1 = int(rows[0]), int(rows[-1])
        self.N = self.r1 - self.r0 + 1
        self.obj = {oi: np.concatenate(
            [np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows],
             np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows]], axis=1)
            for oi in (0, 1)}
        self.tp = {oi: [_tier(v) for v in np.asarray(z[f"conf_pos_{oi}"])[rows]]
                   for oi in (0, 1)}
        self.tr = {oi: [_tier(v) for v in np.asarray(z[f"conf_rot_{oi}"])[rows]]
                   for oi in (0, 1)}
        # ★ G-B 起始黄窗 (与 batch 版同一 helper; None 档按绿=2 进变换后原样存回)
        from rl_rebuild.correction import tier_floor as _TF
        _tp = {oi: np.array([(t if t is not None else 2) for t in self.tp[oi]], np.int64)
               for oi in (0, 1)}
        _tr = {oi: np.array([(t if t is not None else 2) for t in self.tr[oi]], np.int64)
               for oi in (0, 1)}
        _tp, _tr, _tm_unused, self.tier_info = _TF.transform(
            _tp, _tr, np.asarray(z["obj_pos_0"], np.float64)[rows], CERT_RISE, tag="PS")
        self.tp = {oi: _tp[oi].tolist() for oi in (0, 1)}
        self.tr = {oi: _tr[oi].tolist() for oi in (0, 1)}
        # [TASK] 主档 = 双物体短板: min over {瓶,盖}×{pos,rot} (机器行按绿处理)
        def _m(a):
            return a if a is not None else 2
        self.tmix = [min(_m(self.tp[0][k]), _m(self.tr[0][k]),
                         _m(self.tp[1][k]), _m(self.tr[1][k]))
                     for k in range(self.N)]
        # [TASK] 人手置信度权重 (形状指引门控; 列缺席=全绿, v1 前兼容)
        def _hcf(col):
            if col not in z:
                return [1.0] * self.N
            return [W_HCONF[_m(_tier(v))]
                    for v in np.asarray(z[col], np.float64)[rows]]
        self.hcf = {"right": _hcf("hand_conf_fin_r"), "left": _hcf("hand_conf_fin_l")}
        self.rest = {oi: self.obj[oi][0].copy() for oi in (0, 1)}
        self.end = {oi: self.obj[oi][-1].copy() for oi in (0, 1)}   # [TASK] placed 目标
        # 释放行 (盖脱离): meta.windows.k_sep (RSI g3 出生点用)
        self.k_sep = self.N - 1
        try:
            meta = json.loads(str(z["meta"]))
            self.k_sep = int(np.clip(meta["windows"]["k_sep"], 1, self.N - 1))
        except Exception:
            pass
        self.kcap = kcap
        self.no_hand_ref = bool(no_hand_ref)
        self.reset()

    # ---- 渐进 RSI: 初始仅 t0, Gate_k EMA 达标解锁 ----
    def entry_table(self, unlocked=()):
        """→ [(交互行, preset集, label)]; preset: 1/2/3, 'placed'。"""
        et = [(0, set(), "t0")]
        u = set(unlocked)
        if 1 in u:
            et.append((0, {1}, "g1"))
        if 2 in u:
            et.append((0, {1, 2}, "g2"))
        if 3 in u:
            # g3 = 释放后首行 (盖已脱, 携带/放置段)
            et.append((min(self.k_sep + 1, self.N - 1), {1, 2, 3}, "g3"))
            et.append((self.N - 1, {1, 2, 3, "placed"}, "ret"))
        return et

    def enter(self, row, preset_ms):
        self.reset()
        self.k = int(row)
        for m in preset_ms:
            if m == "placed":
                self.placed = True
                # 撤退点语义 = "已妥善放好": 快照取母带末行目标
                self.m3_snap = {oi: self.end[oi].copy() for oi in (0, 1)}
            else:
                self.g[m] = True
        if 3 in preset_ms:
            self.released = True

    def reset(self):
        self.k = 0
        self.leash_base = {0: np.zeros(3), 1: np.zeros(3)}
        self._lb_set = False
        self.g = {1: False, 2: False, 3: False, 4: False}
        self.placed = False
        self.released = False
        self.g1_run = 0
        self.wage_paid = 0.0
        self.cert_phase = 0
        self.cert_t = 0
        self.cert_try = 0
        self.cert_wait = 0
        self.cert_pending = False
        self.cert_z0 = {0: 0.0, 1: 0.0}
        self.cert_rel0 = np.zeros(3)
        self.m3_run = 0
        self.m4_run = 0
        self.m3_snap = None
        self.done = False
        self.escort_fail = False
        self._cap_z_prev = None
        self.carry_steps = 0.0      # U41②: 释放后 >=CARRY_PADS 垫触盖的累计步
        self.fall_peak = 0.0        # U41③: 释放后盖单步降幅峰值 (m/步)

    # -- 逐步接口: obj0=瓶(7,) obj1=盖(7,), armq_r/l(7,), pads3=左手>=3/5垫(bool),
    #    wrist_r/l(3,), screw_released=env 螺旋 detach 锁存(bool),
    #    pads_r_cap=右手垫-盖接触>=1垫(bool); None=无信号, 护送判据停用
    #    (离线探针兼容; env 必须显式传) --
    def step(self, obj0, obj1, armq_r, armq_l, pads3, wrist_r, wrist_l,
             screw_released=False, pads_r_cap=None):
        out = {"clock": self.k, "adv": 0.0, "leash": 0.0, "w_obj": None,
               "ms": 0.0, "wage": 0.0, "done": False, "gate_by": None,
               "fail": None, "cert_phase": self.cert_phase, "cert_t": self.cert_t}
        if self.done:
            out["done"] = True
            return out
        k = min(self.k, self.N - 1)
        tier = self.tmix[k]
        out["w_obj"] = 1.0 if self.no_hand_ref else W_OBJ[tier]
        out["w_hand"] = 0.0 if self.no_hand_ref else W_HAND[tier]
        out["w_hconf_r"] = self.hcf["right"][k]
        out["w_hconf_l"] = self.hcf["left"][k]
        earning = self.k < self.N - 1
        # 药A: G1(抓形成) 后首个受管步捕获持握基线 (框架 L5-8 药⑤)
        if self.g[1] and not self._lb_set:
            for _oi, _a in ((0, obj0), (1, obj1)):
                self.leash_base[_oi] = (np.asarray(_a[:3], np.float64)
                                        - self.obj[_oi][k][:3])
            self._lb_set = True
        # ---- 皮筋 (分量化, 逐物体; 仅时钟在走时) ----
        leash = 0.0
        for oi, act in ((0, obj0), (1, obj1)) if earning else ():
            tpos, trot = self.tp[oi][k], self.tr[oi][k]
            ref = self.obj[oi][k]
            _dvec = np.asarray(act[:3]) - ref[:3]
            if self._lb_set:
                _dvec = _dvec - self.leash_base[oi]
            dp = float(np.linalg.norm(_dvec))
            lp = LEASH_POS[tpos if tpos is not None else 2]
            if dp > lp:
                leash -= (dp - lp) / lp
            lr = LEASH_ROT[trot if trot is not None else 2]
            if lr is not None:
                dr = _qang(np.asarray(act[3:7]), ref[3:7])
                if dr > lr:
                    leash -= (dr - lr) / lr
        out["leash"] = max(leash, -3.0)
        # ---- 时钟门 (双变体统一; 红档=宽物门) ----
        if earning:
            ok = True
            gp = GATE_POS if tier > 0 else RED_GATE_POS
            for oi, act in ((0, obj0), (1, obj1)):
                ref = self.obj[oi][k]
                if np.linalg.norm(np.asarray(act[:3]) - ref[:3]) > gp:
                    ok = False
                if tier > 0 and _qang(np.asarray(act[3:7]), ref[3:7]) > GATE_ROT:
                    ok = False
            out["gate_by"] = "obj"
        else:
            ok = False
        # 时钟: G2 前不走 (自主抓稳继承)
        _cap = self.N - 1 if self.kcap is None else min(self.kcap, self.N - 1)
        if earning and ok and self.k < _cap and self.g[2]:
            self.k += 1
            out["adv"] = 1.0
        # ---- G1: [TASK] 左手 >=3/5 垫 稳 G1_HOLD 步 ----
        if not self.g[1]:
            self.g1_run = self.g1_run + 1 if pads3 else 0
            if self.g1_run >= G1_HOLD:
                self.g[1] = True
                out["ms"] += MS_REWARD[1]
        # ---- 站位维持费 ----
        if not self.g[2] and pads3 and self.wage_paid < WAGE_CAP:
            pay = min(WAGE, WAGE_CAP - self.wage_paid)
            out["wage"] = pay
            self.wage_paid += pay
        # ---- G2 认证机 (提升测试; [TASK] 滑移只判左侧: 右手此时尚未进场) ----
        if self.cert_wait > 0:
            self.cert_wait -= 1
        if (self.g[1] and not self.g[2] and self.cert_phase == 0
                and self.cert_wait == 0 and self.cert_try < CERT_TRIES):
            self.cert_phase, self.cert_t = 1, 0
            self.cert_z0 = {0: float(obj0[2]), 1: float(obj1[2])}
            self.cert_rel0 = (np.asarray(wrist_l, np.float64)
                              - np.asarray(obj0[:3]))
        elif self.cert_phase > 0:
            self.cert_t += 1
            if self.cert_phase == 1 and self.cert_t >= CERT_RAMP:
                self.cert_phase, self.cert_t = 2, 0
            elif self.cert_phase == 2 and self.cert_t >= CERT_HOLD:
                rel_l = np.linalg.norm(
                    (np.asarray(wrist_l, np.float64) - np.asarray(obj0[:3]))
                    - self.cert_rel0)
                # 盖被螺旋钉在瓶上, 判双物 z 升 = 瓶被举起且装配未散
                ok5 = (float(obj0[2]) - self.cert_z0[0] >= CERT_RISE
                       and float(obj1[2]) - self.cert_z0[1] >= CERT_RISE
                       and rel_l < CERT_SLIP and pads3)
                if ok5:
                    self.cert_pending = True
                else:
                    self.cert_try += 1
                    self.cert_wait = CERT_WAIT
                self.cert_phase, self.cert_t = 3, 0
            elif self.cert_phase == 3 and self.cert_t >= CERT_RET:
                self.cert_phase, self.cert_t = 0, 0
                if self.cert_pending:
                    self.cert_pending = False
                    self.g[2] = True
                    out["ms"] += MS_REWARD[2]
        out["cert_phase"], out["cert_t"] = self.cert_phase, self.cert_t
        # ==== [TASK] G3 任务核心 = 拧开释放 (螺旋 detach 是物理锁存, 即判) ====
        self.released = self.released or bool(screw_released)
        if self.g[2] and not self.g[3] and self.released:
            self.g[3] = True
            out["ms"] += MS_REWARD[3]
        # ---- [TASK] 护送三件 (T2-3 过带 + U41 持盖/受控下放) ----
        # pads_r_cap = 右垫触盖**计数** (None = 离线探针, 三件全部停用).
        if self.g[3] and not self.placed and pads_r_cap is not None:
            _npads = int(pads_r_cap)
            _cz = float(np.asarray(obj1, np.float64)[2])
            _top = TABLE_Z + ESCORT_BAND
            if self._cap_z_prev is not None:
                _drop = self._cap_z_prev - _cz
                if (self._cap_z_prev > _top >= _cz and _drop > ESCORT_FALL
                        and _npads < 1):
                    self.escort_fail = True         # ① 无接触过带
                self.fall_peak = max(self.fall_peak, _drop)      # ③ 降幅峰值
            if _npads >= CARRY_PADS:
                self.carry_steps += 1.0                          # ② 持盖运输
            self._cap_z_prev = _cz
        out["escort_fail"] = self.escort_fail
        out["carry_steps"] = self.carry_steps
        out["fall_peak"] = self.fall_peak
        _escorted = (self.escort_fail
                     or (pads_r_cap is not None
                         and (self.carry_steps < PLACE_CARRY
                              or self.fall_peak >= PLACE_FALL)))
        # ---- placed: [TASK] 双物体到母带末行目标 (瓶3cm/15° 盖5cm/30°) hold15
        #      + 护送未失败 (右手拿着盖放下, 不是盖自己掉到目标) ----
        if self.g[3] and not self.placed and not _escorted:
            _ok3 = True
            for _oi3, _act3 in ((0, obj0), (1, obj1)):
                _dp3 = np.linalg.norm(np.asarray(_act3[:3])
                                      - self.end[_oi3][:3])
                _tl3 = abs(_axis_tilt(np.asarray(_act3, np.float64)[3:7],
                                      UP_LOCAL[_oi3])
                           - _axis_tilt(self.end[_oi3][3:7], UP_LOCAL[_oi3]))
                if _dp3 > PLACED_POS[_oi3] or _tl3 > PLACED_ROT[_oi3]:
                    _ok3 = False
            if _ok3:
                self.m3_run += 1
                if self.m3_run >= PLACED_HOLD:
                    self.placed = True
                    self.m3_snap = {0: np.asarray(obj0, np.float64).copy(),
                                    1: np.asarray(obj1, np.float64).copy()}
            else:
                self.m3_run = 0
        # ---- G4 = Success: placed 后撤退归位, 物体不被碰倒碰歪 ----
        if self.placed and not self.g[4]:
            _ok4 = (np.abs(np.asarray(armq_r) - self.stance["right"]).max() <= M4_ARM
                    and np.abs(np.asarray(armq_l) - self.stance["left"]).max()
                    <= M4_ARM)
            for _oi4, _act4 in ((0, obj0), (1, obj1)):
                _sn = self.m3_snap[_oi4]
                if (np.linalg.norm(np.asarray(_act4[:3]) - _sn[:3]) > M4_DIST_POS
                        or abs(_axis_tilt(np.asarray(_act4, np.float64)[3:7],
                                          UP_LOCAL[_oi4])
                               - _axis_tilt(_sn[3:7], UP_LOCAL[_oi4]))
                        > M4_DIST_ROT):
                    _ok4 = False
            if _ok4:
                self.m4_run += 1
                if self.m4_run >= M4_HOLD:
                    self.g[4] = True
                    self.done = True
                    out["ms"] += MS_REWARD[4]
                    out["done"] = True
            else:
                self.m4_run = 0
        # ---- 死线 D1/D2pre/D3/D8 ----
        # D2pre: G2 前瓶倾>60° = 倒伏即终 ([TASK] 只判瓶 —— 盖被螺旋钉着同倾角)
        if (not self.g[2] and _axis_tilt(np.asarray(obj0, np.float64)[3:7],
                                         UP_LOCAL[0]) > D2_PRE_TILT):
            out["fail"] = "D2pre_fallen_obj0"
        for oi, act in ((0, obj0), (1, obj1)):
            a = np.asarray(act, np.float64)
            if a[2] < TABLE_Z - D1_DROP:
                out["fail"] = f"D1_drop_obj{oi}"
            if not self.placed:
                if np.linalg.norm(a[:3] - self.obj[oi][k][:3]) > D3_DEV:
                    out["fail"] = f"D3_dev_obj{oi}"
            else:
                sn = self.m3_snap[oi]
                if (np.linalg.norm(a[:3] - sn[:3]) > M4_DIST_POS
                        or abs(_axis_tilt(a[3:7], UP_LOCAL[oi])
                               - _axis_tilt(sn[3:7], UP_LOCAL[oi]))
                        > M4_DIST_ROT):
                    out["fail"] = f"D8_disturb_obj{oi}"
                if abs(_axis_tilt(a[3:7], UP_LOCAL[oi])
                       - _axis_tilt(self.end[oi][3:7], UP_LOCAL[oi])) > D2_TILT:
                    out["fail"] = f"D2_tilt_obj{oi}"
        if out["fail"] is not None:
            self.done = True
            out["done"] = True
        out["clock"] = self.k
        return out
