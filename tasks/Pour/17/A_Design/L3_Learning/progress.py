"""Pour17 Success Tracker (L5-1 换代, 2026-08-27) —— 四Gate阶段机 + 5mm认证测试。

体制沿革: #8 置信门控双参考驱动(绿0.8/0.2 黄0.5/0.5 红0.2/0.8, 皮筋pos3/5/8cm
rot15/30°/红禁, 时钟门5cm/45° 红档人手关节口径20°) 原样保留;
里程碑机 M1-M4 换代为:
  G1 GraspPose形成: 双手各>=3/5垫(>0.5N) 稳10步            +5
  G2 抓稳认证: 5mm提升测试(斜坡5+保持5, 判双物z升>=3mm 且
     手物相对位移<5mm 且垫>=3; 5步放回; 败→20步重握窗, 至多3次) +8
  G3 倒水完成: 倾角>=90° × 口口距<=阈 hold25 (双物体AND)      +10
  placed 平稳放回: 双物回静置3cm/15° hold15 (内部态, 不付奖)
  G4 放稳=Success: 撤退归位 + 物体不被碰倒碰歪 (相对placed快照) +15 终局
时钟: G2 前不走(自主抓稳继承); 渐进RSI: entry_table(unlocked) 按已解锁Gate给出生点。
变体: no_hand_ref 只影响 out["w_obj"]/out["w_hand"](奖励侧输出);
      ★L5-6 起时钟门已双体制统一(红档=宽物门8cm、rot不判), 人手绝对位姿参考
      全面废除 —— 故本旗对判据/成败/死线**零影响**, 评测侧保持默认即可。
消费母带: pour17_reference_v2.npz (v1 亦兼容: 无 human_* 时用主行)。
机器段(approach/retreat)不归本模块管辖 —— 本模块只覆盖 interact 行。
"""
from __future__ import annotations

import numpy as np

TIER_HI, TIER_LO = 70.0, 40.0
W_OBJ = {2: 1.0, 1: 0.5, 0: 0.2}    # L5-6: 高置信=纯物轨(人手0)
W_HAND = {2: 0.0, 1: 0.5, 0: 0.8}   # 人手形状指引权重: 绿0/黄五五开/红主导
LEASH_POS = {2: 0.03, 1: 0.05, 0: 0.08}
LEASH_ROT = {2: np.radians(15), 1: np.radians(30), 0: None}   # 红档 rot 禁入判据
GATE_POS, GATE_ROT = 0.05, np.radians(45)
RED_GATE_POS = 0.08          # P-OBJ 红档宽物门 (无手接管时的口径)
MS_REWARD = {1: 5.0, 2: 8.0, 3: 10.0, 4: 15.0}   # G1/G2/G3/G4(=Success终局)
M2_TILT = np.radians(90)
M2_HOLD = 25          # G3 hold: 重建实测倒水2.2s(33帧)的~75%, 留余量
M3_POS, M3_ROT, M3_HOLD = 0.03, np.radians(15), 15   # placed 判据 (原M3)
M4_ARM, M4_HOLD = np.radians(10), 15          # G4 双臂贴站姿逐关节<10°, hold15
M4_DIST_POS, M4_DIST_ROT = 0.05, np.radians(30)   # 撤退期物体相对placed快照扰动上限
# ---- G1/G2 认证 (L5-1 拍板) ----
G1_HOLD = 10                  # 双手>=3/5垫连续步数
CERT_RAMP, CERT_HOLD, CERT_RET = 8, 5, 8
# 认证幅度 +15mm/判升>=5mm (2026-08-28 显微探针标定: 5mm在握持柔性死区内,
# 右腕实际+2.9mm瓶+0.2mm而β2.0握扛得住15cm提升 —— 幅度必须高出死区才可观测)
CERT_RISE = 0.005             # 双物 z 升 >=5mm (认证行=+15mm)
CERT_SLIP = 0.008             # 手物相对位移 <8mm
CERT_WAIT, CERT_TRIES = 20, 3
WAGE = 0.05                   # 站位维持费 (G2 前, 双手垫>=3 时逐步)
WAGE_CAP = 3.0                # L5-8 药④: 每回合工资总额上限 (< G2的+8, 断躺平诱饵)
# ---- 死线 (#10 全表; D4-D7 属 env 侧接线) ----
D1_DROP = 0.05          # 物体低于桌面 5cm
D2_TILT = np.radians(30)  # 绝对倾倒 (placed后撤退段生效; 交互段豁免)
D3_DEV = 0.35           # 离参考 35cm (瓶峰19.3+15余量)
TABLE_Z = 0.87


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


class PourProgress:
    """时钟=交互行索引, 只进不退。每步喂实测, 吐奖励账目."""

    def __init__(self, npz_path, mouth_local_bot, mouth_local_cup,
                 up_local_bot=(0.0, 1.0, 0.0), mouth_gate=0.12,
                 leash_rot_tilt=False, kcap=None, no_hand_ref=False):
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
        self.tmix = [min(a if a is not None else 2, b if b is not None else 2)
                     for a, b in zip(self.tr[1], self.tp[1])]   # 主档=瓶 min(pos,rot)
        self.rest = {oi: self.obj[oi][0].copy() for oi in (0, 1)}
        self.mouth_b = np.asarray(mouth_local_bot, np.float64)
        self.mouth_c = np.asarray(mouth_local_cup, np.float64)
        self.up_b = np.asarray(up_local_bot, np.float64)
        self.mouth_gate = float(mouth_gate)
        self.leash_rot_tilt = bool(leash_rot_tilt)
        self.kcap = kcap
        self.no_hand_ref = bool(no_hand_ref)
        self.reset()

    # ---- 渐进 RSI (L5-1 拍板3): 初始仅 t0, Gate_k EMA>=0.2 解锁 ----
    def entry_table(self, unlocked=()):
        """unlocked: 已解锁的 Gate 集合 (e.g. {1,2})。返回 [(row, preset, label)]。
        preset 元素: 1/2/3 = G1/G2/G3 预置, 'placed' = 放回预置(快照取静置)。"""
        et = [(0, set(), "t0")]
        u = set(unlocked)
        if 1 in u:
            et.append((0, {1}, "g1"))
        if 2 in u:
            et.append((0, {1, 2}, "g2"))
        if 3 in u:
            ends = [i for i in range(self.N)
                    if _axis_tilt(self.obj[1][i][3:7], self.up_b) >= M2_TILT]
            pour_end = min(max(ends) + 1, self.N - 1) if ends else self.N - 1
            et.append((pour_end, {1, 2, 3}, "g3"))
            et.append((self.N - 1, {1, 2, 3, "placed"}, "ret"))
        return et

    def enter(self, row, preset_ms):
        """按进入点初始化 (env 侧负责把该行位姿写进仿真)."""
        self.reset()
        self.k = int(row)
        for m in preset_ms:
            if m == "placed":
                self.placed = True
                # 撤退点语义 = "已妥善放回": 快照取静置位(recon尾不可信)
                self.m3_snap = {oi: self.rest[oi].copy() for oi in (0, 1)}
            else:
                self.g[m] = True

    def reset(self):
        self.k = 0                       # 时钟 (交互行内索引), 棘轮
        # 药A 持握换基: G2 后皮筋 pos 判"相对抓住时刻偏差的增量"
        self.leash_base = {0: np.zeros(3), 1: np.zeros(3)}
        self._lb_set = False
        self.g = {1: False, 2: False, 3: False, 4: False}
        self.placed = False
        self.g1_run = 0
        self.wage_paid = 0.0
        # 认证机
        self.cert_phase = 0              # 0=idle 1=ramp 2=hold 3=return
        self.cert_t = 0
        self.cert_try = 0
        self.cert_wait = 0
        self.cert_pending = False
        self.cert_z0 = {0: 0.0, 1: 0.0}
        self.cert_rel0 = {"right": np.zeros(3), "left": np.zeros(3)}
        self.m2_run = 0
        self.m3_run = 0
        self.m4_run = 0
        self.m3_snap = None
        self.done = False

    # -- 逐步接口: obj0/obj1 (7,), armq_r/l (7,), pads3 bool(双手各>=3/5垫),
    #    wrist_r/l (3,) 腕世界位置(与 obj 同坐标系) --
    def step(self, obj0, obj1, armq_r, armq_l, pads3, wrist_r, wrist_l):
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
        earning = self.k < self.N - 1
        ok = False
        # 药A: G1(抓形成) 后首个受管步捕获持握基线
        # L5-8 药⑤: 原挂 G2, 而 G2 不可达 → "握"这个动作被永久课税(实测皮筋独大)
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
                leash -= (dp - lp) / lp                     # 连续渐强
            lr = LEASH_ROT[trot if trot is not None else 2]
            if lr is not None:
                if self.leash_rot_tilt:
                    up_l = self.up_b if oi == 1 else np.array([0.0, 1.0, 0.0])
                    dr = abs(_axis_tilt(np.asarray(act, np.float64)[3:7], up_l)
                             - _axis_tilt(ref[3:7], up_l))
                else:
                    dr = _qang(np.asarray(act[3:7]), ref[3:7])
                if dr > lr:
                    leash -= (dr - lr) / lr
        out["leash"] = max(leash, -3.0)                     # 截断
        # ---- 时钟门 (L5-6: 双变体统一; 红档=宽松物门, 绝对位置底线归物轨) ----
        if earning:
            ok = True
            gp = GATE_POS if tier > 0 else RED_GATE_POS
            for oi, act in ((0, obj0), (1, obj1)):
                ref = self.obj[oi][k]
                if np.linalg.norm(np.asarray(act[:3]) - ref[:3]) > gp:
                    ok = False
                if tier > 0 and _qang(np.asarray(act[3:7]), ref[3:7]) > GATE_ROT:
                    ok = False                              # 红档 rot 禁入(形状不可信)
            out["gate_by"] = "obj"
        # 时钟: G2 前不走 (自主抓稳继承; 认证期 phase>0 亦不走, 因 g2 未立)
        _cap = self.N - 1 if self.kcap is None else min(self.kcap, self.N - 1)
        if earning and ok and self.k < _cap and self.g[2]:
            self.k += 1
            out["adv"] = 1.0                                # Δclock earn-only
        # ---- G1: 双手 >=3/5 垫 稳 G1_HOLD 步 ----
        if not self.g[1]:
            self.g1_run = self.g1_run + 1 if pads3 else 0
            if self.g1_run >= G1_HOLD:
                self.g[1] = True
                out["ms"] += MS_REWARD[1]
        # ---- 站位维持费: G2 前, 垫>=3 逐步小额, 每回合封顶 ----
        if not self.g[2] and pads3 and self.wage_paid < WAGE_CAP:
            pay = min(WAGE, WAGE_CAP - self.wage_paid)
            out["wage"] = pay
            self.wage_paid += pay
        # ---- G2 认证机 (5mm 提升测试) ----
        if self.cert_wait > 0:
            self.cert_wait -= 1
        if (self.g[1] and not self.g[2] and self.cert_phase == 0
                and self.cert_wait == 0 and self.cert_try < CERT_TRIES):
            self.cert_phase, self.cert_t = 1, 0
            self.cert_z0 = {0: float(obj0[2]), 1: float(obj1[2])}
            self.cert_rel0 = {
                "right": np.asarray(wrist_r, np.float64) - np.asarray(obj1[:3]),
                "left": np.asarray(wrist_l, np.float64) - np.asarray(obj0[:3])}
        elif self.cert_phase > 0:
            self.cert_t += 1
            if self.cert_phase == 1 and self.cert_t >= CERT_RAMP:
                self.cert_phase, self.cert_t = 2, 0
            elif self.cert_phase == 2 and self.cert_t >= CERT_HOLD:
                rel_r = np.linalg.norm(
                    (np.asarray(wrist_r, np.float64) - np.asarray(obj1[:3]))
                    - self.cert_rel0["right"])
                rel_l = np.linalg.norm(
                    (np.asarray(wrist_l, np.float64) - np.asarray(obj0[:3]))
                    - self.cert_rel0["left"])
                ok5 = (float(obj1[2]) - self.cert_z0[1] >= CERT_RISE
                       and float(obj0[2]) - self.cert_z0[0] >= CERT_RISE
                       and rel_r < CERT_SLIP and rel_l < CERT_SLIP and pads3)
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
        # ---- G3 倒水完成 (原M2判据; 双物体AND=口口距) ----
        if self.g[2] and not self.g[3]:
            tilt = _axis_tilt(np.asarray(obj1[3:7]), self.up_b)
            w, x, y, z_ = np.asarray(obj1[3:7]) / np.linalg.norm(obj1[3:7])
            Rb = np.array([[1-2*(y*y+z_*z_), 2*(x*y-w*z_), 2*(x*z_+w*y)],
                           [2*(x*y+w*z_), 1-2*(x*x+z_*z_), 2*(y*z_-w*x)],
                           [2*(x*z_-w*y), 2*(y*z_+w*x), 1-2*(x*x+y*y)]])
            mb = np.asarray(obj1[:3]) + Rb @ self.mouth_b
            w2, x2, y2, z2 = np.asarray(obj0[3:7]) / np.linalg.norm(obj0[3:7])
            Rc = np.array([[1-2*(y2*y2+z2*z2), 2*(x2*y2-w2*z2), 2*(x2*z2+w2*y2)],
                           [2*(x2*y2+w2*z2), 1-2*(x2*x2+z2*z2), 2*(y2*z2-w2*x2)],
                           [2*(x2*z2-w2*y2), 2*(y2*z2+w2*x2), 1-2*(x2*x2+y2*y2)]])
            mc = np.asarray(obj0[:3]) + Rc @ self.mouth_c
            if tilt >= M2_TILT and np.linalg.norm(mb - mc) <= self.mouth_gate:
                self.m2_run += 1                       # hold: 持续倒水才算真倒
                if self.m2_run >= M2_HOLD:
                    self.g[3] = True
                    out["ms"] += MS_REWARD[3]
            else:
                self.m2_run = 0
        # ---- placed 平稳放回 (原M3判据; 内部态, 不付奖) ----
        if self.g[3] and not self.placed:
            _ok3 = True
            for _oi3, _act3 in ((0, obj0), (1, obj1)):
                _dp3 = np.linalg.norm(np.asarray(_act3[:3]) - self.rest[_oi3][:3])
                _up3 = self.up_b if _oi3 == 1 else np.array([0.0, 1.0, 0.0])
                _tl3 = _axis_tilt(np.asarray(_act3, np.float64)[3:7], _up3)
                if _dp3 > M3_POS or _tl3 > M3_ROT:
                    _ok3 = False
            if _ok3:
                self.m3_run += 1
                if self.m3_run >= M3_HOLD:
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
                _up4 = self.up_b if _oi4 == 1 else np.array([0.0, 1.0, 0.0])
                if (np.linalg.norm(np.asarray(_act4[:3]) - _sn[:3]) > M4_DIST_POS
                        or _axis_tilt(np.asarray(_act4, np.float64)[3:7], _up4)
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
        # ---- 死线 D1/D2/D3/D8 (D2pre: G2前物体倾>60°=倒伏即终) ----
        if not self.g[2]:
            for oi, act in ((0, obj0), (1, obj1)):
                up_ = self.up_b if oi == 1 else np.array([0.0, 1.0, 0.0])
                if _axis_tilt(np.asarray(act, np.float64)[3:7], up_) \
                        > np.radians(60):
                    out["fail"] = f"D2pre_fallen_obj{oi}"
        for oi, act in ((0, obj0), (1, obj1)):
            a = np.asarray(act, np.float64)
            if a[2] < TABLE_Z - D1_DROP:
                out["fail"] = f"D1_drop_obj{oi}"
            if not self.placed:
                if np.linalg.norm(a[:3] - self.obj[oi][k][:3]) > D3_DEV:
                    out["fail"] = f"D3_dev_obj{oi}"
            else:
                sn = self.m3_snap[oi]
                up8 = self.up_b if oi == 1 else np.array([0.0, 1.0, 0.0])
                if (np.linalg.norm(a[:3] - sn[:3]) > M4_DIST_POS
                        or _axis_tilt(a[3:7], up8) > M4_DIST_ROT):
                    out["fail"] = f"D8_disturb_obj{oi}"
                up_ = self.up_b if oi == 1 else np.array([0.0, 1.0, 0.0])
                if _axis_tilt(a[3:7], up_) > D2_TILT:
                    out["fail"] = f"D2_tilt_obj{oi}"
        if out["fail"] is not None:
            self.done = True
            out["done"] = True
        out["clock"] = self.k
        return out
