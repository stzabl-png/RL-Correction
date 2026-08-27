"""Pour17 进度条 (2026-08-27 #8 定稿实现) —— 置信门控双参考驱动。

定稿数值:
  档位阈值      conf(=各自分量) >=70 绿 / >=40 黄 / 否则红
  双参考权重    绿 w_obj=0.8/w_hand=0.2 | 黄 0.5/0.5 | 红 0.2/0.8   (按 min(pos,rot) 档)
  皮筋容差      pos: 绿3/黄5/红8 cm (按 conf_pos 档) ; rot: 绿15/黄30° 红=禁用 (按 conf_rot 档)
  时钟粗容差    5cm / 45° (绿黄=双物体贴物体轨迹; 红=双手贴人手轨迹, 同阈值)
  里程碑        M1 双手抓稳 +5 | M2 真倾倒(倾角>=90° × 口口距<=阈) +8 | M3 平稳放回 +10+终局
  收入          Δclock 棘轮 earn-only (1/行) + 里程碑一次性; 有序强制 M1<M2<M3

消费母带: pour17_reference_v1.npz (source/frame_of_row/conf 列)。
机器段(approach/retreat)不归本模块管辖 —— 本模块只覆盖 interact 行。
"""
from __future__ import annotations

import numpy as np

TIER_HI, TIER_LO = 70.0, 40.0
W_OBJ = {2: 0.8, 1: 0.5, 0: 0.2}
LEASH_POS = {2: 0.03, 1: 0.05, 0: 0.08}
LEASH_ROT = {2: np.radians(15), 1: np.radians(30), 0: None}   # 红档 rot 禁入判据
GATE_POS, GATE_ROT = 0.05, np.radians(45)
MS_REWARD = {1: 5.0, 2: 8.0, 3: 10.0, 4: 15.0}   # M4=Success(撤退归位)终局
M2_TILT = np.radians(90)
M2_HOLD = 25          # 2026-08-27 拍板: 重建实测倒水2.2s(33帧)的~75%, 留余量
M3_POS, M3_ROT, M3_HOLD = 0.03, np.radians(15), 15
M4_ARM, M4_HOLD = np.radians(10), 15          # 双臂贴站姿逐关节<10°, hold15
M4_DIST_POS, M4_DIST_ROT = 0.05, np.radians(30)   # 撤退期物体相对M3快照的扰动上限
# ---- 死线 (2026-08-27 #10 全表拍板; D4-D7 属 env 侧接线) ----
D1_DROP = 0.05          # 物体低于桌面 5cm
D2_TILT = np.radians(30)  # 绝对倾倒 (M3后撤退段生效; 交互段豁免)
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
    """时钟=交互行索引, 只进不退 (行进按进度)。每步喂实测, 吐奖励账目."""

    def __init__(self, npz_path, mouth_local_bot, mouth_local_cup,
                 up_local_bot=(0.0, 1.0, 0.0), mouth_gate=0.12,
                 leash_rot_tilt=False):
        # leash_rot_tilt (E线消融 2026-08-28): 皮筋 rot 改"倾角差"口径(yaw豁免),
        # 默认 False=原全角度。胜出才转正。
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
        self.armq = {s: np.asarray(z[f"{s}_q"], np.float64)[rows]
                     for s in ("right", "left")}
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
        self.reset()

    def entry_table(self):
        """RSI 进入点 (2026-08-27 #11): 自动推导, [(row, preset_ms, label)]。
        缝桶=交互首/末行(脆弱点故意超采); 绿桶=tmix绿段起点; 红黄禁入。
        M2 预置规则: 进入行在倒水段之后(参考倾角剖面自动判)则预置。"""
        ends = [i for i in range(self.N) if _axis_tilt(self.obj[1][i][3:7], self.up_b)
                >= M2_TILT]
        pour_end = max(ends) if ends else self.N
        et = [(0, {1}, "seam1")]
        # 绿桶退役 (2026-08-28 拍板, #11 修订): 中途悬空态物理不可实例化
        # (热身期垫力恒0, 手指够不到面 —— B@1.87M 逐步验尸实证), 出生即送死。
        _ = pour_end  # 保留推导供将来复用
        et.append((self.N - 1, {1, 2, 3}, "seam2_ret"))
        return et

    def enter(self, row, preset_ms):
        """按进入点初始化 (env 侧负责把该行位姿写进仿真)."""
        self.reset()
        self.k = int(row)
        for m in preset_ms:
            self.ms[m] = True
        if 3 in preset_ms:
            # 撤退点语义 = "已妥善放回": 快照取静置位, 不用母带尾行(recon尾不可信)
            self.m3_snap = {oi: self.rest[oi].copy() for oi in (0, 1)}

    def reset(self):
        self.k = 0                       # 时钟 (交互行内索引), 棘轮
        # 药A 持握换基 (2026-08-28 拍板): M1 后皮筋 pos 判"相对抓住时刻偏差的增量"
        # —— 持握固有差(下垂+recon系统差 ~4-5cm)归零, 防"存在税"激励倒挂;
        # 越滑越远照罚。锚逻辑与 D4 grasp_d0 同族。
        self.leash_base = {0: np.zeros(3), 1: np.zeros(3)}
        self._lb_set = False
        self.ms = {1: False, 2: False, 3: False, 4: False}
        self.m2_run = 0
        self.m3_run = 0
        self.m4_run = 0
        self.m3_snap = None                      # M3 时刻双物体位姿快照
        self.done = False

    # -- 逐步接口: 实测 {obj0/obj1: (7,), armq_r/armq_l: (7,), cand_ok: bool} --
    def step(self, obj0, obj1, armq_r, armq_l, cand_ok):
        out = {"clock": self.k, "adv": 0.0, "leash": 0.0, "w_obj": None,
               "ms": 0.0, "done": False, "gate_by": None, "fail": None}
        if self.done:
            out["done"] = True
            return out
        k = min(self.k, self.N - 1)
        tier = self.tmix[k]
        out["w_obj"] = W_OBJ[tier]
        # 时钟耗尽后皮筋/时钟门停判 (2026-08-28 接线口径): 母带尾行是 recon 病值
        # (瓶仍歪53°), 冻结参考继续判偏差 = 奖励"抱着歪瓶不放"、罚正确放回
        # (实测190步累计-26 > M3+M4的+25)。交接给 M3/M4/D8/D1/D3。
        earning = self.k < self.N - 1
        ok = False
        # 药A: M1 后首个受管步捕获持握基线 (放音/缝点出生时 act=ref, b=0 无损)
        if self.ms[1] and not self._lb_set:
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
        # ---- 时钟门 (绿黄=物体贴轨; 红=双手贴人手轨迹; 仅时钟在走时) ----
        if not earning:
            pass
        elif tier > 0:
            ok = True
            for oi, act in ((0, obj0), (1, obj1)):
                ref = self.obj[oi][k]
                if (np.linalg.norm(np.asarray(act[:3]) - ref[:3]) > GATE_POS
                        or _qang(np.asarray(act[3:7]), ref[3:7]) > GATE_ROT):
                    ok = False
            out["gate_by"] = "obj"
        else:
            ok = True
            for s, act in (("right", armq_r), ("left", armq_l)):
                if float(np.abs(np.asarray(act) - self.armq[s][k]).max()) \
                        > np.radians(20):                    # 关节口径手门(红段)
                    ok = False
            out["gate_by"] = "hand"
        # 自主抓稳阶段 (2026-08-29 拍板): M1(双手抓稳)前交互时钟不走 ——
        # 参考停在抓握站位等策略练稳, 不在没抓牢时把物体拖走 (提起墙验尸的结构解)
        if earning and ok and self.k < self.N - 1 and self.ms[1]:
            self.k += 1
            out["adv"] = 1.0                                # Δclock earn-only
        # ---- 里程碑 (有序强制) ----
        if not self.ms[1] and cand_ok:
            self.ms[1] = True
            out["ms"] += MS_REWARD[1]
        if self.ms[1] and not self.ms[2]:
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
                    self.ms[2] = True
                    out["ms"] += MS_REWARD[2]
            else:
                self.m2_run = 0
        if self.ms[2] and not self.ms[3]:
            # 2026-08-27 拍板: 双物体 AND —— 瓶和杯都回各自静置位才算放回
            _ok3 = True
            for _oi3, _act3 in ((0, obj0), (1, obj1)):
                _dp3 = np.linalg.norm(np.asarray(_act3[:3]) - self.rest[_oi3][:3])
                # rot 口径 = 长轴倾角 (yaw 豁免, 2026-08-28 拍板): 旋转对称物体
                # 绕竖轴自转无意义, 全四元数角会把完美放回判死 (自检盲区: 放音
                # 喂的是母带自身 yaw, 抓不到此病)
                _up3 = self.up_b if _oi3 == 1 else np.array([0.0, 1.0, 0.0])
                _tl3 = _axis_tilt(np.asarray(_act3, np.float64)[3:7], _up3)
                if _dp3 > M3_POS or _tl3 > M3_ROT:
                    _ok3 = False
            if _ok3:
                self.m3_run += 1
                if self.m3_run >= M3_HOLD:
                    self.ms[3] = True            # 放回≠功成 (2026-08-28 裁定): 不终局
                    out["ms"] += MS_REWARD[3]
                    self.m3_snap = {0: np.asarray(obj0, np.float64).copy(),
                                    1: np.asarray(obj1, np.float64).copy()}
            else:
                self.m3_run = 0
        # ---- M4 = Success: M3 后撤退归位, 物体不被碰倒碰歪 ----
        if self.ms[3] and not self.ms[4]:
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
                    self.ms[4] = True
                    self.done = True
                    out["ms"] += MS_REWARD[4]
                    out["done"] = True
            else:
                self.m4_run = 0
        # ---- 死线 D1/D2/D3/D8 (判据单一来源; 触发即 fail, env 侧终止扣罚) ----
        # D2交互段补丁 (2026-08-28 拍板): M1 前物体倾角>60° = 没抓住就倒了,
        # 不可挽回即终 (堵"躺桌limbo"白烧600步)。参考放音 M1 前物体全程立正,
        # 铁则零触发; M1 后掉落归 D4。豁免语义不变: 在手里歪是合法的。
        if not self.ms[1]:
            for oi, act in ((0, obj0), (1, obj1)):
                up_ = self.up_b if oi == 1 else np.array([0.0, 1.0, 0.0])
                if _axis_tilt(np.asarray(act, np.float64)[3:7], up_) \
                        > np.radians(60):
                    out["fail"] = f"D2pre_fallen_obj{oi}"
        for oi, act in ((0, obj0), (1, obj1)):
            a = np.asarray(act, np.float64)
            if a[2] < TABLE_Z - D1_DROP:
                out["fail"] = f"D1_drop_obj{oi}"
            if not self.ms[3]:
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
