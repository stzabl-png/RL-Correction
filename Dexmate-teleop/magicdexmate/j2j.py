"""J2J(joint-to-joint 形似直映)映射律 —— 可复用模块。

人体骨架(body24 模型通道)的解剖角逐关节映射到 Vega 手臂,不解 IK:
肩部→j1/j2,上臂自旋→j3,肘屈伸→j4,前臂自旋→j5,腕屈伸→j7,j6 固定 0。
数学与 `sim/dev_j2j_feasibility.py`(2026-08-10 的离线可行性分析)完全相同,
该脚本是这套数学的第一副本并已通过全部自检;**两份暂存**,J2J 接进链路定稿
时以本模块为准、dev 脚本改 import(同 `_SMPL_*` 常量两份暂存的先例)。

用法(任何消费端 —— 预览台、将来的 `--map j2j` —— 都只准调这里,自己不算):

    from magicdexmate.j2j import J2JMapper
    m = J2JMapper()                       # 读 VEGA_URDF 或默认 URDF
    out = m.map_body24(use)               # use = {9/16/17/18/19/20/21: pose7}
    out["right"].q7                       # 执行角(j6 已置 0),None = 无解
    out["right"].flags                    # (肩死区, 肘死区, 腕死区, 无解)

构造约定(与可行性脚本逐条一致,自检可证伪):
  * 人体参考系 = 腰 tracker 去偏航的重力系(process_xr_pose);骨架点取
    body24 模型通道(16/17 肩、18/19 肘、20/21 腕、9 腰),与 PALM_FIX
    的有效通道一致;
  * 机器人侧在 arm_center(胸)系分解,躯干锁 home(胸=绕 y 纯俯仰 +55°,
    自检 F);人体向量预旋转 Ry(-55°) 进胸系 = j1 的 55° 常数偏置;
  * 分解是机器人轴序列的闭式反解(自检 A 与 pinocchio FK 对拍),肘/腕
    曲柄偏距(7.4/4.4cm)用上一帧 j3/j5 闭式补偿;肩心 2cm 公垂线间隙是
    固有几何误差(自检 B2 量上界);
  * 双解族选支带限位罚分(没有罚分时死区里一次抽风会永久甩到反手支);
  * 肘伸直(±15°)时 j3 不可观测,保持上一帧;垂臂即此死区(自检 C)。

运行自检:`.venv/bin/python -m magicdexmate.j2j`(全部尺子先过已知答案)。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from magicdexmate.home_pose import ARM_HOME
from magicdexmate.palm_fix import PALM_FIX
from magicdexmate.pico.xr_pose import process_xr_pose

DEFAULT_URDF = os.environ.get(
    "VEGA_URDF",
    os.path.expanduser("~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf"))
TORSO_HOME = {"torso_j1": 1.047198, "torso_j2": 1.570796, "torso_j3": -0.436332}
# 操作者的腰/肩/肘/腕在 body24 里的下标(SMPL 顺序)
BODY = {"waist": 9, "l_sh": 16, "r_sh": 17, "l_el": 18, "r_el": 19,
        "l_wr": 20, "r_wr": 21}
HANDS = ("right", "left")
EPS = {"right": -1.0, "left": +1.0}        # rpy=pi 翻转折叠出的符号(自检 A)
SIN_DEAD_SH = np.sin(np.radians(15.0))     # 上臂贴 j1 轴 15° 以内 = 肩死区
SIN_DEAD_EL = np.sin(np.radians(15.0))     # 前臂贴肱骨轴 15° 以内 = j3 死区
BETA_DEAD = np.radians(75.0)               # 欧拉中间角接近 ±90° = 腕死区
D2 = np.degrees


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ang_of(Rm) -> float:
    """旋转矩阵的转角[rad]。"""
    return float(np.linalg.norm(pin.log3(np.asarray(Rm))))


def euler_xyz(H):
    """H = Rx(a)·Ry(b)·Rz(c) 的 (a,b,c),b∈[-90°,90°](自检 B1)。"""
    b = np.arcsin(np.clip(H[0, 2], -1.0, 1.0))
    c = np.arctan2(-H[0, 1], H[0, 0])
    a = np.arctan2(-H[1, 2], H[2, 2])
    return float(a), float(b), float(c)


def _solve_ab(A, B, C):
    """A·cos(x)+B·sin(x)=C 的两解;|C| 超可达幅值时贴边求解(方向出锥)。"""
    Rm = np.hypot(A, B)
    if Rm < 1e-9:
        return None
    phi, d = np.arctan2(B, A), np.arccos(np.clip(C / Rm, -1.0, 1.0))
    return wrap(phi - d), wrap(phi + d)


@dataclass
class J2JOut:
    """一只手一帧的映射输出。q7 为执行角(j6=0);beta 是被丢弃的 j6 需求。"""
    q7: np.ndarray | None
    flags: tuple[bool, bool, bool, bool]     # (肩死区, 肘死区, 腕死区, 无解)
    beta: float
    u_g: np.ndarray | None = None            # 重力系上臂/前臂方向(画骨架用)
    f_g: np.ndarray | None = None
    S: np.ndarray | None = None              # 重力系肩/肘/腕点(腰原点)
    E: np.ndarray | None = None
    W: np.ndarray | None = None


class J2JMapper:
    """闭式 J2J 映射器。有状态:上一帧 q7 用于解族连续性 + 曲柄补偿。"""

    def __init__(self, urdf_path: str = DEFAULT_URDF,
                 max_step_deg: float = 2.9):
        self.max_step = np.radians(max_step_deg)   # 执行限速,°/帧
        self.full = pin.buildModelFromUrdf(urdf_path)
        non_arm = ([f"{p}_wheel_j{i}" for p in "BLR" for i in (1, 2)]
                   + [f"torso_j{i}" for i in (1, 2, 3)]
                   + [f"head_j{i}" for i in (1, 2, 3)])
        lock = [self.full.getJointId(n) for n in non_arm
                if self.full.existJointName(n)]
        self.arm = pin.buildReducedModel(self.full, lock, pin.neutral(self.full))
        self.arm_data = self.arm.createData()
        self.full_data = self.full.createData()
        self.full_idxq = {self.full.names[j]: self.full.joints[j].idx_q
                          for j in range(1, self.full.njoints)}
        pin_names = [f"{s}_arm_j{i}" for s in "LR" for i in range(1, 8)]
        self.LO = {n: float(self.arm.lowerPositionLimit[i])
                   for i, n in enumerate(pin_names)}
        self.HI = {n: float(self.arm.upperPositionLimit[i])
                   for i, n in enumerate(pin_names)}
        self.JNAMES = {h: [f"{'R' if h == 'right' else 'L'}_arm_j{i}"
                           for i in range(1, 8)] for h in HANDS}
        self.HOME7 = {h: np.array([ARM_HOME[f"{s}_arm_j{i}"] for i in range(1, 8)])
                      for h, s in (("right", "R"), ("left", "L"))}

        # 胸(arm_center)在躯干 home 下的位姿:必须是绕 y 的纯俯仰(自检 F)
        qf = pin.neutral(self.full)
        for n, v in TORSO_HOME.items():
            qf[self.full_idxq[n]] = v
        pin.forwardKinematics(self.full, self.full_data, qf)
        pin.updateFramePlacements(self.full, self.full_data)
        chest = self.full_data.oMf[self.full.getFrameId("arm_center")]
        self.chest_in_base = pin.SE3(chest.rotation.copy(),
                                     chest.translation.copy())
        self._rv = pin.log3(chest.rotation)
        self.THETA = float(self._rv[1])          # 预期 +55° 俯仰
        self.TILT = ry(self.THETA)               # 胸系 -> 重力系
        self.TILT_T = self.TILT.T                # 重力系 -> 胸系

        # 每只手的链几何(缩减模型 q=0 的 FK,胸系坐标)
        self.GEO: dict[str, dict] = {}
        self._arm_fk(np.zeros(14))
        chest0 = self._frame("arm_center")
        for h, s in (("right", "R"), ("left", "L")):
            rel = lambda n: chest0.inverse() * self._frame(n)   # noqa: E731
            j1o = rel(f"{s}_arm_j1").translation
            j2o = rel(f"{s}_arm_j2").translation
            # 肩心 = j1 轴(方向 y)与 j2 轴(方向 z)公垂线中点
            sh = np.array([(j1o[0] + j2o[0]) / 2.0, j2o[1], j1o[2]])
            el = rel(f"{s}_arm_l4").translation                # 肘点 = j4 原点
            wr = rel(f"{s}_arm_j7").translation                # 腕点 = j7 原点
            ee = rel(f"{s}_ee")
            j3o = rel(f"{s}_arm_j3").translation
            j5o = rel(f"{s}_arm_j5").translation
            self.GEO[h] = {
                "SH": sh, "C": ee.rotation.copy(),
                "t23": j3o - j2o, "t34": el - j3o, "SHl2": sh - j2o,
                "t45": j5o - el, "t57": wr - j5o,
                "Lu": float(np.linalg.norm(el - sh)),
                "Lf": float(np.linalg.norm(wr - el)),
            }
        self.prev = {h: self.HOME7[h].copy() for h in HANDS}

    # ---------------------------------------------------------------- FK
    def _arm_fk(self, q14):
        pin.forwardKinematics(self.arm, self.arm_data, np.asarray(q14, float))
        pin.updateFramePlacements(self.arm, self.arm_data)

    def _frame(self, name) -> pin.SE3:
        return self.arm_data.oMf[self.arm.getFrameId(name)]

    @staticmethod
    def q14(qr, ql):
        return np.concatenate([ql, qr])   # 缩减模型的顺序是 L1..7, R1..7

    def fk_points_chest(self, q_right, q_left) -> dict[str, dict]:
        """两臂执行角 -> 胸系下肩/肘/腕/EE 点与上臂/前臂方向(预览与量尺用)。"""
        self._arm_fk(self.q14(q_right, q_left))
        chest = self._frame("arm_center")
        out = {}
        for h, s in (("right", "R"), ("left", "L")):
            rel = lambda n: chest.inverse() * self._frame(n)   # noqa: E731
            el = rel(f"{s}_arm_l4").translation
            wr = rel(f"{s}_arm_j7").translation
            ee = rel(f"{s}_ee")
            sh = self.GEO[h]["SH"]
            out[h] = {"SH": sh, "EL": el, "WR": wr, "EE": ee.translation,
                      "R_ee": ee.rotation,
                      "u": (el - sh) / max(np.linalg.norm(el - sh), 1e-9),
                      "f": (wr - el) / max(np.linalg.norm(wr - el), 1e-9)}
        return out

    # ---------------------------------------------------------------- 分解
    def decompose(self, h, u_g, f_g, R_ee_g, prev):
        """重力系输入 -> (q7, flags)。q7[5] 存 β(j6 需求);执行时置 0。

        u_g/f_g: 上臂/前臂单位方向;R_ee_g: 腕朝向需求(R_wrist@PALM_FIX);
        prev: 上一帧 q7(解族连续性 + 曲柄补偿的 j3/j5)。
        """
        e = EPS[h]
        G = self.GEO[h]
        nm = self.JNAMES[h]
        LO, HI = self.LO, self.HI
        # 曲柄补偿:肘/腕点的静止方向随上一帧 j3/j5 旋转(偏心 7.4/4.4cm)
        d = G["t23"] + rx(prev[2]) @ G["t34"] - G["SHl2"]
        d = d / np.linalg.norm(d)
        ee_ = G["t45"] + rx(prev[4]) @ G["t57"]
        ee_ = ee_ / np.linalg.norm(ee_)
        u = self.TILT_T @ u_g
        f = self.TILT_T @ f_g
        Ree = self.TILT_T @ R_ee_g
        f_sh = bool(np.hypot(u[0], u[2]) < SIN_DEAD_SH)
        # (j1,j2): u = Ry(e·j1)·Rz(j2)·d;y 分量只含 j2
        sol = _solve_ab(d[1], d[0], u[1])
        if sol is None:
            return None, (f_sh, False, False, True)
        cands = []
        for j2 in sol:
            g = rz(j2) @ d
            a = np.angle((g[0] + 1j * g[2]) * np.conj(u[0] + 1j * u[2]))
            cands.append((wrap(e * a), j2))
        j1, j2 = self._pick(cands, (prev[0], prev[1]),
                            [(LO[nm[0]], HI[nm[0]]), (LO[nm[1]], HI[nm[1]])])
        R_l2 = ry(e * j1) @ rz(j2)
        # (j3,j4): v = Rx(j3)·Ry(j4)·e_;x 分量只含 j4
        v = R_l2.T @ f
        f_el = bool(np.hypot(v[1], v[2]) < SIN_DEAD_EL)
        sol = _solve_ab(ee_[0], ee_[2], v[0])
        if sol is None:
            return None, (f_sh, f_el, False, True)
        cands = []
        for j4 in sol:
            w = ry(j4) @ ee_
            j3 = np.angle((v[1] + 1j * v[2]) * np.conj(w[1] + 1j * w[2]))
            cands.append((wrap(j3), j4))
        if f_el:
            # 肘伸直时上臂自旋不可观测:j3 保持上一帧,避免 atan2 噪声
            # 把肘点甩到 7.4cm 圆的任意相位上
            cands = [(prev[2], b) for _, b in cands]
        j3, j4 = self._pick(cands, (prev[2], prev[3]),
                            [(LO[nm[2]], HI[nm[2]]), (LO[nm[3]], HI[nm[3]])])
        R_l4 = R_l2 @ rx(j3) @ ry(j4)
        # (j5,β,j7): 腕残差的 Rx·Ry·Rz 欧拉分解,β 即 j6 的需求
        H = R_l4.T @ Ree @ G["C"].T
        a, b, c = euler_xyz(H)
        f_wr = bool(abs(b) > BETA_DEAD)
        return np.array([j1, j2, j3, j4, a, e * b, c]), (f_sh, f_el, f_wr, False)

    @staticmethod
    def _pick(cands, prev, lims):
        """双解族选择:连续性 + 限位内优先(限位外记大罚分)。"""
        def score(p):
            s = abs(wrap(p[0] - prev[0])) + abs(wrap(p[1] - prev[1]))
            for v, (lo, hi) in zip(p, lims):
                if v < lo or v > hi:
                    s += 3.0
            return s
        return min(cands, key=score)

    # ---------------------------------------------------------------- 流式接口
    def reset(self):
        self.prev = {h: self.HOME7[h].copy() for h in HANDS}

    def chain_R(self, h, q7):
        """闭式链:胸系下 l2 / l4 / ee 的姿态(自检 A 与 pinocchio 对拍)。"""
        e = EPS[h]
        R_l2 = ry(e * q7[0]) @ rz(q7[1])
        R_l4 = R_l2 @ rx(q7[2]) @ ry(q7[3])
        R_ee = R_l4 @ rx(q7[4]) @ ry(e * q7[5]) @ rz(q7[6]) @ self.GEO[h]["C"]
        return R_l2, R_l4, R_ee

    def map_hand(self, h, u_g, f_g, R_ee_g) -> J2JOut:
        """一只手一帧:更新内部 prev,返回执行角(j6=0)。

        执行式后处理(decompose 的分析数学之外、真机可执行性所需的三步,
        预览台 2026-08-11 抓出的问题):
        1. **向上一帧解绕** —— decompose 的角度天然折叠在 ±180°,而 j1/j3/j5
           量程 ±176°;不解绕的话需求经过 ±180° 时会出现 358° 的假跳变
           (实测帧 179.9°→-178.2°,真实动作只有 1.9°);
        2. **限位钳制** —— 目标永不越限;
        3. **逐关节限速(默认 2.9°/帧,swivel 跟随步的同款钳)** —— 连续代表
           越限而量程内代表在缺口对侧时(±176° 的对侧姿态只差 8°,但关节
           要走 352°),朝它限速回绕,不瞬移也不在墙上永久滞留(贴墙停住
           在拼接素材上实测把右臂臂形 p95 拖到 134°、顶限位 38%);素材
           拼接边界的目标突跳同样被此钳平滑穿过(与 curobo 台架同语义)。
        """
        q_, fl = self.decompose(h, u_g, f_g, R_ee_g, self.prev[h])
        if q_ is None:
            return J2JOut(None, fl, 0.0, u_g, f_g)
        beta = float(q_[5])
        q_exec = q_.copy()
        q_exec[5] = 0.0            # J2J 执行:j6 固定 0
        for k, nm in enumerate(self.JNAMES[h]):
            if k == 5:
                continue
            lo, hi = self.LO[nm], self.HI[nm]
            raw = q_exec[k]
            tgt = float(np.clip(raw, lo, hi))          # 量程内(绕圈)代表
            c = self.prev[h][k] + wrap(raw - self.prev[h][k])
            if lo <= c <= hi:                          # 连续代表可用则优先
                tgt = c
            step = np.clip(tgt - self.prev[h][k],
                           -self.max_step, self.max_step)
            q_exec[k] = float(self.prev[h][k] + step)
        self.prev[h] = q_exec.copy()   # prev = 执行角(曲柄补偿要真实构型)
        self.prev[h][5] = beta         # β 原样保留在 prev(不参与曲柄/连续性)
        return J2JOut(q_exec, fl, beta, u_g, f_g)

    def map_body24(self, use: dict[int, np.ndarray]) -> dict[str, J2JOut]:
        """一帧 body24(已过闸)-> 两只手的执行角。

        use = {9: pose7, 16:…, 17:…, 18:…, 19:…, 20:…, 21:…},pose7 =
        [x,y,z,qx,qy,qz,qw](PICO 原始系);内部用腰去偏航重力系。
        """
        ref = use[BODY["waist"]]
        out: dict[str, J2JOut] = {}
        for h, si, ei, wi in (("left", 16, 18, 20), ("right", 17, 19, 21)):
            S = self._pt(use[si], ref)
            E = self._pt(use[ei], ref)
            W, Rw = self._pose(use[wi], ref)
            u = E - S
            f = W - E
            nu, nf = np.linalg.norm(u), np.linalg.norm(f)
            if nu < 1e-6 or nf < 1e-6:
                out[h] = J2JOut(None, (False, False, False, True), 0.0,
                                S=S, E=E, W=W)
                continue
            r = self.map_hand(h, u / nu, f / nf, Rw @ PALM_FIX[h])
            r.S, r.E, r.W = S, E, W
            out[h] = r
        return out

    @staticmethod
    def _pt(pose7, ref7):
        return process_xr_pose(np.asarray(pose7, float),
                               np.asarray(ref7, float))[:3, 3]

    @staticmethod
    def _pose(pose7, ref7):
        T = process_xr_pose(np.asarray(pose7, float), np.asarray(ref7, float))
        return T[:3, 3], T[:3, :3]


# ---------------------------------------------------------------- 自检 A-D,F
def self_check(mapper: J2JMapper | None = None, verbose=True) -> list[str]:
    """尺子先过已知答案(与 dev_j2j_feasibility.py 的自检同源)。返回失败名单。"""
    m = mapper or J2JMapper()
    failed: list[str] = []

    def check(ok: bool, name: str, detail: str):
        if verbose:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failed.append(name)

    rng = np.random.default_rng(0)
    LO, HI, JN, HOME7 = m.LO, m.HI, m.JNAMES, m.HOME7
    # A. 闭式链 vs pinocchio FK
    worst_r = 0.0
    for _ in range(25):
        q = {h: np.array([rng.uniform(LO[n], HI[n]) for n in JN[h]])
             for h in HANDS}
        m._arm_fk(m.q14(q["right"], q["left"]))
        chest = m._frame("arm_center")
        for h, s in (("right", "R"), ("left", "L")):
            R_l2, R_l4, R_ee = m.chain_R(h, q[h])
            for name, Rm in ((f"{s}_arm_l2", R_l2), (f"{s}_arm_l4", R_l4),
                             (f"{s}_ee", R_ee)):
                Rf = (chest.inverse() * m._frame(name)).rotation
                worst_r = max(worst_r, ang_of(Rm.T @ Rf))
    check(worst_r < 1e-5, "A 闭式链=pinocchio",
          f"25 组随机构型 l2/l4/ee 姿态最大偏差 {worst_r:.2e} rad")
    # B1. 腕欧拉分解回代
    worst = 0.0
    for _ in range(200):
        a = rng.uniform(-3.0, 3.0)
        b = rng.uniform(-1.3, 1.3)
        c = rng.uniform(-1.1, 1.1)
        aa, bb, cc = euler_xyz(rx(a) @ ry(b) @ rz(c))
        worst = max(worst, abs(aa - a), abs(bb - b), abs(cc - c))
    check(worst < 1e-9, "B1 腕欧拉分解回代", f"200 组恢复误差 {worst:.1e} rad")
    # B2. 全链功能往返(随机 q -> 合成人体输入 -> 分解 -> FK 还原方向/朝向)
    e_el, e_fa, e_or = [], [], []
    n_ok = 0
    for _ in range(400):
        h = "right" if rng.uniform() < 0.5 else "left"
        q = np.array([rng.uniform(LO[n] + 0.05, HI[n] - 0.05) for n in JN[h]])
        m._arm_fk(m.q14(q if h == "right" else HOME7["right"],
                        q if h == "left" else HOME7["left"]))
        chest = m._frame("arm_center")
        s = "R" if h == "right" else "L"
        rel = lambda n: chest.inverse() * m._frame(n)   # noqa: E731
        el = rel(f"{s}_arm_l4").translation
        wr = rel(f"{s}_arm_j7").translation
        u = el - m.GEO[h]["SH"]
        f = wr - el
        un = u / np.linalg.norm(u)
        if np.hypot(un[0], un[2]) < 0.35 or np.linalg.norm(f) < 1e-6:
            continue
        u_g = m.TILT @ un
        f_g = m.TILT @ (f / np.linalg.norm(f))
        Ree_g = m.TILT @ rel(f"{s}_ee").rotation
        got = HOME7[h]
        for _i in range(3):
            got2, fl = m.decompose(h, u_g, f_g, Ree_g, got)
            if got2 is None:
                break
            got = got2
        if got2 is None or fl[0] or fl[1] or fl[2]:
            continue
        n_ok += 1
        m._arm_fk(m.q14(got if h == "right" else HOME7["right"],
                        got if h == "left" else HOME7["left"]))
        chest = m._frame("arm_center")
        e_or.append(D2(ang_of(rel(f"{s}_ee").rotation.T @ (m.TILT_T @ Ree_g))))
        gz = got.copy()
        gz[5] = 0.0
        m._arm_fk(m.q14(gz if h == "right" else HOME7["right"],
                        gz if h == "left" else HOME7["left"]))
        chest = m._frame("arm_center")
        el2 = rel(f"{s}_arm_l4").translation
        wr2 = rel(f"{s}_arm_j7").translation
        u2 = el2 - m.GEO[h]["SH"]
        f2 = wr2 - el2
        e_el.append(D2(np.arccos(np.clip(
            u2 @ (m.TILT_T @ u_g) / np.linalg.norm(u2), -1, 1))))
        e_fa.append(D2(np.arccos(np.clip(
            f2 @ (m.TILT_T @ f_g) / np.linalg.norm(f2), -1, 1))))
    p_el, p_fa, p_or = (np.percentile(v, 95) for v in (e_el, e_fa, e_or))
    m_el, m_fa = np.median(e_el), np.median(e_fa)
    check(n_ok > 250 and m_el < 5.0 and m_fa < 5.0
          and p_el < 15.0 and p_fa < 15.0 and p_or < 0.01,
          "B2 全链功能往返",
          f"{n_ok} 组:上臂 p50 {m_el:.1f}°/p95 {p_el:.1f}°,前臂 p50 "
          f"{m_fa:.1f}°/p95 {p_fa:.1f}°,朝向 {p_or:.1e}°")
    # C. 已知答案:垂臂 -> 肘死区被标出 + 上臂指向重力向下
    for h in HANDS:
        q_, fl = m.decompose(h, np.array([0, 0, -1.0]), np.array([0, 0, -1.0]),
                             m.TILT @ m.GEO[h]["C"], HOME7[h])
        check(fl[1], f"C1 {h} 垂臂肘死区", "肘伸直时 j3 不可观测,已标死区")
        qz = q_.copy()
        qz[5] = 0.0
        m._arm_fk(m.q14(qz if h == "right" else HOME7["right"],
                        qz if h == "left" else HOME7["left"]))
        chest = m._frame("arm_center")
        s = "R" if h == "right" else "L"
        el = (chest.inverse() * m._frame(f"{s}_arm_l4")).translation
        u_g = m.TILT @ (el - m.GEO[h]["SH"])
        ang = D2(np.arccos(np.clip(-u_g[2] / np.linalg.norm(u_g), -1, 1)))
        check(ang < 10.0, f"C2 {h} 垂臂方向", f"上臂与重力向下夹角 {ang:.1f}°")
    # D. j6 语义:转 j6 只动指向、不动掌法线
    for h in HANDS:
        qz = HOME7[h].copy()
        qz[4] = qz[5] = qz[6] = 0.0
        _, _, R0 = m.chain_R(h, qz)
        qz6 = qz.copy()
        qz6[5] = 0.5
        _, _, R6 = m.chain_R(h, qz6)
        n_move = D2(np.arccos(np.clip(R0[:, 0] @ R6[:, 0], -1, 1)))
        f_move = D2(np.arccos(np.clip(R0[:, 2] @ R6[:, 2], -1, 1)))
        check(n_move < 0.01 and abs(f_move - D2(0.5)) < 0.01,
              f"D {h} j6 轴语义",
              f"掌法线动 {n_move:.4f}°,指向动 {f_move:.1f}°")
    # F. 胸姿态 = 纯俯仰
    off = D2(abs(m._rv[0]) + abs(m._rv[2]))
    check(off < 0.01 and abs(D2(m.THETA) - 55.0) < 0.5, "F 胸=纯俯仰",
          f"arm_center = 绕 y 俯仰 {D2(m.THETA):.2f}°,其他轴残差 {off:.4f}°")
    return failed


if __name__ == "__main__":
    import sys
    print("== J2J 模块自检(尺子先过已知答案)==")
    bad = self_check()
    print("自检:" + ("全部通过 ✅" if not bad else "失败 ❌ " + ", ".join(bad)))
    sys.exit(1 if bad else 0)
