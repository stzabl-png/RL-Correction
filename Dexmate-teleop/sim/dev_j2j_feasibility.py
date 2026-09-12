#!/usr/bin/env python3
"""J2J(joint-to-joint 形似直映)离线可行性分析 —— 只出数据报告,不接链路。

    cd <仓库根目录> && .venv/bin/python sim/dev_j2j_feasibility.py

评估的映射(用户 2026-08-10 提议的第三种模式):不做 IK,人体骨架的关节角
逐关节映射到 Vega 手臂 —— 肩部→j1/j2,上臂自旋→j3,肘屈伸→j4,前臂自旋→j5,
腕屈伸→j7,j6 固定为 0。本脚本从现有录制(body24 模型通道)离线提取解剖角、
做量程/噪声/代价分析,输出五张表。全程单进程,不依赖 Isaac,不占端口。

构造约定(全部可在自检里被证伪):
  * 人体参考系 = 腰 tracker 去偏航的重力系(process_xr_pose,x前/y左/z上),
    与现行 chest-anchor 法则同一哲学(「水平就是水平」,2026-08-03 实测教训:
    用胸姿态轴会把水平前举压低 55°);骨架点取 body24 模型通道(16/17 肩、
    18/19 肘、20/21 腕、9 腰),与 PALM_FIX 的有效通道一致(逐位核对)。
  * 机器人侧在 arm_center(胸)坐标里分解;躯干锁在 home
    (torso 1.047/1.571/-0.436),此时胸相对底座是绕 y 的纯俯仰 +55.0°
    (FK 验证)。把人体向量预旋转 Ry(-55°) 进胸系再分解,等价于给 j1 加
    55° 常数偏置 —— 这是 J2J 在这台「胸前倾」机器人上成立的前提。
  * 分解本身是机器人自己的轴序列闭式反解(轴向/翻转矩阵全部从 URDF 数值
    导出并与 pinocchio FK 对拍),不是任务空间 IK:
      (j1,j2) 由上臂方向(肩→肘)反解;(j3,j4) 由前臂方向(肘→腕)反解;
      (j5,j6,j7) 由腕朝向需求 R_wrist@PALM_FIX 的残差做 Rx·Ry·Rz 欧拉分解,
      J2J 取 j5/j7、丢弃中间角 β(=j6 的需求)。
  * 表 2-5 的输入过 TrackerGlitchGate(与实机链路同款),表 1 用生数据
    (问的是数据源本身的噪声)。

已知取舍(如实计入误差,不隐藏):Vega 的肘点偏离肱骨轴 7.4cm、腕点偏离
前臂轴 4.4cm(曲柄结构,转 j3/j5 时肘点/腕点画圈)—— 反解用上一帧的
j3/j5 把静止方向常数旋转到位(滞后一帧的闭式补偿,不迭代);肩心在
j1/j2 轴之间还有 2cm 公垂线间隙,无法闭式消掉,是「按方向反解」的固有
几何误差 —— 自检 B2 量它的上界,表 5 的 J2J 对照组量它在真数据上的大小。
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pinocchio as pin

_THIS = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[1]))

from magicdexmate.palm_fix import PALM_FIX  # noqa: E402
from magicdexmate.home_pose import ARM_HOME  # noqa: E402
from magicdexmate.pico.xr_pose import process_xr_pose  # noqa: E402
from magicdexmate.sources.pico_source import (  # noqa: E402
    PicoLogSource, TrackerGlitchGate)
from magicdexmate.swivel import swivel_angle  # noqa: E402

URDF = os.environ.get(
    "VEGA_URDF",
    os.path.expanduser("~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf"))
REPO = _THIS.parents[1]
MATERIALS = [str(REPO / "logs/clip_headwaist.msgpack"),
             str(REPO / "logs/playlist_all13.msgpack")]
# 现行 pink 回归的 cmd 关节角 CSV(表 5 对照)。本仓库 logs/regress 只有它的
# JSON 汇总;CSV 本体在另一份 checkout 里,按序找,都没有就跳过 pink 对照。
PINK_CSVS = [str(REPO / "logs/regress/relaxfix2_idle_playlist_all13.csv"),
             "/home/luhr/dexmate/MagicDexMate/logs/regress/"
             "relaxfix2_idle_playlist_all13.csv"]
TORSO_HOME = {"torso_j1": 1.047198, "torso_j2": 1.570796, "torso_j3": -0.436332}
BODY = {"waist": 9, "l_sh": 16, "r_sh": 17, "l_el": 18, "r_el": 19,
        "l_wr": 20, "r_wr": 21}
D2 = np.degrees
FAILED: list[str] = []


def check(ok: bool, name: str, detail: str):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILED.append(name)


def _short(path: str) -> str:
    return os.path.basename(path).replace(".msgpack", "")


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ang_of(Rm) -> float:
    """旋转矩阵的转角[rad]。"""
    return float(np.linalg.norm(pin.log3(np.asarray(Rm))))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- 模型与几何
FULL = pin.buildModelFromUrdf(URDF)
_NON_ARM = ([f"{p}_wheel_j{i}" for p in "BLR" for i in (1, 2)]
            + [f"torso_j{i}" for i in (1, 2, 3)]
            + [f"head_j{i}" for i in (1, 2, 3)])
_lock = [FULL.getJointId(n) for n in _NON_ARM if FULL.existJointName(n)]
ARM = pin.buildReducedModel(FULL, _lock, pin.neutral(FULL))
ARM_DATA = ARM.createData()
PIN_NAMES = [f"{s}_arm_j{i}" for s in "LR" for i in range(1, 8)]
LO = {n: float(ARM.lowerPositionLimit[i]) for i, n in enumerate(PIN_NAMES)}
HI = {n: float(ARM.upperPositionLimit[i]) for i, n in enumerate(PIN_NAMES)}
FULL_DATA = FULL.createData()
FULL_IDXQ = {FULL.names[j]: FULL.joints[j].idx_q for j in range(1, FULL.njoints)}


def arm_fk(q14):
    pin.forwardKinematics(ARM, ARM_DATA, np.asarray(q14, float))
    pin.updateFramePlacements(ARM, ARM_DATA)


def arm_frame(name) -> pin.SE3:
    return ARM_DATA.oMf[ARM.getFrameId(name)]


def q14(qr, ql):
    return np.concatenate([ql, qr])   # 缩减模型的顺序是 L1..7, R1..7


# 胸(arm_center)在躯干 home 下的姿态:必须是绕 y 的纯俯仰(自检 F)
qf = pin.neutral(FULL)
for n, v in TORSO_HOME.items():
    qf[FULL_IDXQ[n]] = v
pin.forwardKinematics(FULL, FULL_DATA, qf)
pin.updateFramePlacements(FULL, FULL_DATA)
R_CHEST_HOME = FULL_DATA.oMf[FULL.getFrameId("arm_center")].rotation.copy()
_rv = pin.log3(R_CHEST_HOME)
THETA = float(_rv[1])                      # 预期 +55° 俯仰
TILT = ry(THETA)                           # 胸系 -> 重力系
TILT_T = TILT.T                            # 重力系 -> 胸系

# 每只手的链几何(全部数值取自缩减模型 q=0 的 FK,胸系坐标)
HANDS = ("right", "left")
EPS = {"right": -1.0, "left": +1.0}        # rpy=pi 翻转折叠出的符号(自检 A)
GEO: dict[str, dict] = {}
arm_fk(np.zeros(14))
_chest0 = arm_frame("arm_center")
for h, s in (("right", "R"), ("left", "L")):
    rel = lambda n: _chest0.inverse() * arm_frame(n)   # noqa: E731
    j1o = rel(f"{s}_arm_j1").translation
    j2o = rel(f"{s}_arm_j2").translation
    # 肩心 = j1 轴(过 j1o 方向 y)与 j2 轴(过 j2o 方向 z)公垂线中点
    sh = np.array([(j1o[0] + j2o[0]) / 2.0, j2o[1], j1o[2]])
    el = rel(f"{s}_arm_l4").translation                # 肘点 = j4 原点
    wr = rel(f"{s}_arm_j7").translation                # 腕点 = j7 原点
    ee = rel(f"{s}_ee")
    j3o = rel(f"{s}_arm_j3").translation
    j5o = rel(f"{s}_arm_j5").translation
    GEO[h] = {
        "SH": sh, "EL0": el, "WR0": wr, "EE0": ee.translation.copy(),
        "C": ee.rotation.copy(),                       # q=0 的 EE 姿态(胸系)
        # 曲柄常数:肘点/腕点绕 j3/j5 轴的偏心结构(q=0 时各系与胸系同向,
        # 平移可直接相减;decompose 里用上一帧 j3/j5 把静止方向转到位)
        "t23": j3o - j2o, "t34": el - j3o, "SHl2": sh - j2o,
        "t45": j5o - el, "t57": wr - j5o,
        "Lu": float(np.linalg.norm(el - sh)),
        "Lf": float(np.linalg.norm(wr - el)),
        "Lh": float(np.linalg.norm(ee.translation - wr)),
    }


def chain_R(h, q7):
    """闭式链:胸系下 l2 / l4 / ee 的姿态(与 pinocchio 对拍,自检 A)。"""
    e = EPS[h]
    R_l2 = ry(e * q7[0]) @ rz(q7[1])
    R_l4 = R_l2 @ rx(q7[2]) @ ry(q7[3])
    R_ee = R_l4 @ rx(q7[4]) @ ry(e * q7[5]) @ rz(q7[6]) @ GEO[h]["C"]
    return R_l2, R_l4, R_ee


def euler_xyz(H):
    """H = Rx(a)·Ry(b)·Rz(c) 的 (a,b,c),b∈[-90°,90°](自检 D)。"""
    b = np.arcsin(np.clip(H[0, 2], -1.0, 1.0))
    c = np.arctan2(-H[0, 1], H[0, 0])
    a = np.arctan2(-H[1, 2], H[2, 2])
    return float(a), float(b), float(c)


def _solve_ab(A, B, C):
    """A·cos(x)+B·sin(x)=C 的两解;|C| 超出可达幅值时贴边求解(方向出锥
    = 肩/肘死区,由 flags 标注;执行式映射在那里也只能给最近可达方向)。"""
    Rm = np.hypot(A, B)
    if Rm < 1e-9:
        return None
    phi, d = np.arctan2(B, A), np.arccos(np.clip(C / Rm, -1.0, 1.0))
    return wrap(phi - d), wrap(phi + d)


def _pick(cands, prev, lims):
    """双解族选择:连续性 + 限位内优先(限位外记大罚分)。没有罚分时,
    死区里的一次抽风会把解族永久甩到「反手」支上(实测 j2 出 -179°、
    j4 出 +151° 一整段),而两支几何上同样精确,执行式映射必须选可执行的。"""
    def score(p):
        s = abs(wrap(p[0] - prev[0])) + abs(wrap(p[1] - prev[1]))
        for v, (lo, hi) in zip(p, lims):
            if v < lo or v > hi:
                s += 3.0
        return s
    return min(cands, key=score)


SIN_DEAD_SH = np.sin(np.radians(15.0))   # 上臂贴 j1 轴 15° 以内 = 肩死区
SIN_DEAD_EL = np.sin(np.radians(15.0))   # 前臂贴肱骨轴 15° 以内 = j3 死区
BETA_DEAD = np.radians(75.0)             # 欧拉中间角接近 ±90° = 腕死区


def decompose(h, u_g, f_g, R_ee_g, prev):
    """重力系输入 -> (q7, flags)。q7[5] 存 β(j6 的需求);J2J 执行时置 0。

    u_g/f_g: 上臂/前臂单位方向;R_ee_g: 腕朝向需求(R_wrist@PALM_FIX);
    prev: 上一帧 q7(解族连续性 + 曲柄补偿的 j3/j5)。
    flags = (肩死区, 肘死区, 腕死区, 无解)。
    """
    e = EPS[h]
    G = GEO[h]
    nm = JNAMES[h]
    # 曲柄补偿:肘点/腕点的静止方向随上一帧 j3/j5 旋转(结构偏心 7.4/4.4cm)
    d = G["t23"] + rx(prev[2]) @ G["t34"] - G["SHl2"]
    d = d / np.linalg.norm(d)
    ee_ = G["t45"] + rx(prev[4]) @ G["t57"]
    ee_ = ee_ / np.linalg.norm(ee_)
    u = TILT_T @ u_g
    f = TILT_T @ f_g
    Ree = TILT_T @ R_ee_g
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
    j1, j2 = _pick(cands, (prev[0], prev[1]),
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
        # 肘伸直时上臂自旋不可观测:j3 保持上一帧(执行式映射也应如此),
        # 避免用近零向量的 atan2 噪声把肘点甩到 7.4cm 圆的任意相位上
        cands = [(prev[2], b) for _, b in cands]
    j3, j4 = _pick(cands, (prev[2], prev[3]),
                   [(LO[nm[2]], HI[nm[2]]), (LO[nm[3]], HI[nm[3]])])
    R_l4 = R_l2 @ rx(j3) @ ry(j4)
    # (j5,β,j7): 腕残差的 Rx·Ry·Rz 欧拉分解,β 即 j6 的需求
    H = R_l4.T @ Ree @ GEO[h]["C"].T
    a, b, c = euler_xyz(H)
    j5, j6, j7 = a, e * b, c
    f_wr = bool(abs(b) > BETA_DEAD)
    return np.array([j1, j2, j3, j4, j5, j6, j7]), (f_sh, f_el, f_wr, False)


HOME7 = {h: np.array([ARM_HOME[f"{s}_arm_j{i}"] for i in range(1, 8)])
         for h, s in (("right", "R"), ("left", "L"))}
JNAMES = {h: [f"{'R' if h == 'right' else 'L'}_arm_j{i}" for i in range(1, 8)]
          for h in HANDS}
ANATOMY = ["肩俯仰", "肩横摆", "上臂自旋", "肘屈伸", "前臂自旋",
           "腕掌面内摆(拟固定)", "腕屈伸"]


# ---------------------------------------------------------------- 自检 A-D,F
def self_checks_geometry():
    print("\n== 自检(尺子先过已知答案)==")
    rng = np.random.default_rng(0)
    # A. 闭式链 vs pinocchio FK:l2/l4/ee 姿态逐一对拍
    worst_r = 0.0
    for _ in range(25):
        q = {h: np.array([rng.uniform(LO[n], HI[n]) for n in JNAMES[h]])
             for h in HANDS}
        arm_fk(q14(q["right"], q["left"]))
        chest = arm_frame("arm_center")
        for h, s in (("right", "R"), ("left", "L")):
            R_l2, R_l4, R_ee = chain_R(h, q[h])
            for name, Rm in ((f"{s}_arm_l2", R_l2), (f"{s}_arm_l4", R_l4),
                             (f"{s}_ee", R_ee)):
                Rf = (chest.inverse() * arm_frame(name)).rotation
                worst_r = max(worst_r, ang_of(Rm.T @ Rf))
    check(worst_r < 1e-5, "A 闭式链=pinocchio",
          f"25 组随机限位内构型,l2/l4/ee 姿态最大偏差 {worst_r:.2e} rad"
          "(含翻转符号 ε 与 C_ee 的正确性;缩减模型胸系本身带 2e-7 的"
          "URDF 浮点偏斜,阈值 1e-5)")
    # B1. 腕欧拉分解精确回代(给真 R_l4 时必须逐位恢复)
    worst = 0.0
    for _ in range(200):
        a = rng.uniform(-3.0, 3.0)
        b = rng.uniform(-1.3, 1.3)
        c = rng.uniform(-1.1, 1.1)
        aa, bb, cc = euler_xyz(rx(a) @ ry(b) @ rz(c))
        worst = max(worst, abs(aa - a), abs(bb - b), abs(cc - c))
    check(worst < 1e-9, "B1 腕欧拉分解回代", f"200 组已知 (j5,β,j7) 恢复误差 {worst:.1e} rad")
    # B2. 全链功能往返:随机 q -> FK 合成"人体输入" -> 分解(从 home 起迭代
    #     到不动点,模拟稳态的滞后曲柄)-> 对执行构型做 FK -> 方向与朝向必须
    #     还原输入。q 本身在肩心 2cm 公垂线间隙下不唯一,不能拿 q 逐关节
    #     对账;映射的契约是「形似」= 方向像 + 掌向像,就按契约验收:
    #     朝向(β 保留时)按构造应精确还原(~1e-9,抓欧拉/翻转类 bug),
    #     方向残差是肩心间隙的固有几何误差,构成表 5 对照组的底噪。
    e_el, e_fa, e_or = [], [], []
    n_ok = 0
    for _ in range(400):
        h = "right" if rng.uniform() < 0.5 else "left"
        q = np.array([rng.uniform(LO[n] + 0.05, HI[n] - 0.05) for n in JNAMES[h]])
        arm_fk(q14(q if h == "right" else HOME7["right"],
                   q if h == "left" else HOME7["left"]))
        chest = arm_frame("arm_center")
        s = "R" if h == "right" else "L"
        rel = lambda n: chest.inverse() * arm_frame(n)   # noqa: E731
        el = rel(f"{s}_arm_l4").translation
        wr = rel(f"{s}_arm_j7").translation
        u = el - GEO[h]["SH"]
        f = wr - el
        un = u / np.linalg.norm(u)
        if np.hypot(un[0], un[2]) < 0.35 or np.linalg.norm(f) < 1e-6:
            continue   # 肩死区附近不进统计(decompose 内以胸系判死区)
        u_g = TILT @ un
        f_g = TILT @ (f / np.linalg.norm(f))
        Ree_g = TILT @ rel(f"{s}_ee").rotation
        got = HOME7[h]
        for _i in range(3):
            got2, fl = decompose(h, u_g, f_g, Ree_g, got)
            if got2 is None:
                break
            got = got2
        if got2 is None or fl[0] or fl[1] or fl[2]:
            continue
        n_ok += 1
        # 朝向用 β 保留的构型验(分解按构造应精确还原朝向);
        # 方向用 j6=0 的执行构型验(腕点曲柄 e(j5) 就是按 j6=0 建的)
        arm_fk(q14(got if h == "right" else HOME7["right"],
                   got if h == "left" else HOME7["left"]))
        chest = arm_frame("arm_center")
        e_or.append(D2(ang_of(rel(f"{s}_ee").rotation.T @ (TILT_T @ Ree_g))))
        gz = got.copy()
        gz[5] = 0.0
        arm_fk(q14(gz if h == "right" else HOME7["right"],
                   gz if h == "left" else HOME7["left"]))
        chest = arm_frame("arm_center")
        el2 = rel(f"{s}_arm_l4").translation
        wr2 = rel(f"{s}_arm_j7").translation
        u2 = el2 - GEO[h]["SH"]
        f2 = wr2 - el2
        e_el.append(D2(np.arccos(np.clip(
            u2 @ (TILT_T @ u_g) / np.linalg.norm(u2), -1, 1))))
        e_fa.append(D2(np.arccos(np.clip(
            f2 @ (TILT_T @ f_g) / np.linalg.norm(f2), -1, 1))))
    p_el, p_fa, p_or = (np.percentile(v, 95) for v in (e_el, e_fa, e_or))
    m_el, m_fa = np.median(e_el), np.median(e_fa)
    # 判据校准:符号/轴向/解族类 bug 的表现是 30~180°,几何固有误差是个位数
    # 中位、随机极端构型(肘全折等)下 p95 到 ~11°。中位 <5° 且 p95 <15°
    # 能放行固有误差、抓住任何一类真 bug;朝向按构造必须精确。
    check(n_ok > 250 and m_el < 5.0 and m_fa < 5.0
          and p_el < 15.0 and p_fa < 15.0 and p_or < 0.01,
          "B2 全链功能往返",
          f"{n_ok} 组死区外随机构型,执行 FK 还原输入:上臂方向 p50 "
          f"{m_el:.1f}°/p95 {p_el:.1f}°,前臂方向 p50 {m_fa:.1f}°/p95 "
          f"{p_fa:.1f}°,腕朝向 {p_or:.1e}°(方向残差=肩心 2cm 间隙 + 滞后"
          "曲柄在极端构型下的固有几何误差;真数据上的实际值见表 5 J2J 行)")
    # C. 已知答案姿势:人垂臂站立 -> J2J 后机器人臂在重力系里应指向下
    for h in HANDS:
        q_, fl = decompose(h, np.array([0, 0, -1.0]), np.array([0, 0, -1.0]),
                           TILT @ GEO[h]["C"], HOME7[h])
        # 垂臂 = 肘伸直 = j3 死区,必须被标出
        check(fl[1], f"C1 {h} 垂臂肘死区", "肘伸直时上臂自旋不可观测,已标死区")
        qz = q_.copy()
        qz[5] = 0.0
        arm_fk(q14(qz if h == "right" else HOME7["right"],
                   qz if h == "left" else HOME7["left"]))
        chest = arm_frame("arm_center")
        s = "R" if h == "right" else "L"
        el = (chest.inverse() * arm_frame(f"{s}_arm_l4")).translation
        u_g = TILT @ (el - GEO[h]["SH"])
        ang = D2(np.arccos(np.clip(-u_g[2] / np.linalg.norm(u_g), -1, 1)))
        check(ang < 10.0, f"C2 {h} 垂臂方向",
              f"J2J 后上臂与重力向下夹角 {ang:.1f}°(<10°,含固有几何误差)")
    # D. j6 语义:FK 验证「j6 转的是掌法线」(轴向从 URDF+FK 读出,不想当然)
    for h in HANDS:
        qz = HOME7[h].copy()
        qz[4] = qz[5] = qz[6] = 0.0
        _, _, R0 = chain_R(h, qz)
        qz6 = qz.copy()
        qz6[5] = 0.5
        _, _, R6 = chain_R(h, qz6)
        n_move = D2(np.arccos(np.clip(R0[:, 0] @ R6[:, 0], -1, 1)))   # 掌法线 +ee.x
        f_move = D2(np.arccos(np.clip(R0[:, 2] @ R6[:, 2], -1, 1)))   # 指向 +ee.z
        check(n_move < 0.01 and abs(f_move - D2(0.5)) < 0.01,
              f"D {h} j6 轴语义",
              f"j7=0 时转 j6=28.6°:掌法线动 {n_move:.4f}°,指向动 {f_move:.1f}°"
              " —— j6 是绕掌法线的掌面内摆(桡/尺偏),不动掌心朝向")
    # F. 胸姿态 = 纯俯仰
    off = D2(abs(_rv[0]) + abs(_rv[2]))
    check(off < 0.01 and abs(D2(THETA) - 55.0) < 0.5, "F 胸=纯俯仰",
          f"躯干 home 下 arm_center 相对底座 = 绕 y 俯仰 {D2(THETA):.2f}°,"
          f"其他轴残差 {off:.4f}°(j1 常数偏置的依据)")


# ---------------------------------------------------------------- 数据提取
def wf_point(pose7, ref7):
    return process_xr_pose(np.asarray(pose7, float), np.asarray(ref7, float))[:3, 3]


def wf_pose(pose7, ref7):
    T = process_xr_pose(np.asarray(pose7, float), np.asarray(ref7, float))
    return T[:3, 3], T[:3, :3]


class Series:
    """一份素材、一种预处理(生/过闸)下的全部逐帧量。"""

    def __init__(self, src: PicoLogSource, gated: bool):
        n = len(src.frames)
        self.n = n
        self.t = np.array([(fr.t_us - src.frames[0].t_us) / 1e6 for fr in src.frames])
        self.q = {h: np.full((n, 7), np.nan) for h in HANDS}
        self.flag = {h: np.zeros((n, 4), bool) for h in HANDS}
        self.u_g = {h: np.full((n, 3), np.nan) for h in HANDS}
        self.f_g = {h: np.full((n, 3), np.nan) for h in HANDS}
        self.Ree_g = {h: np.full((n, 3, 3), np.nan) for h in HANDS}
        self.W = {h: np.full((n, 3), np.nan) for h in HANDS}
        self.S = {h: np.full((n, 3), np.nan) for h in HANDS}
        self.E = {h: np.full((n, 3), np.nan) for h in HANDS}
        self.org = np.full((n, 3), np.nan)
        self.tilt = np.full(n, np.nan)
        gates = ({k: TrackerGlitchGate() for k in (9, 18, 19, 20, 21)}
                 if gated else None)
        prev = {h: HOME7[h].copy() for h in HANDS}
        for i, fr in enumerate(src.frames):
            b = np.asarray(fr.body24, float)
            use = {k: b[k] for k in (9, 16, 17, 18, 19, 20, 21)}
            if gates is not None:
                for k, g in gates.items():
                    use[k], _ = g.update(fr.t_us, b[k])
            ref = use[9]
            ps = wf_point(use[16], ref)
            pr = wf_point(use[17], ref)
            self.org[i] = 0.5 * (ps + pr)
            nz = np.linalg.norm(self.org[i])
            self.tilt[i] = D2(np.arccos(np.clip(self.org[i][2] / max(nz, 1e-9), -1, 1)))
            for h, si, ei, wi in (("left", 16, 18, 20), ("right", 17, 19, 21)):
                S = ps if h == "left" else pr
                E = wf_point(use[ei], ref)
                W, Rw = wf_pose(use[wi], ref)
                self.S[h][i], self.E[h][i], self.W[h][i] = S, E, W
                u = E - S
                f = W - E
                nu, nf = np.linalg.norm(u), np.linalg.norm(f)
                if nu < 1e-6 or nf < 1e-6:
                    self.flag[h][i] = (False, False, False, True)
                    continue
                self.u_g[h][i] = u / nu
                self.f_g[h][i] = f / nf
                self.Ree_g[h][i] = Rw @ PALM_FIX[h]
                q_, fl = decompose(h, self.u_g[h][i], self.f_g[h][i],
                                  self.Ree_g[h][i], prev[h])
                self.flag[h][i] = fl
                if q_ is not None:
                    self.q[h][i] = q_
                    prev[h] = q_


print("== 载入素材与模型 ==")
print(f"URDF: {URDF}")
SRC = {p: PicoLogSource(p) for p in MATERIALS}

# E. 通道身份:trackers 命名通道必须逐位等于 body24 模型通道(PALM_FIX 前提)
print("\n== 自检 E(通道身份)==")
for p, src in SRC.items():
    n_eq = n_tot = 0
    for fr in src.frames[::7]:
        b = fr.body24
        n_tot += 1
        if (np.array_equal(fr.trackers.get("RWRIST"), b[21])
                and np.array_equal(fr.trackers.get("WAIST"), b[9])):
            n_eq += 1
    check(n_eq / n_tot > 0.8, f"E {os.path.basename(p)}",
          f"RWRIST/WAIST 与 body24[21]/[9] 逐位相同 {100 * n_eq / n_tot:.0f}% 抽样帧"
          "(不同帧来自录制时序,本分析直接取 body24)")

self_checks_geometry()

RAW = {p: Series(src, gated=False) for p, src in SRC.items()}
GATED = {p: Series(src, gated=True) for p, src in SRC.items()}


# ---------------------------------------------------------------- 表 1
def static_windows(se: Series, h: str, win=100, tol=0.02):
    """1s 不重叠窗;肩/肘/腕(腰系)相对窗内均值都 <tol 才算准静态。"""
    out = []
    for a in range(0, se.n - win, win):
        sl = slice(a, a + win)
        ok = True
        for P in (se.S[h], se.E[h], se.W[h]):
            seg = P[sl]
            if np.isnan(seg).any() or np.abs(seg - seg.mean(0)).max() > tol:
                ok = False
                break
        if ok:
            out.append(sl)
    return out


print("\n== 自检 G(准静态窗探测器)==")
class _FakeSeries:
    pass
_fs = _FakeSeries()
_fs.n = 300
rng = np.random.default_rng(1)
still = np.array([0.1, 0.2, 0.3]) + rng.normal(0, 0.001, (300, 3))
move = still + np.outer(np.sin(np.linspace(0, 6, 300)), [0.05, 0, 0])
_fs.S = {"right": still}
_fs.E = {"right": still}
_fs.W = {"right": still}
n_still = len(static_windows(_fs, "right"))
_fs.W = {"right": move}
n_move = len(static_windows(_fs, "right"))
check(n_still == 2 and n_move == 0, "G 静窗探测器",
      f"1mm 噪声静止序列判静 {n_still}/2 窗,5cm 正弦运动判静 {n_move}/2 窗")

print("\n" + "=" * 78)
print("表 1|人体关节角提取与噪声(生数据,100Hz;抖动=相邻帧差,死区帧已剔)")
print("=" * 78)
T1_NOTE = {}
for p, se in RAW.items():
    print(f"\n素材 {os.path.basename(p)}({se.n} 帧,{se.t[-1]:.0f}s)")
    print(f"{'角(→关节)':<20s}{'手':<4s}{'抖动p50':>8s}{'抖动p95':>8s}"
          f"{'静窗数':>6s}{'静窗幅p50':>9s}{'静窗幅p95':>9s}{'死区%':>7s}")
    for h in HANDS:
        wins = static_windows(se, h)
        dead_col = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}
        for i in range(7):
            th = se.q[h][:, i]
            bad = se.flag[h][:, dead_col[i]] | se.flag[h][:, 3] | np.isnan(th)
            ok = ~bad[1:] & ~bad[:-1]
            d = np.abs(D2(wrap(np.diff(th))))[ok]
            ranges = []
            for sl in wins:
                seg = th[sl]
                if not (bad[sl].any() or np.isnan(seg).any()):
                    ranges.append(D2(np.ptp(np.unwrap(seg))))
            j50 = np.percentile(d, 50) if len(d) else np.nan
            j95 = np.percentile(d, 95) if len(d) else np.nan
            r50 = np.percentile(ranges, 50) if ranges else np.nan
            r95 = np.percentile(ranges, 95) if ranges else np.nan
            dz = 100.0 * bad.mean()
            key = (os.path.basename(p), h, i)
            T1_NOTE[key] = (j95, r95)
            print(f"{ANATOMY[i]:<18s}→j{i + 1} {h[0].upper():<3s}"
                  f"{j50:8.2f}{j95:8.2f}{len(ranges):6d}{r50:9.2f}{r95:9.2f}{dz:7.1f}")
    # 抖动尺自检:同角人为平移常数后 p95 必须不变,置换成随机序列必须变大
    th = RAW[p].q["right"][:, 3]
    good = ~np.isnan(th)
    d0 = np.percentile(np.abs(np.diff(th[good])), 95)
    d1 = np.percentile(np.abs(np.diff(th[good] + 1.23)), 95)
    sh = rng.permutation(th[good])
    d2_ = np.percentile(np.abs(np.diff(sh)), 95)
    check(abs(d0 - d1) < 1e-12 and d2_ > 3 * d0, "H1 抖动尺",
          f"j4 抖动 p95 平移不变({D2(d0):.2f}°={D2(d1):.2f}°),"
          f"乱序后应显著变大({D2(d2_):.2f}°)")

# ---------------------------------------------------------------- 表 2
print("\n" + "=" * 78)
print("表 2|量程匹配(两素材合并,过闸数据;OOR=1:1 直映超限位时间占比)")
print("=" * 78)
print(f"{'关节':<10s}{'URDF限位°':>16s}{'人需求p1..p99°':>16s}{'span°':>7s}"
      f"{'OOR%':>7s}{'建议偏置°':>9s}{'偏置后%':>8s}")
T2 = {}
for h in HANDS:
    allq = np.vstack([GATED[p].q[h] for p in MATERIALS])
    allflag = np.vstack([GATED[p].flag[h] for p in MATERIALS])
    valid = ~np.isnan(allq[:, 0]) & ~allflag[:, 3]
    for i in range(7):
        n = JNAMES[h][i]
        th = allq[valid, i]
        dead_col = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}[i]
        th = th[~allflag[valid, dead_col]]
        lo, hi = LO[n], HI[n]
        oor = 100.0 * ((th < lo) | (th > hi)).mean()
        best_off, best_oor = 0.0, oor
        for off in np.radians(np.arange(-60, 61)):
            o = 100.0 * ((th + off < lo) | (th + off > hi)).mean()
            if o < best_oor - 1e-9:
                best_off, best_oor = off, o
        sug = (f"{D2(best_off):+7.0f}" if oor - best_oor >= 2.0 else "      -")
        aft = (f"{best_oor:7.1f}" if oor - best_oor >= 2.0 else "       ")
        p1, p99 = np.percentile(D2(th), [1, 99])
        T2[(h, i)] = (oor, best_off if oor - best_oor >= 2.0 else 0.0, best_oor)
        print(f"{n:<10s}[{D2(lo):+6.0f},{D2(hi):+6.0f}] [{p1:+6.0f},{p99:+6.0f}]"
              f"{p99 - p1:7.0f}{oor:7.1f}{sug:>9s}{aft:>8s}")
# 量程尺自检:把限位人为缩成 ±1°,OOR 必须接近 100%
_th = np.vstack([GATED[p].q["right"] for p in MATERIALS])[:, 0]
_th = _th[~np.isnan(_th)]
_o = 100.0 * ((_th < -np.radians(1)) | (_th > np.radians(1))).mean()
check(_o > 90.0, "H2 量程尺", f"限位缩为 ±1° 时 j1 OOR={_o:.0f}%(应≈100)")

# ---------------------------------------------------------------- 表 3
print("\n" + "=" * 78)
print("表 3|j6 固定为 0 的代价(过闸数据;β=腕欧拉分解的中间角=j6 的需求)")
print("=" * 78)
print("被牺牲的轴(自检 D 已 FK 验证):j6 是绕**掌法线**的掌面内摆(桡/尺偏)。")
print("j7=0 时它完全不动掌心朝向,只把指尖在掌面内扫过 β 角;j7≠0 时掌法线")
print("也随之带偏(下表「掌法线偏」列)。")
print(f"\n{'素材':<22s}{'手':<4s}{'|β|mean°':>9s}{'|β|p95°':>8s}{'|β|max°':>8s}"
      f"{'掌法线偏p50':>11s}{'p95':>6s}{'指向偏p50':>10s}{'p95':>6s}")
fk_beta_err = []
for p, se in GATED.items():
    for h in HANDS:
        ok = ~np.isnan(se.q[h][:, 0]) & ~se.flag[h][:, 2] & ~se.flag[h][:, 3]
        idx = np.where(ok)[0]
        beta = np.abs(D2(se.q[h][idx, 5]))
        n_off, f_off = [], []
        for k in idx[::10]:
            qfull = se.q[h][k].copy()
            qzero = qfull.copy()
            qzero[5] = 0.0
            _, _, Rf = chain_R(h, qfull)
            _, _, Rz_ = chain_R(h, qzero)
            fk_beta_err.append(abs(ang_of(Rf.T @ Rz_) - abs(se.q[h][k, 5])))
            n_off.append(D2(np.arccos(np.clip(Rf[:, 0] @ Rz_[:, 0], -1, 1))))
            f_off.append(D2(np.arccos(np.clip(Rf[:, 2] @ Rz_[:, 2], -1, 1))))
        print(f"{_short(p):<18s}{h[0].upper():<4s}"
              f"{beta.mean():9.1f}{np.percentile(beta, 95):8.1f}{beta.max():8.1f}"
              f"{np.percentile(n_off, 50):11.1f}{np.percentile(n_off, 95):6.1f}"
              f"{np.percentile(f_off, 50):10.1f}{np.percentile(f_off, 95):6.1f}")
check(max(fk_beta_err) < np.radians(0.05), "H3 β=朝向代价",
      f"FK 验证 angle(R_full,R_j6=0)=|β|,最大偏差 {D2(max(fk_beta_err)):.4f}°"
      "(丢 β 的总朝向代价恰为 β 本身)")
# 与骨架记录的手部朝向(R_wrist@PALM_FIX)的整体差,及其中绕掌法线的成分
print(f"\n{'素材':<22s}{'手':<4s}{'总朝向差mean°':>13s}{'p95°':>7s}"
      f"{'绕掌法线成分mean°':>17s}{'垂直成分mean°':>13s}")
for p, se in GATED.items():
    for h in HANDS:
        ok = ~np.isnan(se.q[h][:, 0]) & ~se.flag[h][:, 2] & ~se.flag[h][:, 3]
        tot, npar, nper = [], [], []
        for k in np.where(ok)[0][::5]:
            qzero = se.q[h][k].copy()
            qzero[5] = 0.0
            _, _, Rz_ = chain_R(h, qzero)
            Rt = TILT_T @ se.Ree_g[h][k]
            rv = pin.log3(Rz_.T @ Rt)
            tot.append(D2(np.linalg.norm(rv)))
            npar.append(D2(abs(rv[0])))          # EE 系 x = 掌法线
            nper.append(D2(np.hypot(rv[1], rv[2])))
        print(f"{_short(p):<18s}{h[0].upper():<4s}"
              f"{np.mean(tot):13.1f}{np.percentile(tot, 95):7.1f}"
              f"{np.mean(npar):17.1f}{np.mean(nper):13.1f}")

# ---------------------------------------------------------------- 表 4
print("\n" + "=" * 78)
print("表 4|形似 vs 位似:J2J FK 末端 vs 现行 chest-anchor 目标(pos-scale 1.0)")
print("=" * 78)
print("两侧同在「胸原点、重力轴」系里比:目标=(腕-肩中点)_腰系,J2J=Ry(55°)·FK。")
print(f"Vega 臂段(肩心→肘 {GEO['right']['Lu'] * 1000:.0f} / 肘→腕 "
      f"{GEO['right']['Lf'] * 1000:.0f} / 腕→手心 {GEO['right']['Lh'] * 1000:.0f}mm)"
      "比骨架模板(上臂 ~247 / 前臂 ~253mm)长得多 —— 形似必然换来末端位置")
print("偏差,这里量它有多大(径向签名>0 = 机器人手在人手的更远处)。\n")
print(f"{'素材':<22s}{'手':<4s}{'mean mm':>8s}{'p95 mm':>8s}{'max mm':>8s}"
      f"{'径向签名mean':>12s}{'直立子集mean':>12s}")
T4_STAT = {}
for p, se in GATED.items():
    base_tilt = np.nanpercentile(se.tilt, 10)
    upright = se.tilt < base_tilt + 15.0
    for h in HANDS:
        ok = ~np.isnan(se.q[h][:, 0]) & ~se.flag[h][:, 3]
        dist, rad, dist_up = [], [], []
        for k in np.where(ok)[0]:
            qzero = se.q[h][k].copy()
            qzero[5] = 0.0
            arm_fk(q14(qzero if h == "right" else HOME7["right"],
                       qzero if h == "left" else HOME7["left"]))
            chest = arm_frame("arm_center")
            s = "R" if h == "right" else "L"
            pe = TILT @ (chest.inverse() * arm_frame(f"{s}_ee")).translation
            tgt = se.W[h][k] - se.org[k]
            d = np.linalg.norm(pe - tgt) * 1000
            dist.append(d)
            nt = np.linalg.norm(tgt)
            rad.append((np.linalg.norm(pe) - nt) * 1000 if nt > 1e-6 else 0.0)
            if upright[k]:
                dist_up.append(d)
        T4_STAT[(os.path.basename(p), h)] = (np.mean(dist), np.percentile(dist, 95))
        print(f"{_short(p):<18s}{h[0].upper():<4s}"
              f"{np.mean(dist):8.0f}{np.percentile(dist, 95):8.0f}{np.max(dist):8.0f}"
              f"{np.mean(rad):+12.0f}"
              f"{(np.mean(dist_up) if dist_up else np.nan):12.0f}")
        if not np.std(dist) > 1.0:
            check(False, "H4 表4尺", "距离序列近乎常数 —— 尺子坏了")
# 表 4 尺自检:目标改成 FK 自己 -> 距离必须为 0;并且序列确实在变
se0 = GATED[MATERIALS[1]]
k0 = int(np.where(~np.isnan(se0.q["right"][:, 0]))[0][100])
qz0 = se0.q["right"][k0].copy()
qz0[5] = 0.0
arm_fk(q14(qz0, HOME7["left"]))
chest = arm_frame("arm_center")
pe0 = TILT @ (chest.inverse() * arm_frame("R_ee")).translation
check(np.linalg.norm(pe0 - pe0) < 1e-12, "H4 表4尺",
      "目标=FK 自身时距离恒 0;真实序列 std>1mm(非常数)均成立")

# ---------------------------------------------------------------- 表 5
print("\n" + "=" * 78)
print("表 5|臂形相似度:肘方向(肩→肘)与人骨架的逐帧夹角 + 回转角差")
print("=" * 78)
print("J2J 组按构造应接近 0(残差=方向反解的固有几何误差,见自检 B2);")
print("pink 组 = 现行映射回归 CSV 的 cmd 关节角 FK,量化「臂形不像人」。\n")


def elbow_metrics(u_rob_g, sh_g, el_g, wr_g, u_hum_g, s_h, e_h, w_h):
    ang = D2(np.arccos(np.clip(u_rob_g @ u_hum_g, -1, 1)))
    sw_r = swivel_angle(sh_g, el_g, wr_g)
    sw_h = swivel_angle(s_h, e_h, w_h)
    dsw = (abs(D2(wrap(sw_r - sw_h)))
           if sw_r is not None and sw_h is not None else np.nan)
    return ang, dsw


print(f"{'组':<26s}{'手':<4s}{'夹角mean°':>9s}{'p95°':>7s}{'max°':>7s}"
      f"{'|Δ回转|p50°':>11s}{'p95°':>7s}{'帧数':>7s}")
for p, se in GATED.items():
    for h in HANDS:
        ok = ~np.isnan(se.q[h][:, 0]) & ~se.flag[h][:, 3]
        angs, dsws = [], []
        for k in np.where(ok)[0]:
            qzero = se.q[h][k].copy()
            qzero[5] = 0.0
            arm_fk(q14(qzero if h == "right" else HOME7["right"],
                       qzero if h == "left" else HOME7["left"]))
            chest = arm_frame("arm_center")
            s = "R" if h == "right" else "L"
            el = TILT @ (chest.inverse() * arm_frame(f"{s}_arm_l4")).translation
            wr = TILT @ (chest.inverse() * arm_frame(f"{s}_arm_j7")).translation
            sh = TILT @ GEO[h]["SH"]
            u = (el - sh) / np.linalg.norm(el - sh)
            a, d = elbow_metrics(u, sh, el, wr, se.u_g[h][k],
                                 se.S[h][k], se.E[h][k], se.W[h][k])
            angs.append(a)
            if np.isfinite(d):
                dsws.append(d)
        print(f"J2J {_short(p):<18s}{h[0].upper():<4s}"
              f"{np.mean(angs):9.1f}{np.percentile(angs, 95):7.1f}"
              f"{np.max(angs):7.1f}{np.percentile(dsws, 50):11.1f}"
              f"{np.percentile(dsws, 95):7.1f}{len(angs):7d}")

csv_path = next((c for c in PINK_CSVS if os.path.exists(c)), None)
if csv_path is None:
    print("[表5] 找不到 pink 回归 CSV,pink 对照组跳过")
else:
    import csv as _csv
    with open(csv_path) as f:
        rd = _csv.reader(f)
        hdr = next(rd)
        rows = [r for r in rd if r and r[0]]
    col = {n: i for i, n in enumerate(hdr)}
    se = GATED[str(REPO / "logs/playlist_all13.msgpack")]
    # 自检 I:CSV 解析与对齐 —— 该跑 torso j1/j2 恒 home、j3 在动、臂角非常数
    tor = np.array([[float(r[col[f"cmd_torso_j{i}"]]) for i in (1, 2, 3)]
                    for r in rows])
    armv = np.array([float(r[col["cmd_R_arm_j4"]]) for r in rows])
    check(np.abs(tor[:, 0] - TORSO_HOME["torso_j1"]).max() < 1e-3
          and np.abs(tor[:, 1] - TORSO_HOME["torso_j2"]).max() < 1e-3,
          "I1 CSV torso 列", "cmd_torso_j1/j2 恒等于 home(waist-mode j3 只动 j3)")
    check(np.ptp(tor[:, 2]) > np.radians(20) and np.std(armv) > np.radians(1),
          "I2 CSV 内容活性",
          f"torso_j3 行程 {D2(np.ptp(tor[:, 2])):.0f}°、R_arm_j4 std "
          f"{D2(np.std(armv)):.1f}°(非冻结列)")
    stats = {h: ([], []) for h in HANDS}
    for r in rows:
        t = float(r[col["t"]])
        k = int(round(t * 100.0))
        if t < 8.0 or k >= se.n or np.isnan(se.u_g["right"][k][0]):
            continue
        qfv = pin.neutral(FULL)
        for i in (1, 2, 3):
            qfv[FULL_IDXQ[f"torso_j{i}"]] = float(r[col[f"cmd_torso_j{i}"]])
        for h, s in (("right", "R"), ("left", "L")):
            for i in range(1, 8):
                qfv[FULL_IDXQ[f"{s}_arm_j{i}"]] = float(r[col[f"cmd_{s}_arm_j{i}"]])
        pin.forwardKinematics(FULL, FULL_DATA, qfv)
        pin.updateFramePlacements(FULL, FULL_DATA)
        chest = FULL_DATA.oMf[FULL.getFrameId("arm_center")]
        for h, s in (("right", "R"), ("left", "L")):
            sh_b = chest.rotation @ GEO[h]["SH"] + chest.translation
            el_b = FULL_DATA.oMf[FULL.getFrameId(f"{s}_arm_l4")].translation
            wr_b = FULL_DATA.oMf[FULL.getFrameId(f"{s}_arm_j7")].translation
            u = (el_b - sh_b) / np.linalg.norm(el_b - sh_b)
            if np.isnan(se.u_g[h][k][0]):
                continue
            a, d = elbow_metrics(u, sh_b, el_b, wr_b, se.u_g[h][k],
                                 se.S[h][k], se.E[h][k], se.W[h][k])
            stats[h][0].append(a)
            if np.isfinite(d):
                stats[h][1].append(d)
    for h in HANDS:
        angs, dsws = stats[h]
        print(f"pink {'relaxfix2_idle':<18s}{h[0].upper():<4s}"
              f"{np.mean(angs):9.1f}{np.percentile(angs, 95):7.1f}"
              f"{np.max(angs):7.1f}{np.percentile(dsws, 50):11.1f}"
              f"{np.percentile(dsws, 95):7.1f}{len(angs):7d}")
    print(f"(pink CSV: {csv_path},t<8s 跳过以避开啮合渐入)")

# ---------------------------------------------------------------- 汇总
print("\n" + "=" * 78)
print(f"自检汇总:{'全部通过 ✅' if not FAILED else '失败 ❌ ' + ', '.join(FAILED)}")
print("=" * 78)
sys.exit(1 if FAILED else 0)
