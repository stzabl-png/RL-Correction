"""DexMate(vega_1p_sharpa) 的纯 numpy 正/逆运动学.

为什么不用现成的:
  - cuRobo 装得不完整 (`curobo.wrap` 导不进来), IKSolver 用不了;
  - IsaacLab 的 DifferentialIKController 要拖起整个 Isaac 进程 (~40s 启动 + 独占 GPU),
    离线批量解 20 条 clip × 上百帧不值当。

这份实现只依赖 URDF, 秒级跑完, 且能显式做**关节限位**和**收敛判定** ——
正是"可达性统计"需要的东西。用途:
  1. 离线把重建腕轨迹 IK 成关节轨迹, 统计可达率 (offline_ik.py)
  2. 给 bimanual_align 提供机器人侧的头/手/肩世界位姿, 不必开 Isaac

⚠ 这里算的是**运动学可达**, 不含自碰撞和场景碰撞。IK 说能到, 不代表物理上到得了。
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

URDF_PATH = os.environ.get(
    "VEGA_URDF",
    "/home/lyh/luhr/MagicSim/Third_Party/curobo/curobo/content/assets/robot/"
    "vega_1p_sharpa/vega_1p_sharpa.urdf")

# correction_env_cfg.dexmate_joints 的默认站姿 (度). 躯干姿态会显著抬高肩部,
# 必须带上, 否则可达性算出来是错的 (实测零位肩 z=0.428, 实际 1.301, 差 87cm).
DEFAULT_TORSO_DEG = {"torso_j1": 45.0, "torso_j2": 90.0, "torso_j3": 0.0}
# 手臂默认角 (度): 肩 pitch 左右对称 ±45, 肘同向弯 -90. 用作 IK 的初始热启动,
# 也用作"机器人双手在哪"的参考位姿 (锚定 XY 平移时要的就是这个姿态下的手位).
DEFAULT_ARM_DEG = {"L_arm_j1": 45.0, "R_arm_j1": -45.0,
                   "L_arm_j4": -90.0, "R_arm_j4": -90.0}


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp,     cp * sr,                cp * cr]])


def _T(R, p):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, p
    return M


def quat_to_R(q):
    """(w,x,y,z) -> 3x3."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    """3x3 -> (w,x,y,z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                          (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
        elif i == 1:
            s = np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2]) * 2
            q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                          0.25 * s, (R[1, 2] + R[2, 1]) / s])
        else:
            s = np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2]) * 2
            q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                          (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def _so3_log(R):
    """旋转矩阵 -> 轴角向量 (3,), 模长即角度."""
    c = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > np.pi - 1e-6:                      # 接近 180°, 用对称部分取轴
        A = (R + np.eye(3)) * 0.5
        ax = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(ax))
        ax = A[:, k] / (ax[k] + 1e-12)
        return ax / (np.linalg.norm(ax) + 1e-12) * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (th / (2.0 * np.sin(th)))


class Urdf:
    """最小 URDF 解析 + FK。只处理 revolute/prismatic/fixed/continuous。"""

    def __init__(self, path=URDF_PATH):
        root = ET.parse(path).getroot()
        self.path = path
        self.joints = {}
        self.parent_joint = {}                  # child link -> joint name
        for j in root.findall("joint"):
            n, ty = j.get("name"), j.get("type")
            o = j.find("origin")
            xyz = np.zeros(3) if o is None or o.get("xyz") is None else \
                np.array([float(v) for v in o.get("xyz").split()])
            rpy = np.zeros(3) if o is None or o.get("rpy") is None else \
                np.array([float(v) for v in o.get("rpy").split()])
            a = j.find("axis")
            axis = np.array([1.0, 0.0, 0.0]) if a is None else \
                np.array([float(v) for v in a.get("xyz").split()])
            lim = j.find("limit")
            lo, hi = (-np.pi, np.pi) if ty == "continuous" else (0.0, 0.0)
            if lim is not None:
                lo = float(lim.get("lower", -np.pi))
                hi = float(lim.get("upper", np.pi))
            child = j.find("child").get("link")
            self.joints[n] = dict(
                type=ty, parent=j.find("parent").get("link"), child=child,
                origin=_T(_rpy(*rpy), xyz),
                axis=axis / (np.linalg.norm(axis) + 1e-12), lower=lo, upper=hi)
            self.parent_joint[child] = n
        children = set(self.parent_joint)
        allp = {v["parent"] for v in self.joints.values()}
        self.root = next(iter(allp - children))

    def chain_to(self, link):
        """root -> link 路径上的关节名, 从根到叶."""
        out, cur = [], link
        while cur in self.parent_joint:
            n = self.parent_joint[cur]
            out.append(n)
            cur = self.joints[n]["parent"]
        return out[::-1]

    def chain_between(self, start, end):
        """start(不含) -> end 的关节名. start=None 则从根算起."""
        full = self.chain_to(end)
        if start is None:
            return full
        kids = [self.joints[n]["child"] for n in full]
        if start not in kids:
            raise ValueError(f"{start} 不在 root->{end} 的链上")
        return full[kids.index(start) + 1:]

    def joint_T(self, name, q):
        j = self.joints[name]
        if j["type"] == "fixed":
            return j["origin"]
        if j["type"] == "prismatic":
            return j["origin"] @ _T(np.eye(3), j["axis"] * q)
        k = j["axis"]                            # revolute / continuous: Rodrigues
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
        return j["origin"] @ _T(R, np.zeros(3))

    def fk_chain(self, link, q: dict, base_T=None, start_link=None):
        """返回 link 的世界位姿, 以及沿链每个关节的 (世界原点, 世界轴).

        start_link 给定时, base_T 被解释为**该 link 的世界位姿**, 只算它之后的关节 ——
        用于把手臂锚在实测的 arm_center 上, 不依赖躯干角度的假设.
        """
        T = np.eye(4) if base_T is None else base_T.copy()
        axes = []
        for n in self.chain_between(start_link, link):
            j = self.joints[n]
            T = T @ self.joint_T(n, float(q.get(n, 0.0)))
            if j["type"] != "fixed":
                axes.append((n, T[:3, 3].copy(), T[:3, :3] @ j["axis"]))
        return T, axes

    def link_pose(self, link, q: dict, base_T=None, start_link=None):
        return self.fk_chain(link, q, base_T, start_link)[0]


class ArmIK:
    """单臂 7 自由度 DLS 逆解。躯干/底盘当作固定基座。"""

    def __init__(self, side="right", urdf=None, torso_deg=None,
                 base_pos=(-0.5, 0.0, 0.0), base_yaw_deg=0.0,
                 anchor_link=None, anchor_T=None):
        """anchor_link/anchor_T: 把手臂链锚在某个 link 的**实测**世界位姿上.

        为什么需要: 躯干在仿真里不会精确停在配置的角度 (实测 torso_j2 从 90° 塌到
        83.65°), 整个上半身连着肩一起转, 手臂基座跑掉 8.5cm —— 这时"臂完美跟上
        q_ref"和"末端到达参考位姿"是两回事. 锚在实测 arm_center 上就没有这个假设.
        """
        self.u = urdf or Urdf()
        self.side = side
        P = "R" if side == "right" else "L"
        self.ee_link = f"{side}_hand_C_MC"
        self.fixed = {k: np.deg2rad(v) for k, v in
                      (torso_deg or DEFAULT_TORSO_DEG).items()}
        self.anchor_link = anchor_link
        if anchor_T is not None:
            self.base_T = np.asarray(anchor_T, dtype=float)
        else:
            c, s = np.cos(np.deg2rad(base_yaw_deg)), np.sin(np.deg2rad(base_yaw_deg))
            self.base_T = _T(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]),
                             np.asarray(base_pos, dtype=float))

        chain = self.u.chain_to(self.ee_link)
        self.arm_joints = [n for n in chain if n.startswith(f"{P}_arm_j")
                           and self.u.joints[n]["type"] != "fixed"]
        self.lower = np.array([self.u.joints[n]["lower"] for n in self.arm_joints])
        self.upper = np.array([self.u.joints[n]["upper"] for n in self.arm_joints])
        self.n = len(self.arm_joints)
        self.q_default = np.clip(
            np.array([np.deg2rad(DEFAULT_ARM_DEG.get(n, 0.0)) for n in self.arm_joints]),
            self.lower, self.upper)

    # ---- FK ----
    def _qdict(self, q):
        d = dict(self.fixed)
        d.update({n: float(v) for n, v in zip(self.arm_joints, q)})
        return d

    def fk(self, q):
        """-> (p(3,), R(3,3))"""
        T = self.u.link_pose(self.ee_link, self._qdict(q), self.base_T, self.anchor_link)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def jacobian(self, q):
        """末端几何雅可比 (6,n): 上三行线速度, 下三行角速度 (世界系)."""
        T, axes = self.u.fk_chain(self.ee_link, self._qdict(q), self.base_T,
                                  self.anchor_link)
        pe = T[:3, 3]
        idx = {n: i for i, n in enumerate(self.arm_joints)}
        J = np.zeros((6, self.n))
        for n, o, z in axes:
            if n in idx:
                i = idx[n]
                J[:3, i] = np.cross(z, pe - o)
                J[3:, i] = z
        return J

    def link_pose_world(self, link, q=None):
        d = self._qdict(np.zeros(self.n) if q is None else q)
        return self.u.link_pose(link, d, self.base_T, self.anchor_link)

    # ---- IK ----
    def solve(self, p_tgt, R_tgt, q0=None, iters=200, pos_tol=5e-3, rot_tol=0.05,
              w_rot=0.35, damp=0.05, step_clip=0.25):
        """DLS 逆解, 带限位夹紧。

        w_rot   姿态误差权重。抓取任务里位置比姿态重要得多, 且 7 自由度臂
                在工作空间边缘常常"位置到得了、姿态到不了"; 压低姿态权重能让
                求解器优先保住位置, 更贴近实际训练里 RL 会做的取舍。
        返回 dict: q, pos_err(m), rot_err(rad), ok, iters_used, at_limit
        """
        q = (np.clip(q0, self.lower, self.upper) if q0 is not None
             else np.clip(np.zeros(self.n), self.lower, self.upper))
        pe = re = np.inf
        used = 0
        for used in range(1, iters + 1):
            p, R = self.fk(q)
            ep = p_tgt - p
            er = _so3_log(R_tgt @ R.T)
            pe, re = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
            if pe < pos_tol and re < rot_tol:
                break
            e = np.concatenate([ep, w_rot * er])
            J = self.jacobian(q)
            J[3:] *= w_rot
            dq = J.T @ np.linalg.solve(J @ J.T + (damp ** 2) * np.eye(6), e)
            m = np.max(np.abs(dq))
            if m > step_clip:
                dq *= step_clip / m
            q = np.clip(q + dq, self.lower, self.upper)
        at = np.isclose(q, self.lower, atol=1e-4) | np.isclose(q, self.upper, atol=1e-4)
        return dict(q=q, pos_err=pe, rot_err=re, ok=bool(pe < pos_tol and re < rot_tol),
                    iters_used=used, at_limit=at)

    def solve_traj(self, P, Q, q_init=None, **kw):
        """整条轨迹逐帧求解, 上一帧解作为下一帧热启动 (保证关节连续).

        q_init 默认取默认站姿 —— 第一帧从机器人实际待机姿态出发, 而不是零位,
        否则解出来的第一帧可能落在一个机器人根本不会经过的分支上。
        """
        out = []
        q = self.q_default.copy() if q_init is None else np.asarray(q_init, float).copy()
        for p, quat in zip(P, Q):
            r = self.solve(p, quat_to_R(quat), q0=q, **kw)
            # 只有解得动的帧才继承; 失败帧继续用上一次成功解热启动, 避免烂解传染
            if r["ok"] or q is None:
                q = r["q"].copy()
            out.append(r)
        return out


def robot_anchors(torso_deg=None, base_pos=(-0.5, 0.0, 0.0), base_yaw_deg=0.0,
                  head_link="vega_1p_head_l3"):
    """给 bimanual_align 用的机器人侧锚点 (世界坐标), 不需要开 Isaac.

    -> dict(hands={"left":(3,),"right":(3,)}, head=(3,),
            shoulders={"left":(3,),"right":(3,)}, ik={"left":ArmIK,"right":ArmIK})
    """
    u = Urdf()
    iks = {s: ArmIK(s, urdf=u, torso_deg=torso_deg, base_pos=base_pos,
                    base_yaw_deg=base_yaw_deg) for s in ("left", "right")}
    z = {s: iks[s].q_default for s in iks}       # 默认站姿, 不是零位
    hands = {s: iks[s].fk(z[s])[0] for s in iks}
    sh = {s: iks[s].link_pose_world(f"vega_1p_{'R' if s == 'right' else 'L'}_arm_l1",
                                    z[s])[:3, 3] for s in iks}
    head = iks["right"].link_pose_world(head_link, z["right"])[:3, 3]
    return dict(hands=hands, head=head, shoulders=sh, ik=iks)
