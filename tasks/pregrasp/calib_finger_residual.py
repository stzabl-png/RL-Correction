"""手指关节残差界标定 —— 22 关节全放开之后, 每个关节各自的 Δq 上限。

## 为什么需要它

臂的界是**标定**出来的(`rl_rebuild/correction/calib_arm_residual.py`: 逐关节雅可比 ->
末端位移 95 分位 = 2cm), 而手指的界一直是**一个手挑的统一值** `finger_residual_max = 1.2`
rad(69°) —— 比臂的 0.0154 rad(0.88°)大 78 倍, 且不分关节。

这在合拢参数化时代不要紧(手指的自由度被 `a_c + a_δ` 挡住了, 那 1.2 只在对照实验里生效);
**22 关节全放开之后它就是主要的动作尺度**, 必须按同一套方法标定, 否则:
  · 给指根(MCP)一个过大的权限 —— 转 1° 指垫走 ~1.5mm, 是指尖(DIP)的好几倍;
  · 给指尖一个过小的权限 —— 微调落点恰恰要靠它。

## 做法(与臂的标定同口径, 便于两者可比)

  1. 采样**策略实际会待的位形**: 合拢路径 q_open -> q_close 上取 d ∈ [0, 1.25],
     再加限位内的随机扰动(不只取一条路径, 免得标定只对那条路成立)
  2. 逐关节有限差分算"杠杆臂" |∂pad_f/∂q_j| (m/rad), f = 关节 j 所属的手指
  3. 单关节界 = ε / 杠杆臂的 **90 分位**(不用中位: 界要在最不利位形下也成立)
  4. 22 个关节同时走满会叠加 -> 蒙特卡洛整体缩放, 让**指垫位移的 95 分位 = ε**
     (直接测我们关心的量, 不假设各关节独立)
  5. 校验 20Hz 下界全开时关节速度不超 URDF 限

ε 取 **2cm**, 与臂完全相同 —— 于是 `arm_step_scale=0.25` 一乘, 臂和手指都是
**每步 5mm** 的笛卡尔尺度, 两者由构造可比。

    $PY -m tasks.pregrasp.calib_finger_residual
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from rl_rebuild.correction.ref_builders.replay_grasp import (
    GENERIC_CLOSED, GENERIC_JOINT_ORDER, GENERIC_OPEN, _urdf)

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
_OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "finger_residual_bound.json"))


def _finger_of(joint_name: str) -> str:
    for f in FINGERS:
        if f"_{f}_" in joint_name or joint_name.endswith(f"_{f}"):
            return f
    raise ValueError(f"关节 {joint_name} 无法归属到唯一手指")


def _pad_centroids(hand):
    """各指胶垫**面**在 link 系里的质心。

    ⚠ 不能拿 link 原点当垫的位置: `<elastomer>` 的 STL 顶点离 link 原点中位 **2.35~2.59cm**
    (垫体整个偏在 −y 侧), 而且 **DIP/IP 的转轴正好穿过 link 原点** —— 按原点量, 转 DIP
    指垫"纹丝不动"(实测杠杆臂 0.00mm/rad), 于是这几个关节会被判成"对接触无影响"而拿到
    一个几乎等于整个行程的界。按垫面量, DIP 的杠杆臂约 24mm/rad, 是真实量级。
    (2026-08-15 踩到: 同一个口径错误还让"五垫离瓶面 12~20mm"这组数偏大了约 2.4cm。)
    """
    import re
    from rl_rebuild.correction.kinematics import _urdf_path
    up = _urdf_path()
    txt = open(up).read()
    root = os.path.dirname(up)
    out = {}
    for f in FINGERS:
        m = re.search(rf'<link name="{hand}_{f}_elastomer">.*?<mesh filename="([^"]+)"', txt, re.S)
        import trimesh
        V = np.asarray(trimesh.load(os.path.normpath(os.path.join(root, m.group(1))),
                                    process=False, force="mesh").vertices, float)
        out[f] = V.mean(0)
    return out


def _pads(u, names, q_vec, hand, cen):
    q = {n: float(v) for n, v in zip(names, q_vec)}
    res = {}
    for f in FINGERS:
        L = u.link_pose(f"{hand}_{f}_elastomer", q, np.eye(4), f"{hand}_hand_C_MC")
        res[f] = L[:3, :3] @ cen[f] + L[:3, 3]      # 垫面质心, 不是 link 原点
    return res


def calibrate(eps=0.02, ctrl_hz=20.0, step_scale=0.25, n_cfg=120, n_mc=20000,
              seed=0, hand="right", verbose=True):
    u = _urdf()
    names = [n.replace("right_", f"{hand}_") for n in GENERIC_JOINT_ORDER]
    fid = np.array([FINGERS.index(_finger_of(n)) for n in names])
    lo = np.array([u.joints[n]["lower"] for n in names])
    hi = np.array([u.joints[n]["upper"] for n in names])
    vlim = np.array([u.joints[n].get("velocity") or 0.0 for n in names])
    rng = np.random.default_rng(seed)
    cen = _pad_centroids(hand)

    # ---- 1. 采样位形: 合拢路径 + 限位内扰动 ----
    cfgs = []
    for d in np.linspace(0.0, 1.25, 12):
        base = GENERIC_OPEN + d * (GENERIC_CLOSED - GENERIC_OPEN)
        cfgs.append(np.clip(base, lo, hi))
        for _ in range(n_cfg // 12 - 1):
            cfgs.append(np.clip(base + rng.normal(0, 0.15, len(base)), lo, hi))
    cfgs = np.asarray(cfgs, float)

    # ---- 2. 逐关节杠杆臂 ----
    h = 1e-4
    lever = np.zeros((len(cfgs), len(names)))
    for i, c in enumerate(cfgs):
        p0 = _pads(u, names, c, hand, cen)
        for j in range(len(names)):
            cp = c.copy()
            cp[j] = min(cp[j] + h, hi[j])
            dq = cp[j] - c[j]
            if dq <= 0:
                cp[j] = max(c[j] - h, lo[j])
                dq = c[j] - cp[j]
            if dq <= 0:
                continue
            p1 = _pads(u, names, cp, hand, cen)
            f = FINGERS[fid[j]]
            lever[i, j] = np.linalg.norm(p1[f] - p0[f]) / dq
    lev90 = np.percentile(lever, 90, axis=0)

    # ---- 3. 单关节界 ----
    bound = eps / np.maximum(lev90, 1e-9)
    bound = np.minimum(bound, (hi - lo))            # 不超过关节自身行程

    # ---- 4. 蒙特卡洛整体缩放: 让指垫位移 95 分位 = eps ----
    def p95(b):
        idx = rng.integers(0, len(cfgs), n_mc)
        d = rng.uniform(-1, 1, (n_mc, len(names))) * b
        out = np.zeros(n_mc)
        for k in range(n_mc):
            c = cfgs[idx[k]]
            p0 = _pads(u, names, c, hand, cen)
            p1 = _pads(u, names, np.clip(c + d[k], lo, hi), hand, cen)
            out[k] = max(np.linalg.norm(p1[f] - p0[f]) for f in FINGERS)
        return float(np.percentile(out, 95))

    n_mc = min(n_mc, 2000)                          # FK 是 python 实现, 别跑太久
    cur = p95(bound)
    scale = eps / max(cur, 1e-9)
    bound = bound * scale
    got = p95(bound)

    # ---- 5. 速度校验 ----
    vel = bound * step_scale * ctrl_hz
    viol = [(names[j], float(vel[j]), float(vlim[j]))
            for j in range(len(names)) if vlim[j] > 0 and vel[j] > vlim[j]]

    rep = {
        "eps_m": eps, "ctrl_hz": ctrl_hz, "step_scale": step_scale, "hand": hand,
        "joint_names": [n.replace(f"{hand}_", "right_") for n in names],
        "bound_rad": bound.tolist(),
        "lever_p90_m_per_rad": lev90.tolist(),
        "mc_p95_before_m": cur, "mc_scale": float(scale), "mc_p95_after_m": got,
        "vel_violations": viol,
        "note": "与 calib_arm_residual 同口径(eps=2cm, 95 分位); 每步笛卡尔尺度 = eps*step_scale",
    }
    if verbose:
        print(f"\n{'关节':22s} {'杠杆臂p90':>11s} {'界(rad)':>9s} {'界(度)':>8s} "
              f"{'每步(mm)':>9s} {'行程占比':>8s}")
        for j, n in enumerate(names):
            print(f"  {n.replace(hand+'_',''):20s} {lev90[j]*1000:8.2f}mm/rad "
                  f"{bound[j]:9.4f} {np.degrees(bound[j]):7.2f}° "
                  f"{bound[j]*lev90[j]*step_scale*1000:8.2f} "
                  f"{bound[j]/(hi[j]-lo[j])*100:7.1f}%")
        print(f"\n蒙特卡洛: 缩放前 p95={cur*100:.2f}cm -> 整体×{scale:.3f} -> "
              f"p95={got*100:.2f}cm (目标 {eps*100:.0f}cm)")
        print(f"速度校验(20Hz, ×step_scale): "
              f"{'✅ 全部在限内' if not viol else f'⛔ {len(viol)} 个超限: {viol}'}")
        print(f"对照 —— 臂: 界 0.88°/关节, 每步 5mm; 手指现在: "
              f"中位 {np.degrees(np.median(bound)):.2f}°/关节, 每步 "
              f"{np.median(bound*lev90)*step_scale*1000:.2f}mm")
        print(f"旧的统一值 1.2 rad = 68.8° ⇒ 比标定值大 {1.2/np.median(bound):.0f}×(中位)")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eps", type=float, default=0.02, help="指垫位移 95 分位目标 (m)")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--out", default=_OUT)
    a = ap.parse_args(argv)
    rep = calibrate(eps=a.eps, hand=a.hand)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    print(f"\n-> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
