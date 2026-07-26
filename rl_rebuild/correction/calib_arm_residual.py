"""关节空间残差界标定: 把飞手的"末端 ±2cm"换算成 7 个臂关节各自的 Δq 上限.

为什么不能直接沿用 0.02:
  飞手的 wrist_pos_residual_max=0.02 是**笛卡尔**量, 策略直接给腕位移。
  换成关节残差后, 同样的 Δq 在不同姿态下产生的末端位移完全不同 —— 由雅可比决定。
  肩关节转 1° 末端走 ~0.9cm, 腕关节转 1° 只走 ~0.2cm, 差 4 倍以上。
  给所有关节同一个界, 等于给肩一个过大的权限、给腕一个过小的权限。

做法:
  1. 取第 2 步解出的 q_ref (offline_ik_q.npz), 只用**接触段**的帧 (那里精度要求最高)
  2. 每帧算雅可比, 取每列的位置部分模长 |J_v[:,i]| = 关节 i 的"杠杆臂" (m/rad)
  3. 单关节界 = ε / 杠杆臂的 90 分位  (用 90 分位不用中位: 界要在最不利姿态下也成立)
  4. 但 7 个关节同时走满会叠加. 用蒙特卡洛把整体缩放一遍, 让**末端位移的 95 分位 = ε**
     —— 直接测我们关心的量, 不假设各关节独立或正交
  5. 校验 20Hz 下界全开时的关节速度不超 URDF 限

  $PY -m rl_rebuild.correction.calib_arm_residual
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from rl_rebuild.correction.kinematics import ArmIK

_D = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/offline_ik"))
# URDF 里的关节速度上限 (rad/s), j1..j7
VEL_LIMIT = np.array([2.4, 2.4, 2.7, 2.7, 2.7, 2.7, 2.7])


def calibrate(eps=0.02, ctrl_hz=20.0, n_mc=20000, seed=0,
              q_npz=None, report=None, verbose=True):
    q = np.load(q_npz or f"{_D}/offline_ik_q.npz")
    rep = {r["clip"]: r for r in json.load(open(report or f"{_D}/offline_ik_report.json"))}
    iks = {"right": ArmIK("right"), "left": ArmIK("left")}

    J_all = []
    for c in q.files:
        r = rep[c]
        a, b = r["contact"]
        ik = iks[r["primary"]]
        J_all += [ik.jacobian(q[c][t]) for t in range(a, b + 1)]
    J_all = np.asarray(J_all)                              # (F,6,7)
    lever = np.linalg.norm(J_all[:, :3, :], axis=1)        # (F,7) m/rad

    b_single = eps / np.percentile(lever, 90, axis=0)      # 单关节独走时的界

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(J_all), n_mc)
    Jv = J_all[idx][:, :3, :]

    def p95(scale):
        dq = rng.uniform(-1, 1, (n_mc, 7)) * (b_single * scale)
        return np.percentile(np.linalg.norm(np.einsum("nij,nj->ni", Jv, dq), axis=1), 95)

    lo, hi = 0.01, 2.0                                     # 二分找整体缩放
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if p95(mid) < eps else (lo, mid)
    bound = b_single * lo

    dq = rng.uniform(-1, 1, (n_mc, 7)) * bound
    dx = np.linalg.norm(np.einsum("nij,nj->ni", Jv, dq), axis=1)
    vel = bound * ctrl_hz
    out = dict(bound_rad=bound.tolist(), bound_deg=np.degrees(bound).tolist(),
               scale=float(lo), n_frames=int(len(J_all)),
               lever_p90=np.percentile(lever, 90, axis=0).tolist(),
               dx_p50_cm=float(np.median(dx) * 100), dx_p95_cm=float(np.percentile(dx, 95) * 100),
               dx_max_cm=float(dx.max() * 100),
               vel_rad_s=vel.tolist(), vel_ok=bool((vel < VEL_LIMIT).all()))
    if verbose:
        print(f"样本 {out['n_frames']} 个接触段参考帧 | 末端预算 ε={eps*100:.0f}cm\n")
        print("关节   杠杆臂p90(m/rad)   单关节界      最终界      20Hz速度")
        for i in range(7):
            print(f"  j{i+1}      {out['lever_p90'][i]:.3f}        "
                  f"{np.degrees(b_single[i]):5.2f}°      "
                  f"{out['bound_deg'][i]:5.2f}°     {vel[i]:.2f} rad/s")
        print(f"\n整体缩放 {out['scale']:.3f} (7 关节同走会叠加, 单关节界不能直接用)")
        print(f"末端位移校验: 中位 {out['dx_p50_cm']:.2f}cm  "
              f"95分位 {out['dx_p95_cm']:.2f}cm  最大 {out['dx_max_cm']:.2f}cm")
        print(f"关节速度 vs URDF 上限: {'全部在限内 ✅' if out['vel_ok'] else '有超限 ⚠'}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--eps", type=float, default=0.02, help="末端位移预算 (m), 沿用飞手 ±2cm")
    p.add_argument("--out", default=f"{_D}/arm_residual_bound.json")
    a = p.parse_args()
    r = calibrate(eps=a.eps)
    with open(a.out, "w") as f:
        json.dump(r, f, indent=1)
    print(f"\n-> {a.out}")
