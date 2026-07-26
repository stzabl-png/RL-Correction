"""离线 IK: 把重建腕轨迹解成 DexMate 手臂关节轨迹, 统计可达率.

回答的问题: **换成"真机器人带着训练"之后, 这 20 条 clip 的参考轨迹, 机器人的手臂
够得到吗?** 之前只有一个"掌心到肩的距离 ≤ 臂展"的球形近似, 那是上界很松的估计 ——
7 自由度臂在关节限位和姿态要求下, 实际工作空间远小于一个球。

不开 Isaac: 纯 numpy FK/IK (kinematics.py), 机器人侧锚点也自己 FK 出来,
所以能并行跑、秒级出结果。

  $PY -m rl_rebuild.correction.offline_ik                 # 全部 20 条
  $PY -m rl_rebuild.correction.offline_ik --clips Grasp2  # 单条, 带逐帧明细
  $PY -m rl_rebuild.correction.offline_ik --jobs 6        # 并行 (稳定姿态首算很慢)

⚠ 只算运动学可达, 不含自碰撞/桌面碰撞。IK 说到得了 ≠ 物理上到得了。
"""
from __future__ import annotations

import argparse
import json
import os
import traceback

import numpy as np

from rl_rebuild.correction import bimanual_align as BA
from rl_rebuild.correction import clips
from rl_rebuild.correction import frames as F
from rl_rebuild.correction.kinematics import ArmIK, Urdf, robot_anchors

POS_TOL = 0.005          # 5mm: 比抓取容忍度(毫米级)松一档, 但足以判"够不够得到"
ROT_TOL = np.deg2rad(15)


def solve_clip(name, table_top_z=0.85, grasp_gap=0.045,
               base_pos=(-0.5, 0.0, 0.0), head_anchor="vega_1p_head_l3"):
    e = clips.clip_entry(name)
    A = robot_anchors(base_pos=base_pos, head_link=head_anchor)
    rh = {k: v.tolist() for k, v in A["hands"].items()}

    res = BA.compute_placement(e["npz"], e["mesh"], rh, robot_head=A["head"].tolist(),
                               table_top_z=table_top_z, obj_gap=0.002, grasp_gap=grasp_gap)
    prim = res["primary"]
    J = res["joints"][prim]                       # (T,21,3) 已摆放
    P = J[:, 0]                                   # 腕位
    Q = F.sharpa_base_quat_from_joints(J)         # 腕姿 (wxyz), 与 dexmate_follow 同约定

    # 重建数据本身就带 NaN 帧 (轨迹首尾, 或该手整段没被检出). 这些帧不是 IK 解不出来,
    # 是**根本没有目标**, 必须单独统计, 不能混进可达率里 —— 也必须在拿去训练前处理掉。
    valid = np.isfinite(P).all(1) & np.isfinite(Q).all(1)
    nan_idx = np.flatnonzero(~valid)

    ik = A["ik"][prim]
    sols = ik.solve_traj(P, Q, pos_tol=POS_TOL, rot_tol=ROT_TOL)

    ok = np.array([s["ok"] for s in sols]) & valid
    pe = np.array([s["pos_err"] for s in sols])
    re_ = np.array([s["rot_err"] for s in sols])
    qs = np.stack([s["q"] for s in sols])
    a, b = res["inter"]["per_hand"][prim]          # 接触段
    ctc = np.zeros(len(P), bool)
    ctc[a:b + 1] = True
    cv = ctc & valid                               # 接触段里的有效帧

    def _st(mask, arr, f=lambda x: x):
        s = arr[mask]
        return (float(f(np.median(s))), float(f(s.max()))) if s.size else (float("nan"),) * 2

    # 关节跳变: 相邻帧最大角度差. 大跳 = 解跳到另一个 IK 分支, 轨迹不连续,
    # 直接拿去当参考会让 RL 的残差追不上. 只在连续两帧都有效时才算.
    both = valid[:-1] & valid[1:]
    dq = np.abs(np.diff(qs, axis=0)).max(axis=1)
    jmax = float(np.degrees(dq[both].max())) if both.any() else 0.0
    nlim = np.array([int(s["at_limit"].sum()) for s in sols])

    sh = A["shoulders"][prim]
    reach = np.linalg.norm(P - sh, axis=1)
    p50, pmax = _st(valid, pe, lambda x: x * 100)
    r50, rmax = _st(valid, re_, np.degrees)

    return dict(
        clip=name, primary=prim, T=int(len(P)), grasp_frame=int(res["grasp_frame"]),
        contact=[int(a), int(b)],
        n_nan=int(len(nan_idx)), nan_frames=[int(i) for i in nan_idx[:40]],
        n_nan_contact=int((ctc & ~valid).sum()),
        pos_ok_all=float(ok[valid].mean()) if valid.any() else float("nan"),
        pos_ok_contact=float(ok[cv].mean()) if cv.any() else float("nan"),
        # 位置单独看 (姿态解不出来但位置到得了, 对抓取仍然可用)
        pos_only_all=float((pe[valid] < POS_TOL).mean()) if valid.any() else float("nan"),
        pos_only_contact=float((pe[cv] < POS_TOL).mean()) if cv.any() else float("nan"),
        pos_err_p50_cm=p50, pos_err_max_cm=pmax,
        rot_err_p50_deg=r50, rot_err_max_deg=rmax,
        reach_max_m=float(reach[valid].max()) if valid.any() else float("nan"),
        reach_max_contact_m=float(reach[cv].max()) if cv.any() else float("nan"),
        joint_jump_max_deg=jmax,
        at_limit_frac=float((nlim[valid] > 0).mean()) if valid.any() else float("nan"),
        fail_frames=[int(i) for i in np.flatnonzero(valid & ~ok)][:40],
        q=qs, pos_err=pe, rot_err=re_,
    )


def _fmt(r):
    return (f"{r['clip']:<8} {r['primary']:<5} T={r['T']:<4} "
            f"接触[{r['contact'][0]:>3},{r['contact'][1]:>3}] "
            f"NaN{r['n_nan']:>3}({r['n_nan_contact']}) | "
            f"位+姿 {r['pos_ok_contact']*100:5.1f}% | "
            f"仅位置 {r['pos_only_contact']*100:5.1f}% | "
            f"位置误差 中位{r['pos_err_p50_cm']:5.2f} 最大{r['pos_err_max_cm']:6.2f}cm | "
            f"姿态 中位{r['rot_err_p50_deg']:5.1f}° | "
            f"肩距最大{r['reach_max_contact_m']:.2f}m | "
            f"跳变{r['joint_jump_max_deg']:5.1f}°")


def _one(name):
    try:
        return solve_clip(name)
    except Exception as ex:
        return dict(clip=name, error=f"{type(ex).__name__}: {ex}",
                    tb=traceback.format_exc()[-500:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", nargs="*", default=[f"Grasp{i}" for i in range(20)])
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", default="offline_ik_report.json")
    p.add_argument("--npz", default=None, help="另存关节轨迹 npz (每条 clip 一个 key)")
    a = p.parse_args()

    if a.jobs > 1:
        from multiprocessing import Pool
        with Pool(a.jobs) as pool:
            rs = pool.map(_one, a.clips)
    else:
        rs = []
        for n in a.clips:
            r = _one(n)
            print(_fmt(r) if "error" not in r else f"{n:<8} ❌ {r['error']}", flush=True)
            rs.append(r)

    good = [r for r in rs if "error" not in r]
    bad = [r for r in rs if "error" in r]
    if a.jobs > 1:
        for r in rs:
            print(_fmt(r) if "error" not in r else f"{r['clip']:<8} ❌ {r['error']}")

    print("\n" + "=" * 100)
    if good:
        pc = np.array([r["pos_only_contact"] for r in good])
        fc = np.array([r["pos_ok_contact"] for r in good])
        jm = np.array([r["joint_jump_max_deg"] for r in good])
        nn = np.array([r["n_nan"] for r in good])
        nc = np.array([r["n_nan_contact"] for r in good])
        print(f"{len(good)}/{len(rs)} 条解出 | 接触段**仅位置**可达率: "
              f"中位 {np.median(pc)*100:.1f}%  最差 {pc.min()*100:.1f}%  "
              f"≥95% 的有 {int((pc>=0.95).sum())} 条")
        print(f"接触段**位置+姿态**可达率: 中位 {np.median(fc)*100:.1f}%  "
              f"最差 {fc.min()*100:.1f}%  ≥95% 的有 {int((fc>=0.95).sum())} 条")
        print(f"关节跳变最大值: 中位 {np.median(jm):.1f}°  最大 {jm.max():.1f}° "
              f"(>30° 说明 IK 换分支, 参考轨迹不连续: "
              f"{[r['clip'] for r in good if r['joint_jump_max_deg']>30]})")
        print(f"重建自带 NaN 帧: {int((nn>0).sum())} 条 clip 有, 共 {int(nn.sum())} 帧; "
              f"落在接触段内的 {int(nc.sum())} 帧 "
              f"({[r['clip'] for r in good if r['n_nan']>0]})")
    for r in bad:
        print(f"❌ {r['clip']}: {r['error']}")

    ser = [{k: v for k, v in r.items() if k not in ("q", "pos_err", "rot_err", "tb")}
           for r in rs]
    with open(a.out, "w") as f:
        json.dump(ser, f, indent=1, ensure_ascii=False)
    print(f"\n报告 -> {os.path.abspath(a.out)}")
    if a.npz and good:
        np.savez_compressed(a.npz, **{r["clip"]: r["q"] for r in good})
        print(f"关节轨迹 -> {os.path.abspath(a.npz)}")


if __name__ == "__main__":
    main()
