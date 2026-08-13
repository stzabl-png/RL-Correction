#!/usr/bin/env python3
"""★ confidence 对 **RL 真正承受的误差** 有没有预测力 —— 本轮设计的判据基线。

与 `final_report.py` 的区别是**误差口径**，这一点决定结论正负：

* `final_report.py` 用 **绝对** 位置误差（帧级 rho -0.60，结论"能挑帧"）。
* 本脚本用 **锚定后** 误差：按 `ref_builders/replay_grasp.py` 的现行摆放，物体 XY 在
  **交互开始帧**被对齐到手的抓取锚点，所以那一刻的偏移被归零 —— **绝对误差里的系统性
  深度偏移根本不会进入仿真**。RL 承受的只是"从锚点出发之后轨迹怎么发散"。

同时输出两个口径便于对照。若两者差距大，说明现有判据衡量的是一个 RL 不承受的量。

顺带拆解"运动幅度"（`--amp`），因为路径长度对逐帧抖动极敏感，不拆会把抖动误判成尺度错：
  净幅度 = 轨迹包围盒对角线（抗抖动）  路径长 = 逐帧位移和  抖动指数 = 路径长比/净幅度比
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

RR = Path(__file__).resolve().parents[2]
ARCTIC = Path("/media/lyh/DATA2/arctic/repo/data/arctic_data/data")
META = Path("/tmp/claude-1000/-home-lyh/90060bb3-d0eb-4640-b9d4-aab56bc267bb/scratchpad/arctic15")
ROWS = RR / "Output" / "arctic_eval" / "rows15"


def umeyama(P, Q):
    """相似变换 P->Q（闭式）。只喂相机轨迹 —— 用物体自身拟合会把要测的误差吸收掉。"""
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(X.T @ Y)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return float((S * np.diag(D)).sum() / (X ** 2).sum()), R, mq - float(
        (S * np.diag(D)).sum() / (X ** 2).sum()) * R @ mp


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    return float(np.corrcoef(np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m])))[0, 1])


def streams(take: Path):
    """→ (ours_world, gt_world, 逐帧 conf_pos, 接触起始帧在 f 内的下标, 帧号数组) 或 None。"""
    sub, seq = take.parent.name, take.name
    mp_, ca, npz = META / f"{sub}__{seq}.meta.json", take / "contact_auto.json", ROWS / f"{sub}__{seq}.npz"
    if not (mp_.is_file() and ca.is_file() and npz.is_file()):
        return None
    v2a = {int(r["video_frame"]): int(r["arctic_vidx"])
           for r in json.loads(mp_.read_text())["index"]}
    z = np.load(take / "world_fused.npz", allow_pickle=True)
    W = np.asarray(z["object_ob_in_world"], float)
    c2w = np.asarray(z["c2w"], float)
    obj = seq.split("_")[0]
    Vo = np.asarray(trimesh.load(take / "objects/object_0/object_mesh_scaled_final.obj",
                                 force="mesh").vertices, float).mean(0)
    Vg = np.asarray(trimesh.load(ARCTIC / "meta/object_vtemplates" / obj / "mesh.obj",
                                 force="mesh").vertices, float).mean(0) / 1000.0
    o = np.load(ARCTIC / "raw_seqs" / sub / f"{seq}.object.npy", allow_pickle=True)
    ego = np.load(ARCTIC / "raw_seqs" / sub / f"{seq}.egocam.dist.npy", allow_pickle=True).item()
    T = len(o)
    w2e = np.tile(np.eye(4), (T, 1, 1))
    w2e[:, :3, :3] = np.asarray(ego["R_k_cam_np"], float)
    w2e[:, :3, 3] = np.asarray(ego["T_k_cam_np"], float).reshape(T, 3)
    P = np.tile(np.eye(4), (T, 1, 1))
    P[:, :3, :3] = Rotation.from_rotvec(o[:, 1:4]).as_matrix()
    P[:, :3, 3] = o[:, 4:7] / 1000.0

    f = np.array([t for t in range(len(W)) if t in v2a and v2a[t] < T])
    g = np.array([v2a[t] for t in f])
    s, R, tv = umeyama(c2w[f, :3, 3], np.linalg.inv(w2e)[g, :3, 3])
    ours = s * (R @ (np.einsum("tij,j->ti", W[f, :3, :3], Vo) + W[f, :3, 3]).T).T + tv
    gt = np.einsum("tij,j->ti", P[g, :3, :3], Vg) + P[g, :3, 3]

    ann = json.loads(ca.read_text()).get("annotations", {})
    st = [sg[0] for sd in ("left", "right") for sg in (ann.get(sd) or [])]
    if not st:
        return None
    idx = np.where(f >= min(st))[0]
    if len(idx) < 20:
        return None
    return obj, ours, gt, np.load(npz)["conf_pos"], idx, f


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amp", action="store_true", help="改为拆解运动幅度（净 vs 路径长 vs 抖动）")
    a = ap.parse_args()

    takes = [r for r in (streams(t) for t in
                         sorted((RR / "Output/ReconstructOutput/arctic15").glob("*/*"))) if r]
    if not takes:
        print("(没有可比对的 take)")
        return 0

    if a.amp:
        print(f"{'物体':16}{'净幅度比':>9}{'路径长比':>9}{'抖动指数':>9}{'GT净幅度':>10}{'我们净幅度':>11}")
        n_, p_, j_ = [], [], []
        for obj, ours, gt, _, idx, _ in takes:
            A, B = ours[idx], gt[idx]
            nr = np.linalg.norm(A.max(0) - A.min(0)) / max(np.linalg.norm(B.max(0) - B.min(0)), 1e-9)
            pr = (np.linalg.norm(np.diff(A, axis=0), axis=1).sum()
                  / max(np.linalg.norm(np.diff(B, axis=0), axis=1).sum(), 1e-9))
            n_.append(nr); p_.append(pr); j_.append(pr / max(nr, 1e-9))
            print(f"{obj:16}{nr:>9.2f}{pr:>9.2f}{pr/max(nr,1e-9):>9.2f}"
                  f"{np.linalg.norm(B.max(0)-B.min(0))*1000:>8.0f}mm"
                  f"{np.linalg.norm(A.max(0)-A.min(0))*1000:>9.0f}mm")
        for nm, v, tail in (("净幅度比", n_, "← 真·尺度错"), ("路径长比", p_, ""),
                            ("抖动指数", j_, "← 纯抖动虚增")):
            print(f"{nm}   中位 {np.median(v):.2f}  范围 {min(v):.2f}–{max(v):.2f}   {tail}")
        return 0

    print(f"{'物体':16}{'帧':>6}{'rho(conf,绝对)':>15}{'rho(conf,锚后)':>15}   锚后 低分25%→高分25%")
    CA, EA, EB = [], [], []
    for obj, ours, gt, conf_all, idx, f in takes:
        a0 = idx[0]
        e_abs = np.linalg.norm(ours[idx] - gt[idx], axis=1) * 1000
        e_anc = np.linalg.norm((ours[idx] - ours[a0]) - (gt[idx] - gt[a0]), axis=1) * 1000
        c = (conf_all[f[idx]] if len(conf_all) > f[idx].max()
             else np.full(len(idx), np.nan))
        m = np.isfinite(c) & np.isfinite(e_anc)
        if m.sum() < 20:
            continue
        c, e_abs, e_anc = c[m], e_abs[m], e_anc[m]
        CA.append(c); EA.append(e_abs); EB.append(e_anc)
        lo = e_anc[c <= np.percentile(c, 25)]; hi = e_anc[c >= np.percentile(c, 75)]
        print(f"{obj:16}{len(c):>6}{rho(c,e_abs):>15.2f}{rho(c,e_anc):>15.2f}   "
              f"{np.median(lo):>6.0f}mm → {np.median(hi):>5.0f}mm  "
              f"({np.median(lo)/max(np.median(hi),1e-9):.1f}x)")
    C, A_, B = np.concatenate(CA), np.concatenate(EA), np.concatenate(EB)
    print(f"\n合计 {len(C)} 帧（仅接触后）  rho(conf,绝对) {rho(C,A_):+.2f}   "
          f"★ rho(conf,锚后) {rho(C,B):+.2f}")
    print("  conf_pos 分箱 → 锚后误差中位:")
    for lo_, hi_ in [(0, 20), (20, 40), (40, 60), (60, 80), (80, 101)]:
        k = (C >= lo_) & (C < hi_)
        if k.sum() > 5:
            print(f"    {lo_:3d}-{hi_:<3d} {int(k.sum()):>5}帧   {np.median(B[k]):>6.0f}mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
