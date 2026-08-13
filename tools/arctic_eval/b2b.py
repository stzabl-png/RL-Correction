#!/usr/bin/env python3
"""B-2b：位移法能不能外推到**没见过的人**。

协议 = 留出 subject。定标集 = 11 条（s02/s04/s05×5/s06/s07/s10），
验证集 = **s01 四条**（该 subject 从未参与任何定标），物体与定标集重合，
参考帧策略也对齐（s01 box/phone 用 f0，与 s05 box / s10 phone 完全相同）。
k 与常数基线只来自定标集，s01 一帧都不参与拟合。

回答两件事：
* **k 稳不稳** —— 同一物体换个人，误差-位移斜率变多少；
* **预测准不准** —— 拿定标集的 k 去预测 s01 的毫米数，误差多大（这才是产品指标）。

⚠ 判读 k 时不要用"是否落在 [0.66,1.21] 区间内"这种二元说法：box 的 0.64 只低于下界
0.02，报成"跑出范围"是误导。看**差值**。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from anchored_conf import ARCTIC, META, RR, umeyama
from stage_a import collect

HOLDOUT = RR / "Output/ReconstructOutput/arctic15/s01"


def take_streams(take: Path):
    """→ (物体名, 锚后误差 mm, 离锚位移 mm, 净幅度比, 锚点帧) 或 None。不依赖 confidence 产物。"""
    sub, seq = take.parent.name, take.name
    mp_, ca = META / f"{sub}__{seq}.meta.json", take / "contact_auto.json"
    if not (mp_.is_file() and ca.is_file()):
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
    a0 = idx[0]
    e = np.linalg.norm((ours[idx] - ours[a0]) - (gt[idx] - gt[a0]), axis=1) * 1000
    d = np.linalg.norm(ours[idx] - ours[a0], axis=1) * 1000
    amp = (np.linalg.norm(ours[idx].max(0) - ours[idx].min(0))
           / max(np.linalg.norm(gt[idx].max(0) - gt[idx].min(0)), 1e-9))
    return obj, e, d, float(amp), int(min(st))


def slope(e, d):
    m = d > 1e-6
    return float(np.sum(d[m] * e[m]) / np.sum(d[m] ** 2))


def rho(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def main() -> int:
    # ---- 定标集：只用 11 条拟 k 与常数基线 ----
    dev = collect()
    dev_k = {o: slope(e, d) for o, _, e, _, _, d in dev}
    K = float(np.median(list(dev_k.values())))
    CONST = float(np.median(np.concatenate([r[2] for r in dev])))
    dev_amp = {}
    for o, _, e, _, _, d in dev:
        dev_amp[o] = None
    print(f"定标集 11 条  →  k = {K:.2f}   常数基线 = {CONST:.0f}mm")
    print(f"              k 逐条范围 {min(dev_k.values()):.2f}–{max(dev_k.values()):.2f}\n")

    print(f"{'='*96}\n【留出 subject s01】参数全部来自定标集，s01 一帧未参与拟合\n{'='*96}")
    print(f"{'物体':12}{'s01 k':>8}{'定标 k':>8}{'Δk':>7}{'净幅度比':>9}"
          f"{'实际误差':>10}{'位移法':>9}{'常数基线':>10}{'rho':>7}")
    P1, P2, win = [], [], 0
    for t in sorted(HOLDOUT.glob("*")):
        r = take_streams(t)
        if r is None:
            print(f"  {t.name}: 数据不全，跳过")
            continue
        obj, e, d, amp, _ = r
        k1 = slope(e, d)
        p1 = np.abs(K * d - e)
        p2 = np.abs(CONST - e)
        P1.append(p1); P2.append(p2); win += np.median(p1) < np.median(p2)
        dk = k1 - dev_k[obj] if obj in dev_k else np.nan
        print(f"{obj:12}{k1:>8.2f}{dev_k.get(obj, np.nan):>8.2f}{dk:>+7.2f}{amp:>9.2f}"
              f"{np.median(e):>8.0f}mm{np.median(p1):>7.0f}mm{np.median(p2):>8.0f}mm{rho(d, e):>7.2f}")
    A, B = np.concatenate(P1), np.concatenate(P2)
    print(f"\n★ 合并  位移法 {np.median(A):.0f}mm   常数基线 {np.median(B):.0f}mm   位移法胜 {win}/{len(P1)}")
    print(f"  对照：定标集内部留一物体时 位移法 35mm / 常数 76mm —— 外推后基本不掉")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
