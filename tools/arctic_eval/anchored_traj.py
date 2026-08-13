#!/usr/bin/env python3
"""按 RL 的实际摆放方式测物体轨迹误差：在**接触起始帧**把物体锚到手上之后，还剩多少误差。

RL 摆放（用户描述）：相机位姿与机器人头 XY 对齐 → 手轨迹落到桌面 → **取接触起始帧的手 Pose，
把物体 XY 对齐到它，Z 调到贴桌**。所以手物相对误差在那一刻被归零，**物体的绝对位置误差不再是
RL 承受的量**。真正剩下的是"从锚点出发，物体后续怎么动"。

因此这里测三个量（都在 GT 世界系里，锚定后）：
* **位置发散**：锚定后逐帧的位置残差 —— 轨迹形状错多少
* **运动幅度比**：我们的位移量 / 真值位移量 —— 深度尺度错误会整体放大或缩小运动
* **旋转误差**：对称感知（离散 180° 群），与锚定无关

世界系对齐用**相机轨迹**的闭式 Umeyama（相机是两套世界之间的桥，且实测很准：残差 12mm/2°），
不用物体自身拟合 —— 那会把要测的误差吸收掉。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

RR = Path(__file__).resolve().parents[2]
ARCTIC = Path("/media/lyh/DATA2/arctic/repo/data/arctic_data/data")
META = Path("/tmp/claude-1000/-home-lyh/90060bb3-d0eb-4640-b9d4-aab56bc267bb/scratchpad/arctic15")
SYM = [np.eye(3)] + [Rotation.from_rotvec(np.pi * np.eye(3)[i]).as_matrix() for i in range(3)]


def umeyama(P, Q):
    """相似变换 P->Q（闭式，无局部极小）。"""
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(X.T @ Y)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = float((S * np.diag(D)).sum() / (X ** 2).sum())
    return s, R, mq - s * R @ mp


def main() -> int:
    print(f"{'物体':16}{'接触起始':>8}{'锚后帧':>7}"
          f"{'锚后位置误差(中位/p90)':>22}{'运动幅度比':>11}{'旋转误差':>9}")
    pos_all, amp_all = [], []
    for take in sorted((RR / "Output/ReconstructOutput/arctic15").glob("*/*")):
        sub, seq = take.parent.name, take.name
        mp_ = META / f"{sub}__{seq}.meta.json"
        ca = take / "contact_auto.json"
        if not (mp_.is_file() and ca.is_file()):
            continue
        meta = json.loads(mp_.read_text())
        v2a = {int(r["video_frame"]): int(r["arctic_vidx"]) for r in meta["index"]}
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
        gt_c2w = np.linalg.inv(w2e)
        P = np.tile(np.eye(4), (T, 1, 1))
        P[:, :3, :3] = Rotation.from_rotvec(o[:, 1:4]).as_matrix()
        P[:, :3, 3] = o[:, 4:7] / 1000.0

        f = np.array([t for t in range(len(W)) if t in v2a and v2a[t] < T])
        g = np.array([v2a[t] for t in f])
        # 世界系对齐：只用相机轨迹（物体自身拟合会把要测的误差吸收掉）
        s, R, tv = umeyama(c2w[f, :3, 3], gt_c2w[g, :3, 3])
        ours_w = s * (R @ (np.einsum("tij,j->ti", W[f, :3, :3], Vo) + W[f, :3, 3]).T).T + tv
        gt_w = np.einsum("tij,j->ti", P[g, :3, :3], Vg) + P[g, :3, 3]

        ann = json.loads(ca.read_text()).get("annotations", {})
        starts = [seg[0] for side in ("left", "right") for seg in (ann.get(side) or [])]
        if not starts:
            print(f"{obj:16}  (无接触区间)")
            continue
        t0 = min(starts)
        idx = np.where(f >= t0)[0]
        if len(idx) < 20:
            print(f"{obj:16}  (接触后帧太少)")
            continue
        a = idx[0]
        # ★ 锚定：把两边在接触起始帧对齐（RL 就是这么放的），只看之后怎么发散
        d = np.linalg.norm((ours_w[idx] - ours_w[a]) - (gt_w[idx] - gt_w[a]), axis=1) * 1000
        amp_o = np.linalg.norm(np.diff(ours_w[idx], axis=0), axis=1).sum()
        amp_g = np.linalg.norm(np.diff(gt_w[idx], axis=0), axis=1).sum()
        Dl = np.einsum("tij,tjk->tik", np.linalg.inv(W[f[idx], :3, :3]), P[g[idx], :3, :3])
        Rm = Rotation.from_matrix(Dl).mean().as_matrix()
        er = np.stack([np.degrees(np.linalg.norm(Rotation.from_matrix(
            np.einsum("ij,tjk,kl->til", Rm.T, Dl, S_)).as_rotvec(), axis=1)) for S_ in SYM]).min(0)
        pos_all.append(d)
        amp_all.append(amp_o / max(amp_g, 1e-9))
        print(f"{obj:16}{t0:>8}{len(idx):>7}"
              f"{np.median(d):>13.0f} /{np.percentile(d,90):>7.0f}mm"
              f"{amp_o/max(amp_g,1e-9):>11.2f}{np.median(er):>8.0f}°")
    if pos_all:
        D = np.concatenate(pos_all)
        print(f"\n合计 {len(D)} 帧   锚后位置误差 中位 {np.median(D):.0f}mm  "
              f"p75 {np.percentile(D,75):.0f}  p90 {np.percentile(D,90):.0f}mm")
        print(f"运动幅度比 中位 {np.median(amp_all):.2f}  范围 {min(amp_all):.2f}–{max(amp_all):.2f}"
              f"   (1.0 = 幅度正确)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
