#!/usr/bin/env python3
"""手-物**相对**几何误差（RL 真正消费的量）+ 手轨迹绝对误差。

为什么要单独测"相对"：物体位姿的绝对误差 100-380mm 里，绝大部分是**系统性深度偏移**
（整条 take 一个常数 1.06-1.67）。如果手被同等地推远，手物相对关系就基本保留，
绝对误差再大对抓取也无害。反之若只有物体被推远、手没有，抓取就会穿模或抓空。
**这个差值才是 RL 要承受的误差。**

口径（两侧完全对称，避免引入偏置）：
* 手参考点 = MANO 的全局平移 `trans`。我们的 `world_fused.hand_trans` 与 ARCTIC 的
  `mano.npy[side]["trans"]` 是**同一套参数化**，直接可比（betas 不同带来的偏差约 1cm 量级）。
* 物体参考点 = 各自网格顶点按各自位姿变换后的**质心** —— 同一个物理点，零拟合。
* 全部在**相机系**内比较：绕开世界系对齐，不引入任何拟合自由度。

⚠⚠ **索引陷阱（2026-08-12 修正，此前本文件的结论全部作废）**：
`world_fused.npz` 里 `hand_*` 的长度是 `object_*` 的 **2 倍** —— 手流按 **ARCTIC 30fps
标注帧**（即 `arctic_vidx`，我们记作 `g`）索引，而 object/c2w 按 **15fps 视频帧**（记作 `f`）
索引。用 `f` 去取手，等于取到了整条序列一半时间处的手姿，误差被大幅虚增且不报任何错。
本文件此前用 `ht[i][f]`，算出的"手 268mm / 相对 393mm"是**错的**；修正为 `ht[i][g]` 后
手的绝对误差与相对误差都显著下降（见运行输出）。
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


def gt_streams(subject: str, seq: str):
    o = np.load(ARCTIC / "raw_seqs" / subject / f"{seq}.object.npy", allow_pickle=True)
    ego = np.load(ARCTIC / "raw_seqs" / subject / f"{seq}.egocam.dist.npy", allow_pickle=True).item()
    mano = np.load(ARCTIC / "raw_seqs" / subject / f"{seq}.mano.npy", allow_pickle=True).item()
    T = len(o)
    w2e = np.tile(np.eye(4), (T, 1, 1))
    w2e[:, :3, :3] = np.asarray(ego["R_k_cam_np"], float)
    w2e[:, :3, 3] = np.asarray(ego["T_k_cam_np"], float).reshape(T, 3)
    P = np.tile(np.eye(4), (T, 1, 1))
    P[:, :3, :3] = Rotation.from_rotvec(o[:, 1:4]).as_matrix()
    P[:, :3, 3] = o[:, 4:7] / 1000.0
    return P, w2e, mano


def main() -> int:
    print(f"{'物体':16}{'帧':>5}"
          f"{'手绝对误差':>11}{'物绝对误差':>11}{'★手物相对误差':>14}{'相对/绝对':>10}")
    rel_all, hand_all, obj_all = [], [], []
    for take in sorted((RR / "Output/ReconstructOutput/arctic15").glob("*/*")):
        sub, seq = take.parent.name, take.name
        m = META / f"{sub}__{seq}.meta.json"
        if not m.is_file():
            continue
        meta = json.loads(m.read_text())
        v2a = {int(r["video_frame"]): int(r["arctic_vidx"]) for r in meta["index"]}
        z = np.load(take / "world_fused.npz", allow_pickle=True)
        C = np.asarray(z["object_ob_in_cam"], float)
        c2w = np.asarray(z["c2w"], float)
        w2c = np.linalg.inv(c2w)
        ht = np.asarray(z["hand_trans"], float)
        hv = np.asarray(z["hand_valid"]).astype(bool)

        obj = seq.split("_")[0]
        Vo = np.asarray(trimesh.load(take / "objects/object_0/object_mesh_scaled_final.obj",
                                     force="mesh").vertices, float).mean(0)
        Vg = np.asarray(trimesh.load(ARCTIC / "meta/object_vtemplates" / obj / "mesh.obj",
                                     force="mesh").vertices, float).mean(0) / 1000.0
        P_gt, w2e, mano = gt_streams(sub, seq)

        # 手流长度可能比标注帧少 1-2 帧(取整), 一并纳入上界, 否则末帧越界
        nh = ht.shape[1]
        f = np.array([t for t in range(len(C)) if t in v2a and v2a[t] < min(len(P_gt), nh)])
        g = np.array([v2a[t] for t in f])
        G = np.einsum("tij,tjk->tik", w2e[g], P_gt[g])
        co = np.einsum("tij,j->ti", C[f, :3, :3], Vo) + C[f, :3, 3]      # 我们的物体质心(相机系)
        cg = np.einsum("tij,j->ti", G[:, :3, :3], Vg) + G[:, :3, 3]      # 真值物体质心(相机系)

        rels, hands = [], []
        for i, side in enumerate(("left", "right")):
            ok = hv[i][g]                      # ★ 手一律用 arctic_vidx(g) 索引, 不是 f
            if ok.sum() < 20:
                continue
            ho = np.einsum("tij,tj->ti", w2c[f, :3, :3], ht[i][g]) + w2c[f, :3, 3]
            hg = (np.einsum("tij,tj->ti", w2e[g, :3, :3], np.asarray(mano[side]["trans"], float)[g])
                  + w2e[g, :3, 3])
            hands.append(np.linalg.norm(ho - hg, axis=1)[ok] * 1000)
            # ★ 相对量: 手→物 的向量, 两边各自算, 再比 —— 系统性同向偏移会在这里抵消
            rels.append(np.linalg.norm((co - ho) - (cg - hg), axis=1)[ok] * 1000)
        if not rels:
            continue
        h = np.concatenate(hands); r = np.concatenate(rels)
        oerr = np.linalg.norm(co - cg, axis=1) * 1000
        hand_all.append(h); rel_all.append(r); obj_all.append(oerr)
        print(f"{obj:16}{len(f):>5}{np.median(h):>10.0f}mm{np.median(oerr):>10.0f}mm"
              f"{np.median(r):>13.0f}mm{np.median(r)/max(np.median(oerr),1e-9):>9.2f}")
    if rel_all:
        H = np.concatenate(hand_all); R = np.concatenate(rel_all); O = np.concatenate(obj_all)
        print(f"\n{'合计':16}{'':>5}{np.median(H):>10.0f}mm{np.median(O):>10.0f}mm"
              f"{np.median(R):>13.0f}mm{np.median(R)/np.median(O):>9.2f}")
        print(f"\n  手物相对误差分位: p25 {np.percentile(R,25):.0f}  中位 {np.median(R):.0f}  "
              f"p75 {np.percentile(R,75):.0f}  p90 {np.percentile(R,90):.0f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
