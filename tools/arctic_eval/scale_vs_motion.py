#!/usr/bin/env python3
"""相机移动范围 vs ViPE 世界尺度偏离 —— 把 ARCTIC 与 EgoDex 放在一起看。

**为什么要跨数据集**：只看 ARCTIC 时，相机移动范围全落在 107–408mm 的窄带里，
"移动范围 vs 尺度偏离"的相关只有 −0.16，被我判成"无关"。但那是**量程不够**：
自变量本身没有变化，看不出关系。EgoDex 是走动的第一人称，相机基线大得多，
两者合起来才有足够量程。

**先验**：ViPE 是 SLAM。相机平移基线趋近零时尺度在数学上就不可观测 —— 这不是经验规律，
是三角化的固有性质。所以本脚本是在**验证一个有理由预期的关系**，不是在钓相关。

**为什么重要**：物体 σ 的运行时口径精度 40mm 是在 ARCTIC 上定标的，而 ARCTIC 的
尺度散布是 0.15–1.94（13 倍）。若散布确实由"相机不动"导致，那 40mm 就是**悲观上界**，
在正常拍摄的数据上应当更好 —— 这直接影响我们对工具可用性的判断。

真值来源：ARCTIC = mocap 相机；EgoDex = ARKit（Vision Pro 自身 SLAM，非 mocap，
但相机位姿是该设备最可靠的输出，逐帧对齐残差实测 1mm）。
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr

RR = Path(__file__).resolve().parents[2]
ARCTIC = Path("/media/lyh/DATA2/arctic/repo/data/arctic_data/data")
META = Path("/tmp/claude-1000/-home-lyh/90060bb3-d0eb-4640-b9d4-aab56bc267bb/scratchpad/arctic15")
EGODEX = Path("/home/lyh/Project/V2AP/data/egocentric/egodex/test")
EGO_INTERIM = RR / "Output/ReconstructOutput/interim/egodex_cmp"


def umeyama(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(X.T @ Y)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = float((S * np.diag(D)).sum() / (X ** 2).sum())
    return s, R, mq - s * R @ mp


def arctic_rows():
    out = []
    for take in sorted((RR / "Output/ReconstructOutput/arctic15").glob("*/*")):
        sub, seq = take.parent.name, take.name
        mp_ = META / f"{sub}__{seq}.meta.json"
        if not (mp_.is_file() and (take / "world_fused.npz").is_file()):
            continue
        v2a = {int(r["video_frame"]): int(r["arctic_vidx"])
               for r in json.loads(mp_.read_text())["index"]}
        c2w = np.asarray(np.load(take / "world_fused.npz", allow_pickle=True)["c2w"], float)
        ego = np.load(ARCTIC / "raw_seqs" / sub / f"{seq}.egocam.dist.npy", allow_pickle=True).item()
        T = len(np.asarray(ego["T_k_cam_np"], float).reshape(-1, 3))
        w2e = np.tile(np.eye(4), (T, 1, 1))
        w2e[:, :3, :3] = np.asarray(ego["R_k_cam_np"], float)
        w2e[:, :3, 3] = np.asarray(ego["T_k_cam_np"], float).reshape(T, 3)
        gt = np.linalg.inv(w2e)[:, :3, 3]
        f = np.array([t for t in range(len(c2w)) if t in v2a and v2a[t] < T])
        if len(f) < 40:
            continue
        g = np.array([v2a[t] for t in f])
        s, R, tv = umeyama(c2w[f, :3, 3], gt[g])
        resid = np.median(np.linalg.norm(s * (R @ c2w[f, :3, 3].T).T + tv - gt[g], axis=1))
        span = np.linalg.norm(gt[g].max(0) - gt[g].min(0))
        path = np.linalg.norm(np.diff(gt[g], axis=0), axis=1).sum()
        out.append(("ARCTIC", f"{sub}/{seq.split('_')[0]}", s, span * 1000,
                    path * 1000, resid * 1000))
    return out


def egodex_rows():
    import h5py
    out = []
    for p in sorted(EGO_INTERIM.glob("*")):
        pose = p / "vipe" / "pose" / f"{p.name}.npz"
        if not pose.is_file():
            continue
        task, idx = p.name.rsplit("__", 1)
        h5 = EGODEX / task / f"{idx}.hdf5"
        if not h5.is_file():
            continue
        c2w = np.asarray(np.load(pose, allow_pickle=True)["data"], float)
        with h5py.File(h5, "r") as f:
            gt = f["transforms/camera"][:, :3, 3].astype(float)
        n = min(len(c2w), len(gt))
        if n < 40:
            continue
        s, R, tv = umeyama(c2w[:n, :3, 3], gt[:n])
        resid = np.median(np.linalg.norm(s * (R @ c2w[:n, :3, 3].T).T + tv - gt[:n], axis=1))
        span = np.linalg.norm(gt[:n].max(0) - gt[:n].min(0))
        path = np.linalg.norm(np.diff(gt[:n], axis=0), axis=1).sum()
        out.append(("EgoDex", p.name[:26], s, span * 1000, path * 1000, resid * 1000))
    return out


def main() -> int:
    rows = arctic_rows() + egodex_rows()
    A = np.array([r[3] for r in rows])            # 相机移动范围(包围盒对角线) mm
    P = np.array([r[4] for r in rows])            # 相机路径长 mm
    S = np.array([r[2] for r in rows])            # 尺度 s
    D = np.abs(np.log(S))                         # 尺度偏离(对称: s 与 1/s 同罚)
    RS = np.array([r[5] for r in rows])
    ds = np.array([r[0] for r in rows])

    print(f"{'数据集':10}{'n':>4}{'相机范围中位':>13}{'路径长中位':>12}"
          f"{'尺度 s 范围':>16}{'|log s| 中位':>13}{'对齐残差':>10}")
    for tag in ("ARCTIC", "EgoDex"):
        k = ds == tag
        print(f"{tag:10}{k.sum():>4}{np.median(A[k]):>11.0f}mm{np.median(P[k]):>10.0f}mm"
              f"{f'{S[k].min():.2f}–{S[k].max():.2f}':>16}{np.median(D[k]):>13.2f}"
              f"{np.median(RS[k]):>8.0f}mm")

    print(f"\n★ 合并 {len(rows)} 条：相机移动范围 vs 尺度偏离 |log s|")
    for nm, x in (("包围盒范围", A), ("路径长", P)):
        r, p = spearmanr(x, D)
        print(f"   {nm:10} Spearman {r:+.2f}  p={p:.4f}"
              f"   {'← 负 = 动得越多尺度越准' if r < 0 else ''}")
    print(f"   仅 ARCTIC 内部: {spearmanr(A[ds=='ARCTIC'], D[ds=='ARCTIC']).statistic:+.2f}"
          f"   (量程 {A[ds=='ARCTIC'].min():.0f}–{A[ds=='ARCTIC'].max():.0f}mm, 太窄看不出)")

    print(f"\n{'按相机移动范围分组':22}{'n':>4}{'|log s| 中位':>13}{'s 范围':>16}{'对齐残差':>10}")
    edges = [0, 300, 600, 1200, 1e9]
    names = ["<300mm (几乎不动)", "300–600mm", "600–1200mm", ">1200mm (走动)"]
    for (lo, hi), nm in zip(zip(edges[:-1], edges[1:]), names):
        k = (A >= lo) & (A < hi)
        if k.sum() < 2:
            continue
        print(f"{nm:22}{k.sum():>4}{np.median(D[k]):>13.2f}"
              f"{f'{S[k].min():.2f}–{S[k].max():.2f}':>16}{np.median(RS[k]):>8.0f}mm")
    print("\n  |log s|: 0 = 尺度完全正确; 0.26 ≈ 差 1.3 倍; 0.69 ≈ 差 2 倍")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
