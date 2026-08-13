#!/usr/bin/env python3
"""HaWoR（我们的重建）vs ARKit（EgoDex 标注）—— **量分歧，不判对错**。

⚠⚠ 这不是精度评测。EgoDex 没有 mocap 真值，它的"标注"就是 Apple Vision Pro 的实时追踪
（ARKit 骨架，逐关节自报置信度）。ARKit 精度未知、无标定报告；实测 40 条 take 上腕部
置信度中位 0.99（可用），但**指尖中位仅 0.33、71% 的帧低于 0.5**。
**两台仪器读数不一致时，本脚本无法判断谁错。** 所以只比腕位，不比手指。

它能回答的是：在真实场景（非实验室）下，我们的手部重建与一台条件更好的仪器差多远。
量级本身有意义 —— 我们在 ARCTIC mocap 上测得腕位锚后误差 74mm；若这里的分歧远大于它，
说明真实场景确实比实验室桌面难。

口径与 ARCTIC 侧一致：
* 世界系用**相机轨迹** Umeyama 对齐（相机是两套世界之间唯一的桥）
* 锚定在共同起点，只比"从锚点出发之后怎么走"（绝对偏移在仿真里会被摆放消掉）
* 只取 ARKit 腕置信度达标的帧（默认 ≥0.5），避免拿噪声当参照

数据来源：
* 我们：`interim/egodex_cmp/<vid>/vipe/pose/<vid>.npz` 的 `data` (T,4,4) = 相机 c2w；
        `interim/egodex_cmp/<vid>/hawor/<vid>/world_space_res.pth` —— **纯 pickle**
        （不是 torch 归档，`torch.load` 会报 "Invalid magic number"），list 长度 5：
        [0] trans(2,T,3) [1] rot(2,T,3) [2] pose(2,T,45) [3] betas(2,T,10) [4] valid(2,T)
* ARKit：EgoDex 的 `<task>/<idx>.hdf5`，`transforms/camera` 与 `transforms/{left,right}Hand`
        均为 (T,4,4)。视频与标注逐帧 1:1（均 30fps，已核对 20 条）。
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

RR = Path(__file__).resolve().parents[2]
EGODEX = Path("/home/lyh/Project/V2AP/data/egocentric/egodex/test")
INTERIM = RR / "Output/ReconstructOutput/interim/egodex_cmp"


def umeyama(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(X.T @ Y)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = float((S * np.diag(D)).sum() / (X ** 2).sum())
    return s, R, mq - s * R @ mp


def load_ours(vid: str):
    """→ (c2w (T,4,4), 腕位 (2,T,3), 有效 (2,T)) 或 None。"""
    pose = INTERIM / vid / "vipe" / "pose" / f"{vid}.npz"
    hp = list((INTERIM / vid / "hawor").glob("*/world_space_res.pth"))
    if not (pose.is_file() and hp):
        return None
    z = np.load(pose, allow_pickle=True)
    if "data" not in z.files:
        return None
    with open(hp[0], "rb") as f:               # ⚠ 纯 pickle, 不能用 torch.load
        d = pickle.load(f)
    return (np.asarray(z["data"], float), np.asarray(d[0], float),
            np.asarray(d[4], float) > 0.5)


def load_arkit(task: str, idx: str, thr: float):
    import h5py
    with h5py.File(EGODEX / task / f"{idx}.hdf5", "r") as f:
        cam = f["transforms/camera"][:].astype(float)
        w = np.stack([f["transforms/leftHand"][:, :3, 3],
                      f["transforms/rightHand"][:, :3, 3]]).astype(float)
        c = np.stack([f["confidences/leftHand"][:], f["confidences/rightHand"][:]])
    return cam, w, c >= thr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=0.5)
    a = ap.parse_args()

    vids = sorted(p.name for p in INTERIM.glob("*") if (p / "hawor").is_dir())
    print(f"{'clip':38}{'帧':>5}{'相机残差':>9}{'尺度s':>7}{'手':>4}{'★锚后分歧':>11}{'绝对分歧':>10}{'该手运动范围':>11}")
    CAM, ANC, ABS, SC = [], [], [], []
    for vid in vids:
        task, idx = vid.rsplit("__", 1)
        o = load_ours(vid)
        if o is None:
            continue
        c2w, wr, ok_o = o
        cam_g, wr_g, ok_g = load_arkit(task, idx, a.conf)
        n = min(len(c2w), len(cam_g), wr.shape[1], wr_g.shape[1])
        if n < 40:
            continue
        s, R, tv = umeyama(c2w[:n, :3, 3], cam_g[:n, :3, 3])
        cam_res = np.median(np.linalg.norm(
            (s * (R @ c2w[:n, :3, 3].T).T + tv) - cam_g[:n, :3, 3], axis=1)) * 1000
        CAM.append(cam_res); SC.append(s)
        # ⚠ 挑手必须按**运动量**, 不能按"有效帧最多" ——
        #   闲着不动的手一直在视野里、置信度一直高, 会被选中; 而干活的手进出视野、
        #   有效帧反而少。实测按有效帧挑会选到静止手(ARKit 侧腕位移仅 0.1mm/帧),
        #   两个估计在"静止"上当然一致, 得出的 30mm 分歧是假的。
        best = None
        for h, side in ((0, "L"), (1, "R")):
            m = ok_o[h][:n] & ok_g[h][:n] & np.isfinite(wr[h][:n]).all(1)
            if m.sum() < 40:
                continue
            ours = s * (R @ wr[h][:n].T).T + tv
            gt = wr_g[h][:n]
            i0 = int(np.argmax(m))
            anc = np.linalg.norm((ours - ours[i0]) - (gt - gt[i0]), axis=1)[m] * 1000
            ab = np.linalg.norm(ours - gt, axis=1)[m] * 1000
            span = float(np.linalg.norm(gt[m].max(0) - gt[m].min(0)))   # ARKit 侧运动范围
            if best is None or span > best[0]:
                best = (span, side, anc, ab, m.sum())
        if best is None:
            print(f"{vid[:38]:38}{n:>5}{cam_res:>7.0f}mm{s:>7.2f}   (无可用手帧)")
            continue
        span, side, anc, ab, nv = best
        ANC.append(anc); ABS.append(ab)
        print(f"{vid[:38]:38}{n:>5}{cam_res:>7.0f}mm{s:>7.2f}{side:>4}"
              f"{np.median(anc):>9.0f}mm{np.median(ab):>8.0f}mm{span*1000:>9.0f}mm")
    if not ANC:
        print("\n(还没有可比对的 clip —— 重建可能仍在跑)")
        return 0
    A, B = np.concatenate(ANC), np.concatenate(ABS)
    print(f"\n★ {len(ANC)} 条 / {len(A)} 帧")
    print(f"   锚后分歧  中位 {np.median(A):>4.0f}mm  p75 {np.percentile(A,75):>4.0f}  "
          f"p90 {np.percentile(A,90):>4.0f}mm")
    print(f"   绝对分歧  中位 {np.median(B):>4.0f}mm  p75 {np.percentile(B,75):>4.0f}  "
          f"p90 {np.percentile(B,90):>4.0f}mm")
    print(f"   相机对齐残差 中位 {np.median(CAM):.0f}mm   尺度 s 中位 {np.median(SC):.2f} "
          f"范围 {min(SC):.2f}–{max(SC):.2f}")
    print(f"\n   参照: ARCTIC mocap 上我们的腕位锚后误差 74mm / 绝对 180mm")
    print(f"   ⚠ 本表是**分歧**不是误差 —— ARKit 自身精度未知, 无法判定谁错。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
