#!/usr/bin/env python3
"""三套"接触开始"判据 vs ARCTIC 真值 —— 谁判得准。

**背景**：仓里有三套独立实现在算同一件事，判据完全不同，产物互不相通：
  ① v17A `interaction_episodes.py` —— HOI-DETR 检测"手-物交互链接"稳定 → `frame_plan.json`
  ② `phase/detect.py`            —— 2D mask 重叠(手 mask ∩ 膨胀物体 mask) → `contact_auto.json`
  ③ `scripts/grasp_detect.py`    —— 运动耦合(手物同速 + 相对向量稳定) → `grasp_annotation.json`
它们给出的帧号实测差中位 5 帧、最大 11 帧(15fps 下 733ms)。而 **RL 摆放和 σ 必须用同一帧**，
所以需要知道谁最接近物理真相。

**真值怎么来**：ARCTIC 没有"接触"标注，本脚本从 mocap 真值**推**两个判据：

* **GT-A 手物最小距离**（主判据，最接近物理接触）：MANO 前向出手部顶点(用 ARCTIC 给的
  逐帧 pose/shape)，物体顶点按真值位姿变换，取两者最小距离首次低于阈值的帧。
* **GT-B 物体开始移动**（对照，不依赖 MANO）：真值物体质心速度首次超过阈值的帧。
  它回答的是"物体何时开始被操纵"，正是 σ 的锚点语义所关心的。

两个判据都报，因为它们本身就不是同一件事：手碰到物体 ≠ 物体开始动（可能先碰后拿）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

RR = Path(__file__).resolve().parents[2]
ARCTIC = Path("/media/lyh/DATA2/arctic/repo/data/arctic_data/data")
META = Path("/tmp/claude-1000/-home-lyh/90060bb3-d0eb-4640-b9d4-aab56bc267bb/scratchpad/arctic15")
PLANS = Path("/tmp/claude-1000/-home-lyh/90060bb3-d0eb-4640-b9d4-aab56bc267bb/scratchpad/frameplans")
# smplx.create 期望 <root>/mano/MANO_{LEFT,RIGHT}.pkl 的布局, 用软链接搭出来
MANO_DIR = Path("/tmp/claude-1000/-home-lyh/90060bb3-d0eb-4640-b9d4-aab56bc267bb/scratchpad/manoroot")


def mano_layers():
    # chumpy(smplx 读旧版 MANO pkl 的依赖)用了 Python 3.11 起已删除的 inspect.getargspec。
    # 这是该库在新 Python 上的通用 shim, 不改行为。
    import inspect
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec
    import smplx
    return {s: smplx.create(str(MANO_DIR), model_type="mano", use_pca=False,
                            is_rhand=(s == "right"), flat_hand_mean=True)
            for s in ("left", "right")}


def gt_onsets(sub: str, seq: str, layers, thr_mm: float, vel_mm: float):
    """→ (GT-A 手物最小距离首次<thr 的 arctic 帧, GT-B 物体起动帧) 单位=arctic 30fps 帧号。"""
    o = np.load(ARCTIC / "raw_seqs" / sub / f"{seq}.object.npy", allow_pickle=True)
    mano = np.load(ARCTIC / "raw_seqs" / sub / f"{seq}.mano.npy", allow_pickle=True).item()
    obj = seq.split("_")[0]
    M = trimesh.load(ARCTIC / "meta/object_vtemplates" / obj / "mesh.obj", force="mesh")
    Vo = np.asarray(M.vertices, float) / 1000.0
    step = max(1, len(Vo) // 800)                     # 降采样, 800 点足够定最小距离
    Vo = Vo[::step]
    T = len(o)
    R = Rotation.from_rotvec(o[:, 1:4]).as_matrix()
    t = o[:, 4:7] / 1000.0
    cen = np.einsum("tij,j->ti", R, Vo.mean(0)) + t   # 物体质心轨迹(真值)

    # ---- GT-B: 物体起动 ----
    v = np.r_[0.0, np.linalg.norm(np.diff(cen, axis=0), axis=1)] * 1000 * 30   # mm/s
    vs = np.convolve(v, np.ones(5) / 5, mode="same")
    b = int(np.argmax(vs > vel_mm)) if (vs > vel_mm).any() else -1

    # ---- GT-A: 手物最小距离 ----
    a = -1
    best = None
    for side in ("left", "right"):
        p = mano[side]
        with torch.no_grad():
            out = layers[side](
                global_orient=torch.tensor(np.asarray(p["rot"], np.float32)),
                hand_pose=torch.tensor(np.asarray(p["pose"], np.float32)),
                betas=torch.tensor(np.tile(np.asarray(p["shape"], np.float32), (T, 1))),
                transl=torch.tensor(np.asarray(p["trans"], np.float32)))
        Vh = out.vertices.numpy()[:, ::8]              # (T,~100,3) 降采样
        d = np.empty(T)
        for k in range(T):
            Vok = (R[k] @ Vo.T).T + t[k]
            d[k] = np.min(np.linalg.norm(Vh[k][:, None] - Vok[None], axis=2))
        hit = np.flatnonzero(d * 1000 < thr_mm)
        if len(hit) and (best is None or hit[0] < best):
            best = int(hit[0])
    a = best if best is not None else -1
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thr-mm", type=float, default=15.0, help="GT-A 判接触的手物最小距离")
    ap.add_argument("--vel-mm", type=float, default=30.0, help="GT-B 判起动的物体速度 mm/s")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    layers = mano_layers()
    print(f"{'take':22}{'GT-A手物接触':>12}{'GT-B物体起动':>13}"
          f"{'①v17A/人工':>12}{'②mask重叠':>11}   (均为 15fps 视频帧)")
    rows = []
    takes = sorted((RR / "Output/ReconstructOutput/arctic15").glob("*/*"))
    if a.limit:
        takes = takes[:a.limit]
    for take in takes:
        sub, seq = take.parent.name, take.name
        mp_ = META / f"{sub}__{seq}.meta.json"
        fp = PLANS / f"{sub}__{seq}" / "sam2_object" / "frame_plan.json"
        ca = take / "contact_auto.json"
        if not (mp_.is_file() and fp.is_file() and ca.is_file()):
            continue
        idx = json.loads(mp_.read_text())["index"]
        a2v = {int(r["arctic_vidx"]): int(r["video_frame"]) for r in idx}
        ga, gb = gt_onsets(sub, seq, layers, a.thr_mm, a.vel_mm)

        def to_vid(x):
            if x < 0:
                return None
            k = min(a2v, key=lambda v: abs(v - x))
            return a2v[k]

        va, vb = to_vid(ga), to_vid(gb)
        d1 = json.loads(fp.read_text())["objects"].get("object_0", {}).get("fp_register_frame")
        m1 = None if d1 is None else int(d1) - 10
        ann = json.loads(ca.read_text()).get("annotations", {})
        st = [s[0] for sd in ("left", "right") for s in (ann.get(sd) or [])]
        m2 = min(st) if st else None
        rows.append((f"{sub}/{seq.split('_')[0]}", va, vb, m1, m2))
        f = lambda x: "-" if x is None else str(x)     # noqa: E731
        print(f"{rows[-1][0]:22}{f(va):>12}{f(vb):>13}{f(m1):>12}{f(m2):>11}")

    print(f"\n{'判据':22}{'vs GT-A 手物接触':>18}{'vs GT-B 物体起动':>18}")
    for nm, i in (("① v17A / 人工标注", 3), ("② mask 重叠", 4)):
        for j, gtn in ((1, "A"), (2, "B")):
            pass
        da = np.array([r[i] - r[1] for r in rows if r[i] is not None and r[1] is not None], float)
        db = np.array([r[i] - r[2] for r in rows if r[i] is not None and r[2] is not None], float)
        print(f"{nm:22}{f'{np.median(da):+.0f} 帧 (|中位| {np.median(np.abs(da)):.0f})':>18}"
              f"{f'{np.median(db):+.0f} 帧 (|中位| {np.median(np.abs(db)):.0f})':>18}")
    print("\n  正 = 判得比真值**晚**; 15fps 下 1 帧 = 67ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
