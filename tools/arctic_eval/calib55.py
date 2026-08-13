#!/usr/bin/env python3
"""55 条 ARCTIC grab 上重新定标位移法 —— 物体侧 + **手侧**，留一 subject 交叉验证。

样本从 15 条扩到 55 条（7 个 subject × 11 个物体），协议随之升级：
此前是"11 条定标 / s01 四条留出"的单次留出；现在做 **leave-one-subject-out**，
每个 subject 轮流当验证集，参数只来自其余 subject。这是这批数据能支持的最强协议，
也直接回答"k 换个人还成不成立"。

同时定标**手部** σ。这是 RL 真正需要的那个：`tasks/pregrasp` 跟踪的是手的参考轨迹，
物体的 σ 管不到它。手的锚后误差实测比物体小 1.7-1.8 倍（口径见 --hand 输出）。

⚠ 索引：`world_fused` 的 `hand_*` 按 **ARCTIC 30fps 标注帧**索引，长度是 `object_*` 的
2 倍。必须用 arctic_vidx(g) 取手，用视频帧号(f) 会取到一半时间处的手姿且不报错。
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from anchored_conf import ARCTIC, META, RR, umeyama

TAKES = RR / "Output/ReconstructOutput/arctic15"


def load(take: Path):
    """→ dict(obj, sub, 物体锚后误差/离锚位移, 手锚后误差/离锚位移) 或 None。"""
    sub, seq = take.parent.name, take.name
    mp_, ca = META / f"{sub}__{seq}.meta.json", take / "contact_auto.json"
    if not (mp_.is_file() and ca.is_file() and (take / "world_fused.npz").is_file()):
        return None
    v2a = {int(r["video_frame"]): int(r["arctic_vidx"])
           for r in json.loads(mp_.read_text())["index"]}
    z = np.load(take / "world_fused.npz", allow_pickle=True)
    W = np.asarray(z["object_ob_in_world"], float)
    c2w = np.asarray(z["c2w"], float)
    ht = np.asarray(z["hand_trans"], float)
    hv = np.asarray(z["hand_valid"]).astype(bool)
    nh = ht.shape[1]

    obj = seq.split("_")[0]
    mp = take / "objects/object_0/object_mesh_scaled_final.obj"
    if not mp.is_file():
        return None
    Vo = np.asarray(trimesh.load(mp, force="mesh").vertices, float).mean(0)
    Vg = np.asarray(trimesh.load(ARCTIC / "meta/object_vtemplates" / obj / "mesh.obj",
                                force="mesh").vertices, float).mean(0) / 1000.0
    o = np.load(ARCTIC / "raw_seqs" / sub / f"{seq}.object.npy", allow_pickle=True)
    ego = np.load(ARCTIC / "raw_seqs" / sub / f"{seq}.egocam.dist.npy", allow_pickle=True).item()
    mano = np.load(ARCTIC / "raw_seqs" / sub / f"{seq}.mano.npy", allow_pickle=True).item()
    T = len(o)
    w2e = np.tile(np.eye(4), (T, 1, 1))
    w2e[:, :3, :3] = np.asarray(ego["R_k_cam_np"], float)
    w2e[:, :3, 3] = np.asarray(ego["T_k_cam_np"], float).reshape(T, 3)
    P = np.tile(np.eye(4), (T, 1, 1))
    P[:, :3, :3] = Rotation.from_rotvec(o[:, 1:4]).as_matrix()
    P[:, :3, 3] = o[:, 4:7] / 1000.0

    f = np.array([t for t in range(len(W)) if t in v2a and v2a[t] < min(T, nh)])
    if len(f) < 40:
        return None
    g = np.array([v2a[t] for t in f])
    s, R, tv = umeyama(c2w[f, :3, 3], np.linalg.inv(w2e)[g, :3, 3])

    ann = json.loads(ca.read_text()).get("annotations", {})
    st = [sg[0] for sd in ("left", "right") for sg in (ann.get(sd) or [])]
    if not st:
        return None
    idx = np.where(f >= min(st))[0]
    if len(idx) < 20:
        return None
    gi = g[idx]

    p_raw = np.einsum("tij,j->ti", W[f[idx], :3, :3], Vo) + W[f[idx], :3, 3]   # 我们世界系
    ours = s * (R @ p_raw.T).T + tv                                            # 对齐到真值系
    gt = np.einsum("tij,j->ti", P[gi, :3, :3], Vg) + P[gi, :3, 3]
    # 目标 y = 真实锚后误差(真值米制) —— 两种口径都用它, 不变
    eo = np.linalg.norm((ours - ours[0]) - (gt - gt[0]), axis=1) * 1000
    # ★ 两种回归量 x:
    #   do      = 对齐**之后**量的位移 —— 运行时拿不到(需要真值才能对齐), 只用于对照
    #   do_raw  = 我们自己世界系里量的位移 —— **运行时唯一能拿到的**, 是正确的回归量
    do = np.linalg.norm(ours - ours[0], axis=1) * 1000
    do_raw = np.linalg.norm(p_raw - p_raw[0], axis=1) * 1000

    eh = dh = None
    for i, side in enumerate(("left", "right")):        # 取接触帧最多的那只手
        okm = hv[i][gi]
        if okm.sum() < 50:
            continue
        h_raw = ht[i][gi]
        oh = s * (R @ h_raw.T).T + tv
        gh = np.asarray(mano[side]["trans"], float)[gi]
        b = int(np.argmax(okm))
        e = np.linalg.norm((oh - oh[b]) - (gh - gh[b]), axis=1) * 1000
        d = np.linalg.norm(oh - oh[b], axis=1) * 1000
        d_raw = np.linalg.norm(h_raw - h_raw[b], axis=1) * 1000
        # hand_valid 挡不住 NaN: 个别 take 的 hand_trans/mano trans 含 NaN,
        # 不过滤会让整条链路(斜率/截距/MAE)全变 nan 且不报错。
        fin = okm & np.isfinite(e) & np.isfinite(d)
        if fin.sum() < 50:
            continue
        if eh is None or fin.sum() > len(eh):
            eh, dh, dh_raw = e[fin], d[fin], d_raw[fin]
    return dict(obj=obj, sub=sub, eo=eo, do=do, do_raw=do_raw,
                eh=eh, dh=dh, dh_raw=locals().get("dh_raw"))


def fit(e, d):
    """过原点最小二乘斜率 + 截距版 (e ≈ k·d + b)。返回 (k_origin, k, b)。"""
    m = d > 1e-6
    k0 = float(np.sum(d[m] * e[m]) / np.sum(d[m] ** 2))
    A = np.stack([d[m], np.ones(m.sum())], 1)
    k, b = np.linalg.lstsq(A, e[m], rcond=None)[0]
    return k0, float(k), float(b)


def rho(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20:
        return np.nan
    return float(np.corrcoef(np.argsort(np.argsort(a[m])), np.argsort(np.argsort(b[m])))[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", action="store_true", help="定标手部 σ 而不是物体")
    ap.add_argument("--aligned", action="store_true",
                    help="用**对齐后**位移当回归量(旧口径, 运行时拿不到)。默认用运行时口径。")
    a = ap.parse_args()
    key_e = "eh" if a.hand else "eo"
    key_d = ("dh" if a.hand else "do") + ("" if a.aligned else "_raw")
    what = ("手" if a.hand else "物体") + ("(对齐口径·仅对照)" if a.aligned else "(运行时口径)")

    rows = [r for r in (load(t) for t in sorted(TAKES.glob("*/*"))) if r]
    rows = [r for r in rows if r[key_e] is not None and len(r[key_e]) >= 20]
    print(f"可用 take: {len(rows)} 条  ({what}侧)\n")

    # ---------- 逐 take 斜率 ----------
    per = [(r["sub"], r["obj"], *fit(r[key_e], r[key_d]), float(np.median(r[key_e])))
           for r in rows]
    K = np.array([p[2] for p in per])
    print(f"逐 take 斜率 k: 中位 {np.median(K):.2f}  范围 {K.min():.2f}–{K.max():.2f}  "
          f"p10-p90 {np.percentile(K,10):.2f}–{np.percentile(K,90):.2f}")
    print(f"锚后误差中位: {np.median([p[5] for p in per]):.0f}mm\n")

    # ---------- k 与物体类别是否相关 ----------
    print(f"{'物体':16}{'n':>3}{'k 中位':>8}{'k 范围':>14}{'误差中位':>10}")
    byo = defaultdict(list)
    for sub, obj, k0, k, b, em in per:
        byo[obj].append((k0, em))
    for obj in sorted(byo, key=lambda o: -np.median([x[0] for x in byo[o]])):
        v = np.array([x[0] for x in byo[obj]])
        print(f"{obj:16}{len(v):>3}{np.median(v):>8.2f}"
              f"{f'{v.min():.2f}-{v.max():.2f}':>14}"
              f"{np.median([x[1] for x in byo[obj]]):>8.0f}mm")
    within = np.median([np.std([x[0] for x in byo[o]]) for o in byo if len(byo[o]) > 2])
    print(f"\n  物体内 k 标准差中位 {within:.2f}   全体 k 标准差 {K.std():.2f}"
          f"   → {'类别能解释一部分' if within < 0.8*K.std() else '类别解释不了, 用单一常数即可'}")

    # ---------- 留一 subject 交叉验证 ----------
    subs = sorted({r["sub"] for r in rows})
    print(f"\n{'='*76}\n【留一 subject 交叉验证】参数只来自其余 subject\n{'='*76}")
    print(f"{'留出':>6}{'条数':>5}{'定标k':>8}{'截距':>7}{'预测误差':>10}{'常数基线':>10}{'rho':>7}")
    P1, P2 = [], []
    for s in subs:
        tr = [r for r in rows if r["sub"] != s]
        te = [r for r in rows if r["sub"] == s]
        ks = [fit(r[key_e], r[key_d])[0] for r in tr]
        k = float(np.median(ks))
        res = np.concatenate([r[key_e] - k * r[key_d] for r in tr])
        b = float(np.median(res))
        const = float(np.median(np.concatenate([r[key_e] for r in tr])))
        p1 = np.concatenate([np.abs(k * r[key_d] + b - r[key_e]) for r in te])
        p2 = np.concatenate([np.abs(const - r[key_e]) for r in te])
        rr = rho(np.concatenate([r[key_d] for r in te]), np.concatenate([r[key_e] for r in te]))
        P1.append(p1); P2.append(p2)
        print(f"{s:>6}{len(te):>5}{k:>8.2f}{b:>7.0f}{np.median(p1):>8.0f}mm"
              f"{np.median(p2):>8.0f}mm{rr:>7.2f}")
    A, B = np.concatenate(P1), np.concatenate(P2)
    print(f"\n★ 合并  位移法 {np.median(A):.0f}mm   常数基线 {np.median(B):.0f}mm"
          f"   胜 {sum(np.median(x)<np.median(y) for x,y in zip(P1,P2))}/{len(subs)} 个 subject")

    # ---------- 全量最终常数 ----------
    kk = float(np.median([fit(r[key_e], r[key_d])[0] for r in rows]))
    bb = float(np.median(np.concatenate([r[key_e] - kk * r[key_d] for r in rows])))
    print(f"\n★ 全量定标（写入工具的常数）: 预期误差(mm) = {kk:.3f} × 离锚位移(mm) + {bb:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
