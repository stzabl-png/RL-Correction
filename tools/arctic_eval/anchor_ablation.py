#!/usr/bin/env python3
"""σ 的锚点该用哪个事件 —— 四种锚点的消融，留一 subject 交叉验证。

背景：同事把"接触时刻"拆成了两个事件（提交 `8cf98ee` / 对账表 `5b6761d`）：
  * `first_contact`     mask 重叠，"手碰到"，自动、已挪到重建前段
  * `grasp_established` "抓稳了"，语义上才是 σ 的锚（RL 摆放就发生在这一刻）
本脚本量化：换锚点之后 σ 的预测精度变多少。

四种锚点（前三种是生产环境可得的，第四种是上界）：
  ① first_contact_mask   现行做法（`contact_auto.json` 首段起点）
  ② grasp_established_3d 同事的 3D 自动版（他标注"精度不够只当备胎"）
  ③ onset_plan           人工标注 / v17A（44 条人工 + 11 条 auto，金标准但不可扩展）
  ④ GT_grasp_established ARCTIC 真值推出的"≥3 指接触"（**oracle 上界**，生产拿不到）

口径与 calib55.py 一致：误差 = 锚定后的位置残差（真值米制），回归量 = 我们自己世界系里
量的离锚位移（运行时口径），留一 subject 交叉验证。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from anchored_conf import ARCTIC, META, RR, umeyama

CSV = RR / "Data/arctic_gt_contacts/onset_adjudication.csv"
TAKES = RR / "Output/ReconstructOutput/arctic15"
ANCHORS = [("① first_contact(现行)", "first_contact_mask"),
           ("② grasp_3d(同事自动)", "grasp_established_3d"),
           ("③ onset_plan(人工)", "onset_plan"),
           ("④ GT抓稳(oracle上界)", "GT_grasp_established(>=3指)")]


def streams(take: Path):
    """→ (sub, obj, 我们世界系物体质心 (N,3), 真值物体质心 (N,3), 视频帧号 f) 或 None。"""
    sub, seq = take.parent.name, take.name
    mp_ = META / f"{sub}__{seq}.meta.json"
    mesh = take / "objects/object_0/object_mesh_scaled_final.obj"
    if not (mp_.is_file() and mesh.is_file() and (take / "world_fused.npz").is_file()):
        return None
    v2a = {int(r["video_frame"]): int(r["arctic_vidx"])
           for r in json.loads(mp_.read_text())["index"]}
    z = np.load(take / "world_fused.npz", allow_pickle=True)
    W = np.asarray(z["object_ob_in_world"], float)
    c2w = np.asarray(z["c2w"], float)
    obj = seq.split("_")[0]
    Vo = np.asarray(trimesh.load(mesh, force="mesh").vertices, float).mean(0)
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
    if len(f) < 60:
        return None
    g = np.array([v2a[t] for t in f])
    s, R, tv = umeyama(c2w[f, :3, 3], np.linalg.inv(w2e)[g, :3, 3])
    raw = np.einsum("tij,j->ti", W[f, :3, :3], Vo) + W[f, :3, 3]        # 我们世界系(运行时口径)
    ours = s * (R @ raw.T).T + tv                                       # 对齐到真值系(算误差用)
    gt = np.einsum("tij,j->ti", P[g, :3, :3], Vg) + P[g, :3, 3]
    return sub, obj, raw, ours, gt, f


def build(rows_csv, anchor_col):
    """按给定锚点列构造 (sub, 锚后误差, 离锚位移[运行时口径])。"""
    out = []
    for take in sorted(TAKES.glob("*/*")):
        st = streams(take)
        if st is None:
            continue
        sub, obj, raw, ours, gt, f = st
        key = f"{take.parent.name}__{take.name}"
        r = rows_csv.get(key)
        if r is None:
            continue
        v = r.get(anchor_col, "")
        if v in ("", "-", "None"):
            continue
        a_vid = int(v)
        idx = np.where(f >= a_vid)[0]
        if len(idx) < 20:
            continue
        a0 = idx[0]
        e = np.linalg.norm((ours[idx] - ours[a0]) - (gt[idx] - gt[a0]), axis=1) * 1000
        d = np.linalg.norm(raw[idx] - raw[a0], axis=1) * 1000
        out.append((sub, e, d))
    return out


def loso(data):
    """留一 subject: 参数只来自其余 subject。→ (预测 MAE, 常数基线 MAE, 胜的 subject 数)"""
    subs = sorted({r[0] for r in data})
    P1, P2, win = [], [], 0
    for s in subs:
        tr = [r for r in data if r[0] != s]
        te = [r for r in data if r[0] == s]
        if not tr or not te:
            continue
        ks = []
        for _, e, d in tr:
            m = d > 1e-6
            if m.sum() > 20:
                ks.append(float(np.sum(d[m] * e[m]) / np.sum(d[m] ** 2)))
        k = float(np.median(ks))
        b = float(np.median(np.concatenate([e - k * d for _, e, d in tr])))
        const = float(np.median(np.concatenate([e for _, e, _ in tr])))
        p1 = np.concatenate([np.abs(k * d + b - e) for _, e, d in te])
        p2 = np.concatenate([np.abs(const - e) for _, e, _ in te])
        P1.append(p1); P2.append(p2); win += np.median(p1) < np.median(p2)
    A, B = np.concatenate(P1), np.concatenate(P2)
    return np.median(A), np.median(B), win, len(subs)


def main() -> int:
    rows_csv = {r["take"]: r for r in csv.DictReader(open(CSV))}
    print(f"{'锚点':26}{'take':>5}{'锚后误差中位':>13}{'σ 预测误差':>12}"
          f"{'常数基线':>10}{'胜':>7}")
    base = None
    for nm, col in ANCHORS:
        data = build(rows_csv, col)
        if not data:
            print(f"{nm:26}  (无数据)")
            continue
        E = np.concatenate([r[1] for r in data])
        mae, cb, win, ns = loso(data)
        tag = "" if base is None else f"  ({mae-base:+.0f}mm)"
        if base is None:
            base = mae
        print(f"{nm:26}{len(data):>5}{np.median(E):>11.0f}mm{mae:>10.0f}mm"
              f"{cb:>8.0f}mm{win:>4}/{ns}{tag}")
    print("\n  「锚后误差中位」= 换锚点后 RL 实际承受的误差本身也会变 —— 锚得越靠后，"
          "\n  物体已经动起来的部分被算进去越多。两列要一起看。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
