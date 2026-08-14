#!/usr/bin/env python3
"""导出逐帧对生度 + 手物距离 —— 给阈值标定用（正/负样本都出）。

============================ 为什么需要这个 ============================

`--min-opposition` 现在默认 0.40, 是**在单条 EgoDex clip 上定的**(screw/0 的松手段
对生度 0.34, 取 >0.36)。GraspPose 侧实测该阈值在 ARCTIC 上把 102 只手全拒了 ——
跨数据集不通用。

标定要两头:
  * **下界**(不能高到把真抓握拒掉) —— ARCTIC 有 62 条真值, 但**零负样本**(110 只手
    GT 全部有抓握), 只能定下界;
  * **上界**(不能低到把"扶一下/蹭一下"当抓握) —— 需要负样本, EgoDex 这边有现成的。

本工具出的是**逐帧序列**而不是单个数, 标定时可以按帧用。

============================ 已知的负样本 ============================

| take | 组合 | 帧段 | 为什么是负样本 |
|---|---|---|---|
| screw/0 | object_0×left | f107~f128 | **松手阶段**: ARKit 真值张开度 87→122mm 而瓶径 65.5mm |
| screw/0 | object_0×right | 整段 | 右手在拧**盖**, 跨在盖与瓶口交界, 对瓶身不是抓握 |

正样本(同一批数据, 便于同尺度对比):
  screw/0 object_0×left f42~f61 / object_1×right f19~f29
  pour/17 object_0×left f68~f81 / object_1×right f31~f47

============================ 对生度的定义(口径) ============================

    取五个指尖(OpenPose-21 的 4/8/12/16/20)
    → 各自在物体表面的最近点 → 取该点的**面法向** → 单位化
    → opposition = 1 - ||五个法向的均值||

标定曲线(五指尖均匀铺开 θ 角): θ=180°→0.52, 150°→0.45, 120°→0.25, 90°→0.10。

⚠ 它依赖面法向, 而重建网格实测 **39~46% 的面法向指向体内**(`fix_normals` 修不好)。
  标定前先看 `winding_consistent` 列, 不合格的 take 要单独分组, 否则标的是噪声。

用法:
    python -m ego_pipeline.contact.dump_opposition <take目录> [--out x.csv]
    python -m ego_pipeline.contact.dump_opposition <take1> <take2> ... --out 合并.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

TIPS = [4, 8, 12, 16, 20]          # OpenPose-21 五指尖


def rows_for(recon_dir: Path):
    import trimesh
    from scipy.spatial import cKDTree

    D = Path(recon_dir)
    w = np.load(D / "world_fused.npz", allow_pickle=True)
    rp = D / "replay_world.npz"
    if not rp.is_file():
        rp = Path(str(D).replace("ReconstructOutput", "RetargetOutput")) / "replay_world.npz"
    if not rp.is_file():
        print(f"  [skip] {D}: 缺 replay_world.npz", file=sys.stderr)
        return []
    r = np.load(rp, allow_pickle=True)
    allT = (w["object_ob_in_world_all"] if "object_ob_in_world_all" in w.files
            else w["object_ob_in_world"][None])
    oids = ([str(x) for x in w["object_ids"]] if "object_ids" in w.files else ["object_0"])
    take = f"{D.parent.name}/{D.name}"

    out = []
    for oi, oid in enumerate(oids):
        mp = D / "objects" / oid / "object_mesh_scaled_final.obj"
        if not mp.is_file():
            mp = D / "object_mesh_scaled_final.obj"
        if not mp.is_file():
            continue
        mesh = trimesh.load(mp, process=False, force="mesh")
        Vl, fid = trimesh.sample.sample_surface(mesh, 60000, seed=0)
        Vl = np.asarray(Vl)
        N = np.asarray(mesh.face_normals)[fid]
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        tree = cKDTree(Vl)
        wind = bool(mesh.is_winding_consistent)

        ca = D / f"contact_auto_{oid}.json"
        if not ca.is_file():
            ca = D / "contact_auto.json"
        segs_all = (json.loads(ca.read_text()).get("annotations") or {}) if ca.is_file() else {}

        for side in ("left", "right"):
            hv = r.get(f"mano_verts_{side}") if hasattr(r, "get") else r[f"mano_verts_{side}"]
            tips = r[f"joints_{side}"][:, TIPS]
            segs = segs_all.get(side) or []
            frames = [t for a, b in segs for t in range(a, min(b + 1, allT.shape[1], len(tips)))]
            for t in frames:
                M = allT[oi, t]
                inv = np.linalg.inv(M)
                tl = (inv[:3, :3] @ tips[t].T).T + inv[:3, 3]
                d, i = tree.query(tl)
                nn = N[i]
                oppo = float(1.0 - np.linalg.norm(nn.mean(0)))
                Vw = (M[:3, :3] @ Vl.T).T + M[:3, 3]
                gap3d = float(cKDTree(Vw).query(hv[t])[0].min() * 1000)
                out.append({
                    "take": take, "object_id": oid, "side": side, "frame": t,
                    "opposition": round(oppo, 4),
                    "gap3d_mm": round(gap3d, 3),
                    "tip_gap_mm": round(float(d.mean() * 1000), 2),
                    "winding_consistent": wind,
                })
    return out


LABELS = {   # 已知标注: (take, oid, side) -> [(a, b, label), ...]; 其余留空
    ("screw_unscrew_bottle_cap/0", "object_0", "left"): [(42, 61, "positive"), (107, 128, "negative_release")],
    ("screw_unscrew_bottle_cap/0", "object_0", "right"): [(0, 10 ** 9, "negative_not_grasp")],
    ("screw_unscrew_bottle_cap/0", "object_1", "right"): [(19, 29, "positive")],
    ("pour/17", "object_0", "left"): [(68, 81, "positive")],
    ("pour/17", "object_1", "right"): [(31, 47, "positive")],
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("opposition_samples.csv"))
    a = ap.parse_args(argv)

    rows = []
    for t in a.takes:
        rs = rows_for(t)
        for x in rs:
            key = (x["take"], x["object_id"], x["side"])
            x["label"] = ""
            for lo, hi, lab in LABELS.get(key, []):
                if lo <= x["frame"] <= hi:
                    x["label"] = lab
                    break
        rows += rs
        print(f"  {t}: {len(rs)} 行")
    if not rows:
        print("没有可导出的行", file=sys.stderr)
        return 1
    cols = ["take", "object_id", "side", "frame", "label", "opposition",
            "gap3d_mm", "tip_gap_mm", "winding_consistent"]
    with a.out.open("w", newline="", encoding="utf-8") as fh:
        wri = csv.DictWriter(fh, fieldnames=cols)
        wri.writeheader()
        wri.writerows(rows)
    lab = {}
    for x in rows:
        lab[x["label"] or "(未标注)"] = lab.get(x["label"] or "(未标注)", 0) + 1
    print(f"  -> {a.out}  共 {len(rows)} 行  {lab}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
