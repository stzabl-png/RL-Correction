#!/usr/bin/env python
"""C2 验收: 区域条件合成 vs 原版 Dexonomy —— L2/L3/L4 三层对照。

L2 落区: 每个合成抓取的 obj_cpn_w → 物体规范系, 落在 region.npz 邻域的比例。
        两臂都在"后门过滤前"的全量上算(grasp_data + *_out_of_region 合并),
        否则条件臂被自己的过滤器美化。
L3 抓法: 合成抓取实际用到的指头(hand_cbody 命名解析) vs 选择器 plan 的目标
        (target_digits / palm) —— 量化"用了人的抓法没有"。
L4 真值: 合成抓取指集合 vs ARCTIC GT 抓握模式指集合(Jaccard) + 掌一致。
        GT 只当考卷。

用法: python grasp_accept_eval.py --pairs <obj>:<exp_region>:<exp_plain>:<region.npz>:<take_dir>:<gt_take> ...
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def load_grasps(exp: Path, include_rejected: bool):
    pats = ["grasp_data"] + (["grasp_data_out_of_region"] if include_rejected else [])
    fs = []
    for p in pats:
        fs += glob.glob(str(exp / p / "**" / "*.npy"), recursive=True)
    return [np.load(f, allow_pickle=True).item() for f in sorted(fs)]


def in_region_frac(d, tree, radius):
    pose, scale = np.asarray(d["obj_pose"], float), np.asarray(d["obj_scale"], float)
    Rm = R.from_quat(np.roll(pose[3:7], -1)).as_matrix()
    p = ((np.asarray(d["obj_cpn_w"], float)[:, :3] - pose[:3]) @ Rm) / scale
    dist, _ = tree.query(p)
    return float((dist <= radius).mean())


def grasp_fingers(d):
    """从接触 body 名(right_index_DP / right_hand_C_MC)解析指集合+掌。"""
    bodies = d.get("ho_c", {}).get("bn1", d.get("hand_cbody", []))
    fset, palm = set(), False
    for b in np.atleast_1d(bodies):
        b = str(b)
        if "hand_C_MC" in b or "palm" in b.lower():
            palm = True
            continue
        for f in FINGERS:
            if f"_{f}_" in b:
                fset.add(f)
    return fset, palm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="obj:exp_region:exp_plain:region_npz:take_dir:gt_take")
    ap.add_argument("--gt-root", type=Path,
                    default=Path("/home/lyh/Project/Reconstruct_and_Retarget/Data/arctic_gt_contacts"))
    ap.add_argument("--out", type=Path, default=Path("grasp_accept_report.json"))
    a = ap.parse_args()

    rows = []
    for spec in a.pairs:
        obj, e_reg, e_pla, rnpz, take_dir, gt_take = spec.split(":")
        rz = np.load(rnpz)
        pts = np.asarray(rz["points"])[np.asarray(rz["weight"]) >= float(rz["min_weight"])]
        tree, radius = cKDTree(pts), float(rz["radius"])
        plan_p = Path(take_dir) / "grasp_template_plan.json"
        plan = json.loads(plan_p.read_text())["hands"]["right"] if plan_p.is_file() else None
        gm = json.loads((a.gt_root / gt_take / "gt_summary.json").read_text())[
            "hands"]["right"]["tau_5mm"]["grasp_mode"]
        gt_f, gt_palm = (set(gm["fingers"]), gm["palm_contact_frac"] >= 0.5) if gm else (set(), False)

        row = {"object": obj}
        for arm, exp in (("region", Path(e_reg)), ("plain", Path(e_pla))):
            gs = load_grasps(exp, include_rejected=True)
            if not gs:
                row[arm] = {"n": 0}
                continue
            fr = [in_region_frac(d, tree, radius) for d in gs]
            l3 = l3n = 0
            l4j, l4p = [], []
            for d in gs:
                fset, palm = grasp_fingers(d)
                if plan:
                    tgt, tol = plan["target_digits"], plan["tolerance"]
                    l3 += (abs(len(fset) - tgt) <= tol and palm == plan["vlm"]["palm_contact"])
                    l3n += 1
                if gt_f:
                    l4j.append(len(fset & gt_f) / max(1, len(fset | gt_f)))
                    l4p.append(palm == gt_palm)
            row[arm] = {"n": len(gs),
                        "L2_in_region_mean": round(float(np.mean(fr)), 3),
                        "L2_frac_ge60": round(float(np.mean([f >= 0.6 for f in fr])), 3),
                        "L3_plan_match": round(l3 / l3n, 3) if l3n else None,
                        "L4_gt_jaccard": round(float(np.mean(l4j)), 3) if l4j else None,
                        "L4_palm_ok": round(float(np.mean(l4p)), 3) if l4p else None}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    a.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
