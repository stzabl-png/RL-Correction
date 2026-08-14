#!/usr/bin/env python
"""contact_fingers.json(最终交付口径) vs ARCTIC GT —— 逐指准确率复测。

与 mano_vs_gt.py 的区别: 那个从距离时间线**离线重投票**扫参数网格(标定用);
这个直接读链路真正交付的 contact_fingers.json(含质检门、投票、贴合与否的全部
既定规则), 回答"我们发出去的答案对不对"。

用法:
  python fingers_vs_gt.py --ours <take根> [--takes <清单文件>] [--label 说明]
  # A/B: 同一批 take 的两份产物(如 align vs no-align)分别跑, 比 F1 与指数误差
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GT = Path("/home/lyh/Project/Reconstruct_and_Retarget/Data/arctic_gt_contacts")
DEV = {"box", "laptop", "ketchup", "mixer"}      # 标定物体, held-out 为其余


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", type=Path, required=True, help="take 根目录(下含 <subj>/<seq>/contact/)")
    ap.add_argument("--takes", type=Path, default=None, help="清单文件(每行 s01/box_grab_01)")
    ap.add_argument("--label", default="")
    ap.add_argument("--object-id", default="object_0")
    a = ap.parse_args(argv)

    takes = ([l.strip() for l in a.takes.read_text().split("\n") if l.strip()] if a.takes
             else sorted(f"{p.parent.parent.parent.name}/{p.parent.parent.name}"
                         for p in a.ours.glob("*/*/contact/contact_fingers.json")))
    acc = {g: {"tp": 0, "fp": 0, "fn": 0, "nerr": [], "exact": 0, "n": 0,
               "palm_ok": 0, "unreliable": 0} for g in ("dev", "held")}
    missing = 0
    for t in takes:
        key = t.replace("/", "__")
        cf_p = a.ours / t / "contact" / "contact_fingers.json"
        gt_p = GT / key / "gt_summary.json"
        if not cf_p.is_file() or not gt_p.is_file():
            missing += 1
            continue
        grp = "dev" if key.split("__")[1].split("_grab")[0] in DEV else "held"
        summ = json.loads(gt_p.read_text())["hands"]
        cf = json.loads(cf_p.read_text())["hands"]
        for side in ("left", "right"):
            gm = summ[side]["tau_5mm"]["grasp_mode"]
            r = ((cf.get(side) or {}).get("objects") or {}).get(a.object_id) or {}
            if not gm or r.get("status") != "ok":
                continue
            s = acc[grp]
            if (r.get("quality") or {}).get("reliable") is False:
                s["unreliable"] += 1
                continue
            ours, gt = set(r["fingers"]), set(gm["fingers"])
            s["tp"] += len(ours & gt); s["fp"] += len(ours - gt); s["fn"] += len(gt - ours)
            s["nerr"].append(abs(len(ours) - gm["n_fingers_median"]))
            s["exact"] += (ours == gt); s["n"] += 1
            s["palm_ok"] += (bool(r.get("palm")) == (gm["palm_contact_frac"] >= 0.5))
    print(f"══ {a.label or a.ours}  (缺数据 {missing} 条)")
    for grp in ("dev", "held"):
        s = acc[grp]
        if not s["n"]:
            print(f"  {grp}: 无数据"); continue
        P = s["tp"] / max(1, s["tp"] + s["fp"]); R = s["tp"] / max(1, s["tp"] + s["fn"])
        print(f"  {grp:5s} {s['n']:3d}手: P {P:.2f} R {R:.2f} F1 {2*P*R/max(1e-9,P+R):.2f} "
              f"| 指集合全对 {s['exact']}/{s['n']} ({s['exact']/s['n']:.0%}) "
              f"| 指数误差中位 {np.median(s['nerr']):.0f} "
              f"| 掌对 {s['palm_ok']}/{s['n']} ({s['palm_ok']/s['n']:.0%}) "
              f"| 质检拦下 {s['unreliable']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
