#!/usr/bin/env python
"""Phase B: 我们的 contact 提取 vs ARCTIC GT —— 指级对拍 + 误差账单的数据底座。

对拍口径(全部指级, 不需要网格对应关系):
  * 逐指查准/查全: 在**我们链路自选的提取帧**上, 我们判接触的指集合 vs GT 指集合
    (公平性: 评的是链路交付的东西, 含它自己的选帧);
  * 指数误差: |n_ours − n_GT|;
  * 掌部: 我们的链路结构性测不到(指垫只在指尖) —— 显式报 GT 掌参与率供账单引用;
  * 时间线: 我们的接触区间(视频帧) ↔ GT 接触段(arctic_vidx), 经 meta.index 映射后算 IoU;
  * 阴性检验: GT 无接触的手(scissors 左), 我们有没有幻觉出接触。
τ 纪律: GT 侧 3/5/8/10mm 全套都报, 不挑对我们有利的。

用法:
  python contact_vs_gt.py [--takes all] \
      [--ours <ReconstructOutput/arctic15 本地镜像>] [--gt <arctic_gt_contacts>]
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PAD_MIN_VERTS = 150
TAUS = (3, 5, 8, 10)
TAKES = ["s05__laptop_grab_01", "s07__ketchup_grab_01", "s05__box_grab_01",
         "s04__capsulemachine_grab_01", "s05__espressomachine_grab_01",
         "s05__microwave_grab_01", "s05__mixer_grab_01", "s06__notebook_grab_01",
         "s10__phone_grab_01", "s02__scissors_grab_01", "s05__waffleiron_grab_01"]


def ours_fingers(take_dir: Path, side: str):
    """(接触指集合, 提取帧) —— 与 select_grasp_template.geometric_fingers 同口径。"""
    hits = sorted(glob.glob(str(take_dir / "contact" / f"contact_heatmap_frame*_{side}*.npz")))
    if not hits:
        return None, None
    z = np.load(hits[0], allow_pickle=True)
    w, near = z["vertex_weight"], z["nearest_pad_link"]
    names = [str(x) for x in z["pad_link_names"]]
    hot = w > 0.5
    pads = {names[i].split("_")[1] for i in range(len(names))
            if int(((near == i) & hot).sum()) >= PAD_MIN_VERTS}
    frame = int(Path(hits[0]).name.split("frame")[1][:4])
    return pads, frame


def ours_intervals(take_dir: Path, side: str):
    ca = take_dir / "contact_auto.json"
    if not ca.is_file():
        return []
    return json.loads(ca.read_text())["annotations"].get(side) or []


def to_vidx(frame: int, index: list) -> int | None:
    for row in index:
        if row["video_frame"] == frame:
            return row["arctic_vidx"]
    return None


def temporal_iou(ours_v: set, gt_v: set) -> float:
    if not ours_v and not gt_v:
        return 1.0
    u = ours_v | gt_v
    return len(ours_v & gt_v) / len(u) if u else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", type=Path,
                    default=Path("/home/lyh/Project/Reconstruct_and_Retarget/Data/arctic15_ours"))
    ap.add_argument("--gt", type=Path,
                    default=Path("/home/lyh/Project/Reconstruct_and_Retarget/Data/arctic_gt_contacts"))
    ap.add_argument("--meta", type=Path,
                    default=Path("/home/lyh/Project/Reconstruct_and_Retarget/Data/arctic15_local"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    rows, pooled = [], {t: {"tp": 0, "fp": 0, "fn": 0} for t in TAUS}
    for take in TAKES:
        subj, seq = take.split("__", 1)
        td = a.ours / subj / seq
        gtz = np.load(a.gt / take / "gt_contact.npz", allow_pickle=True)
        meta_p = a.meta / f"{take}.meta.json"
        if not meta_p.is_file():
            print(f"⚠ 缺 meta: {take}")
            continue
        index = json.loads(meta_p.read_text())["index"]

        for si, side in enumerate(("left", "right")):
            pads, frame = ours_fingers(td, side)
            row = {"take": take, "side": side, "our_frame": frame}
            if pads is None:
                row["status"] = "no_extraction"
                # GT 在整段里有没有抓握(漏检检验)
                gt_any = bool(gtz["contact_5mm"][:, si, :5].any())
                row["gt_has_contact"] = gt_any
                row["error_type"] = "miss" if gt_any else "true_negative"
            else:
                vidx = to_vidx(frame, index)
                if vidx is None or vidx >= len(gtz["dists_m"]):
                    row["status"] = "frame_unmapped"
                else:
                    row["status"] = "ok"
                    row["ours"] = sorted(pads)
                    row["n_ours"] = len(pads)
                    for tau in TAUS:
                        gt_set = {FINGERS[i] for i in range(5)
                                  if gtz[f"contact_{tau}mm"][vidx, si, i]}
                        tp = len(pads & gt_set); fp = len(pads - gt_set); fn = len(gt_set - pads)
                        row[f"gt_{tau}mm"] = sorted(gt_set)
                        row[f"prf_{tau}mm"] = [tp, fp, fn]
                        pooled[tau]["tp"] += tp; pooled[tau]["fp"] += fp; pooled[tau]["fn"] += fn
                    row["gt_palm_5mm"] = bool(gtz["contact_5mm"][vidx, si, 5])
                # 时间线 IoU(5mm): 我们的区间帧→vidx 集合 vs GT 接触帧集合
                ours_v = set()
                for s, e in ours_intervals(td, side):
                    for f in range(s, e + 1):
                        v = to_vidx(f, index)
                        if v is not None:
                            ours_v.add(v)
                gt_v = set(np.where(gtz["contact_5mm"][:, si, :5].any(1))[0].tolist())
                row["interval_iou_5mm"] = round(temporal_iou(ours_v, gt_v), 3)
            rows.append(row)

    print(f"{'take':30s} {'手':5s} {'我们':28s} {'GT@5mm':28s} 帧    IoU")
    for r in rows:
        if r.get("status") == "ok":
            print(f"{r['take']:30s} {r['side']:5s} {','.join(r['ours']):28s} "
                  f"{','.join(r['gt_5mm']):28s} f{r['our_frame']:<4d} {r['interval_iou_5mm']}")
        else:
            print(f"{r['take']:30s} {r['side']:5s} [{r['status']}] "
                  f"{'(GT确有接触→漏检!)' if r.get('error_type')=='miss' else '(GT也无→正确阴性)' if r.get('error_type')=='true_negative' else ''}")
    print("\n══ 指级汇总(逐 τ):")
    for tau in TAUS:
        s = pooled[tau]
        P = s["tp"] / max(1, s["tp"] + s["fp"]); R = s["tp"] / max(1, s["tp"] + s["fn"])
        print(f"  τ={tau}mm: precision {P:.2f}  recall {R:.2f}  (tp{s['tp']}/fp{s['fp']}/fn{s['fn']})")
    ious = [r["interval_iou_5mm"] for r in rows if "interval_iou_5mm" in r]
    print(f"  区间 IoU 中位: {np.median(ious):.2f}" if ious else "")
    out = a.out or (a.gt / "contact_vs_gt_report.json")
    out.write_text(json.dumps({"rows": rows, "pooled": pooled}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
