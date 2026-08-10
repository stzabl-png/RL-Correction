#!/usr/bin/env python
"""可信度判据的人眼校准表 —— 抽样出"工具说好"和"工具说坏"的帧, 并排给人核对。

## 为什么需要这一步

到目前为止 `pose_audit.py` 的所有阈值都是拍出来的, 一次人眼校验都没做过。
而在新的路线里(不修正、只判可信度、喂给 RL 当参考权重), **可信度标错比没有可信度更糟** ——
没有时 RL 一视同仁, 标错时它会去认真模仿一段垃圾。所以判据必须先过人眼。

## 做法

不是让人看整段视频(慢且抓不住重点), 而是**分层抽样出对照表**:
  左半张 = 工具判"好"的帧    右半张 = 工具判"坏"的帧
每格都裁到物体附近放大, 画 mask(青) 和 投影(品红) 两条轮廓, 配上判据数值。

人眼只需回答两个问题:
  1. 左边这些, 两条轮廓是不是真的贴合?   (若不贴 => 判据漏报, 阈值太松)
  2. 右边这些, 是不是真的错得离谱?       (若其实没问题 => 判据误报, 阈值太严)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_audit import decimate, load_obj_mask, silhouette  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
COL_MASK = (255, 200, 0)      # 青
COL_PROJ = (200, 60, 235)     # 品红
CELL, CAP = 300, 46


def resolve_video(take: Path) -> Path | None:
    """take 目录 -> 原视频。egodex 三种布局各有各的根, 编号体系不通用。"""
    parts = take.parts
    name, task = parts[-1], parts[-2]
    cands = []
    if "part2" in parts:
        cands.append(Path("/home/lyh/Project/Reconstruct_and_Retarget/Data/Egodex_Part2/part2")
                     / task / f"{name}.mp4")
    base = Path("/home/lyh/Project/V2AP/data/egocentric/egodex/test")
    cands += [base / task / f"{name}.mp4",
              base / task / f"{name.split('_')[0]}.mp4"]
    for c in cands:
        if c.is_file():
            return c
    return None


def crop_cell(frame, mk, pj, pad=0.45):
    """裁到 mask∪投影 的外接框(带边距), 画两条轮廓, 归一到 CELL×CELL。"""
    u = mk | pj
    if not u.any():
        return None
    ys, xs = np.nonzero(u)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    s = int(max(w, h) * (1 + 2 * pad))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    X0, Y0 = cx - s // 2, cy - s // 2
    H, W = frame.shape[:2]
    canvas = np.zeros((s, s, 3), np.uint8)
    sx0, sy0 = max(0, X0), max(0, Y0)
    sx1, sy1 = min(W, X0 + s), min(H, Y0 + s)
    if sx1 <= sx0 or sy1 <= sy0:
        return None
    canvas[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = frame[sy0:sy1, sx0:sx1]
    for m, col in ((mk, COL_MASK), (pj, COL_PROJ)):
        sub = m[sy0:sy1, sx0:sx1].astype(np.uint8)
        big = np.zeros((s, s), np.uint8)
        big[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = sub
        cs, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cs, -1, col, 2, cv2.LINE_AA)
    return cv2.resize(canvas, (CELL, CELL), interpolation=cv2.INTER_AREA)


def pick(per, want_low, k, thr_hi=70, thr_lo=40):
    """按**可信度**分组抽帧(旧版按 modes 分, 但假阳性判据删掉后大部分帧没有 modes)。

    高可信组用来验"说可信的真的准", 低可信组用来验"说不可信的真的有问题"。
    分组为空时退化成按 conf 取头/尾 k 个 —— 保证表永远有东西可看。
    """
    pool = [r for r in per if (r["conf"] < thr_lo) if want_low] if want_low else \
           [r for r in per if r["conf"] >= thr_hi]
    if not pool:
        srt = sorted(per, key=lambda r: r["conf"])
        pool = srt[:max(k, 1)] if want_low else srt[-max(k, 1):]
    pool = sorted(pool, key=lambda r: r["frame"])
    idx = np.linspace(0, len(pool) - 1, min(k, len(pool))).astype(int)
    return [pool[i] for i in sorted(set(idx))]


def build(take: Path, audit: dict, out: Path, k=6, ss=1):
    z = np.load(take / "world_fused.npz", allow_pickle=True)
    K = np.asarray(z["K"], float)
    Tc = np.asarray(z["object_ob_in_cam"], float)
    meshes = (sorted(glob.glob(str(take / "objects" / "*" / "*.obj")))
              or sorted(glob.glob(str(take / "*.obj"))))
    mesh = trimesh.load(meshes[0], force="mesh")
    V0, F0 = decimate(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int))
    s = audit["mesh_scale_fitted"]
    C = V0.mean(0); V = (V0 - C) * s + C
    vid = resolve_video(take)
    cap = cv2.VideoCapture(str(vid)) if vid else None

    good, bad = pick(audit["per_frame"], False, k), pick(audit["per_frame"], True, k)
    ro = audit.get("rot_observability")
    rows = max(len(good), len(bad))
    if rows == 0:
        return None
    sheet = np.full((rows * (CELL + CAP) + 60, 2 * CELL + 30, 3), 22, np.uint8)
    cv2.putText(sheet, "HIGH CONFIDENCE", (CELL // 2 - 78, 26), FONT, 0.62, (120, 255, 120), 2, cv2.LINE_AA)
    if ro is not None:
        cv2.putText(sheet, f"rot-observability {min(ro):.3f}"
                    f"{'  => ROTATION UNOBSERVABLE' if min(ro) < 0.10 else ''}",
                    (8, sheet.shape[0] - 8), FONT, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(sheet, "LOW CONFIDENCE", (CELL + 30 + CELL // 2 - 72, 26), FONT, 0.62, (90, 90, 255), 2, cv2.LINE_AA)

    for col, group in enumerate((good, bad)):
        for r, rec in enumerate(group):
            i = rec["frame"]
            mk = load_obj_mask(take, i)
            if mk is None:
                continue
            pj = silhouette(V, F0, K, Tc[i], mk.shape, ss)
            frame = np.zeros((*mk.shape, 3), np.uint8)
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, f = cap.read()
                if ok:
                    frame = cv2.resize(f, (mk.shape[1], mk.shape[0]))
            cell = crop_cell(frame, mk, pj)
            if cell is None:
                continue
            x = col * (CELL + 30)
            y = 40 + r * (CELL + CAP)
            sheet[y:y + CELL, x:x + CELL] = cell
            cap_txt = [f"f{i}  CONF={rec['conf']}  (pos {rec['conf_pos']} / rot {rec['conf_rot']})",
                       f"exp={rec['explained']:.2f} dc={rec['d_cent_norm']} occl={rec['occl']:.2f}"
                       f"  {','.join(rec['modes'] + rec.get('soft', [])) or '-'}"]
            col_txt = ((90, 90, 255) if rec["conf"] < 40 else
                       (150, 230, 150) if rec["conf"] >= 70 else (170, 200, 255))
            for j, t in enumerate(cap_txt):
                cv2.putText(sheet, t, (x + 4, y + CELL + 18 + j * 18), FONT, 0.42,
                            col_txt, 1, cv2.LINE_AA)
    if cap is not None:
        cap.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet)
    return {"take": str(take), "video": str(vid) if vid else None,
            "n_good_shown": len(good), "n_bad_shown": len(bad),
            "conf_median": audit["conf_median"], "scale": s,
            "scale_reliable": audit.get("scale_reliable", True), "out": str(out)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--take", action="append", required=True,
                    help="ReconstructOutput 下的相对路径, 可重复")
    ap.add_argument("--k", type=int, default=6)
    a = ap.parse_args(argv)
    doc = json.loads(a.audit.read_text())
    by = {t["take"]: t for t in doc["takes"] if "error" not in t}
    root = "/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput"
    for rel in a.take:
        full = f"{root}/{rel}"
        if full not in by:
            print(f"  X 审计里没有 {rel}")
            continue
        r = build(Path(full), by[full], a.out / f"calib_{rel.replace('/', '__')}.png", a.k)
        if r:
            print(f"  OK {rel:46s} conf中位{r['conf_median']:5.1f}  s={r['scale']:.3f}"
                  f"{'' if r['scale_reliable'] else '(不可信)'}  视频={'有' if r['video'] else '无'}"
                  f"  -> {Path(r['out']).name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
