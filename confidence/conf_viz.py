#!/usr/bin/env python
"""可信度叠加视频 —— 原视频 + FP 位姿投影 + 逐帧可信度, 一眼看出哪段可信哪段不可信。

画面:
  投影轮廓   按 conf 上色: 绿(≥70) → 黄(40~70) → 红(<40)
  2D mask    青色细线(参照物, 轮廓和它贴不贴一目了然)
  HUD        conf / conf_pos / conf_rot 数字条 + 当帧判据(遮挡/dc/CT信号)
  底部       conf_pos(绿) 与 conf_rot(品红) 两条曲线 0~100 + 播放头;
             CT 覆盖段画在轴下方(青)

数据一律来自 pose_audit.json —— 视频里显示的就是打分器给的分, 不另算。
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
from pose_calib_sheet import resolve_video  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
STRIP_H = 96
LABEL_W = 70


def conf_color(c):
    """0~100 -> BGR: 红(0) → 黄(50) → 绿(100)。"""
    c = float(np.clip(c, 0, 100)) / 100.0
    if c < 0.5:
        t = c / 0.5
        return (int(40 + 20 * t), int(40 + 190 * t), 230)          # 红->黄
    t = (c - 0.5) / 0.5
    return (int(60 - 20 * t), 230, int(230 - 170 * t))             # 黄->绿


def build_strip(W, n, pf, ct_frames):
    strip = np.full((STRIP_H, W, 3), 24, np.uint8)
    x0, x1, y0, y1 = LABEL_W, W - 10, 8, STRIP_H - 26
    cv2.rectangle(strip, (x0, y0), (x1, y1), (44, 44, 44), -1)
    for v, lab in ((100, "100"), (70, "70"), (30, "30"), (0, "0")):
        y = int(y1 - v / 100 * (y1 - y0))
        cv2.line(strip, (x0, y), (x1, y), (70, 70, 70), 1)
        cv2.putText(strip, lab, (8, y + 4), FONT, 0.32, (130, 130, 130), 1, cv2.LINE_AA)

    def X(i):
        return int(x0 + i / max(1, n - 1) * (x1 - x0))

    for key, col in (("conf_pos", (120, 255, 120)), ("conf_rot", (230, 90, 230))):
        pts = [(X(r["frame"]), int(y1 - r[key] / 100 * (y1 - y0)))
               for r in pf if key in r]
        if len(pts) > 1:
            cv2.polylines(strip, [np.array(pts, np.int32)], False, col, 1, cv2.LINE_AA)
    for i in ct_frames:
        cv2.line(strip, (X(i), y1 + 3), (X(i), y1 + 8), (230, 200, 60), 1)
    cv2.putText(strip, "conf_pos", (LABEL_W, STRIP_H - 6), FONT, 0.38, (120, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(strip, "conf_rot", (LABEL_W + 92, STRIP_H - 6), FONT, 0.38, (230, 90, 230), 1, cv2.LINE_AA)
    cv2.putText(strip, "CT coverage", (LABEL_W + 184, STRIP_H - 6), FONT, 0.38, (230, 200, 60), 1, cv2.LINE_AA)
    return strip, X, (y0, y1)


def render(take: Path, audit: dict, out: Path, scale=0.5, ct_dir: Path | None = None):
    z = np.load(take / "world_fused.npz", allow_pickle=True)
    K = np.asarray(z["K"], float)
    Tc = np.asarray(z["object_ob_in_cam"], float)
    meshes = (sorted(glob.glob(str(take / "objects" / "*" / "*.obj")))
              or sorted(glob.glob(str(take / "*.obj"))))
    mesh = trimesh.load(meshes[0], force="mesh")
    V0, F0 = decimate(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int))
    s = audit["mesh_scale_fitted"]
    C = V0.mean(0)
    V = (V0 - C) * s + C

    pf = audit["per_frame"]
    by = {r["frame"]: r for r in pf}

    ct_frames = []
    if ct_dir is not None:
        ccp = Path(ct_dir) / f"cc_{take.parent.name}_{take.name}.json"
        if ccp.is_file():
            ct_frames = [r["frame"] for r in json.loads(ccp.read_text())["per_frame"]
                         if r.get("n_vis", 0) >= 6]
    vid = resolve_video(take)
    if vid is None:
        return {"take": str(take), "error": "video not found"}
    cap = cv2.VideoCapture(str(vid))
    n = int(min(cap.get(cv2.CAP_PROP_FRAME_COUNT), len(Tc)))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale) // 2 * 2
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale) // 2 * 2

    strip, X, (sy0, sy1) = build_strip(W, n, pf, ct_frames)
    out.mkdir(parents=True, exist_ok=True)
    vp = out / f"conf_{take.parent.name}_{take.name}.mp4"
    vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + STRIP_H))

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        r = by.get(i)
        # mask 参照(青细线)
        mk = load_obj_mask(take, i)
        if mk is not None:
            ms = cv2.resize(mk.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            cs, _ = cv2.findContours(ms, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(small, cs, -1, (255, 200, 0), 1, cv2.LINE_AA)
        # 位姿投影, 按 conf 上色
        conf = r["conf"] if r else 0
        col = conf_color(conf)
        sil = silhouette(V, F0, K, Tc[i], mk.shape if mk is not None else (1080, 1920), 1)
        ss2 = cv2.resize(sil.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        cs, _ = cv2.findContours(ss2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(small, cs, -1, col, 3, cv2.LINE_AA)

        # HUD
        if r:
            lines = [
                (f"frame {i:04d}/{n-1}   t={i/fps:5.2f}s", (210, 210, 210)),
                (f"CONF     {r['conf']:3d}", conf_color(r["conf"])),
                (f"pos {r['conf_pos']:3d}   rot {r['conf_rot']:3d}",
                 (200, 200, 200)),
                (f"occl {r.get('occl', 0):.2f}  dc {r.get('d_cent_norm')}  "
                 f"{','.join(r.get('modes', []) + r.get('soft', [])) or '-'}",
                 (160, 160, 160)),
            ]
        else:
            lines = [(f"frame {i:04d}  (no score)", (140, 140, 140))]
        x, y = 10, 10
        hh = 19 * len(lines) + 10
        ww = 8 + max(int(cv2.getTextSize(t, FONT, 0.5, 1)[0][0]) for t, _ in lines) + 12
        small[y:y+hh, x:x+ww] = (small[y:y+hh, x:x+ww].astype(np.float32) * 0.35).astype(np.uint8)
        for j, (t, c) in enumerate(lines):
            cv2.putText(small, t, (x + 6, y + 19 + j * 19), FONT, 0.5, c, 1, cv2.LINE_AA)
        # conf 色条
        cv2.rectangle(small, (x + ww + 8, y), (x + ww + 26, y + hh), (60, 60, 60), -1)
        fill = int(hh * conf / 100)
        cv2.rectangle(small, (x + ww + 8, y + hh - fill), (x + ww + 26, y + hh), col, -1)

        sp = strip.copy()
        cv2.line(sp, (X(i), sy0), (X(i), sy1), (255, 255, 255), 1)
        vw.write(np.vstack([small, sp]))
    cap.release()
    vw.release()
    return {"take": str(take), "out": str(vp), "n": n}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--take", action="append", required=True)
    ap.add_argument("--ct-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    doc = json.loads(a.audit.read_text())
    root = "/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput"
    by = {t["take"]: t for t in doc["takes"] if "error" not in t}
    for rel in a.take:
        t = by.get(f"{root}/{rel}")
        if not t:
            print(f"  X 审计缺 {rel}")
            continue
        r = render(Path(f"{root}/{rel}"), t, a.out, ct_dir=a.ct_dir)
        print(f"  {'OK' if 'out' in r else 'X'} {rel}  -> {r.get('out', r.get('error'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
