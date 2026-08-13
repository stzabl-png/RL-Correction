#!/usr/bin/env python3
"""把 v17A 自动标注的 mask 叠回原视频, 输出 mp4。

用 manifest 里的 raw_mask(传播原始输出), 不是 mask 字段 —— 后者在被质量门拒掉的帧上是空的,
用它会看不出"传播到底有没有跟住"。逐帧标注状态, 空 mask 的帧打红字, 一眼能看出从哪帧跟丢。
"""
import json, sys, os, cv2, numpy as np
manifest, video, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = json.load(open(manifest))
# frame_idx -> {oid: (raw_mask_path, status)}
per = {}
for fr in d["frames"]:
    fi = int(fr["frame_idx"])
    for oid, o in fr.get("objects", {}).items():
        if isinstance(o, dict):
            per.setdefault(fi, {})[oid] = (o.get("raw_mask"), o.get("status"))
cap = cv2.VideoCapture(video)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
COLORS = {"instance_0001": (0, 0, 255), "instance_0002": (0, 200, 255),
          "instance_0003": (255, 140, 0)}
n_empty = n_has = n_none = 0
for i in range(N):
    ok, im = cap.read()
    if not ok: break
    labs = []
    if i not in per:
        labs.append("(manifest 无此帧)"); n_none += 1
    for oid, (p, stt) in sorted(per.get(i, {}).items()):
        a = None
        if p and p not in ("None", None) and os.path.isfile(p):
            m = cv2.imread(p, 0)
            if m is not None:
                if m.shape[:2] != (H, W):
                    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                a = m > 127
        if a is None or a.sum() == 0:
            labs.append(f"{oid}: EMPTY"); n_empty += 1
        else:
            c = COLORS.get(oid, (255, 0, 255))
            im[a] = (0.5 * im[a] + 0.5 * np.array(c)).astype(np.uint8)
            cnts, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cnts, -1, c, 2)
            labs.append(f"{oid}: {int(a.sum())}px {stt or ''} n={len(cnts)}")
            n_has += 1
    cv2.rectangle(im, (0, 0), (W, 52), (18, 18, 18), -1)
    cv2.putText(im, f"f{i}/{N}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
    bad = any("EMPTY" in s or "无此帧" in s for s in labs)
    cv2.putText(im, " | ".join(labs)[:88], (8, 42), cv2.FONT_HERSHEY_SIMPLEX, .5,
                (0, 80, 255) if bad else (140, 255, 140), 1)
    vw.write(im)
vw.release(); cap.release()
print(f"  帧总数 {N} | 有 mask {n_has} | 空 mask {n_empty} | manifest 缺帧 {n_none}")
print(f"  -> {out}")
