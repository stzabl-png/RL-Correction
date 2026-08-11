#!/usr/bin/env python3
"""视频预检 —— 任何进入重建管线的视频, 统一丢弃开头的连续暗帧(相机自动曝光预热)。

规则(2026-08-10 从 arctic 单条数据上提为通用, 用户裁定"任何视频开头是黑的都不要那些帧"):
  * 只砍**开头连续**低亮度段 —— 中段暗帧是真实内容(手挡镜头), 砍了会让时间轴
    与数据集标注静默错位, 永远保留;
  * 全程记账: 每条被裁的视频旁写 <名字>.preflight.json(丢了几帧/各帧亮度/阈值),
    拿这条数据对任何外部真值时, 帧号 = 原视频帧号 - leading_dark_frames_dropped;
  * 干净视频零开销: staging 镜像里放硬链接, 不复制不转码。

工作方式(reconstruct.sh 自动调用):
  读清单 + 数据根 → 在 staging 根下按**相同相对路径**重建镜像(保 video_id 不变):
  开头无暗帧 → 硬链接原文件; 有 → 重编码裁剪副本 + sidecar。清单原地改写为镜像路径。

  python3 video_preflight.py --list selected.txt --root <数据根> --staging <镜像根>
                             [--dark-thresh 40] [--probe 60]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

DARK_THRESH = 40.0     # 与 arctic_ego_video.py 同源: s05 实测暗帧 18.8 vs 正常 97


def leading_dark_run(video: Path, thresh: float, probe: int) -> tuple[int, list[float]]:
    """开头连续亮度<thresh 的帧数(最多探测 probe 帧), 以及这些帧的亮度。"""
    import cv2

    cap = cv2.VideoCapture(str(video))
    n, means = 0, []
    for _ in range(probe):
        ok, img = cap.read()
        if not ok:
            break
        m = float(img.mean())
        if m >= thresh:
            break
        means.append(round(m, 1))
        n += 1
    cap.release()
    return n, means


def trim_copy(video: Path, out: Path, n_skip: int) -> int:
    """丢开头 n_skip 帧的重编码副本, 返回输出帧数。mp4v 与管线各步一致。"""
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = kept = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i >= n_skip:
            vw.write(img)
            kept += 1
        i += 1
    cap.release()
    vw.release()
    return kept


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", type=Path, required=True, help="视频绝对路径清单, 原地改写")
    ap.add_argument("--root", type=Path, required=True, help="原数据根(算相对路径用)")
    ap.add_argument("--staging", type=Path, required=True, help="镜像根(硬链接/裁剪副本)")
    ap.add_argument("--dark-thresh", type=float, default=DARK_THRESH)
    ap.add_argument("--probe", type=int, default=60, help="最多探测开头多少帧")
    a = ap.parse_args(argv)

    lines = [l.strip() for l in a.list.read_text().splitlines() if l.strip()]
    out_lines, n_trim = [], 0
    for src_s in lines:
        src = Path(src_s)
        if not src.is_file():
            out_lines.append(src_s)          # generic id 等非文件行原样保留
            continue
        try:
            rel = src.resolve().relative_to(a.root.resolve())
        except ValueError:
            rel = Path(src.name)             # root 之外的散视频: 平铺到镜像根
        dst = a.staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        n_skip, means = leading_dark_run(src, a.dark_thresh, a.probe)
        if n_skip == 0:
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            try:
                os.link(src, dst)            # 零拷贝
            except OSError:
                import shutil
                shutil.copy2(src, dst)       # 跨文件系统兜底
        else:
            side = dst.with_suffix(dst.suffix + ".preflight.json")
            if not (dst.is_file() and side.is_file()):   # 幂等: 已裁过就复用
                kept = trim_copy(src, dst, n_skip)
                side.write_text(json.dumps({
                    "source": str(src), "leading_dark_frames_dropped": n_skip,
                    "dropped_frame_means": means, "dark_thresh": a.dark_thresh,
                    "frames_kept": kept,
                    "note": "对外部真值时: 原视频帧号 = 本视频帧号 + dropped",
                }, ensure_ascii=False, indent=1), encoding="utf-8")
            n_trim += 1
            print(f"[preflight] {rel}: 丢开头 {n_skip} 暗帧(亮度 {means}) -> {dst}")
        out_lines.append(str(dst))
    a.list.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[preflight] {len(lines)} 条视频: 裁剪 {n_trim}, 硬链接 {len(lines) - n_trim}; "
          f"清单已指向镜像 {a.staging}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
