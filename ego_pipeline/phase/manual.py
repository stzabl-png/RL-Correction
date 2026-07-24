"""人工标注 provider:读 tools/annotate_grasp_frames.py 产出的 grasp_annotation.json。

grasp_annotation.json 格式:
  {"video": ..., "num_frames": N, "fps": f,
   "annotations": {"left": [[s,e], ...], "right": [[s,e], ...]}}
每个 [s,e] = 一段抓取的 [开始帧, 结束帧](含端点,视频/Tv 时间线)。

同学的自动接触检测就绪后,不改这里 —— 改用 auto provider(见 auto.py)。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import PhaseTracks, segments_to_dense

ANNOTATION_NAME = "grasp_annotation.json"


def find_annotation(take_dir=None, *, annotation_path=None, video_dir=None) -> Path | None:
    """定位 grasp_annotation.json。按优先级:显式路径 > take 目录 > 视频目录 > 从
    world_fused.npz 的 video_id 猜测视频目录。找不到返回 None。"""
    if annotation_path:
        p = Path(annotation_path)
        return p if p.is_file() else None
    for base in (take_dir, video_dir):
        if base:
            c = Path(base) / ANNOTATION_NAME
            if c.is_file():
                return c
    if take_dir:
        wf = Path(take_dir) / "world_fused.npz"
        if wf.is_file():
            try:
                vid = str(np.load(wf, allow_pickle=True)["video_id"])
                vp = Path(vid)
                for cand in (vp if vp.is_dir() else vp.parent,):
                    c = cand / ANNOTATION_NAME
                    if c.is_file():
                        return c
            except Exception:
                pass
    return None


class ManualGraspPhaseProvider:
    name = "manual"

    def available(self, take_dir=None, *, annotation_path=None, video_dir=None) -> bool:
        return find_annotation(take_dir, annotation_path=annotation_path, video_dir=video_dir) is not None

    def load(self, take_dir=None, num_frames=None, *, annotation_path=None, video_dir=None) -> PhaseTracks:
        path = find_annotation(take_dir, annotation_path=annotation_path, video_dir=video_dir)
        if path is None:
            raise FileNotFoundError(f"未找到 {ANNOTATION_NAME}(take_dir={take_dir})")
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        ann = doc.get("annotations", {}) or {}
        n_annot = int(doc.get("num_frames") or 0)
        n = n_annot or (int(num_frames) if num_frames else 0)
        if n <= 0:
            raise ValueError(f"{path}: 无法确定 num_frames")
        tracks = PhaseTracks(
            left=segments_to_dense(ann.get("left"), n),
            right=segments_to_dense(ann.get("right"), n),
            num_frames=n,
            fps=doc.get("fps"),
            source="manual",
            meta={"annotation": str(path), "segments": {"left": ann.get("left", []),
                                                        "right": ann.get("right", [])}},
        )
        # 标注帧数 != 目标重建帧数时,缩放到目标(通常两者相等)。
        if num_frames and int(num_frames) != tracks.num_frames:
            tracks = tracks.resample(int(num_frames))
        return tracks
