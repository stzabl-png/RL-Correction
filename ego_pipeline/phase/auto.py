"""自动接触检测 provider —— 占位契约,等同学的模块就绪后在此接入。

目标:同学研发好"自动标记接触"后,只需在这里把 available()/load() 实现好,
其余全项目(attach 工具、recon_to_replay、下游 RL)无需改动即可从人工切到自动。

===================== 接入契约 =====================
输入:一条 take
  - take_dir: Output/ReconstructOutput/<dataset>/<take>/,内含
      world_fused.npz  (hand_trans/hand_valid/hand_pose … 手轨迹 @Th,
                        object_ob_in_world 物体轨迹 @Tv,object mesh 路径)
  - num_frames: 目标视频/物体时间线长度 Tv(阶段数组应对齐到它)
  - video_dir: 原始视频/散帧目录(可选,若检测需要像素)

输出:phase.types.PhaseTracks
  - left / right: (Tv,) int8 逐帧阶段标签(FREE / CONTACT / 自定义扩展值)
  - num_frames = Tv, source = "auto"
  - 时间线 = 视频/物体帧 (Tv);若模型在手时间线 Th 上出结果,
    用 resample_labels(labels_Th, Tv) 降到 Tv 再构造 PhaseTracks。

推荐接入方式(二选一):
  A. 文件契约:同学的模块把结果写成 take_dir/contact_auto.json
     (结构建议与 grasp_annotation.json 一致,或直接逐帧 {"left":[...],"right":[...]}),
     这里读它 -> PhaseTracks。available() = 该文件存在。
  B. 直接调用:import 同学的函数,传 world_fused.npz / 视频,拿逐帧接触。
     available() = 依赖可导入。

细粒度:若自动版输出多于 2 类(接近/接触/操作/松手),在 types.py 的
PHASE_NAMES 里登记新整数标签即可,下游按 int 处理、不受影响。
====================================================
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import CONTACT, FREE, PhaseTracks, segments_to_dense

AUTO_RESULT_NAME = "contact_auto.json"   # 文件契约(方式 A)


def _find_result(take_dir=None, *, annotation_path=None, video_dir=None) -> Path | None:
    """定位 contact_auto.json。优先级:显式路径 > take 目录 > 视频目录。"""
    if annotation_path:
        p = Path(annotation_path)
        if p.is_file() and p.name == AUTO_RESULT_NAME:
            return p
    for base in (take_dir, video_dir):
        if base:
            c = Path(base) / AUTO_RESULT_NAME
            if c.is_file():
                return c
    return None


def _is_dense(seq, n: int) -> bool:
    """判断一条 left/right 是逐帧 dense 数组(长度==n 且元素是标量)还是区间列表。"""
    if not isinstance(seq, list) or len(seq) != n or n == 0:
        return False
    return all(np.isscalar(x) or isinstance(x, (int, float)) for x in seq)


def _to_dense(seq, n: int) -> np.ndarray:
    """把 left/right(区间列表 或 逐帧数组)统一成 (n,) int8 逐帧标签。"""
    seq = seq or []
    if _is_dense(seq, n):
        return np.asarray(seq).astype(np.int8).reshape(-1)[:n]
    return segments_to_dense(seq, n, on=CONTACT, off=FREE)


class AutoGraspPhaseProvider:
    """消费自动接触检测结果 contact_auto.json(见 phase/detect.py 或同学模块)。

    contact_auto.json 结构(与 grasp_annotation.json 对齐,二者择一):
      A) 区间:{"num_frames":N,"fps":f,"annotations":{"left":[[s,e]..],"right":[..]}}
      B) 逐帧:{"num_frames":N,"left":[0/1/..], "right":[0/1/..]}
    left/right 允许 >2 类的细粒度标签(FREE/CONTACT/自定义),下游按 int 处理。
    """
    name = "auto"

    def available(self, take_dir=None, *, annotation_path=None, video_dir=None) -> bool:
        return _find_result(take_dir, annotation_path=annotation_path, video_dir=video_dir) is not None

    def load(self, take_dir=None, num_frames=None, *, annotation_path=None, video_dir=None) -> PhaseTracks:
        path = _find_result(take_dir, annotation_path=annotation_path, video_dir=video_dir)
        if path is None:
            raise FileNotFoundError(f"未找到 {AUTO_RESULT_NAME}(take_dir={take_dir})")
        doc = json.loads(Path(path).read_text(encoding="utf-8"))

        ann = doc.get("annotations")
        if isinstance(ann, dict):
            left_raw, right_raw = ann.get("left"), ann.get("right")
        else:  # 逐帧顶层形式
            left_raw, right_raw = doc.get("left"), doc.get("right")

        n = int(doc.get("num_frames") or 0) or (int(num_frames) if num_frames else 0)
        if n <= 0:
            raise ValueError(f"{path}: 无法确定 num_frames")

        tracks = PhaseTracks(
            left=_to_dense(left_raw, n),
            right=_to_dense(right_raw, n),
            num_frames=n,
            fps=doc.get("fps"),
            source="auto",
            meta={"result": str(path), "method": doc.get("method"),
                  "params": doc.get("params", {}),
                  "segments": {"left": left_raw if isinstance(left_raw, list) else [],
                               "right": right_raw if isinstance(right_raw, list) else []}},
        )
        if num_frames and int(num_frames) != tracks.num_frames:
            tracks = tracks.resample(int(num_frames))
        return tracks
