"""抓取/接触阶段(grasp phase)的数据类型与时间线工具。

阶段信息 = "人手轨迹上的逐帧语义标签":每一帧、每只手当前处于
未接触 / 接触(抓取)中 等状态。

规范时间线 = 视频/物体帧 (Tv),与 world_fused 的 object_* 及
replay_world 完全对齐(人工标注的帧号本身就是 Tv 帧号)。手轨迹在
world_fused 里是 Th=2*Tv @30fps,需要时用 to_hand_timeline() 上采样。

标签(int8):
  UNKNOWN = -1  无标注/未知(完全没有标注文件时)
  FREE    =  0  未接触(自由移动、接近、松手后)
  CONTACT =  1  接触/抓取中
  2+           预留给更细粒度(APPROACH / MANIPULATE / RELEASE …),
               由自动接触检测(auto provider)按需扩展,下游用 PHASE_NAMES 解释。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UNKNOWN = -1
FREE = 0
CONTACT = 1

PHASE_NAMES = {UNKNOWN: "unknown", FREE: "free", CONTACT: "contact"}

DTYPE = np.int8


def resample_labels(labels: np.ndarray, target_len: int) -> np.ndarray:
    """把逐帧标签最近邻缩放到 target_len(阶段是阶梯函数,最近邻即可)。"""
    labels = np.asarray(labels)
    n = len(labels)
    if n == target_len:
        return labels.astype(DTYPE, copy=True)
    if n == 0:
        return np.full(target_len, UNKNOWN, dtype=DTYPE)
    idx = np.clip(np.floor(np.arange(target_len) * (n / target_len)).astype(int), 0, n - 1)
    return labels[idx].astype(DTYPE, copy=False)


def segments_to_dense(segments, num_frames: int, *, on=CONTACT, off=FREE) -> np.ndarray:
    """把 [[s,e], ...](含端点的抓取区间)填成逐帧标签数组 (num_frames,) int8。

    区间外为 off(默认 FREE),区间内为 on(默认 CONTACT)。空列表 -> 全 off
    (语义:标注过、这只手全程没抓),而非 UNKNOWN。
    """
    arr = np.full(int(num_frames), off, dtype=DTYPE)
    for seg in segments or []:
        if seg is None or len(seg) < 2:
            continue
        s, e = int(seg[0]), int(seg[1])
        s, e = max(0, min(s, e)), min(num_frames - 1, max(s, e))
        if e >= s:
            arr[s:e + 1] = on
    return arr


@dataclass
class PhaseTracks:
    """一条 take 的逐帧阶段信息(规范时间线 = 视频帧 Tv)。

    left / right: (Tv,) int8,index 语义与全项目一致(0=左手 1=右手)。
    """
    left: np.ndarray
    right: np.ndarray
    num_frames: int
    fps: float | None = None
    source: str = "none"          # "manual" | "auto" | "none"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.left = np.asarray(self.left, dtype=DTYPE).reshape(-1)
        self.right = np.asarray(self.right, dtype=DTYPE).reshape(-1)
        if len(self.left) != self.num_frames or len(self.right) != self.num_frames:
            raise ValueError(
                f"phase 长度与 num_frames 不符: left={len(self.left)} "
                f"right={len(self.right)} num_frames={self.num_frames}")

    @property
    def obj(self) -> np.ndarray:
        """物体侧阶段:任一手 CONTACT -> CONTACT;两手都 FREE -> FREE;否则 UNKNOWN。"""
        out = np.full(self.num_frames, UNKNOWN, dtype=DTYPE)
        both = np.stack([self.left, self.right])
        out[np.any(both == CONTACT, axis=0)] = CONTACT
        out[np.all(both == FREE, axis=0)] = FREE
        return out

    def hand_array(self) -> np.ndarray:
        """(2, Tv) int8,[0]=left [1]=right。"""
        return np.stack([self.left, self.right]).astype(DTYPE, copy=False)

    def resample(self, target_frames: int) -> "PhaseTracks":
        """缩放到另一个视频时间线长度(标注 num_frames != Tv 时用)。"""
        if target_frames == self.num_frames:
            return self
        return PhaseTracks(
            left=resample_labels(self.left, target_frames),
            right=resample_labels(self.right, target_frames),
            num_frames=int(target_frames),
            fps=self.fps, source=self.source, meta={**self.meta, "resampled_from": self.num_frames},
        )

    def to_hand_timeline(self, num_hand_frames: int) -> np.ndarray:
        """上采样到手轨迹时间线,返回 (2, Th) int8,对齐 world_fused 的 hand_trans。"""
        return np.stack([
            resample_labels(self.left, num_hand_frames),
            resample_labels(self.right, num_hand_frames),
        ]).astype(DTYPE, copy=False)

    def summary(self) -> str:
        def _one(a):
            return f"{int((a == CONTACT).sum())}接触/{len(a)}帧"
        return f"[{self.source}] L:{_one(self.left)} R:{_one(self.right)}"
