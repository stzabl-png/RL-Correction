"""grasp phase(抓取/接触阶段)—— 人手轨迹上的逐帧语义标签。

现在:人工标注(tools/annotate_grasp_frames.py -> grasp_annotation.json)。
将来:同学的自动接触检测,只需实现 auto.AutoGraspPhaseProvider,下游不变。

用法:
  from phase import load_phase
  tracks = load_phase(take_dir, num_frames=Tv)   # None 表示无标注(向后兼容)
  if tracks:
      tracks.left, tracks.right   # (Tv,) int8  FREE=0 / CONTACT=1

字段落点见 tools/attach_grasp_phase.py 与 ego_pipeline/phase/README.md。
"""
from .types import (  # noqa: F401
    CONTACT, DTYPE, FREE, PHASE_NAMES, UNKNOWN,
    PhaseTracks, resample_labels, segments_to_dense,
)
from .provider import get_provider, load_phase  # noqa: F401
from .manual import ManualGraspPhaseProvider  # noqa: F401
from .auto import AutoGraspPhaseProvider  # noqa: F401
