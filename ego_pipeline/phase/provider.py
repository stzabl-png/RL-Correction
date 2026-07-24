"""Provider 工厂:统一入口,自动在 人工 / 自动 之间选择。

切换逻辑(可用 BIV2AP_PHASE_PROVIDER=auto|manual|none 覆盖):
  prefer="auto"(默认): auto 可用 -> auto;否则 manual 可用 -> manual;否则 None。
一旦同学的 auto provider 就绪(available() 返回 True),现有调用自动切到自动,
无需改任何下游代码。
"""
from __future__ import annotations

import os

from .auto import AutoGraspPhaseProvider
from .manual import ManualGraspPhaseProvider
from .types import PhaseTracks

_REGISTRY = {"manual": ManualGraspPhaseProvider, "auto": AutoGraspPhaseProvider}


def get_provider(name):
    cls = _REGISTRY.get(name)
    return cls() if cls else None


def load_phase(take_dir=None, num_frames=None, *, prefer="auto",
               annotation_path=None, video_dir=None) -> PhaseTracks | None:
    """按优先级挑一个可用 provider 载入阶段信息;都不可用则返回 None(向后兼容)。"""
    forced = os.environ.get("BIV2AP_PHASE_PROVIDER")
    if forced:
        prefer = forced
    if prefer == "none":
        return None

    order = ["auto", "manual"] if prefer == "auto" else [prefer]
    # 显式指定单个 provider 时只试它;prefer=auto 时按 auto->manual 回退。
    if prefer not in ("auto",) and prefer in _REGISTRY:
        order = [prefer]

    for name in order:
        prov = get_provider(name)
        if prov is None:
            continue
        try:
            if prov.available(take_dir, annotation_path=annotation_path, video_dir=video_dir):
                return prov.load(take_dir, num_frames,
                                 annotation_path=annotation_path, video_dir=video_dir)
        except NotImplementedError:
            raise  # auto 半就绪状态:显式冒泡,别静默回退
        except Exception:
            continue
    return None
