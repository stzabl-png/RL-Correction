"""Locked Sweep2 main-ablation settings, adapted from Pour commit 4f2f857."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationSetting:
    name: str
    human_shape: bool
    confidence_mode: str
    warmup_role: str


SETTINGS = {
    "full": AblationSetting("full", True, "reference", "none"),
    "wo_human": AblationSetting("wo_human", False, "reference", "none"),
    "wo_conf": AblationSetting("wo_conf", False, "ones", "none"),
}


def resolve(name: str) -> AblationSetting:
    key = str(name).strip().lower()
    if key not in SETTINGS:
        raise ValueError(f"unknown Sweep2 ablation method {name!r}; choose {sorted(SETTINGS)}")
    return SETTINGS[key]
