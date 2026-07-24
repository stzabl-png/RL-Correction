"""Build SeqRetargeting instances for the SharpaWave from this repo's configs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.seq_retarget import SeqRetargeting

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets" / "robots" / "hands"
CONFIG_DIR = REPO_ROOT / "configs" / "retargeting"


def config_path(hand: str = "right", mode: str = "vector") -> Path:
    assert hand in ("right", "left") and mode in ("vector", "dexpilot")
    return CONFIG_DIR / f"sharpa_wave_{hand}_{mode}.yml"


def build_sharpa_retargeting(
    hand: str = "right", mode: str = "vector", override: dict | None = None
) -> SeqRetargeting:
    path = config_path(hand, mode)
    if not path.exists():
        raise FileNotFoundError(f"retargeting config missing: {path}")
    if not ASSETS_DIR.exists():
        raise FileNotFoundError(
            f"hand URDF assets missing: {ASSETS_DIR} - run scripts/prepare_assets.py first"
        )
    RetargetingConfig.set_default_urdf_dir(str(ASSETS_DIR))
    cfg = RetargetingConfig.load_from_file(str(path), override=override)
    return cfg.build()


def compute_ref_value(retargeting: SeqRetargeting, joint_pos_mano: np.ndarray) -> np.ndarray:
    """Extract the optimizer reference value from (21,3) MANO-convention keypoints.

    Same logic as dex-retargeting's example loop: positions for POSITION mode,
    task-origin difference vectors for VECTOR / DEXPILOT.
    """
    optimizer = retargeting.optimizer
    indices = optimizer.target_link_human_indices
    if optimizer.retargeting_type == "POSITION":
        return joint_pos_mano[indices, :]
    origin_indices = indices[0, :]
    task_indices = indices[1, :]
    return joint_pos_mano[task_indices, :] - joint_pos_mano[origin_indices, :]
