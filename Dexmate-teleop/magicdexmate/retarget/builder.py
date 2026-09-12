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
    hand: str = "right",
    mode: str = "vector",
    override: dict | None = None,
    abduction_weight: float = 1.0,
    pinch_weight: float = 0.0,
) -> SeqRetargeting:
    """Build a SharpaWave SeqRetargeting.

    For VECTOR mode we classify each target vector and set a per-vector weight
    (VectorOptimizer.weight) plus a normalization denominator:

    - Step 4.1 (fj) ABDUCTION vector: task link ends in ``_MP`` (MCP_VL -> MP proximal
      side-swing). Its residual is small (short proximal segment) and would be averaged
      away by the equal-weight vector cost, so we up-weight it by ``abduction_weight``.
    - Step 5b (fj) PINCH vector: origin link is the thumb fingertip (thumb_tip -> finger_tip).
      It directly constrains the thumb<->fingertip GAP so opposition is accurate. Its weight
      is applied CONDITIONALLY at runtime by VectorOptimizer (only while the human is actually
      pinching that finger, gap < ~3cm), mirroring DexPilot's pinch projection / V2AP's
      pinch_loss. ``pinch_weight`` is the weight used WHEN a pinch is detected; open/fist stay
      byte-identical to baseline (gap large -> weight 0 -> A1 zero-cost).
    - PRIMARY vector: everything else = the 5 palm -> fingertip flexion/reach vectors (A1).

    The loss denominator is kept at the PRIMARY (fingertip) count so those vectors retain
    their baseline /N weight and the auxiliary abduction/pinch vectors are ADDED on top,
    not diluting flexion tracking relative to the fixed norm_delta regularization.
    ``abduction_weight=1.0`` AND ``pinch_weight=0.0`` reproduces the exact baseline (the
    pinch vectors contribute 0 to loss and gradient -> byte-identical solution). Configs
    without ``_MP`` / thumb-origin vectors are unaffected (weight stays all ones).
    """
    path = config_path(hand, mode)
    if not path.exists():
        raise FileNotFoundError(f"retargeting config missing: {path}")
    if not ASSETS_DIR.exists():
        raise FileNotFoundError(
            f"hand URDF assets missing: {ASSETS_DIR} - run scripts/prepare_assets.py first"
        )
    RetargetingConfig.set_default_urdf_dir(str(ASSETS_DIR))
    cfg = RetargetingConfig.load_from_file(str(path), override=override)
    retargeting = cfg.build()

    if mode == "vector":
        opt = retargeting.optimizer
        is_abduction = [n.endswith("_MP") for n in opt.task_link_names]
        is_pinch = [o.endswith("thumb_fingertip") for o in opt.origin_link_names]
        if any(is_abduction) or any(is_pinch):
            # BASE weight: primary=1, abduction=abduction_weight, pinch=0. The pinch weight
            # is NOT baked in here -- it is applied conditionally per-frame by the optimizer
            # (only while pinching), so open/fist stay byte-identical to baseline.
            weight = []
            for ab, pi in zip(is_abduction, is_pinch):
                weight.append(0.0 if pi else (abduction_weight if ab else 1.0))
            opt.weight = np.array(weight, dtype=np.float32)
            # Denominator = #PRIMARY (palm->fingertip) vectors only, so those keep their
            # baseline /N weight; abduction + pinch vectors are ADDED on top, not diluting
            # flexion tracking (A1) relative to the fixed norm_delta reg. With
            # abduction_weight->0 and pinch_weight->0 this reproduces the exact baseline.
            opt.loss_denominator = float(
                sum(1 for ab, pi in zip(is_abduction, is_pinch) if not ab and not pi)
            )
            # Step 5b: conditional pinch gating. VectorOptimizer engages `pinch_weight` on a
            # pinch vector only while the human thumb<->finger gap is small (hysteresis).
            opt.pinch_mask = np.array(is_pinch, dtype=bool)
            opt.pinch_weight = float(pinch_weight)

    return retargeting


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
