"""
Stage 3: Gaussian Smooth - reduce hand jitter.

Reuses core functions from tools/smooth_hawor.py:
  - smooth_per_hand(): smooth translation/betas per valid segment
  - smooth_rotations(): smooth rotations in quaternion space
"""
from __future__ import annotations

import os
import sys

import numpy as np

from ego_pipeline.context import EgoContext
from .base import PipelineStage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIV2AP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
TOOLS_DIR = os.path.join(BIV2AP_DIR, "tools")


class SmoothStage(PipelineStage):
    """Gaussian smoothing of MANO parameters."""

    def __init__(self, method: str = "gaussian", sigma: float = 2.0,
                 window: int = 7):
        self.method = method
        self.sigma = sigma
        self.window = window

    def name(self) -> str:
        return "smooth"

    def check_deps(self, ctx: EgoContext) -> list[str]:
        missing = []
        if ctx.mano_trans is None:
            missing.append("mano_trans")
        return missing

    def run(self, ctx: EgoContext) -> EgoContext:
        # Import smooth functions from tools/smooth_hawor.py
        if TOOLS_DIR not in sys.path:
            sys.path.insert(0, TOOLS_DIR)
        from smooth_hawor import smooth_per_hand, smooth_rotations

        side_names = ["left", "right"]

        for h in range(2):
            valid = ctx.mano_valid[h].astype(bool) if ctx.mano_valid is not None \
                else np.ones(ctx.mano_trans.shape[1], dtype=bool)
            n_valid = valid.sum()
            print(f"  {side_names[h]}: {n_valid}/{len(valid)} valid frames")

            if n_valid < 3:
                print(f"    Skip (too few valid)")
                continue

            # Compute pre-smooth jitter for reporting
            t_before = ctx.mano_trans[h][valid]
            jitter_before = np.mean(np.linalg.norm(
                np.diff(t_before, axis=0), axis=1))

            # Smooth translation (Euclidean)
            ctx.mano_trans[h] = smooth_per_hand(
                ctx.mano_trans[h], valid,
                self.method, self.window, self.sigma)

            # Smooth global rotation (quaternion space)
            ctx.mano_rot[h] = smooth_rotations(
                ctx.mano_rot[h], valid,
                self.method, self.window, self.sigma)

            # Smooth hand pose (each of 15 joints independently in quat space)
            hand_pose = ctx.mano_pose[h]  # (N, 45)
            for j in range(15):
                joint_aa = hand_pose[:, j*3:(j+1)*3]
                hand_pose[:, j*3:(j+1)*3] = smooth_rotations(
                    joint_aa, valid,
                    self.method, self.window, self.sigma)
            ctx.mano_pose[h] = hand_pose

            # Smooth betas (mild, Euclidean)
            ctx.mano_betas[h] = smooth_per_hand(
                ctx.mano_betas[h], valid,
                self.method, max(3, self.window // 2), self.sigma * 0.5)

            # Report jitter reduction
            t_after = ctx.mano_trans[h][valid]
            jitter_after = np.mean(np.linalg.norm(
                np.diff(t_after, axis=0), axis=1))
            reduction = (1 - jitter_after / jitter_before) * 100 \
                if jitter_before > 0 else 0
            print(f"    jitter: {jitter_before:.4f} -> {jitter_after:.4f} "
                  f"({reduction:+.1f}%)")

        return ctx
