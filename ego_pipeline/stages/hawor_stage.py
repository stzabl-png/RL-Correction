"""
Stage 2: HaWoR - hand detection, MANO estimation, and infilling.

Wraps the HaWoR pipeline:
  1. detect_track_video() - extract frames + hand detection + tracking
  2. hawor_motion_estimation() - MANO parameter estimation (camera space)
  3. hawor_slam() - SKIPPED (ViPE already provides poses + metric depth)
  4. hawor_infiller() - fill missing frames

Output (OpenCV coordinate system):
  - mano_trans:  (2, N, 3) world-space translation
  - mano_rot:    (2, N, 3) world-space rotation (axis-angle)
  - mano_pose:   (2, N, 45) hand joint angles
  - mano_betas:  (2, N, 10) shape params
  - mano_valid:  (2, N) validity mask
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

from ego_pipeline.context import EgoContext
from ego_pipeline.utils.pose_convert import save_hawor_slam_npz, write_est_focal
from .base import PipelineStage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIV2AP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Default HaWoR paths (can be overridden)
HAWOR_DIR = os.path.join(BIV2AP_DIR, "third_party", "hawor")
HAWOR_CHECKPOINT = os.path.join(HAWOR_DIR, "weights", "hawor", "checkpoints", "hawor.ckpt")
HAWOR_INFILLER = os.path.join(HAWOR_DIR, "weights", "hawor", "checkpoints", "infiller.pt")


class HaWoRStage(PipelineStage):
    """Run HaWoR hand reconstruction, using ViPE poses instead of built-in SLAM."""

    def __init__(
        self,
        checkpoint: str = HAWOR_CHECKPOINT,
        infiller_weight: str = HAWOR_INFILLER,
        hawor_dir: str = HAWOR_DIR,
    ):
        self.checkpoint = checkpoint
        self.infiller_weight = infiller_weight
        self.hawor_dir = hawor_dir

    def name(self) -> str:
        return "hawor"

    def check_deps(self, ctx: EgoContext) -> list[str]:
        missing = []
        if ctx.intrinsics is None:
            missing.append("intrinsics")
        if ctx.poses_c2w is None:
            missing.append("poses_c2w")
        return missing

    def run(self, ctx: EgoContext) -> EgoContext:
        # Add HaWoR to sys.path
        if self.hawor_dir not in sys.path:
            sys.path.insert(0, self.hawor_dir)

        # PyTorch 2.6+ defaults to weights_only=True which blocks omegaconf
        # objects in HaWoR/lightning checkpoints. Override to weights_only=False
        # for trusted local checkpoints.
        _original_torch_load = torch.load
        def _patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)
        torch.load = _patched_torch_load

        # HaWoR uses relative paths (e.g. ./weights/external/detector.pt),
        # so we must cd into hawor_dir during execution.
        # Convert all paths to absolute before chdir.
        prev_cwd = os.getcwd()
        ctx.video_path = os.path.abspath(ctx.video_path)
        ctx.output_dir = os.path.abspath(ctx.output_dir)
        os.chdir(self.hawor_dir)

        from scripts.scripts_test_video.detect_track_video import detect_track_video
        from scripts.scripts_test_video.hawor_video import (
            hawor_motion_estimation,
            hawor_infiller,
        )

        # ── Prepare args namespace (mimics argparse output) ──────────────
        fx = float(ctx.intrinsics[0, 0])
        args = SimpleNamespace(
            video_path=ctx.video_path,
            img_focal=fx,
            input_type="file",
            checkpoint=self.checkpoint,
            infiller_weight=self.infiller_weight,
            vis_mode="world",
        )

        # ── Step 1: Extract frames + detect + track hands ────────────────
        print("  Step 1/4: Detect & track hands...")
        start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
        ctx._hawor_seq_folder = seq_folder
        ctx._hawor_start_idx = start_idx
        ctx._hawor_end_idx = end_idx

        # ── Step 2: MANO estimation in camera space ──────────────────────
        print("  Step 2/4: MANO motion estimation...")
        frame_chunks_all, img_focal = hawor_motion_estimation(
            args, start_idx, end_idx, seq_folder
        )

        # ── Step 3: SKIP hawor_slam — inject ViPE poses instead ──────────
        print("  Step 3/4: Injecting ViPE poses (skipping HaWoR SLAM)...")
        slam_path = os.path.join(
            seq_folder,
            f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz",
        )

        # Write ViPE focal
        write_est_focal(seq_folder, fx)

        # Write ViPE poses in HaWoR SLAM format
        # ViPE may have different frame count than extracted frames.
        # Match to the number of extracted frames.
        n_extracted = len(imgfiles)
        n_vipe = len(ctx.poses_c2w)

        if n_vipe >= n_extracted:
            poses_for_hawor = ctx.poses_c2w[:n_extracted]
        else:
            # Pad last pose if ViPE has fewer frames
            pad = np.tile(ctx.poses_c2w[-1:], (n_extracted - n_vipe, 1, 1))
            poses_for_hawor = np.concatenate([ctx.poses_c2w, pad], axis=0)

        save_hawor_slam_npz(
            poses_for_hawor,
            ctx.intrinsics,
            slam_path,
            start_idx,
            end_idx,
        )

        # ── Step 4: Infiller ─────────────────────────────────────────────
        print("  Step 4/4: Running infiller...")
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = (
            hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
        )

        # ── Store results in context ─────────────────────────────────────
        ctx.mano_trans = np.asarray(pred_trans)   # (2, N, 3)
        ctx.mano_rot = np.asarray(pred_rot)       # (2, N, 3)
        ctx.mano_pose = np.asarray(pred_hand_pose)  # (2, N, 45)
        ctx.mano_betas = np.asarray(pred_betas)   # (2, N, 10)
        ctx.mano_valid = np.asarray(pred_valid)    # (2, N)

        # Convert torch tensors to numpy if needed
        for attr in ["mano_trans", "mano_rot", "mano_pose", "mano_betas", "mano_valid"]:
            val = getattr(ctx, attr)
            if hasattr(val, "numpy"):
                setattr(ctx, attr, val.numpy())

        n_valid_r = (ctx.mano_valid[1] > 0).sum()
        n_valid_l = (ctx.mano_valid[0] > 0).sum()
        N = ctx.mano_trans.shape[1]
        print(f"  MANO output: {N} frames, "
              f"right={n_valid_r}/{N} valid, left={n_valid_l}/{N} valid")

        torch.cuda.empty_cache()
        os.chdir(prev_cwd)
        return ctx
