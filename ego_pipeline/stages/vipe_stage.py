"""
Stage 1: ViPE - camera intrinsics, poses, and metric depth estimation.

Wraps the ViPE pipeline to produce:
  - intrinsics: (3,3) K matrix
  - poses_c2w:  (N,4,4) camera-to-world (OpenCV convention)
  - depth:      (N,H,W) metric depth in meters
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from ego_pipeline.context import EgoContext
from .base import PipelineStage

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIV2AP_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
VIPE_DIR = os.path.join(BIV2AP_DIR, "third_party", "vipe")


class ViPEStage(PipelineStage):
    """Run ViPE pipeline on a single video."""

    def __init__(self, save_depth: bool = True):
        self.save_depth = save_depth

    def name(self) -> str:
        return "vipe"

    def check_deps(self, ctx: EgoContext) -> list[str]:
        missing = []
        if not ctx.video_path or not os.path.isfile(ctx.video_path):
            missing.append(f"video_path (got '{ctx.video_path}')")
        return missing

    def run(self, ctx: EgoContext) -> EgoContext:
        import torch
        sys.path.insert(0, VIPE_DIR)

        from omegaconf import OmegaConf
        from vipe.pipeline import make_pipeline
        from vipe.streams.raw_mp4_stream import RawMp4Stream

        # ── Configure pipeline ───────────────────────────────────────────
        pipeline_cfg = OmegaConf.load(
            os.path.join(VIPE_DIR, "configs", "pipeline", "default.yaml")
        )
        slam_cfg = OmegaConf.load(
            os.path.join(VIPE_DIR, "configs", "slam", "default.yaml")
        )
        pipeline_cfg.slam = OmegaConf.merge(slam_cfg, pipeline_cfg.get("slam", {}))

        pipeline_cfg.output.save_viz = False
        pipeline_cfg.output.save_slam_map = False
        pipeline_cfg.output.save_artifacts = True
        pipeline_cfg.output.skip_exists = False

        # Worker output directory
        vipe_raw_dir = os.path.join(ctx.output_dir, "_vipe_raw")
        pipeline_cfg.output.path = vipe_raw_dir
        os.makedirs(vipe_raw_dir, exist_ok=True)

        if "defaults" in pipeline_cfg:
            del pipeline_cfg["defaults"]

        # ── Build pipeline and run ───────────────────────────────────────
        print("  Loading ViPE models...")
        pipeline = make_pipeline(pipeline_cfg)
        pipeline.return_output_streams = True

        video_name = Path(ctx.video_path).stem
        # Process every frame by default. ViPE's SLAM needs small inter-frame
        # baselines; subsampling (e.g. skip=5 on 15fps video -> 3fps) makes the
        # tracker diverge (garbage focal, zero depth, exploded camera path).
        # Override via ctx.vipe_frame_skip only for high-fps / OOM-constrained runs.
        frame_skip = getattr(ctx, "vipe_frame_skip", 1)
        total_cap = cv2.VideoCapture(str(ctx.video_path))
        total_n = int(total_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_cap.release()
        seek = range(0, total_n, frame_skip) if frame_skip > 1 else None
        stream = RawMp4Stream(Path(ctx.video_path), seek_range=seek, name=video_name)
        n_frames = len(stream)
        print(f"  Video: {ctx.video_path} ({n_frames} frames)")

        print("  Running ViPE pipeline...")
        result = pipeline.run(stream)

        # ── Extract outputs ──────────────────────────────────────────────
        from vipe.utils.io import ArtifactPath

        artifact_path = ArtifactPath(Path(vipe_raw_dir), video_name)

        # Intrinsics
        if artifact_path.intrinsics_path.exists():
            intr_data = np.load(artifact_path.intrinsics_path)
            K_vec = intr_data["data"][0].astype(np.float64)
            K = np.eye(3, dtype=np.float64)
            K[0, 0] = K_vec[0]  # fx
            K[1, 1] = K_vec[1]  # fy
            K[0, 2] = K_vec[2]  # cx
            K[1, 2] = K_vec[3]  # cy
            ctx.intrinsics = K
            print(f"  Intrinsics: fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
                  f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
        else:
            raise RuntimeError("ViPE failed to produce intrinsics")

        # Poses
        if artifact_path.pose_path.exists():
            pose_data = np.load(artifact_path.pose_path)
            ctx.poses_c2w = pose_data["data"].astype(np.float32)
            print(f"  Poses: {ctx.poses_c2w.shape}")
        else:
            raise RuntimeError("ViPE failed to produce poses")

        # Depth
        if self.save_depth and artifact_path.depth_path.exists():
            from vipe.utils.io import read_depth_artifacts
            depth_list = []
            for _, depth_tensor in read_depth_artifacts(artifact_path.depth_path):
                depth_list.append(depth_tensor.numpy())
            if depth_list:
                ctx.depth = np.stack(depth_list).astype(np.float32)
                print(f"  Depth: {ctx.depth.shape} "
                      f"range=[{ctx.depth.min():.3f}, {ctx.depth.max():.3f}]m")

        # Save outputs to output_dir
        out_dir = ctx.output_dir
        np.save(os.path.join(out_dir, "K.npy"), ctx.intrinsics)
        np.save(os.path.join(out_dir, "cam_c2w.npy"), ctx.poses_c2w)
        if ctx.depth is not None:
            np.savez_compressed(os.path.join(out_dir, "depth.npz"),
                                depths=ctx.depth)

        # Cleanup raw ViPE output
        for sub in ["rgb", "pose", "depth", "intrinsics", "mask", "flow", "vipe"]:
            sub_dir = os.path.join(vipe_raw_dir, sub)
            if os.path.isdir(sub_dir):
                shutil.rmtree(sub_dir, ignore_errors=True)

        torch.cuda.empty_cache()
        return ctx
