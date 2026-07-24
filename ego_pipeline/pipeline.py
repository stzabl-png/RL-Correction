"""
EgoPipeline - orchestrator that runs stages in sequence.

Usage:
    pipeline = make_default_pipeline(debug=True)
    result = pipeline.run("video.mp4", "output/")
"""
from __future__ import annotations

import json
import os
import time
from typing import Sequence

import joblib
import numpy as np

from .context import EgoContext
from .stages.base import PipelineStage


class EgoPipeline:
    """Sequentially runs a list of PipelineStage objects."""

    def __init__(self, stages: Sequence[PipelineStage], debug: bool = False):
        self.stages = list(stages)
        self.debug = debug

    def run(self, video_path: str, output_dir: str) -> EgoContext:
        os.makedirs(output_dir, exist_ok=True)
        ctx = EgoContext(video_path=video_path, output_dir=output_dir)

        total_t0 = time.time()
        for i, stage in enumerate(self.stages):
            # Dependency check
            missing = stage.check_deps(ctx)
            if missing:
                raise RuntimeError(
                    f"Stage '{stage.name()}' missing dependencies: {missing}"
                )

            tag = f"{i}_{stage.name()}"
            print(f"\n{'='*60}")
            print(f"  Stage {i}: {stage.name()}")
            print(f"{'='*60}")

            t0 = time.time()
            ctx = stage.run(ctx)
            elapsed = time.time() - t0
            print(f"  -> {stage.name()} done ({elapsed:.1f}s)")

            if self.debug:
                self._save_snapshot(ctx, tag, output_dir)

        # Save final output
        if ctx.mano_trans is not None:
            final_path = os.path.join(output_dir, "world_space_res.pth")
            joblib.dump(
                [ctx.mano_trans, ctx.mano_rot, ctx.mano_pose,
                 ctx.mano_betas, ctx.mano_valid],
                final_path,
            )
            print(f"\nFinal output -> {final_path}")

        total_elapsed = time.time() - total_t0
        print(f"\nPipeline complete ({total_elapsed:.1f}s total)")
        return ctx

    def _save_snapshot(self, ctx: EgoContext, tag: str, output_dir: str):
        """Save intermediate results for debugging."""
        d = os.path.join(output_dir, "debug", tag)
        os.makedirs(d, exist_ok=True)

        if ctx.intrinsics is not None:
            np.save(os.path.join(d, "K.npy"), ctx.intrinsics)

        if ctx.poses_c2w is not None:
            np.save(os.path.join(d, "poses_c2w.npy"), ctx.poses_c2w)

        if ctx.depth is not None:
            depth_stats = {
                "shape": list(ctx.depth.shape),
                "min": float(ctx.depth.min()),
                "max": float(ctx.depth.max()),
                "mean": float(ctx.depth.mean()),
            }
            with open(os.path.join(d, "depth_stats.json"), "w") as f:
                json.dump(depth_stats, f, indent=2)

        if ctx.mano_trans is not None:
            np.savez(
                os.path.join(d, "mano.npz"),
                trans=np.asarray(ctx.mano_trans),
                rot=np.asarray(ctx.mano_rot),
                pose=np.asarray(ctx.mano_pose),
                betas=np.asarray(ctx.mano_betas),
                valid=np.asarray(ctx.mano_valid),
            )

        print(f"  [debug] snapshot -> {d}/")


def make_default_pipeline(debug: bool = False, with_objects: bool = True) -> EgoPipeline:
    """Build the ViPE -> HaWoR -> CoordUnify -> Smooth pipeline.

    When ``with_objects`` is set, the object branch
    (ObjectMask -> ObjectMesh -> ObjectPose) is appended. ObjectMaskStage's
    EgoHOS seed step runs in the isolated ``egohos`` conda env via subprocess;
    the rest of the object branch (SAM2, SAM3D, FoundationPose) runs in biv2ap.
    """
    from .stages.vipe_stage import ViPEStage
    from .stages.hawor_stage import HaWoRStage
    from .stages.coord_stage import CoordUnifyStage
    from .stages.smooth_stage import SmoothStage

    stages = [
        ViPEStage(),
        HaWoRStage(),
        CoordUnifyStage(),
        SmoothStage(method="gaussian", sigma=2.0),
    ]

    if with_objects:
        from .stages.object_mask_stage import ObjectMaskStage
        from .stages.object_mesh_stage import ObjectMeshStage
        from .stages.object_pose_stage import ObjectPoseStage

        stages += [ObjectMaskStage(), ObjectMeshStage(), ObjectPoseStage()]

    return EgoPipeline(stages=stages, debug=debug)
