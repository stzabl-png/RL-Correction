#!/usr/bin/env python3
"""
Ego Pipeline CLI - first-person video to MANO hand parameters.

Pipeline: ViPE -> HaWoR -> CoordUnify -> Gaussian Smooth

Usage:
    # Full pipeline
    XFORMERS_DISABLED=1 python tools/run_ego_pipeline.py \
        --video example/video.mp4 --output output/example/

    # With debug snapshots
    XFORMERS_DISABLED=1 python tools/run_ego_pipeline.py \
        --video example/video.mp4 --output output/example/ --debug

    # Visualize results
    python tools/vis_hand_motion.py --seq_dir output/example/

Output:
    output/
    +-- world_space_res.pth   # Final MANO output (Z-up right-hand)
    +-- K.npy                 # Camera intrinsics
    +-- cam_c2w.npy           # Camera poses (Z-up)
    +-- depth.npz             # Metric depth maps
    +-- debug/                # (if --debug) per-stage snapshots
"""
import argparse
import os
import sys

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIV2AP_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, BIV2AP_DIR)


def main():
    parser = argparse.ArgumentParser(
        description="Ego Pipeline: video -> MANO hand parameters"
    )
    parser.add_argument("--video", required=True, help="Input video (MP4)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--debug", action="store_true",
                        help="Save intermediate snapshots for debugging")
    parser.add_argument("--skip-vipe", action="store_true",
                        help="Skip ViPE (load from existing output dir)")
    parser.add_argument("--smooth-sigma", type=float, default=2.0,
                        help="Gaussian smooth sigma (default: 2.0)")
    args = parser.parse_args()

    from ego_pipeline.context import EgoContext
    from ego_pipeline.pipeline import EgoPipeline
    from ego_pipeline.stages.coord_stage import CoordUnifyStage
    from ego_pipeline.stages.smooth_stage import SmoothStage

    stages = []

    if args.skip_vipe:
        # Load existing ViPE output
        import numpy as np
        print(f"Loading existing ViPE output from {args.output}")
        ctx = EgoContext(video_path=args.video, output_dir=args.output)
        k_path = os.path.join(args.output, "K.npy")
        pose_path = os.path.join(args.output, "cam_c2w.npy")
        depth_path = os.path.join(args.output, "depth.npz")
        if os.path.exists(k_path):
            ctx.intrinsics = np.load(k_path)
        if os.path.exists(pose_path):
            ctx.poses_c2w = np.load(pose_path)
        if os.path.exists(depth_path):
            ctx.depth = np.load(depth_path)["depths"]
    else:
        from ego_pipeline.stages.vipe_stage import ViPEStage
        stages.append(ViPEStage())

    from ego_pipeline.stages.hawor_stage import HaWoRStage
    stages.append(HaWoRStage())
    stages.append(CoordUnifyStage())
    stages.append(SmoothStage(method="gaussian", sigma=args.smooth_sigma))

    pipeline = EgoPipeline(stages=stages, debug=args.debug)

    if args.skip_vipe:
        # Run pipeline with pre-loaded context
        import time
        os.makedirs(args.output, exist_ok=True)
        total_t0 = time.time()
        for i, stage in enumerate(pipeline.stages):
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
            if pipeline.debug:
                pipeline._save_snapshot(ctx, tag, args.output)

        import joblib
        if ctx.mano_trans is not None:
            final_path = os.path.join(args.output, "world_space_res.pth")
            joblib.dump(
                [ctx.mano_trans, ctx.mano_rot, ctx.mano_pose,
                 ctx.mano_betas, ctx.mano_valid],
                final_path,
            )
            print(f"\nFinal output -> {final_path}")
        print(f"\nPipeline complete ({time.time() - total_t0:.1f}s total)")
    else:
        ctx = pipeline.run(args.video, args.output)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    if ctx.intrinsics is not None:
        print(f"  Intrinsics: fx={ctx.intrinsics[0,0]:.1f}")
    if ctx.poses_c2w is not None:
        print(f"  Poses: {ctx.poses_c2w.shape}")
    if ctx.mano_trans is not None:
        N = ctx.mano_trans.shape[1]
        n_r = (ctx.mano_valid[1] > 0).sum() if ctx.mano_valid is not None else "?"
        n_l = (ctx.mano_valid[0] > 0).sum() if ctx.mano_valid is not None else "?"
        print(f"  MANO: {N} frames, right={n_r}/{N}, left={n_l}/{N}")
    print(f"  Coordinate system: Z-up right-hand (X=fwd, Y=left, Z=up)")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
