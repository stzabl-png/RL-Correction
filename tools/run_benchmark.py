#!/usr/bin/env python3
"""
Benchmark Runner: MegaSAM & ViPE on HOI4D-32 + EgoDex-30.

Steps:
  Step 2: MegaSAM on HOI4D-32
  Step 3: ViPE on HOI4D-32
  Step 4: ViPE on EgoDex-30

Usage:
    # Run all steps:
    conda activate biv2ap
    export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):$LD_LIBRARY_PATH
    python tools/run_benchmark.py --step all

    # Run specific step:
    python tools/run_benchmark.py --step megasam-hoi4d
    python tools/run_benchmark.py --step vipe-hoi4d
    python tools/run_benchmark.py --step vipe-egodex
"""

import os, sys, json, argparse, time, subprocess
from pathlib import Path

BIV2AP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_ROOT = os.path.join(BIV2AP_DIR, "Data", "Benchmark")
HOI4D_ROOT = os.path.join(BIV2AP_DIR, "Data", "HOI4D")
MEGASAM_DIR = os.path.join(BIV2AP_DIR, "third_party", "megasam")
VIPE_DIR = os.path.join(BIV2AP_DIR, "third_party", "vipe")


def get_hoi4d_videos():
    """List HOI4D benchmark RGB videos."""
    rgb_dir = os.path.join(BENCH_ROOT, "HOI4D", "rgb")
    vids = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".mp4")])
    return [(os.path.join(rgb_dir, v), v.replace(".mp4", "")) for v, _ in [(v, v) for v in vids]]


def get_egodex_videos():
    """List EgoDex benchmark RGB videos."""
    rgb_dir = os.path.join(BENCH_ROOT, "Egodex", "rgb")
    vids = sorted([f for f in os.listdir(rgb_dir) if f.endswith(".mp4")])
    return [(os.path.join(rgb_dir, v), v.replace(".mp4", "")) for v in vids]


def is_done(out_dir, name):
    d = os.path.join(out_dir, name)
    return os.path.exists(os.path.join(d, "depth.npz")) and os.path.exists(os.path.join(d, "meta.json"))


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: MegaSAM on HOI4D-32
# ═══════════════════════════════════════════════════════════════════════════════

def run_megasam_hoi4d():
    """Run MegaSAM pipeline on HOI4D benchmark videos."""
    out_dir = os.path.join(BENCH_ROOT, "HOI4D", "megasam")
    os.makedirs(out_dir, exist_ok=True)

    videos = get_hoi4d_videos()
    todo = [(v, n) for v, n in videos if not is_done(out_dir, n)]
    print(f"\n━━━ Step 2: MegaSAM on HOI4D ━━━")
    print(f"  Total: {len(videos)}, Todo: {len(todo)}")

    if not todo:
        print("  All done!")
        return

    # Import MegaSAM components
    sys.path.insert(0, MEGASAM_DIR)
    sys.path.insert(0, os.path.join(MEGASAM_DIR, "UniDepth"))
    sys.path.insert(0, os.path.join(MEGASAM_DIR, "base", "droid_slam"))
    sys.path.insert(0, os.path.join(MEGASAM_DIR, "Depth-Anything"))

    import torch, cv2, glob, shutil
    import numpy as np
    from natsort import natsorted

    # Load models once
    print("  Loading models...")
    from run_megasam_egodex import (load_depth_anything, load_unidepth,
                                    run_da_on_dir, run_unidepth_on_dir,
                                    run_droid, _resize_long)

    da_model, da_transform = load_depth_anything()
    ud_model = load_unidepth()

    work_dir = os.path.join(BENCH_ROOT, "HOI4D", "_workdir")
    MAX_FRAMES = 60

    for i, (vid_path, name) in enumerate(todo):
        t0 = time.time()
        print(f"\n  [{i+1}/{len(todo)}] {name}")

        ep_work = os.path.join(work_dir, name)
        ep_out = os.path.join(out_dir, name)
        os.makedirs(ep_out, exist_ok=True)

        try:
            # Extract frames from MP4 (same resize logic as run_megasam_egodex)
            frame_dir = os.path.join(ep_work, "frames")
            os.makedirs(frame_dir, exist_ok=True)
            cap = cv2.VideoCapture(vid_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step = max(1, total // MAX_FRAMES)
            idxs = list(range(0, total, step))[:MAX_FRAMES]
            H_out = W_out = None
            for out_i, idx in enumerate(idxs):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame, H_out, W_out = _resize_long(frame)
                cv2.imwrite(os.path.join(frame_dir, f"frame_{out_i:04d}.png"), frame)
            cap.release()
            n = len(glob.glob(os.path.join(frame_dir, "*.png")))
            W, H = W_out, H_out
            print(f"    Frames: {n} ({W}x{H})")

            # DA
            da_dir = os.path.join(ep_work, "da")
            run_da_on_dir(da_model, da_transform, frame_dir, da_dir)

            # UniDepth: use fixed calibrated_fx (same as EgoDex)
            # Do NOT use GT intrinsics — must be a fair blind comparison
            calib_fx = 249.4

            ud_dir = os.path.join(ep_work, "ud")
            run_unidepth_on_dir(ud_model, frame_dir, ud_dir, calib_fx)

            # DROID-SLAM (signature: frame_dir, da_dir, ud_dir, seq_id, calibrated_fx)
            depths_m, motion_prob, K_opt, cam_c2w = run_droid(
                frame_dir, da_dir, ud_dir, name, calib_fx, opt_intr=True)

            # Save results
            np.savez_compressed(os.path.join(ep_out, "depth.npz"), depths=depths_m)
            np.save(os.path.join(ep_out, "cam_c2w.npy"), cam_c2w)
            np.save(os.path.join(ep_out, "K.npy"), K_opt)
            if motion_prob is not None:
                np.save(os.path.join(ep_out, "motion_prob.npy"), motion_prob)
            meta = {"seq_id": name, "n_frames": len(depths_m), "pipeline": "megasam",
                    "calibrated_fx": float(calib_fx)}
            with open(os.path.join(ep_out, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"    ✅ {len(depths_m)}f, {time.time()-t0:.0f}s")

        except Exception as e:
            print(f"    ❌ {e}")
            import traceback; traceback.print_exc()

        # Cleanup work dir
        if os.path.isdir(ep_work):
            shutil.rmtree(ep_work, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3+4: ViPE on HOI4D-32 / EgoDex-30
# ═══════════════════════════════════════════════════════════════════════════════

def _estimate_safe_max_frames(vid_path, ram_budget_gb=None):
    """Estimate the max number of frames that fit in RAM for ViPE.

    ViPE caches ALL frames across 2 SLAM passes + post-processing.
    Empirical per-frame RAM cost (1080p, measured from OOM at 300f=25GB):
      - RGB tensor on CPU cache: H*W*3*4 (float32) ≈ 24 MB
      - Depth model intermediate buffers: ~20 MB
      - SLAM correlations + flow: ~20 MB
      - Instance/pose/other attrs: ~16 MB
    Total ≈ 80 MB/frame for 1080p.  Scales with resolution.

    Model weights + CUDA context ≈ 8 GB fixed overhead.
    """
    import cv2
    cap = cv2.VideoCapture(str(vid_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Empirical: 80 MB/frame at 1080p, scale proportionally
    ref_pixels = 1920 * 1080
    per_frame_mb = 80.0 * (h * w) / ref_pixels
    per_frame_mb = max(per_frame_mb, 40.0)  # floor

    if ram_budget_gb is None:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024**3)
        # Leave 8 GB for model weights, CUDA context, OS, desktop
        ram_budget_gb = max(avail_gb - 8.0, 4.0)
    # Hard cap: never allocate more than 16 GB to frames (leave room for models)
    ram_budget_gb = min(ram_budget_gb, 16.0)

    max_frames = int((ram_budget_gb * 1024) / per_frame_mb)
    max_frames = max(max_frames, 30)  # absolute minimum
    max_frames = min(max_frames, 200)  # hard cap: never more than 200 frames

    if n_total <= max_frames:
        stride = 1
    else:
        stride = max(1, n_total // max_frames)

    actual_frames = len(range(0, n_total, stride))
    print(f"    RAM budget: {ram_budget_gb:.1f}GB, {per_frame_mb:.0f}MB/f, "
          f"total={n_total}, max_frames={max_frames}, stride={stride}, using={actual_frames}f")
    return n_total, stride


def _run_single_vipe_episode(vid_path, name, out_dir, vipe_dir):
    """Run ViPE on a single episode in an isolated process.

    This function is meant to be called via subprocess to ensure
    complete memory cleanup between episodes.
    """
    import torch
    import numpy as np
    import gc
    import shutil
    from pathlib import Path as P
    from omegaconf import OmegaConf

    sys.path.insert(0, vipe_dir)
    from vipe.pipeline import make_pipeline
    from vipe.streams.raw_mp4_stream import RawMp4Stream
    from vipe.utils.io import read_depth_artifacts, ArtifactPath

    vipe_raw_out = os.path.join(out_dir, "_vipe_raw")
    os.makedirs(vipe_raw_out, exist_ok=True)
    ep_out = os.path.join(out_dir, name)
    os.makedirs(ep_out, exist_ok=True)

    # Build pipeline
    pipeline_cfg = OmegaConf.load(os.path.join(vipe_dir, "configs", "pipeline", "default.yaml"))
    slam_cfg = OmegaConf.load(os.path.join(vipe_dir, "configs", "slam", "default.yaml"))
    pipeline_cfg.slam = OmegaConf.merge(slam_cfg, pipeline_cfg.get("slam", {}))
    pipeline_cfg.output.save_viz = False
    pipeline_cfg.output.save_slam_map = False
    pipeline_cfg.output.save_artifacts = True
    pipeline_cfg.output.skip_exists = False
    pipeline_cfg.init.instance = None
    pipeline_cfg.init.prefetch_queue_size = 4
    pipeline_cfg.slam.buffer = 512
    pipeline_cfg.output.path = vipe_raw_out
    if "defaults" in pipeline_cfg:
        del pipeline_cfg["defaults"]

    pipeline = make_pipeline(pipeline_cfg)
    pipeline.return_output_streams = True

    # Compute safe frame stride
    n_total, stride = _estimate_safe_max_frames(vid_path)
    seek = range(0, n_total, stride) if stride > 1 else None
    stream = RawMp4Stream(P(vid_path), seek_range=seek,
                          name=name.replace("/", "_"))
    result = pipeline.run(stream)

    artifact_path = ArtifactPath(P(vipe_raw_out), name.replace("/", "_"))

    # Extract pose
    if artifact_path.pose_path.exists():
        pose_data = np.load(artifact_path.pose_path)
        cam_c2w = pose_data["data"].astype(np.float32)
        np.save(os.path.join(ep_out, "cam_c2w.npy"), cam_c2w)

    # Extract intrinsics
    if artifact_path.intrinsics_path.exists():
        intr_data = np.load(artifact_path.intrinsics_path)
        K_vec = intr_data["data"][0].astype(np.float32)
        K = np.eye(3, dtype=np.float32)
        K[0, 0], K[1, 1] = K_vec[0], K_vec[1]
        K[0, 2], K[1, 2] = K_vec[2], K_vec[3]
        np.save(os.path.join(ep_out, "K.npy"), K)

    # Extract depth
    n_out = 0
    if artifact_path.depth_path.exists():
        depth_list = []
        for _, depth_tensor in read_depth_artifacts(artifact_path.depth_path):
            depth_list.append(depth_tensor.numpy())
        if depth_list:
            depths = np.stack(depth_list).astype(np.float32)
            np.savez_compressed(os.path.join(ep_out, "depth.npz"), depths=depths)
            n_out = len(depths)

    meta = {"seq_id": name, "n_frames": n_out, "pipeline": "vipe-1.2.0",
            "stride": stride, "total_video_frames": n_total,
            "depth_align_model": str(pipeline_cfg.post.depth_align_model)}
    with open(os.path.join(ep_out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Cleanup raw artifacts
    for sub in ["rgb", "pose", "depth", "intrinsics", "mask", "flow", "vipe"]:
        sub_dir = os.path.join(vipe_raw_out, sub)
        if os.path.isdir(sub_dir):
            shutil.rmtree(sub_dir, ignore_errors=True)

    return n_out


def run_vipe_batch(dataset, videos, out_dir):
    """Run ViPE pipeline on a list of videos using subprocess isolation.

    Each episode runs in a separate subprocess to guarantee complete
    memory release between episodes — preventing OOM accumulation.
    """
    os.makedirs(out_dir, exist_ok=True)

    todo = [(v, n) for v, n in videos if not is_done(out_dir, n)]
    print(f"\n━━━ ViPE on {dataset} ━━━")
    print(f"  Total: {len(videos)}, Todo: {len(todo)}")

    if not todo:
        print("  All done!")
        return

    n_ok = n_fail = 0
    for i, (vid_path, name) in enumerate(todo):
        t0 = time.time()
        print(f"\n  [{i+1}/{len(todo)}] {name}")

        # Run each episode in a subprocess for memory isolation
        worker_code = f'''
import sys, os
sys.path.insert(0, "{BIV2AP_DIR}")
sys.path.insert(0, os.path.join("{BIV2AP_DIR}", "tools"))
os.chdir("{BIV2AP_DIR}")
from run_benchmark import _run_single_vipe_episode, _estimate_safe_max_frames
n_out = _run_single_vipe_episode("{vid_path}", "{name}", "{out_dir}", "{VIPE_DIR}")
print(f"RESULT_FRAMES={{n_out}}")
'''
        try:
            result = subprocess.run(
                [sys.executable, "-c", worker_code],
                capture_output=True, text=True, timeout=600,
                cwd=BIV2AP_DIR,
                env={**os.environ, "PYTHONPATH": f"{VIPE_DIR}:{BIV2AP_DIR}"}
            )

            elapsed = time.time() - t0

            if result.returncode == 0:
                # Parse n_out from stdout
                n_out = 0
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("RESULT_FRAMES="):
                        n_out = int(line.split("=")[1])
                    else:
                        print(f"    {line}")
                print(f"    ✅ {n_out}f, {elapsed:.0f}s")
                n_ok += 1
            else:
                n_fail += 1
                # Print last few lines of stderr
                stderr_lines = result.stderr.strip().split("\n")
                for line in stderr_lines[-10:]:
                    print(f"    {line}")
                print(f"    ❌ exit code {result.returncode}, {elapsed:.0f}s")

        except subprocess.TimeoutExpired:
            n_fail += 1
            print(f"    ❌ timeout (600s)")
        except Exception as e:
            n_fail += 1
            print(f"    ❌ {e}")
            import traceback; traceback.print_exc()

    print(f"\n  Results: ✅ {n_ok}  ❌ {n_fail}")


def run_vipe_hoi4d():
    run_vipe_batch("HOI4D", get_hoi4d_videos(),
                   os.path.join(BENCH_ROOT, "HOI4D", "vipe"))


def run_vipe_egodex():
    run_vipe_batch("EgoDex", get_egodex_videos(),
                   os.path.join(BENCH_ROOT, "Egodex", "vipe"))


# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "megasam-hoi4d", "vipe-hoi4d", "vipe-egodex",
                                 "vipe-all"],
                        help="Which step to run")
    args = parser.parse_args()

    print(f"\n╔{'═'*55}╗")
    print(f"║  MegaSAM vs ViPE Benchmark Runner                  ║")
    print(f"╠{'═'*55}╣")
    print(f"║  Benchmark: {BENCH_ROOT}")
    print(f"╚{'═'*55}╝")

    if args.step in ["all", "vipe-hoi4d", "vipe-all"]:
        run_vipe_hoi4d()

    if args.step in ["all", "vipe-egodex", "vipe-all"]:
        run_vipe_egodex()

    if args.step in ["all", "megasam-hoi4d"]:
        run_megasam_hoi4d()

    print("\n✅ Benchmark run complete!")


if __name__ == "__main__":
    main()
