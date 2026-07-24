#!/usr/bin/env python3
"""
ViPE full-pipeline depth/pose estimation for EgoDex (Reconstruct_and_Retarget).

Runs 3 parallel workers to utilise ~85% of RTX 5090 VRAM (32 GB).
Each worker runs an independent ViPE pipeline on a video at a time.

Pipeline per video:
  1. GeoCalib   → camera intrinsics
  2. TrackAnything → dynamic object segmentation
  3. SLAM (DroidNet) → camera poses + sparse map
  4. Depth alignment → metric depth (UniDepth-L + VDA-ViTS)

Output (per episode):
  {OUT_BASE}/{task}/{ep}/
    depth.npz        depths (N,H,W) float32 [metres]
    cam_c2w.npy      camera-to-world (N,4,4)
    K.npy            intrinsic matrix (3,3) or (4,) [fx,fy,cx,cy]
    meta.json        metadata

Usage:
    conda activate biv2ap
    export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):$LD_LIBRARY_PATH
    export XFORMERS_DISABLED=1
    python tools/run_vipe_egodex.py --resume [--workers 3]
    python tools/run_vipe_egodex.py --dry-run
"""

import os, sys, json, argparse, glob, time, traceback, shutil
import multiprocessing as mp
from pathlib import Path

import numpy as np

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BIV2AP_DIR  = os.path.dirname(SCRIPT_DIR)
VIPE_DIR    = os.path.join(BIV2AP_DIR, "third_party", "vipe")

EGODEX_ROOT = "/home/lyh/Project/Affordance2Grasp/data_hub/RawData/EgoRawData/egodex/test"
OUT_BASE    = os.path.join(BIV2AP_DIR, "Output", "Depth", "ViPE", "Egodex")

# ViPE's native output directory (used by the pipeline internally, then we
# convert its artifacts to our standardised format)
VIPE_RAW_OUT = os.path.join(BIV2AP_DIR, "Output", "Depth", "ViPE", "_vipe_raw")


# ══════════════════════════════════════════════════════════════════════════════
# Episode enumeration
# ══════════════════════════════════════════════════════════════════════════════

def enumerate_egodex():
    """Returns sorted list of dicts {seq_id, mp4_path}."""
    from natsort import natsorted
    eps = []
    for task in natsorted(os.listdir(EGODEX_ROOT)):
        task_dir = os.path.join(EGODEX_ROOT, task)
        if not os.path.isdir(task_dir):
            continue
        for mp4 in natsorted(glob.glob(os.path.join(task_dir, "*.mp4"))):
            stem = os.path.splitext(os.path.basename(mp4))[0]
            eps.append({
                "seq_id": f"{task}/{stem}",
                "mp4": mp4,
            })
    return eps


def is_done(ep):
    out_dir = os.path.join(OUT_BASE, ep["seq_id"])
    return (os.path.exists(os.path.join(out_dir, "depth.npz")) and
            os.path.exists(os.path.join(out_dir, "meta.json")))


# ══════════════════════════════════════════════════════════════════════════════
# Worker: processes one episode at a time
# ══════════════════════════════════════════════════════════════════════════════

def init_worker(worker_id):
    """Initialise the ViPE pipeline (models loaded once per worker)."""
    import torch
    sys.path.insert(0, VIPE_DIR)

    # Set CUDA device (all on GPU 0 for single-GPU setup)
    torch.cuda.set_device(0)

    from omegaconf import OmegaConf
    from vipe.pipeline import make_pipeline

    # Load configs
    pipeline_cfg = OmegaConf.load(os.path.join(VIPE_DIR, "configs", "pipeline", "default.yaml"))
    slam_cfg = OmegaConf.load(os.path.join(VIPE_DIR, "configs", "slam", "default.yaml"))
    pipeline_cfg.slam = OmegaConf.merge(slam_cfg, pipeline_cfg.get("slam", {}))

    # Disable viz to save time
    pipeline_cfg.output.save_viz = False
    pipeline_cfg.output.save_slam_map = False
    pipeline_cfg.output.save_artifacts = True
    pipeline_cfg.output.skip_exists = False

    # Set output path
    worker_out = os.path.join(VIPE_RAW_OUT, f"worker_{worker_id}")
    pipeline_cfg.output.path = worker_out
    os.makedirs(worker_out, exist_ok=True)

    # Remove Hydra-specific keys not consumed by the pipeline constructor
    if "defaults" in pipeline_cfg:
        del pipeline_cfg["defaults"]

    pipeline = make_pipeline(pipeline_cfg)
    pipeline.return_output_streams = True

    return pipeline, pipeline_cfg


def process_episode(ep, pipeline, pipeline_cfg, worker_id):
    """Process a single episode through ViPE and convert output."""
    import torch
    from pathlib import Path
    from vipe.streams.raw_mp4_stream import RawMp4Stream
    from vipe.utils.io import ArtifactPath

    seq_id = ep["seq_id"]
    mp4_path = ep["mp4"]
    out_dir = os.path.join(OUT_BASE, seq_id)
    os.makedirs(out_dir, exist_ok=True)

    # Create video stream
    stream = RawMp4Stream(Path(mp4_path), name=seq_id.replace("/", "_"))

    n_frames = len(stream)
    if n_frames == 0:
        raise RuntimeError("No frames in video")

    # Run pipeline
    result = pipeline.run(stream)

    # Extract outputs from the result
    artifact_path = ArtifactPath(
        Path(pipeline_cfg.output.path),
        seq_id.replace("/", "_")
    )

    # Read pose
    if artifact_path.pose_path.exists():
        pose_data = np.load(artifact_path.pose_path)
        cam_c2w = pose_data["data"].astype(np.float32)  # (N, 4, 4)
        np.save(os.path.join(out_dir, "cam_c2w.npy"), cam_c2w)
    else:
        cam_c2w = None

    # Read intrinsics
    if artifact_path.intrinsics_path.exists():
        intr_data = np.load(artifact_path.intrinsics_path)
        K_vec = intr_data["data"][0].astype(np.float32)  # [fx, fy, cx, cy, ...]
        # Build 3x3 K matrix
        K = np.eye(3, dtype=np.float32)
        K[0, 0] = K_vec[0]
        K[1, 1] = K_vec[1]
        K[0, 2] = K_vec[2]
        K[1, 2] = K_vec[3]
        np.save(os.path.join(out_dir, "K.npy"), K)
    else:
        K = None

    # Read depth from zipped EXR
    if artifact_path.depth_path.exists():
        from vipe.utils.io import read_depth_artifacts
        depth_list = []
        for frame_idx, depth_tensor in read_depth_artifacts(artifact_path.depth_path):
            depth_list.append(depth_tensor.numpy())
        if depth_list:
            depths = np.stack(depth_list).astype(np.float32)
            np.savez_compressed(os.path.join(out_dir, "depth.npz"), depths=depths)
            n_out = len(depths)
        else:
            n_out = 0
    else:
        n_out = 0

    # Save metadata
    meta = {
        "seq_id": seq_id,
        "n_frames": n_out,
        "pipeline": "vipe-1.2.0",
        "depth_align_model": str(pipeline_cfg.post.depth_align_model),
        "slam_keyframe_depth": str(pipeline_cfg.slam.keyframe_depth),
    }
    if K is not None:
        meta["K"] = K.tolist()
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Clean up ViPE's raw output for this episode
    worker_out = pipeline_cfg.output.path
    for sub in ["rgb", "pose", "depth", "intrinsics", "mask", "flow", "vipe"]:
        sub_dir = os.path.join(worker_out, sub)
        if os.path.isdir(sub_dir):
            shutil.rmtree(sub_dir, ignore_errors=True)

    return n_out


def worker_loop(task_queue, result_queue, worker_id, total):
    """Main worker loop: init pipeline, then process episodes from queue."""
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        pipeline, pipeline_cfg = init_worker(worker_id)
        result_queue.put(("ready", worker_id, None))
    except Exception as e:
        result_queue.put(("init_error", worker_id, str(e)))
        return

    while True:
        try:
            item = task_queue.get(timeout=5)
        except Exception:
            break

        if item is None:  # Poison pill
            break

        idx, ep = item
        seq_id = ep["seq_id"]
        t0 = time.time()
        try:
            n_out = process_episode(ep, pipeline, pipeline_cfg, worker_id)
            elapsed = time.time() - t0
            result_queue.put(("ok", idx, {
                "seq": seq_id, "status": "ok",
                "n_frames": n_out, "elapsed_s": round(elapsed, 1),
                "worker": worker_id
            }))
        except Exception as e:
            traceback.print_exc()
            elapsed = time.time() - t0
            result_queue.put(("error", idx, {
                "seq": seq_id, "status": "error",
                "error": str(e), "elapsed_s": round(elapsed, 1),
                "worker": worker_id
            }))

        # Periodic cache clear
        torch.cuda.empty_cache()

    result_queue.put(("done", worker_id, None))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="ViPE depth/pose estimation for EgoDex (multi-worker)."
    )
    parser.add_argument("--resume", action="store_true",
                        help="Skip episodes that already have output")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of parallel workers (default 3 for ~85%% VRAM)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List episodes without processing")
    parser.add_argument("--single", action="store_true",
                        help="Run in single-process mode (for debugging)")
    args = parser.parse_args()

    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(VIPE_RAW_OUT, exist_ok=True)

    episodes = enumerate_egodex()
    print(f"\n  EgoDex input : {EGODEX_ROOT}")
    print(f"  Output dir   : {OUT_BASE}")
    print(f"  Total episodes: {len(episodes)}")

    episodes = episodes[args.start:args.end]

    if args.resume:
        n_before = len(episodes)
        episodes = [ep for ep in episodes if not is_done(ep)]
        print(f"  Resume: skipping {n_before - len(episodes)} done, "
              f"{len(episodes)} remaining")

    print(f"  Will process: {len(episodes)} episodes")
    print(f"  Workers: {args.workers}")

    if args.dry_run:
        for ep in episodes[:30]:
            print(f"    {ep['seq_id']}")
        if len(episodes) > 30:
            print(f"    ... and {len(episodes) - 30} more")
        return

    if len(episodes) == 0:
        print("  Nothing to do.")
        return

    log_path = os.path.join(OUT_BASE, "batch_log.jsonl")

    if args.single:
        # Single-process mode for debugging
        import torch
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        pipeline, pipeline_cfg = init_worker(0)

        n_ok = n_fail = 0
        for i, ep in enumerate(episodes):
            t0 = time.time()
            try:
                n_out = process_episode(ep, pipeline, pipeline_cfg, 0)
                elapsed = time.time() - t0
                n_ok += 1
                log_entry = {"seq": ep["seq_id"], "status": "ok",
                             "n_frames": n_out, "elapsed_s": round(elapsed, 1)}
                print(f"  ✅ [{i+1}/{len(episodes)}] {ep['seq_id']}  "
                      f"{n_out}f  {elapsed:.0f}s")
            except Exception as e:
                traceback.print_exc()
                n_fail += 1
                log_entry = {"seq": ep["seq_id"], "status": "error", "error": str(e)}
                print(f"  ❌ [{i+1}/{len(episodes)}] {ep['seq_id']}  {e}")

            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()

        print(f"\n  ✅ Done: {n_ok}   ❌ Failed: {n_fail}")
        return

    # Multi-process mode
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    # Start workers
    workers = []
    for w in range(args.workers):
        p = ctx.Process(target=worker_loop,
                        args=(task_queue, result_queue, w, len(episodes)))
        p.start()
        workers.append(p)

    # Wait for all workers to initialise
    ready_count = 0
    for _ in range(args.workers):
        msg_type, wid, data = result_queue.get(timeout=300)
        if msg_type == "ready":
            print(f"  🟢 Worker {wid} ready")
            ready_count += 1
        elif msg_type == "init_error":
            print(f"  🔴 Worker {wid} init failed: {data}")
    print(f"  {ready_count}/{args.workers} workers ready\n")

    if ready_count == 0:
        print("  No workers available. Exiting.")
        return

    # Enqueue tasks
    for i, ep in enumerate(episodes):
        task_queue.put((i, ep))

    # Send poison pills
    for _ in range(args.workers):
        task_queue.put(None)

    # Collect results
    n_ok = n_fail = 0
    done_workers = 0
    while done_workers < ready_count:
        msg_type, idx_or_wid, data = result_queue.get(timeout=600)
        if msg_type == "done":
            done_workers += 1
            continue
        elif msg_type == "ok":
            n_ok += 1
            print(f"  ✅ [{n_ok+n_fail}/{len(episodes)}] "
                  f"{data['seq']}  {data['n_frames']}f  "
                  f"{data['elapsed_s']}s  (W{data['worker']})")
        elif msg_type == "error":
            n_fail += 1
            print(f"  ❌ [{n_ok+n_fail}/{len(episodes)}] "
                  f"{data['seq']}  {data['error']}  (W{data['worker']})")

        with open(log_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    # Wait for processes to finish
    for p in workers:
        p.join(timeout=10)

    print(f"\n{'═'*60}")
    print(f"  ✅ Done: {n_ok}   ❌ Failed: {n_fail}")
    print(f"  Output : {OUT_BASE}")
    print(f"  Log    : {log_path}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
