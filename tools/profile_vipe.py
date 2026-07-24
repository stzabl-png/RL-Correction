#!/usr/bin/env python3
"""
ViPE GPU Profiling Script — Measure VRAM usage and throughput per stage.

Profiles each stage of the ViPE pipeline independently:
  Stage 1: GeoCalib (intrinsics estimation)
  Stage 2: TrackAnything (instance segmentation + tracking)
  Stage 3: SLAM (DroidNet feature encoding + frontend/backend)
  Stage 4: Depth alignment (UniDepth-L / Video Depth Anything / PriorDA)

For each stage, we measure:
  - Peak VRAM (MB)
  - Latency per frame (ms)
  - Throughput (FPS)

Usage:
    conda activate biv2ap
    export LD_LIBRARY_PATH=$(python -c "import torch,os; print(os.path.join(os.path.dirname(torch.__file__),'lib'))"):$LD_LIBRARY_PATH
    python tools/profile_vipe.py --video <path_to_mp4> [--max-frames 60]
"""

import os, sys, time, argparse, gc
import numpy as np
import torch
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BIV2AP_DIR = os.path.dirname(SCRIPT_DIR)
VIPE_DIR   = os.path.join(BIV2AP_DIR, "third_party", "vipe")
sys.path.insert(0, VIPE_DIR)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_gpu_mem_mb():
    """Current GPU memory allocated (MB)."""
    return torch.cuda.memory_allocated() / 1024**2

def get_gpu_mem_reserved_mb():
    """Current GPU memory reserved by PyTorch (MB)."""
    return torch.cuda.memory_reserved() / 1024**2

def get_gpu_mem_total_mb():
    """Total GPU memory (MB)."""
    return torch.cuda.get_device_properties(0).total_memory / 1024**2

def reset_peak():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

def get_peak_mb():
    return torch.cuda.max_memory_allocated() / 1024**2


def extract_frames(video_path, max_frames=60, long_dim=640):
    """Extract frames from MP4, resize, return as list of numpy RGB arrays."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)
    idxs = list(range(0, total, step))[:max_frames]
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        if w >= h:
            nw, nh = long_dim, int(round(long_dim * h / w))
        else:
            nh, nw = long_dim, int(round(long_dim * w / h))
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
    return frames, fps


def profile_stage(name, fn, warmup=1, repeats=3):
    """Profile a function: returns (peak_vram_mb, avg_time_s)."""
    # Warmup
    for _ in range(warmup):
        fn()
        torch.cuda.synchronize()

    reset_peak()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak = get_peak_mb()
    avg_time = np.mean(times)
    return peak, avg_time


# ─── Stage profilers ─────────────────────────────────────────────────────────

def profile_geocalib(frames):
    """Profile GeoCalib intrinsics estimation."""
    from vipe.priors.geocalib import GeoCalib

    print("\n━━━ Stage 1: GeoCalib (Intrinsics) ━━━")
    reset_peak()
    baseline = get_gpu_mem_mb()

    model = GeoCalib(weights="pinhole").cuda().eval()
    model_mem = get_gpu_mem_mb() - baseline
    print(f"  Model VRAM: {model_mem:.0f} MB")

    # Prepare input: 3 sample frames
    sample_idxs = [0, min(len(frames)//4, len(frames)-1), min(len(frames)//2, len(frames)-1)]
    sample_batch = torch.stack([
        torch.from_numpy(frames[i]).permute(2, 0, 1).float() / 255.0
        for i in sample_idxs
    ]).cuda()

    def run():
        with torch.no_grad():
            model.calibrate(sample_batch, shared_intrinsics=True)

    peak, avg_time = profile_stage("GeoCalib", run)
    print(f"  Peak VRAM: {peak:.0f} MB")
    print(f"  Time: {avg_time*1000:.1f} ms (for {len(sample_idxs)} frames)")

    del model, sample_batch
    torch.cuda.empty_cache()
    return {"name": "GeoCalib", "model_mb": model_mem, "peak_mb": peak, "time_s": avg_time, "n_frames": len(sample_idxs)}


def profile_track_anything(frames):
    """Profile TrackAnything (GroundingDINO + SAM + AOT)."""
    from vipe.priors.track_anything import TrackAnythingPipeline
    from vipe.streams.base import VideoFrame

    print("\n━━━ Stage 2: TrackAnything (Segmentation) ━━━")
    reset_peak()
    baseline = get_gpu_mem_mb()

    phrases = ["person", "animal", "vehicle"]
    tracker = TrackAnythingPipeline(
        phrases,
        sam_points_per_side=50,
        sam_run_gap=30,
    )
    model_mem = get_gpu_mem_mb() - baseline
    print(f"  Model VRAM: {model_mem:.0f} MB")

    # Profile first frame (detection + SAM + AOT init — heaviest)
    frame_rgb = torch.from_numpy(frames[0]).cuda()
    vf = VideoFrame(raw_frame_idx=0, rgb=frame_rgb)

    reset_peak()
    t0 = time.perf_counter()
    with torch.no_grad():
        tracker.track(vf)
    torch.cuda.synchronize()
    first_time = time.perf_counter() - t0
    first_peak = get_peak_mb()
    print(f"  First frame peak VRAM: {first_peak:.0f} MB, time: {first_time*1000:.1f} ms")

    # Profile subsequent frames (AOT tracking only — lighter)
    if len(frames) > 1:
        times = []
        reset_peak()
        for i in range(1, min(10, len(frames))):
            frame_rgb = torch.from_numpy(frames[i]).cuda()
            vf = VideoFrame(raw_frame_idx=i, rgb=frame_rgb)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                tracker.track(vf)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        track_peak = get_peak_mb()
        track_avg = np.mean(times)
        print(f"  Tracking peak VRAM: {track_peak:.0f} MB, avg: {track_avg*1000:.1f} ms/frame")
    else:
        track_peak = first_peak
        track_avg = first_time

    del tracker
    torch.cuda.empty_cache()
    return {"name": "TrackAnything", "model_mb": model_mem,
            "first_peak_mb": first_peak, "first_time_s": first_time,
            "track_peak_mb": track_peak, "track_avg_s": track_avg}


def profile_droidnet(frames):
    """Profile DroidNet feature/context encoding (SLAM core)."""
    from vipe.slam.networks.droid_net import get_droid_net

    print("\n━━━ Stage 3: DroidNet (SLAM Feature Encoding) ━━━")
    reset_peak()
    baseline = get_gpu_mem_mb()

    device = torch.device("cuda")
    net = get_droid_net(device)
    model_mem = get_gpu_mem_mb() - baseline
    print(f"  Model VRAM: {model_mem:.0f} MB")

    # Prepare SLAM-sized images (384x512 standard resize)
    h0, w0 = frames[0].shape[:2]
    scale = np.sqrt((384 * 512) / (h0 * w0))
    h1 = int(h0 * scale)
    w1 = int(w0 * scale)
    h1 -= h1 % 8
    w1 -= w1 % 8

    def make_tensor(frame):
        img = cv2.resize(frame, (w1, h1), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0

    # Profile single-frame encoding
    img_t = make_tensor(frames[0])

    def run_encode():
        with torch.no_grad():
            fmap = net.encode_features(img_t)
            net_out, inp_out = net.encode_context(img_t)
        return fmap, net_out, inp_out

    peak, avg_time = profile_stage("DroidNet encode", run_encode, warmup=2, repeats=5)
    print(f"  Single frame encode: peak {peak:.0f} MB, time {avg_time*1000:.1f} ms")

    # Profile with buffer (simulating N keyframes in memory)
    for n_kf in [30, 60, 120]:
        if n_kf > len(frames):
            break
        reset_peak()
        # Simulate buffer allocation
        buffers = {
            "fmaps": torch.zeros(n_kf, 128, h1//8, w1//8, device="cuda"),
            "nets":  torch.zeros(n_kf, 128, h1//8, w1//8, device="cuda"),
            "inps":  torch.zeros(n_kf, 128, h1//8, w1//8, device="cuda"),
            "images": torch.zeros(n_kf, 3, h1, w1, device="cuda"),
            "disps":  torch.zeros(n_kf, h1//8, w1//8, device="cuda"),
        }
        buf_mem = get_peak_mb()
        print(f"  Buffer ({n_kf} keyframes): {buf_mem:.0f} MB")
        del buffers
        torch.cuda.empty_cache()

    del net, img_t
    torch.cuda.empty_cache()
    return {"name": "DroidNet", "model_mb": model_mem, "encode_peak_mb": peak,
            "encode_time_s": avg_time, "img_size": (h1, w1)}


def profile_depth_models(frames):
    """Profile depth estimation models used in post-processing."""
    from vipe.priors.depth import DepthEstimationInput, make_depth_model

    results = {}

    for model_name in ["unidepth-l", "metric3d-small"]:
        print(f"\n━━━ Stage 4a: Depth Model '{model_name}' ━━━")
        reset_peak()
        baseline = get_gpu_mem_mb()

        try:
            model = make_depth_model(model_name)
            model_mem = get_gpu_mem_mb() - baseline
            print(f"  Model VRAM: {model_mem:.0f} MB")

            rgb = torch.from_numpy(frames[0]).float().cuda()
            inp = DepthEstimationInput(rgb=rgb)

            def run():
                with torch.no_grad():
                    model.estimate(inp)

            peak, avg_time = profile_stage(model_name, run, warmup=1, repeats=3)
            print(f"  Peak VRAM: {peak:.0f} MB, time: {avg_time*1000:.1f} ms/frame")
            results[model_name] = {"model_mb": model_mem, "peak_mb": peak, "time_s": avg_time}

            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ⚠️ Failed: {e}")
            results[model_name] = {"error": str(e)}

    # Profile VideoDepthAnything (video depth model)
    print(f"\n━━━ Stage 4b: VideoDepthAnything ━━━")
    try:
        from vipe.priors.depth.videodepthanything import VideoDepthAnythingDepthModel
        from vipe.priors.depth import DepthEstimationInput

        for variant in ["vits", "vitl"]:
            print(f"\n  --- VDA-{variant} ---")
            reset_peak()
            baseline = get_gpu_mem_mb()
            model = VideoDepthAnythingDepthModel(model=variant)
            model_mem = get_gpu_mem_mb() - baseline
            print(f"  Model VRAM: {model_mem:.0f} MB")

            for n_frames in [10, 30, 60]:
                if n_frames > len(frames):
                    break
                frame_list = [frames[i] for i in range(n_frames)]
                inp = DepthEstimationInput(video_frame_list=frame_list)

                reset_peak()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model.estimate(inp)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0
                peak = get_peak_mb()
                print(f"    {n_frames} frames: peak {peak:.0f} MB, "
                      f"time {elapsed:.1f}s ({elapsed/n_frames*1000:.0f} ms/f)")
                results[f"vda-{variant}-{n_frames}f"] = {
                    "model_mb": model_mem, "peak_mb": peak,
                    "time_s": elapsed, "n_frames": n_frames
                }
            del model
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ⚠️ Failed: {e}")

    return results


def profile_full_pipeline(video_path, max_frames):
    """Run full ViPE pipeline and measure total VRAM."""
    from omegaconf import OmegaConf
    from vipe.config import validate_typed_config
    from vipe.streams.raw_mp4_stream import RawMp4VideoStream
    from vipe.pipeline import make_pipeline

    print("\n━━━ Full Pipeline Profiling ━━━")

    cfg = OmegaConf.load(os.path.join(VIPE_DIR, "configs", "default.yaml"))
    pipeline_cfg = OmegaConf.load(os.path.join(VIPE_DIR, "configs", "pipeline", "default.yaml"))
    slam_cfg = OmegaConf.load(os.path.join(VIPE_DIR, "configs", "slam", "default.yaml"))

    pipeline_cfg.slam = OmegaConf.merge(slam_cfg, pipeline_cfg.get("slam", {}))
    pipeline_cfg.output.save_viz = False
    pipeline_cfg.output.save_artifacts = False
    pipeline_cfg.output.save_slam_map = False
    pipeline_cfg.output.path = "/tmp/vipe_profile_output"

    reset_peak()
    t0 = time.perf_counter()
    pipeline = make_pipeline(pipeline_cfg)
    pipeline_build_time = time.perf_counter() - t0
    pipeline_build_mem = get_peak_mb()
    print(f"  Pipeline build: {pipeline_build_time:.1f}s, peak VRAM: {pipeline_build_mem:.0f} MB")

    # Create stream
    stream_cfg = OmegaConf.create({
        "paths": [video_path],
        "max_frames": max_frames,
        "instance": "vipe.streams.raw_mp4_stream.RawMp4VideoStream",
    })

    reset_peak()
    t0 = time.perf_counter()
    stream = RawMp4VideoStream(video_path, max_frames=max_frames)
    pipeline.run(stream)
    total_time = time.perf_counter() - t0
    total_peak = get_peak_mb()
    print(f"  Full run: {total_time:.1f}s, peak VRAM: {total_peak:.0f} MB")
    print(f"  Throughput: {max_frames/total_time:.2f} FPS")

    return {"build_time_s": pipeline_build_time, "build_peak_mb": pipeline_build_mem,
            "total_time_s": total_time, "total_peak_mb": total_peak,
            "n_frames": max_frames, "fps": max_frames/total_time}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Profile ViPE GPU usage per stage")
    parser.add_argument("--video", type=str, required=True, help="Path to input MP4")
    parser.add_argument("--max-frames", type=int, default=60, help="Max frames to extract")
    parser.add_argument("--skip-full", action="store_true", help="Skip full pipeline run")
    parser.add_argument("--only", type=str, default=None,
                        help="Only run specific stage: geocalib, track, droid, depth, full")
    args = parser.parse_args()

    assert os.path.exists(args.video), f"Video not found: {args.video}"

    print(f"╔{'═'*60}╗")
    print(f"║  ViPE GPU Profiler — RTX 5090 (32 GB VRAM)              ║")
    print(f"╠{'═'*60}╣")
    print(f"║  Video: {os.path.basename(args.video)[:45]:45s}        ║")
    print(f"║  Max frames: {args.max_frames:<5d}                                   ║")
    print(f"╚{'═'*60}╝")

    # Extract frames
    print("\nExtracting frames...")
    frames, video_fps = extract_frames(args.video, args.max_frames)
    print(f"  Extracted {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}, video FPS: {video_fps:.1f}")

    total_vram = get_gpu_mem_total_mb()
    target_vram = total_vram * 0.85
    print(f"  GPU total: {total_vram:.0f} MB, target 85%: {target_vram:.0f} MB")

    results = {}

    stages = args.only.split(",") if args.only else ["geocalib", "track", "droid", "depth"]

    if "geocalib" in stages:
        results["geocalib"] = profile_geocalib(frames)

    if "track" in stages:
        results["track"] = profile_track_anything(frames)

    if "droid" in stages:
        results["droid"] = profile_droidnet(frames)

    if "depth" in stages:
        results["depth"] = profile_depth_models(frames)

    if "full" in stages and not args.skip_full:
        results["full"] = profile_full_pipeline(args.video, args.max_frames)

    # ─── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'━'*65}")
    print(f"  📊 VRAM USAGE SUMMARY (GPU total: {total_vram:.0f} MB)")
    print(f"{'━'*65}")
    print(f"  {'Stage':<30s} {'Model':>8s} {'Peak':>8s} {'Time':>10s}")
    print(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*10}")

    for key, val in results.items():
        if isinstance(val, dict) and "peak_mb" in val:
            model_s = f"{val.get('model_mb', 0):.0f} MB"
            peak_s  = f"{val['peak_mb']:.0f} MB"
            time_s  = f"{val.get('time_s', 0)*1000:.0f} ms"
            print(f"  {val.get('name', key):<30s} {model_s:>8s} {peak_s:>8s} {time_s:>10s}")
        elif isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, dict) and "peak_mb" in sub_val:
                    peak_s = f"{sub_val['peak_mb']:.0f} MB"
                    time_s = f"{sub_val.get('time_s', 0)*1000:.0f} ms"
                    model_s = f"{sub_val.get('model_mb', 0):.0f} MB"
                    print(f"  {sub_key:<30s} {model_s:>8s} {peak_s:>8s} {time_s:>10s}")

    print(f"\n  85% VRAM target: {target_vram:.0f} MB")
    print(f"{'━'*65}\n")


if __name__ == "__main__":
    main()
