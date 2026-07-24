#!/usr/bin/env python3
"""Run ViPE camera + depth estimation for one video (interim output).

Invoke via ViPE uv (from repo root):

  cd third_party/vipe
  uv run python ../../recon_pipeline/vipe/run_sequence.py --dataset ... --video-id ... --video ... --gpu 0

Or use the shell wrapper: bash recon_pipeline/vipe/run_sequence.sh ...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

RECON_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = RECON_ROOT.parent
VIPE_ROOT = REPO_ROOT / "third_party" / "vipe"

if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))


def _load_vipe():
    import importlib.util

    name = "recon_vipe_pipeline_common"
    module_path = REPO_ROOT / "recon_pipeline" / "_legacy" / "vipe" / "_common.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ViPE helpers from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _estimate_gravity_artifact(step_dir: Path, video_id: str, video_path: Path) -> dict:
    import cv2
    import torch

    from _common.gravity import GRAVITY_SCHEMA_VERSION, vipe_gravity_path
    from _common.io import apply_vipe_focal_flip_c2w, count_video_frames, interpolate_c2w_poses, load_vipe_intrinsics, load_vipe_poses
    from vipe.priors.geocalib import GeoCalib

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    num_frames = count_video_frames(video_path)
    gap = min(int(fps), max(0, (num_frames - 1) // 2))
    sample_indices = np.array(sorted(set([0, gap, min(2 * gap, num_frames - 1)])), dtype=np.int32)
    frames = []
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} for ViPE gravity estimation: {video_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(rgb).float().div(255.0).permute(2, 0, 1))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames available for ViPE gravity estimation: {video_path}")

    model = GeoCalib(weights="pinhole").cuda()
    with torch.inference_mode():
        result = model.calibrate(torch.stack(frames).cuda(), shared_intrinsics=True)
    # GeoCalib names this object "Gravity", but its vec3d is the direction that
    # projects to the image up-field. In OpenCV camera coordinates, the neutral
    # roll/pitch vector is [0, -1, 0], i.e. image-up, not physical gravity.
    up_cam = result["gravity"].vec3d.detach().cpu().numpy().astype(np.float64)
    gravity_cam = -up_cam
    gravity_uncertainty = (
        result.get("gravity_uncertainty").detach().cpu().numpy().astype(np.float64)
        if result.get("gravity_uncertainty") is not None
        else np.zeros((0,), dtype=np.float64)
    )
    if up_cam.ndim == 1:
        up_cam = up_cam.reshape(1, 3)
        gravity_cam = gravity_cam.reshape(1, 3)
    if up_cam.shape[0] == 1 and len(sample_indices) > 1:
        up_cam = np.repeat(up_cam, len(sample_indices), axis=0)
        gravity_cam = np.repeat(gravity_cam, len(sample_indices), axis=0)

    inds, c2w_all = load_vipe_poses(step_dir, video_id)
    _k_mat, flip = load_vipe_intrinsics(step_dir, video_id)
    c2w_all = interpolate_c2w_poses(inds, c2w_all, num_frames)
    if flip:
        c2w_all = np.stack([apply_vipe_focal_flip_c2w(p) for p in c2w_all])
    gravity_world_samples = []
    up_world_samples = []
    for local_idx, frame_idx in enumerate(sample_indices):
        g_cam = gravity_cam[min(local_idx, len(gravity_cam) - 1)]
        u_cam = up_cam[min(local_idx, len(up_cam) - 1)]
        g_world = c2w_all[int(frame_idx), :3, :3] @ g_cam
        u_world = c2w_all[int(frame_idx), :3, :3] @ u_cam
        g_norm = float(np.linalg.norm(g_world))
        u_norm = float(np.linalg.norm(u_world))
        if g_norm > 1e-9 and u_norm > 1e-9:
            gravity_world_samples.append(g_world / g_norm)
            up_world_samples.append(u_world / u_norm)
    if not gravity_world_samples:
        raise RuntimeError("GeoCalib gravity vectors were invalid")
    gravity_world = np.mean(np.stack(gravity_world_samples), axis=0)
    gravity_world = gravity_world / max(float(np.linalg.norm(gravity_world)), 1e-9)
    up_world = np.mean(np.stack(up_world_samples), axis=0)
    up_world = up_world / max(float(np.linalg.norm(up_world)), 1e-9)

    out_path = vipe_gravity_path(step_dir, video_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        schema_version=GRAVITY_SCHEMA_VERSION,
        source="vipe_geocalib",
        sample_frame_indices=sample_indices,
        gravity_cam=gravity_cam,
        up_cam=up_cam,
        gravity_world_samples=np.stack(gravity_world_samples),
        up_world_samples=np.stack(up_world_samples),
        gravity_world=gravity_world,
        up_world=up_world,
        gravity_uncertainty=gravity_uncertainty,
    )
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": GRAVITY_SCHEMA_VERSION,
                "source": "vipe_geocalib",
                "sample_frame_indices": sample_indices.tolist(),
                "gravity_world": gravity_world.tolist(),
                "up_world": up_world.tolist(),
                "gravity_uncertainty": gravity_uncertainty.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"gravity_npz": str(out_path), "gravity_world": gravity_world.tolist(), "up_world": up_world.tolist()}


def run_vipe(job, *, gpu: int, visualize: bool, force: bool) -> dict:
    from _common.gravity import vipe_gravity_path
    from _common.paths import interim_step_dir, is_step_complete, resolve_repo_path, write_step_completion

    vipe = _load_vipe()
    step_dir = interim_step_dir(job.dataset, job.video_id, "vipe")
    video_path = resolve_repo_path(job.video_path)

    if not force and is_step_complete(step_dir, "vipe") and vipe.is_sequence_complete(step_dir, job.video_id):
        if vipe_gravity_path(step_dir, job.video_id).is_file():
            print(f"[vipe] skipped (complete): {job.video_id}", flush=True)
            return {"skipped": True}
        print(f"[vipe] gravity missing; estimating without rerunning ViPE: {job.video_id}", flush=True)
        gravity_extra = _estimate_gravity_artifact(step_dir, job.video_id, video_path)
        write_step_completion(
            step_dir,
            "vipe",
            dataset=job.dataset,
            video_id=job.video_id,
            extra=gravity_extra,
        )
        return gravity_extra

    if not video_path.is_file():
        raise FileNotFoundError(f"RGB video not found: {video_path}")

    step_dir.mkdir(parents=True, exist_ok=True)
    log_path = step_dir / ".logs" / f"{job.video_id}.log"
    config = vipe.VipeRunConfig(
        input_path=video_path,
        output_dir=step_dir.resolve(),
        vipe_root=vipe.VIPE_ROOT,
        discard_nonessential_artifacts=not visualize,
    )
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[vipe] running {job.video_id} → {step_dir}", flush=True)
    vipe.run_vipe_inference(
        video_path,
        config,
        gpu,
        sequence_name=job.video_id,
        loud=False,
        log_path=log_path,
    )

    if not vipe.is_sequence_complete(step_dir, job.video_id):
        log_hint = (
            f"see log {log_path}"
            if log_path.is_file() and log_path.stat().st_size
            else f"log empty at {log_path}"
        )
        raise RuntimeError(f"ViPE artifacts missing under {step_dir} ({log_hint})")

    gravity_extra = _estimate_gravity_artifact(step_dir, job.video_id, video_path)

    from _common.viz import vis_path  # noqa: WPS433 — local import avoids circular deps at module load

    extra = dict(gravity_extra)
    if visualize:
        vis_script = REPO_ROOT / "recon_pipeline" / "_legacy" / "vipe" / "visualize_outputs.py"
        if vis_script.is_file():
            hoi4d_root = REPO_ROOT / "data" / "raw" / "HOI4D"
            subprocess.run(
                [
                    "uv",
                    "run",
                    "python",
                    str(vis_script),
                    "--output-dir",
                    str(step_dir),
                    "--sequence",
                    job.video_id,
                    "--hoi4d-root",
                    str(hoi4d_root),
                    "--overwrite",
                ],
                cwd=str(VIPE_ROOT),
                check=False,
            )
            legacy_vis = step_dir / "vis" / f"{job.video_id}_vis.mp4"
            std_vis = vis_path(step_dir, job.video_id)
            if legacy_vis.is_file() and legacy_vis != std_vis:
                std_vis.parent.mkdir(parents=True, exist_ok=True)
                legacy_vis.rename(std_vis)
            if std_vis.is_file():
                extra["vis_video"] = str(std_vis)

    vipe.discard_vipe_nonessential_artifacts(step_dir, job.video_id)

    completion_extra = dict(extra)
    try:
        from _common.io import count_video_frames

        completion_extra["num_frames"] = count_video_frames(video_path)
    except Exception as exc:
        completion_extra["num_frames_note"] = str(exc)

    write_step_completion(
        step_dir,
        "vipe",
        dataset=job.dataset,
        video_id=job.video_id,
        extra=completion_extra,
    )
    print(f"[vipe] done {job.video_id}", flush=True)
    return {"skipped": False, "output_dir": str(step_dir)}


def main(argv: list[str] | None = None) -> int:
    from _common.dataset import VideoJob
    from _common.paths import resolve_repo_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    job = VideoJob(
        dataset=args.dataset,
        video_id=args.video_id,
        video_path=resolve_repo_path(args.video),
    )
    run_vipe(job, gpu=args.gpu, visualize=args.visualize, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
