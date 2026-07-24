#!/usr/bin/env python3
"""FoundationPose object pose with FP++ tracking by default."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parents[1]
FP_DIR = Path(__file__).resolve().parent
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))
if str(FP_DIR) not in sys.path:
    sys.path.insert(0, str(FP_DIR))

from _common.dataset import VideoJob  # noqa: E402
from _common.io import load_vipe_intrinsics  # noqa: E402
from _common.paths import interim_step_dir, is_step_complete, write_step_completion  # noqa: E402
from _common.viz import encode_fp_pose_vis_mp4, vis_path  # noqa: E402
from sam2_object.sam2_object_common import OBJECT_MASK_ID  # noqa: E402


def _remap_fp_gpu_arg(argv: list[str] | None) -> list[str] | None:
    """Set CUDA visibility before FoundationPose imports; return argv with local --gpu 0."""
    if argv is None:
        argv = sys.argv[1:]
    else:
        argv = list(argv)
    if "--gpu" not in argv:
        return argv
    gpu_idx = argv.index("--gpu")
    if gpu_idx + 1 >= len(argv):
        return argv
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = argv[gpu_idx + 1]
    argv[gpu_idx + 1] = "0"
    return argv


def _render_fp_pose_vis(job: VideoJob, step_dir: Path) -> str:
    from fp_common import scaled_mesh_path

    vipe_dir = interim_step_dir(job.dataset, job.video_id, "vipe")
    obj_dir = interim_step_dir(job.dataset, job.video_id, "sam2_object")
    k_mat, _ = load_vipe_intrinsics(vipe_dir, job.video_id)
    out_vis = encode_fp_pose_vis_mp4(
        video_path=job.video_path,
        ob_in_cam_dir=step_dir / "ob_in_cam",
        masks_dir=obj_dir / "video_segmentation" / "masks",
        mask_filename=f"{OBJECT_MASK_ID}.png",
        k_mat=k_mat,
        mesh_path=scaled_mesh_path(job.dataset, job.video_id),
        start_frame=0,
        out_path=vis_path(step_dir, job.video_id),
    )
    return str(out_vis)


def run_fp_pose(
    job: VideoJob,
    *,
    gpu: int,
    depth_scale: float,
    use_kf: bool,
    kf_scale: float,
    register_iters: int,
    track_iters: int,
    pose_mode: str,
    visualize: bool,
    force: bool,
    vis_only: bool = False,
) -> dict:
    step_dir = interim_step_dir(job.dataset, job.video_id, "fp_pose")
    if vis_only:
        if not (step_dir / "ob_in_cam").is_dir() or not any((step_dir / "ob_in_cam").glob("*.txt")):
            raise FileNotFoundError(f"No fp_pose ob_in_cam outputs under {step_dir}. Run fp_pose first.")
        meta = {"vis_only": True}
        if visualize:
            meta["vis_video"] = _render_fp_pose_vis(job, step_dir)
        return meta

    if not force and is_step_complete(step_dir, "fp_pose"):
        return {"skipped": True}

    from fp_common import run_fp_pp_track

    meta = run_fp_pp_track(
        dataset=job.dataset,
        video_id=job.video_id,
        video_path=job.video_path,
        gpu=gpu,
        depth_scale=depth_scale,
        use_kf=use_kf,
        kf_scale=kf_scale,
        register_iters=register_iters,
        track_iters=track_iters,
        pose_mode=pose_mode,
    )

    if visualize:
        meta["vis_video"] = _render_fp_pose_vis(job, step_dir)

    write_step_completion(step_dir, "fp_pose", dataset=job.dataset, video_id=job.video_id, extra=meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    argv = _remap_fp_gpu_arg(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--depth-scale", type=float, default=1.0, help="Scale ViPE depth to metres")
    parser.add_argument("--no-kf", action="store_true")
    parser.add_argument("--kf-scale", type=float, default=0.05)
    parser.add_argument("--register-iters", type=int, default=5)
    parser.add_argument("--track-iters", type=int, default=1)
    parser.add_argument(
        "--pose-mode",
        choices=("register-each", "track"),
        default="track",
        help="track registers the prompt frame, then tracks forward and backward to cover every frame; "
        "register-each additionally runs FoundationPose register with the per-frame SAM2 mask when available",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--vis-only", action="store_true", help="Re-render fp_pose/vis from existing ob_in_cam")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    print(
        f"[fp_pose] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} local_gpu={args.gpu}",
        flush=True,
    )

    job = VideoJob(dataset=args.dataset, video_id=args.video_id, video_path=args.video.resolve())
    run_fp_pose(
        job,
        gpu=args.gpu,
        depth_scale=args.depth_scale,
        use_kf=not args.no_kf,
        kf_scale=args.kf_scale,
        register_iters=args.register_iters,
        track_iters=args.track_iters,
        pose_mode=args.pose_mode,
        visualize=args.visualize,
        force=args.force,
        vis_only=args.vis_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
