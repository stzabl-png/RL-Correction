#!/usr/bin/env python3
"""Orchestrate the full hand+object reconstruction pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

RECON_ROOT = Path(__file__).resolve().parent
REPO_ROOT = RECON_ROOT.parent
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from _common.dataset import discover_videos, resolve_video_job, write_video_manifest  # noqa: E402
from _common.paths import INTERIM_ROOT, resolve_repo_path  # noqa: E402
from _common.step_launcher import build_step_cmd  # noqa: E402

EGO_ROOT = Path(__file__).resolve().parents[2]   # ego_pipeline/


def _post_sam2_contact_hook(dataset: str, video_id: str) -> None:
    """sam2_object 完成即有全片手/物 mask —— 立刻做 2D 接触检测,
    把 first_contact 写进 frame_plan(GT 实测该判据对首次接触 ±1 帧, 87% 在 ±3 帧内)。
    非致命: 失败只警告, 不挡重建。"""
    take = INTERIM_ROOT / dataset / video_id
    try:
        r = subprocess.run([sys.executable, "-m", "phase.detect", str(take), "--write-plan"],
                           cwd=EGO_ROOT, capture_output=True, text=True, timeout=600)
        tail = (r.stdout or r.stderr).strip().splitlines()
        print(f"[contact-early] {tail[-1] if tail else 'no output'}", flush=True)
        if r.returncode:
            print(f"[contact-early] 警告: detect rc={r.returncode}(不挡重建)", flush=True)
    except Exception as e:
        print(f"[contact-early] 警告: {e}(不挡重建)", flush=True)


STEPS = (
    "vipe",
    "sam3_hands",
    "label",
    "sam2_object",
    "hawor",
    "sam3d",
    "sam3d_scale",
    "fp_pose",
    "fuse",
    "confidence",
    "contact",
)

STEP_SCRIPTS = {
    "vipe": RECON_ROOT / "vipe" / "run_sequence.py",
    "sam3_hands": RECON_ROOT / "sam3_hands" / "run_sequence.py",
    "label": RECON_ROOT / "sam2_object" / "label_object.py",
    "sam2_object": RECON_ROOT / "sam2_object" / "run_sequence.py",
    "hawor": RECON_ROOT / "hawor" / "run_sequence.py",
    "sam3d": RECON_ROOT / "sam3d" / "run_sequence.py",
    "sam3d_scale": RECON_ROOT / "sam3d_scale" / "run_sequence.py",
    "fp_pose": RECON_ROOT / "fp_pose" / "run_sequence.py",
    "fuse": RECON_ROOT / "fuse" / "run_sequence.py",
    "confidence": RECON_ROOT / "confidence" / "run_sequence.py",  # 轨迹可信度打分+平滑, env: hawor
    "contact": RECON_ROOT / "contact" / "run_sequence.py",  # 手↔物接触点提取(2D修3D), CPU, env: hawor
}


def _run_step(
    step: str,
    job,
    *,
    gpu: int,
    visualize: bool,
    label_mode: str | None,
    http_host: str,
    http_port: int,
    extra: list[str],
) -> int:
    script = STEP_SCRIPTS[step]
    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        job.dataset,
        "--video-id",
        job.video_id,
        "--video",
        str(job.video_path),
        "--gpu",
        str(gpu),
    ]
    if step == "label":
        if label_mode is None:
            raise ValueError("--label-mode required for label step")
        cmd.extend(["--label-mode", label_mode, "--http-host", http_host, "--http-port", str(http_port)])
    elif visualize:
        cmd.append("--visualize")
    cmd.extend(extra)
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("input", type=Path, nargs="?", default=None)
    parser.add_argument("--video-list", type=Path, default=None)
    parser.add_argument("--video-id", default=None, help="Single video id to process")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--steps",
        default=",".join(STEPS),
        help=f"Comma-separated steps (default: all). Options: {','.join(STEPS)}",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--visualize", action="store_true", help="Optional per-step visualization")
    parser.add_argument(
        "--label-mode",
        choices=("headed", "http"),
        default=None,
        help="Required when running the label step",
    )
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8765)
    parser.add_argument("--force", action="store_true", help="Re-run steps even if complete")
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args(argv)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = set(steps) - set(STEPS)
    if unknown:
        parser.error(f"Unknown steps: {unknown}")

    if args.video_id and args.input is None:
        job = resolve_video_job(args.dataset, args.video_id, dataset_root=args.dataset_root)
        jobs = [job]
    else:
        jobs = discover_videos(
            args.dataset,
            args.input,
            dataset_root=args.dataset_root,
            video_list=args.video_list,
            sample=args.sample,
            limit=args.limit,
            seed=args.seed,
        )

    if args.sample is not None:
        write_video_manifest(jobs, INTERIM_ROOT / args.dataset / "pipeline_sample_manifest.json", seed=args.seed)

    rc = 0
    for job in jobs:
        print(f"\n=== {job.video_id} ===", flush=True)
        for step in steps:
            script = STEP_SCRIPTS[step]
            step_extra = list(extra)
            if step == "label":
                if args.label_mode is None:
                    print("Skip label (人工标注是 fallback; 自动路线: ego_pipeline/bin/"
                          "auto_label_v17a.py 会在标注缺失时自动出 mask, reconstruct.sh 默认启用; "
                          "确要人工才传 --label-mode headed|http)", flush=True)
                    continue
                cmd = [
                    sys.executable,
                    str(script),
                    "--dataset",
                    job.dataset,
                    "--video-id",
                    job.video_id,
                    "--video",
                    str(resolve_repo_path(job.video_path)),
                    "--label-mode",
                    args.label_mode,
                    "--http-host",
                    args.http_host,
                    "--http-port",
                    str(args.http_port),
                ]
                work_cwd = REPO_ROOT
            else:
                if args.visualize:
                    step_extra.append("--visualize")
                if args.force:
                    step_extra.append("--force")
                cmd, work_cwd = build_step_cmd(
                    step,
                    script,
                    job,
                    gpu_id=args.gpu,
                    dataset_root=args.dataset_root,
                    extra_args=step_extra,
                )
            if args.dry_run:
                print(f"  (cwd {work_cwd})", " ".join(cmd), flush=True)
                continue
            step_rc = subprocess.run(cmd, cwd=str(work_cwd)).returncode
            if step_rc != 0:
                print(f"Step {step} failed with code {step_rc}", flush=True)
                rc = step_rc
                break
            if step == "sam2_object":
                _post_sam2_contact_hook(job.dataset, job.video_id)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
