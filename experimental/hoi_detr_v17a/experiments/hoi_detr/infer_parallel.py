#!/usr/bin/env python3
"""Batch the standalone HOI-DETR probe and report every failed video id."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.hoi_detr.probe_paths import probe_step_dir  # noqa: E402


STEP = "hoi_detr_probe"


def _video_paths(positional: list[Path], video_list: Path | None) -> list[Path]:
    paths = list(positional)
    if video_list is not None:
        for line in video_list.read_text(encoding="utf-8").splitlines():
            entry = line.strip()
            if entry and not entry.startswith("#"):
                paths.append(Path(entry))
    resolved = [path.expanduser().resolve() for path in paths]
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing video(s): {}".format(", ".join(map(str, missing))))
    if not resolved:
        raise ValueError("Pass one or more MP4 paths, or --video-list")
    ids = [path.stem for path in resolved]
    if len(ids) != len(set(ids)):
        raise ValueError("Video filename stems must be unique because they are used as video ids")
    return resolved


def _gpu_ids(value: str) -> list[int]:
    parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("at least one GPU id is required")
    return parsed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", type=Path, nargs="*")
    parser.add_argument("--video-list", type=Path, default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu-ids", type=_gpu_ids, default=[0])
    parser.add_argument("--hoi-detr-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--hf-threshold", type=float, default=0.6)
    parser.add_argument("--fs-threshold", type=float, default=0.92)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--checkpoint-revision", default=None)
    parser.add_argument("--checkpoint-size-bytes", type=int, default=None)
    parser.add_argument("--checkpoint-sha256", default=None)
    parser.add_argument("--checkpoint-authorized", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--loud", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _single_video_command(args: argparse.Namespace, video: Path, gpu_id: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_sequence.py")),
        "--dataset",
        args.dataset,
        "--video-id",
        video.stem,
        "--video",
        str(video),
        "--gpu",
        str(gpu_id),
        "--hoi-detr-root",
        str(args.hoi_detr_root.expanduser().resolve()),
        "--checkpoint",
        str(args.checkpoint.expanduser().resolve()),
        "--score-threshold",
        str(args.score_threshold),
        "--nms-iou",
        str(args.nms_iou),
        "--hf-threshold",
        str(args.hf_threshold),
        "--fs-threshold",
        str(args.fs_threshold),
        "--frame-stride",
        str(args.frame_stride),
    ]
    if args.config is not None:
        command.extend(["--config", str(args.config.expanduser().resolve())])
    if args.source_revision:
        command.extend(["--source-revision", args.source_revision])
    if args.checkpoint_revision:
        command.extend(["--checkpoint-revision", args.checkpoint_revision])
    if args.checkpoint_size_bytes is not None:
        command.extend(["--checkpoint-size-bytes", str(args.checkpoint_size_bytes)])
    if args.checkpoint_sha256:
        command.extend(["--checkpoint-sha256", args.checkpoint_sha256])
    if args.checkpoint_authorized:
        command.append("--checkpoint-authorized")
    if args.visualize:
        command.append("--visualize")
    if args.force:
        command.append("--force")
    return command


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        videos = _video_paths(args.videos, args.video_list)
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    commands = [
        _single_video_command(args, video, args.gpu_ids[index % len(args.gpu_ids)])
        for index, video in enumerate(videos)
    ]
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command))
        return 0

    failed: list[str] = []
    for video, command in zip(videos, commands):
        step_dir = probe_step_dir(args.dataset, video.stem, STEP)
        log_path = step_dir / ".logs" / f"{video.stem}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{video.stem}] starting", flush=True)
        if args.loud:
            result = subprocess.run(command, cwd=str(REPO_ROOT), check=False)
        else:
            with log_path.open("w", encoding="utf-8") as log_file:
                result = subprocess.run(
                    command,
                    cwd=str(REPO_ROOT),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        if result.returncode == 0:
            print(f"[{video.stem}] complete", flush=True)
        else:
            failed.append(video.stem)
            print(f"[{video.stem}] failed (log: {log_path})", flush=True)

    print(f"Done: {len(videos) - len(failed)}/{len(videos)} complete", flush=True)
    print("Failed videos: {}".format(failed if failed else "[]"), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
