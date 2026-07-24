"""Build subprocess commands for reconstruction pipeline steps."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .dataset import VideoJob
from .paths import REPO_ROOT, resolve_repo_path

VIPE_ROOT = REPO_ROOT / "third_party" / "vipe"


def default_run_sequence_cmd(
    script_path: Path,
    job: VideoJob,
    *,
    gpu_id: int,
    dataset_root: Path | None,
    extra_args: list[str],
) -> tuple[list[str], Path]:
    """Return (command argv, subprocess cwd)."""
    import sys

    video = resolve_repo_path(job.video_path)
    cmd = [
        sys.executable,
        str(script_path.resolve()),
        "--dataset",
        job.dataset,
        "--video-id",
        job.video_id,
        "--video",
        str(video),
        "--gpu",
        str(gpu_id),
    ]
    if dataset_root is not None:
        cmd.extend(["--dataset-root", str(dataset_root.resolve())])
    cmd.extend(extra_args)
    return cmd, REPO_ROOT


def vipe_run_sequence_cmd(
    script_path: Path,
    job: VideoJob,
    *,
    gpu_id: int,
    dataset_root: Path | None,
    extra_args: list[str],
) -> tuple[list[str], Path]:
    """ViPE runs under third_party/vipe uv (.venv), not bare system python."""
    video = resolve_repo_path(job.video_path)
    cmd = [
        "uv",
        "run",
        "python",
        str(script_path.resolve()),
        "--dataset",
        job.dataset,
        "--video-id",
        job.video_id,
        "--video",
        str(video),
        "--gpu",
        str(gpu_id),
    ]
    if dataset_root is not None:
        cmd.extend(["--dataset-root", str(dataset_root.resolve())])
    cmd.extend(extra_args)
    return cmd, VIPE_ROOT


STEP_CMD_BUILDERS: dict[str, Callable[..., tuple[list[str], Path]]] = {
    "vipe": vipe_run_sequence_cmd,
}


def build_step_cmd(
    step: str,
    script_path: Path,
    job: VideoJob,
    *,
    gpu_id: int,
    dataset_root: Path | None,
    extra_args: list[str],
) -> tuple[list[str], Path]:
    builder = STEP_CMD_BUILDERS.get(step, default_run_sequence_cmd)
    return builder(
        script_path,
        job,
        gpu_id=gpu_id,
        dataset_root=dataset_root,
        extra_args=extra_args,
    )
