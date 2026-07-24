"""Generic parallel runner for one-video-per-task reconstruction steps."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any, Callable

from .dataset import VideoJob, discover_videos, write_video_manifest
from .paths import REPO_ROOT, is_step_complete
from .step_launcher import build_step_cmd


@dataclass
class ParallelConfig:
    step: str
    dataset: str
    script_rel: str
    gpu_ids: list[int]
    procs_per_gpu: int
    resume: bool
    force: bool
    loud: bool
    dry_run: bool
    extra_args: list[str]


def add_common_parallel_args(parser: argparse.ArgumentParser, *, step: str) -> None:
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g. hoi4d)")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Raw dataset root (default: data/raw/HOI4D for hoi4d)",
    )
    parser.add_argument("input", type=Path, nargs="?", default=None, help="Optional input root or single MP4")
    parser.add_argument("--video-list", type=Path, default=None, help="Text/JSON list of videos or ids")
    parser.add_argument("--sample", type=int, default=None, help="Random sample size")
    parser.add_argument("--limit", type=int, default=None, help="First N videos (sorted)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --sample")
    parser.add_argument("--gpu-ids", type=str, default="0", help="Comma-separated GPU ids")
    parser.add_argument("--procs-per-gpu", type=int, default=1, help="Worker processes per GPU")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true", help="Re-run even if step complete")
    parser.add_argument("--loud", action="store_true", help="Stream subprocess stdout to terminal")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands only")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=f"Enable optional {step} visualization outputs",
    )


def parse_gpu_list(gpus: str) -> list[int]:
    values = [int(p.strip()) for p in gpus.split(",") if p.strip()]
    if not values:
        raise ValueError("At least one GPU id required in --gpu-ids")
    return values


def _step_complete_fn(step: str) -> Callable[[VideoJob], bool]:
    from .paths import interim_step_dir

    def _check(job: VideoJob) -> bool:
        return is_step_complete(interim_step_dir(job.dataset, job.video_id, step), step)

    return _check


def build_run_sequence_cmd(
    script_path: Path,
    job: VideoJob,
    *,
    step: str,
    gpu_id: int,
    dataset_root: Path | None,
    extra_args: list[str],
) -> tuple[list[str], Path]:
    return build_step_cmd(
        step,
        script_path,
        job,
        gpu_id=gpu_id,
        dataset_root=dataset_root,
        extra_args=extra_args,
    )


@dataclass
class _WorkerState:
    worker_id: int
    gpu_id: int
    script_path: Path
    dataset_root: Path | None
    step: str
    resume: bool
    force: bool
    loud: bool
    extra_args: list[str]
    print_lock: Any
    completed: Any
    failed: Any
    running: Any
    total: int


def _worker_main(task_queue: mp.Queue, cfg: _WorkerState) -> None:
    env_base = os.environ.copy()
    env_base.setdefault("OMP_NUM_THREADS", "1")
    env_base.setdefault("MKL_NUM_THREADS", "1")
    env_base.setdefault("TOKENIZERS_PARALLELISM", "false")

    local_done = 0
    while True:
        try:
            job: VideoJob | None = task_queue.get(timeout=0.5)
        except Empty:
            continue
        if job is None:
            break

        from .paths import interim_step_dir

        step_dir = interim_step_dir(job.dataset, job.video_id, cfg.step)
        if cfg.resume and not cfg.force and is_step_complete(step_dir, cfg.step):
            with cfg.print_lock:
                cfg.completed.value += 1
                print(
                    f"[gpu {cfg.gpu_id} worker {cfg.worker_id}] skipped {job.video_id} (complete) "
                    f"[overall] completed {cfg.completed.value}/{cfg.total}, "
                    f"failed {cfg.failed.value}, running {cfg.running.value}",
                    flush=True,
                )
            continue

        cfg.running.value += 1
        cmd, work_cwd = build_run_sequence_cmd(
            cfg.script_path,
            job,
            step=cfg.step,
            gpu_id=cfg.gpu_id,
            dataset_root=cfg.dataset_root,
            extra_args=cfg.extra_args,
        )
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
        log_path = step_dir / ".logs" / f"{job.video_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if cfg.loud:
                subprocess.run(cmd, cwd=str(work_cwd), env=env, check=True)
            else:
                with log_path.open("w", encoding="utf-8") as log_f:
                    subprocess.run(
                        cmd,
                        cwd=str(work_cwd),
                        env=env,
                        check=True,
                        stdout=log_f,
                        stderr=subprocess.STDOUT,
                    )
            with cfg.print_lock:
                cfg.completed.value += 1
                local_done += 1
                print(
                    f"[gpu {cfg.gpu_id} worker {cfg.worker_id}] completed {job.video_id} "
                    f"(completed {local_done} local) "
                    f"[overall] completed {cfg.completed.value}/{cfg.total}, "
                    f"failed {cfg.failed.value}, running {cfg.running.value - 1}",
                    flush=True,
                )
        except subprocess.CalledProcessError:
            with cfg.print_lock:
                cfg.failed.value += 1
                print(
                    f"[gpu {cfg.gpu_id} worker {cfg.worker_id}] failed {job.video_id} "
                    f"(completed {local_done} local) "
                    f"[overall] completed {cfg.completed.value}/{cfg.total}, "
                    f"failed {cfg.failed.value}, running {cfg.running.value - 1} "
                    f"(log: {log_path})",
                    flush=True,
                )
        finally:
            cfg.running.value -= 1


def run_parallel(
    step: str,
    script_path: Path,
    jobs: list[VideoJob],
    *,
    gpu_ids: list[int],
    procs_per_gpu: int,
    dataset_root: Path | None,
    resume: bool,
    force: bool,
    loud: bool,
    dry_run: bool,
    extra_args: list[str],
) -> int:
    if not jobs:
        print("No tasks.", flush=True)
        return 0

    if dry_run:
        print(f"Dry run: {len(jobs)} tasks, GPUs {gpu_ids}, {procs_per_gpu} proc/GPU", flush=True)
        for i, job in enumerate(jobs[:5]):
            cmd, work_cwd = build_run_sequence_cmd(
                script_path,
                job,
                step=step,
                gpu_id=gpu_ids[i % len(gpu_ids)],
                dataset_root=dataset_root,
                extra_args=extra_args,
            )
            print(f"  (cwd {work_cwd})", " ".join(cmd), flush=True)
        if len(jobs) > 5:
            print(f"  ... and {len(jobs) - 5} more", flush=True)
        return 0

    task_queue: mp.Queue = mp.Queue()
    for job in jobs:
        task_queue.put(job)
    for _ in range(len(gpu_ids) * procs_per_gpu):
        task_queue.put(None)

    ctx = mp.get_context("spawn")
    print_lock = ctx.Lock()
    completed = ctx.Value("i", 0)
    failed = ctx.Value("i", 0)
    running = ctx.Value("i", 0)
    workers: list[mp.Process] = []
    worker_id = 0
    for gpu_id in gpu_ids:
        for _ in range(procs_per_gpu):
            state = _WorkerState(
                worker_id=worker_id,
                gpu_id=gpu_id,
                script_path=script_path,
                dataset_root=dataset_root,
                step=step,
                resume=resume,
                force=force,
                loud=loud,
                extra_args=extra_args,
                print_lock=print_lock,
                completed=completed,
                failed=failed,
                running=running,
                total=len(jobs),
            )
            proc = ctx.Process(target=_worker_main, args=(task_queue, state))
            proc.start()
            workers.append(proc)
            worker_id += 1

    try:
        for proc in workers:
            proc.join()
    except KeyboardInterrupt:
        for proc in workers:
            proc.terminate()
        raise

    print(
        f"Done: completed {completed.value}/{len(jobs)}, failed {failed.value}",
        flush=True,
    )
    return 1 if failed.value else 0


def main_parallel_cli(step: str, script_path: Path, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Parallel {step} over dataset videos")
    add_common_parallel_args(parser, step=step)
    args, extra = parser.parse_known_args(argv)

    try:
        jobs = discover_videos(
            args.dataset,
            args.input,
            dataset_root=args.dataset_root,
            video_list=args.video_list,
            sample=args.sample,
            limit=args.limit,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.sample is not None:
        manifest = REPO_ROOT / "data" / "interim" / args.dataset / f"{step}_sample_manifest.json"
        write_video_manifest(jobs, manifest, seed=args.seed)

    extra_args = list(extra)
    if args.visualize:
        extra_args.append("--visualize")

    gpu_ids = parse_gpu_list(args.gpu_ids)
    return run_parallel(
        step,
        script_path,
        jobs,
        gpu_ids=gpu_ids,
        procs_per_gpu=args.procs_per_gpu,
        dataset_root=args.dataset_root,
        resume=args.resume,
        force=args.force,
        loud=args.loud,
        dry_run=args.dry_run,
        extra_args=extra_args,
    )
