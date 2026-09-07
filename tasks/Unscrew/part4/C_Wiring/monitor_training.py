#!/usr/bin/env python3
"""Checkpoint monitor for long Unscrew training runs.

The monitor has two deliberately separate layers:

1. Cheap TensorBoard/progress snapshots while training keeps running.
2. Deterministic one-environment ``probe_phases.py`` runs at explicit
   milestones, always against an immutable ``ep_*.pth`` checkpoint.

It never stops or restarts training.  A stale run, a failed probe, or a bad
metric is recorded as a warning for a human to judge.

Example (v48):

    python monitor_training.py \
      --run UnscrewHYB_v48:0:HYB \
      --run UnscrewOBJ_v48:1:OBJ \
      --milestones 2,4,8,12,16,20,24,28,30 \
      --out-dir logs/Unscrew_v48_monitor

Create ``<out-dir>/STOP`` to end the monitor cleanly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ROOT = Path(__file__).resolve().parents[4]
PROBE = ROOT / "tasks/Unscrew/part4/B_SmokeTest/probe_phases.py"
DEFAULT_PYTHON = Path("/home/feiyang/isaacsim/python.sh")
CKPT_RE = re.compile(r"^ep_(\d+)_step_(\d+)M_reward_.*\.pth$")
ROW_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+\|")
FORCE_RE = re.compile(r"\bL\s*([0-9]+(?:\.[0-9]+)?)N\b")
TERM_RE = re.compile(r"终止 @t=(\d+) row=(\d+) fail_code=(\d+)")
GMAX_RE = re.compile(r"G链最大=\[([^]]+)\]")

TAGS = (
    "sr_t0/gate1",
    "sr_t0/gate2",
    "sr_t0/gate3",
    "sr_t0/gate4",
    "sr_t0/align",
    "sr/align",
    "curr/ema_t0_g1",
    "curr/ema_t0_g2",
    "curr/ema_t0_g3",
    "curr/ema_cert",
    "prog/clock_frac",
    "n/ep_done_t0",
    "diag/cap_any",
    "diag/n_triad",
    "diag/screw_tau_mNm",
    "diag/screw_deg",
    "diag/screw_unlocked",
    "diag/released",
    "diag/carry_steps",
    "diag/fall_peak",
    "diag/escort_fail",
    "ep_rew/screw",
    "losses/critic_loss",
    "losses/entropy",
)


@dataclass(frozen=True)
class Run:
    name: str
    gpu: int
    variant: str

    @property
    def directory(self) -> Path:
        return ROOT / "logs" / self.name


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def parse_run(value: str) -> Run:
    try:
        name, gpu, variant = value.rsplit(":", 2)
        run = Run(name=name, gpu=int(gpu), variant=variant.upper())
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "--run must be NAME:GPU:VARIANT, e.g. UnscrewHYB_v48:0:HYB"
        ) from exc
    if not name or run.gpu < 0 or run.variant not in {"HYB", "OBJ"}:
        raise argparse.ArgumentTypeError(f"invalid run specification: {value!r}")
    return run


def parse_milestones(value: str) -> list[int]:
    try:
        values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("milestones must be comma-separated integers") from exc
    if not values or values[0] <= 0:
        raise argparse.ArgumentTypeError("milestones must be positive")
    return values


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def read_progress(run: Run) -> tuple[int, float]:
    path = run.directory / "progress_steps.txt"
    try:
        text = path.read_text().strip()
        return int("".join(ch for ch in text if ch.isdigit()) or "0"), path.stat().st_mtime
    except (FileNotFoundError, OSError, ValueError):
        return 0, 0.0


def latest_event(run: Run) -> Path | None:
    files = list((run.directory / "stage1_tb").glob("events.out.tfevents.*"))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def scalar_snapshot(run: Run, rolling: int) -> dict[str, Any]:
    event = latest_event(run)
    if event is None:
        return {"event_error": "no TensorBoard event file"}
    try:
        acc = EventAccumulator(str(event), size_guidance={"scalars": 0})
        acc.Reload()
        available = set(acc.Tags().get("scalars", ()))
        result: dict[str, Any] = {"event_file": str(event.relative_to(ROOT))}
        for tag in TAGS:
            if tag not in available:
                continue
            values = acc.Scalars(tag)
            if not values:
                continue
            tail = values[-rolling:]
            finite = [float(item.value) for item in tail if math.isfinite(float(item.value))]
            result[tag] = {
                "step": int(values[-1].step),
                "last": float(values[-1].value),
                "mean": sum(finite) / len(finite) if finite else None,
                "n": len(finite),
            }
        return result
    except Exception as exc:  # a partly flushed event file is retryable
        return {"event_file": str(event), "event_error": f"{type(exc).__name__}: {exc}"}


def immutable_checkpoint(run: Run, min_age_s: float = 15.0) -> Path | None:
    candidates: list[tuple[int, float, Path]] = []
    now = time.time()
    for path in (run.directory / "stage1_nn").glob("ep_*.pth"):
        match = CKPT_RE.match(path.name)
        if match and now - path.stat().st_mtime >= min_age_s:
            candidates.append((int(match.group(1)), path.stat().st_mtime, path))
    return max(candidates, default=(0, 0.0, None))[2]


def probe_summary(text: str, returncode: int, timed_out: bool) -> dict[str, Any]:
    rows: list[int] = []
    forces: list[float] = []
    terminal: dict[str, int] | None = None
    gmax: list[int] | None = None
    for line in text.splitlines():
        row_match = ROW_RE.match(line)
        if row_match:
            rows.append(int(row_match.group(2)))
        force_match = FORCE_RE.search(line)
        if force_match:
            forces.append(float(force_match.group(1)))
        term_match = TERM_RE.search(line)
        if term_match:
            terminal = {
                "t": int(term_match.group(1)),
                "reported_row": int(term_match.group(2)),
                "fail_code": int(term_match.group(3)),
            }
        chain_match = GMAX_RE.search(line)
        if chain_match:
            gmax = [int(item.strip()) for item in chain_match.group(1).split(",")]
    return {
        "returncode": returncode,
        "timed_out": timed_out,
        "terminal": terminal,
        "gmax": gmax,
        "max_row_seen": max(rows, default=None),
        "max_left_force_n": max(forces, default=None),
    }


def run_probe(
    run: Run,
    checkpoint: Path,
    milestone: int,
    out_dir: Path,
    python: Path,
    timeout_s: int,
) -> dict[str, Any]:
    probe_dir = out_dir / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    log_path = probe_dir / f"{run.variant}_{milestone:03d}M.log"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(run.gpu),
            "TMPDIR": "/home/feiyang/tmp",
            # 监控可跨 clip 复用: 启动时带 MONITOR_UNSCREW_CLIP=<clip> 覆盖
            "UNSCREW_CLIP": os.environ.get("MONITOR_UNSCREW_CLIP", "32"),
            "SHARPA_WANDB": "0",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "PYTHONPATH": str(ROOT),
            "RL_ISAAC_NO_GUARD": "1",
            "POUR_SQUEEZE_FF": "1",
            "POUR_BONUS_NOW": "1",
            "POUR_BONUS_DIST": "1",
            "POUR_VARIANT": run.variant,
            "POUR_UNLOCK": "1,2,3",
            "POUR_PAD_FRIC": "6.0",
        }
    )
    cmd = [
        str(python),
        "-u",
        str(PROBE),
        "--checkpoint",
        str(checkpoint),
        "--steps",
        "900",
        "--every",
        "20",
        "--headless",
    ]
    started = now_iso()
    with log_path.open("w") as log:
        log.write(
            f"[monitor] run={run.name} milestone={milestone}M "
            f"checkpoint={checkpoint.relative_to(ROOT)} started={started}\n"
        )
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
            log.write(f"\n[monitor] probe timeout after {timeout_s}s\n")
    text = log_path.read_text(errors="replace")
    result = probe_summary(text, returncode, timed_out)
    result.update(
        {
            "run": run.name,
            "variant": run.variant,
            "gpu": run.gpu,
            "milestone_m": milestone,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "log": str(log_path.relative_to(ROOT)),
            "started": started,
            "finished": now_iso(),
        }
    )
    atomic_json(probe_dir / f"{run.variant}_{milestone:03d}M.json", result)
    return result


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def log(message: str, logfile: Path) -> None:
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    with logfile.open("a") as handle:
        handle.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--milestones", type=parse_milestones, default=parse_milestones("2,4,8,12,16,20,24,28,30"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--sample-every-steps", type=int, default=250_000)
    parser.add_argument("--rolling", type=int, default=20)
    parser.add_argument("--probe-timeout", type=int, default=1800)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / "monitor.log"
    snapshots = out_dir / "snapshots.jsonl"
    state_path = out_dir / "state.json"
    stop_path = out_dir / "STOP"
    pid_path = out_dir / "monitor.pid"

    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
            os.kill(old_pid, 0)
            print(f"monitor already running as pid {old_pid}", file=sys.stderr)
            return 2
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    pid_path.write_text(f"{os.getpid()}\n")

    state: dict[str, Any] = {"version": 1, "runs": {}}
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text())
            if loaded.get("version") == 1:
                state = loaded
        except (OSError, ValueError):
            log("warning: ignored unreadable state.json", logfile)

    for run in args.run:
        state["runs"].setdefault(
            run.name,
            {"last_sample_bucket": -1, "probes_done": [], "probe_attempts": {}},
        )
    atomic_json(state_path, state)
    log(
        "started; runs=" + ", ".join(f"{r.name}/gpu{r.gpu}/{r.variant}" for r in args.run)
        + f"; milestones={args.milestones}; auto_stop=disabled",
        logfile,
    )

    try:
        while not stop_path.exists():
            all_complete = True
            for run in args.run:
                progress, progress_mtime = read_progress(run)
                run_state = state["runs"][run.name]
                bucket = progress // max(args.sample_every_steps, 1)
                stale_s = time.time() - progress_mtime if progress_mtime else None
                due = [m for m in args.milestones if progress >= m * 1_000_000]
                if any(m not in run_state["probes_done"] for m in args.milestones):
                    all_complete = False

                if bucket > run_state["last_sample_bucket"] or args.once:
                    snapshot = {
                        "time": now_iso(),
                        "run": run.name,
                        "variant": run.variant,
                        "gpu": run.gpu,
                        "progress_steps": progress,
                        "progress_m": progress / 1e6,
                        "progress_stale_s": stale_s,
                        "metrics": scalar_snapshot(run, args.rolling),
                    }
                    append_jsonl(snapshots, snapshot)
                    run_state["last_sample_bucket"] = bucket
                    log(f"snapshot {run.variant} @{progress / 1e6:.3f}M", logfile)

                if stale_s is not None and stale_s > 600:
                    log(f"WARNING {run.name}: progress stale for {stale_s / 60:.1f} min", logfile)

                if not args.no_probe:
                    for milestone in due:
                        if milestone in run_state["probes_done"]:
                            continue
                        attempts = int(run_state["probe_attempts"].get(str(milestone), 0))
                        if attempts >= 2:
                            log(
                                f"WARNING {run.variant} {milestone}M probe failed twice; human review required",
                                logfile,
                            )
                            run_state["probes_done"].append(milestone)
                            break
                        checkpoint = immutable_checkpoint(run)
                        if checkpoint is None:
                            log(f"{run.variant} {milestone}M waiting for immutable checkpoint", logfile)
                            break
                        run_state["probe_attempts"][str(milestone)] = attempts + 1
                        atomic_json(state_path, state)
                        log(
                            f"probe start {run.variant} {milestone}M using {checkpoint.name}",
                            logfile,
                        )
                        result = run_probe(
                            run,
                            checkpoint,
                            milestone,
                            out_dir,
                            args.python,
                            args.probe_timeout,
                        )
                        if result["returncode"] == 0 and result["gmax"] is not None:
                            run_state["probes_done"].append(milestone)
                            log(
                                f"probe done {run.variant} {milestone}M: "
                                f"G={result['gmax']} fail={result['terminal']} "
                                f"max_row={result['max_row_seen']} max_L={result['max_left_force_n']}N",
                                logfile,
                            )
                        else:
                            log(
                                f"WARNING probe failed {run.variant} {milestone}M: "
                                f"rc={result['returncode']} timeout={result['timed_out']}",
                                logfile,
                            )
                        atomic_json(state_path, state)
                        break  # at most one costly probe per run per monitor pass

            atomic_json(state_path, state)
            if args.once:
                break
            if all_complete:
                log("all configured milestone probes complete; monitor exiting", logfile)
                break
            time.sleep(max(args.interval, 10))
    finally:
        try:
            if int(pid_path.read_text().strip()) == os.getpid():
                pid_path.unlink()
        except (FileNotFoundError, OSError, ValueError):
            pass
    log("stopped", logfile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
