"""Shared CLI/task-dispatch plumbing for OCIR Isaac Sim task-module scripts.

Both ``replay_dexycb.py`` and ``setup_dexycb_retarget_scene.py`` are dual-mode
CLI entry points: they can either launch a local, one-shot ``SimulationApp``
(``--mode local``) or submit a job to an already-running, persistent Isaac
control server (``--mode webrtc``, the default). This module holds the
mode-dispatch and job-submission logic shared by both, parameterized by each
script's task name, log-line prefix, and server-restart hint text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Callable

from ocir.sim.control_client import request_json, server_is_running, submit_job
from ocir.sim.isaac_server import launch_simulation_app


SERVER_ONLY_FIELDS = {
    "mode",
    "reuse_instance",
    "livestream",
    "hold_open",
    "hold_open_seconds",
    "control_host",
    "control_port",
    "request_timeout",
    "status",
    "shutdown_server",
}


def apply_mode_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.mode == "webrtc":
        if args.livestream is False:
            parser.error("--mode webrtc cannot be combined with --no-livestream")
        if args.reuse_instance is False:
            parser.error("--mode webrtc cannot be combined with --no-reuse-instance")
        args.livestream = True
        args.reuse_instance = True
        return
    if args.mode == "local":
        if args.livestream is True:
            parser.error("--mode local cannot be combined with --livestream")
        if args.reuse_instance is True:
            parser.error("--mode local cannot be combined with --reuse-instance")
        args.livestream = False
        args.reuse_instance = False
        return
    parser.error(f"unsupported --mode {args.mode!r}; use 'webrtc' or 'local'")


def marshal_params(args: argparse.Namespace) -> dict:
    return {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
        if key not in SERVER_ONLY_FIELDS
    }


def submit_job_to_server(args: argparse.Namespace, *, task: str, log_prefix: str, restart_hint: str) -> int:
    status = request_json(args.control_host, args.control_port, "GET", "/status", timeout=1.0)
    registered = {entry.get("name") for entry in status.get("registered_tasks", [])}
    if task not in registered:
        print(
            f"{log_prefix} control server is running, but it did not register task {task!r}.\n{restart_hint}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    code, _ = submit_job(
        host=args.control_host,
        port=args.control_port,
        task=task,
        params=marshal_params(args),
        request_timeout=float(args.request_timeout),
    )
    return code


def run_sim_cli_main(
    args: argparse.Namespace,
    *,
    log: Callable[[str], None],
    run_batch: Callable[..., dict],
    task: str,
    log_prefix: str,
    restart_hint: str,
) -> int:
    """Shared body for each script's ``main()``: status/shutdown queries,
    local one-shot SimulationApp launch, or job submission to a running
    persistent server."""

    if args.status:
        try:
            print(json.dumps(request_json(args.control_host, args.control_port, "GET", "/status", timeout=1.0), indent=2), flush=True)
            return 0
        except Exception as exc:
            print(f"{log_prefix} no control server at {args.control_host}:{args.control_port}: {exc}", flush=True)
            return 1
    if args.shutdown_server:
        try:
            print(json.dumps(request_json(args.control_host, args.control_port, "POST", "/shutdown", {}, timeout=2.0), indent=2), flush=True)
            return 0
        except Exception as exc:
            print(f"{log_prefix} no control server at {args.control_host}:{args.control_port}: {exc}", flush=True)
            return 1
    log(f"parsed args mode={args.mode} manifest={args.manifest} out_dir={args.out_dir}")
    if args.mode == "local":
        app = launch_simulation_app(args)
        try:
            run_batch(app, args)
            if args.hold_open:
                if args.hold_open_seconds > 0:
                    end = time.monotonic() + float(args.hold_open_seconds)
                    while time.monotonic() < end and app.is_running():
                        app.update()
                        time.sleep(1.0 / 60.0)
                else:
                    while app.is_running():
                        app.update()
                        time.sleep(1.0 / 60.0)
        finally:
            if not args.hold_open:
                app.close()
        return 0
    if server_is_running(args.control_host, args.control_port):
        return submit_job_to_server(args, task=task, log_prefix=log_prefix, restart_hint=restart_hint)
    print(
        f"{log_prefix} no OCIR Isaac control server is running at "
        f"{args.control_host}:{args.control_port}.\nStart the persistent Isaac Sim server first.",
        file=sys.stderr,
        flush=True,
    )
    return 2
