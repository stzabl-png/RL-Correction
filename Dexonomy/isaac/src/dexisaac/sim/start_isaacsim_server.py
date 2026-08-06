#!/usr/bin/env python3
"""Start the persistent OCIR Isaac Sim process and JSON control server."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexisaac.sim.control_client import request_json
from dexisaac.sim.isaac_server import DEFAULT_OCIR_DATA_ROOT, DEFAULT_PROGRESS_LOG, run_persistent_server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["webrtc", "local"], default=os.environ.get("OCIR_ISAACSIM_MODE", "webrtc").lower())
    parser.add_argument("--data-root", type=Path, default=DEFAULT_OCIR_DATA_ROOT)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8765)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--livestream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--stream-ui",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In WebRTC mode, stream the full Isaac Sim UI. Use --no-stream-ui for viewport-only streaming.",
    )
    parser.add_argument("--progress-log", type=Path, default=Path(DEFAULT_PROGRESS_LOG))
    parser.add_argument(
        "--task-module",
        type=Path,
        action="append",
        default=[
            REPO_ROOT / "scripts/simulate_grasp_traj.py",
        ],
        help="Task module path. May be repeated. Each module must define register_sim_tasks(registry).",
    )
    parser.add_argument(
        "--hot-reload-tasks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reload task modules from source before each submitted job.",
    )
    parser.add_argument("--status", action="store_true", help="Print control-server status and exit.")
    parser.add_argument("--shutdown-server", action="store_true", help="Ask the persistent control server to shut down and exit.")
    args = parser.parse_args()
    if args.mode == "webrtc":
        if args.livestream is False:
            parser.error("--mode webrtc cannot be combined with --no-livestream")
        args.livestream = True
    elif args.mode == "local":
        if args.livestream is True:
            parser.error("--mode local cannot be combined with --livestream")
        args.livestream = False
    args.data_root = args.data_root.expanduser()
    return args


def main() -> int:
    args = parse_args()
    if args.status:
        try:
            print(json.dumps(request_json(args.control_host, args.control_port, "GET", "/status", timeout=1.0), indent=2), flush=True)
            return 0
        except Exception as exc:
            print(f"OCIR_SIM_SERVER no control server at {args.control_host}:{args.control_port}: {exc}", flush=True)
            return 1
    if args.shutdown_server:
        try:
            print(json.dumps(request_json(args.control_host, args.control_port, "POST", "/shutdown", {}, timeout=2.0), indent=2), flush=True)
            return 0
        except Exception as exc:
            print(f"OCIR_SIM_SERVER no control server at {args.control_host}:{args.control_port}: {exc}", flush=True)
            return 1
    return run_persistent_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
