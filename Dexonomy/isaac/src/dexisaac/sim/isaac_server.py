#!/usr/bin/env python3
"""Reusable persistent Isaac Sim process and JSON control server."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import importlib.util
import json
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Iterable

from dexisaac.sim.control_client import request_json


ISAAC_STREAMING_EXPERIENCE = "isaacsim.exp.full.streaming.kit"
DEFAULT_OCIR_DATA_ROOT = Path(os.environ.get("OCIR_DATA_ROOT", "/data/users/hangkes2/OCIR")).expanduser()
DEFAULT_PROGRESS_LOG = os.environ.get("OCIR_ISAACSIM_PROGRESS_LOG", "/tmp/ocir_isaacsim_progress.log")


TaskRunner = Callable[[object, argparse.Namespace, Callable[..., None]], dict]
TaskNormalize = Callable[[dict], argparse.Namespace]


@dataclass
class TaskSpec:
    name: str
    run: TaskRunner
    normalize: TaskNormalize
    description: str = ""


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, TaskSpec] = {}

    def register(self, name: str, run: TaskRunner, normalize: TaskNormalize, description: str = "") -> None:
        if name in self._tasks:
            raise ValueError(f"duplicate simulation task registered: {name}")
        self._tasks[name] = TaskSpec(name=name, run=run, normalize=normalize, description=description)

    def get(self, name: str) -> TaskSpec:
        try:
            return self._tasks[name]
        except KeyError as exc:
            raise KeyError(f"unknown simulation task {name!r}; registered tasks: {sorted(self._tasks)}") from exc

    def status(self) -> list[dict]:
        return [{"name": task.name, "description": task.description} for task in self._tasks.values()]


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] OCIR_SIM_SERVER {message}"
    print(line, flush=True)
    progress_log = globals().get("_OCIR_PROGRESS_LOG")
    if progress_log:
        try:
            with Path(progress_log).open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def isaacsim_experience_path(name: str) -> str:
    conda_site_isaacsim = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/isaacsim"
    candidates = [
        Path("/isaac-sim/apps") / name,
        conda_site_isaacsim / "apps" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return name


def load_task_modules(paths: Iterable[Path], tag: str | None = None) -> TaskRegistry:
    registry = TaskRegistry()
    importlib.invalidate_caches()
    load_tag = tag or str(time.time_ns())
    for idx, path in enumerate(paths):
        module_path = path.expanduser().resolve()
        if not module_path.exists():
            raise FileNotFoundError(f"task module does not exist: {module_path}")
        module_name = f"ocir_sim_task_{load_tag}_{idx}_{module_path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import task module: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        source = module_path.read_text(encoding="utf-8")
        code = compile(source, str(module_path), "exec")
        exec(code, module.__dict__)
        register = getattr(module, "register_sim_tasks", None)
        if register is None:
            raise RuntimeError(f"{module_path} must define register_sim_tasks(registry)")
        register(registry)
        log(f"loaded task module {module_path}")
    return registry


def launch_simulation_app(args: argparse.Namespace):
    log(
        f"launching Isaac SimulationApp mode={args.mode} livestream={bool(args.livestream)} "
        f"headless={args.headless} stream_ui={bool(args.stream_ui)} size={args.width}x{args.height}"
    )
    from isaacsim import SimulationApp

    extra_args = []
    if args.livestream:
        extra_args.extend(
            [
                "--/app/livestream/nvcf/quitOnSessionEnded=false",
                "--/app/livestream/nvcf/sessionResumeTimeoutSeconds=3600",
                "--/exts/omni.kit.widget.cache_indicator/check_updates=false",
            ]
        )
    launch_config = {
        "headless": args.headless,
        "width": args.width,
        "height": args.height,
        "disable_viewport_updates": False,
        "extra_args": extra_args,
    }
    if args.livestream:
        launch_config["hide_ui"] = not bool(args.stream_ui)
    experience = isaacsim_experience_path(ISAAC_STREAMING_EXPERIENCE) if args.livestream else ""
    log(f"SimulationApp experience={experience or '<default>'}")
    app = SimulationApp(launch_config, experience=experience)
    log("Isaac SimulationApp launched")
    return app


def wait_for_livestream_ready(app, args: argparse.Namespace, timeout: float = 90.0) -> bool:
    if not args.livestream:
        return True
    deadline = time.monotonic() + float(timeout)
    stop_event = threading.Event()
    result = {"ready": False, "payload": None, "last_error": None}

    def probe_ready() -> None:
        while not stop_event.is_set() and time.monotonic() < deadline:
            try:
                result["payload"] = request_json("127.0.0.1", 8011, "GET", "/v1/streaming/ready", timeout=1.0)
                result["ready"] = True
                stop_event.set()
                return
            except Exception as exc:
                result["last_error"] = exc
                stop_event.wait(0.25)

    probe_thread = threading.Thread(target=probe_ready, name="ocir-isaac-webrtc-ready", daemon=True)
    probe_thread.start()
    last_report = 0.0
    log("waiting for Isaac WebRTC readiness endpoint on 127.0.0.1:8011")
    while time.monotonic() < deadline and not result["ready"]:
        app.update()
        now = time.monotonic()
        if now - last_report >= 5.0:
            last_error = result["last_error"]
            if last_error is None:
                log("waiting for Isaac WebRTC ready endpoint")
            else:
                log(f"waiting for Isaac WebRTC ready endpoint: {type(last_error).__name__}: {last_error}")
            last_report = now
        time.sleep(1.0 / 30.0)
    stop_event.set()
    probe_thread.join(timeout=0.5)
    if result["ready"]:
        payload = result["payload"] or {}
        log(f"Isaac WebRTC ready: {payload.get('statusMessage', payload)}")
        for _ in range(30):
            app.update()
            time.sleep(1.0 / 60.0)
        return True
    log(f"warning: Isaac WebRTC readiness endpoint did not respond within {timeout}s; last error: {result['last_error']}")
    return False


class IsaacControlState:
    def __init__(self, args: argparse.Namespace, registry: TaskRegistry):
        self.args = args
        self.registry = registry
        self.task_module_paths = [Path(path).expanduser().resolve() for path in args.task_module]
        self.hot_reload_tasks = bool(getattr(args, "hot_reload_tasks", True))
        self.task_reload_count = 1
        self.last_task_reload_at = time.time()
        self.last_task_reload_error: dict | None = None
        self.jobs: queue.Queue[dict] = queue.Queue()
        self.job_records: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.busy = False
        self.current_job_id: str | None = None
        self.completed_jobs = 0
        self.last_summary: dict | None = None
        self.last_error: dict | None = None
        self.shutdown_requested = False
        self.started_at = time.time()

    def reload_tasks(self, reason: str) -> TaskRegistry:
        registry = load_task_modules(self.task_module_paths, tag=f"reload_{time.time_ns()}")
        with self.lock:
            self.registry = registry
            self.task_reload_count += 1
            self.last_task_reload_at = time.time()
            self.last_task_reload_error = None
        log(f"{reason}: reloaded task modules")
        return registry

    def get_registry(self) -> TaskRegistry:
        with self.lock:
            return self.registry

    def register_job(self, item: dict) -> None:
        with self.lock:
            self.job_records[item["id"]] = {
                "ok": True,
                "job_id": item["id"],
                "task": item["task"],
                "queued": True,
                "running": False,
                "done": False,
                "submitted_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "progress": {"message": "queued", "phase": "queued", "time": time.time()},
                "response": None,
            }

    def set_job_started(self, job_id: str) -> None:
        with self.lock:
            record = self.job_records.get(job_id)
            if record is not None:
                record.update({"queued": False, "running": True, "started_at": time.time()})

    def set_progress(self, job_id: str, message: str, **extra) -> None:
        with self.lock:
            record = self.job_records.get(job_id)
            if record is not None:
                record["progress"] = {"message": message, "time": time.time(), **extra}

    def set_job_done(self, job_id: str, response: dict) -> None:
        with self.lock:
            record = self.job_records.get(job_id)
            if record is not None:
                record.update({"queued": False, "running": False, "done": True, "finished_at": time.time(), "response": response})

    def job_status(self, job_id: str) -> dict:
        with self.lock:
            record = self.job_records.get(job_id)
            if record is None:
                return {"ok": False, "error": f"unknown job_id {job_id}"}
            return dict(record)

    def status(self) -> dict:
        with self.lock:
            current_progress = None
            if self.current_job_id is not None:
                record = self.job_records.get(self.current_job_id)
                if record is not None:
                    current_progress = record.get("progress")
            return {
                "ok": True,
                "busy": self.busy,
                "current_job_id": self.current_job_id,
                "current_progress": current_progress,
                "completed_jobs": self.completed_jobs,
                "queued_jobs": self.jobs.qsize(),
                "known_jobs": len(self.job_records),
                "last_summary": self.last_summary,
                "last_error": self.last_error,
                "mode": self.args.mode,
                "livestream": bool(self.args.livestream),
                "stream_ui": bool(self.args.stream_ui),
                "data_root": str(self.args.data_root),
                "control_host": self.args.control_host,
                "control_port": self.args.control_port,
                "hot_reload_tasks": self.hot_reload_tasks,
                "task_reload_count": self.task_reload_count,
                "last_task_reload_at": self.last_task_reload_at,
                "last_task_reload_error": self.last_task_reload_error,
                "task_modules": [str(path) for path in self.task_module_paths],
                "registered_tasks": self.registry.status(),
                "uptime_seconds": time.time() - self.started_at,
            }


def make_control_handler(state: IsaacControlState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def _send(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/status":
                self._send(200, state.status())
                return
            if self.path.startswith("/jobs/"):
                job_id = self.path.removeprefix("/jobs/")
                payload = state.job_status(job_id)
                self._send(200 if payload.get("ok", False) else 404, payload)
                return
            self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self) -> None:
            if self.path == "/shutdown":
                state.shutdown_requested = True
                self._send(200, {"ok": True, "shutdown_requested": True})
                return
            if self.path != "/run":
                self._send(404, {"ok": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                task_name = str(payload["task"])
                params = dict(payload.get("params") or {})
                wait_for_result = bool(payload.get("wait", True))
            except Exception as exc:
                self._send(400, {"ok": False, "error": str(exc)})
                return
            item = {"id": f"job-{time.time_ns()}", "task": task_name, "params": params, "args": None, "done": threading.Event(), "response": None}
            state.register_job(item)
            state.jobs.put(item)
            if not wait_for_result:
                self._send(202, {"ok": True, "job_id": item["id"], "status_path": f"/jobs/{item['id']}"})
                return
            item["done"].wait()
            self._send(200, item["response"])

    return Handler


def start_control_server(state: IsaacControlState) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((state.args.control_host, state.args.control_port), make_control_handler(state))
    thread = threading.Thread(target=server.serve_forever, name="ocir-isaac-control", daemon=True)
    thread.start()
    return server


def run_persistent_server(args: argparse.Namespace) -> int:
    globals()["_OCIR_PROGRESS_LOG"] = str(args.progress_log)
    registry = load_task_modules(args.task_module)
    log("starting persistent Isaac server")
    state = IsaacControlState(args, registry)
    server = start_control_server(state)
    app = None
    log(f"control server listening on {args.control_host}:{args.control_port}")
    try:
        app = launch_simulation_app(args)
        wait_for_livestream_ready(app, args)
        while not state.shutdown_requested:
            try:
                item = state.jobs.get(timeout=0.02)
            except queue.Empty:
                app.update()
                time.sleep(1.0 / 60.0)
                continue
            with state.lock:
                state.busy = True
                state.current_job_id = item["id"]
                state.last_error = None
            state.set_job_started(item["id"])
            state.set_progress(item["id"], "job started", phase="start")
            progress = lambda message, _job_id=item["id"], **extra: state.set_progress(_job_id, message, **extra)
            active_phase = "start"
            try:
                if state.hot_reload_tasks:
                    active_phase = "reload"
                    state.set_progress(item["id"], "reloading task modules", phase="reload")
                    registry = state.reload_tasks(item["id"])
                else:
                    registry = state.get_registry()
                active_phase = "resolve"
                task = registry.get(item["task"])
                active_phase = "normalize"
                task_args = task.normalize(item["params"])
                item["args"] = task_args
                active_phase = "run"
                log(f"{item['id']}: running task={item['task']}")
                summary = task.run(app, task_args, progress)
                item["response"] = {"ok": True, "job_id": item["id"], "task": item["task"], "summary": summary}
                state.set_job_done(item["id"], item["response"])
                with state.lock:
                    state.last_summary = summary
                    state.completed_jobs += 1
            except BaseException as exc:
                error_report = {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()}
                if state.hot_reload_tasks and active_phase == "reload":
                    with state.lock:
                        state.last_task_reload_error = error_report
                out_dir = getattr(item.get("args"), "out_dir", None)
                if out_dir is None:
                    raw_out_dir = item.get("params", {}).get("out_dir")
                    if raw_out_dir:
                        out_dir = raw_out_dir
                if out_dir is not None:
                    try:
                        Path(out_dir).mkdir(parents=True, exist_ok=True)
                        (Path(out_dir) / "error_report.json").write_text(json.dumps(error_report, indent=2) + "\n", encoding="utf-8")
                    except Exception:
                        pass
                item["response"] = {"ok": False, "job_id": item["id"], "task": item["task"], **error_report}
                state.set_progress(item["id"], f"job failed: {type(exc).__name__}: {exc}", phase="error")
                state.set_job_done(item["id"], item["response"])
                with state.lock:
                    state.last_error = error_report
                print(error_report["traceback"], flush=True)
            finally:
                with state.lock:
                    state.busy = False
                    state.current_job_id = None
                item["done"].set()
    finally:
        log("shutting down control server and Isaac SimulationApp")
        server.shutdown()
        server.server_close()
        if app is not None:
            app.close()
    return 0
