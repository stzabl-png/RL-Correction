#!/usr/bin/env python3
"""Small JSON client for the OCIR Isaac Sim control server."""

from __future__ import annotations

import json
import time
from pathlib import Path
from urllib import request as urllib_request


def request_json(host: str, port: int, method: str, path: str, payload: dict | None = None, timeout: float = 2.0) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib_request.Request(f"http://{host}:{port}{path}", data=data, headers=headers, method=method)
    with urllib_request.urlopen(req, timeout=None if timeout <= 0 else timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def server_is_running(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        request_json(host, port, "GET", "/status", timeout=timeout)
        return True
    except Exception:
        return False


def submit_job(
    *,
    host: str,
    port: int,
    task: str,
    params: dict,
    request_timeout: float = 0.0,
    poll_interval: float = 2.0,
) -> tuple[int, dict | None]:
    response = request_json(
        host,
        port,
        "POST",
        "/run",
        {"task": task, "params": params, "wait": False},
        timeout=5.0,
    )
    if not response.get("ok", False):
        print(f"OCIR_SIM submit failed: {response.get('error', response)}", flush=True)
        return 1, response

    job_id = response["job_id"]
    print(f"OCIR_SIM submitted {job_id} task={task} to {host}:{port}", flush=True)
    started = time.monotonic()
    last_line = None
    while True:
        if request_timeout > 0 and time.monotonic() - started > float(request_timeout):
            print(f"OCIR_SIM timed out waiting for {job_id} after {request_timeout}s", flush=True)
            return 2, None
        status = request_json(host, port, "GET", f"/jobs/{job_id}", timeout=5.0)
        progress = status.get("progress") or {}
        if progress:
            line = progress.get("message") or json.dumps(progress, sort_keys=True)
            if line != last_line:
                print(f"OCIR_SIM {job_id}: {line}", flush=True)
                last_line = line
        if status.get("done", False):
            result = status.get("response") or {}
            if not result.get("ok", False):
                print(f"OCIR_SIM {job_id} failed: {result.get('error_type', '')} {result.get('error', result)}", flush=True)
                return 1, result
            summary = result.get("summary", {})
            print(f"OCIR_SIM {job_id} completed task={task}", flush=True)
            if summary.get("out_dir"):
                print(f"OCIR_SIM wrote {Path(summary['out_dir']) / 'summary.json'}", flush=True)
            return 0, result
        time.sleep(float(poll_interval))
