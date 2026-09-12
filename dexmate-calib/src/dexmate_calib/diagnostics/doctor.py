from __future__ import annotations

import json
import socket
import subprocess
import time
from dataclasses import asdict

import numpy as np

from dexmate_calib.streaming.zed_stream import ZedStreamClient


def _tcp_probe(host: str, port: int, timeout_s: float) -> dict:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            elapsed = time.perf_counter() - started
            return {"ok": True, "latency_ms": elapsed * 1000.0}
    except OSError as exc:
        return {"ok": False, "error": str(exc)}


def _ssh_probe(target: str, timeout_s: int = 5, proxy_command: str | None = None) -> dict:
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout_s}",
    ]
    if proxy_command is not None:
        command.extend(["-o", f"ProxyCommand={proxy_command}"])
    command.extend([target, "hostname"])
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_s + 2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": result.returncode == 0,
        "hostname": result.stdout.strip() or None,
        "error": result.stderr.strip() or None,
    }


def doctor_network() -> dict:
    return {
        "thor_ssh": _ssh_probe("dexmate@192.168.50.20"),
        "nano_ssh_proxy": _ssh_probe(
            "dexmate-nano@192.168.50.22",
            proxy_command="ssh -W %h:%p dexmate@192.168.50.20",
        ),
        "nano_ssh_direct": _ssh_probe("dexmate-nano@192.168.50.22"),
        "nano_stream_port": _tcp_probe("192.168.50.22", 30000, 3.0),
    }


def doctor_stream(
    host: str,
    port: int,
    *,
    frames: int = 20,
    expected_serial: int | None = 59595115,
    expected_width: int = 1920,
    expected_height: int = 1200,
) -> dict:
    if frames < 2:
        raise ValueError("frames must be >= 2")
    client = ZedStreamClient(
        host,
        port,
        expected_serial=expected_serial,
        expected_width=expected_width,
        expected_height=expected_height,
    )
    received = []
    decode_times = []
    wall_start = time.perf_counter()
    with client:
        for _ in range(frames):
            frame = client.receive()
            decode_start = time.perf_counter()
            image = frame.decode_left_bgr()
            decode_times.append((time.perf_counter() - decode_start) * 1000.0)
            received.append((frame, image.shape))
    elapsed = time.perf_counter() - wall_start

    timestamps = np.asarray([item[0].source_timestamp_ns for item in received], dtype=np.int64)
    deltas = np.diff(timestamps)
    first = received[0][0]
    return {
        "ok": True,
        "host": host,
        "port": port,
        "protocol": first.protocol,
        "camera_serial": first.serial_number,
        "resolution": {"width": first.left_width, "height": first.left_height},
        "decoded_shape": list(received[0][1]),
        "channel_mask": first.channel_mask,
        "segments": [asdict(segment) for segment in first.segments],
        "frames": frames,
        "receive_fps": frames / elapsed,
        "source_fps_median": (
            float(1e9 / np.median(deltas[deltas > 0])) if np.any(deltas > 0) else None
        ),
        "non_monotonic_timestamps": int(np.sum(deltas <= 0)),
        "jpeg_decode_ms_median": float(np.median(decode_times)),
    }


def print_report(report: dict) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))
