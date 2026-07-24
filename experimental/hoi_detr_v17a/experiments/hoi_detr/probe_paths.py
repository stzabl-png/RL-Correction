"""Filesystem helpers kept local so the probe does not register a pipeline step."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.hoi_detr.adapter import write_json_atomic


REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM_ROOT = REPO_ROOT / "data" / "interim"


def probe_step_dir(dataset: str, video_id: str, step: str) -> Path:
    return INTERIM_ROOT / dataset / video_id / step


def completion_marker(step_dir: Path, step: str) -> Path:
    return step_dir / f"{step}_complete.json"


def is_probe_complete(
    step_dir: Path,
    step: str,
    *,
    run_fingerprint: str | None = None,
    required_outputs: tuple[Path, ...] = (),
) -> bool:
    marker = completion_marker(step_dir, step)
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "complete" or payload.get("step") != step:
        return False
    if run_fingerprint is not None and payload.get("run_fingerprint") != run_fingerprint:
        return False
    return all(path.is_file() for path in required_outputs)


def write_probe_completion(
    step_dir: Path,
    step: str,
    *,
    dataset: str,
    video_id: str,
    extra: dict[str, Any],
) -> Path:
    marker = completion_marker(step_dir, step)
    write_json_atomic(
        marker,
        {
            "status": "complete",
            "step": step,
            "dataset": dataset,
            "video_id": video_id,
            **extra,
        },
    )
    return marker
