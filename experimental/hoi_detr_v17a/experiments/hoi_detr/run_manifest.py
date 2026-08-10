"""Minimal provenance record for one v17A segmentation run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    record: dict[str, Any] = {"path": str(resolved), "exists": resolved.is_file()}
    if resolved.is_file():
        stat = resolved.stat()
        record.update(
            size_bytes=stat.st_size,
            modified_time_ns=stat.st_mtime_ns,
            sha256=_sha256(resolved),
        )
    return record


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().resolve())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-c", f"safe.directory={repo_root}", "-C", str(repo_root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    commit = run("rev-parse", "HEAD")
    if commit.returncode != 0:
        return {
            "available": False,
            "root": str(repo_root),
            "reason": commit.stderr.strip(),
        }
    branch = run("symbolic-ref", "--quiet", "--short", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    diff = run("diff", "--binary", "HEAD")
    fingerprint = hashlib.sha256(
        status.stdout.encode() + b"\0" + diff.stdout.encode()
    ).hexdigest()
    return {
        "available": True,
        "root": str(repo_root),
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "dirty": bool(status.stdout),
        "working_tree_fingerprint_sha256": fingerprint,
        "changed_paths": [line[3:] for line in status.stdout.splitlines()],
    }


def _repo_root() -> Path:
    # The linked worktree can be broken, so source hashing below remains the
    # fallback identity even when Git metadata is unavailable.
    return Path(__file__).resolve().parents[2]


def start_manifest(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    command: Sequence[str],
) -> Path:
    entrypoint = Path(__file__).with_name("run_instance_video_segmentation.py")
    payload = {
        "schema_version": "v17a_run_manifest_v1",
        "run_id": str(uuid.uuid4()),
        "stage": "v17a_instance_video_segmentation",
        "status": "running",
        "started_at_utc": _now(),
        "finished_at_utc": None,
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "code": {
            "git": _git_state(_repo_root()),
            "entrypoint": _artifact(entrypoint),
        },
        "command": list(command),
        "parameters": {
            key: _jsonable(value)
            for key, value in vars(args).items()
            if key not in {"video", "detections", "checkpoint", "output_dir"}
        },
        "inputs": {
            "video": _artifact(args.video),
            "hoi_detr_detections": _artifact(args.detections),
        },
        "models": {"sam2_checkpoint": _artifact(args.checkpoint)},
        "expected_outputs": [
            "summary.json",
            "video_mask_sequence/video_mask_sequence.json",
            "video_registry/final_registry.json",
        ],
        "outputs": {},
        "validation": {},
        "failure": None,
    }
    path = output_dir / "run_manifest.running.json"
    _write_atomic(path, payload)
    return path


def finish_manifest(output_dir: Path, summary: Mapping[str, Any]) -> Path | None:
    running = output_dir / "run_manifest.running.json"
    if not running.is_file():
        return None
    payload = json.loads(running.read_text(encoding="utf-8"))
    success = summary.get("status") == "success"
    payload.update(
        status="ready" if success else "failed",
        finished_at_utc=_now(),
        outputs={
            name: _artifact(path)
            for name, path in {
                "summary": output_dir / "summary.json",
                "fatal_summary": output_dir / "fatal_summary.json",
                "video_mask_sequence": output_dir / "video_mask_sequence" / "video_mask_sequence.json",
                "final_registry": output_dir / "video_registry" / "final_registry.json",
            }.items()
            if path.is_file()
        },
        validation={"pipeline_status": summary.get("status")},
        failure=None if success else {
            "status": summary.get("status"),
            "failures": _jsonable(summary.get("failures", [])),
            "failed_videos": _jsonable(summary.get("failed_videos", [])),
        },
    )
    _write_atomic(running, payload)
    final = output_dir / "run_manifest.json"
    os.replace(running, final)
    return final
