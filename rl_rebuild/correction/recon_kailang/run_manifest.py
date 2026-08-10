"""Small manifest writer for the static-reconstruction delivery boundary."""

from __future__ import annotations

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


def artifact(path: str | Path, *, label: str | None = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    record: dict[str, Any] = {
        "path": label or str(resolved),
        "resolved_path": str(resolved),
        "exists": resolved.is_file(),
    }
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


def _repo_state() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-c", f"safe.directory={root}", "-C", str(root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    commit = run("rev-parse", "HEAD")
    if commit.returncode != 0:
        return {"available": False, "root": str(root), "reason": commit.stderr.strip()}
    branch = run("symbolic-ref", "--quiet", "--short", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    diff = run("diff", "--binary", "HEAD")
    return {
        "available": True,
        "root": str(root),
        "commit": commit.stdout.strip(),
        "branch": branch.stdout.strip() if branch.returncode == 0 else None,
        "dirty": bool(status.stdout),
        "working_tree_fingerprint_sha256": hashlib.sha256(
            status.stdout.encode() + b"\0" + diff.stdout.encode()
        ).hexdigest(),
        "changed_paths": [line[3:] for line in status.stdout.splitlines()],
    }


def start_manifest(
    directory: str | Path,
    *,
    final_name: str,
    stage: str,
    entrypoint: str | Path,
    command: Sequence[str],
    inputs: Mapping[str, Mapping[str, Any]],
    parameters: Mapping[str, Any],
    expected_outputs: Sequence[str],
) -> Path:
    target = Path(directory).expanduser().resolve()
    payload = {
        "schema_version": "static_reconstruction_run_manifest_v1",
        "run_id": str(uuid.uuid4()),
        "stage": stage,
        "status": "running",
        "started_at_utc": _now(),
        "finished_at_utc": None,
        "machine": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "code": {
            "git": _repo_state(),
            "entrypoint": artifact(entrypoint),
        },
        "command": list(command),
        "parameters": _jsonable(parameters),
        "inputs": _jsonable(inputs),
        "expected_outputs": list(expected_outputs),
        "outputs": {},
        "validation": {},
        "failure": None,
        "final_name": final_name,
    }
    running = target / f"{Path(final_name).stem}.running.json"
    _write_atomic(running, payload)
    return running


def finish_manifest(
    running: str | Path,
    *,
    status: str,
    outputs: Mapping[str, Mapping[str, Any]],
    validation: Mapping[str, Any],
    failure: Mapping[str, Any] | str | None = None,
) -> Path:
    if status not in {"ready", "failed"}:
        raise ValueError("status must be ready or failed")
    running_path = Path(running).expanduser().resolve()
    payload = json.loads(running_path.read_text(encoding="utf-8"))
    final_name = payload.pop("final_name")
    payload.update(
        status=status,
        finished_at_utc=_now(),
        outputs=_jsonable(outputs),
        validation=_jsonable(validation),
        failure=_jsonable(failure),
    )
    _write_atomic(running_path, payload)
    final = running_path.parent / final_name
    os.replace(running_path, final)
    return final
