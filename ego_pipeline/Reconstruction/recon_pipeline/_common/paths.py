"""Path layout: data/interim/{dataset}/{video_id}/{step}/ and data/reconstruction/{dataset}/{video_id}/."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
# Output roots are env-configurable so a host project (e.g. Bi-V2AP) can route
# artifacts to its own Output/ tree without moving the code. Defaults unchanged.
INTERIM_ROOT = Path(os.environ["RECON_INTERIM_ROOT"]).resolve() if os.environ.get("RECON_INTERIM_ROOT") else REPO_ROOT / "data" / "interim"
FINAL_ROOT = Path(os.environ["RECON_FINAL_ROOT"]).resolve() if os.environ.get("RECON_FINAL_ROOT") else REPO_ROOT / "data" / "reconstruction"

STEP_NAMES = (
    "vipe",
    "sam3_hands",
    "sam2_object",
    "hawor",
    "sam3d",
    "sam3d_scale",
    "fp_pose",
    "fuse",
)

COMPLETION_BASENAME = "{step}_complete.json"


def interim_step_dir(dataset: str, video_id: str, step: str) -> Path:
    return INTERIM_ROOT / dataset / video_id / step


def final_video_dir(dataset: str, video_id: str) -> Path:
    # A host project can mirror the source dataset's own directory nesting in the
    # final tree: a flat take id "A__B__C" is expanded to nested "A/B/C". Opt-in via
    # RECON_FINAL_NESTED=1; default layout (single flat component) is unchanged, so
    # this stays transparent to other consumers of the engine.
    if os.environ.get("RECON_FINAL_NESTED") == "1" and "__" in video_id:
        return FINAL_ROOT.joinpath(dataset, *video_id.split("__"))
    return FINAL_ROOT / dataset / video_id


def completion_marker(step_dir: Path, step: str) -> Path:
    return step_dir / COMPLETION_BASENAME.format(step=step)


def is_step_complete(step_dir: Path, step: str) -> bool:
    marker = completion_marker(step_dir, step)
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "complete"


def write_step_completion(
    step_dir: Path,
    step: str,
    *,
    dataset: str,
    video_id: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    step_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "status": "complete",
        "step": step,
        "dataset": dataset,
        "video_id": video_id,
    }
    if extra:
        payload.update(extra)
    marker = completion_marker(step_dir, step)
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return marker


def worker_log_path(step_dir: Path, video_id: str) -> Path:
    return step_dir / ".logs" / f"{video_id}.log"


def resolve_repo_path(path: Path) -> Path:
    """Resolve a repo-relative path against REPO_ROOT (not process cwd)."""
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()
