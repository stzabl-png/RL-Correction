#!/usr/bin/env python
"""Resolve take paths/ids to canonical HOI4D video ids (one per line).

Each arg may be: a take dir (…/s02/T1), an align_rgb/image.mp4 file, a PARENT dir
(…/s02 — recurses to every take under it), or an already-canonical id / rel path.
Used by reconstruct.sh so the user can name takes by their Data path instead of
hand-writing a --video-list of flat ids.

  python ids_from_paths.py <path-or-id> [<path-or-id> ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> ego_pipeline/
from repo_paths import RECON_PIPELINE, RR_ROOT  # noqa: E402

RECON = RECON_PIPELINE
BIV2AP = RR_ROOT
DATA_ROOT = BIV2AP / "Data" / "HOI4D"
sys.path.insert(0, str(RECON))

from _common.dataset import discover_videos, resolve_video_job  # noqa: E402


def _resolve(arg: str) -> Path | None:
    """A relative path is tried against cwd first, then the Reconstruct_and_Retarget root, so callers
    can pass e.g. 'Data/HOI4D/...' regardless of which dir the wrapper script cd'd to."""
    p = Path(arg)
    if p.exists():
        return p
    alt = BIV2AP / arg
    return alt if alt.exists() else None


def main() -> int:
    seen: set[str] = set()
    for arg in sys.argv[1:]:
        p = _resolve(arg)
        if p is not None:
            jobs = discover_videos("hoi4d", input_path=p, dataset_root=DATA_ROOT)
            ids = [j.video_id for j in jobs]
        else:
            ids = [resolve_video_job("hoi4d", arg, dataset_root=DATA_ROOT).video_id]
        if not ids:
            print(f"[ids_from_paths] no take found under {arg!r}", file=sys.stderr)
            return 1
        for vid in ids:
            if vid not in seen:
                seen.add(vid)
                print(vid)
    return 0 if seen else 1


if __name__ == "__main__":
    raise SystemExit(main())
