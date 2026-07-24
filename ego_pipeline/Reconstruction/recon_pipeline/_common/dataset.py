"""Dataset discovery: videos organized by dataset name then video id."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from .paths import REPO_ROOT

HOI4D_RGB_RELPATH = Path("align_rgb") / "image.mp4"
VIDEO_ID_SEP = "__"


@dataclass(frozen=True)
class VideoJob:
    dataset: str
    video_id: str
    video_path: Path
    rel_path: str | None = None


def _load_vipe_hoi4d_helpers():
    import importlib.util
    import sys

    module_path = REPO_ROOT / "recon_pipeline" / "_legacy" / "vipe" / "_common.py"
    spec = importlib.util.spec_from_file_location("vipe_pipeline_common", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ViPE helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_vipe = _load_vipe_hoi4d_helpers()

hoi4d_name_from_rel = _vipe.hoi4d_name_from_rel
hoi4d_rel_from_name = _vipe.hoi4d_rel_from_name
hoi4d_rgb_video = _vipe.hoi4d_rgb_video
hoi4d_release_root = _vipe.hoi4d_release_root
discover_hoi4d_video_jobs = _vipe.discover_hoi4d_video_jobs
video_jobs_from_list_file = _vipe.video_jobs_from_list_file


def default_dataset_root(dataset: str) -> Path:
    if dataset == "hoi4d":
        return REPO_ROOT / "data" / "raw" / "HOI4D"
    raise ValueError(
        f"Unknown dataset {dataset!r}. Supported: hoi4d. "
        "Pass --dataset-root for custom layouts."
    )


def video_job_from_path(dataset: str, video_path: Path, *, video_id: str | None = None) -> VideoJob:
    video_path = video_path.resolve()
    rel: str | None = None
    if dataset == "hoi4d":
        try:
            rel = _vipe.hoi4d_rel_from_video(video_path)
        except ValueError:
            rel = None
    vid = video_id or (hoi4d_name_from_rel(rel) if rel else video_path.stem)
    return VideoJob(dataset=dataset, video_id=vid, video_path=video_path, rel_path=rel)


def generic_video_id(video_path: Path, root: Path) -> str:
    """Video id for non-hoi4d datasets: MP4 path relative to the dataset root, dropping
    the suffix and joining parts with VIDEO_ID_SEP ('__'). Reversible + collision-free,
    and mirrors the source layout under RECON_FINAL_NESTED (a__b__c -> a/b/c)."""
    rel = video_path.resolve().relative_to(Path(root).resolve())
    return VIDEO_ID_SEP.join(rel.with_suffix("").parts)


def resolve_video_job(
    dataset: str,
    entry: str,
    *,
    dataset_root: Path | None = None,
) -> VideoJob:
    """Resolve a video id, HOI4D rel path, or MP4 path to a VideoJob."""
    root = (dataset_root or default_dataset_root(dataset)).resolve()
    entry = entry.strip()
    path = Path(entry)
    if path.is_file() and path.suffix.lower() == ".mp4":
        vid = None
        if dataset != "hoi4d":
            try:
                vid = generic_video_id(path, root)   # path under root -> nested id
            except ValueError:
                vid = path.stem                      # outside root -> bare stem
        return video_job_from_path(dataset, path, video_id=vid)
    if dataset != "hoi4d":
        # generic dataset: entry is a generic id ("a__b__c") or a relative mp4 path.
        rel = entry.replace(VIDEO_ID_SEP, "/")
        cands = ([root / rel, Path(rel)] if rel.lower().endswith(".mp4")
                 else [root / f"{rel}.mp4", root / rel / "image.mp4"])
        for mp4 in cands:
            if mp4.is_file() and mp4.suffix.lower() == ".mp4":
                return video_job_from_path(dataset, mp4, video_id=generic_video_id(mp4, root))
        raise FileNotFoundError(f"Cannot resolve video entry {entry!r} for dataset {dataset!r} under {root}")
    if dataset == "hoi4d":
        release = hoi4d_release_root(root)
        if path.is_dir():
            mp4 = path / HOI4D_RGB_RELPATH
            if mp4.is_file():
                return video_job_from_path(dataset, mp4)
        if "/" in entry or entry.startswith("ZY"):
            rel = entry.replace(VIDEO_ID_SEP, "/") if VIDEO_ID_SEP in entry and "/" not in entry else entry
            mp4 = hoi4d_rgb_video(root, rel)
            if mp4.is_file():
                return VideoJob(
                    dataset=dataset,
                    video_id=hoi4d_name_from_rel(rel),
                    video_path=mp4,
                    rel_path=rel,
                )
        mp4 = release / entry.replace(VIDEO_ID_SEP, "/") / HOI4D_RGB_RELPATH
        if mp4.is_file():
            rel = entry.replace(VIDEO_ID_SEP, "/")
            return VideoJob(dataset=dataset, video_id=hoi4d_name_from_rel(rel), video_path=mp4, rel_path=rel)
    raise FileNotFoundError(f"Cannot resolve video entry {entry!r} for dataset {dataset!r}")


def discover_videos(
    dataset: str,
    input_path: Path | None = None,
    *,
    dataset_root: Path | None = None,
    video_list: Path | None = None,
    sample: int | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> list[VideoJob]:
    root = (dataset_root or default_dataset_root(dataset)).resolve()
    jobs: list[VideoJob] = []

    if video_list is not None:
        if dataset == "hoi4d":
            for mp4, name in video_jobs_from_list_file(video_list, root):
                jobs.append(VideoJob(dataset=dataset, video_id=name, video_path=mp4, rel_path=hoi4d_rel_from_name(name)))
        else:  # generic: each non-empty line is an MP4 path or a generic id
            for line in Path(video_list).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    jobs.append(resolve_video_job(dataset, line, dataset_root=root))
    elif input_path is not None:
        input_path = input_path.resolve()
        if input_path.is_file() and input_path.suffix.lower() == ".mp4":
            jobs = [resolve_video_job(dataset, str(input_path), dataset_root=root)]
        elif dataset == "hoi4d":
            for mp4, name in discover_hoi4d_video_jobs(input_path, root):
                jobs.append(
                    VideoJob(
                        dataset=dataset,
                        video_id=name,
                        video_path=mp4,
                        rel_path=hoi4d_rel_from_name(name),
                    )
                )
        else:  # generic: recurse for *.mp4 under the given dir
            for mp4 in sorted(input_path.rglob("*.mp4")):
                jobs.append(video_job_from_path(dataset, mp4, video_id=generic_video_id(mp4, root)))
    else:
        if dataset == "hoi4d":
            for mp4, name in discover_hoi4d_video_jobs(hoi4d_release_root(root), root):
                jobs.append(
                    VideoJob(
                        dataset=dataset,
                        video_id=name,
                        video_path=mp4,
                        rel_path=hoi4d_rel_from_name(name),
                    )
                )
        else:  # generic: recurse for *.mp4 under the dataset root
            for mp4 in sorted(root.rglob("*.mp4")):
                jobs.append(video_job_from_path(dataset, mp4, video_id=generic_video_id(mp4, root)))

    if sample is not None:
        if sample < 1:
            raise ValueError("--sample must be >= 1")
        rng = random.Random(seed)
        if len(jobs) > sample:
            jobs = rng.sample(jobs, sample)
        jobs.sort(key=lambda j: j.video_id)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        jobs = jobs[:limit]

    if not jobs:
        raise ValueError(f"No videos discovered for dataset {dataset!r}")
    return jobs


def write_video_manifest(jobs: list[VideoJob], path: Path, *, seed: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "videos": [
            {
                "dataset": j.dataset,
                "video_id": j.video_id,
                "video_path": str(j.video_path),
                "rel_path": j.rel_path,
            }
            for j in jobs
        ],
    }
    if seed is not None:
        payload["seed"] = seed
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
