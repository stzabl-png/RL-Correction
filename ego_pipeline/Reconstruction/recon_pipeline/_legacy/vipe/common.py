"""Shared helpers for HumanVideo2RobotData ViPE pipeline scripts."""

from __future__ import annotations

import fcntl
import json
import os
import random
import shutil
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

SCRIPT_DIR = Path(__file__).resolve().parent

REPO_ROOT = Path(__file__).resolve().parents[3]
VIPE_ROOT = REPO_ROOT / "third_party" / "vipe"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "testing" / "vipe" / "hoi4d"
DEFAULT_HOI4D_ROOT = REPO_ROOT / "data" / "raw" / "HOI4D"
DEFAULT_RGB_ROOT = DEFAULT_HOI4D_ROOT / "HOI4D_release"
DEFAULT_ANNOTATIONS_ROOT = DEFAULT_HOI4D_ROOT / "HOI4D_annotations"
DEFAULT_DEPTH_ROOT = DEFAULT_HOI4D_ROOT / "HOI4D_depth_video"
DEFAULT_BENCHMARK_SUBSET = DEFAULT_HOI4D_ROOT / "test_subset.json"

HOI4D_RGB_RELPATH = Path("align_rgb") / "image.mp4"
HOI4D_DEPTH_RELPATH = Path("align_depth") / "depth_video.avi"
HOI4D_POSE_RELPATH = Path("3Dseg") / "output.log"
HOI4D_SEQUENCE_SEP = "__"

COMPLETION_FILES = ("pose", "depth", "intrinsics", "meta")
# Written by ViPE but not consumed by downstream reconstruction steps.
VIPE_NONESSENTIAL_DIRS = ("rgb", "mask")


def emit_progress(current: int, total: int, event: str, name: str, *, detail: str = "") -> None:
    suffix = f" — {detail}" if detail else ""
    print(f"[{current}/{total}] {event:5s} {name}{suffix}", flush=True)


def ensure_vipe_importable(vipe_root: Path | None = None) -> Path:
    root = (vipe_root or VIPE_ROOT).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"ViPE root not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def list_videos(input_path: Path, limit: int | None = None) -> list[Path]:
    path = input_path.resolve()
    if path.is_file():
        if path.suffix.lower() != ".mp4":
            raise ValueError(f"Expected an MP4 file, got: {path}")
        videos = [path]
    elif not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    else:
        videos = sorted(path.glob("*.mp4"))
        if not videos:
            raise FileNotFoundError(f"No MP4 files found under: {path}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        videos = videos[:limit]
    return videos


def hoi4d_name_from_rel(rel: str) -> str:
    return rel.replace("/", HOI4D_SEQUENCE_SEP)


def hoi4d_rel_from_name(name: str) -> str:
    return name.replace(HOI4D_SEQUENCE_SEP, "/")


def hoi4d_release_root(hoi4d_root: Path) -> Path:
    return hoi4d_root.resolve() / "HOI4D_release"


def hoi4d_annotations_root(hoi4d_root: Path) -> Path:
    return hoi4d_root.resolve() / "HOI4D_annotations"


def hoi4d_depth_root(hoi4d_root: Path) -> Path:
    return hoi4d_root.resolve() / "HOI4D_depth_video"


def hoi4d_rgb_video(hoi4d_root: Path, rel: str) -> Path:
    return hoi4d_release_root(hoi4d_root) / rel / HOI4D_RGB_RELPATH


def hoi4d_depth_video(hoi4d_root: Path, rel: str) -> Path:
    return hoi4d_depth_root(hoi4d_root) / rel / HOI4D_DEPTH_RELPATH


def hoi4d_pose_log(hoi4d_root: Path, rel: str) -> Path:
    return hoi4d_annotations_root(hoi4d_root) / rel / HOI4D_POSE_RELPATH


def hoi4d_rel_from_video(video_path: Path, release_root: Path | None = None) -> str:
    video_path = video_path.resolve()
    if release_root is not None:
        rel_video = video_path.relative_to(release_root.resolve())
    else:
        rel_video = None
        for parent in video_path.parents:
            if parent.name == "HOI4D_release":
                rel_video = video_path.relative_to(parent)
                break
        if rel_video is None:
            raise ValueError(f"Video is not under HOI4D_release: {video_path}")

    parts = rel_video.parts
    if len(parts) < 3 or parts[-2:] != HOI4D_RGB_RELPATH.parts:
        raise ValueError(f"Expected …/align_rgb/image.mp4, got: {video_path}")
    return "/".join(parts[:-2])


def hoi4d_sequence_name_from_video(video_path: Path, release_root: Path | None = None) -> str:
    return hoi4d_name_from_rel(hoi4d_rel_from_video(video_path, release_root))


def load_hoi4d_pose_log(path: Path) -> np.ndarray:
    import numpy as np

    lines = path.read_text().splitlines()
    poses: list[np.ndarray] = []
    idx = 0
    while idx < len(lines):
        parts = lines[idx].split()
        if len(parts) == 3 and all(part.lstrip("-").isdigit() for part in parts):
            mat = [[float(x) for x in lines[idx + offset].split()] for offset in range(1, 5)]
            poses.append(np.array(mat, dtype=np.float64))
            idx += 5
        else:
            idx += 1
    if not poses:
        raise ValueError(f"No poses found in {path}")
    return np.stack(poses)


def _probe_depth_video(avi_path: Path) -> tuple[int, int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames",
            "-of",
            "json",
            str(avi_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return int(stream["width"]), int(stream["height"]), int(stream["nb_frames"])


def load_hoi4d_depth_frames(
    avi_path: Path,
    frame_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Decode HOI4D depth AVI (ffv1 gray16le, millimeters) to float32 meters."""
    import numpy as np

    avi_path = avi_path.resolve()
    if not avi_path.is_file():
        raise FileNotFoundError(f"Depth video not found: {avi_path}")

    width, height, num_frames = _probe_depth_video(avi_path)
    if frame_indices is None:
        indices = np.arange(num_frames, dtype=np.int64)
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(avi_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",
            "-",
        ]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        frames = np.frombuffer(raw, dtype=np.uint16).reshape(num_frames, height, width)
        return frames.astype(np.float32) / 1000.0

    indices = np.asarray(frame_indices, dtype=np.int64)
    out = np.zeros((len(indices), height, width), dtype=np.float32)
    for out_idx, frame_idx in enumerate(indices):
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(avi_path),
            "-vf",
            f"select=eq(n\\,{int(frame_idx)})",
            "-vframes",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray16le",
            "-",
        ]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        out[out_idx] = np.frombuffer(raw, dtype=np.uint16).reshape(height, width).astype(np.float32) / 1000.0
    return out


def hoi4d_annotation_paths(hoi4d_root: Path, sequence_name: str) -> dict[str, Path]:
    rel = hoi4d_rel_from_name(sequence_name)
    return {
        "rel": Path(rel),
        "rgb": hoi4d_rgb_video(hoi4d_root, rel),
        "depth": hoi4d_depth_video(hoi4d_root, rel),
        "pose_log": hoi4d_pose_log(hoi4d_root, rel),
    }


def load_benchmark_subset_rels(subset_path: Path) -> list[str]:
    payload = json.loads(subset_path.read_text(encoding="utf-8"))
    return [entry["rel"] for entry in payload]


def discover_hoi4d_video_jobs(
    input_path: Path,
    hoi4d_root: Path,
    *,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 42,
    subset_path: Path | None = None,
) -> list[tuple[Path, str]]:
    """Return (rgb_mp4_path, filesystem_safe_sequence_name) pairs."""
    input_path = input_path.resolve()
    hoi4d_root = hoi4d_root.resolve()
    release_root = hoi4d_release_root(hoi4d_root)

    if subset_path is not None:
        rels = load_benchmark_subset_rels(subset_path.resolve())
        jobs = [(hoi4d_rgb_video(hoi4d_root, rel), hoi4d_name_from_rel(rel)) for rel in rels]
    elif input_path.is_file():
        rel = hoi4d_rel_from_video(input_path, release_root)
        jobs = [(input_path, hoi4d_name_from_rel(rel))]
    else:
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")
        nested = sorted(input_path.glob("**/align_rgb/image.mp4"))
        if nested:
            jobs = [
                (video, hoi4d_sequence_name_from_video(video, release_root))
                for video in nested
            ]
        else:
            flat = sorted(input_path.glob("*.mp4"))
            if not flat:
                raise FileNotFoundError(
                    f"No HOI4D videos (…/align_rgb/image.mp4) or flat MP4 files under: {input_path}"
                )
            jobs = [(video, video.stem) for video in flat]

    if sample is not None and limit is not None:
        raise ValueError("Use only one of --limit or --sample")
    if sample is not None:
        if sample < 1:
            raise ValueError("--sample must be >= 1")
        rng = random.Random(seed)
        if sample < len(jobs):
            jobs = rng.sample(jobs, sample)
    elif limit is not None:
        if limit < 1:
            raise ValueError("--limit must be >= 1")
        jobs = jobs[:limit]
    return jobs


def write_sample_manifest(
    output_dir: Path,
    jobs: list[tuple[Path, str]],
    *,
    seed: int | None = None,
) -> Path:
    payload = {
        "seed": seed,
        "num_sequences": len(jobs),
        "sequences": [
            {"name": name, "rel": hoi4d_rel_from_name(name), "video": str(video.resolve())}
            for video, name in jobs
        ],
    }
    path = output_dir.resolve() / "sample_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_sample_manifest_names(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [entry["name"] for entry in payload["sequences"]]


def parse_video_list_file(list_path: Path) -> list[str]:
    """Parse a video list file (.txt, .lst, or .json)."""
    list_path = list_path.resolve()
    if not list_path.is_file():
        raise FileNotFoundError(f"Video list file not found: {list_path}")

    suffix = list_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(list_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
        if isinstance(payload, dict) and "sequences" in payload:
            entries: list[str] = []
            for item in payload["sequences"]:
                if isinstance(item, str):
                    entries.append(item.strip())
                elif isinstance(item, dict):
                    if item.get("video"):
                        entries.append(str(item["video"]).strip())
                    elif item.get("rel"):
                        entries.append(str(item["rel"]).strip())
                    elif item.get("name"):
                        entries.append(str(item["name"]).strip())
            return [entry for entry in entries if entry]
        raise ValueError(
            f"Unsupported JSON video list format: {list_path} "
            "(expected a JSON array or {{\"sequences\": [...]}})"
        )

    entries: list[str] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            entries.append(line)
    return entries


def _resolve_video_list_entry(
    entry: str,
    hoi4d_root: Path,
    release_root: Path,
) -> tuple[Path, str]:
    """Resolve one list entry to (mp4_path, sequence_name)."""
    candidate = Path(entry)
    if candidate.suffix.lower() == ".mp4":
        if not candidate.is_file():
            raise FileNotFoundError(f"MP4 not found for list entry: {entry}")
        video_path = candidate.resolve()
        try:
            rel = hoi4d_rel_from_video(video_path, release_root)
            return video_path, hoi4d_name_from_rel(rel)
        except ValueError:
            return video_path, video_path.stem

    if HOI4D_SEQUENCE_SEP in entry:
        rel = hoi4d_rel_from_name(entry)
        video_path = hoi4d_rgb_video(hoi4d_root, rel)
        if not video_path.is_file():
            raise FileNotFoundError(f"RGB video not found for list entry: {entry} -> {video_path}")
        return video_path, hoi4d_name_from_rel(rel)

    if "/" in entry:
        video_path = hoi4d_rgb_video(hoi4d_root, entry)
        if not video_path.is_file():
            raise FileNotFoundError(f"RGB video not found for list entry: {entry} -> {video_path}")
        return video_path, hoi4d_name_from_rel(entry)

    raise ValueError(
        f"Cannot resolve video list entry {entry!r}; "
        "use an MP4 path, HOI4D rel (ZY…/H…/…/T#), or sequence name (ZY…__H…__…__T#)"
    )


def video_jobs_from_list_file(list_path: Path, hoi4d_root: Path) -> list[tuple[Path, str]]:
    """Return (rgb_mp4_path, filesystem_safe_sequence_name) pairs from a list file."""
    hoi4d_root = hoi4d_root.resolve()
    release_root = hoi4d_release_root(hoi4d_root)
    jobs: list[tuple[Path, str]] = []
    for entry in parse_video_list_file(list_path):
        jobs.append(_resolve_video_list_entry(entry, hoi4d_root, release_root))
    if not jobs:
        raise ValueError(f"No videos found in list file: {list_path}")
    return jobs


def stage_hoi4d_video_for_vipe(video_path: Path, sequence_name: str, staging_dir: Path) -> Path:
    """Symlink nested HOI4D image.mp4 to a unique {sequence_name}.mp4 for ViPE artifact naming."""
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged = staging_dir / f"{sequence_name}.mp4"
    if staged.is_symlink() or staged.exists():
        staged.unlink()
    staged.symlink_to(video_path.resolve())
    return staged


def artifact_paths(output_dir: Path, name: str) -> dict[str, Path]:
    base = output_dir.resolve()
    return {
        "pose": base / "pose" / f"{name}.npz",
        "depth": base / "depth" / f"{name}.zip",
        "intrinsics": base / "intrinsics" / f"{name}.npz",
        "meta": base / "vipe" / f"{name}_info.pkl",
        "rgb": base / "rgb" / f"{name}.mp4",
        "mask": base / "mask" / f"{name}.zip",
        "mask_phrases": base / "mask" / f"{name}.txt",
        "viz": base / "vis" / f"{name}_vis.mp4",
    }


def is_sequence_complete(output_dir: Path, name: str) -> bool:
    paths = artifact_paths(output_dir, name)
    return all(paths[key].is_file() for key in COMPLETION_FILES)


@contextmanager
def sequence_lock(output_dir: Path, name: str) -> Iterator[None]:
    """Exclusive lock so two workers never process the same sequence."""
    lock_dir = output_dir / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"{name}.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def write_status(output_dir: Path, name: str, status: str, **extra: object) -> None:
    status_dir = output_dir / ".status"
    status_dir.mkdir(parents=True, exist_ok=True)
    payload = {"sequence": name, "status": status, **extra}
    (status_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def worker_log_path(output_dir: Path, name: str) -> Path:
    return output_dir / ".logs" / f"{name}.log"


def cleanup_run_metadata(output_dir: Path) -> None:
    """Remove ephemeral parallel-run bookkeeping (.locks, .status). Not used by resume."""
    for dirname in (".locks", ".status"):
        path = output_dir / dirname
        if path.is_dir():
            shutil.rmtree(path)


def discard_vipe_nonessential_artifacts(output_dir: Path, name: str) -> None:
    """Drop ViPE copies not needed for reconstruction (rgb/mask copies, staging symlinks)."""
    base = output_dir.resolve()
    for dirname in VIPE_NONESSENTIAL_DIRS:
        path = base / dirname
        if path.is_dir():
            shutil.rmtree(path)
    staging = base / ".input_links"
    if staging.is_dir():
        shutil.rmtree(staging)
    camera_txt = base / "intrinsics" / f"{name}_camera.txt"
    camera_txt.unlink(missing_ok=True)


def setup_script_imports() -> Path:
    """Ensure legacy ViPE helpers are importable when invoked by absolute path."""
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    return SCRIPT_DIR


@dataclass
class VipeRunConfig:
    input_path: Path
    output_dir: Path
    vipe_root: Path = VIPE_ROOT
    pipeline: str = "default"
    frame_start: int = 0
    frame_end: int = 1000
    frame_skip: int = 1
    discard_nonessential_artifacts: bool = False
    extra_overrides: list[str] = field(default_factory=list)

    def build_run_py_command(self, video_path: Path) -> list[str]:
        overrides = [
            f"pipeline={self.pipeline}",
            "streams=raw_mp4_stream",
            f"streams.base_path={video_path}",
            f"streams.frame_start={self.frame_start}",
            f"streams.frame_end={self.frame_end}",
            f"streams.frame_skip={self.frame_skip}",
            f"pipeline.output.path={self.output_dir}",
            "pipeline.output.save_artifacts=true",
            "pipeline.output.save_viz=false",
            "pipeline.slam.visualize=false",
        ]
        overrides.extend(self.extra_overrides)
        return [
            "uv",
            "run",
            "python",
            "run.py",
            *overrides,
        ]


def uv_python_command(vipe_root: Path, script_path: Path, *args: str) -> list[str]:
    return ["uv", "run", "python", str(script_path), *args]


def run_python_script(
    vipe_root: Path,
    script_path: Path,
    script_dir: Path,
    extra_args: list[str],
    *,
    check: bool = True,
    loud: bool = False,
    log_path: Path | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(script_dir) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = uv_python_command(vipe_root, script_path, *extra_args)
    return _run_subprocess(cmd, cwd=vipe_root, env=env, check=check, loud=loud, log_path=log_path)


def _run_subprocess(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    check: bool,
    loud: bool,
    log_path: Path | None,
) -> subprocess.CompletedProcess:
    if loud:
        return subprocess.run(cmd, cwd=str(cwd), env=env, check=check)

    assert log_path is not None, "log_path is required when loud=False"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        return subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            check=check,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def run_vipe_inference(
    video_path: Path,
    config: VipeRunConfig,
    gpu_id: int,
    *,
    sequence_name: str | None = None,
    loud: bool = False,
    log_path: Path | None = None,
) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    name = sequence_name or video_path.stem
    staging_dir = config.output_dir / ".input_links"
    staged_video = stage_hoi4d_video_for_vipe(video_path, name, staging_dir)
    cmd = config.build_run_py_command(staged_video)
    _run_subprocess(
        cmd,
        cwd=config.vipe_root,
        env=env,
        check=True,
        loud=loud,
        log_path=log_path or worker_log_path(config.output_dir, name),
    )
    if config.discard_nonessential_artifacts and is_sequence_complete(config.output_dir, name):
        discard_vipe_nonessential_artifacts(config.output_dir, name)


def parse_gpu_list(gpus: str) -> list[int]:
    values: list[int] = []
    for part in gpus.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise ValueError("At least one GPU id is required, e.g. --gpus 0 or --gpus 0,1")
    return values
