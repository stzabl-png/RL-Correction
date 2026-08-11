#!/usr/bin/env python3
"""Run the isolated, no-training HOI-DETR candidate probe on one RGB video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT,):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from experiments.hoi_detr.probe_paths import (  # noqa: E402
    completion_marker,
    is_probe_complete,
    probe_step_dir,
    write_probe_completion,
)
from experiments.hoi_detr.adapter import (  # noqa: E402
    SCHEMA_VERSION,
    build_report,
    normalize_official_video,
    validate_candidates,
    write_json_atomic,
)
from experiments.hoi_detr.upstream import (  # noqa: E402
    CLASS_NAMES,
    OfficialHoiDetrRuntime,
    resolve_upstream_paths,
)
from experiments.hoi_detr.visualize import render_predictions_video  # noqa: E402


STEP = "hoi_detr_probe"
OFFICIAL_SOURCE_REVISION = "1b367292f3833afd64a204bd4d9d84519541d035"
OFFICIAL_CHECKPOINT_REVISION = "85719ac7bf20b8b67e26206faddf0d9582052046"
OFFICIAL_CHECKPOINT_SIZE = 5_855_053_598
OFFICIAL_CHECKPOINT_SHA256 = "4708fd0ddc5c3d386bad67c31152de58676840b911f6d01219e0092a603277d3"


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("expected a value in [0, 1]")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("expected an integer >= 1")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_revision(source_root: Path, expected_revision: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "HOI-DETR source must be a Git checkout so its revision can be verified"
        ) from exc
    actual_revision = result.stdout.strip().lower()
    if actual_revision != expected_revision.lower():
        raise RuntimeError(
            f"HOI-DETR source revision mismatch: expected {expected_revision}, "
            f"found {actual_revision}"
        )
    dirty = subprocess.run(
        ["git", "-C", str(source_root), "diff", "--quiet", "--"],
        check=False,
    )
    if dirty.returncode != 0:
        raise RuntimeError("HOI-DETR checkout has modified tracked files")
    return actual_revision


def _file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_run_spec(
    *,
    args: argparse.Namespace,
    video_path: Path,
    source_root: Path,
    source_revision: str,
    config: Path,
    checkpoint: Path,
) -> dict:
    return {
        "schema": "hoi_detr_probe_run_v1",
        "dataset": args.dataset,
        "video_id": args.video_id,
        "video": _file_identity(video_path),
        "source_root": str(source_root),
        "source_revision": source_revision,
        "config": {
            **_file_identity(config),
            "sha256": _sha256_file(config),
        },
        "checkpoint": {
            **_file_identity(checkpoint),
            "revision": args.checkpoint_revision,
            "expected_sha256": args.checkpoint_sha256.lower(),
        },
        "score_threshold": args.score_threshold,
        "nms_iou": args.nms_iou,
        "hf_threshold": args.hf_threshold,
        "fs_threshold": args.fs_threshold,
        "frame_stride": args.frame_stride,
        "visualize": bool(args.visualize),
    }


def _run_fingerprint(run_spec: dict) -> str:
    encoded = json.dumps(run_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _remove_previous_outputs(step_dir: Path, video_id: str) -> None:
    for path in (
        step_dir / "upstream_predictions.json",
        step_dir / "detections.json",
        step_dir / "hoi_detr_report.json",
        step_dir / "vis" / f"{video_id}.mp4",
        completion_marker(step_dir, STEP),
    ):
        try:
            path.unlink()
        except FileNotFoundError:  # py3.7 has no unlink(missing_ok=True)
            pass


def _move_staged_file(staging_dir: Path, step_dir: Path, relative_path: Path) -> None:
    source = staging_dir / relative_path
    destination = step_dir / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)


def _cuda_stats() -> tuple[int | None, int | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        torch.cuda.synchronize()
        return int(torch.cuda.memory_allocated()), int(torch.cuda.max_memory_allocated())
    except (ImportError, RuntimeError):
        return None, None


def _run_inference(
    *,
    video_path: Path,
    staging_dir: Path,
    source_root: Path,
    checkpoint: Path,
    config: Path | None,
    device: str,
    score_threshold: float,
    nms_iou: float,
    hf_threshold: float,
    fs_threshold: float,
    frame_stride: int,
) -> tuple[dict, dict]:
    import cv2

    load_started = time.perf_counter()
    runtime = OfficialHoiDetrRuntime.load(
        source_root=source_root,
        checkpoint=checkpoint,
        config=config,
        device=device,
    )
    load_seconds = time.perf_counter() - load_started

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open input video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if width < 1 or height < 1:
        cap.release()
        raise RuntimeError(f"Invalid video dimensions {width}x{height}: {video_path}")

    scratch_image = staging_dir / "scratch_frame.jpg"
    frames = []
    clipped_boxes = 0
    dropped_boxes = 0
    frame_idx = 0
    inference_started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            processed = frame_idx % frame_stride == 0
            if processed:
                prediction = runtime.predict_frame(
                    frame,
                    scratch_image=scratch_image,
                    score_threshold=score_threshold,
                    nms_iou=nms_iou,
                    hf_threshold=hf_threshold,
                    fs_threshold=fs_threshold,
                )
                diagnostics = prediction.pop("diagnostics")
                clipped_boxes += int(diagnostics["clipped_boxes"])
                dropped_boxes += int(diagnostics["dropped_degenerate_boxes"])
            else:
                # Unlike the official demo, skipped frames are intentionally
                # empty. Reusing a prior frame's boxes would be a false result.
                prediction = {"detections": [], "hf": [], "fs": []}
            frames.append(
                {
                    "frame_idx": frame_idx,
                    "processed": processed,
                    **prediction,
                }
            )
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"[{STEP}] processed video frame {frame_idx}", flush=True)
    finally:
        cap.release()
        try:
            scratch_image.unlink()
        except FileNotFoundError:  # py3.7 has no unlink(missing_ok=True)
            pass
    inference_seconds = time.perf_counter() - inference_started

    if frame_idx < 1:
        raise RuntimeError(f"No decodable frames in {video_path}")
    current_cuda_bytes, peak_cuda_bytes = _cuda_stats()
    upstream_payload = {
        "type": "video",
        "video_path": str(video_path),
        "fps": fps,
        "width": width,
        "height": height,
        "num_frames": frame_idx,
        "score_thr": score_threshold,
        "nms_iou": nms_iou,
        "frame_stride": frame_stride,
        "class_names": list(CLASS_NAMES),
        "frames": frames,
    }
    runtime_stats = {
        "reported_num_frames": reported_num_frames,
        "decoded_num_frames": frame_idx,
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "seconds_per_processed_frame": inference_seconds
        / max(1, sum(bool(frame["processed"]) for frame in frames)),
        "cuda_memory_allocated_bytes": current_cuda_bytes,
        "cuda_peak_memory_allocated_bytes": peak_cuda_bytes,
        "clipped_boxes": clipped_boxes,
        "dropped_degenerate_boxes": dropped_boxes,
    }
    return upstream_payload, runtime_stats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hoi-detr-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--score-threshold", type=_probability, default=0.3)
    parser.add_argument("--nms-iou", type=_probability, default=0.5)
    parser.add_argument("--hf-threshold", type=_probability, default=0.6)
    parser.add_argument("--fs-threshold", type=_probability, default=0.92)
    parser.add_argument("--frame-stride", type=_positive_int, default=1)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--checkpoint-authorized",
        action="store_true",
        help="Assert that the checkpoint author has authorized this use",
    )
    parser.add_argument("--source-revision", default=OFFICIAL_SOURCE_REVISION)
    parser.add_argument("--checkpoint-revision", default=OFFICIAL_CHECKPOINT_REVISION)
    parser.add_argument("--checkpoint-size-bytes", type=_positive_int, default=OFFICIAL_CHECKPOINT_SIZE)
    parser.add_argument("--checkpoint-sha256", default=OFFICIAL_CHECKPOINT_SHA256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.checkpoint_authorized:
        print(
            "Refusing to load the checkpoint: the official model card requires "
            "author authorization. Pass --checkpoint-authorized only after that "
            "permission has been obtained.",
            file=sys.stderr,
        )
        return 2

    video_path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"Input video not found: {video_path}", file=sys.stderr)
        return 2
    try:
        source_root, config, checkpoint = resolve_upstream_paths(
            args.hoi_detr_root,
            args.checkpoint,
            args.config,
        )
        source_revision = _verify_source_revision(source_root, args.source_revision)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2
    if checkpoint.stat().st_size != args.checkpoint_size_bytes:
        print(
            f"Preflight failed: checkpoint size is {checkpoint.stat().st_size}, "
            f"expected {args.checkpoint_size_bytes}",
            file=sys.stderr,
        )
        return 2
    if len(args.checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in args.checkpoint_sha256
    ):
        print("Preflight failed: --checkpoint-sha256 must be 64 hexadecimal characters", file=sys.stderr)
        return 2

    run_spec = _build_run_spec(
        args=args,
        video_path=video_path,
        source_root=source_root,
        source_revision=source_revision,
        config=config,
        checkpoint=checkpoint,
    )
    run_fingerprint = _run_fingerprint(run_spec)
    step_dir = probe_step_dir(args.dataset, args.video_id, STEP)
    required_outputs = (
        step_dir / "upstream_predictions.json",
        step_dir / "detections.json",
        step_dir / "hoi_detr_report.json",
    ) + ((step_dir / "vis" / f"{args.video_id}.mp4",) if args.visualize else ())
    if not args.force and is_probe_complete(
        step_dir,
        STEP,
        run_fingerprint=run_fingerprint,
        required_outputs=required_outputs,
    ):
        print(f"Skip {args.video_id}: {STEP} already complete", flush=True)
        return 0

    print(f"Verifying checkpoint SHA-256: {checkpoint}", flush=True)
    actual_checkpoint_sha256 = _sha256_file(checkpoint)
    if actual_checkpoint_sha256.lower() != args.checkpoint_sha256.lower():
        print(
            "Preflight failed: checkpoint SHA-256 mismatch; "
            f"expected {args.checkpoint_sha256.lower()}, found {actual_checkpoint_sha256}",
            file=sys.stderr,
        )
        return 2
    step_dir.mkdir(parents=True, exist_ok=True)
    _remove_previous_outputs(step_dir, args.video_id)
    staging_dir = Path(tempfile.mkdtemp(prefix=".staging-", dir=str(step_dir)))

    inherited_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not inherited_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda:0"
    else:
        device = f"cuda:{args.gpu}"
    committed = False
    try:
        upstream_payload, runtime_stats = _run_inference(
            video_path=video_path,
            staging_dir=staging_dir,
            source_root=source_root,
            checkpoint=checkpoint,
            config=config,
            device=device,
            score_threshold=args.score_threshold,
            nms_iou=args.nms_iou,
            hf_threshold=args.hf_threshold,
            fs_threshold=args.fs_threshold,
            frame_stride=args.frame_stride,
        )
        source_metadata = {
            "source_revision": source_revision,
            "checkpoint_revision": args.checkpoint_revision,
            "source_root": str(source_root),
            "checkpoint_path": str(checkpoint),
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": actual_checkpoint_sha256,
            "config_path": str(config),
            "config_sha256": run_spec["config"]["sha256"],
            "requested_gpu": args.gpu,
            "inherited_cuda_visible_devices": inherited_visible_devices,
            "effective_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": device,
            "hf_threshold": args.hf_threshold,
            "fs_threshold": args.fs_threshold,
            "box_policy": "clip_to_frame_drop_degenerate",
            "runtime": runtime_stats,
        }
        candidates = normalize_official_video(
            upstream_payload,
            dataset=args.dataset,
            video_id=args.video_id,
            video_path=video_path,
            source_metadata=source_metadata,
        )
        validate_candidates(candidates)
        report = build_report(candidates)

        write_json_atomic(staging_dir / "upstream_predictions.json", upstream_payload)
        write_json_atomic(staging_dir / "detections.json", candidates)
        write_json_atomic(staging_dir / "hoi_detr_report.json", report)
        staged_paths = [
            Path("upstream_predictions.json"),
            Path("detections.json"),
            Path("hoi_detr_report.json"),
        ]
        if args.visualize:
            relative_vis = Path("vis") / f"{args.video_id}.mp4"
            render_predictions_video(
                video_path=video_path,
                frames=upstream_payload["frames"],
                output_path=staging_dir / relative_vis,
            )
            staged_paths.append(relative_vis)

        for relative_path in staged_paths:
            _move_staged_file(staging_dir, step_dir, relative_path)
        summary = report["summary"]
        write_probe_completion(
            step_dir,
            STEP,
            dataset=args.dataset,
            video_id=args.video_id,
            extra={
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": run_fingerprint,
                "run_spec": run_spec,
                "num_frames": candidates["video"]["num_frames"],
                "num_processed_frames": summary["num_processed_frames"],
                "num_detections": summary["processed_frames_only"]["num_detections"],
                "num_hf_links": summary["processed_frames_only"]["links_by_kind"]["hf"],
                "num_fs_links": summary["processed_frames_only"]["links_by_kind"]["fs"],
                "detections_json": str(step_dir / "detections.json"),
                "upstream_predictions_json": str(step_dir / "upstream_predictions.json"),
                "report_json": str(step_dir / "hoi_detr_report.json"),
                "vis_video": str(step_dir / "vis" / f"{args.video_id}.mp4")
                if args.visualize
                else None,
            },
        )
        committed = True
        print(json.dumps(summary["processed_frames_only"], indent=2), flush=True)
        return 0
    finally:
        if not committed:
            _remove_previous_outputs(step_dir, args.video_id)
        shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
