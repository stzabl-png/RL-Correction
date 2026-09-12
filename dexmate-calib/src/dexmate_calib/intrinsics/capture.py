from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from dexmate_calib.boards.config import BoardProfile
from dexmate_calib.intrinsics.detector import CharucoDetector, Detection
from dexmate_calib.streaming.zed_stream import ZedStreamClient


@dataclass(frozen=True)
class CaptureSettings:
    output_root: Path
    max_samples: int = 40
    min_corners: int = 20
    min_coverage: float = 0.025
    min_sharpness: float = 80.0
    min_grid_rows: int = 3
    min_grid_cols: int = 3
    min_board_bbox_fraction: float = 0.12
    min_diversity_distance: float = 0.08
    cooldown_s: float = 0.8
    detection_fps: float = 10.0
    preview: bool = True
    auto_capture: bool = True
    streamer_started_by_quickstart: bool = False
    streamer_ssh_route: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _quality_reason(detection: Detection | None, settings: CaptureSettings) -> str | None:
    if detection is None:
        return "board_not_detected"
    if detection.corner_count < settings.min_corners:
        return f"corners<{settings.min_corners}"
    if detection.grid_rows < settings.min_grid_rows:
        return f"grid_rows<{settings.min_grid_rows}"
    if detection.grid_cols < settings.min_grid_cols:
        return f"grid_cols<{settings.min_grid_cols}"
    if detection.board_bbox_fraction < settings.min_board_bbox_fraction:
        return f"board_bbox<{settings.min_board_bbox_fraction:.2f}"
    if detection.coverage_fraction < settings.min_coverage:
        return f"coverage<{settings.min_coverage:.3f}"
    if detection.sharpness < settings.min_sharpness:
        return f"sharpness<{settings.min_sharpness:.1f}"
    return None


def detection_is_due(
    now: float,
    last_detection_monotonic: float,
    settings: CaptureSettings,
) -> bool:
    if settings.detection_fps < 0:
        raise ValueError("detection_fps must be >= 0")
    if not settings.auto_capture or settings.detection_fps == 0:
        return True
    return now - last_detection_monotonic >= 1.0 / settings.detection_fps


def render_capture_preview(
    detector: CharucoDetector,
    image,
    detection: Detection | None,
    *,
    accepted: int,
    target: int,
    state: str,
):
    import cv2

    display = detector.draw(image, detection)
    color = (0, 220, 0) if state in {"ready", "saved"} else (0, 170, 255)
    lines = [f"saved {accepted}/{target} | {state}"]
    if detection is not None:
        lines.extend(
            [
                (
                    f"markers={detection.marker_count} corners={detection.corner_count} "
                    f"grid={detection.grid_cols}x{detection.grid_rows}"
                ),
                (
                    f"coverage={detection.coverage_fraction:.3f} "
                    f"board_bbox={detection.board_bbox_fraction:.2f} "
                    f"sharp={detection.sharpness:.1f} "
                    f"scale={detection.pixels_per_square:.1f}px/square"
                ),
            ]
        )
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (24, 42 + index * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78 if index == 0 else 0.62,
            color if index == 0 else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return display


def _write_manifest(
    path: Path,
    *,
    created_at_utc: str,
    status: str,
    board: BoardProfile,
    board_snapshot: Path,
    identity: dict | None,
    accepted: int,
    frame_index: int,
    settings: CaptureSettings,
    stream_host: str,
    stream_port: int,
) -> None:
    manifest = {
        "schema": "dexmate_calib.intrinsic_session.v1",
        "created_at_utc": created_at_utc,
        "updated_at_utc": _utc_now(),
        "status": status,
        "camera": {
            "name": "head_left",
            "zed_mode": "HD1200",
            "expected_resolution": {"width": 1920, "height": 1200},
            "image_view": "LEFT",
            "image_geometry": "rectified",
            "distortion_model": "none",
            **(identity or {}),
        },
        "streamer": {
            "host": stream_host,
            "port": stream_port,
            "expected_command": (
                "sudo /home/dexmate-nano/zed_stream/build/zed_streamer "
                "--jpeg-quality 100 --max-fps 30 --resolution HD1200 "
                "--no-right --no-depth --no-pc --no-imu"
            ),
            "started_by_quickstart": settings.streamer_started_by_quickstart,
            "ssh_route": settings.streamer_ssh_route,
        },
        "board": {
            "name": board.name,
            "profile_file": board_snapshot.name,
            "profile_sha256": board.sha256,
        },
        "capture": {
            "accepted_samples": accepted,
            "observed_stream_frames": frame_index,
            "settings": {
                "max_samples": settings.max_samples,
                "min_corners": settings.min_corners,
                "min_coverage": settings.min_coverage,
                "min_sharpness": settings.min_sharpness,
                "min_grid_rows": settings.min_grid_rows,
                "min_grid_cols": settings.min_grid_cols,
                "min_board_bbox_fraction": settings.min_board_bbox_fraction,
                "min_diversity_distance": settings.min_diversity_distance,
                "cooldown_s": settings.cooldown_s,
                "detection_fps": settings.detection_fps,
            },
        },
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def capture_session(
    board: BoardProfile,
    client: ZedStreamClient,
    settings: CaptureSettings,
) -> Path:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV and NumPy are required for capture") from exc

    detector = CharucoDetector(board)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_dir = settings.output_root.expanduser().resolve() / f"{stamp}_head_left_HD1200"
    raw_dir = session_dir / "raw"
    preview_dir = session_dir / "previews"
    raw_dir.mkdir(parents=True, exist_ok=False)
    preview_dir.mkdir(parents=True, exist_ok=True)

    board_snapshot = session_dir / "board_profile.yaml"
    board_snapshot.write_text(yaml.safe_dump(board.data, sort_keys=False), encoding="utf-8")
    records_path = session_dir / "frames.jsonl"
    manifest_path = session_dir / "manifest.yaml"
    created_at_utc = _utc_now()

    signatures: list[np.ndarray] = []
    accepted = 0
    frame_index = 0
    last_capture_monotonic = -1e9
    last_detection_monotonic = -1e9
    identity: dict | None = None
    stop_requested = False
    cached_shown = None

    _write_manifest(
        manifest_path,
        created_at_utc=created_at_utc,
        status="in_progress",
        board=board,
        board_snapshot=board_snapshot,
        identity=identity,
        accepted=accepted,
        frame_index=frame_index,
        settings=settings,
        stream_host=client.host,
        stream_port=client.port,
    )

    with records_path.open("w", encoding="utf-8") as records, client:
        while accepted < settings.max_samples and not stop_requested:
            frame = client.receive()
            frame_index += 1
            assert frame.left_width is not None and frame.left_height is not None
            width, height = frame.left_width, frame.left_height
            if identity is None:
                identity = {
                    "protocol": frame.protocol,
                    "camera_serial": frame.serial_number,
                    "width": width,
                    "height": height,
                    "channel_mask": frame.channel_mask,
                }

            now = time.monotonic()
            if not detection_is_due(now, last_detection_monotonic, settings):
                if settings.preview and cached_shown is not None:
                    cv2.imshow("Dexmate HD1200 intrinsic capture", cached_shown)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        stop_requested = True
                continue

            last_detection_monotonic = now
            image = frame.decode_left_bgr()
            detailed_quality = now - last_capture_monotonic >= settings.cooldown_s
            detection = detector.detect(image, detailed_quality=detailed_quality)
            reason = _quality_reason(detection, settings)
            signature = detection.signature(width, height) if detection is not None else None
            if reason is None and signatures:
                nearest = min(float(np.linalg.norm(signature - old)) for old in signatures)
                if nearest < settings.min_diversity_distance:
                    reason = f"redundant<{settings.min_diversity_distance:.3f}"

            eligible = reason is None and now - last_capture_monotonic >= settings.cooldown_s
            manual_capture = False
            display_state = reason or (
                "ready"
                if now - last_capture_monotonic >= settings.cooldown_s
                else f"cooldown<{settings.cooldown_s:.1f}s"
            )
            display = render_capture_preview(
                detector,
                image,
                detection,
                accepted=accepted,
                target=settings.max_samples,
                state=display_state,
            )
            if settings.preview:
                scale = min(1.0, 1280.0 / width, 800.0 / height)
                shown = cv2.resize(display, None, fx=scale, fy=scale) if scale < 1.0 else display
                cached_shown = shown
                cv2.imshow("Dexmate HD1200 intrinsic capture", shown)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    stop_requested = True
                elif key == ord(" "):
                    manual_capture = True

            should_save = eligible and (settings.auto_capture or manual_capture)
            if should_save and frame.left_jpeg is not None and detection is not None:
                filename = f"frame_{accepted:04d}_{frame.source_timestamp_ns}.jpg"
                image_path = raw_dir / filename
                image_path.write_bytes(frame.left_jpeg)
                preview_path = preview_dir / f"frame_{accepted:04d}.jpg"
                cv2.imwrite(str(preview_path), display)
                record = {
                    "sample_index": accepted,
                    "stream_frame_index": frame_index,
                    "image": str(image_path.relative_to(session_dir)),
                    "preview": str(preview_path.relative_to(session_dir)),
                    "source_timestamp_ns": frame.source_timestamp_ns,
                    "receive_timestamp_ns": frame.receive_timestamp_ns,
                    "protocol": frame.protocol,
                    "camera_serial": frame.serial_number,
                    "width": width,
                    "height": height,
                    "channel_mask": frame.channel_mask,
                    "corner_count": detection.corner_count,
                    "marker_count": detection.marker_count,
                    "coverage_fraction": detection.coverage_fraction,
                    "sharpness": detection.sharpness,
                    "grid_rows": detection.grid_rows,
                    "grid_cols": detection.grid_cols,
                    "board_bbox_fraction": detection.board_bbox_fraction,
                    "pixels_per_square": detection.pixels_per_square,
                    "rectified_laplacian_var": detection.rectified_laplacian_var,
                    "rectified_tenengrad_mean": detection.rectified_tenengrad_mean,
                    "centroid_xy": list(detection.centroid_xy),
                    "orientation_rad": detection.orientation_rad,
                }
                records.write(json.dumps(record, separators=(",", ":")) + "\n")
                records.flush()
                signatures.append(signature)
                accepted += 1
                last_capture_monotonic = now
                _write_manifest(
                    manifest_path,
                    created_at_utc=created_at_utc,
                    status="in_progress",
                    board=board,
                    board_snapshot=board_snapshot,
                    identity=identity,
                    accepted=accepted,
                    frame_index=frame_index,
                    settings=settings,
                    stream_host=client.host,
                    stream_port=client.port,
                )

    if settings.preview:
        cv2.destroyAllWindows()
    _write_manifest(
        manifest_path,
        created_at_utc=created_at_utc,
        status="complete" if accepted >= settings.max_samples else "stopped_by_user",
        board=board,
        board_snapshot=board_snapshot,
        identity=identity,
        accepted=accepted,
        frame_index=frame_index,
        settings=settings,
        stream_host=client.host,
        stream_port=client.port,
    )
    return session_dir
