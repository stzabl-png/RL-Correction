"""Manual capture of eye-to-hand calibration samples (Kinect + Dexmate joints).

The operator moves the robot with an external teleop (dexcontrol's keyboard example)
and presses SPACE in the preview window when the board is still and well framed.
Every accepted sample stores the raw color image, the depth image, all joint
positions read around the exposure and the board detection statistics.  Forward
kinematics is *not* evaluated here; it happens at solve time so sessions can be
re-solved with a corrected robot model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from dexmate_calib.boards.config import AnyBoardProfile, create_detector
from dexmate_calib.intrinsics.detector import Detection

SESSION_SCHEMA = "dexmate_calib.handeye_session.v1"


class FrameSource(Protocol):
    def capture(self) -> Any: ...

    @property
    def calibration(self) -> Any: ...


class JointSource(Protocol):
    def read_stationary(self) -> Any: ...


@dataclass(frozen=True)
class HandEyeCaptureSettings:
    output_root: Path
    robot_model: str = "vega_1p"
    base_frame: str = "base"
    target_link: str = "L_ee"
    camera_name: str = "kinect_external"
    max_samples: int = 30
    min_corners: int = 20
    min_grid_rows: int = 3
    min_grid_cols: int = 3
    require_stationary: bool = True
    preview: bool = True
    preview_scale: float = 0.5
    auto_capture: bool = False
    auto_capture_interval_s: float = 1.0
    save_depth: bool = True


def clamp_settings_to_target(
    settings: HandEyeCaptureSettings, board: AnyBoardProfile
) -> HandEyeCaptureSettings:
    """Lower the quality gates to what the target can physically provide.

    The defaults are written for the 10x7 ChArUco board (54 corners); a 4x4 AprilTag grid
    has 64 corners but only a 4x4 tag lattice, and a single tag has 4 corners in a 1x1
    lattice.  Gates above those maxima would reject every frame.
    """
    from dataclasses import replace

    min_corners = min(settings.min_corners, board.expected_corner_count)
    rows = settings.min_grid_rows
    cols = settings.min_grid_cols
    if board.target_type == "apriltag_grid":
        rows = min(rows, board.rows)
        cols = min(cols, board.cols)
    if (min_corners, rows, cols) == (
        settings.min_corners,
        settings.min_grid_rows,
        settings.min_grid_cols,
    ):
        return settings
    return replace(settings, min_corners=min_corners, min_grid_rows=rows, min_grid_cols=cols)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _quality_reason(detection: Detection | None, settings: HandEyeCaptureSettings) -> str | None:
    if detection is None:
        return "no_board"
    if detection.corner_count < settings.min_corners:
        return f"corners<{settings.min_corners}"
    if detection.grid_rows < settings.min_grid_rows or detection.grid_cols < settings.min_grid_cols:
        return "corner_grid_too_small"
    return None


def _detection_stats(detection: Detection) -> dict:
    return {
        "marker_count": int(detection.marker_count),
        "corner_count": int(detection.corner_count),
        "coverage_fraction": float(detection.coverage_fraction),
        "sharpness": float(detection.sharpness),
        "grid_rows": int(detection.grid_rows),
        "grid_cols": int(detection.grid_cols),
        "board_bbox_fraction": float(detection.board_bbox_fraction),
        "pixels_per_square": float(detection.pixels_per_square),
    }


def render_handeye_preview(
    cv2,
    image_bgr: np.ndarray,
    detector,
    detection: Detection | None,
    *,
    accepted: int,
    max_samples: int,
    reason: str | None,
    robot_status: str,
    scale: float,
    extra_lines: list[str] | None = None,
) -> np.ndarray:
    frame = detector.draw(image_bgr, detection)
    if scale != 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lines = [
        f"samples {accepted}/{max_samples}   SPACE=save  q/Esc=quit",
        f"board: {'OK' if reason is None else reason}"
        + (
            f"  corners={detection.corner_count} grid={detection.grid_rows}x{detection.grid_cols}"
            if detection
            else ""
        ),
        f"robot: {robot_status}",
        *(extra_lines or []),
    ]
    for i, text in enumerate(lines):
        y = 28 + 26 * i
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        color = (0, 200, 0) if (i != 1 or reason is None) else (0, 0, 255)
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return frame


def _write_manifest(
    path: Path,
    *,
    created_at_utc: str,
    status: str,
    board: AnyBoardProfile,
    board_snapshot: Path,
    camera_calibration: dict,
    robot_info: dict,
    settings: HandEyeCaptureSettings,
    accepted: int,
) -> None:
    manifest = {
        "schema": SESSION_SCHEMA,
        "created_at_utc": created_at_utc,
        "updated_at_utc": _utc_now(),
        "status": status,
        "calibration_type": "eye_to_hand",
        "camera": {
            "name": settings.camera_name,
            "vendor": camera_calibration.get("vendor"),
            "serial": camera_calibration.get("serial"),
            "color_resolution": camera_calibration.get("color_resolution"),
            "depth_mode": camera_calibration.get("depth_mode"),
            "width": camera_calibration["color"]["width"],
            "height": camera_calibration["color"]["height"],
            "calibration_file": "camera_calibration.json",
        },
        "robot": {
            "model": settings.robot_model,
            "base_frame": settings.base_frame,
            "target_link": settings.target_link,
            **robot_info,
        },
        "board": {
            "name": board.name,
            "target_type": board.target_type,
            "profile_file": board_snapshot.name,
            "profile_sha256": board.sha256,
            "mounted_on": settings.target_link,
        },
        "capture": {
            "accepted_samples": accepted,
            "mode": "auto" if settings.auto_capture else "manual",
            "settings": {
                "max_samples": settings.max_samples,
                "min_corners": settings.min_corners,
                "min_grid_rows": settings.min_grid_rows,
                "min_grid_cols": settings.min_grid_cols,
                "require_stationary": settings.require_stationary,
                "save_depth": settings.save_depth,
            },
        },
    }
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")


def capture_handeye_session(
    board: AnyBoardProfile,
    camera: FrameSource,
    joints: JointSource | None,
    settings: HandEyeCaptureSettings,
    *,
    key_source: Callable[[int], int] | None = None,
    key_handler: Callable[[int], bool] | None = None,
    status_lines: Callable[[], list[str]] | None = None,
    save_guard: Callable[[], str | None] | None = None,
) -> Path:
    """Run the interactive capture loop and return the session directory.

    ``joints`` may be ``None`` for camera-only dry runs; such sessions cannot be solved.
    ``key_source`` overrides ``cv2.waitKey`` (used by tests and headless runs).
    ``key_handler`` receives every key the loop does not use itself (e.g. teleop);
    ``status_lines`` adds preview overlay lines; ``save_guard`` may veto a save by
    returning a reason string.
    """
    import cv2

    detector = create_detector(board)
    settings = clamp_settings_to_target(settings, board)
    calibration = camera.calibration.as_dict()
    created = _utc_now()
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    session_dir = (
        settings.output_root / f"{stamp}_handeye_{settings.camera_name}_{settings.target_link}"
    )
    (session_dir / "color").mkdir(parents=True, exist_ok=False)
    if settings.save_depth:
        (session_dir / "depth").mkdir()
    board_snapshot = session_dir / "board_profile.yaml"
    board_snapshot.write_text(yaml.safe_dump(board.data, sort_keys=False), encoding="utf-8")
    (session_dir / "camera_calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    robot_info = {"joint_source": "dexcontrol" if joints is not None else "none"}
    robot_name = getattr(joints, "robot_name", None) if joints is not None else None
    if robot_name:
        robot_info["robot_name"] = str(robot_name)
    manifest_path = session_dir / "manifest.yaml"
    samples_path = session_dir / "samples.jsonl"

    def write_manifest(status: str, accepted: int) -> None:
        _write_manifest(
            manifest_path,
            created_at_utc=created,
            status=status,
            board=board,
            board_snapshot=board_snapshot,
            camera_calibration=calibration,
            robot_info=robot_info,
            settings=settings,
            accepted=accepted,
        )

    write_manifest("capturing", 0)
    window = f"dexcalib handeye [{settings.camera_name}]"
    if settings.preview:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    wait_key = key_source or (lambda ms: cv2.waitKey(ms) & 0xFF)

    accepted = 0
    last_auto = 0.0
    idle_frames = 0
    # Headless auto-capture (tests / scripted runs) must not spin forever on a target that
    # never passes the gates.
    max_idle_frames = (
        0 if settings.preview or not settings.auto_capture else 200 * max(1, settings.max_samples)
    )
    robot_status = "n/a (no joint source)" if joints is None else "not sampled yet"
    status = "aborted"
    try:
        with samples_path.open("a", encoding="utf-8") as records:
            while accepted < settings.max_samples:
                frame = camera.capture()
                image = frame.color_bgr
                detection = detector.detect(image, detailed_quality=False)
                reason = _quality_reason(detection, settings)
                if max_idle_frames:
                    idle_frames += 1
                    if idle_frames > max_idle_frames:
                        raise RuntimeError(
                            f"auto-capture made no progress in {max_idle_frames} frames "
                            f"(last reason: {reason}); check the target and quality gates"
                        )
                key = wait_key(1)
                if key in (27, ord("q")):
                    status = "stopped_by_user"
                    break
                trigger = key == 32
                if not trigger and key >= 0 and key_handler is not None:
                    key_handler(key)
                if settings.auto_capture and reason is None:
                    now = time.monotonic()
                    if now - last_auto >= settings.auto_capture_interval_s:
                        trigger = True
                        last_auto = now
                if trigger:
                    veto = save_guard() if save_guard is not None else None
                    if reason is not None:
                        robot_status = f"rejected: {reason}"
                    elif veto is not None:
                        robot_status = f"rejected: {veto}"
                    else:
                        joint_sample = joints.read_stationary() if joints is not None else None
                        # Re-grab the frame after confirming the robot is still, so image and
                        # joints bracket the same stationary interval.
                        frame = camera.capture()
                        image = frame.color_bgr
                        detection = detector.detect(image, detailed_quality=True)
                        reason = _quality_reason(detection, settings)
                        stationary_ok = (
                            joint_sample is None
                            or joint_sample.stationary
                            or not settings.require_stationary
                        )
                        if reason is not None:
                            robot_status = f"rejected after re-grab: {reason}"
                        elif not stationary_ok:
                            robot_status = (
                                f"rejected: robot moving ({joint_sample.max_motion_rad:.4f} rad)"
                            )
                        else:
                            record = _save_sample(
                                cv2,
                                session_dir,
                                accepted,
                                frame,
                                detection,
                                joint_sample,
                                settings,
                            )
                            records.write(json.dumps(record, separators=(",", ":")) + "\n")
                            records.flush()
                            accepted += 1
                            idle_frames = 0
                            write_manifest("capturing", accepted)
                            robot_status = f"saved #{accepted - 1}" + (
                                f" (motion {joint_sample.max_motion_rad:.4f} rad)"
                                if joint_sample is not None
                                else ""
                            )
                if settings.preview:
                    preview = render_handeye_preview(
                        cv2,
                        image,
                        detector,
                        detection,
                        accepted=accepted,
                        max_samples=settings.max_samples,
                        reason=reason,
                        robot_status=robot_status,
                        scale=settings.preview_scale,
                        extra_lines=status_lines() if status_lines is not None else None,
                    )
                    cv2.imshow(window, preview)
            else:
                status = "completed"
    finally:
        if settings.preview:
            cv2.destroyWindow(window)
        write_manifest(status, accepted)
    return session_dir


def _save_sample(
    cv2,
    session_dir: Path,
    index: int,
    frame,
    detection: Detection,
    joint_sample,
    settings: HandEyeCaptureSettings,
) -> dict:
    color_name = f"color/{index:04d}.png"
    if not cv2.imwrite(
        str(session_dir / color_name), frame.color_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1]
    ):
        raise OSError(f"Failed to write {color_name}")
    depth_name = None
    if settings.save_depth and getattr(frame, "depth_mm", None) is not None:
        depth_name = f"depth/{index:04d}.png"
        if not cv2.imwrite(str(session_dir / depth_name), frame.depth_mm):
            raise OSError(f"Failed to write {depth_name}")
    record = {
        "sample_index": index,
        "color_image": color_name,
        "depth_image": depth_name,
        "color_timestamp_us": getattr(frame, "color_timestamp_us", None),
        "depth_timestamp_us": getattr(frame, "depth_timestamp_us", None),
        "receive_time_s": getattr(frame, "receive_time_s", None),
        "detection": _detection_stats(detection),
        "joints": None,
    }
    if joint_sample is not None:
        record["joints"] = {
            "positions_rad": joint_sample.positions,
            "stationary": bool(joint_sample.stationary),
            "max_motion_rad": float(joint_sample.max_motion_rad),
            "read_time_s": float(joint_sample.read_time_s),
            "checks": int(joint_sample.checks),
        }
    return record
