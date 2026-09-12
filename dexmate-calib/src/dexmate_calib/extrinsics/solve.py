"""Offline solve of a captured hand-eye session."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from dexmate_calib.boards.config import create_detector, load_board_profile
from dexmate_calib.extrinsics.capture import SESSION_SCHEMA
from dexmate_calib.extrinsics.handeye import (
    HandEyeSolution,
    HandEyeView,
    predicted_T_cam_board,
    project_board,
    solve_hand_eye,
)
from dexmate_calib.geometry.transforms import rotation_angle_deg


@dataclass
class HandEyeSessionResult:
    session_dir: Path
    results_dir: Path
    solution: HandEyeSolution
    robot_model: str
    base_frame: str
    target_link: str
    camera_serial: str
    T_base_depth: np.ndarray | None

    def __str__(self) -> str:
        X = self.solution.T_base_cam
        lines = [
            f"session: {self.session_dir}",
            f"robot: {self.robot_model}  base={self.base_frame}  link={self.target_link}",
            f"camera: {self.camera_serial}",
            f"views: {len(self.solution.inlier_views)} used, {len(self.solution.rejected_views)} rejected",
            f"method: {self.solution.refinement.get('method', 'reprojection')}",
            f"rms reprojection error: {self.solution.rms_px:.4f} px",
            f"init ({self.solution.initialisation['method']}): {self.solution.initialisation['rms_px']:.3f} px",
            "T_base_cam (color camera in robot base, metres):",
        ]
        for row in X:
            lines.append("  [" + ", ".join(f"{v: .6f}" for v in row) + "]")
        loo = self.solution.diagnostics.get("leave_one_out")
        if loo:
            lines.append(
                "leave-one-out spread: rot max {:.4f} deg, trans max {:.2f} mm, held-out rms p90 {:.3f} px".format(
                    loo["T_base_cam_rotation_spread_deg"]["max"],
                    loo["T_base_cam_translation_spread_mm"]["max"],
                    loo["held_out_view_rms_px"]["p90"],
                )
            )
        div = self.solution.diagnostics.get("motion_diversity", {})
        if div:
            lines.append(
                "motion diversity: max rotation {:.1f} deg, axis rank {}, translation span {:.2f} m".format(
                    div.get("max_pairwise_rotation_deg", 0.0),
                    div.get("rotation_axis_rank", 0),
                    div.get("translation_span_m", 0.0),
                )
            )
        lines.append(f"results: {self.results_dir}")
        return "\n".join(lines)


def _load_records(session_dir: Path) -> list[dict]:
    path = session_dir / "samples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_session_views(
    session_dir: Path,
    *,
    kinematics,
    target_link: str,
    min_corners: int = 12,
) -> tuple[list[HandEyeView], list[dict], dict, np.ndarray, np.ndarray, tuple[int, int]]:
    """Detect the board in every sample and attach FK poses.

    Returns ``(views, rejected, manifest, K, D, (width, height))``.
    """
    import cv2

    manifest = yaml.safe_load((session_dir / "manifest.yaml").read_text(encoding="utf-8"))
    if manifest.get("schema") != SESSION_SCHEMA:
        raise ValueError(f"Unexpected session schema {manifest.get('schema')!r}")
    calib = json.loads(
        (session_dir / manifest["camera"]["calibration_file"]).read_text(encoding="utf-8")
    )
    K = np.asarray(calib["color"]["camera_matrix"], dtype=np.float64)
    D = np.asarray(calib["color"]["distortion_coefficients"], dtype=np.float64)
    size = (int(calib["color"]["width"]), int(calib["color"]["height"]))
    board = load_board_profile(session_dir / manifest["board"]["profile_file"])
    if board.sha256 != manifest["board"]["profile_sha256"]:
        raise ValueError("Board profile snapshot hash does not match the manifest")
    detector = create_detector(board)
    min_corners = min(min_corners, board.expected_corner_count)
    required = set(kinematics.required_joints_for(target_link))

    views: list[HandEyeView] = []
    rejected: list[dict] = []
    for record in _load_records(session_dir):
        name = str(record["color_image"])
        joints = record.get("joints")
        if not joints or not joints.get("positions_rad"):
            rejected.append({"view": name, "reason": "no_joint_positions"})
            continue
        positions = {str(k): float(v) for k, v in joints["positions_rad"].items()}
        missing = required - set(positions)
        if missing:
            rejected.append({"view": name, "reason": f"missing_joints:{sorted(missing)}"})
            continue
        image = cv2.imread(str(session_dir / name), cv2.IMREAD_COLOR)
        if image is None or image.shape[1::-1] != size:
            rejected.append({"view": name, "reason": "unreadable_or_wrong_size"})
            continue
        detection = detector.detect(image, detailed_quality=False)
        if detection is None or detection.corner_count < min_corners:
            rejected.append({"view": name, "reason": "insufficient_corners"})
            continue
        object_points, image_points = detector.calibration_points(detection)
        T_base_link = kinematics.frame_pose(positions, target_link, strict=False)
        views.append(HandEyeView(name, object_points, image_points, T_base_link))
    return views, rejected, manifest, K, D, size


def _write_contact_sheet(
    cv2,
    session_dir: Path,
    views: list[HandEyeView],
    solution: HandEyeSolution,
    K: np.ndarray,
    D: np.ndarray,
    output: Path,
    *,
    tile: int = 480,
    columns: int = 5,
) -> None:
    inliers = set(solution.inlier_views)
    tiles = []
    for view in views:
        image = cv2.imread(str(session_dir / view.name), cv2.IMREAD_COLOR)
        if image is None:
            continue
        T_cam_board = predicted_T_cam_board(
            solution.T_base_cam, view.T_base_link, solution.T_link_board
        )
        projected = project_board(view.object_points, T_cam_board, K, D)
        for obs, pred in zip(view.image_points, projected):
            o = tuple(round(v) for v in obs)
            p = tuple(round(v) for v in pred)
            cv2.circle(image, o, 6, (0, 255, 0), 2)
            cv2.circle(image, p, 4, (255, 0, 255), -1)
            cv2.line(image, o, p, (0, 255, 255), 2)
        h, w = image.shape[:2]
        scale = tile / max(h, w)
        thumb = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((tile, tile, 3), dtype=np.uint8)
        canvas[: thumb.shape[0], : thumb.shape[1]] = thumb
        report = next((r for r in solution.view_reports if r["view"] == view.name), None)
        label = view.name.split("/")[-1]
        if report:
            label += f" rms={report['rms_px']:.2f}px"
        if view.name not in inliers:
            label += " REJECTED"
            cv2.rectangle(canvas, (0, 0), (tile - 1, tile - 1), (0, 0, 255), 6)
        cv2.putText(
            canvas, label, (8, tile - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            label,
            (8, tile - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(canvas)
    if not tiles:
        return
    rows = math.ceil(len(tiles) / columns)
    sheet = np.zeros((rows * tile, columns * tile, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, columns)
        sheet[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = t
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def solve_handeye_session(
    session: str | Path,
    *,
    robot_model: str | None = None,
    base_frame: str | None = None,
    target_link: str | None = None,
    min_corners: int = 12,
    min_views: int = 8,
    huber_px: float = 2.0,
    max_view_rms_px: float = 3.0,
    leave_one_out: bool = True,
    kinematics=None,
    method: str = "reprojection",
) -> HandEyeSessionResult:
    """Solve ``T_base_cam`` for a captured session and write ``results/``.

    ``robot_model`` / ``base_frame`` / ``target_link`` default to the values recorded
    in the manifest; pass overrides to re-solve with a corrected robot model.
    ``kinematics`` may inject a pre-built :class:`FrameKinematics` (tests).
    """
    import cv2

    session_dir = Path(session).expanduser().resolve()
    manifest = yaml.safe_load((session_dir / "manifest.yaml").read_text(encoding="utf-8"))
    robot_cfg = manifest.get("robot", {})
    robot_model = robot_model or robot_cfg.get("model", "vega_1p")
    base_frame = base_frame or robot_cfg.get("base_frame", "base")
    target_link = target_link or robot_cfg.get("target_link", "L_ee")
    if kinematics is None:
        from dexmate_calib.robot.kinematics import FrameKinematics

        kinematics = FrameKinematics.from_model(robot_model, base_frame=base_frame)

    views, rejected_pre, manifest, K, D, size = load_session_views(
        session_dir, kinematics=kinematics, target_link=target_link, min_corners=min_corners
    )
    if len(views) < min_views:
        raise ValueError(
            f"Only {len(views)} usable views after detection/FK (need {min_views}); "
            f"rejected: {rejected_pre}"
        )
    solution = solve_hand_eye(
        views,
        K,
        D,
        huber_px=huber_px,
        max_view_rms_px=max_view_rms_px,
        min_views=min_views,
        leave_one_out=leave_one_out,
        method=method,
    )
    solution.rejected_views = rejected_pre + solution.rejected_views

    calib = json.loads(
        (session_dir / manifest["camera"]["calibration_file"]).read_text(encoding="utf-8")
    )
    T_color_depth = calib.get("T_color_depth")
    T_base_depth = None
    if T_color_depth is not None:
        T_base_depth = solution.T_base_cam @ np.asarray(T_color_depth, dtype=np.float64)

    results_dir = session_dir / "results"
    results_dir.mkdir(exist_ok=True)
    camera_serial = str(manifest["camera"].get("serial", "unknown"))
    solved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    X = solution.T_base_cam
    summary = {
        "schema": "dexmate_calib.handeye_result.v1",
        "solved_at_utc": solved_at,
        "calibration_type": "eye_to_hand",
        "method": method,
        "robot": {"model": robot_model, "base_frame": base_frame, "target_link": target_link},
        "camera": {
            "name": manifest["camera"].get("name"),
            "serial": camera_serial,
            "color_resolution": manifest["camera"].get("color_resolution"),
            "width": size[0],
            "height": size[1],
            "camera_matrix": K.tolist(),
            "distortion_coefficients": D.tolist(),
        },
        "board": manifest.get("board"),
        "T_base_cam": X.tolist(),
        "T_base_cam_frame_note": "pose of the Kinect COLOR camera optical frame in the robot base frame; "
        "x right, y down, z forward (OpenCV convention)",
        "T_link_board": solution.T_link_board.tolist(),
        "T_base_depth": None if T_base_depth is None else T_base_depth.tolist(),
        "T_color_depth": T_color_depth,
        "rms_reprojection_error_px": solution.rms_px,
        "views_used": len(solution.inlier_views),
        "views_rejected": len(solution.rejected_views),
        "initialisation": solution.initialisation,
        "refinement": solution.refinement,
        "diagnostics": solution.diagnostics,
        "rejected_views": solution.rejected_views,
    }
    suffix = "" if method == "reprojection" else f"_{method}"
    (results_dir / f"handeye_result{suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    yaml_out = {
        "camera_serial": camera_serial,
        "camera_name": manifest["camera"].get("name"),
        "robot_model": robot_model,
        "base_frame": base_frame,
        "resolution": [size[0], size[1]],
        "K": K.tolist(),
        "distortion_coefficients": D.tolist(),
        "T_base_cam": X.tolist(),
        "T_base_depth": None if T_base_depth is None else T_base_depth.tolist(),
        "rms_reprojection_error_px": round(solution.rms_px, 5),
        "views_used": len(solution.inlier_views),
        "method": method,
        "solved_at_utc": solved_at,
    }
    (results_dir / f"T_base_{manifest['camera'].get('name', 'cam')}.yaml").write_text(
        yaml.safe_dump(yaml_out, sort_keys=False), encoding="utf-8"
    )
    np.save(results_dir / "T_base_cam.npy", X)
    with (results_dir / "per_view.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = [
            "view",
            "status",
            "corners",
            "rms_px",
            "max_px",
            "pnp_rotation_gap_deg",
            "pnp_translation_gap_mm",
            "board_distance_m",
            "reason",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for report in solution.view_reports:
            writer.writerow(
                {"status": "used", **{k: report.get(k, "") for k in fields if k != "status"}}
            )
        for rej in solution.rejected_views:
            writer.writerow(
                {
                    "view": rej["view"],
                    "status": "rejected",
                    "reason": rej.get("reason", ""),
                    "rms_px": rej.get("rms_px", ""),
                }
            )
    _write_contact_sheet(
        cv2, session_dir, views, solution, K, D, results_dir / "reprojection_contact_sheet.jpg"
    )

    # Sanity: X must be rigid.
    if (
        abs(np.linalg.det(X[:3, :3]) - 1.0) > 1e-6
        or rotation_angle_deg(X[:3, :3] @ X[:3, :3].T) > 1e-4
    ):
        raise RuntimeError("Solved T_base_cam is not a proper rigid transform")

    return HandEyeSessionResult(
        session_dir=session_dir,
        results_dir=results_dir,
        solution=solution,
        robot_model=robot_model,
        base_frame=base_frame,
        target_link=target_link,
        camera_serial=camera_serial,
        T_base_depth=T_base_depth,
    )
