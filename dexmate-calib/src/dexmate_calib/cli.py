from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml

from dexmate_calib.boards.config import (
    available_board_profiles,
    create_detector,
    load_board_profile,
    require_charuco,
    resolve_board_profile,
)
from dexmate_calib.diagnostics.doctor import doctor_network, doctor_stream, print_report
from dexmate_calib.extrinsics.config import load_handeye_config
from dexmate_calib.intrinsics.capture import CaptureSettings, capture_session
from dexmate_calib.intrinsics.multi_solve import solve_sessions
from dexmate_calib.intrinsics.solve import solve_session
from dexmate_calib.remote.streamer import RemoteStreamerManager
from dexmate_calib.streaming.zed_stream import ZedStreamClient


def _optional_serial(value: str) -> int | None:
    parsed = int(value)
    return None if parsed == 0 else parsed


def _cmd_board(args: argparse.Namespace) -> int:
    if args.board_command == "list":
        for profile in available_board_profiles():
            aliases = ", ".join(profile.data.get("aliases", [])) or "-"
            print(
                f"{profile.name}\t{profile.target_type}\taliases={aliases}"
                f"\tsha256={profile.sha256[:12]}"
            )
        return 0
    if args.board_command == "show":
        profile = resolve_board_profile(args.board)
        print(yaml.safe_dump(profile.data, sort_keys=False), end="")
        print(f"sha256: {profile.sha256}")
        return 0
    if args.board_command == "validate":
        profile = load_board_profile(args.path)
        create_detector(profile)
        print(f"OK: {profile.name} [{profile.target_type}] ({profile.sha256})")
        return 0
    if args.board_command == "verify":
        import cv2

        profile = resolve_board_profile(args.board)
        image = cv2.imread(str(Path(args.image).expanduser()), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
        detection = create_detector(profile).detect(image)
        result = {
            "ok": detection is not None,
            "board": profile.name,
            "target_type": profile.target_type,
            "expected_markers": profile.expected_marker_count,
            "expected_corners": profile.expected_corner_count,
            "detected_markers": detection.marker_count if detection else 0,
            "detected_corners": detection.corner_count if detection else 0,
        }
        print(json.dumps(result, indent=2))
        return 0 if detection is not None else 2
    if args.board_command == "identify":
        import cv2

        from dexmate_calib.boards.apriltag import identify_markers

        image = cv2.imread(str(Path(args.image).expanduser()), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {args.image}")
        found = identify_markers(image)
        print(json.dumps({"image": str(args.image), "markers": found}, indent=2))
        return 0 if found else 2
    if args.board_command == "render":
        import cv2

        profile = resolve_board_profile(args.board)
        if not hasattr(profile, "render_image"):
            raise ValueError("render currently supports AprilTag grid profiles only")
        pixels_per_m = args.dpi / 0.0254
        image, _ = profile.render_image(pixels_per_m=pixels_per_m, margin_m=args.margin_mm / 1000.0)
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), image):
            raise OSError(f"Failed to write {out}")
        print(
            f"wrote {out}: {image.shape[1]}x{image.shape[0]} px at {args.dpi} dpi "
            f"({image.shape[1] / pixels_per_m * 1000:.1f} x {image.shape[0] / pixels_per_m * 1000:.1f} mm; "
            "print at 100% and measure a tag edge)"
        )
        return 0
    raise AssertionError(args.board_command)


def _cmd_doctor(args: argparse.Namespace) -> int:
    if args.doctor_command == "network":
        report = doctor_network()
        print_report(report)
        nano_ssh_ok = report["nano_ssh_direct"]["ok"] or report["nano_ssh_proxy"]["ok"]
        return 0 if nano_ssh_ok else 2
    report = doctor_stream(
        args.host,
        args.port,
        frames=args.frames,
        expected_serial=args.expected_serial,
        expected_width=args.width,
        expected_height=args.height,
    )
    print_report(report)
    return 0


def _cmd_intrinsics(args: argparse.Namespace) -> int:
    if args.intrinsics_command in {"capture", "quickstart"}:
        if args.manual and args.no_preview:
            raise ValueError("--manual requires the preview window; remove --no-preview")
        if args.detect_fps < 0:
            raise ValueError("--detect-fps must be >= 0")
        board = require_charuco(resolve_board_profile(args.board), "intrinsics capture")
        client = ZedStreamClient(
            args.host,
            args.port,
            expected_serial=args.expected_serial,
            expected_width=1920,
            expected_height=1200,
        )
        settings = CaptureSettings(
            output_root=Path(args.output),
            max_samples=args.samples,
            min_corners=args.min_corners,
            min_coverage=args.min_coverage,
            min_sharpness=args.min_sharpness,
            min_grid_rows=args.min_grid_rows,
            min_grid_cols=args.min_grid_cols,
            min_board_bbox_fraction=args.min_board_bbox_fraction,
            min_diversity_distance=args.min_diversity,
            cooldown_s=args.cooldown,
            detection_fps=args.detect_fps,
            preview=not args.no_preview,
            auto_capture=not args.manual,
        )
        if args.intrinsics_command == "capture":
            session = capture_session(board, client, settings)
            print(session)
            return 0

        manager = RemoteStreamerManager(
            route_preference=args.ssh_route,
            use_sudo=not args.no_sudo,
        )
        started_here = False
        try:
            started_here = manager.ensure_started(
                args.host,
                args.port,
                timeout_s=args.startup_timeout,
            )
            stream_report = doctor_stream(
                args.host,
                args.port,
                frames=args.preflight_frames,
                expected_serial=args.expected_serial,
                expected_width=1920,
                expected_height=1200,
            )
            print("Stream preflight passed:")
            print_report(stream_report)
            settings = replace(
                settings,
                streamer_started_by_quickstart=started_here,
                streamer_ssh_route=manager.select_route().name,
            )
            session = capture_session(board, client, settings)
        finally:
            if manager.process is not None:
                manager.stop_attached(host=args.host, port=args.port)
        print(session)
        if not args.no_solve:
            result = solve_session(session)
            print(result)
        return 0
    if args.intrinsics_command == "solve":
        result = solve_session(
            args.session,
            max_view_error_px=args.max_view_error,
            min_views=args.min_views,
            max_views=args.max_views,
            cross_validation_folds=args.cross_validation_folds,
        )
    else:
        result = solve_sessions(
            args.sessions,
            output=args.output,
            max_view_error_px=args.max_view_error,
            min_views=args.min_views,
            min_views_per_session=args.min_views_per_session,
            max_views_per_session=args.max_views_per_session,
            cross_validation_folds=args.cross_validation_folds,
        )
    print(result)
    return 0


def _cmd_kinect(args: argparse.Namespace) -> int:
    from dexmate_calib.cameras.kinect import KinectCamera, kinect_available

    if not kinect_available():
        raise RuntimeError("pyk4a is not installed; run `uv sync --extra kinect`")
    cfg = load_handeye_config(args.config)["camera"]
    with KinectCamera(
        color_resolution=args.color_resolution or cfg["color_resolution"],
        depth_mode=args.depth_mode or cfg["depth_mode"],
        fps=args.fps or int(cfg.get("fps", 15)),
        expected_serial=None if args.any_serial else str(cfg.get("expected_serial") or "") or None,
    ) as camera:
        info = camera.calibration.as_dict()
        if args.kinect_command == "info":
            print(json.dumps(info, indent=2))
            return 0
        import cv2

        frame = camera.capture()
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), frame.color_bgr)
        print(
            f"wrote {out} ({frame.color_bgr.shape[1]}x{frame.color_bgr.shape[0]}), serial {info['serial']}"
        )
        if frame.depth_mm is not None:
            depth_out = out.with_name(out.stem + "_depth.png")
            cv2.imwrite(str(depth_out), frame.depth_mm)
            print(f"wrote {depth_out}")
        return 0


def _cmd_extrinsics(args: argparse.Namespace) -> int:
    cfg = load_handeye_config(args.config)
    if args.extrinsics_command == "selftest":
        from dexmate_calib.extrinsics.handeye import solve_hand_eye
        from dexmate_calib.extrinsics.synthetic import make_scene
        from dexmate_calib.geometry.transforms import pose_error

        scene = make_scene(
            views=args.views,
            pixel_noise_px=args.pixel_noise,
            fk_rotation_noise_deg=args.fk_rotation_noise,
            fk_translation_noise_m=args.fk_translation_noise,
            outliers=args.outliers,
            seed=args.seed,
        )
        solution = solve_hand_eye(scene.views, scene.camera_matrix, scene.dist_coeffs)
        rot_deg, trans_m = pose_error(solution.T_base_cam, scene.T_base_cam)
        print(
            json.dumps(
                {
                    "views": args.views,
                    "rms_px": solution.rms_px,
                    "T_base_cam_error_deg": rot_deg,
                    "T_base_cam_error_mm": trans_m * 1000.0,
                    "rejected": [r["view"] for r in solution.rejected_views],
                    "expected_outliers": scene.outlier_views,
                    "initialisation_rms_px": solution.initialisation["rms_px"],
                    "leave_one_out": solution.diagnostics.get("leave_one_out"),
                },
                indent=2,
            )
        )
        return 0
    if args.extrinsics_command == "solve":
        from dexmate_calib.extrinsics.solve import solve_handeye_session

        solve_cfg = cfg["solve"]
        result = solve_handeye_session(
            args.session,
            robot_model=args.robot_model,
            base_frame=args.base_frame,
            target_link=args.target_link,
            min_corners=args.min_corners
            if args.min_corners is not None
            else int(solve_cfg["min_corners"]),
            min_views=args.min_views if args.min_views is not None else int(solve_cfg["min_views"]),
            huber_px=args.huber if args.huber is not None else float(solve_cfg["huber_px"]),
            max_view_rms_px=args.max_view_rms
            if args.max_view_rms is not None
            else float(solve_cfg["max_view_rms_px"]),
            leave_one_out=not args.no_leave_one_out and bool(solve_cfg.get("leave_one_out", True)),
            method=args.method or str(solve_cfg.get("method", "reprojection")),
        )
        print(result)
        if args.compare:
            from dexmate_calib.geometry.transforms import pose_error

            primary = result.solution.refinement.get("method", "reprojection")
            other_method = "robotcamcalib" if primary == "reprojection" else "reprojection"
            other = solve_handeye_session(
                args.session,
                robot_model=args.robot_model,
                base_frame=args.base_frame,
                target_link=args.target_link,
                min_corners=args.min_corners
                if args.min_corners is not None
                else int(solve_cfg["min_corners"]),
                min_views=args.min_views
                if args.min_views is not None
                else int(solve_cfg["min_views"]),
                huber_px=args.huber if args.huber is not None else float(solve_cfg["huber_px"]),
                max_view_rms_px=args.max_view_rms
                if args.max_view_rms is not None
                else float(solve_cfg["max_view_rms_px"]),
                leave_one_out=not args.no_leave_one_out
                and bool(solve_cfg.get("leave_one_out", True)),
                method=other_method,
            )
            print("\n--- comparison ---")
            print(other)
            rot_deg, trans_m = pose_error(result.solution.T_base_cam, other.solution.T_base_cam)
            print(
                f"\nT_base_cam difference ({primary} vs {other_method}): "
                f"{rot_deg:.4f} deg, {trans_m * 1000:.2f} mm"
            )
        return 0
    if args.extrinsics_command == "capture":
        from dexmate_calib.cameras.kinect import KinectCamera
        from dexmate_calib.extrinsics.capture import HandEyeCaptureSettings, capture_handeye_session

        cam_cfg, robot_cfg, cap_cfg = cfg["camera"], cfg["robot"], cfg["capture"]
        board = resolve_board_profile(args.board or cfg["board"]["profile"])
        target_link = args.target_link or robot_cfg["target_link"]
        # Gates are clamped to the target's capabilities inside capture_handeye_session.
        settings = HandEyeCaptureSettings(
            output_root=Path(args.output),
            robot_model=args.robot_model or robot_cfg["model"],
            base_frame=args.base_frame or robot_cfg["base_frame"],
            target_link=target_link,
            camera_name=cam_cfg.get("name", "kinect_external"),
            max_samples=args.samples if args.samples is not None else int(cap_cfg["samples"]),
            min_corners=args.min_corners
            if args.min_corners is not None
            else int(cap_cfg["min_corners"]),
            min_grid_rows=int(cap_cfg.get("min_grid_rows", 3)),
            min_grid_cols=int(cap_cfg.get("min_grid_cols", 3)),
            require_stationary=not args.allow_moving
            and bool(cap_cfg.get("require_stationary", True)),
            preview=True,
            preview_scale=args.preview_scale,
            auto_capture=False,
            save_depth=not args.no_depth,
        )
        if args.no_robot:
            if args.teleop:
                raise ValueError("--teleop needs the robot; remove --no-robot")
            joints_ctx = None
        else:
            from dexmate_calib.robot.dexmate import DexmateJointReader
            from dexmate_calib.robot.kinematics import FrameKinematics

            # Fail early if the URDF/link combination is invalid; FK itself runs at solve time.
            fk = FrameKinematics.from_model(settings.robot_model, base_frame=settings.base_frame)
            if not fk.has_frame(target_link):
                raise ValueError(f"Link {target_link!r} not in {settings.robot_model} URDF")
            joints_ctx = DexmateJointReader(
                components=tuple(
                    robot_cfg.get("joint_components", ["torso", "left_arm", "right_arm", "head"])
                ),
                settle_checks=int(cap_cfg.get("settle_checks", 3)),
                settle_interval_s=float(cap_cfg.get("settle_interval_s", 0.15)),
                settle_tolerance_rad=float(cap_cfg.get("settle_tolerance_rad", 0.002)),
            )
        camera = KinectCamera(
            color_resolution=args.color_resolution or cam_cfg["color_resolution"],
            depth_mode="OFF" if args.no_depth else (args.depth_mode or cam_cfg["depth_mode"]),
            fps=args.fps or int(cam_cfg.get("fps", 15)),
            expected_serial=None
            if args.any_serial
            else str(cam_cfg.get("expected_serial") or "") or None,
        )
        with camera:
            if joints_ctx is None:
                session = capture_handeye_session(board, camera, None, settings)
            else:
                with joints_ctx as joints:
                    hooks: dict = {}
                    if args.teleop:
                        from dexmate_calib.robot.teleop import (
                            KeyboardJointTeleop,
                            arm_component_for_link,
                        )

                        arm = arm_component_for_link(target_link)
                        teleop_components = tuple(
                            c for c in (arm, "torso") if c in joints_ctx.components
                        )
                        teleop = KeyboardJointTeleop(
                            joints,
                            teleop_components,
                            step_deg=args.step_deg,
                            velocity_scale=args.velocity_scale,
                        )
                        settle = float(cap_cfg.get("teleop_settle_s", 1.0))

                        def guard(_teleop=teleop, _settle=settle):
                            age = _teleop.seconds_since_command()
                            if age < _settle:
                                return f"wait {_settle - age:.1f}s after last motion command"
                            return None

                        hooks = {
                            "key_handler": teleop.handle_key,
                            "status_lines": lambda _t=teleop: [_t.status()],
                            "save_guard": guard,
                        }
                        print(
                            f"teleop enabled for {teleop_components}: keys go to the preview window "
                            "(w/s step, W/S x4, 0-9 joint, t component, -/= step, SPACE save, q quit)"
                        )
                    session = capture_handeye_session(board, camera, joints, settings, **hooks)
        print(session)
        if args.solve and not args.no_robot:
            from dexmate_calib.extrinsics.solve import solve_handeye_session

            solve_cfg = cfg["solve"]
            print(
                solve_handeye_session(
                    session,
                    min_views=int(solve_cfg["min_views"]),
                    min_corners=int(solve_cfg["min_corners"]),
                    huber_px=float(solve_cfg["huber_px"]),
                    max_view_rms_px=float(solve_cfg["max_view_rms_px"]),
                    leave_one_out=bool(solve_cfg.get("leave_one_out", True)),
                )
            )
        return 0
    raise AssertionError(args.extrinsics_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dexcalib")
    sub = parser.add_subparsers(dest="command", required=True)

    board = sub.add_parser("board", help="Inspect and verify ChArUco board profiles")
    board_sub = board.add_subparsers(dest="board_command", required=True)
    board_sub.add_parser("list")
    show = board_sub.add_parser("show")
    show.add_argument("board")
    validate = board_sub.add_parser("validate")
    validate.add_argument("path")
    verify = board_sub.add_parser("verify")
    verify.add_argument("--board", default="dexmate-10x7")
    verify.add_argument("--image", required=True)
    identify = board_sub.add_parser(
        "identify", help="Sweep all ArUco/AprilTag dictionaries to identify an unknown marker"
    )
    identify.add_argument("--image", required=True)
    render = board_sub.add_parser("render", help="Render a printable AprilTag grid PNG")
    render.add_argument("--board", required=True)
    render.add_argument("--output", required=True)
    render.add_argument("--dpi", type=float, default=300.0)
    render.add_argument("--margin-mm", type=float, default=20.0)

    doctor = sub.add_parser("doctor", help="Read-only network and stream checks")
    doctor_sub = doctor.add_subparsers(dest="doctor_command", required=True)
    doctor_sub.add_parser("network")
    stream = doctor_sub.add_parser("stream")
    stream.add_argument("--host", default="192.168.50.22")
    stream.add_argument("--port", type=int, default=30000)
    stream.add_argument("--frames", type=int, default=20)
    stream.add_argument("--expected-serial", type=_optional_serial, default=59595115)
    stream.add_argument("--width", type=int, default=1920)
    stream.add_argument("--height", type=int, default=1200)

    intrinsics = sub.add_parser("intrinsics", help="Capture and solve head intrinsics")
    intrinsics_sub = intrinsics.add_subparsers(dest="intrinsics_command", required=True)
    capture = intrinsics_sub.add_parser("capture")
    _add_capture_arguments(capture)
    quickstart = intrinsics_sub.add_parser(
        "quickstart", help="SSH-start streamer, validate, capture, stop, then solve"
    )
    _add_capture_arguments(quickstart)
    quickstart.add_argument(
        "--ssh-route",
        choices=("auto", "direct", "proxy"),
        default="auto",
    )
    quickstart.add_argument(
        "--no-sudo",
        action="store_true",
        help="Run streamer as dexmate-nano instead of prompting for remote sudo",
    )
    quickstart.add_argument("--startup-timeout", type=float, default=30.0)
    quickstart.add_argument("--preflight-frames", type=int, default=10)
    quickstart.add_argument(
        "--no-solve",
        action="store_true",
        help="Stop after capture instead of solving the new session",
    )
    solve = intrinsics_sub.add_parser("solve")
    solve.add_argument("session")
    solve.add_argument("--min-views", type=int, default=20)
    solve.add_argument("--max-view-error", type=float, default=0.8)
    solve.add_argument("--max-views", type=int, default=40)
    solve.add_argument("--cross-validation-folds", type=int, default=5)
    solve_multi = intrinsics_sub.add_parser(
        "solve-multi",
        help="Jointly solve one K from multiple compatible sessions",
    )
    solve_multi.add_argument("sessions", nargs="+")
    solve_multi.add_argument("--output", required=True)
    solve_multi.add_argument("--min-views", type=int, default=40)
    solve_multi.add_argument("--min-views-per-session", type=int, default=12)
    solve_multi.add_argument("--max-views-per-session", type=int, default=40)
    solve_multi.add_argument("--max-view-error", type=float, default=0.8)
    solve_multi.add_argument("--cross-validation-folds", type=int, default=5)
    kinect = sub.add_parser("kinect", help="Azure Kinect checks (factory calibration, snapshot)")
    kinect.add_argument(
        "--config", default=None, help="hand-eye config YAML (default configs/handeye.yaml)"
    )
    kinect_sub = kinect.add_subparsers(dest="kinect_command", required=True)
    for name in ("info", "snapshot"):
        k = kinect_sub.add_parser(name)
        k.add_argument("--color-resolution", default=None)
        k.add_argument("--depth-mode", default=None)
        k.add_argument("--fps", type=int, default=None)
        k.add_argument(
            "--any-serial", action="store_true", help="Do not enforce the configured serial"
        )
        if name == "snapshot":
            k.add_argument("--output", default="calibration_data/kinect_snapshot.png")

    extrinsics = sub.add_parser("extrinsics", help="External camera eye-to-hand calibration")
    extrinsics.add_argument(
        "--config", default=None, help="hand-eye config YAML (default configs/handeye.yaml)"
    )
    extrinsics_sub = extrinsics.add_subparsers(dest="extrinsics_command", required=True)
    ex_capture = extrinsics_sub.add_parser(
        "capture", help="Manual Kinect + joint capture (SPACE saves)"
    )
    ex_capture.add_argument("--board", default=None)
    ex_capture.add_argument("--output", default="calibration_data/handeye_kinect")
    ex_capture.add_argument("--samples", type=int, default=None)
    ex_capture.add_argument("--min-corners", type=int, default=None)
    ex_capture.add_argument("--robot-model", default=None)
    ex_capture.add_argument("--base-frame", default=None)
    ex_capture.add_argument("--target-link", default=None)
    ex_capture.add_argument("--color-resolution", default=None)
    ex_capture.add_argument("--depth-mode", default=None)
    ex_capture.add_argument("--fps", type=int, default=None)
    ex_capture.add_argument("--any-serial", action="store_true")
    ex_capture.add_argument(
        "--no-depth", action="store_true", help="Do not stream/save depth images"
    )
    ex_capture.add_argument("--preview-scale", type=float, default=0.5)
    ex_capture.add_argument(
        "--allow-moving", action="store_true", help="Save even if joints changed between reads"
    )
    ex_capture.add_argument(
        "--no-robot", action="store_true", help="Camera-only dry run (session cannot be solved)"
    )
    ex_capture.add_argument(
        "--solve", action="store_true", help="Solve the session right after capture"
    )
    ex_capture.add_argument(
        "--teleop",
        action="store_true",
        help="Drive the arm holding the board (and torso) with keys in the preview window",
    )
    ex_capture.add_argument(
        "--step-deg", type=float, default=0.5, help="Teleop joint step per key event"
    )
    ex_capture.add_argument(
        "--velocity-scale", type=float, default=0.3, help="Teleop motion speed (0-1]"
    )
    ex_solve = extrinsics_sub.add_parser("solve", help="Solve T_base_cam for a captured session")
    ex_solve.add_argument("session")
    ex_solve.add_argument("--robot-model", default=None)
    ex_solve.add_argument("--base-frame", default=None)
    ex_solve.add_argument("--target-link", default=None)
    ex_solve.add_argument("--min-views", type=int, default=None)
    ex_solve.add_argument("--min-corners", type=int, default=None)
    ex_solve.add_argument("--huber", type=float, default=None)
    ex_solve.add_argument("--max-view-rms", type=float, default=None)
    ex_solve.add_argument("--no-leave-one-out", action="store_true")
    ex_solve.add_argument(
        "--method",
        choices=("reprojection", "robotcamcalib"),
        default=None,
        help="reprojection (default): corner reprojection LM; robotcamcalib: verbatim RobotCamCalib solver",
    )
    ex_solve.add_argument(
        "--compare",
        action="store_true",
        help="Also solve with the other method and print the difference",
    )
    ex_self = extrinsics_sub.add_parser("selftest", help="Run the solver on a synthetic scene")
    ex_self.add_argument("--views", type=int, default=25)
    ex_self.add_argument("--pixel-noise", type=float, default=0.4)
    ex_self.add_argument("--fk-rotation-noise", type=float, default=0.03, help="degrees")
    ex_self.add_argument("--fk-translation-noise", type=float, default=0.0005, help="metres")
    ex_self.add_argument("--outliers", type=int, default=0)
    ex_self.add_argument("--seed", type=int, default=0)
    return parser


def _add_capture_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--board", default="dexmate-10x7")
    command.add_argument("--host", default="192.168.50.22")
    command.add_argument("--port", type=int, default=30000)
    command.add_argument("--expected-serial", type=_optional_serial, default=59595115)
    command.add_argument("--output", default="calibration_data/head_left")
    command.add_argument("--samples", type=int, default=40)
    command.add_argument("--min-corners", type=int, default=20)
    command.add_argument("--min-coverage", type=float, default=0.025)
    command.add_argument("--min-sharpness", type=float, default=80.0)
    command.add_argument("--min-grid-rows", type=int, default=3)
    command.add_argument("--min-grid-cols", type=int, default=3)
    command.add_argument(
        "--min-board-bbox",
        dest="min_board_bbox_fraction",
        type=float,
        default=0.12,
    )
    command.add_argument("--min-diversity", type=float, default=0.08)
    command.add_argument("--cooldown", type=float, default=0.8)
    command.add_argument(
        "--detect-fps",
        type=float,
        default=10.0,
        help="Auto-mode ChArUco processing rate; 0 processes every stream frame",
    )
    command.add_argument("--manual", action="store_true")
    command.add_argument("--no-preview", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "board":
            return _cmd_board(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "intrinsics":
            return _cmd_intrinsics(args)
        if args.command == "kinect":
            return _cmd_kinect(args)
        if args.command == "extrinsics":
            return _cmd_extrinsics(args)
        raise AssertionError(args.command)
    except (ConnectionError, OSError, RuntimeError, TypeError, ValueError, TimeoutError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
