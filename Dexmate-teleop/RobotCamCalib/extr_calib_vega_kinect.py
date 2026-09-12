#!/usr/bin/env python
"""Camera <-> robot extrinsics for the Dexmate Vega-1P + Azure Kinect (third-view camera).

Adapts ``extr_calib.py`` (written for xArm6 + RealSense) to this project:

* robot  — Dexmate Vega-1P. Joint angles come from ``scripts/dexmate_observer.py``
  (ZMQ ``tcp://127.0.0.1:5590``, ``{"t_wall_us", "q": {joint_name: rad}}``) so this
  tool runs in the repo ``.venv`` with **no dexcontrol dependency**; the console
  starts the observer automatically when it detects the robot. ``--joints dexcontrol``
  reads in-process instead (then run under the dexcontrol venv).
* camera — Azure Kinect via ``pyk4a`` (``cameras_kinect.KinectCamera``), factory
  colour intrinsics by default. ``--camera mock`` renders the AprilTag board
  synthetically from the live joint feed (GUI dry-run without a camera/robot);
  ``--self-check`` does the whole chain unattended against known ground truth.
* target — the toolkit's 4x4 AprilTag grid (``assets/apriltag_grid/
  compact_apriltag_grid_4x4_tag48mm.yaml``, print at 100 % on A3) rigidly attached
  to the wrist link (default ``R_arm_l7`` / ``L_arm_l7``; the solver estimates the
  board-on-wrist offset, so *where* on the wrist/hand it sits does not matter, only
  that it does not move between samples).
* output — ``configs/camera_extrinsics/kinect_<serial>.yaml`` with
  ``T_base_color`` (the calibrated result), the factory ``T_color_depth`` and their
  product ``T_base_depth`` (the frame ``scripts/kinect_pointcloud.py`` records in),
  plus K/dist, sample count and solver residuals. Samples are also dumped to
  ``<out>.samples.npz`` so the solve can be repeated offline.

Procedure (details in ``docs/VEGA-kinect-extrinsics.md``):

    # terminal 1 (robot on, network up):  observer, or just open the console page
    ROBOT_NAME=dm/vgd1262ab823-1p ~/Dexmate/dexcontrol/.venv/bin/python scripts/dexmate_observer.py
    # terminal 2 (repo root):
    .venv/bin/python RobotCamCalib/extr_calib_vega_kinect.py --arm right
    # browser http://localhost:8080  ->  move the arm (keyboard tool / goto_arm_pose.py),
    # "click_and_append" at >= 8 well-rotated poses, "click_and_save".

Self-check without any hardware (URDF FK -> synthetic board -> detection -> PnP -> solve):

    .venv/bin/python RobotCamCalib/extr_calib_vega_kinect.py --self-check
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent
if str(TOOL_DIR) not in sys.path:  # extr_calib / apriltag_board / cameras import each other bare
    sys.path.insert(0, str(TOOL_DIR))

import cv2  # noqa: E402
import yaml  # noqa: E402
from loguru import logger  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from apriltag_board import AprilTagBoardLayout, load_apriltag_board_yaml  # noqa: E402
from cameras import Camera  # noqa: E402
from extr_calib import (  # noqa: E402
    CamTagCalibrator,
    ExtrinsicsCalibConfig,
    calibrate_cammount_and_tag_prob,
    load_intrinsics_yaml,
    pose_err_deg_m,
)

DEFAULT_BOARD = TOOL_DIR / "assets/apriltag_grid/compact_apriltag_grid_4x4_tag48mm.yaml"
DEFAULT_SUB = "tcp://127.0.0.1:5590"          # scripts/dexmate_observer.py --pub
OUT_DIR = REPO_ROOT / "configs/camera_extrinsics"
ARM_JOINTS = {
    "right": [f"R_arm_j{i}" for i in range(1, 8)],
    "left": [f"L_arm_j{i}" for i in range(1, 8)],
}
TORSO_JOINTS = ["torso_j1", "torso_j2", "torso_j3"]
TAGMOUNT_DEFAULT = {"right": "R_arm_l7", "left": "L_arm_l7"}


# --------------------------------------------------------------------------- URDF
def resolve_urdf(explicit: Optional[str]) -> Path:
    """Same lookup order the rest of the repo uses: flag, $VEGA_URDF, ~/Dexmate, in-repo clone."""
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env = os.environ.get("VEGA_URDF")
    if env:
        candidates.append(Path(env).expanduser())
    candidates += [
        Path("~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser(),
        REPO_ROOT / "dexmate-urdf/robots/humanoid/vega_1p/vega_1p.urdf",
        REPO_ROOT / "dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf",
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "Vega URDF not found. Tried: " + ", ".join(str(c) for c in candidates)
        + ". Pass --urdf, set $VEGA_URDF, or clone dexmate-urdf to ~/Dexmate/."
    )


# ------------------------------------------------------------------ joint sources
class ZmqJointSource:
    """Freshest ``{"t_wall_us", "q": {name: rad}}`` frame from dexmate_observer.py."""

    def __init__(self, addr: str = DEFAULT_SUB) -> None:
        import msgpack
        import zmq

        self._msgpack = msgpack
        self.addr = addr
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.connect(addr)
        self._zmq = zmq
        self.q: Optional[Dict[str, float]] = None
        self.t_recv = 0.0
        self.n = 0

    def latest(self) -> Tuple[Optional[Dict[str, float]], float]:
        """Return (q dict or None, seconds since the last frame)."""
        while True:
            try:
                raw = self.sock.recv(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            msg = self._msgpack.unpackb(raw, raw=False)
            q = msg.get("q") if isinstance(msg, dict) else None
            if q:
                self.q = {str(k): float(v) for k, v in q.items()}
                self.t_recv = time.time()
                self.n += 1
        age = float("inf") if self.q is None else time.time() - self.t_recv
        return self.q, age

    def describe(self) -> str:
        return f"observer {self.addr}"


class DexcontrolJointSource:
    """In-process read via dexcontrol (needs the dexcontrol venv + this toolkit's deps)."""

    def __init__(self) -> None:
        try:
            from dexcontrol.robot import Robot  # noqa: WPS433
        except ImportError as exc:
            raise ImportError(
                "dexcontrol is not importable here. Either run this tool with the "
                "dexcontrol venv (and install viser/yourdfpy/pupil-apriltags/opencv "
                "there), or use --joints zmq with scripts/dexmate_observer.py running."
            ) from exc
        if not os.environ.get("ROBOT_NAME"):
            logger.warning("ROBOT_NAME is not set (SOP: ROBOT_NAME=dm/vgd1262ab823-1p).")
        self.bot = Robot()

    def latest(self) -> Tuple[Optional[Dict[str, float]], float]:
        q: Dict[str, float] = {}
        for comp in ("left_arm", "right_arm", "head", "torso"):
            try:
                q.update({str(k): float(v) for k, v in self.bot.get_joint_pos_dict(comp).items()})
            except Exception:  # noqa: BLE001 - a component may be absent/unpowered
                pass
        return (q or None), 0.0

    def describe(self) -> str:
        return "dexcontrol (in-process, read-only)"


class MockJointSource:
    def __init__(self, q: Optional[Dict[str, float]] = None) -> None:
        self.q: Dict[str, float] = dict(q or {})

    def latest(self) -> Tuple[Optional[Dict[str, float]], float]:
        return dict(self.q), 0.0

    def describe(self) -> str:
        return "mock"


def qpos_from_dict(q: Dict[str, float], actuated_names: Sequence[str]) -> np.ndarray:
    """Full URDF actuated vector (zeros for joints the feed does not carry, e.g. wheels)."""
    return np.array([float(q.get(n, 0.0)) for n in actuated_names], dtype=float)


# --------------------------------------------------------------- mock board camera
def _april_dictionary():
    return cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)


class MockBoardCamera(Camera):
    """Renders the AprilTag board at ``pose_fn()`` = X_CamTag (board pose in the camera frame).

    Corner bookkeeping: for an upright ``cv2.aruco`` marker image, ``pupil_apriltags``
    reports ``Detection.corners`` as [top-right, top-left, bottom-left, bottom-right]
    (measured, cv2's canonical orientation is 180 deg from the AprilTag library's).
    The board YAML pairs ``corners_board_mm[i]`` with ``corners[i]``, so the marker
    bitmap corners are warped onto the projections of ``corners_board_mm`` in exactly
    that order — then the detector's PnP reproduces ``pose_fn()``.
    """

    def __init__(
        self,
        board: AprilTagBoardLayout,
        K: np.ndarray,
        dist: np.ndarray,
        image_size: Tuple[int, int],
        pose_fn: Callable[[], np.ndarray],
        marker_px: int = 160,
    ) -> None:
        self.board = board
        self.K = np.asarray(K, dtype=float).reshape(3, 3)
        self.dist = np.asarray(dist, dtype=float).reshape(-1)
        self.width, self.height = int(image_size[0]), int(image_size[1])
        self.pose_fn = pose_fn
        self.marker_px = int(marker_px)
        dictionary = _april_dictionary()
        self.bitmaps = {
            tag_id: cv2.aruco.generateImageMarker(dictionary, tag_id, self.marker_px)
            for tag_id in board.tags_corners_board_m
        }
        s = float(self.marker_px)
        # bitmap pixel corners in Detection.corners order: TR, TL, BL, BR
        self._src = np.array([[s, 0.0], [0.0, 0.0], [0.0, s], [s, s]], dtype=np.float32)
        self.serial = "mock"

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def read(self) -> np.ndarray:
        return self.render(self.pose_fn())

    def visible(self, X_CamTag: np.ndarray, margin_px: int = 10, min_grazing_deg: float = 20.0) -> bool:
        """All board corners in front of the camera, inside the image, and the plane not
        seen edge-on. Front/back side is left to the detector (a mirrored 36h11 does not decode)."""
        pts = np.vstack([c for c in self.board.tags_corners_board_m.values()])
        pts_cam = (X_CamTag[:3, :3] @ pts.T).T + X_CamTag[:3, 3]
        if np.any(pts_cam[:, 2] <= 0.05):
            return False
        n_cam = X_CamTag[:3, :3] @ np.array([0.0, 0.0, 1.0])
        view = X_CamTag[:3, 3] / max(np.linalg.norm(X_CamTag[:3, 3]), 1e-9)
        if abs(np.dot(n_cam, view)) < np.sin(np.radians(min_grazing_deg)):
            return False
        rvec, _ = cv2.Rodrigues(X_CamTag[:3, :3])
        img, _ = cv2.projectPoints(pts.astype(np.float64), rvec, X_CamTag[:3, 3], self.K, self.dist)
        img = img.reshape(-1, 2)
        ok = (img[:, 0] > margin_px) & (img[:, 0] < self.width - margin_px) & \
             (img[:, 1] > margin_px) & (img[:, 1] < self.height - margin_px)
        return bool(np.all(ok))

    def render(self, X_CamTag: np.ndarray) -> np.ndarray:
        canvas = np.full((self.height, self.width), 255, dtype=np.uint8)
        rvec, _ = cv2.Rodrigues(X_CamTag[:3, :3])
        tvec = X_CamTag[:3, 3].astype(np.float64)
        for tag_id, corners_board in self.board.tags_corners_board_m.items():
            img_pts, _ = cv2.projectPoints(corners_board.astype(np.float64), rvec, tvec, self.K, self.dist)
            dst = img_pts.reshape(4, 2).astype(np.float32)
            H = cv2.getPerspectiveTransform(self._src, dst)
            warped = cv2.warpPerspective(
                self.bitmaps[tag_id], H, (self.width, self.height),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
            )
            canvas = np.minimum(canvas, warped)  # black wins
        return cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)


# ------------------------------------------------------------------- calibrator
class VegaKinectCalibrator(CamTagCalibrator):
    """CamTagCalibrator + project-level output (base<-color, color<-depth, base<-depth)."""

    def __init__(self, config: ExtrinsicsCalibConfig, meta: dict, T_color_depth: Optional[np.ndarray]) -> None:
        super().__init__(config)
        self.meta = dict(meta)
        self.T_color_depth = None if T_color_depth is None else np.asarray(T_color_depth, dtype=float)
        self.last_info: Optional[dict] = None
        self.joint_status = "waiting for joints"
        self.reset_button = self.server.gui.add_button("click_and_reset_samples")
        self.reset_button.on_click(lambda _: self.reset_samples())
        try:
            self.joints_text = self.server.gui.add_text("joints", self.joint_status)
            self.solve_text = self.server.gui.add_text("solve", "0 samples")
        except Exception:  # noqa: BLE001 - very old viser
            self.joints_text = None
            self.solve_text = None

    # -- joints -----------------------------------------------------------
    def set_joint_status(self, status: str) -> None:
        self.joint_status = status
        if self.joints_text is not None and self.joints_text.value != status:
            self.joints_text.value = status

    def block_append(self, reason: str) -> None:
        """Refuse the next append: joints unusable => the FK side of the sample is wrong."""
        self.X_CamTag = None
        self.set_detection_status(f"REJECT {reason}")

    # -- samples ----------------------------------------------------------
    def reset_samples(self) -> None:
        self.X_CamTag_list.clear()
        self.X_WorldCammount_list.clear()
        self.X_WorldTagmount_list.clear()
        self.last_info = None
        logger.info("Cleared all samples.")
        self._update_solve_text()

    def append_and_solve(self) -> None:
        if self.X_CamTag is None:
            logger.warning(f"Not appended: {self.detection_status}")
            return
        self.X_CamTag_list.append(self.X_CamTag.copy())
        self.X_WorldCammount_list.append(self.X_WorldCammount.copy())
        self.X_WorldTagmount_list.append(self.X_WorldTagmount.copy())
        logger.info(f"Appended {len(self.X_CamTag_list)} observations.")
        if len(self.X_CamTag_list) >= self.config.ransac_sample_size:
            self.X_CammountCam, self.X_TagmountTag, self.last_info = calibrate_cammount_and_tag_prob(
                np.asarray(self.X_CamTag_list),
                np.asarray(self.X_WorldCammount_list),
                np.asarray(self.X_WorldTagmount_list),
                max_iters=200,
            )
            logger.info(f"Solved.  X_CammountCam (T_{self.config.cammount_link_name}_color):\n{self.X_CammountCam}")
            logger.info(f"residuals: {self.last_info}")
        self._update_solve_text()

    def _update_solve_text(self) -> None:
        n = len(self.X_CamTag_list)
        if self.last_info is None:
            txt = f"{n} samples (solve starts at {self.config.ransac_sample_size})"
        else:
            i = self.last_info
            t = self.X_CammountCam[:3, 3]
            txt = (f"{n} samples | resid rot {i['rot_err_deg_mean']:.2f}° (max {i['rot_err_deg_max']:.2f}°) "
                   f"trans {i['trans_err_mean']*1000:.1f} mm (max {i['trans_err_max']*1000:.1f} mm) | "
                   f"cam @ base ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m")
        if self.solve_text is not None:
            self.solve_text.value = txt

    # -- output -----------------------------------------------------------
    def result_dict(self) -> dict:
        T_base_color = np.asarray(self.X_CammountCam, dtype=float)
        out = {
            "schema": "dexmate_teleop.camera_extrinsics.v1",
            "created": _dt.datetime.now().isoformat(timespec="seconds"),
            "note": (
                "T_base_color: colour-camera pose in the robot base frame (X_base = T @ X_color). "
                "T_color_depth: factory depth->colour. T_base_depth = T_base_color @ T_color_depth is "
                "the frame scripts/kinect_pointcloud.py records point clouds in (--align depth). "
                "Valid only while neither the camera nor the robot base moves."
            ),
            **self.meta,
            "n_samples": int(len(self.X_CamTag_list)),
            "solver": None if self.last_info is None else {k: float(v) for k, v in self.last_info.items()},
            "K_color": np.asarray(self.K, dtype=float).tolist(),
            "dist_color": np.asarray(self.dist, dtype=float).tolist(),
            "image_size_color": None if self.config.image_size is None else [int(v) for v in self.config.image_size],
            "T_base_color": T_base_color.tolist(),
            "T_tagmount_board": np.asarray(self.X_TagmountTag, dtype=float).tolist(),
            "T_color_depth": None if self.T_color_depth is None else self.T_color_depth.tolist(),
            "T_base_depth": None if self.T_color_depth is None else (T_base_color @ self.T_color_depth).tolist(),
            # RobotCamCalib-native names, so the toolkit's own readers still work
            "X_CammountCam": T_base_color.tolist(),
            "X_TagmountTag": np.asarray(self.X_TagmountTag, dtype=float).tolist(),
        }
        return out

    def save_results(self) -> None:
        if self.last_info is None:
            logger.warning(
                f"Nothing solved yet ({len(self.X_CamTag_list)} samples, need "
                f"{self.config.ransac_sample_size}); not saving."
            )
            return
        out_path = Path(self.config.output_file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            yaml.safe_dump(self.result_dict(), f, sort_keys=False)
        np.savez(
            str(out_path) + ".samples.npz",
            X_CamTag=np.asarray(self.X_CamTag_list),
            X_WorldCammount=np.asarray(self.X_WorldCammount_list),
            X_WorldTagmount=np.asarray(self.X_WorldTagmount_list),
        )
        logger.info(f"Saved {out_path} (+ .samples.npz).")


# --------------------------------------------------------------------- helpers
def look_at(cam_pos: np.ndarray, target: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """T_base_cam for an OpenCV camera (z forward, y down) at cam_pos looking at target."""
    z = np.asarray(target, float) - np.asarray(cam_pos, float)
    z /= np.linalg.norm(z)
    up = np.asarray(up, float)
    x = np.cross(z, up)                # right = forward x world-up   (y down => x = z x up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, cam_pos
    return T


def mock_truth(arm: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
    """Synthetic ground truth shared by --camera mock and --self-check:
    (T_base_cam, T_tagmount_board, K, dist8, image_size). Kinect-like 1080p colour camera on a
    tripod front-right/-left of the robot, board 6 cm off the wrist link."""
    side = -1.0 if arm == "right" else 1.0
    T_base_cam = look_at(np.array([1.9, side * 0.4, 1.35]), np.array([0.45, side * 0.25, 0.95]))
    T_l7_board = np.eye(4)
    T_l7_board[:3, :3] = Rot.from_euler("xyz", [10.0, -80.0, 5.0], degrees=True).as_matrix()
    T_l7_board[:3, 3] = [0.06, 0.02, -0.01]
    K = np.array([[915.0, 0.0, 958.0], [0.0, 914.0, 548.0], [0.0, 0.0, 1.0]])
    dist = np.array([0.05, -0.02, 0.0005, -0.0003, 0.001, 0.0, 0.0, 0.0])
    return T_base_cam, T_l7_board, K, dist, (1920, 1080)


def build_camera(args, mock_state: Optional[dict] = None) -> Tuple[Camera, np.ndarray, np.ndarray, Tuple[int, int], Optional[np.ndarray], dict]:
    """Return (camera, K, dist, image_size, T_color_depth or None, meta)."""
    if args.camera == "mock":
        board = load_apriltag_board_yaml(Path(args.board))
        T_base_cam, T_l7_board, K, dist, size = mock_truth(args.arm)
        mock_state.update({"X_CamTag": np.eye(4), "T_base_cam": T_base_cam, "T_l7_board": T_l7_board})
        cam = MockBoardCamera(board, K, dist, size, pose_fn=lambda: mock_state["X_CamTag"])
        T_cd = np.eye(4); T_cd[:3, 3] = [-0.032, 0.0, 0.002]
        meta = {"camera": {"type": "mock", "serial": "mock", "note": "synthetic board rendered from the joint feed"}}
        return cam, K, dist, size, T_cd, meta
    if args.camera == "kinect":
        from cameras_kinect import KinectCamera

        cam = KinectCamera(serial=args.serial, device_id=args.device_id,
                           color_res=args.color_res, fps=args.fps, depth_mode=args.depth_mode)
        cam.start()
        if args.intrinsics == "factory":
            K, dist, size = cam.intrinsics("color")
            intr_src = "kinect factory"
        else:
            K, dist, size = load_intrinsics_yaml(Path(args.intrinsics))
            intr_src = str(args.intrinsics)
        T_cd = cam.T_color_depth()
        meta = {
            "camera": {"type": "azure_kinect", "serial": cam.serial, "color_res": args.color_res,
                       "depth_mode": args.depth_mode, "fps": args.fps, "intrinsics": intr_src},
        }
        return cam, K, dist, size, T_cd, meta
    raise ValueError(f"unknown camera {args.camera}")


def run_live(args) -> int:
    urdf = resolve_urdf(args.urdf)
    tagmount = args.tagmount_link or TAGMOUNT_DEFAULT[args.arm]
    mock_state: dict = {}
    cam, K, dist, size, T_cd, meta = build_camera(args, mock_state)
    serial = getattr(cam, "serial", None) or "unknown"
    out = Path(args.out) if args.out else OUT_DIR / f"kinect_{serial}.yaml"

    if args.joints == "zmq":
        joints = ZmqJointSource(args.sub)
    else:
        joints = DexcontrolJointSource()
    required = ARM_JOINTS[args.arm] + TORSO_JOINTS

    config = ExtrinsicsCalibConfig(
        robot_urdf_path=urdf,
        cammount_link_name=args.cammount_link,
        tagmount_link_name=tagmount,
        K=K, dist=dist, image_size=size,
        output_file_path=out,
        tag_family="tag36h11",
        tag_board_yaml_path=Path(args.board),
        max_reproj_px=args.max_reproj_px,
        ransac_sample_size=args.min_samples,
    )
    meta.update({
        "robot": {"urdf": str(urdf), "base_link": args.cammount_link, "tagmount_link": tagmount,
                  "arm": args.arm, "joints_from": joints.describe()},
        "board": Path(args.board).stem,
    })
    calib = VegaKinectCalibrator(config, meta, T_cd)
    names = list(calib.urdf_viser.actuated_joint_names)
    missing_in_urdf = [n for n in required if n not in names]
    if missing_in_urdf:
        raise SystemExit(f"URDF {urdf} lacks joints {missing_in_urdf}; wrong model for --arm {args.arm}?")

    logger.info(f"URDF {urdf.name}: {len(names)} actuated joints; base={args.cammount_link} tag mount={tagmount}")
    logger.info(f"joints: {joints.describe()} | camera: {meta['camera']} | out: {out}")
    logger.info("Browser: the viser URL printed above. Move the arm, click_and_append at each pose, click_and_save.")

    period = 1.0 / max(args.hz, 0.5)
    t_log = 0.0
    try:
        while True:
            t0 = time.time()
            q, age = joints.latest()
            if q is None:
                qpos = calib.urdf_viser.qpos
                reason = f"no joint data yet from {joints.describe()}"
            else:
                qpos = qpos_from_dict(q, names)
                missing = [n for n in required if n not in q]
                if missing:
                    reason = f"joint feed lacks {missing}"
                elif age > args.stale_s:
                    reason = f"joint data stale ({age:.1f} s > {args.stale_s} s)"
                else:
                    reason = ""
            if mock_state:  # synthetic camera: board pose follows the URDF FK of the joint feed
                calib.urdf_viser.update(qpos)
                X_base_tag = calib.urdf_viser.get_frame_pose(tagmount)
                mock_state["X_CamTag"] = np.linalg.inv(mock_state["T_base_cam"]) @ X_base_tag @ mock_state["T_l7_board"]
            rgb = cam.read()
            calib.vis_step(rgb, qpos, input_joint_names=None)
            if reason:
                calib.set_joint_status(f"⚠ {reason}")
                calib.block_append(f"joints: {reason} | camera: {calib.detection_status}")
            else:
                arm_deg = np.degrees([q[n] for n in ARM_JOINTS[args.arm]])
                calib.set_joint_status(
                    f"OK {age*1000:.0f} ms old | {args.arm} arm deg " + " ".join(f"{v:.0f}" for v in arm_deg)
                )
            if time.time() - t_log >= args.log_every:
                t_log = time.time()
                logger.info(f"joints: {calib.joint_status} | board: {calib.detection_status} | "
                            f"samples: {len(calib.X_CamTag_list)}")
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        logger.info("stopped by user")
    finally:
        cam.stop()
    return 0


# ------------------------------------------------------------------ self-check
def self_check(args) -> int:
    """No hardware: known camera pose + known board-on-wrist offset, random reachable
    arm poses -> render -> detect -> PnP -> solve, then compare with the truth."""
    rng = np.random.default_rng(args.seed)
    urdf = resolve_urdf(args.urdf)
    board = load_apriltag_board_yaml(DEFAULT_BOARD if not args.board else Path(args.board))
    tagmount = args.tagmount_link or TAGMOUNT_DEFAULT[args.arm]

    # Kinect-like 1080p colour camera with a mild 8-coefficient distortion (exercises the rational
    # path); truth = tripod camera front-right/-left of the robot, board 6 cm off the wrist link.
    T_base_cam_true, T_l7_board_true, K, dist, size = mock_truth(args.arm)

    joints = MockJointSource()
    state = {"X_CamTag": np.eye(4)}
    cam = MockBoardCamera(board, K, dist, size, pose_fn=lambda: state["X_CamTag"])

    config = ExtrinsicsCalibConfig(
        robot_urdf_path=urdf, cammount_link_name=args.cammount_link, tagmount_link_name=tagmount,
        K=K, dist=dist, image_size=size,
        output_file_path=Path(args.out) if args.out else TOOL_DIR / "outputs/vega_kinect_selfcheck.yaml",
        tag_family="tag36h11", tag_board_yaml_path=DEFAULT_BOARD if not args.board else Path(args.board),
        max_reproj_px=args.max_reproj_px, ransac_sample_size=args.min_samples,
    )
    T_cd = np.eye(4); T_cd[:3, 3] = [-0.032, 0.0, 0.002]  # placeholder factory-like depth->colour
    calib = VegaKinectCalibrator(config, {"camera": {"type": "mock"}, "robot": {"urdf": str(urdf), "arm": args.arm,
                                                     "base_link": args.cammount_link, "tagmount_link": tagmount},
                                          "board": board.name}, T_cd)
    names = list(calib.urdf_viser.actuated_joint_names)
    yourdf = calib.urdf_viser.viser_urdf._urdf

    def fk(q: Dict[str, float], link: str) -> np.ndarray:
        yourdf.update_cfg(qpos_from_dict(q, names))
        return np.asarray(yourdf.get_transform(link))

    # nominal "arm forward" pose per side, then random perturbations
    nominal = {"right": [-0.6, -0.4, -0.2, -1.4, -0.6, -0.1, 0.3], "left": [-0.6, 0.4, 0.2, -1.4, 0.6, 0.1, -0.3]}[args.arm]
    n_target = args.self_check_samples
    accepted = 0
    tries = 0
    t_start = time.time()
    while accepted < n_target and tries < 300:
        tries += 1
        q = {n: 0.0 for n in names}
        for jn, v0 in zip(ARM_JOINTS[args.arm], nominal):
            q[jn] = v0 + rng.uniform(-0.45, 0.45)
        q["torso_j1"], q["torso_j2"], q["torso_j3"] = 0.0, 0.0, rng.uniform(-0.1, 0.1)
        X_base_l7 = fk(q, tagmount)
        X_CamTag = np.linalg.inv(T_base_cam_true) @ X_base_l7 @ T_l7_board_true
        if not cam.visible(X_CamTag):
            continue
        state["X_CamTag"] = X_CamTag
        joints.q = q
        rgb = cam.read()
        qpos = qpos_from_dict(q, names)
        calib.vis_step(rgb, qpos, input_joint_names=None)
        if calib.X_CamTag is None:
            logger.warning(f"pose {tries}: {calib.detection_status}")
            continue
        err_deg, err_m = pose_err_deg_m(calib.X_CamTag, X_CamTag)
        if err_deg > 1.0 or err_m > 0.01:
            logger.warning(f"pose {tries}: PnP off by {err_deg:.2f} deg / {err_m*1000:.1f} mm — corner order or intrinsics wrong")
        calib.append_and_solve()
        accepted += 1
        logger.info(f"sample {accepted}/{n_target}: {calib.detection_status}; PnP err {err_deg:.3f} deg {err_m*1000:.2f} mm")

    ok = accepted >= n_target and calib.last_info is not None
    if ok:
        e_rot, e_tr = pose_err_deg_m(calib.X_CammountCam, T_base_cam_true)
        e_rot2, e_tr2 = pose_err_deg_m(calib.X_TagmountTag, T_l7_board_true)
        logger.info(f"T_base_color  error vs truth: {e_rot:.3f} deg, {e_tr*1000:.2f} mm")
        logger.info(f"T_l7_board    error vs truth: {e_rot2:.3f} deg, {e_tr2*1000:.2f} mm")
        ok = e_rot < 0.5 and e_tr < 0.005 and e_rot2 < 0.5 and e_tr2 < 0.005
        calib.save_results()
        res = calib.result_dict()
        T_bd = np.asarray(res["T_base_depth"])
        assert np.allclose(T_bd, calib.X_CammountCam @ T_cd), "T_base_depth composition wrong"
    logger.info(f"self-check {'PASS' if ok else 'FAIL'} ({accepted} samples, {tries} poses tried, {time.time()-t_start:.1f} s)")
    try:
        calib.server.stop()
    except Exception:  # noqa: BLE001
        pass
    return 0 if ok else 1


# ------------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", choices=("right", "left"), default="right", help="which arm carries the board")
    p.add_argument("--urdf", default=None, help="Vega URDF (default: $VEGA_URDF, ~/Dexmate/dexmate-urdf, in-repo clone)")
    p.add_argument("--cammount-link", default="base", help="URDF link the camera is fixed relative to (robot base)")
    p.add_argument("--tagmount-link", default=None, help="URDF link the board is fixed to (default R_arm_l7 / L_arm_l7)")
    p.add_argument("--board", default=str(DEFAULT_BOARD), help="AprilTag board yaml")
    p.add_argument("--camera", choices=("kinect", "mock"), default="kinect", help="mock = synthetic board from the joint feed (dry run)")
    p.add_argument("--serial", default=None, help="Kinect serial (multi-camera); default first device")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--color-res", default="1080P")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--depth-mode", default="NFOV_UNBINNED", help="must match what kinect_pointcloud.py records with")
    p.add_argument("--intrinsics", default="factory", help="'factory' (Kinect calibration) or an intr_calib.py yaml")
    p.add_argument("--joints", choices=("zmq", "dexcontrol"), default="zmq")
    p.add_argument("--sub", default=DEFAULT_SUB, help="dexmate_observer.py publisher address")
    p.add_argument("--stale-s", type=float, default=1.0, help="refuse to append if joints older than this")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--log-every", type=float, default=2.0, help="seconds between terminal status lines")
    p.add_argument("--min-samples", type=int, default=8, help="solve once this many samples are appended")
    p.add_argument("--max-reproj-px", type=float, default=2.0)
    p.add_argument("--out", default=None, help=f"output yaml (default {OUT_DIR}/kinect_<serial>.yaml)")
    p.add_argument("--self-check", action="store_true", help="hardware-free end-to-end check with a synthetic board")
    p.add_argument("--self-check-samples", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_check:
        return self_check(args)
    return run_live(args)


if __name__ == "__main__":
    raise SystemExit(main())
