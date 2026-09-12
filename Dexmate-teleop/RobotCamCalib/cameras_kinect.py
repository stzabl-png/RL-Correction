"""Azure Kinect (pyk4a) camera for RobotCamCalib, plus its factory calibration.

Same ``Camera`` interface as ``cameras.py`` (``start / stop / read -> RGB uint8``),
kept in its own module because ``pyk4a`` needs the Azure Kinect SDK (libk4a) at
import time and must stay optional for the RealSense / OAK / cv2 users.

What the Dexmate-teleop stack needs from the calibration on top of the image:

* ``intrinsics()``  — the factory colour intrinsics ``K`` (3x3), 8 OpenCV
  distortion coefficients ``[k1 k2 p1 p2 k3 k4 k5 k6]`` and the image size the
  colour stream runs at. Good enough for board PnP; run ``intr_calib.py`` if you
  ever need better than factory.
* ``T_color_depth()`` — the factory depth->colour rigid transform, because
  ``scripts/kinect_pointcloud.py`` records point clouds in the **depth** camera
  frame (``--align depth``) while the AprilTag board is seen by the **colour**
  camera. ``T_base_depth = T_base_color @ T_color_depth``.

Device selection mirrors ``scripts/kinect_pointcloud.py``: by ``device_id`` or,
preferably, by ``serial`` (stable across re-plugs; the recorder names its data
directories ``cloud_<serial>``).
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from cameras import Camera


class KinectCamera(Camera):
    """Azure Kinect colour stream via pyk4a."""

    def __init__(
        self,
        serial: Optional[str] = None,
        device_id: int = 0,
        color_res: str = "1080P",
        fps: int = 30,
        depth_mode: str = "NFOV_UNBINNED",
        max_empty_captures: int = 30,
    ) -> None:
        self.serial_wanted = None if serial is None else str(serial)
        self.device_id = int(device_id)
        self.color_res = str(color_res).upper().removeprefix("RES_")
        self.fps = int(fps)
        self.depth_mode = str(depth_mode).upper()
        self.max_empty_captures = int(max_empty_captures)

        self.k4a: Any | None = None
        self.serial: Optional[str] = None
        self._pyk4a: Any | None = None

    # ---- lifecycle -------------------------------------------------------
    @staticmethod
    def _import_pyk4a():
        try:
            import pyk4a  # noqa: WPS433
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise ImportError(
                "pyk4a is not installed (or libk4a is missing). Install the Azure "
                "Kinect SDK (k4a-tools / libk4a1.4-dev) and `uv pip install "
                "--python .venv/bin/python pyk4a`; `scripts/kinect_pointcloud.py "
                "--probe` explains what is missing."
            ) from exc
        return pyk4a

    def _make_config(self, pyk4a):
        from pyk4a import Config

        return Config(
            color_resolution=getattr(pyk4a.ColorResolution, f"RES_{self.color_res}"),
            depth_mode=getattr(pyk4a.DepthMode, self.depth_mode),
            camera_fps=getattr(pyk4a.FPS, f"FPS_{self.fps}"),
            color_format=pyk4a.ImageFormat.COLOR_BGRA32,  # capture.color -> BGRA uint8
            synchronized_images_only=True,
        )

    def start(self) -> None:
        pyk4a = self._import_pyk4a()
        self._pyk4a = pyk4a
        from pyk4a import PyK4A

        n_dev = int(pyk4a.connected_device_count())
        if n_dev == 0:
            raise RuntimeError("No Azure Kinect connected (pyk4a.connected_device_count() == 0).")

        if self.serial_wanted is None:
            k4a = PyK4A(self._make_config(pyk4a), device_id=self.device_id)
            k4a.start()
        else:
            # Open devices one by one until the serial matches; a started device
            # holds the USB exclusively, so stop the non-matching ones at once.
            k4a = None
            seen = []
            for dev in range(n_dev):
                cand = PyK4A(self._make_config(pyk4a), device_id=dev)
                cand.start()
                seen.append(str(cand.serial))
                if str(cand.serial) == self.serial_wanted:
                    k4a = cand
                    break
                cand.stop()
            if k4a is None:
                raise RuntimeError(
                    f"No Azure Kinect with serial {self.serial_wanted}; connected serials: {seen}"
                )
        self.k4a = k4a
        self.serial = str(k4a.serial)

    def stop(self) -> None:
        if self.k4a is not None:
            try:
                self.k4a.stop()
            finally:
                self.k4a = None

    # ---- frames ----------------------------------------------------------
    def read(self) -> np.ndarray:
        """One colour frame as (H, W, 3) RGB uint8."""
        if self.k4a is None:
            raise RuntimeError("Camera not started. Call start() or use the context manager.")
        empty = 0
        while True:
            capture = self.k4a.get_capture()
            color = capture.color
            if color is not None:
                return cv2.cvtColor(color, cv2.COLOR_BGRA2RGB)
            empty += 1
            if empty >= self.max_empty_captures:
                raise RuntimeError(
                    f"{empty} captures in a row had no colour image - USB2 cable, or "
                    "another process holding the device?"
                )

    # ---- factory calibration --------------------------------------------
    def intrinsics(self, camera: str = "color") -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """Factory ``(K, dist8, (width, height))`` for the colour (default) or depth camera.

        ``dist8`` is OpenCV order ``[k1 k2 p1 p2 k3 k4 k5 k6]`` (rational model),
        exactly what ``cv2.solvePnP`` / ``cv2.projectPoints`` expect.
        """
        if self.k4a is None:
            raise RuntimeError("Camera not started.")
        from pyk4a import CalibrationType

        cam_type = CalibrationType.COLOR if camera == "color" else CalibrationType.DEPTH
        calib = self.k4a.calibration
        K = np.asarray(calib.get_camera_matrix(cam_type), dtype=float).reshape(3, 3)
        dist = np.asarray(calib.get_distortion_coefficients(cam_type), dtype=float).reshape(-1)
        if camera == "color":
            size = _color_size(self.color_res)
        else:
            size = _depth_size(self.depth_mode)
        return K, dist, size

    def T_color_depth(self) -> np.ndarray:
        """Factory rigid transform depth-camera -> colour-camera, i.e. ``X_color = T @ X_depth``.

        Uses ``Calibration.get_extrinsic_parameters`` (pyk4a >= 1.4). Older pyk4a
        only exposes ``depth_to_color_3d``; the transform is then recovered by
        pushing the origin and the three unit axes through it.
        """
        if self.k4a is None:
            raise RuntimeError("Camera not started.")
        from pyk4a import CalibrationType

        calib = self.k4a.calibration
        T = np.eye(4)
        if hasattr(calib, "get_extrinsic_parameters"):
            R, t = calib.get_extrinsic_parameters(CalibrationType.DEPTH, CalibrationType.COLOR)
            T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
            T[:3, 3] = np.asarray(t, dtype=float).reshape(3)  # already metres
            return T
        # fallback: probe depth_to_color_3d (millimetres in, millimetres out)
        o = np.asarray(calib.depth_to_color_3d((0.0, 0.0, 0.0)), dtype=float)
        cols = []
        for axis in np.eye(3) * 1000.0:
            cols.append(np.asarray(calib.depth_to_color_3d(tuple(axis)), dtype=float) - o)
        T[:3, :3] = np.stack(cols, axis=1) / 1000.0
        T[:3, 3] = o / 1000.0
        return T


def _color_size(color_res: str) -> Tuple[int, int]:
    table = {
        "720P": (1280, 720),
        "1080P": (1920, 1080),
        "1440P": (2560, 1440),
        "1536P": (2048, 1536),
        "2160P": (3840, 2160),
        "3072P": (4096, 3072),
    }
    return table[color_res]


def _depth_size(depth_mode: str) -> Tuple[int, int]:
    table = {
        "NFOV_2X2BINNED": (320, 288),
        "NFOV_UNBINNED": (640, 576),
        "WFOV_2X2BINNED": (512, 512),
        "WFOV_UNBINNED": (1024, 1024),
    }
    return table[depth_mode]
