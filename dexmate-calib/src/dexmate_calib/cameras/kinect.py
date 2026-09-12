"""Azure Kinect (pyk4a) capture used for external-camera calibration.

The camera is opened in a fixed configuration and the SDK factory calibration is
exported as plain numpy/JSON so a session can be solved offline without the SDK.
Nothing here depends on the robot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

COLOR_RESOLUTIONS = {
    "720P": (1280, 720),
    "1080P": (1920, 1080),
    "1440P": (2560, 1440),
    "1536P": (2048, 1536),
    "2160P": (3840, 2160),
    "3072P": (4096, 3072),
}
DEPTH_MODES = ("NFOV_UNBINNED", "NFOV_2X2BINNED", "WFOV_UNBINNED", "WFOV_2X2BINNED", "OFF")


@dataclass
class KinectFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray | None
    color_timestamp_us: int
    depth_timestamp_us: int | None
    receive_time_s: float


@dataclass
class KinectCalibration:
    serial: str
    color_resolution: str
    depth_mode: str
    color_size: tuple[int, int]
    color_camera_matrix: np.ndarray
    color_distortion: np.ndarray
    depth_size: tuple[int, int] | None
    depth_camera_matrix: np.ndarray | None
    depth_distortion: np.ndarray | None
    T_color_depth: np.ndarray | None
    """Pose of the depth camera in the color camera frame (metres)."""
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        def arr(x):
            return None if x is None else np.asarray(x, dtype=np.float64).tolist()

        return {
            "vendor": "microsoft_azure_kinect",
            "serial": self.serial,
            "color_resolution": self.color_resolution,
            "depth_mode": self.depth_mode,
            "color": {
                "width": self.color_size[0],
                "height": self.color_size[1],
                "camera_matrix": arr(self.color_camera_matrix),
                "distortion_model": "opencv_rational_8",
                "distortion_coefficients": arr(self.color_distortion),
            },
            "depth": None
            if self.depth_size is None
            else {
                "width": self.depth_size[0],
                "height": self.depth_size[1],
                "camera_matrix": arr(self.depth_camera_matrix),
                "distortion_model": "opencv_rational_8",
                "distortion_coefficients": arr(self.depth_distortion),
            },
            "T_color_depth": arr(self.T_color_depth),
            "source": "azure_kinect_sdk_factory_calibration",
        }


def kinect_available() -> bool:
    try:
        import pyk4a  # noqa: F401
    except ImportError:
        return False
    return True


class KinectCamera:
    """Thin context manager around ``pyk4a.PyK4A`` with BGR frames and factory calibration."""

    def __init__(
        self,
        *,
        color_resolution: str = "1536P",
        depth_mode: str = "NFOV_UNBINNED",
        fps: int = 15,
        device_index: int = 0,
        expected_serial: str | None = None,
        synchronized_images_only: bool = True,
    ) -> None:
        if color_resolution not in COLOR_RESOLUTIONS:
            raise ValueError(f"Unknown color resolution {color_resolution!r}")
        if depth_mode not in DEPTH_MODES:
            raise ValueError(f"Unknown depth mode {depth_mode!r}")
        if fps not in (5, 15, 30):
            raise ValueError("Kinect FPS must be 5, 15 or 30")
        self.color_resolution = color_resolution
        self.depth_mode = depth_mode
        self.fps = fps
        self.device_index = device_index
        self.expected_serial = expected_serial
        self.synchronized_images_only = synchronized_images_only and depth_mode != "OFF"
        self._device = None
        self._calibration: KinectCalibration | None = None

    # ----------------------------------------------------------------- lifecycle
    def open(self) -> KinectCamera:
        try:
            import pyk4a
            from pyk4a import FPS, ColorResolution, Config, DepthMode, ImageFormat, PyK4A
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pyk4a is not installed; run `uv sync --extra kinect` (needs libk4a1.4-dev)"
            ) from exc
        if pyk4a.connected_device_count() <= self.device_index:
            raise ConnectionError(
                f"No Azure Kinect at index {self.device_index}; connected: {pyk4a.connected_device_count()}"
            )
        config = Config(
            color_resolution=getattr(ColorResolution, f"RES_{self.color_resolution}"),
            color_format=ImageFormat.COLOR_BGRA32,
            depth_mode=getattr(DepthMode, self.depth_mode),
            camera_fps={5: FPS.FPS_5, 15: FPS.FPS_15, 30: FPS.FPS_30}[self.fps],
            synchronized_images_only=self.synchronized_images_only,
        )
        device = PyK4A(config, device_id=self.device_index)
        device.start()
        serial = str(device.serial)
        if self.expected_serial and serial != self.expected_serial:
            device.stop()
            raise ConnectionError(f"Kinect serial {serial} != expected {self.expected_serial}")
        self._device = device
        self._calibration = self._read_calibration()
        return self

    def close(self) -> None:
        if self._device is not None:
            try:
                self._device.stop()
            finally:
                self._device = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------- frames
    def capture(self, timeout_s: float = 2.0) -> KinectFrame:
        if self._device is None:
            raise RuntimeError("KinectCamera is not open")
        deadline = time.monotonic() + timeout_s
        while True:
            cap = self._device.get_capture(timeout=int(max(1, timeout_s * 1000)))
            color = cap.color
            if color is not None and (self.depth_mode == "OFF" or cap.depth is not None):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("Kinect produced no complete capture")
        bgr = np.ascontiguousarray(color[:, :, :3])
        depth = None if cap.depth is None else np.ascontiguousarray(cap.depth)
        return KinectFrame(
            color_bgr=bgr,
            depth_mm=depth,
            color_timestamp_us=int(cap.color_timestamp_usec),
            depth_timestamp_us=None if depth is None else int(cap.depth_timestamp_usec),
            receive_time_s=time.time(),
        )

    # -------------------------------------------------------------- calibration
    @property
    def calibration(self) -> KinectCalibration:
        if self._calibration is None:
            raise RuntimeError("KinectCamera is not open")
        return self._calibration

    def _read_calibration(self) -> KinectCalibration:
        from pyk4a import CalibrationType

        assert self._device is not None
        calib = self._device.calibration
        color_K = np.asarray(calib.get_camera_matrix(CalibrationType.COLOR), dtype=np.float64)
        color_D = np.asarray(
            calib.get_distortion_coefficients(CalibrationType.COLOR), dtype=np.float64
        )
        color_size = COLOR_RESOLUTIONS[self.color_resolution]
        depth_K = depth_D = depth_size = T_color_depth = None
        if self.depth_mode != "OFF":
            depth_K = np.asarray(calib.get_camera_matrix(CalibrationType.DEPTH), dtype=np.float64)
            depth_D = np.asarray(
                calib.get_distortion_coefficients(CalibrationType.DEPTH), dtype=np.float64
            )
            R, t = calib.get_extrinsic_parameters(CalibrationType.DEPTH, CalibrationType.COLOR)
            T = np.eye(4)
            T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
            t = np.asarray(t, dtype=np.float64).reshape(3)
            # pyk4a returns metres in recent versions and millimetres in older ones.
            if np.max(np.abs(t)) > 1.0:
                t = t / 1000.0
            T[:3, 3] = t
            T_color_depth = T
            depth_size = {
                "NFOV_UNBINNED": (640, 576),
                "NFOV_2X2BINNED": (320, 288),
                "WFOV_UNBINNED": (1024, 1024),
                "WFOV_2X2BINNED": (512, 512),
            }[self.depth_mode]
        return KinectCalibration(
            serial=str(self._device.serial),
            color_resolution=self.color_resolution,
            depth_mode=self.depth_mode,
            color_size=color_size,
            color_camera_matrix=color_K,
            color_distortion=color_D,
            depth_size=depth_size,
            depth_camera_matrix=depth_K,
            depth_distortion=depth_D,
            T_color_depth=T_color_depth,
        )
