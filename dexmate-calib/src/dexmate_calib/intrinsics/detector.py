from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dexmate_calib.boards.config import BoardProfile
from dexmate_calib.intrinsics.quality import measure_board_image_quality


@dataclass(frozen=True)
class Detection:
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray
    marker_count: int
    corner_count: int
    coverage_fraction: float
    sharpness: float
    centroid_xy: tuple[float, float]
    orientation_rad: float
    grid_rows: int
    grid_cols: int
    board_bbox_fraction: float
    pixels_per_square: float
    rectified_laplacian_var: float
    rectified_tenengrad_mean: float
    marker_corners: tuple[np.ndarray, ...]
    marker_ids: np.ndarray | None

    def signature(self, width: int, height: int) -> np.ndarray:
        return np.array(
            [
                self.centroid_xy[0] / width,
                self.centroid_xy[1] / height,
                math.sqrt(max(0.0, self.coverage_fraction)),
                0.5 * math.sin(2.0 * self.orientation_rad),
                0.5 * math.cos(2.0 * self.orientation_rad),
            ],
            dtype=np.float64,
        )


class CharucoDetector:
    def __init__(self, profile: BoardProfile) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("opencv-contrib-python is required") from exc
        self.cv2 = cv2
        self.profile = profile
        self.board, self.dictionary = profile.create_opencv_board()
        detector_params = cv2.aruco.DetectorParameters()
        # OpenCV's default minMarkerPerimeterRate (0.03 of the longest image side) discards
        # markers smaller than ~15 px at 1920-2048 px width, i.e. the 20 mm markers of the
        # dexmate-10x7 board beyond ~1.1 m. 0.01 keeps the board detectable to ~1.6 m at no
        # measurable speed cost.
        detector_params.minMarkerPerimeterRate = 0.01
        if hasattr(cv2.aruco, "CharucoParameters"):
            charuco_params = cv2.aruco.CharucoParameters()
            self.detector = cv2.aruco.CharucoDetector(self.board, charuco_params, detector_params)
        else:  # pragma: no cover - OpenCV < 4.7 compatibility
            self.detector = None
            self.aruco_detector = cv2.aruco.ArucoDetector(self.dictionary, detector_params)

    def detect(self, image_bgr: np.ndarray, *, detailed_quality: bool = True) -> Detection | None:
        cv2 = self.cv2
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if self.detector is not None:
            corners, ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
        else:  # pragma: no cover
            marker_corners, marker_ids, _ = self.aruco_detector.detectMarkers(gray)
            if marker_ids is None:
                return None
            _, corners, ids = cv2.aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self.board
            )
        if corners is None or ids is None or len(corners) < 4:
            return None

        points = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
        order = np.argsort(ids_flat)
        points = points[order]
        ids_flat = ids_flat[order]
        quality = measure_board_image_quality(
            gray,
            points,
            ids_flat,
            self.profile.squares_x,
            self.profile.squares_y,
            detailed=detailed_quality,
        )

        return Detection(
            charuco_corners=points.reshape(-1, 1, 2),
            charuco_ids=ids_flat.reshape(-1, 1),
            marker_count=0 if marker_ids is None else len(marker_ids),
            corner_count=len(points),
            coverage_fraction=quality.image_coverage_fraction,
            sharpness=quality.roi_laplacian_var,
            centroid_xy=quality.centroid_xy,
            orientation_rad=quality.orientation_rad,
            grid_rows=quality.grid_rows,
            grid_cols=quality.grid_cols,
            board_bbox_fraction=quality.board_bbox_fraction,
            pixels_per_square=quality.pixels_per_square,
            rectified_laplacian_var=quality.rectified_laplacian_var,
            rectified_tenengrad_mean=quality.rectified_tenengrad_mean,
            marker_corners=tuple(marker_corners) if marker_corners is not None else (),
            marker_ids=None if marker_ids is None else np.asarray(marker_ids, dtype=np.int32),
        )

    def draw(self, image_bgr: np.ndarray, detection: Detection | None) -> np.ndarray:
        output = image_bgr.copy()
        if detection is not None:
            if detection.marker_ids is not None and detection.marker_corners:
                self.cv2.aruco.drawDetectedMarkers(
                    output,
                    list(detection.marker_corners),
                    detection.marker_ids,
                )
            self.cv2.aruco.drawDetectedCornersCharuco(
                output, detection.charuco_corners, detection.charuco_ids
            )
        return output

    def calibration_points(self, detection: Detection) -> tuple[np.ndarray, np.ndarray]:
        chessboard = np.asarray(self.board.getChessboardCorners(), dtype=np.float32).reshape(-1, 3)
        ids = detection.charuco_ids.reshape(-1)
        if np.any(ids < 0) or np.any(ids >= len(chessboard)):
            raise ValueError("Detected ChArUco corner ID is outside board geometry")
        return chessboard[ids].copy(), detection.charuco_corners.reshape(-1, 2).copy()
