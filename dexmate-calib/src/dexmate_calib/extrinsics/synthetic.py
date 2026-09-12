"""Synthetic eye-to-hand scenes for tests and the ``extrinsics selftest`` command."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmate_calib.extrinsics.handeye import HandEyeView, project_board
from dexmate_calib.geometry.transforms import inv_T, rt_to_T, so3_exp


@dataclass
class SyntheticScene:
    views: list[HandEyeView]
    T_base_cam: np.ndarray
    T_link_board: np.ndarray
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    outlier_views: list[str]


def default_board_points(
    squares_x: int = 10, squares_y: int = 7, square_m: float = 0.027
) -> np.ndarray:
    """Inner ChArUco corners of the dexmate-10x7 board, matching OpenCV ordering."""
    pts = []
    for y in range(1, squares_y):
        for x in range(1, squares_x):
            pts.append([x * square_m, y * square_m, 0.0])
    return np.asarray(pts, dtype=np.float64)


def target_points(target: str) -> np.ndarray:
    """Board points for a named synthetic target."""
    if target == "charuco":
        return default_board_points()
    from dexmate_calib.boards.config import resolve_board_profile

    alias = {"apriltag_4x4": "apriltag-4x4", "single_tag": "single-tag-75"}.get(target, target)
    profile = resolve_board_profile(alias)
    if not hasattr(profile, "object_points"):
        raise ValueError(f"{target!r} is not an AprilTag profile")
    return profile.object_points()


def kinect_like_intrinsics() -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    K = np.array([[976.5, 0.0, 1019.1], [0.0, 976.3, 780.7], [0.0, 0.0, 1.0]])
    D = np.array([0.27, -2.47, 0.0006, -0.0002, 1.48, 0.154, -2.29, 1.40])
    return K, D, (2048, 1536)


def make_scene(
    *,
    views: int = 20,
    pixel_noise_px: float = 0.3,
    fk_rotation_noise_deg: float = 0.0,
    fk_translation_noise_m: float = 0.0,
    outliers: int = 0,
    seed: int = 0,
    max_tilt_rad: float = 0.6,
    target: str = "charuco",
    depth_range_m: tuple[float, float] = (0.6, 1.2),
) -> SyntheticScene:
    """Generate consistent (T_base_link_i, corner observations) for random X and Y.

    The camera sits ~1.5 m in front of the robot looking back at it; board poses are
    sampled directly in the camera frame so every view is in front of the camera,
    then propagated to link poses through the true X and Y.
    """
    rng = np.random.default_rng(seed)
    K, D, size = kinect_like_intrinsics()
    board = target_points(target)
    board_center = board.mean(axis=0)

    # Camera in base: in front, slightly above, looking back towards the robot (-x_base).
    R_bc = so3_exp(np.array([0.0, 0.0, np.pi])) @ so3_exp(np.array([-np.pi / 2 - 0.25, 0.0, 0.0]))
    R_bc = so3_exp(rng.normal(scale=0.05, size=3)) @ R_bc
    t_bc = np.array([1.5, 0.1, 1.2]) + rng.normal(scale=0.05, size=3)
    T_base_cam = rt_to_T(R_bc, t_bc)

    # Board on the link: rotated arbitrarily, offset a few cm.
    T_link_board = rt_to_T(
        so3_exp(rng.uniform(-1.0, 1.0, size=3)), rng.uniform(-0.05, 0.05, size=3)
    )

    view_list: list[HandEyeView] = []
    outlier_names: list[str] = []
    for i in range(views):
        # Board pose in the camera: facing the camera, tilted, 0.6-1.2 m away, spread over image.
        rvec = rng.uniform(-max_tilt_rad, max_tilt_rad, size=3)
        R_cb = so3_exp(rvec)
        depth = rng.uniform(*depth_range_m)
        # Keep the projected board centre inside the image.
        u = rng.uniform(0.2, 0.8) * size[0]
        v = rng.uniform(0.2, 0.8) * size[1]
        center_cam = np.array(
            [(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth]
        )
        t_cb = center_cam - R_cb @ board_center
        T_cam_board = rt_to_T(R_cb, t_cb)
        image_points = project_board(board, T_cam_board, K, D)
        image_points = image_points + rng.normal(scale=pixel_noise_px, size=image_points.shape)

        T_base_link = T_base_cam @ T_cam_board @ inv_T(T_link_board)
        if fk_rotation_noise_deg > 0.0 or fk_translation_noise_m > 0.0:
            dR = so3_exp(rng.normal(scale=np.radians(fk_rotation_noise_deg), size=3))
            dt = rng.normal(scale=fk_translation_noise_m, size=3)
            T_base_link = T_base_link @ rt_to_T(dR, dt)
        name = f"view_{i:03d}"
        if i < outliers:
            # Corrupt the FK pose badly, as if the arm moved between photo and joint read.
            T_base_link = T_base_link @ rt_to_T(
                so3_exp(np.array([0.0, 0.0, 0.15])), np.array([0.03, 0.0, 0.0])
            )
            outlier_names.append(name)
        view_list.append(HandEyeView(name, board, image_points, T_base_link))

    return SyntheticScene(view_list, T_base_cam, T_link_board, K, D, size, outlier_names)
