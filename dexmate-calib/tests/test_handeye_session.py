from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from dexmate_calib.boards.config import resolve_board_profile
from dexmate_calib.extrinsics.capture import HandEyeCaptureSettings, capture_handeye_session
from dexmate_calib.extrinsics.solve import solve_handeye_session
from dexmate_calib.extrinsics.synthetic import make_scene
from dexmate_calib.geometry.transforms import inv_T, pose_error


class FakeCalibration:
    def __init__(self, K, size):
        self.K = K
        self.size = size

    def as_dict(self):
        return {
            "vendor": "fake",
            "serial": "FAKE0001",
            "color_resolution": "1536P",
            "depth_mode": "OFF",
            "color": {
                "width": self.size[0],
                "height": self.size[1],
                "camera_matrix": self.K.tolist(),
                "distortion_model": "none",
                "distortion_coefficients": [0.0] * 5,
            },
            "depth": None,
            "T_color_depth": np.eye(4).tolist(),
        }


@dataclass
class FakeFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray | None
    color_timestamp_us: int
    depth_timestamp_us: int | None
    receive_time_s: float


class FakeCamera:
    """Serves one rendered board image per capture, advancing after each save trigger."""

    def __init__(self, images, K, size):
        self.images = images
        self.calibration = FakeCalibration(K, size)
        self.calls = 0

    def capture(self):
        index = min(self.calls // 2, len(self.images) - 1)  # 2 captures per accepted sample
        self.calls += 1
        return FakeFrame(self.images[index], None, self.calls * 1000, None, float(self.calls))


@dataclass
class FakeJointSample:
    positions: dict
    stationary: bool = True
    max_motion_rad: float = 0.0
    read_time_s: float = 0.0
    checks: int = 3


class FakeJoints:
    def __init__(self, count):
        self.count = count
        self.index = 0

    def read_stationary(self):
        sample = FakeJointSample({"view": float(min(self.index, self.count - 1))})
        self.index += 1
        return sample


class FakeKinematics:
    """Maps the synthetic ``view`` index stored in the joint dict to its known link pose."""

    def __init__(self, poses):
        self.poses = poses

    def required_joints_for(self, frame):
        return ["view"]

    def frame_pose(self, positions, frame, *, strict=True):
        return self.poses[round(positions["view"])]


def _render(board_profile, T_cam_board, K, size):
    if board_profile.target_type == "apriltag_grid":
        board_image, A = board_profile.render_image(pixels_per_m=3000.0)
        h, w = board_image.shape
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        inv = np.linalg.inv(A)
        xy = (inv @ np.c_[src.astype(np.float64), np.ones(4)].T).T[:, :2]
        corners_board = np.c_[xy, np.zeros(4)]
    else:
        board, _ = board_profile.create_opencv_board()
        board_image = board.generateImage((1000, 700), marginSize=0, borderBits=1)
        src = np.array([[0, 0], [999, 0], [999, 699], [0, 699]], dtype=np.float32)
        corners_board = np.array(
            [[0, 0, 0], [0.270, 0, 0], [0.270, 0.189, 0], [0, 0.189, 0]], dtype=np.float64
        )
    rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
    projected, _ = cv2.projectPoints(corners_board, rvec, T_cam_board[:3, 3], K, None)
    H = cv2.getPerspectiveTransform(src, projected.reshape(4, 2).astype(np.float32))
    gray = cv2.warpPerspective(board_image, H, size, flags=cv2.INTER_LINEAR, borderValue=255)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize(
    ("board_alias", "target", "views", "depth"),
    [
        ("dexmate-10x7", "charuco", 14, (0.6, 1.2)),
        ("single-tag-75", "single_tag", 24, (0.5, 0.8)),
    ],
)
def test_capture_and_solve_synthetic_session(tmp_path: Path, board_alias, target, views, depth):
    profile = resolve_board_profile(board_alias)
    scene = make_scene(
        views=views,
        pixel_noise_px=0.0,
        seed=21,
        max_tilt_rad=0.35,
        target=target,
        depth_range_m=depth,
    )
    K = scene.camera_matrix
    size = scene.image_size
    # Render undistorted images (D = 0) so the homography warp is exact.
    images, poses = [], []
    for view in scene.views:
        T_cam_board = inv_T(scene.T_base_cam) @ view.T_base_link @ scene.T_link_board
        images.append(_render(profile, T_cam_board, K, size))
        poses.append(view.T_base_link)

    settings = HandEyeCaptureSettings(
        output_root=tmp_path / "sessions",
        max_samples=len(images),
        preview=False,
        auto_capture=True,
        auto_capture_interval_s=0.0,
        save_depth=False,
        min_corners=min(20, profile.expected_corner_count),
    )
    camera = FakeCamera(images, K, size)
    joints = FakeJoints(len(images))
    session_dir = capture_handeye_session(
        profile, camera, joints, settings, key_source=lambda ms: -1
    )

    manifest = yaml.safe_load((session_dir / "manifest.yaml").read_text())
    assert manifest["status"] == "completed"
    assert manifest["capture"]["accepted_samples"] == len(images)
    records = [json.loads(l) for l in (session_dir / "samples.jsonl").read_text().splitlines()]
    assert len(records) == len(images)
    assert all(r["joints"]["positions_rad"]["view"] == float(i) for i, r in enumerate(records))
    assert (session_dir / "camera_calibration.json").exists()

    result = solve_handeye_session(
        session_dir,
        kinematics=FakeKinematics(poses),
        target_link="L_ee",
        min_views=8,
        leave_one_out=True,
    )
    rot_deg, trans_m = pose_error(result.solution.T_base_cam, scene.T_base_cam)
    assert rot_deg < 0.1, rot_deg
    assert trans_m < 0.003, trans_m
    assert result.solution.rms_px < 1.0
    assert manifest["board"]["target_type"] == profile.target_type
    for name in (
        "handeye_result.json",
        "T_base_kinect_external.yaml",
        "per_view.csv",
        "T_base_cam.npy",
        "reprojection_contact_sheet.jpg",
    ):
        assert (result.results_dir / name).exists(), name
    saved = yaml.safe_load((result.results_dir / "T_base_kinect_external.yaml").read_text())
    assert np.allclose(np.array(saved["T_base_cam"]), result.solution.T_base_cam)
    assert saved["views_used"] == len(result.solution.inlier_views)
    assert saved["method"] == "reprojection"
    # The RobotCamCalib-style solver runs on the same session and lands close by.
    alt = solve_handeye_session(
        session_dir,
        kinematics=FakeKinematics(poses),
        target_link="L_ee",
        min_views=8,
        leave_one_out=False,
        method="robotcamcalib",
    )
    assert (alt.results_dir / "handeye_result_robotcamcalib.json").exists()
    rot_deg, trans_m = pose_error(alt.solution.T_base_cam, result.solution.T_base_cam)
    assert rot_deg < 0.3 and trans_m < 0.01
    print(result)
