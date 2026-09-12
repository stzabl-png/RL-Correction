from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from dexmate_calib.boards.apriltag import (
    AprilTagGridDetector,
    AprilTagGridProfile,
    identify_markers,
)
from dexmate_calib.boards.config import (
    create_detector,
    load_board_profile,
    require_charuco,
    resolve_board_profile,
)
from dexmate_calib.extrinsics.handeye import pnp_board_pose_candidates, solve_hand_eye
from dexmate_calib.extrinsics.synthetic import make_scene
from dexmate_calib.geometry.transforms import pose_error, rt_to_T, so3_exp

K = np.array([[976.5, 0.0, 1019.1], [0.0, 976.3, 780.7], [0.0, 0.0, 1.0]])
SIZE = (2048, 1536)


def _warp_sheet(
    profile: AprilTagGridProfile, T_cam_board: np.ndarray, pixels_per_m: float = 3000.0
):
    """Render the printed sheet and place it in a synthetic camera view (D = 0)."""
    image, A = profile.render_image(pixels_per_m=pixels_per_m)
    h, w = image.shape
    inv = np.linalg.inv(A)
    sheet_uv = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float64)
    sheet_xy = (inv @ np.c_[sheet_uv, np.ones(4)].T).T[:, :2]
    rvec, _ = cv2.Rodrigues(T_cam_board[:3, :3])
    proj, _ = cv2.projectPoints(np.c_[sheet_xy, np.zeros(4)], rvec, T_cam_board[:3, 3], K, None)
    H = cv2.getPerspectiveTransform(
        sheet_uv.astype(np.float32), proj.reshape(4, 2).astype(np.float32)
    )
    frame = cv2.warpPerspective(image, H, SIZE, borderValue=255)
    return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)


def test_profiles_load_and_dispatch():
    grid = resolve_board_profile("apriltag-4x4")
    single = resolve_board_profile("single-tag-75")
    charuco = resolve_board_profile("dexmate-10x7")
    assert isinstance(grid, AprilTagGridProfile) and grid.target_type == "apriltag_grid"
    assert grid.rows == grid.cols == 4 and grid.tag_ids == list(range(16))
    assert grid.expected_corner_count == 64
    assert single.is_single_tag and single.expected_corner_count == 4
    assert single.tags[0].size_m == pytest.approx(0.075)
    assert charuco.target_type == "charuco"
    assert isinstance(create_detector(grid), AprilTagGridDetector)
    assert type(create_detector(charuco)).__name__ == "CharucoDetector"
    with pytest.raises(ValueError, match="ChArUco"):
        require_charuco(grid, "intrinsics")
    # Single tag centred at the origin, corners TL, TR, BR, BL in a y-down frame.
    corners = single.object_points()
    assert np.allclose(corners[:, 2], 0.0)
    assert np.allclose(corners[0, :2], [-0.0375, -0.0375]) and np.allclose(
        corners[2, :2], [0.0375, 0.0375]
    )


def test_profile_validation_errors(tmp_path: Path):
    base = yaml.safe_load(Path("configs/boards/apriltag_36h11_single_75mm.yaml").read_text())
    for mutate, match in [
        (lambda d: d["apriltag"].__setitem__("family", "tag99"), "family"),
        (lambda d: d["layout"].__setitem__("tag_size_m", -1.0), "tag_size_m"),
        (lambda d: d["layout"].__setitem__("tag_id_start", 600), "capacity"),
        (lambda d: d["layout"].__setitem__("rows", 0), "rows"),
    ]:
        data = yaml.safe_load(yaml.safe_dump(base))
        mutate(data)
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump(data))
        with pytest.raises(ValueError, match=match):
            load_board_profile(path)
    # Explicit tag list with two different sizes.
    data = {
        "schema_version": 1,
        "name": "mixed",
        "target_type": "apriltag_grid",
        "apriltag": {"family": "tag36h11"},
        "tags": [
            {"id": 3, "center_m": [0.0, 0.0], "size_m": 0.05},
            {"id": 9, "center_m": [0.1, 0.0], "size_m": 0.03, "col": 1},
        ],
    }
    path = tmp_path / "mixed.yaml"
    path.write_text(yaml.safe_dump(data))
    profile = load_board_profile(path)
    assert profile.tag_ids == [3, 9] and profile.cols == 2
    assert profile.object_points().shape == (8, 3)


@pytest.mark.parametrize("name", ["apriltag-4x4", "single-tag-75"])
def test_render_detect_and_corner_geometry(name: str):
    profile = resolve_board_profile(name)
    T = rt_to_T(so3_exp([0.3, -0.2, 0.1]), [-0.05, 0.02, 0.8])
    frame = _warp_sheet(profile, T)
    detector = create_detector(profile)
    detection = detector.detect(frame)
    assert detection is not None
    assert detection.marker_count == len(profile.tags)
    assert detection.corner_count == 4 * len(profile.tags)
    assert (detection.grid_rows, detection.grid_cols) == (profile.rows, profile.cols)
    obj, img = detector.calibration_points(detection)
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    expected, _ = cv2.projectPoints(obj, rvec, T[:3, 3], K, None)
    err = np.linalg.norm(expected.reshape(-1, 2) - img, axis=1)
    assert err.max() < 1.5, err.max()
    # Drawing must not fail and must not modify the input.
    before = frame.copy()
    detector.draw(frame, detection)
    assert np.array_equal(before, frame)
    # PnP on the detected corners recovers the pose (both IPPE candidates are returned).
    cands = pnp_board_pose_candidates(obj, img, K, None)
    assert 1 <= len(cands) <= 2
    rot_deg, trans_m = pose_error(cands[0], T)
    assert rot_deg < 0.5 and trans_m < 0.003


def test_identify_markers_finds_family_and_id():
    profile = resolve_board_profile("single-tag-75")
    frame = _warp_sheet(profile, rt_to_T(so3_exp([0.1, 0.05, 0.0]), [0.0, 0.0, 0.7]))
    found = identify_markers(frame)
    expected_id = profile.tags[0].tag_id
    hits = [f for f in found if f["dictionary"] == "DICT_APRILTAG_36h11" and f["id"] == expected_id]
    assert hits, found
    assert 60 < hits[0]["side_px"] < 130


def test_single_tag_hand_eye_with_flip_resolution():
    scene = make_scene(
        views=40,
        pixel_noise_px=0.7,
        seed=3,
        target="single_tag",
        depth_range_m=(0.8, 1.1),
        max_tilt_rad=0.25,
    )
    solution = solve_hand_eye(
        scene.views, scene.camera_matrix, scene.dist_coeffs, leave_one_out=False
    )
    rot_deg, trans_m = pose_error(solution.T_base_cam, scene.T_base_cam)
    assert solution.initialisation["pnp_flips_resolved"] > 0
    assert rot_deg < 0.15, rot_deg
    assert trans_m < 0.004, trans_m


def test_apriltag_grid_hand_eye():
    scene = make_scene(views=20, pixel_noise_px=0.3, seed=4, target="apriltag_4x4")
    solution = solve_hand_eye(
        scene.views, scene.camera_matrix, scene.dist_coeffs, leave_one_out=False
    )
    rot_deg, trans_m = pose_error(solution.T_base_cam, scene.T_base_cam)
    assert rot_deg < 0.05 and trans_m < 0.002
