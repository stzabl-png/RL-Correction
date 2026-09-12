from __future__ import annotations

import numpy as np
import pytest

from dexmate_calib.extrinsics.handeye import (
    initial_hand_eye,
    pnp_board_pose,
    solve_hand_eye,
)
from dexmate_calib.extrinsics.synthetic import make_scene
from dexmate_calib.geometry.transforms import pose_error


def test_pnp_recovers_board_pose():
    scene = make_scene(views=3, pixel_noise_px=0.0, seed=3)
    for view in scene.views:
        T = pnp_board_pose(
            view.object_points, view.image_points, scene.camera_matrix, scene.dist_coeffs
        )
        expected = np.linalg.inv(scene.T_base_cam) @ view.T_base_link @ scene.T_link_board
        rot_deg, trans_m = pose_error(T, expected)
        assert rot_deg < 1e-3
        assert trans_m < 1e-4


def test_closed_form_initialisation_convention():
    """If the OpenCV role mapping were wrong the closed-form X would be far off."""
    scene = make_scene(views=15, pixel_noise_px=0.0, seed=4)
    for view in scene.views:
        view.T_cam_board_pnp = pnp_board_pose(
            view.object_points, view.image_points, scene.camera_matrix, scene.dist_coeffs
        )
    candidates = initial_hand_eye(scene.views)
    assert candidates
    best = min(candidates, key=lambda c: pose_error(c[1], scene.T_base_cam)[1])
    rot_deg, trans_m = pose_error(best[1], scene.T_base_cam)
    assert rot_deg < 0.05
    assert trans_m < 0.002
    rot_deg, trans_m = pose_error(best[2], scene.T_link_board)
    assert rot_deg < 0.05
    assert trans_m < 0.002


def test_solve_noise_free_is_exact():
    scene = make_scene(views=16, pixel_noise_px=0.0, seed=5)
    solution = solve_hand_eye(
        scene.views, scene.camera_matrix, scene.dist_coeffs, leave_one_out=False
    )
    rot_deg, trans_m = pose_error(solution.T_base_cam, scene.T_base_cam)
    assert rot_deg < 1e-3
    assert trans_m < 1e-4
    assert solution.rms_px < 1e-3
    assert not solution.rejected_views


def test_solve_with_pixel_and_fk_noise():
    scene = make_scene(
        views=25,
        pixel_noise_px=0.4,
        fk_rotation_noise_deg=0.03,
        fk_translation_noise_m=0.0005,
        seed=6,
    )
    solution = solve_hand_eye(scene.views, scene.camera_matrix, scene.dist_coeffs)
    rot_deg, trans_m = pose_error(solution.T_base_cam, scene.T_base_cam)
    assert rot_deg < 0.08, rot_deg
    assert trans_m < 0.002, trans_m
    assert solution.rms_px < 1.3
    assert solution.refinement["converged"]
    # Refinement must not be worse than the closed-form start.
    assert solution.rms_px <= solution.initialisation["rms_px"] + 1e-9
    loo = solution.diagnostics["leave_one_out"]
    assert loo["T_base_cam_translation_spread_mm"]["max"] < 5.0
    assert solution.diagnostics["motion_diversity"]["rotation_axis_rank"] == 3


def test_solve_rejects_outlier_views():
    scene = make_scene(views=24, pixel_noise_px=0.3, outliers=3, seed=7)
    solution = solve_hand_eye(
        scene.views, scene.camera_matrix, scene.dist_coeffs, leave_one_out=False
    )
    rejected = {r["view"] for r in solution.rejected_views}
    assert set(scene.outlier_views) <= rejected, (scene.outlier_views, solution.rejected_views)
    rot_deg, trans_m = pose_error(solution.T_base_cam, scene.T_base_cam)
    assert rot_deg < 0.1
    assert trans_m < 0.003


def test_solve_requires_minimum_views():
    scene = make_scene(views=5, seed=8)
    with pytest.raises(ValueError):
        solve_hand_eye(scene.views, scene.camera_matrix, scene.dist_coeffs, min_views=8)


def test_robotcamcalib_method_matches_reprojection_on_board():
    """Verbatim RobotCamCalib solver on the same detections lands close to ours."""
    scene = make_scene(views=25, pixel_noise_px=0.4, seed=9)
    rcc = solve_hand_eye(
        scene.views,
        scene.camera_matrix,
        scene.dist_coeffs,
        leave_one_out=False,
        method="robotcamcalib",
    )
    repro = solve_hand_eye(
        scene.views,
        scene.camera_matrix,
        scene.dist_coeffs,
        leave_one_out=False,
        method="reprojection",
    )
    assert rcc.refinement["method"] == "robotcamcalib"
    assert rcc.initialisation["method"] == "robotcamcalib_alternating_ls"
    assert rcc.refinement["iterations"] >= 1
    for sol in (rcc, repro):
        rot_deg, trans_m = pose_error(sol.T_base_cam, scene.T_base_cam)
        assert rot_deg < 0.1, rot_deg
        assert trans_m < 0.003, trans_m
    rot_deg, trans_m = pose_error(rcc.T_base_cam, repro.T_base_cam)
    assert rot_deg < 0.1 and trans_m < 0.003
    with pytest.raises(ValueError):
        solve_hand_eye(scene.views, scene.camera_matrix, scene.dist_coeffs, method="bogus")


def test_robotcamcalib_port_matches_reference_numerics():
    """The ported functions reproduce RobotCamCalib on its own synthetic-style inputs."""
    from dexmate_calib.extrinsics import robotcamcalib as rcc

    rng = np.random.default_rng(0)
    from dexmate_calib.geometry.transforms import rt_to_T, so3_exp

    X_true = rt_to_T(so3_exp(rng.normal(size=3) * 0.5), rng.normal(size=3) * 0.3)  # X_CammountCam
    Y_true = rt_to_T(so3_exp(rng.normal(size=3) * 0.5), rng.normal(size=3) * 0.1)  # X_TagmountTag
    n = 20
    A = np.stack([rt_to_T(so3_exp(rng.normal(size=3)), rng.normal(size=3)) for _ in range(n)])
    # A X_tag = X_cam B  ->  B = X_cam^-1 A X_tag ; add small noise to B
    B = np.stack(
        [
            rcc.inv_T(X_true)
            @ A[i]
            @ Y_true
            @ rt_to_T(so3_exp(rng.normal(size=3) * 1e-3), rng.normal(size=3) * 1e-4)
            for i in range(n)
        ]
    )
    Xc, Xt, info = rcc.calibrate_cammount_and_tag_prob(B, np.repeat(np.eye(4)[None], n, 0), A)
    rot_deg, trans_m = pose_error(Xc, X_true)
    assert rot_deg < 0.1 and trans_m < 0.002
    rot_deg, trans_m = pose_error(Xt, Y_true)
    assert rot_deg < 0.1 and trans_m < 0.002
    assert info["iters"] < 200
