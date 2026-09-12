from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from dexmate_calib.boards.config import resolve_board_profile
from dexmate_calib.intrinsics.detector import CharucoDetector
from dexmate_calib.intrinsics.multi_solve import solve_sessions
from dexmate_calib.intrinsics.solve import (
    _load_observations,
    calibrate_observations,
    solve_session,
)
from dexmate_calib.intrinsics.validation import camera_matrix_difference


def _make_rectified_session(session_dir: Path, views: int = 18, seed: int = 11) -> None:
    profile = resolve_board_profile("dexmate-10x7")
    board, _ = profile.create_opencv_board()
    board_image = board.generateImage((1000, 700), marginSize=0, borderBits=1)
    source_corners = np.asarray(
        [[0.0, 0.0], [999.0, 0.0], [999.0, 699.0], [0.0, 699.0]],
        dtype=np.float32,
    )
    board_corners = np.asarray(
        [[0.0, 0.0, 0.0], [0.270, 0.0, 0.0], [0.270, 0.189, 0.0], [0.0, 0.189, 0.0]],
        dtype=np.float32,
    )
    camera_matrix = np.asarray(
        [[738.0, 0.0, 959.0], [0.0, 739.0, 601.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    raw_dir = session_dir / "raw"
    raw_dir.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    records: list[dict] = []
    for index in range(views):
        rvec = rng.uniform([-0.35, -0.35, -0.25], [0.35, 0.35, 0.25]).astype(np.float64)
        tvec = np.asarray(
            [
                -0.135 + rng.uniform(-0.12, 0.12),
                -0.0945 + rng.uniform(-0.07, 0.07),
                rng.uniform(0.48, 0.80),
            ],
            dtype=np.float64,
        )
        projected, _ = cv2.projectPoints(board_corners, rvec, tvec, camera_matrix, np.zeros(5))
        homography = cv2.getPerspectiveTransform(
            source_corners, projected.reshape(4, 2).astype(np.float32)
        )
        image = cv2.warpPerspective(
            board_image,
            homography,
            (1920, 1200),
            flags=cv2.INTER_LINEAR,
            borderValue=255,
        )
        image_name = f"raw/frame_{index:04d}.jpg"
        assert cv2.imwrite(
            str(session_dir / image_name),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )
        records.append({"sample_index": index, "image": image_name})

    board_file = session_dir / "board_profile.yaml"
    board_file.write_text(yaml.safe_dump(profile.data, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema": "dexmate_calib.intrinsic_session.v1",
        "camera": {
            "name": "head_left",
            "zed_mode": "HD1200",
            "width": 1920,
            "height": 1200,
            "image_view": "LEFT",
            "image_geometry": "rectified",
            "camera_serial": 59595115,
        },
        "board": {
            "name": profile.name,
            "profile_file": board_file.name,
            "profile_sha256": profile.sha256,
        },
    }
    (session_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    (session_dir / "frames.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_session_solver_preserves_dexmate_contract_and_writes_diagnostics(
    tmp_path: Path,
) -> None:
    session_dir = tmp_path / "session"
    _make_rectified_session(session_dir)
    result_path = solve_session(
        session_dir,
        min_views=12,
        max_views=16,
        cross_validation_folds=3,
    )
    result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    assert result["camera"]["zed_mode"] == "HD1200"
    assert result["camera"]["width"] == 1920
    assert result["camera"]["height"] == 1200
    assert result["camera"]["image_view"] == "LEFT"
    assert result["camera"]["image_geometry"] == "rectified"
    assert result["model"]["distortion_model"] == "none"
    assert result["model"]["distortion_coefficients"] == [0.0] * 5
    recovered_k = np.asarray(result["model"]["K"], dtype=np.float64)
    np.testing.assert_allclose(recovered_k[0, 0], 738.0, rtol=0.005)
    np.testing.assert_allclose(recovered_k[1, 1], 739.0, rtol=0.005)
    np.testing.assert_allclose(recovered_k[:2, 2], [959.0, 601.0], atol=2.0)
    assert result["quality"]["rms_reprojection_error_px"] < 0.5
    assert result["quality"]["held_out_median_error_px"] < 0.5
    profile = resolve_board_profile("dexmate-10x7")
    _manifest, image_size, observations, *_rest = _load_observations(
        session_dir, CharucoDetector(profile)
    )
    direct_baseline = calibrate_observations(observations, image_size)
    baseline_difference = camera_matrix_difference(recovered_k, direct_baseline.camera_matrix)
    assert baseline_difference["fx_relative_difference"] < 0.005
    assert baseline_difference["fy_relative_difference"] < 0.005
    assert baseline_difference["principal_point_difference_px"] < 3.0
    assert result["quality"]["cross_validation"]["fold_count"] == 3
    results_dir = session_dir / "results"
    for name in (
        "K.npy",
        "sample_selection.csv",
        "cross_validation.json",
        "quality_summary.json",
        "capture_contact_sheet.jpg",
        "reprojection_contact_sheet.jpg",
    ):
        assert (results_dir / name).is_file()


def test_multi_session_solver_jointly_fits_one_camera_matrix(tmp_path: Path) -> None:
    sessions = [tmp_path / f"session_{index}" for index in range(3)]
    for index, session in enumerate(sessions):
        _make_rectified_session(session, views=14, seed=21 + index)

    output = tmp_path / "pooled"
    result_path = solve_sessions(
        sessions,
        output=output,
        min_views=24,
        min_views_per_session=8,
        max_views_per_session=10,
        cross_validation_folds=3,
    )
    result = yaml.safe_load(result_path.read_text(encoding="utf-8"))
    assert result["schema"] == "dexmate_calib.intrinsics_multi.v1"
    assert result["sources"]["session_count"] == 3
    assert result["model"]["distortion_coefficients"] == [0.0] * 5
    recovered_k = np.asarray(result["model"]["K"], dtype=np.float64)
    np.testing.assert_allclose(recovered_k[0, 0], 738.0, rtol=0.005)
    np.testing.assert_allclose(recovered_k[1, 1], 739.0, rtol=0.005)
    np.testing.assert_allclose(recovered_k[:2, 2], [959.0, 601.0], atol=2.0)
    assert result["quality"]["views_used"] >= 24
    assert result["quality"]["leave_one_session_out"]["fold_count"] == 3
    assert result["quality"]["leave_one_session_out_median_error_px"] < 0.5
    assert set(result["quality"]["views_used_by_session"]) == {
        "session_0",
        "session_1",
        "session_2",
    }
    results_dir = output / "results"
    for name in (
        "K.npy",
        "sample_selection.csv",
        "cross_validation.json",
        "leave_one_session_out.json",
        "quality_summary.json",
        "capture_contact_sheet.jpg",
        "reprojection_contact_sheet.jpg",
    ):
        assert (results_dir / name).is_file()


def test_multi_session_solver_rejects_incompatible_camera_contract(tmp_path: Path) -> None:
    session_a = tmp_path / "session_a"
    session_b = tmp_path / "session_b"
    _make_rectified_session(session_a, views=3)
    _make_rectified_session(session_b, views=3, seed=12)
    manifest_path = session_b / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["camera"]["camera_serial"] = 12345678
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Incompatible session"):
        solve_sessions(
            [session_a, session_b],
            output=tmp_path / "pooled",
            min_views=6,
            min_views_per_session=3,
            max_views_per_session=3,
        )
