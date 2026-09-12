from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from dexmate_calib.boards.config import load_board_profile, resolve_board_profile
from dexmate_calib.intrinsics.capture import render_capture_preview
from dexmate_calib.intrinsics.detector import CharucoDetector


def test_verified_dexmate_board_profile() -> None:
    profile = resolve_board_profile("dexmate-10x7")
    assert profile.squares_x == 10
    assert profile.squares_y == 7
    assert profile.square_length_m == pytest.approx(0.027)
    assert profile.marker_length_m == pytest.approx(0.02025)
    assert profile.dictionary_name == "DICT_5X5_50"
    assert profile.expected_marker_count == 35
    assert profile.expected_corner_count == 54
    assert len(profile.sha256) == 64
    board, _ = profile.create_opencv_board()
    assert len(board.getIds()) == 35
    assert len(board.getChessboardCorners()) == 54


def test_detector_recognizes_generated_verified_board() -> None:
    profile = resolve_board_profile("dexmate-10x7")
    board, _ = profile.create_opencv_board()
    rendered = board.generateImage((1080, 756), marginSize=20, borderBits=1)
    image = cv2.cvtColor(np.asarray(rendered), cv2.COLOR_GRAY2BGR)
    detection = CharucoDetector(profile).detect(image)
    assert detection is not None
    assert detection.marker_count == 35
    assert detection.corner_count == 54
    assert detection.grid_rows == 6
    assert detection.grid_cols == 9
    assert detection.board_bbox_fraction == pytest.approx(1.0)
    assert detection.pixels_per_square > 0
    assert np.isfinite(detection.rectified_laplacian_var)
    fast_detection = CharucoDetector(profile).detect(image, detailed_quality=False)
    assert fast_detection is not None
    assert np.isnan(fast_detection.rectified_laplacian_var)
    assert np.isnan(fast_detection.rectified_tenengrad_mean)
    preview = render_capture_preview(
        CharucoDetector(profile),
        image,
        detection,
        accepted=1,
        target=40,
        state="ready",
    )
    assert preview.shape == image.shape
    assert np.any(preview != image)


def test_rejects_marker_larger_than_square(tmp_path: Path) -> None:
    source = resolve_board_profile("dexmate-10x7")
    data = dict(source.data)
    data["geometry"] = dict(data["geometry"])
    data["geometry"]["marker_length_m"] = 0.030
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="marker_length_m"):
        load_board_profile(path)


def test_rejects_physical_size_mismatch(tmp_path: Path) -> None:
    source = resolve_board_profile("dexmate-10x7")
    data = dict(source.data)
    data["physical"] = dict(data["physical"])
    data["physical"]["board_outer_width_m"] = 0.3
    path = tmp_path / "bad-size.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="board_outer_width_m"):
        load_board_profile(path)
