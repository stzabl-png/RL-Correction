from __future__ import annotations

import numpy as np
import pytest

from dexmate_calib.intrinsics.capture import CaptureSettings, detection_is_due
from dexmate_calib.intrinsics.quality import (
    board_spatial_metrics,
    scale_aware_blur_scores,
    select_pose_diverse_indices,
)
from dexmate_calib.intrinsics.validation import (
    camera_matrix_difference,
    deterministic_folds,
)


def test_full_dexmate_board_spatial_metrics() -> None:
    rows, cols, fraction = board_spatial_metrics(np.arange(54), 10, 7)
    assert rows == 6
    assert cols == 9
    assert fraction == 1.0


def test_pose_diverse_selection_is_deterministic() -> None:
    features = np.asarray(
        [
            [0.1, 0.1, 0.2, 0.0, 0.5],
            [0.5, 0.5, 0.5, 0.0, 0.5],
            [0.9, 0.1, 0.2, 0.0, 0.5],
            [0.1, 0.9, 0.2, 0.0, 0.5],
            [0.9, 0.9, 0.2, 0.0, 0.5],
        ]
    )
    first = select_pose_diverse_indices(features, 3)
    second = select_pose_diverse_indices(features, 3)
    assert first == second
    assert 1 in first  # Largest board coverage is the deterministic seed.


def test_scale_aware_blur_score_rejects_clear_low_tail() -> None:
    laplacian = np.asarray([1.0, *([100.0] * 11)])
    tenengrad = np.asarray([2.0, *([200.0] * 11)])
    scale = np.linspace(30.0, 80.0, 12)
    scores = scale_aware_blur_scores(laplacian, tenengrad, scale)
    assert scores[0] < -3.5
    assert np.nanmedian(scores[1:]) >= 0.0


def test_validation_folds_cover_every_sample_once() -> None:
    folds = deterministic_folds(20, 5)
    held_out = [index for _train, held in folds for index in held]
    assert sorted(held_out) == list(range(20))
    assert all(set(train).isdisjoint(held) for train, held in folds)


def test_camera_matrix_difference_reports_direction_free_magnitude() -> None:
    a = np.asarray([[700.0, 0.0, 960.0], [0.0, 702.0, 600.0], [0.0, 0.0, 1.0]])
    b = np.asarray([[703.5, 0.0, 962.0], [0.0, 698.5, 598.0], [0.0, 0.0, 1.0]])
    metrics = camera_matrix_difference(a, b)
    assert metrics["fx_relative_difference"] > 0
    assert metrics["fy_relative_difference"] > 0
    assert metrics["principal_point_difference_px"] == np.sqrt(8.0)


def test_auto_detection_rate_is_throttled_but_manual_is_not(tmp_path) -> None:
    automatic = CaptureSettings(output_root=tmp_path, detection_fps=10.0)
    assert not detection_is_due(1.05, 1.0, automatic)
    assert detection_is_due(1.11, 1.0, automatic)

    manual = CaptureSettings(output_root=tmp_path, auto_capture=False, detection_fps=10.0)
    assert detection_is_due(1.001, 1.0, manual)

    unthrottled = CaptureSettings(output_root=tmp_path, detection_fps=0.0)
    assert detection_is_due(1.001, 1.0, unthrottled)

    invalid = CaptureSettings(output_root=tmp_path, detection_fps=-1.0)
    with pytest.raises(ValueError, match="detection_fps"):
        detection_is_due(1.0, 0.0, invalid)
