from __future__ import annotations

import cv2
import numpy as np

from dexmate_calib.intrinsics.solve import Observation, calibrate_observations


def test_rectified_solver_recovers_pinhole_intrinsics() -> None:
    rng = np.random.default_rng(7)
    width, height = 1920, 1200
    true_k = np.array([[738.0, 0.0, 959.0], [0.0, 739.0, 601.0], [0.0, 0.0, 1.0]])
    grid = np.array(
        [[x * 0.027, y * 0.027, 0.0] for y in range(6) for x in range(9)],
        dtype=np.float32,
    )
    observations = []
    for index in range(28):
        rvec = rng.uniform([-0.45, -0.45, -0.3], [0.45, 0.45, 0.3]).astype(np.float64)
        tvec = np.array(
            [rng.uniform(-0.14, 0.06), rng.uniform(-0.10, 0.05), rng.uniform(0.45, 0.9)],
            dtype=np.float64,
        )
        points, _ = cv2.projectPoints(grid, rvec, tvec, true_k, np.zeros(5))
        image_points = points.reshape(-1, 2) + rng.normal(0.0, 0.08, (len(grid), 2))
        observations.append(
            Observation(f"synthetic_{index}.jpg", grid, image_points.astype(np.float32))
        )

    fit = calibrate_observations(observations, (width, height))
    assert fit.rms < 0.2
    np.testing.assert_allclose(fit.camera_matrix[0, 0], true_k[0, 0], rtol=0.01)
    np.testing.assert_allclose(fit.camera_matrix[1, 1], true_k[1, 1], rtol=0.01)
    np.testing.assert_allclose(fit.camera_matrix[:2, 2], true_k[:2, 2], atol=4.0)
    np.testing.assert_array_equal(fit.distortion, np.zeros(5))
