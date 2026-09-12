from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoardImageQuality:
    grid_rows: int
    grid_cols: int
    board_bbox_fraction: float
    image_coverage_fraction: float
    centroid_xy: tuple[float, float]
    orientation_rad: float
    pixels_per_square: float
    roi_laplacian_var: float
    rectified_laplacian_var: float
    rectified_tenengrad_mean: float


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    scale = max(1.4826 * mad, float(np.std(finite)) * 0.1, 1e-6)
    return median, scale


def board_spatial_metrics(
    ids: np.ndarray,
    squares_x: int,
    squares_y: int,
) -> tuple[int, int, float]:
    ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
    inner_cols = int(squares_x) - 1
    inner_rows = int(squares_y) - 1
    valid = (ids_flat >= 0) & (ids_flat < inner_cols * inner_rows)
    ids_flat = ids_flat[valid]
    if ids_flat.size == 0:
        return 0, 0, 0.0
    rows = ids_flat // inner_cols
    cols = ids_flat % inner_cols
    row_count = int(np.unique(rows).size)
    col_count = int(np.unique(cols).size)
    bbox_rows = int(np.max(rows) - np.min(rows) + 1)
    bbox_cols = int(np.max(cols) - np.min(cols) + 1)
    bbox_fraction = float((bbox_rows * bbox_cols) / max(inner_rows * inner_cols, 1))
    return row_count, col_count, bbox_fraction


def estimate_pixels_per_square(
    points: np.ndarray,
    ids: np.ndarray,
    squares_x: int,
) -> float:
    import cv2

    points_2d = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
    by_id = {int(corner_id): point for corner_id, point in zip(ids_flat, points_2d)}
    inner_cols = int(squares_x) - 1
    distances: list[float] = []
    for corner_id, point in by_id.items():
        right_id = corner_id + 1
        if corner_id // inner_cols == right_id // inner_cols and right_id in by_id:
            distances.append(float(np.linalg.norm(by_id[right_id] - point)))
        down_id = corner_id + inner_cols
        if down_id in by_id:
            distances.append(float(np.linalg.norm(by_id[down_id] - point)))
    if distances:
        return float(np.median(distances))
    if len(points_2d) >= 3:
        hull_area = float(cv2.contourArea(cv2.convexHull(points_2d.astype(np.float32))))
        return float(math.sqrt(max(hull_area, 1.0) / max(len(points_2d), 1)))
    return 0.0


def _rectified_sharpness(
    gray: np.ndarray,
    points: np.ndarray,
    ids: np.ndarray,
    squares_x: int,
    squares_y: int,
) -> tuple[float, float]:
    import cv2

    points_2d = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    ids_flat = np.asarray(ids, dtype=np.int32).reshape(-1)
    if len(points_2d) < 4:
        return math.nan, math.nan
    inner_cols = int(squares_x) - 1
    pixels_per_square = 80.0
    canonical = np.column_stack(
        (
            (ids_flat % inner_cols + 1) * pixels_per_square,
            (ids_flat // inner_cols + 1) * pixels_per_square,
        )
    ).astype(np.float32)
    homography, _ = cv2.findHomography(points_2d, canonical, method=0)
    if homography is None:
        return math.nan, math.nan
    output_size = (
        round(squares_x * pixels_per_square),
        round(squares_y * pixels_per_square),
    )
    rectified = cv2.warpPerspective(gray, homography, output_size, flags=cv2.INTER_LINEAR)
    mask = np.zeros_like(rectified, dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.round(canonical).astype(np.int32)), 255)
    mask = cv2.erode(mask, np.ones((5, 5), dtype=np.uint8))
    valid = mask > 0
    if int(np.count_nonzero(valid)) < 100:
        return math.nan, math.nan
    laplacian = cv2.Laplacian(rectified, cv2.CV_32F)
    grad_x = cv2.Sobel(rectified, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(rectified, cv2.CV_32F, 0, 1, ksize=3)
    return (
        float(np.var(laplacian[valid])),
        float(np.mean((grad_x * grad_x + grad_y * grad_y)[valid])),
    )


def measure_board_image_quality(
    gray: np.ndarray,
    points: np.ndarray,
    ids: np.ndarray,
    squares_x: int,
    squares_y: int,
    *,
    detailed: bool = True,
) -> BoardImageQuality:
    import cv2

    points_2d = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x_min, y_min = points_2d.min(axis=0)
    x_max, y_max = points_2d.max(axis=0)
    height, width = gray.shape
    coverage = float(max(0.0, x_max - x_min) * max(0.0, y_max - y_min) / (width * height))

    pad = 8
    x0 = max(0, int(x_min) - pad)
    y0 = max(0, int(y_min) - pad)
    x1 = min(width, int(x_max) + pad + 1)
    y1 = min(height, int(y_max) + pad + 1)
    roi = gray[y0:y1, x0:x1]
    roi_laplacian = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0

    centered = points_2d - points_2d.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, -1]
    orientation = float(math.atan2(float(axis[1]), float(axis[0])))
    grid_rows, grid_cols, bbox_fraction = board_spatial_metrics(ids, squares_x, squares_y)
    if detailed:
        rectified_laplacian, tenengrad = _rectified_sharpness(
            gray, points_2d, ids, squares_x, squares_y
        )
    else:
        rectified_laplacian, tenengrad = math.nan, math.nan
    return BoardImageQuality(
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        board_bbox_fraction=bbox_fraction,
        image_coverage_fraction=coverage,
        centroid_xy=(float(points_2d[:, 0].mean()), float(points_2d[:, 1].mean())),
        orientation_rad=orientation,
        pixels_per_square=estimate_pixels_per_square(points_2d, ids, squares_x),
        roi_laplacian_var=roi_laplacian,
        rectified_laplacian_var=rectified_laplacian,
        rectified_tenengrad_mean=tenengrad,
    )


def pose_signature(
    points: np.ndarray,
    image_size: tuple[int, int],
) -> np.ndarray:
    points_2d = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    width, height = image_size
    x_min, y_min = points_2d.min(axis=0)
    x_max, y_max = points_2d.max(axis=0)
    coverage = float(max(0.0, x_max - x_min) * max(0.0, y_max - y_min) / (width * height))
    centered = points_2d - points_2d.mean(axis=0, keepdims=True)
    _, eigenvectors = np.linalg.eigh(centered.T @ centered)
    axis = eigenvectors[:, -1]
    orientation = math.atan2(float(axis[1]), float(axis[0]))
    return np.asarray(
        [
            float(points_2d[:, 0].mean()) / width,
            float(points_2d[:, 1].mean()) / height,
            math.sqrt(max(0.0, coverage)),
            0.5 * math.sin(2.0 * orientation),
            0.5 * math.cos(2.0 * orientation),
        ],
        dtype=np.float64,
    )


def select_pose_diverse_indices(features: np.ndarray, limit: int) -> list[int]:
    vectors = np.asarray(features, dtype=np.float64)
    if vectors.ndim != 2:
        raise ValueError("features must be a 2D array")
    count = len(vectors)
    if limit <= 0:
        raise ValueError("limit must be positive")
    if count <= limit:
        return list(range(count))
    seed = int(np.argmax(vectors[:, 2]))
    selected = [seed]
    minimum_distance = np.linalg.norm(vectors - vectors[seed], axis=1)
    minimum_distance[seed] = -np.inf
    while len(selected) < limit:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.linalg.norm(vectors - vectors[next_index], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf
    return sorted(selected)


def scale_aware_blur_scores(
    laplacian_values: np.ndarray,
    tenengrad_values: np.ndarray,
    pixels_per_square: np.ndarray,
) -> np.ndarray:
    laplacian = np.asarray(laplacian_values, dtype=np.float64)
    tenengrad = np.asarray(tenengrad_values, dtype=np.float64)
    scale = np.asarray(pixels_per_square, dtype=np.float64)
    if not (laplacian.shape == tenengrad.shape == scale.shape):
        raise ValueError("sharpness arrays must have identical shapes")
    scores = np.full(laplacian.shape, np.nan, dtype=np.float64)
    valid = (
        np.isfinite(laplacian)
        & np.isfinite(tenengrad)
        & np.isfinite(scale)
        & (laplacian > 0)
        & (tenengrad > 0)
        & (scale > 0)
    )
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return scores
    if valid_indices.size >= 24:
        bin_count = 3
    elif valid_indices.size >= 12:
        bin_count = 2
    else:
        bin_count = 1
    order = valid_indices[np.argsort(scale[valid_indices])]
    for indices in np.array_split(order, bin_count):
        if len(indices) == 0:
            continue
        log_laplacian = np.log(laplacian[indices])
        log_tenengrad = np.log(tenengrad[indices])
        lap_median, lap_scale = robust_location_scale(log_laplacian)
        ten_median, ten_scale = robust_location_scale(log_tenengrad)
        scores[indices] = 0.5 * (
            (log_laplacian - lap_median) / lap_scale + (log_tenengrad - ten_median) / ten_scale
        )
    return scores
