from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_selection_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "session",
        "image",
        "status",
        "reason",
        "corner_count",
        "grid_rows",
        "grid_cols",
        "board_bbox_fraction",
        "coverage_fraction",
        "pixels_per_square",
        "rectified_laplacian_var",
        "rectified_tenengrad_mean",
        "blur_score",
        "reprojection_error_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _fit_tile(image: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def downscale_for_diagnostics(
    image: np.ndarray,
    max_size: tuple[int, int] = (960, 600),
) -> np.ndarray:
    import cv2

    max_width, max_height = max_size
    scale = min(1.0, max_width / image.shape[1], max_height / image.shape[0])
    if scale >= 1.0:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def write_contact_sheet(
    path: Path,
    images: list[np.ndarray],
    labels: list[str],
    *,
    columns: int = 4,
    tile_size: tuple[int, int] = (360, 240),
) -> None:
    import cv2

    if not images:
        return
    columns = max(1, int(columns))
    tile_width, tile_height = tile_size
    label_height = 34
    rows = (len(images) + columns - 1) // columns
    sheet = np.zeros((rows * (tile_height + label_height), columns * tile_width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        x0 = column * tile_width
        y0 = row * (tile_height + label_height)
        sheet[y0 : y0 + tile_height, x0 : x0 + tile_width] = _fit_tile(
            image, tile_width, tile_height
        )
        cv2.putText(
            sheet,
            labels[index][:55],
            (x0 + 8, y0 + tile_height + 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(path), sheet)


def read_capture_previews(
    session_dir: Path, records: list[dict]
) -> tuple[list[np.ndarray], list[str]]:
    import cv2

    images: list[np.ndarray] = []
    labels: list[str] = []
    for record in records:
        relative = record.get("preview") or record.get("image")
        if not relative:
            continue
        image = cv2.imread(str(session_dir / relative), cv2.IMREAD_COLOR)
        if image is None:
            continue
        images.append(downscale_for_diagnostics(image))
        labels.append(
            f"{record.get('sample_index', len(labels))}: "
            f"corners={record.get('corner_count', '?')} "
            f"coverage={float(record.get('coverage_fraction', 0.0)):.3f}"
        )
    return images, labels
