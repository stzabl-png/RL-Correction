from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from dexmate_calib.boards.config import load_board_profile, require_charuco
from dexmate_calib.intrinsics.detector import CharucoDetector
from dexmate_calib.intrinsics.diagnostics import (
    downscale_for_diagnostics,
    read_capture_previews,
    write_contact_sheet,
    write_json,
    write_selection_csv,
)
from dexmate_calib.intrinsics.quality import (
    pose_signature,
    scale_aware_blur_scores,
    select_pose_diverse_indices,
)
from dexmate_calib.intrinsics.validation import (
    camera_matrix_difference,
    deterministic_folds,
    summarize_camera_matrices,
)


@dataclass(frozen=True)
class Observation:
    image_name: str
    object_points: np.ndarray
    image_points: np.ndarray
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationFit:
    rms: float
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rvecs: tuple[np.ndarray, ...]
    tvecs: tuple[np.ndarray, ...]
    per_view_errors: tuple[float, ...]


def _calibration_flags(cv2) -> int:
    return (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_K4
        | cv2.CALIB_FIX_K5
        | cv2.CALIB_FIX_K6
    )


def calibrate_observations(
    observations: Sequence[Observation], image_size: tuple[int, int]
) -> CalibrationFit:
    import cv2

    if len(observations) < 3:
        raise ValueError("At least 3 observations are required")
    width, height = image_size
    focal_guess = float(max(width, height))
    camera_matrix = np.array(
        [[focal_guess, 0.0, (width - 1) / 2], [0.0, focal_guess, (height - 1) / 2], [0, 0, 1]],
        dtype=np.float64,
    )
    distortion = np.zeros((8, 1), dtype=np.float64)
    rms, camera_matrix, _distortion, rvecs, tvecs = cv2.calibrateCamera(
        [np.asarray(obs.object_points, dtype=np.float32) for obs in observations],
        [np.asarray(obs.image_points, dtype=np.float32) for obs in observations],
        (width, height),
        camera_matrix,
        distortion,
        flags=_calibration_flags(cv2),
    )
    errors = []
    for obs, rvec, tvec in zip(observations, rvecs, tvecs):
        projected, _ = cv2.projectPoints(
            obs.object_points, rvec, tvec, camera_matrix, np.zeros((5, 1))
        )
        delta = projected.reshape(-1, 2) - obs.image_points.reshape(-1, 2)
        errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
    return CalibrationFit(
        rms=float(rms),
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion=np.zeros(5, dtype=np.float64),
        rvecs=tuple(np.asarray(value) for value in rvecs),
        tvecs=tuple(np.asarray(value) for value in tvecs),
        per_view_errors=tuple(errors),
    )


def _detection_metrics(detection) -> dict:
    return {
        "corner_count": int(detection.corner_count),
        "marker_count": int(detection.marker_count),
        "grid_rows": int(detection.grid_rows),
        "grid_cols": int(detection.grid_cols),
        "board_bbox_fraction": float(detection.board_bbox_fraction),
        "coverage_fraction": float(detection.coverage_fraction),
        "pixels_per_square": float(detection.pixels_per_square),
        "roi_laplacian_var": float(detection.sharpness),
        "rectified_laplacian_var": float(detection.rectified_laplacian_var),
        "rectified_tenengrad_mean": float(detection.rectified_tenengrad_mean),
    }


def _load_observations(session_dir: Path, detector: CharucoDetector):
    import cv2

    manifest = yaml.safe_load((session_dir / "manifest.yaml").read_text(encoding="utf-8"))
    camera = manifest["camera"]
    width, height = int(camera["width"]), int(camera["height"])
    if camera.get("zed_mode") != "HD1200" or (width, height) != (1920, 1200):
        raise ValueError(
            f"This calibration profile requires HD1200 1920x1200; session is {width}x{height}"
        )
    if camera.get("image_geometry") != "rectified":
        raise ValueError("This solver is only for rectified LEFT images")
    if camera.get("image_view") != "LEFT":
        raise ValueError("This solver is only for the LEFT image view")

    records = [
        json.loads(line)
        for line in (session_dir / "frames.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    observations: list[Observation] = []
    rejected: list[dict] = []
    selection_rows: list[dict] = []
    for record in records:
        image_name = str(record["image"])
        image_path = session_dir / image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        row = {"image": image_name, "status": "candidate", "reason": ""}
        if image is None or image.shape[1::-1] != (width, height):
            reason = "unreadable_or_wrong_size"
            rejected.append({"image": image_name, "reason": reason})
            row.update(status="rejected", reason=reason)
            selection_rows.append(row)
            continue
        detection = detector.detect(image)
        if detection is None or detection.corner_count < 12:
            reason = "insufficient_corners"
            rejected.append({"image": image_name, "reason": reason})
            row.update(status="rejected", reason=reason)
            selection_rows.append(row)
            continue
        metrics = _detection_metrics(detection)
        row.update(metrics)
        object_points, image_points = detector.calibration_points(detection)
        observations.append(Observation(image_name, object_points, image_points, metrics))
        selection_rows.append(row)
    return manifest, (width, height), observations, rejected, records, selection_rows


def _filter_motion_blur(
    observations: list[Observation],
    min_remaining: int,
    threshold: float = -3.5,
) -> tuple[list[Observation], list[dict]]:
    if len(observations) < 8:
        return list(observations), []
    scores = scale_aware_blur_scores(
        np.asarray([obs.metrics.get("rectified_laplacian_var", math.nan) for obs in observations]),
        np.asarray([obs.metrics.get("rectified_tenengrad_mean", math.nan) for obs in observations]),
        np.asarray([obs.metrics.get("pixels_per_square", math.nan) for obs in observations]),
    )
    for observation, score in zip(observations, scores):
        observation.metrics["blur_score"] = float(score) if np.isfinite(score) else None
    candidates = np.flatnonzero(np.isfinite(scores) & (scores < threshold))
    max_remove = min(
        max(0, len(observations) - min_remaining),
        max(1, math.ceil(0.10 * len(observations))),
    )
    if max_remove <= 0 or candidates.size == 0:
        return list(observations), []
    candidates = candidates[np.argsort(scores[candidates])[:max_remove]]
    rejected_indices = {int(index) for index in candidates}
    rejected = [
        {
            "image": observations[index].image_name,
            "reason": "motion_blur",
            "blur_score": float(scores[index]),
            "threshold": threshold,
        }
        for index in sorted(rejected_indices)
    ]
    active = [obs for index, obs in enumerate(observations) if index not in rejected_indices]
    return active, rejected


def _select_pose_diverse(
    observations: list[Observation],
    image_size: tuple[int, int],
    max_views: int | None,
) -> tuple[list[Observation], list[dict]]:
    if max_views is None or len(observations) <= max_views:
        return list(observations), []
    features = np.asarray(
        [pose_signature(observation.image_points, image_size) for observation in observations]
    )
    selected_indices = set(select_pose_diverse_indices(features, max_views))
    rejected = [
        {"image": observation.image_name, "reason": "pose_redundant"}
        for index, observation in enumerate(observations)
        if index not in selected_indices
    ]
    active = [
        observation for index, observation in enumerate(observations) if index in selected_indices
    ]
    return active, rejected


def _robust_fit(
    observations: list[Observation],
    image_size: tuple[int, int],
    max_view_error_px: float,
    min_remaining: int,
    max_rounds: int = 5,
) -> tuple[CalibrationFit, list[Observation], list[dict], list[dict]]:
    active = list(observations)
    rejected: list[dict] = []
    rounds: list[dict] = []
    for round_index in range(max_rounds):
        fit = calibrate_observations(active, image_size)
        errors = np.asarray(fit.per_view_errors)
        median = float(np.median(errors))
        mad = float(np.median(np.abs(errors - median)))
        robust_limit = median + 3.5 * max(1.4826 * mad, 0.05)
        limit = min(max_view_error_px, robust_limit)
        bad = np.flatnonzero(errors > limit)
        max_remove = min(
            max(0, len(active) - min_remaining),
            max(1, math.ceil(0.10 * len(active))),
        )
        if bad.size > max_remove:
            bad = bad[np.argsort(errors[bad])[::-1][:max_remove]]
        rounds.append(
            {
                "round": round_index + 1,
                "views": len(active),
                "rms_px": fit.rms,
                "median_view_error_px": median,
                "threshold_px": limit,
                "rejected": int(bad.size),
            }
        )
        if bad.size == 0:
            return fit, active, rejected, rounds
        bad_set = {int(index) for index in bad}
        for index in sorted(bad_set):
            rejected.append(
                {
                    "image": active[index].image_name,
                    "reason": "reprojection_outlier",
                    "error_px": float(errors[index]),
                    "threshold_px": limit,
                }
            )
        active = [obs for index, obs in enumerate(active) if index not in bad_set]
    return calibrate_observations(active, image_size), active, rejected, rounds


def _pnp_error(observation: Observation, camera_matrix: np.ndarray) -> float | None:
    import cv2

    ok, rvec, tvec = cv2.solvePnP(
        observation.object_points,
        observation.image_points,
        camera_matrix,
        np.zeros((5, 1)),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    projected, _ = cv2.projectPoints(
        observation.object_points,
        rvec,
        tvec,
        camera_matrix,
        np.zeros((5, 1)),
    )
    delta = projected.reshape(-1, 2) - observation.image_points.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _cross_validate(
    observations: list[Observation],
    image_size: tuple[int, int],
    folds: int,
) -> dict:
    fold_reports: list[dict] = []
    matrices: list[np.ndarray] = []
    all_errors: list[float] = []
    for fold_index, (train_indices, held_indices) in enumerate(
        deterministic_folds(len(observations), folds)
    ):
        train = [observations[index] for index in train_indices]
        held_out = [observations[index] for index in held_indices]
        fit = calibrate_observations(train, image_size)
        errors = [
            error
            for observation in held_out
            if (error := _pnp_error(observation, fit.camera_matrix)) is not None
        ]
        matrices.append(fit.camera_matrix)
        all_errors.extend(errors)
        fold_reports.append(
            {
                "fold": fold_index,
                "train_views": len(train),
                "held_out_views": len(held_out),
                "evaluated_views": len(errors),
                "training_rms_px": fit.rms,
                "held_out_median_error_px": float(np.median(errors)) if errors else None,
                "held_out_max_error_px": max(errors) if errors else None,
                "K": fit.camera_matrix.tolist(),
            }
        )
    return {
        "fold_count": len(fold_reports),
        "evaluated_views": len(all_errors),
        "held_out_median_error_px": float(np.median(all_errors)) if all_errors else None,
        "held_out_p90_error_px": float(np.percentile(all_errors, 90)) if all_errors else None,
        "held_out_max_error_px": max(all_errors) if all_errors else None,
        "camera_matrix_stability": summarize_camera_matrices(matrices),
        "folds": fold_reports,
    }


def _render_reprojection_images(
    session_dir: Path,
    observations: list[Observation],
    fit: CalibrationFit,
) -> tuple[list[np.ndarray], list[str]]:
    import cv2

    images: list[np.ndarray] = []
    labels: list[str] = []
    for observation, rvec, tvec, error in zip(
        observations, fit.rvecs, fit.tvecs, fit.per_view_errors
    ):
        image_path = observation.metrics.get("_image_path")
        if image_path is None:
            image_path = session_dir / observation.image_name
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        projected, _ = cv2.projectPoints(
            observation.object_points,
            rvec,
            tvec,
            fit.camera_matrix,
            np.zeros((5, 1)),
        )
        for observed, predicted in zip(
            observation.image_points.reshape(-1, 2), projected.reshape(-1, 2)
        ):
            observed_xy = tuple(np.round(observed).astype(int))
            predicted_xy = tuple(np.round(predicted).astype(int))
            cv2.circle(image, observed_xy, 4, (0, 255, 0), -1)
            cv2.circle(image, predicted_xy, 3, (255, 0, 255), -1)
            cv2.line(image, observed_xy, predicted_xy, (0, 255, 255), 1)
        images.append(downscale_for_diagnostics(image))
        labels.append(f"{Path(observation.image_name).name} error={error:.3f}px")
    return images, labels


def _update_selection_rows(
    rows: list[dict],
    rejected: list[dict],
    inliers: list[Observation],
    fit: CalibrationFit,
) -> None:
    by_image = {str(row["image"]): row for row in rows}
    for item in rejected:
        row = by_image.get(str(item["image"]))
        if row is not None:
            row.update(status="rejected", reason=item["reason"])
            if "blur_score" in item:
                row["blur_score"] = item["blur_score"]
            if "error_px" in item:
                row["reprojection_error_px"] = item["error_px"]
    for observation, error in zip(inliers, fit.per_view_errors):
        row = by_image.get(observation.image_name)
        if row is not None:
            row.update(
                status="selected",
                reason="",
                blur_score=observation.metrics.get("blur_score"),
                reprojection_error_px=float(error),
            )


def solve_session(
    session: str | Path,
    *,
    max_view_error_px: float = 0.8,
    min_views: int = 20,
    max_views: int | None = 40,
    cross_validation_folds: int = 5,
) -> Path:
    if min_views < 3:
        raise ValueError("min_views must be >= 3")
    if max_views is not None and max_views < min_views:
        raise ValueError("max_views must be >= min_views")
    if max_view_error_px <= 0:
        raise ValueError("max_view_error_px must be positive")
    if cross_validation_folds < 2:
        raise ValueError("cross_validation_folds must be >= 2")
    session_dir = Path(session).expanduser().resolve()
    manifest_path = session_dir / "manifest.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Session manifest not found: {manifest_path}")
    initial_manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    board_path = session_dir / initial_manifest["board"]["profile_file"]
    board = require_charuco(load_board_profile(board_path), "intrinsic solve")
    if board.sha256 != initial_manifest["board"]["profile_sha256"]:
        raise ValueError("Session board profile hash does not match manifest")
    detector = CharucoDetector(board)
    (
        manifest,
        image_size,
        observations,
        detection_rejections,
        records,
        selection_rows,
    ) = _load_observations(session_dir, detector)
    if len(observations) < min_views:
        raise ValueError(f"Need at least {min_views} valid views; found {len(observations)}")

    sharp_inliers, blur_rejections = _filter_motion_blur(observations, min_views)
    diverse_inliers, diversity_rejections = _select_pose_diverse(
        sharp_inliers, image_size, max_views
    )
    if len(diverse_inliers) < min_views:
        raise ValueError(f"Only {len(diverse_inliers)} views remain after quality selection")
    fit, inliers, reprojection_rejections, calibration_rounds = _robust_fit(
        diverse_inliers,
        image_size,
        max_view_error_px,
        min_views,
    )
    if len(inliers) < min_views:
        raise ValueError(f"Only {len(inliers)} views remain after outlier rejection")

    cross_validation = _cross_validate(inliers, image_size, cross_validation_folds)
    split_a = inliers[::2]
    split_b = inliers[1::2]
    split_metrics = None
    if len(split_a) >= 6 and len(split_b) >= 6:
        fit_a = calibrate_observations(split_a, image_size)
        fit_b = calibrate_observations(split_b, image_size)
        split_metrics = camera_matrix_difference(fit_a.camera_matrix, fit_b.camera_matrix)

    held_median = cross_validation["held_out_median_error_px"]
    stability_ok = split_metrics is not None and (
        split_metrics["fx_relative_difference"] < 0.005
        and split_metrics["fy_relative_difference"] < 0.005
        and split_metrics["principal_point_difference_px"] < 3.0
    )
    quality_gates = {
        "enough_views": len(inliers) >= 25,
        "rms_below_0_5_px": fit.rms < 0.5,
        "held_out_median_below_0_5_px": held_median is not None and held_median < 0.5,
        "split_stability": stability_ok,
    }

    results_dir = session_dir / "results"
    results_dir.mkdir(exist_ok=True)
    all_rejections = (
        detection_rejections + blur_rejections + diversity_rejections + reprojection_rejections
    )
    _update_selection_rows(selection_rows, all_rejections, inliers, fit)
    write_selection_csv(results_dir / "sample_selection.csv", selection_rows)
    write_json(results_dir / "cross_validation.json", cross_validation)

    capture_images, capture_labels = read_capture_previews(session_dir, records)
    write_contact_sheet(results_dir / "capture_contact_sheet.jpg", capture_images, capture_labels)
    reprojection_images, reprojection_labels = _render_reprojection_images(
        session_dir, inliers, fit
    )
    write_contact_sheet(
        results_dir / "reprojection_contact_sheet.jpg",
        reprojection_images,
        reprojection_labels,
    )

    quality_summary = {
        "views_recorded": len(records),
        "views_detected": len(observations),
        "views_after_blur_filter": len(sharp_inliers),
        "views_after_pose_selection": len(diverse_inliers),
        "views_used": len(inliers),
        "rejection_counts": {
            reason: sum(1 for item in all_rejections if item["reason"] == reason)
            for reason in sorted({item["reason"] for item in all_rejections})
        },
        "calibration_rounds": calibration_rounds,
        "cross_validation": cross_validation,
    }
    write_json(results_dir / "quality_summary.json", quality_summary)

    width, height = image_size
    result = {
        "schema": "dexmate_calib.intrinsics.v1",
        "camera": {
            "name": "head_left",
            "serial": manifest["camera"].get("camera_serial"),
            "zed_mode": "HD1200",
            "width": width,
            "height": height,
            "image_view": "LEFT",
            "image_geometry": "rectified",
        },
        "model": {
            "type": "pinhole",
            "distortion_model": "none",
            "K": fit.camera_matrix.tolist(),
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            "fx": float(fit.camera_matrix[0, 0]),
            "fy": float(fit.camera_matrix[1, 1]),
            "cx": float(fit.camera_matrix[0, 2]),
            "cy": float(fit.camera_matrix[1, 2]),
        },
        "board": {"name": board.name, "profile_sha256": board.sha256},
        "quality": {
            "rms_reprojection_error_px": fit.rms,
            "views_detected": len(observations),
            "views_used": len(inliers),
            "held_out_views": cross_validation["evaluated_views"],
            "held_out_median_error_px": held_median,
            "held_out_max_error_px": cross_validation["held_out_max_error_px"],
            "cross_validation": cross_validation,
            "split_stability": split_metrics,
            "calibration_rounds": calibration_rounds,
            "gates": quality_gates,
            "all_gates_pass": all(quality_gates.values()),
            "thresholds": {
                "max_view_error_px": max_view_error_px,
                "min_views": min_views,
                "max_views": max_views,
                "cross_validation_folds": cross_validation_folds,
            },
        },
        "diagnostics": {
            "sample_selection_csv": "sample_selection.csv",
            "cross_validation_json": "cross_validation.json",
            "quality_summary_json": "quality_summary.json",
            "capture_contact_sheet": "capture_contact_sheet.jpg",
            "reprojection_contact_sheet": "reprojection_contact_sheet.jpg",
        },
        "rejected_views": all_rejections,
    }
    stem = f"intrinsics_head_left_HD1200_{width}x{height}"
    json_path = results_dir / f"{stem}.json"
    yaml_path = results_dir / f"{stem}.yaml"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    np.save(results_dir / "K.npy", fit.camera_matrix)

    with (results_dir / "reprojection_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "rms_error_px"])
        for observation, error in zip(inliers, fit.per_view_errors):
            writer.writerow([observation.image_name, f"{error:.9f}"])
    return yaml_path
