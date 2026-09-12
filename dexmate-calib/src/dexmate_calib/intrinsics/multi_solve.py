from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dexmate_calib.boards.config import load_board_profile, require_charuco
from dexmate_calib.intrinsics.detector import CharucoDetector
from dexmate_calib.intrinsics.diagnostics import (
    downscale_for_diagnostics,
    write_contact_sheet,
    write_json,
    write_selection_csv,
)
from dexmate_calib.intrinsics.solve import (
    CalibrationFit,
    Observation,
    _cross_validate,
    _filter_motion_blur,
    _load_observations,
    _pnp_error,
    _render_reprojection_images,
    _robust_fit,
    _select_pose_diverse,
    _update_selection_rows,
    calibrate_observations,
)
from dexmate_calib.intrinsics.validation import (
    camera_matrix_difference,
    summarize_camera_matrices,
)


@dataclass(frozen=True)
class SessionSource:
    name: str
    path: Path
    manifest: dict[str, Any]
    board_path: Path


def _camera_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    camera = manifest["camera"]
    return {
        "camera_serial": camera.get("camera_serial"),
        "zed_mode": camera.get("zed_mode"),
        "width": int(camera["width"]),
        "height": int(camera["height"]),
        "image_view": camera.get("image_view"),
        "image_geometry": camera.get("image_geometry"),
        "board_profile_sha256": manifest["board"].get("profile_sha256"),
    }


def _load_sources(sessions: list[str | Path]) -> tuple[list[SessionSource], dict[str, Any]]:
    if len(sessions) < 2:
        raise ValueError("solve-multi requires at least two sessions")
    sources: list[SessionSource] = []
    resolved_paths: set[Path] = set()
    names: set[str] = set()
    expected_contract: dict[str, Any] | None = None
    for session in sessions:
        session_dir = Path(session).expanduser().resolve()
        if session_dir in resolved_paths:
            raise ValueError(f"Duplicate session: {session_dir}")
        manifest_path = session_dir / "manifest.yaml"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Session manifest not found: {manifest_path}")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        board_path = session_dir / manifest["board"]["profile_file"]
        board = require_charuco(load_board_profile(board_path), "intrinsic solve-multi")
        if board.sha256 != manifest["board"].get("profile_sha256"):
            raise ValueError(f"Session board profile hash does not match manifest: {session_dir}")
        contract = _camera_contract(manifest)
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            differences = {
                key: {"expected": expected_contract[key], "actual": contract[key]}
                for key in expected_contract
                if contract[key] != expected_contract[key]
            }
            raise ValueError(f"Incompatible session {session_dir}: {differences}")
        name = session_dir.name
        if name in names:
            raise ValueError(f"Session directory names must be unique: {name}")
        resolved_paths.add(session_dir)
        names.add(name)
        sources.append(SessionSource(name, session_dir, manifest, board_path))
    assert expected_contract is not None
    return sources, expected_contract


def _qualify_observation(source: SessionSource, observation: Observation) -> Observation:
    metrics = dict(observation.metrics)
    metrics["session"] = source.name
    metrics["_image_path"] = str(source.path / observation.image_name)
    return Observation(
        image_name=f"{source.name}/{observation.image_name}",
        object_points=observation.object_points,
        image_points=observation.image_points,
        metrics=metrics,
    )


def _qualify_items(source: SessionSource, items: list[dict]) -> list[dict]:
    qualified = []
    for item in items:
        value = dict(item)
        value["session"] = source.name
        value["image"] = f"{source.name}/{item['image']}"
        qualified.append(value)
    return qualified


def _attach_session(items: list[dict], session_name: str) -> list[dict]:
    return [{**item, "session": session_name} for item in items]


def _session_name(observation: Observation) -> str:
    return str(observation.metrics["session"])


def _leave_one_session_out(
    observations: list[Observation],
    image_size: tuple[int, int],
    session_names: list[str],
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    matrices: list[np.ndarray] = []
    all_errors: list[float] = []
    for held_session in session_names:
        train = [obs for obs in observations if _session_name(obs) != held_session]
        held_out = [obs for obs in observations if _session_name(obs) == held_session]
        if len(train) < 3 or not held_out:
            continue
        fit = calibrate_observations(train, image_size)
        errors = [
            error
            for observation in held_out
            if (error := _pnp_error(observation, fit.camera_matrix)) is not None
        ]
        matrices.append(fit.camera_matrix)
        all_errors.extend(errors)
        reports.append(
            {
                "held_out_session": held_session,
                "train_views": len(train),
                "held_out_views": len(held_out),
                "evaluated_views": len(errors),
                "training_rms_px": fit.rms,
                "held_out_median_error_px": float(np.median(errors)) if errors else None,
                "held_out_p90_error_px": (float(np.percentile(errors, 90)) if errors else None),
                "held_out_max_error_px": max(errors) if errors else None,
                "K": fit.camera_matrix.tolist(),
            }
        )
    return {
        "fold_count": len(reports),
        "evaluated_views": len(all_errors),
        "held_out_median_error_px": float(np.median(all_errors)) if all_errors else None,
        "held_out_p90_error_px": (float(np.percentile(all_errors, 90)) if all_errors else None),
        "held_out_max_error_px": max(all_errors) if all_errors else None,
        "camera_matrix_stability": summarize_camera_matrices(matrices),
        "folds": reports,
    }


def _stratified_split(
    observations: list[Observation], session_names: list[str]
) -> tuple[list[Observation], list[Observation]]:
    split_a: list[Observation] = []
    split_b: list[Observation] = []
    for session_name in session_names:
        group = [obs for obs in observations if _session_name(obs) == session_name]
        split_a.extend(group[::2])
        split_b.extend(group[1::2])
    return split_a, split_b


def _diagnostic_subset(
    observations: list[Observation], fit: CalibrationFit, maximum: int = 40
) -> tuple[list[Observation], CalibrationFit]:
    if len(observations) <= maximum:
        return observations, fit
    indices = np.linspace(0, len(observations) - 1, maximum, dtype=int)
    selected = [observations[index] for index in indices]
    selected_fit = CalibrationFit(
        rms=fit.rms,
        camera_matrix=fit.camera_matrix,
        distortion=fit.distortion,
        rvecs=tuple(fit.rvecs[index] for index in indices),
        tvecs=tuple(fit.tvecs[index] for index in indices),
        per_view_errors=tuple(fit.per_view_errors[index] for index in indices),
    )
    return selected, selected_fit


def _read_observation_images(
    observations: list[Observation],
) -> tuple[list[np.ndarray], list[str]]:
    import cv2

    images: list[np.ndarray] = []
    labels: list[str] = []
    for observation in observations:
        image = cv2.imread(str(observation.metrics["_image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        images.append(downscale_for_diagnostics(image))
        labels.append(observation.image_name)
    return images, labels


def solve_sessions(
    sessions: list[str | Path],
    *,
    output: str | Path,
    max_view_error_px: float = 0.8,
    min_views: int = 40,
    min_views_per_session: int = 12,
    max_views_per_session: int = 40,
    cross_validation_folds: int = 5,
) -> Path:
    if min_views < 3:
        raise ValueError("min_views must be >= 3")
    if min_views_per_session < 3:
        raise ValueError("min_views_per_session must be >= 3")
    if max_views_per_session < min_views_per_session:
        raise ValueError("max_views_per_session must be >= min_views_per_session")
    if max_view_error_px <= 0:
        raise ValueError("max_view_error_px must be positive")
    if cross_validation_folds < 2:
        raise ValueError("cross_validation_folds must be >= 2")

    sources, contract = _load_sources(sessions)
    output_dir = Path(output).expanduser().resolve()
    if output_dir in {source.path for source in sources}:
        raise ValueError("Multi-session output must not overwrite a source session")
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    all_observations: list[Observation] = []
    all_rejections: list[dict] = []
    selection_rows: list[dict] = []
    session_summaries: list[dict[str, Any]] = []
    for source in sources:
        board = require_charuco(load_board_profile(source.board_path), "intrinsic solve-multi")
        detector = CharucoDetector(board)
        (
            _manifest,
            image_size,
            observations,
            detection_rejections,
            records,
            rows,
        ) = _load_observations(source.path, detector)
        if image_size != (contract["width"], contract["height"]):
            raise ValueError(f"Unexpected image size in {source.path}: {image_size}")
        if len(observations) < min_views_per_session:
            raise ValueError(
                f"Session {source.name} needs at least {min_views_per_session} valid views; "
                f"found {len(observations)}"
            )
        qualified_observations = [
            _qualify_observation(source, observation) for observation in observations
        ]
        qualified_rejections = _qualify_items(source, detection_rejections)
        qualified_rows = _qualify_items(source, rows)
        sharp, blur_rejections = _filter_motion_blur(qualified_observations, min_views_per_session)
        diverse, diversity_rejections = _select_pose_diverse(
            sharp,
            image_size,
            max_views_per_session,
        )
        if len(diverse) < min_views_per_session:
            raise ValueError(
                f"Session {source.name} has only {len(diverse)} views after quality selection"
            )
        all_observations.extend(diverse)
        all_rejections.extend(
            qualified_rejections
            + _attach_session(blur_rejections, source.name)
            + _attach_session(diversity_rejections, source.name)
        )
        selection_rows.extend(qualified_rows)
        session_summaries.append(
            {
                "session": source.name,
                "path": str(source.path),
                "views_recorded": len(records),
                "views_detected": len(observations),
                "views_after_blur_filter": len(sharp),
                "views_after_pose_selection": len(diverse),
            }
        )

    if len(all_observations) < min_views:
        raise ValueError(f"Need at least {min_views} pooled views; found {len(all_observations)}")
    image_size = (int(contract["width"]), int(contract["height"]))
    fit, inliers, reprojection_rejections, calibration_rounds = _robust_fit(
        all_observations,
        image_size,
        max_view_error_px,
        min_views,
    )
    all_rejections.extend(
        [
            {
                **item,
                "session": item["image"].split("/", 1)[0],
            }
            for item in reprojection_rejections
        ]
    )

    session_names = [source.name for source in sources]
    used_counts = Counter(_session_name(observation) for observation in inliers)
    for summary in session_summaries:
        summary["views_used"] = used_counts[summary["session"]]

    cross_validation = _cross_validate(inliers, image_size, cross_validation_folds)
    leave_one_session_out = _leave_one_session_out(inliers, image_size, session_names)
    split_a, split_b = _stratified_split(inliers, session_names)
    split_metrics = None
    if len(split_a) >= 6 and len(split_b) >= 6:
        fit_a = calibrate_observations(split_a, image_size)
        fit_b = calibrate_observations(split_b, image_size)
        split_metrics = camera_matrix_difference(fit_a.camera_matrix, fit_b.camera_matrix)

    held_median = cross_validation["held_out_median_error_px"]
    loso_median = leave_one_session_out["held_out_median_error_px"]
    loso_stability = leave_one_session_out["camera_matrix_stability"]
    split_stability_ok = split_metrics is not None and (
        split_metrics["fx_relative_difference"] < 0.005
        and split_metrics["fy_relative_difference"] < 0.005
        and split_metrics["principal_point_difference_px"] < 3.0
    )
    loso_stability_ok = loso_stability is not None and (
        loso_stability["fx_relative_range"] < 0.005
        and loso_stability["fy_relative_range"] < 0.005
        and loso_stability["principal_point_span_px"] < 3.0
    )
    quality_gates = {
        "enough_sessions": len(sources) >= 2,
        "enough_views": len(inliers) >= max(25, min_views),
        "each_session_has_enough_views": all(
            used_counts[name] >= min_views_per_session for name in session_names
        ),
        "rms_below_0_5_px": fit.rms < 0.5,
        "held_out_median_below_0_5_px": held_median is not None and held_median < 0.5,
        "leave_one_session_out_median_below_0_5_px": (
            loso_median is not None and loso_median < 0.5
        ),
        "split_stability": split_stability_ok,
        "leave_one_session_out_stability": loso_stability_ok,
    }

    _update_selection_rows(selection_rows, all_rejections, inliers, fit)
    write_selection_csv(results_dir / "sample_selection.csv", selection_rows)
    write_json(results_dir / "cross_validation.json", cross_validation)
    write_json(results_dir / "leave_one_session_out.json", leave_one_session_out)

    diagnostic_observations, diagnostic_fit = _diagnostic_subset(inliers, fit)
    capture_images, capture_labels = _read_observation_images(diagnostic_observations)
    write_contact_sheet(results_dir / "capture_contact_sheet.jpg", capture_images, capture_labels)
    reprojection_images, reprojection_labels = _render_reprojection_images(
        output_dir, diagnostic_observations, diagnostic_fit
    )
    write_contact_sheet(
        results_dir / "reprojection_contact_sheet.jpg",
        reprojection_images,
        reprojection_labels,
    )

    quality_summary = {
        "sessions": session_summaries,
        "views_after_balanced_pose_selection": len(all_observations),
        "views_used": len(inliers),
        "rejection_counts": {
            reason: sum(1 for item in all_rejections if item["reason"] == reason)
            for reason in sorted({item["reason"] for item in all_rejections})
        },
        "calibration_rounds": calibration_rounds,
        "cross_validation": cross_validation,
        "leave_one_session_out": leave_one_session_out,
    }
    write_json(results_dir / "quality_summary.json", quality_summary)

    result = {
        "schema": "dexmate_calib.intrinsics_multi.v1",
        "camera": {
            "name": "head_left",
            "serial": contract["camera_serial"],
            "zed_mode": contract["zed_mode"],
            "width": contract["width"],
            "height": contract["height"],
            "image_view": contract["image_view"],
            "image_geometry": contract["image_geometry"],
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
        "board": {
            "name": sources[0].manifest["board"]["name"],
            "profile_sha256": contract["board_profile_sha256"],
        },
        "sources": {
            "session_count": len(sources),
            "sessions": session_summaries,
            "selection_policy": "per-session blur filter and pose-diverse cap, then pooled fit",
        },
        "quality": {
            "rms_reprojection_error_px": fit.rms,
            "views_used": len(inliers),
            "views_used_by_session": dict(used_counts),
            "held_out_median_error_px": held_median,
            "held_out_p90_error_px": cross_validation["held_out_p90_error_px"],
            "leave_one_session_out_median_error_px": loso_median,
            "leave_one_session_out_p90_error_px": leave_one_session_out["held_out_p90_error_px"],
            "cross_validation": cross_validation,
            "leave_one_session_out": leave_one_session_out,
            "split_stability": split_metrics,
            "calibration_rounds": calibration_rounds,
            "gates": quality_gates,
            "all_gates_pass": all(quality_gates.values()),
            "thresholds": {
                "max_view_error_px": max_view_error_px,
                "min_views": min_views,
                "min_views_per_session": min_views_per_session,
                "max_views_per_session": max_views_per_session,
                "cross_validation_folds": cross_validation_folds,
            },
        },
        "diagnostics": {
            "sample_selection_csv": "sample_selection.csv",
            "cross_validation_json": "cross_validation.json",
            "leave_one_session_out_json": "leave_one_session_out.json",
            "quality_summary_json": "quality_summary.json",
            "capture_contact_sheet": "capture_contact_sheet.jpg",
            "reprojection_contact_sheet": "reprojection_contact_sheet.jpg",
            "contact_sheet_views": len(diagnostic_observations),
        },
        "rejected_views": all_rejections,
    }

    pooled_manifest = {
        "schema": "dexmate_calib.intrinsic_multi_session.v1",
        "source_sessions": [{"name": source.name, "path": str(source.path)} for source in sources],
        "camera_contract": contract,
        "settings": result["quality"]["thresholds"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(pooled_manifest, sort_keys=False), encoding="utf-8"
    )

    stem = (
        f"intrinsics_head_left_{contract['zed_mode']}_"
        f"{contract['width']}x{contract['height']}_pooled"
    )
    json_path = results_dir / f"{stem}.json"
    yaml_path = results_dir / f"{stem}.yaml"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    yaml_path.write_text(yaml.safe_dump(result, sort_keys=False), encoding="utf-8")
    np.save(results_dir / "K.npy", fit.camera_matrix)

    with (results_dir / "reprojection_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["session", "image", "rms_error_px"])
        for observation, error in zip(inliers, fit.per_view_errors):
            writer.writerow([_session_name(observation), observation.image_name, f"{error:.9f}"])
    return yaml_path
