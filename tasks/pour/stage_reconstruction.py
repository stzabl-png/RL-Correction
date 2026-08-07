"""Merge validated two-object reconstruction outputs into a pour reference.

Track NPZ contract (one file per object):

* ``pose``: ``[T,7]`` xyz+wxyz or ``[T,4,4]`` rigid transforms.
* ``source_frame``: monotonically increasing EgoDex frame numbers.
* ``confidence``: one value per pose.
* ``coordinate_frame``: scalar string, ``camera`` or ``arkit_world``.
* ``quaternion_order``: scalar string ``wxyz`` when pose is ``[T,7]``.

Geometry JSON holds the two :class:`PourObjectSpec` dictionaries and contact
regions in object-local coordinates.  The strict schema is intentional: data
that cannot prove time and frame alignment must not silently become a reward.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from tasks.pour.reference import PourReference, rotation_matrix_to_wxyz
from tasks.pour.scene import PourObjectSpec, PourSceneManifest


def _scalar_text(value: np.ndarray, name: str) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar string")
    return str(array.item())


def _quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def _pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    result = np.broadcast_to(np.eye(4), (len(pose), 4, 4)).copy()
    result[:, :3, :3] = _quat_to_matrix(pose[:, 3:7])
    result[:, :3, 3] = pose[:, :3]
    return result


def _matrix_to_pose(transform: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [transform[:, :3, 3], rotation_matrix_to_wxyz(transform[:, :3, :3])],
        axis=1,
    ).astype(np.float32)


def _validate_pose(pose: np.ndarray, name: str) -> None:
    if pose.ndim != 2 or pose.shape[1] != 7 or not np.isfinite(pose).all():
        raise ValueError(f"{name} pose must be finite [T,7], got {pose.shape}")
    norm = np.linalg.norm(pose[:, 3:7], axis=1)
    if not np.allclose(norm, 1.0, atol=2.0e-3):
        raise ValueError(f"{name} contains non-normalized quaternions")


def _interp_pose(
    pose: np.ndarray, source_frame: np.ndarray, target_frame: np.ndarray
) -> np.ndarray:
    if target_frame[0] < source_frame[0] or target_frame[-1] > source_frame[-1]:
        raise ValueError("track does not cover the full reference time range")
    hi = np.searchsorted(source_frame, target_frame, side="right")
    hi = hi.clip(1, len(source_frame) - 1)
    lo = hi - 1
    alpha = (
        (target_frame - source_frame[lo])
        / np.maximum(source_frame[hi] - source_frame[lo], 1.0e-9)
    )[:, None]
    position = (1.0 - alpha) * pose[lo, :3] + alpha * pose[hi, :3]
    q0, q1 = pose[lo, 3:7].copy(), pose[hi, 3:7].copy()
    q1 = np.where((q0 * q1).sum(axis=1, keepdims=True) < 0.0, -q1, q1)
    quaternion = (1.0 - alpha) * q0 + alpha * q1
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    return np.concatenate([position, quaternion], axis=1).astype(np.float32)


def _camera_pose_at(reference: PourReference, source_frame: np.ndarray) -> np.ndarray:
    return _interp_pose(reference.camera_pose, reference.source_frame, source_frame)


def load_object_track(
    path: str | Path,
    reference: PourReference,
    *,
    minimum_confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load, frame-convert, time-align, and confidence-gate one track."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        required = {"pose", "source_frame", "confidence", "coordinate_frame"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{source.name} missing {sorted(missing)}")
        raw = np.asarray(data["pose"], dtype=np.float64)
        frame = np.asarray(data["source_frame"], dtype=np.float64)
        confidence = np.asarray(data["confidence"], dtype=np.float64)
        coordinate_frame = _scalar_text(data["coordinate_frame"], "coordinate_frame")
        if raw.ndim == 3 and raw.shape[1:] == (4, 4):
            pose = _matrix_to_pose(raw)
        elif raw.ndim == 2 and raw.shape[1] == 7:
            if "quaternion_order" not in data.files:
                raise KeyError(f"{source.name} [T,7] pose requires quaternion_order")
            order = _scalar_text(data["quaternion_order"], "quaternion_order")
            if order != "wxyz":
                raise ValueError(f"{source.name}: quaternion_order must be wxyz")
            pose = raw.astype(np.float32)
        else:
            raise ValueError(f"{source.name}: pose shape {raw.shape} is unsupported")

    _validate_pose(pose, source.name)
    if frame.shape != (len(pose),) or confidence.shape != (len(pose),):
        raise ValueError(f"{source.name}: source_frame/confidence length mismatch")
    if len(frame) < 2 or not np.all(np.diff(frame) > 0.0):
        raise ValueError(f"{source.name}: source_frame must be strictly increasing")
    if not np.isfinite(confidence).all() or np.any((confidence < 0.0) | (confidence > 1.0)):
        raise ValueError(f"{source.name}: confidence must be finite in [0,1]")

    if coordinate_frame == "camera":
        camera = _pose_to_matrix(_camera_pose_at(reference, frame))
        pose = _matrix_to_pose(camera @ _pose_to_matrix(pose))
    elif coordinate_frame != "arkit_world":
        raise ValueError(
            f"{source.name}: coordinate_frame must be camera or arkit_world"
        )

    aligned_pose = _interp_pose(pose, frame, reference.source_frame)
    aligned_confidence = np.interp(
        reference.source_frame, frame, confidence
    ).astype(np.float32)
    if float(aligned_confidence.min()) < minimum_confidence:
        raise ValueError(
            f"{source.name}: aligned confidence min {aligned_confidence.min():.3f} "
            f"< {minimum_confidence:.3f}"
        )
    return aligned_pose, aligned_confidence


def _yaw_mapping(
    cup_video: np.ndarray,
    bottle_video: np.ndarray,
    cup_sim: np.ndarray,
    bottle_sim: np.ndarray,
    *,
    maximum_residual_m: float,
) -> np.ndarray:
    source_delta = bottle_video[:3] - cup_video[:3]
    target_delta = bottle_sim[:3] - cup_sim[:3]
    if np.linalg.norm(source_delta[:2]) < 1.0e-4 or np.linalg.norm(target_delta[:2]) < 1.0e-4:
        raise ValueError("cup/bottle initial centers cannot define a horizontal mapping")
    yaw = np.arctan2(target_delta[1], target_delta[0]) - np.arctan2(
        source_delta[1], source_delta[0]
    )
    rotation = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    source = np.stack([cup_video[:3], bottle_video[:3]])
    target = np.stack([cup_sim[:3], bottle_sim[:3]])
    translation = (target - source @ rotation.T).mean(axis=0)
    residual = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    if float(residual.max()) > maximum_residual_m:
        raise ValueError(
            f"single ARKit-to-sim transform residual {residual.max() * 100:.2f}cm "
            f"> {maximum_residual_m * 100:.2f}cm"
        )
    quaternion = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
    return np.r_[translation, quaternion].astype(np.float32)


def _load_geometry(path: str | Path, demo_id: str) -> tuple[dict, dict[str, np.ndarray], np.ndarray]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if int(value.get("schema_version", -1)) != 1:
        raise ValueError("geometry schema_version must be 1")
    if str(value.get("demo_id")) != str(demo_id):
        raise ValueError("geometry demo_id does not match reference")
    contacts = value.get("contacts", {})
    if contacts.get("coordinate_frame") != "object_local":
        raise ValueError("contact regions must use object_local coordinates")
    points: dict[str, np.ndarray] = {}
    confidence = []
    expected = {"left": "cup", "right": "bottle"}
    for side in ("left", "right"):
        entry = contacts.get(side, {})
        if entry.get("object") != expected[side]:
            raise ValueError(f"{side} contact must refer to {expected[side]}")
        region = np.asarray(entry.get("points"), dtype=np.float32)
        score = float(entry.get("confidence", 0.0))
        if region.ndim != 2 or region.shape[1] != 3 or len(region) == 0:
            raise ValueError(f"{side} contact points must be non-empty [N,3]")
        if not np.isfinite(region).all() or not 0.0 < score <= 1.0:
            raise ValueError(f"{side} contact region/confidence is invalid")
        points[side] = region
        confidence.append(score)
    return value, points, np.asarray(confidence, dtype=np.float32)


def merge_reconstruction(
    reference_path: str | Path,
    scene_path: str | Path,
    *,
    cup_track_path: str | Path,
    bottle_track_path: str | Path,
    geometry_path: str | Path,
    minimum_confidence: float = 0.5,
    maximum_mapping_residual_m: float = 0.03,
) -> tuple[PourReference, PourSceneManifest]:
    reference = PourReference.load(reference_path)
    scene = PourSceneManifest.load(scene_path, resolve_relative=False)
    if scene.demo_id != reference.demo_id:
        raise ValueError("scene and reference demo_id differ")
    cup_pose, cup_confidence = load_object_track(
        cup_track_path, reference, minimum_confidence=minimum_confidence
    )
    bottle_pose, bottle_confidence = load_object_track(
        bottle_track_path, reference, minimum_confidence=minimum_confidence
    )
    geometry, contacts, contact_confidence = _load_geometry(
        geometry_path, reference.demo_id
    )
    cup = PourObjectSpec(**geometry["cup"])
    bottle = PourObjectSpec(**geometry["bottle"])
    transform = _yaw_mapping(
        cup_pose[0],
        bottle_pose[0],
        np.asarray(cup.initial_pose_wxyz, dtype=np.float64),
        np.asarray(bottle.initial_pose_wxyz, dtype=np.float64),
        maximum_residual_m=maximum_mapping_residual_m,
    )
    merged_reference = replace(
        reference,
        cup_pose=cup_pose,
        bottle_pose=bottle_pose,
        object_confidence=np.stack([cup_confidence, bottle_confidence], axis=1),
        video_contact_left=contacts["left"],
        video_contact_right=contacts["right"],
        video_contact_confidence=contact_confidence,
    )
    merged_reference.validate()
    if not merged_reference.training_ready:
        raise RuntimeError("merged reference did not pass the training-ready gate")
    merged_scene = PourSceneManifest(
        demo_id=scene.demo_id,
        status="pending_grasp_approval",
        reference_npz=scene.reference_npz,
        left_grasp_prior="",
        right_grasp_prior="",
        cup=cup,
        bottle=bottle,
        video_to_sim_wxyz=transform.tolist(),
    )
    merged_scene.validate(require_assets=False)
    merged_reference.save(reference_path)
    merged_scene.save(scene_path)
    return merged_reference, merged_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--cup-track", required=True)
    parser.add_argument("--bottle-track", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.5)
    parser.add_argument("--maximum-mapping-residual-m", type=float, default=0.03)
    args = parser.parse_args()
    reference, scene = merge_reconstruction(
        args.reference,
        args.scene,
        cup_track_path=args.cup_track,
        bottle_track_path=args.bottle_track,
        geometry_path=args.geometry,
        minimum_confidence=args.minimum_confidence,
        maximum_mapping_residual_m=args.maximum_mapping_residual_m,
    )
    print(
        f"[pour-stage] demo={reference.demo_id} frames={reference.length} "
        f"status={scene.status} training_ready={reference.training_ready}"
    )


if __name__ == "__main__":
    main()
