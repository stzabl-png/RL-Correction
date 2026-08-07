"""Adapt Step-2 multi-object reconstruction into Task-5 reference inputs.

Step 2 exports object tracks in its gravity-aligned ViPE world while EgoDex
hands/camera are in a stationary ARKit world.  This adapter fits one rigid
world-to-world transform from the two camera trajectories (including the fixed
OpenCV-to-ARKit camera-axis conversion), rejects poor synchronization, exports
cup/bottle tracks in ARKit world, and estimates video contact regions by
projecting EgoDex fingertips to the reconstructed object meshes.

Opening geometry, physical properties, USD paths, and desired simulation
initial poses remain explicit in ``--object-spec``; they are never guessed from
the mesh bounding box.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tasks.pour.reference import HAND_JOINT_SUFFIXES, PourReference
from tasks.pour.scene import PourObjectSpec
from tasks.pour.stage_reconstruction import _interp_pose, _matrix_to_pose, _pose_to_matrix


TIP_INDICES = np.asarray(
    [HAND_JOINT_SUFFIXES.index(name) for name in (
        "ThumbTip",
        "IndexFingerTip",
        "MiddleFingerTip",
        "RingFingerTip",
        "LittleFingerTip",
    )],
    dtype=np.int64,
)


def _rotation_angle_deg(rotation: np.ndarray) -> np.ndarray:
    cosine = ((np.trace(rotation, axis1=-2, axis2=-1) - 1.0) / 2.0).clip(-1.0, 1.0)
    return np.rad2deg(np.arccos(cosine))


def _average_rotation(rotation: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(rotation.sum(axis=0))
    value = left @ right_t
    if np.linalg.det(value) < 0.0:
        left[:, -1] *= -1.0
        value = left @ right_t
    return value


def fit_step2_world_to_arkit(
    step2_c2w: np.ndarray,
    reference: PourReference,
    *,
    maximum_translation_p95_m: float = 0.05,
    maximum_rotation_p95_deg: float = 5.0,
) -> tuple[np.ndarray, dict]:
    """Fit one rigid transform from synchronized camera poses."""
    step2_c2w = np.asarray(step2_c2w, dtype=np.float64)
    if step2_c2w.ndim != 3 or step2_c2w.shape[1:] != (4, 4):
        raise ValueError("Step-2 c2w must be [T,4,4]")
    frames = np.rint(reference.source_frame).astype(np.int64)
    if frames.min() < 0 or frames.max() >= len(step2_c2w):
        raise ValueError("Step-2 camera track does not cover the EgoDex reference")

    # Step-2 declares camera +Z as forward (OpenCV axes); ARKit camera forward
    # is -Z with +Y up.  D maps ARKit camera coordinates into OpenCV camera.
    camera_axis = np.diag([1.0, -1.0, -1.0, 1.0])
    step2_camera_arkit_axes = step2_c2w[frames] @ camera_axis
    arkit_camera = _pose_to_matrix(reference.camera_pose)
    candidates = arkit_camera @ np.linalg.inv(step2_camera_arkit_axes)
    rotation = _average_rotation(candidates[:, :3, :3])
    translation = np.median(candidates[:, :3, 3], axis=0)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    predicted = transform @ step2_camera_arkit_axes
    translation_error = np.linalg.norm(
        predicted[:, :3, 3] - arkit_camera[:, :3, 3], axis=1
    )
    rotation_error = _rotation_angle_deg(
        predicted[:, :3, :3].transpose(0, 2, 1) @ arkit_camera[:, :3, :3]
    )
    report = {
        "schema_version": 1,
        "camera_axis_conversion": "opencv_plus_z_to_arkit_minus_z",
        "num_camera_samples": int(len(frames)),
        "translation_median_m": float(np.median(translation_error)),
        "translation_p95_m": float(np.quantile(translation_error, 0.95)),
        "translation_max_m": float(translation_error.max()),
        "rotation_median_deg": float(np.median(rotation_error)),
        "rotation_p95_deg": float(np.quantile(rotation_error, 0.95)),
        "rotation_max_deg": float(rotation_error.max()),
        "step2_world_to_arkit": transform.tolist(),
    }
    if report["translation_p95_m"] > maximum_translation_p95_m:
        raise ValueError(
            f"Step2/ARKit camera translation p95 {report['translation_p95_m']:.3f}m "
            f"> {maximum_translation_p95_m:.3f}m"
        )
    if report["rotation_p95_deg"] > maximum_rotation_p95_deg:
        raise ValueError(
            f"Step2/ARKit camera rotation p95 {report['rotation_p95_deg']:.2f}deg "
            f"> {maximum_rotation_p95_deg:.2f}deg"
        )
    return transform, report


def _scalar(value: np.ndarray) -> str:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"expected scalar, got {array.shape}")
    raw = array.item()
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _object_track(
    data: np.lib.npyio.NpzFile,
    object_id: str,
    world_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    object_ids = [_scalar(np.asarray(value)) for value in np.asarray(data["object_ids"])]
    if object_id not in object_ids:
        raise KeyError(f"object id {object_id!r} not in {object_ids}")
    index = object_ids.index(object_id)
    pose_all = np.asarray(data["object_ob_in_world_all"], dtype=np.float64)
    frame_all = np.asarray(data["object_frame_indices_all"], dtype=np.float64)
    valid_all = np.asarray(data["object_valid_all"], dtype=bool)
    valid = valid_all[index]
    pose = pose_all[index, valid]
    frame = frame_all[index, valid]
    if len(frame) < 2 or not np.all(np.diff(frame) > 0.0):
        raise ValueError(f"{object_id} track must have increasing frames")
    return world_transform @ pose, frame


def _mesh_path(fused_path: Path, data: np.lib.npyio.NpzFile, object_id: str) -> Path:
    object_ids = [_scalar(np.asarray(value)) for value in np.asarray(data["object_ids"])]
    index = object_ids.index(object_id)
    filename = _scalar(np.asarray(data["object_mesh_filenames"])[index])
    path = (fused_path.parent / filename).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _obj_vertices(path: Path) -> np.ndarray:
    vertices = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) >= 4:
                vertices.append([float(fields[1]), float(fields[2]), float(fields[3])])
    value = np.asarray(vertices, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 4:
        raise ValueError(f"mesh has no usable vertices: {path}")
    return value


def estimate_contact_region(
    reference: PourReference,
    *,
    side: str,
    object_pose: np.ndarray,
    mesh_vertices: np.ndarray,
    distance_threshold_m: float = 0.03,
    maximum_points: int = 128,
) -> tuple[np.ndarray, float, dict]:
    """Project video fingertips to the nearest reconstructed mesh vertices."""
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    joints = reference.left_joints if side == "left" else reference.right_joints
    side_index = 0 if side == "left" else 1
    tips_world = joints[:, TIP_INDICES]
    object_matrix = _pose_to_matrix(object_pose)
    rotation = object_matrix[:, :3, :3]
    translation = object_matrix[:, :3, 3]
    tips_local = np.einsum(
        "tij,tfj->tfi", rotation.transpose(0, 2, 1), tips_world - translation[:, None]
    )

    selected: dict[tuple[int, int, int], tuple[float, np.ndarray]] = {}
    observations = 0
    for frame in range(reference.length):
        if reference.hand_confidence[frame, side_index] <= 0.0:
            continue
        for tip in tips_local[frame]:
            distance = np.linalg.norm(mesh_vertices - tip, axis=1)
            nearest_index = int(distance.argmin())
            nearest_distance = float(distance[nearest_index])
            if nearest_distance > distance_threshold_m:
                continue
            observations += 1
            point = mesh_vertices[nearest_index]
            score = float(reference.hand_confidence[frame, side_index]) * np.exp(
                -nearest_distance / max(distance_threshold_m / 2.0, 1.0e-6)
            )
            key = tuple(np.rint(point / 0.002).astype(np.int64))
            if key not in selected or score > selected[key][0]:
                selected[key] = (score, point.copy())
    if not selected:
        raise RuntimeError(
            f"{side} has no fingertip-to-mesh contact within {distance_threshold_m:.3f}m"
        )
    ordered = sorted(selected.values(), key=lambda item: item[0], reverse=True)
    ordered = ordered[:maximum_points]
    points = np.stack([item[1] for item in ordered]).astype(np.float32)
    confidence = float(np.mean([item[0] for item in ordered]))
    diagnostics = {
        "raw_contact_observations": observations,
        "unique_contact_points": len(selected),
        "retained_contact_points": len(points),
        "distance_threshold_m": distance_threshold_m,
        "confidence": confidence,
    }
    return points, confidence, diagnostics


def adapt_step2(
    fused_path: str | Path,
    reference_path: str | Path,
    object_spec_path: str | Path,
    output_dir: str | Path,
    *,
    cup_id: str,
    bottle_id: str,
    maximum_translation_p95_m: float = 0.05,
    maximum_rotation_p95_deg: float = 5.0,
    contact_distance_m: float = 0.03,
) -> dict:
    fused_path = Path(fused_path).resolve()
    reference = PourReference.load(reference_path)
    specs = json.loads(Path(object_spec_path).read_text(encoding="utf-8"))
    if int(specs.get("schema_version", -1)) != 1:
        raise ValueError("object spec schema_version must be 1")
    if str(specs.get("demo_id")) != reference.demo_id:
        raise ValueError("object spec demo_id does not match reference")

    with np.load(fused_path, allow_pickle=False) as data:
        if _scalar(data["coordinate_frame"]) != "gravity_z_up_world":
            raise ValueError("Step-2 fused reconstruction must use gravity_z_up_world")
        world_transform, alignment = fit_step2_world_to_arkit(
            data["c2w"],
            reference,
            maximum_translation_p95_m=maximum_translation_p95_m,
            maximum_rotation_p95_deg=maximum_rotation_p95_deg,
        )
        cup_matrix, cup_frame = _object_track(data, cup_id, world_transform)
        bottle_matrix, bottle_frame = _object_track(data, bottle_id, world_transform)
        cup_mesh = _mesh_path(fused_path, data, cup_id)
        bottle_mesh = _mesh_path(fused_path, data, bottle_id)

    for label, frame in (("cup", cup_frame), ("bottle", bottle_frame)):
        if frame[0] > reference.source_frame[0] or frame[-1] < reference.source_frame[-1]:
            raise ValueError(f"{label} Step-2 track does not cover the reference timeline")

    cup_pose = _interp_pose(_matrix_to_pose(cup_matrix), cup_frame, reference.source_frame)
    bottle_pose = _interp_pose(
        _matrix_to_pose(bottle_matrix), bottle_frame, reference.source_frame
    )
    cup_contact, cup_confidence, cup_contact_diag = estimate_contact_region(
        reference,
        side="left",
        object_pose=cup_pose,
        mesh_vertices=_obj_vertices(cup_mesh),
        distance_threshold_m=contact_distance_m,
    )
    bottle_contact, bottle_confidence, bottle_contact_diag = estimate_contact_region(
        reference,
        side="right",
        object_pose=bottle_pose,
        mesh_vertices=_obj_vertices(bottle_mesh),
        distance_threshold_m=contact_distance_m,
    )

    cup_spec = dict(specs["cup"])
    bottle_spec = dict(specs["bottle"])
    cup_spec["mesh"] = str(cup_mesh)
    bottle_spec["mesh"] = str(bottle_mesh)
    PourObjectSpec(**cup_spec).validate(require_assets=True, require_geometry=True)
    PourObjectSpec(**bottle_spec).validate(require_assets=True, require_geometry=True)

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "cup_track.npz",
        pose=cup_matrix,
        source_frame=cup_frame,
        confidence=np.ones(len(cup_frame), dtype=np.float32),
        coordinate_frame=np.array("arkit_world"),
    )
    np.savez_compressed(
        output / "bottle_track.npz",
        pose=bottle_matrix,
        source_frame=bottle_frame,
        confidence=np.ones(len(bottle_frame), dtype=np.float32),
        coordinate_frame=np.array("arkit_world"),
    )
    geometry = {
        "schema_version": 1,
        "demo_id": reference.demo_id,
        "cup": cup_spec,
        "bottle": bottle_spec,
        "contacts": {
            "coordinate_frame": "object_local",
            "left": {
                "object": "cup",
                "points": cup_contact.tolist(),
                "confidence": cup_confidence,
            },
            "right": {
                "object": "bottle",
                "points": bottle_contact.tolist(),
                "confidence": bottle_confidence,
            },
        },
    }
    (output / "geometry.json").write_text(
        json.dumps(geometry, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "demo_id": reference.demo_id,
        "cup_object_id": cup_id,
        "bottle_object_id": bottle_id,
        "alignment": alignment,
        "contacts": {"left": cup_contact_diag, "right": bottle_contact_diag},
    }
    (output / "adapter_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fused", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--object-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cup-id", required=True)
    parser.add_argument("--bottle-id", required=True)
    parser.add_argument("--maximum-translation-p95-m", type=float, default=0.05)
    parser.add_argument("--maximum-rotation-p95-deg", type=float, default=5.0)
    parser.add_argument("--contact-distance-m", type=float, default=0.03)
    args = parser.parse_args()
    report = adapt_step2(
        args.fused,
        args.reference,
        args.object_spec,
        args.output_dir,
        cup_id=args.cup_id,
        bottle_id=args.bottle_id,
        maximum_translation_p95_m=args.maximum_translation_p95_m,
        maximum_rotation_p95_deg=args.maximum_rotation_p95_deg,
        contact_distance_m=args.contact_distance_m,
    )
    print(
        f"[pour-step2] demo={report['demo_id']} cup={report['cup_object_id']} "
        f"bottle={report['bottle_object_id']} output={Path(args.output_dir).resolve()}"
    )


if __name__ == "__main__":
    main()
