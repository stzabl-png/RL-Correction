"""Build and validate Task-5 video reference artifacts.

EgoDex stores every transform in one stationary ARKit origin frame.  We keep
that frame (including the camera trajectory) and resample from 30 Hz to the
20 Hz policy rate.  Camera-frame reconstruction tracks are transformed back
to this stationary frame by ``stage_reconstruction``.  Object tracks and
video contact regions are optional here because raw EgoDex HDF5 does not
contain them; the builder never invents either signal.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tasks.pour.enums import PourPhase


SCHEMA_VERSION = 2
HAND_JOINT_SUFFIXES = (
    "Hand",
    "ThumbKnuckle",
    "ThumbIntermediateBase",
    "ThumbIntermediateTip",
    "ThumbTip",
    "IndexFingerMetacarpal",
    "IndexFingerKnuckle",
    "IndexFingerIntermediateBase",
    "IndexFingerIntermediateTip",
    "IndexFingerTip",
    "MiddleFingerMetacarpal",
    "MiddleFingerKnuckle",
    "MiddleFingerIntermediateBase",
    "MiddleFingerIntermediateTip",
    "MiddleFingerTip",
    "RingFingerMetacarpal",
    "RingFingerKnuckle",
    "RingFingerIntermediateBase",
    "RingFingerIntermediateTip",
    "RingFingerTip",
    "LittleFingerMetacarpal",
    "LittleFingerKnuckle",
    "LittleFingerIntermediateBase",
    "LittleFingerIntermediateTip",
    "LittleFingerTip",
)


def _validate_transforms(name: str, value: np.ndarray) -> None:
    if value.ndim != 3 or value.shape[1:] != (4, 4):
        raise ValueError(f"{name} must be [T,4,4], got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN/Inf")
    last = value[:, 3]
    expected = np.zeros_like(last)
    expected[:, 3] = 1.0
    if not np.allclose(last, expected, atol=1.0e-4):
        raise ValueError(f"{name} has invalid homogeneous rows")
    rotation = value[:, :3, :3]
    eye = np.eye(3, dtype=rotation.dtype)
    if np.max(np.abs(rotation @ np.swapaxes(rotation, 1, 2) - eye)) > 2.0e-3:
        raise ValueError(f"{name} rotations are not orthonormal")


def rotation_matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """Convert one or more 3x3 rotations to sign-continuous wxyz quaternions."""

    matrices = np.asarray(rotation, dtype=np.float64)
    flat = matrices.reshape(-1, 3, 3)
    result = np.empty((len(flat), 4), dtype=np.float64)
    for index, matrix in enumerate(flat):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quat = np.array(
                [
                    0.25 * scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
        else:
            diagonal = np.diag(matrix)
            axis = int(np.argmax(diagonal))
            j, k = (axis + 1) % 3, (axis + 2) % 3
            scale = np.sqrt(1.0 + matrix[axis, axis] - matrix[j, j] - matrix[k, k]) * 2.0
            quat = np.zeros(4)
            quat[0] = (matrix[k, j] - matrix[j, k]) / scale
            quat[axis + 1] = 0.25 * scale
            quat[j + 1] = (matrix[j, axis] + matrix[axis, j]) / scale
            quat[k + 1] = (matrix[k, axis] + matrix[axis, k]) / scale
        result[index] = quat / np.linalg.norm(quat)
    for index in range(1, len(result)):
        if np.dot(result[index - 1], result[index]) < 0.0:
            result[index] *= -1.0
    return result.reshape(matrices.shape[:-2] + (4,)).astype(np.float32)


def _smooth_positions(value: np.ndarray, width: int = 5) -> np.ndarray:
    if width <= 1:
        return value.copy()
    pad = width // 2
    padded = np.pad(value, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.full(width, 1.0 / width, dtype=np.float64)
    return np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(3)], axis=1)


def _resample_linear(value: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    source_t = np.arange(len(value), dtype=np.float64) / source_hz
    count = int(round((len(value) - 1) * target_hz / source_hz)) + 1
    target_t = np.arange(count, dtype=np.float64) / target_hz
    target_t = np.minimum(target_t, source_t[-1])
    flat = value.reshape(len(value), -1)
    out = np.stack([np.interp(target_t, source_t, flat[:, index]) for index in range(flat.shape[1])], axis=1)
    return out.reshape((count,) + value.shape[1:]).astype(np.float32)


def _resample_quaternion(value: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    source_t = np.arange(len(value), dtype=np.float64) / source_hz
    count = int(round((len(value) - 1) * target_hz / source_hz)) + 1
    target_t = np.minimum(np.arange(count, dtype=np.float64) / target_hz, source_t[-1])
    hi = np.searchsorted(source_t, target_t, side="right").clip(1, len(value) - 1)
    lo = hi - 1
    alpha = ((target_t - source_t[lo]) / np.maximum(source_t[hi] - source_t[lo], 1.0e-9))[:, None]
    q0, q1 = value[lo].astype(np.float64), value[hi].astype(np.float64)
    q1 = np.where((q0 * q1).sum(axis=1, keepdims=True) < 0.0, -q1, q1)
    out = (1.0 - alpha) * q0 + alpha * q1
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out.astype(np.float32)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    padded = np.r_[False, mask.astype(bool), False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    runs = list(zip(changes[::2], changes[1::2] - 1, strict=True))
    if not runs:
        raise ValueError("no sustained pour rotation found")
    return max(runs, key=lambda item: item[1] - item[0])


def infer_video_phases(
    right_wrist_camera: np.ndarray,
    *,
    core_tilt_deg: float = 60.0,
    boundary_tilt_deg: float = 45.0,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Infer ALIGN/POUR/RETURN from the sustained right-wrist rotation."""

    rotation = right_wrist_camera[:, :3, :3]
    relative = np.einsum("ij,tjk->tik", rotation[0].T, rotation)
    cosine = ((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0).clip(-1.0, 1.0)
    tilt = np.arccos(cosine)
    core_start, core_end = _longest_true_run(np.rad2deg(tilt) >= core_tilt_deg)
    expanded = np.rad2deg(tilt) >= boundary_tilt_deg
    start = core_start
    while start > 0 and expanded[start - 1]:
        start -= 1
    end = core_end
    while end + 1 < len(expanded) and expanded[end + 1]:
        end += 1
    phase = np.full(len(tilt), int(PourPhase.ALIGN), dtype=np.int64)
    phase[start : end + 1] = int(PourPhase.POUR)
    phase[end + 1 :] = int(PourPhase.RETURN)
    return phase, tilt.astype(np.float32), (start, end)


@dataclass
class PourReference:
    demo_id: str
    fps: float
    camera_intrinsic: np.ndarray
    camera_pose: np.ndarray
    left_wrist: np.ndarray
    right_wrist: np.ndarray
    left_joints: np.ndarray
    right_joints: np.ndarray
    video_phase: np.ndarray
    right_tilt_rad: np.ndarray
    hand_confidence: np.ndarray
    source_frame: np.ndarray
    bottle_pose: np.ndarray
    cup_pose: np.ndarray
    object_confidence: np.ndarray
    video_contact_left: np.ndarray
    video_contact_right: np.ndarray
    video_contact_confidence: np.ndarray

    @property
    def length(self) -> int:
        return int(len(self.video_phase))

    @property
    def has_object_tracks(self) -> bool:
        return bool(np.isfinite(self.bottle_pose).all() and np.isfinite(self.cup_pose).all())

    @property
    def has_video_contacts(self) -> bool:
        return bool(
            len(self.video_contact_left) > 0
            and len(self.video_contact_right) > 0
            and np.isfinite(self.video_contact_left).all()
            and np.isfinite(self.video_contact_right).all()
            and np.all(self.video_contact_confidence > 0.0)
        )

    @property
    def training_ready(self) -> bool:
        return bool(
            self.has_object_tracks
            and self.has_video_contacts
            and np.all(self.object_confidence > 0.0)
        )

    def validate(self) -> None:
        length = self.length
        expected = {
            "camera_pose": (length, 7),
            "left_wrist": (length, 7),
            "right_wrist": (length, 7),
            "left_joints": (length, len(HAND_JOINT_SUFFIXES), 3),
            "right_joints": (length, len(HAND_JOINT_SUFFIXES), 3),
            "right_tilt_rad": (length,),
            "hand_confidence": (length, 2),
            "source_frame": (length,),
            "bottle_pose": (length, 7),
            "cup_pose": (length, 7),
            "object_confidence": (length, 2),
            "video_contact_confidence": (2,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} != {shape}")
        for name in ("video_contact_left", "video_contact_right"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 2 or value.shape[1] != 3:
                raise ValueError(f"{name} must be [N,3], got {value.shape}")
        allowed = {int(PourPhase.ALIGN), int(PourPhase.POUR), int(PourPhase.RETURN)}
        if not set(np.unique(self.video_phase)).issubset(allowed):
            raise ValueError("video_phase contains a non-video phase")
        if not np.isfinite(self.left_wrist).all() or not np.isfinite(self.right_wrist).all():
            raise ValueError("wrist reference contains NaN/Inf")

    def save(self, path: str | Path) -> None:
        self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            schema_version=np.array(SCHEMA_VERSION, dtype=np.int64),
            demo_id=np.array(self.demo_id),
            fps=np.array(self.fps, dtype=np.float32),
            camera_intrinsic=self.camera_intrinsic.astype(np.float32),
            camera_pose=self.camera_pose.astype(np.float32),
            left_wrist=self.left_wrist.astype(np.float32),
            right_wrist=self.right_wrist.astype(np.float32),
            left_joints=self.left_joints.astype(np.float32),
            right_joints=self.right_joints.astype(np.float32),
            video_phase=self.video_phase.astype(np.int64),
            right_tilt_rad=self.right_tilt_rad.astype(np.float32),
            hand_confidence=self.hand_confidence.astype(np.float32),
            source_frame=self.source_frame.astype(np.float32),
            bottle_pose=self.bottle_pose.astype(np.float32),
            cup_pose=self.cup_pose.astype(np.float32),
            object_confidence=self.object_confidence.astype(np.float32),
            video_contact_left=self.video_contact_left.astype(np.float32),
            video_contact_right=self.video_contact_right.astype(np.float32),
            video_contact_confidence=self.video_contact_confidence.astype(np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PourReference":
        with np.load(path, allow_pickle=False) as data:
            version = int(data["schema_version"])
            if version != SCHEMA_VERSION:
                raise ValueError(f"reference schema {version} != supported {SCHEMA_VERSION}")
            artifact = cls(
                demo_id=str(data["demo_id"]),
                fps=float(data["fps"]),
                camera_intrinsic=data["camera_intrinsic"],
                camera_pose=data["camera_pose"],
                left_wrist=data["left_wrist"],
                right_wrist=data["right_wrist"],
                left_joints=data["left_joints"],
                right_joints=data["right_joints"],
                video_phase=data["video_phase"],
                right_tilt_rad=data["right_tilt_rad"],
                hand_confidence=data["hand_confidence"],
                source_frame=data["source_frame"],
                bottle_pose=data["bottle_pose"],
                cup_pose=data["cup_pose"],
                object_confidence=data["object_confidence"],
                video_contact_left=data["video_contact_left"],
                video_contact_right=data["video_contact_right"],
                video_contact_confidence=data["video_contact_confidence"],
            )
        artifact.validate()
        return artifact


def build_egodex_reference(
    hdf5_path: str | Path,
    *,
    demo_id: str,
    source_hz: float = 30.0,
    target_hz: float = 20.0,
) -> PourReference:
    import h5py

    with h5py.File(hdf5_path, "r") as data:
        camera = data["transforms/camera"][...].astype(np.float64)
        _validate_transforms("camera", camera)
        wrists: dict[str, np.ndarray] = {}
        joints: dict[str, np.ndarray] = {}
        confidences: list[np.ndarray] = []
        for side in ("left", "right"):
            wrist_world = data[f"transforms/{side}Hand"][...].astype(np.float64)
            _validate_transforms(f"{side}Hand", wrist_world)
            wrists[side] = wrist_world
            joint_positions = []
            for suffix in HAND_JOINT_SUFFIXES:
                transform = data[f"transforms/{side}{suffix}"][...].astype(np.float64)
                _validate_transforms(f"{side}{suffix}", transform)
                joint_positions.append(transform[:, :3, 3])
            joints[side] = np.stack(joint_positions, axis=1)
            confidence_key = f"confidences/{side}Hand"
            confidences.append(
                data[confidence_key][...].astype(np.float32)
                if confidence_key in data
                else np.ones(len(camera), dtype=np.float32)
            )
        intrinsic = data["camera/intrinsic"][...].astype(np.float32)

    raw_phase, raw_tilt, _ = infer_video_phases(wrists["right"])
    left_pos = _resample_linear(_smooth_positions(wrists["left"][:, :3, 3]), source_hz, target_hz)
    right_pos = _resample_linear(_smooth_positions(wrists["right"][:, :3, 3]), source_hz, target_hz)
    left_quat = _resample_quaternion(rotation_matrix_to_wxyz(wrists["left"][:, :3, :3]), source_hz, target_hz)
    right_quat = _resample_quaternion(rotation_matrix_to_wxyz(wrists["right"][:, :3, :3]), source_hz, target_hz)
    frame = _resample_linear(np.arange(len(camera), dtype=np.float32)[:, None], source_hz, target_hz)[:, 0]
    nearest = np.rint(frame).astype(np.int64).clip(0, len(camera) - 1)
    length = len(frame)
    missing_pose = np.full((length, 7), np.nan, dtype=np.float32)
    artifact = PourReference(
        demo_id=demo_id,
        fps=target_hz,
        camera_intrinsic=intrinsic,
        camera_pose=np.concatenate(
            [
                _resample_linear(
                    _smooth_positions(camera[:, :3, 3]), source_hz, target_hz
                ),
                _resample_quaternion(
                    rotation_matrix_to_wxyz(camera[:, :3, :3]),
                    source_hz,
                    target_hz,
                ),
            ],
            axis=1,
        ),
        left_wrist=np.concatenate([left_pos, left_quat], axis=1),
        right_wrist=np.concatenate([right_pos, right_quat], axis=1),
        left_joints=_resample_linear(joints["left"], source_hz, target_hz),
        right_joints=_resample_linear(joints["right"], source_hz, target_hz),
        video_phase=raw_phase[nearest],
        right_tilt_rad=_resample_linear(raw_tilt[:, None], source_hz, target_hz)[:, 0],
        hand_confidence=_resample_linear(np.stack(confidences, axis=1), source_hz, target_hz),
        source_frame=frame,
        bottle_pose=missing_pose.copy(),
        cup_pose=missing_pose.copy(),
        object_confidence=np.zeros((length, 2), dtype=np.float32),
        video_contact_left=np.empty((0, 3), dtype=np.float32),
        video_contact_right=np.empty((0, 3), dtype=np.float32),
        video_contact_confidence=np.zeros(2, dtype=np.float32),
    )
    artifact.validate()
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--demo-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-hz", type=float, default=20.0)
    args = parser.parse_args()
    reference = build_egodex_reference(
        args.hdf5, demo_id=args.demo_id, target_hz=args.target_hz
    )
    reference.save(args.output)
    pour_frames = np.flatnonzero(reference.video_phase == int(PourPhase.POUR))
    print(
        f"[pour-reference] {args.demo_id}: {reference.length} frames @ {reference.fps:g}Hz "
        f"pour=[{pour_frames[0]},{pour_frames[-1]}] training_ready={reference.training_ready}"
    )


if __name__ == "__main__":
    main()
