"""Shared DexYCB manifest/label reading helpers.

These were originally private helpers of the standalone affordance extractor;
they are shared here so the ``export_grasp_sequences`` exporter (and any other
DexYCB consumer) can read the selected-subset manifest, per-frame labels, and
per-sequence MANO calibration without duplicating the logic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ocir.dexycb.mano_model import ManoModel, load_mano_betas, load_mano_model


def load_manifest(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing DexYCB manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sequence_by_id(manifest: dict, sequence_id: str) -> dict:
    for sequence in manifest["sequences"]:
        if sequence["sequence_id"] == sequence_id:
            return sequence
    raise KeyError(f"sequence_id not found in manifest: {sequence_id}")


def load_sequence_mano(manifest: dict, sequence: dict) -> tuple[ManoModel, np.ndarray, dict]:
    side = str(sequence.get("mano_side", "right")).lower()
    model_key = "MANO_LEFT.pkl" if side == "left" else "MANO_RIGHT.pkl"
    model = load_mano_model(manifest["mano_models"][model_key], side=side)

    calib = sequence.get("mano_calib")
    calib_name = calib[0] if isinstance(calib, list) else calib
    if not calib_name:
        raise ValueError(f"sequence {sequence['sequence_id']} has no MANO calibration entry")
    betas_path = Path(manifest["selected_root"]) / "calibration" / f"mano_{calib_name}" / "mano.yml"
    betas = load_mano_betas(betas_path)
    return model, betas, {"side": side, "model_path": manifest["mano_models"][model_key], "betas_path": str(betas_path)}


def read_label(sequence: dict, frame_id: int) -> dict[str, np.ndarray]:
    path = Path(sequence["path"]) / sequence["canonical_camera"] / f"labels_{frame_id:06d}.npz"
    data = np.load(path)
    return {key: data[key] for key in data.files}


def valid_mano_frame(label: dict[str, np.ndarray]) -> bool:
    joints = np.asarray(label["joint_3d"][0], dtype=float)
    pose_m = np.asarray(label["pose_m"], dtype=float)
    if not np.isfinite(joints).all() or np.any(joints < -0.5):
        return False
    if not np.isfinite(pose_m).all() or np.allclose(pose_m, 0.0):
        return False
    return True


def hand_camera_to_object(hand_points_cam: np.ndarray, pose_y: np.ndarray) -> np.ndarray:
    """Transform camera-frame hand points into the object's canonical model frame."""

    pose_y = np.asarray(pose_y, dtype=float)
    rot = pose_y[:3, :3]
    pos = pose_y[:3, 3]
    return (np.asarray(hand_points_cam, dtype=float) - pos[None, :]) @ rot
