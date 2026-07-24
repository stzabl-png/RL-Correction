"""Identity DexYCB manifest for world-frame trajectories.

Stage B (``simulate_grasp_traj``) maps a trajectory from "camera frame" into
Isaac world via a DexYCB apriltag extrinsic looked up in a manifest. When a
trajectory is already authored in a z-up world frame (e.g. the RL_Correction /
egodex frame), we want that mapping to be the identity. ``DexYCBFrameMapper``
computes ``camera_to_tag = tag_R.T @ cam_R`` and ``t = tag_R.T @ (cam_t - tag_t)``
-- so making the apriltag and camera extrinsics the SAME ``[I|0]`` matrix yields
rotation = I, translation = 0, i.e. camera frame == world frame.

``ensure_identity_manifest`` creates/updates such a manifest and registers a
sequence id, so no manual manifest editing is ever needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

_IDENTITY_3x4 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
_CAMERA = "cam0"
_EXTRINSICS = "identity"


def ensure_identity_manifest(root: str | Path, sequence_id: str) -> Path:
    """Ensure an identity DexYCB manifest exists under ``root`` and contains an
    entry for ``sequence_id``. Returns the manifest.json path (absolute)."""

    root = Path(root)
    calib = root / "calibration" / f"extrinsics_{_EXTRINSICS}"
    calib.mkdir(parents=True, exist_ok=True)

    ext_yml = calib / "extrinsics.yml"
    if not ext_yml.exists():
        ext_yml.write_text(
            yaml.safe_dump({"extrinsics": {"apriltag": _IDENTITY_3x4, _CAMERA: _IDENTITY_3x4}}),
            encoding="utf-8",
        )
    meta_yml = root / "meta_identity.yml"
    if not meta_yml.exists():
        meta_yml.write_text(yaml.safe_dump({"extrinsics": _EXTRINSICS}), encoding="utf-8")

    manifest = root / "manifest.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else {}
    data["selected_root"] = str(root.resolve())
    seqs = data.setdefault("sequences", [])
    if not any(s.get("sequence_id") == sequence_id for s in seqs):
        seqs.append({
            "sequence_id": sequence_id,
            "canonical_camera": _CAMERA,
            "meta": str(meta_yml.resolve()),
        })
    manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return manifest.resolve()
