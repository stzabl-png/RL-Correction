#!/usr/bin/env python3
"""Export DexYCB human-demo trajectories into grasp-synthesis sequence dirs.

For each manifest sequence this writes ``human_demo.npz`` (schema documented
in :mod:`ocir.grasp_synthesis.anchored_bodex.demo_data`) into the
corresponding ``sequences/<sequence_id>/`` directory and records it in that
directory's ``sequence.json``. All hand data (MANO vertices + 21 keypoints)
is pre-transformed into the grasped object's canonical model frame, so the
anchored-BODex pipeline never needs the MANO pkl or camera calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from ocir.dexycb.labels import (
    hand_camera_to_object,
    load_manifest,
    load_sequence_mano,
    read_label,
    sequence_by_id,
    valid_mano_frame,
)
from ocir.dexycb.mano_model import pose_m_to_vertices_and_joints

DEFAULT_DATA_ROOT = Path("/data/users/hangkes2/OCIR")
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "processed_data/dex_ycb/manifests/selected_5_sequences.json"
DEFAULT_SEQUENCES_ROOT = DEFAULT_DATA_ROOT / "processed_data/dex_ycb/sequences"
DEXYCB_SEQUENCE_SUFFIX = ("processed_data", "dex_ycb", "sequences")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] OCIR_EXPORT_DEMO {message}", flush=True)


def merge_sequence_json(sequence_dir: Path, updates: dict) -> None:
    path = sequence_dir / "sequence.json"
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    meta.update(updates)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def infer_data_root_from_sequences_root(sequences_root: Path) -> Path | None:
    """Infer OCIR data root from ``.../processed_data/dex_ycb/sequences``."""

    parts = sequences_root.expanduser().resolve().parts
    if len(parts) < len(DEXYCB_SEQUENCE_SUFFIX):
        return None
    if tuple(parts[-len(DEXYCB_SEQUENCE_SUFFIX):]) != DEXYCB_SEQUENCE_SUFFIX:
        return None
    return Path(*parts[:-len(DEXYCB_SEQUENCE_SUFFIX)])


def default_manifest_for_sequences_root(sequences_root: Path) -> Path:
    data_root = infer_data_root_from_sequences_root(sequences_root)
    if data_root is None:
        return DEFAULT_MANIFEST
    return data_root / "processed_data/dex_ycb/manifests/selected_5_sequences.json"


def _replace_path_prefix(value: str, old_root: str, new_root: str) -> str:
    old_root = old_root.rstrip("/")
    if value == old_root:
        return new_root
    if value.startswith(old_root + "/"):
        return new_root.rstrip("/") + value[len(old_root):]
    return value


def localize_manifest_paths(manifest: dict, data_root: Path) -> dict:
    """Return a copy of a DexYCB manifest whose absolute paths point at this machine.

    Prepared manifests often contain the absolute ``data_root`` from the
    machine that generated them. For portable copied datasets, replace that
    prefix in-memory with the data root inferred from the local sequence root.
    The manifest file itself is left untouched.
    """

    old_roots = []
    value = manifest.get("data_root")
    if isinstance(value, str) and value:
        old_roots.append(value)
    old_roots.append(str(DEFAULT_DATA_ROOT))

    seen = set()
    old_roots = [root for root in old_roots if not (root in seen or seen.add(root))]
    new_root = str(data_root.expanduser().resolve())

    def visit(value):
        if isinstance(value, dict):
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        if isinstance(value, str):
            out = value
            for old_root in old_roots:
                out = _replace_path_prefix(out, old_root, new_root)
            return out
        return value

    localized = visit(manifest)
    localized["data_root"] = new_root
    return localized


def export_sequence_demo(manifest: dict, sequence: dict, sequences_root: Path, force: bool) -> dict:
    seq_id = sequence["sequence_id"]
    sequence_dir = sequences_root / seq_id
    if not sequence_dir.is_dir():
        raise FileNotFoundError(f"sequence directory not found: {sequence_dir} (run prepare_dexycb_subset first)")
    demo_path = sequence_dir / "human_demo.npz"
    target_idx = int(sequence.get("ycb_grasp_ind", 0) or 0)
    target_info = sequence["ycb_models"][target_idx]
    if demo_path.exists() and not force:
        merge_sequence_json(
            sequence_dir,
            {
                "sequence_id": seq_id,
                "object_name": target_info["name"],
                "human_demo": demo_path.name,
                "mano_side": str(sequence.get("mano_side", "right")),
                "subject": sequence.get("subject"),
                "canonical_camera": sequence.get("canonical_camera"),
            },
        )
        log(f"{seq_id}: {demo_path.name} exists; use --force to overwrite")
        return {"sequence_id": seq_id, "skipped": True, "demo_path": str(demo_path)}

    mano_model, mano_betas, mano_report = load_sequence_mano(manifest, sequence)
    vertex_part_ids = np.argmax(mano_model.weights, axis=1).astype(np.int8)

    frame_ids = np.asarray([int(f) for f in sequence["frame_ids"]], dtype=np.int32)
    t = frame_ids.shape[0]
    pose_m_camera = np.full((t, 51), np.nan, dtype=np.float32)
    object_pose_camera = np.full((t, 4, 4), np.nan, dtype=np.float32)
    hand_joints_object = np.full((t, 21, 3), np.nan, dtype=np.float32)
    hand_vertices_object = np.full((t, 778, 3), np.nan, dtype=np.float32)
    valid_mask = np.zeros((t,), dtype=bool)

    for i, frame_id in enumerate(frame_ids):
        label = read_label(sequence, int(frame_id))
        if not valid_mano_frame(label):
            continue
        pose_y = np.asarray(label["pose_y"], dtype=float)
        if target_idx >= pose_y.shape[0] or not np.isfinite(pose_y[target_idx]).all():
            continue
        pose_m = np.asarray(label["pose_m"][0], dtype=float)
        vertices_cam, joints_cam = pose_m_to_vertices_and_joints(mano_model, pose_m, mano_betas)
        pose_m_camera[i] = pose_m.astype(np.float32)
        object_pose_camera[i, :3, :] = pose_y[target_idx].astype(np.float32)
        object_pose_camera[i, 3, :] = [0.0, 0.0, 0.0, 1.0]
        hand_vertices_object[i] = hand_camera_to_object(vertices_cam, pose_y[target_idx]).astype(np.float32)
        hand_joints_object[i] = hand_camera_to_object(joints_cam, pose_y[target_idx]).astype(np.float32)
        valid_mask[i] = True

    if not valid_mask.any():
        raise ValueError(f"{seq_id}: no valid MANO+object frames; nothing to export")

    np.savez_compressed(
        demo_path,
        frame_ids=frame_ids,
        mano_side=str(sequence.get("mano_side", "right")),
        betas=np.asarray(mano_betas, dtype=np.float32)[:10],
        pose_m_camera=pose_m_camera,
        object_pose_camera=object_pose_camera,
        hand_joints_object=hand_joints_object,
        hand_vertices_object=hand_vertices_object,
        vertex_part_ids=vertex_part_ids,
        valid_mask=valid_mask,
    )
    merge_sequence_json(
        sequence_dir,
        {
            "sequence_id": seq_id,
            "object_name": target_info["name"],
            "human_demo": demo_path.name,
            "mano_side": str(sequence.get("mano_side", "right")),
            "subject": sequence.get("subject"),
            "canonical_camera": sequence.get("canonical_camera"),
        },
    )
    report = {
        "sequence_id": seq_id,
        "demo_path": str(demo_path),
        "object_name": target_info["name"],
        "num_frames": int(t),
        "num_valid_frames": int(valid_mask.sum()),
        "mano": mano_report,
    }
    log(f"{seq_id}: wrote {demo_path.name} ({report['num_valid_frames']}/{t} valid frames, object {target_info['name']})")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--sequences-root", type=Path, default=DEFAULT_SEQUENCES_ROOT, help="Root holding the per-sequence grasp-synthesis input dirs.")
    parser.add_argument("--sequence-id", action="append", default=None, help="DexYCB sequence id. Omit to export all manifest sequences.")
    parser.add_argument("--localize-manifest-paths", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequences_root = args.sequences_root.expanduser()
    manifest_path = args.manifest.expanduser() if args.manifest is not None else default_manifest_for_sequences_root(sequences_root)
    manifest = load_manifest(manifest_path)
    local_data_root = infer_data_root_from_sequences_root(sequences_root)
    if args.localize_manifest_paths and local_data_root is not None:
        manifest = localize_manifest_paths(manifest, local_data_root)
    if args.sequence_id:
        sequences = [sequence_by_id(manifest, seq_id) for seq_id in args.sequence_id]
    else:
        sequences = list(manifest["sequences"])
    reports = [export_sequence_demo(manifest, sequence, sequences_root, bool(args.force)) for sequence in sequences]
    exported = [r for r in reports if not r.get("skipped")]
    log(f"exported {len(exported)}/{len(reports)} sequences")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
