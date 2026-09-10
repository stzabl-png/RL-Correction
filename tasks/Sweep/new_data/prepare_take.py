"""Validate and convert one Sweep Part-4 take to the existing Sweep prior interface.

The delivered GraspPose candidates are in the Dexonomy canonical object frame.
`region_rank.json` carries the exact canonical transform used during synthesis, so
this script materializes that metadata and delegates the numerical conversion to
the already validated `tasks/pregrasp/make_prior.py` implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, np.float64) / np.linalg.norm(quat)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])


def validate_grasp(path: Path, expected_hand: str) -> None:
    grasp = np.load(path, allow_pickle=True).item()
    expected = {
        "grasp_qpos": (1, 29),
        "squeeze_qpos": (1, 29),
        "pregrasp_qpos": (6, 29),
        "obj_pose": (7,),
        "obj_scale": (3,),
    }
    for key, shape in expected.items():
        actual = np.asarray(grasp[key]).shape
        if actual != shape:
            raise ValueError(f"{path}: {key} is {actual}, expected {shape}")
    hand = str(grasp.get("hand_name", ""))
    side_matches = ((expected_hand == "left" and "left" in hand)
                    or (expected_hand == "right" and hand and "left" not in hand))
    if not side_matches:
        raise ValueError(f"{path}: hand_name={hand!r}, expected {expected_hand}")


def selected_row(rank_path: Path, filename: str) -> tuple[dict, dict]:
    payload = json.loads(rank_path.read_text())
    rows = [row for row in payload["rows"] if row["file"] == filename]
    if len(rows) != 1:
        raise ValueError(f"{rank_path}: expected one row for {filename}, got {len(rows)}")
    row = rows[0]
    if not row["functional"] or row.get("duplicate_of"):
        raise ValueError(f"selected candidate is not a distinct functional grasp: {row}")
    canonical = payload.get("canonical_frame") or {}
    if "canonical_from_input_rot_wxyz" not in canonical:
        raise ValueError(f"{rank_path}: missing exact canonical transform")
    return row, canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--take", required=True, choices=("32", "80", "128", "180"))
    args = parser.parse_args()

    config_path = CONFIG_DIR / f"take_{args.take}.json"
    config = json.loads(config_path.read_text())
    data_dir = resolve(config["data_dir"])
    output = {"take": int(args.take), "config": str(config_path.relative_to(ROOT)), "tools": {}}
    replay = np.load(data_dir / "replay_world.npz", allow_pickle=True)
    if float(np.asarray(replay["fps"])) != 15.0:
        raise ValueError(f"{data_dir}: expected 15 Hz replay")
    if [str(value) for value in replay["object_ids"]] != ["object_0", "object_1"]:
        raise ValueError(f"{data_dir}: unexpected object ordering")

    # Part-4 lacks only the phase channels expected by the successful Sweep2
    # loader. Materialize an immutable, task-owned interface copy; SweepEnv later
    # replaces the generic loader motion with the validated bimanual reference.
    replay_interface = resolve(config["training_replay"])
    replay_interface.parent.mkdir(parents=True, exist_ok=True)
    count = len(replay["frames"])
    phase = np.ones(count, dtype=np.int8)
    phase[max(0, count - 10):] = 2
    replay_payload = {key: replay[key] for key in replay.files}
    replay_payload.update(
        phase_left=phase.copy(), phase_right=phase.copy(),
        phase_obj=np.ones(count, dtype=np.int8),
        phase_source=np.asarray(["task3_interface"], dtype="<U15"),
    )
    np.savez_compressed(replay_interface, **replay_payload)

    for role in ("dustpan", "broom"):
        spec = config["grasppose"][role]
        delivery = resolve(spec["delivery_dir"])
        candidate = delivery / "grasp_data" / spec["candidate"]
        rank_path = delivery / "region_rank.json"
        validate_grasp(candidate, spec["hand"])
        row, canonical = selected_row(rank_path, spec["candidate"])

        info_path = resolve(spec["canonical_info"])
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps(canonical, indent=2) + "\n")
        prior_path = resolve(spec["prior"])
        prior_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            sys.executable,
            str(ROOT / "tasks/pregrasp/make_prior.py"),
            "--grasp_npy", str(candidate),
            "--info_json", str(info_path),
            "--out", str(prior_path),
        ], check=True)
        prior = np.load(prior_path, allow_pickle=False)
        for key, shape in {"grasp": (29,), "squeeze": (29,), "pregrasp": (6, 29)}.items():
            if prior[key].shape != shape:
                raise ValueError(f"{prior_path}: {key}={prior[key].shape}, expected {shape}")
        output["tools"][role] = {
            "object_id": spec["object_id"],
            "candidate": str(candidate.relative_to(ROOT)),
            "candidate_sha256": sha256(candidate),
            "prior": str(prior_path.relative_to(ROOT)),
            "prior_sha256": sha256(prior_path),
            "rank": int(row["rank"]),
            "demo_angle_deg": float(row["demo_angle_deg"]),
            "in_region_frac": row.get("in_region_frac"),
            "fingertip_dp_contacts": int(row["fingertip_dp_contacts"]),
            "thumb_contact": bool(row["thumb_contact"]),
        }

    poses = np.asarray(replay["obj_pose_all"], np.float64)
    pan_mesh = trimesh.load(data_dir / "objects/object_0/object_mesh_scaled_final.obj",
                            force="mesh", process=False)
    pan_pose0 = poses[0, 0]
    pan_world0 = (np.asarray(pan_mesh.vertices) @ quat_to_matrix(pan_pose0[3:7]).T
                  + pan_pose0[:3])
    scene_table_z = float(pan_world0[:, 2].min())
    scene_layout = {
        "schema_version": "sweep_held_scene_v1",
        "scene_table_z": scene_table_z,
        "source_frame": 0,
        "objects": {
            f"object_{index}": {
                "identity": "dustpan" if index == 0 else "broom",
                "anchor_hand": "left" if index == 0 else "right",
                "pos": poses[index, 0, :3].tolist(),
                "quat_wxyz": poses[index, 0, 3:7].tolist(),
            } for index in (0, 1)
        },
    }
    layout_path = resolve(config["scene_layout"])
    layout_path.parent.mkdir(parents=True, exist_ok=True)
    layout_path.write_text(json.dumps(scene_layout, indent=2) + "\n")
    output["source"] = {
        "fps": 15.0,
        "scene_table_z": scene_table_z,
        "scene_layout": str(layout_path.relative_to(ROOT)),
        "training_replay": str(replay_interface.relative_to(ROOT)),
        "training_replay_sha256": sha256(replay_interface),
        "mesh_extents_m": {
            f"object_{index}": np.asarray(trimesh.load(
                data_dir / f"objects/object_{index}/object_mesh_scaled_final.obj",
                force="mesh", process=False).extents).tolist()
            for index in (0, 1)
        },
    }

    manifest = resolve(config["prepared_manifest"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[prepare_take] validated take {args.take} -> {manifest}")


if __name__ == "__main__":
    main()
