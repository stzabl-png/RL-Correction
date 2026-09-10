"""Rigidly adapt the accepted take9 tool/grasp pair to an accepted Task3 trajectory.

The transform maps the take9 grasp frame exactly onto the recipient's accepted
grasp frame.  Applying the same transform to the mesh and grasp prior preserves
the take9 hand/tool geometry while leaving the recipient object poses and arm
trajectory byte-for-byte unchanged.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def rot(q_wxyz):
    q = np.asarray(q_wxyz, float)
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def quat_wxyz(matrix):
    q = Rotation.from_matrix(matrix).as_quat()
    return q[[3, 0, 1, 2]]


p = argparse.ArgumentParser()
p.add_argument("--recipient", required=True)
p.add_argument("--take", required=True, choices=("32", "36", "80"))
p.add_argument("--out-config", required=True)
a = p.parse_args()

root = Path(__file__).resolve().parents[3]
recipient = json.loads((root / a.recipient).read_text())
donor_path = root / "tasks/Sweep/new_data/configs/take9_powerdisk_fixed_training_v1.json"
donor = json.loads(donor_path.read_text())
out_config = root / a.out_config
asset_root = root / f"tasks/Sweep/new_data/assets/take_{a.take}/universal_take9_fixed_v1"
prior_root = root / f"tasks/Sweep/new_data/priors/universal_take9_fixed_v1/take_{a.take}"
asset_root.mkdir(parents=True, exist_ok=True)
prior_root.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, glob.glob(
    "/home/msc-auto/rlcorr-venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*"
)[0])
from pxr import Usd, UsdGeom, Vt  # noqa: E402

report = {
    "take": a.take,
    "recipient_config": a.recipient,
    "donor_config": str(donor_path.relative_to(root)),
    "reference_unchanged": recipient["reference"],
    "roles": {},
}

new_priors = {}
new_assets = {}
for role in ("broom", "dustpan"):
    old_prior_path = root / recipient["grasppose"][role]["prior"]
    donor_prior_path = root / donor["grasppose"][role]["prior"]
    old = np.load(old_prior_path, allow_pickle=True)
    src = np.load(donor_prior_path, allow_pickle=True)
    go = np.asarray(old["grasp"], float)
    gs = np.asarray(src["grasp"], float)
    R = rot(go[3:7]) @ rot(gs[3:7]).T
    t = go[:3] - R @ gs[:3]
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-8)
    assert np.linalg.det(R) > .999999

    prior = {k: np.asarray(src[k]).copy() for k in src.files}
    for key in ("grasp", "squeeze", "pregrasp"):
        if key not in prior:
            continue
        q = np.atleast_2d(np.asarray(prior[key], float)).copy()
        q[:, :3] = q[:, :3] @ R.T + t
        for i in range(len(q)):
            q[i, 3:7] = quat_wxyz(R @ rot(q[i, 3:7]))
        prior[key] = q[0] if np.asarray(src[key]).ndim == 1 else q
    for key in ("contact_pos", "contact_centroid"):
        if key in prior:
            prior[key] = np.asarray(prior[key]) @ R.T + t
    if "contact_normal" in prior:
        prior["contact_normal"] = np.asarray(prior["contact_normal"]) @ R.T
    prior["universal_take9_to_recipient_R"] = R
    prior["universal_take9_to_recipient_t"] = t
    prior["frame_schema"] = np.array(
        "take9 universal input -> recipient accepted input; exact grasp-frame alignment"
    )
    prior_path = prior_root / f"{role}.npz"
    np.savez(prior_path, **prior)
    assert np.allclose(prior["grasp"][:7], go[:7], atol=2e-6)
    assert np.array_equal(prior["grasp"][7:], src["grasp"][7:])
    new_priors[role] = str(prior_path.relative_to(root))

    src_mesh_path = root / donor["assets"][f"{role}_mesh"]
    mesh = trimesh.load(src_mesh_path, force="mesh", process=False)
    old_faces = np.asarray(mesh.faces).copy()
    mesh.vertices = np.asarray(mesh.vertices) @ R.T + t
    role_dir = asset_root / role
    role_dir.mkdir(exist_ok=True)
    mesh_path = role_dir / "object_mesh_scaled_final.obj"
    mesh.export(mesh_path)
    assert np.array_equal(mesh.faces, old_faces)
    new_assets[f"{role}_mesh"] = str(mesh_path.relative_to(root))

    src_usd = root / donor["assets"][f"{role}_usd"]
    stage = Usd.Stage.Open(str(src_usd))
    stage = Usd.Stage.Open(stage.Flatten())
    rigid = stage.GetDefaultPrim()
    cache = UsdGeom.XformCache()
    transformed = 0
    max_identity_error = 0.0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        geom = UsdGeom.Mesh(prim)
        mesh_to_root = np.asarray(
            cache.GetLocalToWorldTransform(prim)
            * cache.GetLocalToWorldTransform(rigid).GetInverse()
        )
        max_identity_error = max(max_identity_error,
                                 float(np.abs(mesh_to_root - np.eye(4)).max()))
        assert np.allclose(mesh_to_root, np.eye(4), atol=1e-6), str(prim.GetPath())
        points = np.asarray(geom.GetPointsAttr().Get(), float)
        points = points @ R.T + t
        geom.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        geom.GetNormalsAttr().Clear()
        geom.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(
            np.stack((points.min(0), points.max(0))).astype(np.float32)))
        transformed += 1
    assert transformed > 0
    usd_path = role_dir / f"{role}.usd"
    stage.GetRootLayer().Export(str(usd_path))
    new_assets[f"{role}_usd"] = str(usd_path.relative_to(root))

    report["roles"][role] = {
        "R": R.tolist(), "t_m": t.tolist(),
        "donor_prior": str(donor_prior_path.relative_to(root)),
        "recipient_anchor_prior": str(old_prior_path.relative_to(root)),
        "output_prior": str(prior_path.relative_to(root)),
        "output_mesh": str(mesh_path.relative_to(root)),
        "output_usd": str(usd_path.relative_to(root)),
        "grasp_pose_error_m": float(np.linalg.norm(prior["grasp"][:3] - go[:3])),
        "grasp_rotation_error_deg": float(np.degrees(
            Rotation.from_matrix(rot(prior["grasp"][3:7]).T @ rot(go[3:7])).magnitude())),
        "take9_fingers_exact": True,
        "mesh_vertices": int(len(mesh.vertices)), "mesh_faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "usd_mesh_count": transformed,
        "usd_mesh_to_root_max_identity_error": max_identity_error,
    }

cfg = recipient.copy()
cfg["task_name"] = f"Task3Take{a.take}UniversalFixed"
cfg["assets"] = new_assets
for role in ("broom", "dustpan"):
    spec = dict(donor["grasppose"][role])
    spec["prior"] = new_priors[role]
    cfg["grasppose"][role] = spec
cfg["grip_mode"] = "fixed"
cfg["scripted_prelude_steps"] = 0
cfg["finger_pose"] = "grasp"
cfg["grip_tighten_rad"] = 0.0
cfg["hand_friction"] = donor.get("hand_friction", 10.0)
cfg["thin_entry_collision"] = donor["thin_entry_collision"]
cfg["universal_asset_substitution"] = {
    "version": "take9_fixed_v1",
    "method": "exact grasp-frame rigid alignment",
    "recipient_object_poses_unchanged": True,
    "recipient_arm_trajectory_unchanged": True,
    "take9_tool_grasp_geometry_preserved": True,
    "status": "pending zero-residual video review",
}
cfg["acceptance"] = "pending_zero_residual_video_review"
cfg["cube_status"] = "provisional_disabled"
out_config.parent.mkdir(parents=True, exist_ok=True)
out_config.write_text(json.dumps(cfg, indent=2) + "\n")
(asset_root / "substitution_audit.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
