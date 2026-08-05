"""Export Dexonomy grasp npy files to ocir-grasp-synthesis Stage B trajectory dirs.

  python tools/export_isaac_traj.py --exp-dir output/pp2_sharpa_wave --data grasp_data \
      [--out-root output/pp2_sharpa_wave/isaac_traj] [--limit N]

Each grasp npy becomes <out-root>/<name>/trajectory.{npz,json} following
src/ocir/grasp_traj/trajectory_schema.py. Authored in the Dexonomy object-canonical
frame (z-up, object COM at origin) — Stage B auto-snaps absolute z so the object
bottom lands on the table; only hand-relative-to-object geometry matters.

Sequence layout (dt=1/30):
  approach(1): 20 frames — wrist descends from +6cm above pregrasp[0], fingers open
  close(2):    interpolated pregrasp[0..K] -> grasp_qpos
  squeeze(3):  15 frames — fingers grasp -> squeeze targets, wrist fixed
  carry(4):    45 frames — wrist rises +8cm, fingers hold squeeze targets
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

JOINT_ORDER = [
    "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE", "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA", "right_pinky_PIP",
    "right_pinky_DIP",
]


def interp_qpos29(keys, n_frames):
    """Interpolate a (K,29) keyframe array to n_frames (linear pos/joints, slerp quat)."""
    keys = np.asarray(keys, np.float64)
    k = len(keys)
    t_key = np.linspace(0.0, 1.0, k)
    t_out = np.linspace(0.0, 1.0, n_frames)
    pos = np.stack([np.interp(t_out, t_key, keys[:, i]) for i in range(3)], axis=1)
    joints = np.stack([np.interp(t_out, t_key, keys[:, i]) for i in range(7, 29)], axis=1)
    rots = R.from_quat(np.roll(keys[:, 3:7], -1, axis=1))  # wxyz -> xyzw
    quat = np.roll(Slerp(t_key, rots)(t_out).as_quat(), 1, axis=1)  # -> wxyz
    return pos, quat, joints


def export_one(npy_path, out_dir, mesh_path, approach_h, lift_h):
    d = np.load(npy_path, allow_pickle=True).item()
    pre = np.asarray(d["pregrasp_qpos"], np.float64)      # (K,29) open -> near
    grasp = np.asarray(d["grasp_qpos"], np.float64)[0]
    squeeze = np.asarray(d["squeeze_qpos"], np.float64)[0]

    # approach: descend from +approach_h above pregrasp[0]
    start = pre[0].copy()
    start[2] += approach_h
    a_pos, a_quat, a_j = interp_qpos29([start, pre[0]], 20)
    # close: through the pregrasp sequence into the grasp pose
    c_pos, c_quat, c_j = interp_qpos29(np.concatenate([pre, grasp[None]], axis=0), 28)
    # squeeze: fingers only
    s_pos, s_quat, s_j = interp_qpos29([grasp, squeeze], 15)
    s_pos[:] = grasp[:3]
    s_quat[:] = grasp[3:7]
    # carry: wrist straight up, fingers hold squeeze
    lift_end = squeeze.copy()
    lift_end[:3] = grasp[:3]
    lift_end[3:7] = grasp[3:7]
    lift_end[2] += lift_h
    carry_start = lift_end.copy()
    carry_start[2] -= lift_h
    y_pos, y_quat, y_j = interp_qpos29([carry_start, lift_end], 45)

    hand_pos = np.concatenate([a_pos, c_pos, s_pos, y_pos])
    hand_quat = np.concatenate([a_quat, c_quat, s_quat, y_quat])
    fingers = np.concatenate([a_j, c_j, s_j, y_j])
    segment = np.concatenate([
        np.full(len(a_pos), 1), np.full(len(c_pos), 2),
        np.full(len(s_pos), 3), np.full(len(y_pos), 4),
    ]).astype(np.int8)
    T = len(hand_pos)

    obj_pos = np.tile(np.zeros(3), (T, 1))
    obj_quat = np.tile(np.array([1.0, 0, 0, 0]), (T, 1))

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_dir, "trajectory.npz"),
        hand_pos_camera=hand_pos, hand_quat_camera=hand_quat,
        finger_targets=fingers, object_pos_camera=obj_pos,
        object_quat_camera=obj_quat, segment=segment)
    meta = {
        "dt": 1.0 / 30.0,
        "joint_order": JOINT_ORDER,
        "grasp_json": os.path.abspath(npy_path),
        "sequence_dir": f"external/{os.path.basename(out_dir)}",
        "switch_frame_index": 0,
        "grasp_root_tf": np.eye(4).tolist(),
        "extra_metadata": {
            "object_mesh": os.path.abspath(mesh_path),
            "object_name": "object",
            "world_frame": "dexonomy_object_canonical_z_up",
            "source": "dexonomy_sharpa_wave",
        },
        "num_steps": T,
        "segment_names": ["retarget", "approach", "close", "squeeze", "carry"],
    }
    with open(os.path.join(out_dir, "trajectory.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--data", default="grasp_data")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--object-mesh", default=None,
                    help="override object mesh path (default: from scene cfg processed_data)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--approach-h", type=float, default=0.06)
    ap.add_argument("--lift-h", type=float, default=0.08)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.exp_dir}/{args.data}/**/*.npy", recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if args.limit:
        files = files[:args.limit]
    out_root = args.out_root or os.path.join(args.exp_dir, "isaac_traj")

    mesh_path = args.object_mesh
    if mesh_path is None:
        d0 = np.load(files[0], allow_pickle=True).item()
        scene = np.load(str(d0["scene_path"]), allow_pickle=True).item()
        for o in scene["scene"].values():
            if o.get("type") == "rigid_object":
                mesh_path = os.path.join(os.path.dirname(str(d0["scene_path"])), o["file_path"])
                mesh_path = os.path.abspath(mesh_path)
    assert mesh_path and os.path.exists(mesh_path), f"object mesh not found: {mesh_path}"

    for f in files:
        name = os.path.basename(f).replace(".npy", "")
        T = export_one(f, os.path.join(out_root, name), mesh_path,
                       args.approach_h, args.lift_h)
        print(f"exported {name}: {T} frames")
    print(f"\n{len(files)} trajectories -> {out_root}\nobject mesh: {mesh_path}")


if __name__ == "__main__":
    main()
