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

#: 22 维手指关节序 (SharpaWave). 左右手同序, 只差前缀 —— 由 --side 决定。
_JOINT_SUFFIXES = [
    "thumb_CMC_FE", "thumb_CMC_AA", "thumb_MCP_FE", "thumb_MCP_AA", "thumb_IP",
    "index_MCP_FE", "index_MCP_AA", "index_PIP", "index_DIP",
    "middle_MCP_FE", "middle_MCP_AA", "middle_PIP", "middle_DIP",
    "ring_MCP_FE", "ring_MCP_AA", "ring_PIP", "ring_DIP",
    "pinky_CMC", "pinky_MCP_FE", "pinky_MCP_AA", "pinky_PIP", "pinky_DIP",
]
JOINT_ORDER: list[str] = ["right_" + j for j in _JOINT_SUFFIXES]   # 默认右手


def set_side(side: str) -> None:
    """按手别填 JOINT_ORDER (写进 trajectory.json, Isaac 侧照它对关节)."""
    assert side in ("left", "right"), side
    JOINT_ORDER[:] = [f"{side}_{j}" for j in _JOINT_SUFFIXES]


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


def export_one(npy_path, out_dir, mesh_path, approach_h, lift_h,
               squeeze_frames=15, carry=True):
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
    s_pos, s_quat, s_j = interp_qpos29([grasp, squeeze], squeeze_frames)
    s_pos[:] = grasp[:3]
    s_quat[:] = grasp[3:7]
    parts = [(a_pos, a_quat, a_j, 1), (c_pos, c_quat, c_j, 2), (s_pos, s_quat, s_j, 3)]
    if carry:
        # carry: wrist straight up, fingers hold squeeze
        lift_end = squeeze.copy()
        lift_end[:3] = grasp[:3]
        lift_end[3:7] = grasp[3:7]
        lift_end[2] += lift_h
        carry_start = lift_end.copy()
        carry_start[2] -= lift_h
        y_pos, y_quat, y_j = interp_qpos29([carry_start, lift_end], 45)
        parts.append((y_pos, y_quat, y_j, 4))

    hand_pos = np.concatenate([p[0] for p in parts])
    hand_quat = np.concatenate([p[1] for p in parts])
    fingers = np.concatenate([p[2] for p in parts])
    segment = np.concatenate([
        np.full(len(p[0]), p[3]) for p in parts
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
    ap.add_argument("--side", default="right", choices=("left", "right"),
                    help="手别; 决定 trajectory.json 里的 22 维关节名前缀")
    ap.add_argument("--data", default="grasp_data")
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--object-mesh", default=None,
                    help="override object mesh path (default: from scene cfg processed_data)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--approach-h", type=float, default=0.06)
    ap.add_argument("--lift-h", type=float, default=0.08)
    ap.add_argument("--squeeze-frames", type=int, default=15,
                    help="squeeze 段帧数; contact 评估模式建议 60 (2s 缓慢收紧)")
    ap.add_argument("--no-carry", action="store_true",
                    help="不导出 carry 抬升段 (配 Isaac --eval-mode contact: 只评接触质量不评提起)")
    args = ap.parse_args()
    set_side(args.side)

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
                       args.approach_h, args.lift_h,
                       squeeze_frames=args.squeeze_frames, carry=not args.no_carry)
        print(f"exported {name}: {T} frames")
    print(f"\n{len(files)} trajectories -> {out_root}\nobject mesh: {mesh_path}")


if __name__ == "__main__":
    main()
