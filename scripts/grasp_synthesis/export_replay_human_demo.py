#!/usr/bin/env python3
"""Convert a Reconstruct_and_Retarget ``replay_world.npz`` (reconstructed MANO hand + object
trajectory, world frame) into anchored-BODex ``human_demo.npz`` (MANO in the
object-canonical frame + object pose), so ``anchored_bodex`` can anchor grasp
synthesis to the reconstructed human hand -- fixing thumb-down grasps that the
region-only affordance seeding produces.

replay_world.npz keys used: joints_right (T,21,3) world, mano_verts_right
(T,778,3) world, obj_pose (T,7) [pos, quat wxyz] world, valid_right (T,), frames.
The MANO 21-joint order matches the human_demo convention (wrist, thumb*4,
index*4, middle*4, ring*4, pinky*4). vertex_part_ids is a fixed MANO property,
cached at assets/mano/mano_right_vertex_part_ids.npy.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
PART_IDS_NPY = REPO / "assets" / "mano" / "mano_right_vertex_part_ids.npy"


def quat_wxyz_to_rotmat(q):
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def build_human_demo(replay_npz: Path, out_npz: Path, side: str = "right") -> Path:
    replay = np.load(str(replay_npz), allow_pickle=True)
    joints_w = np.asarray(replay[f"joints_{side}"], dtype=np.float64)          # (T,21,3)
    verts_w = np.asarray(replay[f"mano_verts_{side}"], dtype=np.float64)       # (T,778,3)
    obj = np.asarray(replay["obj_pose"], dtype=np.float64)                     # (T,7) [pos, quat wxyz]
    valid = np.asarray(replay[f"valid_{side}"]).astype(bool)                   # (T,)
    frames = np.asarray(replay["frames"], dtype=np.int32) if "frames" in replay.files \
        else np.arange(len(joints_w), dtype=np.int32)
    T = joints_w.shape[0]

    part_ids = np.load(str(PART_IDS_NPY)).astype(np.int8)                      # (778,)

    hand_joints_object = np.full((T, 21, 3), np.nan, dtype=np.float64)
    hand_vertices_object = np.full((T, 778, 3), np.nan, dtype=np.float64)
    object_pose_camera = np.tile(np.eye(4), (T, 1, 1)).astype(np.float64)

    for t in range(T):
        p = obj[t, :3]
        R = quat_wxyz_to_rotmat(obj[t, 3:7])
        object_pose_camera[t, :3, :3] = R
        object_pose_camera[t, :3, 3] = p
        if valid[t]:
            # object-canonical = R^T (world - p)  == (world - p) @ R  for row vectors
            hand_joints_object[t] = (joints_w[t] - p) @ R
            hand_vertices_object[t] = (verts_w[t] - p) @ R

    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_npz),
        frame_ids=frames.astype(np.int32),
        mano_side=str(side),
        betas=np.zeros(10, dtype=np.float64),
        object_pose_camera=object_pose_camera.astype(np.float64),
        hand_joints_object=hand_joints_object.astype(np.float64),
        hand_vertices_object=hand_vertices_object.astype(np.float64),
        vertex_part_ids=part_ids,
        valid_mask=valid,
    )
    print(f"[human_demo] wrote {out_npz} (T={T}, valid={int(valid.sum())}, side={side})", flush=True)
    return out_npz


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--replay-npz", required=True)
    ap.add_argument("--out", required=True, help="output human_demo.npz path (usually <sequence-dir>/human_demo.npz)")
    ap.add_argument("--side", default="right", choices=["right", "left"])
    args = ap.parse_args(argv)
    if not PART_IDS_NPY.exists():
        print(f"error: missing {PART_IDS_NPY} (MANO vertex_part_ids cache)", file=sys.stderr)
        return 2
    build_human_demo(Path(args.replay_npz), Path(args.out), side=args.side)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
