#!/usr/bin/env python3
"""Render fuse visualization (requires hawor env for MANO hand geometry)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RECON_ROOT = Path(__file__).resolve().parents[1]
if str(RECON_ROOT) not in sys.path:
    sys.path.insert(0, str(RECON_ROOT))

from _common.hawor_mano import compute_hand_sequence_geometry  # noqa: E402
from _common.io import apply_vipe_focal_flip_c2w, count_video_frames, interpolate_c2w_poses, load_vipe_intrinsics, load_vipe_poses  # noqa: E402
from _common.paths import interim_step_dir  # noqa: E402
from _common.viz import encode_fuse_vis_mp4, vis_path  # noqa: E402
from sam2_object.sam2_object_common import OBJECT_MASK_ID  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hawor-world", type=Path, required=True)
    args = parser.parse_args(argv)

    video_path = args.video.resolve()
    vipe_dir = interim_step_dir(args.dataset, args.video_id, "vipe")
    fp_dir = interim_step_dir(args.dataset, args.video_id, "fp_pose")
    sam3d_scale_dir = interim_step_dir(args.dataset, args.video_id, "sam3d_scale")
    object_dir = interim_step_dir(args.dataset, args.video_id, "sam2_object")
    fuse_dir = interim_step_dir(args.dataset, args.video_id, "fuse")

    inds, c2w_all = load_vipe_poses(vipe_dir, args.video_id)
    k_mat, flip = load_vipe_intrinsics(vipe_dir, args.video_id)
    num_frames = count_video_frames(video_path)
    c2w_all = interpolate_c2w_poses(inds, c2w_all, num_frames)
    if flip:
        c2w_all = np.stack([apply_vipe_focal_flip_c2w(p) for p in c2w_all])

    ob_dir = fp_dir / "ob_in_cam"
    frame_indices = sorted(int(p.stem) for p in ob_dir.glob("*.txt"))
    mesh_path = sam3d_scale_dir / "object_mesh_scaled_final.obj"
    hand_geom = compute_hand_sequence_geometry(
        world_result_path=args.hawor_world.resolve(),
        c2w_all=c2w_all,
        num_video_frames=num_frames,
        video_path=video_path,
    )
    out = encode_fuse_vis_mp4(
        video_path=video_path,
        out_path=vis_path(fuse_dir, args.video_id),
        ob_in_cam_dir=ob_dir,
        frame_indices=frame_indices,
        k_mat=k_mat,
        mesh_path=mesh_path,
        object_masks_dir=object_dir / "video_segmentation" / "masks",
        object_mask_filename=f"{OBJECT_MASK_ID}.png",
        hand_geometry=hand_geom,
    )
    print(json.dumps({"vis_video": str(out), "hands_rendered": hand_geom is not None}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
