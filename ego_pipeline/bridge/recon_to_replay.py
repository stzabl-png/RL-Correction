#!/usr/bin/env python
"""Reconstruction output -> Retargeting input.

Convert one recon `world_fused.npz` into a `replay_world.npz` that MagicDexMate's
`sim/retarget_isaacsim.py` (and HaWoRSource) consume. Run in the **hawor** conda env
(needs MANO/smplx via hawor.utils.process).

  conda run -n hawor python recon_to_replay.py --in <dir-or-world_fused.npz> --out <dir-or-replay_world.npz>

world_fused.npz (gravity_z_up_world, [0]=left [1]=right):
  hand_trans (2,Th,3) hand_rot (2,Th,3) hand_pose (2,Th,45) hand_betas (2,Th,10) hand_valid (2,Th)
  object_ob_in_world (Tv,4,4) num_frames=Tv          (hands Th=2*Tv @30fps, video Tv @15fps)
replay_world.npz:
  joints_left/right (Tv,21,3) f32 world m, OpenPose/MediaPipe order
  valid_left/right (Tv,) f32 · obj_pose (Tv,7) [x,y,z,qw,qx,qy,qz] world · frames (Tv,) i32 · fps f32
  mano_verts_right (Tv,778,3) f32 (if available)
"""
import argparse
import os
import sys

import numpy as np
import torch
from scipy.spatial.transform import Rotation

# 让 `from phase import load_phase` 可用(ego_pipeline/phase),独立于 cwd。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DEFAULT_HAWOR = os.path.expanduser("~/Project/Reconstruct_and_Retarget/third_party/hawor")


def _resolve_io(inp: str, out: str) -> tuple[str, str]:
    if os.path.isdir(inp):
        inp = os.path.join(inp, "world_fused.npz")
    if out is None:
        out = os.path.join(os.path.dirname(inp), "replay_world.npz")
    elif os.path.isdir(out) or out.endswith(os.sep):
        out = os.path.join(out, "replay_world.npz")
    return inp, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, help="world_fused.npz or its directory")
    ap.add_argument("--out", default=None, help="replay_world.npz or directory (default: alongside input)")
    ap.add_argument("--fps", type=float, default=15.0, help="video/object fps recorded in output")
    ap.add_argument("--hawor-dir", default=_DEFAULT_HAWOR)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    inp, out = _resolve_io(args.inp, args.out)
    inp, out = os.path.abspath(inp), os.path.abspath(out)
    d = np.load(inp, allow_pickle=True)

    # HaWoR's MANO loads from a cwd-relative `_DATA/` path -> run from the hawor repo root.
    os.chdir(args.hawor_dir)

    use_cuda = not args.cpu and torch.cuda.is_available()
    ht = torch.from_numpy(np.asarray(d["hand_trans"])).float()
    hr = torch.from_numpy(np.asarray(d["hand_rot"])).float()
    hp = torch.from_numpy(np.asarray(d["hand_pose"])).float()
    hb = torch.from_numpy(np.asarray(d["hand_betas"])).float()
    hv = np.asarray(d["hand_valid"]).astype(np.float32)  # (2,Th)
    Th = ht.shape[1]

    sys.path.insert(0, args.hawor_dir)
    from hawor.utils.process import run_mano, run_mano_left  # noqa: E402

    out_l = run_mano_left(ht[0:1], hr[0:1], hp[0:1], betas=hb[0:1], use_cuda=use_cuda)
    out_r = run_mano(ht[1:2], hr[1:2], hp[1:2], betas=hb[1:2], use_cuda=use_cuda)
    jl_full = out_l["joints"][0].detach().cpu().numpy().astype(np.float32)  # (Th,21,3)
    jr_full = out_r["joints"][0].detach().cpu().numpy().astype(np.float32)

    # object / camera timeline = the canonical replay timeline
    obj_w = np.asarray(d["object_ob_in_world"], dtype=np.float64)  # (Tv,4,4)
    Tv = int(d["num_frames"]) if "num_frames" in d.files else obj_w.shape[0]

    # hands @Th(30fps) -> video @Tv(15fps): nearest source frame (Th==2*Tv -> 2i)
    src = np.clip(np.floor(np.arange(Tv) * (Th / Tv)).astype(int), 0, Th - 1)
    jl = jl_full[src].copy()
    jr = jr_full[src].copy()
    valid_l = hv[0, src]
    valid_r = hv[1, src]
    # Invalid frames carry zero-pose MANO junk (a hand at the origin). Mark them
    # NaN so downstream retarget/vis treat them as "not tracked" (existence comes
    # from the valid mask), instead of rendering a degenerate hand at the origin.
    jl[valid_l < 0.5] = np.nan
    jr[valid_r < 0.5] = np.nan

    # object pose -> (Tv,7) [x,y,z, qw,qx,qy,qz] (already gravity_z_up_world)
    obj_pose = np.zeros((Tv, 7), dtype=np.float32)
    obj_pose[:, :3] = obj_w[:, :3, 3]
    q_xyzw = Rotation.from_matrix(obj_w[:, :3, :3]).as_quat()  # scalar-last
    obj_pose[:, 3] = q_xyzw[:, 3]
    obj_pose[:, 4:7] = q_xyzw[:, :3]

    payload = dict(
        joints_left=jl, joints_right=jr,
        valid_left=valid_l.astype(np.float32), valid_right=valid_r.astype(np.float32),
        obj_pose=obj_pose,
        frames=np.arange(Tv, dtype=np.int32),
        fps=np.float32(args.fps),
    )
    # Per-frame reconstruction confidence (1 real, 0.5 interpolated, 0.3 held, 0 ignored)
    # from fuse's trajectory cleaning, so downstream RL correction knows which frames
    # to trust. Hand confidence is @Th -> resampled to the video timeline like valid_*.
    if "hand_confidence" in d.files:
        hc = np.asarray(d["hand_confidence"], dtype=np.float32)  # (2,Th)
        payload["confidence_left"] = hc[0, src]
        payload["confidence_right"] = hc[1, src]
    if "object_confidence" in d.files:
        payload["obj_confidence"] = np.asarray(d["object_confidence"], dtype=np.float32)

    # 抓取/接触阶段(人手轨迹的逐帧语义标签)。可用则写入,无标注则静默跳过(向后兼容)。
    # 现在=人工标注(grasp_annotation.json),同学的自动接触检测就绪后自动切换。
    try:
        from phase import load_phase
        _tracks = load_phase(os.path.dirname(inp), num_frames=Tv)
        if _tracks is not None:
            payload["phase_left"] = _tracks.left       # (Tv,) int8  FREE=0/CONTACT=1
            payload["phase_right"] = _tracks.right
            payload["phase_obj"] = _tracks.obj
            payload["phase_source"] = np.array(_tracks.source)
            print(f"[recon_to_replay] grasp phase: {_tracks.summary()}")
    except Exception as _e:
        print(f"[recon_to_replay] grasp phase skipped: {_e}")
    if "vertices" in out_r:
        payload["mano_verts_right"] = out_r["vertices"][0].detach().cpu().numpy().astype(np.float32)[src]
    if "vertices" in out_l:
        payload["mano_verts_left"] = out_l["vertices"][0].detach().cpu().numpy().astype(np.float32)[src]
    try:  # MANO face topology (constant) — needed by sim/vis_mano_sim.py
        from hawor.utils.process import get_mano_faces
        payload["mano_faces"] = np.asarray(get_mano_faces(), dtype=np.int32)
    except Exception as exc:
        print(f"[recon_to_replay] mano_faces unavailable: {exc}")

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    np.savez_compressed(out, **payload)
    print(f"[recon_to_replay] {inp}\n  -> {out}\n  joints L/R {jl.shape}/{jr.shape} | "
          f"Tv={Tv} Th={Th} | valid L/R={int(valid_l.sum())}/{int(valid_r.sum())} | fps={args.fps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
