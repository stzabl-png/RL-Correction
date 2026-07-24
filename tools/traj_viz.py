#!/usr/bin/env python
"""Trajectory-vs-video alignment visualizer.

Layered overlay for checking whether reconstructed hand/object trajectories line up
with the source video:

  base layer        : the input MP4, played / written as-is
  + --mano          : project MANO hand joints (camera frame) -> 2D skeleton
  + --mano-traj     : draw each wrist's projected trajectory trail
  + --mesh PATH     : project a (scaled) object mesh -> 2D, posed per --traj obj_pose
  + --obj-traj      : draw the object-center projected trajectory trail

All overlays are projected with the camera intrinsics (--intrinsics, default = HOI4D
RealSense D435i). Trajectories are expected in the CAMERA frame, in meters (this is the
frame HOI4D / our hoi4d_to_replay.py store, before any sim world transform).

The trajectory .npz may contain (any subset):
    joints_left / joints_right : (N,21,3) camera-frame MANO joints (OpenPose order)
    valid_left  / valid_right  : (N,) 0/1   (gap-filled frames not drawn)
    obj_pose                   : (N,7) [x,y,z, qw,qx,qy,qz] camera frame
    frames                     : (N,) int    video-frame index for each row
                                 (if absent, row i -> video frame i)

Examples
--------
# HOI4D two-hand chair carry: MANO joints + wrist trails, NO mesh / NO object track
python tools/traj_viz.py \
    --video Data/HOI4D/HOI4D_release/ZY20210800001/H1/C20/N27/S295/s03/T4/align_rgb/image.mp4 \
    --traj third_party/MagicDexMate/outputs/hoi4d_C20N27/replay_hoi4d_2hand.npz \
    --mano --mano-traj --out outputs/traj_viz_chair.mp4

# later: add the scaled object mesh + its trajectory
python tools/traj_viz.py --video ... --traj ... --mano --mano-traj \
    --mesh chair_027.obj --mesh-scale 1.0 --obj-traj --out ...
"""
import argparse
import os
import sys

import cv2
import numpy as np

# HOI4D RealSense D435i intrinsics (1920x1080), shared across sessions
# (matches tools/prepare_gt_for_eval.py).
HOI4D_K = np.array([[1376.8, 0.0, 967.7],
                    [0.0, 1376.2, 526.8],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
HOI4D_K_RES = (1920, 1080)

# MediaPipe / OpenPose 21-joint hand skeleton (0 = wrist).
HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (0, 9), (9, 10), (10, 11), (11, 12),       # middle
    (0, 13), (13, 14), (14, 15), (15, 16),     # ring
    (0, 17), (17, 18), (18, 19), (19, 20),     # pinky
    (5, 9), (9, 13), (13, 17),                 # palm arches
]
# BGR colors (OpenCV). Source A (--traj) vs source B (--traj2) get distinct hues.
COL = {"right": (60, 220, 60),     # green
       "left": (255, 140, 50),     # blue
       "obj": (40, 200, 255),      # amber
       "mesh": (180, 180, 180)}    # gray
COL_B = {"right": (60, 60, 235),   # red   (source B right)
         "left": (200, 60, 235),   # magenta (source B left)
         "obj": (200, 200, 60)}    # teal


def parse_K(spec, frame_wh):
    if spec is None:
        K = HOI4D_K.copy()
        kw, kh = HOI4D_K_RES
    elif os.path.exists(spec):
        a = np.load(spec)
        K = a.reshape(3, 3) if a.size == 9 else np.array(
            [[a[0], 0, a[2]], [0, a[1], a[3]], [0, 0, 1]], float)
        kw, kh = frame_wh
    else:
        fx, fy, cx, cy = (float(x) for x in spec.split(","))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        kw, kh = frame_wh
    # rescale K if it was defined for a different resolution than the frames
    fw, fh = frame_wh
    if (kw, kh) != (fw, fh):
        K = K.copy()
        K[0] *= fw / kw
        K[1] *= fh / kh
    return K


def project(P, K):
    """(...,3) camera-frame points -> (...,2) pixels, plus depth Z."""
    P = np.asarray(P, float)
    Z = P[..., 2]
    Zc = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
    u = P[..., 0] / Zc * K[0, 0] + K[0, 2]
    v = P[..., 1] / Zc * K[1, 1] + K[1, 2]
    return np.stack([u, v], -1), Z


def quat_to_R(q):
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def draw_hand(img, j2d, Z, color, valid):
    if not valid:
        return
    h, w = img.shape[:2]
    ok = (Z > 0)
    for a, b in HAND_BONES:
        if ok[a] and ok[b]:
            pa, pb = j2d[a], j2d[b]
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, 2, cv2.LINE_AA)
    for k in range(len(j2d)):
        if ok[k] and -50 <= j2d[k, 0] <= w + 50 and -50 <= j2d[k, 1] <= h + 50:
            r = 5 if k == 0 else 3                      # wrist a bit bigger
            cv2.circle(img, (int(j2d[k, 0]), int(j2d[k, 1])), r, color, -1, cv2.LINE_AA)


def draw_trail(img, pts2d, Z, color, thick=2):
    pts = [p for p, z in zip(pts2d, Z) if z > 0]
    for i in range(1, len(pts)):
        cv2.line(img, (int(pts[i - 1][0]), int(pts[i - 1][1])),
                 (int(pts[i][0]), int(pts[i][1])), color, thick, cv2.LINE_AA)


def load_source(path, K_spec, frame_wh, colmap, label, flip_yz=False):
    """Load a trajectory npz and project its hands/object with intrinsics K_spec."""
    d = np.load(path, allow_pickle=True)
    N = next((len(d[k]) for k in ("joints_left", "joints_right", "obj_pose") if k in d.files), None)
    if N is None:
        sys.exit(f"{path}: no joints_left/right/obj_pose")
    frames = np.asarray(d["frames"]).astype(int) if "frames" in d.files else np.arange(N)
    K = parse_K(K_spec, frame_wh)
    proj = {}
    for s in ("left", "right"):
        if f"joints_{s}" in d.files:
            j = np.asarray(d[f"joints_{s}"], float).copy()
            if flip_yz:                                   # HaWoR internal -> OpenCV image frame
                j[..., 1] *= -1; j[..., 2] *= -1
            uv, Z = project(j, K)
            v = d[f"valid_{s}"] if f"valid_{s}" in d.files else np.ones(len(j))
            proj[s] = (uv, Z, np.asarray(v))
    obj = None
    if "obj_pose" in d.files:
        op = np.asarray(d["obj_pose"], float)
        ouv, oz = project(op[:, :3], K)
        obj = (op, ouv, oz)
    return dict(d=d, N=N, row_of={int(f): i for i, f in enumerate(frames)},
                frames=frames, K=K, proj=proj, obj=obj, col=colmap, label=label)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="base MP4")
    ap.add_argument("--traj", required=True, help="trajectory .npz (source A, camera frame)")
    ap.add_argument("--intrinsics", default=None,
                    help="'fx,fy,cx,cy', a .npy path, or omit for HOI4D default")
    ap.add_argument("--label", default="A", help="legend label for --traj (e.g. GT)")
    ap.add_argument("--traj2", default=None, help="second trajectory .npz (source B) to compare")
    ap.add_argument("--intrinsics2", default=None, help="intrinsics for --traj2")
    ap.add_argument("--label2", default="B", help="legend label for --traj2 (e.g. HaWoR)")
    ap.add_argument("--flip2", action="store_true",
                    help="negate Y,Z of --traj2 joints (HaWoR internal -> OpenCV image frame)")
    ap.add_argument("--mano", action="store_true", help="draw MANO hand skeletons")
    ap.add_argument("--mano-traj", action="store_true", help="draw wrist trajectory trails")
    ap.add_argument("--trail", type=int, default=-1, help="trail length in frames (-1 = full)")
    ap.add_argument("--mesh", default=None, help="object mesh (.obj/.ply/.glb), source A only")
    ap.add_argument("--mesh-scale", type=float, default=1.0)
    ap.add_argument("--obj-traj", action="store_true", help="draw object-center trajectory")
    ap.add_argument("--scale", type=float, default=1.0, help="output downscale factor")
    ap.add_argument("--out", default=None, help="output mp4 (default: outputs/<traj>_viz.mp4)")
    ap.add_argument("--show", action="store_true", help="also open a live window")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nfrm = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    srcs = [load_source(args.traj, args.intrinsics, (W, H), COL, args.label)]
    if args.traj2:
        srcs.append(load_source(args.traj2, args.intrinsics2, (W, H), COL_B, args.label2, args.flip2))
    fps = float(srcs[0]["d"]["fps"]) if "fps" in srcs[0]["d"].files else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
    for sc in srcs:
        print(f"[viz] source '{sc['label']}': N={sc['N']} frames {sc['frames'].min()}..{sc['frames'].max()} "
              f"hands={list(sc['proj'])} K fx={sc['K'][0,0]:.1f} cx={sc['K'][0,2]:.1f}")

    mesh_v = None
    if args.mesh:
        import trimesh
        m = trimesh.load(args.mesh, force="mesh")
        mesh_v = np.asarray(m.vertices, float) * args.mesh_scale
        if len(mesh_v) > 4000:
            mesh_v = mesh_v[np.random.default_rng(0).choice(len(mesh_v), 4000, replace=False)]

    out_path = args.out or os.path.join(
        "outputs", os.path.splitext(os.path.basename(args.traj))[0] + "_viz.mp4")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    ow, oh = int(W * args.scale), int(H * args.scale)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (ow, oh))

    drawn = 0
    for vf in range(nfrm):
        ok, img = cap.read()
        if not ok:
            break
        any_row = False
        for sc in srcs:
            row = sc["row_of"].get(vf)
            if row is None:
                continue
            any_row = True
            lo = 0 if args.trail < 0 else max(0, row - args.trail)
            for s, (uv, Z, val) in sc["proj"].items():
                if args.mano_traj:
                    draw_trail(img, uv[lo:row + 1, 0], Z[lo:row + 1, 0], sc["col"][s], 2)
                if args.mano:
                    draw_hand(img, uv[row], Z[row], sc["col"][s], bool(val[row]))
            if sc["obj"] is not None and args.obj_traj:
                op, ouv, oz = sc["obj"]
                draw_trail(img, ouv[lo:row + 1], oz[lo:row + 1], sc["col"]["obj"], 3)
            if mesh_v is not None and sc is srcs[0] and sc["obj"] is not None:
                op = sc["obj"][0]
                R = quat_to_R(op[row, 3:7])
                muv, mz = project(mesh_v @ R.T + op[row, :3], sc["K"])
                for p, z in zip(muv, mz):
                    if z > 0 and 0 <= p[0] < W and 0 <= p[1] < H:
                        cv2.circle(img, (int(p[0]), int(p[1])), 1, COL["mesh"], -1)
        drawn += any_row
        # HUD: frame tag + per-source legend (right/left swatch colors)
        cv2.putText(img, f"frame {vf}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, f"frame {vf}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        for i, sc in enumerate(srcs):
            y = 80 + i * 36
            cv2.putText(img, f"{sc['label']}  R", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        sc["col"]["right"], 2, cv2.LINE_AA)
            cv2.putText(img, "/L", (180, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, sc["col"]["left"], 2, cv2.LINE_AA)

        out = cv2.resize(img, (ow, oh)) if args.scale != 1.0 else img
        writer.write(out)
        if args.show:
            cv2.imshow("traj_viz", out)
            if (cv2.waitKey(int(1000 / max(fps, 1))) & 0xFF) in (ord("q"), 27):
                break
    cap.release(); writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"[viz] wrote {out_path}  ({drawn} frames with overlays)")


if __name__ == "__main__":
    main()
