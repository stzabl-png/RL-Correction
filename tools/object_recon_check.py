#!/usr/bin/env python
"""Is a take's OBJECT reconstruction usable, and if not, what is wrong with it?

The contact heatmap is painted on the object mesh and posed by the object track, so both
have to be right or the heat is meaningless. This separates the two failure modes:

  wrong MESH (scale/shape)   the projected silhouette is consistently the wrong SIZE at
                             every frame -> re-label SAM3D on a better frame, or rescale
  wrong TRACK (pose/depth)   the size ratio DRIFTS over time -> the pose estimate is
                             sliding in depth; re-labelling will not fix it

Reported per frame: the area ratio projected/observed, and the linear scale that would
reconcile them (sqrt of the area ratio). A constant ratio means scale; a drifting ratio
means the track.

Frames inside the grasp are marked, because the hand eats into the observed mask there
and inflates the ratio for reasons that have nothing to do with the reconstruction.

Usage:
  python tools/object_recon_check.py <ReconstructOutput/take-dir> [--side right] [--out PNG]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ego_pipeline"))
from contact import frames as F                                        # noqa: E402
from contact import observe2d as O                                     # noqa: E402
from contact import viz                                                # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", type=Path)
    ap.add_argument("--side", default="right", choices=["right", "left"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-frames", type=int, default=6, help="frames to render")
    a = ap.parse_args()

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh

    take = F.load_take(a.take, side=a.side, require_qpos=False)
    if take.video_path is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from contact_align_heatmap import guess_video
        take.video_path = guess_video(take.recon_dir)
    mesh = trimesh.load(take.obj_mesh_path, force="mesh", process=False)
    V = np.asarray(mesh.vertices)

    grasp = set(take.contact_frames().tolist()) if take.intervals else set()
    rows = []
    for f in range(take.Tv):
        om = cv2.imread(str(take.mask_path("object", f)), cv2.IMREAD_UNCHANGED)
        hm = cv2.imread(str(take.mask_path("hand", f)), cv2.IMREAD_UNCHANGED)
        if om is None:
            continue
        hw = om.shape[:2]
        T = take.obj_T_world[f]
        uv, z, inb = O.project_points(V @ T[:3, :3].T + T[:3, 3], take.c2w[f], take.K, hw)
        proj = O.splat_mask(uv, inb & (z > 0), hw, radius=2, close=15)
        seen = np.ones(hw, bool) if hm is None else ~(hm > 0)
        pm, mm = (om > 0) & seen, proj & seen
        ratio = mm.sum() / max(pm.sum(), 1)
        rows.append((f, int(pm.sum()), int(mm.sum()), ratio, np.sqrt(ratio), f in grasp))

    arr = np.array([(r[0], r[3], r[4]) for r in rows if not r[5]])   # grasp-free frames
    print(f"[obj]    {take.recon_dir.name}: mesh {len(V)} verts, extents "
          f"{np.round(mesh.extents, 3).tolist()} m, watertight={mesh.is_watertight}")
    if len(arr):
        print(f"[obj]    outside the grasp: area ratio proj/observed median {np.median(arr[:,1]):.2f} "
              f"(range {arr[:,1].min():.2f}-{arr[:,1].max():.2f})")
        print(f"[obj]    implied linear scale correction: median {1/np.median(arr[:,2]):.2f}x "
              f"(range {1/arr[:,2].max():.2f}-{1/arr[:,2].min():.2f}x)")
        drift = arr[:, 1].max() / max(arr[:, 1].min(), 1e-6)
        # A drifting ratio does NOT by itself implicate the pose track: an anisotropic
        # mesh projects to wildly different areas as the object turns, so rotation has
        # to be ruled out before blaming the track.
        aniso = float(max(mesh.extents) / max(min(mesh.extents), 1e-9))
        R0 = take.obj_T_world[0][:3, :3]
        turn = max(np.degrees(np.arccos(np.clip(
            (np.trace(R0.T @ take.obj_T_world[f][:3, :3]) - 1) / 2, -1, 1)))
            for f in range(take.Tv))
        print(f"[obj]    mesh anisotropy {aniso:.2f}:1, object turns up to {turn:.0f}deg, "
              f"area-ratio drift {drift:.1f}x")
        if drift < 1.6:
            print("[obj]    -> steady ratio: MESH SCALE is the whole story")
        elif aniso > 1.5 and turn > 60:
            print("[obj]    -> the drift is explained by turning an anisotropic mesh; "
                  "MESH SHAPE is the prime suspect, the pose track is not implicated")
        else:
            print("[obj]    -> drift is not explained by rotation: the POSE TRACK is suspect")

        lbl = take.recon_dir / "masks/objects/label_prompt.json"
        if lbl.exists():
            import json
            pr = json.loads(lbl.read_text())["objects"][0]
            inside = any(a0 <= pr["frame_idx"] <= b0 for a0, b0 in take.intervals)
            print(f"[obj]    SAM3D was labelled on frame {pr['frame_idx']}"
                  + ("  <-- INSIDE the grasp: the hand covers part of the object there, so "
                     "the single-view mesh is built from a partial silhouette" if inside else ""))

    show = np.linspace(0, len(rows) - 1, a.n_frames).astype(int)
    fig = plt.figure(figsize=(4.6 * a.n_frames / 2, 9.6), constrained_layout=True)
    gs = fig.add_gridspec(3, max(1, a.n_frames // 2))
    for i, ri in enumerate(show):
        f, mpx, ppx, ratio, lin, ing = rows[ri]
        ax = fig.add_subplot(gs[i // (a.n_frames // 2), i % (a.n_frames // 2)])
        img = viz.load_video_frame(take.video_path, f)
        om = cv2.imread(str(take.mask_path("object", f)), cv2.IMREAD_UNCHANGED) > 0
        hw = om.shape
        T = take.obj_T_world[f]
        uv, z, inb = O.project_points(V @ T[:3, :3].T + T[:3, 3], take.c2w[f], take.K, hw)
        proj = O.splat_mask(uv, inb & (z > 0), hw, radius=2, close=15)
        viz._bg(ax, img, hw, f"f={f}{' (grasping)' if ing else ''}  observed {mpx}px  "
                             f"projected {ppx}px  ratio {ratio:.1f}x")
        viz.overlay_mask(ax, om, (0.2, 1.0, 0.3), 0.55)
        viz.overlay_mask(ax, proj, (1.0, 0.3, 0.1), 0.40)

    ax = fig.add_subplot(gs[2, :])
    fs = [r[0] for r in rows]
    ax.plot(fs, [r[3] for r in rows], lw=1.4, color="#d1495b", label="area ratio projected/observed")
    ax.axhline(1.0, color="#2a9d8f", ls="--", lw=1, label="1.0 = perfect")
    for a0, b0 in take.intervals:
        ax.axvspan(a0, b0, color="#888", alpha=0.18)
    ax.set_yscale("log")
    ax.set_xlabel("frame")
    ax.set_ylabel("area ratio (log)")
    ax.legend(fontsize=8)
    ax.set_title("grey band = annotated grasp (hand eats the observed mask there, "
                 "so the ratio is inflated for reasons unrelated to the reconstruction)",
                 fontsize=9)

    fig.suptitle(f"{take.recon_dir.name}: object reconstruction check  |  green = SAM2 "
                 f"observed mask, red = projected mesh", fontsize=11)
    out = a.out or (take.retarget_dir / "object_recon_check.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[out]    {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
