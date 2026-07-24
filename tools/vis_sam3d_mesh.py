#!/usr/bin/env python3
"""
Visualize a SAM3D output mesh as a 4-angle turntable, stitched into one PNG.

Renders the mesh from 4 azimuths (0/90/180/270 deg, fixed elevation) and tiles
them 2x2 into a single image. Uses pyrender (EGL offscreen, normal-shaded) when
available, else falls back to a matplotlib Poly3DCollection render (no GPU/display
needed).

Run:
  ~/anaconda3/envs/biv2ap/bin/python tools/vis_sam3d_mesh.py <mesh.obj|.ply> [--out out.png]
  # default mesh = the C3 full-chain output; default out = <mesh_dir>/mesh_views.png
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import trimesh

BIV2AP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MESH = os.path.join(
    BIV2AP, "Output/obj_branch_full/ZY20210800001_H1_C3_N23_S19_s01_T1",
    "objects/sam3d_input/masks/object/object.obj")

AZIMUTHS = [0, 90, 180, 270]
ELEV = 20.0
RES = 640


def load_normalized(path):
    """Load mesh, recenter to origin, scale to unit bounding-sphere radius."""
    m = trimesh.load(path, force="mesh")
    m.apply_translation(-m.bounding_sphere.primitive.center)
    r = float(m.bounding_sphere.primitive.radius) or 1.0
    m.apply_scale(1.0 / r)
    return m


def cam_pose(azim_deg, elev_deg, dist):
    """OpenGL-convention camera pose looking at origin (camera +z points to eye)."""
    a, e = math.radians(azim_deg), math.radians(elev_deg)
    eye = np.array([dist * math.cos(e) * math.sin(a),
                    dist * math.sin(e),
                    dist * math.cos(e) * math.cos(a)])
    fwd = eye / (np.linalg.norm(eye) + 1e-9)          # target is origin
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, fwd); right /= np.linalg.norm(right) + 1e-9
    newup = np.cross(fwd, right)
    M = np.eye(4)
    M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, newup, fwd, eye
    return M


def render_pyrender(mesh):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyrender
    rmesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    renderer = pyrender.OffscreenRenderer(RES, RES)
    imgs = []
    for az in AZIMUTHS:
        scene = pyrender.Scene(bg_color=[245, 245, 245, 255],
                               ambient_light=[0.35, 0.35, 0.35])
        scene.add(rmesh)
        pose = cam_pose(az, ELEV, dist=2.6)
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.5), pose=pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=pose)
        color, _ = renderer.render(scene)
        imgs.append(np.asarray(color)[..., :3])
    renderer.delete()
    return imgs


def render_matplotlib(mesh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    v, f = mesh.vertices, mesh.faces
    imgs = []
    for az in AZIMUTHS:
        fig = plt.figure(figsize=(RES / 100, RES / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        tris = v[f]
        pc = Poly3DCollection(tris, alpha=1.0, facecolor=(0.7, 0.72, 0.78),
                              edgecolor=(0.3, 0.3, 0.3), linewidths=0.05)
        ax.add_collection3d(pc)
        lim = 0.9
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1]); ax.view_init(elev=ELEV, azim=az)
        ax.set_axis_off()
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        imgs.append(buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3].copy())
        plt.close(fig)
    return imgs


def tile_2x2(imgs, labels, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, img, lab in zip(axes.ravel(), imgs, labels):
        ax.imshow(img); ax.set_title(lab, fontsize=12); ax.axis("off")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", nargs="?", default=DEFAULT_MESH)
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", choices=["auto", "pyrender", "matplotlib"], default="auto")
    args = ap.parse_args()

    mesh = load_normalized(args.mesh)
    print(f"[mesh] {args.mesh}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces")

    imgs, used = None, None
    if args.backend in ("auto", "pyrender"):
        try:
            imgs = render_pyrender(mesh); used = "pyrender"
        except Exception as e:
            print(f"[warn] pyrender failed ({type(e).__name__}: {e}); falling back")
            if args.backend == "pyrender":
                raise
    if imgs is None:
        imgs = render_matplotlib(mesh); used = "matplotlib"

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.mesh)), "mesh_views.png")
    labels = [f"azim {a}°" for a in AZIMUTHS]
    fig = tile_2x2(imgs, labels,
                   f"SAM3D mesh: {os.path.basename(args.mesh)}  "
                   f"({len(mesh.faces)} faces, {used})")
    fig.savefig(out, dpi=100)
    print(f"[out] {out}  (backend={used})")


if __name__ == "__main__":
    main()
