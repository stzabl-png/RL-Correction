"""Self-contained affordance prediction on an object mesh (vendored into OCIR).

Samples a point cloud from the mesh surface, unit-sphere-normalizes it (matching
training), runs the vendored checkpoint, and writes:
  <out>/affordance.npz       points (N,3) + points_raw (N,3) + heatmap (N,) + center + scale
  <out>/affordance_pred.png  (if matplotlib available)  headless 3D scatter
  <out>/affordance_pred.ply  (if open3d available)      colored point cloud

Run in the affordance conda env (Sonata / spconv-cu128 / torch_scatter), e.g.:
  conda run -n <affordance-env> \
    python -m ocir.affordance.predict --mesh obj.obj --ckpt assets/affordance/model.pt --out <dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh

from ocir.affordance.affordance_model import build_mvp_model
from ocir.affordance.normalize import normalize_unit_sphere

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT = REPO_ROOT / "assets" / "affordance" / "model.pt"


def load_model(ckpt_path, device="cuda"):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    mcfg = dict(cfg["model"])
    mcfg["use_normals"] = cfg["data"]["use_normals"]
    model = build_mvp_model(mcfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, cfg


def sample_mesh(mesh_path, n_points, seed=0):
    m = trimesh.load(mesh_path, process=False, force="mesh")
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(tuple(m.geometry.values()))
    pts, fidx = trimesh.sample.sample_surface(m, n_points, seed=seed)
    normals = m.face_normals[fidx]
    return pts.astype(np.float32), normals.astype(np.float32), m


def predict_to_dir(mesh_path, out_dir, ckpt=DEFAULT_CKPT, n_points=2048, seed=0, viz=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(ckpt, device)
    use_normals = cfg["data"]["use_normals"]
    n = n_points or cfg["data"]["n_points"]

    pts_raw, normals_raw, mesh = sample_mesh(mesh_path, n, seed)
    print(f"mesh: {mesh.vertices.shape[0]} verts / {mesh.faces.shape[0]} faces -> sampled {len(pts_raw)} pts", flush=True)

    pts, center, scale = normalize_unit_sphere(pts_raw)
    coords = torch.from_numpy(pts)[None].to(device)
    normals = torch.from_numpy(normals_raw)[None].to(device) if use_normals else None
    with torch.no_grad():
        prob = torch.sigmoid(model(coords, normals))[0].cpu().numpy()

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "affordance.npz",
             points=pts, points_raw=pts_raw, heatmap=prob.astype(np.float32),
             center=center.astype(np.float32), scale=np.float32(scale))
    print(f"affordance stats: min={prob.min():.3f} max={prob.max():.3f} "
          f"mean={prob.mean():.3f} frac>0.5={(prob > 0.5).mean():.3f}", flush=True)

    if viz:
        _write_viz(out, pts, pts_raw, prob)
    print(f"saved -> {out}/  (affordance.npz)", flush=True)
    return out / "affordance.npz"


def _write_viz(out, pts, pts_raw, prob):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=prob, cmap="jet", vmin=0, vmax=1, s=6, depthshade=False)
        ax.set_title(f"Affordance max={prob.max():.2f} mean={prob.mean():.2f} >0.5:{(prob > 0.5).mean() * 100:.1f}%", fontsize=10)
        ax.set_axis_off()
        lim = float(np.abs(pts).max())
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        fig.tight_layout(); fig.savefig(out / "affordance_pred.png", dpi=120); plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] png skipped ({type(exc).__name__}: {exc})", flush=True)
    try:
        import open3d as o3d
        import matplotlib.pyplot as plt
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_raw)
        pcd.colors = o3d.utility.Vector3dVector(plt.cm.jet(np.clip(prob, 0, 1))[:, :3])
        o3d.io.write_point_cloud(str(out / "affordance_pred.ply"), pcd)
    except Exception as exc:  # noqa: BLE001
        print(f"[viz] ply skipped ({type(exc).__name__}: {exc})", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-viz", action="store_true")
    args = ap.parse_args(argv)
    predict_to_dir(args.mesh, args.out, ckpt=args.ckpt, n_points=args.n, seed=args.seed, viz=not args.no_viz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
