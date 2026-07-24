#!/usr/bin/env python
"""Mesh (.obj) -> USD for the replay script's --object-usd.

pxr is only importable once Kit is bootstrapped in this install, so this boots a
headless SimulationApp, then writes a single UsdGeom.Mesh under /Object.

  OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python scripts/obj_to_usd.py \
      --in chair/027.obj --out chair_027.usd [--scale 1.0]

Physics (collision + rigid body) is NOT baked here - sim/retarget_isaacsim.py adds
it at runtime for --mode physics, so one USD serves both render and physics modes.
"""
import argparse

import numpy as np


def load_obj(path):
    verts, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                for k in range(1, len(idx) - 1):  # fan-triangulate
                    faces.append([idx[0], idx[k], idx[k + 1]])
    return np.asarray(verts, np.float64), np.asarray(faces, np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    v, f = load_obj(args.inp)
    v *= args.scale
    print(f"loaded {args.inp}: {len(v)} verts, {len(f)} tris  bbox(m)="
          f"{(v.max(0) - v.min(0)).round(3).tolist()}")

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    from pxr import Usd, UsdGeom, Vt  # noqa: E402

    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    xform = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(xform.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/Object/mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.reshape(-1).astype(np.int32)))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    stage.GetRootLayer().Save()
    print(f"saved {args.out}  (defaultPrim=/Object)")
    app.close()


if __name__ == "__main__":
    main()
