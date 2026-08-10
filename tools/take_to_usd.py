#!/usr/bin/env python
"""Convert every object mesh of a reconstruction take to USD, in trajectory order.

A paired scene (bottle + cap) reconstructs several objects in one pass, so the take dir
holds objects/object_0/, objects/object_1/, ... The replay script matches USDs to
obj_pose_all by position, so the file names must sort in the same order -- hence
object_0.usd, object_1.usd rather than mesh-derived names.

Booting Kit costs ~8 s, and obj_to_usd.py boots it once per call, so this is just a loop
that reports what it produced; the cost is unavoidable without rewriting that script.

Usage:
  python tools/take_to_usd.py <ReconstructOutput take> [--out-dir <RetargetOutput take>]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ego_pipeline"))
from repo_paths import RR_ROOT, RECON_OUTPUT, RETARGET_OUTPUT, ISAAC_PYTHON  # noqa: E402

OBJ_TO_USD = Path(RR_ROOT) / "ego_pipeline/Retargeting/scripts/obj_to_usd.py"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", type=Path, help="ReconstructOutput take dir")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: the mirrored RetargetOutput take dir")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    take = a.take.resolve()
    out = a.out_dir or (Path(RETARGET_OUTPUT) / take.relative_to(Path(RECON_OUTPUT).resolve()))
    out.mkdir(parents=True, exist_ok=True)

    objs = sorted((take / "objects").glob("object_*/object_mesh_scaled_final.obj"))
    if not objs:                                        # single-object take
        single = take / "object_mesh_scaled_final.obj"
        if not single.exists():
            raise SystemExit(f"no object mesh under {take}")
        objs = [single]

    made = []
    for i, mesh in enumerate(objs):
        name = mesh.parent.name if mesh.parent.name.startswith("object_") else "object_0"
        dst = out / f"{name}.usd"
        if dst.exists() and not a.force:
            print(f"[usd]    {dst.name} exists (skip; --force to redo)")
            made.append(dst)
            continue
        print(f"[usd]    {mesh.parent.name if mesh.parent.name.startswith('object_') else take.name}"
              f" -> {dst.name} ...", flush=True)
        r = subprocess.run([str(ISAAC_PYTHON), str(OBJ_TO_USD), "--in", str(mesh), "--out", str(dst)],
                           env={"OMNI_KIT_ACCEPT_EULA": "YES", "PATH": "/usr/bin:/bin",
                                "HOME": str(Path.home())},
                           capture_output=True, text=True)
        if r.returncode != 0 or not dst.exists():
            print(r.stdout[-800:] + r.stderr[-800:])
            raise SystemExit(f"obj_to_usd failed for {mesh}")
        made.append(dst)

    print(f"\n{len(made)} USD(s) in {out}:")
    for p in made:
        print(f"  {p.name}  {p.stat().st_size/1e6:.1f} MB")
    if len(made) > 1:
        print("\nrun_retarget.sh picks up all *.usd in the dir, sorted, so object_0.usd "
              "lines up with obj_pose_all[0].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
