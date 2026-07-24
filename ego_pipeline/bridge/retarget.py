#!/usr/bin/env python
"""Batch: Reconstruction outputs -> Retargeting / sim inputs.

Scans the recon tree (ReconstructOutput) for `<...>/world_fused.npz` and, per take,
writes the sim-ready data into a SEPARATE retarget tree (RetargetOutput) at the same
relative path, so both trees mirror the source dataset's nesting:
    <recon-root>/<dataset>/<.../take>/world_fused.npz  (+ object_mesh_scaled_final.obj)
        -> <retarget-root>/<dataset>/<.../take>/replay_world.npz   (via recon_to_replay.py, hawor env)
                                              /object.usd           (via obj_to_usd.py, .venv-isaac)

  python retarget.py [--recon-root DIR] [--retarget-root DIR] [--limit N] [--skip-usd] [--force]

Roots default to $RECON_FINAL_ROOT / $RETARGET_FINAL_ROOT (else Reconstruct_and_Retarget/Output/{Reconstruct,Retarget}Output).
Cross-env steps run as subprocesses (hawor conda env / MagicDexMate .venv-isaac), so this
driver itself runs under any python.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIV2AP = Path("/home/lyh/Project/Reconstruct_and_Retarget")
DEFAULT_RECON_ROOT = Path(os.environ.get("RECON_FINAL_ROOT", BIV2AP / "Output" / "ReconstructOutput"))
DEFAULT_RETARGET_ROOT = Path(os.environ.get("RETARGET_FINAL_ROOT", BIV2AP / "Output" / "RetargetOutput"))
HAWOR_PY = "/home/lyh/anaconda3/envs/hawor/bin/python"
ISAAC_PY = BIV2AP / "third_party" / "MagicDexMate" / ".venv-isaac" / "bin" / "python"
OBJ_TO_USD = BIV2AP / "ego_pipeline" / "Retargeting" / "scripts" / "obj_to_usd.py"


def find_takes(recon_root: Path) -> list[Path]:
    return sorted(p.parent for p in recon_root.rglob("world_fused.npz") if "interim" not in p.parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="*", type=Path,
                    help="specific take dir(s) under --recon-root (or a parent dir to expand); "
                         "default: all takes under --recon-root")
    ap.add_argument("--recon-root", type=Path, default=DEFAULT_RECON_ROOT, help="recon output tree to scan")
    ap.add_argument("--retarget-root", type=Path, default=DEFAULT_RETARGET_ROOT, help="where to write sim data (mirrors recon layout)")
    ap.add_argument("--limit", type=int, default=None, help="process only the first N takes")
    ap.add_argument("--skip-usd", action="store_true", help="skip obj.obj -> object.usd (Isaac) step")
    ap.add_argument("--force", action="store_true", help="rebuild even if replay_world.npz/object.usd exist")
    args = ap.parse_args()

    if args.takes:                                    # explicit selection: each take dir or a parent to expand
        takes = []
        for t in args.takes:
            t = t.resolve()
            takes.extend([t] if (t / "world_fused.npz").exists() else find_takes(t))
    else:
        takes = find_takes(args.recon_root)
    if args.limit:
        takes = takes[: args.limit]
    if not takes:
        print(f"[retarget] no world_fused.npz under {args.recon_root}", file=sys.stderr)
        return 1
    print(f"[retarget] {len(takes)} take(s): {args.recon_root} -> {args.retarget_root}")

    ok = 0
    for i, take in enumerate(takes, 1):
        rel = take.relative_to(args.recon_root)            # <dataset>/<.../take>
        dest = args.retarget_root / rel
        tag = f"[{i}/{len(takes)}] {rel}"
        replay = dest / "replay_world.npz"
        usd = dest / "object.usd"
        mesh = take / "object_mesh_scaled_final.obj"
        try:
            dest.mkdir(parents=True, exist_ok=True)
            if args.force or not replay.exists():
                print(f"{tag}: recon_to_replay …")
                subprocess.run([HAWOR_PY, str(HERE / "recon_to_replay.py"),
                                "--in", str(take), "--out", str(replay)], check=True)
            else:
                print(f"{tag}: replay_world.npz exists (skip; --force to redo)")

            if not args.skip_usd and mesh.exists() and (args.force or not usd.exists()):
                print(f"{tag}: obj_to_usd …")
                env = {**os.environ, "OMNI_KIT_ACCEPT_EULA": "YES"}
                subprocess.run([str(ISAAC_PY), str(OBJ_TO_USD),
                                "--in", str(mesh), "--out", str(usd)], check=True, env=env)
            ok += 1
        except subprocess.CalledProcessError as e:
            print(f"{tag}: FAILED ({e})", file=sys.stderr)

    print(f"[retarget] done: {ok}/{len(takes)} takes -> replay_world.npz"
          + ("" if args.skip_usd else " + object.usd"))
    return 0 if ok == len(takes) else 2


if __name__ == "__main__":
    raise SystemExit(main())
