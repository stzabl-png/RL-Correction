#!/usr/bin/env python3
"""Build the tuned Sharpa hand USD: OCIR structure + BODex collision model.

Derives ``right_sharpa_wave_tuned.usd`` from two sources:

* **Structure source** -- OCIR's ``right_sharpa_wave.usd`` (Isaac URDF-importer
  layout): 22 finger DOFs, 33 rigid bodies, palm-rooted, ``root_joint``, link
  names/transforms/masses/visuals/joint limits, and the baked per-joint tuned
  ``maxForce`` drive limits. All preserved verbatim.
* **Collider source** -- Articulation_Bodex's ``sharpa_right_tuned_instanceable
  .usd``. Its distal (``*_DP``) and elastomer collision meshes differ from
  OCIR's (contact-shaped pads); the local-to-link transform of every collider
  mesh is bit-identical between the two assets (verified), so porting is a
  pure points/topology replacement.

The output is a single self-contained, fully de-instanced layer where every
collision mesh is authored as PhysX ``convexDecomposition`` (minThickness
0.002, hullVertexLimit 64, maxConvexHulls 16) with contactOffset 0.004 /
restOffset 0.001 -- the settings the reference pipeline ships. OCIR's current
asset carries plain ``convexHull`` colliders with no offsets, which fill in
the fingertip-pad concavities.

A ``provenance.json`` is written beside the output: md5 of both source files
plus per-link collider geometry hashes. Re-running the build hard-fails if a
source hash changed (delete the provenance file to accept new sources).

Runs without an Isaac boot: pxr is bootstrapped from the isaacsim package's
``omni.usd.libs`` extension (the script re-execs itself once with
LD_LIBRARY_PATH pointing at its native libs). Invoke via the isaacsim env:

    scripts/run_isaacsim_conda.sh scripts/isaac/build_tuned_hand_usd.py
    scripts/run_isaacsim_conda.sh scripts/isaac/build_tuned_hand_usd.py --verify
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "assets/robots/hands/sharpa_wave/usd/right/right_sharpa_wave/right_sharpa_wave.usd"
DEFAULT_BODEX_SRC = Path(
    os.environ.get("OCIR_ARTICULATION_BODEX_ROOT", Path.home() / "OCIR/third_party/Articulation_Bodex")
) / "assets/hands/sharpa_right_tuned_instanceable.usd"
DEFAULT_OUT = REPO_ROOT / "assets/robots/hands/sharpa_wave/usd/right/right_sharpa_wave_tuned/right_sharpa_wave_tuned.usd"

#: PhysX collision settings of the reference asset (authored by attribute
#: name so plain usd-core / omni.usd.libs suffices -- no PhysxSchema needed).
COLLISION_SETTINGS = {
    "physxConvexDecompositionCollision:minThickness": ("float", 0.002),
    "physxConvexDecompositionCollision:hullVertexLimit": ("int", 64),
    "physxConvexDecompositionCollision:maxConvexHulls": ("int", 16),
    "physxCollision:contactOffset": ("float", 0.004),
    "physxCollision:restOffset": ("float", 0.001),
}
PHYSX_API_TOKENS = ("PhysxConvexDecompositionCollisionAPI", "PhysxCollisionAPI")

#: Links whose collider GEOMETRY is taken from the BODex asset (the grasping
#: surfaces); all other links keep OCIR geometry and only get the collision
#: approximation/settings above.
BODEX_GEOMETRY_LINKS = tuple(
    f"right_{finger}_{part}"
    for finger in ("index", "middle", "ring", "pinky", "thumb")
    for part in ("DP", "elastomer")
)

#: The tuned per-joint drive maxForce values the asset must carry (audited;
#: the runtime reads them from the USD -- this is only a tripwire against a
#: silently re-exported asset with wrong limits).
EXPECTED_MAX_EFFORTS = {
    "MCP_FE": 1.864, "MCP_AA": 1.864,
    "PIP": 0.638, "IP": 0.638,
    "DIP": 0.189369,
    "pinky_CMC": 0.5285,
    "thumb_CMC_FE": 3.3, "thumb_CMC_AA": 3.3,
}
VIRTUAL_JOINT_SUFFIXES = ("x_joint", "y_joint", "z_joint", "roll_joint", "pitch_joint", "yaw_joint")

EXPECTED_FINGER_DOFS = 22
EXPECTED_RIGID_BODIES = 33
EXPECTED_COLLIDERS = 26


def _bootstrap_pxr() -> None:
    """Make ``pxr`` importable from the isaacsim extscache without booting
    Isaac. The native libs need LD_LIBRARY_PATH at process start, so re-exec
    once if it isn't set yet."""

    try:
        import pxr  # noqa: F401
        return
    except ImportError:
        pass
    candidates = []
    for base in sys.path + [str(Path(p) / "site-packages") for p in glob.glob(str(Path(sys.prefix) / "lib/python*"))]:
        candidates.extend(glob.glob(str(Path(base) / "isaacsim/extscache/omni.usd.libs-*")))
    if not candidates:
        raise SystemExit("cannot find omni.usd.libs in the isaacsim extscache; run via scripts/run_isaacsim_conda.sh")
    lib_root = sorted(candidates)[-1]
    sys.path.insert(0, lib_root)
    lib_bin = str(Path(lib_root) / "bin")
    if lib_bin not in os.environ.get("LD_LIBRARY_PATH", ""):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = lib_bin + os.pathsep + env.get("LD_LIBRARY_PATH", "")
        env["_OCIR_PXR_REEXEC"] = "1"
        os.execve(sys.executable, [sys.executable] + sys.argv, env)
    try:
        import pxr  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"pxr import failed even with omni.usd.libs on the path: {exc}")


def expected_effort_for(joint_name: str) -> float | None:
    if joint_name == "right_pinky_CMC":
        return EXPECTED_MAX_EFFORTS["pinky_CMC"]
    if joint_name.startswith("right_thumb_CMC"):
        return EXPECTED_MAX_EFFORTS["thumb_CMC_FE"]
    for suffix in ("MCP_FE", "MCP_AA", "PIP", "DIP"):
        if joint_name.endswith(suffix):
            return EXPECTED_MAX_EFFORTS[suffix]
    if joint_name.endswith("_IP"):
        return EXPECTED_MAX_EFFORTS["IP"]
    return None


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def geometry_hash(points, counts, indices) -> str:
    h = hashlib.md5()
    h.update(len(points).to_bytes(8, "little"))
    h.update(len(indices).to_bytes(8, "little"))
    for v in points:
        h.update(f"{v[0]:.7f},{v[1]:.7f},{v[2]:.7f};".encode())
    h.update(",".join(str(int(i)) for i in indices).encode())
    h.update(",".join(str(int(c)) for c in counts).encode())
    return h.hexdigest()


def link_of(prim):
    from pxr import UsdPhysics

    cur = prim.GetParent()
    while cur and cur.IsValid():
        if cur.HasAPI(UsdPhysics.RigidBodyAPI):
            return cur
        cur = cur.GetParent()
    return None


def collect_collider_meshes(stage):
    """{link_name: mesh_prim} for every collision mesh (instance proxies ok)."""

    from pxr import Usd, UsdGeom, UsdPhysics

    out = {}
    root = stage.GetDefaultPrim()
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not (p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI)):
            continue
        link = link_of(p)
        if link is None:
            continue
        if link.GetName() in out:
            raise SystemExit(f"multiple collision meshes under link {link.GetName()}; expected exactly one")
        out[link.GetName()] = p
    return out


def mesh_geometry(prim):
    points = prim.GetAttribute("points").Get()
    counts = prim.GetAttribute("faceVertexCounts").Get()
    indices = prim.GetAttribute("faceVertexIndices").Get()
    if not points or not counts or not indices:
        raise SystemExit(f"collision mesh {prim.GetPath()} has no geometry")
    return points, counts, indices


def build(src: Path, bodex_src: Path, out: Path) -> dict:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    provenance_path = out.parent / "provenance.json"
    src_md5, bodex_md5 = md5_file(src), md5_file(bodex_src)
    if provenance_path.exists():
        old = json.loads(provenance_path.read_text(encoding="utf-8"))
        stale = {
            name
            for name, (old_hash, new_hash) in {
                "structure_source": (old.get("structure_source_md5"), src_md5),
                "collider_source": (old.get("collider_source_md5"), bodex_md5),
            }.items()
            if old_hash != new_hash
        }
        if stale:
            raise SystemExit(
                f"source USD(s) changed since the committed build: {sorted(stale)}. "
                f"Inspect the new sources, then delete {provenance_path} to accept them."
            )

    bodex_stage = Usd.Stage.Open(str(bodex_src))
    bodex_colliders = collect_collider_meshes(bodex_stage)
    missing = [l for l in BODEX_GEOMETRY_LINKS if l not in bodex_colliders]
    if missing:
        raise SystemExit(f"BODex asset lacks collision meshes for {missing}")

    # De-instance in memory, then flatten: a clean single layer with real,
    # semantically-pathed prims (build_hand de-instances at load anyway).
    src_stage = Usd.Stage.Open(str(src))
    for p in list(src_stage.Traverse()):
        if p.IsInstanceable():
            p.SetInstanceable(False)
    flat_layer = src_stage.Flatten()
    stage = Usd.Stage.Open(flat_layer)

    ours = collect_collider_meshes(stage)
    if len(ours) != EXPECTED_COLLIDERS:
        raise SystemExit(f"structure source composed {len(ours)} collision meshes, expected {EXPECTED_COLLIDERS}")

    link_hashes = {}
    replaced = []
    for link_name, mesh in sorted(ours.items()):
        if link_name in BODEX_GEOMETRY_LINKS:
            b_points, b_counts, b_indices = mesh_geometry(bodex_colliders[link_name])
            mesh.GetAttribute("points").Set(Vt.Vec3fArray(b_points))
            mesh.GetAttribute("faceVertexCounts").Set(Vt.IntArray(b_counts))
            mesh.GetAttribute("faceVertexIndices").Set(Vt.IntArray(b_indices))
            bounds = Gf.Range3f()
            for v in b_points:
                bounds.UnionWith(Gf.Vec3f(v))
            extent_attr = mesh.GetAttribute("extent") or UsdGeom.Mesh(mesh).CreateExtentAttr()
            extent_attr.Set(Vt.Vec3fArray([bounds.GetMin(), bounds.GetMax()]))
            # Per-vertex data of the old geometry no longer lines up.
            for stale in ("normals",):
                attr = mesh.GetAttribute(stale)
                if attr and attr.HasAuthoredValue():
                    attr.Clear()
            for primvar in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
                if primvar.GetInterpolation() in ("vertex", "faceVarying", "uniform"):
                    primvar.GetAttr().Clear()
            replaced.append(link_name)

        UsdPhysics.CollisionAPI.Apply(mesh)
        UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr().Set(UsdPhysics.Tokens.convexDecomposition)
        for token in PHYSX_API_TOKENS:
            mesh.AddAppliedSchema(token)
        for name, (type_name, value) in COLLISION_SETTINGS.items():
            attr_type = Sdf.ValueTypeNames.Float if type_name == "float" else Sdf.ValueTypeNames.Int
            mesh.CreateAttribute(name, attr_type).Set(value)

        points, counts, indices = mesh_geometry(mesh)
        link_hashes[link_name] = geometry_hash(points, counts, indices)

    out.parent.mkdir(parents=True, exist_ok=True)
    stage.GetRootLayer().Export(str(out))

    provenance = {
        "tool": "scripts/isaac/build_tuned_hand_usd.py",
        "structure_source": str(src),
        "structure_source_md5": src_md5,
        "collider_source": str(bodex_src),
        "collider_source_md5": bodex_md5,
        "collision_settings": {k: v for k, (_, v) in COLLISION_SETTINGS.items()},
        "bodex_geometry_links": sorted(replaced),
        "collider_geometry_md5": link_hashes,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"built {out} ({out.stat().st_size / 1e6:.1f} MB); BODex geometry on {len(replaced)} links; provenance {provenance_path}")
    return provenance


def verify(out: Path) -> int:
    from pxr import Usd, UsdGeom, UsdPhysics

    provenance_path = out.parent / "provenance.json"
    failures = []

    def check(ok: bool, message: str):
        print(("PASS " if ok else "FAIL ") + message)
        if not ok:
            failures.append(message)

    check(out.exists(), f"derived asset exists: {out}")
    check(provenance_path.exists(), f"provenance exists: {provenance_path}")
    if failures:
        return 1
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))

    stage = Usd.Stage.Open(str(out))
    joints, bodies, colliders = {}, 0, {}
    for p in stage.Traverse(Usd.TraverseInstanceProxies()):
        if p.IsA(UsdPhysics.RevoluteJoint):
            joints[p.GetName()] = p.GetAttribute("drive:angular:physics:maxForce").Get()
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies += 1
        if p.IsA(UsdGeom.Mesh) and p.HasAPI(UsdPhysics.CollisionAPI):
            link = link_of(p)
            colliders[link.GetName() if link else str(p.GetPath())] = p

    check(len(joints) == EXPECTED_FINGER_DOFS, f"{EXPECTED_FINGER_DOFS} finger DOFs (found {len(joints)})")
    virtual = [n for n in joints if n.endswith(VIRTUAL_JOINT_SUFFIXES)]
    check(not virtual, f"no virtual base DOFs (found {virtual or 'none'})")
    check(bodies == EXPECTED_RIGID_BODIES, f"{EXPECTED_RIGID_BODIES} rigid bodies (found {bodies})")
    check(len(colliders) == EXPECTED_COLLIDERS, f"{EXPECTED_COLLIDERS} collision meshes (found {len(colliders)})")

    bad_settings = []
    for link_name, mesh in sorted(colliders.items()):
        approx = mesh.GetAttribute("physics:approximation").Get()
        if str(approx) != "convexDecomposition":
            bad_settings.append(f"{link_name}: approximation={approx}")
        for name, (_, expected) in COLLISION_SETTINGS.items():
            value = mesh.GetAttribute(name).Get()
            if value is None or abs(float(value) - float(expected)) > 1e-9:
                bad_settings.append(f"{link_name}: {name}={value} (expected {expected})")
    check(not bad_settings, f"convexDecomposition + settings on every collider ({bad_settings[:4] or 'all ok'})")

    hash_mismatch = []
    for link_name, mesh in sorted(colliders.items()):
        points, counts, indices = mesh_geometry(mesh)
        if geometry_hash(points, counts, indices) != provenance["collider_geometry_md5"].get(link_name):
            hash_mismatch.append(link_name)
    check(not hash_mismatch, f"collider geometry matches provenance hashes ({hash_mismatch or 'all ok'})")
    check(
        sorted(provenance.get("bodex_geometry_links", [])) == sorted(BODEX_GEOMETRY_LINKS),
        "BODex geometry ported on exactly the DP+elastomer links",
    )

    effort_mismatch = []
    for name, value in sorted(joints.items()):
        expected = expected_effort_for(name)
        if expected is None:
            effort_mismatch.append(f"{name}: no expected effort class")
        elif value is None or abs(float(value) - expected) > 1e-4:
            effort_mismatch.append(f"{name}: maxForce={value} (expected {expected})")
    check(not effort_mismatch, f"baked per-joint maxForce equals the tuned set ({effort_mismatch[:4] or 'all ok'})")

    print(("VERIFY OK" if not failures else f"VERIFY FAILED ({len(failures)} failures)"))
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Structure source (OCIR hand USD).")
    parser.add_argument("--bodex-src", type=Path, default=DEFAULT_BODEX_SRC, help="Collider source (Articulation_Bodex tuned USD).")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify", action="store_true", help="Only audit an existing derived asset; build nothing.")
    args = parser.parse_args()

    _bootstrap_pxr()
    if args.verify:
        return verify(args.out)
    for path in (args.src, args.bodex_src):
        if not path.exists():
            print(f"source not found: {path}", file=sys.stderr)
            return 2
    build(args.src, args.bodex_src, args.out)
    return verify(args.out)


if __name__ == "__main__":
    raise SystemExit(main())
