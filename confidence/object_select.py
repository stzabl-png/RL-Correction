"""Select ONE object out of a reconstruction take that may hold several.

The reconstruction grew multi-object support (label_prompt.objects is a list, sam3d and
fp_pose iterate it, world_fused stores object_ob_in_world_all etc.), but every tool in
this directory was written when "a take" meant "an object" and they all still read the
singular npz fields.  On a two-part take that is not merely incomplete, it is WRONG:

  * the singular object_ob_in_cam / object_ob_in_world are object_0 -- verified on
    screw_unscrew_bottle_cap/4, where they equal object_ob_in_world_all[0] exactly;
  * but pose_audit's mask loader ORed EVERY object_*.png together, so object_0's mesh at
    object_0's pose was scored against a silhouette containing object_1 as well.

That mixture attacks the two mask-based criteria directly.  explained is
|mask ∩ proj| / |mask|, so the union puts pixels in the denominator that object_0 can
never cover; d_cent_norm is centroid distance over sqrt(mask area), and the union moves
the centroid toward the other object while inflating the area.  Both get worse the more
of the frame the second object occupies -- i.e. the score degrades for a reason that has
nothing to do with the pose being audited.

Single-object takes resolve to exactly what the old code resolved to (same pose arrays,
same mesh, same single mask), so scores already recorded in pose_audit.json remain
comparable -- including the 29 held-out takes the v3 thresholds were validated against.
Nothing here changes any threshold or any scoring formula.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np

DEFAULT_OBJECT_ID = "object_0"


def object_ids(z) -> list[str]:
    """Object ids in trajectory order; legacy single-object takes report ["object_0"]."""
    if "object_ids" in z.files:
        return [str(x) for x in np.asarray(z["object_ids"]).tolist()]
    return [DEFAULT_OBJECT_ID]


def count(z) -> int:
    return len(object_ids(z))


def poses(z, idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(object_ob_in_cam, object_ob_in_world) for one object, as float arrays."""
    if "object_ob_in_cam_all" in z.files and "object_ob_in_world_all" in z.files:
        return (np.asarray(z["object_ob_in_cam_all"][idx], float),
                np.asarray(z["object_ob_in_world_all"][idx], float))
    if idx != 0:
        raise IndexError(f"take has no per-object arrays; object index {idx} unavailable")
    return (np.asarray(z["object_ob_in_cam"], float),
            np.asarray(z["object_ob_in_world"], float))


def mesh_path(scene: Path, oid: str) -> Path | None:
    """The mesh belonging to one object; falls back to the take-level mesh."""
    for pat in (f"objects/{oid}/*.obj", "*.obj"):
        hits = sorted(glob.glob(str(Path(scene) / pat)))
        if hits:
            return Path(hits[0])
    return None


def obj_mask(scene: Path, frame: int, oid: str):
    """One object's mask -- NOT the union over objects (see module docstring)."""
    import cv2
    p = Path(scene) / "masks" / "objects" / "frames" / f"frame_{frame:06d}_masks" / f"{oid}.png"
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None if a is None else (a > 127)


def resolve(scene: Path, z, spec) -> list[int]:
    """Turn a --object spec into indices. 'all' / None -> every object; else id or index."""
    ids = object_ids(z)
    if spec is None or spec == "all":
        return list(range(len(ids)))
    s = str(spec)
    if s in ids:
        return [ids.index(s)]
    try:
        i = int(s)
    except ValueError:
        raise SystemExit(f"object {s!r} not in {ids}") from None
    if not 0 <= i < len(ids):
        raise SystemExit(f"object index {i} out of range for {ids}")
    return [i]
