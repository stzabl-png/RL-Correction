#!/usr/bin/env python
"""Merge the left- and right-hand contact heatmaps of one take into a single map.

Unscrewing a cap is bimanual: one hand holds the bottle, the other twists the cap. The two
hands are solved independently -- each gets its own best evidence frame, its own alignment
-- but they press on the SAME object, so the thing a downstream grasp planner or reward
function wants is one map over that object, plus the ability to ask which hand did what.

Combining rule: per-vertex MAX, not sum.
  The weight is "how strongly is this vertex contacted", a saturating quantity bounded by 1.
  Summing would push a vertex both hands merely brushed above one either hand pressed hard,
  and would break the shared 0-1 scale the veto and the sigma were calibrated against.
  MAX keeps the scale and answers the question actually being asked.

The per-hand maps stay in the output, so "both hands touched here" is still recoverable
(both_hot below), and so is "this is the holding hand's region".

Usage:
  python tools/contact_merge_hands.py <RetargetOutput take dir> [--out-prefix contact_heatmap_both]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def find_maps(contact_dir: Path) -> dict:
    """The newest heatmap npz per side (a take may hold several frames' worth)."""
    out = {}
    for side in ("left", "right"):
        hits = sorted(contact_dir.glob(f"contact_heatmap_frame*_{side}.npz"),
                      key=lambda p: p.stat().st_mtime)
        if hits:
            out[side] = hits[-1]
    return out


def export_ply(path: Path, verts: np.ndarray, faces: np.ndarray | None, w: np.ndarray) -> None:
    """Vertex-coloured PLY: grey where untouched, red-hot where contacted."""
    c = np.clip(w, 0, 1)
    rgb = np.stack([60 + 195 * c, 60 + 60 * c * (1 - c) * 4, 60 + 20 * (1 - c)], 1)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        head = ["ply", "format ascii 1.0", f"element vertex {len(verts)}",
                "property float x", "property float y", "property float z",
                "property uchar red", "property uchar green", "property uchar blue"]
        if faces is not None and len(faces):
            head += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
        head += ["end_header"]
        f.write(("\n".join(head) + "\n").encode())
        for v, col in zip(verts, rgb):
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {col[0]} {col[1]} {col[2]}\n".encode())
        if faces is not None and len(faces):
            for tri in faces:
                f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n".encode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("take", type=Path, help="RetargetOutput take dir (holds contact/)")
    ap.add_argument("--out-prefix", default="contact_heatmap_both")
    ap.add_argument("--hot", type=float, default=0.5, help="threshold for the hot-vertex counts")
    a = ap.parse_args()

    cdir = a.take.resolve() / "contact"
    maps = find_maps(cdir)
    if len(maps) < 2:
        raise SystemExit(f"need both hands; found {sorted(maps) or 'none'} under {cdir}\n"
                         f"run tools/contact_align_heatmap.py --side left  --stage heatmap\n"
                         f"and  tools/contact_align_heatmap.py --side right --stage heatmap")

    data, W = {}, {}
    for side, p in maps.items():
        d = np.load(p, allow_pickle=True)
        data[side] = d
        key = next((k for k in ("weight", "vertex_weight", "w") if k in d.files), None)
        if key is None:
            raise SystemExit(f"{p} has no weight array; fields = {d.files}")
        W[side] = np.asarray(d[key], dtype=np.float64)
        print(f"[in]     {side:5s} {p.name}  {len(W[side])} verts, "
              f"hot {(W[side] > a.hot).sum()}")

    if len(set(len(w) for w in W.values())) != 1:
        raise SystemExit("the two maps have different vertex counts -- they are not the "
                         "same mesh, so merging them would be meaningless")

    # MAX, not sum -- see the module docstring.
    both = np.maximum(W["left"], W["right"])
    hot_l, hot_r = W["left"] > a.hot, W["right"] > a.hot
    n = len(both)

    src = data["right"]
    verts = np.asarray(src["vertices"]) if "vertices" in src.files else None
    faces = np.asarray(src["faces"]) if "faces" in src.files else None
    if verts is None:
        raise SystemExit(f"no vertex positions in {maps['right']}; fields = {src.files}")

    out_npz = cdir / f"{a.out_prefix}.npz"
    np.savez_compressed(
        out_npz, weight=both, weight_left=W["left"], weight_right=W["right"],
        vertices=verts, **({"faces": faces} if faces is not None else {}),
        combine="max", sources=np.array([str(maps["left"].name), str(maps["right"].name)]))
    out_ply = cdir / f"{a.out_prefix}.ply"
    export_ply(out_ply, verts, faces, both)

    stats = dict(
        vertices=n, hot_threshold=a.hot,
        hot_left=int(hot_l.sum()), hot_right=int(hot_r.sum()),
        hot_either=int((hot_l | hot_r).sum()), hot_both=int((hot_l & hot_r).sum()),
        coverage_either=float((hot_l | hot_r).mean()),
        combine="max",
        sources={k: str(v) for k, v in maps.items()})
    (cdir / f"{a.out_prefix}.json").write_text(json.dumps(stats, indent=2))

    print(f"\n[merge]  hot left {stats['hot_left']}  right {stats['hot_right']}  "
          f"either {stats['hot_either']} ({stats['coverage_either']:.2%} of the surface)  "
          f"both {stats['hot_both']}")
    if stats["hot_both"] == 0:
        print("[merge]  the two hands' regions do not overlap -- expected when one hand "
              "holds the bottle and the other turns the cap")
    for p in (out_npz, out_ply, cdir / f"{a.out_prefix}.json"):
        print(f"[out]    {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
