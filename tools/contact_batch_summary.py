#!/usr/bin/env python
"""Collect the per-take contact reports into one table, and flag the failures.

Reads every stage4_*.json under a RetargetOutput subtree and prints/writes a CSV of the
numbers that decide whether a take is usable:

  bite IoU        did the hand land where the video says it did (>0.35 is the bar)
  pad_min_cm      did any pad actually reach the surface
  penetration_cm  did it sink in
  neg             pads sitting on surface the camera proves is untouched (want 0)
  hand_health     fraction of reconstructed hand joints inside the observed hand mask;
                  near 0 across EgoDex, which is why alignment is a re-placement
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


FIELDS = ["take", "side", "frame", "Tv", "obj_mask_iou", "bite_iou_before", "bite_iou",
          "translation_cm", "magnitude_cm", "pad_min_cm", "penetration_cm", "pads_on_free",
          "hot_verts", "hot_area_pct", "vetoed_verts", "hand_health", "occlusion_pct"]


def collect(root: Path):
    rows = []
    for jp in sorted(root.rglob("stage4_*.json")):
        d = json.loads(jp.read_text())
        a = d.get("alignment") or {}
        h = d.get("heatmap") or {}
        rows.append({
            "take": Path(d["take"]).name, "side": d["side"], "frame": d["frame"], "Tv": d["Tv"],
            "obj_mask_iou": round(d.get("object_mask_iou", float("nan")), 3),
            "bite_iou_before": round(a.get("bite_iou_before", float("nan")), 3),
            "bite_iou": round(a.get("bite_iou_after", float("nan")), 3),
            "translation_cm": a.get("translation_cm"),
            "magnitude_cm": round(a.get("magnitude_cm", float("nan")), 1),
            "pad_min_cm": round(a.get("pad_min_cm_after", float("nan")), 2),
            "penetration_cm": round(a.get("penetration_max_cm", float("nan")), 2),
            "pads_on_free": round(a.get("pads_on_free_surface", float("nan")), 3),
            "hot_verts": h.get("hot_vertices"),
            "hot_area_pct": round(100 * h.get("hot_area_fraction", float("nan")), 2),
            "vetoed_verts": h.get("vetoed_vertices"),
            "hand_health": round(d.get("hand_reprojection_health", {}).get("mean", float("nan")), 3),
            "occlusion_pct": round(100 * d.get("occlusion", {}).get("ratio", float("nan")), 1),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="RetargetOutput subtree to scan")
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--bar", type=float, default=0.35, help="bite IoU acceptance bar")
    a = ap.parse_args()

    rows = collect(a.root)
    if not rows:
        print(f"no stage4_*.json under {a.root}")
        return 1

    w = {k: max(len(k), *(len(str(r[k])) for r in rows)) for k in FIELDS if k != "translation_cm"}
    hdr = "  ".join(k.ljust(w[k]) for k in FIELDS if k != "translation_cm")
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (len(r["take"]), r["take"])):
        print("  ".join(str(r[k]).ljust(w[k]) for k in FIELDS if k != "translation_cm"))

    ok = [r for r in rows if r["bite_iou"] >= a.bar]
    bad = [r for r in rows if not r["bite_iou"] >= a.bar]
    import statistics as st
    print(f"\n{len(rows)} takes | bite IoU >= {a.bar}: {len(ok)} | below: {len(bad)}"
          f" ({', '.join(r['take'] for r in bad) or 'none'})")
    for k in ("bite_iou", "pad_min_cm", "penetration_cm", "magnitude_cm", "hot_area_pct"):
        v = [r[k] for r in rows if r[k] == r[k]]
        if v:
            print(f"  {k:16s} median {st.median(v):7.3f}   min {min(v):7.3f}   max {max(v):7.3f}")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            cw = csv.DictWriter(fh, fieldnames=FIELDS)
            cw.writeheader()
            cw.writerows(rows)
        print(f"\nwrote {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
