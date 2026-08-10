#!/usr/bin/env python
"""v17A automatic interaction masks -> the reconstruction pipeline's label_prompt.json.

This is the piece that lets the object branch run without a human clicking on the
object. It does NOT hand v17A's masks to the pipeline directly, on purpose:

  v17A only produces masks inside the interaction window (measured on take 11: frames
  31-62 of 78), because its seeds come from the hand-object link. The reconstruction
  needs masks for the WHOLE video -- fp_pose tracks the object before and after the
  grasp too. So the useful thing v17A provides is not the masks but the ANSWER TO THE
  QUESTION A HUMAN WOULD OTHERWISE ANSWER: which frame shows the object best, and where
  is it in that frame. Feed that in as a label prompt and the pipeline's own SAM2 pass
  propagates it across the full video exactly as if a person had clicked.

Frame choice follows what the contact work measured: SAM3D reconstructs the mesh from
this single frame, so it must be one where the hand does not cover the object (a
hand-occluded frame gave take 7 a 1.9:1 stretched mesh) and where the object is large.

The click point is the mask's pole of inaccessibility (the deepest interior point by
distance transform), not its centroid -- a centroid can fall outside a C-shaped or
occluded mask, which would prompt SAM2 on background.

Usage:
  python tools/v17a_to_label_prompt.py <video_mask_sequence.json> --step-dir <sam2_object interim dir>
  python tools/v17a_to_label_prompt.py <manifest> --dry-run        # just report the choice
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SCHEMA = "sam2_object_prompt_v2"


def load_manifest(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    if not d.get("frames"):
        raise SystemExit(f"{path}: manifest has no frames (v17A run failed?)")
    return d


def mask_path(base: Path, entry: dict) -> Path | None:
    for key in ("mask", "raw_mask"):
        p = entry.get(key)
        if p:
            p = Path(p)
            return p if p.is_absolute() else base / p
    return None


def interior_point(mask: np.ndarray) -> tuple:
    """Deepest interior pixel -- guaranteed inside, unlike a centroid."""
    import cv2
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(x), float(y), float(dist[y, x])


def rank_frames(manifest: dict, base: Path) -> list:
    """Every accepted (frame, object) with its area, best first."""
    rows = []
    for f in manifest["frames"]:
        for oid, o in (f.get("objects") or {}).items():
            if o.get("status") != "accepted":
                continue
            area = (o.get("metrics") or {}).get("area_pixels") or 0
            rows.append(dict(frame=int(f["frame_idx"]), object_id=oid, area=int(area),
                             mask=mask_path(base, o)))
    rows.sort(key=lambda r: -r["area"])
    return rows


def hand_overlap(mask: np.ndarray, take_dir: Path, frame: int, dilate=9) -> float:
    """Fraction of the object mask the hand touches, if hand masks are available."""
    import cv2
    hm = None
    for side in ("right", "left"):
        p = take_dir / f"masks/hands/frames/frame_{frame:06d}_masks/{side}_hand_0.png"
        h = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if h is not None:
            hm = h if hm is None else np.maximum(hm, h)
    if hm is None:
        return float("nan")
    k = np.ones((2 * dilate + 1,) * 2, np.uint8)
    near = cv2.dilate((hm > 0).astype(np.uint8), k).astype(bool)
    if near.shape != mask.shape:
        near = cv2.resize(near.astype(np.uint8), (mask.shape[1], mask.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    return float((mask & near).sum()) / max(int(mask.sum()), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path, help="v17A video_mask_sequence.json")
    ap.add_argument("--step-dir", type=Path, default=None,
                    help="sam2_object interim dir to write label_prompt.json into")
    ap.add_argument("--recon-take-dir", type=Path, default=None,
                    help="final recon take dir holding masks/hands/, to score hand overlap")
    ap.add_argument("--max-hand-overlap", type=float, default=0.02)
    ap.add_argument("--object-index", type=int, default=None,
                    help="when v17A finds several instances, pick this one (0-based, by "
                         "peak area); default: the largest")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import cv2

    manifest = load_manifest(a.manifest)
    base = a.manifest.parent
    rows = rank_frames(manifest, base)
    if not rows:
        raise SystemExit("no accepted object mask in the manifest")

    ids = sorted({r["object_id"] for r in rows})
    peak = {i: max(r["area"] for r in rows if r["object_id"] == i) for i in ids}
    order = sorted(ids, key=lambda i: -peak[i])
    print(f"[v17a]   instances by peak area: " + ", ".join(f"{i}={peak[i]}px" for i in order))
    chosen_id = order[a.object_index] if a.object_index is not None else order[0]
    if len(ids) > 1 and a.object_index is None:
        print(f"[v17a]   WARNING: {len(ids)} instances found; taking the largest "
              f"({chosen_id}). Use --object-index to override -- the pipeline "
              f"reconstructs ONE object and picking the wrong one is silent.")

    cand = [r for r in rows if r["object_id"] == chosen_id]
    pick = None
    for r in cand:
        if r["mask"] is None or not r["mask"].exists():
            continue
        m = cv2.imread(str(r["mask"]), cv2.IMREAD_UNCHANGED)
        if m is None:
            continue
        m = m > 0
        ov = hand_overlap(m, a.recon_take_dir, r["frame"]) if a.recon_take_dir else float("nan")
        r["hand"] = ov
        if np.isnan(ov) or ov <= a.max_hand_overlap:
            x, y, depth = interior_point(m)
            pick = dict(r, x=x, y=y, depth=depth, mask_arr=m)
            break
    if pick is None:
        raise SystemExit(f"no frame of {chosen_id} passes the hand-overlap bar "
                         f"({a.max_hand_overlap}); v17A's window may be entirely inside the grasp")

    top = cand[:5]
    print(f"[v17a]   top frames for {chosen_id}: "
          + ", ".join(f"f{r['frame']}={r['area']}px" for r in top))
    print(f"[v17a]   chosen frame {pick['frame']}: {pick['area']} px, "
          f"hand overlap {pick['hand']*100:.1f}%" if not np.isnan(pick["hand"])
          else f"[v17a]   chosen frame {pick['frame']}: {pick['area']} px (hand overlap unknown)")
    print(f"[v17a]   click point ({pick['x']:.0f}, {pick['y']:.0f}), "
          f"{pick['depth']:.0f} px clear of the mask boundary")

    prompt = {
        "schema_version": SCHEMA,
        "objects": [{
            "object_id": "object_0",
            "frame_idx": int(pick["frame"]),
            "points": [[pick["x"], pick["y"]]],
            "labels": [1],
            "locked": True,
        }],
        "provenance": {
            "source": "hoi_detr_v17a",
            "manifest": str(a.manifest),
            "instance": chosen_id,
            "instances_found": len(ids),
            "mask_area_px": int(pick["area"]),
            "hand_overlap": None if np.isnan(pick["hand"]) else round(float(pick["hand"]), 4),
            "note": "frame and click point derived automatically from the v17A interaction "
                    "masks; the pipeline's own SAM2 pass still propagates over the full video",
        },
    }
    if a.dry_run or a.step_dir is None:
        print(json.dumps(prompt, indent=2))
        return 0
    a.step_dir.mkdir(parents=True, exist_ok=True)
    out = a.step_dir / "label_prompt.json"
    out.write_text(json.dumps(prompt, indent=2))
    print(f"[out]    {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
