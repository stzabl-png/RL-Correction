#!/usr/bin/env python
"""All of v17A's interaction instances -> ONE multi-object label_prompt.json.

Why not reconstruct each object separately and merge afterwards: ViPE's focal-length
estimate is not deterministic. Two runs over the same video produced intrinsics differing
by 58-98 px and hand trajectories differing by 3-13 cm, so the two objects come out in
DIFFERENT world frames and any merge silently misplaces one relative to the other.

The pipeline already tracks several objects in a single pass (label_prompt.objects is a
list, sam3d/fp_pose iterate it, world_fused stores object_ob_in_world_all). Feeding every
instance in at once keeps them in one consistent frame -- which is what a paired scene
like bottle + cap needs.

Each object gets its OWN best frame: SAM3D reconstructs each mesh from a single view, and
the frame that shows the bottle best is not the one that shows the cap best.

Usage:
  python tools/v17a_multi_object_prompt.py <video_mask_sequence.json> --step-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v17a_to_label_prompt import (SCHEMA, hand_overlap, interior_point,  # noqa: E402
                                  load_manifest, mask_path, rank_frames)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--step-dir", type=Path, default=None)
    ap.add_argument("--recon-take-dir", type=Path, default=None)
    ap.add_argument("--max-hand-overlap", type=float, default=0.02)
    ap.add_argument("--min-area", type=int, default=800,
                    help="drop instances whose peak mask is smaller than this; SAM3D "
                         "cannot build anything usable from a few hundred pixels")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--exclude", action="append", default=[],
                    help="剔除的 v17A 实例 id(可重复); VLM 透明门判空透明的实例走这里")
    a = ap.parse_args()

    import cv2

    manifest = load_manifest(a.manifest)
    base = a.manifest.parent
    rows = rank_frames(manifest, base)
    ids = sorted({r["object_id"] for r in rows})
    if a.exclude:
        dropped = [i for i in ids if i in set(a.exclude)]
        ids = [i for i in ids if i not in set(a.exclude)]
        if dropped:
            print(f"[multi-prompt] 透明门剔除: {dropped}")
        if not ids:
            raise SystemExit("[multi-prompt] X 全部实例被剔除, 无可注册物体")
    peak = {i: max(r["area"] for r in rows if r["object_id"] == i) for i in ids}
    order = sorted(ids, key=lambda i: -peak[i])
    print(f"[v17a]   instances: " + ", ".join(f"{i}={peak[i]}px" for i in order))

    objects, skipped = [], []
    for n, oid in enumerate(order):
        if peak[oid] < a.min_area:
            skipped.append(f"{oid} (peak {peak[oid]}px)")
            continue
        pick = None
        for r in [x for x in rows if x["object_id"] == oid]:
            if r["mask"] is None or not r["mask"].exists():
                continue
            m = cv2.imread(str(r["mask"]), cv2.IMREAD_UNCHANGED)
            if m is None:
                continue
            m = m > 0
            ov = hand_overlap(m, a.recon_take_dir, r["frame"]) if a.recon_take_dir else float("nan")
            if np.isnan(ov) or ov <= a.max_hand_overlap:
                x, y, depth = interior_point(m)
                pick = dict(r, x=x, y=y, depth=depth, hand=ov)
                break
        if pick is None:
            skipped.append(f"{oid} (no frame under the hand-overlap bar)")
            continue
        objects.append({
            "object_id": f"object_{n}",
            "frame_idx": int(pick["frame"]),
            "points": [[pick["x"], pick["y"]]],
            "labels": [1],
            "locked": True,
            "_v17a": {"instance": oid, "area_px": int(pick["area"]),
                      "hand_overlap": None if np.isnan(pick["hand"]) else round(float(pick["hand"]), 4)},
        })
        print(f"[v17a]   object_{n} <- {oid}: frame {pick['frame']}, {pick['area']} px, "
              f"click ({pick['x']:.0f}, {pick['y']:.0f})")
    if skipped:
        print(f"[v17a]   skipped: {'; '.join(skipped)}")
    if not objects:
        raise SystemExit("no usable instance")

    prompt = {"schema_version": SCHEMA,
              "objects": [{k: v for k, v in o.items() if not k.startswith("_")} for o in objects],
              # ★ filter_policy: 用什么透明策略筛的。auto_label 下次进来对不上就重标注 ——
              #   2026-08-14 实测: 策略从 strict 改回 v2 后, 已缓存的标注不会重跑,
              #   于是带着按旧规则剔掉的物体一路跑到底, 全程不报错。
              "provenance": {"source": "hoi_detr_v17a_multi", "manifest": str(a.manifest),
                             "filter_policy": os.environ.get("AUTO_LABEL_FILTER_POLICY", "v2"),
                             "objects": [o["_v17a"] | {"object_id": o["object_id"],
                                                       "frame_idx": o["frame_idx"]} for o in objects],
                             "note": "every instance tracked in ONE reconstruction pass so "
                                     "they share a world frame; per-object label frames "
                                     "because SAM3D builds each mesh from its own view"}}
    if a.dry_run or a.step_dir is None:
        print(json.dumps(prompt, indent=2))
        return 0
    a.step_dir.mkdir(parents=True, exist_ok=True)
    out = a.step_dir / "label_prompt.json"
    out.write_text(json.dumps(prompt, indent=2))
    print(f"[out]    {out}  ({len(objects)} objects)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
