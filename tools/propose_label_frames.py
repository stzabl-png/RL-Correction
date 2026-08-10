#!/usr/bin/env python
"""Propose a better SAM3D labelling frame per take, and render it for review.

SAM3D reconstructs the object mesh from ONE frame, so that frame decides the mesh.
A good one has to satisfy three things at once, and the current pipeline picks none of
them deliberately (18 of 21 takes in this batch were labelled on frame 0):

  unoccluded   outside every grasp interval, hand touching <=2% of the object mask.
               A hand-covered frame gives SAM2 a partial silhouette and SAM3D
               extrapolates a wrong shape -- that is what broke take 7.
  large        more object pixels means more surface detail to reconstruct from.
  sharp        the biggest mask is often the moment the object is being swung through
               the air, where motion blur destroys the texture. Sharpness (Laplacian
               variance inside the mask) DISCOUNTS a frame rather than disqualifying it:
               on take 14 the object is blurry precisely when it is large, and a hard
               cut-off proposed a frame with 2.8x fewer pixels, which is a bad trade for
               a single-view reconstruction.

It maximises pixels x min(1, sharpness / median sharpness), and never proposes a change
when the frame already in use scores at least as well. Outputs, per take, a side-by-side
of the current versus proposed frame plus the proposed mask, so the choice can be
eyeballed before anything is re-run.

Usage:
  python tools/propose_label_frames.py <ReconstructOutput subtree> --out-dir <DIR>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ego_pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact import frames as F                                        # noqa: E402
from contact import viz                                                # noqa: E402
from contact_align_heatmap import guess_video                          # noqa: E402


def annotated_side(take_dir: Path) -> str:
    ga = take_dir / "grasp_annotation.json"
    if ga.exists():
        ann = json.loads(ga.read_text())["annotations"]
        for s in ("right", "left"):
            if ann.get(s):
                return s
    return "right"


def scan(take, take_dir: Path, max_occl: float) -> list:
    import cv2
    grasp = set(take.contact_frames().tolist()) if take.intervals else set()
    k = np.ones((9, 9), np.uint8)
    rows = []
    for f in range(take.Tv):
        om = cv2.imread(str(take.mask_path("object", f)), cv2.IMREAD_UNCHANGED)
        if om is None:
            continue
        om = om > 0
        if not om.any():
            continue
        hm = None
        for s in ("right", "left"):
            h = cv2.imread(str(take_dir / f"masks/hands/frames/frame_{f:06d}_masks/{s}_hand_0.png"),
                           cv2.IMREAD_UNCHANGED)
            if h is not None:
                hm = h if hm is None else np.maximum(hm, h)
        occl = 0.0 if hm is None else float(
            (om & cv2.dilate((hm > 0).astype(np.uint8), k).astype(bool)).sum()) / max(int(om.sum()), 1)
        img = viz.load_video_frame(take.video_path, f)
        sharp = float("nan")
        if img is not None:
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ys, xs = np.nonzero(om)
            crop = g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            if crop.size > 16:
                sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        rows.append(dict(frame=f, px=int(om.sum()), occl=occl, sharp=sharp,
                         in_grasp=f in grasp))
    return rows


def pick(rows, max_occl: float, current=None) -> dict | None:
    """Largest clean mask, discounted by blur.

    Sharpness is a soft factor, not a filter: on take 14 the object is blurry exactly
    when it is large (Laplacian variance 5-7 against a take median of 23), and a hard
    cut-off there threw away every one of the 48 big candidates and proposed a frame with
    2.8x FEWER pixels. Resolution matters more to a single-view reconstruction than
    sharpness does, so blur scales the score down instead of disqualifying a frame.
    """
    cand = [r for r in rows if not r["in_grasp"] and r["occl"] <= max_occl]
    if not cand:
        return None
    sh = np.array([r["sharp"] for r in rows if np.isfinite(r["sharp"])])
    ref = float(np.median(sh)) if len(sh) else 1.0

    def score(r):
        s = 1.0 if not np.isfinite(r["sharp"]) or ref <= 0 else min(1.0, r["sharp"] / ref)
        return r["px"] * s

    best = dict(max(cand, key=score))
    best["sharpness_reference"] = ref
    best["score"] = score(best)
    # never propose a regression: if the frame in use is already clean and bigger, keep it
    if current is not None and not current["in_grasp"] and current["occl"] <= max_occl \
            and score(current) >= best["score"]:
        keep = dict(current)
        keep.update(sharpness_reference=ref, score=score(current), unchanged=True)
        return keep
    best["unchanged"] = False
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-occlusion", type=float, default=0.02)
    ap.add_argument("--include-rejected", action="store_true",
                    help="also process takes marked unusable by take_quality_gate.py")
    a = ap.parse_args()

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a.out_dir.mkdir(parents=True, exist_ok=True)
    takes = sorted({p.parent for p in a.root.rglob("world_fused.npz") if "interim" not in p.parts},
                   key=lambda p: (len(p.name), p.name))

    summary = []
    for td in takes:
        dq = td / "data_quality.json"
        if dq.exists() and json.loads(dq.read_text()).get("verdict") == "rejected" \
                and not a.include_rejected:
            print(f"[skip]   take {td.name}: marked unusable by the quality gate")
            continue
        side = annotated_side(td)
        take = F.load_take(td, side=side, require_qpos=False)
        if take.video_path is None:
            take.video_path = guess_video(td)
        cur_f = json.loads((td / "masks/objects/label_prompt.json").read_text())["objects"][0]["frame_idx"]

        rows = scan(take, td, a.max_occlusion)
        cur = next((r for r in rows if r["frame"] == cur_f), None)
        best = pick(rows, a.max_occlusion, cur)
        if best is None or cur is None:
            print(f"[skip]   take {td.name}: no candidate frame")
            continue

        od = a.out_dir / td.name
        od.mkdir(parents=True, exist_ok=True)
        om = cv2.imread(str(take.mask_path("object", best["frame"])), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(od / f"object_mask_f{best['frame']:04d}.png"), om)
        img = viz.load_video_frame(take.video_path, best["frame"])
        if img is not None:
            cv2.imwrite(str(od / f"frame_f{best['frame']:04d}.png"), img[:, :, ::-1])

        fig, axes = plt.subplots(1, 2, figsize=(17, 5.4), constrained_layout=True)
        for ax, r, tag in ((axes[0], cur, "current"), (axes[1], best, "proposed")):
            m = cv2.imread(str(take.mask_path("object", r["frame"])), cv2.IMREAD_UNCHANGED) > 0
            im = viz.load_video_frame(take.video_path, r["frame"])
            viz._bg(ax, im, m.shape, f"{tag}: frame {r['frame']}   {r['px']} px   "
                                     f"hand {r['occl']*100:.1f}%   sharp {r['sharp']:.0f}"
                                     f"{'   IN GRASP' if r['in_grasp'] else ''}")
            viz.overlay_mask(ax, m, (0.2, 1.0, 0.3), 0.45)
            ys, xs = np.nonzero(m)
            ax.add_patch(plt.Rectangle((xs.min(), ys.min()), xs.max()-xs.min(), ys.max()-ys.min(),
                                       fill=False, ec="#ff3b30", lw=1.6))
        gain = best["px"] / max(cur["px"], 1)
        fig.suptitle(f"take {td.name} ({side})  |  grasp {take.intervals}  |  "
                     + (f"current frame already the best choice"
                        if best.get("unchanged") else
                        f"proposed frame {best['frame']} gives {gain:.1f}x the object pixels")
                     + f"  |  blur reference (median Laplacian var) {best['sharpness_reference']:.0f}",
                     fontsize=11)
        png = a.out_dir / f"take{td.name}_f{cur['frame']}_to_f{best['frame']}.png"
        fig.savefig(png, dpi=110)
        plt.close(fig)

        rec = dict(take=td.name, side=side, intervals=take.intervals,
                   current=cur, proposed={k: v for k, v in best.items()},
                   pixel_gain=round(gain, 2), mask_png=str(od / f"object_mask_f{best['frame']:04d}.png"),
                   figure=str(png))
        (od / "recommended_frame.json").write_text(json.dumps(rec, indent=2, default=float))
        summary.append(rec)
        print(f"[take {td.name:>3}] {cur['frame']:>4} ({cur['px']:>6} px, hand {cur['occl']*100:>4.1f}%) "
              f"-> {best['frame']:>4} ({best['px']:>6} px, hand {best['occl']*100:>4.1f}%)  "
              f"{gain:>4.1f}x")

    (a.out_dir / "label_frame_proposals.json").write_text(json.dumps(summary, indent=2, default=float))
    big = [r for r in summary if r["pixel_gain"] >= 1.33]
    print(f"\n{len(summary)} takes reviewed | {len(big)} would gain >=1.33x object pixels: "
          + ", ".join(f"{r['take']}({r['pixel_gain']}x)" for r in big))
    print(f"wrote {a.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
