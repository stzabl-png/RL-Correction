#!/usr/bin/env python
"""One camera constant per dataset, agreed across takes -- so the dataset is self-consistent.

The intrinsics are a property of the CAMERA, not of a clip: every EgoDex video comes from
the same Vision Pro, so there is one true focal length and ViPE estimates it independently
N times. Per-clip estimates are therefore noise around a constant, and a robust centre of
many is better than any single one -- with no ground truth involved.

Measured on 28 EgoDex takes: 23 estimates land in 724-808 px and 5 land in 1889-1987 px,
with nothing in between. The high cluster is exactly the clips where the head barely moves
(a static camera gives monocular SLAM no parallax, so it fabricates both the motion and the
focal). A median shrugs that off; a mean does not (mean 957.8 vs median 745.0).

WHAT THIS BUYS AND WHAT IT DOES NOT
  consistency  yes -- every clip reconstructed with the same constant is mutually
               comparable, which is what matters when the sim consumes the whole dataset.
  accuracy     no -- the consensus carries the estimator's own bias (+0.65% against the
               EgoDex ground truth here), and that bias does NOT shrink with more samples.
               Absolute scale still leaks in through the robot asset, whose size is real:
               a 10% consensus error would misplace a grasp by ~1 cm on a 10 cm object.
               At 0.65% it is 0.65 mm, i.e. irrelevant.

HOW MANY TAKES
  Not a fixed number. The purpose of the sample is not precision -- it is to be sure you
  are freezing on the MAIN cluster rather than on an outlier. So the report shows the
  cluster structure and the bootstrap spread and lets a human decide; it deliberately does
  not print a "ready" verdict from a magic N.

FREEZING
  Once written, the constant must not drift: a later value would make new clips
  incomparable with old ones, which is the exact thing this exists to prevent. Rewriting
  requires --refreeze and means re-running the whole dataset.

Usage:
  python tools/dataset_camera_consensus.py <ReconstructOutput/dataset dir> [--freeze]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def harvest(root: Path) -> list:
    """Every independent per-clip estimate under `root`.

    Clips already reconstructed WITH a frozen constant are skipped: feeding those back
    would be the estimate validating itself.
    """
    rows = []
    for p in sorted(root.rglob("world_fused.npz")):
        if "interim" in p.parts:
            continue
        try:
            d = np.load(p, allow_pickle=True)
        except Exception as e:
            print(f"[warn] unreadable {p}: {e!r}")
            continue
        if "K" not in d.files:
            continue
        src = str(d["intrinsics_source"]) if "intrinsics_source" in d.files else "per_video"
        K = np.asarray(d["K"], dtype=float)
        rows.append(dict(take=str(p.parent.relative_to(root)), fx=float(K[0, 0]),
                         fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]),
                         frames=int(d["num_frames"]) if "num_frames" in d.files else -1,
                         source=src))
    return [r for r in rows if r["source"] != "dataset_constant"]


def mad_outliers(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad <= 0:
        return np.zeros(len(x), bool)
    return np.abs(x - med) > k * 1.4826 * mad


def cluster_report(x: np.ndarray, keep: np.ndarray) -> dict:
    """Is the surviving set one tight cluster, and how isolated are the rejects?"""
    g = x[keep]
    out = dict(n_total=int(len(x)), n_kept=int(keep.sum()),
               outlier_fraction=float((~keep).mean()),
               median=float(np.median(g)), mean=float(g.mean()),
               std=float(g.std(ddof=1)) if len(g) > 1 else 0.0)
    out["rel_spread"] = out["std"] / out["median"] if out["median"] else float("nan")
    if (~keep).any():
        out["gap_to_nearest_outlier"] = float(
            (np.abs(x[~keep] - out["median"]).min()) / out["median"])
    return out


def bootstrap_median(g: np.ndarray, n_boot=4000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(g, size=(n_boot, len(g)), replace=True), axis=1)
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return dict(ci95=[float(lo), float(hi)],
                half_width_rel=float((hi - lo) / 2 / np.median(g)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path, help="e.g. Output/ReconstructOutput/egodex")
    ap.add_argument("--mad-k", type=float, default=3.0)
    ap.add_argument("--freeze", action="store_true", help="write dataset_camera.json")
    ap.add_argument("--refreeze", action="store_true",
                    help="overwrite an existing constant -- every take must then be re-run")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    rows = harvest(a.dataset_dir)
    if not rows:
        print(f"no independent intrinsics estimates under {a.dataset_dir}")
        return 1
    fx = np.array([r["fx"] for r in rows])
    bad = mad_outliers(fx, a.mad_k)
    keep = ~bad
    rep = cluster_report(fx, keep)
    boot = bootstrap_median(fx[keep])

    for r, o in sorted(zip(rows, bad), key=lambda z: z[0]["fx"]):
        print(f"  {'OUT ' if o else '    '}{r['take']:>44} fx={r['fx']:>8.1f} "
              f"({r['fx']/rep['median']:>5.2f}x median) frames={r['frames']:>4}")

    print(f"\n{rep['n_total']} estimates | kept {rep['n_kept']} | "
          f"outliers {rep['outlier_fraction']:.0%} (MAD k={a.mad_k})")
    print(f"consensus fx = {rep['median']:.1f}   (mean of kept {rep['mean']:.1f}, "
          f"spread {rep['rel_spread']:.1%})")
    print(f"bootstrap 95% CI [{boot['ci95'][0]:.1f}, {boot['ci95'][1]:.1f}] "
          f"= +-{boot['half_width_rel']:.1%}")
    if "gap_to_nearest_outlier" in rep:
        print(f"nearest outlier sits {rep['gap_to_nearest_outlier']:.0%} away from the "
              f"consensus -- a clean gap means the split is unambiguous")

    # what a human should look at before freezing; deliberately not a pass/fail verdict
    print("\nbefore freezing, check:")
    print(f"  * outlier fraction {rep['outlier_fraction']:.0%} -- well under 50%? "
          f"(a dataset of mostly static-camera clips would form a confident WRONG consensus)")
    print(f"  * kept spread {rep['rel_spread']:.1%} -- is this one tight cluster, or a "
          f"broad smear that has no consensus to find?")
    print(f"  * are the rejects explainable? here they were the clips with a near-static "
          f"head, where monocular SLAM has no parallax to work from")

    cxs = {r["cx"] for r in rows}
    cys = {r["cy"] for r in rows}
    fxy = max(abs(r["fx"] - r["fy"]) / r["fx"] for r in rows)
    print(f"  * cx {'constant at %.0f' % cxs.pop() if len(cxs) == 1 else 'VARIES %s' % sorted(cxs)}"
          f", cy {'constant at %.0f' % cys.pop() if len(cys) == 1 else 'VARIES'}"
          f", |fx-fy| <= {fxy:.2%} -- only the focal needs a consensus if these hold")

    out = a.out or (a.dataset_dir / "dataset_camera.json")
    if not a.freeze:
        print(f"\n(dry run -- pass --freeze to write {out})")
        return 0
    if out.exists() and not a.refreeze:
        prev = json.loads(out.read_text())
        print(f"\n{out} already frozen at fx={prev['fx']:.1f} on {prev['frozen_at']}. "
              f"Changing it makes new takes incomparable with the ones already built from "
              f"it; pass --refreeze and re-run the whole dataset if that is intended.")
        return 1
    payload = dict(
        schema_version="dataset_camera_v1",
        dataset=a.dataset_dir.name,
        fx=rep["median"], fy=rep["median"], cx=960.0, cy=540.0,
        image_size=[1920, 1080],
        note="focal is in PIXELS: this constant is only valid for the image size above. "
             "A cropped or rescaled copy of the same footage needs its own constant.",
        method=f"median of {rep['n_kept']} per-clip GeoCalib estimates after MAD "
               f"(k={a.mad_k}) outlier rejection",
        statistics=rep | boot,
        frozen_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        samples=[{k: r[k] for k in ("take", "fx", "frames")}
                 for r, o in zip(rows, bad) if not o],
        rejected=[{k: r[k] for k in ("take", "fx", "frames")}
                  for r, o in zip(rows, bad) if o],
    )
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nfrozen: {out}")
    print("every take must now be reconstructed with this constant -- a mix of frozen and "
          "per-clip intrinsics is exactly the inconsistency this removes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
