"""Affordance-seeded Sharpa grasp synthesis.

For a new object (a sequence dir with an object mesh), this:
  1. predicts the object's expected grasp area with the AffordanceModel
     (cached as ``affordance.npz``; auto-run in its own conda env), then
  2. runs the frozen ``bodex_curobo_v2`` grasp optimizer with seeds restricted
     to that high-affordance region.

It never modifies the frozen ``bodex_curobo_v2`` package -- it only swaps the
seed-pool sampler (see ``affordance_seed.install_affordance_seed_sampler``).

Note on ranking: ``solve_sharpa_bodex`` ranks records by ``rank_score`` (a mix
of similarity terms) which can bury the best force-closure seed; this CLI also
prints the record with the lowest ``grasp_error_max`` so downstream trajectory
generation can pick it.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from ocir.grasp_synthesis.affordance_seed import (
    install_affordance_seed_sampler,
    install_fingertip_contacts,
    install_table_penalty,
    load_affordance_region,
    predict_affordance,
)
from ocir.grasp_synthesis.bodex_curobo_v2.solver import solve_sharpa_bodex
from ocir.grasp_synthesis.object_surface import ObjectSurface


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--affordance-out", type=Path, default=None,
                   help="Where to cache affordance.npz (default: <sequence-dir>/affordance_pred).")
    p.add_argument("--affordance-threshold", type=float, default=0.5,
                   help="Keep object points with predicted affordance > this as the grasp area.")
    p.add_argument("--affordance-n-points", type=int, default=2048)
    p.add_argument("--force-affordance", action="store_true", help="Re-run affordance prediction even if cached.")
    p.add_argument("--seeds", type=int, default=40)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--opt-iters", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--table-up", type=float, nargs=3, default=None,
                   help="Table 'up' direction in the OBJECT frame (x y z). When given, seed points on the "
                        "object's underside are dropped so grasps never approach from below the table.")
    p.add_argument("--min-normal-up", type=float, default=-0.2,
                   help="Keep seed surface points whose outward normal.up >= this (drop down-facing).")
    p.add_argument("--approach-dir", type=float, nargs=3, default=None,
                   help="Human hand approach direction in the OBJECT frame (x y z), from the pre-contact "
                        "wrist path. When given, seed points whose outward normal does not face back toward "
                        "the incoming hand are dropped, narrowing the grasp area to the approached side.")
    p.add_argument("--cone-halfangle-deg", type=float, default=80.0,
                   help="Approach-cone half-angle (deg): keep points with normal.(-approach) >= cos(this). "
                        "Larger = looser (80 keeps the near hemisphere; only drops the far side).")
    p.add_argument("--table-penalty-weight", type=float, default=0.0,
                   help="[opt-① table-in-optimization] Weight of a hand-table collision penalty added to the "
                        "frozen optimizer (relu(depth-below-tabletop)^2 over contact spheres). 0 disables. "
                        "Requires --table-up. e.g. 500 stops the hand diving through the table during optimization.")
    p.add_argument("--fingertip-contacts", action="store_true",
                   help="[opt-② fingertip pinch] Restrict the optimizer to the 5 fingertip contacts (no pads/palm) "
                        "so it builds a TopDown pinch instead of a whole-hand envelope (for small/flat objects).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds <= 0:
        print("--seeds must be positive", file=sys.stderr)
        return 2

    surface = ObjectSurface.from_sequence_dir(args.sequence_dir)
    mesh_path = Path(surface.object_mesh_path)
    affordance_out = args.affordance_out or (args.sequence_dir / "affordance_pred")

    # 1) predict the expected grasp area (cached, own conda env)
    npz = predict_affordance(
        mesh_path, affordance_out,
        n_points=args.affordance_n_points, seed=args.seed, force=args.force_affordance,
    )
    import numpy as np
    up = np.asarray(args.table_up, dtype=float) if args.table_up is not None else None
    approach = np.asarray(args.approach_dir, dtype=float) if args.approach_dir is not None else None
    cone_cos = float(np.cos(np.deg2rad(args.cone_halfangle_deg)))
    points, normals, info = load_affordance_region(
        mesh_path, npz, threshold=args.affordance_threshold,
        world_up_object=up, min_normal_up=args.min_normal_up,
        approach_dir_object=approach, approach_cone_cos=cone_cos,
    )
    print(f"[affordance] region: {info['region_points']}/{info['total_points']} points "
          f"(heat>{args.affordance_threshold}; max={info['heatmap_max']:.3f}) "
          f"bounds {info['region_bounds_min']}..{info['region_bounds_max']}"
          + ("  [top-fraction fallback]" if info["used_top_fraction_fallback"] else ""), flush=True)
    if approach is not None:
        if info["approach_cone_applied"]:
            print(f"[affordance] approach-cone: dir_object={info['approach_dir_object']} "
                  f"halfangle={args.cone_halfangle_deg:.0f}deg  "
                  f"{info['region_points_before_cone']} -> {info['region_points']} points "
                  f"(dropped {info['dropped_by_cone']} on the far side)", flush=True)
        else:
            print(f"[affordance] approach-cone SKIPPED (would leave < min_points; kept "
                  f"{info['region_points']} points)", flush=True)

    # 2) restrict the frozen solver's seed pool to that region, then synthesize
    install_affordance_seed_sampler(points, normals, seed=args.seed)
    # opt-② fingertip pinch: swap the contact-point set to fingertips-only.
    if args.fingertip_contacts:
        install_fingertip_contacts()
        print("[affordance] opt-②: fingertip-only contacts (pinch grasp)", flush=True)
    # opt-①: add a hand-table collision penalty to the optimizer itself.
    if args.table_penalty_weight > 0.0:
        if up is None:
            print("--table-penalty-weight requires --table-up (table 'up' in object frame)", file=sys.stderr)
            return 2
        table_offset = float((surface.points_object_frame @ (up / (np.linalg.norm(up) + 1e-12))).min())
        install_table_penalty(up, table_offset, args.table_penalty_weight)
        print(f"[affordance] opt-①: table penalty in optimization "
              f"(weight={args.table_penalty_weight}, tabletop_offset={table_offset:.4f})", flush=True)
    result = solve_sharpa_bodex(
        args.sequence_dir, args.out_dir,
        seeds=args.seeds, top_k=args.top_k, opt_iters=args.opt_iters, seed=args.seed,
    )

    # record the affordance provenance next to the grasp records
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "affordance_seed_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    # report the best force-closure record (rank_score can bury it)
    records = []
    for path in sorted(glob.glob(str(args.out_dir / "grasp_*.json"))) + \
            sorted(glob.glob(str(args.out_dir / "failed_grasp_*.json"))):
        d = json.loads(Path(path).read_text())
        records.append((Path(path).name, d.get("grasp_error_max"), d.get("dist_error"), d.get("success")))
    print(f"[affordance] done: ok={result.get('ok')} "
          f"successful_seed_count={result.get('successful_seed_count')} out_dir={args.out_dir}", flush=True)
    if records:
        best = min(records, key=lambda r: (r[1] if r[1] is not None else 1e9))
        print(f"[affordance] BEST force-closure record: {best[0]} "
              f"grasp_error_max={best[1]:.4f} dist_error={best[2]:.4f} strict_success={best[3]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
