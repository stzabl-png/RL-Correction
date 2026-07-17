#!/usr/bin/env python3
"""Generate a wrist-rotation robustness trajectory from a vertical-lift result.

The output directory is a normal ``grasp_traj``-compatible trajectory; replay
it with the standard simulator, then score it with
``analyze_rotation_gauntlet.py``:

    scripts/run_isaacsim_conda.sh scripts/isaac/simulate_grasp_traj.py \\
      --trajectory-dir <out-dir> --out-dir <out-dir>/isaac_sim
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback

from ocir.grasp_robustness.rotation_gauntlet import GauntletConfig, build_rotation_gauntlet
from ocir.grasp_traj.trajectory_schema import GraspTrajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grasp-traj-dir", type=Path, required=True, help="Vertical-lift grasp_traj result (trajectory.npz/json) whose lift direction defines world-up.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lift-height", type=float, default=0.15, help="Lift before the rotation program (m).")
    parser.add_argument("--lift-seconds", type=float, default=3.0)
    parser.add_argument("--angles-deg", default="45", help="Comma-separated rotation amplitudes per axis, e.g. '30,60'. Each amplitude runs +/- legs with returns.")
    parser.add_argument("--angular-speed-deg", type=float, default=45.0, help="Peak angular speed of the cosine-eased rotation legs (deg/s).")
    parser.add_argument("--hold-seconds", type=float, default=0.4, help="Stationary hold between legs.")
    parser.add_argument("--axes", default="yaw,pitch,roll", help="Comma-separated subset of yaw,pitch,roll. yaw is about world-up; pitch/roll about two arbitrary horizontal axes.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        print(f"OCIR_GAUNTLET refusing non-empty --out-dir {args.out_dir}; pass --overwrite", file=sys.stderr)
        return 2
    try:
        config = GauntletConfig(
            lift_height_m=float(args.lift_height),
            lift_seconds=float(args.lift_seconds),
            angles_deg=tuple(float(a) for a in str(args.angles_deg).split(",") if a.strip()),
            peak_angular_speed_deg_s=float(args.angular_speed_deg),
            hold_seconds=float(args.hold_seconds),
            axes=tuple(a.strip() for a in str(args.axes).split(",") if a.strip()),
        )
        source = GraspTrajectory.load(args.grasp_traj_dir)
        trajectory, meta = build_rotation_gauntlet(source, config)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        trajectory.save(args.out_dir)
    except Exception as exc:
        print(f"OCIR_GAUNTLET generation failed: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    print(
        f"OCIR_GAUNTLET wrote {args.out_dir} "
        f"(gauntlet_frames={meta['num_gauntlet_frames']}, duration={meta['duration_seconds']:.1f}s, "
        f"phases={len(meta['phases'])})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
