#!/usr/bin/env python3
"""Score a simulated rotation gauntlet: which rotation leg broke the grasp.

Reads the gauntlet trajectory's phase table and the simulator's
``object_track.npz``; the object reference never translates during rotation
legs, so per-phase deviation from the reference is grasp slip.  Writes
``rotation_report.json`` next to the track and prints a per-phase table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def _quat_angle_deg(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(qa * qb, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, -1.0, 1.0)))


def analyze(trajectory_dir: Path, sim_dir: Path, *, pos_threshold_m: float, rot_threshold_deg: float) -> dict:
    meta = json.loads((trajectory_dir / "trajectory.json").read_text(encoding="utf-8"))
    gauntlet = meta.get("extra_metadata", {}).get("rotation_gauntlet")
    if not gauntlet:
        raise ValueError(f"{trajectory_dir}: trajectory has no rotation_gauntlet metadata; regenerate it")
    with np.load(sim_dir / "object_track.npz", allow_pickle=False) as data:
        segments = np.asarray(data["segment"])
        n = segments.size
        actual_pos = np.asarray(data["position_world"], dtype=np.float64)[:n]
        actual_quat = np.asarray(data["orientation_world_wxyz"], dtype=np.float64)[:n]
        ref_pos = np.asarray(data["reference_position_world"], dtype=np.float64)[:n]
        ref_quat = np.asarray(data["reference_orientation_world_wxyz"], dtype=np.float64)[:n]

    pos_err = np.linalg.norm(actual_pos - ref_pos, axis=1)
    rot_err = _quat_angle_deg(actual_quat, ref_quat)
    # Deviation carried in from the grasp/lift itself (measured at the end of
    # the post-lift hold) is subtracted so rotation legs are scored on the
    # slip they ADD, not on the entry bias.
    baseline_end = next((p["end"] for p in gauntlet["phases"] if p["name"] == "post_lift_hold"), None)
    pos_baseline = float(pos_err[baseline_end - 1]) if baseline_end else 0.0
    rot_baseline = float(rot_err[baseline_end - 1]) if baseline_end else 0.0

    rows = []
    first_failure = None
    for phase in gauntlet["phases"]:
        s, e = int(phase["start"]), min(int(phase["end"]), n)
        if e <= s:
            continue
        p_max = float(pos_err[s:e].max()) - pos_baseline
        r_max = float(rot_err[s:e].max()) - rot_baseline
        p_final = float(pos_err[e - 1]) - pos_baseline
        r_final = float(rot_err[e - 1]) - rot_baseline
        exceeded = p_max > pos_threshold_m or r_max > rot_threshold_deg
        if exceeded and first_failure is None and phase.get("kind") == "rotate":
            first_failure = phase["name"]
        rows.append({
            "phase": phase["name"], "kind": phase.get("kind"),
            "added_pos_max_m": round(p_max, 4), "added_pos_final_m": round(p_final, 4),
            "added_rot_max_deg": round(r_max, 2), "added_rot_final_deg": round(r_final, 2),
            "exceeded": bool(exceeded),
        })

    report = {
        "trajectory_dir": str(trajectory_dir),
        "sim_dir": str(sim_dir),
        "pos_threshold_m": float(pos_threshold_m),
        "rot_threshold_deg": float(rot_threshold_deg),
        "entry_baseline_pos_m": round(pos_baseline, 4),
        "entry_baseline_rot_deg": round(rot_baseline, 2),
        "passed": first_failure is None and not any(r["exceeded"] for r in rows),
        "first_failing_phase": first_failure,
        "phases": rows,
    }
    (sim_dir / "rotation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, default=None, help="Defaults to <trajectory-dir>/isaac_sim.")
    parser.add_argument("--pos-threshold", type=float, default=0.03, help="Added object translation that counts as slip (m).")
    parser.add_argument("--rot-threshold-deg", type=float, default=20.0, help="Added object rotation lag that counts as slip (deg).")
    args = parser.parse_args(argv)
    sim_dir = args.sim_dir if args.sim_dir is not None else args.trajectory_dir / "isaac_sim"
    try:
        report = analyze(
            args.trajectory_dir, sim_dir,
            pos_threshold_m=float(args.pos_threshold),
            rot_threshold_deg=float(args.rot_threshold_deg),
        )
    except Exception as exc:
        print(f"OCIR_GAUNTLET analysis failed: {exc}", file=sys.stderr, flush=True)
        return 1
    print(f"{'phase':28s} {'kind':7s} {'+pos_max':9s} {'+rot_max':9s} verdict")
    for row in report["phases"]:
        verdict = "SLIP" if row["exceeded"] else "ok"
        print(f"{row['phase']:28s} {str(row['kind']):7s} {row['added_pos_max_m']:8.3f}m {row['added_rot_max_deg']:8.1f}d {verdict}")
    print(f"\nPASSED: {report['passed']}  first_failing_phase: {report['first_failing_phase']}")
    print(f"report: {sim_dir / 'rotation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
