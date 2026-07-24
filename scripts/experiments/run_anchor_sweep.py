#!/usr/bin/env python3
"""Overnight sweep of anchored-BODex synthesis constraints + human-video anchoring.

Runs a fixed list of one-factor-at-a-time arms around the current anchored_bodex
defaults (plus a pure bodex_curobo_v2 zero-anchor baseline), three sequences each,
and chains grasp_traj generation + the Isaac lift simulation after each arm's
synthesis. Everything is sequential (one GPU job at a time), deadline-bounded,
and resumable: re-running with the same --name skips every unit whose output
already exists, so a morning re-run fills in whatever the deadline cut off.

Output layout (one root per sweep):

    <out-root>/<name>/
      sweep_manifest.json           arms + flags + budget, written at start
      driver.log                    everything the driver prints
      status.jsonl                  one line per unit (done/failed/timeout/skipped)
      logs/<arm>__<seq>.{synth,traj}.log
      synthesis/<arm>/<seq>/        grasp records + summary.json (CLI appends <seq>)
      traj/<arm>/<seq>/             trajectory + isaac_sim/report.json
      results.csv, results.md       aggregation (also re-runnable via --aggregate-only)

Typical launch (12 h budget):

    nohup python3 scripts/experiments/run_anchor_sweep.py --hours 12 \
        > /data/users/hangkes2/OCIR/testing/sweeps/nohup.out 2>&1 &

Uses only the stdlib; the actual jobs run through scripts/run_grasp_synthesis_conda.sh.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
WRAPPER = REPO / "scripts" / "run_grasp_synthesis_conda.sh"
ANCHORED_CLI = REPO / "scripts" / "grasp_synthesis" / "synthesize_sharpa_anchored_bodex.py"
PURE_CLI = REPO / "scripts" / "grasp_synthesis" / "synthesize_sharpa_bodex_curobo_v2.py"
TRAJ_CLI = REPO / "scripts" / "grasp_traj" / "generate_grasp_traj.py"

SEQ_ROOT = Path("/data/users/hangkes2/OCIR/processed_data/dex_ycb/sequences")
DEFAULT_OUT_ROOT = Path("/data/users/hangkes2/OCIR/testing/sweeps")
SEQUENCES = ["20200709_150949", "20200709_142123", "20200709_151724"]
CONTROL_HOST, CONTROL_PORT = "127.0.0.1", 8765

SYNTH_TIMEOUT_S = 1800
SYNTH_TIMEOUT_HEAVY_S = 3600
SIM_TIMEOUT_S = 1500
AGGREGATE_BUFFER_S = 900  # reserved at the end of the budget for aggregation

# Arms in priority order: the deadline cuts from the tail, so single-factor
# probes come before combos and the heavy (seeds/iters) arms sit late.
# kind: "anchored" -> anchored_bodex CLI, "pure" -> bodex_curobo_v2 baseline.
ARMS: list[dict] = [
    {"name": "01_center", "kind": "anchored", "flags": []},
    {"name": "02_pure_baseline", "kind": "pure", "flags": []},
    {"name": "03_fc100", "kind": "anchored", "flags": ["--force-closure-weight", "100"]},
    {"name": "04_fc1000", "kind": "anchored", "flags": ["--force-closure-weight", "1000"]},
    {"name": "05_fc2000", "kind": "anchored", "flags": ["--force-closure-weight", "2000"]},
    {"name": "06_pene300", "kind": "anchored", "flags": ["--penetration-weight", "300"]},
    {"name": "07_pene900", "kind": "anchored", "flags": ["--penetration-weight", "900"]},
    {"name": "08_contact_subset", "kind": "anchored", "flags": ["--contact-subset"]},
    {"name": "09_pose30", "kind": "anchored", "flags": ["--pose-weight", "30"]},
    {"name": "10_pose300", "kind": "anchored", "flags": ["--pose-weight", "300"]},
    {"name": "11_afford0", "kind": "anchored", "flags": ["--affordance-weight", "0"]},
    {"name": "12_afford60", "kind": "anchored", "flags": ["--affordance-weight", "60"]},
    {"name": "13_preclear3mm", "kind": "anchored", "flags": ["--pregrasp-clearance", "0.003"]},
    {"name": "14_preclear8mm", "kind": "anchored", "flags": ["--pregrasp-clearance", "0.008"]},
    {"name": "15_seeds60", "kind": "anchored", "flags": ["--seeds", "60"], "heavy": True},
    {"name": "16_selfcol100", "kind": "anchored", "flags": ["--selfcollision-weight", "100"]},
    {"name": "17_iters1000", "kind": "anchored", "flags": ["--opt-iters", "1000"], "heavy": True},
    {
        "name": "18_wide_jitter",
        "kind": "anchored",
        "flags": [
            "--jitter-pos", "0.03", "--jitter-rot-deg", "30",
            "--jitter-joint", "0.2", "--relax-standoff", "0.03",
        ],
    },
    {
        "name": "19_full_anchor",
        "kind": "anchored",
        "flags": ["--pose-weight", "30", "--contact-subset", "--affordance-weight", "60"],
    },
    {
        "name": "20_fc1000_pene300",
        "kind": "anchored",
        "flags": ["--force-closure-weight", "1000", "--penetration-weight", "300"],
    },
    {
        "name": "21_fc1000_seeds60",
        "kind": "anchored",
        "flags": ["--force-closure-weight", "1000", "--seeds", "60"],
        "heavy": True,
    },
    {
        "name": "22_pose30_pene900",
        "kind": "anchored",
        "flags": ["--pose-weight", "30", "--penetration-weight", "900"],
    },
]


class Driver:
    def __init__(self, root: Path, deadline: float) -> None:
        self.root = root
        self.deadline = deadline
        self.log_path = root / "driver.log"
        self.status_path = root / "status.jsonl"

    def log(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.log_path.open("a") as f:
            f.write(line + "\n")

    def status(self, **fields) -> None:
        fields["ts"] = datetime.now().isoformat(timespec="seconds")
        with self.status_path.open("a") as f:
            f.write(json.dumps(fields) + "\n")

    def time_left(self) -> float:
        return self.deadline - time.time()

    def run_unit(self, cmd: list[str], log_file: Path, timeout_s: int) -> tuple[int | None, float]:
        """Run one job in its own process group; returns (rc or None on timeout, seconds)."""
        start = time.time()
        with log_file.open("w") as lf:
            lf.write("+ " + " ".join(str(c) for c in cmd) + "\n")
            lf.flush()
            proc = subprocess.Popen(
                [str(c) for c in cmd],
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(REPO),
                start_new_session=True,
            )
            try:
                rc = proc.wait(timeout=timeout_s)
                return rc, time.time() - start
            except subprocess.TimeoutExpired:
                self.log(f"  TIMEOUT after {timeout_s}s -> killing process group")
                for sig, grace in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
                    try:
                        os.killpg(proc.pid, sig)
                    except ProcessLookupError:
                        break
                    try:
                        proc.wait(timeout=grace)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                return None, time.time() - start


def isaac_server_up() -> bool:
    try:
        with socket.create_connection((CONTROL_HOST, CONTROL_PORT), timeout=3):
            return True
    except OSError:
        return False


def synth_cmd(arm: dict, seq: str, out_root: Path, extra: list[str]) -> list[str]:
    cli = PURE_CLI if arm["kind"] == "pure" else ANCHORED_CLI
    return [
        WRAPPER, cli,
        "--sequence-dir", SEQ_ROOT / seq,
        "--out-dir", out_root / "synthesis" / arm["name"],
        "--seed", "0",
        "--no-isaac-visualize",
        *arm["flags"],
        *extra,
    ]


def traj_cmd(arm: dict, seq: str, out_root: Path) -> list[str]:
    return [
        WRAPPER, TRAJ_CLI,
        "--sequence-dir", SEQ_ROOT / seq,
        "--synthesis-out-dir", out_root / "synthesis" / arm["name"] / seq,
        "--out-dir", out_root / "traj" / arm["name"] / seq,
    ]


def run_sweep(args: argparse.Namespace, root: Path) -> None:
    arms = [a for a in ARMS if not args.arms or any(s in a["name"] for s in args.arms)]
    seqs = list(SEQUENCES)
    extra_synth: list[str] = []
    if args.smoke:
        arms = arms[:2]  # center + pure baseline: exercises both CLIs and the sim chain
        seqs = seqs[:1]
        extra_synth = ["--seeds", "4", "--opt-iters", "150"]

    deadline = time.time() + args.hours * 3600 - AGGREGATE_BUFFER_S
    drv = Driver(root, deadline)
    (root / "logs").mkdir(parents=True, exist_ok=True)

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "hours": args.hours,
        "smoke": args.smoke,
        "sequences": seqs,
        "extra_synth_flags": extra_synth,
        "arms": arms,
        "synthesis_defaults": "anchored_bodex CLI defaults as of this commit "
        "(fc-weight 500, pose 0, penetration 0, contact-subset off, affordance 20, "
        "selfcollision 1000, 20 seeds / 500 iters, --seed 0, --no-isaac-visualize)",
    }
    (root / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    drv.log(f"sweep root: {root}")
    drv.log(f"{len(arms)} arms x {len(seqs)} sequences, budget {args.hours}h "
            f"(deadline {datetime.fromtimestamp(deadline).strftime('%H:%M:%S')} + aggregation)")

    try:
        for arm in arms:
            timeout_s = SYNTH_TIMEOUT_HEAVY_S if arm.get("heavy") else SYNTH_TIMEOUT_S
            for seq in seqs:
                # --- synthesis unit ---
                summary = root / "synthesis" / arm["name"] / seq / "summary.json"
                unit = {"phase": "synth", "arm": arm["name"], "seq": seq}
                if summary.exists():
                    drv.log(f"[{arm['name']}/{seq}] synthesis exists, skipping")
                    drv.status(**unit, action="skipped_resume")
                elif drv.time_left() < 600:
                    drv.log(f"[{arm['name']}/{seq}] deadline reached, skipping synthesis")
                    drv.status(**unit, action="skipped_deadline")
                else:
                    drv.log(f"[{arm['name']}/{seq}] synthesis ...")
                    rc, secs = drv.run_unit(
                        synth_cmd(arm, seq, root, extra_synth),
                        root / "logs" / f"{arm['name']}__{seq}.synth.log",
                        timeout_s,
                    )
                    action = "timeout" if rc is None else ("done" if rc == 0 else "failed")
                    drv.log(f"[{arm['name']}/{seq}] synthesis {action} rc={rc} ({secs:.0f}s)")
                    drv.status(**unit, action=action, rc=rc, secs=round(secs, 1))

            if args.skip_sims:
                continue
            for seq in seqs:
                # --- traj-gen + Isaac lift sim unit (needs that seq's synthesis) ---
                summary = root / "synthesis" / arm["name"] / seq / "summary.json"
                report = root / "traj" / arm["name"] / seq / "isaac_sim" / "report.json"
                unit = {"phase": "sim", "arm": arm["name"], "seq": seq}
                if report.exists():
                    drv.log(f"[{arm['name']}/{seq}] sim report exists, skipping")
                    drv.status(**unit, action="skipped_resume")
                elif not summary.exists():
                    drv.log(f"[{arm['name']}/{seq}] no synthesis output, skipping sim")
                    drv.status(**unit, action="skipped_no_synth")
                elif drv.time_left() < 600:
                    drv.log(f"[{arm['name']}/{seq}] deadline reached, skipping sim")
                    drv.status(**unit, action="skipped_deadline")
                elif not isaac_server_up():
                    drv.log(f"[{arm['name']}/{seq}] Isaac server {CONTROL_HOST}:{CONTROL_PORT} "
                            "unreachable, skipping sim")
                    drv.status(**unit, action="skipped_server_down")
                else:
                    drv.log(f"[{arm['name']}/{seq}] traj-gen + lift sim ...")
                    rc, secs = drv.run_unit(
                        traj_cmd(arm, seq, root),
                        root / "logs" / f"{arm['name']}__{seq}.traj.log",
                        SIM_TIMEOUT_S,
                    )
                    action = "timeout" if rc is None else ("done" if rc == 0 else "failed")
                    drv.log(f"[{arm['name']}/{seq}] sim {action} rc={rc} ({secs:.0f}s)")
                    drv.status(**unit, action=action, rc=rc, secs=round(secs, 1))
    except KeyboardInterrupt:
        drv.log("interrupted -- aggregating what exists, then exiting")

    aggregate(root, arms, seqs, drv)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _record_errors(seq_dir: Path) -> list[dict]:
    out = []
    for p in sorted(seq_dir.glob("grasp_*.json")) + sorted(seq_dir.glob("failed_grasp_*.json")):
        try:
            rec = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if "grasp_error_max" in rec:
            out.append(
                {
                    "grasp_error_max": rec["grasp_error_max"],
                    "dist_error": rec.get("dist_error"),
                    "rank": rec.get("rank"),
                    "ok": rec.get("ok", False),
                }
            )
    return out


def aggregate(root: Path, arms: list[dict], seqs: list[str], drv: Driver | None = None) -> None:
    rows = []
    for arm in arms:
        for seq in seqs:
            row: dict = {"arm": arm["name"], "seq": seq}
            summary_path = root / "synthesis" / arm["name"] / seq / "summary.json"
            if summary_path.exists():
                try:
                    summary = json.loads(summary_path.read_text())
                except (json.JSONDecodeError, OSError):
                    summary = {}
                ms = summary.get("metric_summary") or {}
                recs = _record_errors(summary_path.parent)
                errs = [r["grasp_error_max"] for r in recs]
                dists = [r["dist_error"] for r in recs if r["dist_error"] is not None]
                rank0 = next((r for r in recs if r.get("rank") == 0), None)
                row.update(
                    synth_done=True,
                    strict_success=summary.get("ok", False),
                    strict_success_count=ms.get("strict_success_count"),
                    n_records=len(recs),
                    best_grasp_error=min(errs) if errs else None,
                    median_grasp_error=statistics.median(errs) if errs else None,
                    best_dist_error=min(dists) if dists else None,
                    rank0_grasp_error=rank0["grasp_error_max"] if rank0 else None,
                    rank0_dist_error=rank0["dist_error"] if rank0 else None,
                )
            else:
                row["synth_done"] = False
            report_path = root / "traj" / arm["name"] / seq / "isaac_sim" / "report.json"
            if report_path.exists():
                try:
                    rep = json.loads(report_path.read_text())
                except (json.JSONDecodeError, OSError):
                    rep = {}
                met = rep.get("metrics") or {}
                row.update(
                    sim_done=True,
                    lifted=met.get("lifted"),
                    sustained_lift=met.get("sustained_lift"),
                    grasp_success=met.get("grasp_success"),
                    max_lift_m=met.get("max_lift_m"),
                    lifted_carry_fraction=met.get("lifted_carry_fraction"),
                    final_obj_pos_err_m=met.get("final_object_position_error_m"),
                    max_contact_drive_err_rad=rep.get("max_contact_drive_target_error_rad"),
                )
            else:
                row["sim_done"] = False
            rows.append(row)

    columns = [
        "arm", "seq", "synth_done", "strict_success", "strict_success_count", "n_records",
        "best_grasp_error", "median_grasp_error", "best_dist_error",
        "rank0_grasp_error", "rank0_dist_error",
        "sim_done", "lifted", "sustained_lift", "grasp_success", "max_lift_m",
        "lifted_carry_fraction", "final_obj_pos_err_m", "max_contact_drive_err_rad",
    ]
    with (root / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})

    def fmt(v, nd=4):
        if v is None or v == "":
            return "-"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, float):
            return f"{v:.{nd}f}"
        return str(v)

    lines = [
        f"# Sweep results -- {root.name}",
        "",
        f"Aggregated {datetime.now().isoformat(timespec='seconds')}. "
        "Strict success gate: grasp_error_max <= 0.001 AND dist_error <= 0.01.",
        "",
        "## Per-arm rollup",
        "",
        "| arm | synth | strict succ | best grasp_err (per seq) | lifts | sustained | max lift m (per seq) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm["name"]]
        done = sum(1 for r in arm_rows if r.get("synth_done"))
        strict = sum(r.get("strict_success_count") or 0 for r in arm_rows)
        best = " / ".join(fmt(r.get("best_grasp_error")) for r in arm_rows)
        lifts = sum(1 for r in arm_rows if r.get("lifted"))
        sustained = sum(1 for r in arm_rows if r.get("sustained_lift"))
        lift_m = " / ".join(fmt(r.get("max_lift_m"), 3) for r in arm_rows)
        lines.append(
            f"| {arm['name']} | {done}/{len(arm_rows)} | {strict} | {best} "
            f"| {lifts} | {sustained} | {lift_m} |"
        )
    lines += [
        "",
        "## Per-run detail",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c)) for c in columns) + " |")
    (root / "results.md").write_text("\n".join(lines) + "\n")

    msg = f"aggregated {len(rows)} rows -> {root / 'results.csv'} and results.md"
    if drv is not None:
        drv.log(msg)
    else:
        print(msg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hours", type=float, default=12.0, help="Total wall-clock budget.")
    parser.add_argument("--name", default=None,
                        help="Sweep name (= output folder). Re-use a previous name to resume it. "
                        "Default: anchor_sweep_<timestamp> (smoke_<timestamp> with --smoke).")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--arms", nargs="*", default=None,
                        help="Substring filter on arm names (e.g. --arms fc1000 pure).")
    parser.add_argument("--smoke", action="store_true",
                        help="Plumbing test: first 2 arms, 1 sequence, 4 seeds / 150 iters, "
                        "sims + aggregation included.")
    parser.add_argument("--skip-sims", action="store_true",
                        help="Synthesis only; sims can be filled in later by re-running "
                        "with the same --name without this flag.")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="Only (re)build results.csv/results.md for an existing sweep "
                        "(requires --name).")
    args = parser.parse_args()

    if args.aggregate_only:
        if not args.name:
            parser.error("--aggregate-only requires --name")
        root = args.out_root / args.name
        if not root.exists():
            parser.error(f"no such sweep root: {root}")
        arms = [a for a in ARMS if not args.arms or any(s in a["name"] for s in args.arms)]
        aggregate(root, arms, SEQUENCES if not args.smoke else SEQUENCES[:1])
        return 0

    name = args.name or (
        ("smoke_" if args.smoke else "anchor_sweep_") + datetime.now().strftime("%Y%m%d_%H%M")
    )
    root = args.out_root / name
    root.mkdir(parents=True, exist_ok=True)
    run_sweep(args, root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
