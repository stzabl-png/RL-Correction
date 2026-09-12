#!/usr/bin/env python3
"""Do Isaac and the Viser bench produce the SAME motion? Per joint, per frame.

The operator's requirement, 2026-08-02:

    "make sure the motion in Isaac - which is what will be sent to the real
     robot - and the motion in Viser are exactly the same"

It matters because the bench is where mapping parameters get chosen by eye. If
the two disagree, every visual judgement made on the bench is worthless. This
has already bitten twice in one day: the bench and an offline harness were fed
different WRIST CHANNELS and reported 109 mm and 17 mm for the same mapping law,
and --waist-bind-up shipped with a help string saying 0.20 while the code said
0.08.

Compared in JOINT space, not pictures: joints are what reaches the robot, and
two renderings can look alike while the commands differ (or the reverse - the
same joints at a different camera angle look nothing alike).

Three quantities, and they answer different questions:

    cmd vs cmd    the MAPPING. Must match. Any difference here is two
                  implementations of one law, i.e. a bug in one of them.
    cmd vs got    Isaac's PD lag. Expected to be small but non-zero; the bench
                  has no physics so it cannot show this at all.
    EE            the same difference in millimetres at the hand, via FK,
                  because 2 deg at the shoulder is not 2 deg at the fingertip.

Produce the two files on the SAME replay:

    .venv-isaac/bin/python sim/teleop_vega_pico.py --source replay \\
        --replay-file logs/clip_headwaist.msgpack --mode trackers \\
        --tracker-left LWRIST --tracker-right RWRIST --tracker-waist WAIST \\
        --map waist-abs --ori-mode palm --control-mode arms+head+waist \\
        --headless --duration 30 --joint-csv /tmp/isaac.csv

    .venv/bin/python scripts/viser_mapping_preview.py \\
        logs/clip_headwaist.msgpack --law waist-abs --control-mode \\
        arms+head+waist --headless-seconds 30 --joint-csv /tmp/viser.csv

    .venv/bin/python sim/dev_compare_isaac_viser.py /tmp/isaac.csv /tmp/viser.csv
"""
from __future__ import annotations

import argparse
import csv
import pathlib
import sys

import numpy as np
import pinocchio as pin

URDF = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
EE = {"right": "R_ee", "left": "L_ee"}


def read(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    head, body = rows[0], rows[1:]
    t = np.array([float(r[0]) for r in body])
    cols = {}
    for i, name in enumerate(head[1:], start=1):
        vals = np.array([float(r[i]) if r[i] != "" else np.nan for r in body])
        cols[name] = vals
    return t, cols


def ee_of(model, data, q_arm: dict, hand: str):
    q = pin.neutral(model)
    for n, v in q_arm.items():
        if model.existJointName(n):
            q[model.joints[model.getJointId(n)].idx_q] = v
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    return data.oMf[model.getFrameId(EE[hand])].translation.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("isaac_csv", type=pathlib.Path)
    ap.add_argument("viser_csv", type=pathlib.Path)
    ap.add_argument("--tol-deg", type=float, default=0.5,
                    help="cmd-vs-cmd difference above this is a mapping bug")
    args = ap.parse_args()

    ti, ci = read(args.isaac_csv)
    tv, cv = read(args.viser_csv)
    joints = [n[4:] for n in ci if n.startswith("cmd_")]
    shared = [j for j in joints if f"cmd_{j}" in cv]
    print(f"[data] isaac {len(ti)} rows, viser {len(tv)} rows, "
          f"{len(shared)}/{len(joints)} joints in common")
    if not shared:
        raise SystemExit("no shared joints - were both files written by the "
                         "same version?")

    # Resample the bench onto Isaac's timebase. They run at different rates and
    # neither is the master; nearest-sample is enough because we are comparing
    # a command that changes slowly against the control period.
    idx = np.clip(np.searchsorted(tv, ti), 0, len(tv) - 1)
    ok = np.abs(tv[idx] - ti) < 0.05
    print(f"[align] {ok.sum()}/{len(ti)} Isaac rows have a bench sample within "
          f"50 ms" + ("" if ok.sum() > 0.5 * len(ti) else
                      "   <-- POOR OVERLAP, the rest of this is not evidence"))
    if ok.sum() < 10:
        raise SystemExit("not enough overlap to compare")

    model = pin.buildModelFromUrdf(str(URDF))
    data = model.createData()

    print(f"\n{'joint':>12} {'cmd vs cmd':>22} {'cmd vs got (Isaac PD)':>24}")
    print(f"{'':>12} {'med':>8} {'p95':>8} {'':>4} {'med':>8} {'p95':>8}")
    worst = []
    for j in shared:
        d_map = np.degrees(ci[f"cmd_{j}"][ok] - cv[f"cmd_{j}"][idx][ok])
        row = f"{j:>12} {np.median(np.abs(d_map)):8.2f} " \
              f"{np.percentile(np.abs(d_map), 95):8.2f}"
        got = ci.get(f"got_{j}")
        if got is not None and not np.all(np.isnan(got)):
            d_pd = np.degrees(ci[f"cmd_{j}"][ok] - got[ok])
            d_pd = d_pd[~np.isnan(d_pd)]
            row += f"     {np.median(np.abs(d_pd)):8.2f} " \
                   f"{np.percentile(np.abs(d_pd), 95):8.2f}"
        flag = "   <-- MAPPING DIFFERS" \
            if np.median(np.abs(d_map)) > args.tol_deg else ""
        print(row + flag)
        worst.append((float(np.median(np.abs(d_map))), j))

    # the same disagreement, in millimetres at the hand
    print("\n[EE] mapping difference at the end-effector (FK on both commands)")
    for hand in ("right", "left"):
        d = []
        for k in np.where(ok)[0][::max(1, int(ok.sum() // 400))]:
            qi = {j: ci[f"cmd_{j}"][k] for j in shared}
            qv = {j: cv[f"cmd_{j}"][idx[k]] for j in shared}
            d.append(np.linalg.norm(ee_of(model, data, qi, hand)
                                    - ee_of(model, data, qv, hand)) * 1000)
        d = np.asarray(d)
        print(f"   {hand:>5}: med {np.median(d):7.1f}  p95 "
              f"{np.percentile(d, 95):7.1f} mm  (n={len(d)})")

    worst.sort(reverse=True)
    bad = [w for w in worst if w[0] > args.tol_deg]
    print()
    if bad:
        print(f"[verdict] {len(bad)} joint(s) disagree by more than "
              f"{args.tol_deg} deg. Worst: "
              + ", ".join(f"{j} {v:.1f}deg" for v, j in bad[:5]))
        print("[verdict] the two are NOT running the same mapping.")
        return 1
    print(f"[verdict] every joint agrees to within {args.tol_deg} deg - the "
          f"bench is showing the motion the consumer sends.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
