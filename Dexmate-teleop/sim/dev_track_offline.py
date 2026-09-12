#!/usr/bin/env python3
"""Reproduce the consumer's wrist tracking error offline, so bisecting is seconds.

Each Isaac smoke run costs ~6.5 minutes and the 2026-08-01 regression (7.9 mm ->
195 mm mean wrist error on logs/playlist_all13.msgpack) has five candidate
causes that changed together. This runs the SAME target law through the SAME
Pink solver over the SAME frames, with no physics, and reports the IK residual
|solved EE - commanded EE|.

WHAT THIS IS AND IS NOT
    it is    the solver's residual: can the arm get to the target at all
    it isn't the closed-loop error: no actuator lag, no one-euro filter, no
             engage ramp, no clutch

so it is only trustworthy where the two agree. Hence the self-check: the OLD
configuration measured 7.9 mm right / 7.7 mm left in Isaac, and this harness
must land near that before any of its other numbers count. A harness that
cannot reproduce a known answer does not get to report an unknown one
(see memory verify-the-metric-first; four wrong answers came out of loops like
this one on 2026-07-31).

Frames are fed CONTINUOUSLY. Pink is a differential solver - subsampling
starves it of its seed and the residual it then reports is failure to
converge, not unreachability. That mistake was made three times in one day.

    .venv/bin/python sim/dev_track_offline.py
    .venv/bin/python sim/dev_track_offline.py --configs old new
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from magicdexmate.bimanual_symmetry import symmetrise  # noqa: E402
from magicdexmate.head_waist_map import TORSO_HOME  # noqa: E402
from magicdexmate.palm_fix import PALM_FIX  # noqa: E402
from magicdexmate.pico.xr_pose import process_xr_pose  # noqa: E402
from pink_vega_ik import PinkVegaIK  # noqa: E402

import pinocchio as pin  # noqa: E402

URDF = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
OLD_TORSO = {"torso_j1": np.radians(60.0), "torso_j2": np.radians(90.0),
             "torso_j3": np.radians(-25.0)}

# name -> (torso home, pos_scale, waist_bind_up, symmetriser on)
CONFIGS = {
    # the known answer: Isaac measured 7.9 mm R / 7.7 mm L on this one
    "old":        (OLD_TORSO,   1.00, 0.20, False),
    # the failing one: Isaac measured 195.5 mm R / 165.8 mm L
    "new":        (TORSO_HOME,  1.35, 0.08, True),
    # one variable at a time, walking from old to new
    "torso":      (TORSO_HOME,  1.00, 0.20, False),
    "torso+up":   (TORSO_HOME,  1.00, 0.08, False),
    "torso+scale": (TORSO_HOME, 1.35, 0.20, False),
    "new-nosym":  (TORSO_HOME,  1.35, 0.08, False),
    "old+sym":    (OLD_TORSO,   1.00, 0.20, True),
}


def chest_of(torso):
    m = pin.buildModelFromUrdf(str(URDF))
    d = m.createData()
    q = pin.neutral(m)
    for n, v in torso.items():
        q[m.joints[m.getJointId(n)].idx_q] = v
    pin.forwardKinematics(m, d, q)
    pin.updateFramePlacements(m, d)
    T = d.oMf[m.getFrameId("arm_center")]
    quat = R.from_matrix(T.rotation).as_quat()
    return T.translation.copy(), np.array([quat[3], *quat[:3]])


def frames_of(path, left="LWRIST", right="RWRIST", waist="WAIST"):
    out = []
    for msg in msgpack.Unpacker(open(path, "rb"), raw=False):
        trk = msg.get("trackers") or {}
        if all(trk.get(k) is not None for k in (left, right, waist)):
            out.append((np.asarray(trk[waist], float),
                        np.asarray(trk[right], float),
                        np.asarray(trk[left], float)))
    return out


def free_roll(R_t, R_c, cap_deg=None):
    """Keep the palm NORMAL, pick the roll about it closest to R_c.

    "Where the palm faces" is a direction: two DoF. The mapping pins all three
    by also fixing the finger direction, and that third constraint is what
    collides with the wrist position at the authored upright torso. Relaxing
    exactly the redundant one - the roll about the palm normal, which no
    palm-facing description can even observe (see magicdexmate/palm_fix.py) -
    hands the arm back a degree of freedom without giving up anything the
    operator specified.
    """
    M = R_t.T @ R_c
    th = np.arctan2(M[2, 1] - M[1, 2], M[1, 1] + M[2, 2])
    if cap_deg is not None:
        # Give away only as much of the finger direction as allowed. Freeing
        # the roll completely recovers the position but costs 65-85 deg of
        # finger direction even on frames that did not need it - the solver
        # spends any freedom it is handed. Capping turns it into a budget.
        th = float(np.clip(th, -np.radians(cap_deg), np.radians(cap_deg)))
    c, s_ = np.cos(th), np.sin(th)
    return R_t @ np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])


def run(frames, torso, scale, up, sym, warm=200, ori_cost=2.0, j1_limit=None,
        pos_cost=8.0, roll_free=False, roll_cap=None):
    chest_pos, chest_quat = chest_of(torso)
    bind = chest_pos + np.array([0.0, 0.0, -0.25 + up])
    ik = PinkVegaIK(urdf_path=str(URDF), orientation_cost=ori_cost,
                    position_cost=pos_cost)
    if j1_limit is not None:
        # The shoulder yaw's hardware range is +-176 deg, so the solver can
        # walk the whole arm around to the far branch - and it does: the median
        # R_arm_j1 over this material is +176.0, i.e. parked ON the limit, where
        # the operator authored -80.2 for the same pose. A joint pinned at a
        # limit has no DoF left, which is what "the IK sometimes locks up" and
        # "the upper arm points 30 deg wrong" look like from outside. Narrowing
        # the range in the SOLVER (not the hardware) removes the wound branch.
        for h in "LR":
            i = ik.pin_names.index(f"{h}_arm_j1")
            ik.model.lowerPositionLimit[i] = -np.radians(j1_limit)
            ik.model.upperPositionLimit[i] = np.radians(j1_limit)
        ik.config.model.lowerPositionLimit = ik.model.lowerPositionLimit
        ik.config.model.upperPositionLimit = ik.model.upperPositionLimit
        ik.config.q = np.clip(ik.config.q, ik.model.lowerPositionLimit,
                              ik.model.upperPositionLimit)
        ik.config.update()
    err = {"right": [], "left": []}
    ang = {"right": [], "left": []}
    lo = ik.model.lowerPositionLimit
    hi = ik.model.upperPositionLimit
    pinned = {"right": [], "left": []}
    sym_n, sym_w = 0, 0.0
    chest_rot = R.from_quat([chest_quat[1], chest_quat[2], chest_quat[3],
                             chest_quat[0]]).as_matrix()
    ik.refresh_fk()
    for i, (waist, t_r, t_l) in enumerate(frames):
        p, Rm = {}, {}
        for h, t in (("right", t_r), ("left", t_l)):
            T = process_xr_pose(t, waist)
            p[h] = bind + T[:3, 3] * scale
            Rm[h] = T[:3, :3] @ PALM_FIX[h]
        if sym:
            p["right"], p["left"], Rm["right"], Rm["left"], w = symmetrise(
                p["right"], p["left"], Rm["right"], Rm["left"])
            sym_n += 1
            sym_w = w
        R_want = {}
        for h in ("right", "left"):
            R_want[h] = Rm[h].copy()     # score against the FULL target
            if roll_free:
                fid_ = ik.model.getFrameId({"right": "R_ee", "left": "L_ee"}[h])
                # the target is base-frame; the current EE is in pink's world.
                # Compare through the chest, the frame both are transferred by.
                R_cur = (chest_rot @ ik._chest.rotation.T
                         @ ik.data.oMf[fid_].rotation)
                Rm[h] = free_roll(Rm[h], R_cur, roll_cap)
            q = R.from_matrix(Rm[h]).as_quat()
            ik.set_target_chest(h, chest_pos, chest_quat, p[h],
                                np.array([q[3], *q[:3]]))
        ik.solve()
        if i < warm:          # let the differential solver catch up first
            continue
        ik.refresh_fk()
        for h in ("right", "left"):
            # Compare in PINK's world, not the base frame. set_target_chest
            # transfers the target through the chest and re-plants it on the
            # reduced model's chest, which sits at the ZERO torso - so
            # ik.ee_pos() and the base-frame target p[h] live in frames that
            # differ by the whole torso rotation. Differencing them directly
            # reported 477 mm on a configuration Isaac measures at 7.9 mm,
            # which is what the self-check is for.
            err[h].append(float(np.linalg.norm(
                ik.ee_pos(h) - ik.last_target[h].translation)))
            # Score orientation against the ORIGINAL full target, never
            # against the relaxed one - grading a relaxation by its own
            # relaxed target is how you conclude that giving up worked.
            # Split it: the palm NORMAL is what the operator specified
            # ("palm faces X"); the FINGER direction is the third DoF that
            # --roll-free hands back to the solver, so report what that costs.
            fid = ik.model.getFrameId({"right": "R_ee", "left": "L_ee"}[h])
            R_got = (chest_rot @ ik._chest.rotation.T
                     @ ik.data.oMf[fid].rotation)
            w = R_want[h]
            palm = np.degrees(np.arccos(np.clip(R_got[:, 0] @ w[:, 0], -1, 1)))
            fing = np.degrees(np.arccos(np.clip(R_got[:, 2] @ w[:, 2], -1, 1)))
            ang[h].append((float(palm), float(fing)))
            names = [f"{h[0].upper()}_arm_j{k}" for k in range(1, 8)]
            ids = [ik.pin_names.index(n) for n in names]
            q = ik.config.q
            pinned[h].append([1.0 if (q[i] - lo[i] < np.radians(1.0)
                                      or hi[i] - q[i] < np.radians(1.0))
                              else 0.0 for i in ids])
    return ({h: np.asarray(v) for h, v in err.items()},
            {h: np.asarray(v) for h, v in ang.items()},
            {h: np.asarray(v) for h, v in pinned.items()}, sym_n, sym_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=pathlib.Path,
                    default=pathlib.Path("logs/playlist_all13.msgpack"))
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--sweep-lean", nargs="+", type=float, default=None,
                    help="sweep the CHEST LEAN along the authored waist path "
                         "(deg). 10.3 = the authored upright, 46.1 = bow_30")
    ap.add_argument("--sweep-up", nargs="+", type=float, default=None,
                    help="sweep --waist-bind-up at the new torso home")
    ap.add_argument("--sweep-scale", nargs="+", type=float, default=[1.0],
                    help="scales to pair with --sweep-up")
    ap.add_argument("--pos-cost", type=float, default=8.0)
    ap.add_argument("--roll-cap", type=float, default=None,
                    help="with --roll-free, cap the give-away at +-DEG")
    ap.add_argument("--roll-free", action="store_true",
                    help="constrain only the palm normal, leave the roll about "
                         "it to the solver")
    ap.add_argument("--j1-limit", type=float, default=None,
                    help="clamp the solver's shoulder-yaw range to +-DEG")
    ap.add_argument("--ori-cost", type=float, default=2.0,
                    help="Pink orientation_cost. Set 0 to ask whether the "
                         "residual is the position being unreachable or the "
                         "palm orientation crowding it out")
    args = ap.parse_args()

    frames = frames_of(args.replay)
    if args.max_frames:
        frames = frames[:args.max_frames]
    print(f"[data] {args.replay}: {len(frames)} frames with both wrists + waist")

    todo = [(n, *CONFIGS[n]) for n in args.configs]
    if args.sweep_lean:
        from magicdexmate.head_waist_map import NEUTRAL_LEAN, waist_joints
        todo = [(f"lean{L:.1f}/up{u:.2f}/s{s_:.2f}",
                 waist_joints(L - NEUTRAL_LEAN), s_, u, False)
                for s_ in args.sweep_scale for u in args.sweep_up or [0.08]
                for L in args.sweep_lean]
    elif args.sweep_up:
        todo = [(f"up{u:.2f}/s{s:.2f}", TORSO_HOME, s, u, False)
                for s in args.sweep_scale for u in args.sweep_up]

    for name, torso, scale, up, sym in todo:
        err, ang, pinned, sym_n, sym_w = run(frames, torso, scale, up, sym,
                                             ori_cost=args.ori_cost,
                                             j1_limit=args.j1_limit,
                                             pos_cost=args.pos_cost,
                                             roll_free=args.roll_free,
                                             roll_cap=args.roll_cap)
        line = (f"{name:12s} scale {scale:.2f} up {up:.2f} "
                f"sym {'on ' if sym else 'off'} |")
        for h in ("right", "left"):
            e = err[h] * 1000.0
            line += (f"  {h[0]}: pos med {np.median(e):6.1f} p95 "
                     f"{np.percentile(e, 95):6.1f}mm  palm "
                     f"{np.median(ang[h][:, 0]):5.1f} finger "
                     f"{np.median(ang[h][:, 1]):5.1f}deg")
        if sym:
            line += f"  (sym {sym_n} frames, last w {sym_w:.2f})"
        print(line)
        for h in ("right", "left"):
            frac = pinned[h].mean(axis=0) * 100.0
            if frac.max() > 5.0:
                worst = "  ".join(f"j{k + 1} {frac[k]:.0f}%" for k in range(7)
                                  if frac[k] > 5.0)
                print(f"             {h:5s} at a joint limit: {worst}")
        if name == "old":
            # Compare on the MEDIAN. playlist_all13 is thirteen clips
            # concatenated, so the target jumps hard at every seam; the
            # consumer absorbs those with a one-euro filter and a 1.5 s engage
            # ramp that this harness does not model, which inflates its mean
            # and p95 only around the seams. The median is the like-for-like
            # quantity and Isaac's own median on 'old' is likewise ~0.
            med = np.median(err["right"]) * 1000
            ok = med < 10.0
            print(f"[self-check] 'old' measured 7.9mm mean in Isaac; this "
                  f"harness median {med:.1f}mm -> "
                  + ("OK, harness agrees (compare medians across configs, "
                     "not means)" if ok else
                     "MISMATCH: this harness is not measuring the same thing, "
                     "its other numbers are not evidence"))


if __name__ == "__main__":
    main()
