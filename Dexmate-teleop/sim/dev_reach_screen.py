#!/usr/bin/env python3
"""Are the waist-abs wrist targets inside the arm's reach? Per config, per frame.

2026-08-01. The 30 s smoke on logs/playlist_all13.msgpack came back with a
195 mm mean wrist tracking error where the same material used to give 7.9 mm.
Four things changed at once (TORSO_HOME, waist lean sign, --pos-scale 1.0->1.35,
--waist-bind-up 0.20->0.08) plus the bimanual symmetriser was wired in. Isaac
runs cost ~6.5 min each, so screen the candidates geometrically first: a target
further from the shoulder than the arm is long CANNOT be tracked, no matter what
the solver does, and that failure shows up as exactly this kind of error.

Reach is measured, not assumed: the arm's max |shoulder -> EE| is sampled over
the URDF joint ranges rather than taken from the 0.771 m quoted in the notes,
because that number was measured at the OLD torso home and the shoulder frame
moves with the chest.

    .venv/bin/python sim/dev_reach_screen.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import msgpack
import numpy as np
import pinocchio as pin

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from magicdexmate.head_waist_map import TORSO_HOME  # noqa: E402
from magicdexmate.pico.xr_pose import process_xr_pose  # noqa: E402

URDF = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
OLD_TORSO = {"torso_j1": np.radians(60.0), "torso_j2": np.radians(90.0),
             "torso_j3": np.radians(-25.0)}
EE = {"right": "R_ee", "left": "L_ee"}
SHOULDER = {"right": "R_arm_j1", "left": "L_arm_j1"}


def build():
    m = pin.buildModelFromUrdf(str(URDF))
    return m, m.createData()


def _q_with(m, torso):
    q = pin.neutral(m)
    for n, v in torso.items():
        q[m.joints[m.getJointId(n)].idx_q] = v
    return q


def chest_and_shoulders(m, d, torso):
    pin.forwardKinematics(m, d, _q_with(m, torso))
    pin.updateFramePlacements(m, d)
    return (d.oMf[m.getFrameId("arm_center")].translation.copy(),
            {h: d.oMf[m.getFrameId(f)].translation.copy()
             for h, f in SHOULDER.items()})


def max_reach(m, d, torso, hand, n=20000, seed=0):
    """Sampled max |shoulder -> EE| over the arm's joint limits, this torso."""
    rng = np.random.default_rng(seed)
    arm = [f"{'R' if hand == 'right' else 'L'}_arm_j{i}" for i in range(1, 8)]
    idx = [m.joints[m.getJointId(a)].idx_q for a in arm]
    lo = np.array([m.lowerPositionLimit[i] for i in idx])
    hi = np.array([m.upperPositionLimit[i] for i in idx])
    q = _q_with(m, torso)
    fe, fs = m.getFrameId(EE[hand]), m.getFrameId(SHOULDER[hand])
    best = 0.0
    for _ in range(n):
        q[idx] = rng.uniform(lo, hi)
        pin.forwardKinematics(m, d, q)
        pin.updateFramePlacements(m, d)
        best = max(best, float(np.linalg.norm(
            d.oMf[fe].translation - d.oMf[fs].translation)))
    return best


def wrists_rel_waist(path, left="LWRIST", right="RWRIST", waist="WAIST"):
    out = {"right": [], "left": []}
    for msg in msgpack.Unpacker(open(path, "rb"), raw=False):
        trk = msg.get("trackers") or {}
        if trk.get(waist) is None:
            continue
        ref = np.asarray(trk[waist], float)
        for h, k in (("right", right), ("left", left)):
            if trk.get(k) is not None:
                out[h].append(process_xr_pose(np.asarray(trk[k], float), ref)[:3, 3])
    return {h: np.asarray(v) for h, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=pathlib.Path,
                    default=pathlib.Path("logs/playlist_all13.msgpack"))
    args = ap.parse_args()

    m, d = build()
    hum = wrists_rel_waist(args.replay)
    print(f"[data] {args.replay}: right {len(hum['right'])} frames, "
          f"left {len(hum['left'])} frames")

    configs = [
        ("OLD torso, scale 1.00, up 0.20", OLD_TORSO, 1.00, 0.20),
        ("NEW torso, scale 1.00, up 0.20", TORSO_HOME, 1.00, 0.20),
        ("NEW torso, scale 1.00, up 0.08", TORSO_HOME, 1.00, 0.08),
        ("NEW torso, scale 1.35, up 0.08", TORSO_HOME, 1.35, 0.08),
        ("NEW torso, scale 1.35, up 0.20", TORSO_HOME, 1.35, 0.20),
    ]
    for label, torso, scale, up in configs:
        chest, sh = chest_and_shoulders(m, d, torso)
        bind = chest + np.array([0.0, 0.0, -0.25 + up])
        print(f"\n--- {label}")
        print(f"    arm_center {np.round(chest, 3)}  bind {np.round(bind, 3)}")
        for h in ("right", "left"):
            reach = max_reach(m, d, torso, h)
            tgt = bind + hum[h] * scale
            dist = np.linalg.norm(tgt - sh[h], axis=1)
            over = dist - reach
            frac = float((over > 0).mean())
            print(f"    {h:5s} reach {reach * 1000:6.1f}mm | "
                  f"|shoulder->target| mean {dist.mean() * 1000:6.1f} "
                  f"max {dist.max() * 1000:6.1f}mm | "
                  f"beyond reach on {frac * 100:5.1f}% of frames, "
                  f"mean excess {over[over > 0].mean() * 1000 if frac else 0.0:6.1f}mm")


if __name__ == "__main__":
    main()
