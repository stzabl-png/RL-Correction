"""Standalone Pink convergence probe for the demo targets (no Isaac).

For each parameter set and each demo delta, drives PinkVegaIK open-loop from
home and reports the remaining EE error after N solve() calls (60Hz ticks).
Distinguishes "converges too slowly" from "static cost-tradeoff undershoot".
Run from MagicDexMate/sim with .venv-isaac python.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import pinocchio as pin  # noqa: E402

from pink_vega_ik import EE_FRAME, PinkVegaIK  # noqa: E402

DELTAS = {  # chest-frame deltas, right hand; left mirrors y
    "fwd 25cm": np.array([0.25, 0.0, 0.0]),
    "lat 30cm": np.array([0.0, -0.30, 0.0]),
    "up 30cm": np.array([0.0, 0.0, 0.30]),
}
TICKS = [60, 180, 600]          # 1s / 3s / 10s at 60Hz


def quat_wxyz(R):
    q = pin.Quaternion(R)
    return [q.w, q.x, q.y, q.z]


def run(params):
    print(f"\n=== {params} ===")
    for label, d_r in DELTAS.items():
        ik = PinkVegaIK(dt=1 / 60, **params)
        pin.framesForwardKinematics(ik.model, ik.data, ik.config.q)
        pin.updateFramePlacements(ik.model, ik.data)
        chest = ik._chest
        cp, cq = chest.translation, quat_wxyz(chest.rotation)
        tgt = {}
        for h, sgn in (("right", 1.0), ("left", -1.0)):
            T0 = ik.data.oMf[ik.model.getFrameId(EE_FRAME[h])]
            d = d_r.copy()
            d[1] *= sgn
            tp = T0.translation + chest.rotation @ d
            tq = quat_wxyz(T0.rotation)
            ik.set_target_chest(h, cp, cq, tp, tq)
            tgt[h] = tp
        line = [f"{label:>9s}"]
        done = 0
        for n in TICKS:
            while done < n:
                ik.solve()
                done += 1
            errs = []
            for h in ("right", "left"):
                pin.framesForwardKinematics(ik.model, ik.data, ik.config.q)
                pin.updateFramePlacements(ik.model, ik.data)
                e = np.linalg.norm(ik.ee_pos(h) - tgt[h]) * 1000
                errs.append(e)
            line.append(f"@{n:4d}t R{errs[0]:6.1f} L{errs[1]:6.1f}mm")
        print("  " + "  ".join(line))


run({})  # defaults: pos8 / ori2 / lm10 / posture1e-2
run({"lm_damping": 1.0})
run({"lm_damping": 0.1})
run({"orientation_cost": 0.5})
run({"lm_damping": 0.1, "orientation_cost": 0.5})
run({"lm_damping": 0.1, "posture_cost": 1e-3})
run({"lm_damping": 0.1, "orientation_cost": 0.5, "posture_cost": 1e-3})
