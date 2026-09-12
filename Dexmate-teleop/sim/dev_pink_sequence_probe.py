"""Sequence probe: does target-transition history change Pink's converged
error, and does the elbow task alter it?  (no Isaac, CPU only)

The Isaac demo shows elbow_cost=0.25 costs overhead +27mm while static CPU
analysis says 0.4mm.  If the pure-Pink sequence reproduces the gap, the path
effect lives in Pink; if not, it lives in the Isaac coupling (open-loop config
drift + PD execution).

Run from MagicDexMate/sim with .venv-isaac python.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import pinocchio as pin  # noqa: E402

from pink_vega_ik import EE_FRAME, ELBOW_FRAME, PinkVegaIK  # noqa: E402

SHOULDER_FRAME = {"right": "vega_1_R_arm_l1", "left": "vega_1_L_arm_l1"}
HOLD = 180  # 3s @ 60Hz, matching the demo hold


def quat_wxyz(R):
    q = pin.Quaternion(R)
    return [q.w, q.x, q.y, q.z]


def ori_err_deg(R_cur, R_tgt):
    c = (np.trace(R_cur.T @ R_tgt) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def frame_pos(ik, name):
    return ik.data.oMf[ik.model.getFrameId(name)].translation.copy()


def refresh(ik):
    pin.framesForwardKinematics(ik.model, ik.data, ik.config.q)
    pin.updateFramePlacements(ik.model, ik.data)


def run_sequence(elbow_w):
    ik = PinkVegaIK(dt=1 / 60, orientation_cost=0.5, elbow_cost=elbow_w)
    refresh(ik)
    chest = ik._chest
    cp, cq = chest.translation.copy(), quat_wxyz(chest.rotation)
    base = {}
    for h in ("right", "left"):
        T0 = ik.data.oMf[ik.model.getFrameId(EE_FRAME[h])]
        base[h] = (T0.translation.copy(), T0.rotation.copy(),
                   frame_pos(ik, SHOULDER_FRAME[h]))

    def seg_targets(name):
        out = {}
        for h, sgn in (("right", 1.0), ("left", -1.0)):
            p0, R0, sh = base[h]
            tp, tR = p0.copy(), R0.copy()
            if name == "fwd":
                tp = p0 + chest.rotation @ np.array([0.25, 0.0, 0.0])
            elif name == "lat":
                tp = p0 + chest.rotation @ np.array([0.0, -0.30 * sgn, 0.0])
            elif name == "up":
                tp = p0 + chest.rotation @ np.array([0.0, 0.0, 0.30])
            elif name == "roll":
                ax = chest.rotation @ np.array([1.0, 0.0, 0.0])
                tR = pin.exp3(np.radians(45.0) * sgn * ax) @ R0
            elif name == "pitch":
                ax = chest.rotation @ np.array([0.0, 1.0, 0.0])
                tR = pin.exp3(np.radians(45.0) * ax) @ R0
            out[h] = (tp, tR, sh)
        return out

    seq = ["fwd", "home", "lat", "home", "up", "home", "roll", "home", "pitch"]
    results = {}
    for name in seq:
        tgts = seg_targets("home" if name == "home" else name)
        for h in ("right", "left"):
            tp, tR, sh = tgts[h]
            ik.set_target_chest(h, cp, cq, tp, quat_wxyz(tR))
            if elbow_w > 0.0:
                ik.set_elbow_target_chest(h, cp, cq, (sh + tp) / 2.0)
        for _ in range(HOLD):
            ik.solve()
        refresh(ik)
        errs = []
        for h in ("right", "left"):
            tp, tR, _ = tgts[h]
            cur = ik.data.oMf[ik.model.getFrameId(EE_FRAME[h])]
            errs.append((np.linalg.norm(cur.translation - tp) * 1000,
                         ori_err_deg(cur.rotation, tR)))
        if name != "home":
            results[name] = errs
    return results


def main():
    for w in (0.0, 0.25):
        r = run_sequence(w)
        print(f"elbow_w={w}: " + "  ".join(
            f"{k} R{v[0][0]:5.1f}mm/{v[0][1]:4.1f}d L{v[1][0]:5.1f}/{v[1][1]:4.1f}"
            for k, v in r.items()))


if __name__ == "__main__":
    main()
