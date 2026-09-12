"""Elbow-task weight sweep (no Isaac, CPU only).

User acceptance criterion (2026-07-27): wrist position AND orientation are
hard requirements; the elbow is a best-effort soft pull.  The Isaac A/B showed
elbow_cost=1.0 costs the wrist +30mm overhead / +6deg roll.  This sweep finds
the largest elbow_cost whose wrist impact is negligible.

Elbow target mimics the mock synthesis: midpoint(shoulder, wrist target),
which is exactly the conflicting pull the Isaac demo produced.

Run from MagicDexMate/sim with .venv-isaac python.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import pinocchio as pin  # noqa: E402

from pink_vega_ik import EE_FRAME, PinkVegaIK  # noqa: E402

SHOULDER_FRAME = {"right": "vega_1_R_arm_l1", "left": "vega_1_L_arm_l1"}
DELTAS = {
    "fwd 25cm": np.array([0.25, 0.0, 0.0]),
    "lat 30cm": np.array([0.0, -0.30, 0.0]),
    "up 30cm": np.array([0.0, 0.0, 0.30]),
    "roll 45d": None,  # position hold, orientation rolled 45deg about chest x
}
WEIGHTS = [0.0, 0.1, 0.25, 0.5, 1.0]
TICKS = 180  # 3s @ 60Hz = demo hold length; the regime the user scores in


def quat_wxyz(R):
    q = pin.Quaternion(R)
    return [q.w, q.x, q.y, q.z]


def ori_err_deg(R_cur, R_tgt):
    cosang = (np.trace(R_cur.T @ R_tgt) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))


def frame_pos(ik, name):
    fid = ik.model.getFrameId(name)
    return ik.data.oMf[fid].translation.copy()


def run_case(label, d_r, w):
    ik = PinkVegaIK(dt=1 / 60, orientation_cost=0.5, elbow_cost=w)
    pin.framesForwardKinematics(ik.model, ik.data, ik.config.q)
    pin.updateFramePlacements(ik.model, ik.data)
    chest = ik._chest
    cp, cq = chest.translation.copy(), quat_wxyz(chest.rotation)
    tgt = {}
    for h, sgn in (("right", 1.0), ("left", -1.0)):
        T0 = ik.data.oMf[ik.model.getFrameId(EE_FRAME[h])]
        p0, R0 = T0.translation.copy(), T0.rotation.copy()
        if d_r is None:  # roll case
            tp = p0
            ax = chest.rotation @ np.array([1.0, 0.0, 0.0])
            tR = pin.exp3(np.radians(45.0) * sgn * ax) @ R0
        else:
            d = d_r.copy()
            d[1] *= sgn
            tp = p0 + chest.rotation @ d
            tR = R0
        ik.set_target_chest(h, cp, cq, tp, quat_wxyz(tR))
        sh = frame_pos(ik, SHOULDER_FRAME[h])
        el_tgt = (sh + tp) / 2.0
        if w > 0.0:
            ik.set_elbow_target_chest(h, cp, cq, el_tgt)
        tgt[h] = (tp, tR, el_tgt)
    for _ in range(TICKS):
        ik.solve()
    pin.framesForwardKinematics(ik.model, ik.data, ik.config.q)
    pin.updateFramePlacements(ik.model, ik.data)
    out = {}
    for h in ("right", "left"):
        tp, tR, el_tgt = tgt[h]
        fid = ik.model.getFrameId(EE_FRAME[h])
        cur = ik.data.oMf[fid]
        from pink_vega_ik import ELBOW_FRAME
        el_err = np.linalg.norm(frame_pos(ik, ELBOW_FRAME[h]) - el_tgt) * 1000
        out[h] = (np.linalg.norm(cur.translation - tp) * 1000,
                  ori_err_deg(cur.rotation, tR), el_err)
    return out


def main():
    hdr = "  ".join(f"w={w:<4}" for w in WEIGHTS)
    print(f"wrist err @{TICKS} ticks (Rpos_mm/Rori_deg | Lpos/Lori) per weight:\n{'':>10s}  {hdr}")
    for label, d_r in DELTAS.items():
        cells, ecells = [], []
        for w in WEIGHTS:
            r = run_case(label, d_r, w)
            cells.append(f"{r['right'][0]:5.1f}/{r['right'][1]:4.1f}|{r['left'][0]:5.1f}/{r['left'][1]:4.1f}")
            ecells.append(f"R{r['right'][2]:5.1f} L{r['left'][2]:5.1f}")
        print(f"  {label:>9s}  " + "  ".join(cells))
        print(f"  {'elbow_mm':>9s}  " + "  ".join(ecells))


if __name__ == "__main__":
    main()
