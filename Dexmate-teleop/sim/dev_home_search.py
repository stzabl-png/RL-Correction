"""Search a better HOME_ARM: all joints well inside limits, EE near the old
home position, and (the scoring criterion) the standalone Pink can actually
execute the demo moves (fwd/lat/up + ori-45) in PHYSICAL axes from it.

Physical axes: Isaac base ~ physical (EE z = hand height checks out). Pin
world is rotated; R_off = R_ee_pin @ R_ee_isaac^-1 at the same q maps
Isaac-base directions into pin world. Run from MagicDexMate/sim (.venv-isaac).
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import pinocchio as pin  # noqa: E402

from pink_vega_ik import EE_FRAME, HOME_ARM, PinkVegaIK  # noqa: E402

# Isaac settled EE pose (right), base frame - from the smoke logs
ISAAC_EE_QUAT_WXYZ = np.array([-0.046, 0.613, 0.723, 0.316])


def rot_from_quat_wxyz(q):
    w, x, y, z = q / np.linalg.norm(q)
    return pin.Quaternion(w, x, y, z).toRotationMatrix()


def fk(ik, q):
    pin.framesForwardKinematics(ik.model, ik.data, q)
    pin.updateFramePlacements(ik.model, ik.data)


def ee_T(ik, hand):
    T = ik.data.oMf[ik.model.getFrameId(EE_FRAME[hand])]
    return pin.SE3(T.rotation.copy(), T.translation.copy())


ik0 = PinkVegaIK(dt=1 / 60)
fk(ik0, ik0.q_home)
T_ee0 = ee_T(ik0, "right")
R_off = T_ee0.rotation @ rot_from_quat_wxyz(ISAAC_EE_QUAT_WXYZ).T
print("R_off @ isaac-x (should match observed (0.143,0,0.204)/0.25 dir):",
      np.round(R_off @ np.array([1.0, 0, 0]), 3))

lo = ik0.model.lowerPositionLimit
hi = ik0.model.upperPositionLimit
names = ik0.pin_names
print("old HOME margins:")
for i, n in enumerate(names):
    q = ik0.q_home[i]
    print(f"  {n}: q={q:+.2f} margin={min(q - lo[i], hi[i] - q):+.2f}")


def probe_home(home_arm, label, deltas_scale=1.0):
    """Return worst-case pink static error [mm] over the demo moves."""
    ik = PinkVegaIK(dt=1 / 60, home_arm=home_arm)
    fk(ik, ik.q_home)
    chest = ik._chest
    cp = chest.translation
    qc = pin.Quaternion(chest.rotation)
    cq = [qc.w, qc.x, qc.y, qc.z]
    moves = {
        "fwd": R_off @ np.array([0.25, 0, 0]) * deltas_scale,
        "latR": R_off @ np.array([0, -0.30, 0]) * deltas_scale,
        "up": R_off @ np.array([0, 0, 0.30]) * deltas_scale,
    }
    worst = 0.0
    detail = []
    for mlabel, d_pin in moves.items():
        ikm = PinkVegaIK(dt=1 / 60, home_arm=home_arm)
        fk(ikm, ikm.q_home)
        tgt = {}
        for h, sgn in (("right", 1.0), ("left", -1.0)):
            T0 = ee_T(ikm, h)
            d = d_pin.copy()
            if mlabel == "latR":
                d = R_off @ np.array([0, -0.30 * sgn, 0]) * deltas_scale
            tp = T0.translation + d
            q0 = pin.Quaternion(T0.rotation)
            ikm.set_target_chest(h, cp, cq, tp, [q0.w, q0.x, q0.y, q0.z])
            tgt[h] = tp
        for _ in range(240):
            ikm.solve()
        fk(ikm, ikm.config.q)
        errs = [np.linalg.norm(ee_T(ikm, h).translation - tgt[h]) * 1000
                for h in ("right", "left")]
        detail.append(f"{mlabel} R{errs[0]:.0f}/L{errs[1]:.0f}")
        worst = max(worst, *errs)
    print(f"{label}: worst {worst:5.0f}mm  ({'  '.join(detail)})")
    return worst


probe_home(HOME_ARM, "old home")

# candidates: unjam the wrists (j6/j7 mid-range), keep an elbow-bent posture
CANDS = {
    "A wrists-zero": {"L_arm_j1": -0.7854, "L_arm_j2": 0.40, "L_arm_j4": -1.40,
                      "R_arm_j1": 0.7854, "R_arm_j2": -0.40, "R_arm_j4": -1.40},
    "B softer": {"L_arm_j1": -0.6, "L_arm_j2": 0.35, "L_arm_j3": 0.3,
                 "L_arm_j4": -1.2, "L_arm_j6": 0.3, "L_arm_j7": -0.2,
                 "R_arm_j1": 0.6, "R_arm_j2": -0.35, "R_arm_j3": -0.3,
                 "R_arm_j4": -1.2, "R_arm_j6": -0.3, "R_arm_j7": 0.2},
    "C mid": {"L_arm_j1": -0.9, "L_arm_j2": 0.5, "L_arm_j4": -1.1,
              "L_arm_j5": -0.3, "R_arm_j1": 0.9, "R_arm_j2": -0.5,
              "R_arm_j4": -1.1, "R_arm_j5": 0.3},
}
for label, cand in CANDS.items():
    ikc = PinkVegaIK(dt=1 / 60, home_arm=cand)
    fk(ikc, ikc.q_home)
    Tr = ee_T(ikc, "right")
    margins = min(min(ikc.q_home[i] - lo[i], hi[i] - ikc.q_home[i])
                  for i in range(len(names)))
    print(f"\n{label}: EE={np.round(Tr.translation, 3)} min_margin={margins:+.2f}")
    probe_home(cand, label)
