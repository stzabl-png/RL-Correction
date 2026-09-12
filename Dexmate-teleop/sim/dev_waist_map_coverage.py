"""Coverage analysis for the user's absolute waist-referenced mapping
(2026-07-28): lock the torso, bind human waist <-> robot waist (torso_j1),
map human wrist-rel-waist onto robot EE-rel-waist at scale s.

For each scale, sample realistic right-wrist positions over the human arm's
envelope and ask Pink (position-only) whether Vega's arm can reach the mapped
target.  Reports coverage % and where the dead zones are.

Run from MagicDexMate/sim with .venv-isaac python.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import pinocchio as pin  # noqa: E402

from pink_vega_ik import EE_FRAME, PinkVegaIK  # noqa: E402

# human geometry in wire frame (x=right, y=up, -z=front), matches the puppet
H_WAIST = np.array([0.0, 1.15, 0.0])
H_SHOULDER_R = np.array([0.18, 1.40, 0.0])
ARM_LEN = 0.54

SCALES = [1.0]
UP_OFFSETS = [0.0, 0.10, 0.15, 0.20, 0.25]  # bind human waist this far above torso_j1
N_SAMPLES = 120
TICKS = 180
OK_MM = 25.0


def frame_T(ik, name):
    return ik.data.oMf[ik.model.getFrameId(name)]


def base_axes_in_pin(ik):
    """Anatomical base axes in pin world: up = trunk axis (waist->chest),
    left = shoulder line (R->L), fwd = left x up."""
    pin.framesForwardKinematics(ik.model, ik.data, ik.config.q)
    pin.updateFramePlacements(ik.model, ik.data)
    waist = frame_T(ik, "torso_j1").translation.copy()
    chest = frame_T(ik, "arm_center").translation.copy()
    shl_l = frame_T(ik, "vega_1_L_arm_l1").translation.copy()
    shl_r = frame_T(ik, "vega_1_R_arm_l1").translation.copy()
    up = chest - waist
    up /= np.linalg.norm(up)
    left = shl_l - shl_r
    left -= (left @ up) * up
    left /= np.linalg.norm(left)
    fwd = np.cross(left, up)
    return waist, fwd, left, up


def sample_wrists(rng):
    """Right-wrist positions over a realistic standing envelope (wire frame):
    front/side hemisphere around the shoulder, hanging-hands included."""
    out = []
    while len(out) < N_SAMPLES:
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        r = rng.uniform(0.25, ARM_LEN)
        p = H_SHOULDER_R + d * r
        if p[2] > 0.10:          # behind the back - skip
            continue
        if p[1] < 0.70:          # below mid-thigh - skip
            continue
        if p[0] < -0.35:         # far across the body - skip
            continue
        out.append(p)
    return np.array(out)


def main():
    rng = np.random.default_rng(0)
    samples = sample_wrists(rng)
    ik0 = PinkVegaIK(dt=1 / 60)
    waist, fwd, left, up = base_axes_in_pin(ik0)
    print(f"robot waist (pin) {np.round(waist, 3)}")

    s = SCALES[0]
    for up_off in UP_OFFSETS:
        anchor = waist + up_off * up
        fails = []
        errs = []
        for p in samples:
            rel = p - H_WAIST  # wire frame
            # wire -> anatomical components: right=x, up=y, front=-z
            r_right, r_up, r_front = rel[0], rel[1], -rel[2]
            tgt = anchor + s * (r_front * fwd + (-r_right) * left + r_up * up)
            ik = PinkVegaIK(dt=1 / 60, orientation_cost=1e-4)
            pin.framesForwardKinematics(ik.model, ik.data, ik.config.q)
            pin.updateFramePlacements(ik.model, ik.data)
            T0 = frame_T(ik, EE_FRAME["right"])
            ik.frame_tasks["right"].set_target(pin.SE3(T0.rotation.copy(), tgt))
            for _ in range(TICKS):
                ik.solve()
            pin.framesForwardKinematics(ik.model, ik.data, ik.config.q)
            pin.updateFramePlacements(ik.model, ik.data)
            e = np.linalg.norm(ik.ee_pos("right") - tgt) * 1000
            errs.append(e)
            if e > OK_MM:
                fails.append((rel, e))
        errs = np.array(errs)
        cov = 100.0 * np.mean(errs <= OK_MM)
        print(f"\nscale {s:.2f} up_off {up_off:+.2f}: coverage {cov:.0f}%  "
              f"err med {np.median(errs):.0f}mm p90 {np.percentile(errs, 90):.0f}mm")
        if fails:
            fr = np.array([f[0] for f in fails])
            fe = np.array([f[1] for f in fails])
            print(f"  {len(fails)} fails, mean miss {fe.mean():.0f}mm; "
                  f"fail centroid rel waist (right/up/front): "
                  f"{fr[:, 0].mean():+.2f}/{fr[:, 1].mean():+.2f}/{-fr[:, 2].mean():+.2f}")
            lo = fr[fr[:, 1] < 0.15]
            hi = fr[fr[:, 1] > 0.45]
            print(f"  fails below chest: {len(lo)}, above chest: {len(hi)}")


if __name__ == "__main__":
    main()
