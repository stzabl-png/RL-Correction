"""Phase-2 closed-loop mapping evaluation on the 2026-07-29 pose capture:
feed the measured human wrist-rel-waist vectors (phase-1 medians) through the
waist-abs mapping into Pink IK and measure, per labeled pose, how well the
robot can realize it - sweeping the bind height h and applying the yaw-bias
correction the data itself revealed (~4-7 deg waist tracker mounting yaw).

Run from MagicDexMate/sim: ../.venv-isaac/bin/python dev_map_pose_capture.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
import pinocchio as pin  # noqa: E402

from pink_vega_ik import EE_FRAME, PinkVegaIK  # noqa: E402

DATA = os.path.join("..", "logs", "pose_capture_20260729",
                    "phase1_body_relative.json")
SKIP = {"raise_high"}          # occlusion-corrupted (audit)
YAW_FIT_POSES = ["front_horizontal", "front_wrist_roll", "hand_raise"]
H_SWEEP = [0.0, 0.10, 0.15, 0.20, 0.25]
TICKS = 240
WAIST_BELOW_CHEST = 0.25       # deployed bind convention (consumer default)


def frame_T(ik, name):
    return ik.data.oMf[ik.model.getFrameId(name)]


def refresh(ik):
    pin.framesForwardKinematics(ik.model, ik.data, ik.config.q)
    pin.updateFramePlacements(ik.model, ik.data)


def anat_axes(ik):
    refresh(ik)
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
    return chest, fwd, left, up


def fit_yaw(res):
    """Yaw bias of the waist reference: mirror-pose sums should have zero
    'left' component; solve the rotation that zeroes them (least squares)."""
    num = den = 0.0
    for slug in YAW_FIT_POSES:
        r = res.get(slug)
        if not r or "RWRIST" not in r or "LWRIST" not in r:
            continue
        sx = r["RWRIST"]["front"] + r["LWRIST"]["front"]
        sy = r["RWRIST"]["left"] + r["LWRIST"]["left"]
        num += sx * sy
        den += sx * sx - sy * sy
    return 0.5 * np.arctan2(2 * num, den) if den else 0.0


def main():
    res = json.load(open(DATA))
    yaw = fit_yaw(res)
    print(f"[yaw-fit] 腰参考偏航修正 = {np.degrees(yaw):+.1f}° "
          f"(由镜像姿势对称性反解)")
    c, s = np.cos(-yaw), np.sin(-yaw)

    ik0 = PinkVegaIK(dt=1 / 60)
    chest, fwd, left, up = anat_axes(ik0)

    poses = [p for p in res if p not in SKIP]
    for h in H_SWEEP:
        bind = chest + (-WAIST_BELOW_CHEST + h) * up
        errs = []
        rows = []
        for slug in poses:
            for hand, key in (("right", "RWRIST"), ("left", "LWRIST")):
                m = res[slug].get(key)
                if m is None:
                    continue
                fx, fy, fz = m["front"], m["left"], m["up"]
                fx, fy = c * fx - s * fy, s * fx + c * fy   # yaw correction
                tgt = bind + fx * fwd + fy * left + fz * up
                ik = PinkVegaIK(dt=1 / 60, orientation_cost=1e-4)
                refresh(ik)
                T0 = frame_T(ik, EE_FRAME[hand])
                q = pin.Quaternion(T0.rotation)
                ik.frame_tasks[hand].set_target(pin.SE3(T0.rotation.copy(), tgt))
                for _ in range(TICKS):
                    ik.solve()
                refresh(ik)
                e = float(np.linalg.norm(ik.ee_pos(hand) - tgt) * 1000)
                errs.append(e)
                rows.append((slug, hand, e))
        errs = np.asarray(errs)
        cov = 100.0 * np.mean(errs <= 25.0)
        print(f"\n== bind h={h:+.2f}: 可达 {cov:.0f}%  err 中位 "
              f"{np.median(errs):.0f}mm  p90 {np.percentile(errs, 90):.0f}mm ==")
        for slug, hand, e in rows:
            if e > 25.0:
                print(f"   miss {slug:<20s} {hand:<5s} {e:6.0f}mm")


if __name__ == "__main__":
    main()
