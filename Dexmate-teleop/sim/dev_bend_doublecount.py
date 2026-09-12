#!/usr/bin/env python3
"""Does bending the waist move the hands when the arms did not move?

A known-answer test, because the metric that was being used cannot fail on this
bug. Wrist tracking error asks "did the hand reach the target it was given" -
and double-counting the operator's bend produces a target that is WRONG but
perfectly reachable, so the hand goes precisely to the wrong place and the
error stays at ~18 mm either way. Measured 2026-08-03: waist-abs 18.0/22.2 mm,
chest-rel 28.8/21.4 mm, on a run where one of the two has a 194 mm defect.

The input here is synthetic and its answer is known: take one real frame and
rotate the whole upper body about the waist, so the operator bends while their
arms hold still RELATIVE TO THEIR CHEST. The robot's hand must then also hold
still relative to ITS chest. Anything else is the defect, in millimetres.

Why waist-abs has one: its target is

    bind(live chest) + (wrist - waist)

and the operator's bend is already inside (wrist - waist), while the bind point
moves again when the robot's chest follows. At the legacy home the bind point
travels 194 mm for 25 deg of bend and 307 mm for 40.

    .venv/bin/python sim/dev_bend_doublecount.py
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import msgpack
import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from magicdexmate.head_waist_map import TORSO_HOME, waist_joints_j3  # noqa: E402
from magicdexmate.palm_fix import PALM_FIX  # noqa: E402
from magicdexmate.pico.xr_pose import process_xr_pose  # noqa: E402
from pink_vega_ik import PinkVegaIK  # noqa: E402

URDF = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
EE = {"right": "R_ee", "left": "L_ee"}
SH = {"left": 16, "right": 17}
WR = {"left": 20, "right": 21}


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
    return (T.translation.copy(), np.array([quat[3], *quat[:3]]),
            T.rotation.copy())


def base_frame(path):
    """One real frame: wrists (4x4, waist-relative) and shoulders (3,)."""
    for msg in msgpack.Unpacker(open(path, "rb"), raw=False):
        trk = msg.get("trackers") or {}
        b24 = msg.get("body24")
        if b24 is None or any(trk.get(k) is None
                              for k in ("RWRIST", "LWRIST", "WAIST")):
            continue
        w = np.asarray(trk["WAIST"], float)
        b24 = np.asarray(b24, float)
        wr = {h: process_xr_pose(np.asarray(trk[k], float), w)
              for h, k in (("right", "RWRIST"), ("left", "LWRIST"))}
        sh = {h: process_xr_pose(b24[SH[h]], w)[:3, 3] for h in SH}
        return wr, sh
    raise SystemExit("no usable frame")


def bend(wr, sh, deg):
    """Rotate the whole upper body about the waist: bending, arms rigid."""
    Rb = R.from_euler("y", np.radians(deg)).as_matrix()
    wr2 = {}
    for h, T in wr.items():
        T2 = np.eye(4)
        T2[:3, :3] = Rb @ T[:3, :3]
        T2[:3, 3] = Rb @ T[:3, 3]
        wr2[h] = T2
    return wr2, {h: Rb @ p for h, p in sh.items()}


def hum_chest(sh):
    org = 0.5 * (sh["left"] + sh["right"])
    y = sh["left"] - sh["right"]
    y = y / np.linalg.norm(y)
    z = org / np.linalg.norm(org)
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)
    return org, np.column_stack([x, y, np.cross(x, y)])


def run(law, wr, sh, leans, scale, up, settle=250):
    """-> {hand: [EE relative to the robot's own chest, per lean]}"""
    ik = PinkVegaIK(urdf_path=str(URDF))
    out = {h: [] for h in EE}
    for lean in leans:
        torso = waist_joints_j3(lean)          # the robot bends the same amount
        cP, cQ, cR = chest_of(torso)
        bind = cP + np.array([0.0, 0.0, -0.25 + up])
        w2, s2 = bend(wr, sh, lean)
        hO, hR = hum_chest(s2)
        for _ in range(settle):
            for h in EE:
                T = w2[h]
                if law == "waist-abs":
                    p = bind + T[:3, 3] * scale
                    Rt = T[:3, :3] @ PALM_FIX[h]
                elif law == "chest-anchor":
                    # origin from the chest, axes from gravity
                    p = cP + (T[:3, 3] - hO) * scale
                    Rt = T[:3, :3] @ PALM_FIX[h]
                else:
                    p = cP + cR @ (hR.T @ (T[:3, 3] - hO)) * scale
                    Rt = cR @ hR.T @ T[:3, :3] @ PALM_FIX[h]
                q4 = R.from_matrix(Rt).as_quat()
                ik.set_target_chest(h, cP, cQ, p, np.array([q4[3], *q4[:3]]))
            ik.solve()
        ik.refresh_fk()
        for h in EE:
            # EE relative to the robot's OWN chest - the quantity that must not
            # change when only the waist bends
            rel = ik._chest.rotation.T @ (
                ik.data.oMf[ik.model.getFrameId(EE[h])].translation
                - ik._chest.translation)
            out[h].append(rel)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=pathlib.Path,
                    default=pathlib.Path("logs/clip_headwaist.msgpack"))
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--bind-up", type=float, default=0.20)
    args = ap.parse_args()

    wr, sh = base_frame(args.replay)
    leans = [0, 10, 20, 30, 40]
    print(f"合成输入:只弯腰,手臂相对胸完全不动。正确答案 = 机器人的手相对"
          f"它自己的胸一动不动(0mm)。\n")
    laws = ("waist-abs", "chest-rel", "chest-anchor")
    print(f"{'弯腰':>6} | " + " | ".join(f"{l+' 右/左':>21}" for l in laws))
    res = {law: run(law, wr, sh, leans, args.scale, args.bind_up)
           for law in laws}
    for i, lean in enumerate(leans):
        cells = []
        for law in laws:
            d = [np.linalg.norm(res[law][h][i] - res[law][h][0]) * 1000
                 for h in ("right", "left")]
            cells.append(f"{d[0]:9.1f} /{d[1]:9.1f}mm")
        print(f"{lean:5d}° | " + " | ".join(f"{c:>21}" for c in cells))
    print("\n(数字 = 手相对机器人自己胸的漂移量。越接近 0 越对。)")


if __name__ == "__main__":
    main()
