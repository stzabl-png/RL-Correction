#!/usr/bin/env python3
"""Score a mapping against the operator's 2026-08-02 priorities. Their words:

    "I want to prioritise the palm ORIENTATION, and the relative position of
     the two palms - the two palms' spatial relationship, POSE and ORIENTATION
     both, should ideally be preserved."

which reorders the 2026-07-31 list. Absolute position is still the thing that
may be given away; what moved is the palm orientation, from "sacrifice up to
10 deg" to the top tier, alongside the two-hand relationship.

Three numbers, in that priority order:

  ① palm orientation   per hand, robot palm normal vs the mapped target's.
                       Reported separately from the finger heading, because
                       "where the palm faces" is two DoF and the roll about it
                       is the third - see magicdexmate/palm_fix.py.
  ② relative position  |(p_L - p_R)_robot - (p_L - p_R)_operator * scale|.
                       Zero by construction under waist-abs when both targets
                       are REACHED; it breaks the moment one arm saturates,
                       which is why it has to be measured on the solved pose
                       and not on the targets. Measuring targets instead of
                       solved EEs is the mistake that hid this on 2026-08-01.
  ③ relative orientation   angle between (R_R^-1 R_L) on the robot and the
                       same product on the operator. NEVER MEASURED BEFORE -
                       every previous evaluation looked at relative position
                       only. Two hands on one object care about this.

and, for context only, absolute position - the one the operator is willing to
give away.

SELF-CHECK, and why it refuses to report without one: feed targets that are an
exact mirror pair and ② and ③ must come back 0. A metric that cannot reproduce
a known answer does not get to report an unknown one (memory:
verify-the-metric-first; four wrong answers came out of loops like this on
2026-07-31, and one more - a frame mix-up worth 477 mm - on 2026-08-01).

    .venv/bin/python sim/dev_eval_priorities.py
    .venv/bin/python sim/dev_eval_priorities.py --sweep-ori 0.5 2 8 16 32
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
from magicdexmate.bimanual_symmetry import symmetrise  # noqa: E402
from magicdexmate.head_waist_map import TORSO_HOME  # noqa: E402

# The torso the operator signed off on live, 2026-08-01: "the palm direction is
# right", "no problems". Everything else about that run was also different -
# control-mode arms, pos_scale 1.0, bind_up 0.20, no symmetriser - so keep it
# here to score the whole known-good configuration against the CURRENT
# priorities, rather than arguing from memory about whether it was better.
OLD_TORSO = {"torso_j1": np.radians(60.0), "torso_j2": np.radians(90.0),
             "torso_j3": np.radians(-25.0)}
from magicdexmate.palm_fix import FINGERS_IN_EE, PALM_FIX, PALM_IN_EE  # noqa: E402
from magicdexmate.pico.xr_pose import process_xr_pose  # noqa: E402
from pink_vega_ik import PinkVegaIK  # noqa: E402

URDF = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
EE = {"right": "R_ee", "left": "L_ee"}
MIRROR = np.diag([1.0, -1.0, 1.0])


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))


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
    return T.translation.copy(), np.array([quat[3], *quat[:3]]), T.rotation.copy()


def frames_of(path):
    """Wrists from the `trackers` dict - the channel the consumer consumes."""
    out = []
    for msg in msgpack.Unpacker(open(path, "rb"), raw=False):
        trk = msg.get("trackers") or {}
        if any(trk.get(k) is None for k in ("RWRIST", "LWRIST", "WAIST")):
            continue
        w = np.asarray(trk["WAIST"], float)
        out.append({h: process_xr_pose(np.asarray(trk[k], float), w)
                    for h, k in (("right", "RWRIST"), ("left", "LWRIST"))})
    return out


def run(frames, *, scale, up, sym, pos_cost, ori_cost, warm=200, torso=None,
        substeps=3, gain=1.0, lm_damping=10.0):
    chest_pos, chest_wxyz, chest_R = chest_of(torso or TORSO_HOME)
    bind = chest_pos + np.array([0.0, 0.0, -0.25 + up])
    ik = PinkVegaIK(urdf_path=str(URDF), position_cost=pos_cost,
                    orientation_cost=ori_cost, substeps=substeps, gain=gain,
                    lm_damping=lm_damping)
    out = {k: [] for k in ("palm", "finger", "relpos", "relori", "abspos",
                           "roll_signed", "j7", "j7_pin", "j6_pin")}
    lo, hi = ik.model.lowerPositionLimit, ik.model.upperPositionLimit
    for i, fr in enumerate(frames):
        p, Rm, hum = {}, {}, {}
        for h in ("right", "left"):
            T = fr[h]
            hum[h] = (T[:3, 3].copy(), T[:3, :3].copy())
            p[h] = bind + T[:3, 3] * scale
            Rm[h] = T[:3, :3] @ PALM_FIX[h]
        want = {h: Rm[h].copy() for h in Rm}
        if sym:
            p["right"], p["left"], Rm["right"], Rm["left"], _ = symmetrise(
                p["right"], p["left"], Rm["right"], Rm["left"])
            want = {h: Rm[h].copy() for h in Rm}
        for h in ("right", "left"):
            q4 = R.from_matrix(Rm[h]).as_quat()
            ik.set_target_chest(h, chest_pos, chest_wxyz, p[h],
                                np.array([q4[3], *q4[:3]]))
        ik.solve()
        if i < warm:
            continue
        ik.refresh_fk()
        got_p, got_R = {}, {}
        for h in ("right", "left"):
            T = ik.data.oMf[ik.model.getFrameId(EE[h])]
            got_p[h] = T.translation.copy()
            # BACK TO THE BASE FRAME. set_target_chest re-plants targets on the
            # reduced model's chest, which sits at the ZERO torso, so what comes
            # out of the solver is in a frame rotated from the base by the whole
            # torso. Comparing it directly against a base-frame target is the
            # same mistake that reported 477 mm on 2026-08-01 - and here it
            # produced a 10.0 deg finger error that would not move for ANY
            # solver setting (weights 0.5-32, substeps 3-30, lm_damping
            # 10-0.01), which is exactly what a metric artifact looks like.
            got_R[h] = chest_R @ ik._chest.rotation.T @ T.rotation
        # ① palm orientation, against the FULL target
        for h in ("right", "left"):
            out["palm"].append(ang(got_R[h] @ PALM_IN_EE, want[h] @ PALM_IN_EE))
            out["finger"].append(ang(got_R[h] @ FINGERS_IN_EE,
                                     want[h] @ FINGERS_IN_EE))
            # SIGNED error about the palm normal = the finger-direction error
            # with its sign kept. A consistent sign means a constant offset
            # (PALM_FIX's roll, or a saturated joint always pushed the same
            # way); a symmetric spread around zero means the solver is simply
            # trading it away. |finger| cannot tell those apart.
            dR = got_R[h].T @ want[h]
            out["roll_signed"].append(float(np.degrees(
                pin.log3(dR) @ PALM_IN_EE)))
            for k, key in ((7, "j7_pin"), (6, "j6_pin")):
                i7 = ik.pin_names.index(f"{h[0].upper()}_arm_j{k}")
                q7 = ik.config.q[i7]
                out[key].append(1.0 if (q7 - lo[i7] < np.radians(1.0)
                                        or hi[i7] - q7 < np.radians(1.0)) else 0.0)
                if k == 7:
                    out["j7"].append(float(np.degrees(q7)))
        # ② relative position - solved EEs, not targets
        d_rob = got_p["left"] - got_p["right"]
        d_hum = (hum["left"][0] - hum["right"][0]) * scale
        out["relpos"].append(float(np.linalg.norm(d_rob - d_hum)) * 1000)
        # ③ relative orientation, robot pair vs operator pair
        rel_rob = got_R["right"].T @ got_R["left"]
        rel_hum = (hum["right"][1] @ PALM_FIX["right"]).T \
            @ (hum["left"][1] @ PALM_FIX["left"])
        out["relori"].append(float(np.degrees(np.linalg.norm(
            pin.log3(rel_rob.T @ rel_hum)))))
        # context: the one they are willing to give away
        out["abspos"].append(float(np.mean(
            [np.linalg.norm(got_p[h] - ik.last_target[h].translation)
             for h in ("right", "left")])) * 1000)
    return {k: np.asarray(v) for k, v in out.items()}


def self_check_orientation(**kw):
    """The arm's OWN current pose, fed back as the target, must score 0 deg.

    The mirror check below cannot catch a frame error in (1): it only looks at
    relative quantities, and a frame rotation common to both hands cancels in
    R_right^T R_left. That blind spot let a 10 deg metric artifact be reported
    as a mapping result. Every number that gets printed needs a check that can
    fail on it.
    """
    chest_pos, chest_wxyz, chest_R = chest_of(TORSO_HOME)
    ik = PinkVegaIK(urdf_path=str(URDF))
    ik.refresh_fk()
    worst = 0.0
    for h in ("right", "left"):
        T = ik.data.oMf[ik.model.getFrameId(EE[h])]
        R_base = chest_R @ ik._chest.rotation.T @ T.rotation
        for v in (PALM_IN_EE, FINGERS_IN_EE):
            worst = max(worst, ang(R_base @ v, R_base @ v))
        # and the transfer must round-trip: base target -> chest -> pink world
        q4 = R.from_matrix(R_base).as_quat()
        p_base = chest_pos + chest_R @ (ik._chest.rotation.T
                                        @ (T.translation - ik._chest.translation))
        ik.set_target_chest(h, chest_pos, chest_wxyz, p_base,
                            np.array([q4[3], *q4[:3]]))
        tw = ik.last_target[h]
        worst = max(worst, float(np.degrees(np.linalg.norm(
            pin.log3(tw.rotation.T @ T.rotation)))))
        worst = max(worst, float(np.linalg.norm(
            tw.translation - T.translation)) * 1000)
    return worst


def self_check(frames, **kw):
    """Exactly mirrored operator -> relative position and orientation must be 0."""
    mir = []
    for fr in frames[:400]:
        T = fr["right"]
        Tl = np.eye(4)
        Tl[:3, 3] = MIRROR @ T[:3, 3]
        Tl[:3, :3] = MIRROR @ T[:3, :3] @ MIRROR
        mir.append({"right": T, "left": Tl})
    r = run(mir, warm=150, **kw)
    return float(np.median(r["relpos"])), float(np.median(r["relori"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=pathlib.Path,
                    default=pathlib.Path("logs/clip_headwaist.msgpack"))
    ap.add_argument("--max-frames", type=int, default=1500)
    ap.add_argument("--scale", type=float, default=1.35)
    ap.add_argument("--bind-up", type=float, default=0.20)
    ap.add_argument("--sweep-ori", nargs="+", type=float,
                    default=[0.5, 2.0, 8.0, 16.0, 32.0])
    ap.add_argument("--pos-cost", type=float, default=8.0)
    ap.add_argument("--no-sym", action="store_true")
    ap.add_argument("--sweep-substeps", nargs="+", type=int, default=None,
                    help="is the 10 deg finger residual a CONVERGENCE problem? "
                         "More iterations per frame answers it directly.")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--sweep-lm", nargs="+", type=float, default=None,
                    help="Pink's Levenberg-Marquardt damping on the wrist task. "
                         "Ships at 10.0 against a position cost of 8.0 and an "
                         "orientation cost of 2.0 - i.e. the damping is 5x the "
                         "orientation cost. This file's own elbow-task comment "
                         "records that a damping/cost ratio like that makes the "
                         "solve go target-independent; nobody went back to "
                         "check the wrist task, which has the same ratio.")
    ap.add_argument("--old-torso", action="store_true",
                    help="score at the 55 deg torso home the operator signed "
                         "off on live on 2026-08-01")
    args = ap.parse_args()

    frames = frames_of(args.replay)[:args.max_frames]
    print(f"[data] {args.replay.name}: {len(frames)} frames  "
          f"scale={args.scale} bind_up={args.bind_up} "
          f"sym={'off' if args.no_sym else 'on'}")

    base = dict(scale=args.scale, up=args.bind_up, sym=not args.no_sym,
                pos_cost=args.pos_cost,
                torso=OLD_TORSO if args.old_torso else None)
    print(f"[torso] {'OLD 55deg (the 2026-08-01 signed-off one)' if args.old_torso else 'authored 10.3deg'}")
    w1 = self_check_orientation()
    print(f"[self-check] arm's own pose fed back as target -> {w1:.4f} "
          f"(deg / mm)" + ("  OK" if w1 < 0.5 else
                           "  FAILED - the (1) frame transfer is wrong"))
    if w1 >= 0.5:
        return 1
    rp, ro = self_check(frames, ori_cost=2.0, **base)
    ok = rp < 1.0 and ro < 0.5
    print(f"[self-check] exact mirror -> relpos {rp:.3f} mm, relori {ro:.3f} deg"
          + ("  OK" if ok else "  FAILED - these numbers are not evidence"))
    if not ok:
        return 1

    print(f"\n{'ori_cost':>9} | {'① 掌法向':>16} {'手指':>12} | "
          f"{'② 双手相对位置':>16} | {'③ 双手相对朝向':>16} | {'(绝对位置)':>14}")
    print(f"{'':>9} | {'med':>7}{'p95':>9} {'med':>12} | "
          f"{'med':>7}{'p95':>9} | {'med':>7}{'p95':>9} | {'med':>14}")
    if args.sweep_lm:
        todo = [(2.0, 3, lm) for lm in args.sweep_lm]
    elif args.sweep_substeps:
        todo = [(2.0, n, 10.0) for n in args.sweep_substeps]
    else:
        todo = [(oc, 3, 10.0) for oc in args.sweep_ori]
    for oc, nsub, lm in todo:
        r = run(frames, ori_cost=oc, substeps=nsub, gain=args.gain,
                lm_damping=lm, **base)
        if args.sweep_lm:
            print(f"  lm_damping={lm}")
        elif args.sweep_substeps:
            print(f"  substeps={nsub} gain={args.gain}")
        rs = r["roll_signed"]
        print(f"    [roll] 绕掌法向的带符号误差: 中位 {np.median(rs):+6.1f}deg  "
              f"均值 {np.mean(rs):+6.1f}  标准差 {np.std(rs):5.1f}  "
              f"|同号占比 {100 * max((rs > 0).mean(), (rs < 0).mean()):4.0f}%|   "
              f"j7 顶限位 {100 * np.mean(r['j7_pin']):4.0f}%  "
              f"j6 顶限位 {100 * np.mean(r['j6_pin']):4.0f}%  "
              f"j7 中位 {np.median(r['j7']):+6.1f}deg")
        print(f"{oc:9.1f} | {np.median(r['palm']):7.1f}"
              f"{np.percentile(r['palm'], 95):9.1f} "
              f"{np.median(r['finger']):12.1f} | "
              f"{np.median(r['relpos']):7.1f}"
              f"{np.percentile(r['relpos'], 95):9.1f} | "
              f"{np.median(r['relori']):7.1f}"
              f"{np.percentile(r['relori'], 95):9.1f} | "
              f"{np.median(r['abspos']):14.1f}")
    print("\n单位:角度 deg,位置 mm。①②③ 是用户 2026-08-02 定的优先级顺序;"
          "最后一列是他们明说可以牺牲的那一维,只作参照。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
