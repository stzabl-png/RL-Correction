#!/usr/bin/env python3
"""Is the arm configuration a function of the TARGET, or of the HISTORY?

2026-08-02. Isaac and the Viser bench were run on the same clip with the same
law and the same parameters and their commanded R_arm_j1 went in opposite
directions from the same seed:

    frame     0    60   140   800
    Isaac  +28.6 +28.6 -130.0  -26.9
    Viser  +31.5 +94.5 +157.3 +176.0   (parked on the +176 deg limit)

Both are valid IK solutions - a 7-DoF arm on a 6-DoF wrist target has a free
dimension and the shoulder can swing either way round. What picked the branch
was WHEN each solver first started moving: the consumer holds its Pink target
frozen until the arm clutch closes (~frame 70), the bench starts at frame 0, so
the first real target each of them saw was a different pose.

That makes the arm configuration a function of history, which means:
  * the bench cannot predict the robot, so parameters chosen on it are guesses
  * two runs of the ROBOT can differ, depending on when the clutch closed
  * left and right pick branches independently - the "arms diverge" report
  * a joint that walks to its limit stops responding - the "IK locks up" report

So this measures determinism directly: run the same clip twice from two
different histories and see whether the two trajectories agree AFTER they have
both been running a while. Then sweep the lever most likely to fix it - the
posture task, currently 1e-2 against a position cost of 8.0, i.e. 800x weaker,
which is far too weak to pick a branch.

Reported per configuration:
    branch spread   max |run A - run B| per joint, degrees, over the tail.
                    0 = deterministic. This is the number that has to come down.
    wrist residual  what the determinism costs in tracking.

    .venv/bin/python sim/dev_branch_determinism.py
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
from magicdexmate.head_waist_map import TORSO_HOME  # noqa: E402
from magicdexmate.palm_fix import PALM_FIX  # noqa: E402
from magicdexmate.pico.xr_pose import process_xr_pose  # noqa: E402
from magicdexmate.swivel import swivel_angle  # noqa: E402
from pink_vega_ik import PinkVegaIK  # noqa: E402

URDF = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
ARM = [f"{s}_arm_j{i}" for s in ("R", "L") for i in range(1, 8)]


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


SMPL = {"sh": {"left": 16, "right": 17}, "el": {"left": 18, "right": 19}}


def load(path):
    """Wrist poses AND the operator's own elbow swivel angle.

    The operator asked the obvious question: PICO reports a whole body, elbows
    included, so why is the solver guessing? It should not be. Every earlier
    attempt fed the elbow in as a WEIGHTED TASK competing with the wrist, and
    all five failed for the same reason - the human's elbow point is often
    somewhere the robot's elbow cannot reach, so the solve had to choose. The
    SWIVEL ANGLE has no such problem: it is dimensionless, so different body
    proportions do not matter, and placing the robot's elbow at the same angle
    on its OWN circle is always reachable.

    Caveat worth keeping in mind: PICO's elbow is inferred, not measured (the
    upper-arm bone length has CV 0.1%, i.e. it is template FK from the
    shoulder), because there is no elbow tracker. It is still the operator's
    own body model rather than a constant.
    """
    out = []
    for msg in msgpack.Unpacker(open(path, "rb"), raw=False):
        trk = msg.get("trackers") or {}
        b24 = msg.get("body24")
        if any(trk.get(k) is None for k in ("RWRIST", "LWRIST", "WAIST")):
            continue
        w = np.asarray(trk["WAIST"], float)
        fr = {h: process_xr_pose(np.asarray(trk[k], float), w)
              for h, k in (("right", "RWRIST"), ("left", "LWRIST"))}
        sw = {}
        if b24 is not None:
            b24 = np.asarray(b24, float)
            for h in ("right", "left"):
                sh = process_xr_pose(b24[SMPL["sh"][h]], w)[:3, 3]
                el = process_xr_pose(b24[SMPL["el"][h]], w)[:3, 3]
                sw[h] = swivel_angle(sh, el, fr[h][:3, 3])
        fr["_swivel"] = sw
        out.append(fr)
    return out


def solve_run(frames, start, posture_cost, scale, up, ramp_frames=90,
              null_bias=0.0, swivel_target="operator"):
    """Solve from `start`, holding still before it. -> (q trace, residual trace)."""
    chest_pos, chest_wxyz = chest_of(TORSO_HOME)
    bind = chest_pos + np.array([0.0, 0.0, -0.25 + up])
    ik = PinkVegaIK(urdf_path=str(URDF), posture_cost=posture_cost,
                    null_bias=null_bias)
    # swivel_target: "operator" = the elbow PICO reports, "const" = a fixed
    # nominal (the first test, which only answered "can a pinned swivel pin the
    # branch at all")
    if null_bias > 0.0 and swivel_target == "const":
        ik.refresh_fk()
        for h in ("right", "left"):
            ik.set_swivel_target(h, ik._robot_swivel(h))
    qs, err = [], []
    home = {}
    for i, fr in enumerate(frames):
        if i < start:
            # what the consumer does before the clutch closes: freeze the
            # target on the current pose, so the solve does not move
            ik.hold_target("right")
            ik.hold_target("left")
            ik.solve()
            qs.append(ik.config.q.copy())
            err.append(0.0)
            continue
        if not home:
            ik.refresh_fk()
            home = {h: ik.ee_pos(h).copy() for h in ("right", "left")}
        a = min(1.0, (i - start) / ramp_frames)
        if null_bias > 0.0 and swivel_target == "operator":
            for h in ("right", "left"):
                ik.set_swivel_target(h, fr.get("_swivel", {}).get(h))
        for h in ("right", "left"):
            T = fr[h]
            p_base = bind + T[:3, 3] * scale
            rel = np.linalg.inv(_se3(chest_pos, chest_wxyz)) @ _h4(p_base)
            del rel
            q4 = R.from_matrix(T[:3, :3] @ PALM_FIX[h]).as_quat()
            ik.set_target_chest(h, chest_pos, chest_wxyz, p_base,
                                np.array([q4[3], *q4[:3]]))
            # ramp in pink's own world, where `home` lives
            tgt = ik.frame_tasks[h].transform_target_to_world
            tgt.translation = (1.0 - a) * home[h] + a * tgt.translation
        ik.solve()
        ik.refresh_fk()
        qs.append(ik.config.q.copy())
        err.append(max(float(np.linalg.norm(
            ik.ee_pos(h) - ik.last_target[h].translation))
            for h in ("right", "left")))
    return np.asarray(qs), np.asarray(err)


def _h4(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


def _se3(pos, quat_wxyz):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3],
                             quat_wxyz[0]]).as_matrix()
    T[:3, 3] = pos
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=pathlib.Path,
                    default=pathlib.Path("logs/clip_headwaist.msgpack"))
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--scale", type=float, default=1.35)
    ap.add_argument("--bind-up", type=float, default=0.20)
    ap.add_argument("--posture", nargs="+", type=float,
                    default=[0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
    ap.add_argument("--swivel", choices=["operator", "const"],
                    default="operator",
                    help="whose elbow decides the redundant DoF. 'operator' "
                         "uses the elbow PICO reports - the input that was "
                         "there all along and that the first null-space test "
                         "did not use.")
    ap.add_argument("--null-bias", nargs="+", type=float, default=None,
                    help="sweep the null-space projection instead of the "
                         "posture cost. The projection lies in the exact null "
                         "space of the wrist task's Jacobian, so unlike the "
                         "posture cost it cannot trade wrist accuracy away.")
    args = ap.parse_args()

    frames = load(args.replay)[:args.max_frames]
    print(f"[data] {len(frames)} frames  scale={args.scale} "
          f"bind_up={args.bind_up}")
    print(f"[test] run A starts solving at frame 0, run B at frame 70 - the "
          f"clutch delay that separated the bench from the consumer\n")
    print(f"{('null_bias' if args.null_bias else 'posture_cost'):>13} "
          f"{'branch spread (deg)':>22} "
          f"{'wrist residual med (mm)':>26}")
    print(f"{'':>13} {'med joint':>10} {'worst':>11} {'A':>13} {'B':>12}")
    todo = ([(0.01, nb) for nb in args.null_bias] if args.null_bias
            else [(pc, 0.0) for pc in args.posture])
    for pc, nb in todo:
        qa, ea = solve_run(frames, 0, pc, args.scale, args.bind_up, null_bias=nb,
                           swivel_target=args.swivel)
        qb, eb = solve_run(frames, 70, pc, args.scale, args.bind_up, null_bias=nb,
                           swivel_target=args.swivel)
        tail = slice(300, None)
        names = list(PinkVegaIK(urdf_path=str(URDF)).pin_names)
        d = np.degrees(np.abs(qa[tail] - qb[tail]))
        per_joint = np.median(d, axis=0)
        worst_i = int(np.argmax(per_joint))
        flag = "" if per_joint.max() < 1.0 else \
               ("   <-- history-dependent" if per_joint.max() > 5.0 else "")
        print(f"{(nb if args.null_bias else pc):13.3f} {np.median(per_joint):10.2f} "
              f"{per_joint[worst_i]:7.1f} {names[worst_i]:>10s} "
              f"{np.median(ea[tail]) * 1000:9.1f} {np.median(eb[tail]) * 1000:11.1f}"
              + flag)


if __name__ == "__main__":
    main()
