#!/usr/bin/env python3
"""Test the operator's decoupling proposal: arms read the hand relative to the CHEST.

Stated 2026-08-02:

    "can't you decouple it - the arm just looks at the hand relative to the
     chest in the body skeleton, the waist only looks at the waist, the head
     only at the head"

Why it should help, given what was measured on 2026-08-01: waist-abs builds the
wrist target as (bind point + wrist-relative-to-WAIST) and expresses it in the
gravity-aligned BASE frame. The arm hangs off the chest, and moving TORSO_HOME
from 55 deg of forward lean to the authored 10.3 deg rotates that chest by 45
deg - which the target does not follow, so the arm has to absorb the 45 deg
itself and runs its wrist joints onto their limits. Position alone is then
solvable (1.6 mm) and orientation alone is solvable (0.0 deg), but the two
together are not (272 mm). Expressing BOTH in the chest frame cancels the 45 deg
on both sides.

Note this is NOT the shoulder-abs / shoulder-rel law that was tried on
2026-07-31 and rejected: those anchored the POSITION at the shoulder but left
the ORIENTATION absolute, and the orientation half is exactly what the
2026-08-01 measurement says matters.

The human chest frame is built geometrically from the two shoulders and the
waist rather than read off body24's spine rotation - PICO's skeleton is a
template FK fit, so a joint's own rotation is inferred where the shoulder
POSITIONS are closer to observed (bone-length CV 0.1% vs the forearm's 11.8%).

PALM_FIX needs no refit: it is applied on the right, in the hand's own frame,
so R_chest_rob @ R_chest_hum^T @ R_wrist @ PALM_FIX reduces to the old
R_wrist @ PALM_FIX exactly when the two chests agree.

    .venv/bin/python sim/dev_chest_rel.py
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
from magicdexmate.palm_fix import FINGERS_IN_EE, PALM_FIX, PALM_IN_EE  # noqa: E402
from magicdexmate.pico.xr_pose import process_xr_pose  # noqa: E402
from pink_vega_ik import PinkVegaIK  # noqa: E402

URDF = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
OLD_TORSO = {"torso_j1": np.radians(60.0), "torso_j2": np.radians(90.0),
             "torso_j3": np.radians(-25.0)}
SMPL = {"sh": {"left": 16, "right": 17}, "el": {"left": 18, "right": 19},
        "wr": {"left": 20, "right": 21}}
EE = {"right": "R_ee", "left": "L_ee"}


def pose7_to_T(p7):
    """body24 row (x y z qx qy qz qw or x y z qw qx qy qz) -> 4x4."""
    v = np.asarray(p7, float)
    T = np.eye(4)
    T[:3, 3] = v[:3]
    T[:3, :3] = R.from_quat(v[3:7]).as_matrix()
    return T


def chest_of_robot(torso):
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


def human_chest(b24_rel):
    """Chest frame from shoulders + waist. b24_rel: 24x4x4 already waist-relative.

    x forward, y left, z up - the same convention the base frame uses, so the
    two chests are directly comparable.
    """
    sl = b24_rel[SMPL["sh"]["left"]][:3, 3]
    sr = b24_rel[SMPL["sh"]["right"]][:3, 3]
    origin = 0.5 * (sl + sr)
    y = sl - sr
    y /= max(np.linalg.norm(y), 1e-9)
    up = origin            # waist is the origin of the waist-relative frame
    z = up / max(np.linalg.norm(up), 1e-9)
    x = np.cross(y, z)
    x /= max(np.linalg.norm(x), 1e-9)
    z = np.cross(x, y)
    return origin, np.column_stack([x, y, z])


def frames_of(path, wrist_src="trackers"):
    """-> list of (body24-relative-to-waist, {hand: wrist 4x4}).

    The wrist comes from the `trackers` dict by default because that is what
    the consumer consumes (scripts/trackers.env -> LWRIST/RWRIST). It is NOT
    the same as body24[20/21]: the two agree bit-for-bit on 85% of frames and
    diverge on the other 14%, where the forearm bridge has repaired a dropout.
    Feeding this harness the raw body24 wrist instead made chest-rel read
    17 mm where the preview bench read 109 mm on the same take - two correct
    implementations, two different inputs. Keep them on the same channel.
    """
    out = []
    for msg in msgpack.Unpacker(open(path, "rb"), raw=False):
        trk = msg.get("trackers") or {}
        b24 = msg.get("body24")
        if b24 is None or trk.get("WAIST") is None:
            continue
        waist = np.asarray(trk["WAIST"], float)
        rel = np.stack([process_xr_pose(r, waist)
                        for r in np.asarray(b24, float)])
        if wrist_src == "trackers" and all(trk.get(k) is not None
                                           for k in ("RWRIST", "LWRIST")):
            wr = {h: process_xr_pose(np.asarray(trk[k], float), waist)
                  for h, k in (("right", "RWRIST"), ("left", "LWRIST"))}
        else:
            wr = {h: rel[SMPL["wr"][h]] for h in ("right", "left")}
        out.append((rel, wr))
    return out


def run(frames, torso, law, scale, up, warm=200):
    chest_pos, chest_wxyz, chest_R = chest_of_robot(torso)
    bind = chest_pos + np.array([0.0, 0.0, -0.25 + up])
    ik = PinkVegaIK(urdf_path=str(URDF))
    m_ = pin.buildModelFromUrdf(str(URDF)); d_ = m_.createData()
    q_ = pin.neutral(m_)
    for n_, v_ in torso.items():
        q_[m_.joints[m_.getJointId(n_)].idx_q] = v_
    pin.forwardKinematics(m_, d_, q_); pin.updateFramePlacements(m_, d_)
    rob_arm = float(np.linalg.norm(
        d_.oMf[m_.getFrameId("R_ee")].translation
        - d_.oMf[m_.getFrameId("R_arm_j1")].translation))
    err = {"right": [], "left": []}
    ori = {"right": [], "left": []}
    for i, (rel, wr) in enumerate(frames):
        h_org, h_R = human_chest(rel)
        for h in ("right", "left"):
            Tw = wr[h]
            if law == "waist-abs":
                p = bind + Tw[:3, 3] * scale
                Rt = Tw[:3, :3] @ PALM_FIX[h]
            else:                                   # chest-rel[-norm]
                k = scale
                if law == "chest-rel-norm":
                    # scale from the OPERATOR'S OWN arm length this frame, not
                    # a hand-tuned constant: a fixed 1.0-1.35 undershoots the
                    # reach-out poses (raise_high lands 474 mm short) and
                    # overshoots the tucked ones, because the two ends of the
                    # range need different numbers. Normalising makes "arm
                    # fully extended" map to "arm fully extended" by
                    # construction, at every extension in between too.
                    sh = rel[SMPL["sh"][h]][:3, 3]
                    el = rel[SMPL["el"][h]][:3, 3]
                    hl = (np.linalg.norm(el - sh)
                          + np.linalg.norm(Tw[:3, 3] - el))
                    k = rob_arm / max(hl, 1e-3)
                p = chest_pos + chest_R @ (h_R.T @ (Tw[:3, 3] - h_org)) * k
                Rt = chest_R @ h_R.T @ Tw[:3, :3] @ PALM_FIX[h]
            q4 = R.from_matrix(Rt).as_quat()
            ik.set_target_chest(h, chest_pos, chest_wxyz, p,
                                np.array([q4[3], *q4[:3]]))
        ik.solve()
        if i < warm:
            continue
        ik.refresh_fk()
        for h in ("right", "left"):
            err[h].append(float(np.linalg.norm(
                ik.ee_pos(h) - ik.last_target[h].translation)))
            Rg = ik.data.oMf[ik.model.getFrameId(EE[h])].rotation
            Rw = ik.last_target[h].rotation
            ori[h].append([float(np.degrees(np.arccos(np.clip(
                (Rg @ v) @ (Rw @ v), -1, 1)))) for v in (PALM_IN_EE, FINGERS_IN_EE)])
    return ({h: np.asarray(v) for h, v in err.items()},
            {h: np.asarray(v) for h, v in ori.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=pathlib.Path,
                    default=pathlib.Path("logs/playlist_all13.msgpack"))
    ap.add_argument("--max-frames", type=int, default=2000)
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 1.2, 1.35])
    ap.add_argument("--wrist-src", choices=["trackers", "body24"],
                    default="trackers",
                    help="trackers = what the consumer consumes (default)")
    args = ap.parse_args()

    frames = frames_of(args.replay, args.wrist_src)[:args.max_frames]
    print(f"[data] {len(frames)} frames, wrist channel = {args.wrist_src}")

    cases = [("waist-abs OLD torso 1.00/0.20", OLD_TORSO, "waist-abs", 1.00, 0.20),
             ("waist-abs NEW torso 1.35/0.08", TORSO_HOME, "waist-abs", 1.35, 0.08),
             ("waist-abs NEW torso 1.35/0.20", TORSO_HOME, "waist-abs", 1.35, 0.20),
             ("waist-abs NEW torso 1.00/0.20", TORSO_HOME, "waist-abs", 1.00, 0.20),
             ("waist-abs NEW torso 1.35/0.26", TORSO_HOME, "waist-abs", 1.35, 0.26)]
    cases += [(f"chest-rel NEW torso s={s:.2f}", TORSO_HOME, "chest-rel", s, 0.0)
              for s in args.scales]
    cases += [("chest-rel OLD torso s=1.00", OLD_TORSO, "chest-rel", 1.0, 0.0),
              ("chest-rel-norm NEW torso", TORSO_HOME, "chest-rel-norm", 1.0, 0.0)]

    for label, torso, law, scale, up in cases:
        err, ori = run(frames, torso, law, scale, up)
        line = f"{label:28s} |"
        for h in ("right", "left"):
            e = err[h] * 1000.0
            o = ori[h]
            line += (f"  {h[0].upper()}: pos med {np.median(e):6.1f} p95 "
                     f"{np.percentile(e, 95):6.1f}mm  palm "
                     f"{np.median(o[:, 0]):4.1f} finger {np.median(o[:, 1]):4.1f}deg")
        print(line)


if __name__ == "__main__":
    main()
