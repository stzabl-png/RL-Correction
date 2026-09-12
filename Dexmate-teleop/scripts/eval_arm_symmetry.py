#!/usr/bin/env python3
"""Per-joint left/right arm symmetry, per motion. The elbow-swivel regression test.

The operator watched the preview and said "the wrists match but the elbows are
way off". They were right, and it is specific: over a real take the mirrored
joint difference |q_left - S*q_right| is

    j3 146 deg   j5 112 deg   j1 88 deg      <- the swivel null space
    j2  13 deg   j4   9 deg                  <- flexion agrees

A 7-DoF arm tracking a 6-DoF wrist pose has one free dimension and it is the
elbow rotating about the shoulder-wrist axis. j1/j3/j5 are that dimension.
Nothing in the wrist target constrains it, so the two arms settle on different
branches from wrist targets only 113 mm / 29 deg apart.

This script exists so that "did the fix help" is a command, not an opinion, and
so it reports per MOTION - a change can help arms-down and wreck arms-overhead.

SELF-CHECK, and why it refuses to run without it
------------------------------------------------
Every number in here comes from a hand-rolled solve loop, and on 2026-07-31 I
got four different wrong answers out of loops like this one (subsampled frames
starve a DIFFERENTIAL solver of its seed; a forgotten constant offset; envelope
saturation read as solver error; a harness on a different mapping law than the
GUI). So before reporting anything it feeds the solver exactly mirrored
targets, where the answer must be zero, and aborts if that comes back above
0.5 deg. A metric that cannot reproduce a known answer does not get to report
an unknown one.

Usage:
    .venv/bin/python scripts/eval_arm_symmetry.py                       # default sweep
    .venv/bin/python scripts/eval_arm_symmetry.py --captures logs/clip_headwaist.msgpack
    .venv/bin/python scripts/eval_arm_symmetry.py --configs off swivel:1.0
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import msgpack
import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sim"))
from magicdexmate.palm_fix import PALM_FIX  # noqa: E402
from magicdexmate.pico.xr_pose import (  # noqa: E402
    mat_to_pos_quat_wxyz, process_xr_pose)
from magicdexmate.swivel import elbow_from_swivel, swivel_angle  # noqa: E402
from pink_vega_ik import EE_FRAME, ELBOW_FRAME, PinkVegaIK  # noqa: E402

IK_URDF = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
# Imported, not re-declared: this constant lived in five files and every
# copy said 55 deg of forward lean while all thirteen authored poses were
# at 10.3 deg. See magicdexmate/head_waist_map.py.
from magicdexmate.head_waist_map import TORSO_HOME  # noqa: E402
# q_left = MIRROR_SIGNS * q_right. Brute-forced over all 128 sign patterns by
# FK and cross-checked against the hand-authored HOME pose - do NOT re-derive
# this from the URDF axis columns, that route gives the wrong answer because
# the origin rpy chain chirality is not in the axis alone.
MIRROR_SIGNS = np.array([-1, -1, -1, 1, -1, -1, -1])
MIRROR = np.diag([1.0, -1.0, 1.0])
# 'the operator was symmetric here': mirror distance between the two
# wrist TARGETS. Below this, the robot is expected to be symmetric too.
SYM_MM = 60.0
SMPL = {"shoulder": {"left": 16, "right": 17},
        "elbow": {"left": 18, "right": 19},
        "wrist": {"left": 20, "right": 21}}


def _ang(a, b) -> float:
    """Angle [deg] between two direction vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip((a @ b) / (na * nb), -1.0, 1.0))))


def load(path: pathlib.Path):
    out = []
    for m in msgpack.Unpacker(open(path, "rb"), raw=False):
        trk = m.get("trackers") or {}
        if all(k in trk for k in ("LWRIST", "RWRIST", "WAIST")):
            out.append(m)
    return out


class Rig:
    """Robot-side constants at the locked torso, read once."""

    def __init__(self, urdf_path):
        self.u = yourdfpy.URDF.load(str(urdf_path), load_meshes=False,
                                    build_scene_graph=True)
        self.joints = [j for j, jo in self.u.joint_map.items() if jo.type != "fixed"]
        self.u.update_cfg(np.array([TORSO_HOME.get(j, 0.0) for j in self.joints]))
        p = lambda n: np.array(self.u.get_transform(n, self.u.base_link))[:3, 3]
        T = np.array(self.u.get_transform("arm_center", self.u.base_link))
        self.chest_pos = T[:3, 3]
        q = R.from_matrix(T[:3, :3]).as_quat()
        self.chest_wxyz = np.array([q[3], *q[:3]])
        self.bind = self.chest_pos + np.array([0.0, 0.0, -0.05])
        self.sh = {h: p(f"vega_1_{h[0].upper()}_arm_l1") for h in ("right", "left")}
        el = {h: p(ELBOW_FRAME[h]) for h in ("right", "left")}
        ee = {h: p(f"{h[0].upper()}_ee") for h in ("right", "left")}
        self.l_up = {h: np.linalg.norm(el[h] - self.sh[h]) for h in ee}
        self.l_fore = {h: np.linalg.norm(ee[h] - el[h]) for h in ee}
        # EQUIVALENT CENTRES. The link frames are not the kinematic centres and
        # using them was wrong: |R_arm_l1 -> elbow| wanders 39 mm and
        # |elbow -> R_ee| 32 mm, which made the arm look like it had
        # non-rigid segments and a 0.71 upper:fore ratio against the human's
        # 0.94. Fitted over 150 random configurations the real picture is
        #     shoulder centre -> elbow      308.2 +- 6.4 mm
        #     elbow -> wrist centre         307.6 +- 10.5 mm
        #     wrist centre -> EE            147.4 +- 0.0 mm
        # i.e. a 1.00 upper:fore ratio, essentially the human's, and the EE is
        # the HAND (SMPL 22/23), not the wrist (20/21).
        self.sh_c = {"right": np.array([-0.070, -0.231, 1.211]),
                     "left": np.array([-0.070, +0.231, 1.211])}
        self.wc_off = np.array([0.003, -0.005, -0.147])   # in the EE frame
        self.L_UP = 0.3082      # fitted |shoulder centre -> elbow|
        self.L_FORE = 0.3076    # fitted |elbow -> wrist centre|
        self.L_HAND = 0.1474    # fitted |wrist centre -> EE|

    def wrist_centre(self, ik, hand):
        T = ik.data.oMf[ik.model.getFrameId(EE_FRAME[hand])]
        off = self.wc_off.copy()
        if hand == "left":
            off[1] = -off[1]
        return T.rotation @ off + T.translation


def solve_run(rig, frames, mode, weight, lm_damping, warm_frac=1 / 3,
              law='waist-abs', pos_scale=1.0):
    """-> dict of metrics.

    The primary one is SWIVEL ERROR: |robot elbow swivel - operator elbow
    swivel|, per arm, per frame. Robot-vs-robot symmetry is the wrong target
    and cost a whole iteration to learn - the operator's own arms are often
    genuinely asymmetric (184 mm mirror difference on the front-horizontal
    take, the largest of the batch), and a solve that returns a symmetric robot
    from an asymmetric operator is not tracking, it is the posture task eating
    the input. What the operator actually asked for:

        "sometimes my own arms are asymmetric too, but when my arms ARE
         symmetric, Dexmate's should be roughly symmetric as well"

    So: track the swivel, and check symmetry CONDITIONALLY - only over the
    frames where the operator was in fact near-symmetric.
    """
    ik = PinkVegaIK(urdf_path=str(IK_URDF), elbow_cost=weight,
                    elbow_lm_damping=lm_damping)
    jd, wr, sw, hasym, seg = [], [], [], [], []
    rel, absd, orie = [], [], []
    warm = int(len(frames) * warm_frac)
    for i, m in enumerate(frames):
        trk = m["trackers"]
        waist = np.asarray(trk["WAIST"], float)
        b24 = m.get("body24")
        tgt, ori, shape_ref, seg_el = {}, {}, None, {}
        for h, k in (("right", "RWRIST"), ("left", "LWRIST")):
            T = process_xr_pose(np.asarray(trk[k], float), waist)
            p_w = mat_to_pos_quat_wxyz(T)[0]
            if law == "seg-rel" and b24:
                pass    # both arms built together below
            elif law == "shape-rel" and b24:
                # Operator priority, stated 2026-07-31:
                #   1. the LEFT-RIGHT wrist relative position is sacred
                #   2. absolute wrist position may drift
                #   3. orientation may give up at most 10 deg
                #   4. subject to those, arm SHAPE wins
                # So: build a shape-exact arm first - each segment along the
                # operator's own segment direction, at the ROBOT's link length,
                # which makes both direction errors zero by construction - and
                # then restore the relative wrist vector DIFFERENTIALLY. Common
                # mode never touches the relative vector, which is exactly why
                # absolute position is the free thing to give away.
                pass    # targets built after the loop, both arms at once
            elif law in ("shoulder-abs", "shoulder-rel") and b24:
                # what the preview台 shows by default: anchor at the SHOULDER
                # and scale only the arm vector, so torso proportions never
                # enter. Comparing against waist-abs is the point of --law.
                p_sh = mat_to_pos_quat_wxyz(process_xr_pose(
                    np.asarray(b24[SMPL["shoulder"][h]], float), waist))[0]
                p_el = mat_to_pos_quat_wxyz(process_xr_pose(
                    np.asarray(b24[SMPL["elbow"][h]], float), waist))[0]
                p_hw = mat_to_pos_quat_wxyz(process_xr_pose(
                    np.asarray(b24[SMPL["wrist"][h]], float), waist))[0]
                human_arm = (np.linalg.norm(p_el - p_sh)
                             + np.linalg.norm(p_hw - p_el))
                k_s = (rig.l_up[h] + rig.l_fore[h]) / max(human_arm, 1e-3)
                tgt[h] = rig.sh[h] + k_s * (p_w - p_sh)
            else:
                tgt[h] = rig.bind + p_w * pos_scale
            M4 = np.eye(4)
            M4[:3, :3] = T[:3, :3] @ PALM_FIX[h]
            ori[h] = mat_to_pos_quat_wxyz(M4)[1]
        if law == "shape-rel" and b24:
            hp = lambda part, hh: mat_to_pos_quat_wxyz(process_xr_pose(
                np.asarray(b24[SMPL[part][hh]], float), waist))[0]
            shp, shp_el = {}, {}
            for h in ("right", "left"):
                sh, el, wr_h = hp("shoulder", h), hp("elbow", h), hp("wrist", h)
                u1, u2 = el - sh, wr_h - el
                n1, n2 = np.linalg.norm(u1), np.linalg.norm(u2)
                if n1 < 1e-6 or n2 < 1e-6:
                    shp = {}
                    break
                # the elbow that makes the shape exact. It sits at exactly
                # l_up from the shoulder, so it is on the reachable sphere BY
                # CONSTRUCTION - unlike the operator's own elbow position,
                # which is what the old --elbow-weight fed and why that blew
                # the wrist out to 475 mm.
                shp_el[h] = rig.sh[h] + rig.l_up[h] * u1 / n1
                shp[h] = shp_el[h] + rig.l_fore[h] * u2 / n2
            if shp:
                r_star = hp("wrist", "left") - hp("wrist", "right")   # 1:1, unscaled
                e = r_star - (shp["left"] - shp["right"])
                tgt["left"] = shp["left"] + 0.5 * e
                tgt["right"] = shp["right"] - 0.5 * e
                shape_ref = shp_el
        if law == "seg-rel" and b24:
            # The operator's three-segment proposal, fully consistent.
            # Every earlier attempt failed the same way: the WRIST target came
            # from one construction (shoulder-abs, whole-arm scaling) and the
            # ELBOW target from another (per-segment lengths), and the two are
            # only simultaneously satisfiable if the forearm happens to be the
            # right length. It never is, so the solver had to drop one - which
            # is why the elbow task never took, at any weight, four times over.
            # Build BOTH from the same segments and the conflict disappears:
            #     elbow        = shoulder centre + L_UP   * u_upperarm
            #     wrist centre = elbow           + L_FORE * u_forearm
            #     EE           = wrist centre    + 147 mm along the hand axis
            # Directions come from the operator, lengths from the robot.
            hq = lambda part, hh: mat_to_pos_quat_wxyz(process_xr_pose(
                np.asarray(b24[SMPL[part][hh]], float), waist))[0]
            ok = True
            for h in ("right", "left"):
                u1 = hq("elbow", h) - hq("shoulder", h)
                u2 = hq("wrist", h) - hq("elbow", h)
                n1, n2 = np.linalg.norm(u1), np.linalg.norm(u2)
                if n1 < 1e-6 or n2 < 1e-6:
                    ok = False
                    break
                seg_el[h] = rig.sh_c[h] + rig.L_UP * u1 / n1
                wc_t = seg_el[h] + rig.L_FORE * u2 / n2
                # EE sits 147 mm from the wrist centre along the commanded
                # hand axis, so the orientation target decides where it goes
                Rt = R.from_quat([ori[h][1], ori[h][2], ori[h][3], ori[h][0]]).as_matrix()
                tgt[h] = wc_t + rig.L_HAND * Rt[:, 2]
            if ok:
                hpw = lambda hh: hq("wrist", hh)
                e = (hpw("left") - hpw("right")) - (tgt["left"] - tgt["right"])
                tgt["left"] = tgt["left"] + 0.5 * e
                tgt["right"] = tgt["right"] - 0.5 * e
                shape_ref = seg_el
            else:
                seg_el = {}
        if law == "shoulder-rel" and b24:
            # Simplest composition, and it beats the hand-built shape-exact
            # law: take the shoulder-abs targets (best arm shape of the three)
            # and apply the SAME differential correction that pins the
            # left-right wrist vector. The two are independent - shape comes
            # from anchoring at the shoulder, the relative vector comes from a
            # differential shift - so composing them keeps both.
            hpw = lambda hh: mat_to_pos_quat_wxyz(process_xr_pose(
                np.asarray(b24[SMPL["wrist"][hh]], float), waist))[0]
            e = (hpw("left") - hpw("right")) - (tgt["left"] - tgt["right"])
            tgt["left"] = tgt["left"] + 0.5 * e
            tgt["right"] = tgt["right"] - 0.5 * e
        ref_tgt = {}
        if b24:
            for h in ("right", "left"):
                hq = lambda part: mat_to_pos_quat_wxyz(process_xr_pose(
                    np.asarray(b24[SMPL[part][h]], float), waist))[0]
                p_sh, p_el, p_hw = hq("shoulder"), hq("elbow"), hq("wrist")
                ha = np.linalg.norm(p_el - p_sh) + np.linalg.norm(p_hw - p_el)
                ref_tgt[h] = rig.sh[h] + ((rig.l_up[h] + rig.l_fore[h])
                                          / max(ha, 1e-3)) * (
                    mat_to_pos_quat_wxyz(process_xr_pose(
                        np.asarray(trk["RWRIST" if h == "right" else "LWRIST"],
                                   float), waist))[0] - p_sh)
        for h in ("right", "left"):
            ik.set_target_chest(h, rig.chest_pos, rig.chest_wxyz, tgt[h], ori[h])
        if weight > 0.0 and b24:
            for h in ("right", "left"):
                hp = lambda part: mat_to_pos_quat_wxyz(process_xr_pose(
                    np.asarray(b24[SMPL[part][h]], float), waist))[0]
                if mode == "position":
                    ik.set_elbow_target_chest(h, rig.chest_pos, rig.chest_wxyz,
                                              rig.bind + hp("elbow"))
                elif mode == "seg":
                    # The operator's three-segment proposal, with the centres
                    # MEASURED rather than guessed: put the elbow on the sphere
                    # of radius 308 mm about the fitted shoulder centre, along
                    # the operator's own upper-arm direction. The elbow really
                    # does live on that sphere (308.2 +- 6.4 mm over 150 random
                    # configurations), so unlike every earlier elbow target
                    # this one cannot fight the wrist task.
                    d = hp("elbow") - hp("shoulder")
                    n = np.linalg.norm(d)
                    if n > 1e-6:
                        ik.set_elbow_target_chest(
                            h, rig.chest_pos, rig.chest_wxyz,
                            rig.sh_c[h] + rig.L_UP * d / n)
                elif mode == "shape" and shape_ref is not None:
                    ik.set_elbow_target_chest(h, rig.chest_pos, rig.chest_wxyz,
                                              shape_ref[h])
                elif mode == "swivel":
                    a = swivel_angle(hp("shoulder"), hp("elbow"), hp("wrist"))
                    if a is None:
                        continue
                    e = elbow_from_swivel(rig.sh[h], tgt[h], rig.l_up[h],
                                          rig.l_fore[h], a)
                    if e is not None:
                        ik.set_elbow_target_chest(h, rig.chest_pos,
                                                  rig.chest_wxyz, e)
        ik.solve()
        if i < warm:
            continue
        ik.refresh_fk()
        q = dict(zip(ik.pin_names, ik.config.q))
        qr = np.array([q[f"R_arm_j{j + 1}"] for j in range(7)])
        ql = np.array([q[f"L_arm_j{j + 1}"] for j in range(7)])
        jd.append(np.degrees(np.abs(ql - MIRROR_SIGNS * qr)))
        wr.append(max(np.linalg.norm(
            ik.ee_pos(h) - ik.frame_tasks[h].transform_target_to_world.translation)
            for h in ("right", "left")) * 1000)
        # The operator's three priorities. Measured on the TARGETS, in the
        # base frame, not on pinocchio's EE: those live in a different frame
        # (set_target_chest re-expresses via the chest) and subtracting them
        # from a base-frame target gave a nonsense 408 mm while the wrist
        # residual read 0.0 mm in the same row. The wrist residual column is
        # what says the robot reached the target; these say whether the target
        # was the right one.
        if b24:
            hpw = lambda hh: mat_to_pos_quat_wxyz(process_xr_pose(
                np.asarray(b24[SMPL["wrist"][hh]], float), waist))[0]
            rel.append(np.linalg.norm((tgt["left"] - tgt["right"])
                                      - (hpw("left") - hpw("right"))) * 1000)
            # absolute give-up = how far this law moved the wrist away from the
            # faithful shoulder-abs placement, the thing being traded away
            absd.append(np.mean([np.linalg.norm(tgt[h] - ref_tgt[h])
                                 for h in ("right", "left")]) * 1000)
        orie.append(max(
            np.degrees(R.from_matrix(
                ik.data.oMf[ik.model.getFrameId(EE_FRAME[h])].rotation.T
                @ ik.frame_tasks[h].transform_target_to_world.rotation).magnitude())
            for h in ("right", "left")))
        # swivel tracking, the metric that does not presume symmetry
        if b24:
            errs = []
            for h in ("right", "left"):
                hp = lambda part: mat_to_pos_quat_wxyz(process_xr_pose(
                    np.asarray(b24[SMPL[part][h]], float), waist))[0]
                a_h = swivel_angle(hp("shoulder"), hp("elbow"), hp("wrist"))
                el_r = ik.data.oMf[ik.model.getFrameId(ELBOW_FRAME[h])].translation
                a_r = swivel_angle(rig.sh[h], el_r, ik.ee_pos(h))
                # segment DIRECTIONS: the plainest statement of "does the arm
                # look like mine". Scale-free, so the 1.33x reach difference
                # never enters, and unlike swivel it covers the whole chain
                # rather than just the redundant dimension.
                wc = rig.wrist_centre(ik, h)
                seg.append([_ang(hp("elbow") - hp("shoulder"),
                                 el_r - rig.sh_c[h]),
                            _ang(hp("wrist") - hp("elbow"), wc - el_r)])
                if a_h is None or a_r is None:
                    continue
                errs.append(abs(np.degrees((a_r - a_h + np.pi) % (2 * np.pi) - np.pi)))
            if errs:
                sw.append(float(np.mean(errs)))
                hasym.append(np.linalg.norm(
                    tgt["left"] - MIRROR @ tgt["right"]) * 1000)
    return dict(jd=np.array(jd), wr=np.array(wr), sw=np.array(sw),
                hasym=np.array(hasym), seg=np.array(seg) if seg else np.zeros((0, 2)),
                rel=np.array(rel), absd=np.array(absd), orie=np.array(orie))


def self_check(rig, frames, tol=0.5) -> float:
    """Mirrored targets in, mirrored joints out. Returns the worst joint [deg]."""
    ik = PinkVegaIK(urdf_path=str(IK_URDF))
    worst = 0.0
    for idx in np.linspace(0, len(frames) - 1, 4, dtype=int):
        trk = frames[idx]["trackers"]
        waist = np.asarray(trk["WAIST"], float)
        T = process_xr_pose(np.asarray(trk["RWRIST"], float), waist)
        p_r = mat_to_pos_quat_wxyz(T)[0]
        R_r = T[:3, :3] @ PALM_FIX["right"]
        for _ in range(400):
            for h, p, Rm in (("right", p_r, R_r),
                             ("left", MIRROR @ p_r, MIRROR @ R_r @ MIRROR)):
                M4 = np.eye(4)
                M4[:3, :3] = Rm
                ik.set_target_chest(h, rig.chest_pos, rig.chest_wxyz,
                                    rig.bind + p, mat_to_pos_quat_wxyz(M4)[1])
            ik.solve()
        ik.refresh_fk()
        q = dict(zip(ik.pin_names, ik.config.q))
        qr = np.array([q[f"R_arm_j{j + 1}"] for j in range(7)])
        ql = np.array([q[f"L_arm_j{j + 1}"] for j in range(7)])
        worst = max(worst, float(np.degrees(np.abs(ql - MIRROR_SIGNS * qr)).max()))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", nargs="+", type=pathlib.Path, default=None,
                    help="takes to evaluate; default = the labelled 2026-07-31 "
                         "poses plus the head/waist clip")
    ap.add_argument("--configs", nargs="+", default=["off", "position:0.25",
                                                     "swivel:0.25", "swivel:1.0"],
                    help="mode:weight, or 'off'. Append :lm to override damping")
    ap.add_argument("--pos-scale", type=float, default=1.0)
    ap.add_argument("--law", choices=["waist-abs", "shoulder-abs", "shape-rel", "shoulder-rel", "seg-rel"],
                    default="waist-abs",
                    help="shoulder-abs is what the preview台 draws by default")
    ap.add_argument("--skip-self-check", action="store_true",
                    help="report anyway if the mirrored-target check fails. "
                         "Only for debugging the check itself")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    caps = args.captures
    if caps is None:
        d = root / "logs/pose_capture_20260731_ori"
        caps = sorted(p for p in d.glob("*take1.msgpack")
                      if "format_test" not in p.name)
        clip = root / "logs/clip_headwaist.msgpack"
        if clip.exists():
            caps.append(clip)
    caps = [c for c in caps if c.exists()]
    if not caps:
        raise SystemExit("no captures found")

    rig = Rig(IK_URDF)
    takes = [(c.stem, load(c)) for c in caps]
    takes = [(n, f) for n, f in takes if len(f) > 30]

    worst = self_check(rig, takes[0][1])
    ok = worst <= 0.5
    print(f"[self-check] mirrored targets -> mirrored joints, worst {worst:.3f} deg "
          + ("OK" if ok else "FAILED"))
    if not ok and not args.skip_self_check:
        raise SystemExit("[self-check] the harness cannot reproduce a known "
                         "answer; its numbers are not evidence. Aborting.")

    for cfg in args.configs:
        parts = cfg.split(":")
        mode = parts[0]
        weight = 0.0 if mode == "off" else float(parts[1])
        lm = float(parts[2]) if len(parts) > 2 else None
        print(f"\n=== {cfg} ===")
        print(f"{'take':>26} {'①左右相对':>10} {'②绝对漂移':>10} "
              f"{'③朝向':>8} {'④上臂':>7} {'④前臂':>7} {'你对称时':>9}")
        allj, allw, allsw, allha, allseg = [], [], [], [], []
        allrel, allabs, allori = [], [], []
        for name, frames in takes:
            r = solve_run(rig, frames, mode, weight, lm, law=args.law,
                          pos_scale=args.pos_scale)
            allj.append(r["jd"]); allw.append(r["wr"])
            allsw.append(r["sw"]); allha.append(r["hasym"]); allseg.append(r["seg"])
            allrel.append(r["rel"]); allabs.append(r["absd"]); allori.append(r["orie"])
            sym = r["jd"][:, [0, 2, 4]].sum(axis=1)
            near = r["hasym"] < SYM_MM if len(r["hasym"]) else np.array([], bool)
            print(f"{name[:26]:>26} "
                  + (f"{np.median(r['rel']):8.0f}mm" if len(r["rel"]) else "        -")
                  + f"{np.median(r['absd']):8.0f}mm"
                  + f"{np.median(r['orie']):7.1f}°"
                  + (f"{np.median(r['seg'][:, 0]):6.0f}°{np.median(r['seg'][:, 1]):7.0f}°"
                     if len(r["seg"]) else "      -      -")
                  + (f"{np.median(sym[:len(near)][near]):8.0f}°"
                     if near.any() else "       -"), flush=True)
        J = np.vstack(allj); W = np.concatenate(allw)
        SW = np.concatenate([a for a in allsw if len(a)])
        SG = np.vstack([a for a in allseg if len(a)])
        HA = np.concatenate([a for a in allha if len(a)])
        sym = J[:, [0, 2, 4]].sum(axis=1)
        n = min(len(sym), len(HA)); near = HA[:n] < SYM_MM
        RL=np.concatenate([a for a in allrel if len(a)]) if any(len(a) for a in allrel) else np.array([0.])
        AD=np.concatenate(allabs); OE=np.concatenate(allori)
        print(f"{'ALL':>26} {np.median(RL):8.0f}mm{np.median(AD):8.0f}mm"
              f"{np.median(OE):7.1f}°{np.median(SG[:, 0]):6.0f}°{np.median(SG[:, 1]):7.0f}°"
              + (f"{np.median(sym[:n][near]):8.0f}°" if near.any() else "       -")
              + f"   [③>10° 的帧 {100 * (OE > 10).mean():.0f}%]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
