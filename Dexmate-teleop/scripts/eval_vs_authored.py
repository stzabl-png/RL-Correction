#!/usr/bin/env python3
"""Score a mapping by how close it lands to the poses the operator AUTHORED.

The objective, in the operator's words: fit toward the reference poses they
posed by hand, not toward their verbal descriptions of them. Descriptions were
a workaround for a batch whose wrists had never been posed (arm_j5/j6/j7 were
0.000 in all six); now that thirteen pairs exist and four of them carry real
wrist angles, the authored configuration IS the target and the mapping should
be judged by how near it gets.

This replaces a set of proxy metrics I had invented - left/right relative
position, absolute drift, segment-direction similarity. Those were my
definitions of "good"; these poses are the operator's, stated concretely, one
per motion. When the two disagree the poses win.

What is compared, per pose:
    EE position   robot end-effector relative to the chest, mm
    EE orientation                                          deg
    arm joints    per-joint, deg, over the 14 arm joints

KNOWN-BAD REFERENCE, excluded from the orientation score by default:
    arms_folded_waist   the authored wrist is rolled ~45 deg off the operator's
                        own corrected description ("palm straight back"); they
                        confirmed the authored pose, not the mapping, is wrong.
    the six 2026-07-30 poses  arm_j5/j6/j7 all 0.000 - the wrist was never
                        posed, so their orientation carries no information.
                        Their POSITION is still good and is scored.

Usage:
    .venv/bin/python scripts/eval_vs_authored.py
    .venv/bin/python scripts/eval_vs_authored.py --law shoulder-rel
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import msgpack
import numpy as np
import yourdfpy
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sim"))
from magicdexmate.head_waist_map import TORSO_HOME  # noqa: E402
from magicdexmate.palm_fix import PALM_FIX  # noqa: E402
from magicdexmate.pico.xr_pose import (  # noqa: E402
    mat_to_pos_quat_wxyz, process_xr_pose)
from pink_vega_ik import EE_FRAME, PinkVegaIK  # noqa: E402

IK_URDF = pathlib.Path(
    "~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf").expanduser()
ARM_JOINTS = [f"{s}_arm_j{i}" for s in ("R", "L") for i in range(1, 8)]
# recording stem -> authored slug
PAIRS = {
    "pose_capture_20260731_ori/01_raise_high_take1": "raise_high",
    "pose_capture_20260731_ori/02_front_horizontal_take1": "front_horizontal",
    "pose_capture_20260731_ori/03_lateral_horizontal_take1": "lateral_horizontal",
    "pose_capture_20260731_ori/04_arms_down_take1": "arms_down",
    "pose_capture_20260731_ori/05_hand_raise_take1": "hand_raise",
    "pose_capture_20260731_pm/01_arms_folded_chest_take1": "arms_folded_chest",
    "pose_capture_20260731_pm/02_arms_folded_waist_take1": "arms_folded_waist",
    "pose_capture_20260731_pm/03_run_ready_take1": "run_ready",
    "pose_capture_20260731_pm/04_pullup_take1": "pullup",
    "pose_capture_20260731_pm/05_bow_30_take1": "bow_30",
    "pose_capture_20260731_pm/06_stand_upright_take1": "stand_upright",
    "pose_capture_20260731_pm/07_look_up_45_take1": "look_up_45",
}
NO_WRIST_TRUTH = {"raise_high", "front_horizontal", "lateral_horizontal",
                  "arms_down", "hand_raise", "arms_folded_waist"}
SMPL = {"shoulder": {"left": 16, "right": 17},
        "elbow": {"left": 18, "right": 19},
        "wrist": {"left": 20, "right": 21}}


def _human_chest(b24, waist):
    """Operator chest frame (x fwd, y left, z up) from shoulders + waist.

    Built from POSITIONS, not from body24's own spine rotation: PICO's skeleton
    is a template FK fit, and the shoulder positions are the part that is
    pinned by observation (upper-arm bone length CV 0.1%) while a joint's own
    rotation is inferred.
    """
    sl = process_xr_pose(b24[SMPL["shoulder"]["left"]], waist)[:3, 3]
    sr = process_xr_pose(b24[SMPL["shoulder"]["right"]], waist)[:3, 3]
    origin = 0.5 * (sl + sr)
    y = sl - sr
    y = y / max(np.linalg.norm(y), 1e-9)
    z = origin / max(np.linalg.norm(origin), 1e-9)   # waist -> shoulders = spine
    x = np.cross(y, z)
    x = x / max(np.linalg.norm(x), 1e-9)
    return origin, np.column_stack([x, y, np.cross(x, y)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--law", choices=["waist-abs", "shoulder-abs", "shoulder-rel",
                                      "chest-rel", "chest-rel-norm"],
                    default="waist-abs")
    ap.add_argument("--bind-up", type=float, default=0.20,
                    help="waist-abs bind height offset above chest-0.25")
    ap.add_argument("--pos-scale", type=float, default=1.0)
    ap.add_argument("--ori-cost", type=float, default=2.0)
    ap.add_argument("--pos-cost", type=float, default=8.0)
    ap.add_argument("--settle", type=int, default=300,
                    help="solver iterations per pose; Pink is differential, so "
                         "each pose must be driven to rest, not sampled once")
    args = ap.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    authored = json.loads((root / "logs/robot_action_poses.json").read_text())

    u = yourdfpy.URDF.load(str(IK_URDF), load_meshes=False, build_scene_graph=True)
    aj = [j for j, jo in u.joint_map.items() if jo.type != "fixed"]
    u.update_cfg(np.array([TORSO_HOME.get(j, 0.0) for j in aj]))
    link = lambda n: np.array(u.get_transform(n, u.base_link))
    chest_T = link("arm_center")
    chest_pos = chest_T[:3, 3]
    cq = R.from_matrix(chest_T[:3, :3]).as_quat()
    chest_wxyz = np.array([cq[3], *cq[:3]])
    bind = chest_pos + np.array([0.0, 0.0, -0.25 + args.bind_up])
    rob_sh = {h: link(f"vega_1_{h[0].upper()}_arm_l1")[:3, 3]
              for h in ("right", "left")}
    rob_arm = float(np.linalg.norm(link("R_ee")[:3, 3] - rob_sh["right"]))

    print(f"[law] {args.law}  bind_up={args.bind_up}  pos_scale={args.pos_scale}")
    print(f"{'pose':22s} {'EE位置':>9} {'EE朝向':>9} {'关节 中位/最大':>16}   备注")
    pos_all, ori_all, jnt_all = [], [], []
    for stem, slug in PAIRS.items():
        path = root / "logs" / f"{stem}.msgpack"
        if not path.exists() or slug not in authored:
            continue
        rows = [m for m in msgpack.Unpacker(open(path, "rb"), raw=False)
                if (m.get("trackers") or {}).get("WAIST") is not None]
        if len(rows) < 30:
            continue
        mid = rows[len(rows) // 3:2 * len(rows) // 3]
        waist = np.mean([np.asarray(m["trackers"]["WAIST"], float) for m in mid], 0)
        b24 = np.mean([np.asarray(m["body24"], float) for m in mid], 0) \
            if mid[0].get("body24") else None

        # --- target from the human, per the law under test -------------------
        tgt, ori = {}, {}
        for h, k in (("right", "RWRIST"), ("left", "LWRIST")):
            T = np.mean([process_xr_pose(np.asarray(m["trackers"][k], float), waist)
                         for m in mid], 0)
            p_w = T[:3, 3]
            if args.law in ("chest-rel", "chest-rel-norm") and b24 is not None:
                # the operator's 2026-08-02 decoupling: the arm reads the hand
                # relative to the CHEST, in the chest frame, so neither the
                # operator's own waist bend nor the robot's TORSO_HOME enters
                # the arm target at all. Orientation goes through the same
                # frame - leaving it absolute is what made shoulder-abs and
                # shoulder-rel break the 10 deg orientation line.
                h_org, h_R = _human_chest(b24, waist)
                k_ = args.pos_scale
                if args.law == "chest-rel-norm":
                    sh_ = process_xr_pose(b24[SMPL["shoulder"][h]], waist)[:3, 3]
                    el_ = process_xr_pose(b24[SMPL["elbow"][h]], waist)[:3, 3]
                    k_ = rob_arm / max(np.linalg.norm(el_ - sh_)
                                       + np.linalg.norm(p_w - el_), 1e-3)
                tgt[h] = chest_pos + chest_T[:3, :3] @ (
                    h_R.T @ (p_w - h_org)) * k_
            elif args.law == "waist-abs" or b24 is None:
                tgt[h] = bind + p_w * args.pos_scale
            else:
                sh = process_xr_pose(b24[SMPL["shoulder"][h]], waist)[:3, 3]
                el = process_xr_pose(b24[SMPL["elbow"][h]], waist)[:3, 3]
                hl = np.linalg.norm(el - sh) + np.linalg.norm(p_w - el)
                tgt[h] = rob_sh[h] + (rob_arm / max(hl, 1e-3)) * (p_w - sh)
            M4 = np.eye(4)
            M4[:3, :3] = T[:3, :3] @ PALM_FIX[h]
            if args.law in ("chest-rel", "chest-rel-norm") and b24 is not None:
                M4[:3, :3] = chest_T[:3, :3] @ _human_chest(b24, waist)[1].T \
                    @ T[:3, :3] @ PALM_FIX[h]
            ori[h] = mat_to_pos_quat_wxyz(M4)[1]
        if args.law == "shoulder-rel" and b24 is not None:
            hw = {h: process_xr_pose(b24[SMPL["wrist"][h]], waist)[:3, 3]
                  for h in ("right", "left")}
            e = (hw["left"] - hw["right"]) - (tgt["left"] - tgt["right"])
            tgt["left"], tgt["right"] = tgt["left"] + 0.5 * e, tgt["right"] - 0.5 * e

        ik = PinkVegaIK(urdf_path=str(IK_URDF),
                        position_cost=args.pos_cost,
                        orientation_cost=args.ori_cost)
        for _ in range(args.settle):
            for h in ("right", "left"):
                ik.set_target_chest(h, chest_pos, chest_wxyz, tgt[h], ori[h])
            ik.solve()
        ik.refresh_fk()
        q_got = dict(zip(ik.pin_names, ik.config.q))

        # --- compare against what the operator authored ----------------------
        ref = authored[slug]["joints"]
        u.update_cfg(np.array([ref.get(j, TORSO_HOME.get(j, 0.0)) for j in aj]))
        # The chest must come from the AUTHORED configuration, not from
        # TORSO_HOME: bow_30 and friends author a different torso, and holding
        # the chest fixed while the arms move with it turns a torso difference
        # into a fake 400 mm arm error.
        ref_chest = link("arm_center")
        dpos, dori = [], []
        for h in ("right", "left"):
            want = link(f"{h[0].upper()}_ee")
            want_rel_p = ref_chest[:3, :3].T @ (want[:3, 3] - ref_chest[:3, 3])
            want_rel_R = ref_chest[:3, :3].T @ want[:3, :3]
            got = ik.data.oMf[ik.model.getFrameId(EE_FRAME[h])]
            got_rel_p = ik._chest.rotation.T @ (got.translation - ik._chest.translation)
            got_rel_R = ik._chest.rotation.T @ got.rotation
            dpos.append(np.linalg.norm(want_rel_p - got_rel_p) * 1000)
            dori.append(np.degrees(
                R.from_matrix(want_rel_R.T @ got_rel_R).magnitude()))
        # wrap to the shorter angle - a raw subtraction reports 256 deg for a
        # pair that is really 104 deg apart
        dj = np.degrees(np.abs([(q_got[n] - ref.get(n, 0.0) + np.pi) % (2 * np.pi) - np.pi
                                for n in ARM_JOINTS]))
        note = "腕无真值" if slug in NO_WRIST_TRUTH else ""
        pos_all += dpos
        jnt_all.append(dj)
        if slug not in NO_WRIST_TRUTH:
            ori_all += dori
        print(f"{slug:22s} {np.mean(dpos):7.0f}mm "
              + (f"{np.mean(dori):7.0f}°" if not note else "      -")
              + f" {np.median(dj):7.0f}°/{dj.max():5.0f}°   {note}")
    print(f"\n{'ALL':22s} {np.mean(pos_all):7.0f}mm "
          + (f"{np.mean(ori_all):7.0f}°" if ori_all else "      -")
          + f" {np.median(np.concatenate(jnt_all)):7.0f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
