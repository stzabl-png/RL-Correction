#!/usr/bin/env python3
"""Solve the tracker->robot mapping from operator-authored pose pairs.

The mapping used to be guessed - a bind point picked from anatomy, a scale
assumed to be 1:1, a yaw fitted from mirror symmetry - and then judged by how
the result looked. This inverts that. The operator poses the robot the way each
static action ought to look (scripts/viser_joint_jog.py -> robot_action_poses
.json); pairing those with the tracker capture of the same action turns the
mapping into a fit with a ground truth that can actually disagree with it.

Model, in robot base axes (x forward, y left, z up) on both sides:

    ee_rel_chest  =  s * Rz(theta) * wrist_rel_ref  +  t

  theta  the operator's body yaw vs the capture frame - absorbs the waist
         tracker's unknown mounting yaw
  s      human-to-robot reach ratio
  t      the robot-side bind point, expressed relative to the chest. This is
         the parameter that was hand-picked as "chest - 0.25 + 0.20"; here it
         is solved for. It also absorbs any CONSTANT error in the human-side
         reference, which is why the reference need not be a perfect waist.

Since the operator held head and waist still across all six static poses (user,
2026-07-30), one fixed reference pose is used for every action rather than a
per-frame one. That removes reference drift from the fit entirely.

Usage:
    .venv/bin/python scripts/fit_mapping_from_poses.py
    .venv/bin/python scripts/fit_mapping_from_poses.py --capture logs/pose_capture_20260729_raw
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import msgpack
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from magicdexmate.actions import STATIC_ACTIONS  # noqa: E402
from magicdexmate.pico.xr_pose import (  # noqa: E402
    R_HEADSET_TO_WORLD, mat_to_pos_quat_wxyz, process_xr_pose)

# body-model channel names, and the raw-channel serials (scripts/trackers.env)
MODEL = {"left": "LWRIST", "right": "RWRIST"}
RAW = {"left": "PC2310MLL4151662G", "right": "PC2310MLL4151712G"}


def frames(path: pathlib.Path):
    with open(path, "rb") as f:
        yield from msgpack.Unpacker(f, raw=False)


def live_rows(path: pathlib.Path, keys: dict[str, str]):
    """Frames where every wanted tracker is present AND changed since the last
    one. The channels freeze bit-identically when a tracker leaves the headset
    FOV; those repeats are not measurements and must not weight a median."""
    out, prev = [], None
    for m in frames(path):
        trk = m.get("trackers") or {}
        if m.get("head") is None or not all(k in trk for k in keys.values()):
            continue
        sig = tuple(tuple(trk[k]) for k in keys.values())
        if sig == prev:
            continue
        prev = sig
        out.append((np.asarray(m["head"], float),
                    {h: np.asarray(trk[k], float) for h, k in keys.items()}))
    return out


def take_files(cap: pathlib.Path, slug: str) -> list[pathlib.Path]:
    return sorted(cap.glob(f"*_{slug}_take*.msgpack"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poses", type=pathlib.Path,
                    default=pathlib.Path("logs/robot_action_poses.json"))
    ap.add_argument("--capture", type=pathlib.Path,
                    default=pathlib.Path("logs/pose_capture_20260729"))
    ap.add_argument("--channel", choices=["model", "raw"], default=None,
                    help="which wrist channel the capture carries "
                         "(default: detect from the first file)")
    ap.add_argument("--max-spread", type=float, default=120.0,
                    help="drop an action whose wrist samples scatter more than "
                         "this [mm] - that is a wandering estimate, not a held "
                         "pose, and it would poison the fit")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="write the fitted parameters here as JSON")
    args = ap.parse_args()

    robot = json.loads(args.poses.read_text())
    print(f"[fit] robot poses: {len(robot)} -> {', '.join(sorted(robot))}")

    probe = sorted(args.capture.glob("*_take*.msgpack"))
    if not probe:
        raise SystemExit(f"no captures in {args.capture}")
    keys_present = set()
    for m in frames(probe[0]):
        keys_present = set((m.get("trackers") or {}).keys())
        break
    channel = args.channel or ("raw" if RAW["left"] in keys_present else "model")
    keys = RAW if channel == "raw" else MODEL
    print(f"[fit] capture: {args.capture}  channel={channel}  "
          f"({', '.join(keys.values())})")

    # ---- one fixed reference for every action (head/waist held still) -------
    heads = []
    for _, slug, _ in STATIC_ACTIONS:
        for f in take_files(args.capture, slug):
            heads += [h for h, _ in live_rows(f, keys)]
    if not heads:
        raise SystemExit("no live frames in any static take")
    heads = np.array(heads)
    ref_pos = np.median(heads[:, :3], axis=0)
    yaws = [R.from_matrix(R_HEADSET_TO_WORLD
                          @ R.from_quat(h[3:7]).as_matrix()
                          @ R_HEADSET_TO_WORLD.T).as_euler("xyz")[2]
            for h in heads]
    ref_yaw = float(np.median(yaws))
    # express that heading back in the raw frame the poses live in
    M = R_HEADSET_TO_WORLD
    ref_rot = R.from_matrix(M.T @ R.from_euler("z", ref_yaw).as_matrix() @ M)
    ref_pose7 = np.concatenate([ref_pos, ref_rot.as_quat()])
    print(f"[fit] fixed reference: pos {np.round(ref_pos, 3).tolist()}, "
          f"heading {np.degrees(ref_yaw):+.1f}deg, from {len(heads)} live frames")

    # ---- human side: one wrist vector per action ---------------------------
    human, quality = {}, {}
    for label, slug, _ in STATIC_ACTIONS:
        rows = []
        for f in take_files(args.capture, slug):
            rows += live_rows(f, keys)
        if len(rows) < 20:
            quality[slug] = f"only {len(rows)} live frames"
            continue
        per_hand, spreads = {}, {}
        for h in ("right", "left"):
            P = np.array([mat_to_pos_quat_wxyz(process_xr_pose(t[h], ref_pose7))[0]
                          for _, t in rows])
            med = np.median(P, axis=0)
            spreads[h] = float(np.median(np.linalg.norm(P - med, axis=1)) * 1000)
            per_hand[h] = med
        worst = max(spreads.values())
        quality[slug] = (f"{len(rows)} frames, spread R {spreads['right']:.0f} / "
                         f"L {spreads['left']:.0f} mm")
        if worst > args.max_spread:
            quality[slug] += "  -> DROPPED (not a held pose)"
            continue
        human[slug] = per_hand

    print("\n[fit] human-side data quality per action:")
    for _, slug, _ in STATIC_ACTIONS:
        mark = "ok  " if slug in human else "SKIP"
        print(f"   {mark} {slug:20s} {quality.get(slug, 'no takes')}")

    pairs = []
    for slug, hh in human.items():
        rp = robot.get(slug)
        if rp is None:
            print(f"[fit] no authored robot pose for {slug} - skipped")
            continue
        for h in ("right", "left"):
            pairs.append((slug, h, hh[h],
                          np.asarray(rp["ee_minus_chest_base_axes"][h], float)))
    if len(pairs) < 4:
        raise SystemExit(f"only {len(pairs)} usable pairs - cannot fit")
    X = np.array([p[2] for p in pairs])
    Y = np.array([p[3] for p in pairs])
    print(f"\n[fit] {len(pairs)} pairs from {len(human)} actions")

    def model(p, X, iso=True):
        th, s = p[0], p[1]
        t = p[2:5]
        S = np.array([s, s, s]) if iso else np.array([p[1], p[1], p[5]])
        c, sn = np.cos(th), np.sin(th)
        Rz = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        return (Rz @ (X * S).T).T + t

    def report(name, p, iso):
        pred = model(p, X, iso)
        err = np.linalg.norm(pred - Y, axis=1) * 1000
        print(f"\n=== {name} ===")
        print(f"  yaw   {np.degrees(p[0]):+7.2f} deg")
        if iso:
            print(f"  scale {p[1]:7.3f}")
        else:
            print(f"  scale 水平 {p[1]:.3f}   竖直 {p[5]:.3f}")
        print(f"  bind  (相对胸, 基座系) {np.round(p[2:5], 4).tolist()}  "
              f"-> 竖直 {p[4]:+.3f} m")
        print(f"  残差  mean {err.mean():6.1f} mm   max {err.max():6.1f} mm")
        for (slug, h, _, _), e in zip(pairs, err):
            print(f"      {slug:20s} {h:5s} {e:7.1f} mm")
        return err

    p0 = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    r1 = least_squares(lambda p: (model(p, X) - Y).ravel(), p0)
    err1 = report("等比例拟合 (yaw, scale, bind)", r1.x, True)

    p0b = np.array([r1.x[0], r1.x[1], *r1.x[2:5], r1.x[1]])
    r2 = least_squares(lambda p: (model(p, X, False) - Y).ravel(), p0b)
    err2 = report("水平/竖直分开比例", r2.x, False)

    chest_z = 1.2239                      # arm_center height, tuck home
    print(f"\n[fit] 解出的绑定点高度 = 胸 {chest_z:.3f} {r1.x[4]:+.3f} "
          f"= {chest_z + r1.x[4]:.3f} m")
    print(f"      现行手调值        = 胸 - 0.25 + 0.20 = {chest_z - 0.05:.3f} m")
    print(f"      差 {abs(chest_z + r1.x[4] - (chest_z - 0.05)) * 1000:.0f} mm")

    if args.out:
        args.out.write_text(json.dumps({
            "capture": str(args.capture), "channel": channel,
            "ref_pos": ref_pos.tolist(), "ref_yaw_deg": float(np.degrees(ref_yaw)),
            "actions_used": sorted(human),
            "isotropic": {"yaw_deg": float(np.degrees(r1.x[0])),
                          "scale": float(r1.x[1]),
                          "bind_rel_chest": r1.x[2:5].tolist(),
                          "resid_mean_mm": float(err1.mean()),
                          "resid_max_mm": float(err1.max())},
            "anisotropic": {"yaw_deg": float(np.degrees(r2.x[0])),
                            "scale_horiz": float(r2.x[1]),
                            "scale_vert": float(r2.x[5]),
                            "bind_rel_chest": r2.x[2:5].tolist(),
                            "resid_mean_mm": float(err2.mean()),
                            "resid_max_mm": float(err2.max())},
        }, ensure_ascii=False, indent=2))
        print(f"\n[fit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
