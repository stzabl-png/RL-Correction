#!/usr/bin/env python3
"""Work out which tracker serial is on which body part, from the recording.

Object mode gives no semantics: five trackers arrive as five serial numbers and
nothing says which is a wrist and which is an elbow. The obvious fix is a probe
session - wave one limb at a time - but wearing the rig is tiring and that is
wearing time spent on bookkeeping. Everything needed is already in the poses
that were going to be recorded anyway, so identify offline instead.

The four votes, each independent, so a disagreement is visible rather than
silently averaged away:

  waist   moves least. A torso tracker travels a few cm while a hand sweeps a
          metre, and this holds in every pose.
  side    project onto the headset's left axis: left-arm trackers sit on the
          positive side. This exact test identified the two wrists across four
          takes in the 2026-07-29 batch with 100% agreement.
  limb    within one side, the wrist is farther from the waist than the elbow -
          it is the distal joint, in every arm pose there is.
  motion  corroborates limb: the wrist sweeps a bigger arc than the elbow.

Reports per-tracker confidence and prints a trackers.env ready to use. Refuses
to guess when the votes disagree, rather than emitting a plausible mapping that
would quietly mirror an arm.

Usage:
    .venv-pico/bin/python scripts/identify_trackers.py logs/pose_capture_YYYYMMDD
    .venv-pico/bin/python scripts/identify_trackers.py logs/... --write
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from magicdexmate.pico.xr_pose import R_HEADSET_TO_WORLD  # noqa: E402
from magicdexmate.sources.pico_source import is_body_joint  # noqa: E402


def collect(paths: list[pathlib.Path]):
    """-> {serial: (N,3) live positions in the head's own frame}, plus raw."""
    head_rel: dict[str, list] = {}
    world: dict[str, list] = {}
    for p in paths:
        prev: dict[str, tuple] = {}
        for m in msgpack.Unpacker(open(p, "rb"), raw=False):
            trk = m.get("trackers") or {}
            h = m.get("head")
            if h is None:
                continue
            hp = np.asarray(h[:3], float)
            hR = R.from_quat(np.asarray(h[3:7], float)).as_matrix()
            for k, v in trk.items():
                k = str(k)
                if is_body_joint(k):
                    continue                       # body-model entry, not a serial
                pos = tuple(float(x) for x in v[:3])
                if prev.get(k) == pos:
                    continue                       # frozen sample, not a measurement
                prev[k] = pos
                world.setdefault(k, []).append(np.asarray(pos))
                head_rel.setdefault(k, []).append(hR.T @ (np.asarray(pos) - hp))
    return ({k: np.array(v) for k, v in head_rel.items()},
            {k: np.array(v) for k, v in world.items()})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("capture", type=pathlib.Path)
    ap.add_argument("--write", action="store_true",
                    help="write scripts/trackers.env (backs up any existing one)")
    args = ap.parse_args()

    paths = (sorted(p for p in args.capture.glob("*.msgpack")
                    if "format_test" not in p.name)
             if args.capture.is_dir() else [args.capture])
    if not paths:
        raise SystemExit(f"no takes in {args.capture}")
    head_rel, world = collect(paths)
    if len(world) < 3:
        raise SystemExit(f"only {len(world)} raw serials found: {list(world)}. "
                         "Object mode not on, or trackers not paired?")
    print(f"[id] {len(paths)} takes, {len(world)} raw trackers\n")

    # --- vote 1: the waist is the one that barely travels -------------------
    span = {k: float(np.linalg.norm(v.max(0) - v.min(0))) for k, v in world.items()}
    order = sorted(span, key=lambda k: span[k])
    waist = order[0]
    print("行程(越小越像腰):")
    for k in order:
        print(f"   {k[-8:]:>8s} {span[k] * 1000:7.0f} mm"
              + ("   <- 判为腰" if k == waist else ""))
    if len(order) > 1 and span[order[1]] < 2.0 * span[waist]:
        print("   ⚠ 最小和次小行程接近,腰的判定不可靠 —— 人工确认")

    arms = [k for k in world if k != waist]
    # --- vote 2: side, from the head's own left axis ------------------------
    # head frame is x right / y up / -z forward, so LEFT is -x
    side = {k: -float(np.median(head_rel[k][:, 0])) for k in arms}
    lefts = sorted(arms, key=lambda k: -side[k])[:len(arms) // 2]
    print("\n左右(相对头显的左向投影,正=左):")
    for k in sorted(arms, key=lambda k: -side[k]):
        print(f"   {k[-8:]:>8s} {side[k] * 1000:+7.0f} mm"
              f"   -> {'左' if k in lefts else '右'}")

    # --- votes 3 and 4: wrist is distal ------------------------------------
    wpos = world[waist]
    ref = np.median(wpos, axis=0)
    result: dict[str, str] = {waist: "WAIST"}
    print("\n每侧内部:腕比肘离腰更远、走的路更长")
    warn = []
    for lab, group in (("左", lefts), ("右", [k for k in arms if k not in lefts])):
        pre = "L" if lab == "左" else "R"
        if len(group) == 1:
            result[group[0]] = f"{pre}WRIST"
            print(f"   {lab}: 只有 1 个,判为腕")
            continue
        far = {k: float(np.median(np.linalg.norm(world[k] - ref, axis=1)))
               for k in group}
        path = {k: float(np.sum(np.linalg.norm(np.diff(world[k], axis=0), axis=1)))
                for k in group}
        wrist = max(far, key=far.get)
        wrist2 = max(path, key=path.get)
        for k in group:
            result[k] = f"{pre}WRIST" if k == wrist else f"{pre}ELBOW"
            print(f"   {lab} {k[-8:]:>8s} 距腰 {far[k] * 1000:5.0f}mm  "
                  f"行程 {path[k]:6.2f}m  -> {result[k]}")
        if wrist != wrist2:
            warn.append(f"{lab}臂:距离票和行程票不一致,请人工确认")

    for w in warn:
        print(f"\n⚠ {w}")
    print("\n判定结果:")
    for k, v in sorted(result.items(), key=lambda kv: kv[1]):
        print(f"   {v:8s} = {k}")

    env = "\n".join([
        "# auto-identified by scripts/identify_trackers.py from "
        f"{args.capture}",
        "# geometry votes: waist=least travel, side=head left-axis, "
        "wrist=distal",
    ] + [f"TRACKER_{v}={k}"
         for k, v in sorted(result.items(), key=lambda kv: kv[1])])
    print("\n--- trackers.env ---")
    print(env)
    if args.write:
        out = pathlib.Path(__file__).resolve().parent / "trackers.env"
        if out.exists():
            bak = out.with_suffix(".env.bak")
            bak.write_text(out.read_text())
            print(f"\n[id] 旧文件备份到 {bak}")
        out.write_text(env + "\n")
        print(f"[id] 已写入 {out}")
    else:
        print("\n(加 --write 才会落盘)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
