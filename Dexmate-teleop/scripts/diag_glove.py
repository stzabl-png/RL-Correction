#!/usr/bin/env python
"""Live glove keypoint diagnostic: is each keypoint actually MOVING?

Connects to the glove and, over --duration seconds, tracks each keypoint's
movement RANGE (max-min per axis, mm) and confidence. Use it to tell a DEAD
sensor (e.g. the thumb barely moves while you wiggle it) from a retarget bug --
independent of any retarget config. Run it on BOTH hands and compare.

  # wiggle your THUMB and FINGERS during the run, then read the per-keypoint range
  PYTHONPATH= .venv/bin/python scripts/diag_glove.py --hand left
  PYTHONPATH= .venv/bin/python scripts/diag_glove.py --hand right   # compare

  # sanity-check the tool itself without hardware (mock keypoints move):
  PYTHONPATH= .venv/bin/python scripts/diag_glove.py --source mock --motion cycle
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from magicdexmate.skeleton import MEDIAPIPE_JOINT_NAMES  # noqa: E402
from magicdexmate.sources import make_source  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["wuji", "mock"], default="wuji")
    p.add_argument("--hand", choices=["right", "left"], default="right")
    p.add_argument("--duration", type=float, default=8.0, help="seconds to sample while you move the hand")
    p.add_argument("--sn", default=None)
    p.add_argument("--address", default=None, help="e.g. 192.168.1.100:50000")
    p.add_argument("--motion", default="cycle", help="mock only")
    args = p.parse_args()

    kwargs = {"motion": args.motion} if args.source == "mock" else {"sn": args.sn, "address": args.address}
    src = make_source(args.source, args.hand, **kwargs)
    src.start()

    print(f"[diag] {args.source} {args.hand} hand: WIGGLE your thumb AND fingers for {args.duration:.0f}s ...")
    kp_min = np.full((21, 3), np.inf)
    kp_max = np.full((21, 3), -np.inf)
    conf_min = np.full(21, np.inf)
    conf_max = np.full(21, -np.inf)
    n = 0
    t0 = time.monotonic()
    last = 0.0
    while time.monotonic() - t0 < args.duration:
        fr = src.get_latest()
        if fr is None:
            time.sleep(0.005)
            continue
        kp_min = np.minimum(kp_min, fr.kp)
        kp_max = np.maximum(kp_max, fr.kp)
        conf_min = np.minimum(conf_min, fr.conf)
        conf_max = np.maximum(conf_max, fr.conf)
        n += 1
        now = time.monotonic()
        if now - last > 0.5:
            last = now
            tt, it = fr.kp[4] * 1000, fr.kp[8] * 1000
            print(f"  live  thumb_tip=[{tt[0]:6.0f}{tt[1]:6.0f}{tt[2]:6.0f}]  "
                  f"index_tip=[{it[0]:6.0f}{it[1]:6.0f}{it[2]:6.0f}] mm  conf(thumb_tip)={fr.conf[4]:.2f}")
    src.stop()

    if n == 0:
        print("[diag] got NO frames -- glove not connected / no data. Check network + SDK.")
        return
    rng = (kp_max - kp_min) * 1000.0  # per-axis movement range in mm
    print(f"\n[diag] {n} frames. Movement RANGE per keypoint over the run (mm) + confidence:")
    print(f"  {'idx':>3} {'name':18s} {'rangeX':>7s}{'rangeY':>7s}{'rangeZ':>7s} {'|range|':>8s} {'conf':>11s}")
    thumb_mag = []
    finger_mag = []
    for i, name in enumerate(MEDIAPIPE_JOINT_NAMES):
        r = rng[i]
        mag = float(np.linalg.norm(r))
        tag = "  <== THUMB" if 1 <= i <= 4 else ""
        if 1 <= i <= 4:
            thumb_mag.append(mag)
        elif i in (8, 12, 16, 20):
            finger_mag.append(mag)
        print(f"  {i:3d} {name:18s} {r[0]:7.1f}{r[1]:7.1f}{r[2]:7.1f} {mag:8.1f} "
              f"{conf_min[i]:.2f}-{conf_max[i]:.2f}{tag}")
    tavg = np.mean(thumb_mag) if thumb_mag else 0.0
    favg = np.mean(finger_mag) if finger_mag else 0.0
    print(f"\n  thumb joints (1-4) avg |range| = {tavg:6.1f} mm")
    print(f"  fingertips (8/12/16/20) avg |range| = {favg:6.1f} mm")
    verdict = ("THUMB LOOKS DEAD (moves far less than fingers) -> glove not tracking this "
               "thumb (calibration/hardware), NOT a retarget bug."
               if favg > 1 and tavg < 0.25 * favg else
               "thumb moves comparably to fingers -> keypoints are alive; look downstream.")
    print(f"  => {verdict}")


if __name__ == "__main__":
    main()
