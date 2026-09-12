#!/usr/bin/env python
"""Offline phase-1 analysis of a pose_capture session: per take, glitch-filter
the wrist/waist streams and reduce each labeled pose to a robust median
"wrist relative to waist" vector - computed with the SAME process_xr_pose
math the live consumer uses, so the numbers ARE the mapping's input.

Output: human-readable table (right/up/front per wrist per pose, symmetry
check) + JSON dump for phase 2 (Pink reachability / robot-pose replay).

Run: .venv-pico/bin/python scripts/analyze_pose_capture.py logs/pose_capture_20260729
"""
import json
import os
import sys

import msgpack
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
from magicdexmate.pico.xr_pose import process_xr_pose  # noqa: E402

JUMP_M = 0.05      # >5cm between consecutive ~100Hz frames = glitch
JUMP_DEG = 20.0    # >20deg/frame orientation = glitch


def sget(m, key):
    return m.get(key) if key in m else m.get(key.encode())


def load_frames(path):
    with open(path, "rb") as f:
        return list(msgpack.Unpacker(f))


def tracker(m, name):
    trk = sget(m, "trackers") or {}
    v = trk.get(name) if name in trk else trk.get(name.encode())
    return None if v is None else np.asarray(v, dtype=np.float64)


def clean_mask(frames, name):
    """Drop frames around physically impossible steps of this tracker."""
    P, Q = [], []
    for m in frames:
        v = tracker(m, name)
        P.append(v[:3] if v is not None else None)
        Q.append(v[3:7] if v is not None else None)
    ok = np.ones(len(frames), dtype=bool)
    for i in range(1, len(frames)):
        if P[i] is None or P[i - 1] is None:
            ok[i] = False
            continue
        if np.linalg.norm(P[i] - P[i - 1]) > JUMP_M:
            ok[i] = ok[i - 1] = False
        d = abs(float(np.dot(Q[i], Q[i - 1])))
        if 2 * np.degrees(np.arccos(min(d, 1.0))) > JUMP_DEG:
            ok[i] = ok[i - 1] = False
    return ok


def body_relative(frames, wrist, ref="WAIST"):
    """Per-frame wrist-rel-waist in the consumer's yaw-compensated frame
    (x=front, y=left, z=up), glitch-filtered."""
    ok = clean_mask(frames, wrist) & clean_mask(frames, ref)
    out = []
    for m, good in zip(frames, ok):
        if not good:
            continue
        w, r = tracker(m, wrist), tracker(m, ref)
        if w is None or r is None:
            continue
        T = process_xr_pose(w, r)
        out.append(T[:3, 3])
    return np.asarray(out), float(ok.mean())


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "logs/pose_capture_20260729"
    manifest = json.load(open(os.path.join(d, "manifest.json")))
    by_pose = {}
    for e in manifest:
        by_pose.setdefault(e["slug"], []).append(e)

    results = {}
    print(f"{'pose':<22s} {'手':>2s} {'前(x)':>7s} {'左(y)':>7s} {'上(z)':>7s}"
          f" {'takes内离散':>9s} {'净帧':>5s}")
    for slug, takes in by_pose.items():
        for wrist, tag in (("RWRIST", "右"), ("LWRIST", "左")):
            meds, keep = [], []
            for e in takes:
                fr = load_frames(os.path.join(d, e["file"]))
                rel, kept = body_relative(fr, wrist)
                if len(rel) < 50:
                    continue
                meds.append(np.median(rel, axis=0))
                keep.append(kept)
            if not meds:
                print(f"{slug:<22s} {tag:>2s}   -- 无可用帧 --")
                continue
            meds = np.asarray(meds)
            med = np.median(meds, axis=0)
            spread = float(np.max(np.linalg.norm(meds - med, axis=1))) * 1000
            results.setdefault(slug, {})[wrist] = {
                "front": float(med[0]), "left": float(med[1]),
                "up": float(med[2]), "spread_mm": spread,
                "kept_frac": float(np.mean(keep)),
            }
            print(f"{slug:<22s} {tag:>2s} {med[0]:+7.3f} {med[1]:+7.3f} "
                  f"{med[2]:+7.3f} {spread:7.0f}mm {np.mean(keep)*100:4.0f}%")

    # left/right symmetry check on the static poses (all mirror poses)
    print("\n对称性(左右腕 |front/up 差| 与 |left 和|,理想=0):")
    for slug, r in results.items():
        if "RWRIST" in r and "LWRIST" in r:
            R, L = r["RWRIST"], r["LWRIST"]
            df = abs(R["front"] - L["front"]) * 1000
            du = abs(R["up"] - L["up"]) * 1000
            dl = abs(R["left"] + L["left"]) * 1000
            print(f"  {slug:<22s} Δ前 {df:4.0f}mm  Δ上 {du:4.0f}mm  Σ左 {dl:4.0f}mm")

    out = os.path.join(d, "phase1_body_relative.json")
    json.dump(results, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"\n[phase1] 已写 {out}")


if __name__ == "__main__":
    main()
