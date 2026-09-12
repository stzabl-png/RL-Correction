#!/usr/bin/env python3
"""Splice capture segments into one continuous take for review playback.

Judging teleop quality means watching a performance, not five disconnected
clips - the 2026-07-28 note is explicit that the standard is the WHOLE run and
the whole arm. Labelled pose captures are recorded one pose at a time, so this
stitches them back into a single stream the replay consumer plays end to end.

Two things have to be handled at a seam:

  timestamps  each capture starts near t=0, so they are re-based onto one
              monotonically increasing clock (both the wall clock `t_us` the
              replay phase-locks to, and `device_ts_ns`, which the stall
              detector watches).

  the jump    consecutive segments end and start in different poses. Butted
              together that is a teleport: the glitch gate rejects it, holds,
              and re-anchors ~0.4 s later - the seam ends up dominating the
              very thing you sat down to watch. So a short bridge is
              interpolated across it (positions lerp, rotations slerp).
              THE BRIDGE IS SYNTHETIC - it is transition filler, never
              evidence. Use --bridge 0 for a strictly-recorded stream.

Usage
-----
  python scripts/make_playlist.py OUT.msgpack IN1.msgpack IN2.msgpack ...
  python scripts/make_playlist.py OUT.msgpack IN*.msgpack --bridge 0
"""
from __future__ import annotations

import argparse
import pathlib

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

DT_US = 10_000          # 100 Hz, the producer's stream rate


def _load(path: pathlib.Path) -> list[dict]:
    with open(path, "rb") as f:
        return list(msgpack.Unpacker(f, raw=False))


def _pose_keys(msg: dict) -> dict[str, list]:
    """Every 7-vector pose in a frame, flattened to one dict for interpolation."""
    out = {}
    for k in ("head", "left", "right"):
        v = msg.get(k)
        if v is not None and len(v) >= 7:
            out[k] = v
    for k, v in (msg.get("trackers") or {}).items():
        if v is not None and len(v) >= 7:
            out[f"trackers/{k}"] = v
    return out


def _lerp_pose(a: list, b: list, u: float) -> list:
    pa, pb = np.asarray(a[:3], float), np.asarray(b[:3], float)
    pos = (1.0 - u) * pa + u * pb
    qa, qb = np.asarray(a[3:7], float), np.asarray(b[3:7], float)
    if np.allclose(qa, 0) or np.allclose(qb, 0):
        quat = qb if u > 0.5 else qa
    else:
        rots = R.from_quat(np.stack([qa / np.linalg.norm(qa),
                                     qb / np.linalg.norm(qb)]))
        quat = Slerp([0.0, 1.0], rots)([u])[0].as_quat()
    return [float(v) for v in (*pos, *quat)]


def _bridge(last: dict, first: dict, n: int) -> list[dict]:
    """n synthetic frames easing from one segment's end into the next's start."""
    pa, pb = _pose_keys(last), _pose_keys(first)
    shared = [k for k in pa if k in pb]
    frames = []
    for i in range(1, n + 1):
        u = i / (n + 1.0)
        u = u * u * (3.0 - 2.0 * u)             # smoothstep: no velocity step
        m = {k: v for k, v in first.items() if k != "trackers"}
        trk = {}
        for k in shared:
            p = _lerp_pose(pa[k], pb[k], u)
            if k.startswith("trackers/"):
                trk[k.split("/", 1)[1]] = p
            else:
                m[k] = p
        if trk:
            m["trackers"] = trk
        m["_bridge"] = True                      # marks fabricated frames
        frames.append(m)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outfile", type=pathlib.Path)
    ap.add_argument("infiles", nargs="+", type=pathlib.Path)
    ap.add_argument("--bridge", type=float, default=0.6,
                    help="seconds of interpolated transition between segments "
                         "(0 = butt them together, recorded frames only)")
    args = ap.parse_args()

    n_bridge = int(round(args.bridge * 1e6 / DT_US))
    out: list[dict] = []
    t_us = 0
    dev = 1_000_000_000
    spans = []

    for idx, path in enumerate(args.infiles):
        msgs = _load(path)
        if not msgs:
            print(f"[skip] {path.name}: empty")
            continue
        if out and n_bridge > 0:
            for m in _bridge(out[-1], msgs[0], n_bridge):
                t_us += DT_US
                dev += DT_US * 1000
                m["t_us"], m["device_ts_ns"] = t_us, dev
                out.append(m)
        start_s = t_us / 1e6
        for m in msgs:
            t_us += DT_US
            dev += DT_US * 1000
            m["t_us"], m["device_ts_ns"] = t_us, dev
            out.append(m)
        spans.append((path.stem, start_s, t_us / 1e6))
        print(f"[seg] {idx + 1}. {path.stem:38s} {start_s:6.1f}s -> {t_us / 1e6:6.1f}s"
              f"  ({len(msgs)} frames)")

    if not out:
        raise SystemExit("nothing to write")
    with open(args.outfile, "wb") as f:
        for m in out:
            f.write(msgpack.packb(m, use_bin_type=True))

    n_fab = sum(1 for m in out if m.get("_bridge"))
    print(f"\n[out] {args.outfile}")
    print(f"[out] {len(out)} frames, {t_us / 1e6:.1f}s @ {1e6 / DT_US:.0f}Hz"
          f"  ({n_fab} bridge frames = {100.0 * n_fab / len(out):.1f}% synthetic)")
    print("[out] segment timeline:")
    for name, a, b in spans:
        print(f"       {a:6.1f}s - {b:6.1f}s  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
