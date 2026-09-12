#!/usr/bin/env python3
"""Rebuild a usable WAIST reference from the headset, for raw-channel captures.

Why this exists
---------------
The 2026-07-29 hybrid rig streams both channels at once: the wrists come from
the raw per-tracker channel (clean in FOV, honest bit-level freeze out of it)
and the waist from the body model. The waist tracker is worn where the headset
cameras can never see it, so the raw waist is permanently frozen - and in the
`_raw` capture batch the model channel was frozen too (the full-body
calibration was never redone after the mode switch). That leaves clean wrists
with no usable origin, and --map waist-abs needs an origin.

The headset IS live in that batch and sits in the same coordinate generation as
the raw wrists, so a waist can be reconstructed from it. The head->waist offset
is not guessed: it is measured from the capture's own labelled poses.

  hands-on-hips: the wrists rest ON the waist, so the wrist midpoint in that
  segment IS the waist. Measured on both takes of the 2026-07-29 batch:
  0.558 / 0.571 m below the head and 0.11 / 0.13 m behind it - anatomically
  sane (eye-to-waist ~0.56 m; the head sits forward of the spine).
  Cross-check on front_horizontal: the reconstruction puts the wrists 0.44 m
  above / 0.52 m in front of the waist, consistent with the 0.57 m arm length
  measured independently on the first batch.

This is an OFFLINE ANALYSIS AID, not a rig component. It lets the mapping and
the robot motion be judged from recorded data without putting the operator back
in the headset. The real session uses the model-channel waist - reconstruct
nothing, redo the full-body calibration instead.

Usage
-----
  python scripts/repair_waist_from_head.py IN.msgpack OUT.msgpack
  python scripts/repair_waist_from_head.py IN.msgpack OUT.msgpack \
      --calib logs/.../05_hands_on_hips_take1.msgpack   # re-measure the offset
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import msgpack
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from magicdexmate.pico.xr_pose import (  # noqa: E402
    R_HEADSET_TO_WORLD, _pose7_to_world, yaw_align_from_wrists)


def _raw_heading(psi: float) -> R:
    """Rotation, expressed in the RAW frame, whose world-frame yaw is `psi`.

    The capture frame is y-up; the consumer converts with world = M @ raw and
    only then reads yaw about world z. Building the heading as a raw-z rotation
    (the obvious-looking mistake) lands as a world ROLL and compensates no yaw
    at all, so go through M explicitly.
    """
    M = R_HEADSET_TO_WORLD
    return R.from_matrix(M.T @ R.from_euler("z", psi).as_matrix() @ M)

# raw-channel wrist serials (scripts/trackers.env; identified 2026-07-29 by
# head-yaw left-axis vote, 100% agreement over 4 takes)
LEFT_SERIAL = "PC2310MLL4151662G"
RIGHT_SERIAL = "PC2310MLL4151712G"
# head -> waist in the BODY frame (XR axes: x right, y up, -z forward),
# measured from the hands-on-hips segments as documented above
DEFAULT_OFFSET = np.array([0.0, -0.565, 0.12])


def _frames(path: pathlib.Path):
    with open(path, "rb") as f:
        yield from msgpack.Unpacker(f, raw=False)


def _live_mask(arr: np.ndarray) -> np.ndarray:
    """True where the sample differs from the previous one. The raw channel
    freezes bit-identically out of FOV, so frozen stretches must not pollute
    a calibration median."""
    if len(arr) == 0:
        return np.zeros(0, dtype=bool)
    return np.array([True] + [not np.array_equal(arr[i], arr[i - 1])
                              for i in range(1, len(arr))])


def _yaw_of(pose7) -> float:
    """World-frame heading of a raw pose, read exactly the way the consumer
    reads it (`process_xr_pose`: convert to z-up first, then yaw about z)."""
    _, rot = _pose7_to_world(pose7)
    return float(R.from_matrix(rot).as_euler("xyz")[2])


def measure_offset(path: pathlib.Path) -> np.ndarray:
    """Head->waist offset from a hands-on-hips capture (wrists rest on the
    waist, so their midpoint is the waist)."""
    head, mid = [], []
    for m in _frames(path):
        trk = m.get("trackers") or {}
        if m.get("head") is None or LEFT_SERIAL not in trk or RIGHT_SERIAL not in trk:
            continue
        head.append(np.asarray(m["head"][:3], dtype=float))
        mid.append((np.asarray(trk[LEFT_SERIAL][:3], dtype=float)
                    + np.asarray(trk[RIGHT_SERIAL][:3], dtype=float)) / 2.0)
    if len(head) < 10:
        raise SystemExit(f"{path}: too few usable frames to calibrate")
    head, mid = np.array(head), np.array(mid)
    live = _live_mask(mid)
    if live.sum() < 10:
        raise SystemExit(f"{path}: wrists frozen throughout - pick a clean take")
    off = np.median(mid[live] - head[live], axis=0)
    print(f"[calib] {path.name}: {live.sum()}/{len(live)} live frames -> "
          f"head->waist offset {np.round(off, 3).tolist()}")
    return off


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infile", type=pathlib.Path)
    ap.add_argument("outfile", type=pathlib.Path)
    ap.add_argument("--calib", type=pathlib.Path, default=None,
                    help="hands-on-hips capture to re-measure the offset from "
                         "(default: the 2026-07-29 measured constant)")
    ap.add_argument("--name", default="WAIST",
                    help="tracker key to write (default WAIST, overwriting the "
                         "dead model-channel entry)")
    ap.add_argument("--live-yaw", action="store_true",
                    help="per-frame headset yaw instead of the segment median. "
                         "Off by default: a head turn is not a body turn, and "
                         "letting it rotate the mapping frame is the 2026-07-20 "
                         "hardware-day fault all over again")
    ap.add_argument("--yaw-from-wrists", action="store_true",
                    help="derive the body heading from the wrist line "
                         "(yaw_align_from_wrists) instead of the headset. The "
                         "headset yaw is NOT the body yaw - on the 2026-07-29 "
                         "front-horizontal take it put the hand pair 12 cm off "
                         "centre (~13 deg). Needs the hands apart and roughly "
                         "level, so it is opt-in per segment")
    args = ap.parse_args()

    offset = measure_offset(args.calib) if args.calib else DEFAULT_OFFSET.copy()

    msgs = list(_frames(args.infile))
    if not msgs:
        raise SystemExit(f"{args.infile}: empty")
    yaws = [_yaw_of(m["head"]) for m in msgs if m.get("head") is not None]
    if not yaws:
        raise SystemExit(f"{args.infile}: no headset pose - cannot reconstruct")
    yaw_const = float(np.median(yaws))
    yaw_src = "headset"
    if args.yaw_from_wrists:
        lp, rp = [], []
        for m in msgs:
            trk = m.get("trackers") or {}
            if LEFT_SERIAL in trk and RIGHT_SERIAL in trk:
                lp.append(np.asarray(trk[LEFT_SERIAL][:3], dtype=float))
                rp.append(np.asarray(trk[RIGHT_SERIAL][:3], dtype=float))
        lp, rp = np.array(lp), np.array(rp)
        live = _live_mask(lp) & _live_mask(rp)
        if live.sum() < 10:
            raise SystemExit("wrists frozen throughout - cannot fit body yaw")
        rot = yaw_align_from_wrists(np.median(lp[live], axis=0),
                                    np.median(rp[live], axis=0))
        # columns live in the RAW frame; take the heading in the world frame
        front = R_HEADSET_TO_WORLD @ rot[:, 0]
        yaw_wr = float(np.arctan2(front[1], front[0]))
        print(f"[yaw] wrist-line heading {np.degrees(yaw_wr):+.1f}deg vs "
              f"headset {np.degrees(yaw_const):+.1f}deg "
              f"(diff {np.degrees(yaw_wr - yaw_const):+.1f}deg)")
        yaw_const, yaw_src = yaw_wr, "wrist-line"

    n_written = 0
    for m in msgs:
        head = m.get("head")
        trk = m.get("trackers")
        if head is None or trk is None:
            continue
        yaw = _yaw_of(head) if args.live_yaw else yaw_const
        rot = _raw_heading(yaw)
        # the offset is anatomical (body frame), so it turns with the body
        pos = np.asarray(head[:3], dtype=float) + rot.apply(offset)
        q = rot.as_quat()                            # scalar-last, yaw only:
        # a level waist tracker carries heading, not the head's pitch/roll
        trk[args.name] = [float(v) for v in (*pos, *q)]
        n_written += 1

    with open(args.outfile, "wb") as f:
        for m in msgs:
            f.write(msgpack.packb(m, use_bin_type=True))

    print(f"[out] {args.outfile}  ({n_written}/{len(msgs)} frames carry a "
          f"reconstructed '{args.name}')")
    print(f"[out] yaw {'per-frame' if args.live_yaw else f'constant {np.degrees(yaw_const):+.1f}deg'}"
          f", yaw source {yaw_src}, offset {np.round(offset, 3).tolist()}")
    print("[note] reconstructed reference - for offline mapping review only; "
          "the live rig must use the model-channel waist")
    return 0


if __name__ == "__main__":
    sys.exit(main())
