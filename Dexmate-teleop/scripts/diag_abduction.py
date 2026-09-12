#!/usr/bin/env python
"""Abduction (MCP_AA) sensitivity probe — offline, no glove, no sim.

The mock glove barely abducts, so diag_retarget.py cannot measure the "small
side-swing deadzone" (problem A2). This probe injects a KNOWN abduction angle
into one finger of a synthetic open hand and reports how much the robot's
MCP_AA joint actually moves in response — i.e. the input->output gain.

Method: take the mock open-hand MANO keypoints, pick a finger, rotate its
MCP/PIP/DIP/TIP about the palm-normal axis through the MCP (pure abduction) by a
sweep of angles, retarget each, and read the resulting robot MCP_AA (SDK order).

Reads:
  - gain = d(robot_MCP_AA) / d(human_abduction). Ideal ~1.0 (rad/rad in-plane).
    Current position-only config is expected to be ~0 for small angles (deadzone).
  - the response curve, so we can see where the deadzone ends.

Usage (teleop venv):
  .venv/bin/python scripts/diag_abduction.py --hand right --fingers index,middle,ring,pinky
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value  # noqa: E402
from magicdexmate.retarget.frames import to_mano  # noqa: E402
from magicdexmate.retarget.mapping import JointMapper  # noqa: E402
from magicdexmate.sources.mock_source import MockGloveSource  # noqa: E402

# MediaPipe/MANO finger keypoint quads (MCP, PIP, DIP, TIP)
_FINGER_IDS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}


def _rot_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation matrix for `angle` rad about unit `axis`."""
    a = axis / (np.linalg.norm(axis) + 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    x, y, z = a
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def _palm_normal(mano: np.ndarray) -> np.ndarray:
    """Palm-plane normal from wrist(0), index_MCP(5), pinky_MCP(17)."""
    v1 = mano[5] - mano[0]
    v2 = mano[17] - mano[0]
    n = np.cross(v1, v2)
    return n / (np.linalg.norm(n) + 1e-12)


def abduct_finger(mano: np.ndarray, finger: str, angle: float) -> np.ndarray:
    """Rotate one finger's MCP..TIP about the palm normal through its MCP."""
    out = mano.copy()
    mcp, pip, dip, tip = _FINGER_IDS[finger]
    axis = _palm_normal(mano)
    pivot = out[mcp].copy()
    R = _rot_about_axis(axis, angle)
    for idx in (mcp, pip, dip, tip):
        out[idx] = pivot + R @ (out[idx] - pivot)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hand", choices=["right", "left"], default="right")
    p.add_argument("--fingers", default="index,middle,ring,pinky")
    p.add_argument("--angles-deg", default="0,2,5,8,10,15",
                   help="abduction angles to inject (deg)")
    p.add_argument("--warm", action="store_true",
                   help="warm-start incrementally (replicates teleop: last_qpos + norm_delta "
                        "regularization). Without it, each angle is solved fresh (optimistic).")
    p.add_argument("--base", default="open", choices=["open", "fist"],
                   help="base motion the abduction is injected on")
    p.add_argument("--base-t", type=float, default=0.0,
                   help="sample time for the base motion; for 'fist': ~1.0 = half-curl (~39deg PIP), "
                        "~2.0 = near-full fist (~78deg PIP). 0.0 = open.")
    args = p.parse_args()

    retargeting = build_sharpa_retargeting(args.hand, "vector")
    mapper = JointMapper(retargeting, args.hand)
    sdk_names = [n.replace(args.hand + "_", "") for n in mapper.sdk_names]

    # Synthetic base hand from the mock source (open, or half-curled 'fist').
    src = MockGloveSource(hand=args.hand, motion=args.base, noise=0.0)
    mano0 = to_mano(src.sample_at(args.base_t).kp, args.hand)

    angles = [float(a) for a in args.angles_deg.split(",")]

    mode = "WARM (teleop-faithful: warm-start + norm_delta reg)" if args.warm \
        else "FRESH (reset each angle: optimistic upper bound)"
    print(f"[cfg] hand={args.hand}  base={args.base}@{args.base_t}  fingers={args.fingers}  mode={mode}")
    print("Response: robot MCP_AA (deg) vs injected human abduction (deg). "
          "gain≈slope near 0; ~0 = DEADZONE.\n")

    for finger in args.fingers.split(","):
        finger = finger.strip()
        aa_name = f"{finger}_MCP_AA"
        row = []
        base = None
        # WARM mode: one finger per fresh sweep, warm-started 0->max incrementally.
        if args.warm:
            retargeting.reset()
        for ang in angles:
            mano = abduct_finger(mano0, finger, np.radians(ang))
            if not args.warm:
                retargeting.reset()  # each angle judged fresh (optimistic)
            ref = compute_ref_value(retargeting, mano)
            q = retargeting.retarget(ref)
            q_sdk = mapper.to_sdk(q)
            aa = float(q_sdk[sdk_names.index(aa_name)])
            if base is None:
                base = aa
            row.append(np.degrees(aa - base))
        small = next((a for a in angles if a > 0), angles[-1])
        gain = row[angles.index(small)] / small if small else 0.0
        curve = "  ".join(f"{a:>4.0f}->{r:+5.1f}" for a, r in zip(angles, row))
        print(f"  {finger:7s} MCP_AA Δ(deg): {curve}    small-angle gain={gain:+.2f}")


if __name__ == "__main__":
    main()
