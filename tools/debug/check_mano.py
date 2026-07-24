#!/usr/bin/env python3
"""Compare MANO params before/after smoothing: jitter and drift.

Usage:
    python tools/debug/check_mano.py \
        --before output/debug/2_coord_unify/mano.npz \
        --after output/debug/3_smooth/mano.npz
"""
import argparse
import numpy as np


def check(before_path, after_path):
    before = np.load(before_path)
    after = np.load(after_path)

    print(f"Before: {before_path}")
    print(f"After:  {after_path}\n")

    for h, side in enumerate(["left", "right"]):
        valid = before["valid"][h].astype(bool)
        n = valid.sum()
        if n < 3:
            print(f"  {side}: skip ({n} valid)")
            continue

        t_b = before["trans"][h][valid]
        t_a = after["trans"][h][valid]

        jitter_b = np.mean(np.linalg.norm(np.diff(t_b, axis=0), axis=1))
        jitter_a = np.mean(np.linalg.norm(np.diff(t_a, axis=0), axis=1))
        reduction = (1 - jitter_a / jitter_b) * 100 if jitter_b > 0 else 0
        drift = np.mean(np.linalg.norm(t_a - t_b, axis=1))

        print(f"  {side}: jitter {jitter_b:.4f} -> {jitter_a:.4f} "
              f"({reduction:+.1f}%)  drift={drift:.4f}m")

        # Rotation check
        r_b = before["rot"][h][valid]
        r_a = after["rot"][h][valid]
        rot_diff = np.mean(np.linalg.norm(r_a - r_b, axis=1))
        print(f"         rot_diff={rot_diff:.4f} rad")

    print("\n  Smooth check done")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    check(p.parse_args().before, p.parse_args().after)
PYEOF'