#!/usr/bin/env python3
"""Verify ViPE output sanity: intrinsics, trajectory, depth.

Usage:
    python tools/debug/check_vipe.py --dir output/debug/0_vipe/
"""
import argparse, os, json
import numpy as np


def check(d):
    print(f"Checking ViPE output in {d}\n")

    # Intrinsics
    K = np.load(os.path.join(d, "K.npy"))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"  Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
    assert 100 < fx < 5000, f"  FAIL: fx={fx} out of range"
    assert abs(fx - fy) / fx < 0.1, f"  FAIL: fx != fy ({fx:.1f} vs {fy:.1f})"
    print("    OK: intrinsics in valid range")

    # Poses
    poses = np.load(os.path.join(d, "poses_c2w.npy"))
    N = len(poses)
    t = poses[:, :3, 3]
    travel = np.linalg.norm(np.diff(t, axis=0), axis=1).sum()
    print(f"\n  Poses: {N} frames, total travel={travel:.3f}m")
    assert travel < 100, f"  FAIL: travel too large ({travel:.1f}m)"

    # Rotation orthogonality
    R = poses[:, :3, :3]
    det = np.linalg.det(R)
    print(f"  Rotation det: mean={det.mean():.6f} std={det.std():.6f}")
    assert np.allclose(det, 1.0, atol=0.01), "  FAIL: non-orthogonal rotations"
    print("    OK: poses valid")

    # Depth stats
    depth_stats_path = os.path.join(d, "depth_stats.json")
    if os.path.exists(depth_stats_path):
        with open(depth_stats_path) as f:
            ds = json.load(f)
        print(f"\n  Depth: shape={ds['shape']} "
              f"range=[{ds['min']:.3f}, {ds['max']:.3f}]m mean={ds['mean']:.3f}m")
        assert ds["min"] >= 0, "  FAIL: negative depth"
        assert ds["max"] < 50, f"  FAIL: depth max={ds['max']:.1f}m too large"
        print("    OK: depth in valid range")

    print("\n  PASS: ViPE output OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    check(p.parse_args().dir)
PYEOF'