#!/usr/bin/env python3
"""Verify OpenCV -> Z-up coordinate transform correctness.

Usage:
    python tools/debug/check_coord.py \
        --opencv output/debug/1_hawor/mano.npz \
        --zup output/debug/2_coord_unify/mano.npz
"""
import argparse
import numpy as np


def check(opencv_path, zup_path):
    cv = np.load(opencv_path)
    zu = np.load(zup_path)

    print(f"OpenCV: {opencv_path}")
    print(f"Z-up:   {zup_path}\n")

    for h, side in enumerate(["left", "right"]):
        valid = cv["valid"][h].astype(bool)
        n = valid.sum()
        if n < 1:
            continue

        t_cv = cv["trans"][h][valid]  # (M, 3) OpenCV: X=right, Y=down, Z=fwd
        t_zu = zu["trans"][h][valid]  # (M, 3) Z-up: X=fwd, Y=left, Z=up

        # Expected: X_new = Z_old, Y_new = -X_old, Z_new = -Y_old
        err_x = np.max(np.abs(t_zu[:, 0] - t_cv[:, 2]))   # X_new = Z_old
        err_y = np.max(np.abs(t_zu[:, 1] - (-t_cv[:, 0]))) # Y_new = -X_old
        err_z = np.max(np.abs(t_zu[:, 2] - (-t_cv[:, 1]))) # Z_new = -Y_old

        print(f"  {side} translation transform errors:")
        print(f"    X_new = Z_old:  max_err={err_x:.6f}")
        print(f"    Y_new = -X_old: max_err={err_y:.6f}")
        print(f"    Z_new = -Y_old: max_err={err_z:.6f}")

        # Check magnitude preservation
        mag_cv = np.linalg.norm(t_cv, axis=1)
        mag_zu = np.linalg.norm(t_zu, axis=1)
        mag_err = np.max(np.abs(mag_cv - mag_zu))
        print(f"    magnitude preservation: max_err={mag_err:.6f}")

        assert err_x < 1e-4, f"FAIL: X transform error {err_x}"
        assert err_y < 1e-4, f"FAIL: Y transform error {err_y}"
        assert err_z < 1e-4, f"FAIL: Z transform error {err_z}"
        assert mag_err < 1e-4, f"FAIL: magnitude not preserved {mag_err}"
        print(f"    OK")

    print("\n  PASS: Coordinate transform correct")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--opencv", required=True)
    p.add_argument("--zup", required=True)
    check(p.parse_args().opencv, p.parse_args().zup)
PYEOF'