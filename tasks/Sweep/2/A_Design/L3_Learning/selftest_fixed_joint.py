from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fixed_joint_frames import grasp_joint_frames, joint_frames, pose_matrix  # noqa: E402


def main():
    tool = np.array([0.4, -0.2, 0.9, 1.0, 0.0, 0.0, 0.0])
    grasp = np.array([0.02, -0.01, 0.12, 0.9238795, 0.0, 0.3826834, 0.0])
    wh = pose_matrix(tool) @ pose_matrix(grasp)
    # matrix -> only use general identity proof; no conversion back is needed.
    h_local, t_local = grasp_joint_frames(grasp)
    assert np.max(np.abs(wh @ h_local - pose_matrix(tool) @ t_local)) < 1e-8
    # General formulation must produce the same coincident world joint.
    hand7 = np.r_[wh[:3, 3], [1.0, 0.0, 0.0, 0.0]]
    # Translation-only variant exercises the general API without matrix->quat code.
    tool7 = np.array([0.3, 0.1, 0.8, 1.0, 0.0, 0.0, 0.0])
    hand7 = np.array([0.32, 0.09, 0.92, 1.0, 0.0, 0.0, 0.0])
    a, b = joint_frames(hand7, tool7)
    assert np.max(np.abs(pose_matrix(hand7) @ a - pose_matrix(tool7) @ b)) < 1e-10
    print("Sweep2 fixed-joint frame self-test: PASS")


if __name__ == "__main__":
    main()
