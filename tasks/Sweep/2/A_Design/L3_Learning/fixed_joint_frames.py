"""Fixed-joint frame math shared by the scene builder and its CPU self-test."""
from __future__ import annotations

import numpy as np


def pose_matrix(pose7):
    p = np.asarray(pose7, dtype=np.float64)
    q = p[3:7] / np.linalg.norm(p[3:7])
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, p[:3]
    return T


def joint_frames(world_hand, world_tool, world_joint=None):
    """Return local hand/tool frames whose world transforms are identical."""
    wh, wt = pose_matrix(world_hand), pose_matrix(world_tool)
    wj = wh if world_joint is None else pose_matrix(world_joint)
    return np.linalg.inv(wh) @ wj, np.linalg.inv(wt) @ wj


def grasp_joint_frames(grasp_hand_in_object):
    """Fast path when the joint anchor is the hand body origin.

    A GraspPose stores ``T_object_hand``.  Thus body0(hand) local frame is identity
    and body1(tool) local frame is exactly ``T_object_hand``.
    """
    return np.eye(4), pose_matrix(grasp_hand_in_object)
