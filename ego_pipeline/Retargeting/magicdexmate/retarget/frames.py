"""Convert glove wrist-frame keypoints to the MANO-ish convention dex-retargeting expects.

dex-retargeting's own MediaPipe pipeline does:

    kp -= kp[wrist]
    R = estimate_frame_from_hand_points(kp)   # "operator" frame from the points themselves
    kp_mano = kp @ R @ OPERATOR2MANO[hand]

Because the operator frame is re-estimated from the keypoints every frame, the
whole transform is invariant to whatever fixed frame the input arrives in.
We reuse exactly that path for the Wuji glove, which sidesteps having to know
the glove's wrist-frame axes convention (X=radial, Z=proximal per Wuji docs)
analytically. See plans/01_retarget_plan.md §R2.
"""

from __future__ import annotations

import numpy as np

from dex_retargeting.constants import OPERATOR2MANO, HandType

_HAND_TYPE = {"right": HandType.right, "left": HandType.left}


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Estimate the wrist 'operator' frame from 21 wrist-centered keypoints.

    Copied verbatim from dex-retargeting (MIT license),
    example/vector_retargeting/single_hand_detector.py, so the runtime does not
    depend on the mediapipe package that module imports.
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    # Compute vector from palm to the first joint of middle finger
    x_vector = points[0] - points[2]

    # Normal fitting with SVD
    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)

    normal = v[2, :]

    # Gram-Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    # We assume that the vector from pinky to index is similar the z axis in MANO convention
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1
    frame = np.stack([x, normal, z], axis=1)
    return frame


def to_mano(kp: np.ndarray, hand: str) -> np.ndarray:
    """(21,3) keypoints in any consistent frame -> (21,3) in MANO convention."""
    centered = kp - kp[0:1, :]
    operator_frame = estimate_frame_from_hand_points(centered)
    return centered @ operator_frame @ OPERATOR2MANO[_HAND_TYPE[hand]]
