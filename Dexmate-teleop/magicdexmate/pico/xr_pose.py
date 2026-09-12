"""Frame math for PICO controller poses (consumer side, no SDK import).

The SDK reports poses in an OpenXR-style right-handed y-up frame:
x = right, y = up, z = backward (forward = -z), quaternions as [qx,qy,qz,qw].

`process_xr_pose` (adapted from GR00T decoupled_wbc PicoStreamer._process_xr_pose)
converts a controller pose into a z-up, headset-yaw-compensated 4x4 transform:
  - position is relative to the headset position,
  - the headset's yaw is removed, so +x is always "the way the operator faces",
    regardless of where they stand or turn in the room.
This makes the teleop mapping independent of the play-space calibration.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# raw (y-up, -z fwd) -> world (z-up, x fwd):  world = M @ raw
#   world_x = -raw_z (forward), world_y = -raw_x (left), world_z = raw_y (up)
R_HEADSET_TO_WORLD = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ],
    dtype=np.float64,
)


def _pose7_to_world(pose7) -> tuple[np.ndarray, np.ndarray]:
    """SDK [x,y,z,qx,qy,qz,qw] -> (pos(3), rot(3x3)) in the z-up world frame."""
    p = np.asarray(pose7, dtype=np.float64)
    xyz, quat = p[:3], p[3:]
    if np.allclose(quat, 0):
        quat = np.array([0.0, 0.0, 0.0, 1.0])
    pos = R_HEADSET_TO_WORLD @ xyz
    rot = R_HEADSET_TO_WORLD @ R.from_quat(quat).as_matrix() @ R_HEADSET_TO_WORLD.T
    return pos, rot


def pose7_to_world_rot(pose7) -> np.ndarray:
    """SDK [x,y,z,qx,qy,qz,qw] -> 3x3 rotation in the z-up world frame.

    Used for the waist tracker's absolute lean: its pitch about world y is the
    torso-lean signal, independent of any reference frame.
    """
    return _pose7_to_world(pose7)[1]


def process_xr_pose(controller_pose7, headset_pose7) -> np.ndarray:
    """Target pose as a 4x4 in the z-up, reference-yaw-compensated frame.

    The reference (headset OR waist tracker) supplies the origin and heading:
    the target is expressed relative to the reference position with the
    reference's yaw removed, so the mapping is invariant to where the operator
    stands / which way they face. Feed the headset for head-relative teleop
    (controllers) or the waist tracker for body-relative teleop (wrist
    trackers, the natural G1-style mapping).
    """
    c_pos, c_rot = _pose7_to_world(controller_pose7)
    h_pos, h_rot = _pose7_to_world(headset_pose7)

    yaw = R.from_matrix(h_rot).as_euler("xyz")[2]
    unyaw = R.from_euler("z", -yaw).as_matrix()

    T = np.eye(4)
    T[:3, :3] = unyaw @ c_rot
    T[:3, 3] = unyaw @ (c_pos - h_pos)
    return T


def mat_to_pos_quat_wxyz(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4x4 -> (pos(3), quat wxyz(4)).  Isaac Lab quaternions are w-first."""
    q_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    return T[:3, 3].copy(), np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def yaw_align_from_wrists(p_left: np.ndarray, p_right: np.ndarray,
                          up: np.ndarray | None = None) -> np.ndarray:
    """Body-yaw alignment from the two wrist positions alone (V2AP-demo's
    calibration trick, teleop_pub.py:238): the operator just stands facing
    forward with hands apart - no T-pose, no button, no trust in any tracker's
    mounting orientation (the waist tracker's yaw is unknown in Object mode).

    Returns R whose columns are (x=front, y=left, z=up) expressed in the
    device/raw frame: left = right-hand -> left-hand line projected onto the
    horizontal plane, front = left x up.

    `up` defaults to the raw PICO frame's +y. Raises if the hands are closer
    than 5 cm or the projected line is degenerate (arms near-vertical).
    """
    p_left = np.asarray(p_left, dtype=np.float64)
    p_right = np.asarray(p_right, dtype=np.float64)
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    up = np.asarray(up, dtype=np.float64) / np.linalg.norm(up)
    y = p_left - p_right
    if np.linalg.norm(y) < 0.05:
        raise ValueError("hands too close together for yaw alignment (<5cm)")
    y = y - (y @ up) * up
    n = np.linalg.norm(y)
    if n < 0.5 * np.linalg.norm(p_left - p_right):
        raise ValueError("hand line nearly vertical - stand with hands apart "
                         "at your sides for yaw alignment")
    y /= n
    x = np.cross(y, up)
    return np.column_stack((x, y, up))


def pitch_delta(rot_now: np.ndarray, rot_ref: np.ndarray) -> float:
    """Lean angle (rad) between two world-frame rotations about world y.

    In the processed frame (x fwd / y left / z up) a forward body lean is a
    positive rotation about +y. The delta is taken as the y component of the
    rotation vector rot_now @ rot_ref.T, which is exact for pure pitch and
    degrades gracefully for combined motions - fine for a chest-tracker ->
    torso-lean mapping that only ever uses small/moderate deltas.
    """
    rv = R.from_matrix(rot_now @ rot_ref.T).as_rotvec()
    return float(rv[1])


class OneEuroFilter:
    """One-euro filter (Casiez et al. 2012) over an nD vector.

    Adaptive low-pass: heavy smoothing at rest (min_cutoff), opens up with
    speed (beta) so fast intentional motion keeps low lag - the standard
    jitter filter for VR teleop targets. Call with the sample and the tick
    dt; reset() on clutch re-engage so stale state can't bleed across grabs.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5,
                 d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.reset()

    def reset(self):
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            return x.copy()
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev, self._dx_prev = x_hat, dx_hat
        return x_hat.copy()


class QuatOneEuroFilter:
    """One-euro on quaternion components (hemisphere-aligned, renormalized).

    Component-wise filtering of sign-consistent unit quaternions is a
    standard approximation of slerp smoothing and exact in the small-angle
    limit; outputs are renormalized so downstream math always sees a unit
    quaternion. Works for any fixed component order (wxyz used here).
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5,
                 d_cutoff: float = 1.0):
        self._f = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self._prev: np.ndarray | None = None

    def reset(self):
        self._f.reset()
        self._prev = None

    def __call__(self, q, dt: float) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if self._prev is not None and float(np.dot(q, self._prev)) < 0.0:
            q = -q
        out = self._f(q, dt)
        out = out / np.linalg.norm(out)
        self._prev = out
        return out
