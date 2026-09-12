"""Bridge short losses of wrist POSITION using the wrist's live ORIENTATION.

The PICO Motion Trackers are tracked inside-out by the headset cameras: position
only exists while the headset can see the tracker, orientation comes from the
tracker's own IMU and never stops. So a dropout is asymmetric - the pose keeps
streaming, but its translation is stale while its rotation is still true. (The
official setup guide is explicit that the cameras must see the trackers:
"look down at the foot motion trackers until the headset cameras recognize
them".) Measured on the 2026-07-29 raw batch: the waist tracker, which sits
where the cameras can never see it, has 1 unique position in 500 frames and
300+ unique orientations.

What this recovers, and what it cannot
--------------------------------------
One IMU cannot observe position. What it can do is exploit the forearm being a
rigid body: over a short window the wrist traces an arc about the elbow, so

    p_wrist(t) = c + R_wrist(t) @ v

with c the elbow and v the (constant) elbow->wrist vector in the tracker's own
frame. Both are linear in the samples, so a live window solves them in closed
form; during a dropout the live rotation keeps driving the arc while c is held.

That assumption holds while the elbow stays put and decays as the shoulder
moves. Measured by leave-one-out over the 20 raw takes - 293 simulated
dropouts, these defaults, against simply freezing the position:

    dropout    freeze med/p90     fused med/p90     p90     p99
      0.25 s    8.4 /  52.7 mm    8.6 /  43.6 mm   -17%    +17%
      0.50 s   16.4 / 103.7 mm   15.8 /  73.9 mm   -29%     -3%
      1.00 s   28.2 / 222.2 mm   23.3 / 117.5 mm   -47%    +12%
      2.00 s   46.5 / 377.0 mm   34.2 / 249.6 mm   -34%     +2%
      3.00 s   53.7 / 410.5 mm   33.4 / 263.0 mm   -36%     -1%

The typical case barely moves - freezing is already near-perfect there. The
win is the tail, which is the case that actually hurts: a 40 cm error is the
arm visibly flying off. p99 is a wash, meaning the worst cases are not made
worse. 56% of windows get rejected by the conditioning checks and fall back to
freezing, which is what keeps that tail honest.

This bridges short gaps; it does not replace tracking. `max_bridge_s` gives up
and reports the estimate untrustworthy, leaving the existing freeze detector to
disengage the arm rather than follow an invented target.

Not used yet: the SDK also reports per-tracker velocity and acceleration
(`get_motion_tracker_data`). Integrating a reported velocity should beat an
inferred arc over the first few hundred ms - the producer only started
forwarding those fields on 2026-07-30, so no capture contains them yet.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def _log_so3(Rm: np.ndarray) -> np.ndarray:
    """Rotation-vector of a rotation matrix (scipy-free: this runs per sample
    in the control loop, and the caller may not have scipy loaded)."""
    c = np.clip((np.trace(Rm) - 1.0) * 0.5, -1.0, 1.0)
    ang = float(np.arccos(c))
    if ang < 1e-8:
        return np.zeros(3)
    axis = np.array([Rm[2, 1] - Rm[1, 2],
                     Rm[0, 2] - Rm[2, 0],
                     Rm[1, 0] - Rm[0, 1]]) / (2.0 * np.sin(ang))
    return axis * ang


class ForearmFuser:
    """Per-tracker position bridge. Feed every sample; ask when position dies.

    Live samples (position changed) refresh the arc fit. Dead samples (position
    bit-identical, orientation still moving) get an estimate from the arc.
    """

    def __init__(self, cal_window_s: float = 1.0, blend_s: float = 0.4,
                 max_bridge_s: float = 1.5, min_samples: int = 30,
                 v_bounds_m: tuple[float, float] = (0.05, 0.80),
                 min_rotation_deg: float = 8.0):
        self.cal_window_s = cal_window_s
        self.blend_s = blend_s
        self.max_bridge_s = max_bridge_s
        self.min_samples = min_samples
        self.v_bounds_m = v_bounds_m
        self.min_rotation_deg = min_rotation_deg
        self._buf: deque[tuple[float, np.ndarray, np.ndarray]] = deque()
        self._last_live: tuple[float, np.ndarray, np.ndarray] | None = None
        self._arc: tuple[np.ndarray, np.ndarray] | None = None   # (elbow, v)
        self.bridging = False

    def _refit(self) -> None:
        """Closed-form solve of p = c + R v over the calibration window.

        Linear in c and v jointly, so it is one least-squares call - no
        iteration, no initial guess, safe to run every live sample.
        """
        if len(self._buf) < self.min_samples:
            self._arc = None
            return
        # Without rotation the system is rank-deficient: p = c + I@v has no
        # unique split, and least-squares happily returns the minimum-norm
        # answer (c = v = p/2), whose radius can land inside any plausible
        # bound. Require the window to actually turn before trusting an arc.
        R0 = self._buf[0][2]
        spread = max(np.linalg.norm(_log_so3(Ri @ R0.T))
                     for _, _, Ri in self._buf)
        if np.degrees(spread) < self.min_rotation_deg:
            self._arc = None
            return
        n = len(self._buf)
        A = np.zeros((3 * n, 6))
        b = np.empty(3 * n)
        for i, (_, p, R) in enumerate(self._buf):
            A[3 * i:3 * i + 3, :3] = np.eye(3)
            A[3 * i:3 * i + 3, 3:] = R
            b[3 * i:3 * i + 3] = p
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        lo, hi = self.v_bounds_m
        # a window with too little rotation fits an arbitrary arc; the radius
        # falling outside forearm-plausible bounds is how that shows up
        self._arc = (x[:3], x[3:]) if lo < np.linalg.norm(x[3:]) < hi else None

    def update(self, t_s: float, pos: np.ndarray, rot: np.ndarray,
               position_live: bool) -> tuple[np.ndarray, bool]:
        """-> (position, trustworthy).

        `trustworthy` is False once the gap outlives `max_bridge_s`; the caller
        should then treat the tracker as lost rather than keep following it.
        """
        pos = np.asarray(pos, dtype=float)
        rot = np.asarray(rot, dtype=float)
        if position_live:
            self._buf.append((t_s, pos.copy(), rot.copy()))
            while self._buf and t_s - self._buf[0][0] > self.cal_window_s:
                self._buf.popleft()
            self._last_live = (t_s, pos.copy(), rot.copy())
            self._refit()
            self.bridging = False
            return pos, True

        if self._last_live is None:
            return pos, False
        t0, p0, _ = self._last_live
        dt = t_s - t0
        self.bridging = True
        if self._arc is None:
            return p0, dt <= self.max_bridge_s
        c, v = self._arc
        arc = c + rot @ v
        # ease out of the frozen position: over the first fraction of a second
        # holding still is more accurate than any arc, and a hard switch would
        # step the IK target
        a = min(1.0, dt / self.blend_s) if self.blend_s > 0 else 1.0
        return (1.0 - a) * p0 + a * arc, dt <= self.max_bridge_s

    @property
    def forearm_len_m(self) -> float | None:
        return None if self._arc is None else float(np.linalg.norm(self._arc[1]))
