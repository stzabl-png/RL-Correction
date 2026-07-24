"""Synthetic Wuji-glove stand-in: anatomically plausible 21-keypoint hand motion.

Lets the full pipeline (frame conversion -> retarget -> mapping -> publish/viz)
run and be tested without hardware. Keypoints are emitted in a rotated
"glove-like" wrist frame (mimicking Wuji's X=radial, Y=palmar, Z=proximal) so
the convention-agnostic conversion in retarget/frames.py is exercised for real.

Geometry is built in an internal frame: +x distal (fingers when open),
+y radial (thumb side, right hand), +z dorsal (back of hand); palm faces -z.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from magicdexmate.skeleton import HandFrame
from magicdexmate.sources.base import GloveSource, LatestSlot

# name: (MCP position, segment lengths MCP->PIP->DIP->TIP, mediapipe ids)
_FINGERS = {
    "index": ((0.092, 0.027, 0.0), (0.042, 0.026, 0.022), (5, 6, 7, 8)),
    "middle": ((0.096, 0.003, 0.0), (0.046, 0.029, 0.024), (9, 10, 11, 12)),
    "ring": ((0.090, -0.021, 0.0), (0.042, 0.027, 0.023), (13, 14, 15, 16)),
    "pinky": ((0.082, -0.043, 0.0), (0.032, 0.021, 0.019), (17, 18, 19, 20)),
}
_THUMB_CMC = np.array([0.030, 0.045, -0.010])
_THUMB_SEGS = (0.046, 0.038, 0.030)
_THUMB_IDS = (1, 2, 3, 4)
# Open-pose thumb direction; curl = opposition swing about +x (radial -> palmar,
# opposing the fingers which curl toward -z) plus in-plane flexion of the chain.
_THUMB_DIR_OPEN = np.array([0.40, 0.92, 0.0]) / np.linalg.norm([0.40, 0.92, 0.0])

# Internal/build frame -> emitted "glove" frame (Wuji-like: X=radial, Y=palmar, Z=proximal).
_BUILD2GLOVE = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

# (thumb_curl, finger_curl) minimizing tip distance per pinch target (grid-searched;
# index reaches ~7 mm, middle ~30 mm; ring/pinky stay 45-60 mm apart - mock thumb
# opposition toward the ulnar side is limited, real-glove data is the reference).
_PINCH_CURL = {"index": (0.95, 0.70), "middle": (1.00, 0.75), "ring": (1.10, 0.65), "pinky": (1.20, 0.47)}
_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def hand_keypoints(curl: np.ndarray) -> np.ndarray:
    """curl: (5,) flexion parameter per finger [thumb..pinky], ~0 open, ~1.3 fist.

    Returns (21,3) keypoints in the build frame (right hand).
    """
    kp = np.zeros((21, 3))

    # Four fingers: planar chains curling toward the palm (-z).
    for i, (name, (mcp, segs, ids)) in enumerate(_FINGERS.items()):
        theta = float(curl[i + 1])
        flex = np.array([1.0, 1.05, 0.7]) * theta
        p = np.array(mcp, dtype=float)
        kp[ids[0]] = p
        phi = 0.0
        for seg_len, d_phi, idx in zip(segs, flex, ids[1:]):
            phi += d_phi
            p = p + seg_len * np.array([np.cos(phi), 0.0, -np.sin(phi)])
            kp[idx] = p

    # Thumb: opposition swing of the whole chain about +x, then in-plane flexion.
    theta_t = float(curl[0])
    swing = _rodrigues(np.array([1.0, 0.0, 0.0]), -1.9 * theta_t)  # +y (radial) -> -z (palmar)
    d0 = swing @ _THUMB_DIR_OPEN
    bend_axis = np.cross(np.array([1.0, 0.0, 0.0]), d0)  # in-plane flexion axis
    bend_axis /= np.linalg.norm(bend_axis)
    flex = np.array([0.20, 0.44, 0.52]) * theta_t
    p = _THUMB_CMC.copy()
    kp[_THUMB_IDS[0]] = p
    phi = 0.0
    for seg_len, d_phi, idx in zip(_THUMB_SEGS, flex, _THUMB_IDS[1:]):
        phi += d_phi
        d = _rodrigues(bend_axis, -phi) @ d0  # bend toward distal(+x)/palmar
        p = p + seg_len * d
        kp[idx] = p

    return kp


def _ramp(t: float, period: float) -> float:
    """0 -> 1 -> 0 raised-cosine over `period` seconds."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * (t % period) / period)


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def mock_wrist_quat(t: float, motion: str = "cycle") -> np.ndarray:
    """Wrist orientation (wxyz quat) - stands in for the glove IMU.

    The Wuji glove provides wrist ORIENTATION only (dorsum IMU, 800 Hz); there
    is no wrist translation. Mock mirrors that: rpy wobble, no position.

    motion == "rpytest": pure single-axis rotations, 3 s per axis
    (roll -> pitch -> yaw, repeating) - used to verify axis mapping of the
    wrist channel end to end.
    """
    if motion == "rpytest":
        phase = int(t // 3.0) % 3
        a = 0.5 * np.sin(2.0 * np.pi * (t % 3.0) / 3.0)
        rpy = [0.0, 0.0, 0.0]
        rpy[phase] = a
        return _euler_to_quat(*rpy)
    roll = 0.25 * np.sin(2.0 * np.pi * t / 6.0)
    pitch = 0.20 * np.sin(2.0 * np.pi * t / 7.0 + 1.0)
    yaw = 0.30 * np.sin(2.0 * np.pi * t / 9.0 + 2.0)
    return _euler_to_quat(roll, pitch, yaw)


def motion_curl(motion: str, t: float) -> np.ndarray:
    """(5,) curl parameters [thumb, index, middle, ring, pinky] at time t."""
    if motion == "open":
        return np.zeros(5)
    if motion == "fist":
        s = _ramp(t, 4.0)
        return np.array([0.8, 1.3, 1.3, 1.3, 1.3]) * s
    if motion == "wave":
        period = 4.0
        curls = [0.5 * _ramp(t, period)]
        for i in range(4):
            curls.append(1.2 * _ramp(t - (i + 1) * period / 8.0, period))
        return np.array(curls)
    if motion == "pinch":
        period = 3.0
        targets = ["index", "middle", "ring", "pinky"]
        target = targets[int(t // period) % 4]
        s = _ramp(t, period)
        thumb_c, finger_c = _PINCH_CURL[target]
        curl = np.full(5, 0.15)
        curl[0] = thumb_c * s
        curl[_FINGER_ORDER.index(target)] = finger_c * s
        return curl
    if motion == "rpytest":
        return np.zeros(5)  # hand stays open; only the wrist moves
    if motion == "cycle":
        schedule = [("open", 2.0), ("fist", 6.0), ("wave", 6.0), ("pinch", 12.0)]
        total = sum(d for _, d in schedule)
        tt = t % total
        for name, dur in schedule:
            if tt < dur:
                return motion_curl(name, tt)
            tt -= dur
    raise ValueError(f"unknown motion: {motion}")


class MockGloveSource(GloveSource):
    def __init__(
        self,
        hand: str = "right",
        motion: str = "cycle",
        rate: float = 120.0,
        noise: float = 0.0005,
        seed: int = 0,
    ):
        assert hand in ("right", "left")
        self.hand = hand
        self.motion = motion
        self.rate = rate
        self.noise = noise
        self._rng = np.random.default_rng(seed)
        self._slot = LatestSlot()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_at(self, t: float) -> HandFrame:
        """Deterministic frame at motion-time t (also used directly by tests)."""
        kp = hand_keypoints(motion_curl(self.motion, t))
        if self.hand == "left":
            kp = kp * np.array([1.0, -1.0, 1.0])
        kp = kp @ _BUILD2GLOVE
        if self.noise > 0:
            kp = kp + self._rng.normal(0.0, self.noise, kp.shape)
        return HandFrame(
            t_us=time.time_ns() // 1000,
            hand=self.hand,
            kp=kp,
            conf=np.ones(21),
            wrist_quat=mock_wrist_quat(t, self.motion),
        )

    def _run(self):
        period = 1.0 / self.rate
        t0 = time.monotonic()
        while not self._stop.is_set():
            self._slot.put(self.sample_at(time.monotonic() - t0))
            time.sleep(period)

    def start(self):
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mock-glove", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_latest(self) -> HandFrame | None:
        return self._slot.get()
