"""PICO frame sources for the Vega arm teleop consumer (py3.11-safe, no SDK).

`PicoFrame` mirrors the producer's raw msgpack message (scripts/
teleop_pico_producer.py). Frame math stays out of here - the sim runner
applies magicdexmate.pico.xr_pose on the raw poses.

Sources:
  PicoZmqSource  - live: SUB (CONFLATE) on the producer's PUB, newest-only.
  PicoLogSource  - offline: replays a producer --record file (raw msgpack
    concatenation) with `sample_at(t)` phase-locked to SIM time.
  MockPicoSource - synthetic raw-frame generator with `sample_at(t)` so the
    sim can phase-lock the motion to SIM time (same trick as MockGloveSource:
    this box's wall clock runs ~30x faster than Isaac steps). Grips close at
    t>=1 s, then both wrists sweep slow circles + a wrist roll, which
    exercises the full raw->yaw-comp->IK path headless.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
import time

import numpy as np


# Every fixed name the producer publishes from the body model, so consumers can
# tell a body-model entry from a raw tracker serial. Classifying by substring
# ("does the key contain WRIST") broke the moment honest joint names arrived -
# PELVIS and L_SHOULDER contain none of the old keywords and were read as
# hardware serials.
BODY_JOINT_NAMES = frozenset({
    "PELVIS", "SPINE3", "NECK", "HEAD",
    "L_SHOULDER", "R_SHOULDER", "L_ELBOW", "R_ELBOW",
    "L_WRIST", "R_WRIST", "L_HAND", "R_HAND", "L_ANKLE", "R_ANKLE",
    # pre-2026-07-31 names, still published as aliases
    "LWRIST", "RWRIST", "WAIST", "LELBOW", "RELBOW",
})


def is_body_joint(key: str) -> bool:
    """True for a body-model joint, False for a raw tracker serial."""
    return str(key) in BODY_JOINT_NAMES


@dataclass
class PicoFrame:
    t_us: int
    head: np.ndarray                    # raw SDK pose [x,y,z,qx,qy,qz,qw], y-up
    left: np.ndarray
    right: np.ndarray
    device_ts_ns: int = 0               # SDK device clock; 0 = unknown (mock)
    left_grip: float = 0.0
    right_grip: float = 0.0
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    buttons: dict = field(default_factory=dict)
    left_joystick: tuple = (0.0, 0.0)
    right_joystick: tuple = (0.0, 0.0)
    trackers: dict = field(default_factory=dict)  # serial -> pose7 (raw y-up)
    # Full SMPL-24 estimate when the producer sends it. Carried because the
    # named joints in `trackers` are producer-version dependent: recordings
    # made before the 2026-07-31 rename have no L_SHOULDER/R_SHOULDER, so a
    # law that reads them works live and silently no-ops on replay - which is
    # exactly how --map shoulder-rel first shipped, applied to 0 frames.
    body24: np.ndarray | None = None

    def controller(self, hand: str) -> np.ndarray:
        return self.left if hand == "left" else self.right

    def grip(self, hand: str) -> float:
        return self.left_grip if hand == "left" else self.right_grip


def frame_from_dict(d: dict) -> PicoFrame:
    """Producer msgpack dict -> PicoFrame (shared by ZMQ and log replay)."""
    return PicoFrame(
        t_us=d["t_us"],
        head=np.asarray(d["head"], dtype=np.float64),
        left=np.asarray(d["left"], dtype=np.float64),
        right=np.asarray(d["right"], dtype=np.float64),
        device_ts_ns=int(d.get("device_ts_ns", 0)),
        left_grip=float(d.get("left_grip", 0.0)),
        right_grip=float(d.get("right_grip", 0.0)),
        left_trigger=float(d.get("left_trigger", 0.0)),
        right_trigger=float(d.get("right_trigger", 0.0)),
        buttons=d.get("buttons", {}),
        left_joystick=tuple(d.get("left_joystick", (0.0, 0.0))),
        right_joystick=tuple(d.get("right_joystick", (0.0, 0.0))),
        trackers={sn: np.asarray(p, dtype=np.float64)
                  for sn, p in d.get("trackers", {}).items()},
        body24=(np.asarray(d["body24"], dtype=np.float64)
                if d.get("body24") else None),
    )


class GripLatch:
    """Hysteresis on the analog grip: >= on_th engages, <= off_th releases.

    A single 0.5 threshold chatters when the operator hovers around it, and
    every spurious release/engage re-zeroes the clutch mid-motion (the arm
    freezes then jumps). 0.6/0.4 keeps the deadman feel from hardware day #1
    while making half-squeezes stable.
    """

    def __init__(self, on_th: float = 0.6, off_th: float = 0.4):
        assert on_th > off_th
        self.on_th, self.off_th = on_th, off_th
        self.engaged = False

    def update(self, value: float) -> bool:
        if self.engaged:
            self.engaged = value > self.off_th
        else:
            self.engaged = value >= self.on_th
        return self.engaged


class DeviceStallDetector:
    """Detects a frozen headset from a stalled SDK device clock.

    The producer's t_us keeps advancing (it wall-clock-stamps every send)
    even when the headset sleeps, so a producer-timestamp staleness check
    can't see a frozen device. The SDK device_ts_ns, by contrast, stops
    changing the instant the headset stops streaming. We track the wall-clock
    time (in producer t_us units) at which device_ts last *changed*; if it
    hasn't changed for longer than `stall_us`, the device is stale.

    device_ts_ns == 0 means the source has no device clock (mock / old
    recordings): those are never reported stale here.
    """

    def __init__(self, stall_ms: float = 300.0):
        self.stall_us = stall_ms * 1000.0
        self._last_device_ts = None
        self._last_change_t_us = None
        self.stalled = False

    def update(self, frame: "PicoFrame") -> bool:
        """Feed the latest frame; return True if the device clock is stalled."""
        if frame is None or frame.device_ts_ns == 0:
            self.stalled = False
            return False
        if frame.device_ts_ns != self._last_device_ts:
            self._last_device_ts = frame.device_ts_ns
            self._last_change_t_us = frame.t_us
        elif self._last_change_t_us is not None:
            if (frame.t_us - self._last_change_t_us) > self.stall_us:
                self.stalled = True
                return True
        self.stalled = False
        return False


class TrackerFreezeDetector:
    """Detects the body-estimate freeze the device clock can't see.

    2026-07-26 live session: the PICO full-body estimate froze solid (all
    tracker poses bit-identical for 15+ s while the operator waved) but the
    headset itself kept streaming - device_ts advanced, so DeviceStallDetector
    saw nothing and the arms silently tracked a dead target. Real sensor data
    always carries noise: a tracker set that is bit-identical for longer than
    `freeze_ms` can only be a frozen estimator.

    device_ts_ns == 0 (mock / old recordings) is exempt - synthetic poses are
    legitimately constant before the trajectory starts.

    `keys` restricts the watch to one channel (or a few). Watching everything
    at once is blind on the 2026-07-29 hybrid rig, where the wrists come from
    the raw per-tracker channel and the waist from the body model: the live
    wrists keep the aggregate value changing, so a dead model channel slips
    through unseen - and in --map waist-abs the waist IS the mapping origin,
    so a frozen one silently sends every target to a dead reference. One
    detector per consumed channel catches that; `label` names it in the
    operator-facing warning.
    """

    def __init__(self, freeze_ms: float = 500.0, keys=None, label: str = "trackers"):
        self.freeze_us = freeze_ms * 1000.0
        self.keys = None if keys is None else set(keys)
        self.label = label
        self._last_key = None
        self._last_change_t_us = None
        self.frozen = False

    def update(self, frame: "PicoFrame | None") -> bool:
        if frame is None or frame.device_ts_ns == 0 or not frame.trackers:
            self.frozen = False
            return False
        watched = sorted(frame.trackers if self.keys is None
                         else self.keys & set(frame.trackers))
        if not watched:
            # channel absent from this frame - nothing to judge (the missing
            # tracker is reported elsewhere; don't cry freeze on no data)
            self.frozen = False
            return False
        key = tuple(np.asarray(frame.trackers[k]).tobytes() for k in watched)
        if key != self._last_key:
            self._last_key = key
            self._last_change_t_us = frame.t_us
        elif self._last_change_t_us is not None:
            if (frame.t_us - self._last_change_t_us) > self.freeze_us:
                self.frozen = True
                return True
        self.frozen = False
        return False


class TrackerGlitchGate:
    """Rejects physically impossible single-step tracker jumps.

    The 2026-07-26 audit of a 47-min live session: the body estimator emitted
    hundreds of wrist steps no human can execute (up to 170 deg rotation and
    >10 cm translation between consecutive updates) while waist/head stayed
    spotless. Fed raw to the IK these spikes yank the arm. Per tracker:

      - a step from the last ACCEPTED pose within human-plausible rates
        (`v_max` m/s, `w_max` deg/s, with small floors for sensor noise)
        is accepted;
      - an implausible step is rejected and the last accepted pose is held;
      - if the implausible pose PERSISTS for `accept_after_ms` (the tracker
        genuinely relocated, e.g. camera reacquire after occlusion), the gate
        adopts it and signals `reanchor=True`: the caller should drop the
        clutch so the arm re-zeroes instead of swinging through the jump.

    Rate thresholds scale with the wall-clock gap between poses, so the gate
    behaves the same whether the consumer samples at 6 Hz (slow sim) or
    100 Hz.
    """

    def __init__(self, v_max: float = 4.0, w_max_deg: float = 500.0,
                 pos_floor: float = 0.05, ang_floor_deg: float = 20.0,
                 accept_after_ms: float = 400.0):
        self.v_max = v_max
        self.w_max = np.deg2rad(w_max_deg)
        self.pos_floor = pos_floor
        self.ang_floor = np.deg2rad(ang_floor_deg)
        self.accept_after_us = accept_after_ms * 1000.0
        self._acc_t_us: int | None = None
        self._acc: np.ndarray | None = None
        self._prev_t_us: int | None = None
        self._prev: np.ndarray | None = None
        self._reject_t_us: int | None = None
        self.n_rejected = 0
        self.n_reanchors = 0

    @staticmethod
    def _ang(q1: np.ndarray, q2: np.ndarray) -> float:
        d = abs(float(np.dot(q1, q2)))
        return 2.0 * float(np.arccos(min(1.0, d)))

    def _within(self, pose7: np.ndarray, ref: np.ndarray, dt: float) -> bool:
        dp = float(np.linalg.norm(pose7[:3] - ref[:3]))
        dang = self._ang(pose7[3:7], ref[3:7])
        return (dp <= max(self.pos_floor, self.v_max * dt)
                and dang <= max(self.ang_floor, self.w_max * dt))

    def update(self, t_us: int, pose7: np.ndarray) -> tuple[np.ndarray, bool]:
        """Feed a candidate pose; return (pose_to_use, reanchor_signal).

        Rates are judged between CONSECUTIVE samples (not vs the last
        accepted pose): judging vs the accepted pose would let a teleport
        quietly become "plausible" once enough time passed while rejecting,
        and the arm would swing through it instead of re-anchoring.
        """
        pose7 = np.asarray(pose7, dtype=np.float64)
        if self._acc is None:
            self._acc = self._prev = pose7
            self._acc_t_us = self._prev_t_us = t_us
            return pose7, False
        dt_step = min(max((t_us - self._prev_t_us) / 1e6, 1e-3), 1.0)
        step_ok = self._within(pose7, self._prev, dt_step)
        near_acc = self._within(pose7, self._acc, dt_step)
        self._prev, self._prev_t_us = pose7, t_us
        if self._reject_t_us is None:
            if step_ok:
                self._acc, self._acc_t_us = pose7, t_us
                return pose7, False
            self._reject_t_us = t_us          # spike begins: hold last accepted
            self.n_rejected += 1
            return self._acc, False
        # in rejection: exit cleanly if the candidate came back near the
        # accepted pose; otherwise wait out the persistence timer
        if near_acc:
            self._reject_t_us = None
            self._acc, self._acc_t_us = pose7, t_us
            return pose7, False
        self.n_rejected += 1
        if (t_us - self._reject_t_us) > self.accept_after_us:
            # sustained new location: adopt it, ask the caller to re-anchor
            self._acc, self._acc_t_us = pose7, t_us
            self._reject_t_us = None
            self.n_reanchors += 1
            return pose7, True
        return self._acc, False


class PicoZmqSource:
    """Newest-only subscriber to the raw PICO producer stream."""

    def __init__(self, sub: str = "tcp://127.0.0.1:5581"):
        self.sub_addr = sub
        self._lock = threading.Lock()
        self._frame: PicoFrame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.n_recv = 0

    def start(self):
        import msgpack
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.connect(self.sub_addr)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")

        def loop():
            poller = zmq.Poller()
            poller.register(sock, zmq.POLLIN)
            while not self._stop.is_set():
                if not dict(poller.poll(timeout=50)):
                    continue
                frame = frame_from_dict(msgpack.unpackb(sock.recv()))
                with self._lock:
                    self._frame = frame
                self.n_recv += 1
            sock.close(0)

        self._thread = threading.Thread(target=loop, name="pico-zmq", daemon=True)
        self._thread.start()
        print(f"[pico-src] subscribed {self.sub_addr} (CONFLATE)")

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def get_latest(self) -> PicoFrame | None:
        with self._lock:
            return self._frame


class PicoLogSource:
    """Replays a producer `--record` file, phase-locked to sim time.

    The file is the raw concatenation of the msgpack frames that went over
    ZMQ (producer writes the exact bytes it sends). Timestamps are re-based
    so playback starts at t=0; `sample_at(t)` returns the newest frame with
    rel_time <= t, holding the first/last frame at the ends (or looping with
    loop=True).
    """

    def __init__(self, path: str, loop: bool = False):
        import msgpack

        self.path = path
        self.loop = loop
        self.frames: list[PicoFrame] = []
        with open(path, "rb") as f:
            try:
                for d in msgpack.Unpacker(f):
                    self.frames.append(frame_from_dict(d))
            except Exception as e:  # truncated tail (recorder killed mid-write)
                print(f"[pico-log] warning: keeping {len(self.frames)} frames, "
                      f"tail unparsable: {e}")
        if not self.frames:
            raise ValueError(f"no frames in {path}")
        t0 = self.frames[0].t_us
        self._rel_t = np.array([(fr.t_us - t0) / 1e6 for fr in self.frames])
        self.duration = float(self._rel_t[-1])
        self._t_start = None  # for wall-clock get_latest fallback
        print(f"[pico-log] {len(self.frames)} frames, {self.duration:.1f}s from {path}"
              + (" (loop)" if loop else ""))

    def start(self):
        self._t_start = time.time()

    def stop(self):
        pass

    def sample_at(self, t: float) -> PicoFrame:
        if self.loop and self.duration > 0:
            t = t % self.duration
        idx = int(np.searchsorted(self._rel_t, t, side="right")) - 1
        return self.frames[max(0, min(idx, len(self.frames) - 1))]

    def get_latest(self) -> PicoFrame:  # wall-clock fallback (unused in sim)
        return self.sample_at(time.time() - (self._t_start or time.time()))


def _quat_xyzw_about(axis, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = 0.5 * angle_rad
    return np.array([*(axis * np.sin(half)), np.cos(half)])


def _quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b for [x,y,z,w] quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


class MockPicoSource:
    """Synthetic raw frames. Head fixed at 1.6 m; wrists orbit + roll.

    Raw frame is y-up, -z forward: right controller starts at
    (x=+0.25, y=1.2, z=-0.35) i.e. 25 cm right, 1.2 m up, 35 cm in front.
    The left wrist is the MIRROR of the right (position x -> -x, rotation
    reflected across the y/z plane) so both arms see equivalent reachability
    - vega's left arm has mirrored joint limits, and a copied (unmirrored)
    rotation used to give the left arm a systematically harder task.

    Publishes three synthetic Swift trackers so the tracker teleop path runs
    end-to-end with no hardware:
      WAIST_SN  - a waist/chest tracker that leans forward/back sinusoidally
                  (drives torso, and is the body root in --mode trackers),
      LWRIST_SN / RWRIST_SN - wrist trackers colocated with the controllers,
                  so --mode trackers exercises the same motion as the
                  controller path (wrist targets relative to the waist).

    `stress()` returns an aggressive variant (bigger sweep, faster, large
    composite roll+pitch wrist rotation) used to probe IK joint-winding.
    """

    ENGAGE_T = 1.0        # grips close here; references get captured
    MOVE_T = 1.5          # motion starts (engage happens at rest)
    # Names match what the live producer synthesizes from the PICO full-body
    # joints (teleop_pico_producer.py BODY_JOINT), so the consumer's default
    # --tracker-left/right/waist work against mock and live streams alike.
    WAIST_SN = "WAIST"
    LWRIST_SN = "LWRIST"
    RWRIST_SN = "RWRIST"
    LELBOW_SN = "LELBOW"
    RELBOW_SN = "RELBOW"
    TRACKER_SN = WAIST_SN   # back-compat alias (older tests/callers)

    def __init__(self, radius: float = 0.10, freq_hz: float = 0.15,
                 roll_deg: float = 30.0, pitch_deg: float = 0.0,
                 lean_deg: float = 15.0):
        self.radius = radius
        self.freq = freq_hz
        self.roll = np.deg2rad(roll_deg)
        self.pitch = np.deg2rad(pitch_deg)
        self.lean = np.deg2rad(lean_deg)
        self.demo_mode = False

    # demo sequence: (t0, t1, offset, roll_deg, pitch_deg) - linear move then
    # hold. Offsets (dx outward, dy up, dz forward[-z]) mirror to both wrists;
    # roll = rotation about raw z, pitch = about raw x (left wrist mirrored).
    # First 30s = position holds, 30-50s = pure wrist-rotation holds (position
    # stays at rest so rotation->position leakage is also visible).
    _DEMO_PERIOD = 50.0
    _DEMO_SEGS = [
        (0.0, 3.0, (0.0, 0.0, 0.0), 0.0, 0.0),      # rest
        (3.0, 5.0, (0.0, 0.0, -0.25), 0.0, 0.0),    # push forward
        (5.0, 8.0, (0.0, 0.0, -0.25), 0.0, 0.0),    # hold
        (8.0, 10.0, (0.0, 0.0, 0.0), 0.0, 0.0),     # back
        (10.0, 13.0, (0.0, 0.0, 0.0), 0.0, 0.0),    # rest
        (13.0, 15.0, (0.30, 0.0, 0.0), 0.0, 0.0),   # lateral raise
        (15.0, 18.0, (0.30, 0.0, 0.0), 0.0, 0.0),   # hold
        (18.0, 20.0, (0.0, 0.0, 0.0), 0.0, 0.0),    # back
        (20.0, 23.0, (0.0, 0.0, 0.0), 0.0, 0.0),    # rest
        (23.0, 25.0, (0.0, 0.30, 0.0), 0.0, 0.0),   # raise overhead-ish
        (25.0, 28.0, (0.0, 0.30, 0.0), 0.0, 0.0),   # hold
        (28.0, 30.0, (0.0, 0.0, 0.0), 0.0, 0.0),    # back
        (30.0, 33.0, (0.0, 0.0, 0.0), 0.0, 0.0),    # rest
        (33.0, 35.0, (0.0, 0.0, 0.0), 45.0, 0.0),   # wrist roll in
        (35.0, 38.0, (0.0, 0.0, 0.0), 45.0, 0.0),   # hold (ori check)
        (38.0, 40.0, (0.0, 0.0, 0.0), 0.0, 0.0),    # back
        (40.0, 42.0, (0.0, 0.0, 0.0), 0.0, 45.0),   # wrist pitch in
        (42.0, 45.0, (0.0, 0.0, 0.0), 0.0, 45.0),   # hold (ori check)
        (45.0, 47.0, (0.0, 0.0, 0.0), 0.0, 0.0),    # back
        (47.0, 50.0, (0.0, 0.0, 0.0), 0.0, 0.0),    # rest
    ]

    def _demo_state(self, t: float):
        """(offset[3], roll_rad, pitch_rad) at demo-local time."""
        tl = max(0.0, t - self.MOVE_T) % self._DEMO_PERIOD
        prev = np.zeros(5)
        for t0, t1, off, roll, pitch in self._DEMO_SEGS:
            tgt = np.array([*off, np.deg2rad(roll), np.deg2rad(pitch)])
            if t0 <= tl < t1:
                a = (tl - t0) / max(t1 - t0, 1e-6)
                cur = prev + (tgt - prev) * a
                return cur[:3], float(cur[3]), float(cur[4])
            prev = tgt
        return prev[:3], float(prev[3]), float(prev[4])

    @classmethod
    def stress(cls) -> "MockPicoSource":
        return cls(radius=0.22, freq_hz=0.30, roll_deg=110.0, pitch_deg=70.0)

    @classmethod
    def demo(cls) -> "MockPicoSource":
        """Legible pose-hold sequence for eyeballing skeleton<->robot mapping
        consistency (forward push / lateral raise / overhead, 3s holds, no
        wrist rotation, static waist). Holds make the comparison immune to a
        slower-than-realtime sim: the robot arrives late but must settle at
        the SAME held position."""
        m = cls(roll_deg=0.0, lean_deg=0.0)
        m.demo_mode = True
        return m

    def start(self):
        pass

    def stop(self):
        pass

    def sample_at(self, t: float) -> PicoFrame:
        head = np.array([0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.0])
        if self.demo_mode:
            # pose-hold sequence + pure wrist-rotation holds, static waist
            off, roll, pitch = self._demo_state(t)
            quat = _quat_xyzw_about([0, 0, 1], roll)
            if pitch:
                quat = _quat_mul_xyzw(_quat_xyzw_about([1, 0, 0], pitch), quat)
            # same mirror rule as circle mode: (x,y,z,w) -> (x,-y,-z,w)
            quat_l = np.array([quat[0], -quat[1], -quat[2], quat[3]])
            left = np.array([-0.25 - off[0], 1.2 + off[1], -0.35 + off[2], *quat_l])
            right = np.array([+0.25 + off[0], 1.2 + off[1], -0.35 + off[2], *quat])
            waist_quat = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            phase = 2.0 * np.pi * self.freq * max(0.0, t - self.MOVE_T)
            # circle in the raw x/y (right/up) plane + gentle push forward (-z)
            dx = self.radius * np.sin(phase)
            dy = self.radius * (1.0 - np.cos(phase))
            dz = -0.05 * np.sin(0.5 * phase)
            quat = _quat_xyzw_about([0, 0, 1], self.roll * np.sin(phase))  # roll about raw z
            if self.pitch:
                quat = _quat_mul_xyzw(
                    _quat_xyzw_about([1, 0, 0], self.pitch * np.sin(0.7 * phase)), quat)
            # mirror across the raw y/z plane: R' = S R S with S = diag(-1,1,1)
            # -> quaternion (x,y,z,w) -> (x,-y,-z,w)
            quat_l = np.array([quat[0], -quat[1], -quat[2], quat[3]])
            left = np.array([-0.25 - dx, 1.2 + dy, -0.35 + dz, *quat_l])
            right = np.array([+0.25 + dx, 1.2 + dy, -0.35 + dz, *quat])
            # waist tracker: forward lean = world +y rotation = raw rotation about -x
            lean = self.lean * np.sin(0.5 * phase)
            waist_quat = _quat_xyzw_about([-1, 0, 0], lean)
        # wrist trackers ride with the controllers (a tracker strapped to the
        # wrist sees ~the same pose as the held controller); slightly below the
        # waist in height so relative-to-waist targets are sane.
        iq4 = (0.0, 0.0, 0.0, 1.0)
        l_sh, r_sh = (-0.18, 1.40, 0.0), (0.18, 1.40, 0.0)
        trackers = {
            self.WAIST_SN: np.array([0.0, 1.05, -0.05, *waist_quat]),
            self.LWRIST_SN: left.copy(),
            self.RWRIST_SN: right.copy(),
            # synthetic elbows: midpoint(shoulder, wrist), matching _fake_body24
            self.LELBOW_SN: np.array([*( (np.asarray(l_sh) + left[:3]) / 2 ), *iq4]),
            self.RELBOW_SN: np.array([*( (np.asarray(r_sh) + right[:3]) / 2 ), *iq4]),
        }
        grip = 1.0 if t >= self.ENGAGE_T else 0.0
        return PicoFrame(
            t_us=int(t * 1e6),
            head=head, left=left, right=right,
            left_grip=grip, right_grip=grip,
            trackers=trackers,
        )

    def get_latest(self) -> PicoFrame:  # wall-clock fallback (unused in sim)
        return self.sample_at(time.time())


class LimbLengthGate:
    """Catches the body model extrapolating an arm, from the forearm length.

    Full-body mode never stops streaming, so freeze detection is blind to its
    worst failure: when a wrist leaves the headset's view the model keeps
    emitting a confident, wrong pose. On the 2026-07-29 batch "arms high" and
    "arms forward" came out 0.168 m apart when they are ~1 m apart in reality,
    and nothing in the pipeline noticed for days.

    It is detectable because PICO's skeleton is only partly rigid. Measured
    over 22431 frames of that batch: upper arm and spine hold to 0.000 mm of
    standard deviation, but elbow->wrist ranges over 128-896 mm. So the wrist
    is not bound by the skeleton, and a forearm that stretches is the model
    reaching for a wrist it cannot see.

    The length is NOT a person constant across a session - it came out 255,
    280, 318 and 342 mm on different takes - so this compares against a
    rolling median of the arm's own recent history, not a fixed value.
    """

    def __init__(self, tol_frac: float = 0.15, window: int = 120,
                 min_samples: int = 30):
        self.tol_frac = tol_frac
        self.window = window
        self.min_samples = min_samples
        self._hist: dict[str, deque] = {}
        self.bad: dict[str, bool] = {}

    def update(self, hand: str, elbow_pos, wrist_pos) -> bool:
        """-> True when this arm's estimate should not be trusted."""
        if elbow_pos is None or wrist_pos is None:
            self.bad[hand] = False
            return False
        d = float(np.linalg.norm(np.asarray(wrist_pos, dtype=float)[:3]
                                 - np.asarray(elbow_pos, dtype=float)[:3]))
        h = self._hist.setdefault(hand, deque(maxlen=self.window))
        if len(h) >= self.min_samples:
            med = float(np.median(h))
            # judge BEFORE admitting the sample, so a run of bad frames cannot
            # drag the reference along with it
            if med > 1e-3 and abs(d - med) / med > self.tol_frac:
                self.bad[hand] = True
                return True
        h.append(d)
        self.bad[hand] = False
        return False
