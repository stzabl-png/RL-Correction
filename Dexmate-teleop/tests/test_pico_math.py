"""No-sim unit tests for the PICO teleop plumbing (numpy/scipy/msgpack only).

Covers the pieces a real-headset session depends on but the Isaac mock smoke
can't check in isolation:
  - xr_pose frame math (axis mapping, yaw-compensation invariance, quat round-trip)
  - producer msgpack message <-> consumer PicoFrame wire contract
  - PicoLogSource replay indexing (incl. loop mode)

Run:  .venv-isaac/bin/python -m pytest tests/test_pico_math.py -q
(any env with numpy+scipy+msgpack+pytest works; no Isaac import here)
"""

import os
import sys

import msgpack
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from magicdexmate.pico.xr_pose import (  # noqa: E402
    OneEuroFilter,
    QuatOneEuroFilter,
    R_HEADSET_TO_WORLD,
    mat_to_pos_quat_wxyz,
    pitch_delta,
    pose7_to_world_rot,
    process_xr_pose,
    yaw_align_from_wrists,
)
from magicdexmate.sources.pico_source import (  # noqa: E402
    DeviceStallDetector,
    TrackerFreezeDetector,
    TrackerGlitchGate,
    GripLatch,
    MockPicoSource,
    PicoFrame,
    PicoLogSource,
    frame_from_dict,
)
from teleop_pico_producer import read_fake_msg  # noqa: E402


# ---------------------------------------------------------------- xr_pose math

def test_headset_to_world_is_proper_rotation():
    M = R_HEADSET_TO_WORLD
    assert np.allclose(M @ M.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(M), 1.0)


def test_axis_mapping():
    M = R_HEADSET_TO_WORLD
    # raw forward (-z) -> world +x ; raw up (+y) -> world +z ; raw right (+x) -> world -y
    assert np.allclose(M @ [0, 0, -1], [1, 0, 0])
    assert np.allclose(M @ [0, 1, 0], [0, 0, 1])
    assert np.allclose(M @ [1, 0, 0], [0, -1, 0])


def test_mat_to_pos_quat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        T = np.eye(4)
        T[:3, :3] = R.random(random_state=rng).as_matrix()
        T[:3, 3] = rng.normal(size=3)
        pos, q_wxyz = mat_to_pos_quat_wxyz(T)
        rot_back = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
        assert np.allclose(pos, T[:3, 3])
        assert np.allclose(rot_back, T[:3, :3], atol=1e-10)


def _pose7(pos, rot_mat):
    q = R.from_matrix(rot_mat).as_quat()  # xyzw
    return np.array([*pos, *q])


def test_yaw_compensation_invariance():
    """Operator walks anywhere / turns in place: same controller pose relative
    to the (yaw-only) headset must give the same processed transform."""
    offset_raw = np.array([0.1, -0.35, -0.45])       # controller rel headset, raw frame
    c_local = R.from_euler("xyz", [20, -35, 50], degrees=True).as_matrix()

    ref = None
    for yaw_deg, h_pos in [(0, [0, 1.6, 0]), (90, [2.0, 1.5, -1.0]),
                           (-137, [-3.0, 1.7, 4.0]), (200, [0.5, 1.4, 0.2])]:
        Ry = R.from_euler("y", yaw_deg, degrees=True).as_matrix()  # raw frame is y-up
        h_pos = np.asarray(h_pos, dtype=np.float64)
        c_pos = h_pos + Ry @ offset_raw
        T = process_xr_pose(_pose7(c_pos, Ry @ c_local), _pose7(h_pos, Ry))
        if ref is None:
            ref = T
        else:
            assert np.allclose(T, ref, atol=1e-9), f"yaw={yaw_deg} broke invariance"


def test_known_geometry():
    """Controller 0.5 m in front of / 0.4 m below the headset, no rotation:
    processed frame must be x=+0.5 (forward), z=-0.4 (down)."""
    head = np.array([0.0, 1.6, 0.0, 0, 0, 0, 1])
    ctrl = np.array([0.0, 1.2, -0.5, 0, 0, 0, 1])
    T = process_xr_pose(ctrl, head)
    assert np.allclose(T[:3, 3], [0.5, 0.0, -0.4], atol=1e-12)
    assert np.allclose(T[:3, :3], np.eye(3), atol=1e-12)


def test_zero_quaternion_guard():
    """SDK reports all-zero quats before the app streams: must not crash."""
    head = np.array([0.0, 1.6, 0.0, 0, 0, 0, 0])
    ctrl = np.array([0.2, 1.2, -0.4, 0, 0, 0, 0])
    T = process_xr_pose(ctrl, head)
    assert np.all(np.isfinite(T))


# ------------------------------------------------------------- wire contract

def test_producer_message_roundtrip_to_frame():
    """read_fake_msg (producer) -> msgpack -> frame_from_dict (consumer):
    the exact wire contract both ends must agree on."""
    msg = read_fake_msg(MockPicoSource(), t=2.0)  # grips closed, motion started
    frame = frame_from_dict(msgpack.unpackb(msgpack.packb(msg)))

    assert frame.t_us == msg["t_us"]
    assert np.allclose(frame.head, msg["head"])
    assert np.allclose(frame.left, msg["left"])
    assert np.allclose(frame.right, msg["right"])
    assert frame.left_grip == pytest.approx(1.0)
    assert frame.grip("right") == pytest.approx(1.0)
    assert np.allclose(frame.controller("left"), msg["left"])
    assert set(frame.buttons) == {"A", "B", "X", "Y", "left_menu", "right_menu",
                                  "left_axis_click", "right_axis_click"}
    assert frame.left_joystick == (0.0, 0.0)
    # poses feed straight into process_xr_pose - must be finite unit-quat 7-vectors
    for p in (frame.head, frame.left, frame.right):
        assert p.shape == (7,)
        assert np.isclose(np.linalg.norm(p[3:]), 1.0, atol=1e-6)


def test_mock_source_quats_normalized_incl_stress():
    for src in (MockPicoSource(), MockPicoSource.stress()):
        for t in np.linspace(0.0, 10.0, 37):
            f = src.sample_at(float(t))
            assert np.isclose(np.linalg.norm(f.left[3:]), 1.0, atol=1e-9)
            assert np.isclose(np.linalg.norm(f.right[3:]), 1.0, atol=1e-9)
    assert MockPicoSource().sample_at(0.0).left_grip == 0.0
    assert MockPicoSource().sample_at(1.5).left_grip == 1.0


# ------------------------------------------------------------ one-euro filter

def test_one_euro_constant_input_is_identity():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    for _ in range(50):
        out = f(np.array([1.0, -2.0, 3.0]), dt=1 / 60)
    assert np.allclose(out, [1.0, -2.0, 3.0], atol=1e-6)


def test_one_euro_reduces_jitter_at_rest():
    rng = np.random.default_rng(1)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    raw, filt = [], []
    for _ in range(600):
        x = rng.normal(0.0, 0.005, size=3)  # 5 mm sensor jitter around rest
        raw.append(x)
        filt.append(f(x, dt=1 / 60))
    raw_std = np.std(np.asarray(raw)[100:], axis=0).mean()
    filt_std = np.std(np.asarray(filt)[100:], axis=0).mean()
    assert filt_std < 0.5 * raw_std, f"filter too weak: {filt_std:.4f} vs {raw_std:.4f}"


def test_one_euro_tracks_fast_motion_with_low_lag():
    """beta opens the cutoff with speed: a fast ramp must be followed closely.
    (1.0, 2.0) are the teleop consumer's defaults - keep them in sync."""
    f = OneEuroFilter(min_cutoff=1.0, beta=2.0)
    dt, v = 1 / 60, 1.0  # 1 m/s sweep
    lag = 0.0
    for i in range(120):
        x = np.array([v * i * dt])
        out = f(x, dt=dt)
        lag = float(x[0] - out[0])
    assert lag < 0.03, f"lag {lag * 1000:.1f}mm at 1 m/s exceeds 30mm"


def test_one_euro_reset_forgets_state():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    for _ in range(20):
        f(np.array([100.0]), dt=1 / 60)
    f.reset()
    assert np.allclose(f(np.array([0.0]), dt=1 / 60), [0.0])


def test_quat_filter_unit_norm_and_hemisphere_continuity():
    rng = np.random.default_rng(2)
    f = QuatOneEuroFilter(min_cutoff=1.0, beta=0.5)
    prev = None
    for i in range(200):
        angle = 0.4 * np.sin(2 * np.pi * 0.25 * i / 60)
        q = R.from_rotvec([0, 0, angle]).as_quat()  # xyzw
        q = np.array([q[3], q[0], q[1], q[2]])       # wxyz
        if rng.random() < 0.3:
            q = -q                                   # sign flips must not glitch
        out = f(q, dt=1 / 60)
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-9)
        if prev is not None:
            assert float(np.dot(out, prev)) > 0.9    # continuous, no jumps
        prev = out


# ------------------------------------------------------------ tracker / torso

def test_pitch_delta_pure_pitch():
    ref = np.eye(3)
    now = R.from_rotvec([0, 0.3, 0]).as_matrix()
    assert pitch_delta(now, ref) == pytest.approx(0.3, abs=1e-9)
    assert pitch_delta(ref, now) == pytest.approx(-0.3, abs=1e-9)
    # pure roll has no pitch component
    roll = R.from_rotvec([0.4, 0, 0]).as_matrix()
    assert pitch_delta(roll, ref) == pytest.approx(0.0, abs=1e-9)


def test_mock_waist_leans_forward_positive_world_pitch():
    """The waist tracker's raw -x rotation must come out as a positive world-y
    (lean forward) pitch in the ABSOLUTE world frame - this is exactly what
    TorsoChannel now reads (pose7_to_world_rot, no reference frame)."""
    src = MockPicoSource(lean_deg=15.0)
    t = src.MOVE_T + 1.0 / (4 * 0.5 * src.freq)  # quarter period of the lean sine
    f = src.sample_at(t)
    assert src.WAIST_SN in f.trackers
    rot_now = pose7_to_world_rot(f.trackers[src.WAIST_SN])
    rot_ref = pose7_to_world_rot(src.sample_at(src.MOVE_T).trackers[src.WAIST_SN])
    assert pitch_delta(rot_now, rot_ref) > 0.1


def test_mock_emits_three_named_trackers():
    f = MockPicoSource().sample_at(2.0)
    assert set(f.trackers) == {MockPicoSource.WAIST_SN, MockPicoSource.LWRIST_SN,
                               MockPicoSource.RWRIST_SN, MockPicoSource.LELBOW_SN,
                               MockPicoSource.RELBOW_SN}
    # wrist trackers colocated with the controllers (so both modes match)
    assert np.allclose(f.trackers[MockPicoSource.LWRIST_SN], f.left)
    assert np.allclose(f.trackers[MockPicoSource.RWRIST_SN], f.right)


def test_body_relative_wrist_is_pose_invariant():
    """Body-relative mapping: a wrist held at a fixed offset relative to the
    waist must give the SAME processed target no matter where the operator
    stands or which way their body faces. This is why the tracker mapping
    feels natural where the head-referenced incremental one felt 'weird'."""
    offset_raw = np.array([0.30, -0.15, -0.40])         # wrist rel waist, raw frame
    wrist_local = R.from_euler("xyz", [15, -25, 40], degrees=True).as_matrix()

    ref = None
    for yaw_deg, w_pos in [(0, [0, 1.05, 0]), (75, [1.5, 1.0, -2.0]),
                           (-160, [-2.0, 1.1, 3.0]), (210, [0.4, 1.0, 0.3])]:
        Ry = R.from_euler("y", yaw_deg, degrees=True).as_matrix()  # raw is y-up
        w_pos = np.asarray(w_pos, dtype=np.float64)
        c_pos = w_pos + Ry @ offset_raw
        T = process_xr_pose(_pose7(c_pos, Ry @ wrist_local), _pose7(w_pos, Ry))
        if ref is None:
            ref = T
        else:
            assert np.allclose(T, ref, atol=1e-9), f"body-yaw {yaw_deg} broke invariance"


def test_tracker_wire_roundtrip():
    msg = read_fake_msg(MockPicoSource(), t=2.0)
    assert msg["trackers"], "fake producer message must carry the mock tracker"
    frame = frame_from_dict(msgpack.unpackb(msgpack.packb(msg)))
    assert set(frame.trackers) == set(msg["trackers"])
    for sn, p in frame.trackers.items():
        assert p.shape == (7,)
        assert np.allclose(p, msg["trackers"][sn])
    # old recordings (no trackers key) parse to an empty dict
    old = {k: v for k, v in msgpack.unpackb(msgpack.packb(msg)).items()
           if k != "trackers"}
    assert frame_from_dict(old).trackers == {}


def test_mock_left_mirrors_right():
    """Left wrist must be the y/z-plane mirror of the right so both arms get
    an equally reachable task (vega's left-arm limits are mirrored)."""
    src = MockPicoSource()
    for t in (2.0, 4.7, 9.3):
        f = src.sample_at(t)
        # raw positions mirror in x
        assert np.allclose(f.left[:3] * [-1, 1, 1], f.right[:3])
        # processed transforms mirror across the world x/z plane (y -> -y)
        TL = process_xr_pose(f.left, f.head)
        TR = process_xr_pose(f.right, f.head)
        S = np.diag([1.0, -1.0, 1.0])
        assert np.allclose(TL[:3, 3], S @ TR[:3, 3], atol=1e-9)
        assert np.allclose(TL[:3, :3], S @ TR[:3, :3] @ S, atol=1e-9)


# ------------------------------------------------------------------ grip latch

def test_grip_latch_hysteresis():
    latch = GripLatch(on_th=0.6, off_th=0.4)
    assert not latch.update(0.5)      # below on threshold: stays released
    assert latch.update(0.65)         # engages
    assert latch.update(0.45)         # hovers above off threshold: holds
    assert not latch.update(0.35)     # releases
    assert not latch.update(0.55)     # needs >= 0.6 again
    assert latch.update(0.60)


# --------------------------------------------------- device stall detection

def _frame(t_us, device_ts_ns):
    z = np.array([0, 0, 0, 0, 0, 0, 1.0])
    return PicoFrame(t_us=t_us, head=z, left=z, right=z, device_ts_ns=device_ts_ns)


def test_device_ts_passes_through_wire():
    msg = read_fake_msg(MockPicoSource(), t=2.0)
    assert "device_ts_ns" in msg
    frame = frame_from_dict(msgpack.unpackb(msgpack.packb(msg)))
    assert frame.device_ts_ns == msg["device_ts_ns"]


def test_stall_detector_flags_frozen_device():
    """Producer keeps stamping t_us (100 Hz), device_ts frozen: must flag."""
    det = DeviceStallDetector(stall_ms=300.0)
    # device advancing normally -> never stalled
    for i in range(10):
        assert not det.update(_frame(t_us=i * 10_000, device_ts_ns=1000 + i))
    # device_ts freezes at 1009; t_us keeps climbing
    frozen = 1009
    t0 = 90_000
    assert not det.update(_frame(t_us=t0 + 100_000, device_ts_ns=frozen))   # 100ms < 300
    assert not det.update(_frame(t_us=t0 + 250_000, device_ts_ns=frozen))   # 250ms
    assert det.update(_frame(t_us=t0 + 400_000, device_ts_ns=frozen))       # 400ms > 300
    assert det.stalled


def test_stall_detector_recovers():
    det = DeviceStallDetector(stall_ms=300.0)
    det.update(_frame(t_us=0, device_ts_ns=5))
    assert det.update(_frame(t_us=500_000, device_ts_ns=5))    # stalled
    # device clock ticks again -> immediately fresh
    assert not det.update(_frame(t_us=510_000, device_ts_ns=6))
    assert not det.stalled


def test_stall_detector_ignores_deviceless_sources():
    """mock / old recordings report device_ts_ns == 0: never stale here."""
    det = DeviceStallDetector(stall_ms=100.0)
    for i in range(20):
        assert not det.update(_frame(t_us=i * 1_000_000, device_ts_ns=0))
    assert not det.update(None)


# ---------------------------------------------------------------- log replay

def _write_log(path, times_s):
    mock = MockPicoSource()
    with open(path, "wb") as f:
        for i, ts in enumerate(times_s):
            msg = read_fake_msg(mock, t=ts)
            msg["t_us"] = int(ts * 1e6)          # deterministic timestamps
            msg["head"][0] = float(i)            # tag frame index for asserts
            f.write(msgpack.packb(msg))


def test_log_source_replay(tmp_path):
    path = str(tmp_path / "session.mpk")
    _write_log(path, [0.0, 0.1, 0.2])
    src = PicoLogSource(path)
    assert len(src.frames) == 3
    assert src.duration == pytest.approx(0.2)
    assert src.sample_at(-1.0).head[0] == 0      # clamps to first
    assert src.sample_at(0.05).head[0] == 0
    assert src.sample_at(0.15).head[0] == 1
    assert src.sample_at(99.0).head[0] == 2      # holds last


def test_log_source_loop(tmp_path):
    path = str(tmp_path / "session.mpk")
    _write_log(path, [0.0, 0.1, 0.2])
    src = PicoLogSource(path, loop=True)
    assert src.sample_at(0.25).head[0] == 0      # 0.25 % 0.2 = 0.05 -> frame 0
    assert src.sample_at(0.35).head[0] == 1


def test_log_source_empty_file_raises(tmp_path):
    path = tmp_path / "empty.mpk"
    path.write_bytes(b"")
    with pytest.raises(ValueError):
        PicoLogSource(str(path))


# ------------------------------------------- tracker value-freeze detection

def _tframe(t_us, device_ts_ns, jitter=0.0):
    z = np.array([0, 0, 0, 0, 0, 0, 1.0])
    trk = {"LWRIST": np.array([0.1 + jitter, 0.2, 0.3, 0, 0, 0, 1.0]),
           "WAIST": np.array([0.0, 0.0, 0.0, 0, 0, 0, 1.0])}
    return PicoFrame(t_us=t_us, head=z, left=z, right=z,
                     device_ts_ns=device_ts_ns, trackers=trk)


def test_freeze_detector_flags_frozen_values():
    """Body estimate freezes (values bit-identical) while device_ts ticks:
    2026-07-26 live failure - DeviceStallDetector can't see it, this must."""
    det = TrackerFreezeDetector(freeze_ms=500.0)
    # live values (jitter changes every frame) -> never frozen
    for i in range(10):
        assert not det.update(_tframe(i * 10_000, 1000 + i, jitter=1e-6 * i))
    # values freeze solid; device clock keeps advancing
    t0 = 100_000
    assert not det.update(_tframe(t0 + 100_000, 2000, jitter=0.5))  # new value
    assert not det.update(_tframe(t0 + 400_000, 2001, jitter=0.5))  # 300ms same
    assert det.update(_tframe(t0 + 700_000, 2002, jitter=0.5))      # 600ms same
    assert det.frozen


def test_freeze_detector_recovers_and_exempts_mock():
    det = TrackerFreezeDetector(freeze_ms=500.0)
    det.update(_tframe(0, 1, jitter=0.1))
    assert det.update(_tframe(600_000, 2, jitter=0.1))       # frozen
    assert not det.update(_tframe(610_000, 3, jitter=0.2))   # value changed
    # mock (device_ts_ns == 0): constant values are legitimate, never frozen
    det2 = TrackerFreezeDetector(freeze_ms=100.0)
    for i in range(10):
        assert not det2.update(_tframe(i * 200_000, 0, jitter=0.0))


def _hybrid_frame(t_us, device_ts_ns, wrist_jitter):
    """2026-07-29 hybrid rig: raw per-tracker wrists (live) + model-channel
    waist/elbows (dead in the _raw batch - full-body calib never redone)."""
    z = np.array([0, 0, 0, 0, 0, 0, 1.0])
    trk = {"RAW-L": np.array([0.1 + wrist_jitter, 0.2, 0.3, 0, 0, 0, 1.0]),
           "RAW-R": np.array([-0.1 - wrist_jitter, 0.2, 0.3, 0, 0, 0, 1.0]),
           "WAIST": np.array([0.0, 0.0, 1.0, 0, 0, 0, 1.0])}   # never changes
    return PicoFrame(t_us=t_us, head=z, left=z, right=z,
                     device_ts_ns=device_ts_ns, trackers=trk)


def test_freeze_detector_aggregate_is_blind_to_one_dead_channel():
    """Regression for the hybrid rig: live raw wrists keep the aggregate value
    changing, so a dead model waist slips past the all-tracker detector. In
    --map waist-abs that waist is the mapping origin - the 2026-07-29 replay
    silently produced 280mm of left-arm error with zero warnings."""
    agg = TrackerFreezeDetector(freeze_ms=500.0)
    for i in range(20):
        assert not agg.update(_hybrid_frame(i * 100_000, 1000 + i, 1e-4 * i))


def test_freeze_detector_per_channel_catches_dead_waist():
    """Same stream, one detector per consumed channel: the waist is caught
    while the live wrists stay clean."""
    waist = TrackerFreezeDetector(freeze_ms=500.0, keys={"WAIST"}, label="waist")
    wrists = [TrackerFreezeDetector(freeze_ms=500.0, keys={k}, label=k)
              for k in ("RAW-L", "RAW-R")]
    frames = [_hybrid_frame(i * 100_000, 1000 + i, 1e-4 * i) for i in range(20)]
    assert not waist.update(frames[0])                 # first sample: no history
    assert not waist.update(frames[3])                 # 300ms identical: under
    assert waist.update(frames[8])                     # 800ms identical: frozen
    assert waist.label == "waist"
    for f in frames:                                   # wrists never flagged
        assert not any(w.update(f) for w in wrists)


def test_freeze_detector_absent_channel_is_not_a_freeze():
    """A channel missing from the stream is a different fault (reported
    elsewhere) - don't cry freeze when there is no data to judge."""
    det = TrackerFreezeDetector(freeze_ms=100.0, keys={"NOPE"}, label="ghost")
    for i in range(10):
        assert not det.update(_hybrid_frame(i * 200_000, 1000 + i, 0.0))
    assert not det.frozen


# ---------------------------------------------------- tracker glitch gate

def _pose(x, qw=1.0, qz=0.0):
    return np.array([x, 0.0, 0.0, 0.0, 0.0, qz, qw])


def test_glitch_gate_passes_plausible_motion():
    gate = TrackerGlitchGate()
    t = 0
    for i in range(50):
        t += 10_000  # 100 Hz
        p, re = gate.update(t, _pose(0.01 * i))  # 1 m/s
        assert not re
        assert np.allclose(p, _pose(0.01 * i))
    assert gate.n_rejected == 0


def test_glitch_gate_rejects_spike_then_recovers():
    gate = TrackerGlitchGate()
    gate.update(0, _pose(0.0))
    # 30cm teleport in 10ms: impossible -> held at last accepted
    p, re = gate.update(10_000, _pose(0.30))
    assert not re and np.allclose(p, _pose(0.0))
    # next frame back to sane: accepted again
    p, re = gate.update(20_000, _pose(0.01))
    assert not re and np.allclose(p, _pose(0.01))
    assert gate.n_rejected == 1 and gate.n_reanchors == 0


def test_glitch_gate_sustained_relocation_signals_reanchor():
    gate = TrackerGlitchGate(accept_after_ms=400.0)
    gate.update(0, _pose(0.0))
    t, re_seen = 0, False
    for _ in range(60):  # 600ms of a persistent far pose
        t += 10_000
        p, re = gate.update(t, _pose(1.0))
        if re:
            re_seen = True
            break
    assert re_seen                       # gave up and adopted the new pose
    assert np.allclose(p, _pose(1.0))
    p, re = gate.update(t + 10_000, _pose(1.0))
    assert not re                        # tracking continues from new home


def test_glitch_gate_rejects_orientation_snap():
    gate = TrackerGlitchGate()
    gate.update(0, _pose(0.0, qw=1.0, qz=0.0))
    # 170deg single-step rotation in 10ms (the audit's signature): rejected
    q170 = np.array([0.0, 0.0, 0.0, 0, 0, np.sin(np.deg2rad(85)), np.cos(np.deg2rad(85))])
    p, re = gate.update(10_000, q170)
    assert not re and np.allclose(p, _pose(0.0, qw=1.0, qz=0.0))


def test_glitch_gate_scales_thresholds_with_dt():
    """Slow consumption (0.1x sim: ~160ms between frames) must not reject
    normal-speed human motion."""
    gate = TrackerGlitchGate()
    gate.update(0, _pose(0.0))
    # 0.3m step over 160ms = 1.9 m/s: plausible, must pass
    p, re = gate.update(160_000, _pose(0.30))
    assert not re and np.allclose(p, _pose(0.30))


def test_yaw_align_from_wrists_neutral_stance():
    # raw PICO frame: x=right, y=up, -z=front. Hands apart at the sides.
    Ra = yaw_align_from_wrists([-0.3, 1.0, 0.0], [0.3, 1.0, 0.0])
    assert np.allclose(Ra[:, 0], [0, 0, -1])   # front = -z
    assert np.allclose(Ra[:, 1], [-1, 0, 0])   # left = -x
    assert np.allclose(Ra[:, 2], [0, 1, 0])    # up = +y
    assert np.allclose(Ra @ Ra.T, np.eye(3), atol=1e-12)


def test_yaw_align_from_wrists_turned_operator():
    # left-hand on the -z side means the operator faces +x (left = up x front:
    # front=+x, up=+y -> left=-z); height offset must be projected away
    Ra = yaw_align_from_wrists([0.0, 1.1, -0.3], [0.0, 0.9, 0.3])
    assert np.allclose(Ra[:, 0], [1, 0, 0])    # front = +x
    assert np.allclose(Ra[:, 1], [0, 0, -1])   # left = -z
    assert np.allclose(Ra[:, 2], [0, 1, 0])


def test_yaw_align_from_wrists_rejects_degenerate():
    with pytest.raises(ValueError):
        yaw_align_from_wrists([0.0, 1.0, 0.0], [0.03, 1.0, 0.0])  # too close
    with pytest.raises(ValueError):
        yaw_align_from_wrists([0.0, 1.4, 0.0], [0.02, 0.8, 0.0])  # vertical


# ------------------------------------------- forearm fusion (position bridge)

def _arc_sample(t, angle, elbow=(0.3, 0.0, 1.0), v=(0.0, 0.0, -0.28)):
    """Wrist swinging on a rigid forearm about a fixed elbow."""
    rot = R.from_euler("y", angle).as_matrix()
    return t, np.asarray(elbow) + rot @ np.asarray(v), rot


def test_forearm_fuser_bridges_a_dropout():
    """Position dies, orientation keeps moving: the arc must carry the wrist.

    Freezing would leave it at the last sample; on the 2026-07-29 raw takes
    that is what drives the 400 mm p90 error this exists to cut."""
    from magicdexmate.pico.arm_fusion import ForearmFuser
    f = ForearmFuser(cal_window_s=1.0, blend_s=0.0, min_samples=20)
    for i in range(60):                       # 0.6 s of live arc
        t, p, rot = _arc_sample(i * 0.01, 0.4 * i * 0.01)
        out, ok = f.update(t, p, rot, position_live=True)
        assert ok and np.allclose(out, p)
    assert f.forearm_len_m == pytest.approx(0.28, abs=1e-3)
    # position dies; feed the true rotation and a stale position
    t, truth, rot = _arc_sample(0.9, 0.4 * 0.9)
    stale = _arc_sample(0.59, 0.4 * 0.59)[1]
    out, ok = f.update(t, stale, rot, position_live=False)
    assert ok
    assert np.linalg.norm(out - truth) < 0.005          # arc recovers it
    assert np.linalg.norm(stale - truth) > 0.03         # freezing would not


def test_forearm_fuser_gives_up_on_long_gaps():
    """Past max_bridge_s the estimate is not trustworthy - say so, so the
    caller disengages instead of following an invented target."""
    from magicdexmate.pico.arm_fusion import ForearmFuser
    f = ForearmFuser(cal_window_s=1.0, max_bridge_s=1.5, min_samples=20)
    for i in range(60):
        t, p, rot = _arc_sample(i * 0.01, 0.4 * i * 0.01)
        f.update(t, p, rot, position_live=True)
    _, ok = f.update(1.5, p, rot, position_live=False)
    assert ok
    _, ok = f.update(3.0, p, rot, position_live=False)
    assert not ok


def test_forearm_fuser_rejects_ill_conditioned_window():
    """A window with no rotation fits an arbitrary arc; the radius bound is
    what catches it, and then the bridge falls back to holding still."""
    from magicdexmate.pico.arm_fusion import ForearmFuser
    f = ForearmFuser(cal_window_s=1.0, min_samples=20, v_bounds_m=(0.05, 0.8))
    rot = np.eye(3)
    p = np.array([0.3, 0.0, 0.72])
    for i in range(60):
        f.update(i * 0.01, p, rot, position_live=True)
    assert f.forearm_len_m is None                      # no arc accepted
    out, ok = f.update(0.7, p, rot, position_live=False)
    assert ok and np.allclose(out, p)                   # holds the last good


# ---------------------------------------- limb-length gate (model collapse)

def _fore(d):
    """elbow at origin, wrist d metres away."""
    return [0.0, 0.0, 0.0], [d, 0.0, 0.0]


def test_limb_gate_passes_a_rigid_forearm():
    """A real forearm keeps its length. Small jitter must not trip the gate,
    or it would disengage constantly on healthy data."""
    from magicdexmate.sources.pico_source import LimbLengthGate
    g = LimbLengthGate()
    rng = np.random.default_rng(0)
    for _ in range(200):
        e, w = _fore(0.28 + rng.normal(0, 0.002))
        assert not g.update("right", e, w)


def test_limb_gate_catches_an_extrapolated_arm():
    """The 2026-07-29 failure: the model reaches for a wrist it cannot see and
    the forearm stretches. Measured range on that batch was 128-896 mm against
    a 0.000 mm rigid upper arm - this is the signature that catches it."""
    from magicdexmate.sources.pico_source import LimbLengthGate
    g = LimbLengthGate()
    for _ in range(60):
        g.update("right", *_fore(0.28))
    assert g.update("right", *_fore(0.50))      # stretched
    assert not g.update("right", *_fore(0.28))  # back on the bone


def test_limb_gate_reference_survives_a_bad_run():
    """A sustained stretch must not become the new normal: the median has to
    be judged against, then updated, or a slow drift teaches the gate to
    accept anything."""
    from magicdexmate.sources.pico_source import LimbLengthGate
    g = LimbLengthGate()
    for _ in range(60):
        g.update("right", *_fore(0.28))
    for _ in range(40):
        assert g.update("right", *_fore(0.55))
