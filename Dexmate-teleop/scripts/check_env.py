#!/usr/bin/env python
"""Environment + pipeline self-test for the teleop venv. Run:

    .venv/bin/python scripts/check_env.py

Checks (in order): imports -> assets -> retargeting build (r/l, vector/dexpilot)
-> FK round-trip accuracy -> mock-glove end-to-end pipeline -> ZMQ loopback
-> retarget timing. Exits non-zero if any mandatory check fails.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, mandatory: bool = True):
    def deco(fn):
        def run():
            try:
                detail = fn() or ""
                RESULTS.append((name, True, str(detail)))
                print(f"  PASS  {name}  {detail}")
            except Exception as e:  # noqa: BLE001
                RESULTS.append((name, not mandatory, f"{type(e).__name__}: {e}"))
                tag = "WARN" if not mandatory else "FAIL"
                print(f"  {tag}  {name}  {type(e).__name__}: {e}")
        return run
    return deco


@check("python/numpy/dex_retargeting/zmq imports")
def check_imports():
    import numpy as np
    import zmq  # noqa: F401
    import dex_retargeting  # noqa: F401

    # numpy 1.26 (Isaac) and 2.x both verified to work - see docs/references/03 §6
    return f"py{sys.version_info.major}.{sys.version_info.minor} numpy={np.__version__}"


@check("optional: wuji_sdk import", mandatory=False)
def check_wuji():
    import wuji_sdk

    return f"wuji-sdk present ({getattr(wuji_sdk, '__version__', '?')})"


@check("optional: sapien import", mandatory=False)
def check_sapien():
    import sapien

    return f"sapien {sapien.__version__}"


@check("sharpa URDF assets prepared")
def check_assets():
    for side in ("right", "left"):
        p = REPO_ROOT / "assets/robots/hands/sharpa_wave" / side / f"{side}_sharpa_wave.urdf"
        assert p.exists(), f"missing {p} - run scripts/prepare_assets.py"
    return "right+left URDFs found"


@check("build retargeting: all 4 configs, 22 dof, SDK joint coverage")
def check_build():
    from magicdexmate.retarget.builder import build_sharpa_retargeting
    from magicdexmate.retarget.mapping import JointMapper

    for hand in ("right", "left"):
        for mode in ("vector", "dexpilot"):
            r = build_sharpa_retargeting(hand, mode)
            names = list(r.joint_names)
            assert len(names) == 22, f"{hand}/{mode}: expected 22 dof, got {len(names)}"
            JointMapper(r, hand)  # raises if SDK joints are not covered
    return "right/left x vector/dexpilot all build"


@check("sequential tracking: mock trajectory -> retarget -> FK fingertip error")
def check_tracking():
    """Measures the operative teleop regime: warm-started sequential retargeting
    along a continuous hand trajectory (NOT single-step jumps to random extreme
    poses - see docs/references/03_dex_retargeting.md 'known issues' for why)."""
    import numpy as np

    from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value
    from magicdexmate.retarget.frames import to_mano
    from magicdexmate.sources.mock_source import MockGloveSource

    retargeting = build_sharpa_retargeting("right", "vector")
    optimizer = retargeting.optimizer
    robot = optimizer.robot
    scaling = optimizer.scaling
    src = MockGloveSource(hand="right", motion="cycle", noise=0.0)

    def fk_vectors(q):
        robot.compute_forward_kinematics(q)
        pos = np.array([robot.get_link_pose(i)[:3, 3] for i in optimizer.computed_link_indices])
        return pos[np.asarray(optimizer.task_link_indices)] - pos[np.asarray(optimizer.origin_link_indices)]

    track = []
    for t in np.linspace(0.0, 26.0, 26 * 30):  # full open/fist/wave/pinch cycle @ 30 Hz
        ref = compute_ref_value(retargeting, to_mano(src.sample_at(float(t)).kp, "right"))
        qpos = retargeting.retarget(ref)
        track.append(np.linalg.norm(fk_vectors(qpos) - ref * scaling, axis=1))
    track = np.array(track) * 1000  # mm, columns: thumb..pinky

    finger_means = track.mean(axis=0)
    msg = f"per-finger mean {np.round(finger_means, 1)} mm, p50 {np.percentile(track, 50):.1f} mm"
    # four fingers must track tightly; thumb is looser (workspace mismatch, tune in R5)
    assert (finger_means[1:] < 15.0).all(), f"four-finger tracking too loose: {msg}"
    assert finger_means[0] < 40.0, f"thumb tracking too loose: {msg}"
    return msg


@check("robot hand size (for scaling_factor)")
def check_hand_size():
    import numpy as np

    from magicdexmate.retarget.builder import build_sharpa_retargeting

    retargeting = build_sharpa_retargeting("right", "vector")
    robot = retargeting.optimizer.robot
    robot.compute_forward_kinematics(np.zeros(robot.dof))
    root = robot.get_link_pose(robot.get_link_index("right_hand_C_MC"))[:3, 3]
    tip = robot.get_link_pose(robot.get_link_index("right_middle_fingertip"))[:3, 3]
    d = float(np.linalg.norm(tip - root))
    return f"wrist->middle fingertip = {d * 1000:.1f} mm; scaling_factor = {d:.3f} / your hand length"


@check("mock pipeline: glove frames -> MANO -> retarget -> SDK order")
def check_mock_pipeline():
    import numpy as np

    from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value
    from magicdexmate.retarget.frames import to_mano
    from magicdexmate.retarget.mapping import JointMapper
    from magicdexmate.sources.mock_source import MockGloveSource

    retargeting = build_sharpa_retargeting("right", "vector")
    mapper = JointMapper(retargeting, "right")
    src = MockGloveSource(hand="right", motion="fist", noise=0.0)

    qpos_open = qpos_fist = None
    for t in np.linspace(0.0, 2.0, 60):  # fist motion: open at t=0 -> closed at t=2
        frame = src.sample_at(float(t))
        ref = compute_ref_value(retargeting, to_mano(frame.kp, "right"))
        q_sdk = mapper.to_sdk(retargeting.retarget(ref))
        assert np.isfinite(q_sdk).all()
        assert (q_sdk >= mapper.lo - 1e-6).all() and (q_sdk <= mapper.hi + 1e-6).all()
        if qpos_open is None:
            qpos_open = q_sdk.copy()
        qpos_fist = q_sdk.copy()

    # Closing the fist must raise flexion joints substantially.
    flex = [i for i, n in enumerate(mapper.sdk_names) if n.endswith(("_FE", "_PIP", "_DIP", "_IP"))]
    delta = float(np.mean(qpos_fist[flex] - qpos_open[flex]))
    assert delta > 0.3, f"fist did not close fingers (mean flexion delta {delta:.3f} rad)"
    return f"fist closes fingers: mean flexion delta {delta:.2f} rad"


@check("ZMQ pub/sub loopback")
def check_zmq():
    import json

    import numpy as np
    import zmq

    from magicdexmate.sinks.qpos_publisher import QposPublisher
    from magicdexmate.retarget.mapping import sharpa_sdk_joint_names

    port = 15901
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://127.0.0.1:{port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    pub = QposPublisher(f"tcp://*:{port}", "right", sharpa_sdk_joint_names("right"))
    time.sleep(0.3)  # late-joiner settle
    pub.send(np.zeros(22), t_capture_us=0)
    msgs = []
    deadline = time.time() + 2.0
    while time.time() < deadline and len(msgs) < 2:
        try:
            msgs.append(json.loads(sub.recv_string(flags=zmq.NOBLOCK)))
        except zmq.Again:
            time.sleep(0.01)
    pub.close()
    sub.close()
    types = {m["type"] for m in msgs}
    assert "hello" in types and "qpos" in types, f"got only {types}"
    return "hello + qpos received"


@check("retarget timing")
def check_timing():
    import numpy as np

    from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value
    from magicdexmate.retarget.frames import to_mano
    from magicdexmate.sources.mock_source import MockGloveSource

    retargeting = build_sharpa_retargeting("right", "vector")
    src = MockGloveSource(hand="right", motion="cycle", noise=0.0)
    # warmup
    for t in np.linspace(0, 1, 10):
        retargeting.retarget(compute_ref_value(retargeting, to_mano(src.sample_at(float(t)).kp, "right")))
    times = []
    for t in np.linspace(0, 8, 100):
        ref = compute_ref_value(retargeting, to_mano(src.sample_at(float(t)).kp, "right"))
        t0 = time.perf_counter()
        retargeting.retarget(ref)
        times.append(time.perf_counter() - t0)
    ms = np.array(times) * 1000
    msg = f"p50 {np.percentile(ms, 50):.2f} ms, p95 {np.percentile(ms, 95):.2f} ms -> max ~{1000 / np.percentile(ms, 95):.0f} Hz"
    assert np.percentile(ms, 95) < 16, f"too slow for 60 Hz: {msg}"
    return msg


def main():
    print(f"== MagicDexMate check_env ({REPO_ROOT}) ==")
    for fn in (
        check_imports, check_wuji, check_sapien, check_assets, check_build,
        check_tracking, check_hand_size, check_mock_pipeline, check_zmq, check_timing,
    ):
        fn()
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"== {len(RESULTS) - len(failed)}/{len(RESULTS)} passed ==")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
