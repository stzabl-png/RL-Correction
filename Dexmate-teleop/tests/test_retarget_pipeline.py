"""Pipeline unit tests. Run in the teleop venv:  .venv/bin/python -m pytest tests/ -q"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from magicdexmate.retarget.frames import to_mano  # noqa: E402
from magicdexmate.skeleton import FINGERTIP_INDICES, joint_name_to_index  # noqa: E402
from magicdexmate.sources.mock_source import MockGloveSource, hand_keypoints, motion_curl  # noqa: E402

ASSETS_READY = (REPO_ROOT / "assets/robots/hands/sharpa_wave/right/right_sharpa_wave.urdf").exists()
needs_assets = pytest.mark.skipif(not ASSETS_READY, reason="run scripts/prepare_assets.py first")


# -- skeleton / names ----------------------------------------------------------

def test_joint_name_mapping_variants():
    assert joint_name_to_index("wrist") == 0
    assert joint_name_to_index("index_finger_mcp") == 5
    assert joint_name_to_index("INDEX_MCP") == 5
    assert joint_name_to_index("right_thumb_tip") == 4
    assert joint_name_to_index("pinky_finger_tip") == 20
    assert joint_name_to_index("little_finger_pip") == 18
    with pytest.raises(KeyError):
        joint_name_to_index("elbow")


# -- mock source ---------------------------------------------------------------

def test_mock_geometry_plausible():
    kp_open = hand_keypoints(motion_curl("open", 0.0))
    assert kp_open.shape == (21, 3)
    # open hand: tips are the farthest point of each finger chain from wrist
    d_tips = np.linalg.norm(kp_open[FINGERTIP_INDICES], axis=1)
    assert (d_tips > 0.08).all() and (d_tips < 0.22).all()
    # fist brings non-thumb tips much closer to the wrist
    kp_fist = hand_keypoints(np.array([0.8, 1.3, 1.3, 1.3, 1.3]))
    d_fist = np.linalg.norm(kp_fist[FINGERTIP_INDICES[1:]], axis=1)
    assert (d_fist < 0.75 * d_tips[1:]).all()


def test_mock_pinch_brings_tips_together():
    src = MockGloveSource(hand="right", motion="pinch", noise=0.0)
    # pinch cycle period 3 s, index first; peak at t=1.5
    kp = src.sample_at(1.5).kp
    pinch_dist = np.linalg.norm(kp[4] - kp[8])
    open_dist = np.linalg.norm(src.sample_at(0.0).kp[4] - src.sample_at(0.0).kp[8])
    assert pinch_dist < 0.3 * open_dist
    # close enough to engage DexPilot's pinch projection (project_dist ~3 cm)
    assert pinch_dist < 0.025


def test_mock_source_threading():
    src = MockGloveSource(hand="right", motion="wave", rate=200.0)
    with src:
        import time

        time.sleep(0.1)
        frame = src.get_latest()
    assert frame is not None
    assert frame.kp.shape == (21, 3)
    assert frame.hand == "right"


# -- frame conversion ----------------------------------------------------------

def test_to_mano_rotation_invariance():
    """Conversion must not depend on the (unknown) glove wrist-frame convention."""
    rng = np.random.default_rng(7)
    kp = hand_keypoints(motion_curl("cycle", 5.0))
    ref = to_mano(kp, "right")
    for _ in range(5):
        # random proper rotation + translation of the raw input
        A = rng.normal(size=(3, 3))
        q, _ = np.linalg.qr(A)
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        moved = kp @ q.T + rng.normal(size=(1, 3))
        np.testing.assert_allclose(to_mano(moved, "right"), ref, atol=1e-9)


def test_to_mano_orientation_semantics():
    """dex-retargeting's MANO convention: open-hand fingers extend along +z,
    thumb toward +y (radial). Verified to match the Sharpa URDF base frame:
    zero-pose right_middle_fingertip is at z=0.203 in right_hand_C_MC."""
    kp = hand_keypoints(motion_curl("open", 0.0))
    mano = to_mano(kp, "right")
    tips = mano[FINGERTIP_INDICES[1:]]  # four fingers
    assert (tips[:, 2] > 0.1).all(), f"fingers not along +z: {tips}"
    assert mano[4, 1] > 0.05, f"thumb not toward +y: {mano[4]}"


# -- retargeting (needs prepared assets) ----------------------------------------

@needs_assets
def test_build_and_mapping():
    from magicdexmate.retarget.builder import build_sharpa_retargeting
    from magicdexmate.retarget.mapping import JointMapper, sharpa_sdk_joint_names

    r = build_sharpa_retargeting("right", "vector")
    assert len(r.joint_names) == 22
    mapper = JointMapper(r, "right")
    assert sorted(mapper.idx.tolist()) == list(range(22))
    assert mapper.sdk_names == sharpa_sdk_joint_names("right")
    assert (mapper.lo < mapper.hi).all()


@needs_assets
@pytest.mark.parametrize("mode", ["vector", "dexpilot"])
def test_mock_to_qpos_end_to_end(mode):
    from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value
    from magicdexmate.retarget.mapping import JointMapper

    retargeting = build_sharpa_retargeting("right", mode)
    mapper = JointMapper(retargeting, "right")
    src = MockGloveSource(hand="right", motion="fist", noise=0.0)

    flex = [i for i, n in enumerate(mapper.sdk_names) if n.endswith(("_FE", "_PIP", "_DIP", "_IP"))]
    q_open = q_fist = None
    for t in np.linspace(0.0, 2.0, 40):
        frame = src.sample_at(float(t))
        ref = compute_ref_value(retargeting, to_mano(frame.kp, "right"))
        q_sdk = mapper.to_sdk(retargeting.retarget(ref))
        assert np.isfinite(q_sdk).all()
        if q_open is None:
            q_open = q_sdk
        q_fist = q_sdk
    delta = float(np.mean(q_fist[flex] - q_open[flex]))
    assert delta > 0.3, f"{mode}: fist did not close fingers (delta={delta:.3f})"


@needs_assets
def test_left_hand_builds_and_runs():
    from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value
    from magicdexmate.retarget.mapping import JointMapper

    retargeting = build_sharpa_retargeting("left", "vector")
    mapper = JointMapper(retargeting, "left")
    src = MockGloveSource(hand="left", motion="fist", noise=0.0)
    for t in (0.0, 1.0, 2.0):
        ref = compute_ref_value(retargeting, to_mano(src.sample_at(t).kp, "left"))
        q = mapper.to_sdk(retargeting.retarget(ref))
        assert np.isfinite(q).all()


# -- zmq schema ----------------------------------------------------------------

def test_publisher_schema():
    import json
    import time

    import zmq

    from magicdexmate.retarget.mapping import sharpa_sdk_joint_names
    from magicdexmate.sinks.qpos_publisher import QposPublisher, names_crc

    names = sharpa_sdk_joint_names("right")
    port = 15917
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.connect(f"tcp://127.0.0.1:{port}")
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    pub = QposPublisher(f"tcp://*:{port}", "right", names)
    time.sleep(0.3)
    pub.send(np.linspace(0, 1, 22), t_capture_us=123, valid=True, wrist_quat=np.array([1, 0, 0, 0]))

    got = {}
    deadline = time.time() + 2.0
    while time.time() < deadline and len(got) < 2:
        try:
            m = json.loads(sub.recv_string(flags=zmq.NOBLOCK))
            got[m["type"]] = m
        except zmq.Again:
            time.sleep(0.01)
    pub.close()
    sub.close()

    assert got["hello"]["joint_names"] == names
    assert got["hello"]["crc"] == names_crc(names)
    q = got["qpos"]
    assert q["crc"] == names_crc(names) and len(q["qpos"]) == 22
    assert q["t_capture_us"] == 123 and q["valid"] is True
    assert q["wrist_quat"] == [1.0, 0.0, 0.0, 0.0]
