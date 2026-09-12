"""本轮新增模块的单测:默认姿势、环境障碍物、触觉差分录制。

    cd <仓库根目录>
    env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
        .venv/bin/python -m pytest tests/test_new_modules.py -q

为什么这三样值得进单测:它们都是**一处改了、别处静默变错**的那种耦合。
默认姿势原本有两份副本;环境障碍物的尺寸填错会让判据恒为真而不说原因;
触觉差分如果读的时候不解码,拿到的是一张看起来正常的差分图。

需要 pinocchio 的那几条会在没有它的环境里自动跳过(只有 .venv-isaac 有)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------- 默认姿势

def test_home_pose_covers_all_14_arm_joints():
    from magicdexmate.home_pose import ARM_HOME
    assert len(ARM_HOME) == 14
    for s in "LR":
        for i in range(1, 8):
            assert f"{s}_arm_j{i}" in ARM_HOME


def test_home_pose_matches_the_two_places_that_used_to_copy_it():
    """曾经有两份副本,各写着一行「MUST stay in sync」。这条锁住它们。"""
    import re
    from magicdexmate.home_pose import ARM_HOME

    scene = (REPO / "sim/vega_scene.py").read_text()
    seg = scene[scene.index("init_state=ArticulationCfg.InitialStateCfg"):
                scene.index("joint_vel=")]
    found = {m.group(1): float(m.group(2))
             for m in re.finditer(r'"([LR]_arm_j\d)":\s*(-?[\d.]+)', seg)}
    assert found, "没能从 vega_scene.py 里解析出 init_state,尺子坏了"
    for n, v in found.items():
        assert ARM_HOME[n] == pytest.approx(v), f"{n} 与 vega_scene 不一致"


def test_home_for_never_includes_the_torso():
    """躯干由操作者手工摆放或由映射驱动,「回默认位」这个动作不该碰它。"""
    from magicdexmate.home_pose import home_for
    got = home_for({"left_arm", "right_arm", "head", "torso"})
    assert not [n for n in got if n.startswith("torso")]
    assert len(got) == 17          # 14 臂 + 3 头


def test_home_for_selects_by_component():
    from magicdexmate.home_pose import home_for
    assert set(home_for({"left_arm"})) == {f"L_arm_j{i}" for i in range(1, 8)}
    assert home_for(set()) == {}


# ---------------------------------------------------- 触觉差分录制(需 h5py)

def _hdw():
    pytest.importorskip("h5py")
    from magicdexmate.recording import HandDataWriter, read_tactile
    return HandDataWriter, read_tactile


@pytest.mark.parametrize("delta", [False, True])
def test_tactile_roundtrip_is_bit_exact(tmp_path, delta):
    """两种编码都必须逐位读得回来。差分那条要是错了,读出来是一张看起来
    正常的差分图,不会报任何错 —— 所以这条断言是硬性的。"""
    HandDataWriter, read_tactile = _hdw()
    import h5py
    rng = np.random.default_rng(0)
    n = 70                          # 跨过 30 帧的关键帧边界
    base = rng.integers(0, 255, size=(5, 240, 320), dtype=np.uint8)
    frames = np.stack([np.clip(base.astype(int) + i, 0, 255).astype(np.uint8)
                       for i in range(n)])
    p = tmp_path / "t.h5"
    w = HandDataWriter(p, "right", [f"j{i}" for i in range(22)],
                       tactile_types=("raw",), tactile_delta=delta).start()
    for i in range(n):
        w.append_tactile({"t_host": i / 30, "dev_ts": np.zeros(5),
                          "fresh": np.ones(5, bool), "raw": frames[i]})
    w.stop()
    with h5py.File(p) as f:
        assert f.attrs["tactile_encoding"] == (
            "delta30+gzip1" if delta else "raw+lzf")
        back = read_tactile(f, "right_hand_tactile_raw")
        assert np.array_equal(back, frames)
        for k in (0, 1, 29, 30, 31, 69):        # 关键帧前后都要对
            one = read_tactile(f, "right_hand_tactile_raw", index=k)
            assert np.array_equal(one, frames[k]), f"第 {k} 帧随机读错了"


def test_delta_encoding_actually_changes_what_is_stored(tmp_path):
    """反证:开了差分之后,不解码直接读**必须**得到不同的数据。
    否则说明差分根本没生效,而上面那条往返测试照样会通过。"""
    HandDataWriter, _ = _hdw()
    import h5py
    rng = np.random.default_rng(1)
    frames = rng.integers(0, 255, size=(8, 5, 240, 320), dtype=np.uint8)
    p = tmp_path / "d.h5"
    w = HandDataWriter(p, "left", [f"j{i}" for i in range(22)],
                       tactile_types=("raw",), tactile_delta=True).start()
    for fr in frames:
        w.append_tactile({"t_host": 0.0, "dev_ts": np.zeros(5),
                          "fresh": np.ones(5, bool), "raw": fr})
    w.stop()
    with h5py.File(p) as f:
        naive = np.asarray(f["left_hand_tactile_raw"])
    assert not np.array_equal(naive, frames), "差分没生效"
    assert np.array_equal(naive[0], frames[0]), "第 0 帧应当是完整图(关键帧)"


# ------------------------------------------------ 环境障碍物(需 pinocchio)

def _guard(**kw):
    pytest.importorskip("pinocchio")
    from magicdexmate.joint_guard import JointGuard
    return JointGuard(control_hz=50, vel_scale=0.4, collision="hold", **kw)


def test_no_table_configured_leaves_the_guard_untouched(monkeypatch):
    monkeypatch.delenv("VEGA_TABLE_HEIGHT", raising=False)
    g = _guard()
    assert getattr(g, "env_names", []) == []


def test_table_is_added_and_home_pose_is_clear(monkeypatch):
    from magicdexmate.home_pose import ARM_HOME
    monkeypatch.setenv("VEGA_TABLE_HEIGHT", "0.76")
    g = _guard()
    assert g.env_names == ["table"]
    assert not g.in_collision(ARM_HOME), "正常桌高下默认姿势不该相撞"
    g.env_self_check(ARM_HOME)                     # 不该抛


def test_self_check_fails_when_the_table_is_inside_the_robot(monkeypatch):
    """尺寸填错最常见的表现是判据恒为真、手臂从第一帧就冻住而不说原因。
    这条锁住「它会当场报错」。"""
    from magicdexmate.home_pose import ARM_HOME
    monkeypatch.setenv("VEGA_TABLE_HEIGHT", "1.33")
    monkeypatch.setenv("VEGA_TABLE_X", "-0.449")   # 正好在默认姿势的手上
    g = _guard()
    assert g.in_collision(ARM_HOME)
    with pytest.raises(SystemExit):
        g.env_self_check(ARM_HOME)
