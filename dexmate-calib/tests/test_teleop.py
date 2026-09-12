from __future__ import annotations

import numpy as np
import pytest

from dexmate_calib.robot.teleop import KeyboardJointTeleop, arm_component_for_link


class FakeComponent:
    def __init__(self):
        self.calls: list[tuple[np.ndarray, bool, float]] = []

    def move_joint_pos(self, pos, *, relative=False, velocity_scale=None):
        self.calls.append((np.asarray(pos, dtype=np.float64), relative, velocity_scale))


class FakeReader:
    def __init__(self):
        self.components = {"left_arm": FakeComponent(), "torso": FakeComponent()}

    def component(self, name):
        return self.components[name]


def test_component_from_link():
    assert arm_component_for_link("L_ee") == "left_arm"
    assert arm_component_for_link("L_arm_l7") == "left_arm"
    assert arm_component_for_link("R_ee") == "right_arm"
    assert arm_component_for_link("torso_l3") == "torso"
    with pytest.raises(ValueError):
        arm_component_for_link("base")


def test_keys_translate_to_relative_steps():
    reader = FakeReader()
    teleop = KeyboardJointTeleop(reader, ("left_arm", "torso"), step_deg=0.5, velocity_scale=0.3)
    assert teleop.active_component == "left_arm"
    assert teleop.joint_index == 6  # last arm joint by default
    assert teleop.seconds_since_command() == float("inf")

    assert teleop.handle_key(ord("3"))
    assert teleop.joint_index == 3
    assert teleop.handle_key(ord("w"))
    assert teleop.handle_key(ord("S"))
    arm_calls = reader.components["left_arm"].calls
    assert len(arm_calls) == 2
    pos, relative, scale = arm_calls[0]
    assert relative and scale == 0.3
    assert pos.shape == (7,)
    assert np.isclose(pos[3], np.deg2rad(0.5)) and np.count_nonzero(pos) == 1
    assert np.isclose(arm_calls[1][0][3], -np.deg2rad(2.0))
    assert teleop.seconds_since_command() < 1.0

    # Cycle to torso; joint index is clamped and steps go to the torso component.
    assert teleop.handle_key(ord("t"))
    assert teleop.active_component == "torso"
    assert teleop.joint_index == 2
    teleop.handle_key(ord("s"))
    assert len(reader.components["torso"].calls) == 1
    assert reader.components["torso"].calls[0][0].shape == (3,)

    # Step size adjustments are clamped to the component maximum.
    for _ in range(10):
        teleop.handle_key(ord("="))
    assert teleop.step_deg <= 1.5
    teleop.handle_key(ord("-"))
    assert teleop.step_deg == pytest.approx(0.75)

    # Unrelated keys are not consumed and do not move anything.
    total = len(arm_calls) + len(reader.components["torso"].calls)
    assert not teleop.handle_key(ord("x"))
    assert not teleop.handle_key(-1)
    assert len(arm_calls) + len(reader.components["torso"].calls) == total
