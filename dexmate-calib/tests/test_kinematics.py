from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pinocchio")
pytest.importorskip("dexmate_urdf")

from dexmate_calib.geometry.transforms import is_rigid, pose_error
from dexmate_calib.robot.kinematics import FrameKinematics, resolve_urdf_path


@pytest.fixture(scope="module")
def fk() -> FrameKinematics:
    return FrameKinematics.from_model("vega_1p")


def test_urdf_resolves_for_supported_models():
    for model in ("vega_1", "vega_1p"):
        assert resolve_urdf_path(model).exists()
    with pytest.raises(ValueError):
        resolve_urdf_path("not_a_robot")


def test_left_ee_chain_and_rigid_pose(fk: FrameKinematics):
    chain = fk.required_joints_for("L_ee")
    assert chain == [
        "torso_j1",
        "torso_j2",
        "torso_j3",
        "L_arm_j1",
        "L_arm_j2",
        "L_arm_j3",
        "L_arm_j4",
        "L_arm_j5",
        "L_arm_j6",
        "L_arm_j7",
    ]
    T = fk.frame_pose({}, "L_ee")
    assert is_rigid(T)
    assert np.allclose(fk.frame_pose({}, "base"), np.eye(4))


def test_unrelated_joints_do_not_move_left_ee(fk: FrameKinematics):
    T0 = fk.frame_pose({}, "L_ee")
    T1 = fk.frame_pose({"R_arm_j1": 0.7, "head_j1": 0.3}, "L_ee")
    assert np.allclose(T0, T1)
    T2 = fk.frame_pose({"L_arm_j1": 0.7}, "L_ee")
    rot_deg, _ = pose_error(T0, T2)
    assert rot_deg > 1.0
    with pytest.raises(KeyError):
        fk.frame_pose({"nonexistent_joint": 0.1}, "L_ee")
    # Non-strict ignores joints outside the URDF (e.g. hand joints from dexcontrol).
    assert np.allclose(fk.frame_pose({"nonexistent_joint": 0.1}, "L_ee", strict=False), T0)


def test_unknown_frame_raises(fk: FrameKinematics):
    with pytest.raises(ValueError):
        fk.frame_pose({}, "no_such_frame")
