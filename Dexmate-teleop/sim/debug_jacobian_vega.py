#!/usr/bin/env python
"""Probe: is the vega_1p articulation fixed-base? Jacobian shape vs bodies?

Prints everything needed to pick the DifferentialIK jacobian row/column
convention for sim/teleop_vega_pico.py (method of sim/debug_ik_vega.py).

Run: OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/debug_jacobian_vega.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from vega_scene import ARM_JOINTS, EE_BODY, VegaSceneCfg  # noqa: E402


def main():
    sim = SimulationContext(SimulationCfg(dt=1 / 240, device=args_cli.device))
    scene = InteractiveScene(VegaSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot: Articulation = scene["robot"]

    for _ in range(10):
        robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    jac = robot.root_physx_view.get_jacobians()
    n_rows = jac.shape[1]
    print(f"[probe] is_fixed_base      = {robot.is_fixed_base}")
    print(f"[probe] num_bodies         = {robot.num_bodies}")
    print(f"[probe] num_joints (dofs)  = {robot.num_joints}")
    print(f"[probe] jacobian shape     = {tuple(jac.shape)}")

    import torch  # noqa: E402

    from isaaclab.controllers import (  # noqa: E402
        DifferentialIKController,
        DifferentialIKControllerCfg,
    )
    from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

    sim_dt = sim.get_physics_dt()

    def reset_and_settle(targets):
        robot.write_joint_state_to_sim(robot.data.default_joint_pos,
                                       robot.data.default_joint_vel)
        for _ in range(120):
            robot.set_joint_position_target(targets)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

    def ee_pose_in_base(body_idx):
        ee = robot.data.body_state_w[:, body_idx, 0:7]
        root = robot.data.root_state_w[:, 0:7]
        return subtract_frame_transforms(root[:, 0:3], root[:, 3:7], ee[:, 0:3], ee[:, 3:7])

    # Two discriminating tests per candidate row (a zero-error hold cannot
    # discriminate: with gravity off the loop is never excited, dq stays 0
    # for ANY jacobian - measured):
    #   step:  command settled pose + (2cm,0,-2cm); correct row converges to
    #          ~mm of the target, a wrong row diverges/oscillates.
    #   long:  command the settled pose for 1440 ticks (6 sim-s, same length
    #          as the main-script hold run that drifted 60-77mm) - reproduces
    #          slow-excitation effects like body_state_w quaternion sign flips.
    def run_case(body_idx, joint_ids, joint_ids_t, row, offset, n_steps):
        targets = robot.data.default_joint_pos.clone()
        reset_and_settle(targets)
        p0, q0 = ee_pose_in_base(body_idx)
        cmd_p = p0.clone() + torch.tensor([offset], device=robot.device)
        cmd_q = q0.clone()
        ik = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                        ik_method="dls"),
            num_envs=1, device=robot.device,
        )
        ik.set_command(torch.cat([cmd_p, cmd_q], dim=-1))
        max_err = 0.0
        for i in range(n_steps):
            if i % 4 == 0:
                J = robot.root_physx_view.get_jacobians()[:, row, :, joint_ids]
                pb, qb = ee_pose_in_base(body_idx)
                targets[0, joint_ids_t] = ik.compute(
                    pb, qb, J, robot.data.joint_pos[:, joint_ids])[0]
                max_err = max(max_err, (pb - cmd_p).norm().item())
            robot.set_joint_position_target(targets)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
        pe, _ = ee_pose_in_base(body_idx)
        dq = (robot.data.joint_pos[0, joint_ids_t]
              - robot.data.default_joint_pos[0, joint_ids_t]).abs().max().item()
        return (pe - cmd_p).norm().item() * 1000, max_err * 1000, dq

    for hand in ("right", "left"):
        cfg = SceneEntityCfg("robot", joint_names=ARM_JOINTS[hand], body_names=[EE_BODY[hand]])
        cfg.resolve(scene)
        body_idx = cfg.body_ids[0]
        joint_ids = cfg.joint_ids
        joint_ids_t = torch.as_tensor(joint_ids, device=robot.device)
        print(f"[probe] {hand}: ee_body_idx={body_idx} joint_ids={joint_ids}")
        for row in (body_idx - 1, body_idx - 2):
            if not (0 <= row < n_rows):
                print(f"[probe]   row {row}: OUT OF RANGE, skipped")
                continue
            e_end, e_max, dq = run_case(body_idx, joint_ids, joint_ids_t, row,
                                        (0.02, 0.0, -0.02), 600)
            print(f"[probe]   row {row} step-test: end_err {e_end:7.1f} mm "
                  f"(max {e_max:6.1f}) max|dq| {dq:.3f} rad")
            e_end, e_max, dq = run_case(body_idx, joint_ids, joint_ids_t, row,
                                        (0.0, 0.0, 0.0), 1440)
            print(f"[probe]   row {row} long-hold: end_err {e_end:7.1f} mm "
                  f"(max {e_max:6.1f}) max|dq| {dq:.3f} rad")

    simulation_app.close()


if __name__ == "__main__":
    main()
