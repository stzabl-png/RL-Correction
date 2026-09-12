#!/usr/bin/env python
"""IK isolation diagnostic for the Vega right arm.

After physics warmup, commands the DifferentialIKController with the EXACT
current EE pose. Expectation: q_des == q_current. Prints:
  - jacobian tensor shape vs num_bodies (fixed-base row layout check)
  - q_des deltas for BOTH jacobian row indices (body_idx-1 and body_idx)
  - 200-tick hold test with the better index: ee error trace

Run: OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/debug_ik_vega.py --headless
"""

import sys

print("MARK1: script started", flush=True)
sys.stdout.flush()

import argparse  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("MARK2: kit launched", flush=True)

import os  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402

from vega_sharpa_scene import EE_BODY, R_ARM_JOINTS, VegaSharpaSceneCfg  # noqa: E402


def main():
    print("MARK3: main entered", flush=True)
    sim = SimulationContext(SimulationCfg(
        dt=1 / 240, render_interval=2, device=args_cli.device,
        physx=PhysxCfg(solver_type=1, max_position_iteration_count=8,
                       max_velocity_iteration_count=4),
    ))
    print("MARK4: sim context created", flush=True)
    scene = InteractiveScene(VegaSharpaSceneCfg(num_envs=1, env_spacing=4.0))
    print("MARK5: scene created", flush=True)
    sim.reset()
    print("MARK6: sim reset done", flush=True)
    robot: Articulation = scene["robot"]
    device = robot.device
    sim_dt = sim.get_physics_dt()
    print("MARK6a: robot handle ok", flush=True)

    arm_cfg = SceneEntityCfg("robot", joint_names=R_ARM_JOINTS, body_names=[EE_BODY["right"]])
    arm_cfg.resolve(scene)
    body_idx = arm_cfg.body_ids[0]
    jids = arm_cfg.joint_ids
    print(f"MARK6b: resolved body_idx={body_idx} jids={list(jids)}", flush=True)

    targets = robot.data.default_joint_pos.clone()
    print("MARK6c: targets cloned", flush=True)
    for i in range(20):
        robot.set_joint_position_target(targets)
        scene.write_data_to_sim(); sim.step(); scene.update(sim_dt)
        if i == 0:
            print("MARK6d: first sim.step ok", flush=True)
        if i == 10:
            j_inloop = robot.root_physx_view.get_jacobians()
            print(f"MARK6d2: in-loop get_jacobians ok shape={tuple(j_inloop.shape)}", flush=True)
    print("MARK6e: warmup done", flush=True)

    jac_full = robot.root_physx_view.get_jacobians()
    print("MARK6f: get_jacobians returned", flush=True)
    nb = robot.num_bodies
    print(f"MARK6g: num_bodies={nb}", flush=True)
    print(f"[diag] num_bodies={nb} jacobian shape={tuple(jac_full.shape)} "
          f"(rows==num_bodies-1 -> fixed base) | ee body '{EE_BODY['right']}' idx={body_idx}")
    print(f"[diag] arm joint ids={list(jids)} names={[robot.joint_names[j] for j in jids]}")

    def ee_pose_b():
        ee_w = robot.data.body_state_w[:, body_idx, 0:7]
        root_w = robot.data.root_state_w[:, 0:7]
        return subtract_frame_transforms(root_w[:, 0:3], root_w[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])

    pos0, quat0 = ee_pose_b()
    print(f"[diag] EE pose in base: pos={[round(v,3) for v in pos0[0].tolist()]} "
          f"quat={[round(v,3) for v in quat0[0].tolist()]}")

    ik = DifferentialIKController(
        DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
        num_envs=1, device=device)
    ik.set_command(torch.cat([pos0, quat0], dim=-1))
    q_cur = robot.data.joint_pos[:, jids]

    for tag, ridx in (("body_idx-1", body_idx - 1), ("body_idx", body_idx)):
        if ridx >= jac_full.shape[1]:
            print(f"[diag] {tag}: row {ridx} out of range"); continue
        jac = jac_full[:, ridx, :, jids]
        q_des = ik.compute(pos0, quat0, jac, q_cur)
        delta = (q_des - q_cur).abs().max().item()
        print(f"[diag] jacobi row {tag}: max|q_des - q_cur| = {delta:.6f} rad")

    # hold test with body_idx-1 (current implementation)
    print("[diag] 200-tick hold test (command = captured pose, row=body_idx-1):")
    for i in range(200):
        jac = robot.root_physx_view.get_jacobians()[:, body_idx - 1, :, jids]
        p, q = ee_pose_b()
        q_des = ik.compute(p, q, jac, robot.data.joint_pos[:, jids])
        targets[0, torch.as_tensor(jids, device=device)] = q_des[0]
        robot.set_joint_position_target(targets)
        scene.write_data_to_sim(); sim.step(); scene.update(sim_dt)
        if i % 40 == 0 or i == 199:
            p2, _ = ee_pose_b()
            print(f"[diag]   tick {i:3d}: ee err {(p2 - pos0).norm().item()*1000:7.1f} mm")

    import threading
    killer = threading.Timer(20.0, lambda: os._exit(0)); killer.daemon = True; killer.start()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:  # noqa: BLE001 - kit swallows excepthook output
        import traceback

        with open("/tmp/ik_exc.txt", "w") as f:
            traceback.print_exc(file=f)
        print(f"EXCEPTION captured to /tmp/ik_exc.txt: {e!r}", flush=True)
        raise
