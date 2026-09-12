"""Behavioral forensics for the wrist-joint sluggishness: every PhysX joint
parameter reads back correct (gains/limits/effort/velocity/armature/friction,
drive target), yet j6 responds ~100x slower than kp/kd predicts.

Two discriminating experiments on R_arm_j6:
  A. zero the drive gains and apply a direct +5 Nm effort: a free joint with
     I~0.01 kgm2 must spin up violently. Crawl here = hidden resistance in the
     joint/link itself (not the drive).
  B. rewrite gains at runtime (stiffness 2000, damping 10) and re-step: no
     behavior change = runtime gain writes never reach the solver.

Run: OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/dev_wrist_forensics.py --headless
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

sys.path.insert(0, "sim")
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402

from vega_scene import VegaSceneCfg  # noqa: E402

JOINT = "R_arm_j6"
PHYS_HZ = 240


def main():
    sim = SimulationContext(SimulationCfg(
        dt=1 / PHYS_HZ, device=args_cli.device,
        physx=PhysxCfg(solver_type=1, max_position_iteration_count=8,
                       max_velocity_iteration_count=4,
                       bounce_threshold_velocity=0.2)))
    scene = InteractiveScene(VegaSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    dt = sim.get_physics_dt()
    jid = robot.joint_names.index(JOINT)
    jid_t = torch.tensor([jid], device=robot.device)
    targets = robot.data.default_joint_pos.clone()
    efforts = torch.zeros_like(targets)

    def tick(n, tag, log_every=12):
        for i in range(n):
            robot.set_joint_position_target(targets)
            robot.set_joint_effort_target(efforts)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            if i % log_every == 0:
                q = robot.data.joint_pos[0, jid].item()
                qd = robot.data.joint_vel[0, jid].item()
                print(f"[{tag}] t={i * dt:5.3f} q={q:+.3f} qd={qd:+.3f}")

    tick(PHYS_HZ, "settle", log_every=80)

    print(f"\n=== A: {JOINT} gains->0, direct effort +5 Nm ===")
    zero = torch.zeros((1, 1), device=robot.device)
    robot.write_joint_stiffness_to_sim(zero, joint_ids=jid_t)
    robot.write_joint_damping_to_sim(zero, joint_ids=jid_t)
    efforts[0, jid] = 5.0
    tick(PHYS_HZ // 2, "A:+5Nm", log_every=6)
    efforts[0, jid] = 0.0

    print(f"\n=== B: gains stiffness=2000 damping=10, step +0.8 ===")
    robot.write_joint_stiffness_to_sim(
        torch.full((1, 1), 2000.0, device=robot.device), joint_ids=jid_t)
    robot.write_joint_damping_to_sim(
        torch.full((1, 1), 10.0, device=robot.device), joint_ids=jid_t)
    tick(PHYS_HZ, "B:settle", log_every=80)
    targets[0, jid] = targets[0, jid].item() + 0.8
    tick(PHYS_HZ, "B:step", log_every=12)

    simulation_app.close()


if __name__ == "__main__":
    main()
