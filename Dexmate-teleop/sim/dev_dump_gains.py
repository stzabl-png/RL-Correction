"""Dump the LIVE PhysX drive parameters (stiffness/damping/max force) for the
arm joints after Isaac Lab applies the ArticulationCfg actuators - settles
whether _ARM_KP/_ARM_KD/effort_limit_sim actually reached the sim (2026-07-28:
step-response shows j6/j7 crawling as if their gains never landed).

Run: OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/dev_dump_gains.py --headless
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, "sim")
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext  # noqa: E402

from vega_scene import VegaSceneCfg  # noqa: E402


def main():
    sim = SimulationContext(SimulationCfg(
        dt=1 / 240, device=args_cli.device,
        physx=PhysxCfg(solver_type=1, max_position_iteration_count=8,
                       max_velocity_iteration_count=4,
                       bounce_threshold_velocity=0.2)))
    scene = InteractiveScene(VegaSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    view = robot.root_physx_view
    stiff = view.get_dof_stiffnesses()[0]
    damp = view.get_dof_dampings()[0]
    maxf = view.get_dof_max_forces()[0]
    armat = view.get_dof_armatures()[0]
    fric = view.get_dof_friction_coefficients()[0]
    maxv = view.get_dof_max_velocities()[0]
    lims = view.get_dof_limits()[0]
    print("\n[gains] joint  stiffness  damping  maxForce  armature  friction  maxVel  limits")
    for i, n in enumerate(robot.joint_names):
        if "arm_j" in n or "torso" in n:
            print(f"[gains] {n:>10s}  {stiff[i].item():9.1f}  {damp[i].item():7.1f}"
                  f"  {maxf[i].item():8.1f}  {armat[i].item():8.4f}  {fric[i].item():8.4f}"
                  f"  {maxv[i].item():8.3f}"
                  f"  [{lims[i, 0].item():+.3f}, {lims[i, 1].item():+.3f}]")
    simulation_app.close()


if __name__ == "__main__":
    main()
