"""Why does torso_j2 stop at 106.7 deg when it is commanded to 120.9?

2026-08-01. The operator authored the upright chest pose as
(torso_j1, j2, j3) = (66.7, 120.9, 43.8) deg. Isaac reports the joint limits as
[0, 180] deg, yet the joint settles at 106.7 - a 14.2 deg shortfall that puts
arm_center 20 cm behind where the mapping thinks it is and leaves the chest
leaning 3.8 deg BACKWARD instead of 10.4 deg forward. The wrist tracking error
went from 8 mm to 195 mm.

A steady-state PD error cannot explain it: stiffness is 8e5, so 0.248 rad of
error implies ~198 kNm of opposing torque. So this sweeps the target, holds,
and reports what is actually achievable, plus the physics flags that would
explain a hard stop.

    OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/dev_torso_range.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--hold-steps", type=int, default=300)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

import sys  # noqa: E402
import os  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vega_scene import VegaSceneCfg  # noqa: E402


def main():
    sim = SimulationContext(SimulationCfg(dt=1.0 / 200.0, device="cuda:0"))
    scene = InteractiveScene(VegaSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    names = robot.data.joint_names
    dt = sim.get_physics_dt()

    print(f"[cfg] sim gravity = {sim.cfg.gravity}")
    print(f"[cfg] joints = {len(names)}")
    j2 = names.index("torso_j2")
    for n in ("torso_j1", "torso_j2", "torso_j3"):
        i = names.index(n)
        lo = np.degrees(robot.data.joint_pos_limits[0, i, 0].item())
        hi = np.degrees(robot.data.joint_pos_limits[0, i, 1].item())
        st = robot.data.joint_stiffness[0, i].item()
        da = robot.data.joint_damping[0, i].item()
        ef = robot.data.joint_effort_limits[0, i].item()
        vl = robot.data.joint_velocity_limits[0, i].item()
        print(f"[drive] {n}: limits [{lo:+.1f}, {hi:+.1f}]deg  kp={st:.3g} "
              f"kd={da:.3g}  effort_lim={ef:.3g}  vel_lim={vl:.3g}")

    targets = robot.data.default_joint_pos.clone()

    def hold(deg, steps):
        targets[0, j2] = float(np.radians(deg))
        for _ in range(steps):
            robot.set_joint_position_target(targets)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
        got = np.degrees(robot.data.joint_pos[0, j2].item())
        tau = robot.data.applied_torque[0, j2].item()
        return got, tau

    print("\n[sweep] commanded -> achieved (300 steps = 1.5 s of hold each)")
    for deg in [0, 30, 60, 80, 90, 95, 100, 105, 106, 107, 108, 110, 120.9, 150, 180]:
        got, tau = hold(deg, args_cli.hold_steps)
        flag = "" if abs(got - deg) < 1.0 else "   <-- BLOCKED"
        print(f"  cmd {deg:7.1f}deg -> got {got:7.1f}deg  "
              f"(applied torque {tau:+.4g}){flag}")

    # does it hold if we walk there slowly instead of jumping?
    print("\n[ramp] walk 106 -> 121 deg in 0.05 deg steps")
    hold(106.0, 300)
    stuck = None
    for deg in np.arange(106.0, 121.01, 0.05):
        targets[0, j2] = float(np.radians(deg))
        for _ in range(4):
            robot.set_joint_position_target(targets)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
        got = np.degrees(robot.data.joint_pos[0, j2].item())
        if deg - got > 1.0 and stuck is None:
            stuck = (deg, got)
    got = np.degrees(robot.data.joint_pos[0, j2].item())
    print(f"  end: cmd 121.0 -> got {got:.2f}deg"
          + (f";  first fell behind at cmd {stuck[0]:.2f} (got {stuck[1]:.2f})"
             if stuck else ";  tracked the whole way"))

    # contact / collision evidence: is anything touching?
    print(f"\n[state] torso_j2 vel {robot.data.joint_vel[0, j2].item():+.4g} rad/s")
    simulation_app.close()


if __name__ == "__main__":
    main()
