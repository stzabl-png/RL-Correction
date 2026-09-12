"""Joint-space step-response probe: bypass Pink/teleop entirely and measure
what the Isaac plant does with a raw position-target step on single arm
joints.  2026-07-27: demo probes show wrist joints crawling at 0.03-0.12
rad/s toward static targets (PD tau should be ~60ms) with no gravity, no
contacts, no self-collision, generous effort/velocity limits - this isolates
whether the sluggishness is in the actuator/solver layer or the task layer.

Run from MagicDexMate root: OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python
sim/dev_step_response.py --headless
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

STEPS = [  # (joint, delta rad) - all end poses well inside limits
    ("R_arm_j7", +0.8),
    ("R_arm_j6", +0.8),
    ("R_arm_j5", +0.8),
    ("R_arm_j1", +0.8),
]
PHYS_HZ = 240
HOLD_S = 2.0
SETTLE_S = 1.5


def main():
    sim_cfg = SimulationCfg(
        dt=1 / PHYS_HZ, render_interval=4, device=args_cli.device,
        physx=PhysxCfg(solver_type=1, max_position_iteration_count=8,
                       max_velocity_iteration_count=4,
                       bounce_threshold_velocity=0.2),
    )
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(VegaSceneCfg(num_envs=1, env_spacing=4.0))
    sim.reset()
    robot: Articulation = scene["robot"]
    dt = sim.get_physics_dt()

    targets = robot.data.default_joint_pos.clone()
    # pin the unstable swerve wheels every step, same as teleop_vega_pico -
    # without this the wheels numerically explode and contaminate the probe
    wheel_ids = [i for i, n in enumerate(robot.joint_names) if "_wheel_j" in n]
    wheel_ids_t = torch.tensor(wheel_ids, device=robot.device)
    wheel_straight = robot.data.default_joint_pos[:, wheel_ids].clone()
    wheel_zero = torch.zeros_like(wheel_straight)

    view = robot.root_physx_view

    def physx_target(jid):
        """Read back what PhysX ACTUALLY holds as the drive target - the last
        unexcluded suspect for the wrist joints parking off-target."""
        for name in ("get_dof_position_targets", "get_dof_targets"):
            fn = getattr(view, name, None)
            if fn is not None:
                try:
                    return fn()[0, jid].item()
                except Exception:
                    continue
        return float("nan")

    def run(n_steps, log_joint=None, tag=""):
        jid = robot.joint_names.index(log_joint) if log_joint else None
        for i in range(n_steps):
            robot.set_joint_position_target(targets)
            if wheel_ids:
                robot.write_joint_state_to_sim(wheel_straight, wheel_zero,
                                               joint_ids=wheel_ids_t)
            scene.write_data_to_sim()
            sim.step()
            scene.update(dt)
            if jid is not None and i % 12 == 0:  # every 50ms
                q = robot.data.joint_pos[0, jid].item()
                qd = robot.data.joint_vel[0, jid].item()
                tg = targets[0, jid].item()
                print(f"[step] {tag} t={i * dt:5.3f} tgt={tg:+.3f} "
                      f"physx_tgt={physx_target(jid):+.3f} "
                      f"q={q:+.3f} err={tg - q:+.3f} qd={qd:+.2f}")

    run(int(2.0 * PHYS_HZ))  # initial settle
    for joint, _ in STEPS:  # verify quiet start before stepping anything
        jid = robot.joint_names.index(joint)
        print(f"[settled] {joint} q={robot.data.joint_pos[0, jid].item():+.3f} "
              f"tgt={targets[0, jid].item():+.3f} "
              f"qd={robot.data.joint_vel[0, jid].item():+.3f}")
    for joint, delta in STEPS:
        jid = robot.joint_names.index(joint)
        base = targets[0, jid].item()
        targets[0, jid] = base + delta
        print(f"\n=== {joint} step {base:+.3f} -> {base + delta:+.3f} ===")
        run(int(HOLD_S * PHYS_HZ), log_joint=joint, tag=joint)
        targets[0, jid] = base
        run(int(SETTLE_S * PHYS_HZ))
    simulation_app.close()


if __name__ == "__main__":
    main()
