"""Isaac smoke test for 1/16 environment Task-5 launches."""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

from tasks.pour.core import CurriculumStage, PourPhase


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", required=True)
parser.add_argument("--num_envs", type=int, choices=(1, 16), default=1)
parser.add_argument("--stage", type=int, choices=range(5), default=int(CurriculumStage.FULL))
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=160)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

slot = isaac_slot("pour-smoke")
app = AppLauncher(args).app

import torch  # noqa: E402

from tasks.pour.cfg import PourTaskCfg, configure_from_manifest  # noqa: E402
from tasks.pour.env import PourTaskEnv  # noqa: E402


cfg = PourTaskCfg()
configure_from_manifest(cfg, args.scene)
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.curriculum_stage = args.stage
env = PourTaskEnv(cfg)

torch.manual_seed(args.seed)
obs, _ = env.reset()
assert obs["policy"].shape == (args.num_envs, 280)
assert obs["priv_info"].shape == (args.num_envs, 23)
assert obs["proprio_hist"].shape == (args.num_envs, cfg.prop_hist_len, 116)

# A repeated seeded reset must select the same perturbation-pool entries.
torch.manual_seed(args.seed + 1)
env.reset()
left_first = env.robot.data.joint_pos[:, env.arm_ids["left"]].clone()
torch.manual_seed(args.seed + 1)
env.reset()
torch.testing.assert_close(
    env.robot.data.joint_pos[:, env.arm_ids["left"]], left_first, atol=1.0e-6, rtol=0.0
)

before = {side: env.arm_target[side].clone() for side in ("left", "right")}
env.task_phase[:] = int(PourPhase.APPROACH)
probe = torch.zeros(args.num_envs, 26, device=env.device)
probe[:, 0] = 0.25
probe[:, 13] = -0.25
env.step(probe)
for side in ("left", "right"):
    if not bool((env.arm_target[side] - before[side]).abs().sum(dim=1).gt(0).all()):
        raise AssertionError(f"{side} arm did not respond to its 13-D action slice")

generator = torch.Generator(device="cpu").manual_seed(args.seed)
reached = set()
for step in range(args.steps):
    action = (
        (torch.rand(args.num_envs, 26, generator=generator) * 2.0 - 1.0)
        .to(env.device)
        .mul_(0.1)
    )
    obs, reward, terminated, truncated, _ = env.step(action)
    reached.update(int(value) for value in env.task_phase.tolist())
    for name, value in (*obs.items(), ("reward", reward)):
        if not bool(torch.isfinite(value).all()):
            raise AssertionError(f"{name} contains NaN/Inf at step {step}")
    total = env.liquid.total()
    torch.testing.assert_close(
        total,
        torch.full_like(total, cfg.liquid.initial_mass),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    for side in ("left", "right"):
        if not bool(torch.isfinite(env._contact_magnitudes(side)).all()):
            raise AssertionError(f"{side} contact sensor contains NaN/Inf")
    if bool((env.cup.data.root_pos_w[:, 2] < cfg.table_top_z - 0.02).any()):
        raise AssertionError("cup penetrated below the table tolerance")
    if bool((env.bottle.data.root_pos_w[:, 2] < cfg.table_top_z - 0.02).any()):
        raise AssertionError("bottle penetrated below the table tolerance")

print(
    f"[pour-smoke] PASS envs={args.num_envs} steps={args.steps} "
    f"phases={sorted(reached)} action=26 obs=280 priv=23"
)
env.close()
app.close()
