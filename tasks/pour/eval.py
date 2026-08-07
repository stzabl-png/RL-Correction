"""Deterministic Task-5 evaluation with episode-level Success Tracker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

from tasks.pour.core import AblationMode


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--run_name", required=True)
parser.add_argument("--ablation", choices=tuple(mode.value for mode in AblationMode), default="full")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--episodes", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

slot = isaac_slot("pour-eval")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.pour.cfg import PourTaskCfg, configure_from_manifest  # noqa: E402
from tasks.pour.core import FailureCode, PourPhase  # noqa: E402
from tasks.pour.env import PourTaskEnv  # noqa: E402
from tasks.pour.tracker import EpisodeRecord, SuccessTracker  # noqa: E402


output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
tracker = SuccessTracker(output / "episodes.jsonl")

here = Path(__file__).resolve().parent
agent_cfg = yaml.safe_load((here / "ppo.yaml").read_text(encoding="utf-8"))
agent_cfg["algorithm"]["num_actors"] = args.num_envs

cfg = PourTaskCfg()
manifest = configure_from_manifest(cfg, args.scene)
cfg.scene.num_envs = args.num_envs
cfg.seed = args.seed
cfg.curriculum_stage = 4
cfg.ablation = args.ablation
raw = PourTaskEnv(cfg)
env = GymStyleEnvWrapper(raw, clip_actions=cfg.clip_actions)
agent = PPO(
    env,
    output_dir=str(output / ".eval_tmp"),
    full_config=ConfigWrapper(agent_cfg, cfg, test=True),
    create_output_dir=False,
)
agent.restore_test(args.checkpoint)
agent.set_eval()

obs = env.reset()
trajectory_return = torch.zeros(args.num_envs, device=raw.device)
contact_return = torch.zeros(args.num_envs, device=raw.device)
episode_count = 0
condition_names = (
    "verify_phase",
    "liquid",
    "spill",
    "left_grasp",
    "right_grasp",
    "cup_upright",
    "bottle_returned",
)

with torch.no_grad():
    while episode_count < args.episodes:
        model_input = {
            "obs": agent.running_mean_std(obs["obs"]),
            "priv_info": obs["priv_info"],
        }
        action = agent.model.act_inference(model_input).clamp(-1.0, 1.0)
        obs, _, done, _ = env.step(action)
        trajectory_return += raw._reward_terms.get(
            "trajectory", torch.zeros_like(trajectory_return)
        )
        contact_return += raw._reward_terms.get(
            "contact", torch.zeros_like(contact_return)
        )
        indices = torch.nonzero(done.bool(), as_tuple=False).squeeze(-1)
        if not len(indices):
            continue
        remaining = args.episodes - episode_count
        indices = indices[:remaining]
        signals = raw._terminal_signals
        records = []
        for index_tensor in indices:
            index = int(index_tensor)
            success = bool(signals["terminal_success"][index])
            failure = int(signals["terminal_failure"][index])
            if not success and failure == int(FailureCode.NONE):
                failure = int(FailureCode.TIMEOUT)
            initial_mass = float(cfg.liquid.initial_mass)
            records.append(
                EpisodeRecord(
                    run_name=args.run_name,
                    demo_id=manifest.demo_id,
                    seed=args.seed,
                    episode=episode_count + len(records),
                    success=success,
                    terminal_phase=PourPhase(
                        int(signals["terminal_phase"][index])
                    ).name,
                    failure_code=failure,
                    steps=int(signals["terminal_steps"][index]),
                    left_grasp=bool(signals["left_grasp"][index]),
                    right_grasp=bool(signals["right_grasp"][index]),
                    cup_fraction=float(signals["terminal_cup_mass"][index]) / initial_mass,
                    spill_fraction=float(signals["terminal_spill_mass"][index]) / initial_mass,
                    cup_tilt_deg=float(torch.rad2deg(signals["cup_tilt"][index])),
                    bottle_tilt_deg=float(torch.rad2deg(signals["bottle_tilt"][index])),
                    trajectory_return=float(trajectory_return[index]),
                    contact_return=float(contact_return[index]),
                    bottle_fraction=float(signals["terminal_bottle_mass"][index]) / initial_mass,
                    cup_pose_wxyz=signals["terminal_cup_pose"][index].cpu().tolist(),
                    bottle_pose_wxyz=signals["terminal_bottle_pose"][index].cpu().tolist(),
                    success_subconditions={
                        name: bool(signals[f"success_condition/{name}"][index])
                        for name in condition_names
                    },
                )
            )
        tracker.append(records)
        trajectory_return[indices] = 0.0
        contact_return[indices] = 0.0
        episode_count += len(records)

summary = tracker.summary()
summary.update(
    {
        "run_name": args.run_name,
        "demo_id": manifest.demo_id,
        "seed": args.seed,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "ablation": args.ablation,
        "deterministic": True,
    }
)
(output / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(
    f"[pour-eval] demo={manifest.demo_id} seed={args.seed} "
    f"success={summary['success_rate'] * 100:.2f}% "
    f"episodes={summary['episodes']} output={output}"
)
env.close()
app.close()
