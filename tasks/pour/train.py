"""Train one Task-5 curriculum stage with the shared 26-D policy.

Stages 1-4 require the preceding checkpoint.  Stages 0-2 stop automatically
after the configured episode count and success gate; this produces a
``curriculum_pass.pth`` checkpoint for the next invocation.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

from tasks.pour.core import AblationMode, CurriculumStage


STAGE_NAMES = {stage.name.lower(): stage for stage in CurriculumStage}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", required=True, help="approved pour11 scene.json")
parser.add_argument("--stage", choices=tuple(STAGE_NAMES), required=True)
parser.add_argument("--ablation", choices=tuple(mode.value for mode in AblationMode), default="full")
parser.add_argument("--name", default="PourBimanual")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_agent_steps", type=int, default=100000000)
parser.add_argument("--load_path", default=None)
parser.add_argument("--gate_min_episodes", type=int, default=1000)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

stage = STAGE_NAMES[args.stage]
if stage > CurriculumStage.SINGLE_GRASP and not args.load_path:
    raise SystemExit("stages 1-4 require --load_path from the preceding stage")
maximum_envs = int(os.environ.get("RL_MAX_ENVS", "1024"))
if args.num_envs > maximum_envs:
    print(f"[pour-train] num_envs {args.num_envs} capped to RL_MAX_ENVS={maximum_envs}")
    args.num_envs = maximum_envs

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

slot = isaac_slot("pour-train")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.pour.cfg import PourTaskCfg, configure_from_manifest  # noqa: E402
from tasks.pour.env import PourTaskEnv  # noqa: E402


class CurriculumPPO(PPO):
    """PPO with an evidence-counted automatic stage gate."""

    def __init__(
        self,
        *positional,
        stage: CurriculumStage,
        min_episodes: int,
        seed: int,
        ablation: str,
        **keyword,
    ):
        super().__init__(*positional, **keyword)
        self.curriculum_stage = stage
        self.gate_min_episodes = int(min_episodes)
        self.run_seed = int(seed)
        self.run_ablation = str(ablation)
        self.gate_passed = False
        self.milestone_60_written = False

    @staticmethod
    def _number(value) -> float:
        return float(value.detach().item()) if torch.is_tensor(value) else float(value)

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        super().write_stats(a_losses, c_losses, b_losses, entropies, kls)
        episodes = self._number(self.extra_info.get("episodes", 0.0))
        success_rate = self._number(self.extra_info.get("success_rate", 0.0))
        if (
            not self.milestone_60_written
            and episodes >= self.gate_min_episodes
            and success_rate >= 0.60
        ):
            milestone = {
                "stage": self.curriculum_stage.name.lower(),
                "seed": self.run_seed,
                "ablation": self.run_ablation,
                "success_rate": success_rate,
                "agent_steps": int(self.agent_steps),
                "episodes": int(episodes),
            }
            Path(self.output_dir, "milestone_60.json").write_text(
                json.dumps(milestone, indent=2) + "\n", encoding="utf-8"
            )
            self.milestone_60_written = True
        if self.curriculum_stage > CurriculumStage.APPROACH_GRASP:
            return
        if episodes < self.gate_min_episodes:
            return
        if self.curriculum_stage == CurriculumStage.SINGLE_GRASP:
            left = self._number(self.extra_info.get("left_grasp_rate", 0.0))
            right = self._number(self.extra_info.get("right_grasp_rate", 0.0))
            passed = min(left, right) >= 0.90
            rates = {"left_grasp_rate": left, "right_grasp_rate": right}
        else:
            success = success_rate
            threshold = 0.85 if self.curriculum_stage == CurriculumStage.DUAL_GRASP else 0.80
            passed = success >= threshold
            rates = {"success_rate": success, "threshold": threshold}
        if passed and not self.gate_passed:
            self.gate_passed = True
            self.max_agent_steps = self.agent_steps + 1
            record = {
                "stage": self.curriculum_stage.name.lower(),
                "agent_steps": int(self.agent_steps),
                "episodes": int(episodes),
                **rates,
            }
            Path(self.output_dir, "curriculum_gate.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[pour-train] curriculum gate passed: {record}")


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
here = Path(__file__).resolve().parent
agent_cfg = yaml.safe_load((here / "ppo.yaml").read_text(encoding="utf-8"))

env_cfg = PourTaskCfg()
manifest = configure_from_manifest(env_cfg, args.scene)
if manifest.demo_id != "11":
    raise SystemExit("training is restricted to canonical demo 11; demo 9 is hold-out only")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
env_cfg.curriculum_stage = int(stage)
env_cfg.ablation = args.ablation

agent_cfg["seed"] = args.seed
agent_cfg["load_path"] = args.load_path
agent_cfg["algorithm"]["experiment_name"] = args.name
agent_cfg["algorithm"]["num_actors"] = args.num_envs
agent_cfg["algorithm"]["minibatch_size"] = min(args.num_envs * 8, 32768)
agent_cfg["algorithm"]["max_agent_steps"] = args.max_agent_steps

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = Path("logs") / args.name / args.ablation / args.stage / f"seed{args.seed}_{timestamp}"
print(
    f"[pour-train] stage={args.stage} ablation={args.ablation} seed={args.seed} "
    f"envs={args.num_envs} log={log_dir}"
)
raw = PourTaskEnv(env_cfg)
env = GymStyleEnvWrapper(raw, clip_actions=env_cfg.clip_actions)
agent = CurriculumPPO(
    env,
    output_dir=str(log_dir),
    full_config=ConfigWrapper(agent_cfg, env_cfg),
    stage=stage,
    min_episodes=args.gate_min_episodes,
    seed=args.seed,
    ablation=args.ablation,
)
agent.epoch_hook = slot.yield_if_paused
if args.load_path:
    agent.restore_train(args.load_path)
agent.train()
if agent.gate_passed:
    agent.save(str(log_dir / "stage1_nn" / "curriculum_pass"))
env.close()
app.close()
