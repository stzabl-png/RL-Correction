"""Record deterministic success/failure episodes for qualitative review."""
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

from tasks.pour.core import AblationMode


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scene", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output_dir", required=True)
parser.add_argument("--ablation", choices=tuple(mode.value for mode in AblationMode), default="full")
parser.add_argument("--episodes", type=int, default=12)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--eye", default="0.75,-1.05,1.55")
parser.add_argument("--lookat", default="0.0,0.0,1.0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

slot = isaac_slot("pour-record")
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from isaaclab.envs import ViewerCfg  # noqa: E402
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.pour.cfg import PourTaskCfg, configure_from_manifest  # noqa: E402
from tasks.pour.core import FAILURE_NAMES, FailureCode  # noqa: E402
from tasks.pour.env import PourTaskEnv  # noqa: E402


output = Path(args.output_dir).resolve()
output.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
agent_cfg = yaml.safe_load((here / "ppo.yaml").read_text(encoding="utf-8"))
agent_cfg["algorithm"]["num_actors"] = 1

cfg = PourTaskCfg()
configure_from_manifest(cfg, args.scene)
cfg.scene.num_envs = 1
cfg.seed = args.seed
cfg.curriculum_stage = 4
cfg.ablation = args.ablation
cfg.viewer = ViewerCfg(
    eye=tuple(float(value) for value in args.eye.split(",")),
    lookat=tuple(float(value) for value in args.lookat.split(",")),
    origin_type="env",
    env_index=0,
    resolution=(960, 720),
)

raw = PourTaskEnv(cfg, render_mode="rgb_array")
raw.metadata["render_fps"] = args.fps
prefix = f"pour_seed{args.seed}"
recording = gym.wrappers.RecordVideo(
    raw,
    video_folder=str(output),
    name_prefix=prefix,
    episode_trigger=lambda episode: episode < args.episodes,
    disable_logger=True,
)
env = GymStyleEnvWrapper(recording, clip_actions=cfg.clip_actions)
agent = PPO(
    env,
    output_dir=str(output / ".record_tmp"),
    full_config=ConfigWrapper(agent_cfg, cfg, test=True),
    create_output_dir=False,
)
agent.restore_test(args.checkpoint)
agent.set_eval()

outcomes = []
obs = env.reset()
with torch.no_grad():
    while len(outcomes) < args.episodes:
        model_input = {
            "obs": agent.running_mean_std(obs["obs"]),
            "priv_info": obs["priv_info"],
        }
        action = agent.model.act_inference(model_input).clamp(-1.0, 1.0)
        obs, _, done, _ = env.step(action)
        if not bool(done[0]):
            continue
        signals = raw._terminal_signals
        if bool(signals["terminal_success"][0]):
            outcomes.append("success")
        else:
            code = FailureCode(int(signals["terminal_failure"][0]))
            if code == FailureCode.NONE:
                code = FailureCode.TIMEOUT
            outcomes.append(f"failure_{FAILURE_NAMES[code]}")
        print(f"[pour-record] episode {len(outcomes)}/{args.episodes}: {outcomes[-1]}")

env.close()
videos = sorted(output.glob(f"{prefix}-episode-*.mp4"))
if len(videos) != len(outcomes):
    print(
        f"[pour-record] warning: finalized {len(videos)} videos for "
        f"{len(outcomes)} outcomes"
    )
for index, (path, outcome) in enumerate(zip(videos, outcomes, strict=False)):
    destination = output / f"seed{args.seed}_{index:03d}_{outcome}.mp4"
    path.replace(destination)
print(f"[pour-record] videos={len(videos)} output={output}")
app.close()
