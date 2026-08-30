"""扭开瓶盖任务的 PPO 训练入口.

  SHARPA_WANDB=0 PYTHONPATH=. $PY -u -m tasks.recon_kailang.bottle_reconstruction.unscrew_train \
      --headless --clip screw_unscrew_cap1_task --name Unscrew0 --num_envs 1024

判读: TensorBoard 的 diag/* (窗口=最近 256 个完整回合):
  diag/release           成功率 (拧满 2 圈释放的回合占比)
  diag/screw_deg         回合内拧动角峰值 (度; 720 = 满)
  diag/cap_contact2_frac ≥2 指尖接触帽的时间占比 (先看它, 见台账 U2)
  diag/rew_*             逐项奖励账本 (U4: 引导不许压过任务核)
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", default="screw_unscrew_cap1_task")
parser.add_argument("--name", default="Unscrew_0")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_agent_steps", type=int, default=None)
parser.add_argument("--kl_threshold", type=float, default=None)
parser.add_argument("--minibatch", type=int, default=None)
parser.add_argument("--load_path", type=str, default=None)
parser.add_argument("--output_root", default="logs/unscrew")
parser.add_argument("--ref", action="store_true",
                    help="v2 轨迹跟随任务 (UnscrewRefTaskEnv)")
parser.add_argument("--dyn", action="store_true", help="Stage B: 动态瓶")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("unscrew_train")
app = AppLauncher(args).app

from datetime import datetime  # noqa: E402

import yaml  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.recon_kailang.bottle_reconstruction.unscrew_env import (  # noqa: E402
    UnscrewTaskCfg,
    UnscrewTaskEnv,
)
from tasks.recon_kailang.bottle_reconstruction.unscrew_ref_env import (  # noqa: E402
    UnscrewDynTaskCfg,
    UnscrewRefTaskCfg,
    UnscrewRefTaskEnv,
)


def main() -> int:
    if os.environ.get("RL_ACC_FINGER", "1") != "1":
        raise ValueError("扭盖任务的 observation_space 假设 RL_ACC_FINGER=1")
    env_cfg = (UnscrewDynTaskCfg() if args.dyn else
               UnscrewRefTaskCfg() if args.ref else UnscrewTaskCfg())
    clips.configure_cfg(env_cfg, args.clip)
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "unscrew_ppo.yaml"), encoding="utf-8") as handle:
        agent_cfg = yaml.safe_load(handle)
    algorithm = agent_cfg["algorithm"]
    agent_cfg["seed"] = args.seed
    agent_cfg["device"] = args.device
    algorithm["experiment_name"] = args.name
    algorithm["num_actors"] = args.num_envs
    batch = args.num_envs * int(algorithm["horizon_length"])
    algorithm["minibatch_size"] = (args.minibatch if args.minibatch
                                   else min(args.num_envs * 8, batch))
    if args.max_agent_steps is not None:
        algorithm["max_agent_steps"] = args.max_agent_steps
    if args.kl_threshold is not None:
        algorithm["kl_threshold"] = args.kl_threshold
    if args.load_path:
        agent_cfg["load_path"] = args.load_path

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.abspath(os.path.join(args.output_root, args.name, stamp))
    print(f"[unscrew-train] clip={args.clip} envs={args.num_envs} "
          f"minibatch={algorithm['minibatch_size']} "
          f"kl={algorithm['kl_threshold']} max_steps={algorithm['max_agent_steps']} "
          f"log_dir={log_dir}", flush=True)

    env_raw = (UnscrewRefTaskEnv if (args.ref or args.dyn)
               else UnscrewTaskEnv)(env_cfg)
    env = GymStyleEnvWrapper(env_raw, clip_actions=env_cfg.clip_actions)
    agent = PPO(env, output_dir=log_dir,
                full_config=ConfigWrapper(agent_cfg, env_cfg))
    agent.epoch_hook = _slot.yield_if_paused
    try:
        agent.train()
        writer = getattr(agent, "writer", None)
        if writer is not None:
            writer.flush()
            writer.close()
        print(f"[unscrew-train] completed: {log_dir}", flush=True)
    except Exception as error:
        print(f"[unscrew-train] FAILED: {error}", file=sys.stderr, flush=True)
        env.close()
        os._exit(1)
    env.close()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
