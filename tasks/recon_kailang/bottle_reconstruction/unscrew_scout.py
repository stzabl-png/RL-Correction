"""两遍法第一遍: 用同一种子跑 rollout, 找出**完整成功**(拧开+放到桌上)的 env.

  $PY -m tasks...unscrew_scout --dyn --checkpoint <ckpt> --num_envs 64 --seed 7

输出各 env 的 released/placed/place_dist/screw 与推荐 env_index;
第二遍用 unscrew_record.py 加 **同样的 --num_envs 与 --seed** + --env_index
即可把该回合录成视频 (rollout 可复现的前提下).
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--clip", default="screw_unscrew_cap1_task")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--ref", action="store_true")
parser.add_argument("--dyn", action="store_true")
parser.add_argument("--episodes", type=int, default=1, help="跟踪前 N 个回合")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("unscrew_scout")
app = AppLauncher(args).app

import torch  # noqa: E402
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

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "unscrew_ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs

if args.dyn:
    args.ref = True
cfg = (UnscrewDynTaskCfg() if args.dyn
       else UnscrewRefTaskCfg() if args.ref else UnscrewTaskCfg())
clips.configure_cfg(cfg, args.clip)
cfg.scene.num_envs = args.num_envs
base = (UnscrewRefTaskEnv if args.ref else UnscrewTaskEnv)(cfg)
base.seed(args.seed)
n_steps = base.ep_total * args.episodes
env = GymStyleEnvWrapper(base, clip_actions=cfg.clip_actions)
agent = PPO(env, output_dir="/tmp/unscrew_scout",
            full_config=ConfigWrapper(agent_cfg, cfg, test=True),
            create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()

obs_dict = env.reset()
raw = env.unwrapped
N = raw.num_envs
dev = raw.device
# 回合 1 的成绩快照 (在该 env 首次 done 的那一刻锁定)
done_once = torch.zeros(N, dtype=torch.bool, device=dev)
snap = {k: torch.zeros(N, device=dev) for k in
        ("released", "placed", "place_d", "screw", "step")}

with torch.no_grad():
    for t in range(n_steps):
        _inp = {"obs": agent.running_mean_std(obs_dict["obs"]),
                "priv_info": obs_dict["priv_info"]}
        if "pointcloud" in obs_dict:
            _inp["pointcloud"] = obs_dict["pointcloud"]
        mu = agent.model.act_inference(_inp)
        # 记录 done 之前的状态 (reset 会洗掉 latch)
        rel = raw.released_latch.float()
        pla = raw.placed_latch.float()
        cap_p = raw.cap.data.root_pos_w - raw.scene.env_origins
        pd = (cap_p - raw.cap_goal).norm(dim=1)
        sc = torch.rad2deg(raw.screw_angle)
        obs_dict, r, done, info = env.step(torch.clamp(mu, -1.0, 1.0))
        fresh = done.bool() & ~done_once
        if fresh.any():
            # placed 终止回合且 latch 在 reset 中被洗掉 —— 终止步奖励含
            # place_bonus(100), 用 r>50 补捕获 (pre-step latch 会漏掉
            # "最后一步才放置成功"的回合, 实测 place_dist 0.8cm 被记 0)
            pla_eff = pla + (r.view(-1) > 50.0).float()
            snap["released"][fresh] = rel[fresh]
            snap["placed"][fresh] = pla_eff.clamp(max=1.0)[fresh]
            snap["place_d"][fresh] = pd[fresh]
            snap["screw"][fresh] = sc[fresh]
            snap["step"][fresh] = float(t)
            done_once |= fresh
        if bool(done_once.all()):
            break

rel, pla = snap["released"], snap["placed"]
pd, sc = snap["place_d"], snap["screw"]
print(f"\n===== scout: {int(done_once.sum())}/{N} env 完成回合1 =====")
print(f"释放率 {float(rel.mean())*100:.1f}%  放置率 {float(pla.mean())*100:.1f}%  "
      f"拧角均 {float(sc.mean()):.0f}°  place_tol {cfg.place_tol*100:.0f}cm")
ok = torch.nonzero(pla > 0).flatten()
if len(ok):
    # 完整成功里挑放置最准的
    best = ok[pd[ok].argmin()]
    print(f"\n✅ 完整成功 env: {[int(i) for i in ok[:20]]}  (共 {len(ok)})")
    print(f"   推荐 --env_index {int(best)}  (place_dist {float(pd[best])*100:.1f}cm, "
          f"拧角 {float(sc[best]):.0f}°, 回合长 {int(snap['step'][best])} 步)")
else:
    rel_i = torch.nonzero(rel > 0).flatten()
    if len(rel_i):
        best = rel_i[pd[rel_i].argmin()]
        print(f"\n⚠ 无完整成功; 释放成功 {len(rel_i)} 个, 最接近目标的 "
              f"--env_index {int(best)} (place_dist {float(pd[best])*100:.1f}cm, "
              f"拧角 {float(sc[best]):.0f}°)")
    else:
        best = sc.argmax()
        print(f"\n⚠ 无释放; 拧角最大 --env_index {int(best)} ({float(sc[best]):.0f}°)")
print("[scout] done")
env.close()
app.close()
