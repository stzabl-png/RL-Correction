"""用冠军策略标定新 cent 定义的量纲: 好抓握在法向版下得几分?"""
import argparse, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True); p.add_argument("--clip", default="Grasp5")
p.add_argument("--grasp_prior", default=""); p.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot; _s = isaac_slot("calib")
app = AppLauncher(a).app
import numpy as np, torch, yaml
from rl_rebuild.algo.ppo.ppo import PPO
from rl_rebuild.correction import clips
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
from tasks.pregrasp.cfg import GraspTaskCfg
from tasks.pregrasp.env import GraspTaskEnv
cfg = GraspTaskCfg(); clips.configure_cfg(cfg, a.clip)
if a.grasp_prior: cfg.grasp_prior_npz = a.grasp_prior
cfg.scene.num_envs = 256
raw = GraspTaskEnv(cfg); raw.gentle = 1.0
env = GymStyleEnvWrapper(raw, clip_actions=cfg.clip_actions)
ac = yaml.safe_load(open("tasks/pregrasp/ppo.yaml")); ac["algorithm"]["num_actors"] = 256
ag = PPO(env, output_dir="/tmp/_calib", full_config=ConfigWrapper(ac, cfg, test=True), create_output_dir=False)
ag.restore_test(os.path.abspath(a.checkpoint)); ag.set_eval()
obs = env.reset(); cn, cc, pads = [], [], []
with torch.no_grad():
    for t in range(a.steps):
        i = {"obs": ag.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
        obs, r, d, _ = env.step(torch.clamp(ag.model.act_inference(i), -1, 1))
        pd, pn = raw._pad_dist_normal()
        _, e, c_norm, _, _ = raw._pad_signals(pn)       # 法向版
        _, _, c_ctr, _, _ = raw._pad_signals(None)      # 旧的指向原点版
        m = e.sum(dim=1) >= 4                            # 只统计"已形成多指接触"的 env
        if m.any():
            cn += c_norm[m].tolist(); cc += c_ctr[m].tolist(); pads += (e[m] > 0).sum(1).tolist()
cn, cc = np.array(cn), np.array(cc)
print(f"\n=== 冠军策略 (100% 成功) 的 cent 分布, n={len(cn)} 帧 (≥4垫接触) ===")
for nm, v in (("法向版 normal", cn), ("旧版 center", cc)):
    print(f"  {nm:<14} 均值 {v.mean():.3f}  中位 {np.median(v):.3f}  "
          f"10分位 {np.percentile(v,10):.3f}  90分位 {np.percentile(v,90):.3f}")
print(f"  建议门槛 (法向版 10~25 分位): {np.percentile(cn,10):.2f} ~ {np.percentile(cn,25):.2f}")
env.close(); app.close()
