"""接近段塌缩的定点诊断: 把两个起步分支**分开**跑, 看是哪一半坏了。

背景 (2026-08-02, `G3_approach_8_5` 9.4M 步塌缩):
  `sr/from_grasp` 从 0.54 掉到 0 再没回来, `pads_now` 1.79 (纯抓取 run 是 4.35),
  `verify/wrist_rise_mm` 3.81 (纯抓取 run 是 16.88). 两个互斥的嫌疑:
    H1 切换逻辑污染 —— 接近段切到抓取时 `_set_arm_center` 改了偏差带中心, 而抬升斜坡
       `q_lift_delta` 是按 `q_pregrasp` 预解的, 两者耦合 ⟹ 切过来的回合抬不动.
    H2 梯度互相淹 —— 两半共享网络, 几乎必然失败的接近回合把抓取半的优势估计拉偏.

判别 (关键: from_grasp 起步的回合**永远不进 PREGRASP**, 切换逻辑对它们根本不执行):
    A 组 (direct_grasp_prob=1.0) 正常  +  B 组 (=0.0) 坏  ⟹ **H1**, 去修代码耦合
    A 组也坏                            ⟹ **H2**, 是策略本身不会抓了, 调课程配比

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.diag_split --headless \\
        --clip Grasp3 --checkpoint logs/G3_approach_8_5/<ts>/stage1_nn/last.pth \\
        --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/8_5.npz --prior_yaw 215
"""
import argparse
import os

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", required=True)
p.add_argument("--clip", default="Grasp3")
p.add_argument("--grasp_prior", default="")
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--steps", type=int, default=420)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot                    # noqa: E402
_slot = isaac_slot("diag_split")
app = AppLauncher(args).app

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
import yaml                                                          # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO                              # noqa: E402
from rl_rebuild.correction import clips                              # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper          # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg, Phase, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv                          # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.scene.num_envs = args.num_envs
raw = GraspTaskEnv(cfg)
raw.gentle = 1.0
env = GymStyleEnvWrapper(raw, clip_actions=cfg.clip_actions)
agent = PPO(env, output_dir="/tmp/_diag_split",
            full_config=ConfigWrapper(agent_cfg, cfg, test=True), create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()
N, dev = raw.num_envs, raw.device


def block(tag, dgp, t0max):
    raw.cfg.direct_grasp_prob = dgp
    raw.cfg.approach_t0_max = t0max
    obs = env.reset()
    succ = ep = cand = 0
    pads, wr, orr, dcmd, bandw = [], [], [], [], []
    with torch.no_grad():
        for _ in range(args.steps):
            inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
            act = agent.model.act_inference(inp)
            obs, _, done, _ = env.step(torch.clamp(act, -1.0, 1.0))
            s = raw._sig
            pads.append(float(s["n_pads"].float().mean()))
            # 进验证段那一刻: 记 q_cmd 离 q_pregrasp 多远 + 偏差带宽度
            iv = (raw.task_phase == Phase.LIFT) & (raw.verify_k == 1)
            if bool(iv.any()):
                dcmd.append(float((raw.q_cmd[iv] - raw.q_pregrasp).abs().max(dim=1)
                                  .values.mean()))
                bandw.append(float((raw.band_hi[iv] - raw.band_lo[iv]).mean()))
            d = done.bool() if torch.is_tensor(done) else torch.tensor(done, device=dev).bool()
            if d.any():
                i = torch.nonzero(d, as_tuple=False).squeeze(-1)
                ep += len(i); succ += int(s["newly_success"][i].sum())
                cand += int(raw.got_candidate[i].sum())
                m = raw.vf_has[i]
                if bool(m.any()):
                    wr += raw.vf_wrist_mm[i][m].tolist()
                    orr += raw.vf_obj_mm[i][m].tolist()
    f = lambda x: (np.mean(x) if len(x) else float("nan"))
    print(f"\n[{tag}]  回合 {ep} | 成功 {succ} ({succ/max(ep,1)*100:.1f}%) | "
          f"候选率 {cand/max(ep,1):.2f}")
    print(f"    pads_now(同时接触) {f(pads):5.2f} | 腕抬升 {f(wr):6.2f}mm | "
          f"物体升 {f(orr):6.2f}mm | 跟随率 {f(orr)/max(f(wr),1e-6):5.3f}")
    print(f"    进验证时 |q_cmd−q_pregrasp|max {np.degrees(f(dcmd)):6.2f}° | "
          f"偏差带宽 {np.degrees(f(bandw)):6.2f}°")
    return succ / max(ep, 1)


print(f"\n{'='*74}\n抬升斜坡(建 env 时预解)第 4 级 = 见上面 [prior] 那行\n{'='*74}")
a1 = block("A 组: 全部从抓取起步 (direct_grasp_prob=1.0, 永不进 PREGRASP)", 1.0, 0.0)
a2 = block("B 组: 全部从接近起步 (direct_grasp_prob=0.0)", 0.0, 0.8)
print(f"\n{'='*74}")
print("判读:  A 正常 & B 坏  ⟹ H1 切换逻辑污染 (去修 _set_arm_center / q_lift 的耦合)")
print("       A 也坏          ⟹ H2 梯度互相淹 (策略本身不会抓了, 调 direct_grasp_prob)")
print(f"{'='*74}")
env.close()
app.close()
