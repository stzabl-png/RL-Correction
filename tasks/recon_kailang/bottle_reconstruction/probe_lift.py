"""pk24: live 瓶 yaw vs 参考瓶 yaw —— U35c 重定向后左手是否跟上了新朝向?

Dyn20@8M 现象: gate 93.7% 但接触≈0, palm_cap 46°, tips 悬停 6.5cm.
假设: 左手携带 yaw 仍是旧习惯 (与 U35c 后的参考差 ~35°), 右手参考按参考轴
对握, 真实盖却在别的方向 -> 掌姿与真实盖错位, 最后 6cm 合不拢.

测 (解锁步): ① live 瓶轴 vs 参考瓶轴夹角 (纯方向);
             ② 分解: 倾角差 vs 绕世界 z 的 yaw 差;
             ③ live 盖轴 vs 右手掌轴 (= palm_cap 复核).
判读: 若 yaw 差 ≈ 35° -> 假设成立, 修左侧 yaw 对齐信号;
      若 yaw 差小 -> 错位另有来源 (右残差在打架), 查右腕跟踪误差.
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("probe_lift")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.recon_kailang.bottle_reconstruction.unscrew_ref_env import (  # noqa: E402
    UnscrewDynTaskCfg,
    UnscrewRefTaskEnv,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "unscrew_ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs

cfg = UnscrewDynTaskCfg()
clips.configure_cfg(cfg, "screw_unscrew_cap1_task")
cfg.scene.num_envs = args.num_envs
base = UnscrewRefTaskEnv(cfg)
n_steps = base.ep_total
env = GymStyleEnvWrapper(base, clip_actions=cfg.clip_actions)
agent = PPO(env, output_dir="/tmp/pk24", full_config=ConfigWrapper(agent_cfg, cfg, test=True),
            create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()

raw = env.unwrapped
dev = raw.device
obs_dict = env.reset()


def axis_of(q):
    x, y, z, w = q[:, 1], q[:, 2], q[:, 3], q[:, 0]
    return torch.stack([2 * (x * z + w * y), 2 * (y * z - w * x),
                        1 - 2 * (x ** 2 + y ** 2)], dim=1)


acc = {k: torch.zeros(1, device=dev) for k in
       ("ax_ang", "yaw_d", "tilt_d", "palm", "wrist_e", "n")}

with torch.no_grad():
    for t in range(n_steps):
        _inp = {"obs": agent.running_mean_std(obs_dict["obs"]),
                "priv_info": obs_dict["priv_info"]}
        if "pointcloud" in obs_dict:
            _inp["pointcloud"] = obs_dict["pointcloud"]
        mu = agent.model.act_inference(_inp)
        obs_dict, r, done, info = env.step(torch.clamp(mu, -1.0, 1.0))

        g = raw.gate_unlocked
        if not g.any():
            continue
        tc = raw.ref_clock.clamp(max=raw.L - 1)
        oq = raw.object.data.root_quat_w
        rq = raw.ref_obj_quat[tc]
        a_live = axis_of(oq)
        a_ref = axis_of(rq)
        ang = torch.rad2deg(torch.arccos(
            (a_live * a_ref).sum(1).clamp(-1, 1)))          # 轴夹角(含方向)
        # 分解: 倾角差 (与世界z的夹角之差) 与 水平投影方位角差 (yaw)
        tilt_l = torch.rad2deg(torch.arccos(a_live[:, 2].clamp(-1, 1)))
        tilt_r = torch.rad2deg(torch.arccos(a_ref[:, 2].clamp(-1, 1)))
        yaw_l = torch.atan2(a_live[:, 1], a_live[:, 0])
        yaw_r = torch.atan2(a_ref[:, 1], a_ref[:, 0])
        dyaw = torch.rad2deg(torch.atan2(torch.sin(yaw_l - yaw_r),
                                         torch.cos(yaw_l - yaw_r)))
        # palm vs live 盖轴
        cq = raw.cap.data.root_quat_w
        a_cap = axis_of(cq)
        hq = raw.wrist_quat_w
        a_hand = axis_of(hq)
        palm = torch.rad2deg(torch.arccos(
            (a_cap * a_hand).sum(1).abs().clamp(max=1)))
        # 右腕离参考位
        org = raw.scene.env_origins
        we = (raw.wrist_pos_w - org - raw.ref_wrist_pos[tc]).norm(dim=1)
        gf = g.float()
        acc["ax_ang"] += (ang * gf).sum(); acc["yaw_d"] += (dyaw.abs() * gf).sum()
        acc["tilt_d"] += ((tilt_l - tilt_r).abs() * gf).sum()
        acc["palm"] += (palm * gf).sum(); acc["wrist_e"] += (we * gf).sum()
        acc["n"] += gf.sum()

n = acc["n"].clamp(min=1)
print("\n===== pk24 判读 (解锁步 n=%d) =====" % int(acc["n"]))
print(f"live瓶轴 vs 参考瓶轴 夹角: {float(acc['ax_ang']/n):5.1f}°")
print(f"  其中 yaw 差 (绕世界z):  {float(acc['yaw_d']/n):5.1f}°   <- U35c 转了 35°")
print(f"  其中 倾角差:            {float(acc['tilt_d']/n):5.1f}°")
print(f"右掌轴 vs live盖轴:       {float(acc['palm']/n):5.1f}°")
print(f"右腕离参考位:             {float(acc['wrist_e']/n)*100:5.1f}cm")
print("判读: yaw差≈35 -> 左手没跟新朝向 (修左yaw信号); yaw差小 -> 右侧问题")
print("[pk24] done")
env.close()
app.close()
