"""checkpoint -> 优化后轨迹导出 (npz). 最终交付物: 可用作下游训练的高质量演示轨迹.

  OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=. $PY -u \
      -m tasks.recon_kailang.bottle_reconstruction.unscrew_export --headless --ref \
      --checkpoint logs/unscrew/UnscrewRef1/<run>/stage1_nn/last.pth

确定性回放 (μ, 无探索噪声) 一条成功回合, 逐控制步 (20Hz) 记录:
  qpos            (T,58)  双臂 7×2 + 双手 22×2 (按 joint_names 序)
  joint_names     (58,)
  right_wrist     (T,7)   右腕世界位姿 (xyz + wxyz)  —— RL 修正后的实际执行轨迹
  left_wrist      (T,7)
  bottle_pose     (T,7)   瓶位姿 (重建轨迹回放)
  cap_pose        (T,7)   盖位姿 (物理仿真结果)
  screw_angle     (T,)    螺旋坐标 (rad; 4π=拧满)
  ref_clock       (T,)    参考帧号 (门控时钟; 冻结窗内重复)
  action          (T,58)  策略残差动作 (右29+左29; v4 起双臂全在动作空间)
  cap_contacts    (T,5)   右指尖-盖接触
  released/placed (T,)    事件标记
  fps=20, 以及来源/口径元数据 (meta json 同名落盘)

⚠ 只导出**成功**回合 (placed). 多试几个 env, 取第一个成功的.
⚠ 左臂是参考重放 (v2.1 台账): 几何跟随瓶轨迹, 抓握力学未经 RL 验证.
"""
import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--clip", type=str, default="screw_unscrew_cap1_task")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--ref", action="store_true", default=True)
parser.add_argument("--dyn", action="store_true",
                    help="Stage B 全物理瓶 (最终交付口径: 零回放)")
parser.add_argument("--out", type=str, default=None,
                    help="默认 <run_dir>/export/optimized_trajectory.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("unscrew_export")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.recon_kailang.bottle_reconstruction.unscrew_ref_env import (  # noqa: E402
    UnscrewDynTaskCfg,
    UnscrewRefTaskCfg,
    UnscrewRefTaskEnv,
)

ckpt = os.path.abspath(args.checkpoint)
run_dir = os.path.dirname(os.path.dirname(ckpt))
out = args.out or os.path.join(run_dir, "export", "optimized_trajectory.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "unscrew_ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs

env_cfg = UnscrewDynTaskCfg() if args.dyn else UnscrewRefTaskCfg()
clips.configure_cfg(env_cfg, args.clip)
env_cfg.scene.num_envs = args.num_envs

env_raw = UnscrewRefTaskEnv(env_cfg)
env = GymStyleEnvWrapper(env_raw, clip_actions=env_cfg.clip_actions)
agent = PPO(env, output_dir=os.path.join(os.path.dirname(out), ".tmp"),
            full_config=ConfigWrapper(agent_cfg, env_cfg, test=True),
            create_output_dir=False)
print(f"[export] loading {ckpt}")
agent.restore_test(ckpt)
agent.set_eval()

jn = list(env_raw.hand.joint_names)
arm_l = [jn.index(f"L_arm_j{i}") for i in range(1, 8)]
arm_r = [jn.index(f"R_arm_j{i}") for i in range(1, 8)]
hand_l = [jn.index(n) for n in jn if n.startswith("left_")]
hand_r = [jn.index(n) for n in jn if n.startswith("right_")]
export_ids = arm_l + hand_l + arm_r + hand_r
export_names = [jn[i] for i in export_ids]
lw_id = list(env_raw.hand.body_names).index("left_hand_C_MC")
rw_id = list(env_raw.hand.body_names).index("right_hand_C_MC")

obs_dict = env.reset()
origins = env_raw.scene.env_origins
buf = {k: [] for k in ("qpos", "right_wrist", "left_wrist", "bottle_pose",
                       "cap_pose", "screw_angle", "ref_clock", "action",
                       "cap_contacts", "released", "placed")}
done_env = -1
with torch.no_grad():
    for t in range(env_raw.ep_total + 2):
        _inp = {"obs": agent.running_mean_std(obs_dict["obs"]),
                "priv_info": obs_dict["priv_info"]}
        if "pointcloud" in obs_dict:
            _inp["pointcloud"] = obs_dict["pointcloud"]
        mu = torch.clamp(agent.model.act_inference(_inp), -1.0, 1.0)
        # 记录 (动作是本步要执行的)
        buf["qpos"].append(env_raw.hand.data.joint_pos[:, export_ids].cpu().numpy())
        buf["right_wrist"].append(torch.cat([
            env_raw.hand.data.body_pos_w[:, rw_id] - origins,
            env_raw.hand.data.body_quat_w[:, rw_id]], dim=1).cpu().numpy())
        buf["left_wrist"].append(torch.cat([
            env_raw.hand.data.body_pos_w[:, lw_id] - origins,
            env_raw.hand.data.body_quat_w[:, lw_id]], dim=1).cpu().numpy())
        buf["bottle_pose"].append(torch.cat([
            env_raw.object.data.root_pos_w - origins,
            env_raw.object.data.root_quat_w], dim=1).cpu().numpy())
        buf["cap_pose"].append(torch.cat([
            env_raw.cap.data.root_pos_w - origins,
            env_raw.cap.data.root_quat_w], dim=1).cpu().numpy())
        buf["screw_angle"].append(env_raw.screw_angle.cpu().numpy())
        buf["ref_clock"].append(env_raw.ref_clock.cpu().numpy())
        buf["action"].append(mu.cpu().numpy())
        buf["cap_contacts"].append(env_raw._cap_contacts().cpu().numpy())
        buf["released"].append(env_raw.released_latch.cpu().numpy())
        buf["placed"].append(env_raw.placed_latch.cpu().numpy())
        obs_dict, r, done, info = env.step(mu)
        if bool(done.any()):
            # placed 在 reset 前的最后记录里为真的 env = 成功回合.
            # r>50 = 终止步含 place_bonus(100) —— 补捕获"最后一步才放置"的
            # 回合 (pre-step latch 会漏), 且不会被 release_bonus(20) 误触.
            # 并且只认**本步确实 done** 的 env (释放与他人终止同步的误报).
            dn = done.cpu().numpy().astype(bool).reshape(-1)
            cand = np.flatnonzero(
                dn & (buf["placed"][-1].astype(bool)
                      | (r.cpu().numpy().reshape(-1) > 50.0)))
            if len(cand):
                done_env = int(cand[0])
                T_end = t + 1
                break
if done_env < 0:
    raise SystemExit("[export] ✗ 没有 env 在一条回合内放置成功, 不导出")

data = {}
for k, v in buf.items():
    arr = np.stack(v[:T_end])                      # (T, N, ...)
    data[k] = arr[:, done_env]
# 选中判据已保证该回合放置成功; 末帧 latch 因 pre-step 采样可能为 False, 补真
data["placed"][-1] = True
data["released"][-1] = True
data["joint_names"] = np.array(export_names)
data["fps"] = np.float32(20.0)
np.savez_compressed(out, **data)
meta = dict(
    source_clip="screw_unscrew_cap1_task",
    source_bundle="datasets/recon_kailang/screw_unscrew_bottle_cap_1",
    checkpoint=os.path.relpath(ckpt, os.getcwd()),
    control_hz=20.0, replay_clock_hz=15.0,
    steps=int(T_end), env_index=done_env,
    released=bool(data["released"][-1]), placed=bool(data["placed"][-1]),
    final_screw_deg=float(np.degrees(data["screw_angle"].max())),
    frame="env 局部系 (桌心=原点, 桌面 z=0.85); 四元数 wxyz",
    caveats=[
        "双臂均为策略控制 (动作空间 58); Stage A 瓶为运动学回放, Stage B 为全物理",
        "盖为物理仿真结果 (解析螺旋 + 粘滞摩擦抽象)",
        "源数据腕部平移为静态填充, 手腕参考由物体轨迹+活姿态流推导",
    ],
)
with open(out.replace(".npz", ".json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print(f"[export] ✅ env{done_env} 成功回合 {T_end} 步 "
      f"(拧角峰值 {meta['final_screw_deg']:.0f}°) -> {out}")
env.close()
app.close()
