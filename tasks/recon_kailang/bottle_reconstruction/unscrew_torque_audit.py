"""U44 力学对账探针: 拧转角的推进是否被真实指尖接触力支撑?

用户质询 (2026-08-31, 看 run7 录像): "扭开了部分, 但那时候手并没有摩擦来旋转".

逐控制步同时记录两个量:
  tau_est  = 螺纹模型内部的力矩估计 (screw_tau_ema)
  tau_cap  = Σ|F_i| × 力臂_i  —— PhysX 真实接触力给出的**物理上限**
             (假设全部力都是切向, 已经是最宽松的上界)
以及 omega 分解: 相对角速度 / 盖绝对 / 瓶绝对 (都投影到螺轴).

判据: 若拧角推进期 tau_cap << 维持该转速所需力矩 (tau_need = τk + b·ω),
      则转动不是指尖摩擦驱动的 = 幻影.

  OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=. \
      CUDA_VISIBLE_DEVICES=1 $PY -u \
      -m tasks.recon_kailang.bottle_reconstruction.unscrew_torque_audit \
      --headless --checkpoint <ckpt> --steps 420
"""
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--clip", type=str, default="screw_unscrew_cap1_task_real")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=420)
parser.add_argument("--env_index", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("unscrew_torque_audit")
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper  # noqa: E402
from tasks.recon_kailang.bottle_reconstruction.unscrew_ref_env import (  # noqa: E402
    UnscrewDynTaskCfg,
    UnscrewRefTaskEnv,
)
from rl_rebuild.algo.ppo.ppo import PPO  # noqa: E402
from rl_rebuild.wrapper.config_wrapper import ConfigWrapper  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(_HERE, "unscrew_ppo.yaml")) as f:
    agent_cfg = yaml.safe_load(f)
agent_cfg["algorithm"]["num_actors"] = args.num_envs

env_cfg = UnscrewDynTaskCfg()
clips.configure_cfg(env_cfg, args.clip)
env_cfg.scene.num_envs = args.num_envs

base = UnscrewRefTaskEnv(env_cfg, render_mode=None)
env = GymStyleEnvWrapper(base, clip_actions=env_cfg.clip_actions)
raw = env.unwrapped
spec = raw.screw_spec
E = args.env_index

agent = PPO(env, output_dir="/tmp/_audit_tmp",
            full_config=ConfigWrapper(agent_cfg, env_cfg, test=True),
            create_output_dir=False)
agent.restore_test(args.checkpoint)
agent.set_eval()

print(f"[audit] spec breakaway={spec.breakaway_torque_nm} kinetic={spec.kinetic_torque_nm} "
      f"viscous={spec.viscous_nms} I_eff={spec.inertia_eff_kgm2}")
print("[audit] 列: 步 | 角° | ω_rel | ω_瓶 | 锁 | τ估计 | τ上限 | τ驱动(有符号) | τ需要 "
      "| 滑移(盖-指,m/s) | n触 | ΣF(N)")
print("[audit] 判据: |τ驱动| < τ需要 且 滑移 > 0 ⇒ 盖跑在手指前面, 转动非指尖驱动 = 幻影")

obs = env.reset()
prev_ang = 0.0
adv_slip: list[float] = []      # 拧角推进步的滑移 (盖面 − 指尖)
adv_n = 0
with torch.no_grad():
    for t in range(args.steps):
        mu = agent.model.act_inference(
            {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]})
        obs, r, done, info = env.step(torch.clamp(mu, -1.0, 1.0))

        # ---- 螺轴 (瓶身局部 +z) ----
        axis_local = torch.zeros(raw.num_envs, 3, device=raw.device)
        axis_local[:, 2] = 1.0
        axis_w = quat_apply(raw.object.data.root_quat_w, axis_local)

        w_cap = (raw.cap.data.root_ang_vel_w * axis_w).sum(dim=1)
        w_bot = (raw.object.data.root_ang_vel_w * axis_w).sum(dim=1)
        w_rel = w_cap - w_bot

        # ---- 真实接触力 -> 物理上限力矩 Σ|F_i|·力臂_i ----
        f = torch.cat([s.data.force_matrix_w.view(raw.num_envs, 1, 3)
                       for s in raw._cap_sensors], dim=1)          # (N,5,3)
        fmag = f.norm(dim=-1)                                       # (N,5)
        tips = raw.tip_pos_w                                        # (N,5,3)
        rel = tips - raw.cap.data.root_pos_w[:, None]
        radial = rel - (rel * axis_w[:, None]).sum(-1, keepdim=True) * axis_w[:, None]
        arm = radial.norm(dim=-1)                                   # (N,5) 到螺轴垂距
        tau_cap = (fmag * arm).sum(dim=1)                           # 全切向的最宽上界
        # ---- 有符号切向驱动力矩: 只有切向分量能驱动螺纹 ----
        rhat = radial / radial.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        that = torch.cross(axis_w[:, None].expand_as(rhat), rhat, dim=-1)  # 切向单位
        ft = (f * that).sum(dim=-1)                                 # (N,5) 有符号
        contact = (fmag > raw.cfg.contact_force_thresh).float()
        tau_drive = (ft * arm * contact).sum(dim=1)                 # 真实驱动力矩
        # ---- 滑移: 盖面切向速度 vs 指尖切向速度 (盖跑得快 = 不是手指在带) ----
        tipv = raw.hand.data.body_lin_vel_w[:, raw.tip_ids]         # (N,5,3)
        v_tip_t = (tipv * that).sum(dim=-1)                         # 指尖切向速度
        capsurf = raw.cap.data.root_lin_vel_w[:, None] + torch.cross(
            raw.cap.data.root_ang_vel_w[:, None].expand_as(rel), rel, dim=-1)
        v_cap_t = (capsurf * that).sum(dim=-1)                      # 盖面切向速度
        slip = ((v_cap_t - v_tip_t) * contact).sum(dim=1) / contact.sum(dim=1).clamp(min=1)

        w_abs = w_rel[E].abs()
        tau_need = spec.kinetic_torque_nm + spec.viscous_nms * float(w_abs)
        ang = float(torch.rad2deg(raw.screw_angle[E]))
        n_c = int(raw._cap_contacts()[E].sum())

        if t % 4 == 0 or abs(ang - prev_ang) > 0.5:
            print(f"[audit] {t:4d} | {ang:7.1f} | {float(w_rel[E]):+6.3f} | "
                  f"{float(w_bot[E]):+6.3f} | "
                  f"{int(raw.screw_locked[E])} | {1000 * float(raw.screw_tau_ema[E]):7.1f} | "
                  f"{1000 * float(tau_cap[E]):7.1f} | {1000 * float(tau_drive[E]):+8.1f} | "
                  f"{1000 * tau_need:7.1f} | {float(slip[E]):+7.3f} | "
                  f"{n_c} | {float(fmag[E].sum()):6.2f}")
        if ang - prev_ang > 0.5 and n_c > 0:      # 有接触且拧角在推进
            adv_slip.append(float(slip[E]))
            adv_n += 1
        prev_ang = ang
        if bool(done[E]) and t > 10:
            print(f"[audit] env{E} 回合结束 @步 {t}")
            break

# ---- U44 新验收门: 推进必须由指尖带动 (指尖跑在盖面前面 => 滑移 < 0) ----
print(f"\n[audit] 推进采样步 n={adv_n}")
if adv_n == 0:
    print("[audit] ⚠ 无推进步 —— 该 ckpt 在修复后的物理下拧不动 (预期: 旧策略靠幻影)")
else:
    pos = sum(1 for s in adv_slip if s > 0)
    frac = pos / adv_n
    mean_slip = sum(adv_slip) / adv_n
    print(f"[audit] 推进步中 滑移>0 (盖跑在手指前面) 占比 = {frac:.1%}  "
          f"均值 = {mean_slip:+.4f} m/s")
    ok = frac < 0.5
    print(f"[audit] {'PASS' if ok else 'FAIL'} U44门: 推进由指尖驱动"
          f" (判据 滑移>0 占比 < 50%)")
print("[audit] 完成")
env.close()
app.close()
