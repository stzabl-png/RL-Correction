"""力学对账探针: 盖的转动是否由真实指尖接触力矩解释?

用户质询 (2026-08-31, 看 run7 录像): "扭开了部分, 但那时候手并没有摩擦来旋转".

逐控制步记录: 模型内部力矩估计 (screw_tau_ema) / 接触力上限 Σ|F|·arm /
有符号切向分量 / 维持当前转速所需力矩 / 滑移 / 接触指数 / 合力.

**验收门 (U44b 更正后)**: Δω 实测 与 (τ_接触 − τ_阻力)·dt/I_eff 预测
**同号率 >= 70%**。

⚠ 已废弃的判据 "滑移<0": force_matrix_w 是**法向力**张量 (IsaacLab 源码
原话 "The normal contact forces"), 证不了摩擦驱动; 且滑移>0 排除不了
**偏心法向推** (手指顶盖沿 = 拨旋钮), 那是合法驱动方式. 滑移仍打印, 仅参考.
⚠ 判 Δω 必须计入螺纹阻力: 2 rad/s 下 drag = τk + b·ω = 75 mN·m,
与接触力矩同量级, 漏掉它会把符号判反 (U44b 教训).

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
print("[audit] 门: Δω 与 (τ_接触−τ_阻力) 同号率 >= 70% (滑移列仅参考, 见 docstring)")

obs = env.reset()
prev_ang = 0.0
adv_slip: list[float] = []      # 拧角推进步的滑移 (盖面 − 指尖)
adv_n = 0
trace: list[tuple] = []         # (步, 角, ω_rel, τ驱动, τ需要, n触)
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
        trace.append((t, ang, float(w_rel[E]), float(1000 * tau_drive[E]),
                      1000 * tau_need, n_c))
        if ang - prev_ang > 0.5 and n_c > 0:      # 有接触且拧角在推进
            adv_slip.append(float(slip[E]))
            adv_n += 1
        prev_ang = ang
        if bool(done[E]) and t > 10:
            print(f"[audit] env{E} 回合结束 @步 {t}")
            break

# ---- 验收门 (U44b 更正后): 一致性检验 —— 盖的角加速度必须由
# "接触力矩 − 螺纹阻力" 解释. 旧的"滑移<0"门已废弃: 它建立在
# force_matrix_w 上, 而那是**法向力**张量, 永远证不了摩擦驱动;
# 且它排除不了偏心法向推 (顶盖沿 = 拨旋钮), 那是合法的驱动方式.
print(f"\n[audit] 推进采样步 n={adv_n}  (参考: 滑移>0 占比 "
      f"{sum(1 for s in adv_slip if s > 0) / max(adv_n, 1):.0%}, 仅供参考不作判据)")
agree = tot = 0
for a, b in zip(trace, trace[1:]):
    if b[0] - a[0] != 1 or a[5] == 0:
        continue
    dw_obs = b[2] - a[2]
    tau_cap = -a[3] / 1000.0                 # 反作用: force_matrix_w 是盖->指尖
    drag = (a[4] / 1000.0) * (1.0 if a[2] > 0 else -1.0)
    dw_pred = (tau_cap - drag) * (12.0 / 240.0) / spec.inertia_eff_kgm2
    if abs(dw_obs) < 1e-3:
        continue
    tot += 1
    agree += (dw_obs > 0) == (dw_pred > 0)
if tot == 0:
    print("[audit] ⚠ 无可用样本 (该 ckpt 在新物理下几乎不转 —— 对旧策略是预期结果)")
else:
    frac = agree / tot
    print(f"[audit] 一致性: Δω 实测与 (τ_接触−τ_阻力) 预测 同号 "
          f"{agree}/{tot} = {frac:.0%}")
    print(f"[audit] {'PASS' if frac >= 0.7 else 'FAIL'} 门: 转动由接触力矩解释"
          f" (判据 同号率 >= 70%)")
print("[audit] 完成")
env.close()
app.close()
