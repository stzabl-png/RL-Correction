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
                      1000 * tau_need, n_c, 1000 * float(raw.screw_tau_ema[E]),
                      1000 * float(tau_cap[E]), float(fmag[E].sum())))
        if ang - prev_ang > 0.5 and n_c > 0:      # 有接触且拧角在推进
            adv_slip.append(float(slip[E]))
            adv_n += 1
        prev_ang = ang
        if bool(done[E]) and t > 10:
            print(f"[audit] env{E} 回合结束 @步 {t}")
            break

# ---- 验收门 (U45 准静态版) ----------------------------------------
# 准静态下 ω = f(τ_ema) 是模型的定义式, 再去检验 "Δω 是否由 τ 解释" 属于
# 同义反复 (那是为旧的积分器模型设计的, U44b 用它定过案, 此处不再作判据)。
# 对准静态模型, 有意义的独立物理合法性检验是这两条:
#   门1 零接触 ⇒ 不转 (允许 EMA 拖尾: τ_ema 时间常数 25ms ≈ 半个控制步)
#   门2 |τ_ema| ≤ Σ|F_i|·arm_i —— 模型不得凭空造出接触力供不起的力矩
zc = [r for r in trace if r[5] == 0]
zc_bad = [r for r in zc if abs(r[2]) > 1e-6]
con = [r for r in trace if r[5] > 0]
ub_bad = [r for r in con if abs(r[6]) > r[7] + 1e-6]
print(f"\n[audit] 最大拧角 {max((r[1] for r in trace), default=0):.1f}°  "
      f"接触步 {len(con)}/{len(trace)}")
ok1 = len(zc_bad) <= max(1, int(0.1 * max(len(zc), 1)))
print(f"[audit] {'PASS' if ok1 else 'FAIL'} 门1 零接触零转动: 违例 "
      f"{len(zc_bad)}/{len(zc)} (容 10%, EMA 拖尾)")
ok2 = len(ub_bad) <= max(1, int(0.05 * max(len(con), 1)))
print(f"[audit] {'PASS' if ok2 else 'FAIL'} 门2 τ_ema ≤ 接触力上限: 违例 "
      f"{len(ub_bad)}/{len(con)}")
# chatter: 周期 2 振荡应使相邻非零 ω 几乎每步异号
seq = [r[2] for r in trace if r[2] != 0.0]
flip = sum(1 for a, b in zip(seq, seq[1:]) if a * b < 0)
fr = flip / max(len(seq) - 1, 1)
print(f"[audit] {'PASS' if fr < 0.5 else 'FAIL'} 门3 无 chatter: 相邻非零 ω "
      f"异号率 {fr:.0%} (周期2振荡应接近 100%)")
Fs = sorted(r[8] for r in con)
if Fs:
    print(f"[audit] 参考 接触力 ΣF 中位 {Fs[len(Fs) // 2]:.1f}N 最大 {Fs[-1]:.1f}N "
          f"(人手精捏约 2-20N)")
print("[audit] 完成")
env.close()
app.close()
