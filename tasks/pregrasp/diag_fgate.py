"""量手指距离门控实际看到的距离 —— 为什么 CTRL2 的 diag/finger_gate 全程 0。

门控用 |wrist_pos_w - object.root_pos_w| 与自标定基准 _fgate_dg 比。纸面上两者同框架,
但 TB 显示门开度恒为 0(= 距离恒 > 2.5×d_g)。这里在真实 env 里把两边都打出来对账。

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.diag_fgate --headless \\
      --clip Grasp3 --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/8_5.npz \\
      --prior_yaw 215 --stance_prefix 60
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", type=str, default="Grasp3")
parser.add_argument("--grasp_prior", type=str, required=True)
parser.add_argument("--prior_yaw", type=float, default=-1.0)
parser.add_argument("--stance_prefix", type=int, default=0)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--force_calib", action="store_true",
                    help="强制应用零位校准 (绕过 参与指<=2 的放行判据), 用于 A/B")
parser.add_argument("--no_calib", action="store_true",
                    help="关掉垫↔接触零位校准 (A/B: 它到底是帮忙还是帮倒忙)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("diag_fgate")
app = AppLauncher(args).app

import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

env_cfg = GraspTaskCfg()
clips.configure_cfg(env_cfg, args.clip)
apply_grasp_prior(env_cfg, args.grasp_prior, args.prior_yaw, approach=True)
env_cfg.stance_prefix_frames = args.stance_prefix
env_cfg.scene.num_envs = args.num_envs
env_cfg.direct_grasp_prob = 1.0      # 全部从 prior 抓握位起步 = 门控本该完全打开的情形
env_cfg.closure_init_max = 0.0
if args.no_calib:
    env_cfg.pad_contact_calib = False   # A/B: 关掉零位校准
if args.force_calib:                    # A/B: 强制开 (绕过包络抓的放行判据)
    env_cfg.pad_calib_max_fingers = 5
    print("[A/B] 强制应用零位校准 (pad_calib_max_fingers=5, 绕过放行判据)")
raw = GraspTaskEnv(env_cfg)
raw.reset()

org = raw.scene.env_origins
w_abs, o_abs = raw.wrist_pos_w, raw.object.data.root_pos_w
d = (w_abs - o_abs).norm(dim=1)
near = raw._fgate_dg * env_cfg.finger_gate_near_k
far = raw._fgate_dg * env_cfg.finger_gate_far_k
gate = ((far - d) / max(far - near, 1e-6)).clamp(0.0, 1.0)

print("\n" + "=" * 72)
print(f"[标定基准] _fgate_dg = {raw._fgate_dg * 100:.2f} cm   "
      f"(= |_grasp_pos_w - obj_init_pos|, 建场景时算)")
print(f"[运行实测] |wrist_pos_w - obj.root_pos_w| = "
      f"{(d * 100).min():.2f} ~ {(d * 100).max():.2f} cm")
print(f"[门开度 ] near={near*100:.1f}cm far={far*100:.1f}cm  ->  gate = "
      f"{gate.min():.3f} ~ {gate.max():.3f}     (0 = 手指冻死)")
print("-" * 72)
print(f"  wrist 绝对 = {[round(x, 4) for x in w_abs[0].tolist()]}")
print(f"  wrist 局部 = {[round(x, 4) for x in (w_abs[0] - org[0]).tolist()]}")
print(f"  obj   绝对 = {[round(x, 4) for x in o_abs[0].tolist()]}")
print(f"  obj   局部 = {[round(x, 4) for x in (o_abs[0] - org[0]).tolist()]}")
print(f"  _grasp_pos_w = {[round(x, 4) for x in raw._grasp_pos_w.tolist()]}")
print(f"  obj_init_pos = {[round(x, 4) for x in raw.obj_init_pos.tolist()]}")
print(f"  env_origins[0] = {[round(x, 4) for x in org[0].tolist()]}")
print("=" * 72 + "\n")

# ---- 零学习基线: 从 GraspPose 起手, 发零动作, 让合拢斜坡自己跑 -------------
# 问的是"这个 prior 在**当前场景**里到底抓不抓得住" —— 与策略好坏无关。
# 冠军时代同一条 clip 的 prior 是抓得住的; 若现在抓不住, 说明场景/校准改动把任务改坏了。
print("[零动作基线] 从 GraspPose 起手, 合拢斜坡自跑 …")
A = int(raw.single_action_space.shape[0])
zero = torch.zeros(raw.num_envs, A, device=raw.device)
hdr = f"{'步':>4} {'closure':>8} {'pads_now':>9} {'gate':>6} {'obj_dz(mm)':>11}"
print(hdr)
obj_z0 = raw.object.data.root_pos_w[:, 2].clone()
for t in range(1, 121):
    raw.step(zero)
    if t % 15 and t != 1:
        continue
    d_now = (raw.wrist_pos_w - raw.object.data.root_pos_w).norm(dim=1)
    g = ((far - d_now) / max(far - near, 1e-6)).clamp(0.0, 1.0)
    pads = raw._sig.get("n_pads")           # 候选判据真正看的量 (指垫数)
    dz = (raw.object.data.root_pos_w[:, 2] - obj_z0) * 1000.0
    print(f"{t:>4} {raw.closure.mean():>8.3f} "
          f"{(float(pads.float().mean()) if pads is not None else float('nan')):>9.2f} "
          f"{g.mean():>6.3f} {dz.mean():>11.2f}")
print("\n(pads 长期 0 = prior 在当前场景抓不住)")

# ---- 指令位形 vs 物理稳态: 到位判据的位置项到底能不能达到 ----
# 2026-08-16: 确定性策略在所有起点上位置差都停在 1.60~1.62cm(门槛 1.04cm), 而朝向
# 0.44°、腕速 100% 达标 ⟹ 卡在**精度**且是**系统性**的(256 env 几乎同一个值)。
# 代码里早有记录:"指令位形 ≠ 物理稳态, 实测 PD 稳态肘角与 IK 解差 8.4°"。
# 这里把臂直接命令到 GraspPose 的精确 IK 解、让物理稳定, 量 d_pos ——
# 若它本身就 >1.04cm, 则**任何策略都过不了这个判据**, 与学习无关。
print("\n[指令 vs 稳态] 把臂命令到 GraspPose 的 IK 解, 静置后量到位误差:")
q_all = raw.hand.data.joint_pos[0].clone()
q_all[raw.arm_jids] = raw._prior_q_grasp
q_all[raw.hand_jids] = raw.q_open
qa = q_all.unsqueeze(0).expand(raw.num_envs, -1).contiguous()
raw.hand.write_joint_state_to_sim(qa, torch.zeros_like(qa))
raw.hand.set_joint_position_target(qa)
raw.arm_tgt[:] = raw._prior_q_grasp
raw.q_cmd[:] = raw._prior_q_grasp
import numpy as _np
for _k in range(120):
    raw.hand.set_joint_position_target(qa)
    raw.hand.write_data_to_sim()
    raw.sim.step(render=False)
    raw.scene.update(0.0)
    if _k in (0, 9, 29, 59, 119):
        dp, dr, _ = raw._align_err()
        _qe = (raw.arm_q - raw._prior_q_grasp.unsqueeze(0)).abs().max(dim=1).values
        print(f"   静置 {_k+1:>3} 步: 位置差 {dp.mean()*100:6.2f}cm  朝向差 "
              f"{_np.degrees(dr.mean().item()):5.2f}°  最大关节误差 "
              f"{_np.degrees(_qe.mean().item()):5.2f}°")
print(f"   判读: 位置差 > {raw.cfg.eps_pos*100:.2f}cm(门槛) ⟹ 该位姿在当前增益下"
      f"**物理上到不了**, 任何策略都过不了判据。")

# ---- 归因: 0 个指垫是"够不着"还是"用错部位碰上了" ----------------------
# pad_d = 每个指垫(elastomer link 原点)到物体表面的最近距离。
#   全部 >1cm  -> 够不着 (手合拢到底仍离物体远)
#   有的 ~0 但 pads_on=0 -> 碰上了但判据不认 (接触力阈值/法向条件没过)
#   物体被顶动而 pad_d 大 -> 用**非指垫部位**(指节/掌)碰的
_pd, _pn, _ = raw._pad_dist_normal()
print("\n[归因] 逐指 指垫->物体表面 最近距离 (cm, 合拢到底时):")
_names = ["thumb", "index", "middle", "ring", "pinky"]
for _i, _nm in enumerate(_names):
    print(f"    {_nm:<7s} {(_pd[:, _i] * 100).mean():>7.2f} cm"
          f"   (最近的 env {(_pd[:, _i] * 100).min():.2f})")
print(f"    -> 五指均值 {(_pd * 100).mean():.2f} cm; "
      f"物体位移 dz={dz.mean():.2f}mm")
print("    判读: 距离大而物体被顶动 = 用非指垫部位碰的; 距离小而 pads=0 = 判据没认。")
app.close()
