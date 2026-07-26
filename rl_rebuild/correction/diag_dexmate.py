"""诊断 DexMate env: 分辨"臂跟不上参考" vs "我的 FK 和 Isaac 对不上".

零残差 baseline 出现 恒定 7.4cm 跟踪误差 + 力矩恒定顶满 时用这个定位.
关键判别: 把 q_actual 喂进自己的 FK, 和 Isaac 报的末端位姿比 ——
  两者一致  -> FK 对, 是臂真的没走到 q_ref (增益/碰撞/限位问题)
  两者不一致 -> FK 错 (基座/躯干/末端 link 对不上), q_ref 本身就是错的目标

  $PY -m rl_rebuild.correction.diag_dexmate --clip Grasp2
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp2")
p.add_argument("--steps", type=int, default=60)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
args.headless = True

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("diag")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.env.dexmate_env import DexmateCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK  # noqa: E402

cfg = DexmateCorrectionEnvCfg()
clips.configure_cfg(cfg, args.clip)
cfg.scene.num_envs = 1
cfg.rsi_prob = 0.0
env = DexmateCorrectionEnv(cfg)
env.reset()
zero = torch.zeros(1, cfg.action_space, device=env.device)

# 用 env **自己**建的那把锚定 IK, 否则比的是两套不同假设 (会看到假的 FK 不一致)
ik = ArmIK(cfg.hand_side, anchor_link="arm_center", anchor_T=env._anchor_T)
org = env.scene.env_origins[0].cpu().numpy()
bn = list(env.hand.body_names)
jn = list(env.hand.joint_names)

print("\n" + "=" * 100)
print("执行器解析结果 + PD 各项 (applied_torque = clip(kp·Δq + kd·Δq̇)):")
for _ in range(5):
    env.step(zero)
d = env.hand.data
kp, kd = d.joint_stiffness[0].cpu().numpy(), d.joint_damping[0].cpu().numpy()
try:
    elim = env.hand.root_physx_view.get_dof_max_forces()[0].cpu().numpy()
except Exception:
    elim = np.full(len(jn), np.nan)
tgt = d.joint_pos_target[0].cpu().numpy()
qa_ = d.joint_pos[0].cpu().numpy()
qv_ = d.joint_vel[0].cpu().numpy()
tau_ = d.applied_torque[0].cpu().numpy()
ce_ = d.computed_torque[0].cpu().numpy() if hasattr(d, "computed_torque") else tau_ * np.nan
print(f"{'关节':<12}{'kp':>9}{'kd':>8}{'力矩上限':>10}{'目标°':>9}{'实际°':>9}"
      f"{'Δq°':>8}{'q̇rad/s':>9}{'kp·Δq':>10}{'kd·Δq̇':>10}{'computed':>10}{'applied':>10}")
for n in [f"R_arm_j{i}" for i in range(1, 8)] + ["L_arm_j1", "L_arm_j5", "torso_j1"]:
    i = jn.index(n)
    dq = tgt[i] - qa_[i]
    print(f"{n:<12}{kp[i]:9.1f}{kd[i]:8.1f}{elim[i]:10.1f}{np.degrees(tgt[i]):9.2f}"
          f"{np.degrees(qa_[i]):9.2f}{np.degrees(dq):8.2f}{qv_[i]:9.3f}"
          f"{kp[i]*dq:10.1f}{-kd[i]*qv_[i]:10.1f}{ce_[i]:10.1f}{tau_[i]:10.1f}")

print("\n" + "=" * 100)
print("前 12 步的终止判据 (episode_length_buf 一直不涨 = 每步都在 reset):")
for k in range(12):
    env.step(zero)
    op = env.object.data.root_pos_w - env.scene.env_origins
    t_ = env._ref_t()
    print(f"  步{k:3d} ep_len={int(env.episode_length_buf[0]):3d} t={int(t_[0]):3d} "
          f"| fell={bool(env._last_fell[0])} obj_div={bool(env._last_obj_div[0])} "
          f"wrist_div={bool(env._last_wrist_div[0])} "
          f"stuck={bool(getattr(env,'_last_stuck',t_*0)[0])} "
          f"err_ctr={int(env.arm_err_ctr[0])} "
          f"| 物体z={float(op[0,2]):.4f} (掉落线 {cfg.table_top_z-cfg.term_obj_fall:.3f})")
env.reset()

for k in range(args.steps):
    env.step(zero)
    if k % 20 != 0 and k != args.steps - 1:
        continue
    t = int(env._ref_t()[0])
    q = env.arm_q[0].cpu().numpy()
    qr = env.q_ref[t].cpu().numpy()
    tau = env.hand.data.applied_torque[0, env.arm_jids].cpu().numpy()
    ee_isaac = env.wrist_pos_w[0].cpu().numpy() - org
    ee_fk_actual = ik.fk(q)[0]                       # 我的 FK 喂 Isaac 的实际关节角
    ee_fk_ref = ik.fk(qr)[0]                         # 我的 FK 喂参考关节角
    ref = env.ref_wrist_pos[t].cpu().numpy()
    print(f"\n--- 步 {k}  参考帧 t={t} ---")
    print(f"  臂关节实际 (度): {np.round(np.degrees(q), 2)}")
    print(f"  臂关节参考 (度): {np.round(np.degrees(qr), 2)}")
    print(f"  逐关节误差 (度): {np.round(np.degrees(q - qr), 2)}   "
          f"|最大| {np.degrees(np.abs(q-qr)).max():.2f}°")
    print(f"  施加力矩 (Nm):   {np.round(tau, 1)}")
    print(f"  归一化力矩:      {np.round(np.abs(tau)/env.arm_effort_limit.cpu().numpy(), 3)}")
    print(f"  末端 Isaac 实测:  {np.round(ee_isaac, 4)}")
    print(f"  末端 我的FK(q实际):{np.round(ee_fk_actual, 4)}   "
          f"差 {np.linalg.norm(ee_isaac-ee_fk_actual)*100:.2f}cm   <-- FK 一致性")
    print(f"  末端 我的FK(q参考):{np.round(ee_fk_ref, 4)}")
    print(f"  参考腕位:         {np.round(ref, 4)}   "
          f"离实测 {np.linalg.norm(ee_isaac-ref)*100:.2f}cm")

# 躯干 / 非交互臂 有没有跑偏
print("\n" + "=" * 100)
print("躯干 + 其它关节 实际 vs 默认:")
qa = env.hand.data.joint_pos[0].cpu().numpy()
qd = env.hand.data.default_joint_pos[0].cpu().numpy()
tau_all = env.hand.data.applied_torque[0].cpu().numpy()
for n in ["torso_j1", "torso_j2", "torso_j3"] + [f"L_arm_j{i}" for i in range(1, 8)]:
    i = jn.index(n)
    print(f"  {n:10s} 实际 {np.degrees(qa[i]):8.2f}°  默认 {np.degrees(qd[i]):8.2f}°  "
          f"差 {np.degrees(qa[i]-qd[i]):+7.2f}°   力矩 {tau_all[i]:9.1f} Nm")

# 机器人有没有和桌子/自己撞上: 看每个 body 的净接触力
print("\n躯干/臂 body 世界位置 (env-local) —— 桌板 z∈[0.81,0.85], |x|,|y|<0.6:")
bp = env.hand.data.body_pos_w[0].cpu().numpy() - org
for n in ["vega_1p_base", "vega_1p_torso_l1", "vega_1p_torso_l2", "vega_1p_torso_l3",
          "arm_center", "vega_1p_R_arm_l1", "R_arm_l3", "R_arm_l4", "R_arm_l5",
          "vega_1p_R_arm_l6", "R_arm_l7", "right_hand_C_MC"]:
    if n in bn:
        v = bp[bn.index(n)]
        inside = abs(v[0]) < 0.6 and abs(v[1]) < 0.6 and 0.81 < v[2] < 0.85
        print(f"  {n:22s} {np.round(v,3)} {'  ⚠ 在桌板体积内' if inside else ''}")


# ---- 手/指有没有插进桌面 ----
# 末端误差方向 = 比参考"高 + 往机器人方向退" -> 典型的被桌面顶住. 逐 body 查.
print("\n" + "=" * 100)
bp = env.hand.data.body_pos_w[0].cpu().numpy() - org
top = cfg.table_top_z
sub = [(n, bp[bn.index(n)]) for n in bn
       if n.startswith(cfg.hand_side) and bp[bn.index(n)][2] < top + 0.02]
print(f"桌面 z={top}. 手部 body 低于 桌面+2cm 的有 {len(sub)} 个:")
for n, v in sorted(sub, key=lambda x: x[1][2])[:14]:
    print(f"  {n:28s} z={v[2]:.4f} {'  ⚠ 在桌面之下' if v[2] < top else ''}")
zs = np.array([bp[bn.index(n)][2] for n in bn if n.startswith(cfg.hand_side)])
print(f"手部 body z 最低 {zs.min():.4f} / 桌面 {top}  -> "
      f"{'插进桌面 ' + str(int((zs<top).sum())) + ' 个 body' if (zs<top).any() else '未插入'}")

ee_i = env.wrist_pos_w[0].cpu().numpy() - org
ref_i = env.ref_wrist_pos[int(env._ref_t()[0])].cpu().numpy()
d = ee_i - ref_i
print(f"\n末端偏离参考: {np.round(d,4)}  (+z=被顶高, -x=往机器人退)  模长 {np.linalg.norm(d)*100:.2f}cm")


# ---- 逐 link 比对 Isaac vs 我的 FK: 第一个分叉的 link 就是病根 ----
print("\n" + "=" * 100)
print("运动学链逐 link 对比 (Isaac 实测 vs 我的 FK 喂同一组实际关节角):")
from rl_rebuild.correction.kinematics import Urdf  # noqa: E402
u = Urdf()
qa_all = env.hand.data.joint_pos[0].cpu().numpy()
qmap = {n: float(qa_all[jn.index(n)]) for n in jn}       # **用实际关节角**, 不用假设值
root_state = env.hand.data.root_state_w[0].cpu().numpy()
print(f"  根 link 世界位姿 pos={np.round(root_state[:3]-org,4)} quat={np.round(root_state[3:7],4)}")
print(f"  cfg 里假设的基座    pos={cfg.dexmate_pos} quat={cfg.dexmate_quat}")
print(f"  dummy_base 关节值: " + "  ".join(
    f"{n}={qmap[n]:+.4f}" for n in jn if n.startswith("dummy_base")))
print(f"  torso 关节值(度):  " + "  ".join(
    f"{n}={np.degrees(qmap[n]):+.2f}" for n in jn if n.startswith("torso_j")))
print(f"\n  {'link':<24}{'Isaac':>34}{'我的FK':>34}{'差(cm)':>9}")
for link in ["vega_1p_base", "vega_1p_torso_l1", "vega_1p_torso_l2", "vega_1p_torso_l3",
             "arm_center", "vega_1p_R_arm_l1", "R_arm_l2", "R_arm_l3", "R_arm_l4",
             "R_arm_l5", "vega_1p_R_arm_l6", "R_arm_l7", "right_hand_C_MC"]:
    if link not in bn or link not in u.parent_joint:
        continue
    pi = env.hand.data.body_pos_w[0, bn.index(link)].cpu().numpy() - org
    pf = u.link_pose(link, qmap, ik.base_T, "arm_center") if link in (
        "vega_1p_R_arm_l1", "R_arm_l2", "R_arm_l3", "R_arm_l4", "R_arm_l5",
        "vega_1p_R_arm_l6", "R_arm_l7", "right_hand_C_MC") else u.link_pose(
        link, qmap, __import__("rl_rebuild.correction.kinematics", fromlist=["x"])._T(
            np.eye(3), np.array(cfg.dexmate_pos, float)))
    print(f"  {link:<24}{str(np.round(pi,4)):>34}{str(np.round(pf,4)):>34}"
          f"{np.linalg.norm(pi-pf)*100:>8.2f}")

sys.stdout.flush()
env.close()
_slot.release()
os._exit(0)
