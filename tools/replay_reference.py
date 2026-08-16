"""零学习**参考回放**基线 —— 开训前必做的那一步(POUR_TRAINING_DESIGN §13.6)。

不带策略, 把参考轨迹**绝对地**灌进机器人(绕开残差机制), 量三件事:

    ① 手够不够得到物体(臂 IK 参考本身可达吗)
    ② 手指按参考合拢时**真的碰到物体**吗(逐指接触力)
    ③ 能不能拿起来(物体高度变化)

**为什么必须先做**: 新范式把残差基线换成"参考", 若参考自己抓不住, 策略就要从一个
抓不住的姿态用高维残差探索出抓握 —— 正是 47_6/4_6 训到全零的形态(合拢轨迹从物体旁
掠过, 接触梯度恒零)。回放能抓住 ⟹ 基线可用; 掠过 ⟹ 提前知道 GraspPose 软项要挑
多重的担子。

两种手指来源可选(--fingers), 直接给出对比:
    ramp  : env 现有的 `ref_finger` —— 其实是 replay_grasp 里的**脚本开→合斜坡**
            (不是人手指姿! 这是本工具查出来的事实)
    human : `ref_qpos_<hand>.npz` 的 **DexPilot 重定向人手指姿**(需先跑 make_ref_qpos)

用法:
    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tools.replay_reference --clip Pour17_bottle \
        --fingers ramp --headless
"""
import argparse

import numpy as np
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--fingers", default="ramp", choices=["ramp", "human"])
p.add_argument("--ref_qpos", default="", help="human 模式的 ref_qpos npz; 空=按 clip 自动推")
p.add_argument("--settle", type=int, default=20, help="回放前静置控制步")
p.add_argument("--hold", type=int, default=30, help="回放后保持控制步(看抓没抓住)")
p.add_argument("--report", default="", help="把结果写成 json")
p.add_argument("--loop", type=int, default=0, help="GUI: 循环回放几遍(0=一遍后停住)")
p.add_argument("--realtime", action="store_true", help="GUI: 按控制频率实时播放")
p.add_argument("--prior", default="", help="GraspPose prior npz(挂上后按 apply_grasp_prior 三件套配置)")
p.add_argument("--prior_yaw", type=float, default=-1.0,
               help="钉死物体 yaw(度), 与训练口径一致; <0 自搜。不给的话回放看到的摆放"
                    "与训练不是同一个, 判读会跑偏")
p.add_argument("--eye", default="1.30,-0.75,1.45")
p.add_argument("--lookat", default="0.40,0.00,0.92")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("replay_ref")
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.scene.num_envs = 1
cfg.rsi_prob = 0.0
if args.prior:
    from tasks.pregrasp.cfg import apply_grasp_prior  # noqa: E402
    apply_grasp_prior(cfg, args.prior, yaw_deg=(args.prior_yaw if args.prior_yaw >= 0 else None))
    print(f"[replay] 挂 GraspPose prior: {args.prior} (pregrasp_align 已关)")
GUI = not args.headless
if GUI:
    from isaaclab.envs import ViewerCfg  # noqa: E402
    cfg.viewer = ViewerCfg(eye=tuple(float(v) for v in args.eye.split(",")),
                           lookat=tuple(float(v) for v in args.lookat.split(",")),
                           origin_type="world", resolution=(1600, 900))
env = GraspTaskEnv(cfg)
env.reset()

dev = env.device
eid = torch.tensor([0], device=dev)
L = int(env.L)
q_arm_ref = env.q_ref                        # (L,7) 参考臂关节(人腕轨迹 IK)
fin_ramp = env.ref_finger                    # (L,22) ← 实为脚本开→合斜坡

# ---- 人手指姿(可选) ----
fin_src, fin_note = fin_ramp, "env.ref_finger(脚本开→合斜坡)"
if args.fingers == "human":
    entry = clips.clip_entry(args.clip)
    hand = entry.get("robot_hand", entry.get("hand", "right"))
    rq = args.ref_qpos or os.path.join(os.path.dirname(entry["npz"]), f"ref_qpos_{hand}.npz")
    d = np.load(rq, allow_pickle=True)
    fq = np.asarray(d["finger_qpos"], np.float32)          # (T,22)
    names = [str(s) for s in d["joint_names"]]
    # 源帧 -> env 帧重采样(与 env 的参考同长)
    idx = np.clip(np.round(np.linspace(0, len(fq) - 1, L)).astype(int), 0, len(fq) - 1)
    fq = fq[idx]
    # 按关节名对齐到 env 的手关节顺序
    jn = list(env.hand.joint_names)
    hand_names = [jn[i] for i in env.hand_jids]
    col = {}
    for k, n in enumerate(names):
        for tgt in hand_names:
            if tgt.endswith(n.replace("right_", "").replace("left_", "")) or tgt == n:
                col[tgt] = k
    if len(col) < len(hand_names):
        print(f"[replay] ⚠ 关节名只对上 {len(col)}/{len(hand_names)}, 缺的用 ramp 顶替")
    arr = np.asarray(fin_ramp.cpu().numpy(), np.float32).copy()
    for j, tgt in enumerate(hand_names):
        if tgt in col:
            arr[:, j] = fq[:, col[tgt]]
    fin_src = torch.tensor(arr, dtype=torch.float32, device=dev)
    fin_note = f"人手 DexPilot 重定向 ({os.path.basename(rq)}, 对上 {len(col)}/{len(hand_names)} 关节)"

print(f"\n[replay] clip={args.clip} 参考 {L} 帧 | 手指来源 = {fin_note}")
print(f"[replay] 抓取起始帧 grasp_start={int(env.grasp_start)}")

sim_dt = env.sim.get_physics_dt()
dec = cfg.decimation


import time  # noqa: E402


def drive(qa, qf, render=False):
    env.hand.set_joint_position_target(qa[None], joint_ids=env.arm_jids)
    env.hand.set_joint_position_target(qf[None], joint_ids=env.hand_jids)
    for k in range(dec):
        env.scene.write_data_to_sim()
        env.sim.step(render=(render and k == dec - 1))
        env.scene.update(sim_dt)
    if render and args.realtime:
        time.sleep(max(0.0, sim_dt * dec))


def probe(t=None):
    """→ (五指接触力 N, 物体高度 m, 腕跟踪误差 m)

    腕跟踪误差 = 实际末端位置 vs 参考腕位 —— 直接回答"①手够不够得到"
    (IK 解不出/够不到时它会是大的且不收敛)。
    """
    f = torch.cat([s.data.net_forces_w[:, 0] for s in env._contact_sensors],
                  dim=0).norm(dim=-1)                       # (5,)
    obj_z = float(env.object.data.root_pos_w[0, 2])
    err = -1.0
    ee = env.hand.data.body_pos_w[0, env.ee_id] - env.scene.env_origins[0]
    if t is not None:
        err = float(torch.norm(ee - env.ref_wrist_pos[min(t, L - 1)]))
    obj = env.object.data.root_pos_w[0] - env.scene.env_origins[0]
    palm_d = float(torch.norm(ee - obj))          # 掌根到物体质心
    return f.cpu().numpy(), obj_z, err, palm_d


# ⚠ 静置前必须先把臂**瞬移**到参考第 0 帧, 不能只发目标位。
#   env.reset() 把臂放在**预抓位**(物体旁边几厘米), 而参考第 0 帧在远处;
#   若直接 drive(q_arm_ref[0]), PD 会把整条臂从物体边上猛拽回起点 ——
#   **一整条胳膊横扫过物体, 把它打飞**。2026-08-16 用户在 GUI 里看到的正是这个,
#   而它**纯属本工具的假象**:训练侧 reset 是 write_joint_state 瞬移, 不会扫。
#   (实测:同一场景零动作静置 40 步, 物体 z 稳在 93.68cm 纹丝不动。)
_q0 = env.hand.data.joint_pos.clone()
_q0[:, env.arm_jids] = q_arm_ref[0]
_q0[:, env.hand_jids] = fin_src[0]
env.hand.write_joint_state_to_sim(_q0, torch.zeros_like(_q0))
env.hand.set_joint_position_target(_q0)
env.hand.write_data_to_sim()
for _ in range(args.settle):
    drive(q_arm_ref[0], fin_src[0], render=GUI)
f0, z0, d0, p0 = probe(0)
print(f"[replay] 静置后: 物体高 {z0*100:.2f}cm  腕跟踪误差 {d0*100:.1f}cm "
      f"(臂已瞬移到参考第 0 帧, 不走扫掠)")

log = []
for t in range(L):
    drive(q_arm_ref[t], fin_src[t], render=GUI)
    f, z, d, pd = probe(t)
    log.append(dict(t=t, force=f.tolist(), z=z, d=d, pd=pd))
for k in range(args.hold):
    drive(q_arm_ref[-1], fin_src[-1], render=GUI)
    f, z, d, pd = probe(L - 1)
    log.append(dict(t=L + k, force=f.tolist(), z=z, d=d, pd=pd))

F = np.array([r["force"] for r in log])          # (steps,5)
Z = np.array([r["z"] for r in log])
D = np.array([r["d"] for r in log])
PD = np.array([r["pd"] for r in log])
gs = int(env.grasp_start)
touched = (F > 0.5).any(axis=0)
print("\n================ 零学习参考回放 · 结果 ================")
print(f"手指来源            : {fin_note}")
print(f"① 腕跟踪误差(够不够得到): 中位 {np.median(D[:L])*100:.1f}cm  最大 "
      f"{D[:L].max()*100:.1f}cm  抓取段中位 {np.median(D[gs:L])*100:.1f}cm")
print(f"①b 掌根到物体质心   : 全程最小 {PD.min()*100:.1f}cm (抓取段最小 "
      f"{PD[gs:L].min()*100:.1f}cm)  ← 物体半径 2.6cm, 掌到指尖约 10cm")
print(f"② 逐指是否接触(>0.5N): {dict(zip(['拇','食','中','无名','小'], touched.tolist()))}")
print(f"   接触力峰值(N)     : {np.round(F.max(axis=0), 2).tolist()}  "
      f"同时受力指数最多 {int(((F > 0.5).sum(axis=1)).max())}")
print(f"③ 物体高度          : 起 {Z[0]*100:.2f}cm -> 末 {Z[-1]*100:.2f}cm  "
      f"最大抬升 {(Z.max()-Z[0])*100:+.2f}cm")
verdict = ("参考自己就能抓住 ⟹ 基线可用" if (touched.sum() >= 2 and (Z.max() - Z[0]) > 0.01)
           else "参考抓不住 ⟹ 基线不可用, GraspPose 软项要挑主要担子")
print(f"判定                : **{verdict}**")
print("=======================================================\n")

# ---- 决定性诊断: **参考腕轨迹**本身到物体有多近(与策略无关) ----
obj_p = (env.object.data.root_pos_w[0] - env.scene.env_origins[0]).cpu().numpy()
rw = env.ref_wrist_pos.cpu().numpy()
dref = np.linalg.norm(rw - obj_p[None], axis=1)
kmin = int(np.argmin(dref))
print(f"④ **参考腕轨迹**到物体质心: 最近 {dref.min()*100:.1f}cm @f{kmin} "
      f"(抓取帧 gs={gs} 处 {dref[min(gs, L-1)]*100:.1f}cm)")
print(f"   物体位置(env系) {np.round(obj_p, 3).tolist()} | 参考腕范围 "
      f"x[{rw[:,0].min():.2f},{rw[:,0].max():.2f}] y[{rw[:,1].min():.2f},{rw[:,1].max():.2f}] "
      f"z[{rw[:,2].min():.2f},{rw[:,2].max():.2f}]")

if args.report:
    json.dump(dict(clip=args.clip, fingers=args.fingers, note=fin_note,
                   obj_pos=obj_p.tolist(),
                   ref_wrist_to_obj_min_cm=float(dref.min() * 100),
                   ref_wrist_to_obj_min_frame=kmin,
                   ref_wrist_to_obj_at_gs_cm=float(dref[min(gs, L - 1)] * 100),
                   wrist_err_med_cm=float(np.median(D[:L]) * 100),
                   wrist_err_max_cm=float(D[:L].max() * 100),
                   palm_obj_min_cm=float(PD.min() * 100),
                   palm_obj_min_after_grasp_cm=float(PD[gs:L].min() * 100),
                   touched=touched.tolist(), force_peak=F.max(axis=0).tolist(),
                   max_pads_together=int(((F > 0.5).sum(axis=1)).max()),
                   z_start_cm=float(Z[0] * 100), z_max_rise_cm=float((Z.max() - Z[0]) * 100),
                   verdict=verdict), open(args.report, "w"), ensure_ascii=False, indent=1)
    print(f"-> {args.report}")

if GUI and args.loop:
    print(f"[replay] GUI 循环回放 {args.loop} 遍 (关窗口即停)")
    for _ in range(args.loop):
        for t in range(L):
            drive(q_arm_ref[t], fin_src[t], render=True)
        for _ in range(20):
            drive(q_arm_ref[-1], fin_src[-1], render=True)
elif GUI:
    print("[replay] 回放结束, 窗口保持 (Ctrl-C 退出)")
    while app.is_running():
        drive(q_arm_ref[-1], fin_src[-1], render=True)

env.close()
app.close()
