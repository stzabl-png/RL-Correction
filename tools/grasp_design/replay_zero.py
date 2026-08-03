"""零动作开环回放: 把设计的关节轨迹直接灌进 env, 看物体动不动.

绕开 _pre_physics_step 的参考/残差机制 —— 这条轨迹是**绝对关节轨迹**, 不是残差.
只驱动右臂 7 关节 + 右手 22 关节, 躯干/左臂保持 env reset 后的目标不变.

  $PY /tmp/.../replay_zero.py --headless
"""
import argparse
import numpy as np

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp2")
p.add_argument("--settle", type=int, default=40, help="放物体后先静置多少控制步")
p.add_argument("--hold", type=int, default=40, help="轨迹跑完后再保持多少控制步")
p.add_argument("--loop", type=int, default=0, help="GUI: 循环回放多少遍 (0=一遍就停在窗口里)")
p.add_argument("--realtime", action="store_true", help="GUI: 按 20Hz 实时节拍播放")
p.add_argument("--mark", action="store_true",
               help="GUI: 画一个跟随物体的标记球 (扁圆环在 GUI 里可能看不见, 见 HANDOFF §2.3)")
p.add_argument("--eye", default="0.55,-0.85,1.45")
p.add_argument("--lookat", default="-0.02,-0.12,0.95")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("replay0")
app = AppLauncher(args).app

import torch  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.env.dexmate_env import DexmateCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402
import time  # noqa: E402

GUI = not args.headless

SP = "/home/lyh/Project/RL_Correction/tools/grasp_design"
T = np.load(f"{SP}/full_traj.npz", allow_pickle=True)
import os
G = np.load(f"{SP}/{os.environ.get('GRIP_NPZ','grip_cfg_v6.npz')}", allow_pickle=True)
ARM = T["arm"]                       # (F,14) [R_arm_j1..7, L_arm_j1..7]
FIN = T["finger"]                    # (F,22)
AJN = [str(s) for s in T["arm_joints"]]
FJN = [str(s) for s in T["finger_joints"]]
PHASE = [str(s) for s in T["phase"]]
dt_traj = float(T["dt"])
OBJ_DES = G["obj"]
WRIST_DES = G["wrist_T"][:3, 3]

cfg = DexmateCorrectionEnvCfg()
clips.configure_cfg(cfg, args.clip)
cfg.anchor_mode = "camera"
cfg.scene.num_envs = 1
cfg.rsi_prob = 0.0
if GUI:
    cfg.viewer = ViewerCfg(eye=tuple(float(v) for v in args.eye.split(",")),
                           lookat=tuple(float(v) for v in args.lookat.split(",")),
                           origin_type="world", resolution=(1600, 900))
env = DexmateCorrectionEnv(cfg)
env.reset()

dec = cfg.decimation
sim_dt = env.sim.get_physics_dt()
ctrl_hz = 1.0 / (sim_dt * dec)
step_ratio = max(int(round((1.0 / dt_traj) / ctrl_hz)), 1)
IDX = list(range(0, len(ARM), step_ratio))
print(f"\n[replay] 轨迹 {len(ARM)} 帧 @{1/dt_traj:.0f}Hz -> env 控制 {ctrl_hz:.0f}Hz, "
      f"取每 {step_ratio} 帧 = {len(IDX)} 控制步 ({len(IDX)/ctrl_hz:.2f}s)")

jn = list(env.hand.joint_names)
arm_ids = [jn.index(n) for n in AJN if n.startswith("R_arm")]
arm_cols = [i for i, n in enumerate(AJN) if n.startswith("R_arm")]
fin_ids = [jn.index(n) for n in FJN]
print(f"[replay] 右臂 {len(arm_ids)} 关节 + 右手 {len(fin_ids)} 关节 (按名字对齐)")

dev = env.device
org = env.scene.env_origins[0]
eid = torch.tensor([0], device=dev)


def put_object():
    q = env.ref_obj_quat[0]
    pose = torch.cat([torch.tensor(OBJ_DES, device=dev, dtype=torch.float32) + org, q])[None]
    env.object.write_root_pose_to_sim(pose, eid)
    env.object.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev), eid)


def set_robot(q_arm, q_fin, teleport=False):
    a = torch.tensor(q_arm, device=dev, dtype=torch.float32)[None]
    f = torch.tensor(q_fin, device=dev, dtype=torch.float32)[None]
    if teleport:
        env.hand.write_joint_state_to_sim(a, torch.zeros_like(a), joint_ids=arm_ids)
        env.hand.write_joint_state_to_sim(f, torch.zeros_like(f), joint_ids=fin_ids)
    env.hand.set_joint_position_target(a, joint_ids=arm_ids)
    env.hand.set_joint_position_target(f, joint_ids=fin_ids)


def phys(n=1, render=False):
    for _ in range(n):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(sim_dt)
    if render and GUI:
        if args.mark:
            try:
                env._draw_sphere("/World/Mark_Obj",
                                 env.object.data.root_pos_w[0].cpu().numpy(),
                                 (0.1, 0.9, 0.2), 0.045)
            except Exception:
                pass
        env.sim.render()


def obj_pos():
    return (env.object.data.root_pos_w[0] - org).cpu().numpy()


def hand_pos():
    bn = list(env.hand.body_names)
    return (env.hand.data.body_pos_w[0, bn.index(f"{cfg.hand_side}_hand_C_MC")] - org).cpu().numpy()


# ---- 起始: 机器人瞬移到轨迹首帧, 物体放到设计位 ----
set_robot(ARM[0][arm_cols], FIN[0], teleport=True)
put_object()
phys(dec)
put_object()
for _ in range(args.settle):
    phys(dec)

TZ = cfg.table_top_z
o0 = obj_pos()
h0 = hand_pos()
print(f"\n[静置 {args.settle} 步后]")
print(f"  物体 {np.round(o0,4).tolist()}  (设计位 {np.round(OBJ_DES,4).tolist()}, "
      f"差 {np.linalg.norm(o0-OBJ_DES)*100:.2f}cm)")
print(f"  手基座 {np.round(h0,4).tolist()}  (轨迹首帧规划位 —— 见下面逐步对比)")

# ---- 回放 ----
def run_once():
    out = []
    tick = 1.0 / ctrl_hz
    for i in IDX:
        t0 = time.time()
        set_robot(ARM[i][arm_cols], FIN[i])
        phys(dec, render=True)
        out.append((PHASE[i], obj_pos().copy(), hand_pos().copy()))
        if args.realtime and GUI:
            d = tick - (time.time() - t0)
            if d > 0:
                time.sleep(d)
    for _ in range(args.hold):
        phys(dec, render=True)
        out.append(("HOLD", obj_pos().copy(), hand_pos().copy()))
    return out


def rewind():
    """回到起始: 机器人瞬移回首帧, 物体放回设计位."""
    set_robot(ARM[0][arm_cols], FIN[0], teleport=True)
    put_object()
    phys(dec, render=True)
    put_object()
    for _ in range(10):
        phys(dec, render=True)


log = run_once()
for r in range(args.loop):
    if GUI and not app.is_running():
        break
    print(f"[replay] 第 {r+2} 遍…", flush=True)
    rewind()
    run_once()

O = np.array([l[1] for l in log])
H = np.array([l[2] for l in log])
ph = np.array([l[0] for l in log])

print(f"\n{'='*76}\n零动作开环回放结果 ({len(log)} 控制步)\n{'='*76}")
print(f"物体起点 {np.round(o0,4).tolist()}   终点 {np.round(O[-1],4).tolist()}")
print(f"  净位移 {np.linalg.norm(O[-1]-o0)*100:.2f}cm   "
      f"最大离起点 {np.linalg.norm(O-o0,axis=1).max()*100:.2f}cm")
print(f"  最高离桌 {(O[:,2].max()-TZ)*100:+.2f}cm   末了离桌 {(O[-1,2]-TZ)*100:+.2f}cm")
print(f"  水平净移 {np.linalg.norm(O[-1,:2]-o0[:2])*100:.2f}cm  "
      f"(设计目标 20.00cm)")
# 三条判据 (pick -> transport -> place), 缺一不可:
lifted = (O[:, 2] - TZ) > 0.05
h0_z, hmax = o0[2] - TZ, (O[:, 2].max() - TZ)
placed = abs(O[-1, 2] - o0[2]) < 0.02          # 末了回到起始离桌高度 = 稳稳放在桌上
moved = np.linalg.norm(O[-1, :2] - o0[:2])
print(f"  ① 抓起 (离桌>5cm 的步数 {int(lifted.sum())}/{len(O)}, 最高 {hmax*100:.2f}cm) "
      f"{'✅' if lifted.any() else '❌ 没抓起来'}")
print(f"  ② 搬运 (水平净移 {moved*100:.2f}cm / 目标 20cm) "
      f"{'✅' if moved > 0.15 else '❌ 没搬到位'}")
print(f"  ③ 放下 (末了离桌 {(O[-1,2]-TZ)*100:+.2f}cm vs 起始 {h0_z*100:+.2f}cm) "
      f"{'✅' if placed else '❌ 没放稳/掉了'}")

print(f"\n{'阶段':<14}{'步':>5}{'物体离桌cm':>12}{'物体离起点cm':>14}{'手基座-规划 差cm':>18}")
for tag in dict.fromkeys(ph):
    m = ph == tag
    j = np.flatnonzero(m)[-1]
    src = IDX[min(j, len(IDX) - 1)] if tag != "HOLD" else len(ARM) - 1
    print(f"{tag:<14}{int(m.sum()):>5}{(O[j,2]-TZ)*100:>12.2f}"
          f"{np.linalg.norm(O[j]-o0)*100:>14.2f}"
          f"{np.linalg.norm(H[j]-T['hand_base_pos'][src])*100:>18.2f}")

# 手臂开环跟踪误差 (规划 vs 实测)
err = np.array([np.linalg.norm(H[k] - T["hand_base_pos"][IDX[k]])
                for k in range(len(IDX))])
print(f"\n[手臂开环跟踪] 手基座 实测 vs 规划: 中位 {np.median(err)*100:.2f}cm  "
      f"最大 {err.max()*100:.2f}cm")
np.savez(f"{SP}/replay_log.npz", obj=O, hand=H, phase=ph, obj0=o0)
print(f"-> {SP}/replay_log.npz")
if GUI:
    print("\n[replay] 回放结束, 窗口保持中 —— 关窗口或 Ctrl-C 退出")
    while app.is_running():
        env.sim.render()
env.close()
