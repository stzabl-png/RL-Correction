"""纯物体轨迹回放 (2026-08-31 用户要求): **只看物体跟着母带的物体轨迹动**。

★ 与已有两个查看器的区别 (为什么要新写一个):
  · `view_objtrack_ik.py`  —— 机器人按物体轨迹解 IK 去驱动, 物体由**物理**携带。
    看的是"机器人跟不跟得上", 不是"轨迹本身长什么样"。
  · `L2_Reference/view_reference.py` —— 读 `replay_world.npz`(原始重建, 无 conf 无 RTS),
    且双手定格在 GraspPose。读的**不是**我们训练用的母带。
  本脚本: 读**训练真正用的母带 npz**, 逐行把两个物体的位姿**运动学**写进去,
  不解 IK、不靠物理携带。看到的就是 `PB.ref_obj` 本身。

★ 复用 `PourEnv` 建场景是**有意的**: 资产、桌高、env 原点、物体静置位全部与训练
  一字不差。本项目自己搭场景踩过坐标系的坑 (见 docs/FRAME_ALIGNMENT.md)。

同时在控制台逐行打印判据关心的量 —— 让眼睛看到的和台账里的数字能对上:
    倾角 · 瓶口↔杯口 水平距/dz/3D距 · 该行的置信档位 · 是否满足 G3_pour

用法:
  cd /home/lyh/Project/RL_Correction
  SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=. $PY \
    tasks/Pour/17/B_SmokeTest/view_objtraj_tape.py \
    --ref tasks/Pour/17/A_Design/L2_Reference/pour17_reference_v3.npz --loop
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--ref", default="tasks/Pour/17/A_Design/L2_Reference/"
                                "pour17_reference_v3.npz")
p.add_argument("--fps", type=float, default=20.0, help="回放帧率")
p.add_argument("--loop", action="store_true", help="循环播放")
p.add_argument("--row0", type=int, default=-1, help="起始行 (-1=交互段起点)")
p.add_argument("--row1", type=int, default=-1, help="结束行 (-1=交互段终点)")
p.add_argument("--all", action="store_true", help="放全链 653 行 (含 Approach/Retreat)")
p.add_argument("--robot_away", action="store_true", help="把机器人挪开, 画面只剩物体")
p.add_argument("--every", type=int, default=5, help="每几行打印一次读数")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

os.environ["POUR_REF_NPZ"] = os.path.abspath(args.ref)
os.environ.setdefault("POUR_SQUEEZE_FF", "1")
app = AppLauncher(args).app

import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "C_Wiring"))
import pour_env as PE  # noqa: E402
sys.path.insert(0, os.path.join(_HERE, "..", "A_Design", "L3_Learning"))
import progress_batch as PBM  # noqa: E402
from progress import (M2_TILT, M2_STRICT_HORIZ,  # noqa: E402
                      M2_STRICT_DZ_LO, M2_STRICT_DZ_HI)

cfg = PE.build_cfg(num_envs=1)
E = PE.PourEnv(cfg)
E.force_entry = [0]
E.reset()
dev = E.device
PB = E.PB
org = E.scene.env_origins

if args.robot_away:
    # 只想看物体 —— 把机器人底座挪到 5m 外(不影响物体, 它们是运动学写入的)
    rp = E.hand.data.root_state_w[:, :7].clone()
    rp[:, 0] -= 5.0
    E.hand.write_root_pose_to_sim(rp)
    print("[view] 机器人已挪开 5m —— 画面只剩物体", flush=True)

N_ROW = PB.N_ROW
r0 = 0 if args.all else (args.row0 if args.row0 >= 0 else 0)
r1 = (E.T_ROW - 1) if args.all else (args.row1 if args.row1 >= 0 else N_ROW - 1)
print(f"[view] 母带 {os.path.basename(args.ref)} · 交互段 {N_ROW} 行 · "
      f"全链 {E.T_ROW} 行 · 本次放 {'全链' if args.all else '交互段'} 行 {r0}~{r1}",
      flush=True)
print(f"[view] G3_pour 判据: 倾角>={np.degrees(M2_TILT):.0f}° 且 水平<="
      f"{M2_STRICT_HORIZ*100:.0f}cm 且 dz∈[{M2_STRICT_DZ_LO*100:.0f},"
      f"{M2_STRICT_DZ_HI*100:.0f}]cm", flush=True)
print(f"[view] {'行':>5s} {'档':>3s} {'倾角°':>7s} {'水平cm':>7s} {'dz cm':>7s} "
      f"{'3D cm':>7s}  G3_pour", flush=True)

dt = 1.0 / max(args.fps, 1e-3)
UPB = PB.up.cpu().numpy().astype(np.float64)


def q2R(q):
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


MB = PB.mb.cpu().numpy().astype(np.float64)
MC = PB.mc.cpu().numpy().astype(np.float64)
UPC = PB.upc.cpu().numpy().astype(np.float64) if hasattr(PB, "upc") else UPB

while True:
    for r in range(r0, r1 + 1):
        ia = r if not args.all else max(0, min(N_ROW - 1, r - E.IA0))
        for oi, art in ((0, E.aux), (1, E.object)):
            pose = PB.ref_obj[oi][ia].clone().unsqueeze(0)
            pose[:, :3] += org
            art.write_root_pose_to_sim(pose)
            art.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        E.scene.write_data_to_sim()
        E.sim.step(render=True)
        E.scene.update(E.sim.get_physics_dt())
        if (r - r0) % args.every == 0:
            b = PB.ref_obj[1][ia].cpu().numpy().astype(np.float64)
            c = PB.ref_obj[0][ia].cpu().numpy().astype(np.float64)
            R1 = q2R(b[3:7])
            v = R1 @ UPB
            tilt = np.degrees(np.arccos(np.clip(v[2] / np.linalg.norm(v), -1, 1)))
            d = (b[:3] + R1 @ MB) - (c[:3] + q2R(c[3:7]) @ MC)
            hz, dz = float(np.linalg.norm(d[:2])), float(d[2])
            ok = (tilt >= np.degrees(M2_TILT) and hz <= M2_STRICT_HORIZ
                  and M2_STRICT_DZ_LO <= dz <= M2_STRICT_DZ_HI)
            t = int(PB.tmix[ia])
            print(f"[view] {r:5d} {'绿黄红'[2-t]:>3s} {tilt:7.1f} {hz*100:7.2f} "
                  f"{dz*100:+7.2f} {np.linalg.norm(d)*100:7.2f}  "
                  f"{'✅' if ok else '·'}", flush=True)
        time.sleep(dt)
    if not args.loop:
        break
    print("[view] ——— 循环 ———", flush=True)
app.close()
