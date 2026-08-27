"""Motion_Planning 运动学回放器 (GUI): 播 Approach_pour17.npz / Retreat_pour17.npz。

⚠ 运动学回放, 不跑物理: 直接写关节状态。看的是"轨迹把手带到哪儿"。
用法见同目录 view_approach.sh / view_retreat.sh。
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--file", required=True, help="Approach_pour17.npz 或 Retreat_pour17.npz")
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--prior_a", default="tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz")
p.add_argument("--yaw_a", type=float, default=19.5)
p.add_argument("--fps", type=float, default=30.0)
p.add_argument("--start_frame", type=int, default=0,
               help="从这一帧起播 (如 81=跳过cuRobo段, 直接从成形梯起点定位开演)")
p.add_argument("--selftest", type=int, default=0,
               help=">0: 只放这么多帧后退出 (headless 自检)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("view_motion")
app = AppLauncher(args).app

import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.approach_only, cfg.action_space = True, 7
apply_grasp_prior(cfg, args.prior_a, args.yaw_a, approach=True)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()

z = np.load(args.file, allow_pickle=True)
jn = list(E.hand.joint_names)
arm_ids = {s: [jn.index(f"{P}_arm_j{i}") for i in range(1, 8)]
           for s, P in (("right", "R"), ("left", "L"))}
fin_names = [str(n) for n in z["fin_names"]]
fin_ids = {s: [jn.index(n.replace("right_", f"{s}_")) for n in fin_names]
           for s in ("right", "left")}
T = len(z["right_q"])
print(f"[view] {args.file} | {T} 行 | 段: "
      + " + ".join(f"{n}({l})" for n, l in zip(z["seg_names"], z["seg_lens"])))

q = E.hand.data.default_joint_pos.clone()
# 先静置 90 步让物体落桌沉降 (出生位=贴桌+2mm 悬空, 不沉降的话每圈开头都晃一次)
for _ in range(90):
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())
# 每遍循环要重置的物体状态 = **沉降后**的稳态
_objs0 = []
for _ob in [E.object] + ([E.aux] if getattr(E, "aux", None) is not None else []):
    _objs0.append((_ob, _ob.data.root_state_w.clone()))
dt = 1.0 / max(args.fps, 1.0)
_f0 = max(0, min(int(args.start_frame), T - 1))
frames = range(_f0, min(T, _f0 + args.selftest) if args.selftest > 0 else T)
def _pose_frame(t=None):
    t = _f0 if t is None else t
    for s in ("right", "left"):
        q[:, arm_ids[s]] = torch.tensor(z[f"{s}_q"][t], dtype=torch.float32,
                                        device=E.device)
        q[:, fin_ids[s]] = torch.tensor(z[f"{s}_f"][t], dtype=torch.float32,
                                        device=E.device)
    E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))

try:
    while True:
        # 循环重置: 手先回 f0 -> 物体归位 -> 静置10步吸收瞬移冲击, 再开播
        _pose_frame(_f0)
        for _ob, _st in _objs0:
            _ob.write_root_pose_to_sim(_st[:, :7])
            _ob.write_root_velocity_to_sim(torch.zeros_like(_st[:, 7:]))
        for _ in range(10):
            E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
            for _ob, _st in _objs0:                   # 定身: 吸收循环同样钳物体
                _ob.write_root_pose_to_sim(_st[:, :7])
                _ob.write_root_velocity_to_sim(torch.zeros_like(_st[:, 7:]))
            E.scene.write_data_to_sim()
            E.sim.step(render=False)
            E.scene.update(E.sim.get_physics_dt())
        if not args.selftest:
            import select
            import sys
            print("[view] 已归位定格 —— 按 Enter 播放", flush=True)
            while True:
                E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
                for _ob, _st in _objs0:               # 定身: 等Enter循环同样钳物体
                    _ob.write_root_pose_to_sim(_st[:, :7])
                    _ob.write_root_velocity_to_sim(torch.zeros_like(_st[:, 7:]))
                E.hand.write_data_to_sim()
                E.sim.step(render=True)
                _r, _, _ = select.select([sys.stdin], [], [], 0.0)
                if _r:
                    sys.stdin.readline()
                    break
        for t in frames:
            for s in ("right", "left"):
                q[:, arm_ids[s]] = torch.tensor(z[f"{s}_q"][t], dtype=torch.float32,
                                                device=E.device)
                q[:, fin_ids[s]] = torch.tensor(z[f"{s}_f"][t], dtype=torch.float32,
                                                device=E.device)
            E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
            E.hand.write_data_to_sim()
            # 编舞语义: 交互前物体定身 —— 每帧钳回沉降稳态, 瞬擦不再累积残转;
            # 且穿模会以网格重叠直接可见, 不被"物体让位"掩盖 (2026-08-27)
            for _ob, _st in _objs0:
                _ob.write_root_pose_to_sim(_st[:, :7])
                _ob.write_root_velocity_to_sim(torch.zeros_like(_st[:, 7:]))
            E.sim.step(render=not args.selftest)
            if not args.selftest:
                time.sleep(dt)
        if args.selftest:
            print("[view] selftest 完成, 退出")
            break
        print("[view] 一遍放完 (Ctrl-C 退出)")
        if not app.is_running():
            break
except KeyboardInterrupt:
    pass
try:
    _slot.release()
except Exception:
    pass
app.close()
raise SystemExit(0)
