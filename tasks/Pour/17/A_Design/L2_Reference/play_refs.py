"""多母带回放器 (2026-09-02): 一个 Isaac 窗口里按 Enter 逐条播放多条 pour17 母带。
直接从母带行驱动 机器人关节 + 双物体位姿 (母带已含解好的 q 与物体轨迹, 无需 IK)。
按 Enter 播放当前条, 播完回定格, 再按 Enter 进下一条 (循环)。Ctrl-C 退出。

用法 (GUI, 不加 --headless):
  cd /home/lyh/Project/RL_Correction
  SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=. /home/lyh/luhr/MagicSim/.venv/bin/python -u \
    tasks/Pour/17/A_Design/L2_Reference/play_refs.py
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--refs", nargs="+", default=[
    "tasks/Pour/17/A_Design/L2_Reference/pour17_reference_v3.npz",       # pos0 原位
    "tasks/Pour/17/A_Design/L2_Reference/pos_variants/pour17_pos1.npz",  # pos1 右前
    "tasks/Pour/17/A_Design/L2_Reference/pos_variants/pour17_pos2.npz",  # pos2 左后
], help="按顺序播放的母带 npz 列表")
p.add_argument("--labels", nargs="+", default=["pos0 原位", "pos1 右前", "pos2 左后"])
p.add_argument("--fps", type=float, default=15.0)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("play_refs")
app = AppLauncher(args).app

import select  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
cfg.approach_only, cfg.action_space = True, 7
apply_grasp_prior(cfg, "tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz", 19.5, approach=True)
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()
for _ in range(60):
    E.scene.write_data_to_sim(); E.sim.step(render=False); E.scene.update(E.sim.get_physics_dt())
W = E.scene.env_origins[0].cpu().numpy().astype(np.float64)
jn = list(E.hand.joint_names)
dev = E.device

# 载入所有母带
TAPES = []
for pth, lab in zip(args.refs, args.labels):
    z = np.load(pth, allow_pickle=True)
    fin = [str(n) for n in z["fin_names"]]
    TAPES.append(dict(lab=lab, z=z, fin=fin, T=len(z["right_q"])))
    print(f"[play] 载入 {lab}: {len(z['right_q'])}行  {pth}", flush=True)


def set_row(z, fin, t):
    q = E.hand.data.default_joint_pos.clone()
    for s_, P_ in (("right", "R"), ("left", "L")):
        for i_ in range(7):
            q[:, jn.index(f"{P_}_arm_j{i_+1}")] = float(z[f"{s_}_q"][t, i_])
        for n_, v_ in zip(fin, z[f"{s_}_f"][t]):
            q[0, jn.index(n_.replace("right_", f"{s_}_"))] = float(v_)
    E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
    E.hand.write_data_to_sim()
    for oi, ob in ((1, E.object), (0, E.aux)):
        pose = np.concatenate([z[f"obj_pos_{oi}"][t] + W, z[f"obj_quat_{oi}"][t]])
        st = torch.tensor(pose, dtype=torch.float32, device=dev).unsqueeze(0)
        ob.write_root_pose_to_sim(st)
        ob.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))


def standby(tape):
    set_row(tape["z"], tape["fin"], 0)
    E.sim.step(render=True)


dt = 1.0 / max(args.fps, 1.0)
idx = 0
try:
    while True:
        tape = TAPES[idx]
        # 定格首帧, 等 Enter
        while True:
            standby(tape)
            print(f"[play] ▶ 待播: [{idx+1}/{len(TAPES)}] {tape['lab']} "
                  f"({tape['T']}行) —— 按 Enter 播放 (Ctrl-C 退出)", flush=True)
            got = False
            for _ in range(1000000):
                standby(tape)
                r, _, _ = select.select([sys.stdin], [], [], 0.0)
                if r:
                    sys.stdin.readline(); got = True; break
                if not app.is_running():
                    break
            if got or not app.is_running():
                break
        if not app.is_running():
            break
        # 播放全程
        print(f"[play] ● 播放 {tape['lab']} ...", flush=True)
        for t in range(tape["T"]):
            set_row(tape["z"], tape["fin"], t)
            E.sim.step(render=True)
            time.sleep(dt)
            if not app.is_running():
                break
        print(f"[play] ✔ {tape['lab']} 播完 —— 按 Enter 进入下一条", flush=True)
        idx = (idx + 1) % len(TAPES)
except KeyboardInterrupt:
    pass
try:
    _slot.release()
except Exception:
    pass
app.close()
raise SystemExit(0)
