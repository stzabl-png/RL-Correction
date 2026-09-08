"""GUI 播放 pour25 approach 段 (Approach_pour25.npz): 双手从张开站姿同时伸手 -> 合拢 GraspPose。
物体固定在 scene_layout 静置位 (approach 段物体不动)。按 Enter 播放/重播, Ctrl-C 退出。
用法(自己终端, 需显示):
  cd /home/lyh/Project/RL_Correction
  SHARPA_WANDB=0 PYTHONPATH=. /home/lyh/luhr/MagicSim/.venv/bin/python \
    tasks/Pour/25/A_Design/L1_Data/Motion_Planning/view_approach.py
"""
import argparse, select, sys, time
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--approach", default="tasks/Pour/25/A_Design/L1_Data/Motion_Planning/Approach_pour25.npz")
p.add_argument("--clip", default="Pour25_bottle")
p.add_argument("--prior_a", default="tasks/pregrasp/priors/Pour25_bottle_thumbfix.npz")
p.add_argument("--yaw_a", type=float, default=90.0)
p.add_argument("--fps", type=float, default=15.0)
p.add_argument("--max_frame", type=int, default=0,
               help="只播到此帧 (含) 后定格, 0=播全程。用于分级检查成形梯。")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
sys.path.insert(0, "/home/lyh/Project/RL_Correction")
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
for _ in range(30):
    E.scene.write_data_to_sim(); E.sim.step(render=False); E.scene.update(E.sim.get_physics_dt())
jn = list(E.hand.joint_names)
dev = E.device

# 物体静置位 (env-init, approach 段保持不动)
prim_p = E.object.data.root_pos_w[0].clone()
prim_q = E.object.data.root_quat_w[0].clone()
aux_p = E.aux.data.root_pos_w[0].clone()
aux_q = E.aux.data.root_quat_w[0].clone()

z = np.load(args.approach, allow_pickle=True)
fin = [str(n) for n in z["fin_names"]]
T = len(z["right_q"])
seg = list(zip([str(x) for x in z["seg_names"]], z["seg_lens"].tolist()))
print(f"[view] Approach {T} 行 | 段 {seg} | 右手=瓶 左手=杯", flush=True)


def hold_objs():
    for ob, pp, qq in ((E.object, prim_p, prim_q), (E.aux, aux_p, aux_q)):
        ob.write_root_pose_to_sim(torch.cat([pp, qq]).unsqueeze(0))
        ob.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))


def set_row(t):
    q = E.hand.data.default_joint_pos.clone()
    for s_, P_ in (("right", "R"), ("left", "L")):
        for i_ in range(7):
            q[:, jn.index(f"{P_}_arm_j{i_+1}")] = float(z[f"{s_}_q"][t, i_])
        for n_, v_ in zip(fin, z[f"{s_}_f"][t]):
            q[0, jn.index(n_.replace("right_", f"{s_}_"))] = float(v_)
    E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
    E.hand.write_data_to_sim()
    hold_objs()


dt = 1.0 / max(args.fps, 1.0)
try:
    while True:
        set_row(0); E.sim.step(render=True)
        print(f"[view] ▶ 待播 approach ({T}行, 双手同时伸手) —— 按 Enter 播放 (Ctrl-C 退出)",
              flush=True)
        got = False
        for _ in range(10**7):
            set_row(0); E.sim.step(render=True)
            r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if r:
                sys.stdin.readline(); got = True; break
            if not app.is_running():
                break
        if not app.is_running():
            break
        _end = T if args.max_frame <= 0 else min(args.max_frame + 1, T)
        print(f"[view] ● 播放 0..{_end-1} ...", flush=True)
        for t in range(_end):
            set_row(t); E.sim.step(render=True); time.sleep(dt)
            if not app.is_running():
                break
        # 定格末帧, 便于检查该级梯位形
        for _ in range(30):
            set_row(_end - 1); E.sim.step(render=True); time.sleep(dt)
            if not app.is_running():
                break
        print(f"[view] ✔ 播到帧{_end-1}并定格 —— 按 Enter 重播", flush=True)
except KeyboardInterrupt:
    pass
app.close()
raise SystemExit(0)
