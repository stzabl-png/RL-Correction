"""人手参考轨迹 vs Dexonomy GraspPose: 它们到底有多接近。

背景: 摆放链里只有"物体 XY(相机锚定+物体听手)"和"物体 yaw(Gate 1, 与视频一致)"来自视频;
`canon_rot` 与 **GraspPose 本身**都是 Dexonomy 在物体网格上算的, 与这次录像无关。
所以"人手轨迹的接近是不是接近 GraspPose"必须实测, 不能假定。

逐帧报: 人手参考腕位/朝向 与 GraspPose 的差, 以及最接近的那一帧在哪。

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.diag_ref_vs_prior --headless \\
        --clip Pour17_bottle --grasp_prior tasks/pregrasp/priors/Pour17_bottle.npz \\
        --prior_yaw 19.5 --stance_prefix 60
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--stance_prefix", type=int, default=60)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("diag_rvp")
app = AppLauncher(args).app

import numpy as np  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402

from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.stance_prefix_frames = args.stance_prefix
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
E = GraspTaskEnv(cfg)
E.reset()

gp = E._grasp_pos_w.cpu().numpy().astype(np.float64)
gq = E._grasp_quat_w.cpu().numpy().astype(np.float64)
W = E.ref_wrist_pos.cpu().numpy().astype(np.float64)
Q = E.ref_wrist_quat.cpu().numpy().astype(np.float64)
obj = E.obj_init_pos.cpu().numpy().astype(np.float64)
gs = int(E.grasp_start)
K = int(args.stance_prefix)


def ang(q1, q2):
    """两个四元数的夹角 (度), 取同半球。"""
    d = abs(float(np.dot(q1, q2)))
    return float(np.degrees(2.0 * np.arccos(np.clip(d, 0.0, 1.0))))


dp = np.linalg.norm(W - gp, axis=1)
da = np.array([ang(q, gq) for q in Q])
k = int(dp.argmin())

print(f"\n{'=' * 78}")
print(f"[人手参考 vs GraspPose] clip={args.clip} | 站姿前缀 K={K} | 交互开始帧 gs={gs}")
print(f"  GraspPose 腕位 {np.round(gp * 100, 1).tolist()}cm | 物体 {np.round(obj * 100, 1).tolist()}cm")
print(f"{'=' * 78}")
print(f"{'帧':>5} {'说明':<14} {'腕位差(cm)':>11} {'朝向差(°)':>10} {'腕->物体(cm)':>13}")
marks = {0: "站姿", K: "人手轨迹起点", gs: "交互开始帧(gs)"}
for t in sorted(set(list(range(0, len(W), max(len(W) // 10, 1))) + list(marks))):
    if t >= len(W):
        continue
    print(f"{t:>5} {marks.get(t, ''):<14} {dp[t] * 100:>11.2f} {da[t]:>10.1f} "
          f"{np.linalg.norm(W[t] - obj) * 100:>13.2f}")
print("-" * 78)
print(f"  ★ 人手轨迹**最接近** GraspPose 的一帧: 第 {k} 帧 "
      f"(腕位差 {dp[k] * 100:.2f}cm, 朝向差 {da[k]:.1f}°)")
print(f"  ★ 在交互开始帧 gs={gs}: 腕位差 {dp[gs] * 100:.2f}cm, 朝向差 {da[gs]:.1f}°")
print(f"\n  判读: 差 <3cm/<15° ⟹ 人手接近确实指向 GraspPose, 可当接近段参考;")
print(f"        差很大 ⟹ 两者是**不同的抓法**, 拿人手轨迹当接近参考会把手带到别处。")
print(f"{'=' * 78}\n")
app.close()
