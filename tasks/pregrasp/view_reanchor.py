"""GUI 检查参考轨迹重锚定 (2026-08-07): 训练用什么构建路径, 这里就走什么.

与 view_ref 的区别: 那个用基类 env (不触发 resting_pose 覆盖/重锚定); 这个建
**GraspTaskEnv + approach 全套** —— resting_pose 覆盖、参考轨迹重锚定、站姿
前缀、prior IK 全部按训练路径生效, 看到的就是修正后训练拿到的东西.

  SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.view_reanchor \
      --clip Screw27_body \
      --grasp_prior tasks/pregrasp/priors/Screw27_body_candidates/35_8_mid_p12.npz \
      --prior_yaw 224 --stance_prefix 60 --play

青线=重锚定后的参考腕轨迹  青球=物体(resting_pose)  黄球=PreGrasp 腕位
--play 时机器人沿 q_ref 从站姿扫到 PreGrasp (只 render 不 step 物理).
核对项: ① 青线终点应落在瓶身旁 (旧 bug 是偏 40cm); ② 手↔物 XY 相对关系
与视频一致 (XY 只随物体整体平移/绕竖轴转, Z 贴桌适配).
"""
import argparse
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Screw27_body")
p.add_argument("--grasp_prior", required=True)
p.add_argument("--prior_yaw", type=float, default=-1.0)
p.add_argument("--stance_prefix", type=int, default=60)
p.add_argument("--t", type=int, default=None, help="停在哪帧 (默认 PreGrasp 帧)")
p.add_argument("--play", action="store_true")
p.add_argument("--fps", type=float, default=20.0)
p.add_argument("--eye", default="0.55,-0.85,1.45")
p.add_argument("--lookat", default="-0.10,-0.08,0.92")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("view_reanchor")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GraspTaskEnv  # noqa: E402

cfg = GraspTaskCfg()
clips.configure_cfg(cfg, args.clip)
apply_grasp_prior(cfg, args.grasp_prior, args.prior_yaw, approach=True)
cfg.stance_prefix_frames = args.stance_prefix
cfg.direct_grasp_prob = 0.0
cfg.scene.num_envs = 1
cfg.obj_jitter_xy = 0.0
cfg.closure_init_max = 0.0
cfg.viewer = ViewerCfg(eye=tuple(float(v) for v in args.eye.split(",")),
                       lookat=tuple(float(v) for v in args.lookat.split(",")),
                       origin_type="world", resolution=(1600, 900))
E = GraspTaskEnv(cfg)
E.reset()

W = E.scene.env_origins[0].cpu().numpy()
L, gs = int(E.L), int(E.grasp_start)
t0 = gs if args.t is None else int(np.clip(args.t, 0, L - 1))
wp = E.ref_wrist_pos.cpu().numpy()
obj = E.obj_init_pos.cpu().numpy()

E._draw_curve("/World/RefTraj", wp + W, (0.1, 0.85, 0.9), 0.004)
E._draw_sphere("/World/MarkObj", obj + W, (0.1, 0.85, 0.9), 0.020)

# 手指钉在复位手型 (模板 pregrasp): 训练里接近段手指由相位机保持 pregrasp 手型,
# 不回放人手手指参考 (人手在视频里交互开始得早, 回放会出现"边飞边合拢"的伪影)
_finger_hold = E.hand.data.joint_pos[:, E.hand_jids].clone()


def pose(t):
    q = E.hand.data.joint_pos.clone()
    q[:, E.arm_jids] = E.q_ref[t]
    q[:, E.hand_jids] = _finger_hold
    E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
    E.hand.set_joint_position_target(q)
    E.hand.write_data_to_sim()


pose(t0)
gap = float(np.linalg.norm(wp[gs][:2] - obj[:2])) * 100
print("\n" + "=" * 74)
print(f"clip={args.clip}  L={L}  站姿前缀={args.stance_prefix}  交互开始帧 gs={gs}")
print(f"物体 (XY=交互帧手锚点, 姿态/Z=resting_pose) {np.round(obj, 4)}")
print(f"腕参考 交互开始帧 ref[{gs}] {np.round(wp[gs], 4)}")
print(f"腕参考交互帧 -> 物体 水平距离 {gap:.1f}cm   (物体听手约定: 应为抓取半径量级 <10cm)")
print("=" * 74)
print("[view_reanchor] 青线=参考腕轨迹 青球=物体 | 关窗口退出"
      + ("  |  播放中" if args.play else ""))

t, dt = t0, 1.0 / max(args.fps, 1e-3)
_last = time.time()
while app.is_running():
    if args.play and time.time() - _last >= dt:
        _last = time.time()
        t = (t + 1) % (gs + 1)     # 站姿 -> PreGrasp 循环 (抓取段不看)
        pose(t)
    E.sim.render()
E.close()
