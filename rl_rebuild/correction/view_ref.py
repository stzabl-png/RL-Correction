"""看 ref builder 实际产出的那个 env —— 训练拿到什么, 这里就显示什么.

与 view_placement 的区别: 那个是"旧摆放 vs 新摆放"的对照工具, 会强制建成 palm 锚
并自己重算一遍新摆放; 这个不重算任何东西, 直接用默认 cfg (camera 锚 + PreGrasp 对齐)
建 env, 机器人摆到 env 自己解好的 `q_ref[t]`. 所以看到的就是训练 reset 后的样子.

  # 停在 PreGrasp 帧 (默认)
  $PY -m rl_rebuild.correction.view_ref --clip Grasp2
  # 停在别的参考帧
  $PY -m rl_rebuild.correction.view_ref --clip Grasp2 --t 88
  # 循环播放整条参考轨迹
  $PY -m rl_rebuild.correction.view_ref --clip Grasp2 --play
  # 关掉 PreGrasp 对齐 (对照: 重建原样的手物相对关系)
  GRASP_PREGRASP_ALIGN=0 $PY -m rl_rebuild.correction.view_ref --clip Grasp2

⚠ 只 render 不 step 物理 —— 手是被"摆"在参考位姿上的, 不是靠电机撑住的.
   要看物理下跟不跟得住, 那是零残差冒烟的事, 不是这个工具.
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Grasp2")
p.add_argument("--t", type=int, default=None, help="停在哪个参考帧 (默认 PreGrasp 帧)")
p.add_argument("--play", action="store_true", help="循环播放整条参考")
p.add_argument("--fps", type=float, default=20.0)
p.add_argument("--eye", default="0.55,-0.85,1.45")
p.add_argument("--lookat", default="-0.10,-0.08,0.92")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("viewref")
app = AppLauncher(args).app

import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ViewerCfg  # noqa: E402

from rl_rebuild.correction import clips, place_camera as PC  # noqa: E402
from rl_rebuild.correction.env.dexmate_env import DexmateCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg  # noqa: E402
from rl_rebuild.correction.ref_builders.replay_grasp import _affordance_target  # noqa: E402

cfg = DexmateCorrectionEnvCfg()
clips.configure_cfg(cfg, args.clip)
cfg.scene.num_envs = 1
cfg.rsi_prob = 0.0
cfg.viewer = ViewerCfg(eye=tuple(float(v) for v in args.eye.split(",")),
                       lookat=tuple(float(v) for v in args.lookat.split(",")),
                       origin_type="world", resolution=(1600, 900))
E = DexmateCorrectionEnv(cfg)
E.reset()

W = E.scene.env_origins[0].cpu().numpy()
hand, TZ, L = cfg.hand_side, cfg.table_top_z, E.L
gs = E.grasp_start
t0 = gs if args.t is None else int(np.clip(args.t, 0, L - 1))

wp = E.ref_wrist_pos.cpu().numpy()
obj = E.obj_init_pos.cpu().numpy()
obj_q = E.obj_init_quat.cpu().numpy()

# ---- 画: 参考腕轨迹 / 物体 / affordance ----
E._draw_curve("/World/RefTraj", wp + W, (0.1, 0.85, 0.9), cfg.traj_width)
E._draw_sphere("/World/MarkObj", obj + W, (0.1, 0.85, 0.9), 0.020)
e = clips.clip_entry(args.clip)
if e.get("affordance"):
    aff_w = PC.affordance_world(obj, obj_q, _affordance_target(e["affordance"]))
    E._draw_sphere("/World/MarkAfford", aff_w + W, (1.0, 0.1, 0.1), 0.012)
E._draw_sphere("/World/MarkPreGrasp", wp[gs] + W, (1.0, 0.9, 0.1), 0.012)


def pose(t):
    """把机器人摆到参考帧 t —— 用 env 自己解好的 q_ref/ref_finger, 不重解 IK."""
    q = E.hand.data.joint_pos.clone()
    q[:, E.arm_jids] = E.q_ref[t]
    q[:, E.hand_jids] = E.ref_finger[t]
    E.hand.write_joint_state_to_sim(q, torch.zeros_like(q))
    E.hand.set_joint_position_target(q)
    E.hand.write_data_to_sim()


pose(t0)
_op = torch.tensor(np.concatenate([obj + W, obj_q]), dtype=torch.float32,
                   device=E.device)[None]
E.object.write_root_pose_to_sim(_op)

print("\n" + "=" * 74)
print(f"clip={args.clip}  交互手={hand}  L={L}  PreGrasp帧 gs={gs}  当前帧 t={t0}")
print(f"锚 anchor_mode={cfg.anchor_mode}  PreGrasp对齐={E.du.ref.builder_wrist_shift is not None}"
      f"  ({PC.PREGRASP_ANCHOR}, hover={PC.PREGRASP_HOVER_GAP*100:.0f}cm)")
print(f"物体 {np.round(obj, 4)}   离桌 {(obj[2]-TZ)*100:.1f}cm")
print(f"腕 ref[{t0}] {np.round(wp[t0], 4)}   离桌 {(wp[t0,2]-TZ)*100:.1f}cm")
print(f"ref builder 总平移 {np.round(E.du.ref.builder_wrist_shift*100, 2).tolist()} cm")
print("=" * 74)
print("[view_ref] 青线=参考腕轨迹 青球=物体 红球=affordance 黄球=PreGrasp腕位")
print("[view_ref] 关窗口或 Ctrl-C 退出" + ("  |  播放中" if args.play else ""))

t, dt = t0, 1.0 / max(args.fps, 1e-3)
_last = time.time()
while app.is_running():
    if args.play and time.time() - _last >= dt:
        _last = time.time()
        t = (t + 1) % L
        pose(t)
    E.sim.render()
E.close()
