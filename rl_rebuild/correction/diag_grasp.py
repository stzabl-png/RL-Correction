"""抓握失败诊断: 零残差跑一遍, 逐相位量化 输入几何 + 物体被拖/甩 + 接触.
回答: 手起始在物体哪(顶部?), 指尖离瓶身多远, 高摩擦下参考动作是否拖飞物体.

  .venv-isaac/bin/python -m rl_rebuild.correction.diag_grasp --clip pp0_human --headless
"""
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--clip", type=str, default="pp0_human")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--pin_object", action="store_true",
                    help="每步把物体钉回初始位 (kinematic), 隔离'物体被撞跑'看真实对齐")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
from rl_rebuild.correction.env.correction_env import SharpaCorrectionEnv  # noqa: E402
from rl_rebuild.correction.env.correction_env_cfg import SharpaCorrectionEnvCfg  # noqa: E402
from rl_rebuild.correction import clips  # noqa: E402

cfg = SharpaCorrectionEnvCfg()
clips.configure_cfg(cfg, args.clip)
cfg.scene.num_envs = args.num_envs
env = SharpaCorrectionEnv(cfg)
env.reset()
N, dev = env.num_envs, env.device
zero = torch.zeros(N, 28, device=dev)

# 物体 AABB 半对角 (判断指尖离表面)
lo, hi = env.du.mesh_path, None
import rl_rebuild.correction.frames as F
vlo, vhi = F.object_aabb_obj(env.du.mesh_path)
half = np.linalg.norm(vhi - vlo) / 2

tip_ids = env.tip_ids
log = {k: [] for k in ["obj_xy_disp", "obj_z", "objdiv", "tip_min_d", "n_contact",
                       "palm_d", "wrist_z", "obj_vel"]}
obj0 = None
obj_pin = torch.cat([env.obj_init_pos + env.scene.env_origins,
                     env.obj_init_quat.expand(N, -1)], dim=1) if args.pin_object else None
with torch.inference_mode():
    for step in range(env.ep_total):
        _, _, _, _, _ = env.step(zero)
        if obj_pin is not None:   # 钉死物体: 每步写回初始位姿 + 清零速度
            env.object.write_root_pose_to_sim(obj_pin, torch.arange(N, device=dev))
            env.object.write_root_velocity_to_sim(torch.zeros(N, 6, device=dev),
                                                  torch.arange(N, device=dev))
        origins = env.scene.env_origins
        op = env.object.data.root_pos_w - origins           # (N,3)
        if obj0 is None:
            obj0 = op.clone()
        t = env._ref_t()
        # 指尖世界位置 -> 到物体中心距离 (减半对角≈到表面)
        tips = env.hand.data.body_pos_w[:, tip_ids] - origins.unsqueeze(1)   # (N,5,3)
        d_tip = (tips - op.unsqueeze(1)).norm(dim=2)         # (N,5)
        palm = env._palm_pos() - origins
        contacts = env._tip_contacts()
        log["obj_xy_disp"].append((op[:, :2] - obj0[:, :2]).norm(dim=1).cpu().numpy())
        log["obj_z"].append(op[:, 2].cpu().numpy())
        log["objdiv"].append((op - env.ref_obj_pos[t])[:, :2].norm(dim=1).cpu().numpy())
        log["tip_min_d"].append(d_tip.min(dim=1)[0].cpu().numpy())
        log["n_contact"].append(contacts.sum(dim=1).cpu().numpy())
        log["palm_d"].append((palm - op).norm(dim=1).cpu().numpy())
        log["wrist_z"].append((env.hand.data.root_pos_w - origins)[:, 2].cpu().numpy())
        log["obj_vel"].append(env.object.data.root_lin_vel_w.norm(dim=1).cpu().numpy())

C = {k: np.stack(v) for k, v in log.items()}   # (T,N)
settle, L, t0 = env.cfg.settle_steps, env.L, env.t0
hold = env.cfg.hold_steps
inter0 = settle + env.du.ref.interaction_seg[0] - t0
ph = {"静置": slice(0, settle), "交互": slice(settle, settle + L - t0),
      "hold": slice(settle + L - t0, settle + L - t0 + hold)}
print(f"\n==== 抓握诊断 {args.clip}  物体半对角={half*100:.1f}cm  "
      f"obj_div阈值={cfg.term_obj_div*100:.0f}cm ====")
print(f"{'量':<16}" + "".join(f"{p:>18}" for p in ph))
for k, unit, s in [("tip_min_d", "cm 指尖-物中心", 100), ("n_contact", "指接触数", 1),
                   ("obj_xy_disp", "cm 物体水平位移", 100), ("obj_z", "m 物体高度", 1),
                   ("objdiv", "cm 偏离参考(水平)", 100), ("obj_vel", "m/s 物速", 1),
                   ("wrist_z", "m 腕高", 1)]:
    row = [(f"{C[k][sl].mean()*s:6.1f}/{C[k][sl].max()*s:<6.1f}" if C[k][sl].size else " - ")
           for sl in ph.values()]
    print(f"{k:<16}" + "".join(f"{r:>18}" for r in row) + f"  {unit}(均/max)")
print(f"\n起始: 指尖-物中心最小 {C['tip_min_d'][0].mean()*100:.1f}cm  "
      f"腕高 {C['wrist_z'][0].mean():.3f}m  物体顶 {(obj0[:,2].mean().item()+ (vhi[2]-vlo[2])/2):.3f}m")
print(f"整段: 指尖曾接触物体? 最大接触数={int(C['n_contact'].max())}  "
      f"零残差下物体最大水平位移={C['obj_xy_disp'].max()*100:.1f}cm  最大物速={C['obj_vel'].max():.2f}m/s")
env.close()
app.close()
