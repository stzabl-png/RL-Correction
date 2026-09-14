"""A0 握持探针入口 (台账 §5.2 A0 / §5.3 H1 H2 H3).

  SHARPA_WANDB=0 $PY tasks/Clean/3/C_Wiring/probe_hold.py --headless [--num_envs 4] [--hold_s 3] [--rows -1]
判据: 静持 + 放音全程 两物体相对手漂移 <1cm/10°, 不掉; 盘倾角 <15°; 放音有贴合/覆盖 (H3 标定原料)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=4)
p.add_argument("--hold_s", type=float, default=3.0)
p.add_argument("--rows", type=int, default=-1, help="放音行数 (-1=全部, 0=不放音)")
p.add_argument("--hulls", type=int, default=128)
p.add_argument("--plate_mass", type=float, default=0.3)
p.add_argument("--mu", type=float, default=1.0)
p.add_argument("--tag", default="v1")
p.add_argument("--beta", type=float, default=1.0, help="合拢剂量: 指目标 = grasp + β(squeeze−grasp)")
p.add_argument("--beta_left", type=float, default=None); p.add_argument("--beta_right", type=float, default=None)
p.add_argument("--trim_left", type=float, nargs=3, default=(0.0, 0.0, 0.0), help="左腕微调 (盘输入系 xyz, m; +y=手相对盘上移=托底指顶紧缘底)")
p.add_argument("--trim_right", type=float, nargs=3, default=(0.0, 0.0, 0.0))
p.add_argument("--render", action="store_true", help="GUI 目检 (不加 --headless): 每物理步渲染, 静持/放音都能看")
p.add_argument("--pause_s", type=float, default=0.0, help="GUI 模式下 复位后先静止观察 N 秒再继续")
p.add_argument("--kinematic", action="store_true",
               help="不开物理: 手和物体全程钉在 GraspPose 相对位姿, 臂按母带行直接写关节状态 (纯目检)")
p.add_argument("--traj", choices=("ref", "human"), default="ref",
               help="kinematic 播什么: ref=机器人母带 (腕由物轨反推); human=人手重定向轨迹 ref_qpos (腕首帧焊接借增量+指流)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
# ★ 物理规矩必须在 import correction_env 之前生效 (PHYS_RULE 是导入期 setdefault)
os.environ.setdefault("SHARPA_WANDB", "0")
os.environ["POUR_OBJ_MASS"] = str(args.plate_mass)     # 主体物 = 盘
os.environ["POUR_OBJ_FRIC"] = str(args.mu)
os.environ["POUR_PAD_FRIC"] = str(args.mu)
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("clean3_probe")
app = AppLauncher(args).app
import numpy as np  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hold_env as HE  # noqa: E402

t0 = time.time()
_cfg = HE.build_cfg(args.num_envs, plate_hulls=args.hulls)
_cfg.clean_trim_left = tuple(args.trim_left); _cfg.clean_trim_right = tuple(args.trim_right)
_cfg.clean_render = bool(args.render)
_cfg.clean_kinematic = bool(args.kinematic)      # 建场景时关物体碰撞
raw = HE.HoldProbeEnv(_cfg)
if args.render and args.pause_s > 0:
    # ⚠ 只能用 sim.render(): 它会临时关掉 playSimulations 只渲染; app.update() 会照常推进物理 (物体会掉)
    if args.kinematic:
        raw.set_object_gravity(False)
    raw.kinematic_place0()
    print(f"[probe_hold] GUI: 已摆到母带 0 行 GraspPose, 静止 {args.pause_s:.0f}s (可调视角, 不步进物理)"); sys.stdout.flush()
    t_end = time.time() + args.pause_s
    while time.time() < t_end:
        raw.sim.render()
if args.kinematic:
    raw.kinematic_replay(args.rows, hold_s=max(args.hold_s, 0.0), traj=args.traj)
    print("[probe_hold] kinematic replay done"); sys.stdout.flush()
    try: _slot.release()
    except Exception: pass
    os._exit(0)
audit = raw.settle_inhand(beta=args.beta, beta_left=args.beta_left, beta_right=args.beta_right)
hold = raw.hold(args.hold_s)
rep = raw.replay(args.rows) if args.rows != 0 else None
out_dir = os.path.join(HE._TASK, "A_Design", "L2_Reference")
res = dict(tag=args.tag, num_envs=args.num_envs, plate_mass=args.plate_mass, mu=args.mu, hulls=args.hulls, beta=args.beta,
           trim_left=list(args.trim_left), trim_right=list(args.trim_right),
           reference=raw.cfg.clean_reference, settle=audit, hold=hold["summary"], hold_log=hold["log"],
           replay=(rep["summary"] if rep else None), wall_s=time.time() - t0)
with open(os.path.join(out_dir, f"probe_hold_{args.tag}.json"), "w") as f:
    json.dump(res, f, indent=1, ensure_ascii=False)
with open(os.path.join(out_dir, f"settled_oh_{args.tag}.json"), "w") as f:
    json.dump(dict(tag=args.tag, reference=raw.cfg.clean_reference, beta=args.beta,
                   settled=hold["summary"]["settled_oh"]), f, indent=1)
if rep:
    np.savez(os.path.join(out_dir, f"probe_hold_{args.tag}_rows.npz"), **rep["rec"])
ok = (max(hold["summary"]["left_pos_cm"], hold["summary"]["right_pos_cm"]) < 1.0
      and max(hold["summary"]["left_rot_deg"], hold["summary"]["right_rot_deg"]) < 10.0
      and sum(hold["summary"]["dropped"].values()) == 0)
print(f"[probe_hold] HOLD {'PASS' if ok else 'FAIL'} | replay={'n/a' if not rep else rep['summary']['first_drop_row']} "
      f"| wall {time.time()-t0:.0f}s | -> probe_hold_{args.tag}.json")
sys.stdout.flush()
try: _slot.release()
except Exception: pass
os._exit(0)
