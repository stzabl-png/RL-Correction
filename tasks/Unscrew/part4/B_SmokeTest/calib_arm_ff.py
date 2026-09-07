"""T2-37 逐行稳态漂移探针 (原为臂前馈标定, 前提已证伪, 保留作诊断工具)。

  母带行硬写 -> 目标 = 行 (+预偏) 保持 HOLD 步 -> 读稳态 q
  -> 预偏 += (行 - 稳态)          (迭代学习控制; 若位移由接触约束决定则不收敛)
一个 env 对应一行 (78 行 = 78 env, 物理 GPU 并行, 一轮 ≈ 单行耗时)。物体逐
子步钉在母带行。**T2-37 结论**: 自由空间双臂跟踪到 ~2mm, 无控制器漂移;
行 212~228 的 3cm 位移是右前臂外壳穿桌 (最深 -1.8cm) 被 PhysX 顶出, 前馈
迭代零效果 —— 用途改为: 重铸母带后复查各行稳态位移是否回到 ~2mm。
输出 npz 仅供诊断 (hist=各轮逐行 [R腕漂移, L腕漂移, dmin写入, dmin稳态])。

  ... calib_arm_ff.py --headless [--iters 0 --hold 25 --out ...]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ["POUR_NO_D6"] = "1"

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--out", default=None)
p.add_argument("--iters", type=int, default=2, help="预偏更新轮数 (另加 1 轮纯复验)")
p.add_argument("--hold", type=int, default=25,
               help="每行保持的控制步数 (0.05s/步; T2-34 实测 ~15 步内稳态)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402

_slot = isaac_slot("calib_arm_ff")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "C_Wiring"))
import task_config as TC  # noqa: E402
import task_env as PE  # noqa: E402

assert os.path.abspath(PE.MASTER) == os.path.abspath(TC.REF_V2), PE.MASTER
with np.load(TC.REF_V2, allow_pickle=True) as z:
    Z = {k: np.asarray(z[k]) for k in
         ("obj_pos_0", "obj_quat_0", "obj_pos_1", "obj_quat_1", "source")}
rows = [int(r) for r in np.where(Z["source"] == 1)[0]]
N = len(rows)
cfg = PE.build_cfg(num_envs=N)
E = PE.UnscrewEnv(cfg)
assert rows == list(range(E.IA0, E.IA1 + 1)), (rows[0], rows[-1], E.IA0, E.IA1)
E.force_entry = [0] * N
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))
DT = E.sim.get_physics_dt()
rows_t = torch.tensor(rows, dtype=torch.long, device=dev)
origins = E.scene.env_origins
zero_vel = torch.zeros(N, 6, device=dev)
obj_pose = {}
for oi in (0, 1):
    pos = torch.tensor(Z[f"obj_pos_{oi}"][rows], dtype=torch.float32, device=dev)
    quat = torch.tensor(Z[f"obj_quat_{oi}"][rows], dtype=torch.float32, device=dev)
    obj_pose[oi] = torch.cat([pos + origins, quat], dim=1)
ARM = E.map_ids_t[:14]


def pin():
    for oi, art in ((0, E.object), (1, E.aux)):
        art.write_root_pose_to_sim(obj_pose[oi])
        art.write_root_velocity_to_sim(zero_vel)


def substep():
    pin()
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(DT)


def wrist():
    bp = E.hand.data.body_pos_w
    return bp[:, E.wid["R"]].clone(), bp[:, E.wid["L"]].clone()


def dmin():
    bn = E.hand.data.body_pos_w
    pads = torch.stack([bn[:, E._pad_bids[i + 5]] for i in range(5)], dim=1)
    return (pads - E.aux.data.root_pos_w.unsqueeze(1)).norm(dim=-1).min(dim=1).values


def settle(corr):
    """硬写母带行 -> 目标=行+corr 保持 -> 返回稳态臂 q (N,14) 与末端漂移 (N,)."""
    qfull = E.hand.data.default_joint_pos.clone()
    qfull[:, E.map_ids_t] = E.ref58[rows_t]
    E.hand.write_joint_state_to_sim(qfull, torch.zeros_like(qfull))
    E.hand.set_joint_position_target(qfull)
    substep()                         # 1 子步 (4ms) 刷新正运动学 = 母带真值
    wr0, wl0 = wrist()
    d0 = dmin()
    tgt = qfull.clone()
    tgt[:, ARM] += corr
    E.hand.set_joint_position_target(tgt)
    for _ in range(args.hold):
        for _ in range(DECI):
            substep()
    q = E.hand.data.joint_pos[:, ARM].clone()
    wr1, wl1 = wrist()
    return q, (wr1 - wr0).norm(dim=-1), (wl1 - wl0).norm(dim=-1), d0, dmin()


crow = E.IA0 + int(E._ps.k_sep)
ci = rows.index(crow)
corr = torch.zeros(E.T_ROW, 14, device=dev)
hist = []
for it in range(args.iters + 1):
    t_it = time.time()
    q, eR, eL, d0, d1 = settle(corr[rows_t])
    if it < args.iters:
        corr[rows_t] += E.ref58[rows_t, :14] - q
    a = torch.stack([eR, eL, d0, d1], dim=1).cpu().numpy()
    tag = "基线" if it == 0 else ("复验" if it == args.iters else f"轮{it}")
    print(f"[calib] {tag} ({time.time() - t_it:.0f}s): 腕漂移 R max "
          f"{a[:, 0].max() * 100:.2f}cm mean {a[:, 0].mean() * 100:.2f}cm | "
          f"L max {a[:, 1].max() * 100:.2f}cm mean {a[:, 1].mean() * 100:.2f}cm "
          f"| c行{crow} dmin {a[ci, 2] * 100:.1f}cm -> {a[ci, 3] * 100:.1f}cm",
          flush=True)
    if it == 0:
        print("[calib] 基线逐行 R 腕漂移 (cm): "
              + " ".join(f"{v * 100:.1f}" for v in a[:, 0]), flush=True)
        print("[calib] 基线逐行 L 腕漂移 (cm): "
              + " ".join(f"{v * 100:.1f}" for v in a[:, 1]), flush=True)
    hist.append(a)

out = args.out or os.path.join(os.path.dirname(TC.REF_V2), "arm_ff_v1.npz")
np.savez(out, corr14=corr.cpu().numpy(), rows=np.array(rows),
         hold=args.hold, iters=args.iters, ref_v2_path=TC.REF_V2,
         hist=np.stack(hist), crow=crow)
print(f"[calib] 预偏表 -> {out} | 预偏 |max| {float(corr.abs().max()):.4f} rad "
      f"| 复验 R 腕漂移 max {hist[-1][:, 0].max() * 100:.2f}cm", flush=True)
try:
    _slot.release()
except Exception:
    pass
sys.stdout.flush()
os._exit(0)
