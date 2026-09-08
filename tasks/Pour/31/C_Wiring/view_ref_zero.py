"""整段参考 GUI 回放 (零动作): 训练环境里从 t0 出生, 残差全零, 机器人纯前馈跟着母带走
接近 -> 交互(倒水) -> 撤离, 物理开着 (抓不抓得住/撒不撒手都是真实结果, 不是动画)。

    SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Pour/31/C_Wiring/view_ref_zero.py
        [--ref <母带npz>]   默认 pour_env 的 MASTER (pour31_reference_v5.npz)
        [--entry 0]         出生点序号 (0 = t0 从头)
        [--loop]            一集播完自动重置再播; 不给则播完停在末态
        [--open_loop]       交互段时钟强制每步 +1 (不等 G2 认证/不看皮筋门控), 纯按母带走
        [--render_once]     每个 env 步只渲 1 帧 (默认渲 6 帧: 12 子步/渲染间隔 2), 更流畅更快
        [--headless --steps N]  无头自检 (只打印, 不开窗)

窗口里按回车开始播 (先把视角摆好); 每 50 步在终端打一行 行号/G链/垫数。
"""
from __future__ import annotations

import argparse
import os
import select
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--ref", default="", help="母带 npz; 空=pour_env 默认 MASTER")
p.add_argument("--entry", type=int, default=0)
p.add_argument("--steps", type=int, default=0, help="最多步数; 0=按母带行数自动(全链+100)")
p.add_argument("--loop", action="store_true")
p.add_argument("--open_loop", action="store_true")
p.add_argument("--render_once", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.ref:
    os.environ["POUR_REF_NPZ"] = os.path.abspath(args.ref)

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("view_ref_zero")
app = AppLauncher(args).app

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pour_env as PE  # noqa: E402

cfg = PE.build_cfg(num_envs=1)
if args.render_once:
    cfg.sim.render_interval = int(cfg.decimation)
    print(f"[回放] --render_once: render_interval={cfg.sim.render_interval} (=decimation)", flush=True)
E = PE.PourEnv(cfg)
labels = [e[4] for e in E.entries]
E.force_entry = [int(args.entry)]
print(f"[回放] 出生点 {labels[int(args.entry)]} | 母带 {E.T_ROW} 行 IA=[{E.IA0},{E.IA1}] RETREAT0={E.RETREAT0}", flush=True)
E.reset()
gui = not args.headless
if gui:
    print("[回放] 摆好视角后在**终端**按回车开始播放 ...", flush=True)
    while True:
        app.update()
        r, _, _ = select.select([sys.stdin], [], [], 0.05)
        if r:
            sys.stdin.readline()
            break
zero = torch.zeros(1, PE.ACT_DIM, device=E.device)
N = int(args.steps) if args.steps > 0 else int(E.T_ROW) + 100
while True:
    E.reset()
    gates = {}
    for t in range(N):
        k_prev = E.PB.k.clone() if args.open_loop else None
        obs, rew, term, trunc, _ = E.step(zero)
        if args.open_loop and int(E.row[0]) >= E.IA0:
            # 交互段: 时钟强制 +1/步 (进度机自己的棘轮结果被覆盖); 到顶后 env 自行开环走 seam2+retreat
            E.PB.k[:] = torch.clamp(k_prev + 1, max=E.PB.N_ROW - 1)
        for m, flag in ((1, E.PB.g1), (2, E.PB.g2), (3, E.PB.g3), ("placed", E.PB.placed), (4, E.PB.g4)):
            if bool(flag[0]) and m not in gates:
                gates[m] = t
                print(f"[回放] t{t:4d} 行{int(E.row[0]):4d}  ★ 达成 {m}", flush=True)
        if t % 50 == 0:
            f = E._pads_f().norm(dim=-1)
            print(f"[回放] t{t:4d} 行{int(E.row[0]):4d}/{E.T_ROW} 垫R={(f[0, :5] > 0.5).sum().item()} "
                  f"垫L={(f[0, 5:] > 0.5).sum().item()} G链={sorted(gates, key=str)}", flush=True)
        if bool(term[0]) or bool(trunc[0]):
            why = "timeout" if bool(trunc[0]) else ("G4" if bool(E.PB.g4[0]) else
                  ("env侧死线" if bool(E._tick_out["fail_env"][0]) else "进度机"))
            print(f"[回放] 一集结束 @t{t} 行{int(E.row[0])} 原因={why} G链={sorted(gates, key=str)}", flush=True)
            break
    if not (gui and args.loop):
        break
    print("[回放] --loop: 重置再播", flush=True)
if gui:
    print("[回放] 播完, 停在末态; 关窗口或 Ctrl+C 退出", flush=True)
    try:
        while app.is_running():
            app.update()
    except KeyboardInterrupt:
        pass
try:
    _slot.release()
except Exception:
    pass
os._exit(0)
