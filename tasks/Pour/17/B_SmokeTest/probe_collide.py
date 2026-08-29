"""L5-25 禁碰传感器冒烟: 验证 D6 绑定已修 + 禁碰传感器真的读得到力。

★为什么必须跑这个: collide_flags 的单元自检只证明"给它力它会红";
证明不了"传感器真的把力送进来"。D6 恰恰是栽在这一步 —— 逻辑没错,
过滤器从未绑上, 5 条线 7900 万步恒为 0 而无人发现。
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

_CW = os.path.dirname(os.path.abspath(__file__)).replace("B_SmokeTest", "C_Wiring")
sys.path.insert(0, _CW)
import pour_env as PE  # noqa: E402

cfg = PE.build_cfg(num_envs=8)
print(f"[probe] 传感器区段: {cfg.sensor_slices}", flush=True)
E = PE.PourEnv(cfg)
print(f"[probe] 实际传感器数={len(E._all_sensors)} 区段={E._slices}", flush=True)

# 逐段报告形状, 空段/零形状即为"没绑上"
for k in ("pad_r", "pad_l", "d6", "noc_arm", "obj_obj"):
    ss = E._seg(k)
    if not ss:
        print(f"[probe] {k:8s} 段为空", flush=True)
        continue
    s0 = ss[0]
    nf = getattr(s0.data, "net_forces_w", None)
    fm = getattr(s0.data, "force_matrix_w", None)
    print(f"[probe] {k:8s} n={len(ss)} net={tuple(nf.shape) if nf is not None else None}"
          f" mat={tuple(fm.shape) if fm is not None else None}", flush=True)

acc = {k: 0 for k in ("arm", "pad", "objobj", "d6")}
mx = {k: 0.0 for k in ("arm", "pad", "objobj", "d6")}
E.reset()
a = torch.zeros(E.num_envs, PE.ACT_DIM, device=E.device)
for t in range(args.steps):
    E.step(a)
    c = E._collide()
    for k in acc:
        acc[k] += int(c[k].sum())
    for k, ss in (("arm", E._seg("noc_arm")), ("d6", E._seg("d6")),
                  ("objobj", E._seg("obj_obj"))):
        if ss:
            at = "net_forces_w" if k == "arm" else "force_matrix_w"
            v = torch.cat([getattr(s.data, at).reshape(E.num_envs, -1, 3)
                           for s in ss], dim=1).nan_to_num(0.0).norm(dim=-1).max()
            mx[k] = max(mx[k], float(v))

print(f"\n[probe] 零动作 {args.steps} 步 × {E.num_envs} env 结果:", flush=True)
for k in acc:
    print(f"[probe]   {k:8s} 触发步数={acc[k]:5d}  观测到的最大力={mx[k]:.3f} N", flush=True)
print("\n[probe] ★判读: 力恒为 0.000 = 传感器没绑上(D6 的老毛病);"
      "\n[probe]        力有非零值 = 通道是活的, 判据才有意义。", flush=True)
sys.stdout.flush()
app.close()
os._exit(0)
