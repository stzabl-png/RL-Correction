"""零动作预检 (#18 运行层, 接线工程首验): 4 env 各从一个 RSI 桶出生, 全程零动作。

自检项:
  A 静置位对账: env 场景物体静置位 vs 母带静置位 (<5mm 铁则, 不然进度机判据全错框架)
  B 观测维 503 / 动作 58 收发正常
  C 机器段死线零误触 (approach 期 D1/D2/D3/D5 不得响)
  D 时钟/行指针推进与 M 链实录 (M1 在力控体制下是否点火 = 数据, 不是断言 ——
    柔顺补测已知右手垫欠实, 贴实是相A探索正业)
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=800)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("pour17_smoke0")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pour_env as PE  # noqa: E402

cfg = PE.build_cfg(num_envs=4)
E = PE.PourEnv(cfg)
# L5-1: 4 env 定点覆盖出生表 (POUR_UNLOCK=1,2,3 时=t0/g1/g2/g3或ret)
labels = [e[4] for e in E.entries]
want = (list(range(len(E.entries))) * 4)[:4]  # v4: 绿点退役, 现役全覆盖
E.force_entry = want
print(f"[零动作] 进入点: {[labels[i] for i in want]} | 全链 {E.T_ROW} 行 "
      f"IA=[{E.IA0},{E.IA1}] RETREAT0={E.RETREAT0}")

# ---- A: 静置位对账 (env 默认态 vs 母带) ----
E.reset()
for _ in range(30):     # 静置几步让物理落定
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())
cup, bot = E._read_objs()
for oi, (name, cur) in enumerate((("杯", cup[0]), ("瓶", bot[0]))):
    ref = E.rest_pose[oi].to(cur.device)
    dp = float((cur[:3] - ref[:3]).norm()) * 100
    print(f"[零动作] A 静置对账 {name}: |Δpos|={dp:.2f}cm (env0=t0, 物体在静置)")

obs, _ = E.reset()
print(f"[零动作] B obs {obs['policy'].shape} (期望 (4,{PE.OBS_DIM}))")

zero = torch.zeros(4, PE.ACT_DIM, device=E.device)
ms_at = {i: {} for i in range(4)}
fail_at = {i: None for i in range(4)}
maxrow = torch.zeros(4, dtype=torch.long)
maxk = torch.zeros(4, dtype=torch.long)
pre_fail = 0
rew_sum = torch.zeros(4)
done_at = {i: None for i in range(4)}
for t in range(args.steps):
    obs, rew, term, trunc, _ = E.step(zero)
    o = E._tick_out
    rew_sum += rew.cpu()
    maxrow = torch.maximum(maxrow, E.row.cpu())
    maxk = torch.maximum(maxk, E.PB.k.cpu())
    for i in range(4):
        if done_at[i] is None:
            for m, flag in ((1, E.PB.g1), (2, E.PB.g2), (3, E.PB.g3),
                            ("p", E.PB.placed), (4, E.PB.g4)):
                if bool(flag[i]) and m not in ms_at[i]:
                    ms_at[i][m] = t
            if bool(o["fail_env"][i]) and (E.row[i] < E.IA0):
                pre_fail += 1
            if bool(term[i]) or bool(trunc[i]):
                done_at[i] = t
                fail_at[i] = ("timeout" if bool(trunc[i]) else
                              ("G4" if bool(E.PB.g4[i]) else
                               ("env侧" if bool(o["fail_env"][i]) else "进度机")))
    if t % 100 == 0:
        f = E._pads_f().norm(dim=-1)
        print(f"[零动作] t{t:4d} row={E.row.cpu().tolist()} "
              f"k={E.PB.k.cpu().tolist()} "
              f"垫R={[(f[i, :5] > 0.5).sum().item() for i in range(4)]} "
              f"垫L={[(f[i, 5:] > 0.5).sum().item() for i in range(4)]}", flush=True)
    if all(v is not None for v in done_at.values()):
        break

print("\n===== 零动作预检报告 =====")
names = [labels[i] for i in want]
for i in range(4):
    print(f"env{i} [{names[i]:10s}] 终:{fail_at[i] or '未终止'}@{done_at[i]} "
          f"maxrow={int(maxrow[i])} maxK={int(maxk[i])} "
          f"G链={sorted(ms_at[i], key=str)} 累计奖励={float(rew_sum[i]):.1f}")
print(f"机器段死线误触 = {pre_fail} (铁则: 0)")
# ★L5-32: 新主判据的接线检查。零动作**不该**通关(参考只是前馈, 抓握靠策略),
#   这里要的不是"过没过", 而是"这几个量算得出来、不抛异常、语义对得上"。
try:
    _rt = E.PB.pop_rates()
    print("[零动作] 主判据接线: " + " ".join(
        f"{k}={_rt[k]:.3f}" for k in ("sr/gate3", "sr/placed", "sr/success",
                                      "sr/g3_loose", "sr/clock_done")
        if k in _rt))
    _need = {"sr/gate3", "sr/placed", "sr/success", "sr/g3_loose",
             "sr_t0/success", "sr_t0/gate3", "sr_t0/placed"}
    _miss = _need - set(_rt)
    print(f"[零动作] 主判据键齐全 = {not _miss}"
          + (f"  ★缺 {sorted(_miss)}" if _miss else ""))
    ok_keys = not _miss
except Exception as _e:
    print(f"[零动作] ❌ 主判据读数抛异常: {type(_e).__name__}: {_e}")
    ok_keys = False
ok = pre_fail == 0 and ok_keys
print("✅ 预检骨架通过 (M链/时钟为实录数据)" if ok else "❌ 机器段死线误触")
try:
    _slot.release()
except Exception:
    pass
app.close()
raise SystemExit(0 if ok else 1)
