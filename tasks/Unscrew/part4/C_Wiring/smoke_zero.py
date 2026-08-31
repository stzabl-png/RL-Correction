"""零动作预检 (#18 运行层, 接线工程首验): 4 env 各从一个 RSI 桶出生, 全程零动作。

自检项:
  A 静置位对账: env 场景物体静置位 vs 母带静置位 (<5mm 铁则, 不然进度机判据全错框架)
  B 观测维 507 / 动作 58 收发正常
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
_slot = isaac_slot("unscrew_smoke0")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import task_env as PE  # noqa: E402

cfg = PE.build_cfg(num_envs=4)
E = PE.UnscrewEnv(cfg)
# L5-1: 4 env 定点覆盖出生表 (POUR_UNLOCK=1,2,3 时=t0/g1/g2/g3或ret)
E.unlocked.update((1, 2, 3))
E._rebuild_entries()
labels = [e[4] for e in E.entries]
want = (list(range(len(E.entries))) * 4)[:4]  # v4: 绿点退役, 现役全覆盖
E.force_entry = want
print(f"[零动作] 进入点: {[labels[i] for i in want]} | 全链 {E.T_ROW} 行 "
      f"IA=[{E.IA0},{E.IA1}] RETREAT0={E.RETREAT0}")

# ---- A: 静置位对账 (env 默认态 vs 母带) ----
E.reset()
for _ in range(30):     # 静置几步让物理落定
    E._SA.apply_screw(E, integrate_angle=False)
    E.scene.write_data_to_sim()
    E.sim.step(render=False)
    E.scene.update(E.sim.get_physics_dt())
E._SA.apply_screw(E, integrate_angle=False)
bot, cap = E._read_objs()
rest_errors_cm = []
for oi, (name, cur) in enumerate((("瓶", bot[0]), ("盖", cap[0]))):
    ref = E.rest_pose[oi].to(cur.device)
    dp = float((cur[:3] - ref[:3]).norm()) * 100
    rest_errors_cm.append(dp)
    print(f"[零动作] A 静置对账 {name}: |Δpos|={dp:.2f}cm (env0=t0, 物体在静置)")
rest_ok = max(rest_errors_cm) < 0.5

obs, _ = E.reset()
assert obs["policy"].shape == (4, PE.OBS_DIM), obs["policy"].shape
print(f"[零动作] B obs {obs['policy'].shape} (期望 (4,{PE.OBS_DIM}))")

zero = torch.zeros(4, PE.ACT_DIM, device=E.device)
ms_at = {i: {} for i in range(4)}
fail_at = {i: None for i in range(4)}
maxrow = torch.zeros(4, dtype=torch.long)
maxk = torch.zeros(4, dtype=torch.long)
pre_fail = 0
seam_fail = 0
_FAILNM = {10: "D1掉瓶", 11: "D1掉盖", 20: "D2瓶倒", 21: "D2盖倒",
           30: "D3瓶偏", 31: "D3盖偏", 4: "D4滑移"}
rew_sum = torch.zeros(4)
done_at = {i: None for i in range(4)}
for t in range(args.steps):
    row_pre = E.row.clone()          # 步进**前**的行号: 失败与 reset 同一步发生,
    obs, rew, term, trunc, _ = E.step(zero)   # 事后读 E.row 只会读到 0
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
            # 铁则的范围 = **cuRobo 背书的机器段** (row < APP_END), 与本文件
            # 开头写的"approach 期不得响"一致。缝1 是"手主动去接触物体"的交接段,
            # 那里有接触是设计内的, 而且手指/臂残差在缝1 对策略开放 —— 缝1 把瓶
            # 碰倒是 RL correction 要学的第一课, 单独统计, 不并进铁则。
            if bool(o["fail_env"][i]) and (row_pre[i] < E.APP_END):
                pre_fail += 1
                _c = int(E.fail_code[i])
                _nm = _FAILNM.get(_c, f"码{_c}")
                print(f"[零动作] ⚠ 机器段死线误触 env{i} @t{t} row={int(row_pre[i])} "
                      f"-> {_nm}", flush=True)
            elif bool(o["fail_env"][i]) and (row_pre[i] < E.IA0):
                seam_fail += 1
                print(f"[零动作] · 缝1 交接段死线 env{i} @t{t} "
                      f"row={int(row_pre[i])} -> "
                      f"{_FAILNM.get(int(E.fail_code[i]), '?')} (记账, 不入铁则)",
                      flush=True)
            if bool(term[i]) or bool(trunc[i]):
                done_at[i] = t
                fail_at[i] = ("timeout" if bool(trunc[i]) else
                              ("G4" if bool(E.PB.g4[i]) else
                               ("env侧" if bool(o["fail_env"][i]) else "进度机")))
    if t % 100 == 0:
        f = E._pads_f().norm(dim=-1)
        print(f"[零动作] t{t:4d} row={E.row.cpu().tolist()} "
              f"k={E.PB.k.cpu().tolist()} "
              f"垫L={[(f[i, :5] > 0.5).sum().item() for i in range(4)]} "
              f"垫R={[(f[i, 5:] > 0.5).sum().item() for i in range(4)]} "
              f"screw={[round(float(v), 1) for v in torch.rad2deg(E.screw_angle).cpu().tolist()]}", flush=True)
    if all(v is not None for v in done_at.values()):
        break

print("\n===== 零动作预检报告 =====")
names = [labels[i] for i in want]
for i in range(4):
    print(f"env{i} [{names[i]:10s}] 终:{fail_at[i] or '未终止'}@{done_at[i]} "
          f"maxrow={int(maxrow[i])} maxK={int(maxk[i])} "
          f"G链={sorted(ms_at[i], key=str)} 累计奖励={float(rew_sum[i]):.1f}")
print(f"机器段死线误触 = {pre_fail} (铁则: 0; 范围=cuRobo 背书的 row<{E.APP_END})")
print(f"缝1 交接段死线 = {seam_fail} (记账项: 零动作合拢会不会碰倒物体; "
      f"手指/臂残差在此段对策略开放, 属 correction 的学习对象)")
ok = rest_ok and pre_fail == 0
print("✅ 预检骨架通过 (M链/时钟为实录数据)" if ok else
      f"❌ 预检失败: rest_ok={rest_ok} pre_fail={pre_fail}")
try:
    _slot.release()
except Exception:
    pass
app.close()
raise SystemExit(0 if ok else 1)
