"""Sweep408 Stage-1 冒烟: 建环境 -> 零动作跑 N 步 -> 逐项核账。

  SHARPA_WANDB=0 PYTHONPATH=. $PY tasks/Sweep/408/C_Wiring/smoke_grip.py --headless [--num_envs 8] [--steps 120]

核的项 (任一不过就 FAIL):
  ① 观测/动作维对得上   ② 手内复位审计: 掌系相对位姿 vs 先验 T_oh 亚厘米
  ③ 钉住期物体不动       ④ 放手后零动作漂移 (只报数, 不设阈 —— 这是策略要学的)
  ⑤ 全程无手/垫低于桌面  ⑥ 横幅: 质量/摩擦覆写生效
"""
from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=8)
p.add_argument("--steps", type=int, default=120)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_config as TC  # noqa: E402
for k, v in TC.PHYS.items():
    os.environ.setdefault(k, v)
os.environ.setdefault("SHARPA_WANDB", "0")
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep408_smoke")
app = AppLauncher(args).app
import numpy as np  # noqa: E402
import torch  # noqa: E402
import grip_env as GE  # noqa: E402

FAIL = []
cfg = GE.build_cfg(args.num_envs)
E = GE.Sweep408GripEnv(cfg)
obs, _ = E.reset()
w = obs["policy"].shape[1]
print(f"[smoke] ⓪ 臂={TC.ARM} soft={TC.USE_SOFT} table={TC.USE_TABLE} contact={TC.USE_CONTACT} | "
      f"成功线=时钟≥{TC.SUCCESS_CLOCK_FRAC:.0%} | 认证=纯位姿 {TC.CERT_POS*100:.1f}cm/{TC.CERT_ROT_DEG:.0f}°")
print(f"[smoke] ① obs 宽度 {w} (期望 {GE.OBS_DIM}) act {GE.ACT_DIM}")
if w != GE.OBS_DIM:
    FAIL.append(f"obs 宽度 {w} != {GE.OBS_DIM}")

# ② 手内复位审计: 掌系里"物体相对手"的位姿 vs **inv(T_oh)**
#    ⚠ 先验的 grasp[:7] 是 "**手在物体系**" 的位姿 (T_oh); _rel() 给的是 "**物体在掌系**", 两者互为逆。
#    直接拿 T_oh 比会得到长度相同、方向不同的假失败 (2026-09-11 踩过: 报 43cm/132°, 其实两者模长都是 22.6cm)。
def _inv_T_oh(g):
    q = g[3:7] / np.linalg.norm(g[3:7])
    w, x, y, z = q
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                  [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                  [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
    return -R.T @ g[:3], np.r_[q[0], -q[1:]]


for side, pri in (("right", TC.PRIOR_BROOM), ("left", TC.PRIOR_PAN)):
    g = np.asarray(np.load(pri)["grasp"], np.float64)
    p_ho, q_ho = _inv_T_oh(g)
    rp, rq = E._rel(side)
    dp = float(torch.linalg.vector_norm(rp - torch.tensor(p_ho, dtype=torch.float32, device=E.device), dim=1).max())
    q0 = torch.tensor(q_ho, dtype=torch.float32, device=E.device).expand_as(rq)
    dr = float(torch.rad2deg(GE._qangle(rq, q0)).max())
    ok = dp < 0.015 and dr < 10.0          # 沉降审计, 不是精确性检查 (母带臂 IK 残差本身就有 ~0.5cm)
    print(f"[smoke] ② {side:5s} 手内复位: 掌系位置差 {dp*100:.3f}cm 姿态差 {dr:.2f}° {'✅' if ok else '❌'}")
    if not ok:
        FAIL.append(f"{side} 手内复位 {dp*100:.2f}cm/{dr:.1f}°")

zero = torch.zeros(E.num_envs, GE.ACT_DIM, device=E.device)
pin_dev, low_min, rel_after = [], [], {s: [] for s in GE.SIDES}
Fpin = {s: [] for s in GE.SIDES}
for t in range(args.steps):
    obs, rew, term, trunc, _ = E.step(zero)
    row = int(E.episode_length_buf[0])
    # ⚠ 逐 env 掩码: release_row 带 ±JITTER 抖动, 拿 env0 的行数去 max 全部 env 会把**已放手**的
    #   env 的正常漂移算进"钉住期位移" (2026-09-11 踩过: 假报 185mm)。
    m = (E.episode_length_buf < E.release_row)
    if bool(m.any()):
        d = max(float(torch.linalg.vector_norm((E._obj_pose(s)[0] - E.nominal[s][0])[m], dim=1).max())
                for s in GE.SIDES)
        pin_dev.append(d)
    if bool(E.latched[0]):
        for s in GE.SIDES:
            rp, _ = E._rel(s)
            rel_after[s].append(float(torch.linalg.vector_norm(rp - E.rel_p0[s], dim=1).max()))
    low_min.append(float((E.hand.data.body_pos_w[:, E._low_ids, 2] - E.cfg.table_top_z).min()))
    if bool(m.any()) and t > TC.K_CLOSE + 5:          # 合拢完成、仍在钉住 -> 指力应已建立
        for s in GE.SIDES:
            Fpin[s].append(E._pad_force_mat(s)[m].mean(0).cpu().numpy())

if pin_dev:
    ok = max(pin_dev) < 0.005      # 母带第0行按设计压桌下 1mm, 解穿残留几 mm 属正常
    print(f"[smoke] ③ 钉住期物体最大位移 {max(pin_dev)*1000:.3f}mm {'✅' if ok else '❌'}")
    if not ok:
        FAIL.append(f"钉住期物体动了 {max(pin_dev)*1000:.1f}mm")
for s in GE.SIDES:
    if rel_after[s]:
        # ⚠ 报中位+最大, 不要只报 max: 零动作下多数 env 稳得住、偶有一个崩掉 20cm, max 被那一个支配,
        #   于是同一台机器两次跑能差一个量级 (2026-09-11 我据此误判成"两台机器物理不同", 白查一轮)。
        v = np.sort(np.asarray(rel_after[s])) * 100
        print(f"[smoke] ④ 放手后**零动作** {s:5s} 掌系漂移 中位 {np.median(v):.2f}cm p90 {np.percentile(v,90):.2f}cm "
              f"max {v[-1]:.2f}cm  (只报数, 这正是策略要学的)")
# ④b 指垫接触力 (钉住期, 合拢后): 这是"被动抓握夹没夹住"的直读量
for s in GE.SIDES:
    if Fpin[s]:
        f = np.mean(Fpin[s], axis=0)
        n_on = int((f > TC.PAD_FTH).sum())
        need = 3 if s == "right" else TC.PAN_PADS_MIN
        print(f"[smoke] ④b {s:5s} 钉住期指垫力 (N) 拇/食/中/无名/小 = {np.round(f, 2)} | "
              f">{TC.PAD_FTH}N 的垫 {n_on} 个 (判据要 {need})" + ("  ⚠ 被动抓握不成立" if n_on < need else "  ✅"))

if TC.USE_TABLE:
    # ⚠ 必须在**钉住瞬间**读 (E._pin() 后不推物理)。在 E.step() 之后读的是过了物理子步的值 ——
    #   钉住是每子步回写, 最后一个子步之后物体会弹开几毫米 (实测扫把 +7mm), 拿它对母带设计值
    #   会得到假失败 (2026-09-11 踩过)。奖励里用的正是那个物理真实值, 这里审计的是"母带→env"这条链。
    E._pin()
    g = {sd: float(E._table_gap(sd).mean()) * 1000 for sd in GE.SIDES}
    tgt = {sd: TC.TABLE_TARGET[sd] * 1000 for sd in GE.SIDES}
    dev = max(abs(g[sd] - tgt[sd]) for sd in GE.SIDES)
    okt = dev < 0.5
    print(f"[smoke] ⑥ 贴桌量 (钉住瞬间, 对母带设计值): 扫把 {g['right']:+.2f}mm (目标 {tgt['right']:+.0f}) | "
          f"簸箕 {g['left']:+.2f}mm (目标 {tgt['left']:+.0f}) | 最大偏差 {dev:.2f}mm {'✅' if okt else '❌'}")
    if not okt:
        FAIL.append(f"贴桌量偏母带设计 {dev:.2f}mm")

ok = min(low_min) > -TC.TABLE_MARGIN
print(f"[smoke] ⑤ 全程 手/垫 离桌最低 {min(low_min)*100:+.2f}cm (死线 {-TC.TABLE_MARGIN*100:+.1f}cm) {'✅' if ok else '❌'}")
if not ok:
    FAIL.append(f"手/垫低于桌面 {min(low_min)*100:.2f}cm")
print(f"[smoke] 死亡分布: rel={int((E.die_kind==1).sum())} drop={int((E.die_kind==2).sum())} table={int((E.die_kind==3).sum())} / {E.num_envs} env")
print("\n[smoke] " + ("PASS ✅" if not FAIL else "FAIL ❌ " + " | ".join(FAIL)), flush=True)
try: _slot.release()
except Exception: pass
os._exit(0 if not FAIL else 1)
