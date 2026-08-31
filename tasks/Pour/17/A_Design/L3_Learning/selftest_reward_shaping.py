"""第十九件自检 (L5-33, 2026-08-31): placed 整形棘轮 —— 验行为不验旗。

★ 立此自检的由来:
earn-only 棘轮这类奖励最典型的三种坏法, 每一种都**静默**:
  ① 可刷 —— 来回晃反复挣同一份钱 (棘轮写错成非单调就会这样)
  ② 白送 —— G3 达成瞬间把"达成前就有的 φ"一次付清 (播种步没豁免就会这样)
  ③ 污染基线 —— 旗关着时行为与基线不逐位一致 (另外五条臂全被改)
mouth 棘轮住在 pour_env(要启 Isaac), 本自检只验 progress_batch 侧的 place 棘轮 +
mouth 的接线文本; 物理行为靠训练回答。

用法: python selftest_reward_shaping.py   (退出码 0 = 全绿)
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from progress import PLACE_SHAPE_K, M3_POS  # noqa: E402

_fail = []


def chk(ok, name, detail):
    print(f"  [{'OK' if ok else 'XX'}] {name}: {detail}", flush=True)
    if not ok:
        _fail.append(name)


REF = os.path.join(_HERE, "..", "L2_Reference", "pour17_reference_v3.npz")


def q2R(q):
    q = np.asarray(q, np.float64); q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def mouth(oi, half):
    z = np.load(REF, allow_pickle=True)
    R = q2R(np.asarray(z[f"obj_quat_{oi}"], np.float64)[0])
    up = np.array([0.0, half, 0.0])
    return up if (R @ up)[2] > (R @ (-up))[2] else -up


N = 4


def build(flag):
    os.environ["POUR_PLACE_SHAPE"] = "1" if flag else "0"
    for m in ("progress_batch",):
        sys.modules.pop(m, None)
    import progress_batch as PBM
    p = PBM.PourProgressBatch(npz_path=REF, num_envs=N, device="cpu",
                              mouth_local_bot=mouth(1, 0.087),
                              mouth_local_cup=mouth(0, 0.066), no_hand_ref=False)
    p.enter(torch.arange(N), torch.full((N,), 200, dtype=torch.long),
            torch.ones(N, dtype=torch.bool), torch.ones(N, dtype=torch.bool),
            torch.ones(N, dtype=torch.bool),          # ★g3 已达成出生
            torch.zeros(N, dtype=torch.bool))
    return p


def step_at(p, pose_b):
    """瓶给定位姿, 杯钉在 rest, 返回 out。"""
    o0 = p.rest[0].unsqueeze(0).repeat(N, 1)
    o1 = torch.tensor(pose_b, dtype=torch.float32).unsqueeze(0).repeat(N, 1)
    return p.step(o0, o1, torch.zeros(N, 7), torch.zeros(N, 7),
                  torch.ones(N, dtype=torch.bool),
                  torch.zeros(N, 3), torch.zeros(N, 3))


print("=" * 78)
print("[selftest] placed 整形棘轮 — 逐项验行为")
print("=" * 78)

REST = None
_pb0 = build(True)
REST = _pb0.rest[1].numpy().astype(np.float64)


def pose(frac_dist, tilt_deg):
    """距 rest 按比例 frac_dist×D0, 倾角 tilt_deg 的瓶位姿。"""
    import math
    h = np.radians(tilt_deg) / 2
    q = np.array([np.cos(h), np.sin(h), 0.0, 0.0])   # 绕X倾斜(长轴Y→竖直用90°基准)
    # 基准: 立着 = 绕X转90°; 倾角 tilt = 90°-额外, 直接构造:
    hh = np.radians(90.0 - tilt_deg) / 2.0 if False else None
    # 简化: 用绕X转 (90°−tilt) 让长轴与竖直夹角= tilt
    a = np.radians(90.0 - tilt_deg) / 2.0
    q = np.array([np.cos(a), np.sin(a), 0.0, 0.0])
    pos = REST[:3].copy(); pos[0] += frac_dist * 0.35
    return np.concatenate([pos, q])


# ---- ① 播种不付钱 ----
print("\n① G3 达成时已有的 φ 不白送 (播种步)")
p = build(True)
o = step_at(p, pose(0.5, 60.0))          # 首个受管步
chk(float(o["place"].sum()) == 0.0, "首步(播种)支付 = 0",
    f"实付 {float(o['place'].sum()):.4f} —— 白送的是'达成 G3 前就有的进度'")

# ---- ② 进步才付钱, 幅度对 ----
print("\n② 进步付钱, 幅度 = K×Δφ")
o2 = step_at(p, pose(0.25, 30.0))        # 距离 0.5→0.25, 倾角 60→30
exp = PLACE_SHAPE_K * (0.5 * 0.25 + 0.5 * (30.0 / 90.0 - 0.0) )  # Δφ=0.125+0.1667
exp = PLACE_SHAPE_K * ((0.5 * (1-0.25) + 0.5 * (1-30/90.0))
                       - (0.5 * (1-0.5) + 0.5 * (1-60/90.0)))
got = float(o2["place"][0])
chk(abs(got - exp) < 1e-3, "支付 = K×Δφ",
    f"期望 {exp:.4f} 实得 {got:.4f}")

# ---- ③ 不可刷: 退回去再回来, 不再付钱 ----
print("\n③ earn-only: 来回晃不挣钱")
step_at(p, pose(0.5, 60.0))              # 退回
o3 = step_at(p, pose(0.25, 30.0))        # 再回来
chk(float(o3["place"].sum()) == 0.0, "回到旧位置支付 = 0",
    f"实付 {float(o3['place'].sum()):.4f} —— 非 0 就是可刷, 策略会学抖动采矿")

# ---- ④ 全程封顶 ----
print("\n④ 全程支付 ≤ K")
p4 = build(True)
tot = 0.0
for fd, td in ((0.9, 85), (0.6, 60), (0.3, 30), (0.1, 10), (0.0, 0)):
    tot += float(step_at(p4, pose(fd, td))["place"][0])
chk(tot <= PLACE_SHAPE_K + 1e-6, f"合计 {tot:.3f} ≤ 封顶 {PLACE_SHAPE_K}",
    "潜函数∈[0,1] ⟹ 总付 ≤ K, 不喧宾夺主(G3 里程碑 10)")

# ---- ⑤ 旗关 = 恒 0 且状态不漂 ----
print("\n⑤ 旗关着与基线逐位一致")
p5 = build(False)
z = 0.0
for fd, td in ((0.9, 85), (0.3, 30), (0.0, 0)):
    z += float(step_at(p5, pose(fd, td))["place"].abs().sum())
chk(z == 0.0 and float(p5.place_pot.abs().sum()) == 0.0,
    "关旗: place 恒 0 且 place_pot 不动",
    f"支付合计 {z} pot={float(p5.place_pot.abs().sum())} —— 污染基线 = 另五条臂全被改")

# ---- ⑥ mouth 接线文本 (行为要启 Isaac, 只验接线) ----
print("\n⑥ mouth 棘轮接线 (文本级)")
src = open(os.path.join(_HERE, "..", "..", "C_Wiring", "pour_env.py"),
           encoding="utf-8").read()
chk('(~_col["objobj"])' in src and "POUR_MOUTH_BONUS" in src,
    "gate 含'瓶杯无接触' (不碰撞的落地)", "碰着杯子那一步棘轮冻结不付钱")
chk("_sd = _gm & (self.mouth_pot <= 0)" in src, "mouth 也是播种不付钱", "与 place 同款")
chk('out["place"] + r_mouth' in src, "两项都进了 rew 总装", "")
chk('"place": 0.0, "mouth": 0.0' in src, "racc 两键齐 (TB 能看到 ep_rew/place|mouth)", "")
try:
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "C_Wiring"))
    import world_fingerprint as WF
    chk({"switches.place_shape", "switches.mouth_bonus"} <= set(WF.CRITICAL),
        "两旗都在指纹 CRITICAL", "跨旗回放奖励地形不同, 必须拒跑")
except Exception as e:
    chk(False, "两旗都在指纹 CRITICAL", f"{type(e).__name__}: {e}")

print("\n" + "=" * 78)
print("★ 覆盖边界: place 棘轮为纯数值行为验证; mouth 棘轮住在 env(要启 Isaac),")
print("  这里只验接线文本 —— 它的行为等价性由发射前的零动作/首段训练曲线兜底。")
print("=" * 78)
if _fail:
    print(f"[selftest] ❌ 失败 {len(_fail)} 项: {_fail}")
    sys.exit(1)
print("[selftest] ✅ 全绿")
