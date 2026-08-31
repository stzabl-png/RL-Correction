"""第十七件自检 (L5-32, 2026-08-31): 新主判据 G3_pour ∧ placed —— 验行为不验旗。

★ 立此自检的由来:
2026-08-31 把主判据从 `G3(3D距) ∧ clock_done` 换成 `G3_pour(倒水几何) ∧ placed`。
换判据是本项目最贵的操作 —— 它同时改了**记账**和**奖励地形**(G3 付 MS_REWARD[3]=10),
而且让全部历史数字作废。L5-27 的教训是"判据可以比参考自身还严而没人发现",
L5-31 的教训是"主判据可以被红档刷而曲线看着很漂亮"。两次都不是代码写错,
是**判据本身说的不是我们以为的那件事**。所以这一件专门验判据的语义。

覆盖:
  ① G3_pour 拒绝"横杵": 倾角够、3D距也够近, 但瓶口在杯口**下方** ⟹ 必须不算
  ② G3_pour 拒绝"太远/太高": 水平>8cm 或 dz>8cm ⟹ 必须不算
  ③ G3_pour 接受真倒水几何(照人类重建实测的方位)
  ④ g3_loose ⊋ g3: 影子严格更松, 且**存在**只过松判据的姿态(否则影子无意义)
  ⑤ 主判据 = g3 ∧ placed, 单独任一项都不算成功
  ⑥ ★同尺性(本次换判据的**根本目的**): 拿 v3 与 v3noconf 两条不同母带各建一个
     PB, 喂**完全相同**的物体位姿序列, g3/placed/success 必须逐位相同
  ⑦ 可达性(防 L5-27 重演): 两条母带自身都能连续满足 G3_pour >= M2_HOLD 步,
     且末态满足 placed

★ 覆盖边界: 纯数值, 不启 Isaac。验的是判据的**语义**, 不验策略能不能做到,
也不验奖励整形是否把策略引向这里 —— 那只有训练能回答。

用法: python selftest_pour_criterion.py   (退出码 0 = 全绿)
"""
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from progress import (M2_TILT, M2_HOLD, M2_STRICT_HORIZ,          # noqa: E402
                      M2_STRICT_DZ_LO, M2_STRICT_DZ_HI, M3_HOLD)
import progress_batch as PBM                                       # noqa: E402

_fail = []


def chk(ok, name, detail):
    print(f"  [{'OK' if ok else 'XX'}] {name}: {detail}", flush=True)
    if not ok:
        _fail.append(name)


REF = os.path.join(_HERE, "..", "L2_Reference", "pour17_reference_%s.npz")
UP = np.array([0.0, 1.0, 0.0])
HALF_B, HALF_C = 0.087, 0.066


def q2R(q):
    q = np.asarray(q, np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def quat_axis(axis, deg):
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2.0
    return np.array([math.cos(h), *(a * math.sin(h))])


def mouth_local(nm, oi, half):
    z = np.load(REF % nm, allow_pickle=True)
    R = q2R(np.asarray(z[f"obj_quat_{oi}"], np.float64)[0])
    up = np.array([0.0, half, 0.0])
    return up if (R @ up)[2] > (R @ (-up))[2] else -up


def build(nm, n=8):
    return PBM.PourProgressBatch(
        npz_path=REF % nm, num_envs=n, device="cpu",
        mouth_local_bot=mouth_local(nm, 1, HALF_B),
        mouth_local_cup=mouth_local(nm, 0, HALF_C), no_hand_ref=False)


print("=" * 78)
print("[selftest] 新主判据 G3_pour ∧ placed — 逐项验语义")
print("=" * 78)

pb = build("v3")
CUP_P = pb.rest[0][:3].numpy().astype(np.float64)
CUP_Q = pb.rest[0][3:7].numpy().astype(np.float64)
MC_W = CUP_P + q2R(CUP_Q) @ pb.mc.numpy().astype(np.float64)
MB_L = pb.mb.numpy().astype(np.float64)


def bottle_for(dz, horiz, tilt_deg, azim=0.0):
    """造一个瓶位姿, 使瓶口相对杯口恰好是 (水平 horiz, 竖直 dz), 倾角 tilt_deg。"""
    q = quat_axis([1.0, 0.0, 0.0], 90.0)                  # 长轴竖直
    q = PBM_qmul(q, quat_axis(UP, azim))
    q = PBM_qmul(q, quat_axis([1.0, 0.0, 0.0], tilt_deg))  # 绕垂直轴倾倒
    R = q2R(q)
    want = MC_W + np.array([horiz, 0.0, dz])              # 想让瓶口落在这
    return np.concatenate([want - R @ MB_L, q])


def PBM_qmul(p, q):
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])


def geom_ok(dz, horiz, tilt_deg):
    """直接照判据算一遍(不经过 PB), 用于逐姿态判定。"""
    b = bottle_for(dz, horiz, tilt_deg)
    R = q2R(b[3:7])
    v = R @ UP
    t = math.acos(max(-1.0, min(1.0, v[2] / np.linalg.norm(v))))
    d = (b[:3] + R @ MB_L) - MC_W
    hz = float(np.linalg.norm(d[:2]))
    strict = (t >= M2_TILT and hz <= M2_STRICT_HORIZ
              and M2_STRICT_DZ_LO <= d[2] <= M2_STRICT_DZ_HI)
    loose = (t >= M2_TILT and float(np.linalg.norm(d)) <= float(pb.mgate))
    return strict, loose, math.degrees(t), hz, float(d[2])


# ---- 前置: 造姿态的算子本身没错(否则整套用例失效) ----
print("\n前置 · 姿态构造器可信")
_s, _l, _t, _hz, _dz = geom_ok(0.04, 0.05, 100.0)
chk(abs(_t - 100.0) < 1e-6 and abs(_hz - 0.05) < 1e-6 and abs(_dz - 0.04) < 1e-6,
    "要什么几何就造出什么几何",
    f"要 (dz 4cm, 水平 5cm, 倾角 100°) → 实得 ({_dz*100:.3f}cm, "
    f"{_hz*100:.3f}cm, {_t:.3f}°)")

# ---- ① 拒绝横杵 ----
print("\n① 拒绝'横杵': 倾角够、3D距也近, 但瓶口在杯口**下方**")
for dz in (-0.02, -0.05, -0.09):
    s, l, t, hz, _ = geom_ok(dz, 0.05, 100.0)
    chk((not s) and l, f"dz={dz*100:+.0f}cm 水平5cm 倾角100° → 新判据拒、旧判据收",
        f"新 {'过' if s else '拒'} / 旧 {'过' if l else '拒'} "
        f"(3D距 {math.hypot(0.05, dz)*100:.1f}cm ≤ 12cm ⟹ 旧口径照收)")

# ---- ② 拒绝太远 / 太高 ----
print("\n② 拒绝'太远'与'太高'")
s, _, _, _, _ = geom_ok(0.04, 0.10, 100.0)
chk(not s, "水平 10cm > 8cm → 拒", "横向凑不够近不算倒进杯里")
s, _, _, _, _ = geom_ok(0.12, 0.05, 100.0)
chk(not s, "dz 12cm > 8cm → 拒", "举太高也不算(会洒)")
s, _, _, _, _ = geom_ok(0.04, 0.05, 80.0)
chk(not s, "倾角 80° < 90° → 拒", "没倒过去")

# ---- ③ 接受真倒水几何 ----
print("\n③ 接受真倒水几何(照人类重建实测方位: dz +1~+4cm, 水平 5~7cm)")
for dz, hz_, td in ((0.0125, 0.056, 95.0), (0.0385, 0.054, 104.0),
                    (0.03, 0.065, 120.0)):
    s, _, _, _, _ = geom_ok(dz, hz_, td)
    chk(s, f"dz {dz*100:+.2f}cm 水平 {hz_*100:.1f}cm 倾角 {td:.0f}° → 过",
        "人类基准(v1)=dz+1.25/水平5.62; v3母带=dz+3.85/水平5.42")

# ---- ④ 影子严格更松, 且非空 ----
print("\n④ g3_loose ⊋ g3 (影子必须更松, 且**存在**只过松判据的姿态)")
grid = [(dz, hz_, td) for dz in np.linspace(-0.10, 0.12, 12)
        for hz_ in np.linspace(0.0, 0.12, 7) for td in (85.0, 95.0, 130.0)]
res = [geom_ok(*g)[:2] for g in grid]
viol = [g for g, (s, l) in zip(grid, res) if s and not l]
only_loose = [g for g, (s, l) in zip(grid, res) if l and not s]
chk(not viol, "严格 ⟹ 松 (无反例)",
    f"扫 {len(grid)} 个姿态, 违反 {len(viol)} 个 —— 有反例说明影子不是'更松'")
chk(len(only_loose) > 0, "存在只过松判据的姿态(影子有信息量)",
    f"{len(only_loose)}/{len(grid)} 个 —— sr/g3_loose − sr/gate3 就是这一批, "
    f"它量的正是'旧口径里有多少是横杵刷出来的'")

# ---- ⑤ 主判据 = g3 ∧ placed ----
print("\n⑤ 主判据 = g3 ∧ placed, 单独任一项都不算成功")
src = open(os.path.join(_HERE, "progress_batch.py"), encoding="utf-8").read()
chk("_succ = self.g3[env_ids] & _plc" in src, "记账里 succ 确实是 g3 ∧ placed",
    "写死在 reset_idx 里; 改成单项会让本行找不到")
chk('"sr/success"' in src and '"sr_t0/success"' in src, "两个口径都在报",
    "sr_t0 才是无课程稀释的真口径")
chk('"sr/clock_done"' in src, "clock_done 仍在报(降级为诊断)",
    "留着是为了看'走完轨迹但没倒水'还在不在发生")
chk("_succ = self.g3[env_ids] & _cdone" not in src, "旧的 g3∧clock_done 已拆除",
    "留着会有两个都叫 success 的量")

# ---- ⑥ ★同尺性: 两条母带喂同样的物体轨迹, 判据结果必须逐位相同 ----
print("\n⑥ ★同尺性 — 换母带不换尺 (本次换判据的根本目的)")
N = 6
pbs = {nm: build(nm, N) for nm in ("v3", "v3noconf")}
chk(np.allclose(pbs["v3"].rest[1].numpy(), pbs["v3noconf"].rest[1].numpy())
    and np.allclose(pbs["v3"].rest[0].numpy(), pbs["v3noconf"].rest[0].numpy()),
    "两条母带的 rest(第0帧) 逐位相同",
    "rest 是 placed 的唯一参照 —— 不同就等于两把尺")

for p in pbs.values():
    p.enter(torch.arange(N), torch.full((N,), 200, dtype=torch.long),
            torch.ones(N, dtype=torch.bool), torch.ones(N, dtype=torch.bool),
            torch.zeros(N, dtype=torch.bool), torch.zeros(N, dtype=torch.bool))

# 同一条"先倒水 hold, 再放回 hold"的物体轨迹, 喂给两个 PB
POUR = bottle_for(0.035, 0.055, 105.0)
REST_B = pbs["v3"].rest[1].numpy().astype(np.float64)
seq = [POUR] * (M2_HOLD + 3) + [REST_B] * (M3_HOLD + 3)
cup_t = torch.tensor(np.tile(np.concatenate([CUP_P, CUP_Q]), (N, 1)),
                     dtype=torch.float32)
out = {}
for nm, p in pbs.items():
    for b in seq:
        p.step(cup_t, torch.tensor(np.tile(b, (N, 1)), dtype=torch.float32),
               torch.zeros(N, 7), torch.zeros(N, 7),
               torch.ones(N, dtype=torch.bool), torch.zeros(N, 3),
               torch.zeros(N, 3))
    out[nm] = (p.g3.clone(), p.placed.clone())
chk(bool(out["v3"][0].all()), "喂真倒水几何 → G3_pour 达成",
    f"{int(out['v3'][0].sum())}/{N} env (hold {M2_HOLD} 步)")
chk(bool(out["v3"][1].all()), "再放回 rest → placed 达成",
    f"{int(out['v3'][1].sum())}/{N} env (hold {M3_HOLD} 步)")
chk(bool((out["v3"][0] == out["v3noconf"][0]).all()
         and (out["v3"][1] == out["v3noconf"][1]).all()),
    "★两条母带对**同一段物体轨迹**判出完全相同的 g3/placed",
    "这就是'六条臂第一次可以直接比'的全部依据 —— 若红, 说明判据仍在读母带形状")

# ---- ⑦ 可达性: 母带自身能过 (防 L5-27 重演) ----
print("\n⑦ 可达性 — 母带自身满足新判据 (L5-27: 判据比参考自身还严)")
for nm in ("v3", "v3noconf"):
    z = np.load(REF % nm, allow_pickle=True)
    Pb, Qb = np.asarray(z["obj_pos_1"], float), np.asarray(z["obj_quat_1"], float)
    Pc, Qc = np.asarray(z["obj_pos_0"], float), np.asarray(z["obj_quat_0"], float)
    mbl, mcl = mouth_local(nm, 1, HALF_B), mouth_local(nm, 0, HALF_C)
    best = cur = 0
    for i in range(len(Pb)):
        R1 = q2R(Qb[i])
        v = R1 @ UP
        t = math.acos(max(-1.0, min(1.0, v[2] / np.linalg.norm(v))))
        d = (Pb[i] + R1 @ mbl) - (Pc[i] + q2R(Qc[i]) @ mcl)
        ok = (t >= M2_TILT and float(np.linalg.norm(d[:2])) <= M2_STRICT_HORIZ
              and M2_STRICT_DZ_LO <= d[2] <= M2_STRICT_DZ_HI)
        cur = cur + 1 if ok else 0
        best = max(best, cur)
    chk(best >= M2_HOLD, f"{nm}: 母带连续满足 G3_pour {best} 行 (需 {M2_HOLD})",
        f"不够就是'判据比参考自身还严', 策略再好也过不去 —— L5-27 就栽在这")
    dend = float(np.linalg.norm(Pb[-1] - Pb[0]))
    chk(dend <= 0.03, f"{nm}: 母带末态回到 rest ({dend*100:.2f}cm ≤ 3cm)",
        "placed 要求物体放回原位; 母带末态不满足就白搭")

print("\n" + "=" * 78)
print("★ 覆盖边界: 纯数值自检, 不启 Isaac。验判据的**语义**(什么该过什么不该过、")
print("  换母带是否同尺、母带自身可不可达), **不验**策略能否做到, 也不验奖励整形")
print("  是否把策略引向这里 —— 那只有训练能回答。")
print("★ 已知不对称(不是缺陷, 是消融本身的一部分, 报结果时必须一起写):")
print("  母带满足 G3_pour 的最长连续行数 v3=42 / v3noconf=89 —— flat 的模仿目标")
print("  把倒水几何保持得久 2.1 倍, 留给策略的余量更大。")
print("=" * 78)
if _fail:
    print(f"[selftest] ❌ 失败 {len(_fail)} 项: {_fail}")
    sys.exit(1)
print("[selftest] ✅ 全绿")
