"""第十八件自检 (L5-32, 2026-08-31): `goal` 臂 (POUR_GOAL_ONLY) —— 验行为不验旗。

★ 立此自检的由来:
`goal` = "只有目标, 没有轨迹"。它要同时关掉**四处**读参考的东西, 每一处漏掉都会
让这条臂静默地变成"半个 straight", 而日志、横幅、TB 全都正常:
  ① adv    时钟推进奖 (稠密, 按参考逐行发)
  ② leash  离参考罚
  ③ r_shape 人手形状指引 (读人手参考; 漏掉它 goal 就还分 oh/o 两版)
  ④ D3_dev 离参考 35cm 死线
外加**最容易漏的第五处**: `self.row = IA0 + PB.k` —— 时钟决定前馈取母带哪一行。
不冻时钟, "没有轨迹"就是假的: 策略照旧按母带的时间表拿到逐行前馈, 只是不为
跟随它付钱而已。写这个旗时这一条是我最后才想到的。

反向同样要验: **旗关着时必须与基线逐位一致**, 否则这个旗会污染另外五条臂。

★ 覆盖边界: 纯数值, 不启 Isaac。验的是 PourProgressBatch 的行为与 pour_env 的
接线文本, **不验物理** —— goal 能不能学出倒水只有训练能回答 (预期它很难, 那是
消融的**结果**, 不是故障)。

用法: python selftest_goal_only.py   (退出码 0 = 全绿)
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import progress_batch as PBM                                       # noqa: E402
from progress import D3_DEV, M2_HOLD                               # noqa: E402

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


N = 6


def build(goal):
    p = PBM.PourProgressBatch(npz_path=REF, num_envs=N, device="cpu",
                              mouth_local_bot=mouth(1, 0.087),
                              mouth_local_cup=mouth(0, 0.066),
                              no_hand_ref=False, goal_only=goal)
    p.enter(torch.arange(N), torch.full((N,), 200, dtype=torch.long),
            torch.ones(N, dtype=torch.bool), torch.ones(N, dtype=torch.bool),
            torch.zeros(N, dtype=torch.bool), torch.zeros(N, dtype=torch.bool))
    return p


def run(p, obj_offset=np.zeros(3), steps=30):
    """喂'物体恰好落在参考行上(+可选偏移)'的位姿, 收集 adv/leash/wage/k。"""
    acc = {"adv": 0.0, "leash": 0.0, "wage": 0.0, "ms": 0.0}
    for _ in range(steps):
        k = int(p.k[0])
        o0 = p.ref_obj[0][k].clone().unsqueeze(0).repeat(N, 1)
        o1 = p.ref_obj[1][k].clone().unsqueeze(0).repeat(N, 1)
        o1[:, :3] += torch.tensor(obj_offset, dtype=torch.float32)
        out = p.step(o0, o1, torch.zeros(N, 7), torch.zeros(N, 7),
                     torch.ones(N, dtype=torch.bool),
                     torch.zeros(N, 3), torch.zeros(N, 3))
        for kk in acc:
            acc[kk] += float(out[kk].sum())
    return acc, int(p.k[0]), out


print("=" * 78)
print("[selftest] `goal` 臂 (POUR_GOAL_ONLY) — 逐项验行为")
print("=" * 78)

pb_base, pb_goal = build(False), build(True)
a_base, k_base, _ = run(pb_base)
a_goal, k_goal, _ = run(pb_goal)

# ---- ① 时钟必须冻死 (最容易漏的一条) ----
print("\n① ★时钟冻死 —— `row = IA0 + k`, 不冻它'没有轨迹'就是假的")
# ★出生行是 200(enter 给的), 所以"冻死"= k 恒等于 200, **不是 0**。
#   第一版我写成了 `k_goal == 0` —— 红的是测试不是代码。同一类错误
#   (拿错基准当"什么都没发生")在本项目已犯过三次, 见 selftest_axis_assumption。
BORN = 200
chk(k_base > BORN, "基线的时钟确实在走(对照, 否则本用例无意义)",
    f"基线 30 步后 k={k_base} (出生行 {BORN})")
chk(k_goal == BORN, "goal 的时钟一步不走",
    f"goal 30 步后 k={k_goal} (出生行 {BORN}) —— 走了就说明前馈仍在按母带时间表"
    f"推进, 策略照旧拿到逐行参考, 只是不为跟随它付钱")

# ---- ② adv / leash 归零 ----
print("\n② 读参考的奖励项归零")
chk(a_base["adv"] > 0 and a_goal["adv"] == 0.0, "adv(时钟推进奖) 基线>0 → goal=0",
    f"基线 {a_base['adv']:.1f} → goal {a_goal['adv']:.1f}")
# leash 要在**偏离参考**时才非零 —— 用零偏移测不出来, 必须给偏移
# ★两段跑法是必须的: `lb`(持握基线) 在 G1 后**第一个受管步**捕获
#   `lb = 物体位姿 − 参考位姿`, 即"换基"。若一上来就带偏移, 偏移会被当成基线
#   一起吃掉, leash 恒为 0 —— 基线组也测不出来, 用例整个失效。
#   所以: 先无偏移跑 3 步让它换基, **之后**再施加偏移。
pb_b2, pb_g2 = build(False), build(True)
for _p in (pb_b2, pb_g2):
    run(_p, steps=3)                       # ← 先换基
lb, _, _ = run(pb_b2, obj_offset=np.array([0.20, 0.0, 0.0]))
lg, _, _ = run(pb_g2, obj_offset=np.array([0.20, 0.0, 0.0]))
chk(lb["leash"] < 0 and lg["leash"] == 0.0,
    "leash(离参考罚) 基线<0 → goal=0",
    f"换基后再偏离 20cm: 基线 {lb['leash']:.2f} → goal {lg['leash']:.2f}")

# ---- ③ 不读参考的项必须**保留** ----
print("\n③ 不读参考的项必须保留(去多了就不是'只去掉轨迹')")
chk(a_goal["wage"] == a_base["wage"],
    "wage(抓握维持费) 两边相同",
    f"基线 {a_base['wage']:.2f} / goal {a_goal['wage']:.2f} —— wage 的条件是 "
    f"`pads3 & ~g2 & active`, **一个字都不读参考**, 去掉它是错的")

# ---- ④ D3_dev 死线关掉 ----
print("\n④ D3_dev(离参考 35cm 死线) 必须关掉")
pb_b3, pb_g3 = build(False), build(True)
far = np.array([D3_DEV * 2.0, 0.0, 0.0])
run(pb_b3, obj_offset=far, steps=3)
run(pb_g3, obj_offset=far, steps=3)
chk(bool(pb_b3.done.all()), "基线: 离参考 70cm 会死(对照)",
    f"done={int(pb_b3.done.sum())}/{N}")
chk(not bool(pb_g3.done.any()), "goal: 同样位置**不死**",
    f"done={int(pb_g3.done.sum())}/{N} —— 没有参考就没有'离参考太远'这回事")

# ---- ⑤ ★旗关着时必须与基线逐位一致 (防污染另外五条臂) ----
print("\n⑤ ★旗关着时与基线逐位一致(防污染另外五条臂)")
pb_off = build(False)
a_off, k_off, _ = run(pb_off)
chk(all(abs(a_off[k] - a_base[k]) < 1e-9 for k in a_off) and k_off == k_base,
    "goal_only=False 与基线**逐位相同**",
    f"adv/leash/wage/ms 与 k 全部一致 (k={k_off})")

# ---- ⑥ 接线: r_shape 与 ARM_FREE 断言 ----
print("\n⑥ env 侧接线 (r_shape 关掉 + 必须同时开 ARM_FREE)")
src = open(os.path.join(_HERE, "..", "..", "C_Wiring", "pour_env.py"),
           encoding="utf-8").read()
chk("if self.hand_dh is not None and not self.goal_only:" in src,
    "r_shape(人手形状指引) 在 goal 下关掉",
    "漏掉它 goal 就还分 oh/o 两版 —— 而'只有目标'里不该有人手轨迹")
chk('assert self.arm_free' in src and 'POUR_GOAL_ONLY' in src,
    "goal_only 必须同时 arm_free, 否则**启动即断言失败**",
    "没有物体轨迹就没有可解的 IK; 只去轨迹不去臂参考 = 还在用轨迹算出来的东西")
# ★顺序检查 (2026-08-31 实测踩到): 第一版把 goal_only 块放在 self.arm_free 赋值
#   **之前**, 断言当场 AttributeError —— 它确实拦住了, 但拦的理由是错的, 而且
#   任何"这段文本在不在"的检查都照样绿。文本自检抓不到顺序, 必须显式比位置。
#   ★两次都栽在同一件事的**不同侧**, 所以这里两组顺序都要验:
#     第一次: 整块放在 arm_free 赋值之前 → `assert self.arm_free` AttributeError
#     第二次: 整块搬到 arm_free 之后     → PB 构造处 `self.goal_only` AttributeError
#   赋值与断言**位置不同, 不能整块搬** —— 赋值要早(PB 构造前), 断言要晚(arm_free 后)。
for _nm, _set, _use in (
        ("arm_free", "self.arm_free = os.environ.get", "assert self.arm_free"),
        ("goal_only", "self.goal_only = os.environ.get", "goal_only=self.goal_only")):
    _i_set, _i_use = src.find(_set), src.find(_use)
    chk(0 <= _i_set < _i_use, f"★`self.{_nm}` 的**赋值**在它的首次使用之前",
        f"赋值@{_i_set} 使用@{_i_use}"
        + ("" if 0 <= _i_set < _i_use else " —— AttributeError, 而且报的是"
           "'属性不存在', 完全不指向真正的问题"))
chk('goal_only=self.goal_only' in src, "旗确实传给了 PourProgressBatch",
    "传漏了就是'横幅打了但什么都没变'")

# ---- ⑦ 指纹 ----
print("\n⑦ 世界指纹")
try:
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "C_Wiring"))
    import world_fingerprint as WF
    chk("switches.goal_only" in WF.CRITICAL, "switches.goal_only 在 CRITICAL 里",
        "冻时钟 = 前馈恒取 IA0 行, 环境动力学与其它臂完全不同; "
        "而 criteria.digest 抓不到它(判据常量一个没变)")
except Exception as e:
    chk(False, "switches.goal_only 在 CRITICAL 里", f"{type(e).__name__}: {e}")

print("\n" + "=" * 78)
print("★ 覆盖边界: 纯数值, 不启 Isaac。**不验** goal 能不能学出倒水 ——")
print("  预期它很难甚至为 0, 那是消融的**结果**(说明物体轨迹是必需的), 不是故障。")
print("★ 判读提示: goal 若也能到非零成功率, 说明前面五条臂的物体轨迹先验")
print("  没有它们看起来那么必要 —— 那会是个比'goal=0'更有意思的结果。")
print("=" * 78)
if _fail:
    print(f"[selftest] ❌ 失败 {len(_fail)} 项: {_fail}")
    sys.exit(1)
print("[selftest] ✅ 全绿")
