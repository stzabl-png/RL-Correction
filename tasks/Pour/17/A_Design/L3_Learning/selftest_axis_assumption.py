"""第十四件自检 (L5-31): 轴对称假设 —— 把一个沉默的物体前提钉成可复现的事实。

★ 立此自检的由来:
`_axis_only_R`(build_ref_v5 反解 IK) 与 `_tilt` / `_axis_tilt`(皮筋 rot 项、G3 倒水、
placed、G4、死线 D2_tilt) **全链**都丢弃"绕物体长轴的自转"。对瓶/杯是正确的
(两者轴对称, 绕长轴转多少不可观测)。但这个前提:

  · 写死在两处不同的文件里 (build_ref_v5.UP_LOCAL 与 PourProgress 的 up_local)
  · 杯子那一路 (`self.upc`) 连参数都没开, 是硬编码的 (0,1,0)
  · 换非轴对称物体 (带把手的杯、勺、盒、壶嘴瓶) 时**判据静默失效** ——
    策略可以把物体绕长轴转任意角度, 所有 Gate 照样全绿, 没有任何检查会红
  · 在 2026-08-30 之前, 世界指纹里**没有任何一项能看见它**

本自检不"修"这个盲区 (对当前物体它是正确设计), 而是:
  ① 证明盲区确实存在, 且范围就是"绕长轴" —— 让它变成有记录的已知限制
  ② 证明 up_local 是**承重**的 (换一个轴, 判据输出就变) —— 防止有人以为它是装饰
  ③ 证明判据对"垂直于长轴的旋转"确实敏感 —— 防止把①误读成"判据根本不判朝向"

用法: python selftest_axis_assumption.py   (退出码 0 = 全绿)
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from progress import _axis_tilt, M2_TILT, M3_ROT, M4_DIST_ROT  # noqa: E402

_fail = []


def chk(ok, name, detail):
    print(f"  [{'OK' if ok else 'XX'}] {name}: {detail}", flush=True)
    if not ok:
        _fail.append(name)


def quat_axis(axis, deg):
    """绕局部 axis 转 deg 度的四元数 (wxyz, 与全项目一致)."""
    a = np.asarray(axis, np.float64)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2.0
    return np.array([math.cos(h), *(a * math.sin(h))])


def qmul(p, q):
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


UP = np.array([0.0, 1.0, 0.0])          # 瓶/杯的长轴 (局部系), 与 env 一致
PERP = np.array([1.0, 0.0, 0.0])        # 任取一条垂直于长轴的局部轴

# ★ 基准必须是"瓶子立着", 不是单位四元数。
#   `_axis_tilt` 量的是"局部长轴映到世界后与**世界竖直轴 Z**的夹角",
#   而长轴是局部 (0,1,0) —— 单位四元数下它是**水平**的, 倾角 90°。
#   (我第一版自检就把单位四元数当成了"立着", 三条用例全红; 红的是测试不是代码。)
Q0 = quat_axis(PERP, 90.0)              # 绕局部X转90°: 长轴(0,1,0) -> 世界(0,0,1) 竖直

print("=" * 78)
print("[selftest] 轴对称假设 — 盲区的范围、承重性、与敏感性")
print("=" * 78)
_b0 = math.degrees(_axis_tilt(Q0, UP))
chk(abs(_b0) < 1e-6, "基准姿态 = 长轴竖直",
    f"倾角 {_b0:.4f}° (单位四元数下长轴是水平的, 倾角 90° —— 基准取错整套用例失效)")

# ---- ① 盲区确实存在: 绕长轴自转, 倾角判据一动不动 ----
print("\n① 盲区: 绕长轴的自转对所有倾角判据不可见")
base = _axis_tilt(Q0, UP)
worst = 0.0
for deg in (15, 30, 90, 137, 180, 359):
    q = qmul(Q0, quat_axis(UP, deg))
    t = _axis_tilt(q, UP)
    worst = max(worst, abs(t - base))
chk(worst < 1e-9, "绕长轴自转 → 倾角恒定",
    f"自转 15°~359° 全部试过, 倾角最大变化 {math.degrees(worst):.2e}° "
    f"⟹ 判据看不见自转 (**这是已知限制, 不是 bug**)")

# 同一件事在三个真判据上的后果, 逐条写明白
q_spin = qmul(Q0, quat_axis(UP, 180.0))
for nm, thr in (("G3 倒水 M2_TILT", M2_TILT), ("placed M3_ROT", M3_ROT),
                ("G4 M4_DIST_ROT", M4_DIST_ROT)):
    same = abs(_axis_tilt(q_spin, UP) - base) < 1e-9
    chk(same, f"{nm} 对 180° 自转不变",
        f"阈值 {math.degrees(thr):.0f}°; 物体倒转 180° 后该判据的输入一字不变")

# ---- ② up_local 是承重的: 换一个轴, 判据就变 ----
print("\n② 承重性: up_local 不是装饰, 换轴会改变判据输出")
# 单挑一个旋转不够: 长轴竖直时, 绕它自转对**任何**垂直轴的"与竖直夹角"都不变,
# 挑错用例会得出"换轴没影响"的假阴性 (我第一版就是这么红的)。改成扫一遍姿态,
# 报两种 up_local 判出来的倾角的最大分歧。
alt = np.array([0.0, 0.0, 1.0])         # 假装长轴是局部 Z
worst_ax, worst_q = 0.0, None
for ax in ([1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.], [1., 0., 1.]):
    for deg in range(0, 360, 15):
        q = qmul(Q0, quat_axis(ax, deg))
        d = abs(_axis_tilt(q, UP) - _axis_tilt(q, alt))
        if d > worst_ax:
            worst_ax, worst_q = d, (ax, deg)
chk(worst_ax > math.radians(30.0), "换 up_local 判出来的倾角会大幅分歧",
    f"扫 5 轴 × 24 角, 最大分歧 {math.degrees(worst_ax):.1f}° "
    f"(出现在绕 {worst_q[0]} 转 {worst_q[1]}°) ⟹ 声明哪个轴, 直接决定判什么")

# ---- ③ 敏感性: 垂直于长轴的旋转必须被判到 (防止把①误读成"根本不判朝向") ----
print("\n③ 敏感性: 垂直于长轴的旋转确实被判到")
for deg in (10, 45, 90):
    q = qmul(Q0, quat_axis(PERP, deg))                # 绕局部 X = 垂直于长轴
    got = math.degrees(_axis_tilt(q, UP))
    chk(abs(got - deg) < 1e-6, f"绕垂直轴 {deg}° → 倾角 {deg}°",
        f"实测 {got:.4f}° ⟹ 真正的倾倒/倒水动作照常可判")

# ---- ④ 组合: 自转 + 倾倒, 只有倾倒那一半算数 ----
print("\n④ 组合: 自转叠加倾倒时, 只有倾倒分量进入判据")
q = qmul(qmul(Q0, quat_axis(UP, 123.0)), quat_axis(PERP, 45.0))
got = math.degrees(_axis_tilt(q, UP))
chk(abs(got - 45.0) < 1e-6, "自转123° + 倾倒45° → 判据只看到 45°",
    f"实测 {got:.4f}°")

# ---- ⑤ 指纹必须能看见这个前提 (2026-08-30 之前看不见) ----
print("\n⑤ 世界指纹必须记录这个前提")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", "C_Wiring"))
    import world_fingerprint as WF
    keys = set(WF.CRITICAL) | set(WF.WARN)
    need = {"geometry.up_local_bot", "geometry.up_local_cup",
            "geometry.axis_spin_ignored"}
    chk(need <= keys, "指纹含 geometry.* 三项",
        f"缺 {sorted(need - keys)}" if need - keys else
        "换了长轴定义的 ckpt 不再会静默'匹配'")
except Exception as e:
    chk(False, "指纹含 geometry.* 三项", f"读不到 world_fingerprint: {type(e).__name__}: {e}")

print("\n" + "=" * 78)
print("★ 覆盖边界 (照 selftest_collide 的规矩显式声明):")
print("  · 本自检只覆盖**判据侧**的轴对称假设 (_axis_tilt)。")
print("    参考生成侧 build_ref_v5._axis_only_R 用的是同一个前提, 但它是离线脚本、")
print("    常量独立, 本自检**证明不了两边一致** —— 一致性由 pour_env 启动时对母带")
print("    meta 里的 up_local 声明做断言来保证 (母带未声明时明示'未知', 不算已核对)。")
print("  · 本自检**不判断物体是不是真的轴对称** —— 那需要网格。瓶/杯成立是人裁的。")
print("=" * 78)

if _fail:
    print(f"[selftest] ❌ 失败 {len(_fail)} 项: {_fail}")
    sys.exit(1)
print("[selftest] ✅ 全绿")
