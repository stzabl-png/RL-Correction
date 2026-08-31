"""第十五件自检 (L5-31): POUR_CONF_FLAT 消融旗 —— 验行为不验旗。

★ 立此自检的由来:
`conf_flat` 是个"只改一处、下游六项自动跟着变"的旗。这种旗最容易出的事故是
**旗接上了、横幅打了, 而某一项没跟着变** —— 因为没有任何东西会报错。
本项目的原话: 「旗接上了 / 横幅打了 / 数出来了 / 预检绿了, 四个都不等于
"它在按你以为的方式工作"」。

所以这里逐项**验行为**: 开旗前后把六个下游量都取出来对比, 每一项都必须
说得出"什么情况下会红"。

覆盖的六项 (全部经由 tp/tr/tmix 索引):
  ① W_OBJ / W_HAND        权重
  ② LEASH_POS             位置皮筋
  ③ LEASH_ROT + rot_ban   朝向皮筋与红档禁判
  ④ 时钟门 GATE_POS       红档宽门失效
  ⑤ DEV_ARM_TIER          残差界 (在 pour_env 侧按 tmix 取, 这里验 tmix 本身)
  ⑥ regime 观测           策略是否还能看见置信度 (验 tmix 是否恒定)

★ 覆盖边界: 本自检**不覆盖母带侧**。用户裁定 flat 连母带一起重造(去掉
conf 驱动的朝向平滑), 那是 build_ref_v5 的离线产物, 由"四道出厂检查 + 母带
md5 进指纹"保证, 不在本自检范围内。

用法: python selftest_conf_flat.py   (退出码 0 = 全绿)
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from progress import (W_OBJ, W_HAND, LEASH_POS, LEASH_ROT,  # noqa: E402
                      GATE_POS, RED_GATE_POS, criteria_digest)

_fail = []


def chk(ok, name, detail):
    print(f"  [{'OK' if ok else 'XX'}] {name}: {detail}", flush=True)
    if not ok:
        _fail.append(name)


def build(flat, no_hand):
    """在给定旗下构造一个 PourProgressBatch, 返回它。

    ★ 必须**重新 import** progress_batch —— 旗是在 __init__ 里读环境变量的,
    而模块级常量不受影响; 但为了排除"模块缓存导致旗没生效"这一类假阴性,
    这里显式重载模块。
    """
    os.environ["POUR_CONF_FLAT"] = "1" if flat else "0"
    for m in ("progress_batch",):
        sys.modules.pop(m, None)
    import progress_batch as PBM
    ref = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "L2_Reference", "pour17_reference_v3.npz"))
    # mouth_local 与 pour_env 同源算法(_mouth_local, half=瓶0.087/杯0.066)
    z = np.load(ref, allow_pickle=True)
    rows = np.where(np.asarray(z["source"]) == 1)[0]

    def mouth(oi, half):
        q = np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows][0]
        w, x, y, zz = q / np.linalg.norm(q)
        R = np.array([[1-2*(y*y+zz*zz), 2*(x*y-w*zz), 2*(x*zz+w*y)],
                      [2*(x*y+w*zz), 1-2*(x*x+zz*zz), 2*(y*zz-w*x)],
                      [2*(x*zz-w*y), 2*(y*zz+w*x), 1-2*(x*x+y*y)]])
        up = np.array([0.0, half, 0.0])
        return up if (R @ up)[2] > (R @ (-up))[2] else -up

    return PBM.PourProgressBatch(
        npz_path=ref, num_envs=4, device="cpu",
        mouth_local_bot=mouth(1, 0.087), mouth_local_cup=mouth(0, 0.066),
        no_hand_ref=no_hand)


print("=" * 78)
print("[selftest] POUR_CONF_FLAT — 逐项验行为")
print("=" * 78)

try:
    import inspect
    import progress_batch as _probe
    _sig = inspect.signature(_probe.PourProgressBatch.__init__)
    _kw = set(_sig.parameters)
except Exception as e:
    print(f"[selftest] ⚠ 读不到 PourProgressBatch 签名: {type(e).__name__}: {e}")
    _kw = set()

# 构造参数名可能与本文件的假设不同 —— 先探明, 不同就明示"未验", 不假装通过
# ★ 探针要双向查: 既查"我要的参数在不在", 也查"有没有我没给的必填参数"。
#   第一版只查了前者 —— 结果被 mouth_local_bot/cup 两个必填参数当场 TypeError。
#   单向的检查看起来通过了, 其实什么都没保证。
PROVIDED = {"npz_path", "num_envs", "device", "no_hand_ref",
            "mouth_local_bot", "mouth_local_cup"}
_required = {k for k, v in _sig.parameters.items()
             if k != "self" and v.default is inspect.Parameter.empty}
_missing_in_test = _required - PROVIDED
_unknown = PROVIDED - _kw
if _missing_in_test or _unknown:
    print(f"  [ii] 实际签名 = {sorted(_kw)}")
    chk(False, "构造签名与本自检假设一致",
        f"自检没给的必填参数 {sorted(_missing_in_test)}; "
        f"自检给了但签名没有的 {sorted(_unknown)} —— **未验, 不是通过**")
    print("=" * 78)
    print("[selftest] ❌ 无法构造被测对象")
    sys.exit(1)

pb_base = build(flat=False, no_hand=False)
pb_flat = build(flat=True, no_hand=False)
pb_flat_o = build(flat=True, no_hand=True)

# ---- 前置: 旗真的被读到了 ----
print("\n前置 · 旗是否生效")
chk(getattr(pb_base, "conf_flat", None) is False, "base 的 conf_flat=False",
    f"实测 {getattr(pb_base,'conf_flat',None)}")
chk(getattr(pb_flat, "conf_flat", None) is True, "flat 的 conf_flat=True",
    f"实测 {getattr(pb_flat,'conf_flat',None)}")

# ---- ⑥ tmix 恒定 (regime 观测的来源) ----
print("\n⑥ 策略还能不能看见置信度")
u_base = sorted(set(pb_base.tmix.tolist()))
u_flat = sorted(set(pb_flat.tmix.tolist()))
chk(len(u_base) > 1, "base 的档位是变化的(否则这个消融无意义)",
    f"出现的档位 {u_base}(2=绿 1=黄 0=红)")
chk(u_flat == [1], "flat 的档位恒为黄档 1",
    f"出现的档位 {u_flat} —— 恒定 ⟹ regime 观测变常量, 策略不再知道可信度")

# ---- ① 权重 ----
print("\n① 权重 W_OBJ / W_HAND")
wo_b = [float(pb_base.WO[t]) for t in pb_base.tmix.tolist()]
wh_b = [float(pb_base.WH[t]) for t in pb_base.tmix.tolist()]
wo_f = [float(pb_flat.WO[t]) for t in pb_flat.tmix.tolist()]
wh_f = [float(pb_flat.WH[t]) for t in pb_flat.tmix.tolist()]
chk(len(set(wo_b)) > 1 and len(set(wo_f)) == 1,
    "带手: base 权重随行变, flat 恒定",
    f"base W_OBJ∈{sorted(set(wo_b))} → flat {sorted(set(wo_f))}; "
    f"base W_HAND∈{sorted(set(wh_b))} → flat {sorted(set(wh_f))}")
chk(abs(wo_f[0] - W_OBJ[1]) < 1e-9 and abs(wh_f[0] - W_HAND[1]) < 1e-9,
    "flat 的权重 = 黄档值",
    f"实测 {wo_f[0]:.2f}/{wh_f[0]:.2f}, 黄档定义 {W_OBJ[1]}/{W_HAND[1]}")

# ★ 不对称: OBJ 变体的 W_OBJ 本来就是平的
wo_fo = [float(pb_flat_o.WO[t]) for t in pb_flat_o.tmix.tolist()]
pb_base_o = build(flat=False, no_hand=True)
wo_bo = [float(pb_base_o.WO[t]) for t in pb_base_o.tmix.tolist()]
chk(len(set(wo_bo)) == 1 and len(set(wo_fo)) == 1,
    "★不对称: 不带手时 base 与 flat 的权重都是平的",
    f"base_o {sorted(set(wo_bo))} / flat_o {sorted(set(wo_fo))} "
    f"⟹ base_o vs flat_o **不差权重这一项**, 只差皮筋/门/残差界/观测。"
    f"判读 2×2 交互项时必须记住(L5-31 已写进台账)")

# ---- ② ③ 皮筋 ----
print("\n②③ 皮筋 位置 / 朝向")
lp_b = [float(pb_base.LP[t]) for t in pb_base.tp[1].tolist()]
lp_f = [float(pb_flat.LP[t]) for t in pb_flat.tp[1].tolist()]
# ★ 浮点比较必须带容差: 这些量在 PB 里是 float32, 取回来是 0.05000000074505806,
#   与 python 的 0.05 **不严格相等**。第一版用了集合精确比较, 红的是测试不是代码。
chk(len(set(lp_b)) > 1 and len(set(lp_f)) == 1
    and abs(lp_f[0] - LEASH_POS[1]) < 1e-6,
    "位置皮筋 base 分档 → flat 恒黄档",
    f"base∈{sorted(set(lp_b))} → flat {sorted(set(lp_f))} (黄档 {LEASH_POS[1]})")
rb_b = [bool(pb_base.rot_ban[t]) for t in pb_base.tr[1].tolist()]
rb_f = [bool(pb_flat.rot_ban[t]) for t in pb_flat.tr[1].tolist()]
chk(any(rb_b) and not any(rb_f),
    "红档'朝向禁判'在 flat 下完全消失",
    f"base 有 {sum(rb_b)} 行禁判 → flat {sum(rb_f)} 行 ⟹ 朝向恢复判定")

# ---- ④ 时钟门 ----
print("\n④ 时钟门(主判据 clock_done 直接依赖它)")
gp_b = [GATE_POS if t > 0 else RED_GATE_POS for t in pb_base.tmix.tolist()]
gp_f = [GATE_POS if t > 0 else RED_GATE_POS for t in pb_flat.tmix.tolist()]
chk(RED_GATE_POS in gp_b and set(gp_f) == {GATE_POS},
    "base 有红档宽门 → flat 全部走标准门",
    f"base 宽门 {gp_b.count(RED_GATE_POS)} 行 ({RED_GATE_POS*100:.0f}cm) → "
    f"flat 0 行, 全部 {GATE_POS*100:.0f}cm ⟹ **判定变严, 这就是它必须列 CRITICAL 的原因**")

# ---- 指纹与 digest ----
print("\n指纹 / digest")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "C_Wiring"))
try:
    import world_fingerprint as WF
    chk("switches.conf_flat" in WF.CRITICAL, "switches.conf_flat 在 CRITICAL 里",
        "拿 flat 训的 ckpt 在 base 环境评测会被拒跑, 不再静默匹配")
except Exception as e:
    chk(False, "switches.conf_flat 在 CRITICAL 里",
        f"读不到 world_fingerprint: {type(e).__name__}: {e}")
# ★钉值随判据一起走: L5-32 把倒水几何三常量并入白名单, digest
#   547163e4c6f98ec5 -> d2025c0b6f10b923。本自检钉的是"**本旗**不改 digest",
#   所以判据换代时更钉值是对的; 但每次更之前必须确认变化来自判据改动本身,
#   而不是"哪里悄悄动了却没人知道" —— 这一条红过一次(2026-08-31), 原因确认
#   为主判据换成 G3_pour ∧ placed, 属预期。
_d0 = criteria_digest()[1]
# ★L5-33 (2026-08-31): 水平门 0.08→0.0675 (圆盘相交, 用户裁定), digest 再变。
#   来源确认: 判据常量改动本身, 属预期。
chk(_d0 == "95b11e5dd755c8e0", "criteria.digest 未被本旗改动",
    f"{_d0[:16]} —— 旗改的是'每行用哪一档', 常量一个没变, "
    f"所以 digest **必须**不变(这正是它抓不到 conf_flat、必须单列的理由)")

print("\n" + "=" * 78)
print("★ 覆盖边界: 只验运行时六项。母带侧(去掉 conf 驱动的朝向平滑)由四道")
print("  出厂检查 + reference.md5 进指纹保证, 不在本自检范围内。")
print("=" * 78)
if _fail:
    print(f"[selftest] ❌ 失败 {len(_fail)} 项: {_fail}")
    sys.exit(1)
print("[selftest] ✅ 全绿")
