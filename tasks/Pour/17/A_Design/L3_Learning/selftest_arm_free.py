"""第十六件自检 (L5-31): POUR_ARM_FREE 消融旗 (straight 臂) —— 验行为不验旗。

★ 立此自检的由来:
`arm_free` 做三件事, 每一件都能**静默地做错而不报错**:
  ① 臂前馈冻结 —— 冻错范围(全程冻)会毁掉 Approach 段, 而日志一切正常
  ② 残差界放大 —— 放大错范围(整体乘)会给接近段 20 倍权限, 2026-08-28 裁定过
     "接近段臂残差同冻"(±2cm 权限曾致撞杯)
  ③ 手指列必须**不冻** —— 冻了就混进第二个变量, 消融不再是单因子
写这个旗时前两条我都写错了一次(无条件冻结 / 整体乘), 都是本自检该抓的。

★ 覆盖边界: 本自检是**纯数值的**, 不启 Isaac。它验的是 `_ff_row` 与 `dev_arm_rows`
的构造逻辑, 不验物理。物理侧(策略真能不能靠残差走出倒水)只能靠训练本身回答。

用法: python selftest_arm_free.py   (退出码 0 = 全绿)
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


_fail = []


def chk(ok, name, detail):
    print(f"  [{'OK' if ok else 'XX'}] {name}: {detail}", flush=True)
    if not ok:
        _fail.append(name)


# ---------------------------------------------------------------------------
# 复刻 pour_env 里的两段构造逻辑 (不启 Isaac)。
# ★ 这是"复刻"不是"调用" —— 所以必须同时验**复刻与真源一致**, 否则本自检会在
#   pour_env 改了之后继续绿, 变成装样子的检查。下面 ⑤ 就是干这个的。
# ---------------------------------------------------------------------------
DEV_ARM_TIER = {2: 0.05, 1: 0.08, 0: 0.10}
DEV_ARM_MACHINE = 0.05
T_ROW, IA0, N_ROW = 653, 190, 273
IA1 = IA0 + N_ROW - 1


def build_dev_rows(arm_free, scale=20.0, tmix=None):
    dev = torch.full((T_ROW,), DEV_ARM_MACHINE)
    for k in range(N_ROW):
        dev[IA0 + k] = DEV_ARM_TIER[int(tmix[k])] * (scale if arm_free else 1.0)
    return dev


def ff_arm(rows, ref58, arm_free):
    """返回给定行的臂列前馈 (N,14)。"""
    ff = ref58[rows].clone()
    if arm_free:
        in_ia = ((rows >= IA0) & (rows <= IA1)).unsqueeze(1).float()
        ff[:, :14] = ff[:, :14] * (1 - in_ia) + ref58[IA0][:14].unsqueeze(0) * in_ia
    return ff


print("=" * 78)
print("[selftest] POUR_ARM_FREE (straight 臂) — 逐项验行为")
print("=" * 78)

# 造一条可辨认的假参考: 每行每列都不同, 这样"冻没冻"一眼可辨
ref58 = torch.arange(T_ROW * 58, dtype=torch.float32).reshape(T_ROW, 58) * 1e-4
rng = np.random.default_rng(17)
tmix = rng.integers(0, 3, N_ROW)

# ---- ① 臂前馈: 交互段冻结 ----
print("\n① 臂前馈在交互段冻结")
ia_rows = torch.arange(IA0, IA1 + 1)
f_on = ff_arm(ia_rows, ref58, True)[:, :14]
f_off = ff_arm(ia_rows, ref58, False)[:, :14]
chk(float((f_on - f_on[0]).abs().max()) < 1e-9, "开旗后交互段臂列**逐行恒定**",
    f"最大逐行差 {float((f_on - f_on[0]).abs().max()):.2e}")
chk(float((f_off - f_off[0]).abs().max()) > 1e-6, "关旗时交互段臂列是变化的(对照)",
    f"最大逐行差 {float((f_off - f_off[0]).abs().max()):.4f} —— 否则①无意义")
chk(float((f_on[0] - ref58[IA0][:14]).abs().max()) < 1e-9, "冻结值 = 交互段首行",
    "不是别的行 —— 冻错行等于给了一个错的抓握姿势")

# ---- ② ★冻结不得外溢到 Approach / Retreat ----
print("\n② 冻结**不得**外溢到 Approach / Retreat (第一版写错过)")
for nm, rows in (("Approach", torch.arange(0, IA0)),
                 ("Retreat", torch.arange(IA1 + 1, T_ROW))):
    a = ff_arm(rows, ref58, True)[:, :14]
    b = ff_arm(rows, ref58, False)[:, :14]
    chk(float((a - b).abs().max()) < 1e-9, f"{nm} 段臂前馈开旗前后**一字不差**",
        f"最大差 {float((a - b).abs().max()):.2e} —— "
        f"冻掉它机器人会从第0步就往抓握姿势走, 而日志一切正常")

# ---- ③ 手指列必须不冻 ----
print("\n③ 手指列不冻 (否则混进第二个变量)")
g_on = ff_arm(ia_rows, ref58, True)[:, 14:]
g_off = ff_arm(ia_rows, ref58, False)[:, 14:]
chk(float((g_on - g_off).abs().max()) < 1e-9, "手指列开旗前后一字不差",
    f"最大差 {float((g_on - g_off).abs().max()):.2e}")
chk(float((g_on - g_on[0]).abs().max()) > 1e-6, "手指列仍然逐行变化",
    f"最大逐行差 {float((g_on - g_on[0]).abs().max()):.4f}")

# ---- ④ 残差界: 只放大交互段, 且保留分档形状 ----
print("\n④ 残差界只放大交互段, 且保留分档形状")
d_on = build_dev_rows(True, 20.0, tmix)
d_off = build_dev_rows(False, 20.0, tmix)
mach = torch.cat([torch.arange(0, IA0), torch.arange(IA1 + 1, T_ROW)])
chk(float((d_on[mach] - d_off[mach]).abs().max()) < 1e-9,
    "机器行(Approach/Retreat)界**未被放大**",
    f"最大差 {float((d_on[mach] - d_off[mach]).abs().max()):.2e}; "
    f"整体乘会给接近段 20 倍权限 —— 2026-08-28 裁定过接近段臂残差同冻(曾致撞杯)")
ia = torch.arange(IA0, IA1 + 1)
ratio = (d_on[ia] / d_off[ia])
chk(float((ratio - 20.0).abs().max()) < 1e-5, "交互行界恰好放大 20 倍",
    f"比值范围 [{float(ratio.min()):.4f}, {float(ratio.max()):.4f}]")
uniq_on = sorted({round(float(x), 4) for x in d_on[ia]})
chk(len(uniq_on) == 3, "★放大后仍是**三档**(只放大不改形状)",
    f"档位 {uniq_on} —— 拍平会让 straight 与 base 多差一个变量")
chk(abs(uniq_on[-1] - 2.0) < 1e-4, "最大档 = 2.0 rad",
    f"实测 {uniq_on[-1]} —— 参考自身相对冻结姿势的最大偏离是 R_j5 的 1.954 rad(112°), "
    f"2.0 恰好覆盖")

# ---- ⑤ ★复刻必须与真源一致 (否则本自检会在 pour_env 改后继续装绿) ----
print("\n⑤ 本自检的复刻常量与 pour_env 真源一致")
try:
    src = open(os.path.join(_HERE, "..", "..", "C_Wiring", "pour_env.py"),
               encoding="utf-8").read()
    ok_t = "DEV_ARM_TIER = {2: 0.05, 1: 0.08, 0: 0.10}" in src
    ok_m = "DEV_ARM_MACHINE" in src
    ok_flag = 'os.environ.get("POUR_ARM_FREE") == "1"' in src
    ok_scale = '"POUR_ARM_FREE_SCALE", "20.0"' in src
    ok_ia = "(r >= self.IA0) & (r <= self.IA1)" in src
    chk(ok_t and ok_m and ok_flag and ok_scale and ok_ia,
        "真源含同名常量/旗/交互段判据",
        f"DEV_ARM_TIER={ok_t} MACHINE={ok_m} 旗={ok_flag} "
        f"默认scale={ok_scale} 交互段限定={ok_ia}")
except Exception as e:
    chk(False, "真源含同名常量/旗/交互段判据",
        f"读不到 pour_env.py: {type(e).__name__}: {e} —— **未验, 不是通过**")

# ---- ⑥ 指纹能看见这两面旗 ----
print("\n⑥ 世界指纹能看见 arm_ref_free / arm_free_scale")
try:
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "C_Wiring"))
    import world_fingerprint as WF
    need = {"switches.arm_ref_free", "switches.arm_free_scale"}
    chk(need <= set(WF.CRITICAL), "两项都在 CRITICAL 里",
        "大界训出的 ckpt 拿到小界环境回放每步都会被钳住, 必须拒跑")
except Exception as e:
    chk(False, "两项都在 CRITICAL 里", f"读不到 world_fingerprint: {type(e).__name__}: {e}")

print("\n" + "=" * 78)
print("★ 覆盖边界: 纯数值自检, 不启 Isaac。验的是 _ff_row 与 dev_arm_rows 的构造")
print("  逻辑与指纹接线, **不验物理** —— 策略能不能靠残差走出倒水, 只有训练能回答。")
print("★ 保留: 认证窗那 21 步的臂参考仍然存在(冻结之后才混合), 否则 G2 认证机制")
print("  无法工作。所以准确说法是「冻结逐行臂参考」, 不是「没有臂参考」。")
print("=" * 78)
if _fail:
    print(f"[selftest] ❌ 失败 {len(_fail)} 项: {_fail}")
    sys.exit(1)
print("[selftest] ✅ 全绿")
