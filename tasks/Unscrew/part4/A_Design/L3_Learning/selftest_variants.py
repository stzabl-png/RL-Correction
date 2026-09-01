"""合法变体 + 反向自检: 参考放音只证"完美能过", 这里再证
① 合法偏差 (物体 2cm 内小偏 + 腕 5mm 漂) 也能过;
② 真错被拦: 拧反方向(不释放)→G3 不立; 盖没送到位→placed 不立;
③ 未释放却报 placed 语义不可能 (G3 是 placed 的前置);
④ 护送信号开着的正路 (右垫全程接触) 照常 G4;
⑤ 盖脱手自由落体到目标 → 无接触快速过带 ⇒ escort_fail, placed 永不立;
⑥ 盖无接触**慢慢**沉到目标 (逐步 <ESCORT_FALL, 过带判据抓不到) ⇒ 持盖步数
   不够 (U41②), placed 仍不立 —— 单靠过带判据会漏的那一路;
⑦ 拿着盖但半途摔下去再接住 (降幅峰值 >PLACE_FALL) ⇒ U41③ 否决。
   (T2-3/U41 同一条用户裁定: 成功=右手**拿着**盖**放**到桌上。)"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "C_Wiring"))
from progress import (CERT_RAMP, CERT_RET, PLACED_HOLD, M4_HOLD,  # noqa: E402
                      ESCORT_FALL, PLACE_CARRY, PLACE_FALL, TABLE_Z,
                      UnscrewProgress)
import task_config as TC  # noqa: E402

NPZ = TC.REF_V2 if os.path.isfile(TC.REF_V2) else TC.REF_V1
z = np.load(NPZ, allow_pickle=True)
rows = np.where(np.asarray(z["source"]) == 1)[0]
obj = {oi: np.concatenate([np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows],
                           np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows]],
                          axis=1) for oi in (0, 1)}
armq = {s: np.asarray(z[f"{s}_q"], np.float64)[rows] for s in ("right", "left")}
stance = {s: np.asarray(z[f"{s}_q"], np.float64)[-1] for s in ("right", "left")}
WOFF = np.array([0.0, 0.0, 0.10])
rng = np.random.default_rng(7)


def run(off_scale=0.0, feed_released=True, cap_end_off=0.0, max_t=1500,
        escort=None):
    """escort: None=无信号(护送停用) / 'hold'=右垫全程接触 /
    'drop'=释放后盖脱手自由落体到终点 (每步 -2.5cm, 无接触) /
    'slowdrop'=无接触但每步只沉 0.5cm (<ESCORT_FALL, 过带判据抓不到) /
    'toss'=右手拿着, 但半途一段 6cm/步 的坠落 (>PLACE_FALL) 后再接住。"""
    P = UnscrewProgress(NPZ)
    t = 0
    drop_z = None
    while t < max_t and not P.g[4]:
        if P.cert_phase == 1:
            dz = 0.015 * min((P.cert_t + 1) / CERT_RAMP, 1.0)
        elif P.cert_phase == 2:
            dz = 0.015
        elif P.cert_phase == 3:
            dz = 0.015 * max(1.0 - (P.cert_t + 1) / CERT_RET, 0.0)
        else:
            dz = 0.0
        k = min(P.k, P.N - 1)
        if P.g[3] and P.k >= P.N - 1:
            o0, o1 = P.end[0].copy(), P.end[1].copy()
            ar, al = stance["right"], stance["left"]
        else:
            o0, o1 = obj[0][k].copy(), obj[1][k].copy()
            ar, al = armq["right"][k], armq["left"][k]
        if off_scale > 0:
            # 只偏 xy: 认证判据是 z 升 >=5mm, 给 z 加 1.5cm 噪声会把"合法偏差"
            # 变成"抵消认证提升"的非法喂法 (那不是本测的命题)
            o0[:2] += rng.uniform(-off_scale, off_scale, 2)
            o1[:2] += rng.uniform(-off_scale, off_scale, 2)
        if cap_end_off and P.g[3]:
            # 从释放起盖就没送到位 (偏移加在时钟尾会漏: placed 在交互尾段
            # 的 hold 里就能锁存, 测不到"没到位"这个命题)
            o1[:2] += cap_end_off
        o0[2] += dz; o1[2] += dz
        rel = feed_released and P.g[2] and k >= P.k_sep
        prc = None
        if escort == "hold":
            prc = 3
        elif escort in ("drop", "slowdrop"):
            prc = 0                      # 释放后全程无右垫接触
            _top = TABLE_Z + 0.03 + 0.005
            if P.g[3] and P.k < P.N - 1 and escort == "slowdrop":
                # 时钟走完前把盖托在**带顶之上**: 否则母带自己的末段下落 (每步
                # 可能 >ESCORT_FALL) 就会触发过带判据, ⑥ 要测的"慢沉"命题落空。
                o1[2] = max(float(o1[2]), _top)
            elif P.g[3] and P.k >= P.N - 1:
                # 释放后盖脱手: xy 到终点, z 逐步下沉。drop = 2.5cm/步 (过带
                # 抓得到); slowdrop = 4mm/步 (<ESCORT_FALL, 过带抓不到 ——
                # 该由 U41② 持盖步数拦下)
                step = 0.025 if escort == "drop" else 0.004
                if drop_z is None:
                    # ⑤ drop 的起点必须在带顶之上才有"过带"可测: 有的 clip (17: 瓶放倒
                    # 后低位拔盖, 母带末行盖离桌仅 0.6cm) 时钟走完时盖本来就在带内,
                    # 从那儿"摔"没有过带事件 —— 那不是判据漏洞, 是测试前提不成立。
                    # 统一从带顶上方 5cm (=离桌 8cm, 手滑脱的量级; 首步先扣 2.5cm 再判, 故需 >带顶+2.5cm) 起落。
                    drop_z = (max(float(o1[2]), _top + 0.05) if escort == "drop"
                              else max(float(o1[2]), _top))
                drop_z = max(drop_z - step, float(P.end[1][2]))
                o1 = P.end[1].copy()
                o1[2] = drop_z
        elif escort == "toss":
            prc = 3          # 手一直在盖上 (过带/持盖两件都满足)
            if P.g[3]:
                # **在 placed 锁存之前**就把盖举高再摔 6cm/步 (>PLACE_FALL=4cm)。
                # 用相对母带位置的高度偏置 (不瞬移到终点): 瞬移会先立 placed,
                # 之后的坠落就变成"扰动已放好的物体"(D8), 测的不是本命题了;
                # 偏置 12cm 也远在 D3(35cm) 之内。
                if drop_z is None:
                    drop_z = 0.12
                o1 = o1.copy()
                o1[2] += drop_z
                drop_z = max(drop_z - 0.06, 0.0)
        # 腕漂 ±2mm: 认证滑移线 8mm, 两帧独立 ±5mm 噪声最坏差 17mm 会误伤
        r = P.step(o0, o1, ar, al, True,
                   o1[:3] + WOFF + rng.uniform(-0.002, 0.002, 3),
                   o0[:3] + WOFF + rng.uniform(-0.002, 0.002, 3),
                   screw_released=rel, pads_r_cap=prc)
        if r["fail"]:
            return P, r["fail"], t
        t += 1
    return P, None, t


# ① 合法偏差
P, fail, t = run(off_scale=0.015)
assert fail is None and P.g[4], f"合法偏差被误拦: fail={fail} G4={P.g[4]} k={P.k}"
print(f"[变体] ① 1.5cm 物体xy偏差+2mm腕漂 -> G4 @{t} ✅")
# ② 不释放 (拧不动/拧反): G3 不立, 时钟仍可走完但不该成功
P, fail, t = run(feed_released=False)
assert not P.g[3] and not P.g[4], f"未释放却 G3={P.g[3]} G4={P.g[4]}"
print(f"[变体] ② 不释放 -> G3 不立 ✅ (t={t}, k={P.k})")
# ③ 盖没送到位: placed 不立
P, fail, t = run(cap_end_off=0.12)
assert P.g[3] and not P.placed and not P.g[4], \
    f"盖偏 12cm 却 placed={P.placed}"
print(f"[变体] ③ 盖离目标 12cm -> placed 不立 ✅")
# ④ 护送正路: 右垫全程接触, 照常走到 G4
P, fail, t = run(escort="hold")
assert fail is None and P.g[4] and not P.escort_fail, \
    f"护送正路被误拦: fail={fail} G4={P.g[4]} escort_fail={P.escort_fail}"
print(f"[变体] ④ 右垫护送 -> G4 @{t} ✅")
# ⑤ 盖脱手自由落体到目标: escort_fail (过带), placed 永不立
P, fail, t = run(escort="drop")
assert P.g[3] and P.escort_fail and not P.placed and not P.g[4], \
    (f"自由落体却 placed={P.placed} escort_fail={P.escort_fail} "
     f"G4={P.g[4]}")
assert P.carry_steps == 0, f"脱手却记了持盖步 {P.carry_steps}"
print(f"[变体] ⑤ 盖脱手自由落体 -> escort_fail(过带)+持盖0步, placed 不立 ✅")
# ⑥ 无接触慢沉: 过带判据抓不到 (降幅 < ESCORT_FALL), 靠 U41② 拦
P, fail, t = run(escort="slowdrop")
assert P.g[3] and not P.placed and not P.g[4], \
    f"无接触慢沉却 placed={P.placed} G4={P.g[4]}"
assert not P.escort_fail, "慢沉不该触发过带判据 (那是 ⑤ 的命题)"
assert fail is None, f"⑥ 不该撞死线: {fail}"
assert P.carry_steps < PLACE_CARRY and P.fall_peak < PLACE_FALL, \
    f"慢沉应只由持盖步数拦下: carry={P.carry_steps} fall={P.fall_peak:.3f}"
print(f"[变体] ⑥ 无接触慢沉 -> 持盖 {P.carry_steps:.0f}<{PLACE_CARRY} 步, "
      f"placed 不立 ✅ (过带判据本身抓不到, 单靠它会漏)")
# ⑦ 拿着但半途坠落再接住: 降幅峰值否决
P, fail, t = run(escort="toss")
assert fail is None, f"⑦ 不该撞死线 (那测的就不是判据了): {fail}"
assert P.g[3] and not P.placed and not P.g[4], \
    f"坠落再接住却 placed={P.placed} G4={P.g[4]}"
assert P.fall_peak >= PLACE_FALL and P.carry_steps >= PLACE_CARRY, \
    f"应由降幅峰值拦下: fall={P.fall_peak:.3f} carry={P.carry_steps}"
print(f"[变体] ⑦ 半途坠落 {P.fall_peak * 100:.0f}cm/步 (>{PLACE_FALL * 100:.0f}) "
      f"-> placed 不立 ✅ (持盖 {P.carry_steps:.0f} 步够, 但没'放'住)")
print("[变体] ★全绿")
