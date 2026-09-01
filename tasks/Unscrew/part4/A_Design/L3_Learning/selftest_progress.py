"""Unscrew Success Tracker 放音自检 (纪律#19): 母带自己当"完美执行"喂进度机。
必须: G1@10步 / 认证一次过 G2 / 时钟走到底 / G3=释放按序(k_sep 处喂 released)
/ 皮筋零违例 / placed+G4 在归位补测段触发 / 收入=38 / 认证反向(不升=3败 g2不立)
/ D3反向。纯 numpy, 无 Isaac。消费 UNSCREW_CLIP 对应的 v1 或 v2 母带。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "C_Wiring"))
from progress import (CERT_RAMP, CERT_HOLD, CERT_RET, CERT_TRIES, G1_HOLD,  # noqa: E402
                      PLACED_HOLD, M4_HOLD, UnscrewProgress)
import task_config as TC  # noqa: E402

NPZ = TC.REF_V2 if os.path.isfile(TC.REF_V2) else TC.REF_V1
z = np.load(NPZ, allow_pickle=True)
rows = np.where(np.asarray(z["source"]) == 1)[0]
obj = {oi: np.concatenate([np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows],
                           np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows]],
                          axis=1)
       for oi in (0, 1)}
armq = {s: np.asarray(z[f"{s}_q"], np.float64)[rows] for s in ("right", "left")}
stance = {s: np.asarray(z[f"{s}_q"], np.float64)[-1] for s in ("right", "left")}
WOFF = {"right": np.array([0.0, 0.0, 0.10]), "left": np.array([0.0, 0.0, 0.10])}


def feed(P, o0, o1, ar, al, dz=0.0, released=False):
    o0 = o0.copy(); o1 = o1.copy()
    o0[2] += dz; o1[2] += dz
    return P.step(o0, o1, ar, al, pads3=True,
                  wrist_r=o1[:3] + WOFF["right"], wrist_l=o0[:3] + WOFF["left"],
                  screw_released=released)


P = UnscrewProgress(NPZ)
print(f"[放音] clip={TC.CLIP_ID} 交互行 {P.N} k_sep={P.k_sep} | 档位: "
      f"绿{P.tmix.count(2)} 黄{P.tmix.count(1)} 红{P.tmix.count(0)}")
led = {"adv": 0.0, "ms": 0.0, "leash": 0.0, "wage": 0.0}
ms_at = {}
# ---- 站位段: G1 -> 认证 -> G2 (完美执行: 认证期双物 +15mm 随斜坡) ----
t = 0
while not P.g[2]:
    assert t < 60, f"站位段 60 步内未立 G2 (phase={P.cert_phase} try={P.cert_try})"
    if P.cert_phase == 1:
        dz = 0.015 * min((P.cert_t + 1) / CERT_RAMP, 1.0)
    elif P.cert_phase == 2:
        dz = 0.015
    elif P.cert_phase == 3:
        dz = 0.015 * max(1.0 - (P.cert_t + 1) / CERT_RET, 0.0)
    else:
        dz = 0.0
    r = feed(P, obj[0][0], obj[1][0], armq["right"][0], armq["left"][0], dz=dz)
    assert r["fail"] is None, f"站位段死线 {r['fail']} @t{t}"
    for kk in ("ms", "wage", "leash", "adv"):
        led[kk] += r[kk]
    for gi in (1, 2):
        if P.g[gi] and gi not in ms_at:
            ms_at[gi] = t
    t += 1
print(f"[放音] G1@{ms_at.get(1)} G2@{ms_at.get(2)} (期望 {G1_HOLD - 1} 与 "
      f"~{G1_HOLD + CERT_RAMP + CERT_HOLD + CERT_RET}) | cert_try={P.cert_try} (期望0)")
assert ms_at[1] == G1_HOLD - 1 and P.cert_try == 0
# ---- 交互段照谱: k_sep 行喂 released ----
while P.k < P.N - 1:
    k = P.k
    r = feed(P, obj[0][k], obj[1][k], armq["right"][k], armq["left"][k],
             released=(k >= P.k_sep))
    assert r["fail"] is None, f"交互段死线 {r['fail']} @k{k}"
    assert r["adv"] == 1.0, f"时钟卡住 @k{k} (gate 违例?)"
    assert r["leash"] == 0.0, f"皮筋违例 @k{k}: {r['leash']}"
    for kk in ("ms", "wage", "leash", "adv"):
        led[kk] += r[kk]
    if P.g[3] and 3 not in ms_at:
        ms_at[3] = k
assert P.g[3] and 3 in ms_at and ms_at[3] >= P.k_sep - 1, \
    f"G3 未按序 (k_sep={P.k_sep}, got {ms_at.get(3)})"
# ---- 归位补测段: 物体钉在末行 + 双臂回站姿 ----
for t2 in range(PLACED_HOLD + M4_HOLD + 6):
    r = feed(P, P.end[0], P.end[1], stance["right"], stance["left"],
             released=True)
    for kk in ("ms", "wage", "leash", "adv"):
        led[kk] += r[kk]
    if P.g[4]:
        break
assert P.placed and P.g[4], f"placed={P.placed} G4={P.g[4]} 未在补测段触发"
total_ms = led["ms"]
print(f"[放音] G链全通: adv={led['adv']:.0f}/{P.N - 1} ms={total_ms:.0f} "
      f"(期望38) wage={led['wage']:.2f} leash={led['leash']:.2f}")
assert led["adv"] == P.N - 1 and total_ms == 38.0

# ---- 反向1: 认证不升 -> 3 次全败, g2 不立 ----
P2 = UnscrewProgress(NPZ)
for t in range(220):
    feed(P2, obj[0][0], obj[1][0], armq["right"][0], armq["left"][0], dz=0.0)
assert P2.g[1] and not P2.g[2] and P2.cert_try == CERT_TRIES, \
    f"认证反向失败: g2={P2.g[2]} try={P2.cert_try}"
print(f"[放音] 反向1 ✅ 不升瓶 -> 认证 {P2.cert_try}/{CERT_TRIES} 全败, G2 不立")
# ---- 反向2: D3 甩飞 ----
P3 = UnscrewProgress(NPZ)
o_bad = obj[0][0].copy(); o_bad[0] += 0.5
r = P3.step(o_bad, obj[1][0], armq["right"][0], armq["left"][0], True,
            obj[1][0][:3], obj[0][0][:3])
assert r["fail"] and r["fail"].startswith("D3"), r["fail"]
print("[放音] 反向2 ✅ 甩飞 50cm -> " + r["fail"])
# ---- 反向3: 未认证不走钟 ----
P4 = UnscrewProgress(NPZ)
for t in range(30):
    r = feed(P4, obj[0][min(t, P4.N - 1)], obj[1][min(t, P4.N - 1)],
             armq["right"][0], armq["left"][0])
assert P4.k == 0 or P4.g[2], "G2 前时钟不许走"
print("[放音] 反向3 ✅ G2 前时钟冻结")
print("[放音] ★全绿")
