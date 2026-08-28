"""Success Tracker 放音自检 (纪律#19, L5-1 换代版): 母带自己当"完美执行"喂进度机。
必须: G1@10步 / 认证一次通过 G2 / 时钟走到底 / G3按序 / 皮筋零违例 /
placed+G4 在归位补测段触发 / 收入=38 / 认证反向(不升=3次全败 g2不立) / D3反向。
纯 numpy, 无 Isaac。消费 v2 母带。
"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from progress import (PourProgress, _axis_tilt, CERT_RAMP, CERT_HOLD, CERT_RET,
                      CERT_TRIES, G1_HOLD)

NPZ = os.path.join(_HERE, "..", "L2_Reference", "pour17_reference_v2.npz")
z = np.load(NPZ, allow_pickle=True)
rows = np.where(np.asarray(z["source"]) == 1)[0]

def pick_mouth(oi, half):
    q = np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows][0]
    w, x, y, zz = q / np.linalg.norm(q)
    R = np.array([[1-2*(y*y+zz*zz), 2*(x*y-w*zz), 2*(x*zz+w*y)],
                  [2*(x*y+w*zz), 1-2*(x*x+zz*zz), 2*(y*zz-w*x)],
                  [2*(x*zz-w*y), 2*(y*zz+w*x), 1-2*(x*x+y*y)]])
    up = np.array([0.0, half, 0.0])
    return up if (R @ up)[2] > (R @ (-up))[2] else -up

mb = pick_mouth(1, 0.087)
mc = pick_mouth(0, 0.066)
obj = {oi: np.concatenate([np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows],
                           np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows]], axis=1)
       for oi in (0, 1)}
armq = {s: np.asarray(z[f"{s}_q"], np.float64)[rows] for s in ("right", "left")}
WOFF = {"right": np.array([0.0, 0.0, 0.10]), "left": np.array([0.0, 0.0, 0.10])}

def feed(P, o0, o1, ar, al, dz=0.0):
    """完美执行喂法: 物体抬 dz, 腕=物+恒偏移(相对零漂)."""
    o0 = o0.copy(); o1 = o1.copy()
    o0[2] += dz; o1[2] += dz
    return P.step(o0, o1, ar, al, pads3=True,
                  wrist_r=o1[:3] + WOFF["right"], wrist_l=o0[:3] + WOFF["left"])

P = PourProgress(NPZ, mouth_local_bot=mb, mouth_local_cup=mc, mouth_gate=0.12)
print(f"[放音] 交互行 {P.N} | 档位: 绿{P.tmix.count(2)} 黄{P.tmix.count(1)} 红{P.tmix.count(0)}")
led = {"adv": 0.0, "ms": 0.0, "leash": 0.0, "wage": 0.0}
gate_hist = {"obj": 0, "hand": 0}
ms_at = {}
# ---- 站位段: G1 -> 认证 -> G2 (完美执行: 认证期物体+5mm随斜坡) ----
t = 0
while not P.g[2]:
    assert t < 60, "站位段 60 步内未立 G2"
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
    led["ms"] += r["ms"]; led["wage"] += r["wage"]; led["leash"] += r["leash"]
    for gi in (1, 2):
        if P.g[gi] and gi not in ms_at:
            ms_at[gi] = t
    t += 1
print(f"[放音] G1@{ms_at.get(1)} G2@{ms_at.get(2)} (期望 {G1_HOLD-1} 与 "
      f"{G1_HOLD+CERT_RAMP+CERT_HOLD+CERT_RET}±1) | 认证尝试后 cert_try={P.cert_try} (期望0)")
# ---- 交互段照谱 ----
mouth_min = 1e9
for tt in range(P.N):
    k = min(P.k, P.N - 1)
    r = feed(P, obj[0][k], obj[1][k], armq["right"][k], armq["left"][k])
    assert r["fail"] is None, f"参考放音触发死线 {r['fail']} @t{tt}"
    led["adv"] += r["adv"]; led["ms"] += r["ms"]; led["leash"] += r["leash"]
    if r["gate_by"]:
        gate_hist[r["gate_by"]] += 1
    if P.g[3] and 3 not in ms_at:
        ms_at[3] = tt
    tilt = _axis_tilt(obj[1][k][3:7], P.up_b)
    if tilt >= np.radians(90):
        q1 = obj[1][k][3:7] / np.linalg.norm(obj[1][k][3:7])
        w, x, y, zz = q1
        R1 = np.array([[1-2*(y*y+zz*zz), 2*(x*y-w*zz), 2*(x*zz+w*y)],
                       [2*(x*y+w*zz), 1-2*(x*x+zz*zz), 2*(y*zz-w*x)],
                       [2*(x*zz-w*y), 2*(y*zz+w*x), 1-2*(x*x+y*y)]])
        q0 = obj[0][k][3:7] / np.linalg.norm(obj[0][k][3:7])
        w2, x2, y2, z2 = q0
        R0 = np.array([[1-2*(y2*y2+z2*z2), 2*(x2*y2-w2*z2), 2*(x2*z2+w2*y2)],
                       [2*(x2*y2+w2*z2), 1-2*(x2*x2+z2*z2), 2*(y2*z2-w2*x2)],
                       [2*(x2*z2-w2*y2), 2*(y2*z2+w2*x2), 1-2*(x2*x2+y2*y2)]])
        d = np.linalg.norm((obj[1][k][:3] + R1 @ mb) - (obj[0][k][:3] + R0 @ mc))
        mouth_min = min(mouth_min, d)
print(f"[放音] 时钟 {P.k}/{P.N-1} | 行进收入 {led['adv']:.0f} | 门统计 obj={gate_hist['obj']}"
      f" hand={gate_hist['hand']} | 皮筋总罚 {led['leash']:.3f} (应=0)"
      f" | 口口最小距 {mouth_min*100:.1f}cm")
# ---- 归位补测段 ----
rest = {oi: obj[oi][0] for oi in (0, 1)}
stance = {s: np.asarray(z[f"{s}_q"], np.float64)[-1] for s in ("right", "left")}
for tt in range(80):
    ar = stance["right"] if P.placed else armq["right"][-1]
    al = stance["left"] if P.placed else armq["left"][-1]
    r = feed(P, rest[0], rest[1], ar, al)
    led["ms"] += r["ms"]
    if P.placed and "p" not in ms_at:
        ms_at["p"] = P.N + tt
    if P.g[4] and 4 not in ms_at:
        ms_at[4] = P.N + tt
print(f"[放音] G1@{ms_at.get(1,'✗')} G2@{ms_at.get(2,'✗')} G3@{ms_at.get(3,'✗')} "
      f"placed@{ms_at.get('p','✗')} G4@{ms_at.get(4,'✗')} | 关收入 {led['ms']:.0f} (期望38)"
      f" | 工资 {led['wage']:.2f}")
# ---- 认证反向: 物体不升 -> 3次全败, g2 不立 ----
P3 = PourProgress(NPZ, mouth_local_bot=mb, mouth_local_cup=mc)
for tt in range(200):
    feed(P3, obj[0][0], obj[1][0], armq["right"][0], armq["left"][0], dz=0.0)
neg_ok = (not P3.g[2]) and P3.cert_try == CERT_TRIES
print(f"[放音] 认证反向: g2={P3.g[2]} cert_try={P3.cert_try} (期望 False/{CERT_TRIES})")
# ---- D3 反向 ----
P2 = PourProgress(NPZ, mouth_local_bot=mb, mouth_local_cup=mc)
_far = obj[1][0].copy(); _far[0] += 0.40
r_d3 = P2.step(obj[0][0], _far, armq["right"][0], armq["left"][0], pads3=True,
               wrist_r=_far[:3] + WOFF["right"], wrist_l=obj[0][0][:3] + WOFF["left"])
print(f"[放音] D3反向: fail={r_d3['fail']} (期望 D3_dev_obj1)")
ok = (P.k == P.N - 1 and abs(led["leash"]) < 1e-9
      and r_d3["fail"] == "D3_dev_obj1" and neg_ok
      and all(m in ms_at for m in (1, 2, 3, "p", 4))
      and ms_at[1] <= ms_at[2] <= ms_at[3] <= ms_at["p"] <= ms_at[4]
      and led["ms"] == 38.0 and P.cert_try == 0)
print("✅ 放音自检全部通过" if ok else "❌ 放音自检未通过 —— 见上")
sys.exit(0 if ok else 1)
