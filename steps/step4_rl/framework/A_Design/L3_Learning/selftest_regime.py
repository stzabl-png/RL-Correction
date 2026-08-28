"""体制自检 (L5-6): 断言实现与"两种参考方式"设计口径逐条一致。纯 numpy。

设计口径 (用户 2026-08-28 复述):
  A. 物体轨迹全程定义任务成败 —— 判据(G1..G4/死线)不含人手行;
  B. 人手轨迹永不做绝对位置参考 —— 时钟门/皮筋/死线均不查人手;
  C. 权重滑动: 高置信 物1.0/手0 (= 纯物轨) | 中 0.5/0.5 | 低 0.2/0.8;
  D. 低置信档物体绝对位置仍被拴住 (宽皮筋 + 甩飞死线), 只是形状不判(rot禁);
  E. P-OBJ = W_HAND 恒 0 (无形状指引), 其余与 P-HYB 全同 (唯一变量)。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from progress import (LEASH_POS, LEASH_ROT, GATE_POS, RED_GATE_POS, D3_DEV,
                      PourProgress, W_HAND, W_OBJ)

NPZ = os.path.join(_HERE, "..", "L2_Reference", "reference_v2.npz")
z = np.load(NPZ, allow_pickle=True)
rows = np.where(np.asarray(z["source"]) == 1)[0]


def pm(oi, half):
    q = np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows][0]
    w, x, y, zz = q / np.linalg.norm(q)
    R = np.array([[1-2*(y*y+zz*zz), 2*(x*y-w*zz), 2*(x*zz+w*y)],
                  [2*(x*y+w*zz), 1-2*(x*x+zz*zz), 2*(y*zz-w*x)],
                  [2*(x*zz-w*y), 2*(y*zz+w*x), 1-2*(x*x+y*y)]])
    u = np.array([0.0, half, 0.0])
    return u if (R @ u)[2] > (R @ (-u))[2] else -u


mb, mc = pm(1, 0.087), pm(0, 0.066)
obj = {oi: np.concatenate([np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows],
                           np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows]], axis=1)
       for oi in (0, 1)}
armq = {s: np.asarray(z[f"{s}_q"], np.float64)[rows] for s in ("right", "left")}
WOFF = np.array([0.0, 0.0, 0.10])
fails = []


def chk(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        fails.append(msg)


# ---- C: 权重表 ----
print("[C] 权重滑动 (物/手)")
chk(W_OBJ[2] == 1.0 and W_HAND[2] == 0.0, "高置信 1.0/0 (= 纯物轨, 人手零份额)")
chk(W_OBJ[1] == 0.5 and W_HAND[1] == 0.5, "中置信 0.5/0.5 (人手形状参考加入)")
chk(W_OBJ[0] == 0.2 and W_HAND[0] == 0.8, "低置信 0.2/0.8 (人手形状主导)")

# ---- D: 低置信仍有绝对位置底线, 但形状不判 ----
print("[D] 低置信档: 位置有底线, 形状不判")
chk(LEASH_POS[0] > LEASH_POS[1] > LEASH_POS[2] and LEASH_POS[0] < D3_DEV,
    f"皮筋pos 逐档放宽 {LEASH_POS[2]}/{LEASH_POS[1]}/{LEASH_POS[0]}m 且 < 甩飞线 {D3_DEV}m")
chk(LEASH_ROT[0] is None, "红档 rot 禁入判据 (物轨形状不可信)")
chk(RED_GATE_POS > GATE_POS, f"红档时钟门放宽 {GATE_POS}→{RED_GATE_POS}m (位置仍在判)")

# ---- A/B: 判据不含人手 ----
print("[A/B] 成败与门控只看物体, 人手不做绝对参考")
P = PourProgress(NPZ, mouth_local_bot=mb, mouth_local_cup=mc)
chk(not hasattr(P, "armq"), "进度机不再持有人手行 (红档臂门已废除)")
# 臂角喂垃圾: 除 G4 归位判据(机器站姿, 非人手行)外, 门/皮筋/死线不得变
res = {}
for tag, jitter in (("真臂", 0.0), ("乱臂", 1.5)):
    P = PourProgress(NPZ, mouth_local_bot=mb, mouth_local_cup=mc)
    P.enter(0, {1, 2})
    acc = {"adv": 0.0, "leash": 0.0, "fail": 0}
    for t in range(P.N):
        k = min(P.k, P.N - 1)
        ar = armq["right"][k] + jitter
        al = armq["left"][k] + jitter
        r = P.step(obj[0][k], obj[1][k], ar, al, pads3=True,
                   wrist_r=obj[1][k][:3] + WOFF, wrist_l=obj[0][k][:3] + WOFF)
        acc["adv"] += r["adv"]
        acc["leash"] += r["leash"]
        acc["fail"] += int(r["fail"] is not None)
    acc["g3"] = P.g[3]
    res[tag] = acc
chk(res["真臂"]["adv"] == res["乱臂"]["adv"],
    f"时钟推进与臂角无关 ({res['真臂']['adv']:.0f} vs {res['乱臂']['adv']:.0f})")
chk(abs(res["真臂"]["leash"] - res["乱臂"]["leash"]) < 1e-9, "皮筋与臂角无关")
chk(res["真臂"]["g3"] == res["乱臂"]["g3"] is True, "G3(任务核心) 与臂角无关")
chk(res["乱臂"]["fail"] == 0, "臂角乱走不触发任何死线 (死线只看物体)")

# ---- E: P-OBJ = W_HAND 恒 0, 其余全同 ----
print("[E] P-OBJ 消融: 唯一变量 = 人手形状指引")
Ph = PourProgress(NPZ, mouth_local_bot=mb, mouth_local_cup=mc, no_hand_ref=False)
Po = PourProgress(NPZ, mouth_local_bot=mb, mouth_local_cup=mc, no_hand_ref=True)
same = {"adv": True, "leash": True, "ms": True}
wh_o, wh_h = [], []
for Pn, box in ((Ph, "h"), (Po, "o")):
    Pn.enter(0, {1, 2})
accs = {"h": [], "o": []}
for t in range(Ph.N):
    for Pn, key in ((Ph, "h"), (Po, "o")):
        k = min(Pn.k, Pn.N - 1)
        r = Pn.step(obj[0][k], obj[1][k], armq["right"][k], armq["left"][k],
                    pads3=True, wrist_r=obj[1][k][:3] + WOFF,
                    wrist_l=obj[0][k][:3] + WOFF)
        accs[key].append((r["adv"], round(r["leash"], 9), r["ms"]))
        (wh_h if key == "h" else wh_o).append(r["w_hand"])
chk(accs["h"] == accs["o"], "两体制的 adv/皮筋/关奖 逐步完全一致")
chk(max(wh_o) == 0.0, "P-OBJ 的 W_HAND 恒 0 (无形状指引)")
chk(max(wh_h) == 0.8 and min(wh_h) == 0.0, "P-HYB 的 W_HAND 随档位在 0~0.8 滑动")

print("\n✅ 体制自检全部通过" if not fails else f"\n❌ {len(fails)} 项不符: {fails}")
sys.exit(0 if not fails else 1)
