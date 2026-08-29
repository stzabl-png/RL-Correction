"""L5-21 自检第八件: ① 训练循环 NaN 防护  ② 认证五项分项计数(双版一致)。

判别句(沿用家族纪律): 任何检查都必须能说出"它在什么情况下会失败"。
所以每一项都配一个**必然该红的输入**, 并顺带证明旧写法在同一输入上确实会坏。
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from progress import (PourProgress, CERT_RAMP, CERT_HOLD, CERT_RET,   # noqa: E402
                      CERT_RISE, CERT_SLIP, CERT_WAIT)
from progress_batch import PourProgressBatch                          # noqa: E402

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


MB, MC = pick_mouth(1, 0.087), pick_mouth(0, 0.066)
OBJ = {oi: np.concatenate(
    [np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows],
     np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows]], axis=1) for oi in (0, 1)}
ARM = {s: np.asarray(z[f"{s}_q"], np.float64)[rows] for s in ("right", "left")}
WOFF = np.array([0.0, 0.0, 0.10])
ok_all = True


# ═════════════ ① 训练循环的 NaN 防护 ═════════════
print("① 训练循环 NaN 防护 —— 空分母窗不得毒化 EMA / 不得靠陈旧证据解锁")


def _ema_new(prev, r, a=0.02):
    return prev if r != r else (1.0 - a) * prev + a * r


nan = float("nan")
# 反例 A: 旧写法吃一个 NaN 就永久坏死 (证明这个检查能红)
old = 0.25
for _ in range(50):
    old = 0.98 * old + 0.02 * nan
bad_old = (old != old)
# 新写法: 连吃 50 个 NaN 后仍是原值
new = 0.25
for _ in range(50):
    new = _ema_new(new, nan)
hold_ok = abs(new - 0.25) < 1e-12
# 新写法仍能被真数据推动 (证明不是"永远不动"的恒真断言)
mv = 0.25
for _ in range(200):
    mv = _ema_new(mv, 1.0)
move_ok = mv > 0.9
# sustain: 空分母窗既不推进也不清零
sustain, ema = 0, 0.35
for r in (nan, nan, 0.40, nan, 0.40, nan):
    ema = _ema_new(ema, r)
    if r != r:
        continue
    sustain = sustain + 1 if ema >= 0.30 else 0
sus_ok = sustain == 2                       # 只有 2 个有数据的窗计入
for lab, v in (("旧写法确实会坏死(此检查能红)", bad_old),
               ("新写法吃50个NaN仍保持原值", hold_ok),
               ("新写法仍能被真数据推动(非恒不动)", move_ok),
               ("空分母窗不推进也不清零 sustain", sus_ok)):
    ok_all &= v
    print(f"   {'✅' if v else '★✗'} {lab}")


# ═════════════ ② 认证五项分项计数 ═════════════
print("\n② 认证五项分项 —— 每次只破坏一项, 必须精确点名那一项")
CASES = [
    ("rise_bot", "瓶不升"), ("rise_cup", "杯不升"),
    ("slip_r", "右腕相对瓶滑移超限"), ("slip_l", "左腕相对杯滑移超限"),
    ("pads", "垫数不足"),
]
for tgt, desc in CASES:
    S = PourProgress(NPZ, mouth_local_bot=MB, mouth_local_cup=MC)
    B = PourProgressBatch(NPZ, num_envs=1, device="cpu",
                          mouth_local_bot=MB, mouth_local_cup=MC)
    fired_b = None
    for t in range(200):
        ks = min(S.k, S.N - 1)
        o0, o1 = OBJ[0][ks].copy(), OBJ[1][ks].copy()
        ph, tt = S.cert_phase, S.cert_t
        dz = (0.015 * min((tt + 1) / CERT_RAMP, 1.0) if ph == 1 else
              0.015 if ph == 2 else
              0.015 * max(1.0 - (tt + 1) / CERT_RET, 0.0) if ph == 3 else 0.0)
        # 只在"判分那一步"破坏目标项, 其余全部合格
        judging = (ph == 2 and tt + 1 >= CERT_HOLD)
        dz0 = dz1 = dz
        wr, wl = o1[:3].copy(), o0[:3].copy()
        pads = True
        if judging:
            if tgt == "rise_bot":
                dz1 = 0.0
            elif tgt == "rise_cup":
                dz0 = 0.0
            elif tgt == "pads":
                pads = False
        o0[2] += dz0
        o1[2] += dz1
        wr = o1[:3] + WOFF + (np.array([0.05, 0, 0]) if
                              (judging and tgt == "slip_r") else 0.0)
        wl = o0[:3] + WOFF + (np.array([0.05, 0, 0]) if
                              (judging and tgt == "slip_l") else 0.0)
        rs = S.step(o0, o1, ARM["right"][ks], ARM["left"][ks],
                    pads3=pads, wrist_r=wr, wrist_l=wl)
        T = lambda a: torch.as_tensor(np.asarray(a, np.float32))[None]  # noqa: E731
        B.step(T(o0), T(o1), T(ARM["right"][ks]), T(ARM["left"][ks]),
               torch.tensor([pads]), T(wr), T(wl))
        if "cert_fail" in rs and fired_b is None:
            fired_b = {k: int(B._acc["cf_" + k]) for k, _ in
                       [(c, 0) for c, _ in CASES]}
            got = [k for k, v in rs["cert_fail"].items() if v]
            hitB = [k for k, v in fired_b.items() if v > 0]
            good = (got == [tgt]) and (hitB == [tgt])
            ok_all &= good
            print(f"   {'✅' if good else '★✗'} 破坏「{desc:14s}」 "
                  f"标量点名={got}  批量点名={hitB}")
            break
    if fired_b is None:
        ok_all = False
        print(f"   ★✗ 破坏「{desc}」 —— 认证从未判分, 用例无效")

# 反例: 五项全合格时不得有任何计数 (证明不是"逢判必红")
S = PourProgress(NPZ, mouth_local_bot=MB, mouth_local_cup=MC)
clean = None
for t in range(200):
    ks = min(S.k, S.N - 1)
    o0, o1 = OBJ[0][ks].copy(), OBJ[1][ks].copy()
    ph, tt = S.cert_phase, S.cert_t
    dz = (0.015 * min((tt + 1) / CERT_RAMP, 1.0) if ph == 1 else
          0.015 if ph == 2 else
          0.015 * max(1.0 - (tt + 1) / CERT_RET, 0.0) if ph == 3 else 0.0)
    o0[2] += dz
    o1[2] += dz
    rs = S.step(o0, o1, ARM["right"][ks], ARM["left"][ks], pads3=True,
                wrist_r=o1[:3] + WOFF, wrist_l=o0[:3] + WOFF)
    if "cert_fail" in rs:
        clean = any(rs["cert_fail"].values()) or any(S.cert_fail_n.values())
        break
good = (clean is False)
ok_all &= good
print(f"   {'✅' if good else '★✗'} 五项全合格时零计数(非逢判必红): "
      f"{'干净' if clean is False else clean}")

print("\n✅ 第八件自检通过: NaN 防护与认证分项均可红可绿"
      if ok_all else "\n★自检未通过 —— 见上")
sys.exit(0 if ok_all else 1)
