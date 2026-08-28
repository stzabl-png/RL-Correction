"""RSI 进入点放音自检 (#11 铁则, L5-1 渐进版): 全解锁 entry_table 的每个进入点
从该行照参考播到底, 必到 G4 且零死线。t0/g1 点位含认证编舞。母带/判据改后必重跑。"""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from progress import PourProgress, CERT_RAMP, CERT_RET

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
stance = {s: np.asarray(z[f"{s}_q"], np.float64)[-1] for s in ("right", "left")}
WOFF = np.array([0.0, 0.0, 0.10])
P = PourProgress(NPZ, mouth_local_bot=mb, mouth_local_cup=mc)
et = P.entry_table(unlocked={1, 2, 3})
print(f"[RSI放音] 全解锁进入点: {[(r_, lb) for r_, _, lb in et]}")
allok = True
for row, ms, label in et:
    P.enter(row, ms)
    T = P.N
    rest = {oi: obj[oi][0] for oi in (0, 1)}
    g4, failmsg = False, None
    for t in range(T + 160):
        k = min(P.k, T - 1)
        if not P.g[2]:                                # 站位+认证编舞
            if P.cert_phase == 1:
                dz = 0.015 * min((P.cert_t + 1) / CERT_RAMP, 1.0)
            elif P.cert_phase == 2:
                dz = 0.015
            elif P.cert_phase == 3:
                dz = 0.015 * max(1.0 - (P.cert_t + 1) / CERT_RET, 0.0)
            else:
                dz = 0.0
            o0 = obj[0][k].copy(); o1 = obj[1][k].copy()
            o0[2] += dz; o1[2] += dz
            ar, al = armq["right"][k], armq["left"][k]
        else:
            end = (P.k >= T - 1) or P.placed
            o0, o1 = (rest[0], rest[1]) if end else (obj[0][k], obj[1][k])
            ar = stance["right"] if P.placed else armq["right"][k]
            al = stance["left"] if P.placed else armq["left"][k]
        r = P.step(o0, o1, ar, al, pads3=True,
                   wrist_r=np.asarray(o1[:3]) + WOFF,
                   wrist_l=np.asarray(o0[:3]) + WOFF)
        if r["fail"]:
            failmsg = r["fail"]
            break
        if P.g[4]:
            g4 = True
            break
    ok = g4 and failmsg is None
    allok &= ok
    print(f"[RSI放音] {label:6s} row={row:3d} 预置{sorted(ms, key=str)} -> "
          f"{'✅' if ok else '❌ fail=' + str(failmsg)}")
print("✅ RSI 全点位放音自洽" if allok else "❌ 有点位不自洽")
sys.exit(0 if allok else 1)
