"""RSI 进入点放音自检 (#11 铁则): 每个进入点从该行照参考播到底,
必到 M4 且零死线。母带重生成后必重跑。"""
import sys
import numpy as np

sys.path.insert(0, "/home/lyh/Project/RL_Correction/tasks/Pour/17/A_Design/L3_Learning")
from progress import PourProgress

NPZ = ("/home/lyh/Project/RL_Correction/tasks/Pour/17/A_Design/"
       "L2_Reference/pour17_reference_v1.npz")
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
P = PourProgress(NPZ, mouth_local_bot=mb, mouth_local_cup=mc)
allok = True
for row, ms, label in P.entry_table():
    P.enter(row, ms)
    T = P.N
    rest = {oi: obj[oi][0] for oi in (0, 1)}
    m4, failmsg = False, None
    for t in range(T + 100):
        k = min(P.k, T - 1)
        end = (P.k >= T - 1) or P.ms[3]
        o0, o1 = (rest[0], rest[1]) if end else (obj[0][k], obj[1][k])
        ar = stance["right"] if P.ms[3] else armq["right"][k]
        al = stance["left"] if P.ms[3] else armq["left"][k]
        r = P.step(o0, o1, ar, al, cand_ok=True)
        if r["fail"]:
            failmsg = r["fail"]
            break
        if P.ms[4]:
            m4 = True
            break
    ok = m4 and failmsg is None
    allok &= ok
    print(f"[RSI放音] {label:12s} row={row:3d} 预置{sorted(ms)} -> "
          f"{'✅' if ok else '❌ fail=' + str(failmsg)}")
print("✅ RSI 全点位放音自洽" if allok else "❌ 有点位不自洽")
sys.exit(0 if allok else 1)
