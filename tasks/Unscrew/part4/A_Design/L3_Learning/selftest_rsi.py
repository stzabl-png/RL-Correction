"""RSI 出生点自检: entry_table 各点位预置后, 从该点位继续完美放音能走完剩余链。"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "C_Wiring"))
from progress import PLACED_HOLD, M4_HOLD, UnscrewProgress  # noqa: E402
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

P0 = UnscrewProgress(NPZ)
table = P0.entry_table(unlocked={1, 2, 3})
print(f"[RSI] 全解锁出生表: {[(r, sorted(map(str, m)), lb) for r, m, lb in table]}")
assert [lb for _, _, lb in table] == ["t0", "g1", "g2", "g3", "ret"]

for row, preset, label in table:
    P = UnscrewProgress(NPZ)
    P.enter(row, preset)
    t, done = 0, False
    from progress import CERT_RAMP, CERT_RET
    while t < 1500 and not P.g[4]:
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
        o0[2] += dz; o1[2] += dz
        rel = P.g[2] and k >= P.k_sep
        r = P.step(o0, o1, ar, al, True, o1[:3] + WOFF, o0[:3] + WOFF,
                   screw_released=rel)
        assert r["fail"] is None, f"[{label}] 死线 {r['fail']} @t{t} k{P.k}"
        t += 1
    assert P.g[4], f"[{label}] {t} 步未通关 (k={P.k}/{P.N - 1})"
    print(f"[RSI] {label:4s} (row {row}, 预置 {sorted(map(str, preset))}) "
          f"-> G4 @{t} 步 ✅")
print("[RSI] ★全点位可走完")
