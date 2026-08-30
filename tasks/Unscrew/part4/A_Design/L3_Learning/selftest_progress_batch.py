"""批量版一致性自检: env0 完美放音与标量版逐位相同; 垃圾 env 零误放。
纯 torch/numpy, 无 Isaac。"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "C_Wiring"))
from progress import CERT_RAMP, CERT_RET, UnscrewProgress  # noqa: E402
from progress_batch import UnscrewProgressBatch  # noqa: E402
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

P = UnscrewProgress(NPZ)
N_ENV = 3
B = UnscrewProgressBatch(NPZ, num_envs=N_ENV, device="cpu")
# env0=完美放音; env1=垃圾(物体乱飞); env2=站位不动
mism = []
led_s = {"adv": 0.0, "ms": 0.0, "leash": 0.0}
led_b = {"adv": 0.0, "ms": 0.0, "leash": 0.0}
T_MAX = 900
for t in range(T_MAX):
    # ---- 标量喂法 (完美执行) ----
    if P.cert_phase == 1:
        dz = 0.015 * min((P.cert_t + 1) / CERT_RAMP, 1.0)
    elif P.cert_phase == 2:
        dz = 0.015
    elif P.cert_phase == 3:
        dz = 0.015 * max(1.0 - (P.cert_t + 1) / CERT_RET, 0.0)
    else:
        dz = 0.0
    if not P.g[2]:
        k = 0
    else:
        k = min(P.k, P.N - 1)
    done_s = P.g[4]
    if done_s:
        o0, o1 = P.end[0], P.end[1]
        ar, al = stance["right"], stance["left"]
    elif P.g[3] and P.k >= P.N - 1:
        o0, o1 = P.end[0], P.end[1]
        ar, al = stance["right"], stance["left"]
    else:
        o0, o1 = obj[0][k], obj[1][k]
        ar, al = armq["right"][k], armq["left"][k]
    o0 = o0.copy(); o1 = o1.copy()
    o0[2] += dz; o1[2] += dz
    rel = P.g[2] and min(P.k, P.N - 1) >= P.k_sep
    rs = P.step(o0, o1, ar, al, True, o1[:3] + WOFF, o0[:3] + WOFF,
                screw_released=rel)
    # ---- 批量喂法: env0 同款, env1 乱飞, env2 停在站位 ----
    o0b = torch.tensor(np.stack([o0, obj[0][0] + [9, 9, 9, 0, 0, 0, 0],
                                 obj[0][0]]), dtype=torch.float32)
    o1b = torch.tensor(np.stack([o1, obj[1][0] + [9, 9, 9, 0, 0, 0, 0],
                                 obj[1][0]]), dtype=torch.float32)
    arb = torch.tensor(np.stack([ar, armq["right"][0], armq["right"][0]]),
                       dtype=torch.float32)
    alb = torch.tensor(np.stack([al, armq["left"][0], armq["left"][0]]),
                       dtype=torch.float32)
    p3 = torch.tensor([True, False, False])
    wr = o1b[:, :3] + torch.tensor(WOFF, dtype=torch.float32)
    wl = o0b[:, :3] + torch.tensor(WOFF, dtype=torch.float32)
    relb = torch.tensor([bool(rel), False, False])
    rb = B.step(o0b, o1b, arb, alb, p3, wr, wl, screw_released=relb)
    for kk in ("adv", "ms", "leash"):
        led_s[kk] += float(rs[kk])
        led_b[kk] += float(rb[kk][0])
        if abs(float(rs[kk]) - float(rb[kk][0])) > 1e-4:
            mism.append((t, kk, float(rs[kk]), float(rb[kk][0])))
    if int(P.k) != int(B.k[0]):
        mism.append((t, "k", int(P.k), int(B.k[0])))
    if P.g[4]:
        break
print(f"[批量] env0: 标量账 {led_s} vs 批量账 {led_b} | 失配 {len(mism)}")
if mism:
    print("  前10:", mism[:10])
assert not mism, "标量/批量失配"
assert P.g[4] and bool(B.g4[0]), "env0 未通关"
# 垃圾 env: 一步都不该有收入 (env1 D3 即死), env2 只挣 G1/工资
assert not bool(B.g2[1]) and bool(B.done[1]), "垃圾 env 未判死"
assert not bool(B.g2[2]), "站位 env 不该过认证"
print("[批量] ★一致性全绿 (env0 逐位同, 垃圾/站位 env 零误放)")
