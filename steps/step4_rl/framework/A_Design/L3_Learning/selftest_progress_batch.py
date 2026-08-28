"""批量版一致性自检 (L5-1): env0-2 完美放音(含认证编舞)须与标量版逐位一致;
env3 喂静止垃圾: G1可立(垫在)、认证3败、G2不立、时钟停滞、无G3+。
另验 TB 逐关率记账 (pop_rates: 挣的口径)。纯 CPU torch。消费 v2 母带。"""
import sys
import numpy as np
import torch

_HERE = __import__("os").path.dirname(__import__("os").path.abspath(__file__))
sys.path.insert(0, _HERE)
from progress import PourProgress, CERT_RAMP, CERT_RET
from progress_batch import PourProgressBatch

NPZ = __import__("os").path.join(_HERE, "..", "L2_Reference", "reference_v2.npz")
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

mb, mc = pick_mouth(1, 0.087), pick_mouth(0, 0.066)
S = PourProgress(NPZ, mouth_local_bot=mb, mouth_local_cup=mc)
B = PourProgressBatch(NPZ, num_envs=4, device="cpu",
                      mouth_local_bot=mb, mouth_local_cup=mc)

obj = {oi: np.concatenate([np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows],
                           np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows]], axis=1)
       for oi in (0, 1)}
armq = {s: np.asarray(z[f"{s}_q"], np.float64)[rows] for s in ("right", "left")}
T = B.N_ROW
rest = {oi: obj[oi][0] for oi in (0, 1)}
STANCE = {s: np.asarray(z[f"{s}_q"], np.float64)[-1] for s in ("right", "left")}
WOFF = np.array([0.0, 0.0, 0.10])

def dz_of(phase, t_):
    if phase == 1:
        return 0.015 * min((t_ + 1) / CERT_RAMP, 1.0)
    if phase == 2:
        return 0.015
    if phase == 3:
        return 0.015 * max(1.0 - (t_ + 1) / CERT_RET, 0.0)
    return 0.0

sum_s = {"adv": 0.0, "ms": 0.0, "leash": 0.0, "wage": 0.0}
sum_b = torch.zeros(4, 4)
mism = 0
TT = T + 130
for t in range(TT):
    ks = min(S.k, T - 1)
    dz_s = dz_of(S.cert_phase, S.cert_t) if not S.g[2] else 0.0
    o0s, o1s = obj[0][ks].copy(), obj[1][ks].copy()
    if t >= T + 40:                                   # 归位补测段
        o0s, o1s = rest[0].copy(), rest[1].copy()
    o0s[2] += dz_s; o1s[2] += dz_s
    _arS = armq["right"][ks]
    _alS = armq["left"][ks]
    if t >= T + 40 and S.placed:
        _arS, _alS = STANCE["right"], STANCE["left"]
    rs = S.step(o0s, o1s, _arS, _alS, pads3=True,
                wrist_r=o1s[:3] + WOFF, wrist_l=o0s[:3] + WOFF)
    for k_ in ("adv", "ms", "leash", "wage"):
        sum_s[k_] += rs[k_]
    # 批量: env0-2 同编舞, env3 喂静止垃圾(垫在但物体不动)
    kb = B.k.clamp(max=T - 1).numpy()
    o0 = np.stack([obj[0][kb[i]].copy() for i in range(4)])
    o1 = np.stack([obj[1][kb[i]].copy() for i in range(4)])
    ar = np.stack([armq["right"][kb[i]] for i in range(4)])
    al = np.stack([armq["left"][kb[i]] for i in range(4)])
    for i in range(3):
        dzb = dz_of(int(B.cert_phase[i]), int(B.cert_t[i])) \
            if not bool(B.g2[i]) else 0.0
        if t >= T + 40:
            o0[i], o1[i] = rest[0].copy(), rest[1].copy()
            if bool(B.placed[i]):
                ar[i], al[i] = STANCE["right"], STANCE["left"]
        o0[i][2] += dzb; o1[i][2] += dzb
    o0[3], o1[3] = rest[0], rest[1]                   # env3 物体永远静止
    ar[3], al[3] = armq["right"][0], armq["left"][0]  # env3 手永远首行
    wr = np.stack([o1[i][:3] + WOFF for i in range(4)])
    wl = np.stack([o0[i][:3] + WOFF for i in range(4)])
    rb = B.step(torch.tensor(o0, dtype=torch.float32),
                torch.tensor(o1, dtype=torch.float32),
                torch.tensor(ar, dtype=torch.float32),
                torch.tensor(al, dtype=torch.float32),
                pads3=torch.tensor([True, True, True, True]),
                wrist_r=torch.tensor(wr, dtype=torch.float32),
                wrist_l=torch.tensor(wl, dtype=torch.float32))
    sum_b += torch.stack([rb["adv"], rb["ms"], rb["leash"], rb["wage"]], dim=1)
    if int(rb["clock"][0]) != S.k:
        mism += 1

print(f"[批量自检] 标量: adv={sum_s['adv']:.0f} ms={sum_s['ms']:.0f} "
      f"leash={sum_s['leash']:.3f} wage={sum_s['wage']:.2f} | G={S.g} placed={S.placed}")
print(f"[批量自检] env0: adv={sum_b[0,0]:.0f} ms={sum_b[0,1]:.0f} "
      f"leash={sum_b[0,2]:.3f} wage={sum_b[0,3]:.2f} | 时钟错位次数={mism}")
print(f"[批量自检] env3(垃圾): ms={sum_b[3,1]:.0f} clock={int(B.k[3])} g2={bool(B.g2[3])} "
      f"cert_try={int(B.cert_try[3])} (期望: G1的+5, 时钟0, g2 False, try=3)")
B.reset_idx(torch.arange(4))
rates = B.pop_rates()
print(f"[批量自检] TB逐关率: {dict((k_, round(v_, 2)) for k_, v_ in rates.items())} "
      f"(期望 gate1=1.0 gate2..4=0.75, env3 拉低)")
ok = (abs(sum_b[0, 0] - sum_s["adv"]) < 1e-3 and abs(sum_b[0, 1] - sum_s["ms"]) < 1e-3
      and abs(sum_b[0, 2].item() - sum_s["leash"]) < 1e-3
      and abs(sum_b[0, 3].item() - sum_s["wage"]) < 1e-3 and mism == 0
      and sum_b[3, 1].item() == 5.0 and int(B.k[3]) == 0
      and abs(rates["sr/gate4"] - 0.75) < 1e-6
      and abs(rates["sr/gate1"] - 1.0) < 1e-6)
print("✅ 批量一致性+TB记账 全部通过" if ok else "❌ 未通过 —— 见上")
sys.exit(0 if ok else 1)
