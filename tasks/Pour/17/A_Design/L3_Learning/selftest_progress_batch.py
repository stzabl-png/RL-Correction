"""批量版一致性自检: env0-2 完美放音须与标量版逐位一致; env3 喂静止垃圾须全零。
另验 TB 逐关率记账 (pop_rates)。纯 CPU torch。"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/lyh/Project/RL_Correction/tasks/Pour/17/A_Design/L3_Learning")
from progress import PourProgress
from progress_batch import PourProgressBatch

NPZ = ("/home/lyh/Project/RL_Correction/tasks/Pour/17/A_Design/"
       "L2_Reference/pour17_reference_v1.npz")
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

sum_s = {"adv": 0.0, "ms": 0.0, "leash": 0.0}
sum_b = torch.zeros(4, 3)
mism = 0
for t in range(T + 80):
    ks = min(S.k, T - 1)
    o0s, o1s = obj[0][ks], obj[1][ks]
    if t >= T:                                        # 归位补测段
        o0s, o1s = rest[0], rest[1]
    _arS = armq["right"][min(ks, T - 1)]
    _alS = armq["left"][min(ks, T - 1)]
    if t >= T and S.ms[3]:
        _arS, _alS = STANCE["right"], STANCE["left"]
    rs = S.step(o0s, o1s, _arS, _alS, cand_ok=True)
    sum_s["adv"] += rs["adv"]; sum_s["ms"] += rs["ms"]; sum_s["leash"] += rs["leash"]
    # 批量: env0-2 同步喂各自时钟行(与标量同源), env3 喂静止垃圾
    kb = B.k.clamp(max=T - 1).numpy()
    o0 = np.stack([obj[0][kb[i]] if (t < T or i == 3) else rest[0] for i in range(4)])
    o1 = np.stack([obj[1][kb[i]] if (t < T or i == 3) else rest[1] for i in range(4)])
    ar = np.stack([armq["right"][kb[i]] for i in range(4)])
    al = np.stack([armq["left"][kb[i]] for i in range(4)])
    if t >= T:
        for i in range(3):
            if bool(B.ms3[i]):
                ar[i], al[i] = STANCE["right"], STANCE["left"]
    if t >= T:
        for i in range(3):
            o0[i], o1[i] = rest[0], rest[1]
    o0[3], o1[3] = rest[0], rest[1]                   # env3 物体永远静止
    ar[3], al[3] = armq["right"][0], armq["left"][0]  # env3 手永远首行
    rb = B.step(torch.tensor(o0, dtype=torch.float32),
                torch.tensor(o1, dtype=torch.float32),
                torch.tensor(ar, dtype=torch.float32),
                torch.tensor(al, dtype=torch.float32),
                cand_ok=torch.tensor([True, True, True, False]))
    sum_b += torch.stack([rb["adv"], rb["ms"], rb["leash"]], dim=1)
    if int(rb["clock"][0]) != S.k:
        mism += 1

print(f"[批量自检] 标量: adv={sum_s['adv']:.0f} ms={sum_s['ms']:.0f} "
      f"leash={sum_s['leash']:.3f} | M1/2/3={S.ms}")
print(f"[批量自检] env0: adv={sum_b[0,0]:.0f} ms={sum_b[0,1]:.0f} "
      f"leash={sum_b[0,2]:.3f} | 时钟错位次数={mism}")
print(f"[批量自检] env3(垃圾): adv={sum_b[3,0]:.0f} ms={sum_b[3,1]:.0f} "
      f"clock={int(B.k[3])} (期望: 时钟停滞, 里程碑0)")
B.reset_idx(torch.arange(4))
rates = B.pop_rates()
print(f"[批量自检] TB逐关率: {dict((k_, round(v_, 2)) for k_, v_ in rates.items())} "
      f"(期望 gate1=0.75 gate2=0.75 gate3=0.75, env3 拉低)")
ok = (abs(sum_b[0, 0] - sum_s["adv"]) < 1e-3 and abs(sum_b[0, 1] - sum_s["ms"]) < 1e-3
      and abs(sum_b[0, 2].item() - sum_s["leash"]) < 1e-3 and mism == 0
      and sum_b[3, 1] == 0 and int(B.k[3]) < 5
      and abs(rates["sr/gate4"] - 0.75) < 1e-6)
print("✅ 批量一致性+TB记账 全部通过" if ok else "❌ 未通过 —— 见上")
sys.exit(0 if ok else 1)
