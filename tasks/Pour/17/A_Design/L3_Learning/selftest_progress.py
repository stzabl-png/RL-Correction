"""进度条放音自检 (运行层纪律#19): 母带自己当"完美执行"喂进度机。
必须: 时钟走到底 / M1-M2 按序触发 / 交互段皮筋零违例 / M3 在归位补测段触发。
纯 numpy, 无 Isaac。
"""
import sys
import numpy as np

sys.path.insert(0, "/home/lyh/Project/RL_Correction/tasks/Pour/17/A_Design/L3_Learning")
from progress import PourProgress, _axis_tilt

NPZ = ("/home/lyh/Project/RL_Correction/tasks/Pour/17/A_Design/"
       "L2_Reference/pour17_reference_v1.npz")
z = np.load(NPZ, allow_pickle=True)
rows = np.where(np.asarray(z["source"]) == 1)[0]

# 口部 local 偏移: 网格长轴 ±half, 取静置时世界 z 更高的一端为口
def pick_mouth(oi, half):
    q = np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows][0]
    w, x, y, zz = q / np.linalg.norm(q)
    R = np.array([[1-2*(y*y+zz*zz), 2*(x*y-w*zz), 2*(x*zz+w*y)],
                  [2*(x*y+w*zz), 1-2*(x*x+zz*zz), 2*(y*zz-w*x)],
                  [2*(x*zz-w*y), 2*(y*zz+w*x), 1-2*(x*x+y*y)]])
    up = np.array([0.0, half, 0.0])
    return up if (R @ up)[2] > (R @ (-up))[2] else -up

mb = pick_mouth(1, 0.087)     # 瓶 17.4/2
mc = pick_mouth(0, 0.066)     # 杯 13.2/2
P = PourProgress(NPZ, mouth_local_bot=mb, mouth_local_cup=mc, mouth_gate=0.12)
print(f"[放音] 交互行 {P.N} | 瓶口local={np.round(mb,3)} 杯口local={np.round(mc,3)}")
print(f"[放音] 档位分布(主档): 绿{P.tmix.count(2)} 黄{P.tmix.count(1)} 红{P.tmix.count(0)}")

obj = {oi: np.concatenate([np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows],
                           np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows]], axis=1)
       for oi in (0, 1)}
armq = {s: np.asarray(z[f"{s}_q"], np.float64)[rows] for s in ("right", "left")}

led = {"adv": 0.0, "ms": 0.0, "leash": 0.0}
gate_hist = {"obj": 0, "hand": 0}
ms_at = {}
mouth_min = 1e9
for t in range(P.N):
    k = min(P.k, P.N - 1)
    r = P.step(obj[0][k], obj[1][k], armq["right"][k], armq["left"][k], cand_ok=True)
    assert r["fail"] is None, f"参考放音触发死线 {r['fail']} @t{t} —— 71°撞60°同族!"
    led["adv"] += r["adv"]
    led["ms"] += r["ms"]
    led["leash"] += r["leash"]
    if r["gate_by"]:
        gate_hist[r["gate_by"]] += 1
    for m in (1, 2, 3):
        if P.ms[m] and m not in ms_at:
            ms_at[m] = t
    tilt = _axis_tilt(obj[1][k][3:7], P.up_b)
    if tilt >= np.radians(90):
        # 记录口口最小距 (校准 mouth_gate 用)
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

print(f"[放音] 交互段结束: 时钟 {P.k}/{P.N-1} | 行进收入 {led['adv']:.0f} "
      f"| 门统计 obj={gate_hist['obj']} hand={gate_hist['hand']} "
      f"| 皮筋总罚 {led['leash']:.3f} (应=0)")
print(f"[放音] 倾角>=90°期间 口口最小距 {mouth_min*100:.1f}cm (gate=12cm)")

# 归位补测段 (合成锚语义: 训练里策略会真放回; 母带尾巴不可信, 用静置位模拟)
rest = {oi: obj[oi][0] for oi in (0, 1)}
zfull = np.load(NPZ, allow_pickle=True)
stance = {s: np.asarray(zfull[f"{s}_q"], np.float64)[-1] for s in ("right", "left")}
for t in range(80):
    # M3 前: 臂留在交互末行; M3 后: 臂喂站姿(模拟撤退完成), 物体保持静置(=M3快照)
    ar = stance["right"] if P.ms[3] else armq["right"][-1]
    al = stance["left"] if P.ms[3] else armq["left"][-1]
    r = P.step(rest[0], rest[1], ar, al, cand_ok=True)
    led["ms"] += r["ms"]
    for m in (3, 4):
        if P.ms[m] and m not in ms_at:
            ms_at[m] = P.N + t
print(f"[放音] 里程碑: M1@{ms_at.get(1,'✗')} M2@{ms_at.get(2,'✗')} "
      f"M3@{ms_at.get(3,'✗')} M4@{ms_at.get(4,'✗')} | 收入 {led['ms']:.0f} (期望 38)")

# 死线反向验证: 40cm 外物体必须触发 D3
P2 = PourProgress(NPZ, mouth_local_bot=mb, mouth_local_cup=mc)
_far = obj[1][0].copy(); _far[0] += 0.40
r_d3 = P2.step(obj[0][0], _far, armq["right"][0], armq["left"][0], cand_ok=True)
print(f"[放音] 死线反向验证: 40cm外物体 -> fail={r_d3['fail']} (期望 D3_dev_obj1)")

ok = (P.k == P.N - 1 and abs(led["leash"]) < 1e-9
      and r_d3["fail"] == "D3_dev_obj1"
      and all(m in ms_at for m in (1, 2, 3, 4))
      and ms_at[1] <= ms_at[2] <= ms_at[3] <= ms_at[4] and led["ms"] == 38.0)
print("✅ 放音自检全部通过" if ok else "❌ 放音自检未通过 —— 见上")
sys.exit(0 if ok else 1)
