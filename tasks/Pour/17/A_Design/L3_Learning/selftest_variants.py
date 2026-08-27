"""合法变体自检 (2026-08-28 立项): 参考放音铁则的盲区补丁。
放音只验证"完美执行能过", 本文件验证"真实执行的合法变体也能过" ——
首例: 立正但绕竖轴自转 40° 的放回 (yaw 豁免拍板的反向验证, 老全角度口径必挂)。
判据改动后必重跑。"""
import sys
import numpy as np

sys.path.insert(0, "/home/lyh/Project/RL_Correction/tasks/Pour/17/A_Design/L3_Learning")
from progress import PourProgress, _axis_tilt

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


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 + y1*w2 + z1*x2 - x1*z2, w1*z2 + z1*w2 + x1*y2 - y1*x2])


P = PourProgress(NPZ, mouth_local_bot=pm(1, 0.087), mouth_local_cup=pm(0, 0.066))
P.enter(P.N - 1, {1, 2})
th = np.radians(40)
qz = np.array([np.cos(th / 2), 0, 0, np.sin(th / 2)])
o0, o1 = P.rest[0].copy(), P.rest[1].copy()
o0[3:7] = qmul(qz, o0[3:7])
o1[3:7] = qmul(qz, o1[3:7])
stance = {s: np.asarray(z[f"{s}_q"], np.float64)[-1] for s in ("right", "left")}
for t in range(40):
    r = P.step(o0, o1, stance["right"], stance["left"], cand_ok=True)
    assert r["fail"] is None, f"yaw变体触发死线 {r['fail']}"
assert P.ms[3] and P.ms[4], "yaw 合法变体未通过 M3/M4"
# 反向的反向: 真歪 25° 的物体不许过 M3 (倾角口径仍要抓真倾倒)
P2 = PourProgress(NPZ, mouth_local_bot=pm(1, 0.087), mouth_local_cup=pm(0, 0.066))
P2.enter(P2.N - 1, {1, 2})
axis = np.array([1.0, 0.0, 0.0])
th2 = np.radians(25)
qx = np.array([np.cos(th2 / 2), *(np.sin(th2 / 2) * axis)])
o1b = P2.rest[1].copy()
o1b[3:7] = qmul(qx, o1b[3:7])
for t in range(20):
    r = P2.step(P2.rest[0], o1b, stance["right"], stance["left"], cand_ok=True)
assert not P2.ms[3], "真歪 25° 竟然过了 M3 (倾角判据失灵)"
# 药A 反向验证: M1 后恒偏 4cm 不罚 / 增偏到 7cm 必罚
P3 = PourProgress(NPZ, mouth_local_bot=pm(1, 0.087), mouth_local_cup=pm(0, 0.066))
P3.enter(5, {1})
off = np.array([0.04, 0.0, 0.0])
rows_all = np.where(np.asarray(z["source"]) == 1)[0]
obj_t = {oi: np.concatenate([np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows_all],
                             np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows_all]],
                            axis=1) for oi in (0, 1)}
armq_t = {s: np.asarray(z[f"{s}_q"], np.float64)[rows_all] for s in ("right", "left")}
tot = 0.0
for t_ in range(20):                      # 恒偏 4cm 跟着参考走
    kk = min(P3.k, P3.N - 1)
    o0 = obj_t[0][kk].copy(); o0[:3] += off
    o1 = obj_t[1][kk].copy(); o1[:3] += off
    r = P3.step(o0, o1, armq_t["right"][kk], armq_t["left"][kk], cand_ok=True)
    tot += r["leash"]
assert abs(tot) < 1e-9, f"恒偏4cm被罚 {tot} (药A失效)"
o1b = obj_t[1][min(P3.k, P3.N-1)].copy(); o1b[:3] += np.array([0.07, 0, 0]) + off - off
o1b[:3] = obj_t[1][min(P3.k, P3.N-1)][:3] + np.array([0.11, 0.0, 0.0])  # 基线0.04→现偏0.11, 增量7cm
r2 = P3.step(obj_t[0][min(P3.k, P3.N-1)], o1b,
             armq_t["right"][min(P3.k, P3.N-1)], armq_t["left"][min(P3.k, P3.N-1)],
             cand_ok=True)
assert r2["leash"] < 0, "增偏7cm未被罚 (纠偏鞭子丢了)"
print("✅ 药A反向验证: 恒偏4cm零罚 / 增偏7cm有罚")
print("✅ 合法变体自检: yaw40°放回通过 M3+M4 / 真歪25°被正确拦下")
