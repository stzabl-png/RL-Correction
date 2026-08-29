"""★闸门自审 (L5-15, 判据家族第6件): 证明检查本身不是恒真断言。

判别法 (RL_Training 提炼): **任何检查都必须能说出"它在什么情况下会失败"** ——
说不出失败条件的检查就是恒真断言, 而它的绿灯零信息量。
可执行形式: **造一个必然该红的输入喂进去, 看它红不红。**

本文件审的是"峰值保全检查"(build_ref_v5.py 的出厂检查之一): 它上线以来只见过绿
(八项 0% 损失), 从未被证明能红。

判别句: 任何检查都必须能说出"它在什么情况下会失败" —— 说不出的就是恒真断言。
峰值保全检查上线以来只见过绿(八项 0% 损失), 从未被证明能红。
"""
import sys

import numpy as np

FEAT_TOL = 0.05
UP = np.array([0.0, 1.0, 0.0])


def _task_features(P, Q):
    tl = []
    for q in Q:
        w, x, y, z = q / np.linalg.norm(q)
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                      [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                      [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])
        v = R @ UP
        tl.append(np.degrees(np.arccos(np.clip(v[2] / max(np.linalg.norm(v), 1e-9),
                                               -1, 1))))
    tl = np.array(tl)
    return {"倾角峰值": float(tl.max()), "倾角行程": float(tl.max() - tl.min()),
            "高度行程": float((P[:, 2].max() - P[:, 2].min()) * 100),
            "水平行程": float(np.linalg.norm(P[:, :2] - P[0, :2], axis=1).max() * 100)}


def gate(before, after, tag):
    bad = []
    for k in before:
        rel = (before[k] - after[k]) / max(abs(before[k]), 1e-9)
        if rel > FEAT_TOL:
            bad.append(f"{k} {before[k]:.1f}->{after[k]:.1f}({-rel*100:+.1f}%)")
    print(f"  {tag}: " + ("★红 " + " | ".join(bad) if bad else "绿"))
    return bool(bad)


# ★路径必须相对 __file__: 早先写死本机绝对路径, 本机永远绿、三台远端永远
# FileNotFoundError —— "只在作者机器上有效的检查"等于没有检查。
import os  # noqa: E402
_HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(_HERE, "..", "L2_Reference", "pour17_reference_v2.npz"))
rows = np.where(np.asarray(d["source"]) == 1)[0]
P = np.asarray(d["obj_pos_1"], np.float64)[rows]
Q = np.asarray(d["obj_quat_1"], np.float64)[rows]
base = _task_features(P, Q)
print(f"基线特征: {dict((k, round(v, 1)) for k, v in base.items())}\n")

print("对照组(应该绿):")
ok0 = gate(base, _task_features(P, Q), "原样不动")

print("\n必然该红的输入(每种破坏一个任务特征):")
fired = []
# ① 削掉倾角峰值: 把倾角最大的那些帧换成中位帧的朝向
tl = np.array([_task_features(P[i:i+1], Q[i:i+1])["倾角峰值"] for i in range(len(Q))])
Q1 = Q.copy()
hi = np.argsort(tl)[-int(len(tl) * 0.15):]
Q1[hi] = Q[int(np.argsort(tl)[len(tl) // 2])]
fired.append(gate(base, _task_features(P, Q1), "① 削掉倾角峰值(前15%高倾角帧压平)"))
# ② 压掉高度行程: z 向中位收缩一半
P2 = P.copy()
P2[:, 2] = np.median(P[:, 2]) + (P[:, 2] - np.median(P[:, 2])) * 0.5
fired.append(gate(base, _task_features(P2, Q), "② 高度行程压一半"))
# ③ 压掉水平行程
P3 = P.copy()
P3[:, :2] = P[0, :2] + (P[:, :2] - P[0, :2]) * 0.5
fired.append(gate(base, _task_features(P3, Q), "③ 水平行程压一半"))
# ④ 只损失 3% (低于 5% 容差) —— 应该保持绿, 验它不是"逢改必红"
P4 = P.copy()
P4[:, 2] = np.median(P[:, 2]) + (P[:, 2] - np.median(P[:, 2])) * 0.97
ok4 = gate(base, _task_features(P4, Q), "④ 只损失3%(应保持绿)")

print(f"\n判定: 对照绿={not ok0} | 三种破坏全红={all(fired)} | 3%不误报={not ok4}")
_ok = (not ok0) and all(fired) and (not ok4)
print("✅ 该闸门可红可绿, 不是恒真断言" if _ok else "★该闸门有问题 —— 见上")
sys.exit(0 if _ok else 1)
