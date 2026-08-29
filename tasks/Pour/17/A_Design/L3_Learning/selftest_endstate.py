"""L5-27 自检第十三件: 第四道出厂检查(末态-判据一致性)必须能红能绿。

★这道检查的存在理由: 前三道(关节连续性 / 峰值保全 / IK 精度)查的都是
"带子自己顺不顺", 没有一道查"带子的结尾能不能通过判分标准"。
v2 母带正是死在这里 —— 照着带子做必然扣分, 想拿分就得违抗带子。

按家族纪律, 用**真母带 v2 的实测数据**当"必然该红"的输入(它确实该红),
再造修好的版本验证该绿。
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from progress import endstate_consistency, M3_POS, M3_ROT, M3_HOLD  # noqa: E402

NPZ = os.path.join(_HERE, "..", "L2_Reference", "pour17_reference_v2.npz")
z = np.load(NPZ, allow_pickle=True)
rows = np.where(np.asarray(z["source"]) == 1)[0]
ok_all = True


def show(tag, ok, inf, want):
    global ok_all
    good = (ok == want)
    ok_all &= good
    print(f"  {'✅' if good else '★✗'} {tag:32s} {'绿' if ok else '红'} "
          f"(期望{'绿' if want else '红'})  末行{inf['end_pos_cm']:.2f}cm "
          f"倾角{inf['end_tilt_deg']:.1f}° 末尾连续{inf['tail_ok_rows']}行")


print("① ★真实反例: 现役 v2 母带(未修)必须红 —— 它就是 G4 封顶的原因")
for oi, nm in ((1, "瓶 object_1"), (0, "杯 object_0")):
    P = np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows]
    Q = np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows]
    ok, inf = endstate_consistency(P, Q)
    # 瓶必须红(5.6cm), 杯位置合格但末尾保持不足也可能红 —— 分别声明期望
    want = False if oi == 1 else ok      # 杯的真实结果原样记录, 瓶必须红
    if oi == 1:
        show(f"v2 {nm}(应红)", ok, inf, False)
        if ok:
            print("     ★这条若变绿说明检查失效 —— v2 实测瓶末行距首行 5.60cm")
    else:
        print(f"  ·  v2 {nm} 实测: {'绿' if ok else '红'} "
              f"末行{inf['end_pos_cm']:.2f}cm 末尾连续{inf['tail_ok_rows']}行"
              + ("" if ok else f"  原因={inf['bad']}"))

print("\n② 修好的输入必须绿(把末尾 30 行余弦归位到首行)")


def qslerp(q0, q1, a):
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)


def home(P, Q, K=30):
    P, Q = P.copy(), Q.copy()
    n = len(P)
    for i in range(n - K, n):
        a = 0.5 * (1 - np.cos(np.pi * (i - (n - K) + 1) / K))
        P[i] = (1 - a) * P[i] + a * P[0]
        Q[i] = qslerp(Q[i], Q[0], a)
    return P, Q


for oi, nm in ((1, "瓶"), (0, "杯")):
    P = np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows]
    Q = np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows]
    ok, inf = endstate_consistency(*home(P, Q))
    show(f"归位后 {nm}(应绿)", ok, inf, True)

print("\n③ 边界: 恰好卡在容差上/下必须分得开")
n = 200
Q = np.tile(np.array([1.0, 0, 0, 0]), (n, 1))
for dist_cm, want in ((2.9, True), (3.1, False)):
    P = np.zeros((n, 3))
    P[-1] = [dist_cm / 100.0, 0, 0]          # 只有末行偏
    P[-40:] = np.linspace(0, dist_cm / 100.0, 40)[:, None] * np.array([1.0, 0, 0])
    ok, inf = endstate_consistency(P, Q)
    show(f"末行偏 {dist_cm}cm", ok, inf, want)

print("\n④ 位置达标但末尾保持不足必须红(placed 要连续 15 行)")
P = np.zeros((n, 3))
P[-8:] = 0.0
P[-30:-8] = np.array([0.05, 0, 0])          # 倒数第 9~30 行偏 5cm
ok, inf = endstate_consistency(P, Q)
show(f"末行达标但仅连续 8 行", ok, inf, False)

print("\n✅ 第十三件自检通过: 第四道出厂检查可红可绿"
      if ok_all else "\n★自检未通过 —— 见上")
sys.exit(0 if ok_all else 1)
