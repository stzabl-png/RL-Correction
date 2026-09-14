"""CPU 自检: Clean/3 判据合约 (无 Isaac)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from progress_batch import CleanGeometry, CleanProgressBatch, CleanSignals  # noqa: E402

Q_CI = torch.tensor([0.7071068, 0.7071068, 0.0, 0.0])          # 输入 +y -> 规范 +z
Q_ID = torch.tensor([[1.0, 0.0, 0.0, 0.0]])


def q_from_axis(axis, deg):
    a = np.radians(deg); v = np.asarray(axis, float) / np.linalg.norm(axis)
    return torch.tensor([[np.cos(a / 2), *(np.sin(a / 2) * v)]], dtype=torch.float32)


def qmul(a, b):
    from progress_batch import quat_mul
    return quat_mul(a, b)


def main():
    g = CleanGeometry()
    S = CleanSignals(1, "cpu", Q_CI, g)
    P = CleanProgressBatch(1, "cpu", S, g)
    ids = torch.tensor([0])
    plate_p = torch.tensor([[0.0, 0.0, 1.0]])
    plate_q = qmul(Q_ID, Q_CI)                    # 盘水平: 规范系 = 单位 -> 输入系 = R_ci
    P.reset(ids, plate_p)
    # 海绵平放在碟心 (规范 z = 平底 -0.73cm + face_off): 应"擦到"
    z_on = float(S.seat_height(torch.zeros(1, 2), torch.zeros(1))[0]) + 0.001   # 刚体落座: 端头搭在斜坡上
    sp_q = qmul(Q_ID, Q_CI)
    sig = S(plate_p, plate_q, plate_p + torch.tensor([[0.0, 0.0, z_on]]), sp_q)
    assert bool(sig["contact"][0]) and abs(float(sig["gap_min"][0]) - 0.001) < 2e-3, sig["gap_min"]
    assert 0.027 < z_on < 0.030, z_on            # 端头搭缘: 比碟心平底高 ~1.6cm
    assert float(sig["plate_tilt"][0]) < 1e-4
    out = P.step(sig, plate_p, torch.tensor([False]))
    assert out["gates"][0].tolist()[0] is True and float(out["coverage"][0]) > 0.05
    # 悬空 3cm: 不擦到, 覆盖不涨, 行程不涨
    sig2 = S(plate_p, plate_q, plate_p + torch.tensor([[0.0, 0.0, z_on + 0.03]]), sp_q)   # 悬空 3cm
    assert not bool(sig2["contact"][0])
    c0 = float(P.coverage()[0]); out2 = P.step(sig2, plate_p, torch.tensor([False]))
    assert float(out2["coverage"][0]) == c0 and float(out2["travel"][0]) == 0.0
    # 贴着盘面横扫: 覆盖单调涨, 行程累计, 出盘沿部分不计
    P.reset(ids, plate_p)
    prev = 0.0
    for x in np.linspace(-0.10, 0.10, 21):
        zx = float(S.seat_height(torch.tensor([[x, 0.0]]), torch.zeros(1))[0]) + 0.001
        sigx = S(plate_p, plate_q, plate_p + torch.tensor([[x, 0.0, zx]], dtype=torch.float32), sp_q)
        o = P.step(sigx, plate_p, torch.tensor([False]))
        assert float(o["coverage"][0]) >= prev - 1e-6; prev = float(o["coverage"][0])
    assert 0.15 < prev < 0.9, prev
    assert 0.15 < float(o["travel"][0]) < 0.21, o["travel"]        # 接触中的行程 ≈ 盘内 17cm
    # 盘倾 20°: hold_ok 假, 成功不可点亮
    tilt_q = qmul(q_from_axis([1, 0, 0], 20.0), Q_CI)
    sigt = S(plate_p, tilt_q, plate_p + torch.tensor([[0.0, 0.0, z_on]]), sp_q)
    ot = P.step(sigt, plate_p, torch.tensor([False]))
    assert not bool(ot["tilt_ok"][0]) and abs(float(np.degrees(sigt["plate_tilt"][0])) - 20.0) < 0.5
    # 海绵斜 30° 立起: 只有一条边擦到 (足迹点部分)
    sq30 = qmul(qmul(Q_ID, q_from_axis([1, 0, 0], 30.0)), Q_CI)
    sig30 = S(plate_p, plate_q, plate_p + torch.tensor([[0.0, 0.0, z_on + 0.015]]), sq30)   # 斜边下探 3.3cm, 抬 1.5cm 后只剩一条边擦到
    assert 0 < int(sig30["touch"][0].sum()) < sig30["touch"].shape[1] // 2
    print("Clean3 progress self-test: PASS")


if __name__ == "__main__":
    main()
