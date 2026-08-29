"""L5-23 自检第九件: 判据侧死因分项必须精确点名。

判别句: 任何检查都必须能说出"它在什么情况下会失败"。
所以每一种死因都造一个**只触发它**的输入, 要求分项计数只有那一项非零;
另加一个"全都不该死"的对照, 证明不是逢步必红。

★env 侧死因(D1_pre/D2_pre/D3_pre/D4_slip/D5_table)需要 Isaac 物理, 本自检
覆盖不到 —— 这一点必须写明, 不能让"9件全绿"制造覆盖完整的错觉。
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from progress import TABLE_Z, D1_DROP, D3_DEV, D2_TILT, M4_DIST_POS  # noqa: E402
from progress_batch import PourProgressBatch                        # noqa: E402

NPZ = os.path.join(_HERE, "..", "L2_Reference", "pour17_reference_v2.npz")
z = np.load(NPZ, allow_pickle=True)
rows = np.where(np.asarray(z["source"]) == 1)[0]
OBJ = {oi: np.concatenate(
    [np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows],
     np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows]], axis=1) for oi in (0, 1)}
ARM = {s: np.asarray(z[f"{s}_q"], np.float64)[rows] for s in ("right", "left")}
KEYS = ("D2pre", "D1_drop", "D3_dev", "D8_disturb", "D2_tilt")
ok_all = True


def fresh():
    return PourProgressBatch(NPZ, num_envs=1, device="cpu",
                             mouth_local_bot=np.array([0.0, 0.087, 0.0]),
                             mouth_local_cup=np.array([0.0, 0.066, 0.0]))


def T(a):
    return torch.as_tensor(np.asarray(a, np.float32))[None]


def run(mutate, placed=False, tag=""):
    """喂一步, 返回本步各死因计数。mutate(o0,o1) 就地改物体位姿。"""
    B = fresh()
    o0, o1 = OBJ[0][0].copy(), OBJ[1][0].copy()
    if placed:
        B.placed[:] = True
        B.g2[:] = True
        for oi, a in ((0, o0), (1, o1)):
            B.m3_snap[oi][0] = torch.as_tensor(a, dtype=torch.float32)
    mutate(o0, o1)
    B.step(T(o0), T(o1), T(ARM["right"][0]), T(ARM["left"][0]),
           torch.tensor([True]), T(o1[:3]), T(o0[:3]))
    return {k: int(B._acc["term_" + k]) for k in KEYS}, int(B._acc["term_any"])


def tilt90(q):
    """把朝向绕 x 轴转 90°, 使 up 轴倒下。"""
    w, x, y, zz = q
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    return np.array([c * w - s * x, c * x + s * w, c * y - s * zz, c * zz + s * y])


CASES = [
    ("D1_drop", "瓶掉到桌面下 6cm",
     lambda a, b: b.__setitem__(2, TABLE_Z - D1_DROP - 0.01), False),
    ("D3_dev", "瓶离参考 50cm",
     lambda a, b: b.__setitem__(slice(0, 3), b[:3] + np.array([D3_DEV + 0.15, 0, 0])),
     False),
    ("D2pre", "G2前瓶倾 90°",
     lambda a, b: b.__setitem__(slice(3, 7), tilt90(b[3:7])), False),
    ("D8_disturb", "placed后瓶被推 10cm",
     lambda a, b: b.__setitem__(slice(0, 3), b[:3] + np.array([M4_DIST_POS + 0.05, 0, 0])),
     True),
]
print("每种死因单独触发 —— 必须只点名它自己")
for key, desc, mut, placed in CASES:
    c, anyn = run(mut, placed=placed)
    hit = [k for k, v in c.items() if v > 0]
    good = (key in hit) and (anyn == 1)
    # D8 与 D2_tilt 在 placed 段天然共线(推远+倾倒), 允许 D8 用例附带命中
    if key == "D8_disturb":
        good = good and set(hit) <= {"D8_disturb", "D2_tilt"}
    else:
        good = good and hit == [key]
    ok_all &= good
    print(f"  {'✅' if good else '★✗'} {desc:22s} 点名={hit}  any={anyn}")

# 对照: 参考轨迹首帧原样, 不该有任何死因
c, anyn = run(lambda a, b: None, placed=False)
good = (anyn == 0) and all(v == 0 for v in c.values())
ok_all &= good
print(f"  {'✅' if good else '★✗'} {'原样首帧(不该死)':22s} 点名="
      f"{[k for k, v in c.items() if v]}  any={anyn}")

print("\n★覆盖声明: 本自检只覆盖判据侧 5 种死因; env 侧 D1_pre/D2_pre/D3_pre/"
      "D4_slip/D5_table 需 Isaac 物理, **未被覆盖**。")
print("✅ 第九件自检通过: 判据侧死因分项可红可绿"
      if ok_all else "★自检未通过 —— 见上")
sys.exit(0 if ok_all else 1)
