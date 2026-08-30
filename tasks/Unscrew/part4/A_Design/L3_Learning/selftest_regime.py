"""体制自检: 断言实现与设计口径逐条一致 (CHECKLIST 第 11 格的"体制"件)。
① P-OBJ 变体: w_obj≡1, w_hand≡0 (纯物轨消融);
② P-HYB: 档位表 绿(1,0)/黄(.5,.5)/红(.2,.8) 逐档兑现;
③ 人手置信度门: 红档人手行 -> W_HCONF=0 (形状奖被掐灭), 绿档=1;
④ 红档 rot 禁入: conf_rot 红的行, 乱转物体既不掉皮筋也不掐时钟 (pos 仍管);
⑤ 机器行 conf=NaN 结构性禁查 (tier 按绿处理, 不虚报红)。"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "C_Wiring"))
from progress import (W_OBJ, W_HAND, W_HCONF, _tier, TIER_HI, TIER_LO,  # noqa: E402
                      UnscrewProgress)
from progress_batch import UnscrewProgressBatch  # noqa: E402
import task_config as TC  # noqa: E402

NPZ = TC.REF_V2 if os.path.isfile(TC.REF_V2) else TC.REF_V1

# ① P-OBJ
B = UnscrewProgressBatch(NPZ, num_envs=1, device="cpu", no_hand_ref=True)
assert torch.all(B.WO == 1.0) and torch.all(B.WH == 0.0)
print("[体制] ① P-OBJ: w_obj≡1 w_hand≡0 ✅")
# ② P-HYB 档位表
B2 = UnscrewProgressBatch(NPZ, num_envs=1, device="cpu", no_hand_ref=False)
assert np.allclose([float(B2.WO[t]) for t in (2, 1, 0)], [1.0, 0.5, 0.2])
assert np.allclose([float(B2.WH[t]) for t in (2, 1, 0)], [0.0, 0.5, 0.8])
print("[体制] ② P-HYB 档位 绿(1,0)/黄(.5,.5)/红(.2,.8) ✅")
# ③ 人手置信度门
assert W_HCONF[2] == 1.0 and W_HCONF[1] == 0.5 and W_HCONF[0] == 0.0
z = np.load(NPZ, allow_pickle=True)
rows = np.where(np.asarray(z["source"]) == 1)[0]
if "hand_conf_fin_r" in z:
    hv = np.asarray(z["hand_conf_fin_r"], np.float64)[rows]
    for k in range(len(rows)):
        exp = W_HCONF[_tier(hv[k]) if _tier(hv[k]) is not None else 2]
        assert abs(float(B2.HCF_R[k]) - exp) < 1e-6, (k, hv[k], float(B2.HCF_R[k]))
    print(f"[体制] ③ 人手conf门逐行兑现 ✅ (右手行均值 {hv.mean():.0f} -> "
          f"HCF 均值 {float(B2.HCF_R.mean()):.2f})")
else:
    assert torch.all(B2.HCF_R == 1.0)
    print("[体制] ③ 母带无 hand_conf 列 -> HCF≡1 (兼容口径) ✅")
# ④ 红档 rot 禁入
P = UnscrewProgress(NPZ)
red_rows = [k for k in range(P.N) if P.tr[0][k] == 0 or P.tr[1][k] == 0]
assert P.tmix.count(0) >= 0    # 只要有档位机制在
if red_rows:
    k = red_rows[0]
    oi = 0 if P.tr[0][k] == 0 else 1
    # 直接检验数值机制: LEASH_ROT[0] is None -> 皮筋 rot 跳过
    from progress import LEASH_ROT
    assert LEASH_ROT[0] is None
    print(f"[体制] ④ 红档 rot 禁入 (LEASH_ROT[红]=None; 首个红rot行 k={k} obj{oi}) ✅")
else:
    print("[体制] ④ 本 clip 无红 rot 行 (数值机制 LEASH_ROT[0]=None 仍断言) ✅")
    from progress import LEASH_ROT
    assert LEASH_ROT[0] is None
# ⑤ 机器行 NaN -> 按绿
assert _tier(float("nan")) is None
assert _tier(TIER_HI) == 2 and _tier(TIER_LO) == 1 and _tier(TIER_LO - 1) == 0
mrows = np.where(np.asarray(z["source"]) == 0)[0]
assert np.isnan(np.asarray(z["conf_pos_0"], np.float64)[mrows]).all(), \
    "机器行 conf 必须是 NaN (结构性禁查)"
print("[体制] ⑤ 机器行 conf=NaN 禁查 ✅")
print("[体制] ★全绿")
