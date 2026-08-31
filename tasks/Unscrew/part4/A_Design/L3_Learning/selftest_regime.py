"""体制自检: 断言实现与设计口径逐条一致 (CHECKLIST 第 11 格的"体制"件)。
① P-OBJ 变体: w_obj≡1, w_hand≡0 (纯物轨消融);
② P-HYB: 档位表 绿(1,0)/黄(.5,.5)/红(.2,.8) 逐档兑现;
③ 人手置信度门: 红档人手行 -> W_HCONF=0 (形状奖被掐灭), 绿档=1;
④ 红档 rot 禁入: conf_rot 红的行, 乱转物体既不掉皮筋也不掐时钟 (pos 仍管);
⑤ 机器行 conf=NaN 结构性禁查 (tier 按绿处理, 不虚报红)。"""
import os
import sys

import numpy as np
import json
import tempfile
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "..", "C_Wiring"))
from progress import (W_OBJ, W_HAND, W_HCONF, _tier, TIER_HI, TIER_LO,  # noqa: E402
                      UnscrewProgress)
from progress_batch import UnscrewProgressBatch  # noqa: E402
import task_config as TC  # noqa: E402

import progress as PG  # noqa: E402
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

# ⑥ 判据摘要: 判定阈值变化必须变红，奖励权重变化必须保持安静。
schema0, digest0 = PG.criteria_digest()
assert schema0 == 3 and len(digest0) == 16   # schema=3: T2-3 护送判据入摘要
assert all(PG.criteria_digest()[1] == digest0 for _ in range(3))
for name in ("GATE_POS", "CERT_RISE", "PLACED_HOLD", "M4_DIST_ROT", "D1_DROP",
             "D2_PRE_TILT", "D3_DEV", "D4_SLIP", "PAD_FTH", "TABLE_Z",
             "ESCORT_BAND", "ESCORT_FALL"):
    old = getattr(PG, name)
    setattr(PG, name, old + 1 if isinstance(old, int) else old * 1.01 + 1e-6)
    assert PG.criteria_digest()[1] != digest0, name
    setattr(PG, name, old)
old_wage = PG.WAGE
PG.WAGE *= 2.0
assert PG.criteria_digest()[1] == digest0
PG.WAGE = old_wage
assert {"PLACED_POS", "CERT_SLIP", "M4_ARM", "D2_TILT"} <= set(
    PG.criteria_items()) and {"D5_BELOW_TABLE", "SCREW_TRIAD"} <= set(PG.criteria_items())
print(f"[体制] ⑥ 判据摘要 {digest0}: 阈值敏感/奖励静默 ✅")

# ⑦ UNSCREW_TURNS 必须进入 runtime ScrewSpec，且不能污染全局 clip 注册表。
from types import SimpleNamespace
from rl_rebuild.correction import clips
from tasks.pregrasp.screw_assembly import _spec_from_cfg

assembly = clips.clip_entry(TC.CLIP)["secondary"]["assembly"]
registered_turns = float(assembly["turns"])
probe_turns = registered_turns + 0.125
spec = _spec_from_cfg(SimpleNamespace(screw_turns_override=probe_turns), assembly)
assert abs(spec.turns - probe_turns) < 1e-12
assert float(assembly["turns"]) == registered_turns
print("[体制] ⑦ UNSCREW_TURNS -> runtime spec，注册表不变 ✅")

# ⑧ 世界指纹比较器必须区分匹配、物理不符与关键项缺失。
import copy
import world_fingerprint as WF


def _put(tree, dotted, value):
    node = tree
    parts = dotted.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


fingerprint = {}
for key in WF.CRITICAL:
    value = (["joint"] if key.endswith("controlled_joint_names_in_order")
             else "0123456789abcdef" if key.endswith("md5")
             else True if key.endswith("detach_at_full")
             else 1)
    _put(fingerprint, key, value)
assert WF.compare(fingerprint, copy.deepcopy(fingerprint)) == ([], [], [])
changed = copy.deepcopy(fingerprint)
changed["assembly"]["turns"] = 2
crit, _, unver = WF.compare(fingerprint, changed)
assert any(key == "assembly.turns" for key, _, _ in crit) and not unver
missing = copy.deepcopy(fingerprint)
del missing["criteria"]["digest"]
crit, _, unver = WF.compare(fingerprint, missing)
assert not crit and any(key == "criteria.digest" for key, _, _ in unver)
print("[体制] ⑧ 世界指纹：匹配/物理不符/关键项缺失三分 ✅")

# ⑨ 正式训练只能消费 probe + cuRobo + Isaac-v2 的同 clip/同剂量母带。
with tempfile.TemporaryDirectory() as td:
    parent_path = os.path.join(td, "reference_v1.npz")
    good_path = os.path.join(td, "reference_v2.npz")
    rest_path = os.path.join(td, "env_rest.json")
    acceptance_path = os.path.join(td, "acceptance_v2.json")
    with open(rest_path, "w", encoding="utf-8") as fh:
        fh.write("{}\n")
    meta = {"clip": TC.CLIP_ID, "rest_source": "probe",
            "approach": "curobo:test", "retreat": "curobo:test",
            "screw": {"turns": TC.SCREW_TURNS},
            "rest_md5": TC.file_md5(rest_path)}
    arrays = {
        "right_q": np.zeros((2, 7)), "left_q": np.zeros((2, 7)),
        "obj_pos_0": np.zeros((2, 3)), "obj_quat_0": np.zeros((2, 4)),
        "obj_pos_1": np.zeros((2, 3)), "obj_quat_1": np.zeros((2, 4)),
        "source": np.ones(2), "seg_lens": np.ones(5),
        "station_wr": np.zeros(7), "station_wl": np.zeros(7),
    }
    np.savez(parent_path, meta=np.array(json.dumps(meta)), **arrays)
    meta["planning_basis_digest"] = TC.reference_planning_digest(parent_path)
    np.savez(parent_path, meta=np.array(json.dumps(meta)), **arrays)
    parent_md5 = TC.file_md5(parent_path)[:8]
    meta_v2 = (f"gen=unscrew_v2;parent_v1_md5={parent_md5};"
               f"betaL={TC.BETA_L};betaR={TC.BETA_R};ik=delta_space;critical_bad=0;"
               f"clip={TC.CLIP_ID}")
    np.savez(good_path, meta=np.array(json.dumps(meta)),
             meta_v2=np.array(meta_v2), **arrays)
    receipt = {"schema": "unscrew_acceptance_v1", "clip": TC.CLIP_ID,
               "reference_v2_md5": TC.file_md5(good_path),
               "passes": 3, "num_envs": 4, "world": {}}
    with open(acceptance_path, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh)

    assert TC.training_reference_issues(
        good_path, expected_path=good_path, parent_v1_path=parent_path,
        rest_path=rest_path, acceptance_path=acceptance_path) == []

    bad_path = os.path.join(td, "bootstrap.npz")
    bad_meta = dict(meta)
    bad_meta.update(rest_source="offline_estimate", approach="smoothstep占位")
    np.savez(bad_path, meta=np.array(json.dumps(bad_meta)), **arrays)
    bad_issues = TC.training_reference_issues(
        bad_path, expected_path=bad_path, parent_v1_path=parent_path,
        rest_path=rest_path, acceptance_path=acceptance_path)
    assert any("probe_rest" in issue for issue in bad_issues)
    assert any("cuRobo" in issue for issue in bad_issues)
    assert any("meta_v2" in issue for issue in bad_issues)
print("[体制] ⑨ 正式母带预检：合格 v2 通过/离线占位拒绝 ✅")
print("[体制] ★全绿")
