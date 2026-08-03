"""给 DexMate USD 补碰撞过滤对 (filteredPairs), 让自碰撞能真正开起来.

## 为什么需要

`enabledSelfCollisions=True` 时 PhysX 会检测**所有**连杆两两碰撞, 包括关节直连的父子
连杆 —— 而父子连杆在关节处几何重叠是完全正常的 (拇指掌骨当然嵌在手掌里). 没有过滤对,
PhysX 会拼命把它们推开.

2026-07-28 实测 (diag_selfcol.py 差分):
  开自碰撞后拇指链从根部被顶开, 指尖偏移 3.11cm;
  内力顺运动链传到肩, R_arm_j1 平均力矩 **+70 Nm** (上限 150);
  臂追不上目标 -> 96.5% 回合以 term/stuck 终止, 成功率 0.

  collision_audit.py 网格级测距 (静置位姿) 找到的互穿对:
    right_hand_C_MC ↔ right_thumb_MC     0.084 cm   ← 父子 (拇指 CMC)
    R_arm_l8        ↔ right_hand_C_MC    0.095 cm   ← 父子 (腕)
    torso_l2        ↔ torso_l3           0.098 cm   ← 父子
    right_index_PP  ↔ right_middle_PP    0.072 cm   ← 同层相邻手指, 非父子

## 过滤哪些

1. **父子连杆** —— 从 URDF 的 joint parent/child 生成
2. **祖孙连杆** —— 隔一个关节的也常年重叠 (可选, --grandparent)
3. **静置位姿下距离 < thresh 但非父子的** —— 比如相邻手指的近节, 天然贴着

不过滤的: 真正需要检测的那些 (指尖之间、手指与另一只手、手与桌面).

  $PY tools/add_collision_filters.py --check     # 只列出要加哪些, 不写
  $PY tools/add_collision_filters.py             # 就地写进派生资产
"""
from __future__ import annotations

import argparse
import os
import shutil
import xml.etree.ElementTree as ET
from itertools import combinations

import numpy as np
from pxr import Usd, UsdPhysics

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
USD = os.path.join(_REPO, "assets", "vega_1p_sharpa_fixedtorso.usd")
URDF = os.environ.get(
    "VEGA_URDF",
    "/home/lyh/luhr/MagicSim/Third_Party/curobo/curobo/content/assets/robot/"
    "vega_1p_sharpa/vega_1p_sharpa.urdf")


def urdf_adjacency(urdf_path, grandparent=False):
    """-> (父子对集合, child->parent 映射)."""
    root = ET.parse(urdf_path).getroot()
    parent = {}
    for j in root.findall("joint"):
        p = j.find("parent").get("link")
        c = j.find("child").get("link")
        parent[c] = p
    pairs = {frozenset((c, p)) for c, p in parent.items()}
    if grandparent:
        for c, p in parent.items():
            g = parent.get(p)
            if g:
                pairs.add(frozenset((c, g)))
    return pairs, parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usd", default=USD)
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--state", default=None,
                    help="diag_selfcol 存的 npz; 给了就把静置位姿下贴合的非父子对也过滤")
    ap.add_argument("--thresh", type=float, default=0.003, help="m, 贴合判定阈值")
    ap.add_argument("--grandparent", action="store_true", help="连祖孙对也过滤")
    ap.add_argument("--check", action="store_true", help="只看, 不写")
    a = ap.parse_args()

    adj, _ = urdf_adjacency(a.urdf, a.grandparent)
    print(f"URDF {a.urdf}\n  父子对{'(含祖孙)' if a.grandparent else ''} {len(adj)} 组")

    stage = Usd.Stage.Open(a.usd)
    # 连杆名 -> prim 路径 (只认有刚体的)
    link_path = {pr.GetName(): pr.GetPath() for pr in stage.Traverse()
                 if pr.HasAPI(UsdPhysics.RigidBodyAPI)}
    print(f"USD  {a.usd}\n  刚体连杆 {len(link_path)} 个")

    todo = {p for p in adj if all(n in link_path for n in p)}
    print(f"  其中两端都在 USD 里的父子对: {len(todo)}")

    extra = set()
    if a.state:
        from rl_rebuild.correction.collision_audit import audit
        rows = audit(a.state, step=0, topk=0)
        for dist, na, nb in rows:
            if dist < a.thresh and frozenset((na, nb)) not in adj:
                extra.add(frozenset((na, nb)))
        print(f"  静置位姿下贴合(<{a.thresh*100:.1f}cm)的非父子对: {len(extra)}")
        for pr in sorted(extra, key=lambda s: sorted(s)):
            print(f"      {' ↔ '.join(sorted(pr))}")
    todo |= extra

    # 幂等: 已经有的就不重复加
    existing = set()
    for pr in stage.Traverse():
        if pr.HasAPI(UsdPhysics.FilteredPairsAPI):
            rel = UsdPhysics.FilteredPairsAPI(pr).GetFilteredPairsRel()
            for t in (rel.GetTargets() if rel else []):
                existing.add(frozenset((pr.GetName(), t.name)))
    new = todo - existing
    print(f"\n已存在的过滤对 {len(existing)};  本次要加 {len(new)}")
    if a.check:
        for p in sorted(new, key=lambda s: sorted(s))[:40]:
            print("   +", " ↔ ".join(sorted(p)))
        if len(new) > 40:
            print(f"   ... 其余 {len(new)-40} 组")
        print("\n(--check 模式, 未写入)")
        return

    if not new:
        print("没有要加的, 退出.")
        return
    bak = a.usd + ".bak"
    if not os.path.exists(bak):
        shutil.copy2(a.usd, bak)
        print(f"已备份 -> {bak}")
    for pair in new:
        x, y = sorted(pair)
        prim = stage.GetPrimAtPath(link_path[x])
        api = UsdPhysics.FilteredPairsAPI.Apply(prim)
        api.CreateFilteredPairsRel().AddTarget(link_path[y])
    stage.GetRootLayer().Save()
    print(f"已写入 {len(new)} 组过滤对 -> {a.usd}")


if __name__ == "__main__":
    main()
