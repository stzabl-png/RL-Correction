#!/usr/bin/env python
"""USD 依赖分析 + flatten(自包含化)。用 flatten_usd.sh 调起(它负责 pxr 的路径引导)。

对每个输入 USD:
  1) 用 UsdUtils.ComputeAllDependencies 列出全部外部依赖(sublayer/reference/payload/贴图);
  2) 无依赖 -> 报告"已自包含", 不产出新文件;
     有依赖 -> flatten 成 <name>.flat.usd(单文件, 无外部引用);
  3) 对 flatten 前后各统计一遍结构指纹(prim 数 / 关节数 / 带质量的 prim / 带碰撞的 prim),
     打印对照 —— **注意: 这只是结构自检, 不能替代物理一致性验证**。
     物理一致性必须跑零动作回放对拍(见 README)。
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from pxr import Sdf, Usd, UsdPhysics, UsdUtils


def fingerprint(stage: Usd.Stage) -> dict:
    n_prim = n_joint = n_mass = n_coll = 0
    for pr in stage.Traverse():
        n_prim += 1
        if pr.HasAPI(UsdPhysics.MassAPI):
            n_mass += 1
        if pr.HasAPI(UsdPhysics.CollisionAPI):
            n_coll += 1
        if pr.IsA(UsdPhysics.Joint):
            n_joint += 1
    return {"prims": n_prim, "joints": n_joint, "mass_api": n_mass, "collision_api": n_coll}


def deps_of(path: Path):
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    ext = [Path(l.identifier) for l in layers if Path(l.identifier).resolve() != path.resolve()]
    return ext, [Path(a) for a in assets], list(unresolved)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("usd", nargs="+", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args()
    rc = 0
    for u in a.usd:
        u = u.resolve()
        print(f"\n=== {u.name} ===")
        if not u.is_file():
            print("  !! 文件不存在"); rc = 1; continue
        ext, assets, unres = deps_of(u)
        print(f"  外部 layer 依赖 : {len(ext)}")
        for x in ext[:10]:
            print(f"      {x}")
        print(f"  资产引用(贴图等): {len(assets)}")
        for x in assets[:10]:
            print(f"      {x}")
        if unres:
            print(f"  ⚠ 未解析引用   : {len(unres)}")
            for x in unres[:10]:
                print(f"      {x}")
            rc = 1

        st = Usd.Stage.Open(str(u))
        fp0 = fingerprint(st)
        print(f"  结构指纹(原始)  : {fp0}")

        if not ext and not assets:
            print("  ✅ 已自包含 —— 无需 flatten")
            continue

        out = (a.out_dir or u.parent) / (u.stem + ".flat.usd")
        flat = st.Flatten()
        flat.Export(str(out))
        st2 = Usd.Stage.Open(str(out))
        fp1 = fingerprint(st2)
        same = fp0 == fp1
        h = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"  -> {out.name}  ({out.stat().st_size/1e6:.2f} MB)")
        print(f"  结构指纹(flat)  : {fp1}  {'✅ 一致' if same else '❌ 不一致'}")
        print(f"  sha256          : {h}")
        if not same:
            rc = 1
        e2, a2, u2 = deps_of(out)
        print(f"  flat 后残留依赖 : layer {len(e2)} / asset {len(a2)} / 未解析 {len(u2)}")
        if e2 or a2:
            print("  ⚠ flatten 后仍有外部引用"); rc = 1
    print("\n⚠ 结构指纹只是自检, **物理一致性必须跑零动作回放对拍**"
          "(四个出生点的累计奖励与终止步逐位一致, 口径见 smoke_zero.py)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
