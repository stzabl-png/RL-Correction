"""把手臂连杆的碰撞几何补进 DexMate USD.

为什么需要:
  实测这个 USD 里 **52 个碰撞体全部在手上**, 手臂/躯干/底盘的 `collisions` Xform 是**空的**.
  后果是手臂可以**穿过桌子、穿过自己**. 对 RL 来说这是致命的 —— 策略会找到"把手臂从桌面
  穿过去"的解并被判成功, 训出来的轨迹在真机上直接撞桌.

  URDF 里 **78/110 个 link 有 collision 网格**(手臂躯干全都有), 是 URDF→USD 转换时丢的.
  这个工具把它们读回来补上.

只补**会动的手臂连杆**:
  躯干/底盘已经在 make_fixed_torso_usd.py 里变成 fixed joint, 它们与桌沿有重叠,
  加了碰撞只会产生一堆永久接触(既不影响运动, 又白费算力).

  $PY tools/add_arm_collision.py --check     # 只看要加哪些, 不写
  $PY tools/add_arm_collision.py             # 就地补进 assets/ 里那份派生资产
"""
from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics, Vt

URDF = os.environ.get(
    "VEGA_URDF",
    "/home/lyh/luhr/MagicSim/Third_Party/curobo/curobo/content/assets/robot/"
    "vega_1p_sharpa/vega_1p_sharpa.urdf")
USD = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets",
                                   "vega_1p_sharpa_fixedtorso.usd"))
# 只处理手臂 (会动的). 躯干/底盘/头已固定, 见模块 docstring.
PREFIX = ("R_arm_l", "L_arm_l", "vega_1p_R_arm_l", "vega_1p_L_arm_l")


def _rpy_quat(r, p, y):
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    return Gf.Quatd(cr * cp * cy + sr * sp * sy,
                    Gf.Vec3d(sr * cp * cy - cr * sp * sy,
                             cr * sp * cy + sr * cp * sy,
                             cr * cp * sy - sr * sp * cy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--usd", default=USD)
    ap.add_argument("--approx", default="convexHull",
                    choices=("convexHull", "convexDecomposition", "boundingCube"),
                    help="碰撞近似. 手臂连杆是长条形, convexHull 足够且最省")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    import trimesh
    root = ET.parse(a.urdf).getroot()
    base = os.path.dirname(os.path.abspath(a.urdf))
    stage = Usd.Stage.Open(a.usd)

    jobs = []
    for link in root.findall("link"):
        name = link.get("name")
        if not name.startswith(PREFIX):
            continue
        for i, col in enumerate(link.findall("collision")):
            g = col.find("geometry")
            m = g.find("mesh") if g is not None else None
            if m is None:
                continue
            # vega_1p_sharpa 那份 URDF 引用的 meshes/vega_1p/ 是空的, 网格实际在
            # 隔壁 robot/vega_1p/meshes/ 下 (只有本体资产带, sharpa 版没随附). 依次找.
            rel = m.get("filename")
            cands = [os.path.normpath(os.path.join(base, rel)),
                     os.path.normpath(os.path.join(
                         base, "..", "vega_1p", "meshes", os.path.basename(rel)))]
            f = next((c for c in cands if os.path.exists(c)), cands[0])
            o = col.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) \
                if o is not None else np.zeros(3)
            rpy = np.array([float(v) for v in (o.get("rpy") or "0 0 0").split()]) \
                if o is not None else np.zeros(3)
            jobs.append((name, i, f, xyz, rpy))

    # ⚠ 手臂 link 在这个 USD 里是 **instance proxy**(左右臂共用原型), 不能直接往里写
    # ("authoring to an instance proxy is not allowed"). 先对相关 link 取消实例化.
    if not a.check:
        n_un = 0
        for prim in stage.Traverse():
            if prim.IsInstanceable():
                prim.SetInstanceable(False)
                n_un += 1
        if n_un:
            print(f"已取消 {n_un} 个 prim 的实例化 (否则无法写入碰撞几何)")

    print(f"URDF {a.urdf}\nUSD  {a.usd}\n找到 {len(jobs)} 个手臂碰撞网格\n")
    n_ok = 0
    for name, i, f, xyz, rpy in jobs:
        prim = stage.GetPrimAtPath(f"/vega_1p_sharpa/{name}")
        if not prim.IsValid():
            print(f"  ⚠ USD 里没有 link {name}, 跳过")
            continue
        if not os.path.exists(f):
            print(f"  ⚠ 网格不存在 {f}, 跳过")
            continue
        mesh = trimesh.load(f, force="mesh", process=False)
        v, fc = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        print(f"  {name:<22} {os.path.basename(f):<48} 顶点{len(v):>6} 面{len(fc):>6}"
              f"  origin={np.round(xyz,3)}")
        n_ok += 1
        if a.check:
            continue
        path = f"/vega_1p_sharpa/{name}/collisions/col_{i}"
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp().Set(Gf.Vec3d(*xyz))
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_rpy_quat(*rpy))
        gm = UsdGeom.Mesh.Define(stage, path + "/mesh")
        gm.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*p) for p in v]))
        gm.CreateFaceVertexIndicesAttr().Set(Vt.IntArray(fc.reshape(-1).tolist()))
        gm.CreateFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(fc)))
        gm.CreatePurposeAttr().Set(UsdGeom.Tokens.guide)      # 只做碰撞, 不渲染
        UsdPhysics.CollisionAPI.Apply(gm.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(gm.GetPrim()).CreateApproximationAttr(a.approx)

    if not a.check:
        stage.GetRootLayer().Save()
        n = sum(1 for p in stage.Traverse()
                if any("PhysicsCollisionAPI" in s for s in p.GetAppliedSchemas()))
        print(f"\n已补 {n_ok} 个手臂碰撞体 (近似={a.approx}); USD 碰撞体总数 {n}")
        print("验收: 手臂不该再穿过桌子; 跑 eval 看零残差成功率有没有掉")


if __name__ == "__main__":
    main()
