"""生成"躯干锁死"版 DexMate USD —— 把不参与任务的关节换成 fixed joint.

为什么要这么做:
  躯干在仿真里**撑不住**. 实测: reset 把 torso 硬掰回 [45,90,0], 之后约 60 步内又塌到
  [46.56, 83.83, -0.09] 的平衡点, 手臂基座随之移动 **10.55cm** —— 而这 60 步正好是
  "从站姿接近物体"的那一段, 等于手在往前伸的同时肩膀在往后退.

  这不是物理: torso_j2 以上的重力矩只有 ~30Nm 量级, 而驱动 kp=1e5 Nm/rad, 6.17° 的
  偏差要 10800Nm 才解释得通. 关节限位也设了(PhysX 里确实是 ±0.097°)却没被强制执行.
  求解器迭代 8/32/128 三档结果完全一样, 所以也不是迭代不够.

  躯干在这个任务里**全程不动**, 与其和求解器较劲, 不如直接从自由度里去掉:
  换成 fixed joint 之后它连"能不能撑住"这个问题都不存在了.

做法: **展平**原 USD 到 assets/ (不动 MagicSim 的共享资产), 把指定关节
  PhysicsRevoluteJoint/PhysicsPrismaticJoint -> PhysicsFixedJoint,
  并把**锁定角度烘进 localRot0** —— fixed joint 约束的是 frame0 == frame1,
  而 revolute 在角度 θ 时是 frame0·R(axis,θ) == frame1, 所以
  localRot0' = localRot0 ⊗ R(axis, θ).

  $PY tools/make_fixed_torso_usd.py            # 生成到 assets/
  $PY tools/make_fixed_torso_usd.py --check    # 只打印会改哪些关节, 不写文件
"""
from __future__ import annotations

import argparse
import math
import os

from pxr import Gf, Usd, UsdPhysics

SRC = os.path.join(os.environ.get("MAGICSIM_ASSETS", "/home/lyh/luhr/MagicSim/Assets"),
                   "Robots", "vega_1p_sharpa.usd")
DST = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets",
                                   "vega_1p_sharpa_fixedtorso.usd"))
# 关节名 -> 锁定值 (revolute 用度, prismatic 用米). 必须与 cfg.dexmate_joints 的躯干项一致.
LOCK = {
    "torso_j1": 40.5196, "torso_j2": 73.6595, "torso_j3": 0.3896,
    "dummy_base_prismatic_x_joint": 0.0, "dummy_base_prismatic_y_joint": 0.0,
    "dummy_base_revolute_z_joint": 0.0,
    "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0,
}
AXIS_VEC = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}


def _axis_quat(axis: str, deg: float) -> Gf.Quatf:
    x, y, z = AXIS_VEC[axis]
    h = math.radians(deg) * 0.5
    s = math.sin(h)
    return Gf.Quatf(math.cos(h), Gf.Vec3f(x * s, y * s, z * s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--check", action="store_true", help="只看不写")
    ap.add_argument("--selfcol", action="store_true",
                    help="打开整机自碰撞 + 同侧过滤. ⚠ 目前**会打坏抓取** "
                         "(Grasp2 零残差 100%%->0%%, 指尖接触 1.46->0.02), 默认关闭. "
                         "过滤对写了 3280 条也没救回来, 原因未定位.")
    a = ap.parse_args()

    if not a.check:
        os.makedirs(os.path.dirname(a.dst), exist_ok=True)
        # ⚠ **必须 Flatten, 不能 shutil.copy** (2026-08-03 修).
        #   MagicSim 的 DexMate 资产有两种布局:
        #     ① 已展平的单文件 (本机就是这种, 160 个 reference 全是层内的)
        #     ② 模块化: 顶层 vega_1p_sharpa.usd 引用同级
        #        configuration/vega_1p_sharpa_{base,physics,robot,sensor}.usd
        #   单文件拷贝把 USD 从它的引用树里拽出来 —— 对 ① 无害, 对 ② 会让所有
        #   visuals/collisions 引用悬空. Isaac 不报错致命, 只是每个 link 退回默认
        #   变换 ⟹ **机器人散架、双手往外摊开**. 同事直接 clone 后就是这个症状.
        #   Flatten() 把组合结果烘进单层, 两种布局产出一致, 跨机器可搬.
        Usd.Stage.Open(a.src).Flatten().Export(a.dst)
        path = a.dst
    else:
        path = a.src

    stage = Usd.Stage.Open(path)
    todo = [p for p in stage.Traverse() if p.GetName() in LOCK
            and "Joint" in p.GetTypeName()]
    print(f"源 {a.src}\n目标 {a.dst if not a.check else '(--check, 不写)'}")
    print(f"找到 {len(todo)}/{len(LOCK)} 个待锁关节\n")

    changed = []
    for prim in todo:
        name = prim.GetName()
        val = LOCK[name]
        ty = prim.GetTypeName()
        j = UsdPhysics.Joint(prim)
        b0 = j.GetBody0Rel().GetTargets()
        b1 = j.GetBody1Rel().GetTargets()
        lp0 = j.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0)
        lr0 = j.GetLocalRot0Attr().Get() or Gf.Quatf(1, 0, 0, 0)
        lp1 = j.GetLocalPos1Attr().Get() or Gf.Vec3f(0, 0, 0)
        lr1 = j.GetLocalRot1Attr().Get() or Gf.Quatf(1, 0, 0, 0)
        axis = "X"
        if ty == "PhysicsRevoluteJoint":
            axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get() or "X"
            lr0 = lr0 * _axis_quat(axis, val)          # 把角度烘进 frame0
        elif ty == "PhysicsPrismaticJoint":
            axis = UsdPhysics.PrismaticJoint(prim).GetAxisAttr().Get() or "X"
            v = AXIS_VEC[axis]
            # 平移量在 frame0 的**局部**方向上, 要转到 body0 系再叠到 localPos0
            d = Gf.Quatf(lr0).Transform(Gf.Vec3f(v[0] * val, v[1] * val, v[2] * val)) \
                if hasattr(Gf.Quatf(lr0), "Transform") else Gf.Vec3f(*[c * val for c in v])
            lp0 = Gf.Vec3f(lp0[0] + d[0], lp0[1] + d[1], lp0[2] + d[2])
        changed.append((name, ty, axis, val))
        if a.check:
            continue
        p = prim.GetPath()
        stage.RemovePrim(p)                            # 换类型只能重建 prim
        fj = UsdPhysics.FixedJoint.Define(stage, p)
        fj.CreateBody0Rel().SetTargets(b0)
        fj.CreateBody1Rel().SetTargets(b1)
        fj.CreateLocalPos0Attr().Set(lp0)
        fj.CreateLocalRot0Attr().Set(lr0)
        fj.CreateLocalPos1Attr().Set(lp1)
        fj.CreateLocalRot1Attr().Set(lr1)

    for n, ty, ax, v in changed:
        unit = "m" if "prismatic" in ty.lower() else "°"
        print(f"  {n:<32} {ty:<24} 轴{ax}  锁在 {v}{unit}")

    # ---- 自碰撞: 打开, 但把"同一只手内部"过滤掉 ----
    # 原 USD 是 enabledSelfCollisions=False, 结果**左右手会互相穿模**.
    # 但这是个**整机开关**: 直接打开的话同一只手的手指之间也开始互撞, 合拢被挡住
    # —— 实测 Grasp2 零残差成功率从 100% 掉到 0%, 指尖接触 1.46 -> 0.03.
    # 所以打开之后必须用 FilteredPairsAPI 把同手内部的对排除, 只保留
    # 左手↔右手 和 手↔手臂 —— 那才是会穿模的地方.
    if not a.check and a.selfcol:
        for prim in stage.Traverse():
            for at in prim.GetAttributes():
                if at.GetName() == "physxArticulation:enabledSelfCollisions":
                    at.Set(True)
                    print(f"\n自碰撞已打开: {prim.GetPath()}")
        # 按**身体一侧**分组: 同侧的手 + 手臂全部互相过滤.
        # 只过滤同手内部是不够的 —— 手臂补上碰撞几何之后, 手的根部和自己那条臂的
        # l7/l8 凸包在静止时就重叠, 一开自碰撞就把手臂顶住 (实测仍是 0%).
        # 保留的是: 左侧 ↔ 右侧, 也就是"左右手/左右臂互相穿模"这个真问题.
        SIDE = {"left": ("left_", "L_arm_", "vega_1p_L_arm_"),
                "right": ("right_", "R_arm_", "vega_1p_R_arm_")}
        groups = {"left": [], "right": []}
        for prim in stage.Traverse():
            n = prim.GetName()
            if prim.GetParent() and prim.GetParent().GetName() == "vega_1p_sharpa":
                for side, pfx in SIDE.items():
                    if n.startswith(pfx):
                        groups[side].append(prim)
        n_pair = 0
        for side, links in groups.items():
            for i, p0 in enumerate(links):
                others = [l.GetPath() for j, l in enumerate(links) if j != i]
                if not others:
                    continue
                UsdPhysics.FilteredPairsAPI.Apply(p0).CreateFilteredPairsRel(
                    ).SetTargets(others)
                n_pair += len(others)
            print(f"  {side} 侧 {len(links)} 个 link (手+臂), 过滤同侧内部对 "
                  f"{len(links)*(len(links)-1)} 条")
        print(f"共写入 {n_pair} 条过滤对 -> 只剩**左侧↔右侧**的自碰撞")

    if not a.check:
        stage.GetRootLayer().Save()
        print(f"\n已写出 {a.dst}  ({os.path.getsize(a.dst)/1e6:.1f} MB)")
        print("验收: 建 env 后 arm_center 应恒为 [-0.4588, 0, 1.2996] 且全程不动")


if __name__ == "__main__":
    main()
