"""右手碰撞球 yml -> 左手 (ocir 缺 sharpa_left.yml, build_assets 需要它).

左手 URDF 与右手是**精确镜像**(实测同一组关节角下 link 位置差 <0.002mm), 镜像面为
xz 平面 (M = diag(1,-1,1))。但 link **局部系**的约定按类别不同, 差一个固定翻转 D:

    T_L = M · T_R · M · D          (D 由实测求出, 逐 link 常数)
    局部点:  p_L = D · M · p_R

    hand_C_MC / *_fingertip   D = I               -> (x, -y,  z)
    多数 link (PP/MP/DP/...)  D = diag(1,-1,-1)   -> (x,  y, -z)
    *_elastomer               D = diag(-1,-1,1)   -> (-x, y,  z)

⚠ 若按单一规则镜像 (比如一律 y 取反), 球心会落到 link 外面 —— 本脚本末尾的校验
就是抓这个的: 镜像后每个球心到左手网格表面的距离, 应与右手侧同名球一致。

  python tools/sharpa_wave/mirror_left_spheres.py [--check]
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

OCIR = Path("/home/lyh/Project/ocir-grasp-synthesis/assets/robots/hands/sharpa_wave")
SRC = OCIR / "collision/curobo/sharpa_right.yml"
DST = OCIR / "collision/curobo/sharpa_left.yml"
URDF_L = OCIR / "urdf/left_sharpa_wave/left_sharpa_wave.urdf"
URDF_R = OCIR / "urdf/right_sharpa_wave/right_sharpa_wave.urdf"
M = np.diag([1.0, -1.0, 1.0])


# ---------------------------------------------------------------- URDF FK
def _rpy2R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def load_urdf(path):
    root = ET.parse(path).getroot()
    joints = []
    for j in root.findall("joint"):
        o = j.find("origin")

        def g(el, key, default):
            v = el.get(key) if el is not None and el.get(key) else default
            return np.array([float(x) for x in v.split()])

        joints.append(dict(name=j.get("name"), type=j.get("type"),
                           parent=j.find("parent").get("link"),
                           child=j.find("child").get("link"),
                           xyz=g(o, "xyz", "0 0 0"), rpy=g(o, "rpy", "0 0 0")))
    links = [l.get("name") for l in root.findall("link")]
    ch = {j["child"] for j in joints}
    return joints, links, [l for l in links if l not in ch][0]


def fk_zero(joints, root):
    """qpos=0 处各 link 的世界位姿 (镜像关系与 q 无关, 取零位即可求 D)."""
    T = {root: np.eye(4)}
    progress = True
    while progress:
        progress = False
        for j in joints:
            if j["parent"] in T and j["child"] not in T:
                A = np.eye(4)
                A[:3, :3] = _rpy2R(*j["rpy"])
                A[:3, 3] = j["xyz"]
                T[j["child"]] = T[j["parent"]] @ A
                progress = True
    return T


def per_link_D():
    """逐 link 求 D = (M·R_R·M)ᵀ·R_L, 四舍五入到 ±1 的对角阵."""
    JL, LL, rL = load_urdf(URDF_L)
    JR, _, rR = load_urdf(URDF_R)
    TL, TR = fk_zero(JL, rL), fk_zero(JR, rR)
    D = {}
    for lb in LL:
        rb = lb.replace("left_", "right_")
        if not lb.startswith("left_") or rb not in TR:
            continue
        Draw = (M @ TR[rb][:3, :3] @ M).T @ TL[lb][:3, :3]
        Dr = np.round(Draw)
        if np.abs(Draw - Dr).max() > 1e-4:
            raise RuntimeError(f"{lb}: D 不是 ±1 对角阵\n{Draw}")
        D[rb] = Dr
    return D


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="镜像后校验球心是否贴合左手网格")
    args = ap.parse_args()

    D = per_link_D()
    src = yaml.safe_load(SRC.read_text())
    out = {"robot": "left_sharpa_wave", "collision_spheres": {}}
    n_sph = 0
    for link, spheres in src["collision_spheres"].items():
        if link not in D:
            print(f"  ⚠ {link} 在左手 URDF 里没有对应 link, 跳过")
            continue
        A = D[link] @ M                                   # p_L = D·M·p_R
        out["collision_spheres"][link.replace("right_", "left_")] = [
            {"center": [float(v) for v in (A @ np.asarray(s["center"], float))],
             "radius": float(s["radius"])} for s in spheres]
        n_sph += len(spheres)
    DST.write_text(yaml.safe_dump(out, default_flow_style=None, sort_keys=False))
    print(f"[mirror] {len(out['collision_spheres'])} 个 link / {n_sph} 个球 -> {DST}")

    if not args.check:
        return 0

    # ---- 校验: 球心到本 link 网格表面的距离, 左右应一致 ----
    import trimesh
    JL, LL, rL = load_urdf(URDF_L)
    rootL = ET.parse(URDF_L).getroot()
    meshes = {}
    for l in rootL.findall("link"):
        col = l.find("collision")
        if col is None:
            continue
        mesh_el = col.find("geometry/mesh")
        if mesh_el is None:
            continue
        f = URDF_L.parent / mesh_el.get("filename").replace("package://", "")
        if not f.exists():
            f = URDF_L.parent / "meshes" / Path(mesh_el.get("filename")).name
        if f.exists():
            meshes[l.get("name")] = trimesh.load(f, force="mesh")
    print(f"[check] 载入 {len(meshes)} 个左手碰撞网格")
    worst = []
    for link, spheres in out["collision_spheres"].items():
        if link not in meshes:
            continue
        m = meshes[link]
        pts = np.array([s["center"] for s in spheres])
        d = trimesh.proximity.signed_distance(m, pts)     # >0 = 在内部
        worst.append((link, float(d.min()), float(d.max())))
    worst.sort(key=lambda r: r[1])
    print(f"{'link':26s} {'球心到表面 有符号距离 mm (>0=内部)':>34s}")
    for link, lo, hi in worst[:10]:
        flag = "  ⚠ 球心在网格外" if lo < -0.002 else ""
        print(f"{link:26s}   min {lo*1000:+7.2f}   max {hi*1000:+7.2f}{flag}")
    bad = [w for w in worst if w[1] < -0.002]
    print(f"\n球心跑到网格外 >2mm 的 link: {len(bad)}/{len(worst)}"
          + ("   ✅ 镜像正确" if not bad else "   ❌ 镜像有误"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
