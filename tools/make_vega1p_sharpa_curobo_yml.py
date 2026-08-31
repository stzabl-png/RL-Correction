"""从仓库自带 URDF 生成 vega_1p_sharpa 的 cuRobo 机器人配置 (跨机自给自足)。

  ~/miniforge3/envs/isaac/bin/python tools/make_vega1p_sharpa_curobo_yml.py

背景: curobo_plan_worker 原用 MagicSim 树里的 magicsim_vega1p_sharpa.yml
(/home/lyh 机器)。那套 `curobo.motion_planner` API 实为 **NVlabs/curobo 新版
主线** (v0.8+, warp 内核免编译), 2026-08-30 已 pip -e 装进本机 isaac 环境
(~/WorkSpace/curobo)。缺的只是机器人配置, 本脚本生成:

    datasets/vega_urdf/vega_1p_sharpa_curobo.yml   (robot_cfg 包裹, worker 直接吃)

网格缺口的处理 (仓库 URDF 只带了 sharpa 手 STL, vega_1p 机体 obj 没入库):
  - 臂 (L/R_arm_l1..l8) + 头 (head_l1..l3): 从 ~/WorkSpace/vega_curobo/assets
    (github.com/luaiabuelsamen/vega_curobo, Dexmate 官方资产, LICENSE-dexmate)
    按名拷到缺失文件名 —— 两份 URDF 的 link 系与网格原点**逐位相同**
    (L_arm_l2/l4 实测: inertial origin 与 mesh identity origin 完全一致);
  - 底座/雷达/相机/轮/躯干件: 1cm 立方 stub (只为让 parser 起得来), 拟合出的
    球事后删除; 躯干改为**FK 推导的手工球** (肩点+躯干柱, 自碰撞用 —— worker
    运行时会关躯干对世界的碰撞, 这些球只挡"臂扫躯干")。
碰撞球 = MorphIt 自动拟合 (手/臂用真网格)。与 MagicSim 人工调的那份存在差异,
首次规划后若"起点即碰撞/过保守", 用 RobotBuilder.refit_link_spheres 排查。
"""
from __future__ import annotations

import os
import re
import shutil
import sys

import numpy as np
import yaml

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)
URDF = os.path.join(REPO, "datasets", "vega_urdf", "vega_1p_sharpa",
                    "vega_1p_sharpa.urdf")
MESH_ROOT = os.path.dirname(URDF)
VC = os.path.expanduser("~/WorkSpace/vega_curobo/assets/meshes")
OUT = os.path.join(REPO, "datasets", "vega_urdf", "vega_1p_sharpa_curobo.yml")
TOOL_FRAMES = ["right_hand_C_MC", "left_hand_C_MC"]
LOCK_ZERO = ["dummy_base_prismatic_x_joint", "dummy_base_prismatic_y_joint",
             "dummy_base_revolute_z_joint",
             "B_wheel_j1", "B_wheel_j2", "F_wheel_j1", "F_wheel_j2",
             "L_wheel_j1", "L_wheel_j2", "R_wheel_j1", "R_wheel_j2"]
STUB = ("v -0.005 -0.005 -0.005\nv 0.005 -0.005 -0.005\nv 0.005 0.005 -0.005\n"
        "v -0.005 0.005 -0.005\nv -0.005 -0.005 0.005\nv 0.005 -0.005 0.005\n"
        "v 0.005 0.005 0.005\nv -0.005 0.005 0.005\n"
        "f 1 2 3\nf 1 3 4\nf 5 7 6\nf 5 8 7\nf 1 5 6\nf 1 6 2\n"
        "f 2 6 7\nf 2 7 3\nf 3 7 8\nf 3 8 4\nf 4 8 5\nf 4 5 1\n")
# 只有这些 link 的球最终保留 stub 拟合外的来源; stub link 的球一律删
STUB_DROP_PAT = re.compile(
    r"^(vega_1p_(base|back_lidar|front_lidar|B_wheel|mobile)|.*_base_cam"
    r"|[LRBF]_wheel|zed_|dummy_)")


def fill_meshes():
    """把缺失网格补齐: 臂/头 <- vega_curobo 真网格; 其余 <- stub."""
    src = {  # 目标文件名片段 -> vega_curobo 文件
        "vega_1p_L_arm_l1_visuals_L_arm_l1": "L_arm_l1.obj",
        "vega_1p_L_arm_l1_collisions_L_arm_l1": "collision_L_arm_l1.obj",
        "vega_1p_R_arm_l1_visuals_R_arm_l1": "R_arm_l1.obj",
        "vega_1p_R_arm_l1_collisions_R_arm_l1": "collision_R_arm_l1.obj",
        "vega_1p_L_arm_l6_visuals_L_arm_l6": "L_arm_l6.obj",
        "vega_1p_L_arm_l6_collisions_L_arm_l6": "collision_L_arm_l6.obj",
        "vega_1p_R_arm_l6_visuals_R_arm_l6": "R_arm_l6.obj",
        "vega_1p_R_arm_l6_collisions_R_arm_l6": "collision_R_arm_l6.obj",
    }
    for side in ("L", "R"):
        for k in (2, 3, 4, 5, 7, 8):
            src[f"{side}_arm_l{k}_visuals_arm_l{k}"] = f"arm_l{k}.obj"
            src[f"{side}_arm_l{k}_collisions_arm_l{k}"] = f"collision_arm_l{k}.obj"
    for k in (1, 2, 3):   # 头: 第 0 件用真网格, 其余件 stub
        src[f"vega_1p_head_l{k}_visuals_head_l{k}"] = f"head_l{k}.obj"
        src[f"vega_1p_head_l{k}_collisions_head_l{k}"] = f"head_l{k}.obj"
        src[f"head_l{k}_0"] = f"head_l{k}.obj"
    urdf = open(URDF).read()
    need = sorted(set(re.findall(r'filename="([^"]+)"', urdf)))
    n_cp = n_st = 0
    for f in need:
        p = os.path.normpath(os.path.join(MESH_ROOT, f))
        if os.path.isfile(p):
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        base = os.path.basename(p)
        hit = None
        for key, vc_name in src.items():
            if base.startswith(key):
                hit = os.path.join(VC, vc_name)
                break
        if hit and os.path.isfile(hit):
            shutil.copyfile(hit, p)
            n_cp += 1
        else:
            open(p, "w").write(STUB)
            n_st += 1
    print(f"[yml] 网格补齐: 真网格拷贝 {n_cp} | stub {n_st}")


def torso_spheres():
    """FK 推导躯干手工球 (vega_1p_torso_l3 系): 肩连线 + 躯干柱下延。"""
    from rl_rebuild.correction.kinematics import Urdf
    u = Urdf(URDF)
    # 双肩 = L/R_arm_j1 的关节原点; 用 fk_chain 到 torso_l3 与 arm_l1 求相对
    q0 = {}
    T_t3 = u.link_pose("vega_1p_torso_l3", q0)
    sph = []
    for P in ("L", "R"):
        T_s = u.link_pose(f"vega_1p_{P}_arm_l1", q0)
        rel = np.linalg.inv(T_t3) @ T_s
        c = rel[:3, 3]
        sph.append({"center": [float(v) for v in c], "radius": 0.07})
    mid = np.mean([s["center"] for s in sph], axis=0)
    for dz in (0.0, -0.12, -0.24, -0.36):
        sph.append({"center": [float(mid[0]), float(mid[1]),
                               float(mid[2] + dz)], "radius": 0.10})
    return sph


def main():
    fill_meshes()
    from curobo.robot_builder import RobotBuilder

    print(f"[yml] URDF: {URDF}")
    b = RobotBuilder(URDF, MESH_ROOT, tool_frames=TOOL_FRAMES)
    print(f"[yml] links={len(b._link_names)} mesh_links={len(b._mesh_link_names)}")
    print("[yml] MorphIt 碰撞球拟合中 (GPU)...", flush=True)
    b.fit_collision_spheres()
    print("[yml] 自碰撞矩阵...", flush=True)
    b.compute_collision_matrix()
    cfg = b.build()
    b.save(cfg, OUT)
    with open(OUT) as f:
        d = yaml.safe_load(f)
    kin = d["kinematics"] if "kinematics" in d else d
    # stub link 的球删掉; 躯干换 FK 手工球
    cs = kin.get("collision_spheres") or {}
    dropped = [k for k in list(cs) if STUB_DROP_PAT.match(k)]
    for k in dropped:
        del cs[k]
    cs["vega_1p_torso_l3"] = torso_spheres()
    cs.pop("vega_1p_torso_l1", None)
    cs.pop("vega_1p_torso_l2", None)
    kin["collision_spheres"] = cs
    if kin.get("collision_link_names"):
        kin["collision_link_names"] = [n for n in kin["collision_link_names"]
                                       if n in cs]
    # 站姿亚毫米擦碰对 (RobotDebugger 实测: 肘弯 l5-l7 0.19mm + 相邻指节):
    # 拟合球微突出的伪碰撞; 规划时手指锁死, 忽略无害
    ig = {k: list(v) for k, v in (kin.get("self_collision_ignore") or {}).items()}
    _pairs = [("R_arm_l5", "R_arm_l7"), ("L_arm_l5", "L_arm_l7")]
    for _s in ("left", "right"):
        _pairs += [(f"{_s}_index_MP", f"{_s}_middle_PP"),
                   (f"{_s}_middle_PP", f"{_s}_ring_MP"),
                   (f"{_s}_middle_MP", f"{_s}_ring_DP"),
                   (f"{_s}_index_PP", f"{_s}_middle_MP"),
                   (f"{_s}_ring_PP", f"{_s}_pinky_MP"),
                   (f"{_s}_ring_MP", f"{_s}_pinky_PP")]
    for _a, _b in _pairs:
        ig.setdefault(_a, [])
        if _b not in ig[_a]:
            ig[_a].append(_b)
    kin["self_collision_ignore"] = ig
    lj = dict(kin.get("lock_joints") or {})
    cs_joints = set((kin.get("cspace") or {}).get("joint_names") or [])
    for n in LOCK_ZERO:
        if n in cs_joints:          # 猜名不许进 yml (F_wheel 不存在的教训)
            lj[n] = 0.0
    # 入库文件不得固化生成机器的绝对路径；worker 会按仓库根重定位。
    kin["asset_root_path"] = "datasets/vega_urdf/vega_1p_sharpa"
    kin["urdf_path"] = (
        "datasets/vega_urdf/vega_1p_sharpa/vega_1p_sharpa.urdf")
    kin["lock_joints"] = lj
    with open(OUT, "w") as f:
        f.write("# tools/make_vega1p_sharpa_curobo_yml.py 生成 (NVlabs curobo "
                "RobotBuilder / MorphIt)\n# 源: datasets/vega_urdf/vega_1p_sharpa"
                " URDF; 臂/头网格自 vega_curobo (Dexmate 官方资产)\n")
        yaml.safe_dump({"robot_cfg": {"kinematics": kin}}, f,
                       default_flow_style=None, sort_keys=False)
    n_sph = sum(len(v) for v in cs.values())
    print(f"[yml] ✅ {OUT}")
    print(f"[yml] 碰撞球 {n_sph} 颗 / {len(cs)} links | stub 球已删 "
          f"{len(dropped)} links | lock {len(lj)} 关节")
    for k in sorted(cs):
        if "arm" in k or "torso" in k or "hand_C" in k:
            ext = np.asarray([s["center"] for s in cs[k]], float)
            print(f"   {k:26s} {len(cs[k]):3d} 球 | 范围 "
                  f"{np.round(ext.min(0), 2).tolist()}~"
                  f"{np.round(ext.max(0), 2).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
