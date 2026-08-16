"""在 MuJoCo GUI 里看**物体的摆放**: 规范系姿态 + 虚拟桌面 + 前/上轴。

摆放约定(2026-08-15 定):
  * 姿态 = 重建里**前 10 帧位姿的中位数**, 再吸附到最近的稳定静置姿态
    (`upright_from_recon.py`; 抓握窗里物体常已被拿起, 不能用)
  * 竖轴: up -> +Z
  * 方位: front -> **-X**。front 来自重建世界系的 +X("相机前向"), 与 RL 场景一致 ——
    机器人底座 (-0.5,0,0)、ZED 相机 (-0.41,0,1.40) 都在 -X 侧, 双手伸向 +X。
  * 桌面: 规范系 z = 物体最低点。Dexonomy 导入时就是这么放虚拟平面的。

画面里的虚拟桌面是 **2m x 2m 半透明面片, 没有物理碰撞**, 只为看清物体是不是稳稳
立在桌上。真正参与碰撞的是 scene_cfg 里的 `table` 平面(无限大), 两者位置一致。

用法(MUJOCO_GL 不能是 egl):
  env -u MUJOCO_GL DISPLAY=:1 python tools/show_placement_gui.py --oid <oid> [--hand <xml>]
"""

import argparse
import glob
import os
import sys

import mujoco
import mujoco.viewer
import numpy as np

DEXO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DEXO)
os.chdir(DEXO)
from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv  # noqa: E402


def add_line(scn, p0, p1, rgba, width=0.003):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(p0, float), np.asarray(p1, float))
    scn.ngeom += 1


def add_sphere(scn, p, r, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0]),
                        np.asarray(p, float), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def add_box(scn, center, half, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.asarray(half, float),
                        np.asarray(center, float), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    scn.ngeom += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oid", required=True)
    ap.add_argument("--hand", default="assets/hand/sharpa_wave_v2_left/left.xml")
    ap.add_argument("--scene", default="tabletop", choices=("tabletop", "floating"))
    ap.add_argument("--table-size", type=float, default=2.0, help="虚拟桌面边长(m)")
    ap.add_argument("--alpha", type=float, default=0.25)
    ap.add_argument("--recon", default=None, help="给了就把视频提取的接触点画到物体上")
    ap.add_argument("--object", default="object_0")
    ap.add_argument("--side", default="left")
    ap.add_argument("--hot", type=float, default=0.5, help="热点权重门槛(>=画红, 以下画暗)")
    ap.add_argument("--max-pts", type=int, default=1500)
    ap.add_argument("--dome", type=float, default=0.15,
                    help="半透明半球的半径(m), 0=不画。表示撒点/接近方向的覆盖范围。")
    ap.add_argument("--dome-center", default="contact", choices=("contact", "object"),
                    help="半球球心: contact=接触点质心(默认), object=物体质心")
    ap.add_argument("--show-sampling", action="store_true", default=True,
                    help="画出**允许撒点的区域**(约束 A): 表面采样点里距接触热点 <= radius 的。"
                         "绿色=通过(会进入优化), 灰色=被前门筛掉。radius/min_weight 直接读"
                         "processed_data/<oid>/region.npz, 与 sample_init_pose 用的是同一份。")
    ap.add_argument("--no-show-sampling", dest="show_sampling", action="store_false")
    ap.add_argument("--n-surf", type=int, default=2500, help="画多少个表面采样点")
    a = ap.parse_args()

    sp = sorted(glob.glob(f"assets/object/custom/scene_cfg/{a.oid}/{a.scene}/*.npy"))
    if not sp:
        raise SystemExit(f"找不到 scene_cfg: {a.oid}/{a.scene}")
    sc = np.load(sp[0], allow_pickle=True).item()
    base = os.path.dirname(sp[0])
    for _v in sc["scene"].values():
        for _k in ("file_path", "xml_path", "urdf_path", "info_path"):
            if _k in _v:
                _v[_k] = os.path.normpath(os.path.join(base, _v[_k]))

    import trimesh
    mesh = trimesh.load(f"assets/object/custom/processed_data/{a.oid}/mesh/simplified.obj",
                        force="mesh", process=False)
    V = np.asarray(mesh.vertices)
    z_table = float(V[:, 2].min())
    ext = np.ptp(V, axis=0)
    print(f"\n=== {a.oid} 摆放 ===")
    print(f"  规范系尺寸  x {ext[0]*100:.1f}  y {ext[1]*100:.1f}  z {ext[2]*100:.1f} cm"
          f"   -> {'直立' if ext[2] > max(ext[0], ext[1]) else '★侧躺'}")
    print(f"  物体最低点 z = {z_table*100:.1f} cm  (虚拟桌面放在这个高度)")
    print(f"  -X 方向 = 朝向人体/机器人 (RL 里机器人底座在 -0.5,0,0)")
    print(f"\n  灰色半透明面片 = 2m x 2m 虚拟桌面, **无物理碰撞**, 仅供观察")
    print(f"  坐标轴: X 红 / Y 绿 / Z 蓝  (-X 侧 = 人体/机器人所在方向)")

    # 视频提取的接触点 -> 规范系
    # probe_local 存在**输入网格系**里, 规范系的变换记在 info/simplified.json:
    #     p_canon = R_c2i^T @ (p_input - com)
    hot_pts = warm_pts = None
    if a.recon:
        import json
        z = np.load(f"{a.recon}/contact/contact_v2_{a.object}_{a.side}.npz", allow_pickle=True)
        P = np.asarray(z["probe_local"], float)
        W = np.asarray(z["weight"], float)
        info = json.load(open(f"assets/object/custom/processed_data/{a.oid}/info/simplified.json"))
        com = np.asarray(info["com_offset"], float)
        qw = np.asarray(info["canonical_from_input_rot_wxyz"], float)
        w_, x_, y_, z_ = qw
        R_c2i = np.array([[1-2*(y_*y_+z_*z_), 2*(x_*y_-w_*z_), 2*(x_*z_+w_*y_)],
                          [2*(x_*y_+w_*z_), 1-2*(x_*x_+z_*z_), 2*(y_*z_-w_*x_)],
                          [2*(x_*z_-w_*y_), 2*(y_*z_+w_*x_), 1-2*(x_*x_+y_*y_)]])
        # ⚠ info 里存的 canonical_from_input_rot_wxyz **就是 输入系→规范系** 的旋转
        #   (名字即语义), 不要再当成反向的用。行向量写法要转置:
        #       p_canon = R_i2c @ (p_input - com)   →   (P - com) @ R_i2c.T
        #   用错方向实测: 接触点离规范网格表面中位 8.74mm(飘在旁边), 改对后 0.18mm(贴合)。
        Pc = (P - com) @ R_c2i.T
        rs = np.random.RandomState(0)
        hot_pts = Pc[W >= a.hot]
        warm_pts = Pc[(W > 0.05) & (W < a.hot)]
        for nm, arr in (("hot", hot_pts), ("warm", warm_pts)):
            if len(arr) > a.max_pts:
                arr = arr[rs.choice(len(arr), a.max_pts, replace=False)]
            if nm == "hot":
                hot_pts = arr
            else:
                warm_pts = arr
        zz = hot_pts[:, 2]
        print(f"  接触点: 热点 {len(hot_pts)} 个(红) / 次级 {len(warm_pts)} 个(暗红)")
        print(f"    热点高度 z ∈ [{zz.min()*100:+.1f}, {zz.max()*100:+.1f}] cm"
              f"  = 桌面往上 {(zz.min()-z_table)*100:.1f} ~ {(zz.max()-z_table)*100:.1f} cm\n")

    # 撒点分类可视化 —— 判据与 sample_init_pose 完全一致(读同一份 region.npz)
    #   有效  绿   落区内 + 非底面 + 非内壁 -> 真正会进优化
    #   底面  橙   外法向朝下, 贴桌那一面, 手伸不进去
    #   内壁  紫   法向背离主轴, 空心物体的腔内, 手伸不进去
    #   落区外 灰  距接触热点超过 radius
    surf = {}
    rp = f"assets/object/custom/processed_data/{a.oid}/region.npz"
    if a.show_sampling and os.path.isfile(rp):
        from scipy.spatial import cKDTree
        rz = np.load(rp)
        rpts = np.asarray(rz["points"], float)
        rw = np.asarray(rz["weight"], float)
        radius = float(rz["radius"]) if "radius" in rz else 0.02
        minw = float(rz["min_weight"]) if "min_weight" in rz else 0.2
        hotr = rpts[rw >= minw]
        sp_, ti_ = trimesh.sample.sample_surface_even(mesh, a.n_surf)
        fn = mesh.face_normals[ti_]                          # 外法向
        dd, _ = cKDTree(hotr).query(sp_)
        inreg = dd <= radius
        bottom = fn[:, 2] < -0.7
        ax = int(np.argmax(np.ptp(V, axis=0)))
        o = [i for i in range(3) if i != ax]
        rr = sp_[:, o] - V[:, o].mean(0)
        rn = rr / np.maximum(np.linalg.norm(rr, axis=1, keepdims=True), 1e-9)
        inner = (fn[:, o] * rn).sum(1) < -0.3
        surf["ok"] = sp_[inreg & ~bottom & ~inner]
        surf["bottom"] = sp_[inreg & bottom]
        surf["inner"] = sp_[inreg & ~bottom & inner]
        surf["out"] = sp_[~inreg]
        n = len(sp_)
        print(f"  撒点分类 (radius={radius*100:.0f}cm, 权重阈值 {minw:.2f}, 采样 {n} 点):")
        print(f"    ★有效  绿 {len(surf['ok']):5d}  {len(surf['ok'])/n:6.1%}   落区内+外壁 -> 进优化")
        print(f"     底面  橙 {len(surf['bottom']):5d}  {len(surf['bottom'])/n:6.1%}   贴桌, 手伸不进去")
        print(f"     内壁  紫 {len(surf['inner']):5d}  {len(surf['inner'])/n:6.1%}   杯腔内, 手伸不进去")
        print(f"     区外  灰 {len(surf['out']):5d}  {len(surf['out'])/n:6.1%}   离接触点太远\n")

    dome_c = np.zeros(3)
    if a.dome > 0:
        if a.dome_center == "contact" and hot_pts is not None and len(hot_pts):
            dome_c = hot_pts.mean(0)
        else:
            dome_c = V.mean(0)
        print(f"  半透明半球: 半径 {a.dome*100:.0f}cm, 球心 {a.dome_center} "
              f"{np.round(dome_c*100,1).tolist()} cm  —— 撒点/接近方向的覆盖范围\n")

    env = MuJoCo_OptEnv(hand_cfg=HandCfg(xml_path=a.hand, freejoint=True),
                        scene_cfg=sc, sim_cfg=MuJoCo_OptCfg())
    m, dat = env._model, env._data
    dat.qpos[:] = 0
    dat.qpos[2] = 2.0                      # 手挪到画面外, 只看物体摆放
    mujoco.mj_forward(m, dat)

    with mujoco.viewer.launch_passive(m, dat) as viewer:
        while viewer.is_running():
            scn = viewer.user_scn
            scn.ngeom = 0
            h = a.table_size / 2
            add_box(scn, [0, 0, z_table - 0.001], [h, h, 0.001],
                    [0.55, 0.55, 0.60, a.alpha])                  # 虚拟桌面
            # 坐标轴按通用约定: X 红 / Y 绿 / Z 蓝
            add_line(scn, [0, 0, z_table], [0.30, 0, z_table], [0.95, 0.15, 0.15, 1.0], 0.005)
            add_line(scn, [0, 0, z_table], [0, 0.30, z_table], [0.15, 0.85, 0.20, 1.0], 0.005)
            add_line(scn, [0, 0, z_table], [0, 0, z_table + 0.30], [0.20, 0.35, 1.0, 1.0], 0.005)
            if a.dome > 0:                                        # 半透明半球(撒点覆盖)
                c = dome_c
                for it in range(1, 7):                            # 纬线 6 圈(0~90°)
                    th = np.pi / 2 * it / 6
                    r_ = a.dome * np.sin(th); zz_ = a.dome * np.cos(th)
                    prev = None
                    for k in range(37):
                        ph = 2 * np.pi * k / 36
                        pt = c + np.array([r_ * np.cos(ph), r_ * np.sin(ph), zz_])
                        if prev is not None:
                            add_line(scn, prev, pt, [0.35, 0.75, 1.0, 0.30], 0.0010)
                        prev = pt
                for k in range(12):                               # 经线 12 条
                    ph = 2 * np.pi * k / 12
                    prev = None
                    for it in range(0, 13):
                        th = np.pi / 2 * it / 12
                        pt = c + np.array([a.dome * np.sin(th) * np.cos(ph),
                                           a.dome * np.sin(th) * np.sin(ph),
                                           a.dome * np.cos(th)])
                        if prev is not None:
                            add_line(scn, prev, pt, [0.35, 0.75, 1.0, 0.30], 0.0010)
                        prev = pt
            if surf:                                          # 撒点分类
                for p in surf["out"]:
                    add_sphere(scn, p, 0.0010, [0.45, 0.45, 0.48, 0.30])
                for p in surf["inner"]:
                    add_sphere(scn, p, 0.0016, [0.70, 0.25, 0.95, 0.80])
                for p in surf["bottom"]:
                    add_sphere(scn, p, 0.0016, [1.00, 0.55, 0.05, 0.90])
                for p in surf["ok"]:
                    add_sphere(scn, p, 0.0022, [0.15, 0.95, 0.35, 0.95])
            if hot_pts is not None:                               # 视频接触点
                for p in warm_pts:
                    add_sphere(scn, p, 0.0015, [0.55, 0.10, 0.10, 0.55])
                for p in hot_pts:
                    add_sphere(scn, p, 0.0022, [1.0, 0.12, 0.12, 0.95])
            for k in range(-2, 3):                                # 桌面网格线
                add_line(scn, [k * 0.25, -h, z_table], [k * 0.25, h, z_table],
                         [0.7, 0.7, 0.75, 0.30], 0.0012)
                add_line(scn, [-h, k * 0.25, z_table], [h, k * 0.25, z_table],
                         [0.7, 0.7, 0.75, 0.30], 0.0012)
            viewer.sync()


if __name__ == "__main__":
    main()
