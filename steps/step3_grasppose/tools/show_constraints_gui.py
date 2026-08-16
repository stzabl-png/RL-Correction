"""在 MuJoCo GUI 里把**采样约束**画出来 —— 接触区、朝向人体的方向、小臂圆锥。

对应的设计(2026-08-15 讨论):
  ① 落区    : 采样点必须落在视频接触热点的 r 邻域内
  ② 虎口朝下 : thumb_z > -0.25 (现有判据, 前移到优化之前)
  ③ 小臂朝向 : 小臂方向(腕→肘, 手根系 -z)与「朝向人体」夹角 <= 60°
     ★为什么约束朝向而不是腕点位置: top-down 抓取时腕点在物体正上方,
       位置方位角是退化的(头顶转一点就翻前后); 而小臂指向永远有意义 ——
       人在 -X 侧从上往下抓, 小臂也必须朝 -X 斜上方伸出去, 不可能朝 +X。

画面里:
  红色小球   视频接触热点(落区约束 ①)
  蓝色粗箭头 「朝向人体」方向 front (= RL 场景里机器人所在的 -X 侧)
  蓝色细线锥 允许的小臂方向范围(半角 60°)
  绿/红箭头  当前抓取的小臂方向 —— 绿=在锥内(通过), 红=在锥外(拒绝)

用法(注意 MUJOCO_GL 不能是 egl):
  env -u MUJOCO_GL DISPLAY=:1 python tools/show_constraints_gui.py \\
      --grasp-dir <dir> --hand <xml> [--recon <take> --object object_0 --side left]
      [--cone-deg 60] [--dwell 2.5]
"""

import argparse
import glob
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

DEXO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DEXO)
os.chdir(DEXO)
from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv  # noqa: E402


def quat_wxyz_to_R(q):
    w, x, y, z = np.asarray(q, float)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def add_line(scn, p0, p1, rgba, width=0.0015):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.eye(3).flatten(), np.asarray(rgba, np.float32))
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp-dir", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--recon", default=None, help="给了就画视频接触热点")
    ap.add_argument("--object", default="object_0")
    ap.add_argument("--side", default="left")
    ap.add_argument("--oid", default=None, help="导入的 oid, 用于把接触点换到规范系")
    ap.add_argument("--front", type=float, nargs=3, default=[-1.0, 0.0, 0.0],
                    help="「朝向人体」方向。缺省 -X = RL 场景里机器人所在侧。"
                         "规范系方位角钉死之后, 这里应改用 upright_from_recon 输出的 front。")
    ap.add_argument("--cone-deg", type=float, default=60.0)
    ap.add_argument("--dwell", type=float, default=2.5, help="每个抓取停留几秒")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.grasp_dir}/**/*_grasp.npy", recursive=True))[:a.n]
    if not files:
        raise SystemExit(f"{a.grasp_dir} 里没有 *_grasp.npy")

    d0 = np.load(files[0], allow_pickle=True).item()
    sc = np.load(d0["scene_path"], allow_pickle=True).item()
    base = os.path.dirname(d0["scene_path"])
    for _v in sc["scene"].values():
        for _k in ("file_path", "xml_path", "urdf_path", "info_path"):
            if _k in _v:
                _v[_k] = os.path.normpath(os.path.join(base, _v[_k]))
    env = MuJoCo_OptEnv(hand_cfg=HandCfg(xml_path=a.hand, freejoint=True),
                        scene_cfg=sc, sim_cfg=MuJoCo_OptCfg())
    m, dat = env._model, env._data

    # 接触热点 -> 规范系
    hot = None
    if a.recon and a.oid:
        z = np.load(f"{a.recon}/contact/contact_v2_{a.object}_{a.side}.npz", allow_pickle=True)
        P = np.asarray(z["probe_local"], float)
        W = np.asarray(z["weight"], float)
        P = P[W >= 0.5]
        import json
        info = json.load(open(f"assets/object/custom/processed_data/{a.oid}/info/simplified.json"))
        R_c2i = quat_wxyz_to_R(info["canonical_from_input_rot_wxyz"])
        com = np.asarray(info["com_offset"], float)
        hot = (P - com) @ R_c2i.T
        if len(hot) > 600:
            hot = hot[np.random.RandomState(0).choice(len(hot), 600, replace=False)]

    front = np.asarray(a.front, float)
    front /= np.linalg.norm(front)
    e1 = np.cross(front, [0, 0, 1.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(front, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(front, e1)
    half = np.radians(a.cone_deg)

    print(f"\n约束可视化: 小臂圆锥半角 {a.cone_deg:.0f}°, front = {np.round(front, 2)}")
    print(f"抓取 {len(files)} 个, 每个停 {a.dwell}s\n")

    with mujoco.viewer.launch_passive(m, dat) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        idx, t_last = 0, 0.0
        while viewer.is_running():
            if time.time() - t_last > a.dwell:
                t_last = time.time()
                d = np.load(files[idx % len(files)], allow_pickle=True).item()
                q = np.asarray(d["grasp_qpos"], float).reshape(-1)
                dat.qpos[:] = 0
                dat.qpos[:len(q)] = q
                dat.qvel[:] = 0
                mujoco.mj_forward(m, dat)

                R = quat_wxyz_to_R(q[3:7])
                fa = R @ np.array([0.0, 0.0, -1.0])          # 小臂方向 = 手根系 -z
                ang = np.degrees(np.arccos(np.clip(fa @ front, -1, 1)))
                ok = ang <= a.cone_deg
                cpn = np.asarray(d["hand_cpn_w"], float)[:, :3]
                ctr = cpn.mean(0)

                scn = viewer.user_scn
                scn.ngeom = 0
                if hot is not None:                           # ① 接触热点
                    for p in hot:
                        add_sphere(scn, p, 0.0018, [0.9, 0.12, 0.12, 0.85])
                add_line(scn, ctr, ctr + front * 0.16,        # front 方向
                         [0.15, 0.4, 1.0, 0.95], 0.004)
                for k in range(24):                            # ③ 小臂允许锥
                    ph = 2 * np.pi * k / 24
                    dirk = (np.cos(half) * front
                            + np.sin(half) * (np.cos(ph) * e1 + np.sin(ph) * e2))
                    add_line(scn, ctr, ctr + dirk * 0.16, [0.15, 0.4, 1.0, 0.5], 0.0012)
                col = [0.1, 0.9, 0.2, 1.0] if ok else [1.0, 0.15, 0.1, 1.0]
                add_line(scn, ctr, ctr + fa * 0.16, col, 0.005)   # 当前小臂方向

                print(f"  [{idx % len(files) + 1}/{len(files)}] "
                      f"{os.path.basename(files[idx % len(files)])}  "
                      f"小臂与front夹角 {ang:5.1f}°  -> {'✓通过' if ok else '✗拒绝'}", flush=True)
                idx += 1
            viewer.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
