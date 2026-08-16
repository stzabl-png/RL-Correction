"""按 GUI 的方式给 GraspPose 出图 —— 多视角 + MuJoCo 自己算的接触点。

取代 `render_contact_vs_grasp.py` 的单视角两栏图。那套图害我们误判了三次:
  ① 手落在物体背面 -> 被物体轮廓裁掉, 看着像插进去
  ② 空心物体的开口 -> 透过杯口看到后方的手, 看着像手在杯子里
  ③ 物体半透明也救不了 ②, 因为问题不是遮挡而是"开口是个洞"
真正的裁判是 MuJoCo 自己的接触判定, 所以这里把接触点直接画在图上,
并且一次出 4 个方位角 —— 4 个角度同时看着像穿模, 才值得怀疑。

图上:
  绿色小球 = MuJoCo 判定的接触点(mjVIS_CONTACTPOINT)。**没有接触点就是没碰上**;
  标题栏打印每个接触的 dist: 正=间隙, 负=穿透。

用法:
  MUJOCO_GL=egl python tools/render_grasp_views.py --grasp-dir <dir> --hand <xml> \\
      --out-dir <out> [--n 16] [--azimuths 0 90 180 270] [--size 480]
"""

import argparse
import glob
import os

import mujoco
import numpy as np


def load_env(npy, hand_xml):
    import sys
    dexo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, dexo)
    from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv
    d = np.load(npy, allow_pickle=True).item()
    sp = d["scene_path"]
    sc = np.load(sp, allow_pickle=True).item()
    base = os.path.dirname(sp)
    for _v in sc["scene"].values():                 # scene_cfg 存的是相对路径
        for _k in ("file_path", "xml_path", "urdf_path", "info_path"):
            if _k in _v:
                _v[_k] = os.path.normpath(os.path.join(base, _v[_k]))
    env = MuJoCo_OptEnv(hand_cfg=HandCfg(xml_path=hand_xml, freejoint=True),
                        scene_cfg=sc, sim_cfg=MuJoCo_OptCfg())
    return env, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasp-dir", required=True, help="含 *_grasp.npy 的目录(递归)")
    ap.add_argument("--hand", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n", type=int, default=16, help="最多渲染几个抓取")
    ap.add_argument("--azimuths", type=float, nargs="*", default=[0, 90, 180, 270])
    ap.add_argument("--elevation", type=float, default=-15.0)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--distance", type=float, default=0.34)
    ap.add_argument("--show-force", action="store_true", help="画接触力箭头(默认关, 太长会糊图)")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.grasp_dir}/**/*_grasp.npy", recursive=True))
    if not files:
        raise SystemExit(f"{a.grasp_dir} 里没有 *_grasp.npy")
    files = files[:a.n]
    os.makedirs(a.out_dir, exist_ok=True)

    env, _ = load_env(files[0], a.hand)
    m, dat = env._model, env._data
    m.vis.global_.offwidth = a.size
    m.vis.global_.offheight = a.size
    m.vis.headlight.ambient[:] = 0.5
    m.vis.headlight.diffuse[:] = 0.7
    renderer = mujoco.Renderer(m, a.size, a.size)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True   # ★接触点
    # 接触力箭头默认关: 它按力大小画成很长的线, 会糊住整张图(实测比物体还长几倍)。
    # 要看力的方向再用 --show-force 打开。
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = bool(a.show_force)
    m.vis.scale.contactwidth = 0.02      # 接触点画小一点, 别盖住指尖
    m.vis.scale.contactheight = 0.006

    def bname(g):
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""

    from PIL import Image
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        q = np.asarray(d["grasp_qpos"], float).reshape(-1)
        dat.qpos[:] = 0
        dat.qpos[:len(q)] = q
        dat.qvel[:] = 0
        mujoco.mj_forward(m, dat)

        cons = []
        for i in range(dat.ncon):
            c = dat.contact[i]
            b1, b2 = bname(c.geom1), bname(c.geom2)
            if b1.startswith("hand") != b2.startswith("hand"):
                cons.append(c.dist * 1000)
        worst = min(cons) if cons else float("nan")

        tiles = []
        for az in a.azimuths:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = np.asarray(d["ext_center"], float)
            cam.distance, cam.azimuth, cam.elevation = a.distance, az, a.elevation
            renderer.update_scene(dat, camera=cam, scene_option=opt)
            tiles.append(renderer.render())
        grid = np.concatenate([np.concatenate(tiles[:2], 1),
                               np.concatenate(tiles[2:4], 1)], 0) \
            if len(tiles) == 4 else np.concatenate(tiles, 1)
        name = os.path.basename(f).replace(".npy", "")
        Image.fromarray(grid).save(os.path.join(a.out_dir, f"{name}.png"))
        tag = "★穿透" if (cons and worst < 0) else "间隙"
        print(f"  {name}: 手-物接触 {len(cons)} 个, 最小 dist {worst:+.2f}mm ({tag})")
    print(f"-> {a.out_dir}  ({len(files)} 张, 每张 {len(a.azimuths)} 个视角)")


if __name__ == "__main__":
    main()
