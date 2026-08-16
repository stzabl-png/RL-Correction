"""把手的抓取模板**按参与手指分组**渲染出来, 供人工挑选。

为什么按手指分组: 38 个模板里很多是同一类抓法的变体(比如"五指+掌"就有 13 个)。
按"哪几根手指参与接触"分组之后, 组内是可比的, 一眼能看出哪些实质重复、哪些确实不同。

图上:
  * 手摆成该模板的姿态(raw_anno 里的 qpos), **不放物体** —— 看的是抓型本身
  * 橙色小球 = 该模板定义的接触点(hand_cpn_w), 大小固定
  * 每个模板 4 个视角拼一张

用法:
  MUJOCO_GL=egl python tools/render_templates_by_group.py --hand assets/hand/sharpa_wave_v2_left \\
      --xml left.xml --out-dir output/tmpl_groups [--size 360]
"""

import argparse
import glob
import os
from collections import defaultdict

import mujoco
import numpy as np
import yaml

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def finger_key(bodies):
    s = set()
    for x in bodies:
        x = str(x).replace("left_", "").replace("right_", "")
        for k in FINGERS:
            if x.startswith(k):
                s.add(k)
        if "C_MC" in x or x.startswith("hand"):
            s.add("palm")
    return tuple([k for k in FINGERS if k in s] + (["palm"] if "palm" in s else []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", required=True, help="手目录, 如 assets/hand/sharpa_wave_v2_left")
    ap.add_argument("--xml", default="left.xml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--size", type=int, default=360)
    ap.add_argument("--azimuths", type=float, nargs="*", default=[0, 90, 180, 270])
    ap.add_argument("--elevation", type=float, default=-15.0)
    ap.add_argument("--distance", type=float, default=0.38)
    a = ap.parse_args()

    mj = mujoco.MjModel.from_xml_path(f"{a.hand}/{a.xml}")
    dat = mujoco.MjData(mj)
    mj.vis.global_.offwidth = a.size
    mj.vis.global_.offheight = a.size
    mj.vis.headlight.ambient[:] = 0.5
    mj.vis.headlight.diffuse[:] = 0.7
    renderer = mujoco.Renderer(mj, a.size, a.size)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)

    from PIL import Image
    groups = defaultdict(list)
    for f in sorted(glob.glob(f"{a.hand}/init_tmpl/*.npy")):
        nm = os.path.basename(f)[:-4]
        t = np.load(f, allow_pickle=True).item()
        groups[finger_key(t["hand_cbody"])].append((nm, t))

    order = sorted(groups.items(), key=lambda kv: (-len(kv[0]), -len(kv[1])))
    for key, items in order:
        gname = "+".join(key)
        gdir = os.path.join(a.out_dir, f"{len(key)}指_{gname}")
        os.makedirs(gdir, exist_ok=True)
        print(f"\n【{gname}】 {len(items)} 个 -> {gdir}")
        for nm, t in sorted(items):
            tq = np.asarray(yaml.safe_load(open(f"{a.hand}/raw_anno/{nm}.yaml"))["qpos"], float)
            dat.qpos[:] = 0
            dat.qpos[:len(tq)] = tq
            mujoco.mj_forward(mj, dat)
            cpn = np.asarray(t["hand_cpn_w"], float)[:, :3]
            tiles = []
            for az in a.azimuths:
                cam = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(cam)
                cam.lookat[:] = cpn.mean(0)
                cam.distance, cam.azimuth, cam.elevation = a.distance, az, a.elevation
                renderer.update_scene(dat, camera=cam, scene_option=opt)
                scn = renderer.scene
                for p in cpn:                      # 模板定义的接触点
                    if scn.ngeom >= scn.maxgeom:
                        break
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                        np.array([0.004, 0, 0]), np.asarray(p, float),
                                        np.eye(3).flatten(),
                                        np.array([1.0, 0.5, 0.05, 1.0], np.float32))
                    scn.ngeom += 1
                tiles.append(renderer.render())
            grid = np.concatenate([np.concatenate(tiles[:2], 1),
                                   np.concatenate(tiles[2:4], 1)], 0) \
                if len(tiles) == 4 else np.concatenate(tiles, 1)
            Image.fromarray(grid).save(os.path.join(gdir, f"{nm}.png"))
            print(f"    {nm:<28} 接触点 {len(cpn)}")
    print(f"\n-> {a.out_dir}")


if __name__ == "__main__":
    main()
