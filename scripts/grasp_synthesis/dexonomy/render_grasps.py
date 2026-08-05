"""Render synthesized grasps with the virtual table drawn as a translucent pink plane.

  MUJOCO_GL=egl python tools/render_grasps.py --exp-dir output/pp2_sharpa_wave \
      [--data succ_grasp] [--limit 8] [--hand-xml assets/hand/sharpa_wave/right.xml]

Outputs <exp-dir>/renders/<name>.png (two views stacked) and a montage.png.
The pink plane is the `virtual_plane` from the scene cfg — the half-space the
gen_init plane filter keeps the hand skeleton out of (i.e. the table).
"""

import argparse
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--data", default="succ_grasp", choices=["succ_grasp", "grasp_data", "init_data"])
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--hand-xml", default="assets/hand/sharpa_wave/right.xml")
    ap.add_argument("--distance", type=float, default=0.32)
    args = ap.parse_args()

    import mujoco
    import imageio.v2 as imageio
    from dexonomy.sim import MuJoCo_OptEnv, MuJoCo_OptCfg, HandCfg
    from dexonomy.util.file_util import load_scene_cfg

    files = sorted(glob.glob(f"{args.exp_dir}/{args.data}/**/*.npy", recursive=True))[:args.limit]
    if not files:
        print(f"no npy under {args.exp_dir}/{args.data}")
        return
    outdir = f"{args.exp_dir}/renders"
    os.makedirs(outdir, exist_ok=True)

    d0 = np.load(files[0], allow_pickle=True).item()
    scene_cfg = load_scene_cfg(str(d0["scene_path"]))
    vplane = None
    for o in scene_cfg["scene"].values():
        if o.get("type") == "rigid_object" and "virtual_plane" in o:
            vplane = np.asarray(o["virtual_plane"], np.float64)  # [px,py,pz, qw,qx,qy,qz], plane +z = up
        elif o.get("type") == "plane":
            vplane = None  # real plane geom exists in the model; recolored pink below

    env = MuJoCo_OptEnv(hand_cfg=HandCfg(xml_path=args.hand_xml, freejoint=True),
                        scene_cfg=scene_cfg, sim_cfg=MuJoCo_OptCfg())
    env._model.vis.global_.offwidth = 640
    env._model.vis.global_.offheight = 640
    env._model.vis.headlight.ambient[:] = 0.5
    env._model.vis.headlight.diffuse[:] = 0.7
    # real table plane geoms (scene type "plane") -> translucent pink
    for gi in range(env._model.ngeom):
        if env._model.geom(gi).name.startswith("plane-"):
            env._model.geom_rgba[gi] = [1.0, 0.45, 0.72, 0.35]
    renderer = mujoco.Renderer(env._model, 640, 640)

    def draw_plane(scene):
        if vplane is None:
            return
        g = scene.geoms[scene.ngeom]
        mat = R.from_quat(np.roll(vplane[3:7], -1)).as_matrix()
        # thin box, top face flush with the table surface
        pos = vplane[:3] - mat @ np.array([0, 0, 0.0011])
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX,
                            np.array([0.14, 0.14, 0.001]), pos, mat.flatten(),
                            np.array([1.0, 0.45, 0.72, 0.35], dtype=np.float32))
        scene.ngeom += 1

    tiles = []
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        env.reset_qpos(d["grasp_qpos"][0].astype(np.float32))
        mujoco.mj_forward(env._model, env._data)
        views = []
        for az, el in ((120, -25), (240, -25)):
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = d["ext_center"]
            cam.distance = args.distance
            cam.azimuth, cam.elevation = az, el
            renderer.update_scene(env._data, cam)
            draw_plane(renderer.scene)
            views.append(renderer.render().copy())
        img = np.concatenate(views, axis=0)
        name = os.path.basename(f).replace(".npy", "")
        imageio.imwrite(f"{outdir}/{name}.png", img)
        tiles.append(img)
        print(f"rendered {name}")
    imageio.imwrite(f"{outdir}/montage.png", np.concatenate(tiles, axis=1))
    print(f"saved {len(tiles)} renders + montage to {outdir}")


if __name__ == "__main__":
    main()
