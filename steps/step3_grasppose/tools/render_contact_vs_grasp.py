"""左右对照渲染: 左=视频提取的接触区(region.npz 红点), 右=合成 GraspPose。

对每个经 Isaac 接触验证的抓取(isaac_traj 里的 30 上限集)出一张 1280x640:
左幅把手挪出画面只留物体+接触点, 右幅同一相机下手在 grasp_qpos。

  MUJOCO_GL=egl python tools/render_contact_vs_grasp.py \
      --exp-dir output/phone_selt_sharpa_wave --out-dir output/sel_gallery
"""

import argparse
import glob
import os

import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hand-xml", default="assets/hand/sharpa_wave/right.xml")
    ap.add_argument("--distance", type=float, default=0.34)
    ap.add_argument("--azimuth", type=float, default=160.0,
                    help="相机方位角。★默认单一视角会造成'手在物体背后'被误看成穿模 —— "
                         "有争议时换 2~3 个方位角再看(2026-08-15 踩过)。")
    ap.add_argument("--elevation", type=float, default=-12.0)
    ap.add_argument("--obj-alpha", type=float, default=1.0,
                    help="物体透明度(1=不透明)。★手落在物体背面时, 不透明渲染会让遮挡看起来"
                         "像穿模 —— 2026-08-15 因此误判过三次。有争议时设 0.35 一次看清。")
    ap.add_argument("--max-pts", type=int, default=300)
    args = ap.parse_args()

    import mujoco
    import imageio.v2 as imageio
    from dexonomy.sim import MuJoCo_OptEnv, MuJoCo_OptCfg, HandCfg
    from dexonomy.util.file_util import load_scene_cfg

    # 只渲染 Isaac 验证集里的抓取(与双口径表同集)
    traj_root = os.path.join(args.exp_dir, "isaac_traj")
    names = sorted(os.path.basename(d.rstrip("/"))
                   for d in glob.glob(f"{traj_root}/*/") if os.path.isdir(d))
    allf = sorted(glob.glob(f"{args.exp_dir}/grasp_data/**/*.npy", recursive=True))
    files = ([f for f in allf if os.path.basename(f).replace(".npy", "") in names]
             if names else allf)          # 没跑 Isaac 时直接渲全部候选
    if not files:
        print(f"{args.exp_dir}: 无匹配抓取")
        return

    d0 = np.load(files[0], allow_pickle=True).item()
    scene_cfg = load_scene_cfg(str(d0["scene_path"]))
    region = (scene_cfg.get("task") or {}).get("region")
    rpts_canon = None
    if region:
        rp = region["path"]
        if not os.path.isfile(rp):                       # 远端同步来的绝对路径 → 本地回退
            oid = scene_cfg["task"]["obj_name"]
            rp = f"assets/object/custom/processed_data/{oid}/region.npz"
        rz = np.load(rp)
        pts, w = np.asarray(rz["points"]), np.asarray(rz["weight"])
        pts = pts[w >= float(rz["min_weight"])]
        if len(pts) > args.max_pts:                      # MjvScene geom 上限
            pts = pts[np.random.RandomState(0).choice(len(pts), args.max_pts, replace=False)]
        rpts_canon = pts

    env = MuJoCo_OptEnv(hand_cfg=HandCfg(xml_path=args.hand_xml, freejoint=True),
                        scene_cfg=scene_cfg, sim_cfg=MuJoCo_OptCfg())
    env._model.vis.global_.offwidth = 640
    env._model.vis.global_.offheight = 640
    env._model.vis.headlight.ambient[:] = 0.5
    env._model.vis.headlight.diffuse[:] = 0.7
    if args.obj_alpha < 1.0:
        # 只把**物体**设半透明, 手保持不透明。
        # ⚠ 不能用 "body 名以 left_/right_ 开头" 反选 —— 合成场景里手的刚体名未必带这个前缀,
        #   实测把手也一起透明了。改成正选: 物体的刚体名就是 scene_cfg 里的 obj_name。
        _obj_names = set()
        _t = scene_cfg.get("task") if isinstance(scene_cfg, dict) else None
        if _t and _t.get("obj_name"):
            _obj_names.add(str(_t["obj_name"]))
        for _k in (scene_cfg.get("scene") or {}):
            if _k != "table":
                _obj_names.add(str(_k))
        for g in range(env._model.ngeom):
            bname = mujoco.mj_id2name(env._model, mujoco.mjtObj.mjOBJ_BODY,
                                      env._model.geom_bodyid[g]) or ""
            if any(o in bname for o in _obj_names):
                env._model.geom_rgba[g, 3] = args.obj_alpha
    renderer = mujoco.Renderer(env._model, 640, 640)
    os.makedirs(args.out_dir, exist_ok=True)

    tag = os.path.basename(args.exp_dir.rstrip("/")).replace("_sharpa_wave", "")

    def draw_pts(scene, pts_w):
        for p in pts_w:
            if scene.ngeom >= scene.maxgeom - 1:
                break
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.002, 0, 0]), np.asarray(p, np.float64),
                                np.eye(3).flatten(),
                                np.array([0.9, 0.12, 0.12, 0.9], dtype=np.float32))
            scene.ngeom += 1

    n = 0
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        pose, scale = np.asarray(d["obj_pose"], float), np.asarray(d["obj_scale"], float)
        Rm = R.from_quat(np.roll(pose[3:7], -1)).as_matrix()
        pts_w = (rpts_canon * scale) @ Rm.T + pose[:3] if rpts_canon is not None else []

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = d["ext_center"]
        cam.distance, cam.azimuth, cam.elevation = args.distance, args.azimuth, args.elevation

        q = d["grasp_qpos"][0].astype(np.float32)
        panels = []
        for hand_visible in (False, True):
            qq = q.copy()
            if not hand_visible:
                qq[2] += 5.0                             # 手挪出画面(freejoint 根 z)
            env.reset_qpos(qq)
            mujoco.mj_forward(env._model, env._data)
            renderer.update_scene(env._data, cam)
            if not hand_visible:
                draw_pts(renderer.scene, pts_w)
            else:
                cpn = np.asarray(d["obj_cpn_w"], float)[:, :3]
                for pp in cpn:                       # 合成规划的接触点(绿)
                    if renderer.scene.ngeom >= renderer.scene.maxgeom - 1:
                        break
                    g = renderer.scene.geoms[renderer.scene.ngeom]
                    import mujoco as _mj
                    _mj.mjv_initGeom(g, _mj.mjtGeom.mjGEOM_SPHERE,
                                     np.array([0.004, 0, 0]), pp, np.eye(3).flatten(),
                                     np.array([0.1, 0.9, 0.2, 1.0], dtype=np.float32))
                    renderer.scene.ngeom += 1
            panels.append(renderer.render().copy())
        img = np.concatenate(panels, axis=1)             # 左接触点 | 右GraspPose
        name = os.path.basename(f).replace(".npy", "")
        imageio.imwrite(f"{args.out_dir}/{tag}_{name}.png", img)
        n += 1
    print(f"{tag}: {n} 张 -> {args.out_dir}")


if __name__ == "__main__":
    main()
