"""Post-filter: remove grasps whose hand geometry goes below the table plane.

  MUJOCO_GL=egl python tools/filter_plane_clearance.py --exp-dir output/pp2_sharpa_wave \
      [--data succ_grasp] [--thre -0.002] [--check-pregrasp]

For every grasp npy, FK the hand at grasp_qpos (and optionally each pregrasp_qpos)
and compute the minimum signed clearance of all hand collision-mesh vertices to the
virtual table plane. Files with clearance < --thre are moved to <data>_below_plane/
(mirrored layout). Prints a per-file report.
"""

import argparse
import glob
import os
import shutil

import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--data", default="succ_grasp")
    ap.add_argument("--thre", type=float, default=-0.002,
                    help="min allowed clearance (m); negative allows slight dip")
    ap.add_argument("--check-pregrasp", action="store_true")
    ap.add_argument("--hand-xml", default="assets/hand/sharpa_wave/right.xml")
    args = ap.parse_args()

    import mujoco
    from dexonomy.sim import MuJoCo_OptEnv, MuJoCo_OptCfg, HandCfg
    from dexonomy.util.file_util import load_scene_cfg

    files = sorted(glob.glob(f"{args.exp_dir}/{args.data}/**/*.npy", recursive=True))
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        print("no files")
        return

    d0 = np.load(files[0], allow_pickle=True).item()
    scene_cfg = load_scene_cfg(str(d0["scene_path"]))
    plane = None
    for o in scene_cfg["scene"].values():
        if o.get("type") == "rigid_object" and "virtual_plane" in o:
            plane = np.asarray(o["virtual_plane"], np.float64)
        elif o.get("type") == "plane":
            plane = np.asarray(o["pose"], np.float64)
    if plane is None:
        print("scene has no table plane; nothing to filter")
        return
    p0 = plane[:3]
    n = R.from_quat(np.roll(plane[3:7], -1)).as_matrix()[:, 2]  # plane +z = up

    env = MuJoCo_OptEnv(hand_cfg=HandCfg(xml_path=args.hand_xml, freejoint=True),
                        scene_cfg=scene_cfg, sim_cfg=MuJoCo_OptCfg())
    model, data = env._model, env._data

    # collect hand collision-geom vertices (local frames) once
    hand_geoms = []
    for gi in range(model.ngeom):
        g = model.geom(gi)
        if g.contype[0] == 0 and g.conaffinity[0] == 0:
            continue
        body_name = model.body(g.bodyid[0]).name
        if not body_name.startswith(env._cfg.hand_prefix):
            continue
        if g.type[0] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = g.dataid[0]
            adr, num = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            v = model.mesh_vert[adr: adr + num]
            hand_geoms.append((gi, v))
    if not hand_geoms:
        print("no hand collision meshes found")
        return

    def min_clearance(qpos29):
        env.reset_qpos(qpos29.astype(np.float32))
        mujoco.mj_forward(model, data)
        lo = np.inf
        for gi, v in hand_geoms:
            xmat = data.geom_xmat[gi].reshape(3, 3)
            w = v @ xmat.T + data.geom_xpos[gi]
            lo = min(lo, float(((w - p0) @ n).min()))
        return lo

    reject_root = f"{args.exp_dir}/{args.data}_below_plane"
    kept, moved = 0, 0
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        c = min_clearance(d["grasp_qpos"][0])
        if args.check_pregrasp:
            for q in d["pregrasp_qpos"]:
                c = min(c, min_clearance(q))
        ok = c >= args.thre
        print(f"{os.path.basename(f):22s} clearance={c*1000:+7.1f}mm  {'OK' if ok else 'BELOW -> moved'}")
        if ok:
            kept += 1
        else:
            dst = f.replace(f"/{args.data}/", f"/{args.data}_below_plane/")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(os.path.realpath(f), dst)
            if os.path.islink(f):
                os.unlink(f)
            moved += 1
    print(f"\nkept {kept}, moved {moved} -> {reject_root}")


if __name__ == "__main__":
    main()
