"""Import an arbitrary object mesh into Dexonomy assets (generic version of the
Bi-V2AP-specific build_asset in recon_dexonomy_pipeline.py).

Example — a donut lying flat on the table (mesh z-up):
  python tools/import_object.py --mesh donut.obj --oid donut

Writes assets/object/custom/{processed_data,scene_cfg}/<oid>/... with both a
floating and a tabletop scene (tabletop carries a virtual_plane so gen_init
rejects grasps that reach below the table).

--up is the direction, IN THE INPUT MESH FRAME, that points away from the table
when the object rests in your scene (default 0 0 1 = mesh is modeled resting).

Note: the mesh is re-centered at its center of mass; the applied offset is saved
as `com_offset` in info/simplified.json. To map a synthesized grasp back into
your scene:  T_scene_hand = T_scene_object_inputframe * Trans(com_offset) * T_canon_hand
"""

import argparse
import json
import os

import numpy as np
import trimesh

ASSET_ROOT = os.path.join(os.path.dirname(__file__), "..", "assets", "object", "custom")


def quat_from_z_to(v):
    z = np.array([0.0, 0.0, 1.0])
    d = float(np.dot(z, v))
    if d > 1 - 1e-8:
        return np.array([1.0, 0, 0, 0])
    if d < -1 + 1e-8:
        return np.array([0.0, 1.0, 0, 0])
    axis = np.cross(z, v)
    axis /= np.linalg.norm(axis) + 1e-12
    ang = np.arccos(np.clip(d, -1, 1))
    s = np.sin(ang / 2)
    return np.array([np.cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--oid", required=True)
    ap.add_argument("--up", type=float, nargs=3, default=[0, 0, 1],
                    help="table-up direction in the input mesh frame")
    ap.add_argument("--auto-up", action="store_true",
                    help="derive --up from the most probable trimesh stable resting pose")
    ap.add_argument("--scale", type=float, default=1.0, help="uniform pre-scale on the mesh")
    ap.add_argument("--mass", type=float, default=0.1)
    ap.add_argument("--coacd-threshold", type=float, default=0.05)
    ap.add_argument("--coacd-preprocess", default="auto", choices=("auto", "on", "off"),
                    help="CoACD 预处理模式。'auto' 遇到退化三角面(零面积薄片)会在流形检查里"
                         "断言崩溃 (bottle_cap 实测: 3668 面里 60 个退化面 -> core dump); "
                         "'on' 强制体素重网格化, 能修好这类输入")
    ap.add_argument("--coacd-resolution", type=int, default=50,
                    help="预处理体素分辨率; 薄壁/细螺纹物体可能要调高到 100~200")
    args = ap.parse_args()

    import coacd  # after trimesh/numpy (OpenMP)

    oid = args.oid
    pdir = os.path.abspath(f"{ASSET_ROOT}/processed_data/{oid}")
    for sub in ("mesh", "urdf/meshes", "info"):
        os.makedirs(f"{pdir}/{sub}", exist_ok=True)
    scdir_f = os.path.abspath(f"{ASSET_ROOT}/scene_cfg/{oid}/floating")
    scdir_t = os.path.abspath(f"{ASSET_ROOT}/scene_cfg/{oid}/tabletop")
    os.makedirs(scdir_f, exist_ok=True)
    os.makedirs(scdir_t, exist_ok=True)

    m = trimesh.load(args.mesh, force="mesh")
    if args.scale != 1.0:
        m.apply_scale(args.scale)
    if args.auto_up:
        Ts, probs = trimesh.poses.compute_stable_poses(m)
        ups = [T[:3, :3].T @ np.array([0.0, 0.0, 1.0]) for T in Ts]
        pick, why = 0, f"most probable (p={probs[0]:.2f})"
        # disambiguate with the video pose estimate when available: the scene pose
        # follows the video, not physical stability (e.g. a bottle standing upright)
        wf = os.path.join(os.path.dirname(os.path.abspath(args.mesh)), "world_fused.npz")
        if os.path.exists(wf):
            z = np.load(wf, allow_pickle=True)
            if "object_ob_in_world" in z.files:
                ob = z["object_ob_in_world"]
                valid = z["object_pose_valid_by_frame"][0] \
                    if "object_pose_valid_by_frame" in z.files else np.ones(len(ob), bool)
                Rv = ob[int(np.argmax(valid))][:3, :3]
                vup = Rv.T @ np.array([0.0, 0.0, 1.0])
                vup /= np.linalg.norm(vup)
                cand = [i for i, p in enumerate(probs) if p >= 0.02] or [0]
                angs = {i: np.degrees(np.arccos(np.clip(np.dot(ups[i], vup), -1, 1)))
                        for i in cand}
                pick = min(angs, key=angs.get)
                why = f"closest to video up ({angs[pick]:.0f} deg, p={probs[pick]:.2f})"
                if angs[pick] > 45:
                    print(f"[import] WARNING: chosen stable pose is {angs[pick]:.0f} deg from "
                          f"the video estimate — verify the resting orientation manually (--up)")
        args.up = ups[pick].tolist()
        print(f"[import] auto-up: {np.round(args.up, 4).tolist()}  [{why}]")
    com = m.center_mass.copy()
    m.apply_translation(-com)  # canonical frame: COM at origin

    # canonicalize to the resting pose: rotate so that --up becomes +z, i.e. the
    # canonical frame IS the physical resting frame (table = horizontal plane below).
    # Mapping back to the input mesh frame: p_input = R_c2i @ p_canon + com
    up_in = np.asarray(args.up, np.float64)
    up_in /= np.linalg.norm(up_in)
    q_align = quat_from_z_to(up_in)          # rotates +z -> up_in  (canon -> input)
    from scipy.spatial.transform import Rotation as Rot
    R_c2i = Rot.from_quat(np.roll(q_align, -1)).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_c2i.T                       # input -> canon
    m.apply_transform(T)
    m.export(f"{pdir}/mesh/simplified.obj")
    m.export(f"{pdir}/mesh/normalized.obj")

    parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces),
                            threshold=args.coacd_threshold,
                            preprocess_mode=args.coacd_preprocess,
                            preprocess_resolution=args.coacd_resolution)
    names = []
    for i, (v, f) in enumerate(parts):
        nm = f"convex_piece_{i:03d}.obj"
        names.append(nm)
        trimesh.Trimesh(v, f).export(f"{pdir}/urdf/meshes/{nm}")

    asset = "\n".join(f'    <mesh name="{n}" file="meshes/{n}"/>' for n in names)
    geoms = []
    for i, n in enumerate(names):
        geoms.append(f'      <geom name="object_visual_{i}" type="mesh" contype="0" conaffinity="0" density="0" mesh="{n}"/>')
        geoms.append(f'      <geom name="object_collision_{i}" type="mesh" mesh="{n}"/>')
    open(f"{pdir}/urdf/coacd.xml", "w").write(
        f'<mujoco model="MuJoCo Model">\n  <compiler angle="radian" meshdir="."/>\n'
        f'  <asset>\n{asset}\n  </asset>\n  <worldbody>\n    <body name="object">\n'
        f'{chr(10).join(geoms)}\n    </body>\n  </worldbody>\n</mujoco>\n')
    open(f"{pdir}/urdf/coacd.urdf", "w").write('<robot name="object"></robot>\n')

    obb = m.bounding_box_oriented.primitive.extents.tolist()
    json.dump({"gravity_center": [0.0, 0.0, 0.0], "obb": obb, "scale": 1.0,
               "density": float(args.mass / max(m.volume, 1e-9)), "mass": args.mass,
               "com_offset": com.tolist(),
               "canonical_from_input_rot_wxyz": np.roll(
                   Rot.from_matrix(R_c2i.T).as_quat(), 1).tolist()},
              open(f"{pdir}/info/simplified.json", "w"), indent=1)

    # canonical frame is z-up by construction: table plane at the lowest vertex.
    # The tabletop scene gets a REAL plane object (not just virtual_plane), so the
    # table exists in all three stages: init plane filter, grasp refinement
    # (MuJoCo collision, 2cm margin cushion) and eval.
    zmin = float(m.vertices[:, 2].min())
    plane_pose = np.array([0, 0, zmin, 1, 0, 0, 0], np.float32)

    rp = f"../../../processed_data/{oid}"
    base = {"type": "rigid_object",
            "file_path": f"{rp}/mesh/simplified.obj", "xml_path": f"{rp}/urdf/coacd.xml",
            "urdf_path": f"{rp}/urdf/coacd.urdf", "info_path": f"{rp}/info/simplified.json",
            "scale": np.array([1.0, 1.0, 1.0], np.float32),
            "pose": np.array([0, 0, 0, 1, 0, 0, 0], np.float32)}
    np.save(f"{scdir_f}/scale010.npy",
            {"scene": {oid: dict(base)}, "scene_id": f"{oid}/floating/scale010",
             "task": {"type": "force_closure", "obj_name": oid}})
    # tabletop scene carries BOTH plane representations:
    # - virtual_plane on the object: consumed by the init-stage skeleton filter
    #   (with filter.collision.hard_plane=True it is a hard constraint);
    # - a real "plane" object: a MuJoCo collision plane, so the object rests on
    #   the table during grasp refinement and eval (EvalCfg plane_margin=0; for
    #   thin objects run op=grasp with a small cushion, e.g. op.grasp.plane_margin=0.008).
    tinfo = dict(base)
    tinfo["virtual_plane"] = plane_pose
    np.save(f"{scdir_t}/scale010.npy",
            {"scene": {oid: tinfo, "table": {"type": "plane", "pose": plane_pose}},
             "scene_id": f"{oid}/tabletop/scale010",
             "task": {"type": "force_closure", "obj_name": oid}})
    print(f"[import] {oid}: {len(names)} convex parts, obb={np.round(obb, 3).tolist()}, "
          f"com_offset={np.round(com, 4).tolist()}")
    print(f"[import] scene_cfg: {scdir_t}/scale010.npy (+ floating)")


if __name__ == "__main__":
    main()
