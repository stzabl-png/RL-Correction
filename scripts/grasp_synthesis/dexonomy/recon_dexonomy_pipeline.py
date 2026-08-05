#!/usr/bin/env python
"""
Consolidated pipeline: Bi-V2AP reconstruction -> Dexonomy asset -> grasp synthesis
-> validation -> OCIR-format per-object output.

Adds TABLETOP (table-aware) support: an object-frame `virtual_plane` derived from
the reconstructed on-table resting orientation (world_fused.npz). Dexonomy's init
`check_plane` then rejects grasps whose hand skeleton dips below that table.

Stages (run any subset via --stages):
  assets   : mesh -> COACD collision + info + floating/tabletop scene_cfg
  synth    : dexrun op=init -> op=grasp -> op=eval   (per template, over all objects)
  export   : OCIR-format bundle per (object, template) + summary.json

Usage:
  python tools/recon_dexonomy_pipeline.py \
      --recon-base /home/lyh/Project/Bi-V2AP/Output/ReconstructOutput/egodex/part2/basic_pick_place \
      --objects 0-19 --templates 1_Large_Diameter --scene tabletop \
      --exp-name recon_pp --gpu 0 --stages assets synth export
"""
import os, sys, json, argparse, subprocess, glob, shutil
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
ASSET_ROOT = "assets/object/recon"
HAND_XML = "assets/hand/shadow/right.xml"

JOINTS = ["rh_FFJ4","rh_FFJ3","rh_FFJ2","rh_FFJ1","rh_MFJ4","rh_MFJ3","rh_MFJ2","rh_MFJ1",
          "rh_RFJ4","rh_RFJ3","rh_RFJ2","rh_RFJ1","rh_LFJ5","rh_LFJ4","rh_LFJ3","rh_LFJ2",
          "rh_LFJ1","rh_THJ5","rh_THJ4","rh_THJ3","rh_THJ2","rh_THJ1"]


# ----------------------------------------------------------------------------- helpers
def quat_from_z_to(vec):
    """wxyz quaternion of the shortest-arc rotation mapping +z -> vec (unit)."""
    z = np.array([0.0, 0.0, 1.0]); v = vec / (np.linalg.norm(vec) + 1e-12)
    d = float(np.dot(z, v))
    if d > 1 - 1e-8:
        return np.array([1.0, 0, 0, 0])
    if d < -1 + 1e-8:
        return np.array([0.0, 1.0, 0, 0])          # 180 deg about x
    axis = np.cross(z, v); axis /= (np.linalg.norm(axis) + 1e-12)
    ang = np.arccos(np.clip(d, -1, 1)); s = np.sin(ang / 2)
    return np.array([np.cos(ang / 2), axis[0]*s, axis[1]*s, axis[2]*s])


def resting_up_in_object_frame(recon_dir):
    """world +z expressed in the object canonical frame, from the first valid
    on-table frame of world_fused.npz. Returns unit 3-vector or None."""
    wf = os.path.join(recon_dir, "world_fused.npz")
    if not os.path.exists(wf):
        return None
    z = np.load(wf, allow_pickle=True)
    if "object_ob_in_world" not in z.files:
        return None
    ob = z["object_ob_in_world"]                       # (T,4,4) object->world
    valid = z["object_pose_valid_by_frame"][0] if "object_pose_valid_by_frame" in z.files \
        else np.ones(len(ob), bool)
    idx = int(np.argmax(valid)) if valid.any() else 0
    R = ob[idx][:3, :3]                                # object->world rotation
    up_obj = R.T @ np.array([0.0, 0.0, 1.0])           # world-up in object frame
    n = np.linalg.norm(up_obj)
    return up_obj / n if n > 1e-6 else None


# ----------------------------------------------------------------------------- stage: assets
def build_asset(recon_dir, oid, threshold=0.05):
    import trimesh, coacd
    src = os.path.join(recon_dir, "object_mesh_scaled_final.obj")
    pdir = f"{ASSET_ROOT}/processed_data/{oid}"
    os.makedirs(f"{pdir}/mesh", exist_ok=True)
    os.makedirs(f"{pdir}/urdf/meshes", exist_ok=True)
    os.makedirs(f"{pdir}/info", exist_ok=True)
    scdir_f = f"{ASSET_ROOT}/scene_cfg/{oid}/floating"; os.makedirs(scdir_f, exist_ok=True)
    scdir_t = f"{ASSET_ROOT}/scene_cfg/{oid}/tabletop"; os.makedirs(scdir_t, exist_ok=True)

    m = trimesh.load(src, force="mesh")
    m.apply_translation(-m.center_mass)                # center at COM
    m.export(f"{pdir}/mesh/simplified.obj"); m.export(f"{pdir}/mesh/normalized.obj")

    parts = coacd.run_coacd(coacd.Mesh(m.vertices, m.faces), threshold=threshold)
    names = []
    for i, (v, f) in enumerate(parts):
        nm = f"convex_piece_{i:03d}.obj"; names.append(nm)
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
    mass = 0.1
    json.dump({"gravity_center": [0.0, 0.0, 0.0], "obb": obb, "scale": 1.0,
               "density": float(mass / max(m.volume, 1e-9)), "mass": mass},
              open(f"{pdir}/info/simplified.json", "w"), indent=1)

    # virtual_plane in object frame from recon resting orientation
    up = resting_up_in_object_frame(recon_dir)
    vplane = None
    if up is not None:
        contact = m.vertices[np.argmin(m.vertices @ up)]       # lowest point along up
        q = quat_from_z_to(up)                                 # plane +z -> up
        vplane = np.concatenate([contact.astype(np.float32), q.astype(np.float32)])

    rp = f"../../../processed_data/{oid}"
    base = {"type": "rigid_object",
            "file_path": f"{rp}/mesh/simplified.obj", "xml_path": f"{rp}/urdf/coacd.xml",
            "urdf_path": f"{rp}/urdf/coacd.urdf", "info_path": f"{rp}/info/simplified.json",
            "scale": np.array([1.0, 1.0, 1.0], np.float32),
            "pose": np.array([0, 0, 0, 1, 0, 0, 0], np.float32)}
    # floating
    np.save(f"{scdir_f}/scale010.npy",
            {"scene": {oid: dict(base)}, "scene_id": f"{oid}/floating/scale010",
             "task": {"type": "force_closure", "obj_name": oid}})
    # tabletop (adds virtual_plane if available)
    tinfo = dict(base)
    if vplane is not None:
        tinfo["virtual_plane"] = vplane
    np.save(f"{scdir_t}/scale010.npy",
            {"scene": {oid: tinfo}, "scene_id": f"{oid}/tabletop/scale010",
             "task": {"type": "force_closure", "obj_name": oid}})
    return {"oid": oid, "n_coacd": len(names), "obb": obb,
            "has_virtual_plane": vplane is not None}


# ----------------------------------------------------------------------------- stage: synth
def run_dexrun(op, exp_name, template, gpu, scene, n_cfg, extra=None):
    cfg_path = f"{ASSET_ROOT}/scene_cfg/**/{scene}/*.npy"
    cmd = ["dexrun", f"op={op}", "hand=shadow", f"exp_name={exp_name}"]
    if op == "init":
        cmd += [f"tmpl_name={template}", f"init_gpu=[{gpu}]",
                f"op.object.cfg_path={cfg_path}", f"op.object.n_cfg={n_cfg}"]
    if extra:
        cmd += extra
    env = dict(os.environ, MUJOCO_GL="egl")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


# ----------------------------------------------------------------------------- stage: export (OCIR format)
def export_object(save_dir, oid, template, out_root):
    import mujoco, imageio
    from dexonomy.sim import MuJoCo_OptEnv, MuJoCo_OptCfg, HandCfg
    from dexonomy.util.file_util import load_scene_cfg
    succ = sorted(glob.glob(f"{save_dir}/succ_grasp/{template}/{oid}/**/*.npy", recursive=True))
    if not succ:
        return {"oid": oid, "template": template, "n_validated": 0}
    outdir = os.path.join(out_root, oid, template); os.makedirs(outdir, exist_ok=True)

    # all validated grasp poses
    G = [np.load(f, allow_pickle=True).item() for f in succ]
    np.savez(os.path.join(outdir, "all_grasps.npz"),
             grasp_qpos=np.stack([g["grasp_qpos"][0] for g in G]),
             squeeze_qpos=np.stack([g["squeeze_qpos"][0] for g in G]),
             joint_order_full=np.array(["base_x","base_y","base_z","base_qw","base_qx","base_qy","base_qz"]+JOINTS))

    d = G[0]                                            # representative grasp
    keyq = np.concatenate([d["pregrasp_qpos"], d["grasp_qpos"], d["squeeze_qpos"]], 0)
    kseg = [0]*len(d["pregrasp_qpos"]) + [1] + [2]
    K = 12; dq = []; ds = []
    for i in range(len(keyq)-1):
        for t in range(K):
            a = t/K; dq.append(keyq[i]*(1-a)+keyq[i+1]*a); ds.append(int(kseg[i]))
    dq.append(keyq[-1]); ds.append(int(kseg[-1]))
    dq = np.asarray(dq, np.float64); ds = np.asarray(ds, np.int8); T = len(dq)
    hand = dq[:, 7:29].copy()
    np.savez(os.path.join(outdir, "finger_track.npz"),
             desired_position_rad=hand, driven_target_rad=hand, actual_position_rad=hand,
             joint_order=np.asarray(JOINTS, dtype="<U19"), segment=ds)
    pos = np.tile(np.asarray(d["obj_pose"][:3], np.float64), (T, 1))
    ori = np.tile(np.asarray(d["obj_pose"][3:7], np.float64), (T, 1))
    np.savez(os.path.join(outdir, "object_track.npz"),
             position_world=pos, orientation_world_wxyz=ori,
             reference_position_world=pos, reference_orientation_world_wxyz=ori,
             segment=ds, carry_mask=(ds >= 1))

    # render video + screenshot
    e = MuJoCo_OptEnv(hand_cfg=HandCfg(xml_path=HAND_XML, freejoint=True),
                      scene_cfg=load_scene_cfg(d["scene_path"]), sim_cfg=MuJoCo_OptCfg())
    e._model.vis.global_.offwidth = 600; e._model.vis.global_.offheight = 600
    e._model.vis.headlight.ambient[:] = 0.5; e._model.vis.headlight.diffuse[:] = 0.7
    center = np.asarray(d["ext_center"], np.float64); frames = []
    for q in dq:
        e.reset_qpos(q.astype(np.float32)); mujoco.mj_forward(e._model, e._data)
        r = mujoco.Renderer(e._model, 600, 600); cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
        cam.lookat[:] = center; cam.distance = 0.34; cam.azimuth = 130; cam.elevation = -18
        r.update_scene(e._data, cam); frames.append(r.render()); r.close()
    gi = int(np.where(ds == 1)[0][0])
    imageio.imwrite(os.path.join(outdir, "screenshot.png"), frames[gi])
    imageio.mimsave(os.path.join(outdir, "video.mp4"), frames, fps=20, quality=8, macro_block_size=1)

    rep = {"ok": True, "task": "dexonomy_grasp_pose", "source": "Dexonomy", "hand": "shadow",
           "grasp_type": template, "object_id": oid, "scene_path": d["scene_path"],
           "obj_scale": np.asarray(d["obj_scale"]).tolist(),
           "obj_pose_wxyz": np.asarray(d["obj_pose"]).tolist(),
           "num_hand_joints": 22, "joint_order": JOINTS,
           "n_validated_grasps": len(G), "representative_frames": T, "fps": 20,
           "phases": {"pregrasp": int((ds == 0).sum()), "grasp": int((ds == 1).sum()),
                      "squeeze": int((ds == 2).sum())},
           "num_contacts": int(len(d.get("ho_c", {}).get("pos", []))),
           "kinematic_playback": True,
           "note": "finger_track desired==driven==actual (kinematic). object fixed in canonical frame."}
    json.dump(rep, open(os.path.join(outdir, "report.json"), "w"), indent=1)
    return {"oid": oid, "template": template, "n_validated": len(G),
            "num_contacts": rep["num_contacts"]}


# ----------------------------------------------------------------------------- main
def parse_objs(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b)+1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon-base", required=True)
    ap.add_argument("--objects", default="0-19")
    ap.add_argument("--templates", nargs="+", default=["1_Large_Diameter"])
    ap.add_argument("--scene", choices=["floating", "tabletop"], default="tabletop")
    ap.add_argument("--exp-name", default="recon_pp")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--oid-prefix", default="egodex_pp")
    ap.add_argument("--stages", nargs="+", default=["assets", "synth", "export"])
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    objs = parse_objs(args.objects)
    oids = [f"{args.oid_prefix}_{i}" for i in objs]
    save_dir = f"output/{args.exp_name}_shadow"
    out_root = args.out_root or f"{save_dir}/ocir_style"
    summary = {"exp": args.exp_name, "scene": args.scene, "templates": args.templates,
               "objects": {}, "assets": {}}

    if "assets" in args.stages:
        print("=== STAGE assets ===", flush=True)
        for i, oid in zip(objs, oids):
            rd = os.path.join(args.recon_base, str(i))
            try:
                info = build_asset(rd, oid)
                summary["assets"][oid] = info
                print(f"  [{oid}] coacd={info['n_coacd']} vplane={info['has_virtual_plane']}", flush=True)
            except Exception as ex:
                summary["assets"][oid] = {"error": str(ex)}
                print(f"  [{oid}] ERROR {ex}", flush=True)

    if "synth" in args.stages:
        print("=== STAGE synth ===", flush=True)
        for tmpl in args.templates:
            print(f"--- template {tmpl} ---", flush=True)
            for op in ["init", "grasp", "eval"]:
                rc, log = run_dexrun(op, args.exp_name, tmpl, args.gpu, args.scene,
                                     n_cfg=max(len(objs), 20))
                tail = "\n".join(l for l in log.splitlines()
                                 if not any(s in l for s in
                                            ["cuDeviceGetUuid", "entry point",
                                             "not supported in the installed"]))[-400:]
                print(f"  op={op} rc={rc}\n{tail}", flush=True)

    if "export" in args.stages:
        print("=== STAGE export ===", flush=True)
        os.makedirs(out_root, exist_ok=True)
        for oid in oids:
            summary["objects"].setdefault(oid, {})
            for tmpl in args.templates:
                res = export_object(save_dir, oid, tmpl, out_root)
                summary["objects"][oid][tmpl] = res
                print(f"  [{oid}/{tmpl}] validated={res['n_validated']}", flush=True)

    # roll-up
    tot = {t: 0 for t in args.templates}
    per_obj_any = 0
    for oid in oids:
        got = False
        for t in args.templates:
            n = summary["objects"].get(oid, {}).get(t, {}).get("n_validated", 0)
            tot[t] += n; got = got or n > 0
        per_obj_any += int(got)
    summary["totals"] = {"per_template": tot,
                         "objects_with_grasp": per_obj_any, "n_objects": len(oids)}
    json.dump(summary, open(f"{save_dir}/summary.json", "w"), indent=1, default=str)
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary["totals"], indent=1), flush=True)
    print(f"summary -> {save_dir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
