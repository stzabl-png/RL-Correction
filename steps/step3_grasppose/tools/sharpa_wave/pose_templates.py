"""Pose candidate grasp-template qpos for sharpa_wave, render them, and print diagnostics
(thumb-finger pad gaps, normal opposition, self-penetration).
Run: MUJOCO_GL=egl /home/lyh/anaconda3/envs/dexonomy/bin/python tools/sharpa_wave/pose_templates.py <outdir>
"""

import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml

OUT = Path(__file__).resolve().parents[2] / "assets/hand/sharpa_wave"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")

JOINTS = [
    "right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE", "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE", "right_index_MCP_AA", "right_index_PIP", "right_index_DIP",
    "right_middle_MCP_FE", "right_middle_MCP_AA", "right_middle_PIP", "right_middle_DIP",
    "right_ring_MCP_FE", "right_ring_MCP_AA", "right_ring_PIP", "right_ring_DIP",
    "right_pinky_CMC", "right_pinky_MCP_FE", "right_pinky_MCP_AA", "right_pinky_PIP",
    "right_pinky_DIP",
]

# distal pad keypoint per finger (clean pad normals; ocir sphere ordering is inconsistent)
TIP_KP = {"right_thumb_DP": 1, "right_index_DP": 0, "right_middle_DP": 1,
          "right_ring_DP": 0, "right_pinky_DP": 1}


def make_qpos(thumb, fingers, pinky_cmc=0.05):
    """thumb=[CMC_FE,CMC_AA,MCP_FE,MCP_AA,IP]; fingers={name:[MCP_FE,AA,PIP,DIP]}"""
    q = dict(zip(JOINTS, np.zeros(len(JOINTS))))
    for n, v in zip(["CMC_FE", "CMC_AA", "MCP_FE", "MCP_AA", "IP"], thumb):
        q[f"right_thumb_{n}"] = v
    for fname, vals in fingers.items():
        for n, v in zip(["MCP_FE", "MCP_AA", "PIP", "DIP"], vals):
            q[f"right_{fname}_{n}"] = v
    q["right_pinky_CMC"] = pinky_cmc
    return q


TEMPLATES = {
    "1_Large_Diameter": make_qpos(
        thumb=[1.68, -0.19, -0.35, -0.01, 0.0],
        fingers={f: [0.75, 0.0, 0.75, 0.4] for f in ["index", "middle", "ring", "pinky"]}),
    "3_Medium_Wrap": make_qpos(
        thumb=[1.36, -0.32, -0.33, -0.28, 0.07],
        fingers={f: [1.0, 0.0, 1.1, 0.6] for f in ["index", "middle", "ring", "pinky"]}),
    "fingertip_mid": make_qpos(
        thumb=[1.66, 0.15, -0.51, 0.08, 0.11],
        fingers={f: [0.6, 0.0, 0.7, 0.5] for f in ["index", "middle", "ring", "pinky"]}),
    "fingertip_small": make_qpos(
        thumb=[1.05, 0.13, -0.52, -0.22, 0.01],
        fingers={f: [0.8, 0.0, 0.9, 0.7] for f in ["index", "middle", "ring", "pinky"]}),
}

spec = mujoco.MjSpec.from_file(str(OUT / "right.xml"))
spec.visual.global_.offwidth = 1280
spec.visual.global_.offheight = 1280
spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]
model = spec.compile()
data = mujoco.MjData(model)
kp = yaml.safe_load((OUT / "keypoint.yaml").read_text())
renderer = mujoco.Renderer(model, height=960, width=960)


def kp_world(body, idx):
    bid = model.body(body).id
    Rw = data.xmat[bid].reshape(3, 3)
    p = data.xpos[bid] + Rw @ np.array(kp[body][idx][:3])
    n = Rw @ np.array(kp[body][idx][3:])
    return p, n


for tname, qdict in TEMPLATES.items():
    data.qpos[:] = 0
    for jname, v in qdict.items():
        jid = model.joint(jname).id
        lo, hi = model.jnt_range[jid]
        data.qpos[model.jnt_qposadr[jid]] = np.clip(v, lo, hi)
    mujoco.mj_forward(model, data)

    pen = [(model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name,
            c.dist) for c in data.contact[:data.ncon] if c.dist < -5e-4]
    pth, nth = kp_world("right_thumb_DP", TIP_KP["right_thumb_DP"])
    print(f"\n=== {tname} ===  self-pen(>0.5mm): {pen if pen else 'none'}")
    for f in ["index", "middle", "ring", "pinky"]:
        b = f"right_{f}_DP"
        p, n = kp_world(b, TIP_KP[b])
        gap = np.linalg.norm(p - pth)
        print(f"   thumb<->{f:6s} pad gap={gap*100:5.1f}cm  normal_dot={np.dot(n, nth):+.2f}")

    for vname, (azi, elev) in {"palmside": (180, -20), "thumbside": (250, -25)}.items():
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.03, 0, 0.1]
        cam.distance = 0.4
        cam.azimuth, cam.elevation = azi, elev
        renderer.update_scene(data, camera=cam)
        scene = renderer.scene
        for body, idx in TIP_KP.items():
            p, n = kp_world(body, idx)
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                np.zeros(9), np.array([1, 0, 0, 1], dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.0015, p, p + 0.02 * n)
            scene.ngeom += 1
        imageio.imwrite(SCRATCH / f"tmpl_{tname}_{vname}.png", renderer.render())

print("\nqpos vectors (Dexonomy joint order):")
for tname, qdict in TEMPLATES.items():
    q = [round(float(qdict[j]), 3) for j in JOINTS]
    print(f"{tname}: {q}")
