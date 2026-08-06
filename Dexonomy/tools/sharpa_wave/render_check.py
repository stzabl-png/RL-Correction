"""Offscreen sanity renders for the generated sharpa_wave MJCF:
- neutral + curled pose from 3 views
- the 11 BODex contact keypoints drawn as arrows (pos + normal, body frame -> world)
Run: MUJOCO_GL=egl /home/lyh/anaconda3/envs/dexonomy/bin/python tools/sharpa_wave/render_check.py
"""

import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

OUT = Path("/home/lyh/Project/Dexonomy/assets/hand/sharpa_wave")
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp")

CONTACT_KEYPOINTS = [  # (body, keypoint index) — mirrors ocir SHARPA_CONTACT_POINTS
    ("right_pinky_DP", 1), ("right_pinky_PP", 0),
    ("right_ring_PP", 0), ("right_ring_DP", 1),
    ("right_middle_PP", 0), ("right_middle_DP", 1),
    ("right_index_PP", 0), ("right_index_DP", 1),
    ("right_thumb_PP", 0), ("right_thumb_DP", 1),
    ("right_hand_C_MC", 0),
]

spec = mujoco.MjSpec.from_file(str(OUT / "right.xml"))
spec.visual.global_.offwidth = 1280
spec.visual.global_.offheight = 1280
spec.visual.headlight.ambient = [0.4, 0.4, 0.4]
spec.visual.headlight.diffuse = [0.7, 0.7, 0.7]
model = spec.compile()
data = mujoco.MjData(model)
kp = yaml.safe_load((OUT / "keypoint.yaml").read_text())

curl = np.zeros(model.nq)
for i in range(model.njnt):
    name = model.joint(i).name
    if any(s in name for s in ["_FE", "PIP", "DIP", "_IP", "CMC"]) and "AA" not in name:
        lo, hi = model.jnt_range[i]
        curl[i] = lo + 0.6 * (hi - lo)

renderer = mujoco.Renderer(model, height=720, width=720)
views = {  # (azimuth, elevation)
    "az0": (0, -15),
    "az90": (90, -15),
    "az180": (180, -15),
    "az270": (270, -15),
}

def draw_keypoints(scene):
    for body, idx in CONTACT_KEYPOINTS:
        p_local = np.array(kp[body][idx][:3])
        n_local = np.array(kp[body][idx][3:])
        bid = model.body(body).id
        Rw = data.xmat[bid].reshape(3, 3)
        p = data.xpos[bid] + Rw @ p_local
        n = Rw @ n_local
        for scale, rgba in ((1.0, [1, 0, 0, 1]),):
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                                np.zeros(9), np.array(rgba, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.0015, p, p + 0.02 * n)
            scene.ngeom += 1

for pose_name, q in (("neutral", np.zeros(model.nq)), ("curled", curl)):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    for vname, (azi, elev) in views.items():
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0.09]
        cam.distance = 0.42
        cam.azimuth, cam.elevation = azi, elev
        renderer.update_scene(data, camera=cam)
        draw_keypoints(renderer.scene)
        img = renderer.render()
        import imageio.v2 as imageio
        f = SCRATCH / f"sharpa_{pose_name}_{vname}.png"
        imageio.imwrite(f, img)
        print("wrote", f)
