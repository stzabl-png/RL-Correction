"""Random-search thumb joint angles for each template posture.
Score: pad opposition (thumb vs index/middle), pad gap near target, no self-penetration.
Run: MUJOCO_GL=egl /home/lyh/anaconda3/envs/dexonomy/bin/python tools/sharpa_wave/search_thumb.py
"""

from pathlib import Path

import mujoco
import numpy as np
import yaml

OUT = Path("/home/lyh/Project/Dexonomy/assets/hand/sharpa_wave")
rng = np.random.default_rng(0)

TIP_KP = {"right_thumb_DP": 1, "right_index_DP": 0, "right_middle_DP": 1,
          "right_ring_DP": 0, "right_pinky_DP": 1}
THUMB = ["right_thumb_CMC_FE", "right_thumb_CMC_AA", "right_thumb_MCP_FE",
         "right_thumb_MCP_AA", "right_thumb_IP"]

# fingers fixed per template: {finger_joint_suffix_values}, target gap (m), opposition target
CASES = {
    "fingertip_small": dict(fingers=[0.8, 0.0, 0.9, 0.7], gap=0.030, ref=["index", "middle"]),
    "fingertip_mid": dict(fingers=[0.6, 0.0, 0.7, 0.5], gap=0.045, ref=["index", "middle"]),
    "3_Medium_Wrap": dict(fingers=[1.0, 0.0, 1.1, 0.6], gap=0.045, ref=["middle", "ring"]),
    "1_Large_Diameter": dict(fingers=[0.75, 0.0, 0.75, 0.4], gap=0.065, ref=["middle", "ring"]),
}

model = mujoco.MjModel.from_xml_path(str(OUT / "right.xml"))
data = mujoco.MjData(model)
kp = yaml.safe_load((OUT / "keypoint.yaml").read_text())
jid = {model.joint(i).name: i for i in range(model.njnt)}


def set_qpos(fingers, thumb_q, pinky_cmc=0.05):
    data.qpos[:] = 0
    for f in ["index", "middle", "ring", "pinky"]:
        for suf, v in zip(["MCP_FE", "MCP_AA", "PIP", "DIP"], fingers):
            data.qpos[model.jnt_qposadr[jid[f"right_{f}_{suf}"]]] = v
    data.qpos[model.jnt_qposadr[jid["right_pinky_CMC"]]] = pinky_cmc
    for name, v in zip(THUMB, thumb_q):
        data.qpos[model.jnt_qposadr[jid[name]]] = v
    mujoco.mj_forward(model, data)


def kp_world(body, idx):
    bid = model.body(body).id
    Rw = data.xmat[bid].reshape(3, 3)
    return data.xpos[bid] + Rw @ np.array(kp[body][idx][:3]), Rw @ np.array(kp[body][idx][3:])


for tname, cfg in CASES.items():
    lo = np.array([model.jnt_range[jid[n]][0] for n in THUMB])
    hi = np.array([model.jnt_range[jid[n]][1] for n in THUMB])
    best, best_q = 1e9, None
    for it in range(4000):
        tq = rng.uniform(lo, hi)
        set_qpos(cfg["fingers"], tq)
        pen = sum(-c.dist for c in data.contact[:data.ncon] if c.dist < -5e-4)
        pth, nth = kp_world("right_thumb_DP", 1)
        score = 200.0 * pen
        for f in cfg["ref"]:
            b = f"right_{f}_DP"
            p, n = kp_world(b, TIP_KP[b])
            gap = np.linalg.norm(p - pth)
            score += np.dot(n, nth) + 6.0 * abs(gap - cfg["gap"])
        if score < best:
            best, best_q = score, tq
    set_qpos(cfg["fingers"], best_q)
    pth, nth = kp_world("right_thumb_DP", 1)
    pen = [(model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name,
            round(c.dist * 1000, 1)) for c in data.contact[:data.ncon] if c.dist < -5e-4]
    print(f"\n=== {tname} === thumb={np.round(best_q, 2).tolist()} score={best:.2f} pen={pen or 'none'}")
    for f in ["index", "middle", "ring", "pinky"]:
        p, n = kp_world(f"right_{f}_DP", TIP_KP[f"right_{f}_DP"])
        print(f"   thumb<->{f:6s} gap={np.linalg.norm(p - pth)*100:5.1f}cm dot={np.dot(n, nth):+.2f}")
