"""在 MuJoCo GUI 里打开一个 GraspPose —— 停在抓握姿态, 默认显示接触点/接触力。

为什么需要它: 离线渲染 (`render_contact_vs_grasp.py`) 是单一视角的静态图, 手落在物体
背面、或者透过空心物体的开口看到后方的手, 都会被误看成穿模 (2026-08-15 因此来回误判
三次)。GUI 里能自由转视角 + 看 MuJoCo 自己算的接触点, 是最终裁判。

用法:
    MUJOCO_GL 必须**不是** egl (那是无头渲染, GUI 起不来):
    env -u MUJOCO_GL DISPLAY=:1 python tools/show_grasp_gui.py <grasp.npy> [hand.xml]

GUI 里:
    左侧 Rendering 面板  勾 Contact Point / Contact Force  看 MuJoCo 判定的接触
                        勾 Convex Hull                     看真正参与碰撞的几何
    姿态被逐帧锁死(不跑物理), 手不会掉下去。
"""
import os
import sys

import mujoco
import mujoco.viewer
import numpy as np

DEXO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DEXO)
os.chdir(DEXO)
from dexonomy.sim import HandCfg, MuJoCo_OptCfg, MuJoCo_OptEnv  # noqa: E402

NPY = sys.argv[1] if len(sys.argv) > 1 else \
    "output/v2up_cup_sharpa_wave_v2_left/sub/grasp_data/20_3_grasp.npy"
HAND = sys.argv[2] if len(sys.argv) > 2 else "assets/hand/sharpa_wave_v2_left/left.xml"

d = np.load(NPY, allow_pickle=True).item()
sp = d["scene_path"]
sc = np.load(sp, allow_pickle=True).item()
base = os.path.dirname(sp)
for _k, _v in sc["scene"].items():          # scene_cfg 里存的是相对路径
    for _kk in ("file_path", "xml_path", "urdf_path", "info_path"):
        if _kk in _v:
            _v[_kk] = os.path.normpath(os.path.join(base, _v[_kk]))

env = MuJoCo_OptEnv(hand_cfg=HandCfg(xml_path=HAND, freejoint=True),
                    scene_cfg=sc, sim_cfg=MuJoCo_OptCfg())
m, dat = env._model, env._data
q = np.asarray(d["grasp_qpos"], float).reshape(-1)
dat.qpos[:len(q)] = q
mujoco.mj_forward(m, dat)


def bname(g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or ""


print(f"\n=== {os.path.basename(NPY)}  hand={os.path.basename(HAND)} ===", flush=True)
print(f"MuJoCo 检出接触对 {dat.ncon} 个:", flush=True)
for i in range(dat.ncon):
    c = dat.contact[i]
    print(f"   dist = {c.dist * 1000:+.2f} mm   {bname(c.geom1)} ↔ {bname(c.geom2)}"
          f"   ({'★穿透' if c.dist < 0 else '间隙'})", flush=True)
print("\nGUI 已打开: 拖拽转视角; 左侧 Rendering 勾 Contact Point 看接触点。\n", flush=True)

with mujoco.viewer.launch_passive(m, dat) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    while viewer.is_running():
        dat.qpos[:len(q)] = q               # 锁死姿态, 不让物理把手推开
        dat.qvel[:] = 0
        mujoco.mj_forward(m, dat)
        viewer.sync()
