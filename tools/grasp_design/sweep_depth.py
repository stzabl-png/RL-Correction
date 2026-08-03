"""合拢深度扫描: 抓深一点手指戳桌, 抓浅一点夹不住. 找那条线在哪."""
import contextlib, io, sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, "/home/lyh/Project/RL_Correction")
from rl_rebuild.correction import clips, place_camera as PC, frames as F
from rl_rebuild.correction.ref_builders.replay_grasp import (
    _affordance_target, grasp_center_local, hand_lowest_world,
    GENERIC_OPEN, GENERIC_CLOSED)
from rl_rebuild.correction.kinematics import Urdf, quat_to_R, _T

CLIP, TZ, HOVER = "Grasp2", 0.85, 0.03
cfg = SimpleNamespace(clip_name=CLIP, target_hz=20.0, table_top_z=TZ, clearance=0.015,
                      freeze_wrist=False, hover_gap=None, anchor_mode="camera")
with contextlib.redirect_stdout(io.StringIO()):
    du = clips.load_data_unit(cfg)
r, e, hand = du.ref, clips.clip_entry(CLIP), clips.interact_hand(CLIP)
gs = min(int(r.interaction_seg[0]), r.L - 1)
wq = r.track_wrist[:, 3:7].astype(np.float64)
obj = du.object_init_pose[:3].astype(np.float64)
oq = r.track_object[0, 3:7].astype(np.float64)
aff_w = PC.affordance_world(obj, oq, _affordance_target(e["affordance"]))
q_gs = wq[gs]

# 物体几何 (平放后的世界包围盒)
verts = F.load_obj_verts(e["mesh"])
vr = F.rot_apply(np.broadcast_to(oq, (len(verts), 4)), verts) + obj
print(f"物体世界包围盒 z [{vr[:,2].min():.4f}, {vr[:,2].max():.4f}] "
      f"(离桌 {(vr[:,2].min()-TZ)*100:+.2f} ~ {(vr[:,2].max()-TZ)*100:+.2f}cm)  "
      f"xy 半径 {np.linalg.norm(vr[:,:2]-obj[:2],axis=1).max()*100:.2f}cm")
print(f"affordance 世界 z {aff_w[2]:.4f} (离桌 {(aff_w[2]-TZ)*100:.2f}cm)\n")

u = Urdf()
JN = [n.replace("right_", f"{hand}_") for n in
      ["right_thumb_CMC_FE","right_thumb_CMC_AA","right_thumb_MCP_FE","right_thumb_MCP_AA",
       "right_thumb_IP","right_index_MCP_FE","right_index_MCP_AA","right_index_PIP",
       "right_index_DIP","right_middle_MCP_FE","right_middle_MCP_AA","right_middle_PIP",
       "right_middle_DIP","right_ring_MCP_FE","right_ring_MCP_AA","right_ring_PIP",
       "right_ring_DIP","right_pinky_CMC","right_pinky_MCP_FE","right_pinky_MCP_AA",
       "right_pinky_PIP","right_pinky_DIP"]]
FING = ("thumb", "index", "middle", "ring", "pinky")


def tips(wp, qvec):
    q = {n: float(v) for n, v in zip(JN, qvec)}
    T = _T(quat_to_R(q_gs), wp)
    return np.stack([u.link_pose(f"{hand}_{f}_elastomer", q, T, f"{hand}_hand_C_MC")[:3, 3]
                     for f in FING])


print(f"{'合拢中心高于aff':>15}{'手最低−桌':>12}{'指尖最低−桌':>13}{'指尖在物体外沿?':>16}  各指尖离桌(cm)")
for d in (0.0, 0.005, 0.010, 0.015, 0.020, 0.025):
    # 腕位: 让合拢终态的 grasp center 落在 aff + d
    gc = grasp_center_local(hand, GENERIC_CLOSED)
    wp = aff_w + np.array([0, 0, d]) - quat_to_R(q_gs) @ gc
    lo_c = hand_lowest_world(wp, q_gs, hand, qvec=GENERIC_CLOSED)
    tp = tips(wp, GENERIC_CLOSED)
    rad = np.linalg.norm(tp[:, :2] - obj[:2], axis=1)
    n_out = int((rad > 0.0375).sum())
    print(f"{d*100:>13.1f}cm{(lo_c-TZ)*100:>11.2f}cm{(tp[:,2].min()-TZ)*100:>12.2f}cm"
          f"{f'{n_out}/5':>16}  {np.round((tp[:,2]-TZ)*100,1).tolist()}")

print(f"\n各指尖水平半径 (物体半径 3.75cm):")
gc = grasp_center_local(hand, GENERIC_CLOSED)
wp0 = aff_w - quat_to_R(q_gs) @ gc
tp0 = tips(wp0, GENERIC_CLOSED)
for f, t in zip(FING, tp0):
    print(f"  {f:<7} 离物体中心 {np.linalg.norm(t[:2]-obj[:2])*100:5.2f}cm  离桌 {(t[2]-TZ)*100:+6.2f}cm")
