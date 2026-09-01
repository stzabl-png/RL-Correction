"""Dexonomy DELIVER (screw17_*) -> prior npz, **含 com_offset 平移** (make_prior.py 假设规范系无平移, 这批不成立).

用系统 python3 (numpy2 pickle):
  python3 tasks/Unscrew/17/A_Design/L1_Data/GraspPose/make_prior_screw17.py
输出 (物体输入系 = CAD 系, 瓶底/盖底在原点, canon_rot=单位):
  tasks/pregrasp/priors/Screw17_bottle_left.npz   (左手, 1_Large_Diameter__v4_7_14)
  tasks/pregrasp/priors/Screw17_cap_right.npz     (右手, 33_Inferior_Pincer__1_47; 盖 CAD 系 = 装配态盖底原点)
"""
import json, os, sys, numpy as np
D = "/home/lyh/Project/Dexonomy/output/DELIVER"
OUT = "tasks/pregrasp/priors"
PICK = {"Screw17_bottle_left": ("screw17_bottle_left", "1_Large_Diameter__v4_7_14_grasp.npy"),
        "Screw17_cap_right": ("screw17_cap_right", "33_Inferior_Pincer__1_47_grasp.npy")}
for name, (deliver, f) in PICK.items():
    rr = json.load(open(f"{D}/{deliver}/region_rank.json"))
    com = np.asarray(rr["canonical_frame"]["com_offset"], np.float64)
    rot = np.asarray(rr["canonical_frame"]["canonical_from_input_rot_wxyz"], np.float64)
    assert np.allclose(rot, [1, 0, 0, 0], atol=1e-6), rot
    g = np.load(f"{D}/{deliver}/grasp_data/{f}", allow_pickle=True).item()
    def to_input(rows):
        rows = np.atleast_2d(np.asarray(rows, np.float64)).copy()
        rows[:, :3] += com          # 规范系(质心原点) -> 输入系(CAD): 只差平移
        return rows
    hoc = g["ho_c"]
    cpos = np.asarray(hoc["pos"], np.float64) + com
    cnrm = np.asarray(hoc["normal"], np.float64)
    out = dict(grasp=to_input(g["grasp_qpos"])[0], squeeze=to_input(g["squeeze_qpos"])[0],
               pregrasp=to_input(g["pregrasp_qpos"]), contact_pos=cpos, contact_normal=cnrm,
               contact_centroid=cpos.mean(0), canon_rot=rot, com_offset=com,
               hand_name=np.array(g["hand_name"]), tmpl=np.array(g["tmpl_name"]),
               source=np.frombuffer(f"{D}/{deliver}/grasp_data/{f}".encode(), dtype=np.uint8))
    np.savez(f"{OUT}/{name}.npz", **out)
    print(f"[prior] {name}: hand={g['hand_name']} tmpl={g['tmpl_name']} com={np.round(com,4)} "
          f"grasp wrist(输入系)={np.round(out['grasp'][:3],4)} quat={np.round(out['grasp'][3:7],4)} "
          f"contacts {len(cpos)} centroid {np.round(out['contact_centroid'],4)} pregrasp {out['pregrasp'].shape}")
