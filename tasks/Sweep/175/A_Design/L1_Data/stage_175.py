"""take 175 入库 = "同一批实物, 只换动作" (2026-09-13, 用户: 用 r2; 台账 tasks/Sweep/175/A_Design/DECISIONS.md §1).

  网格/USD/碰撞缓存: 逐字复制 datasets/sweep408 的 r2 版 (ICP 证明 175 与 408 是同一把刷子/同一个簸箕: 残差 0.28/0.19cm)。
  物轨: 175 的 RTS 平滑轨 (raw 网格系) 用带尺度 ICP (raw175 -> 408 入库网格) 的旋转/平移配到 r2 网格系 (不带尺度: 物体在世界里
        变回 r2 真实大小), 再按 build_reference.load_track 的约定 (p_obj = t_rec + R_rec·com/s) 反算 t_rec 写回 rts npz。
  索引: 175 上游 object_0=簸箕/object_1=扫把, 入库后统一成 408 约定 object_0=扫把(右)/object_1=簸箕(左)。
  replay_world: 同 408 副本, 保持 raw 系原轨 (只做索引交换); scene_layout: 同 schema, 数值按 175 第 0 帧。
"""
import json, os, shutil, sys
import numpy as np, trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
UP = "/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_part4/sweep_dustpan/175"
SRC = os.path.join(ROOT, "datasets/sweep408"); DST = os.path.join(ROOT, "datasets/sweep175")
DEXO = "/home/lyh/Project/Dexonomy/assets/object/custom/processed_data"
ICP = json.load(open(os.path.join(ROOT, "scratchpad/icp_175_to_408staged.json")))
# 入库索引 -> (175 上游索引, ICP 键, Dexonomy r2 名, builder SPEC 的 s)
ROLE = {0: dict(up=1, icp="175_broom", dx="p4t408_broom_r2", s=1.30, name="broom"),
        1: dict(up=0, icp="175_pan_id", dx="p4t408_dustpan_r2", s=1.25, name="pan")}
if os.path.exists(DST): shutil.rmtree(DST)
os.makedirs(os.path.join(DST, "poseqa"))
for d in ("objects", "cache"): shutil.copytree(os.path.join(SRC, d), os.path.join(DST, d))
for cfg in ("objects/object_0/config.yaml", "objects/object_1/config.yaml", "cache/config.yaml"):
    p = os.path.join(DST, cfg)
    if os.path.exists(p): open(p, "w").write(open(p).read().replace("datasets/sweep408", "datasets/sweep175"))
for f in ("ref_qpos_left.npz", "ref_qpos_right.npz", "confidence_complete.json", "PROVENANCE.md", "world_summary.json", "world_fused.npz"):   # world_fused: env 相机锚定要 c2w
    shutil.copy(os.path.join(UP, f), DST)
for f in os.listdir(os.path.join(UP, "poseqa")):
    if f.endswith(".json"): shutil.copy(os.path.join(UP, "poseqa", f), os.path.join(DST, "poseqa"))
reg = {}
for oi, r in ROLE.items():
    z = np.load(os.path.join(UP, "poseqa", f"rts_sweep_dustpan_175_object_{r['up']}.npz"))
    s, Rp, tp = float(ICP[r["icp"]]["s"]), np.asarray(ICP[r["icp"]]["R"]), np.asarray(ICP[r["icp"]]["t"])
    com = np.asarray(json.load(open(f"{DEXO}/{r['dx']}/info/simplified.json"))["com_offset"], np.float64) / r["s"]
    out = {k: z[k] for k in z.files}
    for key in ("object_ob_in_world_smooth", "object_ob_in_world_filtered"):
        T = np.asarray(z[key], np.float64); Tn = T.copy()
        R175, t175 = T[:, :3, :3], T[:, :3, 3]
        R_r2 = R175 @ Rp.T                                   # r2 网格系在世界里的姿态
        p_r2 = np.einsum("nij,j->ni", R175, -Rp.T @ tp / s) + t175   # r2 网格原点 (=raw175 系里 ICP 的原点像) 的世界位置
        Tn[:, :3, :3] = R_r2; Tn[:, :3, 3] = p_r2 - np.einsum("nij,j->ni", R_r2, com)   # 让 builder 的 t+R·com 正好落回 p_r2
        out[key] = Tn.astype(T.dtype)
        if key.endswith("smooth"): reg[oi] = (R_r2, p_r2)
    out["registration_note"] = np.asarray(f"raw175 obj{r['up']} -> 408 r2 入库网格 (ICP s={s:.4f}, 旋转 {np.degrees(np.arccos(np.clip((np.trace(Rp)-1)/2,-1,1))):.2f}°); "
                                          f"t 已按 build_reference.load_track (p=t+R·com/{r['s']}) 反算; 索引: 入库 object_{oi} = 上游 object_{r['up']}")
    np.savez(os.path.join(DST, "poseqa", f"rts_sweep_dustpan_175_object_{oi}.npz"), **out)
    # 核验: 第 0 帧, r2 入库网格按 (R_r2,p_r2) 摆到世界 vs 175 raw 网格按原轨摆到世界, 最近点残差应 ≈ ICP 残差
    m_r2 = trimesh.load(os.path.join(DST, "objects", f"object_{oi}", "object_mesh_scaled_final.obj"), process=False)
    m_raw = trimesh.load(os.path.join(UP, "objects", f"object_{r['up']}", "object_mesh_scaled_final.obj"), process=False)
    rng = np.random.RandomState(0)
    A = np.asarray(m_r2.vertices)[rng.choice(len(m_r2.vertices), 8000)] @ reg[oi][0][0].T + reg[oi][1][0]
    T0 = np.asarray(z["object_ob_in_world_smooth"][0], np.float64)
    B = np.asarray(m_raw.vertices)[rng.choice(len(m_raw.vertices), 8000)] @ T0[:3, :3].T + T0[:3, 3]
    d1, _ = cKDTree(B).query(A); d2, _ = cKDTree(A).query(B)
    print(f"[stage] object_{oi} ({r['name']}): ICP 残差 {ICP[r['icp']]['cost']*100:.2f}cm 倍率 {s:.3f} | 第0帧世界系核验: r2 网格 vs raw 网格 最近点 {np.mean(d1)*100:.2f}/{np.mean(d2)*100:.2f}cm "
          f"(尺度差本身会贡献 ~{(abs(s-1)*0.5*np.linalg.norm(m_raw.bounding_box.extents)/2)*100:.1f}cm) | r2 原点世界 {np.round(reg[oi][1][0],4).tolist()}")
# replay_world: 索引交换到 408 约定, 保持 raw 系
w = np.load(os.path.join(UP, "replay_world.npz"), allow_pickle=True); rw = {k: w[k] for k in w.files}
perm = [ROLE[0]["up"], ROLE[1]["up"]]
for k in ("obj_pose_all", "obj_verts_local", "obj_valid_all"): rw[k] = np.asarray(w[k])[perm]
rw["obj_pose"] = np.asarray(w["obj_pose_all"])[ROLE[0]["up"]]
rw["object_ids"] = np.asarray(["object_0", "object_1"]); rw["index_note"] = np.asarray("入库 object_0=扫把(上游 object_1), object_1=簸箕(上游 object_0); 位姿仍是 raw 网格系 (同 408 副本)")
np.savez(os.path.join(DST, "replay_world.npz"), **rw)
# scene_layout: 沿用 408 schema, 数值换 175 第 0 帧 (r2 系, 加同一 shift 仅供目检; 消费者只吃相对偏移与 quat)
lay = json.load(open(os.path.join(SRC, "scene_layout.json")))
# 右腕锚定: 175 的右腕 f0 平移到 408 layout 的右腕位 (机器人同一站姿), 不是抄 408 的 shift (抄数会让物体落在 IK 不可达处, 先验闸失败)
shift = np.asarray(lay["objects"]["object_0"]["wrist_pos"]) - np.asarray(np.load(os.path.join(UP, "ref_qpos_right.npz"), allow_pickle=True)["wrist_pos"][0])
lay["schema_version"] = "sweep175_held_scene_v1"
lay["scene_table_source"] = "175 开头双手已各握一物 (同 408); 网格=408 r2, 物轨 ICP 配到 r2 系; objects.pos = 第0帧 r2 系位姿 + 右腕锚定 shift"
lay["registration"]["shift_m"] = shift.tolist(); lay["registration"]["shift_source"] = "stage_175: 408 layout 右腕 − 175 raw 右腕 f0"
for oi, r in ROLE.items():
    o = lay["objects"][f"object_{oi}"]
    o["pos"] = (reg[oi][1][0] + shift).tolist(); o["quat_wxyz"] = Rot.from_matrix(reg[oi][0][0]).as_quat()[[3, 0, 1, 2]].tolist()
    side = "right" if oi == 0 else "left"
    q = np.load(os.path.join(UP, f"ref_qpos_{side}.npz"), allow_pickle=True)
    o["wrist_pos"] = (np.asarray(q["wrist_pos"][0]) + shift).tolist(); o["wrist_quat_wxyz"] = np.asarray(q["wrist_quat_wxyz"][0]).tolist()
json.dump(lay, open(os.path.join(DST, "scene_layout_175_native.json"), "w"), ensure_ascii=False, indent=1)   # 175 自己的第 0 帧数值, 仅供目检
# 实际喂环境的 layout 直接沿用 408 的数: 它只用于基类先验闸 (规范姿 + yaw 自搜 + GraspPose IK) 与 aux 相对偏移, 焊死后按母带重摆;
# 用 175 原生数值时 yaw 自搜到 150° 后 IK 不可达 (闸失败), 同物同先验用 408 的数 110° 就过 (2026-09-14 00:30 UTC)
lay8 = json.load(open(os.path.join(SRC, "scene_layout.json"))); lay8["schema_version"] = "sweep175_held_scene_v1 (numbers = 408 layout)"
json.dump(lay8, open(os.path.join(DST, "scene_layout.json"), "w"), ensure_ascii=False, indent=1)
rel408 = np.asarray(json.load(open(os.path.join(SRC, "scene_layout.json")))["objects"]["object_1"]["pos"]) - np.asarray(json.load(open(os.path.join(SRC, "scene_layout.json")))["objects"]["object_0"]["pos"])
rel175 = reg[1][1][0] - reg[0][1][0]
print(f"[stage] 第0帧 簸箕-扫把 相对偏移: 408 {np.round(rel408*100,1).tolist()}cm | 175 {np.round(rel175*100,1).tolist()}cm")
print(f"[stage] → {DST}: " + ", ".join(sorted(os.listdir(DST))))
