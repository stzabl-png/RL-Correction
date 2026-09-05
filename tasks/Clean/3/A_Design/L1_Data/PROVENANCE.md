# Clean/3 数据来源 (未入库, 指针版)

- 源 take: `/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_auto/clean_tableware/3`
  (recon_world_v5, 300 帧, manifest 2026-09-05 全绿; 五条横评见 A_Design/DECISIONS.md §0)
- 物轨: `poseqa/rts_clean_tableware_3_object_{0,1}.npz` (+conf); 人手: `ref_qpos_{left,right}.npz` (静腕变体!)
- 接触: `contact/` (C1 路线: contact_fingers + expected_area + affordance 图 + grasp_prompt 给 Dexonomy)
- 网格: `objects/object_{0,1}/object_mesh_scaled_final.obj` (无纹理产物, 无 CAD 对照)
- 入库动作 (A1, 待做): 拷 take → `datasets/clean_tableware/3/`; scene_layout 按 env 桌高重算 (勿用上游桌高, unscrew17 教训); 强制核 manifest 存在性 (UCB venv 缺失静默 skip 的教训, 见源 README)
