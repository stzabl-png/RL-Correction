# RL-Correction

**Optimization-Guided Object-Aware RL Trajectory Correction**

From noisy egocentric video reconstruction to high-quality sim-verified robot trajectories.

![Pipeline overview](RL_Correction.png)

## Pipeline

1. **Egocentric Human Video** — input
2. **Noisy Reconstruction** — MANO pose, partial object point cloud, object 6D pose, hand-object trajectory, affordance / contact prior → `τ_raw`
3. **a. Object Geometry** (mesh, point cloud, SDF, normals, scale/shape feature → `O`) · **b. Optimization Prior** (contact consistency, non-penetration, smoothness, joint limits, grasp/IK feasibility → `τ_opt`)
4. **Object-Conditioned RL Correction** — residual correction on top of the optimization prior: `τ_corr = τ_opt + Δτ_RL`
5. **Isaac Sim Verification** — real contact, non-penetration, grasp stability, collision-free motion, task completion, fidelity to video intent
6. **Sim-Verified Data** — `D_verified = {τ_corr | s_i > δ}`
7. **Data Curation / Deduplication** — geometry, contact pattern, grasp pose, trajectory similarity, outcome diversity
8. **High-Quality Sim-Verified Data** — `D_high-quality`, for robot policy training and dataset construction

A trajectory quality reward (`+ contact + stability + progress + success − penetration − collision − joint-limit − jerk − deviation`) closes the loop, with a fidelity term keeping the corrected trajectory close to the original video intent.

---

## 本分支：Step 2 — Noisy Reconstruction

从 egocentric 人手视频重建出（带噪声、物理上不一致的）`τ_raw`：

- MANO 手部姿态
- 物体部分点云与 6D pose
- hand-object 轨迹
- affordance / contact prior

同时产出 Step 3a 需要的 **Object Geometry** `O`：mesh、完整点云、SDF、表面法向、scale / shape feature。

输出对应 HuggingFace 数据集中的 `Data/<split>/<sample_id>/reconstruction/`（见 `Step1_DataInput` 分支）。

### 本分支代码结构（并入的 336 个文件）

```
ego_pipeline/          重建 + retarget 主流程
  repo_paths.py/.sh    ★ 所有路径的统一出口（RR_ROOT 自动推导，clone 即用）
  reconstruct.sh       重建入口（8 步：vipe→sam3_hands→sam2_object→hawor→sam3d→sam3d_scale→fp_pose→fuse）
  retarget.sh          retarget 入口
  label.sh             物体 mask 人工标注
  bridge/retarget.py   recon 产物 → replay_world.npz + object.usd
  phase/               抓取接触阶段（grasp phase）子系统
    manual.py          人工标注 provider（当前流程使用）
    detect.py          recon-only 自动接触检测（2D 手/物 mask 邻接，分左右手）
    comotion_gate.py   手-物主动协同运动闸：判「真抓 vs 靠近/搁着」，带物体实例身份
    auto.py            自动 provider（消费 contact_auto.json）
  Reconstruction/      重建各 stage
  Retargeting/         retarget + SharpaWave 手资产（vendored）
tools/                 标注、attach phase、批处理、可视化
experimental/hoi_detr_v17a/   HOI-DETR v17A 自动交互分割（**未接入主流程**，见其 SETUP_AND_USAGE.md）
```

**当前流程的 Mask 与 Phase 仍为人工标注**；自动化（v17A / co-motion gate）已打包但未接入。

### 环境与依赖

- 上手先读 **`USAGE.md`**（三步全流程）。
- **`third_party/` 不入 git**（体积过大），需自行准备，见 **`SETUP.md`**。
- 所有路径走 `ego_pipeline/repo_paths.py`；clone 到任何位置无需配置即可运行，外部依赖用同名环境变量覆盖。
