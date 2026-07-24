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
