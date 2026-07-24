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

## 本分支：Step 4 — RL Correction

在优化先验之上做 object-conditioned 的残差修正（keyframe residual correction）：

```
π_θ(τ_raw, τ_opt, O, C) → Δτ_RL
τ_corr = τ_opt + Δτ_RL
```

策略条件包括 hand-object state、contact / affordance prior、phase（pre-grasp / contact / grasp / lift / place）。

轨迹质量奖励：

```
R = + contact + stability + progress + success
    − penetration − collision − joint-limit − jerk − deviation
```

其中 fidelity 项约束修正后的轨迹不偏离原视频意图。修正结果经 Isaac Sim 验证（Step 5）后进入 `D_verified`，再去重得到 `D_high-quality`。
