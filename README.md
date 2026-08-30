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

## Where the code lives

This default branch (`main`) holds only the overview above. **The implementation lives on
the per-stage branches** — pipeline step 4 (the RL correction itself) is on
`Step4_RL_Correction`, together with the training code, the design ledger and the criteria.

| Branch | Contents |
|---|---|
| `Step1_DataInput` | data ingestion |
| `Step2_NoisyRecon` | video → noisy HOI reconstruction (pipeline step 2) |
| `Step3_Dexonomy` | grasp pose generation (step 3b) |
| **`Step4_RL_Correction`** | **RL correction — training code, design ledger, success criteria, self-tests** |

```bash
git clone git@github.com:stzabl-png/RL-Correction.git
cd RL-Correction && git checkout Step4_RL_Correction
git lfs pull      # reference trajectories / USD assets are LFS-tracked
```

**If you want to reuse our training design**, start here:
[`docs/TRAINING_DESIGN_GUIDE.md`](https://github.com/stzabl-png/RL-Correction/blob/Step4_RL_Correction/docs/TRAINING_DESIGN_GUIDE.md)
on `Step4_RL_Correction` — branch map, code layout, the five core training ideas
(residual policy · four-gate phase machine · progressive RSI birth-point curriculum ·
reliability-weighted reference channels · world fingerprint), how to launch a run,
and the pitfalls that cost us whole training runs.
