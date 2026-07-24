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

## 本分支：Step 1 — Data Input

本分支不直接存放数据，数据统一托管在 HuggingFace：

**https://huggingface.co/datasets/UCBProject/RL_Correction/tree/main**

### 数据结构

```
Data/
└── EgoDex Part2/
    └── <sample_id>/
        ├── reconstruction/            # Step 2 产物：noisy reconstruction
        │   ├── human_demo.npz         # 人手 / 物体轨迹（MANO pose、6D pose）
        │   ├── affordance.npz         # affordance / contact prior
        │   ├── object.obj             # 物体 mesh
        │   └── object_coacd_parts.npz # CoACD 凸分解，用作仿真碰撞体
        └── grasp_pose/                # Step 3 产物：优化后的抓取轨迹
            ├── trajectory.json
            ├── trajectory.npz
            └── isaac_sim/             # Isaac Sim 验证结果
                ├── scene.usd
                ├── finger_track.npz
                ├── object_track.npz
                ├── report.json
                ├── screenshot.png
                └── video.mp4
```

### 下载

```bash
pip install -U "huggingface_hub[cli]"

# 全量
hf download UCBProject/RL_Correction --repo-type dataset --local-dir ./data

# 只取单个样本
hf download UCBProject/RL_Correction --repo-type dataset \
  --include "Data/EgoDex Part2/0/*" --local-dir ./data
```

Python：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="UCBProject/RL_Correction",
    repo_type="dataset",
    local_dir="./data",
)
```

上传：`hf upload UCBProject/RL_Correction <local_path> <path_in_repo> --repo-type dataset`
