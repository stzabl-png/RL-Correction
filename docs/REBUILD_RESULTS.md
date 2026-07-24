# SHARPA RL rebuild — staged build on top of the working sharpa-rl-lab design

This is **our RL, rebuilt** as a clean fork of the validated `../sharpa-rl-lab` (so it inherits the
robust behavior: stable-grasp cache + torque control + adaptive gravity curriculum + **tactile contact
observations**, which we keep for later distillation). On top of that baseline we add, gated and tested
stage by stage: **(2)** hand/object point cloud + masks, **(3)** a world-model auxiliary loss (the policy
predicts object pose), **(4)** multi-object (multi-scale) generalization.

Package renamed `rl_isaaclab` -> `rl_rebuild` so it coexists with the original. Runs in the `sharpa`
conda env. Single RTX 4090, headless. All metrics evaluated at **FULL gravity (-9.81)**, 64 envs, 600
steps, deterministic policy.

## Eval metric key
- **drop frac**: per-step fraction of envs whose object fell below the hold band (≈0 = held the whole time).
- **yaw angvel**: object angular velocity about +Z (the rotation we want), rad/s.
- **rotate_reward**: clip(yaw, -0.5, 0.5) — saturates at 0.5 when spinning fast.

## Results

| stage | task | obs added | train reward (best) | full-g eval: drop / yaw / rotate_rew | verdict |
|---|---|---|---|---|---|
| official | Isaac-Inhand-Rotate-Sharpa-Wave-v0 (pretrained, distilled) | tactile | — | 0.000 / 0.99 / 0.48 | reference |
| **1 baseline** | `...-Sharpa-Wave-v0` | tactile (proprio+contacts) | 725 | (teacher; see below) | works |
| **2 point cloud** | `...-Sharpa-Wave-PC-v0` | + hand/object point cloud + 2-ch mask | 296 | 0.00005 / 0.347 / 0.370 | **PASS** (held + rotating) |
| **3 world model** | `...-Sharpa-Wave-WM-v0` | + pose-prediction aux loss | 472 | 0.00003 / **1.532** / **0.492** | **PASS** (best — aux helps) |
| **4 multi-scale** | `...-Sharpa-Wave-WM-Multi-v0` | 8 object sizes (0.4-0.6) | 391 | **0.00000** / **1.744** / 0.489 | **PASS** (generalizes) |

**Stage 2 (point cloud + masks).** Egocentric (wrist-frame) point cloud: 128 object surface points
(cylinder, transformed by the object pose each step) + 33 hand body-link points (FK), each tagged with a
2-channel hand/object mask -> 161 points x 5. A lightweight PointNet branch (per-point MLP -> max-pool ->
128-d) is fused into the HORA policy MLP. Threaded through PPO + experience buffer; gated by
`enable_pointcloud`. Result: still reaches full gravity, holds the object (drop ~0), rotates it
(yaw 0.35). Rotation is weaker than the baseline teacher in the same 20M-step budget (the point cloud
makes optimization harder and the curriculum reached full g later), but it clearly works.

**Stage 3 (world model).** The policy net gets a head that predicts the object pose (pos[3] + quat[4],
wrist frame) from the actor latent; PPO adds an MSE+orientation auxiliary loss (`enable_world_model`,
coef 1.0). Far from hurting, it **improved** the policy: best reward 472 (vs 296 for pc-only) and at full
gravity it holds (drop ~0) and rotates at **1.53 rad/s** (rotate_reward 0.49, saturating the clip — faster
than the official pretrained's 0.99). The pose-prediction auxiliary shapes a richer latent. The wm loss
plateaus ~0.45 (good position, rough orientation — expected for a fast-spinning object).

**Stage 4 (multi-scale generalization).** Same pc + world-model design, but 8 object sizes per batch
(cylinder scale linspace 0.4-0.6, provided cache `sharpa_grasp_linspace_0.4-0.6-8.npy`); env i gets size
i%8 and the object point cloud is scaled to match. The adaptive gravity curriculum still drove all 8
sizes to **full gravity**, and at full g the policy **holds every size (drop 0.00000)** and rotates fast
(yaw 1.74 rad/s, rotate_reward 0.49). So the pc+world-model policy generalizes across object sizes.
Caveat — true cross-*shape* generalization (sphere/cube vs cylinder) is a follow-up: it needs (a) a
per-shape grasp cache from `gen_grasp.py`, and (b) the object point cloud sampled from the actual shape
(our `_compute_pointcloud` currently samples a cylinder); the multi-scale result is size generalization
on the cylinder, which is what the provided assets/cache support out of the box.

## How to reproduce (from the rebuild root)
```bash
PY=/home/magics/miniconda3/envs/sharpa/bin/python
ENV="OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0"
# train any stage (task ids above); 4096 envs fits a 24GB 4090
env $ENV $PY rl_rebuild/scripts/train.py --task <task-id> --num_envs 4096 --headless --device cuda:0 --max_agent_steps 20000000
# eval at full gravity
env $ENV $PY rl_rebuild/scripts/eval_metrics.py --task <task-id> --num_envs 64 --eval_steps 600 --load_path <run>/stage1_nn/best.pth --device cuda:0
# render a video (close-up)
env $ENV $PY rl_rebuild/scripts/render_video.py --task <task-id> --num_envs 4 --video_length 300 --load_path <run>/stage1_nn/best.pth --device cuda:0 --out_dir videos_<stage>
```

## Videos (close-up, the hand rotating the cylinder at full gravity)
- official replication: `../sharpa-rl-lab/videos_official/render_Isaac-Inhand-Rotate-Sharpa-Wave-v0-step-0.mp4`
- stage 1 baseline:   `videos_stage1_baseline/render_Isaac-Inhand-Rotate-Sharpa-Wave-v0-step-0.mp4`
- stage 2 point cloud:`videos_stage2_pointcloud/render_Isaac-Inhand-Rotate-Sharpa-Wave-PC-v0-step-0.mp4`
- stage 3 world model: `videos_stage3_worldmodel/render_Isaac-Inhand-Rotate-Sharpa-Wave-WM-v0-step-0.mp4`
- stage 4 multi-scale: `videos_stage4_multiscale/render_Isaac-Inhand-Rotate-Sharpa-Wave-WM-Multi-v0-step-0.mp4`
All verified non-blank with inter-frame motion (the object visibly spins; stage 3/4 show the fastest spin).

## Net result
All four staged additions PASS their gate: the rebuild keeps the working sharpa-rl-lab behavior (robust
hold + rotation at full gravity, tactile obs preserved for later distillation) and successfully layers on
the point cloud + masks, the world-model auxiliary (which *improved* rotation), and multi-size
generalization. The point-cloud + world-model pipeline is threaded through env -> model (PointNet branch +
pose-prediction head) -> PPO -> experience buffer, all gated by `enable_pointcloud` / `enable_world_model`
so the baseline is untouched when off.
