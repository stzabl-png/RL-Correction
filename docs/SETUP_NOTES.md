# sharpa-rl-lab — local replication notes (4090, conda env `sharpa`)

Replicated from https://github.com/sharpa-robotics/sharpa-rl-lab (commit 95ccda3, tag v1.0.0).
Goal: get the official SharpaWave in-hand cylinder-rotation RL training + eval running locally and
confirm it works. **Status: WORKS end-to-end** (pretrained eval + train-from-scratch + gen_grasp).

## Environment (REUSED, not rebuilt)
Runs in the existing **`sharpa`** conda env (`/home/magics/miniconda3/envs/sharpa/bin/python`):
isaacsim 5.1.0.0 / **isaaclab v2.3.2** (in the repo's "release/2.3.0 tested" range) / torch 2.7.0+cu128.
The repo's `pyproject.toml` has zero hard deps; `tensorboardX 2.6.5` was already present. Installed the
package editable:
```bash
cd /home/magics/mt_dir/Dynrotate/sharpa-rl-lab
/home/magics/miniconda3/envs/sharpa/bin/python -m pip install -e .   # registers `rl_rebuild`
```
No collision with the sibling `sharpa_isaaclab` package (different package name + different gym task ids).

## Fixes applied (replication issues found + fixed)
1. **`bin/pip` is broken in the `sharpa` env** (`ModuleNotFoundError: No module named 'pip'` from the
   console script). Use `python -m pip ...` instead — pip 26.1.1 itself is fine.
2. **CPU/GPU device mismatch in `set_external_force_and_torque`** (newer IsaacLab warp kernel requires
   all args on the same device). In `rl_rebuild/tasks/inhand_rotate/sharpa_wave_env.py`
   `_pre_physics_step`, the random-force call created `torques=torch.zeros(self.num_envs,1,3)` on **CPU**
   while `forces` was on CUDA → `RuntimeError: ... input array for argument 'torques' is on device=cpu`.
   Fixed: `torch.zeros(self.num_envs, 1, 3, device=self.device)`.
3. **Pretrained checkpoints are stage-2 (distilled / ProprioAdapt) policies** — they contain
   `adapt_tconv.*` + `sa_mean_std`. Load/eval them with `--algorithm ProprioAdapt` (loading them as a
   bare PPO model fails with "Unexpected key(s) adapt_tconv...").

## Verified results (single RTX 4090, headless)
- **Pretrained eval** (`pretrained/0.5-0.5-1.pth`, ProprioAdapt, 64 envs, full gravity −9.81):
  object **held with 0 drops**, **yaw angvel ≈ 0.99 rad/s**, rotate_reward ≈ 0.48 (near the 0.5 clip).
  Script: `rl_rebuild/scripts/eval_metrics.py` (added — a bounded version of `play.py` that exits after
  `--eval_steps` and prints rotation/hold metrics, since `play.py`'s `test()` loops forever).
- **Train from scratch** (`Isaac-Inhand-Rotate-Sharpa-Wave-v0`, 4096 envs, 12M steps ≈ 9 min):
  FPS ~23k, reward 1.8 → 388, and the **adaptive gravity curriculum ramped −0.05 → −10 (full g, capped)**
  with no drops. Stage-1 checkpoint: `logs/debug/<ts>/stage1_nn/best.pth`.
- **gen_grasp** (`Isaac-Inhand-Rotate-Grasp-Sharpa-Wave-v0`): confirmed it builds the grasp cache (cache
  size accumulates, 0 errors). The repo already ships `cache/sharpa_grasp_linspace_0.5-0.5-1.npy`, so a
  full 50k regeneration is unnecessary for training.

## How to run (from the repo root)
```bash
cd /home/magics/mt_dir/Dynrotate/sharpa-rl-lab
PY=/home/magics/miniconda3/envs/sharpa/bin/python
COMMON="OMNI_KIT_ACCEPT_EULA=YES PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0"

# eval pretrained (held + rotating, full gravity)
env $COMMON $PY rl_rebuild/scripts/eval_metrics.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 \
  --num_envs 64 --eval_steps 600 --algorithm ProprioAdapt --load_path pretrained/0.5-0.5-1.pth \
  --headless --device cuda:0

# train stage-1 PPO (4096 envs fit in 24 GB; default 16384 will OOM a 4090)
env $COMMON $PY rl_rebuild/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 \
  --num_envs 4096 --headless --device cuda:0 --max_agent_steps 300000000

# distill stage-2 (ProprioAdapt) from a stage-1 checkpoint
env $COMMON $PY rl_rebuild/scripts/train.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 \
  --num_envs 4096 --headless --device cuda:0 --algorithm ProprioAdapt --load_path <stage1>/best.pth
```
Notes: default `--num_envs 16384` OOMs a 24 GB card → use 4096. `minibatch_size = min(num_envs*8, 32768)`.
```
