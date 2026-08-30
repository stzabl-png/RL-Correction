# Current task

## Goal and boundaries

Build `tasks/Sweep/2` from the current Pour17 residual-RL implementation. The robot starts with a dustpan fixed to the left hand and a broom fixed to the right hand using validated GraspPose transforms. A fixed 1 cm cube is swept into the dustpan. The current scope excludes cube-position randomization and grasp learning.

Completion is measured by deterministic evaluation over at least 512 episodes, with the cube fully inside the dustpan and stable, at a success rate of at least 50%.

## Environment and important versions

- Remote host: `msc-a6000` (`mscauto-Lambda-Vector`), user `msc-auto`.
- Project root: `/home/msc-auto/RL_sweep`.
- Git baseline: `Step4_RL_Correction` at `4b1daa75ea0625dff10e387f41cfd7320c191d96`.
- Working branch: `sweep-task`.
- Source dataset: `/home/msc-auto/RL_Correction/datasets/sweep_2_better`.
- No package installation or environment mutation is authorized.

## Project/data flow

Verified input flow: Sweep2 video reconstruction provides dustpan (`object_0`) and broom (`object_1`) trajectories, per-object position/rotation confidence, retargeted hand motion, object USD/mesh assets, and GraspPose candidates. The task will convert the smoothed tool trajectories into a robot reference using fixed hand-tool transforms and arm IK, then train a 14-D bimanual arm residual policy. Fingers remain fixed.

## Key code paths

- Existing reference implementation: `tasks/Pour/17/`.
- Existing Sweep registrations: `rl_rebuild/correction/clips.py`.
- New task root: `tasks/Sweep/2/`.
- Runtime videos: `outputs_video/` only.
- Training logs, checkpoints, TensorBoard, and run data: `logs/` only.

## Demo and validation commands

Commands will be added after the remote environment and new entry points are verified.

## Training, tmux, logs, and checkpoints

Persistent training will use a unique task-owned tmux session after static, deterministic, 1-env, multi-env, and short-training checks pass. The exact GPU, command, log, checkpoint path, and resume instructions will be recorded before launch.

## Current status and next checks

- Requirements and implementation plan approved.
- Repository bootstrap in progress.
- Next: import and verify Sweep2 data and resolve runtime environment/assets.
