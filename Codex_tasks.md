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

Pure CPU contracts (verified):

```bash
/home/msc-auto/miniconda3/envs/isaac/bin/python \
  tasks/Sweep/2/A_Design/L3_Learning/selftest_fixed_joint.py
/home/msc-auto/miniconda3/envs/isaac/bin/python \
  tasks/Sweep/2/A_Design/L3_Learning/selftest_progress.py
```

## Training, tmux, logs, and checkpoints

Persistent training will use a unique task-owned tmux session after static, deterministic, 1-env, multi-env, and short-training checks pass. The exact GPU, command, log, checkpoint path, and resume instructions will be recorded before launch.

## Verified runtime and dataset state

- Runtime: `/home/msc-auto/miniconda3/envs/isaac/bin/python` (Python 3.11,
  PyTorch 2.7.0+cu128) with Isaac Lab from `/home/msc-auto/MagicSim_IsaacLab`.
- The repository-local launch comments and `env_a6000.sh` contain stale paths;
  Sweep launchers must set their own project root and interpreter explicitly.
- Sweep2 was copied into `datasets/sweep_2_better` (4,982 files, about 136 MB).
- Source video is 300 frames at 30 fps (10 seconds). The retarget NPZ says 15 fps;
  reference construction therefore preserves source frame indices and explicitly
  resamples to the 20 Hz control clock instead of trusting that metadata silently.
- `object_0` is the dustpan (left hand); `object_1` is the broom (right hand).
- The tracked robot USD and both GraspPose prior blobs were hydrated and hash-checked.

## Historical failure evidence used by this implementation

- The deleted predecessor's fixed joints snapped each tool by 20.8--21.1 cm on
  reset. New joints must be derived from the GraspPose hand-in-object transform,
  and an executable reset-offset assertion must reject millimetre-scale mismatch.
- Its zero-residual broom missed the 1 cm cube by at least 2.4 cm and never produced
  particle entry/success. The new fixed cube location is derived from the actual
  broom corridor and pan mouth, not copied from that task.
- It accumulated high pan/reference shaping reward while true success remained zero.
  New task progress and clock advancement are tied to cube motion and containment.
- One old branch ran without GraspPose and with an inconsistent seven-dimensional
  active action path. Sweep2 fixes both fingers from validated priors and exposes
  exactly 14 arm residuals to both policy and trainer.

## Current status and next checks

- Repository, data, output directories, and task records are bootstrapped.
- Both A6000 GPUs are currently owned by other users' active jobs; no job has been
  interrupted and no Isaac/GPU process will start until a slot is safely available.
- Pure geometry/progress and Fixed Joint frame contracts are implemented and their
  CPU self-tests pass.
- Sweep clip paths are now self-contained and the P-OBJ reference builder is
  implemented with explicit 30 Hz source to 20 Hz control resampling.
- The physical environment is statically implemented: one fixed cube, two physical
  tool attachments, frozen fingers, 14 arm actions, P-OBJ confidence bounds, and
  cube-grounded task reward/termination. Runtime construction contains a strict
  3 mm attachment/reset assertion.
- Runtime reference generation is waiting for a safe GPU slot; no foreign process
  will be interrupted. In parallel, physical scene wiring can be implemented and
  statically checked.
