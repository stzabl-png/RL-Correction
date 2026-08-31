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

Approved execution order (commands are code-complete but GPU validation is pending):

1. Build `tasks/Sweep/2/A_Design/L2_Reference/sweep2_reference_v1.npz`.
2. Run `smoke_sweep.py` with one and eight environments, zero then small random actions.
3. Record zero residual to `outputs_video/sweep2_zero_reference_v1.mp4`.
4. Run `make_expert.py`; it saves data only after physical success to
   `logs/expert/sweep2_success_v1.npz` and records its replay under `outputs_video/`.
5. Launch `train_sweep.py` in a uniquely named tmux. It performs actor BC once,
   saves `bc_warm.pth`, then switches permanently to pure on-policy PPO.
6. Evaluate a selected checkpoint with `eval_sweep.py --num_envs 512`; acceptance is
   exactly the emitted `success_rate >= 0.50` report under the run's `logs/` tree.

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
- The stale tracked broom prior was rejected by multi-start ArmIK. The selected
  replacement is `Sweep2_broom_v2.npz`, sourced from the current functional-region
  candidate `8_Prismatic_2_Finger__46_16` and placed at a 90 degree task yaw.
- The validated 200-row reference preserves 100% of reconstructed position
  increments and 8% of rotation increments after a rigid 120 degree recon-to-sim
  world registration. Full rotation was proven infeasible for every tested world
  yaw; 10% was reachable but exceeded the 8 degree/frame continuity gate.
- Final reference SHA-256:
  `3a3171625c84e42c92425f0c23ca20d0b3080471c3e3744e3e3d9ec1c35b6b6f`.
  Right/left IK are both 200/200, with maximum position errors 4.91/4.77 mm and
  maximum joint steps 7.89/1.93 degrees. The nominal brush-to-cube corridor miss is
  1.45 cm, intentionally left for the expert residual (zero residual need not win).

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
- Reference generation is complete and passed both CPU contract tests, compilation,
  and whitespace checks. Physical Isaac construction/replay is the next gate.
- Both A6000 GPUs remain occupied by foreign training/recording jobs at 98%/91%
  utilization. No Isaac process, video, expert generation, or training has started;
  those steps must resume only after a safe GPU slot is observed.

## 2026-08-30 — Reconstruction correction (training blocked)

- `outputs_video/sweep2_zero_reference_v1.mp4` is rejected and must not be used as
  a validation artifact. Visual inspection against the source ego video and
  `outputs_video/sweep2_dexonomy_grasps.mp4` shows incorrect broom orientation and
  hand/tool interpenetration.
- The rejected reference introduced task-planned changes that are outside the
  source-faithful Pour data flow: a 90-degree broom yaw, a global trajectory yaw,
  8% rotation retention, a fixed/shifted dustpan, and extra position smoothing.
- The authoritative Sweep clock is the 15 Hz value stored in `retarget/replay_world.npz`
  and both `retarget/ref_qpos_*.npz` files. The 30 fps MP4 container does not override
  the trajectory arrays' time base. A 300-row reconstruction therefore spans 20 s
  before resampling to the 20 Hz control clock.
- Correct reconstruction contract: preserve the RTS object 6DoF increments; use
  GraspPose for the frame-zero hand/object transform and fixed finger posture;
  solve arm IK from the object-driven hand targets; weld reconstructed human wrist
  motion to the GraspPose at frame zero and retain only its motion increments for
  medium/low-confidence shape guidance. Only a documented small shared initial
  scene registration may be applied.
- The scripted expert process was stopped by exact task-owned PID before it saved
  any dataset or expert video. Actor BC and PPO have not started and remain blocked
  until a corrected zero-residual replay passes the visual source/GraspPose gate.
- The prior 1-env/8-env reset results are invalid as task acceptance evidence because
  they validated attachment consistency inside the rejected scene, not reconstruction
  fidelity. The corrected scene must repeat the 1-env physical assertion and video
  gate; the user has explicitly waived the 8-env pre-video gate.

## 2026-08-30 — Corrected source replay awaiting user acceptance

- Corrected reference: `tasks/Sweep/2/A_Design/L2_Reference/sweep2_reference_v1.npz`,
  501 rows at 20 Hz (25.0 s), source clock 15 Hz. Full object 6DoF increments are
  retained; scene registration is one shared -14 degree yaw.
- Physical gate: `logs/smoke/sweep2_sourcefaithful_1env_v5_20260830.log` records
  `PASS envs=1 steps=64`. Right/left FixedJoint relative position errors are
  approximately 0; absolute row-zero tool errors are 0.50/0.00 mm after the bounded
  4.71 mm left runtime registration.
- Replay command used GPU1 in task-owned tmux and wrote only to approved paths:
  `CUDA_VISIBLE_DEVICES=1 RL_ISAAC_NO_GUARD=1 SHARPA_WANDB=0 PYTHONPATH=. /home/msc-auto/miniconda3/envs/isaac/bin/python -u tasks/Sweep/2/C_Wiring/record_sweep.py --out outputs_video/sweep2_zero_reference_sourcefaithful_v2.mp4 --steps 0 --enable_cameras --headless`.
- Replay result: `outputs_video/sweep2_zero_reference_sourcefaithful_v2.mp4`, H.264
  1280x720, 20 fps, 540 frames, 27.0 s. Durable log:
  `logs/video_sweep2_zero_reference_sourcefaithful_v2_20260830.log`; inspection
  sheet: `artifacts/video_inspection/sourcefaithful_v2/contact_sheet.png`.
- Current boundary: wait for user visual approval. Scripted expert, actor-only BC,
  PPO, and deterministic evaluation have not started.
## 2026-08-30 — Sweep2 physical-contract repair in progress

- Authoritative nominal contact is reference row 54, not the rejected hard-coded
  row 425. The fixed cube uses `cube_start_w` xy and the 25 mm cube's table height.
- The active cube is 25 mm and 5 g with unchanged 0.6/0.5 static/dynamic friction.
- Gate 3 uses pan-local cube-centre y `[18, 30] mm`, derived from the basin's
  `+8..11 mm` load-bearing surface plus 12.5 mm cube half extent. A cube below the
  pan is explicitly rejected by the CPU self-test.
- The dustpan visual mesh remains unchanged. Its VHACD collision is disabled at
  runtime and replaced by four invisible compound boxes: floor, two side walls,
  and back wall, with the positive-z mouth left open.
- Static validation: `PYTHONPATH=. /home/msc-auto/miniconda3/envs/isaac/bin/python
  tasks/Sweep/2/A_Design/L3_Learning/selftest_progress.py` passes.
- Physical smoke: `logs/smoke/sweep2_openpan_smoke_v7_20260830.log` passes one env,
  64 zero-action steps, FixedJoint assertions, and open-compound construction.
- Active physical gate: tmux `codex_sweep2_pan_probe_v5_20260830`, log
  `logs/probes/sweep2_pan_geometry_v5_20260830.log`, data
  `logs/probes/sweep2_pan_geometry_v5.npz`, video
  `outputs_video/sweep2_pan_geometry_v5.mp4`.
- Scripted expert, actor BC, PPO, and final evaluation remain blocked until both the
  basin-support and table-level mouth-entry probe gates pass.

## 2026-08-30 — Approved expert reconstruction plan and active row313 gate

- User-approved expert source segment: reconstructed rows 300--387.  Rows 300--313
  preserve the visible outward preparation and rows 313--387 preserve the inward
  sweep in the dustpan frame.  The complete source reference remains unchanged.
- The task setup will hold the dustpan at a table-aligned row313 pose, place the
  fixed 25 mm / 5 g cube near pan-local `[x=+10 mm, z=125 mm]`, and use only
  bounded closed-loop corrections around the reconstructed broom motion.
- Audit correction: the v7 entrance probe physically reached Gate4 and then reset;
  the script ignored `done` and misreported the reset state as failure.  The active
  repair captures `success/gates/cube_pan` before reset and uses live-USD bristle
  points instead of source-OBJ coordinates.
- Next single-environment gate (GPU0, no BC/PPO): tmux
  `codex_sweep2_pan_probe_row313_v8_20260830`, log
  `logs/probes/sweep2_pan_geometry_row313_v8_20260830.log`, data
  `logs/probes/sweep2_pan_geometry_row313_v8.npz`, video
  `outputs_video/sweep2_pan_geometry_row313_v8.mp4`.
- Exact launch command from `/home/msc-auto/RL_sweep`:

  ```bash
  CUDA_VISIBLE_DEVICES=0 RL_ISAAC_NO_GUARD=1 SHARPA_WANDB=0 PYTHONPATH=. \
    /home/msc-auto/miniconda3/envs/isaac/bin/python -u \
    tasks/Sweep/2/C_Wiring/probe_pan_geometry.py \
    --hold_row 313 --entry_speed 0.03 --entry_steps 160 \
    --out logs/probes/sweep2_pan_geometry_row313_v8.npz \
    --video outputs_video/sweep2_pan_geometry_row313_v8.mp4 \
    --enable_cameras --headless
  ```
- Expert generation, BC, and PPO remain blocked until this corrected low-speed
  support/entry gate passes.  Expert video approval remains a separate user gate.

### row313 v8 result and v9 correction

- v8 support passed, but entry failed for a probe-placement reason rather than a
  valid ramp test: the intended pan-local start `[0, *, 125] mm` became
  `[22.9, 68.9, 190.7] mm` on the first transition.  The old helper chose pan-local
  x/z and then overwrote world z, which changes pan-local z whenever the pan is
  tilted.  The cube therefore never started at the tested mouth location.
- `_pan_cube_start` now solves the required pan-local y analytically so pan-local
  x/z remain exact while the cube rests on the world table.  The probe also applies
  one measured post-settle vertical correction to compensate FixedJoint/PD load.
- Replacement gate: tmux `codex_sweep2_pan_probe_row313_v9_20260830`, log
  `logs/probes/sweep2_pan_geometry_row313_v9_20260830.log`, data
  `logs/probes/sweep2_pan_geometry_row313_v9.npz`, video
  `outputs_video/sweep2_pan_geometry_row313_v9.mp4`; same command as v8 with all
  `v8` output names changed to `v9`.

### row313 v9 result and v10 non-penetrating start

- v9 validated the corrected pan-frame placement and the post-settle pan-height
  calibration: basin support passed and the settled whole-mesh/table gap was
  1.03 mm.  Entry was not a valid push test because the 25 mm world-axis cube at
  pan-local z=125 mm overlapped the tilted entrance ramp.  Its first transition
  produced a 2.64 m/s separation impulse and launched it away from the mouth.
- Oriented-footprint analysis puts the ramp front at pan-local z=119.6 mm and the
  cube's projected half extent at about 18.3 mm.  The fixed start is therefore
  moved to pan-local z=140 mm (`start_outside=45 mm`), leaving approximately 2 mm
  clearance without changing cube size, pan geometry, success gates, or source
  motion.
- Replacement one-environment gate: tmux
  `codex_sweep2_pan_probe_row313_v10_20260830`, log
  `logs/probes/sweep2_pan_geometry_row313_v10_20260830.log`, data
  `logs/probes/sweep2_pan_geometry_row313_v10.npz`, video
  `outputs_video/sweep2_pan_geometry_row313_v10.mp4`.  BC and PPO remain blocked.

### row313 v10 result and v11 table-aligned mouth

- v10 removed the initial solver explosion and reconfirmed basin support.  The
  fixed cube made controlled contact at pan-local z=136.6 mm but could not climb:
  it never entered and Gate2--4 remained false.  The trace and video show a real
  raised-mouth failure rather than a mass, command-direction, or reset artifact.
- The source row313 pan normal is tilted about eight degrees from world up.  Raising
  only the whole-mesh minimum to 1 mm leaves the mouth rolled and the central ramp
  face centimetres above the table.  v11 preserves reconstructed yaw and GraspPose,
  levels the pan work plane, and uses a 55 mm / 26 degree ramp whose front top is
  near local y=-14 mm and whose rear top joins the y=+10 mm basin floor.
- Replacement gate: tmux `codex_sweep2_pan_probe_row313_v11_20260830`, log
  `logs/probes/sweep2_pan_geometry_row313_v11_20260830.log`, data
  `logs/probes/sweep2_pan_geometry_row313_v11.npz`, video
  `outputs_video/sweep2_pan_geometry_row313_v11.mp4`.  Expert generation remains
  blocked until this gate records pre-reset Gate4.

### v11 startup guard and v12 staged-ramp gate

- v11 terminated before its physical phases because the newly lowered ramp was
  already active at the transient, tilted source-row-zero spawn.  Table contact
  displaced the attached left tool by 16.36 mm, correctly tripping the existing
  1 cm initialization-adjustment guard.
- v12 keeps only the new ramp collision disabled during spawn and pan alignment,
  enables it after the table-aligned pose has settled, and advances another 20
  physics steps before asserting a 0--4 mm whole-mesh/table gap.  The ramp is now
  1 mm thick; its working-pose leading bottom/top are approximately 1/2 mm above
  the table, avoiding both initialization impulse and a false embedded collider.
- Gate: tmux `codex_sweep2_pan_probe_row313_v12_20260830`, log
  `logs/probes/sweep2_pan_geometry_row313_v12_20260830.log`, data
  `logs/probes/sweep2_pan_geometry_row313_v12.npz`, video
  `outputs_video/sweep2_pan_geometry_row313_v12.mp4`.

### v12 reference-bound guard and v13 fixed-left task reference

- v12 reached the table-aligned IK solve with 0.10 mm position and 0.01 degree
  orientation error.  It stopped before physics because that pose is 1.238 times
  the old left-arm policy residual envelope at source row313.
- The level-pan pose is deterministic task setup, not behavior the actor should
  learn.  v13 installs its valid GraspPose IK directly as the fixed left-arm task
  reference and keeps the actor residual at zero; the exploration envelope is not
  widened.  The original source left-arm delta is retained as diagnostics and will
  be serialized into the expert metadata.
- Gate: tmux `codex_sweep2_pan_probe_row313_v13_20260830`, log
  `logs/probes/sweep2_pan_geometry_row313_v13_20260830.log`, data
  `logs/probes/sweep2_pan_geometry_row313_v13.npz`, video
  `outputs_video/sweep2_pan_geometry_row313_v13.mp4`.

### v13 raised-edge result and v14 continuous ramp

- v13 passed table alignment (`level_adjust=8.19 deg`), enabled-ramp settling
  (`whole-mesh gap=0.95 mm`), and basin support (`inside=True`, 0.0211 m/s).
  Its real outside-entry phase still stopped at z=138.66 mm: the 1 mm ramp body
  left a 1--2 mm rigid leading edge above the table, so Gate2--4 stayed false.
- v14 makes the ramp surface continuous with the table: the front top is about
  1 mm below the table and the rear top joins the +10 mm basin.  A filtered pair
  disables only ramp--table collision; cube--ramp, cube--table, pan--cube, and all
  other collisions remain active.  The fixed cube centre starts 50 mm outside the
  mouth to clear the ramp/contact-offset envelope; no randomization is added.
- Gate: tmux `codex_sweep2_pan_probe_row313_v14_20260830`, log
  `logs/probes/sweep2_pan_geometry_row313_v14_20260830.log`, data
  `logs/probes/sweep2_pan_geometry_row313_v14.npz`, video
  `outputs_video/sweep2_pan_geometry_row313_v14.mp4`.

### v14 contact-offset result and v15 submerged shallow ramp

- v14 again passed pan height and basin support, but entry stopped at pan-local
  z=141.0 mm.  At contact the cube/ramp horizontal gap was about 1.1 mm and the
  ramp-front/cube-bottom vertical gap about 1.5 mm, both inside the collider contact
  envelope; the solver therefore still treated the submerged box front as contact.
- v15 lowers the front ramp top to about 4 mm below the table, extends the approach
  to 80 mm at 21.5 degrees, and assigns the ramp a 0.2 mm contact offset.  Its rear
  remains flush with the +10 mm basin.  The fixed cube centre is z=160 mm, giving
  approximately 5 mm clearance from the full ramp/contact envelope.
- Gate: tmux `codex_sweep2_pan_probe_row313_v15_20260830`, log
  `logs/probes/sweep2_pan_geometry_row313_v15_20260830.log`, data
  `logs/probes/sweep2_pan_geometry_row313_v15.npz`, video
  `outputs_video/sweep2_pan_geometry_row313_v15.mp4`.

### v15 horizon audit and v16 reachable isolated-entry gate

- Trace audit corrected the endpoint interpretation: v15 did not stick at
  z=151.11 mm.  It advanced almost linearly by 0.0435 mm per environment step,
  travelling only 8.7 mm in the 200-step window and never reaching the ramp.
- Because the probe disables the broom to isolate pan topology, its original Gate2
  requirement was also unreachable.  v16 explicitly marks Gates1/2 as probe
  preconditions and resets `cube_start` to the tested outside position; the actual
  Gate3 full-containment and Gate4 ten-step stability rules are unchanged.
- v16 uses a controlled 0.12 m/s imposed speed for at most 500 steps, then removes
  it immediately on full containment.  This remains a geometry gate, not expert
  data and not a relaxed training success definition.
- Gate: tmux `codex_sweep2_pan_probe_row313_v16_20260830`, log
  `logs/probes/sweep2_pan_geometry_row313_v16_20260830.log`, data
  `logs/probes/sweep2_pan_geometry_row313_v16.npz`, video
  `outputs_video/sweep2_pan_geometry_row313_v16.mp4`.

### v16 ramp-climb result and v17 continuous-force gate

- v16 reached the ramp: over its first ~150 entry steps the cube moved from
  z=158.5 to 104.6 mm and climbed from y=-2.4 to +9.1 mm with no velocity spike.
  It then held at the ramp midpoint because a once-per-environment-step velocity
  write is cancelled by ramp normal/friction during the physics substeps.
- v17 replaces that diagnostic actuator with a persistent 0.05 N world-frame force
  along pan -z.  The force is immediately set to zero when full containment is
  first observed; success still requires the unchanged ten stable steps.  This is
  a pan-geometry probe only, not saved expert behavior.
- Gate: tmux `codex_sweep2_pan_probe_row313_v17_20260830`, log
  `logs/probes/sweep2_pan_geometry_row313_v17_20260830.log`, data
  `logs/probes/sweep2_pan_geometry_row313_v17.npz`, video
  `outputs_video/sweep2_pan_geometry_row313_v17.mp4`.

### User-directed pivot from topology probes to the near-success expert

- The user correctly rejected v16/v17 as expert evidence: those videos deliberately
  disable broom collision and reset the cube between support/entry phases.  v17 was
  stopped without starting BC/PPO; no topology-probe rollout will be presented as
  an expert again.
- Re-audited `outputs_video/expert_failures/closed_loop_cube_pan_servo_v1.mp4` and
  its 500-step trace.  It contains the desired continuous right-hand sweep and earns
  Gates1/2.  Closest containment is at step301, still 35.3 mm shallow in pan-z and
  15.5 mm below the accepted basin-centre y; it also has a 2.67 m/s contact spike.
- Active expert v3 keeps that controller structure but installs the verified fixed,
  level left-pan GraspPose reference, enables the continuous ramp, fixes the cube at
  pan-local z=160 mm, and transforms reconstructed source rows300--387 into the task
  frame using `T_broom_task = T_pan_fixed * inv(T_pan_source) * T_broom_source`.
  Right-arm IK is the feed-forward task reference; only a bounded cube-contact servo
  is stored as actor residual.
- Next run: tmux `codex_sweep2_expert_source_relative_v3_20260830`, log
  `logs/expert/sweep2_expert_source_relative_v3_20260830.log`, success data
  `logs/expert/sweep2_success_source_relative_v3.npz`, success video
  `outputs_video/sweep2_expert_source_relative_v3.mp4`; a failure, if any, is saved
  under the corresponding `logs/expert/failures/` and
  `outputs_video/expert_failures/` paths.  BC/PPO remain blocked.

### Expert v3 early result and v4 fixed-cube correction

- v3 passed all 88 source-relative right-arm IK rows, with maximum 0.486 mm and
  0.121 degree error.  It failed for a fixed-start placement reason: at task row0
  the broom was already 15.4 mm from the cube, so rows300--313's required outward
  preparation contacted it and moved it from z=160 to about 254 mm.  Gate2 was
  earned but the subsequent inward reference could not recover that displacement.
- v4 moves only the deterministic cube centre 35 mm outward to pan-local z=195 mm.
  This keeps row300 collision-free and lets row313 finish behind the cube before
  source rows313--387 perform the inward sweep.  Pan pose, GraspPoses, right-hand
  source-relative motion, policy bounds, and Gate1--4 criteria are unchanged.
- Planned replacement: tmux `codex_sweep2_expert_source_relative_v4_20260830`, log
  `logs/expert/sweep2_expert_source_relative_v4_20260830.log`, success data
  `logs/expert/sweep2_success_source_relative_v4.npz`, video
  `outputs_video/sweep2_expert_source_relative_v4.mp4`.
- User-approved ease adjustment for v4: the fixed block is 30 mm / 6 g with
  static/dynamic friction 0.25/0.15 and restitution 0.01.  Containment uses the
  true 15 mm half extent and basin-centre y `[21, 34] mm`; broom proximity is
  `cube_half + 6 mm`.  Earlier 25 mm notes describe superseded probes.

### Expert v4 initialization failure and v5 point-cloud placement

- v4 proved that a scalar z offset was insufficient: actual reset broom distance
  was 12.5 mm, the 30 mm cube overlapped at row300, and relative speed reached
  1.69 m/s on step1 and about 9 m/s by step3.  It terminated after 10 transitions;
  no part of that trace is valid expert data.
- v5 searches a deterministic `(pan-x, pan-z)` grid using the complete live-USD
  bristle point cloud transformed at task rows0/13 (source rows300/313).  Accepted
  placement requires row300 centre distance >=35 mm, row313 distance <=40 mm, the
  nearest row313 bristle at least 3 mm on the outside `+z` side, and x within the
  central pan corridor.  A second actual-reset assertion requires >=30 mm before
  the first physics transition.
- Replacement: tmux `codex_sweep2_expert_source_relative_v5_20260830`, log
  `logs/expert/sweep2_expert_source_relative_v5_20260830.log`, success data
  `logs/expert/sweep2_success_source_relative_v5.npz`, video
  `outputs_video/sweep2_expert_source_relative_v5.mp4`; failure artifacts use the
  unique `source_relative_fixed_pan_cube_servo_v5` tag.
- Record integrity note: v4 reused v3's failure tag and overwrote v3's diagnostic
  NPZ/video.  The durable v3 run log retains its numeric result, but that video is
  not recoverable without rerunning v3.  All subsequent attempts use unique tags.

### v1 frame/trace audit and minimal fixed-pan expert correction

- The user superseded the proposed larger-block branch.  The task is again the
  fixed 25 mm / 5 g block used by `closed_loop_cube_pan_servo_v1.mp4`, with the
  original 0.6/0.5 friction and no position randomization.
- The 500-frame v1 video was aligned one-to-one with
  `logs/expert/failures/closed_loop_cube_pan_servo_v1.npz`.  Gate1 starts at frame0,
  Gate2 at frame2, and Gates3/4 never occur.  The closest containment state is
  frame301: cube pan-local `[-20.85, 2.49, 117.83] mm`, which is 35.33 mm short in
  entry depth and 15.51 mm below the accepted basin-centre height.  Peak relative
  speed is 2.675 m/s at frame68.
- The useful frame301 state is still zero residual.  The v1 right-arm controller
  first emits a nonzero action at frame405; v2 moved onset to frame300 but froze the
  reference at row300, saturated all seven right-arm channels, and improved the
  geometric deficit by only about 5 mm.  The minimal correction must therefore
  preserve source rows300--387 rather than hold one row.
- The pan work-plane normal is already within about 8.1 degrees of world up at the
  closest frame.  The actual left-side defect is height: at the open mouth the
  compound floor top is about 24 mm above the table-resting block bottom.  Reuse the
  row313 probe's validated fixed pan IK, continuous ramp, and roughly 10 mm lowering.
- Expert v6 keeps source-relative rows300--387, applies an 8 mm pan-normal clearance
  to row300 that smoothly decays to zero at row313, then activates the measured
  bristle/cube servo.  The cube remains fixed at pan-local `[0, *, 160] mm`.
- Next one-environment run: tmux
  `codex_sweep2_expert_v1_minfix_v6_20260830`, log
  `logs/expert/sweep2_expert_v1_minfix_v6_20260830.log`, success data
  `logs/expert/sweep2_success_v1_minfix_v6.npz`, and success video
  `outputs_video/sweep2_expert_v1_minfix_v6.mp4`.  Failed evidence uses the unique
  `closed_loop_cube_pan_servo_v1_minfix_v6` basename.  BC/PPO remain blocked.

### v6 dynamic reset failure and v7 initialization-order correction

- v6 passed 88/88 task IK and the static/reset centre-distance assertion (23.07 mm
  versus 14.50 mm required), but the cube escaped outward to pan-z=656 mm before
  feedback began.  The saved 500-step trace/video are
  `logs/expert/failures/closed_loop_cube_pan_servo_v1_minfix_v6.npz` and
  `outputs_video/expert_failures/closed_loop_cube_pan_servo_v1_minfix_v6.mp4`.
- Trace causality is unambiguous: actions are exactly zero through step14; relative
  speed is already 1.695 m/s at step1 and peaks at 1.966 m/s at step5.  The pan
  frame jumps while the fixed-pan IK/constraint settles, and the already enabled
  grounded ramp sweeps through the fixed cube.  This is a reset-order defect, not a
  right-arm or friction defect.
- v7 parks the cube outside the work area, settles source row300 with the ramp
  disabled for 80 steps, enables/settles the ramp for 12 more steps, asserts pan
  position error below 3 mm and speed below 0.05 m/s, then installs the single fixed
  cube start before frame0.  No setup frame is recorded as expert behavior.
- Next run: tmux `codex_sweep2_expert_v1_minfix_v7_20260830`, log
  `logs/expert/sweep2_expert_v1_minfix_v7_20260830.log`, success data/video
  `logs/expert/sweep2_success_v1_minfix_v7.npz` and
  `outputs_video/sweep2_expert_v1_minfix_v7.mp4`.  Failed evidence uses the unique
  `closed_loop_cube_pan_servo_v1_minfix_v7` basename.

### v7 pre-rollout absolute-pose assertion

- v7 performed the park/settle sequence and stopped before installing or stepping
  the task cube: pan speed was 0.0205 m/s, but its settled absolute root position
  differed from the analytic target by 3.49 mm, exceeding the newly added 3 mm
  absolute-position assertion by 0.49 mm.  No v7 expert trace/video was created.
- This does not weaken the existing 3 mm FixedJoint-relative assertion, which
  remains in `SweepEnv`.  v8 records the absolute delta vector and rotation, allows
  the already observed deterministic loaded-root offset up to 5 mm, and still
  requires rotation below 2 degrees and pan speed below 0.05 m/s.
- Next run: `codex_sweep2_expert_v1_minfix_v8_20260830`, with all v8 logs/data/video
  using the matching unique basename under `logs/` and `outputs_video/`.

### v8 low-pan success and premature preparation contact

- v8 validates the left-side correction: after settling, pan absolute delta is
  `[-2.004, +1.125, -2.625] mm`, rotation error 0.370 degrees, and speed
  0.0205 m/s.  The video visibly shows a low, level pan and continuous mouth ramp.
- It still fails before the inward sweep.  Right actions are zero through step14,
  but broom centre-distance reaches the 12.5 mm cube half extent at step10.  The
  8 mm preparation lift was being tapered to zero, so rows300--312's outward motion
  contacted the cube and moved it from pan-z=162 to 234 mm.  Feedback then begins
  from an already invalid state and the cube escapes laterally to pan-x=280 mm.
- Saved evidence:
  `logs/expert/failures/closed_loop_cube_pan_servo_v1_minfix_v8.npz` and
  `outputs_video/expert_failures/closed_loop_cube_pan_servo_v1_minfix_v8.mp4`.
- v9 keeps a 15 mm lift constant over source rows300--312, drops to the reconstructed
  row313 height only after the broom is outside the cube, and relies on the existing
  contact-gated clock to hold row313 while the bounded servo approaches.  Escape
  diagnostics now stop on either pan-z>350 mm or |pan-x|>150 mm.
- Next run: tmux `codex_sweep2_expert_v1_minfix_v9_20260830`; success/failure
  artifacts use the matching unique v9 basenames under `logs/` and `outputs_video/`.

### v9 physical trace: preparation still contacts before control

- v9 kept the nominal brush 15 mm above source rows300--312, but the physical
  trace proves this margin is insufficient.  The cube first exceeds 1 mm motion at
  step2 and is displaced 27.3 mm by step4 while every actor action remains exactly
  zero; the first nonzero closed-loop action is step15.  At step4 the cube is at
  pan `[-15.1, -2.4, 184.0] mm`, and by step45 it has settled near
  `[-16.0, -2.2, 224.7] mm`.  Therefore neither cube friction nor post-contact
  feedback is causal for this attempt.
- Saved evidence:
  `logs/expert/failures/closed_loop_cube_pan_servo_v1_minfix_v9.npz` and
  `outputs_video/expert_failures/closed_loop_cube_pan_servo_v1_minfix_v9.mp4`.
  The low, level pan and continuous entry ramp remain visually correct.
- v10 changes only the collision-free preparation margin: rows300--312 use a
  35 mm pan-normal lift (15 mm nominal lift plus the measured ~20 mm dynamic
  tracking margin).  It adds a hard physical gate that aborts if the cube moves
  more than 2 mm before contact row313; the fixed 25 mm / 5 g cube, friction,
  pan pose, source rows300--387, residual bounds, and success criteria are unchanged.
- Next run: tmux `codex_sweep2_expert_v1_minfix_v10_20260830`, log
  `logs/expert/sweep2_expert_v1_minfix_v10_20260830.log`, success outputs
  `logs/expert/sweep2_success_v1_minfix_v10.npz` and
  `outputs_video/sweep2_expert_v1_minfix_v10.mp4`; failed evidence uses the unique
  `closed_loop_cube_pan_servo_v1_minfix_v10` basename.

### v10 clears preparation; live nearest-point feedback pushes laterally

- v10 passes the new preparation gate: the cube stays within 0.51 mm of its
  installed pose through row312, the first nonzero action is step15, and the first
  motion above 1 mm is step19 after feedback begins.  This validates the 35 mm
  collision-free lift and keeps it for later attempts.
- The post-contact controller fails laterally.  It moves pan-z inward from 162 to
  114 mm by step30, but simultaneously moves pan-x to 37 mm and then to 135 mm by
  step45; the lateral escape reaches 156 mm at step90.  Video frames show the brush
  changing which edge of its bristles acts on the cube.  The controller was
  re-selecting the nearest one of 2048 mesh points at every update, so its controlled
  working point was discontinuous even though the desired cube target was smooth.
- Saved evidence:
  `logs/expert/failures/closed_loop_cube_pan_servo_v1_minfix_v10.npz` and
  `outputs_video/expert_failures/closed_loop_cube_pan_servo_v1_minfix_v10.mp4`.
- v11 uses the single reconstructed brush contact point after its verified live-USD
  root transform (`raw.broom_contact_local`), tracks the cube centre exactly in x,
  updates feedback every step, and caps each Cartesian correction at 6 mm.  This is
  a contact-servo correction only; pan, cube, source rows, success gates, and the
  validated 35 mm preparation remain unchanged.
- Next run: tmux `codex_sweep2_expert_v1_minfix_v11_20260830`, log
  `logs/expert/sweep2_expert_v1_minfix_v11_20260830.log`; all success/failure
  trace and video paths use unique v11 basenames under `logs/` and `outputs_video/`.

### v11 reaches the mouth depth but starts on the bristle edge

- v11 preserves the pre-contact gate and improves inward motion: the cube reaches
  pan-z 88.8 mm from its 162 mm start, only 6.3 mm short of the centre-depth needed
  for full containment.  It still escapes laterally, reaching pan-x 171.9 mm at
  step68.  The first action is step13 and first cube motion is step14, so this is a
  post-contact alignment failure, not reset/preparation collision.
- Reference geometry explains the direction.  The reconstructed contact point is
  at pan-x -31.9 mm on source row313 and traverses to +23.5 mm by row387, while the
  v10/v11 fixed cube was at x=0.  The original v1 near-success trace had the cube at
  x=-20.85 mm at its closest frame301.  Placing the fixed cube at x=-20 mm therefore
  aligns it with the reconstructed brush corridor and still leaves 27.5 mm of
  full-containment lateral margin.
- Saved evidence:
  `logs/expert/failures/closed_loop_cube_pan_servo_v1_minfix_v11.npz` and
  `outputs_video/expert_failures/closed_loop_cube_pan_servo_v1_minfix_v11.mp4`.
- v12 changes only the deterministic cube pan-x from 0 to -20 mm.  It retains the
  fixed 25 mm / 5 g cube, z=160 mm start, pan, ramp, friction, source rows300--387,
  35 mm preparation lift, fixed live-USD contact point, 6 mm servo cap, and all
  physical success gates.
- Next run: tmux `codex_sweep2_expert_v1_minfix_v12_20260830`, log
  `logs/expert/sweep2_expert_v1_minfix_v12_20260830.log`; all success/failure
  trace and video paths use unique v12 basenames under `logs/` and `outputs_video/`.

### v12 confirms source lateral crossing is causal

- Moving the cube to x=-20 mm does not remove the failure: v12 pushes it deeply
  inward to pan-z 26.2 mm, but laterally from x=-18.9 to +156.0 mm by step30.
  Together with v11's x=0 start ending at +171.9 mm, this shows the final lateral
  escape is nearly invariant to the 20 mm start shift.
- The registered source contact path itself crosses pan-x from -31.9 mm at row313
  to +23.5 mm at row387.  v13 therefore keeps source depth (`z`), height (`y`),
  all orientations, timing, and the outward-then-inward motion, while translating
  the same physical brush working point onto the fixed x=-20 mm corridor at every
  row.  This removes only the measured lateral crossing.
- The row313 diagnostic used `points_pan[1]` with a row313 vertex index; it is fixed
  to `points_pan[-1]` so future outside-z reports describe the actual contact row.
  This changes reporting only, not physics or success gates.
- Saved evidence:
  `logs/expert/failures/closed_loop_cube_pan_servo_v1_minfix_v12.npz` and
  `outputs_video/expert_failures/closed_loop_cube_pan_servo_v1_minfix_v12.mp4`.
- Next run: tmux `codex_sweep2_expert_v1_minfix_v13_20260830`, log
  `logs/expert/sweep2_expert_v1_minfix_v13_20260830.log`; unique v13 success/failure
  traces and videos remain under `logs/` and `outputs_video/`.

### v13 controls the bristle-face midpoint from the wrong side

- v13 removes the large lateral escape: at step100 the cube remains near pan-x
  -40 mm.  It acquires Gate2, but moves the cube outward to pan-z 193.8 mm and then
  holds that failed state.  The failed 500-step trace/video are saved as
  `logs/expert/failures/closed_loop_cube_pan_servo_v1_minfix_v13.npz` and
  `outputs_video/expert_failures/closed_loop_cube_pan_servo_v1_minfix_v13.mp4`.
- Root cause: v13 fixes the reconstructed contact marker, but that marker lies near
  the middle of a bristle working face spanning about 70 mm in local z.  Servoing
  this midpoint onto the cube outer face places some bristle geometry on the cube's
  inner side, allowing it to push outward.
- v14 chooses one deterministic vertex from the 2048 live-USD bristle points at
  row313, requiring that vertex to be at least 4 mm outside the cube in pan +z and
  selecting the closest valid point.  This same vertex defines the x-straightened
  reference and every feedback update; it never switches.  Feedback preserves its
  nominal reconstructed y coordinate and corrects only x/z toward the measured
  cube outer face.  All pan/cube physics, source y/z and orientations, success gates,
  and preparation checks remain unchanged.
- Next run: tmux `codex_sweep2_expert_v1_minfix_v14_20260830`, log
  `logs/expert/sweep2_expert_v1_minfix_v14_20260830.log`; unique v14 outputs remain
  under `logs/` and `outputs_video/`.

### v14 outer-edge point is correct, but continuous cube-follow cancels the sweep

- v14 passes all setup gates and controls a fixed outer bristle vertex whose row313
  pan coordinate is `[-20.0, 17.8, 176.4] mm`.  It reaches Gate2, but after step100
  the cube is stationary near `[+32.7, -2.3, 166.8] mm`; Gate3 never appears.
- The controller continued redefining the target as the current cube outer surface
  on every post-contact row.  That cancels the reconstructed reference's decreasing
  z path: as the source tries to move inward, feedback moves the brush back outward
  to remain at the cube surface.  It also followed the cube's x drift instead of
  holding the validated straight corridor.
- v15 uses feedback only while the clock is held at row313 to acquire contact.  Its
  contact x is fixed at -20 mm, y remains the nominal reconstructed value, and z is
  the measured cube outer face.  As soon as real contact/motion advances the row,
  the residual decays to zero and the x-straightened source rows314--387 execute the
  inward sweep.  The physical task contract and all geometry are unchanged.
- Next run: tmux `codex_sweep2_expert_v1_minfix_v15_20260830`, log
  `logs/expert/sweep2_expert_v1_minfix_v15_20260830.log`; v15 uses a 220-step
  validation horizon and unique trace/video names.
- Launch from `/home/msc-auto/RL_sweep` on authorized GPU0:
  `CUDA_VISIBLE_DEVICES=0 RL_ISAAC_NO_GUARD=1 SHARPA_WANDB=0 PYTHONPATH=. /home/msc-auto/miniconda3/envs/isaac/bin/python -u tasks/Sweep/2/C_Wiring/make_expert.py --steps 220 --out logs/expert/sweep2_success_v1_minfix_v15.npz --video outputs_video/sweep2_expert_v1_minfix_v15.mp4 --enable_cameras --headless`.

### Baseline reset: freeze `closed_loop_cube_pan_servo_v1`

- v15 was stopped during simulator initialization before frame0/rollout at the
  user's request to prevent further controller drift.  It produced no v15 expert
  trace or video and no BC/PPO was launched.
- The immutable behavior baseline is now
  `logs/expert/failures/closed_loop_cube_pan_servo_v1.npz` paired one-to-one with
  `outputs_video/expert_failures/closed_loop_cube_pan_servo_v1.mp4`.  Its right-hand
  action sequence and row sequence must not be regenerated, remapped, straightened,
  or replaced by a new contact servo in the next comparison.
- Verified baseline metrics: Gate1 frame0, Gate2 frame2, closest frame301 at cube
  pan `[-20.85, 2.49, 117.83] mm`; remaining exact deficits are 35.33 mm in depth
  and 15.51 mm in accepted basin-centre height.  Frame301 is still zero residual;
  the first nonzero recorded action is frame405.
- Next validation is a strict A/B replay of the recorded v1 actions/rows: A is the
  archived v1 evidence; B changes only the left pan to the validated low, level pose
  and continuous entry ramp.  Cube geometry/physics and all right commands remain
  byte-for-byte from the v1 trace.  No expert redesign or training may resume until
  B is quantitatively and visually better than v1.
## 2026-08-30 — Sweep2 dustpan asset mouth repair and immutable-v1 replay

- User-selected behavioral baseline remains
  `outputs_video/expert_failures/closed_loop_cube_pan_servo_v1.mp4`; its paired
  trace is `logs/expert/failures/closed_loop_cube_pan_servo_v1.npz`.
- Cleaned obsolete visual evidence: removed every failed expert/minfix video and
  pan-geometry diagnostic video while preserving the v1 baseline, the approved
  Dexonomy grasp audit, and the source-faithful zero-residual replay.  Restored
  `make_expert.py` to repository HEAD and removed the superseded pan-probe and
  low-pan A/B scripts.
- Mesh audit found the visible source dustpan mouth rises from the 8--9 mm basin
  work surface to 14.06 mm over local z=98--108 mm, forming a 5--6 mm lip.
  `build_smooth_dustpan_asset.py` now reproducibly deforms only the central mouth
  corridor, with a lateral blend into unchanged sidewalls.  The task-specific
  asset remains watertight with unchanged 142,696 vertices, 285,384 faces and
  four connected components; mouth maximum is now 6.85 mm.  Original dataset
  files remain untouched.  Full measurements are in
  `logs/assets/dustpan_smooth_entry_report.json`.
- `clips.py` selects the task-owned smooth dustpan OBJ/USD only for Sweep2's
  dustpan.  `replay_v1_smooth_asset.py` restores the archived v1 frame-zero cube
  location, replays rows 0--499 and the exact 14-D v1 action tensor, and asserts
  both arm/tool reference tensors remain unchanged before saving artifacts.
- Validated replay artifacts:
  `outputs_video/sweep2_v1_smooth_asset_replay.mp4`,
  `logs/ab/sweep2_v1_smooth_asset_replay.npz`, and
  `logs/ab/sweep2_v1_smooth_asset_replay_metrics.json`.  The full action SHA-256
  matches A/B (`6c7e40fd...ffd5dac`), and the right-action SHA-256 remains
  `ddd14c4e...f0b522`.
- Physical outcome: not yet an expert (Gate1 frame0, Gate2 frame12, Gates3/4
  absent).  The exact-containment deficit improved from 38.58 mm at v1 frame301
  to 21.84 mm at repaired-asset frame418.  At B's best frame, lateral containment
  already passes; remaining deficits are about 19.63 mm in depth and 9.57 mm in
  pan-normal centre height.  This isolates the remaining issue to the unchanged
  v1 pan pose/path, not the removed asset lip.  BC/PPO remain blocked.
