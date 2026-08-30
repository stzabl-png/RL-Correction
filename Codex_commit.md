# Engineering journal

## 2026-08-30 — Sweep task bootstrap

- Goal: Add a fixed-start bimanual sweep task on top of the verified Pour17 residual-RL stack.
- Changes: Created the task-owned engineering records and output-directory policy on branch `sweep-task`.
- Reasoning: Videos, training outputs, datasets, and generated artifacts must remain outside Git history while source and design records remain reviewable.
- Validation: Baseline branch `Step4_RL_Correction` resolved to `4b1daa75ea0625dff10e387f41cfd7320c191d96`; the new branch started from a clean worktree.
- Remaining: Import Sweep2 data, build the scene/reference, validate replay, implement the tracker and training path.

## 2026-08-30 — Evidence gate before environment implementation

- Goal: Freeze the facts that the new implementation must satisfy before reusing
  any predecessor design.
- Changes: Recorded the verified interpreter, dataset timebase discrepancy, asset
  hydration state, GPU ownership, and the measurable failure modes of the deleted
  Sweep prototype.
- Reasoning: The predecessor's shaped reward looked healthy while physical success
  was zero, so implementation gates must be expressed in cube/tool geometry and
  Fixed Joint consistency rather than reward magnitude.
- Validation: Read actual NPZ/video metadata and predecessor trace summaries; checked
  current GPU processes without changing them.
- Remaining: Pure geometry tests, reference generation, physical replay, BC warmup,
  PPO training, and 512-episode deterministic evaluation.

## 2026-08-30 — Sweep physical progress contract

- Goal: Give Sweep2 one simulator-independent source of truth for progress, reward,
  containment, stability, and Fixed Joint frames.
- Changes: Added vectorized pan-frame geometry, four latched task gates, earn-only
  cube progress reward, full-footprint containment, a 0.5-second stability terminal,
  GraspPose-derived joint-frame math, and CPU self-tests.
- Reasoning: Reference following and pan pose alone produced reward farming in the
  predecessor. The new task contract cannot finish without real cube displacement
  followed by stable physical containment.
- Validation: Both `selftest_progress.py` and `selftest_fixed_joint.py` pass under
  the authorized Isaac Python environment; `git diff --check` is clean.
- Remaining: Wire these contracts into the Isaac scene and validate actual reset
  transforms/contact behavior before training.

## 2026-08-30 — Self-contained Sweep clip and P-OBJ reference builder

- Goal: Remove developer-specific data paths and turn the reconstruction into the
  exact bimanual reference consumed by the new task.
- Changes: Repointed both Sweep2 clip registrations to `datasets/sweep_2_better`,
  used the dataset's stated 0.1 kg tool masses, and added an explicit 30-to-20 Hz
  RTS resampler plus GraspPose-locked continuous arm IK builder.
- Reasoning: A shared world transform preserves reconstructed broom/pan geometry;
  solving hands from each tool's GraspPose keeps the physical attachment and
  P-OBJ reference mutually consistent. Human finger motion is intentionally absent.
- Validation: Python compilation and `git diff --check` pass. Runtime IK validation
  is pending because both GPUs are actively occupied by other users.
- Remaining: Run the builder at the first safe GPU slot, inspect its IK report, then
  wire the generated reference into the physical scene.

## 2026-08-30 — Physical Sweep environment wiring (static gate)

- Goal: Implement the fixed-start scene and 14-D residual control path without
  inheriting Pour's task-specific success machinery.
- Changes: Added a Sweep environment with a 1 cm/1 g cube, physical broom/right-hand
  and pan/left-hand FixedJoints, own-hand collision filtering, frozen GraspPose
  fingers, confidence-dependent asymmetric arm residual bounds, exact cube/tool
  policy observations, and immediate stable-containment termination.
- Reasoning: Starting every body in a constraint-consistent row-zero pose prevents
  FixedJoint snap. The task clock plays open-loop only up to nominal brush contact;
  after that it requires real brush proximity or cube displacement.
- Validation: Python compilation and whitespace checks pass. Runtime assertions will
  reject more than 3 mm tool/reference or attachment-frame error at construction.
- Remaining: Generate the reference on a free GPU, run the physical reset audit,
  then tune the deterministic expert push and record zero-residual replay.

## 2026-08-30 — Replay, expert-BC, PPO, and acceptance entry points

- Goal: Make the approved learning sequence executable end to end with artifact
  paths enforced in code.
- Changes: Added zero/checkpoint video recording, a physically validated scripted
  residual expert generator, one-shot actor-only BC warmup, pure on-policy PPO
  training, finite smoke tests, and deterministic 512-episode evaluation.
- Reasoning: Expert files are saved only after the simulator reaches the same stable
  containment terminal used by evaluation. BC runs before PPO and is absent from all
  rollout/update code, matching the approved simplification.
- Validation: Every new entry point compiles and `git diff --check` passes. Video,
  expert, training, and evaluation paths are guarded to `outputs_video/` or `logs/`.
- Remaining: Runtime validation and trajectory tuning at the first safe GPU slot.
