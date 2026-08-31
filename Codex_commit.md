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

## 2026-08-30 — Validated Sweep GraspPose and robot-space reference

- Goal: Replace the stale broom grasp and produce a reference that is jointly
  reachable, continuous, task-aligned, and faithful to the reconstruction.
- Changes: Selected the current functional-region broom candidate
  `8_Prismatic_2_Finger__46_16`; added deterministic grasp/yaw and world-yaw
  diagnostics; retained full reconstructed position increments and the maximum
  rotation fraction that passes the continuity gate; replaced greedy IK restarts
  with sparse multi-solution pools, global branch dynamic programming, PCHIP seeds,
  and dense refinement. The validated reference is now generated only after every
  hard assertion passes.
- Reasoning: The old prior was unreachable by 22.3 cm, while full reconstructed
  rotation made at least one key frame unreachable at every tested world yaw.
  A 120 degree rigid world registration plus 8% rotation retention keeps the full
  position curve, brings the broom within 1.45 cm of the easy cube, and leaves a
  small, learnable downward/inward correction for actor BC and residual PPO.
- Validation: Both arms are 200/200 IK-valid. Right/left maximum position errors are
  4.91/4.77 mm and maximum joint steps are 7.89/1.93 degrees. The generated NPZ hash
  is `3a3171625c84e42c92425f0c23ca20d0b3080471c3e3744e3e3d9ec1c35b6b6f`;
  both CPU self-tests, Python compilation, and `git diff --check` pass.
- Remaining: Run the 1-env physical reset/replay gate and record its video once a
  foreign-job-free GPU is available, then generate the successful expert and train.

## 2026-08-30 — Source-faithful Sweep2 reconstruction and replay gate

- Goal: Replace the rejected altered scene with the Pour-style object-driven data
  flow and produce a physically validated source-trajectory replay for user review.
- Changes: Rebuilt the 501-row reference from the 15 Hz RTS object tracks with one
  shared -14 degree scene registration, full 6DoF increments, GraspPose-locked arm
  IK, and separately aligned human-wrist shape references. Fixed attached-tool
  initialization now preserves the task-provided reconstructed object pose instead
  of replacing it with the generic Dexonomy resting pose. Reset settling cancels
  load-induced arm PD sag, and the left tool receives a measured 4.71 mm constant
  runtime registration (also applied to cube start) while preserving every motion
  increment.
- Reasoning: The generic pregrasp path silently replaced the reconstructed broom
  pose and produced 27.8 cm grasp IK error. After preserving the reconstructed pose,
  the same grasp error is 0.07 cm. The remaining left-tool discrepancy was a fixed
  analytic-FK versus Isaac-chain translation, not FixedJoint looseness.
- Validation: NPZ hash `c765ce3c8a36568d47cf6d2e943669bdf7ab99b507046f5d5a505be29edb00b9`;
  both object-driven arms and both human-shape arms have 100% IK validity. The
  1-env 64-step physical smoke passes; reset absolute tool errors are 0.50 mm right
  and 0 mm left after registration, with FixedJoint relative errors near numerical
  zero. Recorded H.264 1280x720, 20 fps, 540-frame replay at
  `outputs_video/sweep2_zero_reference_sourcefaithful_v2.mp4` (SHA-256
  `2537c52b117a092aaedc1c89be63ac8f366837664e85153973578ac8a3c67393`).
- Remaining: User visual acceptance. Do not generate the scripted expert, run BC,
  or start PPO before that acceptance.
## 2026-08-30 — Sweep2 physical contract correction and open-pan probe

- Goal: remove scene-level causes that made a physically valid expert impossible
  before attempting another scripted trajectory.
- Changes: `sweep_env.py` now consumes the reference NPZ's measured contact row and
  cube start, uses a 5 g / 25 mm cube, measures proximity on a dense bristle work
  face, and replaces the dustpan VHACD collision with an explicit open compound
  floor/side/back collider. `progress_batch.py` now requires the cube centre above
  the measured load-bearing basin surface instead of accepting it below the pan.
  `probe_pan_geometry.py` adds a one-environment support and mouth-entry physical
  gate with all NPZ/log/video outputs in approved project directories.
- Reasoning: failed expert traces proved Gate 2 contact but never Gate 3. The old
  random brush proxy overreported distance by 6--7 mm; the old y criterion described
  the mesh underside; and the converted pan's convex decomposition stopped a driven
  cube outside the visible mouth in one physics step.
- Validation: CPU progress self-test passes. One-environment 64-step smoke passes
  with 3 mm FixedJoint assertions and `open pan compound collision:
  disabled_meshes=1; boxes=4`. The first physical support probe passed; the repeated
  open-mouth probe is running in `codex_sweep2_pan_probe_v5_20260830`.
- Remaining: require both support and entry gates to pass before changing or running
  the scripted expert. BC and PPO remain blocked.

## 2026-08-30 — Pre-reset probe evidence and live-USD bristle coordinates

- Goal: remove two measurement errors before planning the row300--387 expert.
- Changes: `probe_pan_geometry.py` now accepts a source hold row, performs a
  low-speed mouth-entry test, records pre-reset success/gates/stability, and stops
  on termination.  `sweep_env.py` now transforms the selected bristle work face
  through the live USD mesh hierarchy into the broom rigid-root frame.
- Reasoning: v7 reached stable containment but was mislabeled after automatic
  reset; source OBJ coordinates are not owned by the converted USD rigid root.
- Validation: both changed entry points pass `py_compile` and `git diff --check`.
  The row313 physical gate is recorded in `Codex_tasks.md` and is pending.
- Remaining: run the row313 support/low-speed-entry gate, inspect its video and
  trace, then build the source-relative expert only if that gate passes.

## 2026-08-30 — Exact pan-frame fixed-cube placement

- Goal: make the low-speed mouth probe test the approved fixed cube location.
- Changes: `_pan_cube_start` now solves pan-local y from the required world table
  height while preserving exact pan-local x/z.  The row313 probe measures the
  settled live pan gap and performs one bounded vertical IK correction before the
  support and entry stages.
- Reasoning: v8 overwrote world z after constructing a pan-local point; with a
  tilted pan this moved the start from local z=125 mm to about 191 mm, so its entry
  failure did not test the ramp.
- Validation: syntax and whitespace checks pass.  v8 still provides a valid basin
  support result; replacement v9 physical entry validation is pending.
- Remaining: require v9 Gate4 before expert construction.

## 2026-08-30 — Remove fixed-cube/ramp initial penetration

- Goal: make the row313 entry gate begin from a physically separated fixed cube.
- Changes: `SweepGeometry.start_outside` is 45 mm, placing the 25 mm cube centre
  at pan-local z=140 mm while preserving the approved fixed-task setup.
- Reasoning: v9's exact z=125 mm placement exposed a second issue: the cube is
  world-axis aligned while the pan is tilted, so its pan-z projected half extent
  is about 18.3 mm.  The prior start penetrated the ramp front at z=119.6 mm and
  generated a 2.64 m/s solver impulse before the scripted entry began.
- Validation: task-contract self-test, `py_compile`, and `git diff --check` pass.
  The v10 one-environment physical support/entry gate is pending.
- Remaining: require pre-reset Gate4 and inspect the trace/video before generating
  the source-relative expert.

## 2026-08-30 — Level the pan work plane and ground the mouth ramp

- Goal: eliminate the real raised-lip failure isolated by the v10 physical probe.
- Changes: the physical probe preserves source yaw but aligns the dustpan work-plane
  normal with world up.  The compound entry ramp is lengthened from 35 to 55 mm,
  lowered at its leading edge to the mesh underside/table height, and joined to the
  +10 mm basin floor at 26 degrees.
- Reasoning: v10 began without penetration and moved only 3.4 mm before stopping at
  z=136.6 mm.  Whole-mesh-minimum calibration alone hid the source pose's roughly
  eight-degree roll/pitch; at the cube corridor the old ramp presented a raised
  vertical face.  The task requires the pan entrance, not an arbitrary mesh corner,
  to sit smoothly on the table.
- Validation: v10 support passed at pan-local `[-5.3, 22.5, 68.5] mm`; entry failed
  cleanly with no solver launch.  Static checks for v11 are pending, followed by its
  one-environment pre-reset Gate4 test.
- Remaining: do not generate the expert or train until v11 passes and its video is
  inspected.

## 2026-08-30 — Stage grounded-ramp activation after pan alignment

- Goal: retain the smooth working-pose mouth without disturbing constraint-safe
  source-row-zero initialization.
- Changes: the entry ramp spawns collision-disabled and the physical probe enables
  it only after pan leveling and height calibration.  Ramp thickness is reduced to
  1 mm and the probe advances 20 settling steps plus a 0--4 mm live-gap assertion.
- Reasoning: v11's lowered ramp contacted the table while the pan was still in its
  tilted transient spawn pose, moving the left tool 16.36 mm and tripping the 1 cm
  registration bound before the intended test began.
- Validation: static checks are pending, followed by the v12 support/entry probe.
- Remaining: production expert/reset wiring must activate the ramp at the same
  table-aligned task boundary; BC/PPO remain blocked.

## 2026-08-30 — Make level-pan IK a fixed task reference

- Goal: represent deterministic scene setup as feed-forward reference rather than
  consuming or enlarging the actor's residual budget.
- Changes: the probe installs the valid table-aligned left-arm GraspPose IK at the
  held task row and commands zero left policy residual.  It still records the joint
  delta and ratio relative to the original reconstructed row for provenance.
- Reasoning: v12's IK was physically valid and precise but measured 1.238 times the
  old left residual envelope.  A fixed dustpan is part of the approved easy task
  setup; asking the actor to learn pan leveling would contradict that design.
- Validation: static checks and the v13 physical gate are pending.
- Remaining: serialize the same fixed-left reference into the expert/task reference
  only after the physical entry gate passes.

## 2026-08-30 — Remove the final rigid lip with a filtered continuous ramp

- Goal: let the table-supported cube enter the verified basin without a sharp
  collision edge.
- Changes: the 60 mm entry ramp begins about 1 mm below the table and ends at the
  +10 mm basin floor.  Only its collision pair with the static table is filtered;
  every cube contact remains physical.  The deterministic cube start moves 5 mm
  farther out to clear contact offsets.
- Reasoning: v13 proved pan leveling, live height, ramp activation, and basin
  support, but the remaining 1 mm leading face stopped the low-speed cube at
  z=138.66 mm before any containment gate.
- Validation: task self-test/static checks and the v14 physical gate are pending.
- Remaining: require v14 pre-reset Gate4 before expert generation.

## 2026-08-30 — Move the ramp front outside the contact-offset envelope

- Goal: ensure the first cube contact is the sloped top surface, not a speculative
  contact against the box front.
- Changes: the filtered ramp is now 80 mm long at 21.5 degrees, begins 4 mm below
  the table, uses a 0.2 mm contact offset, and still ends at the basin floor.  The
  deterministic cube start is z=160 mm.
- Reasoning: v14 stopped at z=141.0 mm because its nominally submerged front and
  cube bottom were still within the combined contact-offset envelope.
- Validation: static checks and v15 physical Gate4 are pending.
- Remaining: no expert/BC/PPO until the entry gate passes.

## 2026-08-30 — Make the isolated entry probe reachable and semantically valid

- Goal: test the unchanged containment/stability gates after actually traversing
  the fixed outside-to-basin distance.
- Changes: the pan-only probe records its tested outside position as `cube_start`,
  preconditions Gates1/2 because the broom is deliberately disabled, and retains
  the real Gate3/4 logic.  The v16 launch uses 0.12 m/s for at most 500 steps.
- Reasoning: v15 moved steadily rather than sticking, but its 200-step horizon
  covered only 8.7 mm.  It also could never earn the broom-proximity Gate2 in an
  isolated no-broom test.
- Validation: static checks and the v16 pre-reset Gate4 probe are pending.
- Remaining: after Gate4, inspect the trace/video and only then build the physical
  broom expert with all four gates earned normally.

## 2026-08-30 — Use continuous force for the ramp topology probe

- Goal: distinguish ramp traversability from the substep semantics of root-velocity
  writes.
- Changes: `probe_pan_geometry.py` supports a persistent world-frame entry force;
  it clears that force immediately on full containment before evaluating Gate4.
- Reasoning: v16 physically climbed from y=-2.4 to +9.1 mm and z=158.5 to 104.6 mm,
  proving the ramp is reached and sloped, but its velocity command was cancelled by
  contact/friction within each environment step.
- Validation: static checks and v17 with 0.05 N are pending.
- Remaining: physical broom expert must earn Gates1--4 without probe preconditions.

## 2026-08-30 — Rebase the near-success broom servo onto the fixed-pan task frame

- Goal: improve the user-selected right-hand expert trajectory instead of producing
  more pan-only diagnostic videos.
- Changes: `make_expert.py` maps source rows300--387 through their relative
  pan-to-broom transforms onto the validated fixed level pan, solves GraspPose IK as
  the feed-forward right-arm reference, fixes the left-arm reference to the validated
  pan IK, enables the continuous ramp, and places the deterministic cube at z=160 mm.
  The existing cube-face servo remains a bounded residual and all physical gates are
  earned normally.
- Reasoning: `closed_loop_cube_pan_servo_v1.mp4` already shows the desired continuous
  sweep and earns Gates1/2; its remaining deficit comes from the moving/high pan and
  incomplete depth, not from absence of right-hand contact.
- Validation: local syntax compilation passes.  Remote static checks, 88-row IK
  coverage, and the one-environment physical expert rollout are pending.
- Remaining: inspect every failed/successful v3 trace and video before BC/PPO.

## 2026-08-30 — Separate the cube from the reconstructed outward preparation

- Goal: preserve rows300--313's visible outward setup without pushing the task cube
  in the wrong direction.
- Changes: expert task setup places the fixed cube at pan-local z=195 mm, 35 mm
  beyond the generic ramp-probe start; no randomization is introduced.
- Reasoning: v3 began with broom distance 15.4 mm and the outward preparation pushed
  the cube from z=160 to about 254 mm before the inward sweep began.
- Validation: syntax compilation is pending; v3 failure artifacts will be retained,
  followed by the otherwise identical v4 physical rollout.
- Remaining: require v4 Gate4 and video inspection before expert acceptance.

## 2026-08-30 — Make the fixed block easier to acquire and slide

- Goal: implement the user's approved easy fixed-block task before expert v4.
- Changes: block size increases from 25 to 30 mm, mass is 6 g, friction changes
  from 0.6/0.5 to 0.25/0.15, restitution is 0.01, and contact offset is 1.2 mm.
  The geometry contract uses the 15 mm half extent, y-centre bounds `[21,34] mm`,
  and a size-aware broom proximity threshold.
- Reasoning: the larger visible face is easier for the source-derived bristles to
  acquire, while lower friction makes ramp traversal easier without changing the
  fixed pan, reference motion, or success semantics.
- Validation: CPU contract self-test, syntax, whitespace, then physical expert v4
  are pending.
- Remaining: consider a rounded convex asset only if the enlarged low-friction cube
  still catches; do not add unnecessary asset complexity first.

## 2026-08-30 — Select the fixed cube from full row300/313 bristle geometry

- Goal: guarantee collision-free outward preparation and reachable outside contact
  using 3D geometry rather than a scalar pan-z guess.
- Changes: expert setup searches fixed central-corridor candidates against the live
  2048-point bristle face at task rows0/13.  It enforces row300 clearance, row313
  reachability/outside direction, and repeats a live reset-distance assertion before
  any physics step.  Failure artifacts now use the unique v5 tag.
- Reasoning: v4's z=195 mm guess still yielded only 12.5 mm centre distance and an
  immediate 9 m/s separation event because the oriented bristle face spans all
  three axes.
- Validation: local syntax compilation passes; remote static checks and v5 are
  pending.
- Remaining: require normal-speed Gate1--4 plus video inspection before acceptance.

## 2026-08-30 — Return to the 25 mm v1 expert and repair only timing/pan height

- Goal: use the user-selected near-success v1 behavior as the direct diagnostic
  baseline instead of changing the block or replacing the right-hand motion.
- Changes: restore the 25 mm / 5 g block and its 0.6/0.5 friction; keep the validated
  fixed low pan and ramp; preserve reconstructed rows300--387; add only an 8 mm
  preparation clearance that decays by row313; start the live-bristle contact servo
  at row313.  The fixed cube is centred at pan-local z=160 mm.
- Reasoning: aligned video/trace evidence shows frame301 is the closest state while
  actions are still zero, whereas v1 feedback starts only at frame405.  The closest
  state is 35.33 mm shallow and 15.51 mm below the basin centre.  The pan is already
  near level but its mouth is too high, so cube enlargement does not address the
  root cause.
- Validation: local syntax compilation passes.  CPU contract, remote syntax/diff,
  task IK/reset clearance, and the physical one-environment rollout are pending.
- Remaining: require Gate4 and inspect the complete video/trace before actor BC.

## 2026-08-30 — Settle the low pan before installing the fixed cube

- Goal: remove the zero-action outward launch observed in v6 without changing the
  source motion, block, friction, controller bounds, or physical success gates.
- Changes: `make_expert.py` now parks the block outside the work area, settles the
  fixed-pan row with its grounded ramp disabled, enables and settles the ramp, then
  installs the one fixed cube start before recorded frame0.  It asserts pan position
  and speed, and saves a bounded diagnostic failure immediately after an outward
  escape rather than wasting the full render horizon.
- Reasoning: v6 actions are zero through step14, while relative speed reaches
  1.695 m/s on step1 and 1.966 m/s on step5.  The cube therefore cannot have been
  launched by the right-arm servo; the moving ramp during reset is causal.
- Validation: v6 failure trace/video saved.  Local syntax is pending, followed by
  CPU/static gates and the one-environment v7 rollout.
- Remaining: inspect v7 trace/video and require Gate4 before actor BC.

## 2026-08-30 — Separate absolute loaded-root tolerance from FixedJoint tolerance

- Goal: let the settled task reach its physical rollout without weakening the
  required 3 mm tool/hand FixedJoint assertion.
- Changes: log the pan position delta vector and rotation; require absolute root
  error below 5 mm, rotation below 2 degrees, and speed below 0.05 m/s.
- Reasoning: v7 stopped before cube installation at 3.49 mm absolute error and
  0.0205 m/s.  This is a different invariant from the already passing relative
  FixedJoint error and should not reuse its 3 mm threshold.
- Validation: local syntax, remote static checks, and v8 physical rollout pending.

## 2026-08-30 — Keep the outward preparation collision-free until row313

- Goal: retain the user-approved source motion while preventing it from pushing the
  fixed cube in the wrong direction before the inward sweep begins.
- Changes: keep a 15 mm pan-normal lift throughout rows300--312 instead of tapering
  the 8 mm lift; return to source height at row313, where contact-gated row advance
  lets the bounded servo align before rows313--387 continue.  Report minimum static
  clearance across every preparation row and abort both depth and lateral escapes.
- Reasoning: v8 proves the low pan/ramp is stable, but action-free steps2--14 already
  move the cube; frame10 reaches centre-distance 12.69 mm and frame15 shows the cube
  displaced outward before feedback.  The taper, not the inward sweep, is causal.
- Validation: v8 failure NPZ/video inspected frame-by-frame.  v9 static and physical
  gates pending.

## 2026-08-30 — Size preparation clearance from the v9 physical trace

- Goal: prevent right-brush contact before the reconstructed inward sweep begins,
  without changing the accepted pan geometry, fixed cube, source sweep, or task
  success contract.
- Changes: `make_expert.py` raises source rows300--312 by 35 mm in the pan-normal
  direction and aborts any rollout where the cube moves more than 2 mm before
  contact row313.  The attempt/controller and failure artifacts advance to unique
  v10 names.
- Reasoning: v9 moves the cube 27.3 mm by step4 with zero actions; its first
  closed-loop action is step15.  A 15 mm nominal lift therefore lacks about 20 mm
  of physical tracking/transient margin and the old run could incorrectly unlock
  the post-contact reference clock through the generic moved gate.
- Validation: v9 trace and montage inspected; local syntax, CPU/static gates, and
  the one-environment v10 physical rollout are pending.
- Remaining: require no pre-contact motion, then Gate4 stable containment and full
  video/trace inspection before actor BC.

## 2026-08-30 — Stabilize the physical brush working point

- Goal: retain v10's validated collision-free preparation while removing the
  lateral push created by a discontinuous contact-point selector.
- Changes: `make_expert.py` uses `raw.broom_contact_local`, the source contact point
  transformed through the live USD hierarchy, for every post-contact target.  It
  removes the 0.75 lateral bias, recomputes feedback each step, caps each Cartesian
  correction at 6 mm, and advances the attempt/artifact names to v11.
- Reasoning: v10 begins cube motion only after feedback, initially makes useful
  inward progress (pan-z 162 to 114 mm), but its nearest-of-2048 working point
  changes across bristle edges and sends pan-x from 1 to 135 mm.  A fixed physical
  point makes the target continuous and x-centred.
- Validation: v10 trace and montage inspected.  Local syntax, remote static gates,
  and the one-environment v11 physical rollout are pending.
- Remaining: require Gate4 and inspect the complete v11 video/trace before BC.

## 2026-08-30 — Align the fixed cube with the reconstructed brush corridor

- Goal: remove edge contact while preserving the source right-hand trajectory and
  the validated low-pan task geometry.
- Changes: `make_expert.py` moves only the deterministic cube pan-x coordinate from
  0 to -20 mm and advances attempt/artifact names to v12.
- Reasoning: v11 pushes inward to pan-z 88.8 mm but escapes laterally.  Source row313
  places its registered contact point at x=-31.9 mm, and the original v1 closest
  state places the cube at x=-20.85 mm.  The old x=0 start therefore contacts the
  bristle edge; x=-20 mm matches observed source geometry and remains inside the
  exact containment corridor.
- Validation: v11 trace, video montage, and source contact-point path inspected.
  Local syntax, remote static gates, and the one-environment v12 rollout are pending.
- Remaining: require Gate4 and full video/trace inspection before actor BC.

## 2026-08-30 — Remove only the causal lateral component of the source sweep

- Goal: convert the now-verified deep push into a contained straight push while
  retaining the reconstructed trajectory's task-relevant motion and pose shape.
- Changes: `make_expert.py` translates the registered brush contact point to the
  fixed pan-x=-20 mm corridor on every row.  It preserves source y/z, orientations,
  rows, timing, pan/cube physics, and success criteria.  It also fixes the row313
  outside-z diagnostic to index the actual row313 point cloud and advances unique
  attempt/artifact names to v13.
- Reasoning: v11 (cube x=0) and v12 (cube x=-20 mm) both end near pan-x +160--172
  mm, while v12 reaches z=26.2 mm.  The source contact point itself crosses 55 mm
  laterally, so eliminating that component is the minimal causal change.
- Validation: v12 physical trace and source contact path inspected.  Local syntax,
  remote static gates, and the one-environment v13 rollout are pending.
- Remaining: require Gate4 and inspect full trace/video before actor BC.

## 2026-08-30 — Control one outer bristle edge instead of the face midpoint

- Goal: retain v13's corrected lateral corridor but make its contact force point
  inward rather than straddling the cube.
- Changes: `make_expert.py` selects one deterministic live-USD bristle vertex at
  row313 that lies at least 4 mm outside the cube.  The same vertex defines every
  reference x correction and feedback target.  Feedback preserves source y and
  adjusts only x/z; attempt/artifact names advance to v14.
- Reasoning: v13 keeps x bounded and reaches Gate2, yet pushes z outward to 193.8
  mm.  Its controlled reconstructed marker is at the middle of a ~70 mm-wide face,
  so placing that marker on the outer surface puts bristles on both sides.  A fixed
  outer-edge vertex removes both midpoint straddling and v10's vertex switching.
- Validation: v13 trace inspected and full failure artifacts saved.  Local syntax,
  remote static gates, and one-environment v14 rollout are pending.
- Remaining: require Gate4 and inspect full trace/video before actor BC.

## 2026-08-30 — Limit feedback to contact acquisition

- Goal: stop the closed-loop target from canceling the reconstructed inward sweep.
- Changes: `make_expert.py` applies the fixed outer-edge servo only while held at
  contact row313, fixes its x target at -20 mm, and clears the acquisition residual
  after the task clock advances.  Source rows314--387 then execute unchanged apart
  from the already justified x-straightening.  Attempt/artifact names advance to v15.
- Reasoning: v14 reaches Gate2 but stalls at z=166.8 mm because continuously targeting
  the current cube outer surface pulls the brush outward as the source moves inward.
  Following measured cube x also reintroduces lateral drift.
- Validation: v14 setup and physical trace through step150 inspected; complete
  failure artifact save is still running.  Local/remote gates and v15 rollout pending.
- Remaining: require Gate4 and inspect full trace/video before actor BC.

## 2026-08-30 — Freeze the v1 right-hand behavior as the regression baseline

- Goal: prevent iterative expert-controller changes from moving farther away from
  the user-approved `closed_loop_cube_pan_servo_v1.mp4` behavior.
- Changes: stopped v15 before rollout and froze further code changes.  The next
  implementation must replay the archived v1 actions/rows unchanged and vary only
  the validated left dustpan pose/entry collision in a strict A/B comparison.
- Reasoning: v1 already gives the best natural right-hand sweep and its closest
  frame is zero residual; later contact-point and lateral-path controllers changed
  the causal variable instead of isolating the pan defect.
- Validation: v15 has no trace/video; its task-owned tmux and Isaac process were
  stopped exactly.  Archived v1 trace/video pairing and metrics were re-verified.
- Remaining: implement and inspect the isolated v1-action low-pan A/B replay before
  any further expert generation, BC, PPO, or Git commit.
## 2026-08-30 — Replace Sweep2's raised-mouth dustpan scan with a smooth task asset

- Added `tasks/Sweep/2/A_Design/build_smooth_dustpan_asset.py` and its generated
  task-owned OBJ/USD under `tasks/Sweep/2/assets/dustpan_smooth_entry/`.  The
  source dataset asset is immutable; 8,203 mouth vertices are lowered by at most
  7.76 mm along a continuous longitudinal profile with a lateral sidewall blend.
- Updated `rl_rebuild/correction/clips.py` so Sweep2's secondary dustpan uses the
  repaired asset and its independent converted USD.
- Added `tasks/Sweep/2/C_Wiring/replay_v1_smooth_asset.py`, which hard-locks the
  archived v1 row/action sequences and asserts neither arm nor tool reference is
  mutated.  Videos remain under `outputs_video/`; traces, metrics and logs remain
  under `logs/`.
- Validation: builder topology/watertight assertions, Python compilation,
  `git diff --check`, one-environment FixedJoint assertions (<3 mm), exact 500-row
  and action-hash assertions, and full rendered physical replay.  The repaired
  asset reduces closest containment deficit 38.58 -> 21.84 mm but does not yet
  satisfy Gate3/4, so it is not labeled or consumed as an expert.
- Cleanup: removed failed expert/minfix/pan-probe videos, removed superseded
  diagnostic scripts, and restored cumulative failed `make_expert.py` experiments
  to HEAD.  Preserved v1 and the two approved reconstruction/grasp videos.
