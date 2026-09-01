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

## 2026-08-30 — Match the physical dustpan mouth to the repaired asset

- Goal: test whether the remaining near-entry failure is caused by the invisible
  collision surface while replaying the immutable v1 behavior byte-for-byte.
- Changes: `sweep_env.py` replaces the disabled 80 mm proxy ramp with an enabled
  28 mm local wedge spanning z=80--108 mm and y=8.5--6 mm; its floor joins the
  wedge continuously at z=80 mm.  `replay_v1_smooth_asset.py` asserts the collider
  is enabled and records its geometry in metrics.
- Validation: Python compilation, `git diff --check`, analytic endpoint assertions,
  one-environment physical reset, FixedJoint assertions (<3 mm), exact row/action
  hashes, 500-frame video, and trace inspection.  The best exact-containment
  deficit improves 21.84 -> 19.06 mm, but Gate3/4 remain false.
- Remaining: this isolated collision fix is useful but insufficient.  Preserve the
  resulting video for user review; do not change the right controller or start
  BC/PPO until the next controlled variable is approved.

## 2026-08-30 — Reject pre-roll settling that breaks the near-success state

- Goal: remove the visible initial cube impulse by parking the cube while the
  row-zero tools and short entry collider settle.
- Experiment: enabled the entry only after two attachment-settle passes, restored
  the exact archived cube pose with zero velocity, and replayed all 500 immutable
  v1 rows/actions.
- Result: early displacement fell 59.06 -> 30.84 mm but remained visible, while
  best containment regressed 19.06 -> 38.20 mm.  The initialization transient is
  part of the state that lets the unchanged v1 trajectory nearly enter later.
- Resolution: reverted the experimental code to the validated physical-entry
  replay, retained trace/metrics/log for diagnosis, and deleted the failed video.

## 2026-08-30 — Test a right-only inward extension of immutable v1

- Goal: clear the remaining 19.06 mm mouth-depth deficit without changing the
  left hand, cube, pan asset/collider, physics or success criteria.
- Changes: added `replay_v1_right_depth_extension.py`, which reconstructs v1's
  nominal cumulative residual and uses live-anchor ArmIK to add one smooth
  pan-local right-hand depth offset.  It records plan, bound and action hashes in
  task-owned metrics and never mutates source references.
- Validation: 25 mm passed the original envelope and reduced deficit to 4.63 mm.
  The user-authorized 40 mm run stayed within URDF limits and a 3 deg maximum
  confidence-envelope excess, but achieved only 4.78 mm.  Both recorded 500 rows,
  kept every left action zero, and failed Gate3/4.
- Remaining: extra depth command has saturated physical progress; test lateral
  brush/cube contact alignment as a separate next variable before expert/BC/PPO.

## 2026-08-30 — Redefine terminal success at entry and export two experts

- Goal: encode the user-approved operational definition that the sweep succeeds
  as soon as the cube enters the dustpan, then stop instead of allowing it to fall
  back out later.
- Changes: added a geometrically guarded `entered` signal in
  `progress_batch.py`; retained `fully_inside` unchanged for strict diagnostics;
  made Gate3 and Gate4 trigger on the same first-entry step; updated the CPU
  contract tests; parameterized the right-depth replay and truncated its trace and
  video exactly at first success.
- Validation: CPU self-test PASS.  Formal 25 mm replay succeeded at frame384 and
  formal 40 mm replay at frame382.  Each trace is an exact source-row prefix,
  contains one terminal `entered` frame, has simultaneous Gate3/4, has no left
  actions, and has the same frame count as its 20 fps 1280x720 video.
- Artifacts: `outputs_video/sweep2_expert_entry{25,40}_v1.mp4`; all matching
  traces, metrics and run logs are under `logs/expert/`.
- Git state: not committed because the touched task-contract files overlap the
  preserved pre-existing dirty Sweep work; no unrelated change was staged.

## 2026-08-30 — Sweep2 arm-residual RL pipeline

- Extended Sweep2 state/reward design and added a strict 4-second Actor freeze.
- Added separate Critic sizing and policy-update masks to the shared PPO stack.
- Replaced actor-only BC with weighted Actor BC plus success/failure Critic
  return regression; added no-video physical transition collection and manifest.
- Added `SWEEP_TRAJECTORY_PLAYBOOK.md` and `DECISIONS.md` so future reconstructed
  Sweep trajectories reuse the validated asset/expert/training workflow.
- Validation completed so far: Python compilation, `git diff --check`, physical
  recollection of two successful experts plus canonical failure, and 1-env PPO.

## 2026-08-30 — 3M 训练诊断与中文流程收敛

- 正式训练每跨过 3M agent steps 保存不可变 checkpoint 和指标 JSON，并由
  `autorecord_sweep.sh` 生成 `outputs_video/` 下的确定性视频以及 run 内逐步 NPZ trace。
- `record_sweep.py` 新增 trace 输出，包含 observation、privileged state、action、reward、
  row、Gate、success、cube-pan、Actor mask 与累计 residual。
- 正式 run 写入 `world.json`，固定 reference、簸箕资产、transition hash、policy I/O、
  固定方块位置、4 秒前缀和成功定义。
- 将 `Codex_tasks.md` 和 `SWEEP_TRAJECTORY_PLAYBOOK.md` 重写为中文，只保留最终确定的
  Sweep2 数据流以及可复用于新 Sweep 重建轨迹的处理和训练方法。
- 修复多环境簸箕 collider 的 clone xform 重复以及逐环境 full-stage traversal 的平方复杂度。
- 静态验证：`py_compile`、`bash -n`、`git diff --check` 通过。正式 1024-env run 已在
  tmux `sweep2_ppo1024_seed42_v3_20260830` 启动。

## 2026-08-30 — 将 Sweep2 成功处理过程固化为详细案例

- 在 `SWEEP_TRAJECTORY_PLAYBOOK.md` 增加完整 Sweep2 case study，记录从错误回放、
  canonical v1 near-miss 选择、簸箕 mesh/collider 根因、严格 A/B、25/40 mm 右臂
  depth extension、`entered` 成功定义，到 transition、warmup 和 PPO 的完整因果链。
- 记录被证伪的方案及其量化结果，包括重叠 proxy ramp、误删早期接触状态、pan-only
  probe 冒充 expert、同时改多变量，以及 commanded depth 不等于 cube penetration。
- 明确区分可复用的诊断流程与不可照搬的 Sweep2 专属数值，供后续每条新 Sweep 重建
  轨迹独立生成 reference、expert、transition 和 residual policy。
- 验证：Markdown fence 平衡、九个案例小节齐全、关键数值与 expert metrics、工程台账和
  mistakes 记录交叉核对，`git diff --check` 通过。

## 2026-08-31 — 修复 3M 诊断并恢复正式训练

- 修复 `train_sweep.py` 诊断 JSON 路径缺少 `torch` import 的 NameError。
- 新增 resume-only `--initial_agent_steps` / `--initial_epoch`，并让 PPO 保留全局 step，
  避免恢复后从零计数或重新进入 Critic-only warmup。
- PPO checkpoint 从此保存并恢复 optimizer state、agent steps、epoch 和 learning rate；
  旧 checkpoint 缺少这些字段时继续兼容。
- 从已成功写出的 `diag_0003M.pth` 补齐 3M JSON、deterministic MP4 和 rollout NPZ，
  再从 3,014,656 steps / epoch91 恢复 1024-env PPO。
- 验证：Python compilation、`git diff --check`、checkpoint/JSON 非空、视频 418 帧、
  rollout NPZ 非空；3M deterministic rollout 为 Gate `[1,1,0,0]`。

## 2026-08-31 — 统一 Sweep 训练节点产物并清理冗余快照

- 目标：让每个保留训练步数都有自包含的 checkpoint/debug 包与统一视觉目录，后续录像固定同时产出斜视视频和桌心俯视连续帧。
- 重要改动：`train_sweep.py` 将每 3M checkpoint/metrics 写入 `logs/checkpoints/Sweep2__<YYYYMMDD>_policy_<XXXXM>/`，并关闭通用 `ep_* / best / last` 冗余保存；`autorecord_sweep.sh` 将 `policy.mp4` 与 `topdown_frames/` 写入 `outputs_video/<同名节点>/`，将 `rollout.npz` 与 `record.log` 写回 checkpoint 节点；`record_sweep.py` 支持失败策略保存 terminal 前连续俯视帧。
- 迁移与清理：只保留 3M、12M、15M、48M 四个训练节点；迁移前后 checkpoint、rollout、record log 和 MP4 均通过 SHA-256 对照。删除其他步数视频、diagnostics 和所有周期/reward checkpoint；完整保留 `logs/expert/`、TensorBoard、run metadata 以及四个批准的基准/专家视频。
- 补录验证：3M 保存第 406–417 帧并明确为失败 terminal；12M、15M、48M 分别保存首次成功前第 257–268、221–232、211–222 帧。每个节点均为 12 张互不相同的 1280×720 PNG，人工抽查确认 cube、dustpan mouth 和 broom 同时可见。
- 验证：`py_compile`、`bash -n`、`git diff --check`、节点文件清单、JSON checkpoint 路径、PNG 数量/唯一 SHA-256、MP4/NPZ 非空和任务进程退出检查。
- Git 状态：未提交、未推送；保留工作区原有未提交 Sweep 改动。

## 2026-08-31 — 完整整理 Sweep Residual RL 训练算法

- 将 `SWEEP_TRAJECTORY_PLAYBOOK.md` 的算法部分扩写为可独立执行的训练说明，明确 reference 加累计 residual 的控制形式、confidence step/deviation envelope、4 秒 scripted prelude 和接触同步 reference row。
- 按真实实现记录 asymmetric Actor-Critic：Actor 使用 191 维 observation 与前 8 维 task embedding，Critic 使用 191+22 维完整真值；补充网络宽度、Gaussian exploration、normalization 与 deterministic inference。
- 逐项整理四级 Success Tracker、ratcheted reward、confidence tracking、人手 shape prior、动作/左臂正则，以及 expert transition、带权 Actor BC、成功/失败 Critic warmup 和 pure on-policy PPO 更新。
- 增加正式启动、3M 诊断包、Gate 漏斗排错、checkpoint 恢复、512 回合 deterministic 验收和“达到 50% 不会自动早停”的操作说明。
- 验证：章节结构、Markdown fence、关键维度/超参数/路径与 `sweep_env.py`、`progress_batch.py`、`bc_warmup.py`、`train_sweep.py`、`ppo.py`、`ppo_sweep.yaml`、`eval_sweep.py` 交叉核对；`git diff --check` 通过。

## 2026-08-31 — Deep20 成功条件与末段推动奖励

- 目标：把第一阶段“cube 中心刚进入”提升为“cube 完整进入后，入口侧整面再深入 20 mm”，并让 reward 对最后几毫米和扫把实际推动继续提供梯度。
- 重要改动：`progress_batch.py` 保留 Gate3 `entered` 并将 Gate4 改为 `deep_inside`，新增 ratcheted `deep_progress`；`sweep_env.py` 用扫把距离、后方方向、横向和高度对齐加权 episode 新 inward depth，替换一次性速度峰值 push shaping；新增 Deep20/fully-inside/扫把辅助诊断。
- 专家：Actor 只 BC 40 mm；Critic 使用按新 reward 重采的 25/40 mm near-success 和 canonical failure，不使用 48M rollout，也不恢复 48M checkpoint。
- 录像：`record_sweep.py` 在真实 Deep20 terminal 后只为斜视 MP4 重复最终帧 2 秒，并补充 `next_*`、`entered/fully_inside/deep_inside/deep_margin/deep_progress/broom_assisted_progress` trace。
- GPU 安全：修复 Sweep 自动录像绕过 guard 的旧行为；训练现在响应 `pause.request`，只在 epoch 边界释放本任务槽位，录像退出并静置后恢复。用户明确授权与 `feiyang` 共享 GPU0 且不得中断外部进程；流水线据此直接启动，但本任务训练与录像之间仍禁止重叠。
- 验证：Python compilation、CPU Success Tracker contract、`bash -n` 和 `git diff --check` 已通过；GPU transition/smoke 与 1024-env 已在 task-owned tmux 排队等待安全计算槽位。

## 2026-08-31 — 新 Codex 统一交接入口

- 新增根目录 `CODEX_HANDOFF.md`，将分散在工程日志、Playbook、旧 Pour 指南和运行目录中的当前事实整理为单一入口。
- 文档明确 Deep20 目标与判据、reference 到 residual PPO 的数据流、Actor/Critic 输入与 warmup、1024-env 当前 run、3M 产物规范、512 回合验收、GPU 共享边界和最近待办。
- 同步 `Codex_tasks.md` 的实时状态：Deep20 transition 和 1-env smoke 已完成；1024-env 已完成初始化和 BC/critic 数据预热，BC checkpoint 已落盘，当前处于 critic-only warmup，禁止重复 launch。
- 此次只修改文档，不修改训练代码、不重启进程、不触碰其他用户 GPU 作业。

## 2026-08-31 — 回退到旧 15M entry-success 训练方案

- 按用户决定撤销 Deep20 作为训练目标，恢复首次几何有效 entered 时 Gate3/Gate4 同步成功并立即终止；Deep20 几何量仅保留为诊断，不再参与成功或 task reward。
- 训练重新使用 logs/expert/transitions/ 中旧方案的 25/40 mm success 与 canonical failure transition，从随机初始化重新做 Actor/Critic warmup，不加载旧 checkpoint。
- 保留诊断基础设施与录像展示逻辑：真实 rollout 在成功当步终止，MP4 额外重复终止帧 40 次（20 FPS 下 2 秒）。修复录像摘要误读 terminal reset 后 Gate 的问题。
- 验证：Python compile、CPU tracker self-test、git diff --check 通过；旧 15M checkpoint 在恢复环境中于 step232 重现 entry success，trace Gate [1,1,1,1]，视频 233 个物理帧后追加 40 帧。

## 2026-08-31 — 修复成功录像冻结帧 off-by-one

- 发现 DirectRLEnv 在 terminal step 返回前已经自动 reset；旧 recorder 在 env.step 后渲染，误把 reset 画面 frame_0232 当作成功冻结源。
- recorder 现在检测 terminal 后跳过 post-reset render，保留最后一个有效物理画面 frame_0231，并以该画面追加 40 帧。
- 回归验证：topdown 最后一帧为 frame_0231.png，MP4 共 272 帧（232 个有效画面 + 40 帧冻结），terminal trace Gate [1,1,1,1]。首次 v1 初始化 run 已停止，改用独立 v2 目录从头重启，避免日志混写。

## 2026-09-01 — Success 收紧为 fully-inside，并接入 15M Actor expert

- 用户指出 3M 冻结画面只显示浅进入。核查发现 recorder 未显示真实 terminal，且旧 entered 判据允许中心跨口但 footprint 未完整进入。
- Gate3 保留 entered；Gate4/success 改为 fully_inside。新增从中心入门到完整 footprint 清口的 earn-only full_progress，Deep20 仍不启用。
- recorder 在专用 suppress-reset 模式下渲染真实 terminal state；topdown 3M 回归 frame_0300 显示 fully_inside，trace cube_pan.z=80.793 mm。
- Actor BC 改为同时学习 40 mm fully-inside expert 与旧 15M fully-inside rollout；Critic 使用两条 full-success、25 mm near-success 和 canonical failure。所有 transition 按当前 reward 重采。
- 主视频相机改为机器人左前方略高的中景，覆盖上半身、双臂和桌面操作区。
