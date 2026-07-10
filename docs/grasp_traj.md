# Grasp trajectory generation + physics simulation (`grasp_traj`)

Turns one synthesized anchored-BODex grasp into a full manipulation --
approach from the human video demo, retarget the hand along the way, close
and squeeze onto the grasp pose, then carry the object along its recorded
trajectory -- and renders it as a video in Isaac Sim **with real PhysX
physics** on the object (gravity, collision, a table, tuned friction), not
just kinematic replay.

## Two-stage architecture

```text
sequence dir + grasp record  --[grasp-synthesis env, torch/CUDA]-->  trajectory.npz/.json   (Stage A)
                                                                       |
                                                            [isaacsim env, PhysX]           (Stage B)
                                                                       v
                                                  video.mp4 / report.json (lift/drop metrics)
```

- **Stage A** (`src/ocir/grasp_traj/`): trajectory generation. Torch/CUDA;
  reuses `anchored_bodex` (demo loading, retargeting, affordance) and
  `bodex_curobo_v2` (SDF contact queries) and cuRobo v2's `MotionPlanner`.
- **Stage B** (`src/ocir/isaac/simulate_grasp_traj.py`): physics playback in
  Isaac Sim. Registers as server task `grasp_traj_simulation` (see
  [Isaac Sim infrastructure](isaac_sim.md)).
- `trajectory_schema.py` is the only module imported by both stages and must
  stay torch- and pxr-free.

## Stage A: trajectory pipeline

The generated trajectory is labeled per step with one of five segments
(`retarget`, `approach`, `close`, `squeeze`, `carry`):

1. **Retarget replay with smooth long-horizon opening.** The human demo is
   retargeted frame-by-frame (fingertip-fit IK). Over the last
   `--open-horizon-seconds` (default 1.0s) of the replay each frame blends
   the retargeted joints toward the wide-open pregrasp (all flexion joints
   scaled toward 0 rad by `--pregrasp-open-fraction`; spread/thumb-rotation
   channels keep their grasp values), reaching fully open exactly at the
   switch frame -- there is no separate in-place opening action next to the
   object.
2. **Switch-frame selection.** Starting `--approach-seconds` before the
   detected grasp frame, the search walks backward through the demo until
   the FULLY OPEN hand clears the object by `--open-clearance` (default 5cm,
   checked against the object SDF with all 37 hand collision spheres) and
   precedes the demo's first hand-object contact frame.
3. **Stage poses come from the record.** Four-stage records (see
   [anchored_bodex.md](anchored_bodex.md#four-stage-grasp-poses)) already
   carry `pregrasp` / `grasp` (contact-retreated) / `squeeze` poses computed
   at synthesis time, and the generator consumes them directly
   (`uses_record_stages: true` in the report). Legacy single-action records
   fall back to the generator's own repair: wrist back-off along the
   approach axis until the open hand clears (`wrist_backoff_m`) plus a
   finger contact projection (`contact_close_fraction`), with
   `--squeeze-delta` building the squeeze target.
4. **Planned transit (cuRobo v2, fail-closed).** The open hand flies from
   the switch pose to an adaptive **preapproach** point -- the record's
   pregrasp wrist backed off along its palm axis just far enough for the
   wide-open hand to clear the SDF (`preapproach_backoff_m`; the
   Articulation_Bodex step-back, computed from geometry instead of a fixed
   distance). The hand is modeled as a floating-base robot
   (`right_sharpa_wave_floating.generated.urdf`, a 6-DOF virtual joint chain
   -- the MagicSim `SharpaWaveFloating` construction), fingers locked wide
   open, the object mesh loaded as a collision obstacle; the planned path is
   arc-length-resampled under the `--max-wrist-speed` cap. cuRobo failure
   here is a hard error (`--planner linear` is an explicit debugging
   alternative, not a silent fallback). The path is SDF-validated en route
   (the arrival tail next to the object is excluded from the threshold
   check).
5. **Close + squeeze (all under the simulation's softened close gains).**
   Step-in: preapproach -> pregrasp wrist while the fingers blend wide-open
   -> the synthesized pregrasp posture. Then the wrist moves pregrasp ->
   contact-grasp pose -- cuRobo again when it finds a plan
   (`pregrasp_to_grasp_planner: curobo`), with straight interpolation as the
   DESIGNED fallback since this leg ends essentially on the contact boundary
   where collision-constrained planning is expected to be infeasible for
   some grasps. Fingers then close in place to the contact-grasp posture
   (`--final-close-seconds`), and squeeze ramps to the record's squeeze
   stage (a bounded drive-force request past contact).
6. **Carry.** The wrist follows `object_pose_camera(t) @ grasp_root_tf`
   (recorded object trajectory composed with the rigid hand-to-object
   transform at the CONTACT grasp pose), blended out of the squeeze-end pose
   over `--carry-blend-seconds`; the recorded object trajectory is densified
   wherever the composed hand motion would exceed `--max-wrist-speed`, so a
   fast recorded carry cannot yank a marginal friction grasp loose. The
   object is carried by contact friction alone.

### Stage A CLI

```bash
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_traj/generate_grasp_traj.py \
  --sequence-dir /path/to/sequences/<sequence_id> \
  --synthesis-out-dir /path/to/anchored_bodex_output/<sequence_id> \
  --out-dir /path/to/grasp_traj_output/<sequence_id>
```

`--synthesis-out-dir` resolves the grasp record from that dir's
`summary.json` (`grasp_json` on success, else `failed_grasp_json`) -- or
pass `--grasp-json` directly. Key flags (defaults in parentheses):
`--fps` (30), `--approach-seconds` (1.0, minimum lead time / plan duration),
`--close-seconds` / `--squeeze-seconds` (0.3), `--final-close-seconds`
(0.4), `--pregrasp-open-fraction` (1.0),
`--approach-clearance` (0.003m, en-route clearance for the planned
approach), `--open-clearance` (0.05m), `--open-horizon-seconds` (1.0),
`--planner {curobo,linear}` (curobo, fail-closed),
`--max-wrist-speed` (0.25 m/s),
`--carry-blend-seconds` (0.3), `--carry-start {grasp_frame,pickup_frame}`
(grasp_frame). `--squeeze-delta` (0.15 rad) and `--near-contact-margin`
(0.003m) only apply to legacy records without stages. Writes
`trajectory.npz` (per-step hand/object pos+quat, finger targets, segment
labels) + `trajectory.json` (switch frame, clearance/transit/stage report,
config).

By default (`--simulate`, on) it also submits the trajectory to Isaac Sim
(`--isaac-mode {server,standalone}` + `--isaac-*` passthrough flags for
every Stage B option). Pass `--no-simulate` to only generate the trajectory.
The console prints progress plus one final core-metrics line; the full
report stays in `report.json`.

## Stage B: physics simulation

```bash
scripts/run_isaacsim_conda.sh scripts/isaac/simulate_grasp_traj.py \
  --mode local \
  --trajectory-dir /path/to/grasp_traj_output/<sequence_id> \
  --out-dir /path/to/some/output/dir \
  --sequence-id <sequence_id>
```

### Hand physics

The Sharpa Wave USD's palm link is rigidly fixed to the world by a
zero-offset `PhysicsFixedJoint` (`root_joint`). The simulation deactivates
that joint, applies `ArticulationRootAPI` at runtime (the asset's own copy
lives in an authoring layer outside the loaded USD's stack), and drives the
resulting **floating-base articulation** through the PhysX tensor API
(`SingleArticulation`): root pose + finite-difference root velocities every
step, finger joints on PD position drives (radians). Reduced-coordinate
solving means links physically cannot separate -- unlike teleporting
authored USD transforms, which this Isaac version does not reliably honor.
Per-link velocity/depenetration caps and articulation solver iterations
20/10 keep contact-heavy squeezes stable. Each `app.update()` advances sim
time 1/60s, so `--sim-steps-per-frame 2` plays a 30fps trajectory in real
time. During the close segment the finger drives are softened
(`--close-joint-stiffness`/`--close-joint-max-force`) so an early-touching
finger stalls instead of shoving the object; full gains return for
squeeze/carry.

### Collision

- **Object**: `--object-collision sdf` (default) -- exact SDF triangle-mesh
  collision at `--sdf-resolution` (256). Convex decomposition (`convex`)
  bridges concavities, which wedges the object inside the closed hand's hull
  volume and makes it follow the hand ("suction") or pop out violently.
- **Hand**: baked convex-hull colliders, inflated by `--hand-rest-offset`
  (0.001m rest offset on every collider -- the "+1mm on the hand collision
  mesh" physical-gripper calibration) so fingers never enter the
  deep-penetration regime.
- **Debug mode**: `--carry-mode kinematic` turns the object into a pure
  visual prim (no collision) teleported along the reference trajectory,
  validating trajectory/frame-mapping geometry only
  (`final_object_position_error_m` should be exactly 0).

### Other Stage B flags

`--object-mass` (explicit override; known YCB objects use their published
mass, otherwise mesh volume x `--object-density` 700 kg/m^3),
`--friction` (2.0, both sides, multiply combine),
`--joint-stiffness/-damping/-max-force/-armature/-friction`
(80/20/300/0.01/0.05), `--lift-threshold`/`--drop-threshold`
(0.02m/0.005m), `--tabletop-z` (0.0), `--time-steps-per-second` (120, PhysX
substep rate, keep a multiple of 60), `--capture-every` (1),
`--settle-steps` (60), `--video-fps` (trajectory fps).

### Outputs and diagnostics

Writes `video.mp4` (H.264), `screenshot.png`, `scene.usd`, and `report.json`
with a `metrics` block: `lifted` (any carry step above `--lift-threshold`),
`sustained_lift` (>= 5 consecutive lifted steps), `grasp_success`
(sustained and not dropped), `object_dropped`, `max_lift_m` /
`final_carry_lift_m` / `num_lifted_carry_steps` /
`max_consecutive_lifted_steps` / `lifted_carry_fraction`, and
`final_object_position_error_m` -- plus `max_joint_tracking_error_rad` /
`joint_tracking_error_per_step` (drive-target vs. actual joint positions;
values exploding past ~1 rad indicate solver instability, small fractions
of a rad are normal contact stall). A `grasp_success: false` result is a
genuine, useful finding: it means the synthesized grasp does not hold the
object under real physics (common for grasps that did not reach
anchored-BODex's own strict force-closure success), not necessarily a
simulation bug.

---

## Changelog

Design iterations in chronological order, with the observed result of each
-- kept so we can trace which changes produced the best behavior. Test set:
the three available DexYCB sequences 20200709_151724 (wood block),
20200709_142123 (potted meat can), 20200709_150949 (mug), all with
`failed_grasp` records from anchored-BODex (their grasp poses penetrate the
object by 5-13mm -- the recurring root cause below).

### v1 -- initial implementation (2026-07-09)

Kinematic-wrapper wrist: palm link marked PhysX-kinematic, ancestor Xform
teleported every frame, finger targets written as per-step USD `DriveAPI`
degrees. Stage A closed the fingers from a barely-relaxed pregrasp while
flying standoff -> grasp, with the standoff derived from the *switch* pose.

**Result: broken.** Hand links visibly separated (a kinematic link inside a
live-ish articulation destabilizes the joint solver), fingers jittered at
high frequency, and the close sweep knocked every object over. No lifts.

### v2 -- articulation-driven hand + trajectory geometry fixes (2026-07-10)

- Stage B: deactivate `root_joint`, apply `ArticulationRootAPI` at runtime
  (discovered the loaded USD has none -- it lives in an unloaded authoring
  layer), drive the floating-base articulation through the PhysX tensor API
  (root pose + finite-difference velocities; joint targets in radians).
- Physics-time accounting fixed: each `app.update()` advances 1/60s
  regardless of `timeStepsPerSecond` (substep granularity only), so
  velocities use the *simulated* dt and `--sim-steps-per-frame 2` = real
  time for 30fps.
- Stability caps: `maxDepenetrationVelocity` 2 m/s, per-link velocity caps,
  `maxJointVelocity`, articulation solver iterations 20/10, self-collisions
  off, matching caps on the object (without them a squeeze pinch ejected the
  object at 36 m/s). Kinematic carry-mode object made collision-free
  (fingers squeezing a per-frame-teleported infinite-mass collider detonated
  the articulation after ~1s of contact).
- Stage A: standoff derived from the *grasp* pose (was: switch pose, which
  made the close sweep in sideways); 0.25 m/s wrist speed cap;
  squeeze->carry pose blend (27mm boundary jump -> 3mm).

**Result: hand mechanically sound** (joint tracking <= 0.4 rad, zero
explosions, kinematic-mode object error exactly 0), object no longer
knocked over by the approach -- but the near-closed fingers still pushed the
object during close, and nothing lifted.

### v3 -- wide-open pregrasp, close in place (2026-07-10)

Pregrasp opens all flexion joints toward 0 rad
(`--pregrasp-open-fraction`); the open hand flies all the way to the grasp
wrist pose and only closes there.

**Result: first genuine wrap acquisitions** (fingers visibly around the
block, still upright). Block rose ~6-8mm then slipped during carry; harder
squeeze (0.25-0.35 rad) ejected the object instead of holding it.

### v4 -- retreat/open + cuRobo transit planning (2026-07-10, refined away in v6)

Discrete retreat along the palm axis until open-hand clearance, in-place
opening there, then a cuRobo v2 `MotionPlanner` transit (floating-base
6-DOF virtual-chain URDF, fingers locked open, object mesh as obstacle)
replacing straight-line interpolation; straight-line + via-point kept as
automatic fallback. cuRobo integration required: passing a self-collision-
free metrics config (the `self_collision_check=False` flag does not reach
the metrics rollout that gates success), task configs passed as file paths
(dicts are mutated in place by multiple consumers), +/-2pi virtual joint
limits (rpy-chart boundary IK failures), arc-length resampling (the
interpolated result contains dwell frames).

**Result: collision-free arcing transits** (5-8cm SDF clearance along the
path). Opening still happened too close to the object on some sequences.

### v5 -- contact-projected close + compliant fingers (2026-07-10)

Measured the grasp records against the object SDF: grasp-pose clearance
-5.6/-12.6/-4.8mm across the three sequences -- even the *palm* placement
penetrates. Added: wrist back-off along the approach axis until the open
hand clears (6-18mm on the test set), two-stage close to the SDF
zero-clearance posture (fast to 3mm margin, slow final), synthesized
grasp/squeeze joints demoted to bounded force requests, close-segment drive
softening (stiffness 20 / max force 60), reach-leg SDF check.

**Result: object stays upright through close and early carry** with visible
surface contact; carry drift down to 0.20-0.36m (from multi-meter
ejections); still no sustained lift.

### v6 -- smooth long-horizon opening + collision overhaul (2026-07-10, commits 03a798c + 8634a38)

- Opening blended smoothly over `--open-horizon-seconds` into the tail of
  the retarget replay (no discrete retreat/open phases at all); switch frame
  selected for *open-hand* clearance (`--open-clearance` 5cm), so the hand
  is always fully open well before it nears the object.
- Object collision switched to **SDF triangle mesh** (convex decomposition
  bridged concavities -- the object got wedged inside the closed hand's hull
  volume, following the hand around, "mysteriously sucked up", then fired
  out at the depenetration cap). Hand colliders inflated by a **2mm rest
  offset** (physical gripper calibration).

**Result: best so far -- first genuine lifts.** Wood block lifted ~5cm and
meat can ~2cm above resting height during carry (then dropped -- the
repaired failed-grasp poses are necessarily shallower than synthesized);
mug rose ~2cm briefly. No suction, no interpenetration, no ejections, no
solver warnings on any sequence. Note: results on marginal grasps are
run-to-run sensitive (the same trajectory produced lifted=true at 0.170m
max z in one run and lifted=false at 0.133m in another).

### v7 -- synthesis-provided four-stage poses + staged approach (2026-07-10)

The stage poses moved UPSTREAM into anchored-BODex itself (see
[anchored_bodex.md](anchored_bodex.md#four-stage-grasp-poses)): every record
now carries `pregrasp` (true mid-optimization snapshot at the ~1cm-standoff
stage, via an optimizer-core subclass reproducing BODex's `save_qpos`),
`raw_grasp`, `grasp` (SDF-retreated to non-penetrating contact along the
pregrasp->raw path), and `squeeze` (Articulation-BODex per-joint
extrapolation with a 0.15 rad flexion floor). The generator consumes them
directly and its own wrist/finger contact projection became a
legacy-records-only fallback. Segment b became: cuRobo plan (fail-closed,
per the concurrent hand-tuned approach rework) to an adaptive
**preapproach** (pregrasp wrist backed off along its palm axis until the
wide-open hand clears -- a fixed-goal plan to the pregrasp wrist itself
failed because that pose is only collision-free with its own near-closed
fingers), step-in blending fingers open->pregrasp, a second cuRobo attempt
for the short pregrasp->grasp leg (straight interpolation as designed
fallback at the contact boundary), in-place close to the contact posture,
squeeze to the record's stage. Concurrent hand edits also added recorded-
carry densification under the wrist speed cap, published YCB masses, and
richer lift metrics (`sustained_lift`, `grasp_success`, ...).

**Result:** on the fresh 40-seed/500-iter 151724 record, cuRobo planned
both legs (en-route clearance 2.5cm), the block stayed upright through
close/squeeze, lifted ~3.1cm briefly, then dropped
(`grasp_success: false`) -- acquisition mechanics all work; sustained hold
still bounded by grasp-record quality (0 strictly-successful seeds on this
sequence).

### v8 -- stage-compatible contact offset (2026-07-10)

The stage-driven record's `grasp` pose is deliberately at 0mm SDF clearance.
The v6 2mm hand rest offset, combined with the object's 1mm rest offset,
created a 3mm physical separation requirement that the new contact pose did
not satisfy. Close therefore stalled by ~0.81rad, and restoring full
squeeze/carry gains caused a 101rad joint-tracking blow-up.

The default hand rest offset is now **1mm** (2mm total pair separation).
On the staged wood-block record this eliminated the blow-up (maximum joint
tracking error 0.74rad) and produced a 10.6cm peak lift before the known
marginal failed grasp slipped. This is the current physics baseline; pass
`--isaac-hand-rest-offset 0.002` to reproduce the former calibration.

### v9 -- isolate staged-grasp visualization rows (2026-07-10)

The anchored-BODex 4x3 renderer authors visibility changes before each stage
row. Isaac's camera could capture its previous frame before those changes had
been rendered, making every row appear to stack all four stage hands. The
renderer now advances the app twice after each visibility switch. Verified on
the staged wood-block record: each front/side/top row contains exactly its
selected `pregrasp`, `raw_grasp`, `grasp`, or `squeeze` hand.

### Tooling (2026-07-10, commits 3da7b95 + 3946a12)

Videos encoded H.264/yuv420p via ffmpeg; console output reduced to progress
lines + one core-metrics summary (full diagnostics stay in `report.json`).

### Known bottleneck / next candidates

- All three test records are `failed_grasp` outputs whose synthesized poses
  penetrate the object; sustained lifts most likely require
  strictly-successful upstream grasp records.
- Deferred structural option: a compliant 6-DOF *driven* wrist (MagicSim
  `SharpaWaveFloating` gains) instead of tensor-API root teleports, so the
  wrist yields on contact conflict rather than forcing penetration.
