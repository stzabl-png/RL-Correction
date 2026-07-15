# Grasp trajectory generation + physics simulation (`grasp_traj`)

Turns one synthesized anchored-BODex grasp into a full manipulation --
approach from the human video demo, retarget the hand along the way, close
and squeeze onto the grasp pose, let the contacts settle, then carry: by
default a straight vertical 20cm lift (the simplified pick task), or the
object's recorded demo trajectory (`--carry-style demo`) -- and renders it
as a video in Isaac Sim **with real PhysX physics** on the object (gravity,
collision, a table, tuned friction), not just kinematic replay.

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
3. **Stage poses come from the record.** Stage records (see
   [anchored_bodex.md](anchored_bodex.md#three-stage-grasp-poses)) already
   carry `pregrasp` (opened clear of the object) / `grasp` (the fully
   optimized action, possibly slightly penetrating) / `squeeze` poses
   computed at synthesis time, and the generator consumes them directly
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
6. **Settle.** After the squeeze ramp the wrist parks at the squeeze-end
   pose with the squeeze targets held for `--settle-seconds` so the physics
   contacts converge before any load transfer (appended to the squeeze
   segment: same drive gains, and the carry-only lift metrics stay clean).
7. **Carry** (`--carry-style`, default `vertical_lift`):
   - `vertical_lift`: the wrist rises straight up (world +z) by
     `--carry-lift-height` over `--carry-lift-seconds` with a cosine ease
     (zero boundary velocity, so no squeeze->carry blend or jerk; peak speed
     `pi/2 * height/duration`, kept under `--max-wrist-speed` by growing the
     step count), then holds for `--carry-hold-seconds`. Orientation and
     finger targets stay at their squeeze values. The simplified pick task:
     it removes the demo-path mismatch that was measured yanking grasps
     loose at carry entry (the demo object pose at the carry-start frame
     sits 2-5cm from where the simulated object actually rests). "Up" in
     the camera frame comes from the same DexYCB apriltag extrinsics Stage
     B's frame mapper uses (`--dexycb-manifest`; auto-resolved next to the
     sequences root, then the Stage B default manifest; fails closed if
     missing -- the cameras are tilted, so no axis-aligned fallback is
     close).
   - `demo`: the original behavior -- the wrist follows
     `object_pose_camera(t) @ grasp_root_tf` (recorded object trajectory
     composed with the rigid hand-to-object transform at the CONTACT grasp
     pose), blended out of the squeeze-end pose over
     `--carry-blend-seconds`; the recorded object trajectory is densified
     wherever the composed hand motion would exceed `--max-wrist-speed`.
     The object is carried by contact friction alone.

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
(1.0), `--pregrasp-open-fraction` (1.0),
`--approach-clearance` (0.003m, en-route clearance for the planned
approach), `--open-clearance` (0.05m), `--open-horizon-seconds` (1.0),
`--planner {curobo,linear}` (curobo, fail-closed),
`--max-wrist-speed` (0.25 m/s), `--settle-seconds` (1.0),
`--carry-style {vertical_lift,demo}` (vertical_lift),
`--carry-lift-height` (0.20m) / `--carry-lift-seconds` (2.0) /
`--carry-hold-seconds` (1.0) / `--dexycb-manifest` (auto) for the vertical
lift, `--carry-blend-seconds` (0.3) and
`--carry-start {grasp_frame,pickup_frame}` (grasp_frame) for the demo
carry. `--squeeze-delta` (0.15 rad) and `--near-contact-margin`
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

The hand asset is the **Articulation_Bodex tuned Sharpa USD, verbatim**
(`assets/robots/hands/sharpa_wave/usd/right/bodex_reference/sharpa_right_tuned_instanceable.usd`,
routed through `sharpa_wave_right.yml`; `provenance.json` records the source
hash): 22 finger DOFs plus the asset's own palm world-anchor
`PhysicsFixedJoint` and passive 6-DOF virtual base chain, making the hand a
**fixed-base articulation**. The simulation drives it exactly like the
reference validator (`ref/sharpa_tabletop.py`), by **kinematic anchor
transport**: each trajectory frame rewrites the `/World/Hand` wrapper
Xform's translate/orient ops, the palm anchor follows the wrapper, and the
whole hand teleports rigidly -- there are no root dynamics to stabilize and
no root velocities to command. Articulation solver iterations are 20/10 and
hand link self-collision is on by default (`--hand-self-collisions`). Each
`app.update()` advances sim time 1/60s, so `--sim-steps-per-frame 2` plays
a 30fps trajectory in real time.

**Finger drives**: the reference's per-joint soft PD table
(`SHARPA_PER_JOINT_DRIVES` in the sim module -- MCP 14/2.6, PIP 4.5/0.9,
DIP 2.0/0.45, thumb CMC_FE 26/5, pinky CMC 3/0.7 Nm/rad, converted to USD's
per-degree drive units at authoring), with `maxForce` equal to the asset's
baked tuned effort limits (MCP 1.864, PIP/thumb-IP 0.638, DIP 0.189, pinky
CMC 0.5285, thumb CMC 3.3 Nm), armature `--joint-armature` 0.001, joint
friction `--joint-friction` 0.0. Soft gains keep contact joints out of
permanent force saturation; the caps, not the gains, bound the grip. The
same gains apply in every segment (no close-phase softening -- the gains
are already soft). Targets are commanded the reference way: writing
`drive:angular:physics:targetPosition` (degrees) on the joint prims each
step; a `SingleArticulation` view is kept only for joint-state readback and
the initial joint teleport. Resolved per-joint caps are recorded in
`report.json` under `hand.resolved_max_efforts`.

Finger drives target the synthesized stage poses directly: during squeeze
and carry each joint holds the record's squeeze angle as its position-drive
target, so a blocked joint presses persistently up to its tuned effort cap.
The v13 contact-aware target governor (`--contact-aware-finger-targets`,
default OFF) is retained as an option: it recomputes each drive target
before every physics update and limits it to `--contact-target-lead-rad`
(0.03rad) from the joint's actual position.

### Collision

- **Object**: `--object-collision convex` (default) -- convex decomposition
  (minThickness 2mm, hullVertexLimit 64, hull count
  `--convex-decomp-max-hulls` 32), matching the reference validators. `sdf`
  is the optional alternative: exact SDF triangle-mesh collision at
  `--sdf-resolution` (256), keeping concavities (a mug's opening/handle)
  hollow where hulls would bridge them. 4mm contact / 1mm rest offsets and
  a `--contact-slop` 0.2 penetration deadband (suppresses resting-contact
  jitter/creep; 0 disables) in either case.
- **Hand**: all 26 collider meshes are **baked into the asset** as PhysX
  `convexDecomposition` (minThickness 2mm, hullVertexLimit 64,
  maxConvexHulls 16) with 4mm contact / 1mm rest offsets; the runtime only
  counts them (hard error if an asset ships none). `--hand-rest-offset` is
  an explicit override only -- by default the baked offsets are kept.
  `--hand-usd` swaps in an alternative asset (e.g. the old URDF-import
  `right_sharpa_wave/right_sharpa_wave.usd` for A/B runs).
- **Hand-table collision is filtered out by default**
  (`--hand-table-collision` to re-enable) via UsdPhysics collision groups,
  as in the reference validator: the trajectory may skim the tabletop and
  palm/finger scraping only injects contact noise. The object still
  collides with both hand and table.
- **Debug mode**: `--carry-mode kinematic` turns the object into a pure
  visual prim (no collision) teleported along the reference trajectory,
  validating trajectory/frame-mapping geometry only
  (`final_object_position_error_m` should be exactly 0).

### Friction

`--friction-target both` (default, the `ref/sharpa_tabletop.py` setup): ONE
material at `--friction` (3.0, restitution 0) binds to all 26 hand colliders
AND the object, with `--friction-combine-mode multiply` -- the hand-object
pair therefore sees `friction^2` (an effective 9.0 supergrip) while
object-table sees `friction x 0.5`. `--friction-target object` is the
Articulation_Bodex `open_by_handle`-style alternative: the material binds to
the object only, and every pair the object touches (table included) sees
`--friction`.

### Other Stage B flags

`--object-mass` (explicit override; known YCB objects use their published
mass, otherwise mesh volume x `--object-density` 700 kg/m^3),
`--gravity` (9.81 m/s^2; the reference validator uses 30 as a ~3g stress
load), `--joint-armature`/`--joint-friction` (0.001/0.0, the reference
drive-table values; stiffness/damping/effort caps come from the per-joint
table and are not CLI-tunable),
`--lift-threshold`/`--drop-threshold`
(0.02m/0.005m), `--tabletop-z` (0.0), `--time-steps-per-second` (120, PhysX
substep rate, keep a multiple of 60), `--capture-every` (1),
`--settle-steps` (60), `--video-fps` (trajectory fps),
`--contact-aware-finger-targets` (off; see Hand physics), and
`--contact-target-lead-rad` (0.03rad).

### Outputs and diagnostics

Writes `video.mp4` (H.264), `screenshot.png`, `scene.usd`, and `report.json`
with a `metrics` block: `lifted` (any carry step above `--lift-threshold`),
`sustained_lift` (>= 5 consecutive lifted steps), `grasp_success`
(sustained and not dropped), `object_dropped`, `max_lift_m` /
`final_carry_lift_m` / `num_lifted_carry_steps` /
`max_consecutive_lifted_steps` / `lifted_carry_fraction`, and
`final_object_position_error_m` -- plus `max_joint_tracking_error_rad` /
`joint_tracking_error_per_step` (drive-target vs. actual joint positions;
these compare against the original synthesized trajectory and may be large
when contact correctly stalls a finger). `max_contact_drive_target_error_rad`
compares the bounded target actually sent to PhysX and is the relevant
contact-controller stability diagnostic. A `grasp_success: false` result is a
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
object by 5-13mm -- the recurring root cause below, fixed at the source in
v15).

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
tracking error 0.74rad). Its 10.6cm peak was a brief pinch-ejection rather
than a retained grasp, so it is explicitly not counted as success. This is
the current physics baseline; pass
`--isaac-hand-rest-offset 0.002` to reproduce the former calibration.

### v9 -- isolate staged-grasp visualization rows (2026-07-10)

The anchored-BODex 4x3 renderer authors visibility changes before each stage
row. Isaac's renderer ignored root-Xform inherited visibility for the baked
stage meshes, making every row appear to stack all four hands. The renderer
now authors visibility directly on every mesh and waits for the render/camera
pipeline to consume those edits. Verified on the staged wood-block record:
each front/side/top row contains exactly its selected `pregrasp`,
`raw_grasp`, `grasp`, or `squeeze` hand.

### v10 -- all-sequence staged physics benchmark (2026-07-10)

Regenerated the four-stage anchored-BODex records (40 seeds, 500 iterations)
and ran the right-hand trajectory + dynamic SDF physics on all three
sequences at the v8 contact calibration. All use record stages and stayed
stable: maximum joint tracking error was 0.82rad (can), 0.87rad (mug), and
0.74rad (wood block), with no solver blow-up. The can and mug did not reach
the 2cm lift threshold; the wood block briefly pinch-ejected then dropped.

A ranked-candidate and squeeze sweep did not find a stable wood-block grasp:
candidate 003 never lifted and destabilized after falling; squeeze scales
0.5, 0.8, and 1.0 yielded respectively no lift, a 3.4cm transient lift, and
a 10.6cm pinch-ejection. The remaining blocker is strict grasp feasibility
upstream: every current record is `failed_grasp` (`successful_seed_count: 0`),
not a collision or frame-mapping failure in the trajectory stack.

### v11 -- expanded upstream-search check (2026-07-10)

A versioned 100-seed / 1000-iteration wood-block synthesis search also
produced zero strict successes and a worse top failure (`grasp_error_max`
0.0229 versus 0.0171 for the current 40-seed record). It was deliberately
not promoted. Retention now requires a change to anchored-BODex's grasp
feasibility objective or constraints, not more trajectory playback tuning.

### v12 -- anchored-BODex feasibility ablations (2026-07-10)

The contact-subset and human-guidance hypotheses were tested without changing
any defaults. Restoring all 11 Sharpa contact points improved the wood-block
force-closure residual from 0.0171 to 0.00557 but still produced zero strict
successes; its dynamic trajectory lifted only 3.7cm for two frames before
dropping. A 100-seed / 1000-iteration full-contact run regressed to 0.00652.

Finally, a human-seeded full-contact run with both pose and affordance
guidance disabled also failed (0.00617) and moved 6.9cm from the anchor.
The failure therefore persists across contact-model, search-budget, and
guidance ablations. The current defaults remain the human-guided contact
subset pipeline; a proper improvement needs a new feasible-grasp objective,
not threshold relaxation or a physics-side workaround.

### v13 -- contact-aware finger impedance targets (2026-07-11)

Finger drives previously received the full synthesized squeeze angle once per
trajectory frame. A blocked finger could therefore retain a large position
error for two physics updates, accumulating enough spring/depenetration energy
to pass through the object or pinch-eject it. Squeeze and carry now govern
every joint independently before every physics update: the virtual position
target stays within 0.03rad of the actual joint while preserving the
trajectory target as the closing direction. The soft-gain close remains
unbounded so the fingers can traverse free space and reach contact.

The staged can, mug, and wood-block rollouts were regenerated and visually
inspected. All three now show physical contact followed by pushing, tipping,
or release instead of hand-object pass-through or object launch. Contact-phase
targets were limited to 0.03rad when authored at each physics update; the
post-update contact drive error remained below 0.03rad for the can and mug
and peaked at 0.093rad while the wood block physically tipped. None of the
failed upstream grasps was falsely reported as a successful retained lift.
Leads of 0.06 and 0.09rad were rejected because they moved or destabilized
the object without producing retention.

### v14 -- restore full soft close before governed squeeze (2026-07-11)

The initial v13 controller also governed the close segment. That prevented
free-space joints from traversing quickly enough: the wood-block contact pose
ended with 0.224rad maximum joint error and visibly open fingers. The soft-gain
close now receives the full trajectory target; bounded impedance starts only
at squeeze and remains active through carry. `--final-close-seconds` increased
from 0.4s to 1.0s so the compliant drives can settle onto contact before the
wrist begins carrying.

With the corrected default, the wood-block close ended within 0.038rad of the
recorded contact posture and achieved six consecutive lifted carry frames
before the known failed grasp slipped. Can and mug remained non-retaining but
stable, without hand-object pass-through or pinch-ejection. Per-step
`finger_track.npz` now stores desired, governed, and actual joint positions to
make future controller regressions directly measurable.

### v15 -- in-optimization non-penetration penalty + shy-of-contact stages (2026-07-11)

Following a comparison against UltraDexGrasp (InternRobotics; a BODex +
cuRobo rollout pipeline), two root-cause fixes moved into anchored-BODex
itself:

- **Non-penetration energy in the optimizer** (`--penetration-weight`,
  default 3000): `relu(-sdf)^2` summed over all 37 hand collision spheres,
  differentiable through the exact SDF gradient, added as a guidance cost
  (frozen `bodex_curobo_v2` untouched). The staged cost's symmetric
  `(dist-target)^2` term was indifferent between stopping at the surface
  and overshooting into it; UltraDexGrasp's upstream BODex serializes a
  `pene_error` field our port never had. **Result: converged-pose
  penetration fell from -5.6/-12.6/-4.8mm to under -1.3mm on every top-8
  record of all three sequences** -- the defect that motivated the entire
  downstream repair chain (v5 wrist back-off, v7 stage retreat) is now
  fixed at the source.
- **Grasp stage kept shy of contact** (`--contact-clearance`, 2mm) and
  **squeeze delta taken from the full raw-pregrasp closing motion** (was
  grasp-pregrasp, which collapses to the flat floor when the retreat lands
  on the snapshot), adopting UltraDexGrasp's stance that the tightest
  converged pose is a force direction, not a configuration to physically
  reach. When the whole pregrasp->raw path sits inside the margin (now
  common), the pregrasp snapshot itself becomes the commanded grasp.
  Trajectory generation skips cuRobo for the now-frequently-degenerate
  sub-2mm pregrasp->grasp leg (`interp_short`).

**Result: geometry fixed at the cause; outcomes unchanged.** All three
sequences run stably (contact drive error 0.028-0.039rad) but none lifts:
the wood block is knocked over during the step-in. A seeded margin
ablation (0.5mm vs 2mm contact clearance -- deterministic, identical
optimization) produced **bit-identical** physics metrics, proving the
knock-over happens before the differing close targets matter; the margin
is not the blocker. Two residual leads: (a) still 0 strictly-successful
seeds -- the fingertip-on-top block grasps lack force closure regardless
of penetration; (b) ranking can now surface poor-force-closure seeds
(142123's top-3 have grasp_error ~1.09 while ranks 3+ sit at ~0.035)
because `rank_score` mixes similarity terms -- grasp-error-aware ranking
is a cheap next candidate.

### v16 -- tuned hand asset: BODex colliders + baked per-joint efforts (2026-07-11)

Two properties ported from the proven Articulation_Bodex reference hand,
after auditing both USDs directly (pxr from the isaacsim `omni.usd.libs`
extension, no Isaac boot):

- **Derived tuned asset** (`scripts/isaac/build_tuned_hand_usd.py` ->
  `assets/.../right_sharpa_wave_tuned/`): the audit showed both assets have
  the SAME 26 collider links with identical local-to-link transforms; 16 are
  geometry-identical and only differed in settings (ours plain `convexHull`,
  no offsets -- hulls fill the fingertip-pad concavities), while the 5
  distal + 5 elastomer meshes (the grasping surfaces) genuinely differ.
  The build flattens our asset, swaps in the BODex pad geometry, and bakes
  `convexDecomposition` (2mm minThickness / 64 hull verts / 16 hulls) with
  4mm/1mm contact/rest offsets on all 26. `provenance.json` + `--verify`
  audit (DOF/body/collider counts, settings, geometry hashes, tuned effort
  values) guard regeneration. `--hand-rest-offset` became an explicit
  override (default: keep baked offsets).
- **Baked per-joint effort limits** (`--joint-effort-profile baked`,
  default): the audit's second finding was that our USD already ships the
  exact BODex-tuned per-joint `maxForce` values (MCP 1.864 / PIP 0.638 /
  DIP 0.189 / pinky CMC 0.5285 / thumb CMC 3.3 Nm) -- the runtime simply
  clobbered them with scalar 300 (60 during close). Now the tuned vector is
  read from the asset and applied from the first close step through
  squeeze/carry. **Free-space segments keep the scalar authority**: the
  first attempt applied tuned caps in every segment and the mug sequence's
  fast switch->pregrasp finger swing lagged >1 rad, sweeping the still-
  closed fingers into the object (6.6m ejection during approach); caps are
  for bounding contact forces, not free-space tracking. `uniform` restores
  the old scalars for regression runs.

Also in v16, the **friction setup now matches the reference**: the
high-friction material (2.0/2.0, restitution 0) binds to the object only,
the hand keeps its ordinary asset material, and the object material's
combine mode is `max` (`--friction-combine-mode`) -- so the hand-object
pair sees exactly 2.0. Previously BOTH sides carried 2.0 with `multiply`
combine, giving an effective pair friction of 4.0: strong enough to torque
the object around a single early fingertip contact, a plausible source of
the weird on-contact motions.

A/B on the three sequences (identical Stage A trajectories; old = old asset
+ uniform efforts + both-sides multiply friction, new = full v16): all runs
stable, the mug ejection from the interim every-segment-caps attempt is
eliminated, and pre-carry object disturbance and final object error improve
or hold on every sequence (block: 3.7mm -> 1.8mm disturbance, 0.32 -> 0.24m
final error; meat can: 0.36 -> 0.25m; mug: 0.13 -> 0.11m disturbance,
0.24 -> 0.22m). One finding to be honest about: the old config "lifted" the
wood block (3cm, sustained) -- that lift reproduced under the new
asset/efforts but **disappeared with the friction fix**, i.e. it was an
artifact of the unphysical 4.0 pair friction gluing the block to the
fingertips, not a real grasp. Under reference-faithful physics none of the
three failed-force-closure records lifts, consistent with the known
bottleneck below. `report.json` now records `hand.joint_effort_profile` +
`hand.resolved_max_efforts` and the hand/object friction material setup.

### v17 -- per-pair realistic friction: pad-only material (2026-07-11)

The v16 object-material setup had a scoping problem: a material's combine
mode governs EVERY pair its collider participates in, so the object's
2.0/`max` material also set object-table friction to 2.0 (up from the
multiply-era 1.0) and gave the hard palm/phalanx shells the same grip as the
silicone fingertips. v17 models the physical pairs instead
(`--friction-target pads`, the new default): the high-friction material
(1.2/1.2, restitution 0, combine `max`) binds to only the 10
grasping-surface hand colliders -- the 5 `*_DP` distals and 5 `*_elastomer`
fingertip pads, exactly the links whose BODex geometry v16 ported -- and
nothing else carries a material. Resulting pair frictions: pad-object 1.2
(realistic for silicone elastomer on hard surfaces, 4x margin over the
synthesis QP's mu=0.3 force-closure cone), shell-object 0.5, object-table
0.5 (PhysX default material, `average` combine). `--friction-target object`
restores the v16 behavior with `--friction`. This is strictly harsher
everywhere except the pad contact itself -- it measures synthesis quality
more nakedly. Validation on the wood block (identical Stage A trajectory):
exactly 10 colliders bound (report `hand_friction.bound_collider_paths`),
run stable, and both contact metrics improved vs v16 (pre-carry disturbance
4.0mm -> 2.7mm, max contact drive-target error 0.19 -> 0.04 rad); still no
lift, as expected for a failed-force-closure record.

### v18 -- direct stage-pose targeting by default (2026-07-11)

`--contact-aware-finger-targets` now defaults OFF: squeeze/carry drives
target the synthesized squeeze pose directly instead of the governed
current+0.03rad lead. Rationale: the v13 governor bounded spring load when
efforts were the uniform 300 Nm scalar; under the baked per-joint caps
(v16) contact force is bounded by the caps regardless of target error, so
governing mostly threw away grip -- it throttled the thumb CMC (cap 3.3 Nm
above the 80 x 0.03 = 2.4 Nm governed ceiling) and let saturated joints
back off under disturbance instead of holding their caps. Validation on
the wood block (same Stage A trajectory, v17 friction): stable, no
blow-ups, slightly tighter grip (max carry lift 0.9mm -> 2.7mm) for
slightly more close-phase disturbance (2.7mm -> 5.3mm); the held ~0.79 rad
contact drive-target error is the squeeze overdrive doing its job, not a
tracking fault. The governor remains available via
`--contact-aware-finger-targets` for regression comparison.

### v19 -- Articulation_Bodex asset verbatim + reference-validator physics (2026-07-12 -- 2026-07-14)

The simulation setup was aligned with the proven reference validator
`ref/sharpa_tabletop.py` (Articulation_Bodex), replacing the v16-v18
per-pair-realistic experiments:

- **Hand asset replaced outright** (commits c4b297b + 901c59d): the default
  is now Articulation_Bodex's own tuned USD, copied verbatim to
  `bodex_reference/sharpa_right_tuned_instanceable.usd` and then stripped
  in-file of its palm world-anchor joint and passive 6-DOF virtual base
  chain (13 prims) -- a pure 22-DOF floating-base hand with the tuned
  colliders, offsets, and per-joint effort limits baked in. The v16 derived
  asset (`right_sharpa_wave_tuned/`) and its build script were removed as
  superseded.
- **Friction = reference recipe**: one 3.0/3.0 material, combine `multiply`,
  bound to all 26 hand colliders AND the object (`--friction-target both`,
  default) -- effective hand-object pair friction 9.0. The v17 pad-only and
  whole-hand modes were removed; `object` remains as the A/B alternative.
  The transient compliant-fingertip-material experiment was removed too:
  the hand is fully rigid.
- **Object collision default = convex decomposition** (the reference's
  choice); SDF became the explicit alternative. Object gains the
  reference's `contactSlopCoefficient` 0.2 (`--contact-slop`) penetration
  deadband against resting-contact jitter.
- **Hand-table collision filtered by default** via UsdPhysics collision
  groups (`--hand-table-collision` re-enables), exactly the reference's
  ground-hand mechanism.
- **Gravity parameterized** (`--gravity`, default 9.81; the reference's 30
  m/s^2 stress load is one flag away). Hand self-collision on by default.
- **Live-view shake fixed**: the root was re-teleported to the same frame
  pose every physics substep while carrying forward velocity -- a 60Hz
  sawtooth, visible in the live viewport but hidden in recorded videos by
  fixed-phase per-frame capture. The commanded root pose now interpolates
  across substeps (linear position + slerp orientation) toward the next
  frame, consistent with the finite-difference velocities.
- Dead code removed with the new asset: runtime hand-collision rebuilding
  (baked colliders are now required), runtime virtual-chain deactivation,
  and duplicate report keys.

### v20 -- reference driving method: anchored transport + soft per-joint drives (2026-07-14)

The hand and finger driving switched to the reference validator's exact
method, replacing floating-base tensor-API driving:

- **Asset restored to the pristine BODex USD** (verbatim copy, source md5
  091f0b9d...): the palm world-anchor `FixedJoint` and passive 6-DOF
  virtual chain are baked back in (v19 had stripped them for floating-base
  driving). The hand is a fixed-base articulation again.
- **Wrist = kinematic anchor transport**: the `/World/Hand` wrapper Xform
  is rewritten once per trajectory frame and the anchor follows it. No root
  velocities, no substep pose interpolation, no per-link stability caps --
  an anchored wrist cannot sag, wobble, or need re-pinning, which also
  removes the residual contact-driven rotational jitter of the floating
  method.
- **Fingers = reference soft per-joint drive table**
  (`SHARPA_PER_JOINT_DRIVES`: MCP 14/2.6 ... thumb CMC 26/5 Nm/rad,
  x pi/180 into USD per-degree units; maxForce = baked tuned caps; armature
  0.001, friction 0.0), commanded by `drive:angular:physics:targetPosition`
  writes in degrees. This closes the last big divergence from the
  reference: the previous 80/20 gains were ~4600 Nm/rad effective while
  authored (per-degree units) and left contact joints permanently
  force-saturated. Segment-based gain switching, the `uniform` effort
  profile, and `--joint-stiffness/-damping/-max-force/--close-*` flags were
  removed with it.
- The `SingleArticulation` view remains read-only (joint-state metrics,
  initial joint teleport, the optional target governor).

### Tooling (2026-07-10, commits 3da7b95 + 3946a12)

Videos encoded H.264/yuv420p via ffmpeg; console output reduced to progress
lines + one core-metrics summary (full diagnostics stay in `report.json`).

### Known bottleneck / next candidates

- All three test records remain `failed_grasp` outputs (0 strictly
  successful seeds). Since v15 their poses no longer meaningfully penetrate
  the object (sub-millimeter), and as of v20 the physics stack mirrors the
  proven reference validator end to end: its exact hand asset, driving
  method (anchored transport + soft per-joint drives), friction recipe,
  collision settings, and contact slop. The residual blocker is
  force-closure quality itself: sustained carries most likely require
  strictly-successful upstream grasp records. Remaining deliberate
  differences from the reference: gravity 9.81 vs its 30 m/s^2 stress
  load, per-sequence object masses vs its fixed 0.5 kg, and the trajectory
  task itself vs its staged batch protocol with retrieval-force probes.
- Ranking mixes similarity terms into `rank_score` and can promote
  poor-force-closure seeds over much better ones (see v15) --
  grasp-error-aware ranking is a cheap next candidate.
- Deferred structural option: a compliant 6-DOF *driven* wrist (MagicSim
  `SharpaWaveFloating` gains) instead of tensor-API root teleports, so the
  wrist yields on contact conflict rather than forcing penetration.
