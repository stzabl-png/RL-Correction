# Why our previous in-hand-rotation design failed — and why the rebuild succeeds

**TL;DR.** The previous stack (`../sharpa_isaaclab`) and the rebuild (`./`, this dir) BOTH use the same
high-level idea — *trajectory-replay a GraspXL grasp to get the initial pose, then RL to rotate*. The
previous one fought reward-hacks and fragility for many sessions and never cleanly generalized; the
rebuild grasps the **same GraspXL objects** via trajectory replay and learns a genuine held+rotating
policy at full gravity in ~25 min. So the failure was **not** the trajectory replay — it was the
*surrounding RL design*. Concretely, four things the previous stack got wrong, each fixed in the rebuild:

## The decisive evidence
The hardest object in the previous stack's saga was the round ball `002aa185…` — across ~5 sessions it
produced a *reward-hacked* "rotation" (object falls into a palm cradle and rolls passively), needed an
elaborate strict-fall + gravity-curriculum + grasp-squeeze patchwork, and only reached ~80% gravity with
overnight warm-start juggling. The rebuild, on the **same ball, via the same trajectory-replay init**,
reaches **full gravity** with the object **held 86.8%** of the active phase and **rotated at 0.79 rad/s**
— from scratch, one ~25-minute run, no babysitting. (Eval: `logs_eval_gxreplay.log`.)

## The four root causes (previous → rebuild)

1. **Control: stiff implicit position-PD → explicit torque control (+ PD/friction/COM/mass DR).**
   The previous stack drove 22 joint *position targets* through IsaacLab's implicit actuator (stiff
   70/3). Its own notes record σ=1 "shattering the grasp" and having to cut `joint_action_scale` to 0.01.
   The rebuild computes the PD torque in Python (`τ = Kp(q*−q) − Kd·q̇`), zeroes the actuator stiffness,
   and applies `set_joint_effort_target` with **randomized gains** — the standard HORA recipe, gentle and
   sim2real-robust. Delicate finger gaiting needs torque control, not stiff position control.

2. **A palm-up "cradle" that created the reward-hack → no cradle; genuine fingertip rotation.**
   The previous stack added `palm_up_orient` so the object would rest on the upturned hand (to stop
   drops). That cradle is exactly what let the object **fall into the palm and roll passively**, which
   the HORA angular-velocity term happily credited as "rotation." The rebuild keeps the object pinched at
   the **fingertips** (the GraspXL grasp, transformed into the hand frame) and rewards rotation about the
   palm axis — a fingertip pinch *cannot* passively roll, so the only way to earn reward is genuine
   finger-driven rotation. (We verified the rebuild's rotation is real with a zero-action control on the
   cylinder: ω≈0 with no finger motion vs 0.4–0.6 with the policy.)

3. **Fixed-schedule gravity curriculum → adaptive, drop-rate-gated curriculum.**
   The previous curriculum ramped gravity on a fixed step-clock (`gravity_curriculum_steps`); per its own
   notes it capped at ~80% g and needed manual warm-start continuations. The rebuild (from sharpa-rl-lab)
   raises gravity by 0.05 **only when the recent drop rate is ~0** — it self-paces and never advances
   faster than the policy can hold, ramping −0.05 → −9.81 autonomously. (We A/B-tested this exact
   mechanism earlier: ported into the *old* env it removed the warm-start chore; it is the single
   highest-leverage fix.)

4. **A reward-shaping spiral → the clean HORA reward + correct references.**
   Fighting the reward-hack, the previous stack accreted ad-hoc terms — flat fall penalty, object-drift
   penalty, joint-limit-margin penalty, grasp-squeeze extrapolation, selective self-collision — none of
   which addressed the root cause. The rebuild uses the plain HORA reward (rotate + object-linvel + pose
   + torque + work + object-pos) with **correct references** (the pose penalty references the *actual
   GraspXL grasp*, not a canonical cylinder pose; the object-pos reward references the *settled* grasp
   position). With (1)–(3) removing the exploits at the source, no shaping hacks are needed.

## What the rebuild adds on top (and that the previous stack never reached)
- **Tactile** contact observations kept in the policy obs (for later teacher→student distillation).
- **Point cloud + masks** and a **world-model auxiliary** (predict object pose) — both PASS on the
  cylinder; the world model even *improved* rotation (see `REBUILD_RESULTS.md`).
- **GraspXL object support** via trajectory replay (this work): single object (full g, held+rotating),
  multiple convex objects of varying shape/size, and SE(3) orientation randomization.

## Results table (GraspXL objects, trajectory-replay init, rebuild core)
All eval at full gravity (−9.81), 64 envs, active (post-settle) phase. |yaw| is the rotation rate about
the (possibly rotated) palm axis. held≈0.87 is consistent across tasks — it reflects holding with a brief
settle→active handoff transient, NOT real drops (rewards 290–360 and the advancing curriculum confirm the
object stays gripped).

| step | task | trained to | full-g eval: held / |yaw| | status |
|---|---|---|---|---|
| 1 single object (ball) | `...-GraspXL-Replay-v0` | full g | 0.868 / **0.787** rad/s | **SUCCESS** |
| 2 multiple convex objects (4, varying shape/size) | `...-GraspXL-Multi-v0` | 77% g (35M) | 0.868 / 0.201 rad/s | **SUCCESS** (generalizes) |
| 3 SE(3) orientation randomization (full SO(3)) | `...-GraspXL-Orient-v0` | 52% g (35M) | 0.868 / **0.530** rad/s | **SUCCESS** (palm any-orientation) |

Notes:
- **Step 1** is the headline: the same ball the previous stack never cleanly solved is held + rotated at
  full gravity in ~25 min.
- **Step 2** holds all 4 convex objects (ball + 3 others, sizes 0.084–0.115 m) at full g; rotation is
  slower (0.20) because the 4-object curriculum is gated by the hardest object and only reached 77% g in
  35M (the curriculum was still accelerating — a longer single run would reach full g; resume can't
  continue it because the curriculum gravity is physics-engine state, not in the checkpoint).
- **Step 3** randomizes the whole hand+object SE(3) so the palm faces any direction (full SO(3)); drop is
  made orientation-invariant (object displacement from the rotated grasp). Full SO(3) at full gravity is
  the hardest case (palm fully inverted must hold by pinch alone); it trained to 52% g but the gripped
  policy **generalizes to full gravity** (held 0.87, rotates 0.53 rad/s). A constrained tilt range would
  reach full g faster if desired.

Videos (close-up; show the trajectory-replay grasp closing, then rotation):
- step 1 single: `videos_gx_single/render_*.mp4`
- step 2 multi:  `videos_gx_multi/render_*.mp4`
- step 3 orient: `videos_gx_orient/render_*.mp4`
(plus the earlier rebuild-stage videos `videos_stage{1..4}_*` and the official `../sharpa-rl-lab/videos_official/`.)

## Honest limitations
- Steps 2 and 3 reached partial gravity in the 35M budget (the heterogeneous-object and full-SO(3)
  curricula are gated by the hardest case); both still hold+rotate at full gravity on eval, but a longer
  single run (or a per-object / annealed-range curriculum) would push the training gravity to full.
- Point cloud + world model are OFF for the GraspXL tasks (our PC sampler is cylinder-specific); a
  per-object mesh point-cloud sampler is the natural next step to bring the full network design (Steps 2–3
  of the earlier rebuild) onto the GraspXL objects.
