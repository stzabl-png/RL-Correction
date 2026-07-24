# In-hand rotation on GraspXL objects — what actually went wrong (2026-06-29)

> **TL;DR (read this).** Two things were true and neither was what we thought:
> 1. The "sustained rotation on the single object" we believed we had **was never real rotation** — it was
>    the ball being *held with jitter*, mislabeled by the old `|yaw|` metric (absolute angular speed about
>    world-z). Under a correct *signed* metric it nets **0.036 rad/s** over a full episode.
> 2. The method/code is **fine** — the official **and** our rebuild rotate a cylinder at **0.98 rad/s**
>    (signed == |yaw|, zero drops), and a ball/cube at 0.65. The reason our **GraspXL** experiments only
>    *hold* is the **grasp**: the GraspXL dataset gives a pick-up *fingertip-pinch / cradle* grasp, which is
>    great for holding but cannot be rotated. The rotation task needs an **enclosing grasp** (what the
>    official `gen_grasp` cache produces). **Same code, different grasp ⇒ rotate vs hold.**
>
> So this was never an RL/reward/SO(3) problem. It is a **grasp-initialization** problem.

---

## 1. The metric bug that sent us down the wrong path
`eval_metrics.py` reported `|yaw|` = `abs(angvel about world-z)`, summed per step. Taking the absolute value
makes fast **back-and-forth jitter** read as fast "rotation." That is the entire origin of the fictitious
"0.79–0.98 rad/s" on the ball. (The official `eval_metrics.py` actually *also* prints the **signed** mean —
we were reading the wrong line.) Fix / always use: `rl_rebuild/scripts/orient_diag_eval.py` measures the
**per-episode CUMULATIVE SIGNED rotation about the (per-env) `rot_axis`**, plus held-fraction, active-time
milestones, and a zero-action control. Verdict rule: `signed ≈ |yaw|` ⇒ real rotation; `signed ≪ |yaw|` ⇒
jitter/holding. (Also fixed: `SHARPA_ORIENT_CAP` env-var so eval/render can force the orientation cap, which
is not stored in the checkpoint.)

## 2. The single-object policy HOLDS — it never rotated (3 independent confirmations)
Checkpoint `logs/debug/2026-06-28_02-02-07/stage1_nn/last.pth`, its own home task (`...-GraspXL-Replay-v0`),
palm-up, full gravity:

| evidence | value | meaning |
|---|---|---|
| signed yaw about rot_axis | **0.036 rad/s** | ≈0.08 rev over 12.8 s; **flat** across milestones (0.42→0.45→0.51 rad) |
| `\|yaw\|` about rot_axis | 0.69 rad/s | jitter — what the old metric called "rotation" |
| training `rotate_reward` (= raw mean signed angvel·rot_axis, clip ±0.5) | **0.089** | earned ~no rotation reward; its 337/episode came from **not dropping** |
| **video** patch-track (`videos_verify_single/`) | patch **pinned**: x-spread 17 px, y-spread 4 px, visible 400/401 frames | a ball spinning ≥0.5 rad/s would sweep ~80 px and hide half the time; it does neither ⇒ **not rotating** |

## 3. The method works — the DECISIVE baseline
Same `eval_metrics.py`, full gravity, no DR, ProprioAdapt:

| checkpoint | task | signed yaw | \|yaw\| | rotate_reward | drops | verdict |
|---|---|---|---|---|---|---|
| official `pretrained/0.5-0.5-1` | cylinder | **0.982** | 0.982 | 0.48 (≈±0.5 clip) | **0** | genuine sustained rotation |
| official `ckpts/cylinder_ball_cube` | cyl+ball+cube | **0.642** | 0.642 | 0.36 | few | rotates a **ball/cube** too |
| **rebuild** `pretrained/0.5-0.5-1` | cylinder | **0.982** | 0.982 | 0.48 | **0** | **our code reproduces it exactly** |
| **rebuild** `ckpts/cylinder_ball_cube` | cyl+ball+cube | 0.652 | 0.652 | 0.36 | few | same |

`signed == |yaw|` ⇒ pure directional rotation, no jitter. **The rebuild is not broken** — it rotates a
cylinder at 0.98 and a ball at 0.65. Video: `videos_verify_cylinder/` (the cylinder's "B" face sweeps away
and is regripped — real rotation).

## 4. Root cause: the GRASP, not the reward/object/code
- **Cylinder/ball/cube (rotate):** grasp comes from `gen_grasp` — the object is dropped into the hand and
  **settles into an enclosing grasp** (fingers wrap around it). The policy can gait it around → 0.98 / 0.65.
- **GraspXL (hold):** the grasp comes from the **GraspXL dataset** (the approach/pick-up trajectory). Both our
  modes use it — `replay` (replays the approach then settles) and `cache` (resets to the settled dataset
  grasp). Visually it is a **cradle/fingertip pinch**: the ball rests on top of open fingertips
  (`videos_verify_single/`). That is stable to *hold* but there is nothing to *gait against* → it can only be
  held, and rotating it shears it out.
- **Confirming experiment** (`...-GraspXL-Sustained-v0`, §5): even a rotation-favorable reward (rotate 3,
  capped speed, bounded anti-drop centering, hand-frame-gravity obs, from scratch) only pushed the
  pinch-grasp to `rotate_reward 0.114` (~0.11 rad/s) — vs **0.48** for the enclosing-grasp cylinder. The
  reward lever is nearly maxed; the grasp is the ceiling.
- This also explains the earlier "two regimes": survival-dominant reward ⇒ hold (signed ~0.03–0.09);
  rotation-dominant reward (`OrientScratch`, rotate 6) ⇒ the policy forces rotation and **shears the pinch
  out in ~3.6 s** ("one-time rotation"). Both are the same pinch-grasp limitation seen from two reward sides.

## 5. The sustained-rotation attempt and why it plateaued
`SharpaWaveGraspXLSustainedCfg` / `...-GraspXL-Sustained-v0`, from scratch, 1024 envs. Targets the drop
directly: rotate 3.0 + **capped** angvel_clip ±0.5 (don't pay for ball-shearing fast spins), a **bounded**
centering reward `exp(-disp/0.03) ∈ [0,1]` (new `center_reward_bounded` flag — the per-step
still-in-grasp signal the exploding `1/(disp+ε)` term could never provide), strong anti-drift, and hand-frame
gravity obs (SO(3)-equivariance). Result: rotate_reward rose 0.034 → **0.114** but stayed centering-dominated
— i.e. it holds better, rotates a little, **does not reach sustained rotation**, because the pinch grasp is
the ceiling (see §4). Stopped at ~26 M steps (`logs/debug/2026-06-29_04-51-09`).

## 6. The actual fix (recommended) and alternatives
1. **Give the GraspXL objects an enclosing, rotation-affording grasp** — the real fix. Use the official
   `gen_grasp` recipe on each GraspXL object (drop it into the hand, let it settle into an enclosing grasp,
   record the cache), exactly as the cylinder/ball/cube grasps were made, **instead of importing the GraspXL
   approach grasp**. Then train with the unchanged official reward. This is the path the baseline proves works
   (a ball already rotates at 0.65 this way). Caveats to handle: GraspXL ball mass is 0.10 kg (cylinder 0.05),
   and object scale/extent vary — the settle/drop band and friction may need a per-object pass.
2. **Reframe the GraspXL deliverable as stable *grasping*, not rotation.** What we *did* achieve on GraspXL
   objects is robust holding across objects and (with hand-frame gravity obs) across orientations — held
   0.88–0.92. If the goal is grasp transfer, that already works; rotation needs (1).
3. **Always measure signed-about-rot_axis (+ video patch-track). Treat `|yaw|` as deprecated** for this task.

## 7. DECISION (user, 2026-06-29): keep the trajectory-replay grasp → deliver as stable GRASPING
We will **not** build the enclosing-grasp fix. The GraspXL deliverable is **stable grasping** (holding the
object through the episode), not in-hand rotation — rotation is not achievable from the imported pick-up
pinch (§4), and reaching it would mean replacing the trajectory-replay grasp we deliberately use.

**What the GraspXL stack DOES deliver (verified, signed diagnostic, full gravity):**
| setting | policy | held (active) | episodes surviving full 13.5 s |
|---|---|---|---|
| palm-up | single-object Replay | 0.88 | 83 % |
| tilt cap 0°→179° (per-env random) | single-object Replay | **0.88, flat across all tilts** | 77 % |
| orientation curriculum + hand-frame gravity obs | gravws3 (`logs/debug/2026-06-29_03-19-27`) | **0.92** | **90 %** |

i.e. the policy **holds the GraspXL ball stably across full gravity and the full SE(3) range of palm
orientations** (held ~0.88–0.92, episodes mostly survive the full horizon). The hand-frame-gravity-obs
policy (`...-OrientGrav-v0`, ckpt `logs/debug/2026-06-29_03-19-27/stage1_nn/last.pth`) is the best grasper.
This is the deliverable. (Rotation remains available later via §6 step 1 if priorities change.)

## 8. Files / artifacts
- Diagnostic: `rl_rebuild/scripts/orient_diag_eval.py` (`--orient_cap`, `--zero_action`, `--gravity_z`).
  Bounded-center reward: `sharpa_wave_env.py` (`center_reward_bounded`, `center_reward_sigma`).
- Sustained cfg/task: `SharpaWaveGraspXLSustainedCfg` / `Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Sustained-v0`.
- **Videos:** `videos_verify_cylinder/` (cylinder **rotating**, signed 0.98 — the method works);
  `videos_verify_single/` (GraspXL ball **held**, not rotating — pinned green patch);
  `videos_orient_rotating/` (OrientScratch: forces rotation on the pinch, then drops in ~3.6 s).
- Baselines re-verified: official & rebuild `pretrained/0.5-0.5-1.pth` (0.98), `ckpts/cylinder_ball_cube.pth`
  (0.65). Deprecated metric: `eval_metrics.py`'s `|yaw|` line.
</content>
