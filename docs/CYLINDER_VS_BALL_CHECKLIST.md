# Cylinder-vs-Ball diagnostic checklist (run after the cylinder completes)

Purpose: capture *why the cylinder rotates and the ball only holds*, isolate the single cause, and — if it
is the **initial-pose problem** — explain mechanistically why the hand never re-poses into a rotatable grip.

Both runs use the **identical Stage-4 design** (same network, reward, curriculum, DR, control). By
construction the **only intended differences are the object and its initial grasp pose**. So the diff below
should be short; anything else that differs is an accidental confound to flag.

How to use: fill the **Observed** cells from the completed cylinder run + a matched ball checkpoint, then work
Part D. Commands to collect everything are in Part E.

---

## Part A — Design choices (should be SAME; verify, don't assume)

| Design choice | Value (both) | SAME? | Note if it differs |
|---|---|---|---|
| Algorithm | PPO + KL-adaptive LR, GAE | ☐ | |
| Network | ActorCritic: proprio(192) + priv_mlp[256,128,8] + PointNet(5→128) → MLP[512,256,128] | ☐ | |
| World-model head | on, predicts obj pose(7), aux MSE × coef 1.0 | ☐ | |
| Point cloud | 161 pts × 5 (xyz+2-ch mask), wrist frame | ☐ | **cyl: sampled on cylinder surface / ball: sampled on real mesh** |
| Control | torque/effort target, `action_scale 1/24` | ☐ | |
| Reward | 6-term HORA, verbatim (no center_reward_bounded / squeeze / orient) | ☐ | |
| Curriculum | adaptive drop-gated gravity (−0.05 → −10) | ☐ | |
| Init mode | cyl: grasp cache `cache/sharpa_grasp_linspace` / **ball: trajectory replay → endpoint** | ☐ | **DIFFERENT (intended)** |
| Orientation | palm-up, `rot_axis=(0,0,1)`, no SE(3) randomization | ☐ | |

## Part B — Settings: hand, collisions, materials

| Setting | Cylinder | Ball | SAME? |
|---|---|---|---|
| Hand USD | `assets/SharpaWave/right_sharpa_wave.usda` | same | ☐ |
| Hand init (wrist) pose | `((0,0,0.5),(0.819,0,−0.574,0))` palm-up | same | ☐ |
| Hand `disable_gravity` | True (floating) | same | ☐ |
| **Self-collision** | `enabled_self_collisions=True` | same | ☐ |
| Hand contact_offset / rest_offset | 0.002 / 0.0 | same | ☐ |
| Actuators | `IdealPDActuatorCfg(stiffness=None, damping=None)` → torque | same | ☐ |
| Tactile sensors | 5 elastomers, filter on `/object` | same | ☐ |
| Elastomer / metal / object friction | 0.8 / 0.1 / **0.5** | 0.8 / 0.1 / **0.5** (DR same) | ☐ |
| Friction DR | ×[0.5, 2.0] | same | ☐ |
| **Object collision approx** | cylinder native (convex) | **`convexDecomposition`** (mesh) | ☐ **DIFFERENT** |
| **Object mass** | **0.05 kg** | **0.10 kg** (`OBJECT_MASS`) + DR mass[0.01,0.25] | ☐ **DIFFERENT** |
| Object size | cylinder r≈0.04, h≈0.064 × **scale 0.5** | ball ≈ **0.053 m**, **scale 1.0 (medium/original)** | ☐ **DIFFERENT** |
| Object init pos | `(−0.0956, −0.0052, 0.619)` | from replay endpoint (settled) | ☐ |
| **Reset height band** | `[0.59906, 0.63906]` (≈4 cm absolute) | `width 0.16` recentered on cached z | ☐ **DIFFERENT** |
| `replicate_physics` | False | False | ☐ |

## Part C — Parameters (numeric; should be SAME unless noted)

| Param | Value | SAME? |
|---|---|---|
| num_envs / seed / max_agent_steps | 1024 / 42 / 100M | ☐ |
| batch = 8×envs / minibatch min(8×envs,32768) / mini_epochs | 8192 / 8192 / 5 | ☐ |
| dt / decimation / control step / episode | 1/240 / 12 / 0.05 s / 400 steps (20 s) | ☐ |
| lr / γ / τ / kl / e_clip / critic_coef / entropy / bounds | 5e-3 / 0.99 / 0.95 / 0.02 / 0.2 / 4 / 0 / 1e-4 | ☐ |
| Reward scales (rotate/linvel/pose/torque/work/objpos) | +2.5 / −0.3 / −0.4 / −0.1 / −0.5 / +0.003 | ☐ |
| angvel_clip | ±0.5 | ☐ |
| DR: pd / com / force_scale | ×[0.5,2] / ±0.01 / 2 | ☐ |

## Part D — The key question: does cylinder rotate and ball hold? Why?

### D0. Confirm the outcome (don't trust total reward — the centering term inflates it)

| Metric (eval at full g, signed about `rot_axis`) | Cylinder | Ball | Verdict |
|---|---|---|---|
| **Signed cumulative rotation** (`orient_diag_eval.py`) | ____ | ____ | cyl ≫ ball ⇒ confirmed |
| Held fraction (object stays in band) | ____ | ____ | both high ⇒ ball *holds* |
| Final gravity reached by curriculum | ____ | ____ | ball reaching full g ⇒ stable hold, not rotation |
| Per-term means: `rotate_reward` vs `object_pos_diff` | ____ | ____ | ball: objpos ≫ rotate ⇒ holding dominates |

> Sanity: rotate reward maxes at `clip(0.5)×2.5 = 1.25`; the centering term `0.003/(disp+0.001)` ≈ **3.0 when
> held dead-still** (disp→0). So a perfect *holder* out-scores a perfect *rotator*. High total reward on the
> ball ≠ rotation. **Always read the signed-rotation metric and the per-term split, not total reward.**

### D1. Is the ball grasp a pinch or an enclosing grip? (geometry of the initial pose)

- [ ] **Contacts at the endpoint:** how many of the 5 elastomers are in contact, and contact force per finger?
      Cylinder = several fingers wrapping (enclosing); ball ≈ 2–3 fingertips (pinch). Expect ball pinch.
- [ ] **Finger-object wrap:** distance from each fingertip to the object center vs object radius. Enclosing ⇒
      fingertips distributed *around* the object; pinch ⇒ two opposing contacts, large gaps elsewhere.
- [ ] **Object seat:** is the object between the fingers (manipulable) or resting against the palm (cradled)?

### D2. Is the hand *pinned* to the bad pose by the reward? (why it can't adjust) — primary hypothesis

The −0.4 `pos_diff_penalty = Σ(q − q_default)²` is taken against **`hand.data.default_joint_pos`**, which the
GraspXL env **sets to the settled replay endpoint = the pinch** (`sharpa_wave_graspxl_env.py`, settle block).
So re-posing into an enclosing grip means leaving `q_default` ⇒ a *quadratic penalty that grows with the very
motion needed to fix the grasp*. For the cylinder, `q_default` is already an enclosing, rotatable grasp, so the
same anchor *helps*. Test it:

- [ ] **Confirm the reference:** verify ball `default_joint_pos` == replay endpoint (pinch), cyl == enclosing grasp.
- [ ] **Finger excursion over an episode:** mean/max `|q − q_default|` per finger joint. Ball ≈ 0 (pinned at
      pinch) ⇒ the policy never attempts a re-grasp. Cylinder shows cyclic excursion (rotating gait).
- [ ] **Reward-gradient argument:** with pinch as reference, the net of (`pos_diff −0.4` + `torque −0.1` +
      `work −0.5`) penalizing the re-grasp vs (`rotate +2.5`, capped at 1.25) + (`objpos`, *maximised by NOT
      moving*) — does holding dominate the achievable return? Estimate from the per-term means in D0.
- [ ] **Counterfactual (decouple the anchor):** if a quick test sets `q_default` to a neutral/enclosing pose
      (not the pinch) and the ball starts to rotate, the pose-anchor is the cause. (Diagnostic only — not the
      faithful design; note it, don't ship it.)

### D3. Is the pinch geometrically rotatable at all? (even if it tried)

- [ ] Could this hand **enclose** a 5.3 cm ball at the endpoint object position without first moving the
      object? (finger reach vs object pose). If not, the *initial grasp/placement* is unrecoverable in-place,
      and the centering penalty (which fights moving the object) makes it a dead end.
- [ ] Compare to cylinder: the enclosing grasp already has the object centered in the finger cage ⇒ small
      finger motions produce object spin. The pinch lacks the moment arm / opposing contacts to spin the ball.

### D4. Conclusion to write

- [ ] State the single cause (expected: **initial grasp = a fingertip pinch, used as the pose-penalty
      reference**, so the reward optimum is to hold; secondary confounds: mass 2×, mesh collision, larger object).
- [ ] If initial-pose-problem confirmed, record **why the hand can't adjust**: (a) `pos_diff` anchors it at the
      pinch, (b) torque/work penalties tax the re-grasp, (c) centering reward rewards stillness, (d) possibly no
      in-place enclosing solution exists. Re-grasping is *off the reward gradient*, not physically blocked.
- [ ] Fix directions (for a later decision, not this run): supply an **enclosing gen_grasp** for the ball like
      the cylinder; and/or set the pose-penalty reference to a neutral/enclosing pose; and/or a squeeze
      curriculum. Keep DR and the reward otherwise verbatim (clean inheritance).

## Part E — Commands to collect the observations

```bash
PY=/home/magics/miniconda3/envs/sharpa/bin/python
CK_CYL=$(ls -t logs/debug/*/stage1_nn/*.pth | head -1)        # or the cyl run's best.pth
CK_BALL=<ball checkpoint at a comparable agent-step>

# D0: correct rotation metric + held fraction, full gravity, signed about rot_axis
$PY rl_rebuild/scripts/orient_diag_eval.py --task Isaac-Inhand-Rotate-Sharpa-Wave-WM-v0          --num_envs 64 --load_path "$CK_CYL"
$PY rl_rebuild/scripts/orient_diag_eval.py --task Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Replay-PCWM-v0 --num_envs 64 --load_path "$CK_BALL"

# Per-term reward + drop/contacts: read from wandb (project sharpa-pcwm-confirm) or the TB scalars:
#   extras logged: rotate_reward, object_pos_diff, pos_diff_penalty, torque_penalty, work_penalty,
#   roll/pitch/yaw, gravity_z, height_reset_upper/lower.   (sharpa_wave_env.py _get_rewards/_get_dones)

# Visual confirmation (pinch vs enclosing, hold vs spin):
$PY rl_rebuild/scripts/render_video.py --task Isaac-Inhand-Rotate-Sharpa-Wave-WM-v0          --num_envs 4 --video_length 320 --load_path "$CK_CYL"  --out_dir videos_cyl_confirm
$PY rl_rebuild/scripts/render_video.py --task Isaac-Inhand-Rotate-Sharpa-Wave-GraspXL-Replay-PCWM-v0 --num_envs 4 --video_length 320 --load_path "$CK_BALL" --out_dir videos_ball_confirm
```

> Eval gotchas: `eval_metrics.py` reports `|yaw|` — **deprecated** (jitter reads as rotation; use
> `orient_diag_eval.py`). Eval sets gravity −9.81 and `gravity_curriculum False`. 4096 envs carb-crashes; use
> ≤64 for eval. Don't `setsid`; only kill your own processes.
