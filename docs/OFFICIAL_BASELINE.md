# Official SharpaWave tutorial — our guidance baseline (verified 2026-06-29)

The official tutorial (`Dynrotate/sharpa-rl-lab`) is our ground-truth baseline for in-hand rotation. These
numbers are **re-verified** (signed metric, full gravity, no DR, ProprioAdapt, 64 envs × 500 steps), and the
rebuild reproduces them exactly — use them to sanity-check any change.

## Verified baseline numbers (signed yaw about rot_axis, the metric that matters)
| checkpoint | object(s) | signed yaw (rad/s) | rotate_reward (clip ±0.5) | drops | note |
|---|---|---|---|---|---|
| `pretrained/0.5-0.5-1.pth` | cylinder | **0.98** | 0.48 (saturated) | 0 | the canonical "it rotates" |
| `pretrained/0.4-0.6-8.pth` | cylinder ×8 scales | — | — | — | multi-scale generalization |
| `ckpts/cylinder_ball_cube.pth` | cylinder + **ball** + cube | **0.64** | 0.36 | few | proves non-cylinders rotate |

`signed ≈ |yaw|` for all of these ⇒ pure directional rotation (no jitter). **Rebuild matches** (0.98 / 0.65).

## The recipe (what the baseline does, and why it works)
1. **Grasp = `gen_grasp` enclosing grasp.** The object is dropped into the hand and **settles into an
   enclosing grasp** (fingers wrap around it); the settled hand-joints + object-pose are saved to a cache
   (`cache/sharpa_grasp_linspace_*.npy`). Training resets to these grasps. **This is the part that makes
   rotation possible** — there is geometry to gait against. (See `gen_grasp.py`: it just runs the env with
   zero actions and records the settle.)
2. **Reward (official, in `sharpa_wave_env_cfg.py`):** `rotate_reward_scale 2.5`, `angvel_clip ±0.5`
   (rotate term = clipped `angvel · rot_axis`), `object_linvel −0.3`, `pos_diff −0.4`, `torque −0.1`,
   `work −0.5`, `object_pos (1/(disp+ε)) 0.003`. `rot_axis = (0,0,1)`.
3. **Curriculum:** adaptive gravity curriculum (raise g as the drop rate falls).
4. **Distillation:** stage-1 PPO (privileged) → stage-2 ProprioAdapt (proprio+tactile student). The pretrained
   ckpts are stage-2 → eval/render with `--algorithm ProprioAdapt`.
5. **Metric:** `eval_metrics.py` prints BOTH `mean yaw angvel (about z)` (SIGNED — **use this**) and
   `mean |yaw|` (absolute — **do not** judge rotation by this; jitter inflates it).

## How to reproduce / sanity-check (from `sharpa-rl-lab` or `sharpa_rl_rebuild`)
```
python rl_*/scripts/eval_metrics.py --task Isaac-Inhand-Rotate-Sharpa-Wave-v0 \
  --num_envs 64 --eval_steps 500 --algorithm ProprioAdapt \
  --load_path pretrained/0.5-0.5-1.pth --device cuda:0
# expect: mean yaw angvel (about z) ~0.98, drops ~0
```

## Implication for our GraspXL work
Our GraspXL objects only **hold** because they import the GraspXL **pick-up** grasp (a fingertip pinch),
not a `gen_grasp` enclosing grasp. To get rotation on GraspXL objects, **apply step 1 of this recipe to each
GraspXL object** (settle it into the hand, build a grasp cache) and keep steps 2–5 unchanged. See
`ORIENT_FINDINGS.md` §4/§6.
</content>
