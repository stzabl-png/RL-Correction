# Water-bottle Step4 scene

This isolated Step4 task loads a two-part PCO-1810 bottle into the official
DexMate environment. It contains the final physical scene and its validation
tools; it does not contain the upstream video segmentation, reconstruction, or
CAD-generation workflow.

## Scene variants

- `water_bottle_twist_static`: bottle body and cap are separate rigid objects
  at the two trajectory-derived table placements. This is the placement and
  passive-physics baseline.
- `water_bottle_twist_assembled`: the cap starts closed and engaged. Positive
  relative rotation raises it by 3.18 mm per turn and releases it after two
  turns (6.36 mm of constrained travel).
- `water_bottle_twist_screw_on`: the cap starts free at the right-hand
  trajectory placement. It remains free until the robot aligns it with the
  thread entrance; after capture, negative relative rotation lowers it by the
  same pitch until fully closed.

The PhysX rack-and-pinion joint is not Direct-GPU compatible. The two screw
variants therefore enforce the same helical relation with batched GPU tensor
writes:

```text
z = z_closed + pitch * angle / (2*pi)
```

Capture requires radial error at most 3 mm, axial error at most 3 mm, tilt at
most 10 degrees, and yaw error at most 30 degrees. A free-cap analytic barrier
prevents a wrongly aligned cap from passing down through the bottle envelope.
Visible CAD threads do not use triangle-on-triangle collision because that
contact is unstable and conflicts with the analytic constraint at training
scale. Robot/cap and table/cap contacts remain enabled.

The task exposes `screw_angle`, `screw_engaged`, and `screw_has_depth` tensors.
It deliberately does not add a twist reward or task-success definition; those
belong to the training task that consumes this scene.

## Validation

Run dependency-light tests:

```bash
PYTHONPATH=. python -m unittest \
  tasks.recon_kailang.bottle_reconstruction.test_screw_joint \
  rl_rebuild.correction.recon_kailang.tests.test_static_reconstruction
```

Run the passive placement/physics audit:

```bash
OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH=. \
python -m tasks.recon_kailang.bottle_reconstruction.smoke --headless \
  --clip water_bottle_twist_static --report /tmp/bottle_static.json
```

Run both Direct-GPU screw audits:

```bash
OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=. \
python -m tasks.recon_kailang.bottle_reconstruction.screw_smoke --headless \
  --clip water_bottle_twist_assembled --torque-nm 0.003 --steps 1500 \
  --settle-steps 0 --report /tmp/bottle_unscrew.json

OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=. \
python -m tasks.recon_kailang.bottle_reconstruction.screw_smoke --headless \
  --clip water_bottle_twist_screw_on --torque-nm 0.003 --steps 1500 \
  --settle-steps 0 --report /tmp/bottle_screw_on.json
```

Add `--num-envs 1024` to the second command for the training-scale audit.
Detailed source, placement, physics, and A6000 results are recorded in
`datasets/recon_kailang/water_bottle_twist_static/VALIDATION.md`.

Record a close-up mechanism video (the applied torque is for visualization,
not a robot policy):

```bash
OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=. \
python -m tasks.recon_kailang.bottle_reconstruction.record_scene --headless \
  --clip water_bottle_twist_assembled --screw-demo \
  --output /tmp/water_bottle_unscrew.mp4 \
  --eye 0.35,0.55,1.25 --lookat=-0.10,0.00,0.98
```
