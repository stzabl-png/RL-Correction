# Task 5: bimanual pouring

This directory is isolated from the existing single-hand tasks. It implements
one 26-D policy (left 13, then right 13), a seven-phase pour episode, a
conservative analytic liquid proxy, curriculum gates, video ablations, and
episode-level evaluation artifacts.

## Safety gates

Training refuses to start unless all of the following are true:

1. The reference has synchronized ARKit-world hand, camera, cup, and bottle
   trajectories plus non-empty object-local video contact regions.
2. One rigid ARKit-world to simulation transform fits both reconstructed
   objects within 3 cm.
3. The scene status is `ready` and all mesh/USD/reference/prior paths exist.
4. Both Dexonomy priors have a human approval record whose SHA-256 matches the
   exact prior and whose independent Isaac Gate-2 result has at least four
   pads and positive `Q_star`.

`pour/11` is the only training reference. `pour/9` is hold-out evaluation
only; `train.py` enforces this rule.

## Reference and reconstruction

Build the v2 hand/camera references from raw EgoDex HDF5:

```bash
python -m tasks.pour.reference \
  --hdf5 /data/Egodex/raw/test/pour/11.hdf5 \
  --demo-id 11 \
  --output tasks/pour/references/pour11/reference.npz
```

Raw EgoDex transforms stay in the stationary ARKit world frame. A reconstructed
object track NPZ has the strict fields below:

- `pose`: `[T,7]` xyz+wxyz or `[T,4,4]`.
- `source_frame`: strictly increasing EgoDex frame numbers.
- `confidence`: `[T]` values in `[0,1]`.
- `coordinate_frame`: scalar `camera` or `arkit_world`.
- `quaternion_order`: scalar `wxyz` for a `[T,7]` pose.

`geometry.json` schema 1 contains complete `cup` and `bottle` object specs and:

```json
{
  "contacts": {
    "coordinate_frame": "object_local",
    "left": {"object": "cup", "points": [[0, 0, 0]], "confidence": 0.8},
    "right": {"object": "bottle", "points": [[0, 0, 0]], "confidence": 0.8}
  }
}
```

Merge only after reconstruction, time synchronization, and frame validation:

```bash
python -m tasks.pour.stage_reconstruction \
  --reference tasks/pour/references/pour11/reference.npz \
  --scene tasks/pour/references/pour11/scene.json \
  --cup-track /reconstruction/pour11/cup_track.npz \
  --bottle-track /reconstruction/pour11/bottle_track.npz \
  --geometry /reconstruction/pour11/geometry.json
```

This advances the manifest only to `pending_grasp_approval`.

The current Step-2 reconstruction branch already supports multiple SAM2 object
slots. Label `cup` and `bottle` separately, reconstruct both, then adapt its
`world_fused.npz` package. The object spec supplies reviewed opening geometry,
physical values, converted USD paths, and desired simulation initial poses:

```json
{
  "schema_version": 1,
  "demo_id": "11",
  "cup": {
    "label": "cup", "mesh": "replaced from Step2", "usd": "/assets/cup.usd",
    "mass_kg": 0.2, "friction": 0.5,
    "initial_pose_wxyz": [0, 0.18, 0.9, 1, 0, 0, 0],
    "opening_center_local": [0, 0, 0.1],
    "opening_axis_local": [0, 0, 1], "opening_radius_m": 0.035
  },
  "bottle": {
    "label": "bottle", "mesh": "replaced from Step2", "usd": "/assets/bottle.usd",
    "mass_kg": 0.3, "friction": 0.5,
    "initial_pose_wxyz": [0, -0.18, 0.9, 1, 0, 0, 0],
    "opening_center_local": [0, 0, 0.15],
    "opening_axis_local": [0, 0, 1], "opening_radius_m": 0.015
  }
}
```

The numeric values above illustrate the schema only; measure them from the
reconstructed assets and approve them before running the adapter.

```bash
python -m tasks.pour.step2_adapter \
  --fused <step2/world_fused.npz> \
  --reference tasks/pour/references/pour11/reference.npz \
  --object-spec <pour11/object_spec.json> --output-dir <pour11/staged> \
  --cup-id object_0 --bottle-id object_1
```

The adapter rejects camera synchronization/frame fits whose translation p95 is
over 5 cm or rotation p95 is over 5 degrees. It exports two ARKit-world tracks,
contact regions obtained from video fingertips projected to each mesh, and an
alignment report. Feed those outputs to `stage_reconstruction` above. Repeat
the entire reconstruction independently for demo 9; do not reuse demo 11
tracks or contacts.

## Manual Dexonomy approval

Task 5 pins the self-contained Step-3 Dexonomy checkout at `a09b5ab`. Verify it,
then run the template that a human selected from previews (for example,
`3_Medium_Wrap` or `1_Large_Diameter`):

```bash
python -m tasks.pour.dexonomy_adapter check --dexonomy-root /opt/Dexonomy
python -m tasks.pour.dexonomy_adapter generate \
  --dexonomy-root /opt/Dexonomy --mesh <cup.obj> --object-id pour11_cup \
  --side left --template 3_Medium_Wrap
python -m tasks.pour.dexonomy_adapter generate \
  --dexonomy-root /opt/Dexonomy --mesh <bottle.obj> --object-id pour11_bottle \
  --side right --template 1_Large_Diameter
```

The adapter corrects a Step-3 wrapper issue by re-exporting left candidates
with explicit left joint names. Convert the reviewed candidates with the same
side check:

```bash
python -m tasks.pour.dexonomy_adapter convert \
  --grasp-npy <cup_candidate.npy> --info-json <cup/simplified.json> \
  --output <cup_left.npz> --side left
python -m tasks.pour.dexonomy_adapter convert \
  --grasp-npy <bottle_candidate.npy> --info-json <bottle/simplified.json> \
  --output <bottle_right.npz> --side right
```

Run both candidates through side-specific IK (Gate-1) and independent Task-5
Isaac physics (Gate-2). The screen uses the actual bimanual robot, object
assets, contact sensors, and materials. A candidate must lift its object off
the table and hold it with the same residual setting; the screen cannot mark
a scene approved:

```bash
CUDA_VISIBLE_DEVICES=1 python -m tasks.pour.screen_prior \
  --scene <pour11/scene.json> --left-prior <cup_left.npz> \
  --right-prior <bottle_right.npz> --left-template 3_Medium_Wrap \
  --right-template 1_Large_Diameter --output-dir <screen-dir> --headless
```

If a right-hand candidate is mirrored as a fallback, `mirroring.py` marks it
as requiring this same physical screen; mirroring is never approval.

Each approval report contains:

```json
{
  "schema_version": 1,
  "hand_side": "left",
  "object": "cup",
  "template": "3_Medium_Wrap",
  "prior_sha256": "...",
  "gate1_pass": true,
  "gate2_pass": true,
  "stable_lift_pass": true,
  "pads_star": 4.0,
  "q_star": 0.1,
  "object_lift_m": 0.02
}
```

Apply the explicit selections:

```bash
python -m tasks.pour.approve_grasps \
  --scene tasks/pour/references/pour11/scene.json \
  --left-prior /priors/cup_left.npz \
  --right-prior /priors/bottle_right.npz \
  --left-report <screen-dir/left_screen.json> \
  --right-report <screen-dir/right_screen.json> \
  --approved-by "human reviewer"
```

## VM environment and smoke

The isolated VM environment is:

```text
/home/kailang/.local/miniconda3/envs/rl-correction-pour
Isaac Sim 5.1.0
Isaac Lab v2.3.2 (37ddf62)
PyTorch 2.7.0+cu128
```

The non-Kit GPU check is:

```bash
python -m tasks.pour.m0_check --cuda-device 1
```

After the account owner accepts the NVIDIA Omniverse EULA, run both Isaac
smokes on the currently idle GPU:

```bash
CUDA_VISIBLE_DEVICES=1 python -m tasks.pour.smoke \
  --scene tasks/pour/references/pour11/scene.json --num_envs 1 --headless
CUDA_VISIBLE_DEVICES=1 python -m tasks.pour.smoke \
  --scene tasks/pour/references/pour11/scene.json --num_envs 16 --headless
```

## Curriculum training

The default PPO settings retain `KL threshold=0.02`, 1,024 environments,
60 stance-prefix frames, and the ±3 cm/±15 degree reachable pregrasp pool.
Only lower `num_envs` to the largest stable power of two after a measured OOM.

Run the five stages in order, passing `curriculum_pass.pth` or the final
checkpoint from the preceding stage. Stages 0-2 automatically stop after at
least 1,000 evaluated training episodes and the 90%/85%/80% gate.

```bash
python -m tasks.pour.train --scene <pour11/scene.json> \
  --stage single_grasp --ablation full --seed 42 --headless
python -m tasks.pour.train --scene <pour11/scene.json> \
  --stage dual_grasp --ablation full --seed 42 \
  --load_path <stage0/curriculum_pass.pth> --headless
python -m tasks.pour.train --scene <pour11/scene.json> \
  --stage approach_grasp --ablation full --seed 42 \
  --load_path <stage1/curriculum_pass.pth> --headless
python -m tasks.pour.train --scene <pour11/scene.json> \
  --stage align_pour --ablation full --seed 42 \
  --load_path <stage2/curriculum_pass.pth> --headless
python -m tasks.pour.train --scene <pour11/scene.json> \
  --stage full --ablation full --seed 42 \
  --load_path <stage3/checkpoint.pth> --headless
```

Use seeds `42,43,44` and the same stage budgets for `full`, `no_trajectory`,
`no_contact`, and `pure_rl`. Checkpoints are observation-compatible across all
four modes; exact liquid state and success subconditions remain critic-only.

## Evaluation and artifacts

Run each final checkpoint on demo 11 and its independently built demo 9 scene:

```bash
python -m tasks.pour.eval --scene <scene.json> --checkpoint <checkpoint.pth> \
  --output_dir <eval-dir> --run_name <name> --ablation full \
  --seed 42 --episodes 1000 --headless
```

The output contains `episodes.jsonl` and `summary.json`, including phase,
two-hand grasp state, both object poses, conservative liquid allocation,
success subconditions, and the deterministic failure reason. Record visual
examples separately:

```bash
python -m tasks.pour.record --scene <scene.json> --checkpoint <checkpoint.pth> \
  --output_dir <video-dir> --seed 42 --episodes 12 --headless
```

Aggregate all seeds, hold-out results, ablations, failure counts, and first
60%-success milestones:

```bash
python -m tasks.pour.aggregate --evaluation-root <eval-root> \
  --milestone-root logs --output-dir <report-dir>
```

## Local verification

```bash
python -m unittest discover -s tasks/pour/tests -v
```

The tests cover action splitting, left/right mapping, phase reward gates,
liquid mass conservation, success/failure precedence, artifact round trips,
coordinate-frame reconstruction merge, side-verified Dexonomy conversion,
manual prior approval, and tracker output. No existing single-hand file is
modified by this task.
