# recon_kailang: Static reconstruction into Step4

该目录集中存放 Kailang 为静态物体 `reconstruction → Step4 RL` 新增的独立功能、测试和
说明。必须接入现有环境的兼容修改仍原位更新在 `clips.py`、loader 和 env 中；新增入口统一
放在署名目录，便于组内识别作者和 review 边界。

This path stages one static object reconstructed from an egocentric video into
the existing Step4 table/robot environment. It does not rebuild the table and
does not use the reconstructed human trajectory to command the robot.

## Scope and upstream contract

Episode selection and best-mask selection belong to Step3. This integration
starts after those choices have produced a static-object reconstruction.

Required inputs:

- reconstruction: `object_mesh_scaled_final.obj` and `world_fused.npz`;
- retarget: `replay_world.npz` with `phase_left/right`, hand joints, object pose,
  frames, and FPS;
- optional retarget: `ref_qpos.npz`;
- optional reconstruction provenance: `world_summary.json` and
  `reconstruction_complete.json`.

`recon_kailang/stage_static_reconstruction.py` copies only these files into
`TrainingData/<dataset>/<name>` through an atomic staging directory. A failed
stage cannot leave a partially registered data unit.

## Placement contract

`ref_builders/static_reconstruction.py` implements the object-only placement:

1. Find the earliest frame where exactly one of `phase_left/right == 1`. A tie
   must be resolved by passing an explicit hand; it is never guessed.
2. Use joint 0 (wrist/root) of that hand after the existing replay-to-table
   alignment and temporal resampling.
3. Preserve the FoundationPose quaternion, changing it only for the declared
   coordinate-frame conversion and unit normalization. Stable-pose projection
   is disabled for this data source.
4. Align the rotated mesh AABB centre to the wrist XY.
5. Translate Z so the rotated mesh's lowest vertex is 2 mm above the configured
   table top.
6. Reject the clip if centre/wrist XY error exceeds 1 mm, bottom-gap error
   exceeds 1 mm, or any mesh vertex lies outside the table.

For this source, `DataUnit.ref.interaction_seg[0]` is also set to that mapped
phase frame. The generic replay loader's hand/mesh-distance estimate is not
allowed to override the placement clock.

The human wrist trajectory is a placement marker and a visual review overlay.
For `place_mode=object_only`, DexMate keeps its configured default joint pose;
human hand joints are not replayed and IK is not run. The placed object is not
shifted to make it reachable. A reach-gate failure aborts environment creation.

This source-specific path is isolated from the existing Setting B
affordance/fingertip placement and does not change that behavior.

## Stage and register a clip

Run from the repository root:

```bash
python -m rl_rebuild.correction.recon_kailang.stage_static_reconstruction \
  egodex task1_static_smoke \
  --reconstruction /path/to/ReconstructOutput/egodex/test/basic_pick_place/1 \
  --retarget /path/to/RetargetOutput/egodex/test/basic_pick_place/1 \
  --src-clip egodex/test/basic_pick_place/1 \
  --mass-kg 0.2 \
  --friction 0.5
```

After staging succeeds, add one `_td_static(...)` entry to `clips.py`.
Mass and friction default to the values in the staged `meta.json`; explicit
arguments in `clips.py` override them.

The first environment load converts the OBJ through the existing mesh-converter
path and caches a rigid, collidable `cache/object.usd`.

## Verification commands

The EULA variable below must only be set after the operator has accepted the
NVIDIA Isaac Sim/Omniverse EULA.

```bash
python -m unittest \
  rl_rebuild.correction.recon_kailang.tests.test_static_reconstruction

export OMNI_KIT_ACCEPT_EULA=YES
export SHARPA_WANDB=0
export RL_ISAAC_NO_GUARD=1

python -m tasks.recon_kailang.static_reconstruction.smoke \
  --clip task1_static_smoke --steps 80 \
  --report /path/to/physics_smoke.json \
  --headless --device cuda:0

python -m tasks.recon_kailang.static_reconstruction.train \
  --clip task1_static_smoke --name task1_static_minppo \
  --num_envs 8 --horizon 32 --max_agent_steps 1024 \
  --output_root /path/to/training_output \
  --headless --device cuda:0

python -m tasks.recon_kailang.static_reconstruction.record_scene \
  --clip task1_static_smoke \
  --output /path/to/task1_static_scene.mp4 \
  --preview /path/to/task1_static_scene_preview.png \
  --headless --device cuda:0
```

On a shared Isaac installation, pass a private Kit portable root through
`--kit_args` as required by that machine.

## Smoke-test record

The initial development clip is
`egodex/test/basic_pick_place/1` (`basic_pick_place/1.mp4`).

- source: 160 frames at 30 FPS; aligned: 107 frames at 20 FPS;
- first interaction: right hand, source frame 52, aligned frame 35;
- placed centre XY: `(0.0787566, 0.0290795)` m; construction error: 0 mm;
- rotated-mesh bottom gap before settling: 2.000 mm;
- mesh extents: approximately `0.200 x 0.096 x 0.019` m;
- runtime physics: 0.2 kg, friction 0.5, positive inertia,
  convex-decomposition collision, gravity -9.81 m/s^2;
- after 80 simulation steps: instantiated XY error 0.143 mm, bottom/table error
  -0.244 mm, XY drift 3.421 mm, final speed `1.56e-5 m/s`, table margin
  41.34 cm, finite state, and no early reset;
- robot maximum joint drift: 0.0038 rad, with the official fixed-torso USD;
- PPO smoke test: 8 environments, 1024 agent steps, three epoch checkpoints
  plus `best.pth`/`last.pth`; all 226,733 tensor elements in `last.pth` finite;
- review video: H.264, 1280x720, 147 frames at 20 FPS (7.35 seconds), with
  the wrist path, first interaction point, and initial object centre overlaid.

## Repository boundary

Commit source and tests only:

- `rl_rebuild/correction/recon_kailang/`;
- `tasks/recon_kailang/static_reconstruction/`;
- the isolated `clips.py`, loader, and environment changes;
- `tests/test_static_reconstruction.py`;
- this document.

Do not commit `TrainingData`, logs, generated OBJ/USD/NPZ/video/checkpoints,
private Python packages, or local copies of robot source assets.

The robot asset must come from Git LFS on commit `f6db7de` or newer. On the
verified `4b43270` baseline, `assets/vega_1p_sharpa_fixedtorso.usd` is a
25,553,426-byte USD crate with no unresolved external-reference symptoms. Do
not regenerate it with the pre-fix copy script.
