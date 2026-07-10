# Anchored BODex (`anchored_bodex`): human-demo-guided grasp synthesis

A second synthesis pipeline built **on top of** (never modifying)
`bodex_curobo_v2`: it reads a reconstructed human video demonstration (MANO
hand + object trajectory) from the sequence directory and anchors the same
bilevel optimization to it. Seeds come from retargeted human contact-frame
poses, the contact-point set and an affordance heatmap come from where the
human actually touched the object, and annealed guidance energies keep the
result similar to the demo -- while the force-closure QP still decides
success exactly as in the pure pipeline.

```text
sequence directory (object mesh + human_demo.npz)
    -> solve_sharpa_anchored_bodex() -> grasp_*.json / summary.json (+ affordance.npz cache)
```

## Modules

All under `src/ocir/grasp_synthesis/anchored_bodex/`:

| Module | Role |
| --- | --- |
| `demo_data.py` | `HumanDemo`: `human_demo.npz` loader (schema documented in its docstring) |
| `affordance.py` | affordance heatmap + per-frame contact detection, lazily cached to `affordance.npz` |
| `demo_analysis.py` | grasp window detection (grasp/pickup frames) + human contact roles |
| `retarget.py` | MANO->Sharpa wrist calibration + fingertip-fit IK (`HandFitter`) |
| `seed_generator.py` | relaxed retargeted contact-frame seeds (`relax_joint_mask`) |
| `guidance.py`, `rollout.py`, `solver.py` | anchored energies + cuRobo wiring |
| `synthesize_sharpa_anchored_bodex.py` | CLI |

## Demo data preparation (once per sequence set)

Export `human_demo.npz` from DexYCB labels into each sequence directory:

```bash
scripts/run_grasp_synthesis_conda.sh scripts/dexycb/export_grasp_sequences.py \
  --manifest ${OCIR_DATA_ROOT}/processed_data/dex_ycb/manifests/selected_5_sequences.json \
  --sequences-root ${OCIR_DATA_ROOT}/processed_data/dex_ycb/sequences
```

It writes per-frame MANO vertices/keypoints + object pose (pre-transformed
into the object canonical frame) and records the file in `sequence.json`.
The `affordance.npz` cache is computed lazily by the anchored solver on
first run, or ahead of time (optionally with debug `.ply`/`.png` renders):

```bash
scripts/run_grasp_synthesis_conda.sh -m ocir.grasp_synthesis.anchored_bodex.affordance \
  --sequences-root ${OCIR_DATA_ROOT}/processed_data/dex_ycb/sequences --debug-viz
```

For demonstrations from any other reconstruction pipeline, produce
`human_demo.npz` matching the schema in `demo_data.py`.

## One-time calibration

Generate the MANO->Sharpa wrist calibration (writes
`assets/.../grasp_synthesis/bodex/mano_transfer.yml` plus an overlay PNG for
eyeballing; hard-fails on convention errors):

```bash
scripts/run_grasp_synthesis_conda.sh scripts/grasp_synthesis/calibrate_mano_sharpa.py
```

## Running it

Same input/output/Isaac flags as the pure CLI, on sequence dirs that contain
`human_demo.npz`:

```bash
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_synthesis/synthesize_sharpa_anchored_bodex.py \
  --sequence-dir /path/to/sequences/<sequence_id> \
  --out-dir /path/to/output_root \
  --seeds 40 --top-k 8 --opt-iters 500
```

## How it differs from the pure pipeline

- **Seeding**: no object-surface sampling. Seeds are human hand poses from
  demo frames where the hand contacts the object (contact frames are detected
  and cached in `affordance.npz`), retargeted to the Sharpa hand (wrist pose
  relative to the object frame + finger joints via fingertip-fit IK), then
  *relaxed* out of contact (`--relax-flexion`, default 0.15 rad opened on
  flexion joints; `--relax-standoff`, default 1.5 cm wrist pull-back) and
  jittered (`--jitter-pos/--jitter-rot-deg/--jitter-joint`). Seed #0 is the
  unjittered grasp-frame pose. Each seed keeps its un-relaxed retargeted pose
  as its *anchor*.
- **Contact points**: the active subset of the 11 Sharpa contact points is
  selected per sequence from which human hand parts (fingertips/pads/palm)
  actually touched the object, and the force-closure QP's pressure
  constraints are regenerated for that subset (`--no-contact-subset`
  restores all 11).
- **Guidance energies** (on top of the unchanged BODex staged cost):
  an annealed pose prior toward each seed's anchor (`--pose-weight`, zero by
  the stage-0->1 contact switch) and an affordance attraction pulling
  fingertip/pad contact spheres toward the high-heatmap region
  (`--affordance-weight`, `--afford-tau`, decayed over stages 1->2).
- **Success is unchanged**: strict success is still pure force-closure +
  contact distance. Similarity only affects *ranking* among successful seeds
  (`--rank-affordance-weight`, `--rank-pose-weight`) and is reported in the
  records (`affordance_coverage`, `pose_similarity`, `anchor_frame_id`,
  `rank_score`, `retarget_report`, ...).

## Visualization

The CLI submits `anchored_grasp_visualization` jobs (falling back to the
plain task on servers that cannot load it), driven by
`scripts/isaac/visualize_anchored_grasp.py`. On top of the base grasp scene
it adds the object points colored by the affordance heatmap, the human MANO
hand at the seed's anchor frame (green point cloud), and a translucent blue
ghost of the retargeted anchor pose next to the optimized grasp -- all
present in the interactive/exported scene (`scene.usd`, toggleable there),
but the **saved screenshot** deliberately shows only the object, the final
grasp, and the human demo point cloud (the ghost hand and affordance heatmap
are hidden just for that capture, then restored). Like the base visualizer,
the screenshot is a single 1x3 composite of front/side/top orthogonal views.
Same two modes as the base visualizer (see
[Isaac Sim infrastructure](isaac_sim.md)); extra flags:
`--show-affordance/--show-demo-hand/--show-anchor-hand`, `--anchor-opacity`,
`--demo-frame`. It also renders pure-BODex records, skipping whichever
overlays lack data.

## Constraints

- Right-hand demos only (the Sharpa asset is a right hand); left-hand
  sequences fail with a clear error.
- Requires the same CUDA backend as the pure pipeline.
