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
- **Contact points**: all 11 Sharpa contact points stay active by default,
  same as the pure pipeline, so the force-closure QP can recruit opposition
  the human demo did not exhibit. `--contact-subset` opts back in to the
  human-guided restriction: the active subset is selected per sequence from
  which human hand parts (fingertips/pads/palm) actually touched the object,
  and the QP's pressure constraints are regenerated for that subset.
- **Force-closure weight** (`--force-closure-weight`, default **500**,
  original BODex/pure pipeline: 100): overrides only the first entry of
  original BODex's `[grasp, dist, regu]` staged-cost weight triple (still
  `[?, 1000, 10]` otherwise); this is an optimization-time weight on the
  stage-0-only QP energy cost, pulling the optimizer harder toward low
  grasp energy while contact targets are still being negotiated -- it does
  **not** change the pass/fail success threshold itself (`grasp_error` is
  computed independent of any weight; `compute_success` checks
  `grasp_error_max <= 0.001` regardless).
- **Guidance energies** (on top of the unchanged BODex staged cost, which
  as in original BODex runs the force-closure QP energy in stage 0 only and
  has stages 1-2 track the contact targets frozen at the stage-0 switch):
  an annealed pose prior toward each seed's anchor (`--pose-weight`, default
  **0.0 -- disabled**; when enabled, anneals to zero by the stage-0->1
  contact switch) -- note this only controls the optimization-time pull
  toward the anchor pose; seeds themselves are still generated from the
  human demo's retargeted contact frames regardless of this weight -- an
  affordance attraction pulling fingertip/pad contact spheres toward the
  high-heatmap region (`--affordance-weight`, `--afford-tau`, decayed over
  stages 1->2), and an **asymmetric non-penetration penalty**
  (`--penetration-weight`, default **900**): `relu(-signed_distance)^2`
  summed over ALL 37 hand collision spheres (their own all-sphere FK,
  differentiable through the exact SDF gradient). The base staged cost's
  distance term is a symmetric `(dist - target)^2` that is indifferent
  between stopping at the surface and overshooting into the mesh; this term
  supplies the missing asymmetry. Its schedule is ramped **in**, not out:
  zero through stage 0, linearly increased across stage 1, full weight only
  in the final distance=0 stage -- this schedule won a three-way trial on the
  three test sequences (see git history for the trial writeup). Active in
  stage 0 it fights the pose prior / QP sphere-by-sphere and contorts the
  pose; a variant keeping the force-closure QP live in all three stages (with
  no penalty) was also tried and dropped. Note this penalty softens the
  **final** grasp's penetration but by construction cannot clean the stage-0
  `pregrasp` snapshot (captured at `opt_progress == 0.6`, before the ramp) --
  that is handled geometrically by the pregrasp retreat (`--pregrasp-clearance`,
  see the four-stage table). `--pose-weight` still defaults to **0**: a
  stronger pose prior was found to pull finger geometry back into penetrating
  retarget anchors on hard cases. Pass a positive value to re-enable it.
- **Self-collision** (`--selfcollision-weight`, default 1000): pairwise
  sphere-vs-sphere overlap energy, `relu(min_dist - center_dist)^2` summed
  over every pair of hand-collision spheres EXCEPT same-link spheres and
  URDF-adjacent links (fingers/palm segments that share a joint surface and
  are expected to sit close in most poses). Adjacency is derived once,
  directly from the asset's own URDF joint tree (`build_self_collision_pairs`
  in `guidance.py`) -- not reused from `bodex_curobo_v2`'s own
  `self_collision_ignore` map, which is keyed to a finer virtual-link naming
  scheme (`*_MCP_VL`, `*_elastomer`, ...) that doesn't correspond 1:1 to this
  pipeline's coarser collision-sphere set. Unlike every other guidance term
  this one is **not scheduled**: fingers must never interpenetrate each
  other at any point in the optimization, so it is at full weight in all
  three stages.
- **Success is unchanged**: strict success is still pure force-closure +
  contact distance. Similarity only affects *ranking* among successful seeds
  (`--rank-affordance-weight`, `--rank-pose-weight`) and is reported in the
  records (`affordance_coverage`, `pose_similarity`, `anchor_frame_id`,
  `rank_score`, `retarget_report`, ...).

## Four-stage grasp poses

Every grasp record carries a `stages` dict with four wrist+finger poses
(`grasp_stages.py`; each stage is `{position, orientation (wxyz),
joints {name: rad}}` in the object canonical frame), so downstream consumers
(the `grasp_traj` pipeline) never have to repair or invent poses themselves:

| Stage | Origin |
| --- | --- |
| `raw_grasp` | The fully optimized final action, unmodified (identical to the record's `action`). Usually penetrates the object by several mm. |
| `pregrasp` | Mid-optimization snapshot taken the moment the staged contact cost enters its middle (1cm-standoff) stage -- i.e. the pose optimized under the initial ~2cm-standoff target (the end of the optimizer's first, force-closure-scored phase). Reproduces BODex's own `save_qpos`/`mid_result` mechanism by *subclassing* the frozen optimizer core (`SnapshotBodexNewtonOpt`), never modifying it. Because the snapshot precedes the penetration-penalty ramp it frequently sits inside the object (object-dependent, some seeds 5-18mm deep), so it is **retreated straight along the palm approach axis** (joints frozen; only the wrist position moves) to the smallest backoff whose full-hand SDF clearance reaches `--pregrasp-clearance` (default 5mm) -- a guaranteed collision-free pre-grasp pose. `stage_report` records `pregrasp_snapshot_clearance_m` (before retreat), `pregrasp_backoff_m`, and the achieved `pregrasp_clearance_m`. |
| `grasp` | `raw_grasp` retreated along the pregrasp -> raw interpolation path to the largest fraction whose full-hand SDF clearance (all 37 collision spheres) is still >= `--contact-clearance` (2mm): deliberately SHY of the contact boundary -- the tightest converged pose is never a configuration to physically reach (UltraDexGrasp's stance); contact force comes from squeeze. Since the `pregrasp` end of the path is now retreated clear of the object, a valid retreat point at the contact margin always exists (unless the pregrasp retreat itself hit its backoff cap, in which case the pregrasp pose is reused). |
| `squeeze` | Articulation-BODex extrapolation: `grasp + clamp(raw_grasp - pregrasp, min=--squeeze-min)` per joint -- the delta is the optimizer's FULL closing motion (its intended force direction, including whatever the retreat removed), the 0.15 rad floor applied to flexion channels only (never abduction/adduction), clamped to joint limits. A bounded drive-force request past contact. |

`stage_report` records the retreat fraction and the SDF clearance of each
stage. The Isaac visualization renders records with stages as a **4x3 grid**
(one labeled row per stage x front/side/top orthogonal views); the exported
`scene.usd` contains all four stage hands with only `grasp` visible by
default (the rest are toggleable).

## Visualization

The CLI submits `anchored_grasp_visualization` jobs (falling back to the
plain task on servers that cannot load it), driven by
`scripts/isaac/visualize_anchored_grasp.py`. On top of the base grasp scene
it adds the object points colored by the affordance heatmap, the human MANO
hand at the seed's anchor frame (green point cloud), and a translucent blue
ghost of the retargeted anchor pose next to the optimized grasp -- all
present in the interactive/exported scene (`scene.usd`, toggleable there),
but the **saved screenshot** deliberately shows only the object, the grasp
hand, and the human demo point cloud (the ghost hand and affordance heatmap
are hidden just for that capture, then restored). For four-stage records
the screenshot is a labeled **4x3 grid** (pregrasp / raw_grasp / grasp /
squeeze rows x front/side/top views, individual rows also saved as
`stage_<name>.png`); records without stages keep the single 1x3 composite.
Same two modes as the base visualizer (see
[Isaac Sim infrastructure](isaac_sim.md)); extra flags:
`--show-affordance/--show-demo-hand/--show-anchor-hand`, `--anchor-opacity`,
`--demo-frame`. It also renders pure-BODex records, skipping whichever
overlays lack data.

## Constraints

- Right-hand demos only (the Sharpa asset is a right hand); left-hand
  sequences fail with a clear error.
- Requires the same CUDA backend as the pure pipeline.
