# OCIR Grasp Synthesis (cuRobo v2)

Sharpa Wave dexterous-hand grasp synthesis on official NVLabs cuRobo v2, with
Isaac Sim visualization support. This is an independent port of the BODex
grasp-synthesis algorithm onto official cuRobo v2 -- it never imports
anything from BODex itself. Split out from a larger OCIR repository that also
carries an original-BODex reference implementation and unrelated DexYCB
replay/retargeting work; this workspace keeps only the cuRobo v2 port and
what it needs to run and visualize.

## Overview

The pipeline takes one manipulated-object mesh at a time ("a sequence"),
runs a BODex-style bilevel grasp-synthesis optimization (staged contact
cost + force-closure QP, ported to run natively on official cuRobo v2's
optimizer core) to find Sharpa Wave hand poses that grasp it, and optionally
renders the best result in Isaac Sim.

```text
sequence directory (object mesh) -> solve_sharpa_bodex() -> grasp_*.json / summary.json
                                                           -> Isaac Sim visualization (optional)
```

A second, human-demonstration-guided pipeline (**anchored BODex**, see
[Anchored BODex](#anchored-bodex-human-demo-guided-synthesis)) additionally
reads a reconstructed human video demonstration (MANO hand + object
trajectory) from the sequence directory and anchors the same optimization to
it: seeds come from retargeted human contact-frame poses, the contact-point
set and an affordance heatmap come from where the human actually touched the
object, and annealed guidance energies keep the result similar to the demo
while the force-closure QP still decides success.

```text
sequence directory (object mesh + human_demo.npz)
    -> solve_sharpa_anchored_bodex() -> grasp_*.json / summary.json (+ affordance.npz cache)
```

Object meshes are convex-decomposed with `coacd` for contact evaluation;
the hand's own link meshes currently use a single whole-mesh convex hull
each (see [Known Limitations](#known-limitations)).

## Repository Layout

```text
src/ocir/
  grasp_synthesis/
    assets.py            # Sharpa Wave asset-config resolution (URDF, USD, collision, joint limits)
    object_surface.py    # generic "sequence directory" -> object mesh + surface points loader
    synthesize_sharpa_bodex_curobo_v2.py   # CLI entry point (pure object-only pipeline)
    bodex_curobo_v2/     # the grasp-synthesis algorithm itself:
                          # grasp-energy QP, staged contact cost (mesh-mesh + sphere-mesh),
                          # BODex-faithful momentum optimizer, seed generator, cuRobo v2 rollout wiring
    anchored_bodex/      # human-demo-guided pipeline built on top of (never modifying) bodex_curobo_v2:
                          # demo_data.py (human_demo.npz loader), affordance.py (heatmap + contact
                          # frames, lazily cached per sequence), demo_analysis.py (grasp window +
                          # contact roles), retarget.py (MANO->Sharpa calibration + fingertip-fit IK),
                          # seed_generator.py (relaxed retargeted contact-frame seeds), guidance.py,
                          # rollout.py, solver.py, synthesize_sharpa_anchored_bodex.py (CLI)
  isaac/
    visualize_grasp.py   # renders a synthesized grasp in Isaac Sim (persistent-server or standalone)
    replay_dexycb.py, sim_cli.py   # shared geometry/CLI helpers visualize_grasp.py depends on
  sim/
    control_client.py, isaac_server.py, start_isaacsim_server.py   # persistent Isaac control server
  dexycb/
    prepare_dexycb_subset.py, mano_model.py, labels.py   # DexYCB subset prep + label/MANO helpers
    export_grasp_sequences.py   # DexYCB labels -> per-sequence human_demo.npz exporter
                                 # (needed once per sequence before anchored-BODex runs)

scripts/            # thin conda-env wrappers around the modules above (same CLI flags)
assets/robots/hands/sharpa_wave/   # hand URDF, meshes, USD, collision spheres, manip configs
third_party/curobo  # official NVLabs cuRobo v2 (git submodule)
```

`bodex_curobo_v2` never imports anything from BODex. See its module
docstrings for algorithm details (mesh processing, staged optimization,
force-closure QP, seed generation).

## Installation

1. Initialize the cuRobo submodule (official NVLabs cuRobo, not a fork):
   ```bash
   git submodule update --init --recursive third_party/curobo
   ```
2. Create/use a single conda environment for everything -- grasp synthesis
   and Isaac Sim visualization both run in it. All of this repo's helper
   scripts (`scripts/run_grasp_synthesis_conda.sh`,
   `scripts/run_isaacsim_conda.sh`) default to an environment named
   `env_isaacsim`, overridable via the `OCIR_ISAACSIM_CONDA_ENV` /
   `OCIR_GRASP_SYNTHESIS_CONDA_ENV` / `OCIR_CUROBO_CONDA_ENV` env vars if you
   name yours differently. It needs:

   | Package | Notes |
   | --- | --- |
   | `isaacsim` | Isaac Sim itself (tested against 5.1.x). Only needed for the visualization half; grasp synthesis alone doesn't import it. |
   | `torch` | CUDA build. Must be importable before `coacd` in-process (see below). |
   | `warp-lang` | cuRobo v2's GPU kernels (mesh SDF queries, etc.) run on this. |
   | `coal` | GJK/EPA convex-convex distance. Installed from **conda-forge**, not pip (`conda install -c conda-forge coal`) -- a plain `pip install coal` will not get you this package. |
   | `coacd` | Convex decomposition of object meshes (`pip install coacd`). |
   | `trimesh`, `PyYAML`, `numpy` | Mesh I/O and config loading. |

   None of this is declared in this repo's own `pyproject.toml` (which only
   lists `numpy`/`PyYAML`) -- the heavier ML/robotics/sim stack is assumed
   to already be present in the environment, not managed or pinned by this
   repo. Once the environment has all of the above, install this package
   itself into it:
   ```bash
   pip install -e .
   ```
3. `coacd` must be importable *after* `torch` has already been imported in
   the same process -- it bundles its own native OpenMP runtime that
   segfaults if loaded first. `contact_world.py` already orders its imports
   this way; if you import `coacd` yourself elsewhere, import `torch` first.

Grasp synthesis requires a CUDA-capable GPU (`solve_sharpa_bodex` raises if
`torch.cuda.is_available()` is false).

## Data Preparation

Grasp synthesis is dataset-agnostic. It reads a **sequence directory** --
any directory that provides one object mesh:

```text
<sequence_dir>/
  <object_name>.obj (or .stl)   # required: exactly one mesh file, unless overridden below
  points.xyz                    # optional: pre-sampled surface points, one "x y z" per line
  sequence.json                 # optional: {"object_mesh": "...", "points": "...", ...} overrides
  human_demo.npz                # anchored BODex only: MANO hand + object trajectory in the
                                 # object canonical frame (schema in anchored_bodex/demo_data.py)
  affordance.npz                # anchored BODex only: heatmap + contact-frame cache; created
                                 # automatically on first run, reused afterwards
```

Resolution rules (`ObjectSurface.from_sequence_dir`, in
`src/ocir/grasp_synthesis/object_surface.py`):

- **Object mesh**: if `sequence.json` sets `"object_mesh"` (path relative to
  the sequence dir, or absolute), that file is used. Otherwise the loader
  looks for exactly one `*.obj`/`*.stl` file directly in the sequence
  directory -- it errors if there are zero or more than one, so a
  directory with multiple mesh candidates must set `"object_mesh"`
  explicitly to disambiguate.
- **Surface points**: if `sequence.json` sets `"points"`, that file is used.
  Otherwise `points.xyz` in the sequence directory is used if present.
  Otherwise 2048 points are sampled directly from the mesh surface
  (`trimesh.sample.sample_surface`). These points are only used for a
  rough object bounding-box/center-of-mass estimate (root-pose seed bounds,
  gravity center) -- they do not otherwise affect contact evaluation.
- **Metadata**: `sequence_id` defaults to the sequence directory's name;
  `object_name` defaults to the mesh file's stem. Any other keys in
  `sequence.json` are passed through into `metadata` and end up in the
  output grasp records.

Batch runs treat every immediate subdirectory of a given root as one
sequence (see [Quick Start](#quick-start)).

### Regenerating sequences from DexYCB

`src/ocir/dexycb/prepare_dexycb_subset.py` regenerates the object-mesh
sequence dirs and the manifest from raw DexYCB archives (`--help` documents
subject/camera/sequence-count flags). For anchored BODex, additionally run
the demo exporter once per sequence set:

```bash
scripts/run_grasp_synthesis_conda.sh scripts/dexycb/export_grasp_sequences.py \
  --manifest ${OCIR_DATA_ROOT}/processed_data/dex_ycb/manifests/selected_5_sequences.json \
  --sequences-root ${OCIR_DATA_ROOT}/processed_data/dex_ycb/sequences
```

It writes `human_demo.npz` (per-frame MANO vertices/keypoints + object pose,
pre-transformed into the object canonical frame) into each sequence dir and
records it in `sequence.json`. The `affordance.npz` cache is then computed
lazily by the anchored solver itself, or ahead of time (optionally with debug
`.ply`/`.png` renders) via:

```bash
scripts/run_grasp_synthesis_conda.sh -m ocir.grasp_synthesis.anchored_bodex.affordance \
  --sequences-root ${OCIR_DATA_ROOT}/processed_data/dex_ycb/sequences --debug-viz
```

To build a sequence directory by hand for a new object, all you need is the
object mesh (`.obj`/`.stl`) in its own directory; `points.xyz`/
`sequence.json` are optional refinements. The pure pipeline needs nothing
else; anchored BODex needs `human_demo.npz` from whatever reconstruction
pipeline produced the demonstration.

## Quick Start

Set `OCIR_DATA_ROOT` to wherever your sequence inputs and outputs should
live (an external data root, not part of this repo; several scripts default
paths under it, e.g. `${OCIR_DATA_ROOT}/testing/...`, but no script
requires this exact layout -- `--sequence-dir`/`--sequences-root`/
`--out-dir` accept any path).

Check the backend is available (official-cuRobo-v2 import isolation, all
`bodex_curobo_v2` modules, `coal`/`coacd`/`warp` importable) without running
synthesis:

```bash
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_synthesis/synthesize_sharpa_bodex_curobo_v2.py --check-only
```

Start the persistent Isaac Sim control server (needed for the default
`--isaac-mode server`; skip this if you'll use `--isaac-mode standalone`):

```bash
scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py --width 1280 --height 720
```

Run grasp synthesis for a single sequence. `--sequence-dir`/
`--sequences-root` (exactly one, mutually exclusive) and `--out-dir` are
always required -- there is no default input source or output directory:

```bash
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_synthesis/synthesize_sharpa_bodex_curobo_v2.py \
  --sequence-dir /path/to/sequences/<sequence_id> \
  --out-dir /path/to/output_root \
  --seeds 20 --top-k 8 --opt-iters 500
```

Run grasp synthesis in batch mode (every immediate subdirectory of the given
root is treated as one sequence):

```bash
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_synthesis/synthesize_sharpa_bodex_curobo_v2.py \
  --sequences-root /path/to/sequences \
  --out-dir /path/to/output_root \
  --seeds 20 --top-k 8 --opt-iters 500
```

## Common Workflows

### `synthesize_sharpa_bodex_curobo_v2.py` flags

| Flag | Meaning |
| --- | --- |
| `--sequence-dir <path>` | Run exactly one sequence directory. |
| `--sequences-root <path>` | Batch mode: run every immediate subdirectory as a sequence. |
| `--out-dir <path>` | Required. Per-sequence output goes to `<out-dir>/<sequence_name>/`. |
| `--object-mesh <path>` | Override the auto-discovered object mesh. Only valid with `--sequence-dir`. |
| `--asset-config <path>` | Override the Sharpa Wave hand config yaml (defaults to `assets/robots/hands/sharpa_wave/sharpa_wave_right.yml`). |
| `--seeds <int>` (default 20) | Number of parallel grasp-pose seeds optimized per sequence. |
| `--top-k <int>` (default 8) | How many top-scoring successful grasps to keep/write. |
| `--opt-iters <int>` (default 500) | Optimizer iterations. Progress is logged every 50 (`bodex_newton iter N/total`). |
| `--seed <int>` (default 0) | RNG seed for `torch`/`numpy`. |
| `--grasp-threshold`, `--distance-threshold` | Strict-success thresholds (grasp error, contact distance). |
| `--isaac-visualize` / `--no-isaac-visualize` (default on) | Whether to visualize the best (or best-failed) grasp after solving. |
| `--isaac-visualize-failed` / `--no-...` (default on) | If no seed strictly succeeded, visualize the best-ranked failed seed instead of skipping. |
| `--isaac-mode {server,standalone}` (default `server`) | `server` submits to the persistent control server (see below). `standalone` launches a one-shot local Isaac Sim window per grasp instead -- for a headed machine with no persistent server running. |
| `--control-host`, `--control-port` (default `127.0.0.1:8765`) | Persistent server address, used when `--isaac-mode server`. |
| `--isaac-width`, `--isaac-height`, `--isaac-tabletop-z`, `--isaac-hold-open[-seconds]`, `--isaac-show-object-points` | Visualization rendering options, passed through to `visualize_grasp.py`. The hold is timed (default 10 s); in standalone mode the one-shot Isaac window closes itself afterwards so batch runs continue unattended. |
| `--check-only` | Only validate the backend; no sequence/out-dir needed. |
| `--strict-success-exit-code` | Exit 1 if any sequence had zero strictly-successful seeds. |

Exit code is nonzero if any sequence's solver crashed or its visualization
failed (or, with `--strict-success-exit-code`, if any sequence had zero
successful seeds); otherwise 0, even if some individual sequences report
`"ok": false` in their summary.

### Anchored BODex: human-demo-guided synthesis

`scripts/grasp_synthesis/synthesize_sharpa_anchored_bodex.py` runs the
human-demonstration-guided variant. One-time setup: generate the
MANO->Sharpa wrist calibration (writes
`assets/.../grasp_synthesis/bodex/mano_transfer.yml` plus an overlay PNG for
eyeballing; hard-fails on convention errors):

```bash
scripts/run_grasp_synthesis_conda.sh scripts/grasp_synthesis/calibrate_mano_sharpa.py
```

Then run it exactly like the pure CLI (same input/output/Isaac flags), on
sequence dirs that contain `human_demo.npz`:

```bash
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_synthesis/synthesize_sharpa_anchored_bodex.py \
  --sequence-dir /path/to/sequences/<sequence_id> \
  --out-dir /path/to/output_root \
  --seeds 40 --top-k 8 --opt-iters 500
```

How it differs from the pure pipeline:

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
- **Visualization**: the CLI submits `anchored_grasp_visualization` jobs
  (falling back to the plain task on servers that cannot load it), driven by
  `scripts/isaac/visualize_anchored_grasp.py`. On top of the base grasp
  scene it adds the object points colored by the affordance heatmap, the
  human MANO hand at the seed's anchor frame (green point cloud), and a
  translucent blue ghost of the retargeted anchor pose next to the optimized
  grasp -- all present in the interactive/exported scene (`scene.usd`,
  toggleable there), but the **saved screenshot** deliberately shows only the
  object, the final grasp, and the human demo point cloud (the ghost hand
  and affordance heatmap are hidden just for that capture, then restored).
  Like the base visualizer, the screenshot is a single 1x3 composite of
  front/side/top orthogonal views. Same two modes as the base visualizer --
  persistent server (`--mode webrtc`, picked up by a hot-reloading running
  server without restart) or one-shot pop-up window (`--mode local`); extra
  flags: `--show-affordance/--show-demo-hand/--show-anchor-hand`,
  `--anchor-opacity`, `--demo-frame`. It also renders pure-BODex records,
  skipping whichever overlays lack data.

Right-hand demos only (the Sharpa asset is a right hand); left-hand
sequences fail with a clear error. Requires the same CUDA backend as the
pure pipeline.

### Isaac Sim visualization: persistent server vs. standalone

Two ways to render a grasp, both driving the same
`src/ocir/isaac/visualize_grasp.py`:

- **Persistent server** (`--isaac-mode server`, the default): start it once
  with `scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py`,
  then every synthesis run submits a `grasp_pose_visualization` job to it
  over a small JSON control API (`src/ocir/sim/control_client.py`,
  `GET /status`, `POST /run`, `POST /shutdown`, `GET /jobs/<id>`). Keeps one Isaac Sim
  instance warm across many sequences -- much faster for batch runs.
  `start_isaacsim_server.py --status` prints server status;
  `--shutdown-server` asks it to exit.
- **Standalone** (`--isaac-mode standalone`, or run
  `scripts/isaac/visualize_grasp.py --mode local` directly): launches a
  fresh, one-shot Isaac Sim window per grasp and exits when done. No
  persistent server needed -- use this to deploy on another headed machine
  without keeping a background Isaac Sim process running. Slower per-grasp
  (full Isaac Sim startup each time) but has no server dependency.

`visualize_grasp.py` can also be run by hand against a single grasp record:

```bash
scripts/run_isaacsim_conda.sh scripts/isaac/visualize_grasp.py \
  --mode local \
  --sequence-dir /path/to/sequences/<sequence_id> \
  --grasp-json /path/to/output_root/<sequence_id>/grasp_000.json \
  --out-dir /path/to/some/output/dir
```

Important flags: `--tabletop-z`, `--show-table`/`--show-object-points`,
`--hold-open[-seconds]` (timed viewport hold after rendering; add
`--hold-open-until-closed` in `--mode local` to instead keep the window
alive until you close it manually),
`--camera-distance-scale`/`--camera-target-offset`/`--camera-focus-max-ratio`
(camera framing). With `--mode webrtc` (submits to the persistent server),
`--use-raw-object-pose` plus `--manifest`/`--sequence-id`/`--frame-id`
optionally replay the object's real recorded pose from a DexYCB manifest
instead of dropping it onto the table at the origin -- this is unrelated to
how the sequence's object mesh itself is resolved.

**Camera / screenshot**: the saved screenshot is a single 1-row x 3-column
image (`isaac_grasp.png`) combining 3 mutually orthogonal views -- front
(along -Y), side (along +X), and top (straight down +Z), each labeled --
rather than one oblique shot. `--camera-distance-scale` (default 1.5) sets
how far the camera sits from the focus target, as a multiple of the scene's
bounding radius. Framing is object-centered: if including the hand would
expand the box beyond `--camera-focus-max-ratio` (default 1.6) times the
object's own extent -- which happens for a poorly-converged grasp whose hand
ends up far from the object -- the camera frames on the object alone instead
of zooming out to fit both, so the shot stays close and legible.

## Configuration

The hand asset is described by a single yaml config, default
`assets/robots/hands/sharpa_wave/sharpa_wave_right.yml`, loaded by
`src/ocir/grasp_synthesis/assets.py`. Fields you're likely to touch:

- `urdf_path`, `usd_path` -- hand geometry, relative to the config file's directory.
- `base_link`, `ee_link` -- kinematic root/end-effector link names.
- `joint_order` -- the 22 Sharpa Wave joint names, in the order actions are ordered.
- `joint_limits` -- per-joint `[lower, upper]` radians.
- `contact_points.config_path`, `collision_spheres.path` -- contact-point and analytic-sphere-collision configs (`assets/robots/hands/sharpa_wave/contact_points/`, `.../collision/curobo/`).
- `bodex.<key>` -- paths to the BODex-side configs (`grasp_synthesis_config`, `robot_config`, `hand_pose_transfer`), under `assets/robots/hands/sharpa_wave/grasp_synthesis/bodex/`.

Pass `--asset-config <path>` to either script to point at a different hand
config; `SharpaWaveAsset` validates that every referenced path exists at
load time.

## Outputs

Given `--out-dir <root>`, each sequence writes to `<root>/<sequence_name>/`:

- `grasp_<rank>.json` (rank `000`, `001`, ...) -- one per top-`k` strictly
  successful seed, best score first. Each record includes the full 29-D
  action (7-D root pose + joint angles), `score`, `success`,
  `grasp_error_max`, `dist_error`, `object_mesh`, `sequence_dir`,
  `sequence_id`/`object_name` (from the sequence's metadata), and
  `metric_summary`.
- `failed_grasp_<rank>.json` -- written instead, ranked by score, if zero
  seeds strictly succeeded.
- `summary.json` (per sequence) -- `ok`, the best grasp record's fields
  merged in on success (or `error`/`top_failed_grasps` on failure),
  `seed_count`, `top_k`, `top_grasps` (list of `{rank, seed_index, score,
  grasp_json}`), `opt_iters`.
- `isaac_visualization/` (if `--isaac-visualize`) -- `isaac_grasp.png` (one
  1x3 composite of front/side/top screenshots), `scene.usd` (full USD
  stage), `report.json` (object/hand mesh counts, camera framing, resolved
  object pose, `score`).

The top-level run also writes `<root>/summary.json`: `ok`, `curobo_path`,
`solver_failures`, `failed_visualizations`, `strict_failures`, and `runs`
(each sequence's per-sequence summary, embedded).

## Known Limitations

- **Robot links use a single whole-mesh convex hull each**, while the
  object mesh is convex-decomposed into multiple parts via `coacd`
  (normalize -> decompose -> rescale; see `_coacd_convex_parts`/
  `_load_convex_parts` in
  `src/ocir/grasp_synthesis/bodex_curobo_v2/contact_world.py`). This
  under-approximates contact for concave hand-link regions. See that
  module's docstring.
- `coacd` must be imported after `torch` (see the import-order comment in
  `contact_world.py`) or it segfaults due to a native OpenMP/runtime
  library conflict -- see [Installation](#installation).
- If you run more than one checkout of this codebase on the same machine,
  make sure the persistent Isaac Sim server is actually started from the
  checkout you're editing (`start_isaacsim_server.py --status` echoes back
  its loaded `task_modules` paths) -- a stale server process serving an
  older checkout will silently run outdated code and fail in confusing
  ways (e.g. a removed CLI flag reported as "required").
