# OCIR Grasp Synthesis (cuRobo v2)

Sharpa Wave dexterous-hand grasp synthesis on official NVLabs cuRobo v2, with
Isaac Sim visualization and physics simulation. This is an independent port
of the BODex grasp-synthesis algorithm onto official cuRobo v2 -- it never
imports anything from BODex itself. Split out from a larger OCIR repository;
this workspace keeps only the cuRobo v2 port and what it needs to run,
visualize, and simulate.

## Overview

Four pipelines, each consuming a **sequence directory** (one manipulated
object, optionally with a reconstructed human demonstration):

1. **BODex on cuRobo v2** (`bodex_curobo_v2`) -- the core bilevel grasp
   optimization (staged contact cost + force-closure QP) over one object
   mesh, producing ranked Sharpa Wave grasp poses.
   -> [docs/bodex_curobo_v2.md](docs/bodex_curobo_v2.md)
2. **Anchored BODex** (`anchored_bodex`) -- the same optimization anchored to
   a human video demonstration: seeds from retargeted human contact-frame
   poses, per-sequence contact-point subsets and an affordance heatmap from
   where the human actually touched, annealed similarity guidance -- with
   force closure still deciding success.
   -> [docs/anchored_bodex.md](docs/anchored_bodex.md)
3. **Grasp trajectory + physics simulation** (`grasp_traj`) -- turns one
   synthesized grasp plus its demo into a full manipulation trajectory
   (retarget -> open -> planned transit -> close -> grasp hold -> carry) and
   plays it back in Isaac Sim with real PhysX physics, reporting lift/drop
   metrics and a video. -> [docs/grasp_traj.md](docs/grasp_traj.md)
4. **Closed-loop full trajectory** (`full_traj`) -- preserves an existing
   completed grasp, replaces the vertical lift with the recorded MANO/object
   carry path, and adjusts only the wrist online so the dynamic object follows
   that translation+orientation path without frame-timing correspondence.
   -> [docs/full_traj.md](docs/full_traj.md)

```text
sequence dir (object mesh)                    --[1]-->  grasp_*.json / summary.json
sequence dir (object mesh + human_demo.npz)   --[2]-->  grasp_*.json / summary.json
grasp record + sequence dir                   --[3]-->  trajectory.npz -> video.mp4 / report.json
completed grasp trajectory + human demo       --[4]-->  full_traj reference -> closed-loop video / path metrics
```

All Isaac-facing steps run either against a **persistent Isaac Sim server**
(fast batch runs) or as **one-shot standalone instances** (no background
process) -- see [docs/isaac_sim.md](docs/isaac_sim.md).

## Documentation

| Document | Contents |
| --- | --- |
| [docs/bodex_curobo_v2.md](docs/bodex_curobo_v2.md) | Core algorithm, CLI flags, outputs, hand asset config, limitations |
| [docs/anchored_bodex.md](docs/anchored_bodex.md) | Human-demo-guided synthesis: demo data prep, calibration, guidance, visualization |
| [docs/grasp_traj.md](docs/grasp_traj.md) | Trajectory generation + PhysX simulation, all flags, diagnostics, **changelog of design iterations** |
| [docs/full_traj.md](docs/full_traj.md) | Carry-only closed-loop wrist control, MANO/object reference generation, path metrics |
| [docs/isaac_sim.md](docs/isaac_sim.md) | Persistent server / standalone modes, visualization, video encoding, troubleshooting |

## Repository Layout

```text
src/ocir/
  grasp_synthesis/
    assets.py            # Sharpa Wave asset-config resolution (URDF, USD, collision, joint limits)
    object_surface.py    # generic "sequence directory" -> object mesh + surface points loader
    synthesize_sharpa_bodex_curobo_v2.py   # CLI entry point (pure object-only pipeline)
    bodex_curobo_v2/     # the grasp-synthesis algorithm itself (frozen; never modified by new features)
    anchored_bodex/      # human-demo-guided pipeline built on top of bodex_curobo_v2
  grasp_traj/            # Stage A trajectory generation (schema, segments, clearance,
                          # switch-frame search, cuRobo transit planner, generator, CLI)
  full_traj/             # MANO/object path reference, SE(3) controller/metrics, Isaac task
  isaac/
    visualize_grasp.py   # grasp visualization (also the server's task-registration hub)
    visualize_anchored_grasp.py   # anchored-variant visualization overlays
    simulate_grasp_traj.py        # Stage B trajectory physics simulation
    replay_dexycb.py, sim_cli.py  # shared frame-mapping/camera/video/CLI helpers
  sim/
    control_client.py, isaac_server.py, start_isaacsim_server.py   # persistent Isaac control server
  dexycb/
    prepare_dexycb_subset.py, mano_model.py, labels.py   # DexYCB subset prep + label/MANO helpers
    export_grasp_sequences.py   # DexYCB labels -> per-sequence human_demo.npz exporter

scripts/            # thin conda-env wrappers around the modules above (same CLI flags)
assets/robots/hands/sharpa_wave/   # hand URDF, meshes, USD, collision spheres, manip configs
docs/               # detailed per-subsystem documentation (see table above)
third_party/curobo  # official NVLabs cuRobo v2 (git submodule)
```

## Installation

1. Initialize the cuRobo submodule (official NVLabs cuRobo, not a fork):
   ```bash
   git submodule update --init --recursive third_party/curobo
   ```
2. Create/use a single conda environment for everything -- grasp synthesis
   and Isaac Sim both run in it. All of this repo's helper scripts
   (`scripts/run_grasp_synthesis_conda.sh`, `scripts/run_isaacsim_conda.sh`)
   default to an environment named `env_isaacsim`, overridable via the
   `OCIR_ISAACSIM_CONDA_ENV` / `OCIR_GRASP_SYNTHESIS_CONDA_ENV` /
   `OCIR_CUROBO_CONDA_ENV` env vars. It needs:

   | Package | Notes |
   | --- | --- |
   | `isaacsim` | Isaac Sim itself (tested against 5.1.x). Only needed for visualization/simulation; grasp synthesis alone doesn't import it. |
   | `torch` | CUDA build. Must be importable before `coacd` in-process (see below). |
   | `warp-lang` | cuRobo v2's GPU kernels (mesh SDF queries, etc.) run on this. |
   | `coal` | GJK/EPA convex-convex distance. Installed from **conda-forge**, not pip (`conda install -c conda-forge coal`) -- a plain `pip install coal` will not get you this package. |
   | `coacd` | Convex decomposition of object meshes (`pip install coacd`). |
   | `trimesh`, `PyYAML`, `numpy` | Mesh I/O and config loading. |
   | `ffmpeg` (system binary) | H.264 video encoding for simulation/replay outputs (OpenCV fallback exists but is not H.264). |

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

Grasp synthesis and trajectory generation require a CUDA-capable GPU.

Verify the backend without running anything:

```bash
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_synthesis/synthesize_sharpa_bodex_curobo_v2.py --check-only
```

## Data Preparation

All pipelines read a **sequence directory** -- any directory that provides
one object mesh:

```text
<sequence_dir>/
  <object_name>.obj (or .stl)   # required: exactly one mesh file, unless sequence.json overrides
  points.xyz                    # optional: pre-sampled surface points
  sequence.json                 # optional: {"object_mesh": "...", "points": "...", ...} overrides
  human_demo.npz                # anchored BODex / grasp_traj: MANO hand + object trajectory
  affordance.npz                # heatmap + contact-frame cache; created automatically, reused
```

Resolution rules live in `src/ocir/grasp_synthesis/object_surface.py`
(`ObjectSurface.from_sequence_dir`): `sequence.json` overrides win, otherwise
exactly one `*.obj`/`*.stl` must exist; surface points fall back to
`points.xyz` then to 2048 mesh samples (used only for seed bounds, not
contact evaluation). Batch runs treat every immediate subdirectory of a
given root as one sequence.

To regenerate DexYCB sequence dirs and manifests from raw archives, use
`src/ocir/dexycb/prepare_dexycb_subset.py`; for the demo-driven pipelines,
additionally export `human_demo.npz` per sequence -- see
[docs/anchored_bodex.md](docs/anchored_bodex.md#demo-data-preparation-once-per-sequence-set).

For a new object from any source: put its mesh alone in a directory. The
pure pipeline needs nothing else; the demo-driven pipelines need
`human_demo.npz` from whatever reconstruction produced the demonstration.

## Deployment

Set `OCIR_DATA_ROOT` to wherever sequence inputs and outputs live (an
external data root, not part of this repo; several scripts default paths
under it, but every CLI accepts explicit `--sequence-dir` /
`--sequences-root` / `--out-dir` paths).

**Persistent Isaac Sim server** (recommended for batch runs; keeps one Isaac
instance warm and hot-reloads task code per job):

```bash
scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py --width 1280 --height 720
```

Every pipeline defaults to `--isaac-mode server` and submits jobs to it. On
a machine without a background server (e.g. a headed workstation), pass
`--isaac-mode standalone` instead -- each job launches a one-shot Isaac Sim
instance and closes itself. Details, the control API, and troubleshooting
(including the stale-server-from-another-checkout pitfall):
[docs/isaac_sim.md](docs/isaac_sim.md).

## Quick Start

```bash
# 1. Pure grasp synthesis (one sequence; use --sequences-root for batch)
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_synthesis/synthesize_sharpa_bodex_curobo_v2.py \
  --sequence-dir ${OCIR_DATA_ROOT}/processed_data/dex_ycb/sequences/<sequence_id> \
  --out-dir ${OCIR_DATA_ROOT}/testing/grasp_synthesis/bodex_v2 \
  --seeds 20 --top-k 8 --opt-iters 500

# 2. Human-demo-anchored synthesis (sequence dir must contain human_demo.npz)
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_synthesis/synthesize_sharpa_anchored_bodex.py \
  --sequence-dir ${OCIR_DATA_ROOT}/processed_data/dex_ycb/sequences/<sequence_id> \
  --out-dir ${OCIR_DATA_ROOT}/testing/grasp_synthesis/anchored_bodex \
  --seeds 40 --top-k 8 --opt-iters 500

# 3. Trajectory generation + physics simulation from a synthesized grasp
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_traj/generate_grasp_traj.py \
  --sequence-dir ${OCIR_DATA_ROOT}/processed_data/dex_ycb/sequences/<sequence_id> \
  --synthesis-out-dir ${OCIR_DATA_ROOT}/testing/grasp_synthesis/anchored_bodex/<sequence_id> \
  --out-dir ${OCIR_DATA_ROOT}/testing/grasp_traj/<sequence_id>

# 4. Replace one completed lift with a MANO/object full-trajectory reference
scripts/run_grasp_synthesis_conda.sh \
  scripts/full_traj/generate_full_traj.py \
  --grasp-traj-dir ${OCIR_DATA_ROOT}/testing/grasp_traj/<sequence_id>/grasp_pose_1 \
  --out-dir ${OCIR_DATA_ROOT}/testing/full_traj/<sequence_id>/grasp_pose_1

# 5. Run carry-only closed-loop wrist control (submits to the Isaac server)
scripts/run_isaacsim_conda.sh \
  scripts/full_traj/simulate_full_traj.py \
  --trajectory-dir ${OCIR_DATA_ROOT}/testing/full_traj/<sequence_id>/grasp_pose_1 \
  --out-dir ${OCIR_DATA_ROOT}/testing/full_traj/<sequence_id>/grasp_pose_1/isaac_sim
```

Each pipeline's flags, outputs, and internals are documented in its
[docs/](docs/) page.
