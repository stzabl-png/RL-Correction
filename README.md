# OCIR Grasp Synthesis (cuRobo v2)

Sharpa Wave hand grasp synthesis on official NVLabs cuRobo v2, with Isaac Sim
visualization support. Split out from the main OCIR repository, which also
carries an original-BODex reference implementation and unrelated DexYCB
replay/retargeting work; this workspace keeps only the cuRobo v2 port and
what it needs to run and visualize.

## Repository Layout

```text
src/ocir/
  grasp_synthesis/
    assets.py, object_surface.py     # asset-path resolution, surface-artifact loading
    synthesize_sharpa_bodex_curobo_v2.py   # CLI entry point
    bodex_curobo_v2/                  # the grasp-synthesis algorithm itself:
                                       # grasp-energy QP, staged contact cost, BODex-faithful
                                       # optimizer, seed generator, cuRobo v2 rollout wiring
  isaac/
    visualize_grasp.py                # renders a synthesized grasp in Isaac Sim
    replay_dexycb.py, sim_cli.py      # shared geometry/CLI helpers visualize_grasp.py depends on
  sim/
    control_client.py, isaac_server.py, start_isaacsim_server.py   # persistent Isaac control server
  dexycb/
    prepare_dexycb_subset.py, mano_model.py   # DexYCB subset prep (only needed to regenerate inputs)
  affordance/
    extract_dexycb_object_affordance.py       # produces the surface_artifact .npz grasp synthesis reads

scripts/            # thin conda-env wrappers around the above
assets/robots/hands/sharpa_wave/   # hand URDF, meshes, USD, collision spheres, manip configs
third_party/curobo  # official NVLabs cuRobo v2 (git submodule)
```

`bodex_curobo_v2` never imports anything from BODex — it's an independent
port of the BODex grasp-synthesis algorithm onto official cuRobo v2. See its
module docstrings for the algorithm details (mesh processing, staged
optimization, force-closure QP, seed generation).

## Environment

Everything runs in one conda env (`env_isaacsim` on the original OCIR host).
Set `OCIR_DATA_ROOT` to wherever DexYCB raw/processed data and generated
outputs live (shared external data root, not part of this repo):

```text
${OCIR_DATA_ROOT}/
  raw_data/dex_ycb/, raw_data/mano_models/
  processed_data/dex_ycb/manifests/, processed_data/dex_ycb/selected/
  testing/grasp_synthesis/sharpa_wave_bodex_curobo_v2/
```

## Quick Start

Start the persistent Isaac Sim control server:

```bash
scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py --width 1280 --height 720
```

Run grasp synthesis (defaults to the `selected_5_sequences.json` manifest,
i.e. all selected objects, and submits results to the running server for
visualization):

```bash
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_synthesis/synthesize_sharpa_bodex_curobo_v2.py \
  --seeds 20 --top-k 8 --opt-iters 500
```

Add `--check-only` to validate the backend (official-cuRobo-v2 import
isolation, all `bodex_curobo_v2` modules) without running synthesis.

## Known Limitation

Mesh-mode contact currently uses a single convex hull per link/object
(via the standalone `coal` package) rather than a full multi-piece convex
decomposition — an under-approximation for concave regions. See
`src/ocir/grasp_synthesis/bodex_curobo_v2/contact_world.py`'s module
docstring.
