# BODex on cuRobo v2 (`bodex_curobo_v2`)

The core grasp-synthesis algorithm: an independent port of BODex's bilevel
grasp optimization onto official NVLabs cuRobo v2. It never imports anything
from BODex itself. This is the "pure" pipeline -- object mesh in, ranked
Sharpa Wave grasp poses out, no human demonstration involved.

```text
sequence directory (object mesh) -> solve_sharpa_bodex() -> grasp_*.json / summary.json
                                                           -> Isaac Sim visualization (optional)
```

## Algorithm overview

- **Bilevel optimization**: an outer BODex-faithful momentum optimizer over
  the 29-D action (7-D root pose + 22 joint angles) wraps an inner
  force-closure QP (grasp-energy) evaluated at the current contact
  configuration. Success is strict: force closure within
  `--grasp-threshold` plus contact distance within `--distance-threshold`.
- **Staged contact cost**: optimization proceeds through stages (free-space
  shaping -> contact acquisition -> refinement), combining mesh-mesh
  (convex-hull GJK/EPA via `coal`) and sphere-mesh (warp SDF) distance
  queries. See the module docstrings in
  `src/ocir/grasp_synthesis/bodex_curobo_v2/` for the exact staging.
- **Object meshes are convex-decomposed** with `coacd`
  (normalize -> decompose -> rescale; `_coacd_convex_parts` in
  `contact_world.py`) for contact evaluation. The decomposition is **cached
  on disk next to the object mesh** (`<stem>_coacd_parts.npz`, keyed by mesh
  content hash + CoACD parameters, invalidated automatically when either
  changes) and reused by every pipeline touching the same object -- grasp
  synthesis constructs two contact worlds per run and the trajectory
  generator a third, so a sequence's object is decomposed once ever instead
  of three times per run. Cache write failures (read-only object folder)
  degrade to a warning + in-memory decomposition. Hand links use a single
  whole-mesh convex hull each (see Limitations).
- **Seeding**: root poses sampled around the object from surface
  points/bounding geometry, joints from a canonical open posture
  (`seed_generator.py`).

Key modules (all under `src/ocir/grasp_synthesis/bodex_curobo_v2/`, frozen --
new features go in sibling packages that import it, never modify it):

| Module | Role |
| --- | --- |
| `contact_world.py` | `SingleObjectContactWorld`: coacd parts, warp SDF, GJK/EPA contact queries |
| `grasp_cost.py` | staged contact cost + force-closure QP (`ContactBuffer`) |
| `solver.py` | `solve_sharpa_bodex()` orchestration, yaml loading, ranking |
| `seed_generator.py` | root/joint seed sampling |
| `rollout.py` | cuRobo v2 optimizer-core wiring |

## Running it

```bash
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_synthesis/synthesize_sharpa_bodex_curobo_v2.py \
  --sequence-dir /path/to/sequences/<sequence_id> \
  --out-dir /path/to/output_root \
  --seeds 20 --top-k 8 --opt-iters 500
```

Batch mode: replace `--sequence-dir` with `--sequences-root <root>` (every
immediate subdirectory is one sequence).

### CLI flags

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
| `--isaac-mode {server,standalone}` (default `server`) | See [Isaac Sim infrastructure](isaac_sim.md). |
| `--control-host`, `--control-port` (default `127.0.0.1:8765`) | Persistent server address. |
| `--isaac-width`, `--isaac-height`, `--isaac-tabletop-z`, `--isaac-hold-open[-seconds]`, `--isaac-show-object-points` | Rendering options passed through to `visualize_grasp.py`. |
| `--check-only` | Only validate the backend; no sequence/out-dir needed. |
| `--strict-success-exit-code` | Exit 1 if any sequence had zero strictly-successful seeds. |

Exit code is nonzero if any sequence's solver crashed or its visualization
failed (or, with `--strict-success-exit-code`, if any sequence had zero
successful seeds); otherwise 0, even if some individual sequences report
`"ok": false` in their summary.

## Outputs

Given `--out-dir <root>`, each sequence writes to `<root>/<sequence_name>/`:

- `grasp_<rank>.json` (rank `000`, `001`, ...) -- one per top-`k` strictly
  successful seed, best score first. Each record includes the full 29-D
  action (7-D root pose + joint angles), `score`, `success`,
  `grasp_error_max`, `dist_error`, `object_mesh`, `sequence_dir`,
  `sequence_id`/`object_name` (from the sequence's metadata), and
  `metric_summary`.
- `failed_grasp_<rank>.json` -- written instead, ranked by score, if zero
  seeds strictly succeeded. **Downstream consumers (grasp_traj) accept these
  too, but their quality is unverified by construction** -- fingers/palm may
  penetrate the object (see [grasp_traj.md](grasp_traj.md)).
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

## Hand asset configuration

The hand is described by a single yaml config, default
`assets/robots/hands/sharpa_wave/sharpa_wave_right.yml`, loaded by
`src/ocir/grasp_synthesis/assets.py`. Fields you're likely to touch:

- `urdf_path`, `usd_path` -- hand geometry, relative to the config file's directory.
- `base_link`, `ee_link` -- kinematic root/end-effector link names.
- `joint_order` -- the 22 Sharpa Wave joint names, in the order actions are ordered.
- `joint_limits` -- per-joint `[lower, upper]` radians.
- `contact_points.config_path`, `collision_spheres.path` -- contact-point and
  analytic-sphere-collision configs
  (`assets/robots/hands/sharpa_wave/contact_points/`, `.../collision/curobo/`).
- `bodex.<key>` -- paths to the BODex-side configs (`grasp_synthesis_config`,
  `robot_config`, `hand_pose_transfer`), under
  `assets/robots/hands/sharpa_wave/grasp_synthesis/bodex/`.

Pass `--asset-config <path>` to any pipeline CLI to point at a different
hand config; `SharpaWaveAsset` validates that every referenced path exists
at load time.

## Known limitations

- **Robot links use a single whole-mesh convex hull each**, while the object
  mesh is convex-decomposed into multiple parts via `coacd`. This
  under-approximates contact for concave hand-link regions. See
  `contact_world.py`'s docstring.
- `coacd` must be imported after `torch` (native OpenMP runtime conflict --
  `contact_world.py` orders its imports accordingly; do the same anywhere
  else you import `coacd`).
- CUDA required: `solve_sharpa_bodex` raises if
  `torch.cuda.is_available()` is false.
