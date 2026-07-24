# Affordance-seeded grasp synthesis

Grasp synthesis (the pure `bodex_curobo_v2` pipeline) samples its seeds over the
**whole** object surface, so it can converge on awkward grasps (e.g. hooking a
cup rim) that lack force closure. This adds an **expected grasp area** step: a
vendored affordance model predicts where a human would grab the object, and
grasp seeds are drawn only from that region.

```text
object mesh --[affordance model, sonata env]--> affordance.npz (per-point heatmap)
                                                     |
                          high-affordance points ----+--> BODex seed pool (region-restricted)
                                                                |
                                          bodex_curobo_v2 optimize --> ranked grasp records
```

## One command

```bash
scripts/run_grasp_synthesis_conda.sh \
  scripts/grasp_synthesis/synthesize_affordance_seeded.py \
  --sequence-dir /path/to/sequences/<id> \
  --out-dir /path/to/out/<id> \
  --affordance-threshold 0.5 --seeds 40 --top-k 8 --opt-iters 500
```

It (1) predicts the object's expected grasp area if not cached
(`<sequence-dir>/affordance_pred/affordance.npz`), (2) restricts the seed pool to
points with affordance `> --affordance-threshold`, (3) runs the frozen BODex
optimizer, and (4) prints the record with the lowest `grasp_error_max` (the
default `rank_score` can bury the best force-closure seed) and writes
`affordance_seed_info.json` next to the grasp records.

## Full pipeline: reconstructed take → grasp → lift video (one command)

For a Reconstruct_and_Retarget-style take (reconstruction + retarget), this produces the complete
approach→grasp→lift physics video in one command — object placed at its stable
pose on the table, the hand starting from the **reconstructed** initial pose,
cuRobo driving it to the affordance-seeded grasp, then a +Z lift:

```bash
scripts/run_grasp_synthesis_conda.sh scripts/grasp_traj/reconstructed_grasp_video.py \
  --recon-dir    <ReconstructOutput>/.../<take> \
  --retarget-dir <RetargetOutput>/.../<take> \
  --sequence-id  <name>
```

It chains: decimate mesh → affordance-seeded synthesis (best force-closure
record) → Stage-A trajectory → auto identity manifest → Stage-B PhysX video
(`<work>/grasp_traj/<name>/isaac_sim/video.mp4`). `ref_qpos.npz` is optional —
when absent the reconstructed initial hand pose is derived from
`replay_world.npz`'s MANO joints (SharpaWave base-frame convention), so the
hand-object relative start always comes from the reconstruction. Flags:
`--carry-lift-height` (0.10), `--seeds`, `--opt-iters`, `--reuse-synthesis`.

## Model + checkpoint (vendored)

- Code: `src/ocir/affordance/` (`affordance_model.py`, `heads.py`,
  `sonata_backbone.py`, `normalize.py`, `predict.py`) — vendored, no external
  repo needed.
- Checkpoint: `assets/affordance/model.pt` (**git-lfs**; `git lfs pull` to fetch).
  Swap in a better model by replacing this file — see
  [`assets/affordance/README.md`](../assets/affordance/README.md).

## Environment

The Sonata / spconv-cu128 / torch_scatter stack is heavy and is **not** in the
grasp-synthesis / Isaac env, so prediction runs as a subprocess in a dedicated
conda env. Set it up once (see [`envs/affordance-requirements.txt`](../envs/affordance-requirements.txt))
and point OCIR at it:

| env var | default | meaning |
| --- | --- | --- |
| `OCIR_AFFORDANCE_ENV` | `deximit` | conda env that has the sonata stack |
| `OCIR_AFFORDANCE_CKPT` | `assets/affordance/model.pt` | checkpoint path |
| `OCIR_AFFORDANCE_EXTRA_PYTHONPATH` | *(unset)* | extra paths, only if `sonata` is importable via a checkout rather than pip-installed |

The subprocess runs with a **clean** `PYTHONPATH` (only OCIR `src` + the extra),
so the Isaac env's bundled numpy/torch cannot shadow the affordance env's own.

`sonata` fetches its pretrained PTv3 encoder from HuggingFace (`facebook/sonata`)
on first use and caches it (`~/.cache/sonata`).

## Notes

- `affordance.npz` is cached; pass `--force-affordance` to re-predict.
- If the threshold selects too few points, the loader falls back to the top 25%.
- The affordance region's points are projected onto the mesh and given outward
  face normals, so seeds approach from outside the surface (same convention as
  the whole-surface seeder).
