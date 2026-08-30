# EgoDex screw/unscrew bottle cap clip 1

This is the repository-normalized snapshot of
`egodex_auto/screw_unscrew_bottle_cap/1`: a 104-frame first-person take in
which the left hand holds a bottle body and the right hand unscrews the cap and
places it on the table.

The snapshot preserves the final Step2/Step3 reconstruction, quality evidence,
contact priors, and Step4 retargeting inputs. It is intentionally stored next
to the existing frozen reconstruction datasets rather than under
`TrainingData/`, which is a generated, ignored runtime staging area.

## Layout

| Path | Meaning |
|---|---|
| `meta.json` | Repository-facing identity, object mapping, timing, quality summary, and integration status. |
| `reconstruction/world_fused.npz` | Source reconstruction: calibrated camera poses/intrinsics, both hands, and both objects' 6-DoF tracks in metre-scale `gravity_z_up_world`. |
| `reconstruction/object_valid_measured.npz` | Per-object/per-frame confidence support: trustworthy mask, occlusion, and position confidence. |
| `reconstruction/objects/object_0/` | Final bottle-body CAD mesh. |
| `reconstruction/objects/object_1/` | Final bottle-cap CAD mesh. |
| `reconstruction/object_mesh_scaled_final.obj` | Upstream primary-object compatibility alias; byte-identical to the object-0 bottle body. |
| `reconstruction/masks/` | 104 frames of left/right hand masks and bottle-body/cap masks, plus their manifests. |
| `reconstruction/contact/` | Reconstructed hand-object contact evidence, surface-probe weights, point clouds, heatmaps, and summaries. |
| `reconstruction/poseqa/` | Per-object RTS-smoothed pose tracks and framewise uncertainty/confidence evidence. |
| `reconstruction/*.json` | Completion markers, world summary, contact intervals, confidence summary, and frame-scan record. |
| `retarget/replay_world.npz` | Isaac replay input containing both hands, both object poses, phase labels, and MANO geometry. |
| `retarget/ref_qpos_left.npz` | Left SharpaWave 22-DoF reference joints. |
| `retarget/ref_qpos_right.npz` | Right SharpaWave 22-DoF reference joints. |
| `retarget/ref_qpos.npz` | Repository-loader compatibility alias, byte-identical to the right-hand file because the task-driving cap hand is right. |
| `retarget/object_0.usd`, `object_1.usd` | Self-contained Isaac assets for the bottle body and cap. |
| `provenance/` | Unmodified upstream bundle notes, data lineage, tuning notes, and retrieval-registry excerpt. |
| `THIRD_PARTY_LICENSES/` | License notice for the CAD substitute. |
| `SHA256SUMS` | Hashes for every tracked file in this snapshot. |

## Object and contact mapping

- `object_0` is the bottle body, held by the **left** hand. The accepted
  contact-v2 evidence has opposition `0.45485` and hand-mask agreement
  `0.38046`.
- `object_1` is the bottle cap, manipulated by the **right** hand. Its accepted
  contact-v2 evidence has opposition `0.54211` and hand-mask agreement
  `0.51735`.
- The measured trustworthy-frame counts are 67/104 for the body and 86/104 for
  the cap. Exact intervals and raw values remain in
  `object_valid_measured.npz` and `provenance/README_USAGE.md`.

The masks and contact files are reconstruction evidence and priors; they are
not collision/contact truth from Isaac. Isaac must still test whether a robot
can physically realize the reference.

## Important limitations

1. The cap's position track is strong, but its rotation about the nearly
   symmetric screw axis is not visually observable during the actual twist.
   The upstream analysis measures only about -109 degrees from frames 0-35,
   whereas a real opening is typically one or two full turns. Do not use the
   visual cap quaternion as turn-count ground truth; learn/verify screw progress
   through robot-hand motion and the simulator's task state.
2. The bottle body declares `rotation_free_axes=[2]`; axial self-rotation is an
   intentional free dimension, not a missing value.
3. The upstream lineage file records device-level hand wrists, but explicitly
   warns that its MANO finger-pose field contains placeholders because the
   ARKit-25-to-MANO mapping was not implemented. For robot fingers, use the
   final `ref_qpos_left/right.npz` retarget outputs and validate them visually.
4. The source description says 104 frames at 30 FPS, while the final
   `replay_world.npz` stores `fps=15.0` without reducing the frame count. The
   files are preserved verbatim. Resolve this clock mismatch before any reward,
   velocity, jerk, or episode-duration calculation that depends on time.
5. `confidence_complete.json` is the current two-object confidence authority.
   The aggregate `88/61` line in `UPSTREAM_PROVENANCE.md` is stale lineage text
   from an earlier pass and must not override the current per-object results.

## What was deliberately not imported

- `viewer/`: a standalone replay program plus duplicate SharpaWave assets; this
  repository already owns its Isaac/robot runtime.
- `data/.snapshot_before_rerun/`: superseded intermediate output.
- top-level `data/replay_world.npz` and `data/ref_qpos_*.npz`: redundant older
  copies; `data/retarget/` is identified by the upstream README as final.
- `cad/bottle_*.obj`: byte-identical duplicates of the final meshes under
  `reconstruction/objects/`; the registry excerpt and license were retained.
- `conf_*.mp4`, `framescan.log`, `__pycache__/`: generated review media, logs,
  and Python cache. The confidence video remains recoverable from the original
  source archive whose hash is recorded in `meta.json`.

## Step4 status

This directory is a data snapshot, not a registered training clip yet. The
current generic `static_reconstruction` loader follows one primary object, and
the existing water-bottle task adapter uses a separately validated static scene
and analytic screw model. Registering this take before choosing its two-object
initialization, mass/friction assumptions, screw pitch/travel, and 30-vs-15 FPS
policy would silently train the wrong task.

The next integration step should consume `obj_pose_all` from
`retarget/replay_world.npz`, use object 0 as the held body and object 1 as the
right-hand cap, add an explicit helical task state, and then register a new clip
without overwriting `water_bottle_twist_*`.
