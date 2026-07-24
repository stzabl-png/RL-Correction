# Pipeline steps

## Overview

```text
RGB video
  ├─ vipe           camera intrinsics, c2w poses, depth maps
  ├─ sam3_hands     text-prompt left/right hand masks
  └─ sam2_object    manual point labels → SAM2-propagated object masks

hawor             ViPE camera + ViPE depth handoff, SAM3 hand-mask filter, no infiller → world MANO
sam3d             raw mesh per labeled object from object mask + ViPE depth
sam3d_scale       metric scale/orientation per object via clicked-frame depth + single-frame FP
fp_pose           per-object FoundationPose register + FP++ track (centroid + 6D KF)
fuse              T_obj_world = c2w @ T_obj_cam; gravity-align camera/objects/MANO; export final NPZ + copied meshes
```

## Step dependencies

| Step | Requires |
|------|----------|
| vipe | RGB video |
| sam3_hands | RGB video |
| label | RGB video |
| sam2_object | label_prompt.json |
| hawor | vipe, sam3_hands |
| sam3d | sam2_object, vipe |
| sam3d_scale | sam3d, sam2_object, vipe |
| fp_pose | vipe, sam2_object, sam3d_scale |
| fuse | vipe, hawor, fp_pose, sam3d_scale |

After `vipe`, `sam3_hands`, and `sam2_object` are complete, `hawor` and `sam3d` can run independently. `sam3d_scale` waits for `sam3d`; `fp_pose` waits for `sam3d_scale`; `fuse` waits for `vipe`, `hawor`, `fp_pose`, and `sam3d_scale`.

`sam2_object` is named for the backend it uses: SAM2 video predictor (`third_party/sam2`) handles both point-prompt preview and mask propagation. The prompt schema supports multiple objects; each object has its own prompt frame, points, propagated mask files, mesh, and pose track. SAM3 is still used for `sam3_hands`, and SAM3D is a separate object-mesh reconstruction model.

For batch runs, [`run_batch_queue.py`](../run_batch_queue.py) is the recommended orchestrator. It starts CPU-based HTTP labeling first, then enqueues each video for reconstruction as soon as its `sam2_object/label_prompt.json` is saved. The runner uses memory-aware GPU worker slots: by default it uses one worker slot per GPU id, and `--workers-per-gpu 2` allows safe same-GPU overlap when the per-step memory estimates fit the budget. Visualization is off unless `--visualize` is passed.

## Required data and assets

Current video discovery is implemented for HOI4D. The default root is:

```text
data/raw/HOI4D/
```

Each selected HOI4D sequence must contain:

```text
data/raw/HOI4D/HOI4D_release/<sequence>/align_rgb/image.mp4
```

The runner accepts HOI4D artifact ids such as
`ZY20210800001__H1__C11__N07__S185__s02__T2`, slash-separated HOI4D relative
paths, or direct MP4 paths. Use `--dataset-root` when the raw HOI4D root is
elsewhere.

Model/runtime assets are step-specific:

| Step | Required assets |
|---|---|
| vipe | ViPE submodule and uv environment |
| sam3_hands | SAM3 environment, model access/checkpoints |
| sam2_object | SAM2 installed in the `sam3` env, `third_party/sam2/checkpoints/sam2.1_hiera_large.pt` by default |
| hawor | HaWoR environment and MANO files |
| sam3d | SAM3D environment and checkpoints |
| sam3d_scale / fp_pose | FoundationPose environment and weights |
| fuse | Existing interim outputs from all dependency steps |

See [SETUP.md](SETUP.md) for installation details.

## Coordinate frames

- **ViPE world**: monocular SLAM frame from ViPE `pose/*.npz`; ViPE also writes `gravity/{video_id}.npz` from GeoCalib gravity estimates.
- **Gravity z-up world**: final export frame. Fusion rotates ViPE-world quantities so +Z is physical up, then applies a yaw around +Z so the middle-frame camera focal axis projected into the horizontal plane becomes world +X. GeoCalib's raw `Gravity.vec3d` is interpreted as an up direction because it projects to the image up-field; the pipeline stores physical gravity as the opposite direction and validates that it maps to -Z.
- **HaWoR hands**: HaWoR is run with ViPE camera poses injected as its SLAM trajectory and ViPE depth converted to compact inverse-depth/disparity in the HaWoR SLAM artifact. HaWoR extracts the source RGB at 30 fps, so both the injected ViPE trajectory and depth timeline are sampled by source-video time before HaWoR converts camera-space MANO to world-space MANO. Fusion then gravity-aligns MANO global translation/root orientation into the z-up final frame without the legacy HaWoR visualization flip.
- **FP output**: `objects/{object_id}/ob_in_cam/*.txt` = object→camera per frame (does not consume ViPE poses directly). The top-level `ob_in_cam/*.txt` alias is kept for `object_0`.
- **Fusion**: computes ViPE-world object-to-world as `T_obj_world(t) = T_c2w(t) @ T_obj_cam(t)` for every labeled object, then applies `T_gravity_z_up_from_vipe_world` to camera/object/hand globals before writing final outputs. The transform includes both gravity +Z alignment and the middle-frame camera-forward XY yaw.

## FP++ design

`fp_pose` uses the Python NVlabs FoundationPose backend with the original
register-once + FP++ tracking path by default:

1. `--pose-mode track` is the default. For each labeled object, it registers that object's prompt frame with FoundationPose on the object's scaled mesh from `sam3d_scale`.
2. It then tracks forward to the end of the video and backward to frame 0, using per-frame SAM2 mask centroids plus a 6D Kalman filter to initialize FoundationPose `track_one(...)`. The masks are not used as a hard visibility gate and generated poses are not rejected by 2D mask consistency checks.
3. Outputs are written as per-frame object-to-camera transforms for every frame under `fp_pose/objects/{object_id}/ob_in_cam/XXXXXX.txt`; `object_0` is also copied to the legacy top-level `fp_pose/ob_in_cam/` path.

The slower diagnostic mode `--pose-mode register-each` is still available on
`fp_pose/run_sequence.py`. It runs FoundationPose `register(...)` on every frame
using that frame's SAM2 propagated object mask, ViPE depth, ViPE intrinsics, and
the scaled mesh.

Isaac ROS TensorRT from branch `STEP_6_pose` is **not** wired into `recon_pipeline`; see `fp_pose/ISAAC_ROS.md` for historical notes only.

## Entry points

| Step | Single | Parallel |
|------|--------|----------|
| vipe | `vipe/run_sequence.py` | `vipe/infer_parallel.py` |
| sam3_hands | `sam3_hands/run_sequence.py` | `sam3_hands/infer_parallel.py` |
| label | `sam2_object/label_object.py` | manual / per-video |
| sam2_object | `sam2_object/run_sequence.py` | `sam2_object/infer_parallel.py` |
| hawor | `hawor/run_sequence.py` | `hawor/infer_parallel.py` |
| sam3d | `sam3d/run_sequence.py` | `sam3d/infer_parallel.py` |
| sam3d_scale | `sam3d_scale/run_sequence.py` | — |
| fp_pose | `fp_pose/run_sequence.py` | `fp_pose/infer_parallel.py` |
| fuse | `fuse/run_sequence.py` | `fuse/infer_parallel.py` |
| all | `run_pipeline.py` | run parallel scripts per step |

## Completion markers

Each interim step writes `{step}_complete.json` with `"status": "complete"`. Parallel runners skip videos when this file exists (`--resume`, default on).

Fusion writes `data/reconstruction/{dataset}/{video_id}/world_fused.npz`, copies each scaled object mesh under `objects/{object_id}/`, keeps the top-level mesh as the `object_0` compatibility alias, writes `world_summary.json`, and then writes `reconstruction_complete.json` after final validation. Final `c2w`, `object_ob_in_world`, `object_ob_in_world_all`, and MANO globals are in `gravity_z_up_world`; original ViPE-world arrays are retained under explicit `*_vipe_world` keys. Final validation checks schema, single-object compatibility fields, multi-object array shapes/meshes, gravity alignment fields, middle-frame camera-forward XY alignment, and that sampled valid MANO hand roots are in front of the final cameras. The batch queue uses this final marker for resume and deletes per-video interim outputs by default after successful validation. Pass `--visualize` on the fuse step to also write `data/interim/{dataset}/{video_id}/fuse/vis/{video_id}.mp4`, and use `run_batch_queue.py --keep-interim` if you want that interim visualization to survive a successful batch run.

As soon as a video's manual label is saved, the batch queue preserves the small
object prompt under:

```text
data/object_labels/{dataset}/{video_id}/sam2_object/label_prompt.json
```

`run_batch_queue.py --skip-label` restores this prompt back into interim storage
when needed, so labels can be reused without keeping heavy masks, depth, or
per-step logs.
