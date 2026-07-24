# Data layout

All paths are relative to the repository root.

## Interim (`data/interim/`)

Interim directories are working storage. Single-step scripts keep their normal outputs for debugging, but the batch queue runner deletes `data/interim/{dataset}/{video_id}/` by default after the final reconstruction validates. As soon as each video label is saved, the batch runner copies the lightweight object prompt to `data/object_labels/{dataset}/{video_id}/sam2_object/label_prompt.json`, so later `--skip-label` runs can reuse the click labels without keeping masks, depth, logs, or other heavy step outputs. Use `run_batch_queue.py --keep-interim` when you need to inspect per-step artifacts after a successful batch video.

```text
data/interim/{dataset}/{video_id}/
├── vipe/
│   ├── vipe_complete.json
│   ├── pose/{video_id}.npz
│   ├── depth/{video_id}.zip
│   ├── intrinsics/{video_id}.npz
│   └── vipe/{video_id}_info.pkl
│   # optional (testing): vis/{video_id}.mp4
│   # discarded after run: rgb/, mask/, .input_links/
├── sam3_hands/
│   ├── sam3_hands_complete.json
│   ├── {video_id}/video_segmentation/masks/...
│   # optional: vis/{video_id}.mp4  (L+R hands)
├── sam2_object/
│   ├── label_prompt.json
│   ├── sam2_object_complete.json
│   ├── video_segmentation/masks/frame_XXXXXX_masks/object_0.png
│   ├── video_segmentation/masks/frame_XXXXXX_masks/object_1.png
│   # optional: vis/{video_id}.mp4
├── hawor/
│   ├── hawor_complete.json
│   └── {video_id}/world_space_res.pth
│   # optional: vis/{video_id}.mp4  (L+R MANO)
├── sam3d/
│   ├── object_mesh_raw.obj
│   └── sam3d_meta.json
│   └── objects/{object_id}/object_mesh_raw.obj
│   # optional: vis/{video_id}_recon.png  (raw mesh preview)
├── sam3d_scale/
│   ├── object_mesh_scaled_stage1.obj
│   ├── object_mesh_scaled_final.obj
│   ├── scale_fpalign_scale_metadata.json
│   ├── objects/{object_id}/object_mesh_scaled_final.obj
│   ├── observed_object_pointcloud_ref.ply
│   ├── debug_stage1_scale_overlay.png
│   ├── debug_foundationpose_overlay.png
│   ├── debug_final_scale_overlay.png
│   └── debug_scale_3d.png
├── fp_pose/
│   ├── ob_in_cam/XXXXXX.txt
│   ├── fp_pose_meta.json
│   ├── fp_pose_complete.json
│   └── objects/{object_id}/
│       ├── ob_in_cam/XXXXXX.txt
│   # optional: vis/{video_id}.mp4
└── fuse/
    └── fuse_complete.json
    # optional: vis/{video_id}.mp4
```

## Final (`data/reconstruction/`)

```text
data/reconstruction/{dataset}/{video_id}/
├── world_fused.npz       # c2w, K, object poses, hand MANO params, mesh filename
├── object_mesh_scaled_final.obj
├── objects/{object_id}/object_mesh_scaled_final.obj
├── masks/
│   ├── masks_manifest.json
│   ├── objects/
│   │   ├── label_prompt.json
│   │   ├── sam2_object_complete.json
│   │   └── frames/frame_XXXXXX_masks/{object_id}.png
│   └── hands/
│       ├── hand_masks_complete.json
│       ├── sam3_hands_complete.json
│       └── frames/frame_XXXXXX_masks/{left_hand_0,right_hand_0}.png
├── world_summary.json
├── reconstruction_complete.json
└── vis/{video_id}_final_reconstruction.mp4   # optional, standalone final visualizer
```

## Label cache (`data/object_labels/`)

```text
data/object_labels/{dataset}/{video_id}/sam2_object/
└── label_prompt.json
```

This cache is intentionally small and durable. It stores the SAM2 object prompt
frame, click points, and foreground/background labels for every labeled object:

```json
{
  "schema_version": "sam2_object_prompt_v2",
  "objects": [
    {
      "object_id": "object_0",
      "frame_idx": 42,
      "points": [[320.5, 240.0]],
      "labels": [1],
      "locked": true
    }
  ]
}
```

The batch runner restores this file back into
`data/interim/{dataset}/{video_id}/sam2_object/label_prompt.json` when
`--skip-label` is used and the interim prompt is missing.

The primary final coordinate frame is `gravity_z_up_world`. It is the ViPE world
transformed by `T_gravity_z_up_from_vipe_world` so +Z points upward, opposite
physical gravity, and then yawed around +Z so the middle-frame camera focal axis
(camera +Z), projected into the gravity-aligned XY plane, becomes final world
+X. With this right-handed frame, +Y is `+Z x +X`, so it points camera-left for
that middle-frame reference. GeoCalib's raw `Gravity.vec3d` is treated as an up
direction because it projects to the image up-field; `gravity_direction_vipe_world`
stores the opposite, physical down direction. The main camera, object, and hand
outputs are all represented in this same frame:

- `c2w`: camera-to-world in `gravity_z_up_world`
- `object_ob_in_world`: object-to-world in `gravity_z_up_world`
- `hand_trans`: MANO global translation in `gravity_z_up_world`
- `hand_rot`: MANO global/root axis-angle in `gravity_z_up_world`

Original/debug coordinate-frame fields are kept with explicit names, for
example `c2w_vipe_world`, `object_ob_in_vipe_world`,
`hand_trans_hawor_world`, and `hand_rot_hawor_world`.

`world_fused.npz` keys:

| Key | Description |
|-----|-------------|
| `schema_version` | final package schema, currently `recon_world_v4` |
| `dataset`, `video_id` | source identifiers |
| `coordinate_frame` | `gravity_z_up_world` |
| `source_coordinate_frame` | `vipe_world` |
| `num_frames` | raw video frame count |
| `c2w` | `(T, 4, 4)` gravity-aligned z-up camera-to-world per frame |
| `c2w_vipe_world` | original ViPE-world camera-to-world poses |
| `K` | `(3, 3)` intrinsics |
| `object_ob_in_cam` | `(N, 4, 4)` object pose in camera frame |
| `object_ob_in_world` | `(N, 4, 4)` object pose in gravity-aligned z-up world frame |
| `object_ob_in_vipe_world` | original ViPE-world object poses |
| `object_frame_indices` | frame indices for object poses |
| `num_objects`, `object_ids` | number of labeled/reconstructed objects and their ids |
| `object_mesh_filenames` | per-object mesh paths relative to the final directory |
| `object_ob_in_cam_all` | `(O, N_max, 4, 4)` padded per-object object-to-camera poses |
| `object_ob_in_world_all` | `(O, N_max, 4, 4)` padded per-object poses in `gravity_z_up_world` |
| `object_ob_in_vipe_world_all` | `(O, N_max, 4, 4)` padded per-object poses in original ViPE world |
| `object_frame_indices_all`, `object_valid_all` | per-object frame index table and valid mask for the padded arrays |
| `object_visible_by_frame` | `(O, T)` visibility-style table derived from FP pose frames when no FP visibility sidecar exists |
| `object_pose_valid_by_frame` | `(O, T)` frames that have an exported FP pose |
| `T_gravity_z_up_from_vipe_world` | `(4, 4)` transform applied to ViPE world-frame quantities; includes gravity roll/pitch and the middle-frame camera-forward yaw |
| `gravity_direction_vipe_world`, `up_direction_vipe_world` | Physical down/up directions in ViPE world; validation requires `T_gravity_z_up_from_vipe_world @ gravity = -Z` and `... @ up = +Z` |
| `world_xy_alignment_mode`, `world_xy_alignment_frame_index` | XY yaw rule, currently `middle_camera_forward_projected_xy`, and the source video frame used for the yaw reference |
| `world_x_direction_vipe_world`, `world_y_direction_vipe_world`, `world_z_direction_vipe_world` | Final world basis directions expressed in the original ViPE world |
| `world_xy_reference_camera_forward_*`, `world_xy_alignment_yaw_angle_rad` | Middle-frame focal-axis diagnostics used to derive the final yaw |
| `gravity_cam_samples`, `up_cam_samples` | Sampled physical down/up directions in camera coordinates |
| `mesh_filename`, `mesh_path` | copied scaled object mesh filename in this directory |
| `hand_trans`, `hand_rot`, `hand_pose`, `hand_betas`, `hand_valid` | MANO params `(2, T_hawor, …)` in `gravity_z_up_world` when hawor step ran |
| `hand_trans_hawor_world`, `hand_rot_hawor_world` | HaWoR MANO globals before gravity alignment; HaWoR is run with time-aligned ViPE camera poses and ViPE inverse-depth/disparity in its SLAM artifact |
| `hand_coordinate_frame` | `gravity_z_up_world` for current final MANO globals |
| `hand_source_coordinate_frame` | `hawor_world_vipe_camera` for current outputs |
| `hand_camera_timeline` | optional marker for repaired or versioned HaWoR camera-timeline outputs |

The raw video frame count and the HaWoR hand-frame count may differ. For current
HOI4D test outputs the raw video has 300 frames while HaWoR MANO has 600 frames.
When overlaying hands on video frames, sample the hand arrays by normalized time
rather than assuming a one-to-one frame index.

`reconstruction_complete.json` is the final resume marker used by the batch queue. It is written only after `world_fused.npz`, `world_summary.json`, and `object_mesh_scaled_final.obj` validate.

`masks/` is copied by `fuse` from the interim SAM2 object masks and SAM3 hand masks before interim cleanup. The manifest records relative file templates for per-frame object masks and left/right hand masks.

For compatibility, the single-object fields and top-level `object_mesh_scaled_final.obj` always refer to the primary object (`object_0`). Use the `*_all` arrays and `objects/{object_id}/` meshes when a video has more than one labeled object.

Per-step visualization MP4s live under `data/interim/.../{step}/vis/` only when
`--visualize` is used. `sam3d` writes a single PNG preview instead of an MP4,
and `sam3d_scale` writes debug PNGs for scale/orientation inspection. These
visualization/debug files are working artifacts and are deleted by default after
a successful batch-queue reconstruction unless `--keep-interim` is used.

The standalone final visualizer writes durable inspection videos under
`data/reconstruction/{dataset}/{video_id}/vis/`. It is run separately with
`recon_pipeline/visualize_final_reconstruction.py` and is not created by default
during batch reconstruction.

The object-mask step is `sam2_object` because it uses SAM2 for manual point-prompt preview and video mask propagation. `sam3_hands` remains SAM3-backed, and `sam3d`/`sam3d_scale` refer to the separate SAM3D object-mesh stages.

## Dataset name

Pass `--dataset hoi4d` (or future dataset ids). Video id for HOI4D uses the existing `__`-separated sequence name, e.g. `ZY20210800001__H1__C11__N07__S185__s02__T2`.

## Logs

Parallel runs write per-video logs to `data/interim/{dataset}/{video_id}/{step}/.logs/{video_id}.log` when not using `--loud`.
