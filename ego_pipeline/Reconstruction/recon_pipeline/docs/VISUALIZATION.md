# Visualization policy

Most `recon_pipeline` steps follow the same rules:

1. **Off by default** — pass `--visualize` (or `run_pipeline.py --visualize`).
2. **One artifact per step** — written under `{step_dir}/vis/` for standard visualizations (PNG only for single-frame SAM3D).
3. **Complete summary** — each visualization shows *all* outputs that step produces, combined in that one file.
4. **No viz cruft** — no temporary PNG frame sequences kept on disk.

`sam3d_scale` is the exception: it intentionally keeps several debug PNGs because scale/orientation estimation is usually inspected frame-by-frame rather than as a video.

## Per-step contents

| Step | Output | What you see |
|------|--------|--------------|
| **vipe** | `vis/{video_id}.mp4` | 2×2 grid: RGB, segmentation (if ViPE produced masks), depth colormap, camera trajectory |
| **sam3_hands** | `vis/{video_id}.mp4` | RGB + **both** hand masks (left=green, right=orange) on every frame |
| **sam2_object** | `vis/{video_id}.mp4` | RGB + propagated object mask; click prompts drawn on the label frame |
| **hawor** | `vis/{video_id}.mp4` | RGB + **both** hand MANO meshes (via HaWoR renderer) |
| **sam3d** | `vis/{video_id}_recon.png` | Side-by-side: label-frame RGB+mask \| raw SAM3D mesh |
| **sam3d_scale** | debug PNGs in `sam3d_scale/` | clicked-frame depth/mesh scale overlays and 3D scale plot |
| **fp_pose** | `vis/{video_id}.mp4` | RGB + light object mask + **semi-transparent mesh** in FP pose + **3D bbox** + **RGB axes** |
| **fuse** | `vis/{video_id}.mp4` | RGB + light object mask + **semi-transparent object mesh** + **MANO hand skeletons** (depth-ordered with object) |

## Final reconstruction visualization

Batch runs do not create visualization by default. After a final package exists under
`data/reconstruction/{dataset}/{video_id}/`, render an inspection video with:

```bash
conda activate hawor
python3 recon_pipeline/visualize_final_reconstruction.py \
  --dataset hoi4d \
  --video-id ZY20210800001__H1__C11__N07__S185__s02__T2
```

The output is written to
`data/reconstruction/{dataset}/{video_id}/vis/{video_id}_final_reconstruction.mp4`.
By default it writes a fast raw-video overlay with projected point clouds for
the MANO hands and every object mesh in the final package, plus the hand
skeleton, per-object axes, and hand axes. Axis colors are X=red, Y=green, Z=blue.
World-frame axes are hidden by default; pass `--draw-world-axis` when you need
them.

Render a whole list:

```bash
conda activate hawor
python3 recon_pipeline/visualize_final_reconstruction.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt
```

This is the command to use after a forced batch rerun if you want to refresh
the final inspection videos for every video in the list.

Use `--layout side-by-side` for the heavier diagnostic view with an additional
reconstruction-only panel. `--hand-source final` is the default and verifies the
canonical final fused MANO globals; `--hand-source raw-hawor` is a debugging
fallback that projects the pre-gravity HaWoR globals with the ViPE camera.

Surface rendering options:

| Flag | Meaning |
|------|---------|
| `--surface-mode points` | Default; render object and MANO as projected point clouds |
| `--surface-mode mesh` | Render triangular surface overlays |
| `--object-point-stride 8` | Default object vertex subsampling for point-cloud mode |
| `--hand-point-stride 1` | Default MANO vertex subsampling for point-cloud mode |
| `--point-radius 2` | Point radius in pixels |
| `--accurate-raster` | Slower triangle depth sorting for mesh mode |

Use `--no-hands` to render object/world axes only when the HaWoR/MANO runtime is
not available.

## SAM2 mask diagnosis

To inspect SAM2 object propagation without marking the main pipeline step
complete, run `sam2_object/run_sequence.py --diagnostic-only --visualize`. It
copies the saved prompt into `sam2_object_diagnostic/`, writes propagated masks
there, and writes one overlay MP4 under `sam2_object_diagnostic/vis/`. By
default, the diagnostic MP4 shows all labeled objects together with distinct
colors.

Use `--visualize-object-id object_1` only when you want to inspect one object
alone.

## Interim retention (inference)

Same minimum-necessary rule for non-viz artifacts — see [DATA_LAYOUT.md](DATA_LAYOUT.md).

Downstream steps always read **raw dataset RGB** (`data/raw/.../align_rgb/image.mp4`), not copies under interim dirs. Fuse visualizations are written to `data/interim/{dataset}/{video_id}/fuse/vis/{video_id}.mp4`; final reconstruction outputs stay under `data/reconstruction/`.
