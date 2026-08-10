# Full Hand + Object Reconstruction Pipeline

End-to-end reconstruction from RGB video: **ViPE camera/depth → SAM3 hand masks → SAM2 object masks → HaWoR MANO with ViPE camera/depth → SAM3D mesh → FoundationPose/FP++ object pose → unified world export**.

This package lives under `recon_pipeline/` and keeps the active reconstruction code self-contained, with a small `_legacy/` compatibility helper set for ViPE/SAM3/HaWoR utility functions. It writes dataset-scoped working files under `data/interim/` and durable outputs under `data/reconstruction/`.

## Documentation

| Doc | Description |
|-----|-------------|
| [docs/PIPELINE.md](docs/PIPELINE.md) | Step order, dependencies, coordinate frames |
| [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md) | Interim and final output paths |
| [docs/SETUP.md](docs/SETUP.md) | Conda envs, submodules, FoundationPose |
| [docs/LABELING.md](docs/LABELING.md) | Manual SAM2 object labeling — **fallback only**; 默认走 v17A 自动标注(见 ego_pipeline/bin/auto_label_v17a.py) |
| [docs/BATCH_QUEUE.md](docs/BATCH_QUEUE.md) | Hybrid batch runner, memory-aware GPU scheduling, measured resource peaks |
| [docs/VISUALIZATION.md](docs/VISUALIZATION.md) | Per-step `--visualize` outputs and retention rules |

## Quick start: full batch

The recommended entry point for multiple videos is the hybrid queue runner. It
starts HTTP labeling first, then reconstructs each video as soon as its label is
saved. Use this for a fresh run from raw videos:

```bash
python3 recon_pipeline/run_batch_queue.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt \
  --gpu-ids 0,1 \
  --http-host 0.0.0.0 \
  --http-port 8765
```

`run_batch_queue.py` uses HTTP labeling by default and does not auto-detect GPU count. If `--gpu-ids` is omitted it uses GPU `0` only; pass every GPU you want it to use. The default is one reconstruction worker slot per listed GPU. For already-labeled batches on the tested A6000 machine, use `--workers-per-gpu 2` to enable the memory-aware same-GPU overlap scheduler. Visualization is off by default and successful videos are compacted after final validation unless `--keep-interim` is passed.

Open the labeling page at `http://127.0.0.1:8765/`. From a remote laptop, forward the port first:

```bash
ssh -L 8765:127.0.0.1:8765 user@remote-host
```

Rerun reconstruction for videos that already have labels, without starting the
labeling UI:

```bash
python3 recon_pipeline/run_batch_queue.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt \
  --gpu-ids 0,1 \
  --skip-label \
  --force \
  --workers-per-gpu 2
```

This reuses existing `sam2_object/label_prompt.json` files and reruns completed
steps. If the per-video interim prompt was cleaned by a previous successful
batch run, the runner restores it automatically from
`data/object_labels/{dataset}/{video_id}/sam2_object/label_prompt.json`. This
command is required when you want existing `fp_pose` outputs regenerated with
the current default prompt-frame registration + bidirectional FP++ tracking behavior. The worker overlap
uses conservative per-step GPU memory reservations; see
[docs/BATCH_QUEUE.md](docs/BATCH_QUEUE.md) for the measured scheduler trials.

Relabel videos that already have interim prompts by using `--force-label
--force`. `--force-label` clears the selected videos' interim prompt files
before starting the HTTP UI; `--force` reruns downstream reconstruction with the
new labels:

```bash
python3 recon_pipeline/run_batch_queue.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt \
  --gpu-ids 0,1 \
  --http-host 0.0.0.0 \
  --http-port 8765 \
  --force-label \
  --force
```

## Required data

The current discovery code is implemented for HOI4D. The default dataset root
is:

```text
data/raw/HOI4D/
```

At minimum, every selected sequence must contain the RGB video:

```text
data/raw/HOI4D/HOI4D_release/<sequence>/align_rgb/image.mp4
```

For HOI4D, `--video-list` entries may be any of:

```text
ZY20210800001__H1__C11__N07__S185__s02__T2
ZY20210800001/H1/C11/N07/S185/s02/T2
/absolute/or/repo/relative/path/to/image.mp4
```

Use `--dataset-root /path/to/HOI4D` when the raw data root is not
`data/raw/HOI4D`. Keep raw data under `data/raw/`; cleanup only targets
pipeline working outputs and success logs.

## Labeling only

> ⚠ **人工标注是 fallback，默认路线是全自动**：`reconstruct.sh` 会在标注缺失时自动调
> `ego_pipeline/bin/auto_label_v17a.py`（v17A 发现实例选帧 → 取 mask 内切极点写
> label_prompt.json → 本管线 SAM2 传播全片，无需任何人工点选）。独立工具版是
> `tools/v17a_to_label_prompt.py`。只有自动失败或需要人工裁决时才用下面的交互工具。

The object-labeling backend is SAM2, installed in the integrated `sam3` environment. SAM2 preview runs after each click/undo. CPU preview is the default so labeling does not reserve GPU memory while reconstruction workers run. The labeling UI supports multiple object slots (`1`-`9`, `o` for new object, `x`/`Delete` to delete the active object); each object can be prompted on a different frame and is reconstructed downstream.

HTTP browser UI for a video list:

```bash
conda activate sam3
python3 recon_pipeline/sam2_object/label_object.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt \
  --label-mode http \
  --http-host 0.0.0.0 \
  --http-port 8765
```

HTTP browser UI with GPU SAM2 preview:

```bash
conda activate sam3
python3 recon_pipeline/sam2_object/label_object.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt \
  --label-mode http \
  --http-host 0.0.0.0 \
  --http-port 8765 \
  --preview-device cuda \
  --gpu 0
```

Local browser UI with GPU SAM2 preview:

```bash
conda activate sam3
python3 recon_pipeline/sam2_object/label_object.py \
  --dataset hoi4d \
  --video-id ZY20210800001__H1__C11__N07__S185__s02__T2 \
  --label-mode headed \
  --preview-device cuda \
  --gpu 0
```

See [docs/LABELING.md](docs/LABELING.md) for controls and prompt file format.

## Kailang 的静态物体重建扩展

Step3 v17A mask adapter、SAM-only 最终尺度契约和相应测试集中在署名目录
[`../recon_kailang/`](../recon_kailang/README.md)。既有 runner 的兼容性修改仍保留在原位，
但所有新增独立入口都从该目录启动，便于区分作者和后续维护边界。

## Common batch flags

| Flag | Use |
|---|---|
| `--gpu-ids 0,1` | Explicit GPUs for reconstruction workers; no auto-detection |
| `--workers-per-gpu 2` | Allow memory-checked same-GPU overlap for already-labeled batches |
| `--skip-label` | Reuse existing/cached labels and do not start the labeling UI |
| `--force-label` | Clear selected interim labels before starting the labeling UI; requires `--force` |
| `--force` | Rerun completed reconstruction steps |
| `--keep-interim` | Keep heavy per-video interim outputs after final validation |
| `--keep-success-logs` | Keep successful per-step logs under the batch directory |
| `--visualize` | Enable per-step debug visualization; off by default |
| `--preview-device cuda --preview-gpu 0` | Use GPU SAM2 preview during HTTP labeling |
| `--no-mask-preview` | Save click prompts without interactive SAM2 mask preview |
| `--dry-run` | Print the commands the runner would launch |

See [docs/BATCH_QUEUE.md](docs/BATCH_QUEUE.md) for the complete runner
behavior, resource policy, resume rules, and measured peak usage.

## Single-step examples

Use these when debugging one step at a time.

**ViPE** — run under `third_party/vipe` uv (conda `cu128` provides CUDA + `uv`):

```bash
conda activate cu128
cd third_party/vipe
uv run python ../../recon_pipeline/vipe/run_sequence.py \
  --dataset hoi4d --video-id ZY20210800001__H1__C11__N07__S185__s02__T2 \
  --video ../../data/raw/HOI4D/HOI4D_release/.../align_rgb/image.mp4 \
  --gpu 0
```

**SAM3 hands** — conda `sam3` env (`hf auth login` + accept `facebook/sam3.1` license):

```bash
conda activate sam3
python3 recon_pipeline/sam3_hands/run_sequence.py --dataset hoi4d --video-id ... --video ... --gpu 0
```

The default hand-mask model is `sam3.1`. On Blackwell GPUs such as RTX 5090,
the hand step automatically uses `sam3` instead to avoid known `sam3.1`
floating-point precision failures. Non-Blackwell GPUs keep the original
`sam3.1` path.

```bash
# SAM2 object masks use the sam3 env, where SAM2 is installed editable.
# Labeling refreshes the SAM2 preview after each click by default, using CPU unless --preview-device cuda is passed.
conda activate sam3
# 2) Manual object label + SAM2 propagation (local window OR remote HTTP — see docs/LABELING.md)
python3 recon_pipeline/sam2_object/label_object.py \
  --dataset hoi4d --video-id ... --label-mode headed

python3 recon_pipeline/sam2_object/run_sequence.py --dataset hoi4d --video-id ... --video ... --gpu 0

# 3) Hand + object reconstruction
python3 recon_pipeline/hawor/run_sequence.py --dataset hoi4d --video-id ... --video ... --gpu 0
python3 recon_pipeline/sam3d/run_sequence.py --dataset hoi4d --video-id ... --video ... --gpu 0
python3 recon_pipeline/sam3d_scale/run_sequence.py --dataset hoi4d --video-id ... --video ... --gpu 0
python3 recon_pipeline/fp_pose/run_sequence.py --dataset hoi4d --video-id ... --video ... --gpu 0

# 4) World fusion (NPZ only; pass --visualize for test MP4)
python3 recon_pipeline/fuse/run_sequence.py --dataset hoi4d --video-id ... --video ...
```

See [docs/BATCH_QUEUE.md](docs/BATCH_QUEUE.md) for scheduling rules, logs, resume behavior, and measured resource peaks.

Batch queue runs are compact by default: after `fuse` validates the final package, the runner keeps `data/reconstruction/{dataset}/{video_id}/` and deletes per-video interim outputs plus successful step logs. Add `--keep-interim` or `--keep-success-logs` only when debugging.

Or run all steps (after labeling) via:

```bash
python3 recon_pipeline/run_pipeline.py --dataset hoi4d --video-id ... \
  --steps vipe,sam3_hands,sam2_object,hawor,sam3d,sam3d_scale,fp_pose,fuse --gpu 0
```

## Batch / parallel

Each step has `infer_parallel.py` with `--gpu-ids`, `--procs-per-gpu`, `--resume`, `--loud`, `--dry-run`:

```bash
python3 recon_pipeline/vipe/infer_parallel.py --dataset hoi4d --sample 10 --seed 42 \
  --gpu-ids 0 --procs-per-gpu 1 --dry-run
```

Task unit: **one video** within a dataset.

## Output roots

| Root | Purpose |
|------|---------|
| `data/interim/{dataset}/{video_id}/{step}/` | Per-step working artifacts; deleted by default after successful batch queue final validation |
| `data/reconstruction/{dataset}/{video_id}/` | Durable final package: gravity-aligned z-up fused world NPZ, copied scaled object mesh, summary, completion marker |

The main final file is `world_fused.npz`. The primary exported world frame is `gravity_z_up_world`, produced by rotating the ViPE world so +Z points up and yawing around +Z so the middle-frame camera focal axis projects to world +X. Use `c2w`, `object_ob_in_world`, `hand_trans`, and `hand_rot` for downstream reconstruction; these are all in the same gravity-aligned world frame. For multi-object videos, `object_0` is also exposed through the legacy single-object fields, while `object_ids`, `object_mesh_filenames`, `object_ob_in_world_all`, `object_ob_in_cam_all`, and `object_valid_all` describe every reconstructed object. Full video-frame visibility is stored in `object_visible_by_frame` and `object_pose_valid_by_frame`. Original ViPE/debug fields are retained with explicit names such as `c2w_vipe_world`, `object_ob_in_vipe_world`, and `hand_trans_hawor_world`.

## Visualization defaults

All steps are **inference-only by default**. See [docs/VISUALIZATION.md](docs/VISUALIZATION.md) for the shared contract (`--visualize` → one file under `{step}/vis/`).

| Step | `--visualize` output |
|------|----------------------|
| vipe | 2×2: RGB, seg, depth, camera path |
| sam3_hands | L+R hand masks on RGB |
| sam2_object | Object mask + label clicks |
| hawor | L+R MANO meshes on RGB |
| sam3d | RGB+mask \| raw SAM3D mesh (PNG) |
| sam3d_scale | Scale/debug overlays + scaled mesh artifacts |
| fp_pose | Object mask + pose axes |
| fuse | Object mask + mesh wireframe |

Final reconstruction packages can be visualized after the batch run with:

```bash
conda activate hawor
python3 recon_pipeline/visualize_final_reconstruction.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt
```

This writes one overlay MP4 per final output under `data/reconstruction/{dataset}/{video_id}/vis/`. The default final visualizer is fast and uses projected point clouds for every reconstructed object and the MANO surfaces. Pass `--surface-mode mesh` to use triangular mesh overlays, `--layout side-by-side` to add a reconstruction-only panel, `--draw-world-axis` to draw global axes, or `--no-hands` when MANO runtime dependencies are unavailable.

## Interim retention

Downstream steps read the **raw dataset RGB video**, not copies under interim dirs. Nonessential artifacts are discarded automatically during compact batch queue runs (see [docs/DATA_LAYOUT.md](docs/DATA_LAYOUT.md)).

## Submodules

Uses existing repo submodules under `third_party/` plus:

- `third_party/sam2` — manual point-prompt object mask preview/propagation
- `third_party/FoundationPose` — object pose backend for `fp_pose`

Initialize:

```bash
git submodule update --init third_party/vipe third_party/sam3 third_party/hawor \
  third_party/sam2 third_party/sam-3d-objects third_party/FoundationPose
```

See [docs/SETUP.md](docs/SETUP.md) for per-step conda environments.
