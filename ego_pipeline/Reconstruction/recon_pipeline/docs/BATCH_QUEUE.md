# Hybrid batch queue runner

`recon_pipeline/run_batch_queue.py` is the recommended entry point for batch reconstruction. It starts manual labeling first, then launches reconstruction for each video as soon as that video's label prompt is saved.

The runner is intentionally conservative:

- labeling uses one HTTP page and SAM2 CPU preview by default;
- reconstruction worker slots are assigned from `--gpu-ids` and `--workers-per-gpu`;
- GPU steps use an approximate per-step memory reservation before launching;
- when multiple steps are waiting on the same GPU, later pipeline stages are prioritized when they can fit;
- completed steps are skipped unless `--force` is passed;
- visualization is off unless `--visualize` is passed;
- successful videos are compacted by default after final-output validation;
- logs and generated batch manifests are written under `data/interim/{dataset}/batch_queue/{timestamp}/`.

This avoids the observed GPU-memory overflow cases while still allowing safe same-GPU overlap for lighter steps.

## Basic usage

Run the HOI4D test batch on two GPUs. This is the recommended command for a
fresh full-pipeline run:

```bash
python3 recon_pipeline/run_batch_queue.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt \
  --gpu-ids 0,1 \
  --http-host 0.0.0.0 \
  --http-port 8765
```

`run_batch_queue.py` defaults to HTTP labeling (`--label-mode http`) and currently only supports HTTP for batch labeling. The local headed browser UI is available through `sam2_object/label_object.py`; see [LABELING.md](LABELING.md).

The runner prints the local browser URL:

```text
Open labeling UI: http://127.0.0.1:8765/
```

For remote use, forward the port from your laptop:

```bash
ssh -L 8765:127.0.0.1:8765 user@remote-host
```

Then open `http://127.0.0.1:8765/`.

## Required inputs

Current dataset discovery is implemented for HOI4D. By default, the runner uses:

```text
data/raw/HOI4D/
```

Each selected sequence must resolve to an RGB video:

```text
data/raw/HOI4D/HOI4D_release/<sequence>/align_rgb/image.mp4
```

Select videos with one of these inputs:

| Input | Meaning |
|---|---|
| `--video-list data/video_lists/test_seq.txt` | Text/JSON list of HOI4D ids, HOI4D relative paths, or MP4 paths |
| `--video-id ZY...__T2` | One HOI4D video id |
| positional `input` | Directory or MP4 path to discover/process |
| `--sample N --seed 42` | Randomly sample discovered videos |
| `--limit N` | Keep only the first N discovered/listed videos |
| `--dataset-root /path/to/HOI4D` | Override the default raw HOI4D root |

HOI4D list entries may use either artifact ids or raw relative paths:

```text
ZY20210800001__H1__C11__N07__S185__s02__T2
ZY20210800001/H1/C11/N07/S185/s02/T2
```

The pipeline also requires the third-party models, conda environments,
FoundationPose weights, SAM2 checkpoint, SAM3/SAM3D assets, and HaWoR MANO
files described in [SETUP.md](SETUP.md).

## What happens

1. The runner writes a batch manifest and a generated label list under `data/interim/{dataset}/batch_queue/{timestamp}/`.
2. It starts `sam2_object/label_object.py` in the `sam3` env with `--preview-device cpu`.
3. It watches for `data/interim/{dataset}/{video_id}/sam2_object/label_prompt.json`.
4. When a prompt appears, the runner immediately copies it to `data/object_labels/{dataset}/{video_id}/sam2_object/label_prompt.json`, then enqueues the video.
5. GPU workers run the requested reconstruction steps in order.
6. Each step writes its normal completion marker under `data/interim/{dataset}/{video_id}/{step}/`.
7. `fuse` writes the durable final reconstruction under `data/reconstruction/{dataset}/{video_id}/`.
8. After final validation succeeds, the runner deletes `data/interim/{dataset}/{video_id}/` and success logs by default; the lightweight prompt has already been cached.

Default reconstruction order:

```text
vipe -> sam3_hands -> sam2_object -> hawor -> sam3d -> sam3d_scale -> fp_pose -> fuse
```

`fuse` is CPU-only, but it is run by the same worker immediately after `fp_pose` because it is short and keeps the per-video log flow simple.

Final validation requires:

```text
data/reconstruction/{dataset}/{video_id}/
├── world_fused.npz
├── object_mesh_scaled_final.obj
├── objects/{object_id}/object_mesh_scaled_final.obj
├── masks/masks_manifest.json
├── masks/objects/frames/frame_XXXXXX_masks/{object_id}.png
├── masks/hands/frames/frame_XXXXXX_masks/{left_hand_0,right_hand_0}.png
├── world_summary.json
└── reconstruction_complete.json
```

`world_fused.npz` is the self-contained reconstruction package. It stores camera intrinsics and gravity-aligned z-up `c2w`, object poses in camera/world frames, frame indices, per-object mesh filenames, the ViPE gravity alignment transform, the middle-frame camera-forward XY yaw metadata, and MANO parameters in the same z-up world frame when available. For compatibility, `object_0` is also exposed through the legacy single-object fields and top-level mesh. `fuse` also copies the 2D SAM2 object masks and SAM3 hand masks into `masks/` so they survive interim cleanup. The final completion marker is written only after these files validate.

The batch queue does not create final visualization videos unless `--visualize`
is passed for per-step debug outputs. For the recommended final inspection
video, run the standalone visualizer after reconstruction; see
[VISUALIZATION.md](VISUALIZATION.md).

## Resource policy

The runner uses a memory-aware queue rather than strict one-video-per-GPU scheduling:

| Resource | Policy |
|---|---|
| Labeling | CPU SAM2 preview by default; no GPU reservation |
| GPU worker slots | `--workers-per-gpu` worker slots are created for each id in `--gpu-ids` |
| GPU memory | Each GPU has an approximate reservation budget, default `43000` MB |
| GPU steps | Each step reserves its estimated GPU memory before launch; waiters block when the budget would be exceeded |
| Same-GPU ordering | Later pipeline stages are preferred over earlier stages when both can fit; the releasing worker also gets a short continuation window so another video does not immediately steal the GPU |
| CPU-only steps | `fuse` runs after its dependencies complete |
| Resume | Skip completed steps unless `--force` is passed |

This means the runner can use both GPUs when at least two labeled videos are ready, and can also overlap safe combinations on the same GPU. Large combinations such as ViPE + ViPE, SAM3 hands + ViPE, and SAM3 hands + FoundationPose are blocked by the reservation model.

The runner does not auto-detect GPU count. If `--gpu-ids` is omitted it uses GPU `0` only. Pass all GPUs you want it to use, for example `--gpu-ids 0,1`.

The default estimates are intentionally conservative and can be overridden for experiments:

```text
vipe=33000,sam3_hands=32000,sam2_object=9000,hawor=8000,sam3d=22000,sam3d_scale=10000,fp_pose=18000
```

Use `--step-gpu-mem-mb step=MB,...` only after measuring the target machine. A lower `fp_pose` estimate was faster in one trial but allowed `fp_pose` to overlap with `sam3_hands` and caused a CUDA OOM during FoundationPose registration.

## Recommended assignment mode

For the current two-A6000 test machine and already-labeled HOI4D batches, the recommended assignment mode is:

```bash
python3 recon_pipeline/run_batch_queue.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt \
  --gpu-ids 0,1 \
  --skip-label \
  --force \
  --workers-per-gpu 2
```

This mode creates two worker slots per listed GPU, but it does not blindly run every ready step. Each GPU step first reserves the configured memory estimate against a `43000` MB per-GPU budget. If the next step does not fit, that worker waits and prints the reason:

```text
wait for GPU 0 memory (32000+18000>43000 MB)
wait for GPU 0 higher-priority step
wait for GPU 0 continuation priority (5.0s)
```

The current default reservations are:

| Step | Reservation MB | Notes |
|---|---:|---|
| `vipe` | 33000 | Heavy; blocks another heavy step on the same GPU |
| `sam3_hands` | 32000 | Heavy; must not overlap with `fp_pose` |
| `sam2_object` | 9000 | Can overlap with some heavier steps |
| `hawor` | 8000 | Low VRAM, can overlap, but may contend for CPU |
| `sam3d` | 22000 | Moderate; can overlap with `fp_pose` |
| `sam3d_scale` | 10000 | Short; can overlap with `sam3_hands` or `fp_pose` |
| `fp_pose` | 18000 | Conservative because FoundationPose can spike during registration |

The priority rule is intentionally simple: a video that is farther through the pipeline has priority over an earlier-stage video when both are waiting and the higher-priority step can fit. A worker that just released a GPU reservation also gets a short continuation window so another worker does not immediately steal the GPU before the same video can start its next step.

## Per-step resource profile

Measured on one HOI4D video with the default `fp_pose --pose-mode track` path:

```text
ZY20210800001__H1__C11__N07__S185__s02__T2
```

Hardware sampled: GPU 0, NVIDIA RTX A6000 48 GB. CPU percentage is summed across the process tree, so `800%` means roughly eight fully used CPU cores. GPU values were sampled once per second with `nvidia-smi`.

| Step | Elapsed s | Peak CPU % | Peak RAM MB | Peak GPU0 mem MB | Peak GPU0 util % |
|---|---:|---:|---:|---:|---:|
| label_cpu_http | 25.9 | 214.7 | 6998.3 | 535 | 6 |
| vipe | 200.6 | 828.2 | 36812.5 | 32747 | 100 |
| sam3_hands | 122.7 | 289.9 | 8033.6 | 31109 | 100 |
| sam2_object | 32.6 | 312.6 | 7938.7 | 8741 | 100 |
| hawor | 148.8 | 316.0 | 13650.0 | 7525 | 100 |
| sam3d | 62.0 | 610.6 | 11151.7 | 21057 | 100 |
| sam3d_scale | 23.7 | 802.9 | 6844.2 | 8963 | 100 |
| fp_pose | 679.8 | 484.5 | 6661.3 | 32699 | 100 |
| fuse | 3.4 | 324.0 | 279.0 | 30 | 0 |

`fp_pose` defaults to `--pose-mode track`, which registers the labeled prompt
frame and then tracks both forward and backward to cover every frame, consuming
each frame's SAM2 mask centroid (plus a 6D Kalman filter) to initialise every
`track_one` step -- the full-video mask propagation is used on every frame. The slower per-frame registration mode is available as
`fp_pose/run_sequence.py --pose-mode register-each` for diagnostics.

The raw sampling logs were generated during the profiling run and are not part
of the minimal retained output set.

## Important flags

`--step-arg STEP:FLAG`(可重复) 把额外参数透传给任一步, 例如
`--step-arg=fp_pose:--pose-mode=register-each`。经 reconstruct.sh 时必须用 `=` 连写形式
(它的透传是单 token)。这是批量队列此前唯一缺失的机制。


| Flag | Meaning |
|---|---|
| `--dataset hoi4d` | Dataset adapter; current discovery code supports HOI4D |
| `--dataset-root /path/to/HOI4D` | Override `data/raw/HOI4D` |
| `--video-list path.txt` | Process videos from a list file |
| `--video-id VIDEO_ID` | Process one resolved video id |
| `--sample N`, `--limit N`, `--seed 42` | Select a subset from discovered/listed videos |
| `--gpu-ids 0,1` | List GPUs available to the reconstruction worker slots |
| `--workers-per-gpu 2` | Start two reconstruction worker slots per listed GPU; default is `1` |
| `--gpu-mem-budget-mb 43000` | Approximate per-GPU scheduler memory budget |
| `--step-gpu-mem-mb fp_pose=18000,...` | Override per-step GPU memory estimates for scheduler experiments |
| `--label-mode http` | Default and currently the only supported batch labeling mode |
| `--skip-label` | Do not start the labeling UI; process videos that already have `label_prompt.json`, restoring from `data/object_labels/` when the interim prompt is missing |
| `--force-label` | Delete selected interim `label_prompt.json` files before starting the labeling UI; requires `--force` |
| `--steps ...` | Run a subset of reconstruction steps after labeling |
| `--http-host 0.0.0.0`, `--http-port 8765` | Host/port for the labeling server |
| `--no-browser` | Start HTTP labeling without trying to open a browser |
| `--preview-device cpu` | Default; use CPU SAM2 preview while labeling |
| `--preview-device cuda --preview-gpu 0` | Use GPU SAM2 preview during labeling |
| `--no-mask-preview` | Label points only, with no SAM2 preview overlay |
| `--poll-interval 2.0` | Seconds between label-completion polling checks |
| `--visualize` | Pass `--visualize` to reconstruction steps that support it; off by default |
| `--force` | Re-run steps even when completion markers already exist |
| `--keep-interim` | Keep `data/interim/{dataset}/{video_id}/` after final validation; use this when inspecting SAM2 masks or FP pose outputs |
| `--keep-success-logs` | Keep successful per-step batch logs |
| `--dry-run` | Print label and reconstruction commands without running them |

## Logs and resume behavior

For each run, the batch directory contains:

```text
data/interim/{dataset}/batch_queue/{timestamp}/
├── manifest.json
├── label_video_list.txt
├── label_server.log
├── batch_summary.json
├── batch_summary.md
└── logs/{video_id}/{step}.gpu{N}.log
```

By default, success logs are removed after a video reaches a validated final reconstruction. Failed-video logs are kept so failures can be debugged.

If the runner is stopped, rerun the same command. Videos with existing labels are immediately eligible for reconstruction, completed steps are skipped unless `--force` is used, and videos with `data/reconstruction/{dataset}/{video_id}/reconstruction_complete.json` are treated as done. Because final-complete videos no longer need their interim directories, the batch runner may delete stale per-video interim folders on resume.

Aggressive cleanup requires the default step list or any custom `--steps` list that includes `fuse`. If you intentionally run a partial batch without `fuse`, add `--keep-interim`.

Use `--skip-label` when all videos are already labeled:

```bash
python3 recon_pipeline/run_batch_queue.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt \
  --gpu-ids 0,1 \
  --skip-label \
  --workers-per-gpu 2
```

Use `--skip-label --force` to reuse existing labels and rerun completed
reconstruction steps. This is the recommended command after changing the
implementation of a downstream step such as `fp_pose`:

```bash
python3 recon_pipeline/run_batch_queue.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt \
  --gpu-ids 0,1 \
  --skip-label \
  --force \
  --workers-per-gpu 2
```

Use `--force-label --force` when you want to manually label the same videos
again. This clears the selected videos' interim prompt files before starting the
HTTP labeling UI, so old prompts do not immediately enqueue reconstruction:

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

With the current code, forced reruns regenerate `fp_pose` using the default
`--pose-mode track` prompt-frame registration + bidirectional FP++ tracking path.
Per-frame registration (`--pose-mode register-each`) is reachable from `fp_pose/run_sequence.py` directly, or through `run_pipeline.py` which forwards unknown flags to the step; the batch queue (`run_batch_queue.py`) has no extra-arg passthrough and always runs `track`.

## Design notes

This is not a full global scheduler. Worker slots are still assigned to a GPU, and a running video keeps its step order. The scheduler only decides when same-GPU workers may launch their next GPU step based on approximate memory reservations and step priority.

The current design improves utilization over strict step-by-step batching and over one-task-per-GPU batching: reconstruction can begin as soon as the first video is labeled, while the same browser page continues labeling the remaining videos, and lighter same-GPU overlaps can run when they fit the memory budget.

On the three-video HOI4D test list, measured with `--skip-label --force --gpu-ids 0,1`, the safe memory-aware scheduler with `--workers-per-gpu 2` completed in `1660.1s` versus `2362.2s` for one worker per GPU.
