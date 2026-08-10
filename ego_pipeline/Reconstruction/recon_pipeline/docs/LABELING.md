# Manual SAM2 object labeling

> ⚠ 本文档只覆盖 **fallback 人工路线**。物体 mask 默认全自动（无需人点）：
> `ego_pipeline/bin/auto_label_v17a.py`（reconstruct.sh 自动调用）或
> `tools/v17a_to_label_prompt.py`。别据本文推断"每条视频都要人工标注"。

Object masks use **SAM2 point prompts** (positive = object, negative = background). Label one or more objects, then `sam2_object/run_sequence.py` propagates each object through the clip and writes the `sam2_object/` interim output consumed by SAM3D, scaling, FoundationPose, and fusion.

SAM2 is the object-mask backend. SAM3 remains the hand-mask backend in `sam3_hands`, and SAM3D is a separate object-mesh model.

## Modes

| Mode | Flag | Use when |
|------|------|----------|
| Headed | `--label-mode headed` | Local machine with display (same browser UI, auto-opened locally) |
| HTTP | `--label-mode http` | Remote SSH server — browser UI + port forwarding |

`label_object.py` requires `--label-mode` when run directly. The full batch queue (`run_batch_queue.py`) uses HTTP labeling by default and currently starts one browser page for the whole video list.

Preview device:

- Default: `--preview-device cpu`, also available as `--cpu-only`.
- GPU preview: `--preview-device cuda --gpu <id>`.
- No preview inference: `--no-mask-preview`.

When preview is enabled, SAM2 inference runs after each click and undo so the mask overlay updates immediately.

`headed` and `http` use the same HTML interface and controls. `headed` binds to
`127.0.0.1` and opens the browser locally; `http` lets you choose the host for
remote SSH/port-forward use.

## Controls

| Key | Action |
|-----|--------|
| Left click | Add point (object if mode=object) |
| Right click | Background point |
| `p` / `n` | Switch to object / background mode |
| `a` / `d` | Previous / next frame |
| `q` / `e` | Previous / next 5 frames |
| `[` / `]` | Previous / next frame |
| `u` | Undo last point |
| `c` | Clear all points |
| `1`-`9` | Select object slot |
| `o` | Add a new object |
| `x` / `Delete` / `Backspace` | Delete the current object |
| `l` | Save/lock the current object locally |
| `m` | Unlock/modify the current object |
| `s` | Save prompt / load next batch video |
| `Esc` | Cancel |

When mask preview is enabled, both modes refresh the SAM2 preview after each
click and after undo. After **Save object** locks an object locally, the page
keeps that object's preview mask and draws it in a stable object-specific color
when you are viewing that object's prompt frame. The active editable object
continues to use the blue live preview.

Run the headed local browser UI for one video on a local machine with a display:

```bash
conda activate sam3
python3 recon_pipeline/sam2_object/label_object.py \
  --dataset hoi4d \
  --video-id ZY20210800001__H1__C11__N07__S185__s02__T2 \
  --label-mode headed
```

Run the headed local browser UI with GPU SAM2 preview:

```bash
conda activate sam3
python3 recon_pipeline/sam2_object/label_object.py \
  --dataset hoi4d \
  --video-id ZY20210800001__H1__C11__N07__S185__s02__T2 \
  --label-mode headed \
  --preview-device cuda \
  --gpu 0
```

Headed mode also accepts `--video-list`. It keeps one browser page open, saves
the current video's `label_prompt.json`, then automatically loads the next
video in that same page. SAM2 preview is loaded once and reused across the
whole headed batch, matching the HTTP flow. For remote labeling, prefer HTTP
mode.

## HTTP mode (remote SSH)

For batch labeling, HTTP mode is the default workflow. The server opens one browser page, saves the current video's `label_prompt.json`, then automatically loads the next video in the same page.

On the server, use the integrated `sam3` env. This environment contains SAM3 for hand masks and the editable SAM2 install used by object labeling:

```bash
conda activate sam3
# Labeling preview uses CPU by default; add --preview-device cuda --gpu 0 only if you want GPU preview.
python3 recon_pipeline/sam2_object/label_object.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt \
  --label-mode http \
  --http-host 0.0.0.0 \
  --http-port 8765
```

The batch queue starts this same HTTP labeling flow automatically, so the full pipeline command does not need a `--label-mode` flag:

```bash
python3 recon_pipeline/run_batch_queue.py \
  --dataset hoi4d \
  --video-list data/video_lists/test_seq.txt \
  --gpu-ids 0,1 \
  --http-host 0.0.0.0 \
  --http-port 8765
```

If these videos already have interim `label_prompt.json` files and you want to
label them again, add `--force-label --force`. `--force-label` clears the
selected interim prompts before opening the page, and `--force` reruns
reconstruction after the new prompts are saved.

For one video, replace `--video-list ...` with `--video-id <VIDEO_ID>` or `--video <path/to/image.mp4>`.

By default, each click or undo refreshes the SAM2 mask preview on CPU, so labeling does not reserve GPU memory while later reconstruction tasks run. The browser opens as soon as the HTTP server is ready; SAM2 loads in the background. The server prewarms the SAM2 video state for the current video and reuses it across objects and prompt frames, so the first click should not pay the full video initialization cost when prewarm has finished. Pass `--preview-device cuda --gpu 0` to use GPU preview, or pass `--no-mask-preview` to skip SAM2 preview inference while clicking (points only, no overlay).

HTTP mode with GPU SAM2 preview:

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

On your laptop:

```bash
ssh -L 8765:127.0.0.1:8765 user@remote-host
```

Open `http://127.0.0.1:8765/` in a browser.

| Control | Action |
|---------|--------|
| Slider (drag) | Preview frame number only |
| Slider (release) / ◀ ▶ | Jump to frame |
| `a` / `d` or arrow keys | Previous / next frame |
| `q` / `e` | Previous / next 5 frames |
| `[` / `]` | Previous / next frame |
| Left click | Object point + refresh preview |
| Right click | Background point + refresh preview |
| `u` | Undo last point + refresh preview |
| `c` or **Clear all** | Remove all points |
| `1`-`9` or object dropdown | Select the active object |
| `o` or **New object** | Add the next object slot |
| `x`, `Delete`, `Backspace`, or **Delete object** | Delete the active object |
| `l` or **Save object** | Lock the current object's points and keep its colored preview in the page |
| `m` or **Edit object** | Unlock the current object for changes |
| `s` or **Save & next** | Write prompt file and load the next batch video |

**Save & next** writes:

```text
data/interim/{dataset}/{video_id}/sam2_object/label_prompt.json
```

When the last video is saved, the server reports that all videos are labeled and the page closes if the browser allows scripts to close it.

During a full batch queue run, this prompt is copied to the lightweight durable
cache as soon as the batch runner detects that the video has been labeled:

```text
data/object_labels/{dataset}/{video_id}/sam2_object/label_prompt.json
```

Later `run_batch_queue.py --skip-label` runs restore the cached prompt back into
`data/interim/.../sam2_object/` automatically if the interim prompt was deleted.

## Multi-object labeling workflow

Use object slots when one video contains multiple reconstructable objects:

1. Pick object `1` (`object_0` internally), jump to a frame where it is visible, click foreground/background points, and wait for the SAM2 preview.
2. Press `l` or **Save object** to lock that object's points locally in the page and keep its colored preview visible on its prompt frame.
3. Press `o` or **New object** for the next object, then jump to any frame where that object is visible. Different objects do not need to use the same prompt frame.
4. Repeat for all objects, then press `s` / **Save & next** once to write the video-level prompt and advance to the next video.

Use `x`, `Delete`, `Backspace`, or **Delete object** to remove the active
object slot before the video-level save. If the last object is deleted, the
page creates a fresh empty `object_0` slot so the prompt can still be edited.

The per-object **Save object** action is a local page lock to prevent
accidental edits and repeated labels. Its colored preview mask is transient UI
state only; the durable file is still written by the video-level **Save & next**
action and stores the prompt points, not preview images.

## Prompt file format

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
    },
    {
      "object_id": "object_1",
      "frame_idx": 105,
      "points": [[280.0, 210.0], [300.0, 230.0]],
      "labels": [1, 0],
      "locked": true
    }
  ]
}
```

`labels`: `1` = foreground (object), `0` = background. Older single-object prompt files with top-level `frame_idx`, `points`, and `labels` are still accepted and are treated as `object_0`.

## After labeling

```bash
conda activate sam3
python3 recon_pipeline/sam2_object/run_sequence.py \
  --dataset hoi4d --video-id ... --video ... --gpu 0
```

For a multi-object prompt, this step runs SAM2 propagation once per object and writes one mask per object under each frame directory (`object_0.png`, `object_1.png`, ...). Downstream object stages reconstruct every labeled object. Legacy top-level mesh/pose aliases still point to `object_0` for compatibility.

By default, SAM2 uses:

```text
third_party/sam2/checkpoints/sam2.1_hiera_large.pt
configs/sam2.1/sam2.1_hiera_l.yaml
```

Override these with `--sam2-checkpoint` and `--sam2-model-cfg` if you use a smaller checkpoint.

Or batch (skips videos without `label_prompt.json`):

```bash
conda activate sam3
python3 recon_pipeline/sam2_object/infer_parallel.py \
  --dataset hoi4d --video-list my_videos.txt --gpu-ids 0
```
