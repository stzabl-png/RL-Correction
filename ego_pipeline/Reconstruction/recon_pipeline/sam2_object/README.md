# sam2_object interface

`sam2_object` consumes the upstream point annotations in `label_prompt.json` and
uses the SAM2 video predictor to propagate one mask per object across the complete
video. It does not choose interaction episodes, classify material, or decide which
objects should be reconstructed; those decisions belong to the upstream HOI/VLM
stages.

## Input contract

The prompt is stored at:

```text
<RECON_INTERIM_ROOT>/<dataset>/<video_id>/sam2_object/label_prompt.json
```

Use schema `sam2_object_prompt_v2`. Every retained object has an independent ID,
prompt frame and set of positive/background points:

```json
{
  "schema_version": "sam2_object_prompt_v2",
  "objects": [
    {
      "object_id": "dustpan",
      "frame_idx": 55,
      "points": [[620.0, 430.0]],
      "labels": [1],
      "locked": true,
      "name": "dustpan"
    },
    {
      "object_id": "broom",
      "frame_idx": 61,
      "points": [[810.0, 390.0], [760.0, 420.0]],
      "labels": [1, 0],
      "locked": true,
      "name": "broom"
    }
  ]
}
```

Requirements:

- `object_id` is unique and contains only letters, numbers, `.`, `_`, or `-`;
- `frame_idx` is a valid zero-based video frame;
- `points` and `labels` have equal non-zero lengths;
- label `1` is a positive object point and label `0` is a background point;
- every object has at least one positive point;
- point coordinates are finite and lie inside the video frame.

The legacy single-object JSON shape remains accepted and is normalized to
`object_0`.

## Run only this step

From the repository root, after the prompt exists:

```bash
./ego_pipeline/reconstruct.sh <video.mp4> \
  --dataset <dataset> --root <video-root> \
  --steps=sam2_object --no-auto-label --keep-interim \
  --visualize --gpu-ids 0
```

For a direct single-video diagnostic call:

```bash
python ego_pipeline/Reconstruction/recon_pipeline/sam2_object/run_sequence.py \
  --dataset <dataset> --video-id <video_id> --video <video.mp4> \
  --gpu 0 --visualize --force
```

`third_party/sam2` and
`third_party/sam2/checkpoints/sam2.1_hiera_large.pt` are external assets and are
not stored in Git. Batch execution runs this step in the `codetr` conda environment,
which must contain PyTorch, the editable SAM2 package, and `eva-decord`. The same
environment is used by the upstream v17A SAM2 propagation.

## Output contract

```text
sam2_object/
├── label_prompt.json
├── sam2_object_complete.json
├── video_segmentation/masks/
│   ├── frame_000000_masks/<object_id>.png
│   ├── frame_000001_masks/<object_id>.png
│   └── ...
└── vis/<video_id>_objects.mp4        # when --visualize is enabled
```

Every frame directory contains one grayscale PNG for every object ID. A frame in
which SAM2 finds no foreground is represented by an all-zero PNG rather than a
missing file. The completion marker records `num_objects`, `object_ids`, per-object
prompt metadata and `detected_frames`.

Retries and `--force` runs preserve `label_prompt.json` and `frame_plan.json`, but
clear generated masks and visualization first. The old completion marker is
invalidated before propagation, so a failed rerun cannot appear successful.
