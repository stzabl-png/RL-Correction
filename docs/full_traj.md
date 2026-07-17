# Closed-loop full-trajectory carry

`ocir.full_traj` converts an existing completed grasp trajectory into a
MANO/object carry reference, then tracks the demonstrated object path in
Isaac Sim by changing only the Sharpa wrist pose.

This is an experimental simulation controller. It is useful after a grasp
trajectory already closes the hand successfully; it does not synthesize or
repair grasps.

## Grasp and activation semantics

Current grasp records contain two poses: `pregrasp` and `grasp`. There is no
overclosed squeeze pose or squeeze action. Trajectory segment value `3` is a
legacy schema label named `squeeze`, but its current behavior is only a
stationary hold at the final grasp posture while contacts settle.

Full-trajectory control is disabled for every pre-carry frame. It activates
on the first frame whose segment is `carry` (`4`). At activation it copies
the final 22-DoF grasp target and commands that same vector for the entire
carry. Actual finger joints can deflect under their soft PD drives and
contact forces, but their commanded targets do not change.

## Pipeline

### 1. Generate the reference

```bash
scripts/run_grasp_synthesis_conda.sh \
  scripts/full_traj/generate_full_traj.py \
  --grasp-traj-dir ${OCIR_DATA_ROOT}/testing/grasp_traj/<sequence_id>/grasp_pose_1 \
  --out-dir ${OCIR_DATA_ROOT}/testing/full_traj/<sequence_id>/grasp_pose_1
```

The input directory must contain `trajectory.npz` and `trajectory.json` from
`grasp_traj`. Generation:

- copies every frame before carry without modifying any array;
- derives the post-grasp MANO wrist frame from the 21 recorded keypoints;
- applies `assets/robots/hands/sharpa_wave/grasp_synthesis/bodex/mano_transfer.yml`
  to obtain the nominal Sharpa base pose;
- loads the recorded object poses from `human_demo.npz`;
- resamples synchronized object/wrist paths to at most 5 mm and 3 degrees per
  path interval by default; and
- replaces the old vertical-lift carry with the resulting reference while
  holding the final grasp targets.

Use `--overwrite` for a non-empty output directory. The generator otherwise
fails rather than mixing artifacts from different references.

### 2. Simulate with carry-only feedback

Against the persistent server:

```bash
scripts/run_isaacsim_conda.sh \
  scripts/full_traj/simulate_full_traj.py \
  --trajectory-dir ${OCIR_DATA_ROOT}/testing/full_traj/<sequence_id>/grasp_pose_1 \
  --out-dir ${OCIR_DATA_ROOT}/testing/full_traj/<sequence_id>/grasp_pose_1/isaac_sim
```

One-shot local Isaac Sim:

```bash
scripts/run_isaacsim_conda.sh \
  scripts/full_traj/simulate_full_traj.py \
  --mode local \
  --trajectory-dir /path/to/full_traj/output \
  --out-dir /path/to/simulation/output
```

The persistent server's default `visualize_grasp.py` registration hub now
also exposes `full_traj_simulation`.

## Controller behavior

The video is treated as an ordered geometric path through SE(3), not as a
frame-clock signal. The controller is a monotonic pure-pursuit follower:
each update it projects the actual object pose onto the reference path
(searching a bounded forward window, never moving backward) and steers the
wrist toward a lookahead carrot ahead of that projection. Progress therefore
follows the object itself -- it never waits for the object to enter a tight
tolerance band around each individual path point. An object further than the
projection acceptance radius from the path (for example after a dropped
grasp) makes no projection progress; a stall-escape timeout then creeps one
path point forward so a single unreachable region cannot consume the entire
run, and total carry time is still capped at a multiple of the source carry
duration.

At the first carry update, the controller measures the actual object and
wrist poses and aligns the video object path to that measured object pose.
By default (`--hold-alignment`) this alignment is kept for the whole carry,
so the controller tracks the demonstrated relative motion rather than
absolute camera-calibrated coordinates. `--no-hold-alignment` restores the
old behavior of removing the alignment smoothly over `--catchup-seconds` so
the target converges to the absolute recorded path.

The nominal wrist replays MANO's changing wrist-to-object transform on top
of the synthesized grasp relation measured at carry entry. Bounded PI terms
correct object translation and orientation: the proportional term acts on
the error to the lookahead carrot (this is what propels pursuit), while the
integral term acts on the cross-track error to the projected path point, so
the standing lookahead offset cannot wind it up. Orientation corrections
rotate the wrist about the object's actual position rather than the wrist
origin, so they do not inject translation error through the hand-object
lever arm. Wrist translation, rotation, speed, and acceleration are limited.
Excessive hand-object slip is reported as a lost grasp, but feedback remains
active and continues bounded pursuit.

The run finishes when the projection reaches the end of the path and the
final cross-track error is within tolerance (reported as
`final_pose_within_tolerance`), when the last point stalls past the
stall-escape timeout, or at the total timeout.

Important defaults:

| Control | Default |
| --- | --- |
| Pursuit lookahead | 2 cm translation, 10 degrees rotation |
| Projection window / acceptance radius | 40 path points / 5 cm |
| Projection rotation weight | 0.05 m per radian |
| Cross-track tolerance (coverage + completion) | 8 mm, 5 degrees |
| Stall-escape timeout | 1 second |
| Total timeout | 3x source duration |
| PI gains, translation | `kp=0.6`, `ki=0.15` |
| PI gains, rotation | `kp=0.6`, `ki=0.15` |
| Maximum feedback correction | 5 cm, 20 degrees |
| Maximum wrist speed | 0.25 m/s, 90 degrees/s |
| Lost-grasp threshold | 5 cm or 30 degrees for 5 control updates |

`--carry-mode friction` is required. Contact-aware finger-target rewriting
is rejected because it violates the fixed-target contract.

## Frames and pose conventions

Generated reference files store camera-frame positions in metres and
quaternions in `wxyz` order. The simulator maps both object and wrist paths
through the same DexYCB camera-to-Isaac-world calibration and table-height
offset used by `grasp_traj`.

Homogeneous transforms map local coordinates into their named parent frame.
For example, the demonstrated wrist-to-object relation is computed as
`inverse(T_world_object) @ T_world_wrist`.

## Outputs and evaluation

Generation writes compatible `trajectory.npz/json` plus:

- `full_traj_reference.npz`: recorded object and MANO wrist paths, source
  frame coordinates, and the fixed finger target;
- `full_traj_reference.json`: source paths, carry boundary, calibration, and
  resampling metadata.

Simulation writes the usual video, screenshot, scene, and report plus:

- `controlled_wrist_track.npz`: nominal and commanded wrist poses, carrot and
  projection path indices, carrot and cross-track errors;
- `object_track.npz`: actual/reference object positions and orientations;
- `finger_track.npz`: desired, driven, and actual finger positions.

`report.json.full_traj_controller` includes completion/timeouts, carrot and
cross-track error summaries (coverage fractions are computed from cross-track
error against the projected path point), the held alignment magnitude,
saturation counts, lost-grasp state, and an exact `finger_targets_constant`
check. `ordered_path_metrics` compares the simulated object path against the
reference the controller actually tracked (the aligned path when
`--hold-alignment` is on) using monotonic dynamic time warping, so its
translation/orientation RMSE, maxima, and path coverage measure ordered
geometric agreement without requiring video and simulation frames to have the
same timing.

## Limitations

- Wrist-only feedback cannot recover object degrees of freedom that the
  current contacts do not control.
- Once contact is lost, continued pursuit may not recover the object; the run
  remains marked `lost_grasp` even though the requested chase policy stays
  active.
- Controller defaults are conservative starting values, not tuned gains for
  every object and grasp.
- Isaac validation is required for final gain selection; pure-NumPy tests
  cover reference preservation, fixed finger targets, SE(3) math, projection
  progress and pursuit, alignment holding, stall-escape timeouts, and
  lost-grasp reporting.
