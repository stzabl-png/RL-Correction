# Open-loop full-trajectory carry

`ocir.full_traj` replaces the vertical-lift carry of an existing
`grasp_traj` result with the recorded MANO wrist motion. The complete wrist
command is generated before Isaac Sim starts. During simulation the object
remains dynamic, but its measured pose never changes the wrist command.

Use this pipeline after `grasp_traj` has produced candidate trajectories. A
failed source grasp remains a failed grasp: `full_traj` retargets carry motion
and does not synthesize or repair finger contacts.

## Control contract

Everything before the first `carry` frame is copied exactly from the selected
source candidate. Segment value `3` remains the stationary final-grasp hold;
there is no separate squeeze action or overclosed finger pose.

At carry entry:

- the first retargeted wrist pose equals the final grasp-prefix wrist pose;
- the first aligned object reference equals the final prefix object reference;
- all 22 finger targets are copied from the final prefix frame and held
  exactly constant for the complete carry; and
- the object is driven only by gravity, collision, and friction.

The simulator reads object poses for recording and metrics only. There is no
PI controller, path projection, lookahead target, slip recovery, or other
feedback from the object to the hand.

## Wrist retargeting

The demonstrated object path is aligned to the nominal source object pose at
carry entry:

```text
T_C_O_ref[k] =
    T_C_O_source[0] @ inverse(T_C_O_demo[0]) @ T_C_O_demo[k]
```

This aligned path is an evaluation reference, not an object command, in
friction mode.

`--retarget-mode object` (default) drives the wrist so this object path
replays through the synthesized grasp relation, held rigid:

```text
T_C_W_robot[k] = T_C_O_ref[k] @ inverse(T_C_O_source[0]) @ T_C_W_robot[0]
```

The human's wrist motion *relative to the object* (in-hand adjustment plus
MANO wrist-keypoint noise, measured at 30+ mm / ~10 degrees over a DexYCB
carry) is deliberately discarded: with fixed finger targets it is pure
commanded slip that pries the grasp open.

`--retarget-mode wrist` replays the recorded MANO wrist motion verbatim
instead:

```text
T_C_W_robot[k] =
    T_C_W_robot[0] @ inverse(T_C_W_mano[0]) @ T_C_W_mano[k]
```

This preserves the full relative MANO wrist translation and orientation, at
the cost of commanding that wrist-object drift into the rigid grasp.

## Timing

The recorded human carry is too dynamic for the fixed-target PD grasp: the
clip is cut at the grasp frame, so it starts at its full instantaneous wrist
speed (0.16-0.28 m/s measured on DexYCB sequences) and sustains 2-3x the
speed of the proven vertical lift, with 40-85 deg/s wrist rotation. Replayed
one-to-one, all tested carries shear the object out of the grasp within
0.5 s.

Generation therefore retimes the carry while preserving its geometric path
exactly:

- `--time-scale` (default `3.0`) stretches the carry duration relative to the
  source video clock by inserting interpolated rows (the schema has one fixed
  `dt`, so duration is expressed in row count). At 3x the peak wrist speed
  (~0.13 m/s on the tested sequences) sits under the proven vertical-lift
  envelope (0.157 m/s); at 2x (~0.20 m/s) the tested grasp still slipped at
  peak speed;
- `--ease-in` / `--ease-out` (default `0.3` s each) ramp the path speed from
  and back to zero with cosine profiles, removing the velocity step between
  the stationary grasp hold and the moving carry. Each ease is capped at 45%
  of the scaled carry duration.

`--time-scale 1.0 --ease-in 0 --ease-out 0` replays the raw video timing. If
valid video frames have gaps, the missing integer frames are interpolated
before retiming. The reference arrays are retimed together with the command,
so they stay pointwise-synchronized.

The default 30 Hz trajectory is interpolated across two `app.update()` calls
per row, giving 60 Hz wrist commands.

## Generate

```bash
scripts/run_grasp_synthesis_conda.sh \
  scripts/full_traj/generate_full_traj.py \
  --grasp-traj-dir ${OCIR_DATA_ROOT}/testing/grasp_traj/<sequence_id>/grasp_pose_1 \
  --out-dir ${OCIR_DATA_ROOT}/testing/full_traj/<sequence_id>/grasp_pose_1
```

The input directory must contain `trajectory.npz` and `trajectory.json` from
`grasp_traj`. Generation writes a compatible replacement trajectory plus:

- `full_traj_reference.npz`: raw synchronized MANO wrist and object paths,
  source frame indices, and the fixed finger target;
- `full_traj_reference.json`: carry boundary, source paths, timing, and
  calibration metadata.

The command refuses a non-empty output directory unless `--overwrite` is
passed.

## Simulate

Against the persistent Isaac server:

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

`--carry-mode friction` is required. Contact-aware finger target rewriting is
rejected because it violates the fixed-target carry contract.

The physical scene, hand asset, object collision, friction, drive settings,
solver rates, camera, and video path all come from the same
`simulate_grasp_traj` implementation used by the source trajectory.

## Outputs and metrics

Simulation writes:

- `video.mp4`, `screenshot.png`, and `scene.usd`;
- `object_track.npz` with actual and reference object position and orientation;
- `finger_track.npz` with desired, driven, and actual finger positions;
- `report.json` with normal grasp/lift metrics and `full_traj_open_loop`.

`full_traj_open_loop` reports:

- `object_pose_used_for_control: false` and
  `wrist_commands_precomputed: true`;
- an exact `finger_targets_constant` check;
- time-aligned translation/orientation errors at the commanded timestamps;
- ordered dynamic-time-warping path metrics for geometric path agreement
  without requiring equal progression speed.

The evaluation tolerances can be changed with
`--path-position-tolerance` and `--path-orientation-tolerance-deg`. They affect
metrics only and never affect simulation commands.

## Limitations

- Object displacement during grasping and slip during carry are not corrected.
- A successful vertical lift does not guarantee that the same contacts can
  withstand the recorded wrist translations and rotations.
- The kinematic wrist follows the precomputed command even after the object is
  dropped.
- Candidates should be evaluated independently; the candidate index is part
  of both the source and output directory.
