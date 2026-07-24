# Rotation-gauntlet grasp robustness test

`ocir.grasp_robustness` stress-tests a synthesized grasp with scripted wrist
rotations, independently of the `full_traj` demo-retargeting pipeline. The
demo carries mostly translate the object in one orientation; passing them
does not show the grasp can resist torsion. The gauntlet holds the object at
a fixed lifted position and sweeps the wrist orientation about the object's
center, so every recorded deviation is grasp slip attributable to a specific
rotation leg.

This is an experimental evaluation tool. It reuses the standard
`grasp_traj_simulation` task; no server-side changes are required.

## Program

1. The completed grasp prefix (approach, close, hold) of a **vertical-lift
   `grasp_traj` result** is copied unchanged. The lift's camera-frame
   displacement recovers world-up, so no calibration files are needed; a
   retargeted `full_traj` trajectory is rejected as input.
2. The object is lifted `--lift-height` (default 0.15 m) with a cosine ease
   over `--lift-seconds` (default 3 s) -- the proven envelope.
3. For each axis in `--axes` (default `yaw,pitch,roll`; yaw is about
   world-up, pitch/roll about two arbitrary horizontal axes) and each
   amplitude in `--angles-deg` (default `45`), the wrist rotates about the
   held object's center through four cosine-eased legs: `+A`, return, `-A`,
   return, with `--hold-seconds` pauses between legs. Peak angular speed is
   capped at `--angular-speed-deg` (default 45 deg/s).
4. Finger targets stay exactly at the grasp pose throughout; the object is
   dynamic under friction.

Because the rotation pivots on the object's center, the object *reference*
never translates during rotation legs -- the wrist swings on the hand-object
lever arm instead (peak linear speed ~0.1 m/s at the defaults).

## Run

```bash
export OCIR_DATA_ROOT=/data/users/hangkes2/OCIR

# 1. generate (pure NumPy)
PYTHONPATH=src python3 scripts/grasp_robustness/generate_rotation_gauntlet.py \
  --grasp-traj-dir ${OCIR_DATA_ROOT}/testing/grasp_traj/<sequence_id>/grasp_pose_<N> \
  --out-dir ${OCIR_DATA_ROOT}/testing/grasp_robustness/<sequence_id>/grasp_pose_<N> \
  --overwrite

# 2. simulate with the ordinary grasp_traj simulator
scripts/run_isaacsim_conda.sh scripts/isaac/simulate_grasp_traj.py \
  --trajectory-dir ${OCIR_DATA_ROOT}/testing/grasp_robustness/<sequence_id>/grasp_pose_<N> \
  --out-dir ${OCIR_DATA_ROOT}/testing/grasp_robustness/<sequence_id>/grasp_pose_<N>/isaac_sim

# 3. score each rotation leg (pure NumPy)
PYTHONPATH=src python3 scripts/grasp_robustness/analyze_rotation_gauntlet.py \
  --trajectory-dir ${OCIR_DATA_ROOT}/testing/grasp_robustness/<sequence_id>/grasp_pose_<N>
```

Useful sweeps: `--angles-deg 30,60,90` runs escalating amplitudes per axis;
`--axes pitch` isolates one axis; `--angular-speed-deg 90` tests dynamic
torsion. At the defaults the gauntlet is ~28 s of simulated time (~830
frames), several times longer than a demo carry.

## Scoring

`analyze_rotation_gauntlet.py` reads the phase table from `trajectory.json`
and the simulator's `object_track.npz`. Per phase it reports the added
object translation and rotation error relative to the reference, minus the
baseline deviation measured at the end of the post-lift hold (so the grasp's
entry bias is not charged to the rotation legs). A phase exceeding
`--pos-threshold` (default 3 cm) or `--rot-threshold-deg` (default 20 deg)
counts as slip; the first failing *rotation* phase is reported as
`first_failing_phase`. Results go to `rotation_report.json` next to the
track and to stdout as a table.

Orientation scoring depends on the object-orientation track being correct;
it requires the row-vector-convention fix in `read_object_world_pose`
(2026-07-17) -- rerun old simulations rather than analyzing stale tracks.

## Limitations

- The pitch/roll axis headings are arbitrary (only world-up is recovered);
  two runs of the same gauntlet are comparable, but "pitch" does not mean a
  specific world direction across sequences.
- The wrist path is not collision-checked; large amplitudes at low lift
  heights can swing the hand toward the table. The defaults (0.15 m lift,
  45 deg) keep clear on the tested sequences.
- Slip thresholds are starting values, not tuned per object.
