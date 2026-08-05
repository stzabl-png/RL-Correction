# Water-bottle Step4 validation

This dataset is the final two-part PCO-1810 bottle scene derived for
`water_bottle.mp4` (170 frames at 30 fps). The published scope begins at the
Step4-ready assets and replay data; upstream segmentation and asset-generation
files are intentionally excluded.

## Placement evidence

- Bottle body: left interaction begins at source frame 15. The left hand is
  invalid there, so frame 31, the first valid frame in the same interaction
  window, is used for table XY placement.
- Cap: right interaction and placement both use source frame 13.
- Mesh AABB centres are aligned to the selected wrist XY coordinates. The
  static baseline starts each mesh 2 mm above the table.
- The hand trajectories are placement evidence only. They do not initialize or
  command the robot.

The A6000 passive-scene audit measured body/cap XY errors of 0.0 mm and less
than 0.001 mm. Both rigid bodies settled onto the table without penetration or
explosive motion. The body/cap masses are 0.53 kg and 0.003 kg; friction is 0.5
and 0.4; gravity, positive inertia, collision geometry, table bounds, and the
official DexMate asset all passed.

## Screw model

The CAD meshes contain matching right-handed PCO-1810 threads. Direct-GPU
PhysX rejects its built-in rack-and-pinion constraint, so Step4 uses an
equivalent batched tensor relation with a 3.18 mm pitch, two turns, and 6.36 mm
total constrained travel. It supports:

- pre-engaged closed cap to full unscrew and release;
- free cap to alignment-gated capture, two-turn screw-on, and closed retention.

Capture thresholds are 3 mm radial, 3 mm axial, 10 degrees tilt, and 30 degrees
yaw. Before capture, a GPU analytic barrier blocks a wrongly aligned cap from
passing through the bottle envelope. Body/cap mesh collision is filtered to
avoid fighting the analytic helix; robot/cap and table/cap contacts remain
active.

The temporary CAD substitute was generated with BOSL2. Its BSD-2-Clause
redistribution notice is bundled in `THIRD_PARTY_LICENSES/BOSL2_LICENSE.txt`.
Hashes for the two meshes and two replay assets are bundled in `SHA256SUMS`.

## A6000 results

All commands ran in the official DexMate Step4 environment using Isaac Sim
Direct-GPU and exited successfully.

| Audit | Result | Key measurements |
|---|---|---|
| Pre-engaged unscrew, 1 env | Pass | 2.016 turns; released; engaged-state coupling error 0.000153 mm; radial drift 0.0000037 mm |
| Free-cap capture and screw-on, 1 env | Pass | initially free; misaligned penetration blocked; captured; -2.000 turns; -6.36005 mm travel; retained closed; coupling error 0.000087 mm |
| Free-cap capture and screw-on, 1024 envs | Pass | 1024 caps and 1024 GPU screw mechanisms; complete two-turn motion; no non-finite state; coupling and radial-drift gates passed |
| Actual `DirectRLEnv.step()` path | Pass | screw coordinate advanced through the training decimation loop; observations remained finite |

The analytic screw mechanism is a scalable physical abstraction, not a claim
that PhysX resolves individual thread-tooth contacts. Mass values are explicit
engineering assumptions rather than estimates from the source video. No RL
success-rate claim is made here; this deliverable verifies the scene, physical
properties, placement, engagement state, and GPU-trainable execution path.
