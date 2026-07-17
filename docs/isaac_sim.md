# Isaac Sim infrastructure: persistent server, standalone mode, visualization

All Isaac-facing tools in this repo (grasp visualization, anchored
visualization, grasp-trajectory physics simulation, closed-loop full-trajectory
simulation, DexYCB replay) share the
same execution infrastructure and dual-mode CLI.

## Persistent server vs. standalone

- **Persistent server** (`--isaac-mode server`, the default everywhere):
  start it once with

  ```bash
  scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py --width 1280 --height 720
  ```

  then every synthesis/trajectory run submits jobs to it over a small JSON
  control API (`src/ocir/sim/control_client.py`: `GET /status`, `POST /run`,
  `POST /shutdown`, `GET /jobs/<id>`). Keeps one Isaac Sim instance warm
  across many sequences -- much faster for batch runs.
  `start_isaacsim_server.py --status` prints server status;
  `--shutdown-server` asks it to exit.

  **Task hot-reload**: the server's default task module is
  `visualize_grasp.py`, whose `register_sim_tasks` chains
  `importlib.reload(...)` registrations for `grasp_pose_visualization`,
  `anchored_grasp_visualization`, `grasp_traj_simulation`, and
  `full_traj_simulation`, and whose
  module top additionally reloads the shared helper modules
  (`replay_dexycb`, `sim_cli`) -- plain from-imports would otherwise keep
  binding against the stale copies in `sys.modules`. A running server
  therefore picks up code changes (including helper-level ones like the
  video encoder) per job submission, without a restart.

- **Standalone** (`--isaac-mode standalone`, or run any Isaac script with
  `--mode local` directly): launches a fresh, one-shot Isaac Sim instance
  per job and exits when done. No persistent server needed -- use this to
  deploy on another headed machine without keeping a background Isaac Sim
  process running. Slower per-job (full Isaac Sim startup each time) but has
  no server dependency. The hold-open behavior is timed
  (`--hold-open-seconds`, default 10 s), so batch runs continue unattended;
  add `--hold-open-until-closed` to instead keep the window until closed
  manually.

## Grasp visualization (`visualize_grasp.py`)

Renders one synthesized grasp record with the object:

```bash
scripts/run_isaacsim_conda.sh scripts/isaac/visualize_grasp.py \
  --mode local \
  --sequence-dir /path/to/sequences/<sequence_id> \
  --grasp-json /path/to/output_root/<sequence_id>/grasp_000.json \
  --out-dir /path/to/some/output/dir
```

Important flags: `--tabletop-z`, `--show-table`/`--show-object-points`,
`--hold-open[-seconds]`,
`--camera-distance-scale`/`--camera-target-offset`/`--camera-focus-max-ratio`
(camera framing). With `--mode webrtc` (submits to the persistent server),
`--use-raw-object-pose` plus `--manifest`/`--sequence-id`/`--frame-id`
optionally replay the object's real recorded pose from a DexYCB manifest
instead of dropping it onto the table at the origin.

**Camera / screenshot**: the saved screenshot is a single 1-row x 3-column
image (`isaac_grasp.png`) combining 3 mutually orthogonal views -- front
(along -Y), side (along +X), and top (straight down +Z), each labeled --
rather than one oblique shot. `--camera-distance-scale` (default 1.5) sets
how far the camera sits from the focus target, as a multiple of the scene's
bounding radius. Framing is object-centered: if including the hand would
expand the box beyond `--camera-focus-max-ratio` (default 1.6) times the
object's own extent -- which happens for a poorly-converged grasp whose hand
ends up far from the object -- the camera frames on the object alone instead
of zooming out to fit both, so the shot stays close and legible.

## Video encoding

All videos (trajectory simulation, DexYCB replay) are written by the shared
`write_video` helper (`src/ocir/isaac/replay_dexycb.py`) as
**H.264/yuv420p mp4 with faststart** via ffmpeg (browser/Slack-playable),
falling back to OpenCV `avc1`/`mp4v` only if ffmpeg is unavailable.

## Troubleshooting

- If you run more than one checkout of this codebase on the same machine,
  make sure the persistent Isaac Sim server is actually started from the
  checkout you're editing (`start_isaacsim_server.py --status` echoes back
  its loaded `task_modules` paths) -- a stale server process serving an
  older checkout will silently run outdated code and fail in confusing ways
  (e.g. a removed CLI flag reported as "required").
- Camera repositioning inside a running app must go through
  `isaacsim.core.utils.viewports.set_camera_view`; raw camera xform ops go
  stale after `Camera.initialize()`.
