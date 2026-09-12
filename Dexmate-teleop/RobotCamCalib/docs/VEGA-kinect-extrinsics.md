# Camera ↔ robot extrinsics for the Dexmate Vega-1P + Azure Kinect

`RobotCamCalib/extr_calib_vega_kinect.py` — the RobotCamCalib third-view (eye-on-base)
hand-eye calibration adapted to this project. Written 2026-08-15; fills the gap listed in
`docs/PROJECT_HANDOFF.md §8.4` ("camera↔robot extrinsics have not been done").

## What you get

`configs/camera_extrinsics/kinect_<serial>.yaml`:

| key | meaning |
|---|---|
| `T_base_color` | pose of the Kinect **colour** camera in the robot `base` link frame (`X_base = T @ X_color`) — the calibrated quantity |
| `T_color_depth` | factory depth→colour rigid transform read from the device |
| `T_base_depth` | `T_base_color @ T_color_depth` — the frame `scripts/kinect_pointcloud.py --align depth` records point clouds in (int16 mm) |
| `T_tagmount_board` | board pose on the wrist link (nuisance parameter; sanity check that it is stable between runs) |
| `K_color`, `dist_color`, `image_size_color` | colour intrinsics used (factory by default; 8 OpenCV coefficients) |
| `n_samples`, `solver` | number of appended poses and residual statistics (`rot_err_deg_*`, `trans_err_*` in m) |
| `camera`, `robot`, `board` | serial / resolution / depth mode, URDF / links / arm / joint source, board name |

Plus `<yaml>.samples.npz` with every appended `(X_CamTag, X_WorldCammount, X_WorldTagmount)` so the
solve can be redone offline. Load in code with `magicdexmate.camera_extrinsics.load(serial_or_path)`.

Frames: `base` = URDF root link (chassis). Everything the peer's stack draws in viser is in that
frame. The calibration is valid **only while the camera and the robot chassis both stay put**;
after any bump, tripod move or robot drive, redo it (it takes ~10 minutes).

## Hardware

* Print `RobotCamCalib/assets/apriltag_grid/compact_apriltag_grid_4x4_tag48mm_a3.pdf` on A3 at
  **100 % scale** (laser printer, "actual size"), glue it flat to a rigid plate.
* Fix the plate rigidly to the **wrist / hand of the arm you calibrate with** (default link
  `R_arm_l7` for `--arm right`, `L_arm_l7` for `--arm left`). Where exactly does not matter — the
  solver estimates the offset — but it must not move between samples. A clamp on the Sharpa hand
  flange, or a plate screwed to the wrist, both work; a hand-held board does not.
* Kinect on a tripod where it will stay for data collection, looking at the manipulation
  workspace; the board must be visible over the whole range of arm poses you will sample.
* Same `--depth-mode` / `--color-res` as the recorder uses (`NFOV_UNBINNED`, `1080P` here — the
  recorder's `--color-res` default is `720P`; intrinsics are rescaled automatically, `T_base_color`
  does not depend on resolution, `T_color_depth` depends on the depth mode).

## Software prerequisites

```bash
bash scripts/setup_env.sh                       # .venv: viser, yourdfpy, msgpack + pupil-apriltags, loguru, opencv-contrib, pyk4a
.venv/bin/python scripts/kinect_pointcloud.py --probe   # libk4a / udev / pyk4a / device present?
sudo bash scripts/setup_robot_net.sh            # 192.168.50.100/24 etc. on enp3s0
```

`pyk4a` needs the Azure Kinect SDK (`libk4a`, `k4a-tools`) installed system-wide first; on this
desktop it was **not** installed as of 2026-08-15 (`--probe` says what is missing).

Joint angles come from `scripts/dexmate_observer.py` (read-only, publishes
`{"t_wall_us","q":{joint: rad}}` on `tcp://127.0.0.1:5590`); the console starts it automatically
when it detects the robot, or start it by hand:

```bash
ROBOT_NAME=dm/vgd1262ab823-1p ~/Dexmate/dexcontrol/.venv/bin/python scripts/dexmate_observer.py
```

This keeps dexcontrol out of the calibration tool's environment (the peer's rule: dexcontrol only
in its own venv). `--joints dexcontrol` reads in-process instead, if you run the tool under the
dexcontrol venv with the toolkit deps installed there.

## Procedure

```bash
cd <repo root>
.venv/bin/python RobotCamCalib/extr_calib_vega_kinect.py --arm right          # or --arm left
# multi-camera: one run per device
.venv/bin/python RobotCamCalib/extr_calib_vega_kinect.py --arm right --serial 000123456789
```

Open the viser URL it prints (`http://localhost:8080`, next free port if taken). The page shows the
Vega URDF at the live joint angles, the camera frustum with the live image at the current
estimate, and three GUI rows:

* `joints` — `OK <age> ms | right arm deg …` when the observer feed is fresh and carries all
  seven arm joints + the three torso joints; otherwise `⚠ …` and appends are refused.
* `detection` — `OK 16 tags, reprojection 0.xx px` or `REJECT …` (board not seen, blurred, or
  reprojection above `--max-reproj-px 2.0`).
* `solve` — sample count, residuals, camera position in the base frame once ≥ `--min-samples 8`.

Loop, per sample:

1. Move the arm to a new pose with the keyboard tool
   (`keyboard_joint_control.py --component right_arm`, `SOP.md §2`) or `goto_arm_pose.py`;
   this tool never commands the robot.
2. Wait for the arm to settle (the joint row updates at 15 Hz; the `age` should be < 100 ms).
3. Both rows `OK` → click **`click_and_append`**. The terminal logs the count and, from 8 samples,
   the current `T_base_color` and residuals.
4. Repeat for **≥ 8, better 12–15 poses**. Rotate the board a lot (±30–45° about several axes),
   translate it across the workspace the camera will observe during data collection, and keep all
   16 tags inside the image. Rotational diversity is what makes the hand-eye problem well
   conditioned; pure translations do not constrain it.
5. Residuals should settle around `rot_err_deg_mean` < 0.3° and `trans_err_mean` < 5 mm on the
   real robot (synthetic self-check: 0.1° / 1 mm). A single sample with a much larger residual is
   a bad pose (arm still moving, board flexed): `click_and_reset_samples` and redo, or continue —
   the solver is Huber-robust.
6. **`click_and_save`** → `configs/camera_extrinsics/kinect_<serial>.yaml`.

Sanity check afterwards: with the arm at a known pose, transform the wrist position
`FK(base←R_arm_l7)` into the depth frame with `inv(T_base_depth)` and confirm the point lands on
the wrist in a live point cloud (`kinect_pointcloud.py --live`). Rendering the robot URDF into the
recorder's viser view with `T_base_depth` is the natural next integration step (not done yet;
`scripts/viser_isaac_mirror.py:186-207` currently places clouds with a hard-coded offset).

## Dry run / self-check without hardware

```bash
# unattended: URDF FK -> synthetic board -> pupil_apriltags -> bundle PnP -> solver, vs ground truth
.venv/bin/python RobotCamCalib/extr_calib_vega_kinect.py --self-check            # also in scripts/check_all.py
# interactive GUI with a synthetic camera fed by a live joint stream (learn the buttons)
.venv/bin/python RobotCamCalib/extr_calib_vega_kinect.py --camera mock --joints zmq
```

Measured 2026-08-15 (right and left arm): `T_base_color` recovered to 0.03–0.06° / 2 mm and the
board-on-wrist offset to 0.05–0.07° / <1 mm from 12 synthetic poses, PnP reprojection 0.1–0.6 px.

## What was changed in RobotCamCalib for this

* `extr_calib.py::ViserUrdfUser` — continuous joints (Vega's drive wheels) report `None` limits;
  now treated as ±π so the URDF loads.
* `extr_calib.py::CamTagCalibrator` — accepts 5/8/12/14 OpenCV distortion coefficients (Kinect
  factory calibration is the 8-coefficient rational model), not only 5.
* New `cameras_kinect.py` (`KinectCamera`: pyk4a colour stream, factory intrinsics, depth→colour
  extrinsics, selection by serial like the recorder), new `extr_calib_vega_kinect.py` (this tool),
  new `magicdexmate/camera_extrinsics.py` (loader), `scripts/check_all.py` entry,
  `scripts/setup_env.sh` deps.
* Everything else (`apriltag_board.py`, the probabilistic solver, the board assets, the xArm6
  examples) is unchanged.

## Pitfalls

* The joint feed must include `torso_j1..3` — the FK from `base` goes through the torso. If the
  observer cannot read the torso, the tool refuses to append (`joint feed lacks [...]`).
* Never move the chassis (or push the robot) after calibrating; `base` is the chassis.
* Ambient light: the Kinect colour stream is auto-exposed; avoid motion blur (wait after moving).
* Board on the wrong arm / wrong `--arm`: the tool checks the URDF has the joints, not which arm
  physically carries the board — the residuals will simply be huge.
* Factory intrinsics are good to a fraction of a pixel at 1080p; if residuals stay > 1° with
  clean samples, run `intr_calib_charuco.py` and pass `--intrinsics outputs/intrinsics.yaml`.
