# Arms + Sharpa hands data collection — codebase analysis & operating set

> Written 2026-08-15 on `msc-Alienware-Aurora-R13` (wired NIC `enp3s0`) for the Dexterous ViTac Assembly project.
> Scope: teleoperate **only the two Vega-1P arms + Sharpa Wave hands** and record **visual + tactile** manipulation data.
> Continues the peer's docs: `README.md`, `docs/PROJECT_HANDOFF.md`, `SOP.md` (hands), `SOP_wholebody_teleop.md` (arms/head/waist).
> All `file:line` refs are against commit `c1192cf` (main).

---

## 0. TL;DR

- The peer (luhr / chenfj, Jun–Aug 2026) built **two independent ZMQ pipelines** — *arms/head/waist* (PICO headset + wrist trackers → mapping + Pink IK in Isaac Lab → `dexmate_bridge.py` → dexcontrol) and *hands* (Wuji gloves → dex-retargeting → `sharpa_real_runner.py` → Sharpa SDK, with tactile HDF5 recording) — plus one browser console (`scripts/vega_console.py`, :8086) that starts/stops everything and broadcasts recording sessions to every writer (`data/sessions/<name>/`).
- **Joints:** each arm is **7-DoF**, URDF `L/R_arm_j1..j7`, dexcontrol indexes them **0–6 per component**. Index 0 is the shoulder (`j1`), **not** the base. Base/torso/head/chassis are separate components you simply don't select. "Arms only" = all 7 joints per arm.
- **Hardware-validated** (real robot, 2026-07-27 / 08-10): keyboard arm parking + saved-pose restore, glove → Sharpa hand teleop, tactile recording + viser replay, Kinect capture, PICO arm mapping in sim signed off by an operator, and one 72-min live arm-bridge session (which set the 60 °/s follow-speed cap). **Never validated:** the console's real-robot buttons, velocity feed-forward / PID scaling in the bridge, camera↔robot extrinsics, multi-camera recording (mock only), and the arm↔hand **clap sync test** — so merged episodes are *not yet* synchronized data.
- **This desktop is not yet runnable for the arm path**: code hard-codes `~/Dexmate/{dexcontrol,dexmate-urdf}` + a dexcontrol venv that don't exist here (both repos were cloned into the repo root instead), the robot's Zenoh config (`~/.dexmate/comm/zenoh/*.dzcfg`) is missing, `pyk4a` is not in `.venv`, and two SDK-version bugs will bite on first contact (§6).

---

## 1. Joint naming / indexing (answer to "joints 1–6, 0 = base?")

Source: `dexmate-urdf/robots/humanoid/vega_1p/vega_1p.urdf` (limits byte-identical to `vega_1/vega_1.urdf`, which is what all repo code loads); dexcontrol per-component index from `dexbot_utils` configs; keyboard tool `dexcontrol/examples/advanced_examples/keyboard_joint_control.py:51-55` (`num_joints: 7` for arms).

| dexcontrol component | idx | URDF joint | role | pos limits (rad) | vel (rad/s) |
|---|---|---|---|---|---|
| `left_arm` / `right_arm` | 0 | `*_arm_j1` | shoulder | ±3.071 | 2.4 |
| | 1 | `*_arm_j2` | shoulder | L −0.453/1.553, R −1.553/0.453 | 2.4 |
| | 2 | `*_arm_j3` | upper-arm roll | ±3.071 | 2.7 |
| | 3 | `*_arm_j4` | elbow | −3.071/0.244 | 2.7 |
| | 4 | `*_arm_j5` | forearm roll | ±3.071 | 2.7 |
| | 5 | `*_arm_j6` | wrist | **±1.396 (±80°)** | 2.7 |
| | 6 | `*_arm_j7` | wrist roll | L −1.378/1.117, R −1.117/1.378 | 2.7 |
| `head` | 0–2 | `head_j1..j3` | pan/tilt/roll | ±1.483; ±2.792; −1.378/1.483 | 3.2 |
| `torso` | 0–2 | `torso_j1..j3` | sagittal fold | 0/1.571; 0/3.142; ±1.571 | 0.9 |
| `chassis` | 0–3 | `L/R_wheel_j1` (steer), `L/R_wheel_j2` (drive) | omni base | steer ±2.722; drive continuous | — |

- There is **no actuated joint 0** anywhere; `base` is the root *link*. The only `*_j0` names (`arm_center_j0`, `L/R_ee_j0`) are **fixed** joints (`vega_1p.urdf:243,565,785`).
- Repo joint-order arrays are name-keyed and consistent: `scripts/dexmate_bridge.py:67-73` (`ARM` 7+7, `HEAD` 3, `TORSO` 3, `ARM_VMAX`), `magicdexmate/home_pose.py:17-24` (14 arm + 3 head; torso deliberately absent), `sim/pink_vega_ik.py:50` (`ARM_PIN_ORDER` L1..7,R1..7), `sim/teleop_vega_pico.py:506-508` (`JOINT_DUMP`, R first). Everything crosses process boundaries as `{joint_name: value}` dicts, so order never matters on the wire.
- The wrist pair `j6/j7` is where the peer's limit-pinning problem lives (human forearm rolls 150–180°, `j6` has 160° total) — see `PROJECT_HANDOFF.md §8.1`.

---

## 2. How the peer built it (timeline → architecture)

| When | What (evidence) |
|---|---|
| 2026-06-11 | Roadmap `plans/00_roadmap.md`: decouple retarget from sim/real via ZMQ; dex-retargeting `vector` (+dexpilot) for Wuji→Sharpa; Isaac Lab scene; Sharpa real hand; Vega arm via dexcontrol; recording later. |
| Jun–Jul | Hand line (`fj_work_claude.md`, not in repo): Wuji source, retarget configs, Sharpa real sink, tactile writer, viser replay. **Real-hand + tactile validated 2026-07-27** (`SOP.md:6`). |
| Jul 19–Aug 3 | PICO whole-body line (`docs/PICO_teleop.md`, `PROJECT_HANDOFF.md §5`): producer, mapping law (chest-anchor, absolute palm orientation), Pink IK, joint guard, head/waist. Operator sign-off on hardware for arms+head+waist mapping. |
| Aug 4–9 | Replay-then-confirm flow (`replay_check.py`), one-command console, episode reset, unified recording, device detection, real-arm bridge instrumentation. |
| Aug 10–11 | cuRobo IK backend behind `--ik`, wrist relax state machine, bridge follow-speed 150→60 °/s after a dangerous 72-min live session (`687416d`), bench-port isolation + live-robot guards, Kinect multi-camera capture/record/view (`c1192cf`). |

Architecture (ports are ZMQ; every publisher broadcasts, nothing blocks; a subscriber that hears nothing for 1.5 s on the control channel treats the switch as OFF):

```
ARMS   PICO headset+trackers → scripts/teleop_pico_producer.py (.venv-pico) ─:5581→
       sim/teleop_vega_pico.py (.venv-isaac; mapping + Pink/cuRobo IK + joint_guard; Isaac = executor only) ─:5583 {q, cmd, engaged, mode, arm}→
         ├─ scripts/viser_isaac_mirror.py (embedded in console page :8086)
         └─ scripts/dexmate_bridge.py --live (dexcontrol venv) → Robot().{left,right}_arm.set_joint_pos[_vel] @100 Hz
       control channel :5584 (console → consumer/bridge/recorders: {recording, session, epoch, shutdown, drive switch})
HANDS  Wuji gloves (192.168.1.100/.101, UDP via wuji_sdk) → scripts/teleop_retarget.py (.venv, 60 Hz) ─:5556 R / :5557 L {qpos[22], wrist_quat, crc}→
       scripts/sharpa_real_runner.py (.venv + LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib; 20 Hz) → Sharpa hands 192.168.10.10 (L) / .20 (R)
       + tactile thread 30 Hz → HDF5
VISION Azure Kinect via pyk4a → scripts/kinect_pointcloud.py (.venv): view stream :5591, per-camera cloud recorder (int16 mm .npz)
CONSOLE scripts/vega_console.py (.venv) — supervises all children as process groups (no tmux), logs in logs/console/, device probe every 3 s
```

Design rules the peer enforces (`PROJECT_HANDOFF.md §2, §9`): *only broadcasts, never waits*; *one implementation per quantity* (mirror computes nothing; `home_pose.py` is the only default pose); every switch prints how often it fired; never SIGINT Isaac (kills `nvidia_uvm`).

---

## 3. The operative set (components you will actually run)

| # | Operative | Process / entry | Interpreter | I/O | Status on HW |
|---|---|---|---|---|---|
| A1 | Park an arm by keyboard | `dexcontrol/examples/advanced_examples/keyboard_joint_control.py --component right_arm` (`0-6` select joint, hold `w`/`s`, `q` quit) | dexcontrol venv, `ROBOT_NAME=dm/vgd1262ab823-1p` | Zenoh 192.168.50.20:7447 | ✅ validated |
| A2 | Save / restore named arm pose | `goto_arm_pose.py --list / <name> / --save` + `arm_poses.json` (`right_arm_teleop` = [−48.4,−21.1,−11.3,−91.9,−51.7,−4.7,17.0]°) | dexcontrol venv | motion plugin (smoothing, gravity comp) | ✅ validated on SDK 0.5.0rc1; **broken on 0.5.0** (§6) |
| A3 | Live arm teleop (sim) | `sim/teleop_vega_pico.py --control-mode arms` (console: tick 左臂/右臂 only) | `.venv-isaac` | :5581 in, :5583 out | ✅ sim sign-off |
| A4 | Live arm drive (real) | `scripts/dexmate_bridge.py --live --components left_arm,right_arm` (states off→syncing 8 °/s→following ≤60 °/s; `--stale-ms 200`, e-stop + hand-gap guards) | dexcontrol venv | :5583/:5584 in, Zenoh out | ⚠ one live session (08-10); console button never used with HW |
| A5 | Record→preview→replay on robot | `scripts/replay_check.py --record/--preview/--play` (+ `sim/replay_in_isaac.py`) | `.venv` (+ isaac for step 3) | publishes on :5583 like the sim | ✅ flow exists; sim-free way to move real arms |
| A6 | Read-only real-arm echo | `scripts/dexmate_observer.py` (auto-managed by console, :5590, 15 Hz) | dexcontrol venv | — | ⚠ never with live robot |
| H1 | Glove self-test | `PYTHONPATH= .venv/bin/python scripts/diag_glove.py --hand right --duration 12` | `.venv` | UDP | ✅ |
| H2 | Hand retarget publisher | `scripts/teleop_retarget.py --source wuji --hand right --pinch-weight 20 --relax-distal [--pub tcp://*:5557 for left]`; dual: `teleop_retarget_dual.py` | `.venv` | :5556/:5557 | ✅ (dual: mock only) |
| H3 | Real hand + tactile recording | `LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib PYTHONPATH= .venv/bin/python scripts/sharpa_real_runner.py --hand right --sub tcp://127.0.0.1:5556 --record data/teleop [--tactile-delta]` (tactile default `--tactile f6,deform,raw`, `--tactile-hz 30`; keys `e` engage, `w` freeze, `q` home, `r` record, `x` quit) | `.venv` | :5556 in, Sharpa SDK out, `.h5` | ✅ validated |
| H4 | Replay tactile episode | `scripts/replay_hand_viser.py <h5>` → http://localhost:8080 | `.venv` | — | ✅ |
| V1 | Camera capture / view / record | `scripts/kinect_pointcloud.py --probe / --live (:8087) / --pub-view tcp://*:5591 --record-session tcp://127.0.0.1:5584` | `.venv` (needs pyk4a) | ZMQ, `.npz` | ✅ capture; ⚠ multi-cam mock only; ⚠ no extrinsics |
| C1 | Console (single entry) | `.venv/bin/python scripts/vega_console.py` → http://localhost:8086 | `.venv` | spawns all above | ✅ sim; ⚠ real buttons unproven |
| C2 | Session recording | console 「开始新一段」→「开始录制」; or standalone runner `r` | — | `data/sessions/<name>/` | ✅ writers tested; ⚠ sync unverified |
| C3 | Merge / sync check | `scripts/merge_episode.py data/sessions/<n>` (→ `merged.h5`, 50 Hz NN resample), `scripts/check_clap_sync.py` (<20 ms pass) | `.venv` | — | ⚠ clap never on real data |
| T1 | No-hardware self test | `.venv/bin/python scripts/check_all.py` (18 checks; refuses to run while a live bridge/runner is up) | `.venv` | — | ✅ |
| N1 | Network bring-up | `sudo bash scripts/setup_robot_net.sh` (adds 192.168.50.100/24, 192.168.10.240/24, 192.168.5.100/24 on `enp3s0`) | — | — | run after every cable/reboot |

Console body-part checkboxes map to `--control-mode` (`arms` / `arms+head` / `arms+head+waist`) and to the bridge's `--components`; for your scope tick **左臂 + 右臂 + 控制左手/右手** only. Chassis is never driven by any code path (`VEGA_WELD_WHEELS=1`).

---

## 4. Two ways to run an arms+hands session

**Mode 1 — validated recipe (what the peer collected data with, `SOP.md`):** arms are *parked*, hands are teleoperated.
1. `sudo bash scripts/setup_robot_net.sh`; robot on, e-stop released; `ping 192.168.50.20`, `ping 192.168.10.10/.20`, `ping 192.168.1.100/.101`.
2. Park arm(s): A1 keyboard, or A2 `goto_arm_pose.py right_arm_teleop` (confirm range clear; robot-side motion plugin moves at scale 0.2).
3. Terminal 1: H2 retarget publisher for each hand. Terminal 2: H3 runner (tactile on by default: f6,deform,raw; add `--tactile-delta` for −49 % disk), press `e` to engage, `r` to start/stop an episode; ~1 s stall at start = tactile zero-calibration (fingertips must not touch anything).
4. Camera: V1 recorder pointed at the same session (`--record-session tcp://127.0.0.1:5584`) if you use the console's session broadcast; otherwise run it standalone.
5. After each take: H4 replay; check `Touch started: result=0` in the runner log if tactile is all zeros.

**Mode 2 — live arm + hand teleop (the whole-body stack, restricted to arms):** needs PICO headset + trackers (full-body calibration in headset), Isaac Lab (`.venv-isaac`), and the dexcontrol venv.
1. `vega_console.py`; tick 左臂/右臂 + hands; wrist mapping = 绝对 (default); optional 桌面高度 for table collision.
2. 「启动」(sim comes up in ~30 s), verify 链路/操作者/跟随状态 ✅.
3. Before first real drive use A5 (record → preview → replay in Isaac → replay on robot with `dexmate_bridge.py --live` and the page's 驱动真机 switch).
4. Then 「连接手臂」/「连接真手」, 「开始新一段」(homes to `home_pose`, resets IK), 「开始录制」.
5. Stop only via 「全部停止」/ shutdown broadcast — never Ctrl-C the Isaac process.

Safety layers already in place (keep them): SDK-side joint-limit clipping (`dexcontrol/src/dexcontrol/core/arm.py:324-330`); bridge slew limits from *measured* dt, stale-data disarm, e-stop check, min hand gap 0.08 m; consumer `joint_guard.py` (per-joint velocity clamp, 249-pair self-collision hold, table obstacle); Sharpa sink 0.9-scaled limits + 0.05 rad/step + `speed_coeff 0.3` + watchdog freeze; console refuses to connect when the device isn't detected; `check_all.py` refuses to run benches while a live bridge/runner exists.

---

## 5. Data streams and formats

`data/sessions/<name>/` (name = operator's or `YYYYmmdd_HHMMSS__HHMMSS`), all stamped with **host wall clock**:

| Stream | File | Writer | Notes |
|---|---|---|---|
| Arm commands | `arm.msgpack` | `replay_check.py --record` (started by console) | `_t` wall; `cmd` dict of 20 joints, `engaged`, `mode` |
| Real robot joints | `robot.msgpack` | `dexmate_bridge.py` @ `--rec-hz 50` | `t_wall_us`, pos/vel/current |
| PICO | `pico.msgpack` | producer | copy also in `logs/pico_*.msgpack` |
| Glove raw | `glove_{left,right}.msgpack` | `teleop_retarget.py` | 21×3 keypoints + conf + wrist quat |
| Hand joints + tactile | `<name>_{left,right}.h5` | `sharpa_real_runner.py` (`hand_data_writer.py:10-27`) | `hand_timestamp`, `{side}_hand_target_joint_positions (N,22)`, `{side}_hand_joint_positions (N,22)`, `tactile_timestamp`, `{side}_hand_tactile_f6 (M,5,6) f32` [Fx Fy Fz Mx My Mz per fingertip], `_deform (M,5,240,240) u8`, `_raw (M,5,240,320) u8`; `tactile_encoding` = `raw+lzf` or `delta30+gzip1` — **read tactile only via `magicdexmate.recording.read_tactile`** |
| Point cloud | `cloud_<serial>/NNNNNN.npz` + `cloud_t_<serial>.msgpack` | `kinect_pointcloud.py` | int16 **millimetres**, depth-camera optical frame, `--rec-hz 10`, voxel 1 cm |
| Merged | `merged.h5` | `merge_episode.py` | NN-resampled to 50 Hz over the *intersection*; heterogeneous rows as JSON; tactile/cloud by index only — review format, not training-ready |

Standalone runner (no console) writes `data/teleop/episode_XXXX/episode_XXXX_{left,right}.h5` instead.

---

## 6. State of this desktop (2026-08-15) — what blocks the arm path

| Item | Expected by code | Here | Fix |
|---|---|---|---|
| dexcontrol SDK + its venv | `~/Dexmate/dexcontrol/.venv/bin/python` (`vega_console.py:41`, SOP §2/§3.5+) | `~/Dexmate` absent; repo-root `./dexcontrol` (0.5.0) has no venv; `import dexcontrol` in `.venv` resolves to an empty namespace package | `mkdir ~/Dexmate && ln -s $PWD/dexcontrol ~/Dexmate/dexcontrol` then create its venv (`uv venv --python 3.11 ~/Dexmate/dexcontrol/.venv && uv pip install --python ~/Dexmate/dexcontrol/.venv/bin/python -e ~/Dexmate/dexcontrol`) — or point `DEXCONTROL_PY` at the existing conda env `/home/msc/anaconda3/envs/dexcontrol/bin/python` (dexcontrol 0.5.0 editable from `/home/msc/jiakaichen/dexcontrol`, dexcomm, dexbot_utils, dexmate_urdf all present) |
| Vega URDF | `~/Dexmate/dexmate-urdf/robots/humanoid/vega_1/vega_1.urdf` (`sim/pink_vega_ik.py:44`, `sim/vega_scene.py:64`, `joint_guard.py:47`, ~10 scripts, only some honour `$VEGA_URDF`) | repo-root `./dexmate-urdf` | `ln -s $PWD/dexmate-urdf ~/Dexmate/dexmate-urdf` |
| Robot Zenoh config | `~/.dexmate/comm/zenoh/dm_vgd1262ab823-1p.dzcfg` (auto-loaded, `dexcontrol/src/dexcontrol/__init__.py:56-72`) | missing | copy from the lab laptop (`luhr-Legion`) or obtain from Dexmate; plus `export ROBOT_NAME=dm/vgd1262ab823-1p` |
| Camera binding | `pyk4a` in `.venv` (`SETUP.md`, commit `6c3a62c`) | not installed | `uv pip install --python .venv/bin/python pyk4a` (needs libk4a + udev rule; `kinect_pointcloud.py --probe`) |
| Wired NIC | `enp3s0` in `vega_console.py:48`, `setup_robot_net.sh:10` | ✅ switched from `enp49s0` (staged) | cable to the Dexmate switch still unplugged (`NO-CARRIER`) |
| Sharpa SDK 5.0.8, wuji_sdk, avahi-browse, PICO service | present | ✅ | — |
| git index | — | `dexcontrol/` and `dexmate-urdf/` are **staged as gitlinks** (nested repos, no `.gitmodules`) — a commit would push dangling submodule pointers | `git rm --cached dexcontrol dexmate-urdf` and add both to `.gitignore` (or move them to `~/Dexmate`) |

Code bugs that will surface on first hardware contact with the vendored **dexcontrol 0.5.0**:
1. `goto_arm_pose.py:109` calls `comp.set_joint_target(target, scale=…, tracked=True)` — removed in 0.5.0 (`dexcontrol/CHANGELOG.md:34-37`, no alias → `AttributeError`). Replace with `comp.move_to_joint_pos(target, velocity_scale=args.scale)` (returns a `MotionHandle`).
2. `scripts/dexmate_bridge.py:145` reads `getattr(c, "joint_pos_limits", None)`; the SDK property is `joint_pos_limit` (singular, `(N,2)` array, `dexcontrol/src/dexcontrol/core/component.py:288`). So the bridge's own limit clamp (`:459-461`, `:538-539`) is dead code and the "已从机器人读取…限位" line never prints; only the SDK's internal clip protects the arm. Fix the name and unpack `lim[:,0], lim[:,1]`.
3. Documentation still carries the previous operator's absolute paths (`/home/luhr/…` in `SOP.md:82-83,171-177,313-314`, `goto_arm_pose.py:8-17`).

---

## 7. Gaps vs. what ViTac assembly data needs — suggested order of work

1. **Make the arm path runnable here** (§6 table), then re-run `check_all.py` (18/18) and `SOP.md` §1–§3 keyboard parking against the robot — the lowest-risk first contact.
2. **Fix the two 0.5.0 bugs** above before `goto_arm_pose.py` / `dexmate_bridge.py --live`.
3. **Camera ↔ robot extrinsics** — the handoff's explicit open item (`PROJECT_HANDOFF.md §8.4`; console labels every camera "尚未校准,暂不连接"). **Done 2026-08-15 (tool, not yet run on hardware):** `RobotCamCalib/extr_calib_vega_kinect.py` adapts your `RobotCamCalib` hand-eye toolkit to Vega + Kinect (joints from `dexmate_observer.py`, factory Kinect intrinsics, board on `R_arm_l7`/`L_arm_l7`, output `configs/camera_extrinsics/kinect_<serial>.yaml` with `T_base_color` / `T_color_depth` / `T_base_depth`; loader `magicdexmate.camera_extrinsics`; hardware-free `--self-check` in `check_all.py`). Procedure: `RobotCamCalib/docs/VEGA-kinect-extrinsics.md`. Remaining: run it on the real robot, then make `viser_isaac_mirror.py` / the recorder consume `T_base_depth` instead of the hard-coded cloud offset.
4. **Run the clap test** on a real take (`check_clap_sync.py`, <20 ms) before trusting `merged.h5`; add hardware sync (master/subordinate cable) if you use two Kinects — today each camera thread free-runs.
5. **Decide the arm input for assembly:** Mode 1 (parked arms, hand-only teleop) is validated and gives clean tactile data but no reach; Mode 2 (PICO live arms) is 7 % wrist-limit-pinned on the peer's benchmark and its real-arm bridge has one live session — plan a wearing test with `--follow-speed 60` and feed-forward *off* first (`PROJECT_HANDOFF.md §8.3`).
6. **Training-ready export:** `merge_episode.py` keeps rows as JSON and tactile/cloud by index; a per-episode exporter (joints + f6 + deform + RGB-D at a fixed rate) is a small script on top of `read_tactile()`.
7. Sharpa hands are not in the arm collision model (URDF ends at `l8`) — relevant once both arms work close together over an assembly jig; `joint_guard.py --min-hand-gap` (0.08 m) is the only guard.
