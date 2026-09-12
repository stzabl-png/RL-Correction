# MagicDexMate — handover notes

> Handover for the Vega-1 whole-body teleoperation work.
> Written 2026-08-09. The codebase is **`<仓库根目录>`**.

Whole-body teleoperation of a **Dexmate Vega-1** humanoid: a PICO headset and
wrist trackers drive the two 7-DoF arms, the 3-DoF head and the 3-joint sagittal
torso; a pair of **wuji** gloves drive two **Sharpa Wave** dexterous hands.
Everything is validated in Isaac Lab first, then streamed to the real robot over
`dexcontrol`. Recorded episodes are meant to become VLA training data.

**This file lives in `~/magicsim/GR00T-WholeBodyControl/` (the fork that holds
the engineering log), but the code it describes lives in
`<仓库根目录>/`.** All paths below are absolute for that reason.

This file is the map. Two other documents go deeper and are kept current:

| Document | What it holds |
|---|---|
| `<仓库根目录>/SOP_wholebody_teleop.md` | **Operator manual.** How to start, what "good" looks like, what to do when it isn't. Deliberately no theory. |
| `~/magicsim/GR00T-WholeBodyControl/workflow_fj.md` (next to this file) | **Engineering log.** Every decision, every measurement, every hypothesis that turned out wrong. Reverse-chronological. Long, but it is the reason we don't re-run failed experiments. |
| `~/dexmate/fj_work_claude.md` | The hand/glove line's own log (separate pipeline, 1250 lines). |

---

## 1. Where the code is

**The codebase is `<仓库根目录>`.** Nearly all work happens here.
The other paths are dependencies, not workspaces.

| Path | Role |
|---|---|
| **`<仓库根目录>`** | **This repo. The codebase.** Mapping, IK, sim, teleop, recording, all tooling. |
| `~/magicsim/GR00T-WholeBodyControl` | Fork of NVlabs GR00T-WholeBodyControl. Holds `workflow_fj.md`; also the vendored PICO SDK (`external_dependencies/`). |
| `~/Dexmate/dexcontrol` | Vendor SDK for the real robot (zenoh/dexcomm). **Has its own venv; never install it into ours.** |
| `~/Dexmate/dexmate-urdf` | Official Vega-1 URDF — the authority for joint limits. |
| `~/magicsim/MagicSim` | Isaac USD assets (`vega_1p_weldwheels.usd`). |
| `~/dexmate/V2AP-demo`, `~/magicsim/T-Rex-main` | **Reference implementations from another group, same robot, same hands.** Read-only. See §7. |

### Environments — picking the wrong one is the most common failure

| Purpose | Interpreter |
|---|---|
| Isaac sim / the teleop consumer | `.venv-isaac/bin/python` |
| Offline analysis, viewers, gloves, recording | `.venv/bin/python` |
| PICO producer (needs a cp310 `.so`) | `.venv-pico/bin/python` |
| Real robot | `~/Dexmate/dexcontrol/.venv/bin/python` |

`dexcontrol` pulls zenoh/dexcomm and **must stay out of `.venv-isaac`** (that env
pins numpy 1.26 and an onnxruntime built against CUDA 12; it is fragile). The
real robot therefore runs as a separate process talking ZMQ.

Glove/hand scripts need `PYTHONPATH=` in front (ROS's `PYTHONPATH` leaks in and
breaks imports). `scripts/check_all.py` strips it centrally.

---

## 2. How it fits together

```
PICO headset + wrist trackers
   │  scripts/teleop_pico_producer.py        (.venv-pico)     ZMQ :5581
   ▼
sim/teleop_vega_pico.py                       (.venv-isaac)
   │  mapping law + Pink IK — both plain numpy/pinocchio on the CPU.
   │  Isaac Lab is only the simulated EXECUTOR; it is never in the mapping.
   ├─► Isaac (physics, joint feedback)
   ├─► ZMQ :5583  { q measured, cmd commanded, engaged, mode, arm targets }
   │        ├─► scripts/viser_isaac_mirror.py     browser :8086  ← what you watch
   │        └─► scripts/dexmate_bridge.py         the real robot (separate process)
   └─◄ ZMQ :5584  control channel (browser → consumer + bridge + recorders)

wuji gloves ─► scripts/teleop_retarget.py (.venv) ─► ZMQ :5556/:5557 ─► Sharpa hands
```

Two rules hold this together and should not be broken:

- **Only broadcasts, never waits.** No process blocks on another. Any one can be
  restarted alone. The control channel is stateless: the publisher re-sends
  everything every tick, and a subscriber that hasn't heard for 1.5 s treats it
  as "off" — so closing the browser stops the robot rather than leaving it
  running on the last command.
- **One implementation per quantity.** The mirror computes *nothing*; it only
  poses a URDF with joint angles the consumer publishes. A second implementation
  of a mapping always drifts from the first — that has already misled us twice.

---

## 3. Entry points

```bash
cd <仓库根目录>

# The operator entry point. Zero arguments; everything else is in the browser.
.venv/bin/python scripts/vega_console.py            # browser :8086

# Everything that can be checked without hardware. 13 checks, ~2 minutes.
.venv/bin/python scripts/check_all.py               # --quick for the 6-second subset

# Side-by-side robot playback: baseline vs a fix, joints pinned at limits called out.
.venv/bin/python scripts/viser_verify.py            # browser :8090

# Regression bench: same clips, full metric set, before/after.
.venv/bin/python scripts/regress_teleop.py --save logs/regress/after.json
.venv/bin/python scripts/regress_teleop.py --compare logs/regress/base1.json logs/regress/after.json
```

**Run the regression on an idle machine.** See §6, item 2 — machine load can
change the numbers by 30×, and it disguises itself as noise.

---

## 4. Key files

*Paths in this section are relative to `<仓库根目录>`.*

### Mapping and control
| File | What it is |
|---|---|
| `sim/teleop_vega_pico.py` | The consumer. Mapping law, clutch, head/waist channels, filters, and every runtime switch. ~2600 lines, the heart of the system. |
| `sim/pink_vega_ik.py` | Pink (QP differential IK) driver for the two arms. 14-joint reduced pinocchio model. |
| `magicdexmate/home_pose.py` | **The default pose, and the only copy of it.** Pure Python, importable from every venv. Used by the sim, the solver and the real-robot bridge. |
| `magicdexmate/palm_fix.py` | Operator-wrist → robot-EE orientation constants. Palm normal = `+ee.x`, fingers = `+ee.z` (three independent proofs, 2026-08-01). |
| `magicdexmate/joint_guard.py` | Velocity clamp, self-collision (249 link pairs), environment obstacles (table/walls) with a fail-loud self-check. |
| `magicdexmate/control_link.py` | The `:5584` control channel. One-way, stateless, fail-safe. |
| `scripts/dexmate_bridge.py` | The real robot. Three states (off / syncing / following), rate limits in deg/s from *measured* elapsed time, velocity feedforward, PID readout, per-second diagnostics. |

### Viewers
| File | What it is |
|---|---|
| `scripts/vega_console.py` | The console. Process supervision + the embedded mirror. Part selection, device detection, epoch/recording controls. |
| `scripts/viser_isaac_mirror.py` | The robot view. Computes nothing. Also draws the **operator's skeleton** and palm/finger arrows. |
| `scripts/viser_verify.py` | Two robots side by side from two joint CSVs, limit-pinned joints and per-frame jumps called out. |

### Recording
| File | What it is |
|---|---|
| `magicdexmate/recording/hand_data_writer.py` | Threaded HDF5 writer for one hand (joints + tactile). Optional delta+keyframe tactile encoding (−49 % disk, bit-exact). |
| `magicdexmate/recording/__init__.py` | `read_tactile()` — **always read tactile through this**; a delta-encoded frame read raw looks like a plausible image and is silently wrong. |
| `scripts/merge_episode.py` | Offline merge of all streams onto one wall-clock timeline. |
| `scripts/check_clap_sync.py` | Measures the residual timing offset between the hand and arm streams. Has a `--self-check`. |

### Diagnostics (`sim/dev_*.py`)
Every one is re-runnable and self-checking. `dev_limit_lock.py` (limit pinning),
`dev_lag.py` (filter latency, with an ASCII plot), `dev_bridge_sync.py` (bridge
state machine), `dev_undefined_names.py` (the NameError class this repo keeps
hitting), plus recording and console smokes. `scripts/check_all.py` runs them all.

---

## 5. Brief history

- **Jul 19 – 30** — PICO pipeline built. Wheel instability found to be the root
  cause of all residual error (welded wheels became the default). Mapping law
  settled on an absolute waist frame; tracker plumbing (raw wrists + model waist)
  resolved.
- **Jul 31 – Aug 1** — Orientation closed: a 90° error was traced to the constant
  being fitted on a different tracker channel than the one consumed at runtime.
  Operator signed off on hardware: *"the palm direction is right."* Head and
  waist added.
- **Aug 2 – 3** — Priorities re-stated by the operator (wrist orientation first,
  then the two hands' relative pose, absolute position last). Real-time factor,
  filter dt and head oscillation all fixed. `chest-anchor` mapping adopted.
  Arms + head + waist signed off on hardware.
- **Aug 4 – 6** — Replay-then-confirm flow before touching hardware. Console
  built (one command, zero arguments). V2AP mechanisms ported behind switches.
- **Aug 8 – 9** — V2AP and T-Rex read line by line. Episode reset, environment
  obstacles, unified recording, device detection, operator skeleton. The
  latency mechanism found (velocity feedforward + PID). Limit-pinning
  quantified, nine fixes tried.

---

## 6. Solved

1. **One command, zero arguments.** `vega_console.py`. Part selection, device
   status, real-robot connect, recording and stop are all page controls.
2. **The page shows the mapping, not just the robot.** The operator's SMPL-24
   skeleton is drawn beside the robot with white (palm normal) and yellow
   (finger direction) arrows, plus a text readout. Same computation as
   `viser_body_review.py`, not a second copy.
3. **Chained part selection.** arms → head → waist, expressed by *disabling*
   rather than auto-checking. Sharpa hand independent.
4. **Device detection that occupies nothing.** Vega via its own mDNS broadcast
   (the same channel `dexcontrol` uses, so it also survives the robot's IP
   changing); Sharpa via ping (they are **network** devices, not USB, left/right
   by IP); camera by device count, serial only when no capture process holds it.
   The connect buttons now refuse to start when the device isn't there.
5. **Unified recording.** One directory per take under `data/sessions/<name>/`,
   named by the operator or automatically from start/end time, broadcast to
   every recorder. Six streams: arm commands, Sharpa joints + tactile, real
   robot joints, raw glove skeleton, PICO, point cloud. Three of those had no
   recorder at all before. `merge_episode.py` merges them offline.
   - Timebase: **host wall clock everywhere.** The sim stream's `t_us` is
     *simulation* time (measured at 2.02× wall clock, and the ratio varies), so
     a `t_wall_us` field was added; aligning on `t_us` would be silently wrong.
   - Storage chosen by measurement: tactile as inter-frame delta + keyframes
     (−49 % disk, bit-exact, random access preserved); point cloud as int16
     millimetres uncompressed (1.0 ms/frame vs 61 ms for compressed float32,
     and smaller — millimetres are the sensor's native unit).
6. **Real-robot bridge is instrumented and its unit bug is fixed.** Rate limits
   are now deg/s computed from *measured* elapsed time (the loop runs at 98 Hz,
   not the nominal 100, so the old per-tick limit was 147 deg/s, not 150). It
   prints measured loop rate, rate-limiter saturation, and commanded-vs-measured
   lag every second — in live mode, which it never did before.
7. **An episode concept.** "Start a new take" walks the robot back to the
   default pose, resets the solver configuration, and re-anchors — so a bad
   posture cannot be inherited across takes. Ported from the reference
   implementations (§7).
8. **Everything above is checkable without hardware.** `check_all.py`, 13/13.

---

## 7. What the reference implementations do (and what we took)

`~/dexmate/V2AP-demo` and `~/magicsim/T-Rex-main` are another group's teleop
stack for **the same robot and the same hands**. T-Rex is the published version;
their teleop control layer is byte-for-byte the same as V2AP's, and it is the
recipe they collected 100 hours of data with.

**Their IK adds no constraints.** `SelfCollisionBarrier` is imported and never
used (comment: it slows Pink down); there is no constraint on the redundant
elbow DoF. What they do instead:

| | Them | Us |
|---|---|---|
| FrameTask gain | 0.2 (a first-order low-pass inside the solver) | 1.0 |
| position : orientation | 50 : 1 | 16 : 1 |
| Smoothness posture task (target = last solution) | cost 0.2 | implemented, off |
| Output velocity clamp | **0.4 × hardware limit, always on** (≈55–62 deg/s) | implemented, off |
| Every episode | **OMPL replans the robot back to a fixed default joint pose**, then rebuilds the IK seed *and* the mapping anchor from it | nothing (until now) |
| Torso / head | locked for the whole session | teleoperated (our requirement) |
| Filters | none at all | two one-euro stages |
| Operator | watches a screen, no headset | wears the headset |

Taken and verified: the **episode reset**, **environment obstacles** in the
collision model (with T-Rex's "must not collide at the default pose" self-check),
**tactile-as-compressed-media** (we chose delta+gzip over their lossless video),
and **per-take recording segmentation**.

Not taken: their **rigid CNC wrist mount** (a machined part — their
tracker-to-hand transform is a designed constant; ours is "however the
controller got strapped to the back of the hand this time", which changes every
wearing), and **wrist cameras** (we have none).

**One number worth remembering:** their production system deliberately clamps
the robot to **0.4× its hardware limit**. "Make the robot keep up with the
operator" is not a problem they try to solve — they slow the operator down.

---

## 8. Unsolved

### 8.1 Joints twist into limits on large motions — **wrist mechanism RESOLVED 2026-08-10** (see end of section)

Measured on the standard clip: **14.15 % of frames have at least one arm joint
within 1° of a limit, with the longest continuous stretch 5.2 seconds.** The
joints that pin are the **wrist** ones (`*_arm_j6`, ±80°; `*_arm_j7`, −79…64°),
not the null-space triple — so this is **orientation reachability**, not IK
branch history. A human forearm rolls 150–180°; the robot's j6 has ±80°.

Nine approaches, measured on the same material (`sim/dev_limit_lock.py`):

| Approach | Pinned frames | Verdict |
|---|---|---|
| baseline | 14.15 % | — |
| re-seed solver at engage | 14.18 % | no effect |
| smoothness posture task | 14.24 % | no effect |
| episode homing every 15 s | 13.97 % | no effect |
| solver gain 0.2 (copied from them) | 21.40 % | worse |
| orientation scaling (anchor-referenced) | 71.24 % | much worse |
| orientation scaling (home-referenced) | 11.31 % | worse than baseline |
| incremental orientation (`--ori-mode track`) | 96.98 % | much worse |
| **relax orientation weight only when saturated** | **2.45 %** | **works — but see below** |

The pattern is itself the finding: **changing solver history does nothing;
continuously changing the orientation demand is worse; only relaxing at the
moment of saturation helps.**

**That fix initially appeared to fail under machine load** (81.25 % pinned with
four cores busy, vs 14.15→14.20 % without the switch) and was retracted for a
day. **On 2026-08-10 the real cause was found, and it was not load.** The
trigger read the minimum limit-distance over **all seven** joints of the arm,
while the failure it treats only involves the wrist pair. That creates a latch:
any perturbation that briefly pins a *shoulder* joint (j2) engages the
relaxation → the arm, freed of its orientation constraint, wanders while the
3-DoF position task is still satisfied → the shoulder stays pinned → the
relaxation never releases. In every bad run the dominant pinned joint was j2
(4275/4279 frames, 35 s continuous), and the wrist error doubled on exactly the
latched arm; one 65 %-pinned run happened at real-time factor 2.37, i.e. with
essentially no load. Load was only the perturbation that kicked the bistable
system into its absorbing state — which is why EMA-smoothing the trigger (a
single-frame-noise remedy) never helped.

**The fix is three parts** (`sim/pink_vega_ik.py::_relax_pinned`): the trigger
reads **j6/j7 only**; the relaxation curve gets a **2° floor** (fully relaxed
below 2°, so the equilibrium sits *outside* the 1° pinned criterion instead of
grinding the wall); and while saturated the orientation **target is slerped
toward the current orientation** (at full relaxation the task produces zero push
regardless of how far out of range the demand is — weight scaling alone loses to
a 150° error even at 1 % weight). A five-assertion mechanism regression,
`sim/dev_relax_latch.py`, was written to fail on the old code first and is now
part of `check_all.py` (14 checks).

Verified idle **and** under four busy cores, twice (playlist_all13): pinned
frames **7.00 % idle / 7.23 % / 6.92 % loaded** (old code: 2.45 % idle,
47–81 % perturbed), wrist pinned joint-frames down 95 % (1192→57; the residue
is sub-0.2 s transient touches, no lock-in), wrist error at baseline
(17.7/12.8 mm), jumps 3.56→3.67 % (within noise). On the heavier
clip_headwaist material the wrist reduction is 78 % (1787→396), and 94 % of the
residue co-occurs with same-arm shoulder/elbow pinning — that is **position
saturation** (the weight-8 position task recruiting wrist joints for a few
extra millimetres of reach), a separate mechanism being addressed by
`--relax-pos-at-limit` (see below). The remaining ~7 % on playlist is
shoulder/elbow reach saturation, bit-for-bit equal to the baseline's
shoulder/elbow share (756→753 joint-frames), so the fix introduced no
wandering. The guard combo is back on by
default in the console (`--self-collision hold --relax-at-limit 0.01
--relax-margin 3.0`). Wearing sign-off on feel is still pending.

The physical gap remains: a human forearm rolls 150–180°, `*_arm_j6` has ±80°.
The reference group, on the same robot, works around it rather than solving it
(torso locked, every take starts mid-range, operator clamped to 0.4× hardware
speed); with the fix, saturation now degrades gracefully (hold 2° off the wall,
recover instantly) instead of latching.

**Position-side counterpart, same day** (`--relax-pos-at-limit`, **off by
default, pending wearing sign-off**): when a *shoulder* joint (j1–j3)
saturates — the operator reaching beyond the workspace — the position target is
slerped toward the current EE and the position weight drops 8→0.08 (target =
current leaves no wander freedom, so no latch; the pinned joint is pinned by
the very task being relaxed, so it self-releases). Two measured missteps on the
way: blending without the weight drop freezes the arm *on* the wall (a weight-8
"stay here" task is a stiff damper posture cannot beat), and triggering on the
elbow j4 sacrifices tracking during normal tucked poses (left wrist error
+47 %). Final numbers (idle / 4-core loaded identical): playlist 7.00→3.53 %
pinned with longest run 4.9→0.7 s; clip_headwaist 45.94→11.44 %, longest
4.6→0.8 s — every multi-second grind on both materials is gone. Cost: left
wrist +1.1 mm, jumps +1.1 pp — hence off by default until the operator judges
feel. Mechanism assertions live in `sim/dev_relax_latch.py` (8 total).

**The reference design was then tested in full, and it does not transfer.**
Their approach is incremental orientation anchored at the default pose
(`--ori-mode track --anchor home`) plus a physical reset each take. All three
parts exist here now. Measured on the same material, with each candidate
explanation controlled in turn:

| Condition | incremental | absolute |
|---|---|---|
| anchor = EE at engage | 96.98 % | 14.15 % |
| anchor = default pose (**their design**) | 97.52 % | — |
| short clip (32 s) | 79.62 % | 60.44 % |
| torso locked | 96.66 % | 14.17 % |

Incremental is 80–97 % under every condition; absolute is 14–60 %. Wrist error
also goes from 18/13 mm to 50/47 mm. **The same mathematical form that works for
them diverges here.**

The most likely reason — and it needs new information, not more parameter
sweeps — is that **the tracker data is not comparable**. They use a Vive tracker
per wrist, an independent 6-DoF measurement. We use PICO's body model: measured
2026-07-31, the same held pose 15 seconds apart differs by **3.8–18.5° in wrist
orientation**, and forearm length has a CV of 11.8 % (the wrist is the only
joint in the chain not constrained by the skeleton). **Incremental mapping
integrates that error frame over frame; absolute mapping is independent each
frame and does not accumulate.**

Both modes are selectable — `--wrist-map {absolute,incremental}`, or the
"腕部朝向" dropdown in the console. The numbers above are in the flag's help
text and in the SOP so nobody re-derives them. Absolute remains the default.

### 8.2 Latency — mechanism found, not verified on hardware

- `dexcontrol` exposes **`set_joint_pos_vel()`** (position + velocity
  feedforward) and **`get_pid()` / `set_pid()`** (arm position-loop P gain,
  0.1–4×). Neither had ever been used. A position loop tracking a moving target
  lags in proportion to velocity ÷ Kp — which is exactly "fine when slow, can't
  keep up when fast". The bridge now uses feedforward by default and prints the
  factory P multipliers at startup. **Never tested on hardware.**
- The consumer's own lag was measured for the first time: **80 ms**, from
  two one-euro stages. Relaxing them to 4.0 / 15.0 Hz recovers most of it at a
  cost inside the noise floor (jitter 3.56 % → 3.59 %). **Defaults unchanged** —
  the operator reported "the palm wobbles a lot" once before when filtering was
  effectively disabled by a dt bug, so this needs a wearing test, not a
  measurement.
- Structural, and worth knowing: **the mirror shows Isaac's achieved joint
  angles, and the USD gives Isaac a velocity limit 57× the real robot's.** "The
  sim looks right" is therefore not evidence about the robot.

### 8.3 Requires hardware

1. First real-robot drive: read `get_pid()`, enable feedforward, watch the new
   "robot lag" readout, only then consider `--arm-pid-scale`.
2. **Clap test.** All streams share a wall clock, so the *timestamps* align — but
   the acquisition delays differ, and aligned timestamps are not aligned motion.
   `check_clap_sync.py` is ready and self-checked against synthetic data;
   it has never seen a real recording. **Until it has, merged episodes must not
   be treated as synchronised data.**
3. The device panel's "detected" branch has only ever run in the negative (the
   wired NIC is down on this machine).
4. Every new switch needs a wearing test to settle its default.
5. `--hands both` (both hands in one process): wiring verified with mock gloves,
   never run with real ones.

### 8.4 Known gaps

- The Sharpa hands are **not in the collision model** (the URDF ends at `l8`).
- Point cloud is in the depth camera's optical frame; **camera↔robot extrinsics
  have not been done**, and the depth values have never been checked against a
  tape measure.
- The waist gain 0.7 was back-computed from one piece of verbal feedback, not
  fitted.
- ~90 files are still untracked in git, pending review.

---

## 9. Things this repo has been burned by

Written down because they recur.

1. **"It runs" ≠ "it did anything."** Nine separate times a feature was wired up
   and never executed, always with exit code 0. Every switch must carry a
   readout that says how many times it fired, and print `<-- NEVER FIRED` when
   it didn't.
2. **Verify the ruler before quoting the number.** Multiple conclusions have come
   from broken measurements. Recent examples: a limit-pinning comparison read off
   a CSV that was *still being written* (the tool now refuses files touched in
   the last 10 s); the clap-sync ruler had a resolution of 50 ms while its own
   pass threshold was 20 ms (fixed with sub-sample interpolation).
3. **Bench conditions must match use conditions.** The `--relax-at-limit` default
   was set from an idle-machine measurement and reverted the same day when a
   loaded run showed 81 % instead of 2.45 %.
4. **One quantity, one implementation.** The default pose used to exist in two
   files, each with a "MUST stay in sync" comment.
5. **Never SIGINT a long-running Isaac.** It corrupts `nvidia_uvm` (`nvidia-smi`
   looks fine, CUDA reports unknown error). Six occurrences. The console now
   broadcasts a shutdown request and waits; recovery is
   `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm` (both halves need sudo).
6. **`pkill -f <pattern>` matches your own shell** and kills it (exit 144). Use
   `ps -eo pid,comm,args | awk '$2 ~ /^python/ && /name/ {print $1}'`.
7. **Read the reference before theorising.** The hand-side log and `SOP.md`
   already contained the answer more than once while time was spent guessing.
