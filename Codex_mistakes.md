# Durable lessons

## GPU 空闲检测必须要求稳定窗口，自动录像不得绕过 guard

- 背景/现象：Deep20 transition collector 启动后，检查发现它与 GPU0 上其他用户的 Isaac 作业短暂重叠。
- 错误假设或操作：一次性查询 compute app 被当成可占用证明；同时旧 Sweep 自动录像脚本显式设置 `RL_ISAAC_NO_GUARD=1`。
- 根因：外部作业的 CUDA context 可在采样瞬间变化，一次查询存在竞态；录像脚本没有实现训练侧 pause/yield 协议。
- 修正：立即只停止本任务 collector；默认启动脚本改为 GPU0 连续 60 秒无 compute app 且 utilization 不高于 5% 才继续；随后用户明确确认有权共享 GPU，才用显式 `SWEEP_ALLOW_SHARED_GPU=1` 重新启动。训练在 epoch 边界调用 `yield_if_paused()`，录像不再绕过 guard。
- 预防/检查方法：远端流水线启动后必须再次核对 `nvidia-smi` 的 PID/owner；未获得资源所有者明确共享授权时必须稳定等待。即使获准共享，也不得停止、降优先级或操作外部 PID，并仍要保证本任务训练和录像不重叠。

## Bootstrap must account for client filesystem and rsync versions

- Context/symptom: The first transfer from the authorized macOS clone did not produce a clean Linux worktree.
- Incorrect assumption or action: The transfer command assumed `rsync --protect-args` was available and that a case-insensitive macOS checkout could contain both `tasks/Pour` and `tasks/pour`.
- Root cause: The bundled macOS rsync is older and the repository contains paths that differ only by case; the Linux target correctly reported the missing lowercase files.
- Correction: Use portable rsync arguments, reconstruct case-colliding tracked paths from the Git object database on Linux, and verify `git status` on the destination before editing.
- Prevention/check: Treat the destination Linux `git status --short --branch` as a mandatory bootstrap gate and never use a macOS working tree as the sole completeness check for a case-sensitive repository.

## Interrupting SSH did not terminate the remote CPU diagnostic

- Context/symptom: A long Sweep reference audit was interrupted from the local SSH
  client, then a revised audit was launched. Process inspection showed both exact
  `build_reference.py --world_yaw_deg -80 --audit_side right` jobs still running.
- Incorrect assumption or action: I assumed sending Ctrl-C to the local non-PTY SSH
  session had also terminated the remote Python child.
- Root cause: The SSH frontend exited without propagating termination to the remote
  process, leaving the first CPU-only diagnostic orphaned.
- Correction: Resolved the two task-owned PIDs and terminated only those exact PIDs;
  verified both were gone and did not touch any foreign process.
- Prevention/check: After interrupting a remote long-running command, always inspect
  the exact command line and ownership before relaunching. Use task-owned tmux for
  persistent work and exact PID targeting for cleanup; never use broad process kills.

## Geometry optimization replaced the reconstructed Sweep task

- Context/symptom: The first zero-residual Sweep video had both hands visibly
  intersecting their tools and the broom in an orientation absent from both the ego
  video and the accepted Dexonomy replay.
- Incorrect assumption or action: I optimized IK reachability and cube-path geometry
  by adding a 90-degree broom yaw, changing the global world yaw, suppressing most
  reconstructed rotation, smoothing translation, shifting the pan, and finally
  freezing the pan.
- Root cause: Reset-frame and geometry assertions were treated as sufficient gates,
  while source-video/GraspPose visual fidelity was not made a prerequisite. The
  resulting checks proved only that a self-consistent altered scene could reset.
- Correction: Reject the generated reference/video and rebuild from the Pour rule:
  object 6DoF trajectory drives GraspPose-locked wrist IK; human wrists contribute
  aligned motion shape only at reduced object confidence. Permit only a small,
  explicit shared initial scene registration.
- Prevention/check: Before physical smoke or reward tuning, compare frame zero and
  several motion frames against the ego video, the selected GraspPose render, and the
  accepted task replay. Any per-tool yaw, frozen object track, or reduced trajectory
  component requires explicit user approval and a recorded falsification test.

## Long CPU diagnostics must write durable output

- Context/symptom: A multi-start raw-trajectory IK audit outlived its non-interactive
  SSH client, and its buffered stdout was lost even though the exact task-owned child
  completed normally.
- Incorrect assumption or action: I treated a many-row, multi-seed IK sweep as a
  short foreground diagnostic.
- Root cause: Runtime was underestimated and stdout was not redirected to the
  project log tree before launch.
- Correction: Discard the result as non-auditable; rerun with a unique task-owned
  tmux session and a log under `logs/reference/`.
- Prevention/check: Any remote diagnostic expected to exceed one minute must have a
  durable log and named task-owned session before launch, even when it is CPU-only.

## Fixed-attached tasks must preserve their reconstructed object pose

- Context/symptom: The corrected Sweep reference passed offline IK, but physical
  initialization reported 27.8 cm grasp IK error before the Sweep reset ran.
- Incorrect assumption or action: The fixed-attached Sweep config still allowed the
  inherited pregrasp loader to replace the task-provided first-frame object pose
  with a generic Dexonomy canonical resting pose.
- Root cause: The parent loader was designed for approach/grasp tasks where it owns
  object placement; fixed-attached trajectory tasks already own that placement.
- Correction: When `fixed_attached_tools` is set, retain `object_cfg.init_state` and
  use GraspPose only as the physical object-hand transform. The resulting IK error
  is 0.07 cm without weakening the 2 cm reachability assertion.
- Prevention/check: Physical smoke logs must show the explicit reconstructed-pose
  preservation message and a passing grasp IK audit before attachment assertions.
## Never transform source OBJ vertices as if they were USD rigid-root vertices

- Context/symptom: offline analysis first reported the dustpan 19.6 mm above the
  table, while a runtime probe later reported penetration.
- Incorrect assumption or action: source OBJ vertices were multiplied directly by
  the simulated `Aux` rigid-root pose.
- Root cause: MeshConverter inserts an internal transform below the USD rigid root;
  OBJ local coordinates and rigid-root local coordinates are not identical.
- Correction: recover mesh points from the live USD stage using the complete
  `mesh local -> world -> Aux root` transform chain before measuring placement.
- Prevention/check: every mesh-to-world physical assertion must name and verify its
  coordinate owner; use the live USD hierarchy for converted assets.

## Runtime task clocks must consume the reference's measured contact row

- Context/symptom: physical traces reached Gate 2 around step 50--60, but the task
  and experts treated rows 300--425 as nominal contact.
- Incorrect assumption or action: `SweepEnv` replaced NPZ `contact_row=54` with a
  hand-authored constant 425 and rebuilt the cube start from that row.
- Root cause: a later geometry experiment was allowed to override authoritative
  reference metadata without a consistency assertion.
- Correction: load `contact_row` and `cube_start_w` directly from the reference NPZ.
- Prevention/check: assert that runtime phase boundaries originate from serialized
  reference metadata; never duplicate them as constants in environment wiring.

## Probe success must be captured before vector-environment auto reset

- Context/symptom: the v7 dustpan probe showed the cube inside and stable in the
  video/trace, but saved `entry_ok=False` from a later outside state.
- Incorrect assumption or action: the probe ignored `done`, continued stepping,
  and evaluated only the final raw simulator state.
- Root cause: the environment wrapper automatically reset immediately after Gate4;
  the final raw state belonged to the next episode, not the successful rollout.
- Correction: read the environment tick's pre-reset `success`, gates, cube-in-pan,
  and stable-run values on every step; stop immediately on termination.
- Prevention/check: every physical probe and expert must pair state with the same
  transition and treat any post-reset state as a new episode.

## World-table height must not overwrite a pan-frame placement component

- Context/symptom: the row313 v8 probe requested pan-local z=125 mm but measured
  about 191 mm after the first transition, and the low-speed cube never reached the
  mouth.
- Incorrect assumption or action: code transformed a pan-local point to world,
  then overwrote its world z coordinate to put the cube on the table.
- Root cause: the dustpan is tilted, so changing world z also changes pan-local y
  and z; the requested pan-local mouth coordinate was destroyed.
- Correction: keep pan-local x/z fixed and solve pan-local y from the desired world
  table height using the pan's world up-axis component.
- Prevention/check: every fixed start must assert its measured pan-local x/z after
  the first physics transition before interpreting any contact result.

## Collision clearance must use the oriented footprint

- Context/symptom: after fixing the coordinate transform, v9 started at the exact
  requested pan-local z=125 mm but immediately reached 2.64 m/s and flew outward.
- Incorrect assumption or action: clearance was estimated from the scalar 12.5 mm
  cube half-size as if the cube axes and tilted pan axes were aligned.
- Root cause: the world-axis cube projects about 18.3 mm onto pan-local z, so its
  inner face initially overlapped the ramp front by several millimetres.
- Correction: place the fixed centre at z=140 mm, beyond the complete projected
  footprint, before applying the low-speed entry command.
- Prevention/check: calculate or measure oriented bounding-box separation in the
  collision owner's frame and assert a positive first-frame gap before interpreting
  forces, friction, or controller quality.

## A global mesh minimum does not prove the entire pan mouth is grounded

- Context/symptom: v10 had a measured 1.03 mm pan-mesh/table minimum and a stable
  basin, yet the cube stopped at the first entrance contact.
- Incorrect assumption or action: vertical translation was treated as sufficient
  table alignment while retaining the reconstructed pan's roll and pitch.
- Root cause: only one low mesh corner was near the table; the working mouth and
  central collision ramp remained raised, and the short ramp exposed a vertical
  leading face above the cube's support plane.
- Correction: preserve source yaw but align the work-plane normal to world up, then
  connect the measured local mesh underside to the basin with a longer ramp.
- Prevention/check: report mouth-edge height across its full width and continuity
  to the basin; never substitute a single global-minimum gap for that assertion.

## Working-pose colliders must not be active during an incompatible spawn pose

- Context/symptom: v11 stopped during environment construction with a 16.36 mm
  left-tool registration shift before the level-pan probe ran.
- Incorrect assumption or action: the grounded ramp was enabled immediately even
  though generic environment construction still begins at tilted source row zero.
- Root cause: a collider designed for the later table-aligned expert pose contacted
  the table during the transient spawn and pushed the FixedJoint/arm system.
- Correction: spawn that ramp disabled and enable it only after reaching and
  settling the table-aligned task pose.
- Prevention/check: validate collision-free initialization separately from the
  working pose, and explicitly stage pose-specific collision geometry activation.

## Deterministic scene registration is not a policy residual

- Context/symptom: v12 found a precise, joint-valid level-pan GraspPose IK but
  rejected it at 1.238 times the old left-arm residual envelope.
- Incorrect assumption or action: the probe represented the fixed task setup as a
  residual from the reconstructed row and applied actor exploration limits to it.
- Root cause: feed-forward task-reference construction and learnable correction
  were conflated.
- Correction: install the level-pan IK as the fixed left-arm task reference and
  retain zero actor residual; record the source delta only as provenance.
- Prevention/check: every offset must declare whether it belongs to registration,
  feed-forward reference, or policy action before any bound is applied.

## A thin ramp can still expose an unclimbable rigid leading face

- Context/symptom: v13 passed pan leveling and basin support, but the outside cube
  stopped at z=138.66 mm against a nominally 1 mm-thick entry ramp.
- Incorrect assumption or action: reducing the ramp thickness was treated as
  equivalent to making its surface continuous with the table.
- Root cause: any above-table box front remains a sharp vertical collision edge;
  low-speed rigid contact need not climb it even when the edge is visually tiny.
- Correction: extend the ramp surface below the table, filter only the ramp--table
  collision pair, and keep its front farther than the combined contact offsets from
  the table-supported cube.
- Prevention/check: verify the height field and contact-offset envelope seen along
  the cube approach path, not just collider thickness or visual appearance.

## Do not interpret a rollout endpoint without checking its slope and reachability

- Context/symptom: v15 ended at z=151.11 mm and was initially described as another
  ramp stop.
- Incorrect assumption or action: the final state was interpreted without checking
  per-step displacement or whether the rollout had reached the ramp at all.
- Root cause: the nominal velocity is written once per environment step and table
  friction acts during substeps; effective progress was only 0.0435 mm/step.
- Correction: inspect the full trace derivative, calculate the required horizon,
  and make pan-only Gate1/2 explicit preconditions when the broom is disabled.
- Prevention/check: before diagnosing contact, prove that the tested geometry was
  reached and that every prerequisite gate is attainable in the isolated setup.

## A once-per-control-step velocity write is not a persistent contact push

- Context/symptom: v16 climbed the ramp to y=9.1 mm and then remained near
  z=104.6 mm despite rewriting a nominal inward velocity each environment step.
- Incorrect assumption or action: root velocity assignment was treated as an
  actuator that applies through all physics substeps.
- Root cause: ramp normal and friction cancel the assigned velocity after the first
  substep; no sustained force remains for the rest of the control interval.
- Correction: use a bounded persistent force for the isolated topology probe and
  clear it immediately on containment.
- Prevention/check: specify whether a diagnostic command is state assignment,
  impulse, force, or closed-loop actuator, and validate its substep persistence.

## Pan-only probes must not be presented as expert-trajectory progress

- Context/symptom: the user observed that v16 showed a static right hand and a cube
  being reset directly into the pan, unlike the requested expert sweep.
- Incorrect assumption or action: topology validation was allowed to dominate the
  iteration sequence and its videos were discussed alongside expert progress.
- Root cause: a useful internal gate (support/ramp debugging) was confused with the
  user-facing deliverable (continuous reconstructed right-hand sweep).
- Correction: stop v17, return to the near-success
  `closed_loop_cube_pan_servo_v1` controller, and use pan-probe results only as scene
  setup underneath a real broom rollout.
- Prevention/check: every reported video must state whether the cube is reset and
  whether the broom collision/controller is active; only continuous Gate1--4 broom
  rollouts may be called expert candidates.

## Preserve a collision-free interval for the source outward preparation

- Context/symptom: expert v3 immediately earned proximity and moved the cube outward
  from z=160 to roughly 254 mm during rows300--313.
- Incorrect assumption or action: the cube was placed at the ramp-probe start without
  checking clearance from the broom at the start of the selected source segment.
- Root cause: row300's bristles were already within 15.4 mm, so the intended
  no-contact outward setup became an outward push.
- Correction: place the fixed cube at z=195 mm so row313 reaches its outside face
  before the inward segment starts.
- Prevention/check: assert broom--cube clearance at the first expert row and contact
  proximity at the inward-sweep boundary as separate preconditions.

## Do not infer 3D bristle clearance from a scalar pan-z offset

- Context/symptom: moving the cube from z=160 to 195 mm reduced actual row300
  broom distance from 15.4 to 12.5 mm and produced a roughly 9 m/s launch.
- Incorrect assumption or action: clearance was estimated from the nominal sweep
  direction without evaluating the oriented, extended bristle work face.
- Root cause: the live bristle cloud spans x/y/z; another point became the nearest
  collider after the z-only move.
- Correction: search the full transformed point cloud at both row300 and row313 and
  assert the live reset distance before stepping physics.
- Prevention/check: use minimum distance to the collision-owned work face for all
  initial placement decisions.

## Every expert attempt needs a unique failure-artifact tag

- Context/symptom: v4 saved under the v3 controller tag and overwrote v3's failed
  NPZ/video.
- Incorrect assumption or action: the failure filename was tied only to controller
  family, not attempt version/configuration.
- Root cause: output identity omitted task geometry and run version.
- Correction: v5 and later attempts use unique versioned tags; the v3 numeric result
  remains in its durable run log, but its overwritten video is not recoverable
  without a rerun.
- Prevention/check: assert destination nonexistence or include the run version in
  every log, NPZ, checkpoint, and video basename before launch.

## Use the correct transformed axis when auditing a tool quaternion

- Context/symptom: an initial read of the v1 trace described the dustpan as nearly
  vertical (95--103 degrees) even though the frames and row313 probe showed a mostly
  level work plane.
- Incorrect assumption or action: the world-y component of transformed local +y
  was used as though it were its world-z component.
- Root cause: for scalar-first `wxyz`, local +y's world-z component is matrix entry
  `R[2,1] = 2*(y*z + w*x)`, not `R[1,1] = 1-2*(x*x + z*z)`.
- Correction: recompute the complete rotation matrix and compare the transformed
  pan-local +y axis with world +z.  The v1 closest frame is about 8.1 degrees from
  level; its real defect is vertical placement of the mouth/floor.
- Prevention/check: print the full transformed basis and cross-check it against a
  known probe pose before changing any IK target from a quaternion-derived angle.

## Never enable grounded task collision before a changed GraspPose settles

- Context/symptom: v6 passed a static 23.07 mm broom/cube clearance check but the
  fixed cube launched outward at 1.97 m/s in the first five transitions.
- Incorrect assumption or action: the grounded dustpan ramp was enabled while reset
  was still reconciling the source spawn, new fixed-pan IK, arm PD, and FixedJoint.
- Root cause: centre-distance checks covered broom geometry only.  The moving ramp
  swept through the nearby cube before the right-arm controller emitted any action.
- Correction: park the cube, settle the attached tools with the ramp disabled,
  enable and settle the ramp, assert pan pose/speed, then install the fixed task
  cube before recorded frame0.
- Prevention/check: every task-specific collider enabled after spawn needs a
  zero-action dynamic reset gate; static separation from the active tool is not
  sufficient evidence of a collision-free first transition.

## Row-specific geometry diagnostics must index the same row

- Context/symptom: the expert report labelled a value `row313_broom_outside_z_mm`,
  but the coordinate came from preparation row301 using a vertex index selected on
  row313.
- Incorrect assumption or action: `points_pan[1][idx13]` was used after computing
  `idx13` from `points_pan[-1]`.
- Root cause: the row index and vertex index were separated across two statements
  and the preparation index survived a prior diagnostic rewrite.
- Correction: compute the coordinate with `points_pan[-1][idx13]`.
- Prevention/check: derive every row-labelled diagnostic from one explicitly named
  row array and include a static sanity print for the source row number.

## Keep an immutable behavioral A/B baseline during controller iteration

- Context/symptom: successive v10--v15 expert attempts changed contact-point
  selection, source-path lateral motion, cube placement, and feedback timing while
  trying to fix a near-successful v1 rollout.
- Incorrect assumption or action: each failed controller became the starting point
  for the next one, instead of replaying v1 unchanged and isolating the dustpan fix.
- Root cause: useful per-attempt diagnostics were treated as justification for
  cumulative redesign even though v1's best frame still used zero residual.
- Correction: stop v15, freeze the archived v1 NPZ/video as the right-hand source of
  truth, and permit only a one-variable low-pan/entry A/B replay next.
- Prevention/check: before accepting a controller change, replay the immutable
  baseline in the same scene and require quantitative plus visual improvement with
  every non-target variable held constant.
## Repair visible task-asset defects in the asset, not with an overlapping proxy

- Context/symptom: the v1 dustpan visibly had a raised mouth lip, but the first
  attempted A/B added and enabled an invisible smooth ramp around the unchanged
  mesh.  It kicked the attached pan by 41.16 mm on frame1 and ejected the cube.
- Incorrect assumption/action: treated a reconstructed mesh defect as only a
  collision-topology problem, then tried root/child table filtering and delayed
  activation without first repairing the visible source geometry.
- Root cause: the original central mouth vertices rise from an 8--9 mm basin top
  to 14.06 mm.  The added proxy also overlapped the initial v1 broom/tool geometry;
  filtering table descendants did not change the failure trace.
- Correction: removed both failed videos and superseded scripts, reverted the
  ineffective filter change, generated a watertight task-owned asset whose mouth
  continuously falls to 6.85 mm, and reran the immutable v1 actions/rows.
- Prevention/check: when a user identifies an asset-level visual defect, first
  measure and render the asset cross-section, preserve its topology and grasp
  frame, create a non-destructive task copy, and validate with a behavioral A/B
  whose command tensors are hash-locked.

## Do not treat a repeatable early contact as a disposable reset transient

- Context/symptom: the physical-entry v1 replay visibly moved the cube during its
  first frames, so a parked-cube/tool-settle pre-roll was tried before frame0.
- Incorrect assumption/action: assumed the early motion could be removed without
  changing the later near-success state because v1 actions and rows stayed exact.
- Root cause: the disabled-entry and enabled-entry traces had identical cube world
  motion for the first 50 frames; the source broom motion, not ramp activation,
  produces the early contact and establishes the cube state used later.
- Correction: reject and revert the pre-roll version after it reduced early motion
  only partially and worsened best containment from 19.06 to 38.20 mm.
- Prevention/check: before suppressing an apparent initialization transient, A/B
  its full state trajectory against the best rollout and verify it is not a causal
  task contact.  Preserve post-contact state equivalence, not only command hashes.

## Commanded task-space depth is not equivalent to cube penetration

- Context/symptom: a 25 mm right-hand inward extension left only 4.63 mm of strict
  containment deficit, suggesting a larger 40 mm command should finish insertion.
- Incorrect assumption/action: extrapolated cube motion linearly from commanded
  Cartesian tool displacement after contact.
- Root cause: at the pan mouth the brush/cube contact is marginal; increasing the
  command from 25 to 40 mm changed IK/residual motion but did not preserve an
  effective inward contact normal.  Best deficit slightly worsened to 4.78 mm.
- Correction: stop increasing depth amplitude and inspect lateral/work-face contact
  alignment around frames385--395 as the next isolated variable.
- Prevention/check: after every task-space amplitude A/B, compare both command and
  object response.  Treat a flat/non-monotonic object response as contact loss,
  not evidence that still more command is needed.

## Artifact arguments must include their approved project-root prefix

- Context/symptom: the first formal-expert tmux attempt failed before simulation
  because `--baseline`, `--trace`, and `--video` were passed as subpaths beneath
  their approved roots, while the script validates root-relative paths.
- Root cause: treated the arguments as relative to `logs/`/`outputs_video/`, but
  `under()` resolves them relative to the project root and then checks containment.
- Correction: waited for both short-lived failed processes to exit, then relaunched
  serially with `logs/...` and `outputs_video/...`; no physical rollout or formal
  artifact had been produced by the failed attempt.
- Prevention/check: inspect each artifact path constructor before launch and run a
  cheap path-resolution check before paying Isaac startup cost.

## 2026-08-30 — Transition recollection corrections

- First collector attempt used an internal signal key name that differed from
  the environment (`pan_clearance` versus `mouth_clearance`).  The strict schema
  gate stopped before writing a dataset; the collector now uses the canonical key.
- The first physical expert recollection silently inherited a later-edited cube
  start from the reference NPZ and therefore did not reproduce the accepted
  expert.  The collector now restores and pins the archived v1 easy-start cube
  pose from the canonical action source before reset.
- Isaac Sim 5.1 plugin teardown hung after durable transition/training artifacts
  were complete.  Task-owned processes were terminated by exact PID only, and
  training now exits at the established clean post-checkpoint process boundary.

## 2026-08-31 — 3M 诊断钩子缺少运行时导入

- 现象：训练在 3,014,656 steps 首次跨过 3M 时，已经保存
  `diag_0003M.pth`，随后在构造诊断 JSON 的 `torch.stack()` 处触发 NameError 并退出。
- 根因：静态 `py_compile` 无法发现函数运行到特定 3M 分支时才解析的未定义全局名；
  `train_sweep.py` 使用了 `torch` 却没有 import。
- 修复：显式导入 `torch`，补齐当次 JSON/video/trace，并从精确 3M checkpoint 恢复。
  checkpoint 以后同时存 optimizer 与 step/epoch/lr，避免旧格式恢复丢失训练状态。
- 预防：所有稀疏触发的诊断分支必须在小步 smoke 中用降低阈值强制执行一次；不能把
  “模块可编译”当成“定时分支已验证”。

## Terminal 后渲染会把 reset 状态误当作成功冻结帧

- 背景：15M entry 回归在 terminal transition step232 成功，但 DirectRLEnv 在 env.step 返回前已自动 reset。
- 症状：录像器随后 render，生成的 frame_0232 是 reset 画面；正确最后有效图像是 frame_0231。
- 修正：检测 done 后保存 transition，但跳过 post-reset render，以最后一个有效 pre-reset RGB 图像追加 40 帧。
- 预防：录像验收同时检查 trace terminal Gate、topdown 最大编号和 MP4 物理帧数；不能只看 success_step 日志。

## 只看最后一张已渲染帧会误判 terminal 几何

- 现象：3M trace 在 step300 fully_inside，但视频最后只显示 step299 的浅进入。
- 根因：DirectRLEnv 在 done 返回前 reset；上一版为避免 reset 画面直接跳过 terminal render，因此展示的是 terminal action 前状态。
- 修正：录制环境专用 suppress_terminal_reset，保持训练判据不变但允许渲染真实 terminal physics，再按 tick terminal 边界退出。
- 预防：同时核对 terminal cube_pan/fully_inside、topdown terminal 图像与视频冻结源。
