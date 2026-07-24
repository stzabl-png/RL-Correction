# 帧对齐核对表（Reconstruct_and_Retarget · BODex/cuRobo · magicdexmate · IsaacLab）

> 目的：把四套系统的位姿对齐进 **IsaacLab RL env 系**（唯一权威 TARGET）。所有结论来自工作流逐文件核对(2026-07-16)，
> 带 file:line。**动手前用一个 `frames.py` + 002aa185 单测锁死这些变换**（PHASE1 doc 已列为 B1）。

## TL;DR
- **三系统统一**：z-up、米、弧度，存储四元数 **wxyz**（scalar-first）→ **无轴/单位转换**。
- 真正的活：**刚性重锚**（重心化 + re-yaw 到 hand_init_pose + 加 env_origins）+ **几个 xyzw↔wxyz shim** + **帧率重采样(30→15)** + **Reconstruct_and_Retarget 用 Path B 不用 Path A** + **BODex 物体 +90°about-X 对账**。

## 权威 TARGET：IsaacLab RL env 系
- **帧**：每-env 局部 = world − `scene.env_origins`（`env_spacing=0.75`, `replicate_physics=False`）。所有 obs/reward env-相对；写 sim 时加回 env_origins，读时减。
- **锚**：env 内一切锚在**固定 hand-init root pose** = pos `(0,0,0.5)`、rot wxyz `(0.819152,0,-0.5735764,0)`（≈70° about −Y）(`sharpa_wave_env_cfg.py:80,108`)。
- z-up 右手系；米/弧度/千克；四元数 **wxyz**（`quat_to_rotmat` 取 q[:,0..3]=w,x,y,z；`grasp.py:5` 全 WXYZ）。
- **物体 mesh 原点居中**（`sharpa_dataset.py:38`）；`object_pos` env-相对、但 `object_rot` 是**原始 world quat**（`sharpa_wave_env.py:432-433`，clone 不单独旋转才自洽）。
- 手根("wrist")= 浮动手 articulation root：`root_pos_w/root_quat_w`。egocentric wrist 系就是它。
- 22 关节序：articulation `actuated_dof_indices`（cfg:211-234, thumb5/index4/middle4/ring4/pinky5）；数据集原生 `SHARPA_USD_JOINT_NAMES`(index,middle,pinky,ring,thumb) 经 `art_perm` 重排。
- ⚠️ **文档旧数字全错**：物体位置 `[0,0,0.07]`(PHASE1) 和 `[0,0.55,0.80]`(DEXMATE_WORKSPACE) 是**不同帧**，且都 ≠ env 实际 object init `(-0.0956,-0.0052,0.619)`（hand root z=0.5）。**以 env-local 为准，两个 doc 数字作废。**

## Reconstruct_and_Retarget 视频重建 ① —— 有两条输出路径，**必须用 Path B**
- **Path A（ego_pipeline，world_space_res.pth）= 别用**：CoordUnify 固定 OpenCV→Zup 轴交换、假 z-up（假设首帧相机水平）、**物体从没抬进 world（留在相机系）** → 手/物体不一致。
- **Path B（recon_pipeline，`world_fused.npz`/`replay_world.npz`）= 真输入**：ViPE(度量深度+SLAM)→HaWoR(MANO)→FoundationPose(物体6D)→fuse。
- **帧**：导出系 `gravity_z_up_world`：+Z=−重力（真重力，GeoCalib）、+X=中间帧相机朝向投影到 XY、+Y 右手补全。**原点=ViPE gauge（首相机附近，任意大偏移）**；T_gravity 仅旋转不平移。
- 米；四元数：原始重建**无四元数**（物体=4×4矩阵、MANO=axis-angle）；边界处出现且**两处序相反**：HaWoR SLAM handoff = **xyzw**(scalar-last, `pose_convert.py:39`)，replay `obj_pose`=**wxyz**(`recon_to_replay.py:82-87`)。
- 手 30fps vs 物体/视频 15fps（`Th=2*Tv`，重采样）。MANO 侧 index **[0]=左 [1]=右**。
- **→ IsaacLab**：Path B 已是 z-up 右手米，无需轴交换。(1) 物体旋转 xyzw→wxyz；(2) 手根 axis-angle→wxyz + 30→15fps 重采样；(3) **重锚**：world 原点是任意 gauge → 先重心化到锚点(物体质心或 t0 抓取腕位)，`p_isaac=R_yaw@(p_world−p_anchor)+env_origin`，手和物体**同一变换**保 HOI 几何；(4) **re-yaw**：world +X 是相机朝向(任意) → 选 R_z(θ) 把手/物体摆进 env 的 hand_init_pose；(5) 重力已 −Z。
- **陷阱**：① 别喂 Path A（物体留相机系）；② HaWoR world flip `[1,-1,-1]` 只在可视化路径、导出手 globals **没加** → 若假设不成立手会 Y/Z 翻；③ ViPE 负焦距=OpenCV/GL 轴翻，需 `diag(1,-1,-1)`；④ `combine_hands_object.py` 假设手物同相机系，别和 fuse 输出混。

## BODex/cuRobo grasp（Sharpa 右手，浮动手无臂）—— 若保留才需要
- `sharpa_right.yml`：`base_link=ee_link=right_hand_C_MC`，`use_root_pose:True`（自由浮动）。
- **帧**：cuRobo **world/scene 系**（物体按 scene 位姿摆放，**非** robot-base、**非**保证原点居中）。`contact_point`/`contact_frame`/`robot_pose` 都在此系。
- z-up、米、弧度、**wxyz**（`curobo/types/math.py:52`；DOF 布局 `[x,y,z, qw,qx,qy,qz, q0..q21]`，w 在 index 3）。
- `robot_pose` shape `[G,3stage,29]`，stage=[pregrasp粗/grasp精/squeeze终]，**physic_check 用 stage 1**；squeeze 是外推非解算。
- 11 接触：`contact_point[P,3]` **world 系**表面点；`contact_frame[P,3,3]` 列=[法向,切1,切2] world；`contact_force[P,3]`=单位力沿法向、**在 contact_frame 里**（非 world）。
- **浮动手关键**：grasp 合成直接优化手浮动 root 的 world 位姿 → **robot_pose 就是 world 位姿，1:1 映射到 IsaacLab 浮动手 env**（root=robot_pose[:7]，joints=qpos[7:]）。无需 base 变换（那是 dexmate 7-DOF 臂才有）。
- **→ IsaacLab**：已 z-up/米/wxyz。(1) 变 object-相对：`T_obj_hand=inv(P_obj_scene)@P_hand_scene`；(2) IsaacLab 里 `P_hand_world=env_origin ⊕ (P_obj_isaac @ T_obj_hand)` 设浮动 root，joints=stage qpos；(3) 只平移加 env_origin。
- **陷阱**：DGN 网格 Y-up → 物体被 **+90° about-X** 重定向（`physic_check` 物体 orient `[0.7071,0.7071,0,0]`）；IsaacLab 必须用**同一物体朝向**或全部 object-相对，否则手物错位。`data/sharpa_extracted/*/graspxl_final_poses.json` 是**另一条 GraspXL 管线**（object z=0.07 + wrist pose），**别和 BODex .npy 混**。

## magicdexmate 重定向 ①-手指 —— 输出契约
- **输入**：3D 手关键点 `(21,3)` 米，MediaPipe/OpenPose-21 序（只用 wrist+5 指尖 6 点）。MANO 由上游 `hawor_to_joints.py`/`hoi4d_to_replay.py` 过 MANO layer 转成 21 关节。
- **输出**：**仅 22 手指关节角**（弧度，SDK/URDF 序=thumb-first，=GRASPXL 序；`_reorder_named_values` 重排到 SHARPA_USD 序）。**无手腕、无物体**。
- **帧**：手指角**帧不变**（每帧重估 operator 系）；相对 URDF base link `right_hand_C_MC`。
- **手腕6D + 物体**只在离线 Isaac replay(`retarget_isaacsim.py`)拼上：`base_pos(T,3)`、`base_quat(T,4)wxyz`（从 npz `wrist_*` 或 `base_rot_from_joints` 几何推导：+z=腕→MCP质心、+y=index-pinky MCP、+x=掌法向），`obj_pose(T,7)` 来自数据集。
- **⚠️ 欠约束不可行来源**：只有 腕+5指尖 约束 22 DOF → **中节(PIP/DIP)+外展(AA)+CMC 是优化器猜的**。这（连同穿模）就是 RL 要修的。
- **失败证据**（=项目动机）：`retarget_isaacsim.py --mode physics` 把手指当 PD 目标、**手腕 teleport**、物体落下，"能否抓起由接触决定"；`--log-contact` 专门区分 压缩/穿透(overlap)/爆飞 三种失败。

## 一句话行动项
建 `frames.py`：定义 `T_world_recon→isaac`(重心化+re-yaw+env_origin)、`xyzw→wxyz`、`resample_30_15`、（若保留 BODex）`T_bodexscene→isaac`(object-相对) + `+90°about-X 对账`；用 002aa185 写单测断言：一条 replay_world.npz 的手+物体落进 env-local 后，手指与物体几何关系不变。
