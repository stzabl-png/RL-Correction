# 重建 → 训练环境:通用对齐流程

> 拿到任意一条重建,如何自动摆进训练环境。**不针对单条 clip 调参**。
> 实现:`rl_rebuild/correction/bimanual_align.py`;可视化/落地:`correction_env.py::_place_bimanual`
> 首次验证数据:`egodex/part2/basic_pick_place/2`(甜甜圈)

## 核心原则

**只用有物理/几何约束的通道,不用自由悬空的量。**

| 通道 | 可信? | 依据 |
|---|---|---|
| 相机位姿 `c2w` | ✅ | SLAM/ViPE 有几何约束 |
| 交互手 + 接触段 | ✅ | 物体给了接触锚点 |
| mesh 几何 | ✅ | 确定性 |
| MANO 骨长 | ✅ | 实测掌长 10.08cm,左右手一致到 0.01cm |
| **非交互手的绝对位置** | ❌ | 无接触约束会漂。实测左手被整体推近右手 19.5cm |
| **重建的物体朝向** | ❌ | 实测与任何稳定静置姿态差 ~100° |
| **重建的手-物相对位置** | ❌ | 实测掌心离物体最近 8.6cm,抓取帧 16.7cm——手从没真正碰到物体 |

一只手可不可信,取决于**它有没有接触约束**,而不是它动不动。
(左手全程不动是原视频事实,重建还原对了;错的是它的绝对位置。)

## 输入

```
RetargetOutput/<clip>/replay_world.npz     joints_left/right (T,21,3), obj_pose (T,7),
                                           phase_left/right (T,), fps
ReconstructOutput/<clip>/world_fused.npz   c2w (T,4,4), world_xy_alignment_mode,
                                           coordinate_frame
ReconstructOutput/<clip>/object_mesh_scaled_final.obj
机器人侧(从 Articulation 实测)             头 link / 左右手 link / 肩 link 的世界位姿
```

## 流程

### Step 0 — 校验重建的世界系对齐(决定要不要转 yaw)

```
world_xy_alignment_mode == "middle_camera_forward_projected_xy"
coordinate_frame        == "gravity_z_up_world"
   → yaw = 0            重建的 +X 已经是人的正前方,机器人也朝 +X,天然对上
   → 否则 fallback:     ① 物体搬运方向→+X  ② 交互手接近方向→+X  ③ 不转
```

**这一步最容易做错。** 上游管线已经用中间帧的相机前向把世界 XY 对齐过了
(本 clip 实测相机前向平均只偏离 +X **3.29°**)。自己再发明朝向准则
(双手连线 / 物体搬运方向)去转它,反而会把正确的相对关系转歪——
盒子本来就在右前方(搬运 −43.8°),硬转到 0° 等于抹掉真实几何。

### Step 1 — 从 `phase_*` 检测交互结构

```
hands         = 有 phase==1 的手     → 自动分派 单手 / 双手 / 哪只手
grasp_frame   = 首次接触帧
release_frame = 最后接触帧
```

全流程的分派依据,不看手的身份、不需要人工标注。

### Step 2 — XY 平移:**人头 → 机器人头**

```
dxy = 机器人头 link 世界 XY − 相机(人头)在 grasp_frame 的 XY
```

**为什么锚头不锚手**:人和机器人"手相对身体"的位置差别很大——实测这条 clip
里人的双手几乎都在身体中线上(相对头 y≈±0.03),而机器人双手左右分开 ±0.23。
锚在手上会把人整个推偏(实测偏 31.8cm);锚在头上等于"让机器人站在人当时的位置",
演示被放进机器人的第一人称坐标系。**且完全不依赖哪只手交互。**

相机位姿缺失时退回锚在交互手掌心 + 固定前伸量(`anchor_mode` 会记录用了哪种)。

### Step 3 — Z 平移:抓取距离

```
dz = obj_rest_z + grasp_gap − 掌心_z(grasp_frame)
```

`grasp_gap`(cfg,默认 0.045)= 抓取帧掌心距物体**中心**的目标距离。
**调它 = 整体升降人手轨迹与所有标记,物体不动**(物体独立贴桌)。

### Step 4 — 物体姿态:稳定静置姿态

```
trimesh.poses.compute_stable_poses(mesh) → 枚举全部稳定姿态
选与重建姿态**测地距最小**的那个(保留朝向信息;对不对称物体重要)
```

⚠ 该函数要跑 1~2 分钟,**必须缓存**到 mesh 同目录 `stable_poses.json`(带 mtime 校验)。
直接在 Isaac 进程里现算会阻塞主循环,触发中止回调 → **段错误 exit 139**。

### Step 5 — 物体位置:re-anchor 到掌心

```
XY = 交互手掌心 @grasp_frame        Z = 贴桌(该稳定姿态下的静置高度)
```

消掉重建那 10cm+ 的手-物系统误差。

### 非交互手

**不做参考跟踪**,机器人对应臂保持默认姿态。
重建说"它没动"可信,"它在哪"不可信 → 只保留语义,不继承位置误差。

## 落地时必须一起做的三件事

1. **覆盖 `ref_obj_pos` / `ref_obj_quat` / `ref_obj_vel`**
   只写 `obj_init_pos` + `write_root_pose_to_sim` 不够——冻结窗每个物理步都会用
   `ref_obj_pos[t]` 把物体钉回旧 re-anchor 位置。
2. **读机器人 body 位姿前先刷新**
   `write_joint_state_to_sim` 只写物理引擎,`data.body_pos_w` 有缓存。
   必须 `sim.step(render=False)` + `dexmate.update(dt)`,否则读到旧值(实测差 2.3cm)。
3. **摆姿势用 Articulation 张量 API,不能用 USD DriveAPI**
   GPU 仿真开着 `eENABLE_DIRECT_GPU_API`,PhysX 拒绝 USD 路径的 `setDriveTarget`——
   属性写进去了但完全不生效。

## 质量门(`placement_report`)

自动判定这条 clip 能不能直接训,不人工看每一条。

| 指标 | 阈值 | 适用 |
|---|---|---|
| `grasp_gap_m` | 与目标差 <5mm | 全部 |
| `palm_z_min` | ≥ 桌面 | 全部 |
| `obj_within_table` | ≤ 桌半宽 | 全部 |
| `reach_max_m` | ≤ 臂展 | 有肩位时 |
| `yaw_tier` | == 0(用了重建自带对齐) | 全部 |
| `cam_fwd_dev_deg` | < 20° | 有相机时 |
| `stable_vs_recon_deg` | 记录不拦(>60° 说明重建朝向不可信) | 全部 |
| `hand_axis_vs_task_deg` | > 45° | **仅双手任务** |

**肩位必须从 Articulation 实测取**,不能用 URDF 零位偏移——躯干姿态会把肩抬高
(本 clip 实测肩 z=1.301,零位算出来是 0.428,差 87cm,导致 reach 被严重误判)。

## 已知失效条件

- **单手 clip 里非交互手的位置不可用** → 已规避(不跟踪、不作约束)
- **双手同时交互**:两只手都有接触锚点,反而更好办;但 `hand_axis_vs_task_deg`
  这条门才有意义
- **物体几乎不平移**(原地旋转类任务)→ Step 0 的 fallback ①② 都会失效,退到"不转"
- **重建管线换了 XY 对齐模式** → Step 0 会检测到并退回 fallback,不会静默出错
- **旋转对称物体**:稳定姿态"选最接近重建的"没有信息量(本 clip 两个稳定姿态
  概率 0.524/0.476,几乎等价);对不对称物体才要紧

## 本 clip 的验收结果

```
yaw_tier 0(重建自带对齐)   cam_fwd_dev 3.29°   锚定 head
物体离肩 64.2 cm ✅        抓取帧掌心离肩 61.2 cm ✅(臂展 75.5,余量 14cm)
训练用到的 57 帧: 1 帧超限 2.5cm(1.8%)—— 判定可接受,交给 RL 妥协
(质量门里原有一项 err_over_residual 已删除, 理由见下)
```

## 已删除的错误指标:`err_over_residual`

曾经有一项 `err_over_residual`,算的是"接触段内每帧掌心到**物体初始位置**的距离
偏离 `grasp_gap` 多少"。**这个指标是错的**,已删除:

- 接触段里手是**抓着物体在搬运**的,离开物体初始位置本来就是正常任务行为,
  不是误差。实测它与搬运距离成正比(接触段越长数值越大),和数据质量无关。
- 它想防的"重建手-物误差超残差权限"这件事,**已被 Step 5 的 re-anchor 解决** ——
  物体就放在抓取帧掌心正下方,抓取时刻误差按构造为 0,由 `grasp_gap_m` 验证。
- 更根本的是:**参考轨迹只是粗略模板,策略靠残差自己找可行解**。质量门不该
  逐帧约束手物距离,只该管"抓取时刻够不够得到"和"轨迹在不在工作空间内"。

## 未解决

**`ref_obj_pos` 是常量。** 接触段里物体被拿起来搬走,环境里却把它钉在桌面。
这不影响抓取(物体被握住后跟着手走),但造成观测里 3 维恒零的死通道
(`ref_obj_pos[t5] − ref_obj_pos[t]`),且 `lam_traj` 因此被置 0。
要做"带轨迹抓"(搬运段)时必须先恢复物体参考轨迹。
