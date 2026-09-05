# sweep_dustpan/2 重建数据包 · RL 使用指南

一条 EgoDex 第一视角"扫地"(簸箕+扫帚)视频的完整重建，供 RL 训练与仿真环境初始化。
打包日期 2026-08-21，源 take：`egodex_auto/sweep_dustpan/2`（EgoDex test split, sweep_dustpan/2.mp4）。

## 一句话质量档案

全库扫地数据里最好的一条：**两个物体均"位置 good + 旋转 good"**（confidence 中位 87/87 与
73/73，量程 0~100），零证伪帧，尺度拟合可靠，**两物体都没有旋转自由轴**（三轴全部可观测，
6DoF 参考全维可用——这在瓶杯类数据里很少见）。逐帧 σ 中位：位置 3.3mm、旋转 1.1°。

## ⚠ 给环境初始化的最重要事实

**两个物体从第 0 帧起就已经在手里，且全程 300 帧从未放下**（接触区间 left/right 均为
[0,299]）。所以仿真初始化应当：

> **把物体直接生成在手中（按第 0 帧位姿 + 第 0 帧手 qpos 构成持握状态），不要让策略自己去抓。**
> 这条数据里没有"接近-抓取"阶段可供模仿，任务从"已持握"开始。

第 0 帧初始状态（世界系，米制，重力 Z-up）：

| | 位置 xyz | 四元数 wxyz |
|---|---|---|
| object_0（簸箕，左手持） | [0.557, -0.100, 0.817] | [0.577, 0.745, 0.207, 0.262] |
| object_1（扫帚，右手持） | [0.567, -0.224, 0.877] | [0.428, 0.386, 0.641, 0.507] |

手的初始位姿取 `ref_qpos_*.npz` 的 `wrist_pos[0]` / `wrist_quat_wxyz[0]` / `finger_qpos[0]`。

## 手-物对应（已标注，不用猜）

| 物体 | 手 | 身份 | 分数 pos/rot | 接触判据 |
|---|---|---|---|---|
| object_0 | **左手** | 簸箕 | 87/87 | opposition 0.41, mask 一致性 0.38 |
| object_1 | **右手** | 扫帚 | 73/73 | opposition 0.63, mask 一致性 0.75 |

出处：`contact/contact_v2_summary.json`（四种"物体×手"组合逐一判过，另两组 no_contact）。

## 坐标系与单位

- 世界系：`gravity_z_up_world`——重力对齐、Z 轴朝上、米制。EgoDex 设备标定给定
  （非视觉估计），尺度可信。所有轨迹(手/物体/相机)同一世界系。
- 帧率 **15 fps**，共 **300 帧**（20 秒）。
- 四元数一律 **wxyz** 顺序。

## 文件清单与读法

### retarget/ —— 仿真直接可用（推荐入口）

- `replay_world.npz`
  - `joints_left/right (300,21,3)` 双手 21 关节世界系轨迹（OpenPose 顺序，0=腕）
  - `obj_pose_all (2,300,7)` 两物体位姿 [x,y,z,qw,qx,qy,qz]，第 0 维序同 `object_ids`
  - `valid_left/right (300,)` 逐帧有效位（本条全 1）
  - 另含 `mano_verts_left/right`、`confidence_*`、`phase_*`（阶段标签）等
- `ref_qpos_left.npz` / `ref_qpos_right.npz` —— SharpaWave 手参考
  - `finger_qpos (300,22)` 22 自由度关节角（弧度），`joint_names (22,)` 给出顺序
    （如 right_thumb_CMC_FE…）
  - `wrist_pos (300,3)` / `wrist_quat_wxyz (300,4)`：腕位姿，**recon_world 系**
    （=上述世界系；`frame_of_reference` 字段自述）
  - 质量：右手最大帧间跳变 0.09 rad，左手 0.03 rad，300/300 帧有效
- `object_0.usd` / `object_1.usd` —— Isaac 用物体资产（已拟合真实尺度）

### 重建层

- `world_fused.npz` 全量融合轨迹（相机+双手 MANO+物体），含坐标系变换元数据
- `object_mesh_scaled_final.obj`、`objects/` 物体网格
- `PROVENANCE.md` 出处台账：相机/内参/重力/手=EgoDex 设备给定；深度/物体=重建。
  手指 45 维由 ARKit 25 关节逐帧拟合（真实开合，非占位）

### 可信度层（RL 加权用）

- `poseqa/rts_sweep_dustpan_2_object_{0,1}.npz`
  - `object_ob_in_world_smooth` RTS 平滑后的物体轨迹（建议用它而非原始）
  - `sigma_pos_reported_m` / `sigma_rot_reported_deg` 逐帧 σ——reward shaping /
    软信任度按这个加权（σ 用法详见仓内 `confidence/CONFIDENCE_GUIDE.md`）
- `confidence_complete.json` take 级裁决快照（与全局 manifest 同步，带出处）

### 接触层

- `contact_auto_object_{0,1}.json`：`annotations.left/right` = 接触帧区间；
  `per_frame` = 逐帧接触强度
- `contact/contact_v2_object_{0_left,1_right}.npz|.ply`：物体表面接触区域点云
  （GraspPose/贴合初始化可用）

## Isaac 里肉眼验收（可选）

```bash
cd <解包目录上一级>
OMNI_KIT_ACCEPT_EULA=YES <isaac-venv>/bin/python \
  ego_pipeline/Retargeting/sim/retarget_isaacsim.py \
  --traj sweep_2_better/retarget/replay_world.npz \
  --object-usd sweep_2_better/retarget/object_0.usd sweep_2_better/retarget/object_1.usd \
  --mode render
# ENTER 播放, q+ENTER 退出。--mode physics 可测物理持握。
```

## 已知边界（写给较真的人）

- 旋转 "good" 档按 ARCTIC 真值标定对应**真实误差中位约 20°**（P75 约 24°）——做倾角类
  硬判据时留裕度；逐帧 σ 加权比硬阈值更稳。
- confidence 打分标定物体为 ARCTIC 大件，扫帚簸箕类的跨类别迁移未单独验证。
- 本条 pos 与 rot 的 confidence 中位数值相等（87/87、73/73），疑似打分链共用路径，
  待复核；不影响"两通道均可用"的结论。
- 视频里没有"放下/拿起"事件，不适合训练抓取阶段。
