# 数据来源 · egodex_auto/sweep_dustpan__2

帧数 300 · variant `dev` · 生成于本目录的 `world_fused.npz`

**⚠ 本条数据不是纯重建。** 下表逐项说明每个量是我们估出来的，还是数据集直接给的。
拿它做精度结论前请先看这张表。

| 内容 | 来源 | 说明 |
|---|---|---|
| 相机位姿 c2w | **数据集给定** | EgoDex `transforms/camera`（设备 SLAM），非 ViPE 估计 |
| 相机内参 K | **数据集给定** | EgoDex `camera/intrinsic`（设备标定），非 ViPE 估计 |
| 世界系 / 重力 | **数据集给定** | EgoDex 本身重力对齐米制，仅做 Y-up→Z-up 换轴；原点在地面而非首帧相机 |
| 深度 | ****重建**** | ViPE 估计，再按 depth_scale=1.160754894791285 缩放到真实米制 |
| 双手轨迹 | **数据集给定** | EgoDex ARKit 手部追踪（腕位）；有效帧比例 left 100% right 100% ⚠ 手指 45 维为**零占位**，ARKit 25 关节→MANO 映射尚未实现 |
| 物体 mask | ****重建**** | HOI-DETR 找交互 + SAM2 传播（v17A 全自动） |
| 物体网格 | ****重建**** | SAM3D 单帧重建 |
| 物体尺度 | ****重建**** | sam3d_scale（单帧深度 + 单帧 FoundationPose） |
| 物体位姿 | ****重建**** | FoundationPose 逐帧跟踪 |
| 接触区间 | ****重建**** | mask 重叠检测；已产出 contact_auto.json |
| 材质判定 object_0001 | **VLM 推断** | ? conf=high 建议过滤=False（只记录，未删数据） |
| 材质判定 object_0002 | **VLM 推断** | ? conf=high 建议过滤=False（只记录，未删数据） |
| conf_pos / conf_rot | ****估计**** | 87.0 / 87.0，判定 good |

## 不在本条数据里的东西

* **Retrieval（资产库检索）** —— 本条**未使用**资产（物体网格是 SAM3D 重建的）。原因: VLM 判 part_change=none, 不需要分件资产。资产库在 `ego_pipeline/Retargeting/assets/retrieval/`。
* **物体真值** —— EgoDex 只给物体的文字名（`llm_objects`），没有位姿/网格/尺寸。物体那条线的精度只能靠 ARCTIC 验。

任务描述：Gather beads using a sweeper brush and push them into a dustpan on a wooden tablecloth.
物体（LLM 标注）：['beads', 'sweeper', 'dustpan']
