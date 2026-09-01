# 数据来源 · egodex_part4/screw_unscrew_bottle_cap__32

帧数 125 · variant `dev` · 生成于本目录的 `world_fused.npz`

**⚠ 本条数据不是纯重建。** 下表逐项说明每个量是我们估出来的，还是数据集直接给的。
拿它做精度结论前请先看这张表。

| 内容 | 来源 | 说明 |
|---|---|---|
| 相机位姿 c2w | **数据集给定** | EgoDex `transforms/camera`（设备 SLAM），非 ViPE 估计 |
| 相机内参 K | **数据集给定** | EgoDex `camera/intrinsic`（设备标定），非 ViPE 估计 |
| 世界系 / 重力 | **数据集给定** | EgoDex 本身重力对齐米制，仅做 Y-up→Z-up 换轴；原点在地面而非首帧相机 |
| 深度 | ****重建**** | ViPE 估计，再按 depth_scale=0.31568202611378826 缩放到真实米制 |
| 双手轨迹 | **数据集给定** | EgoDex ARKit 手部追踪（腕位）；有效帧比例 left 100% right 100%；手指 45 维由 ARKit 25 关节拟合（arkit_to_mano，腕系逐帧拟合），仅当拟合失败才退回零占位（重建日志有醒目警告） |
| 物体 mask | ****重建**** | HOI-DETR 找交互 + SAM2 传播（v17A 全自动） |
| 物体网格 | **资产库检索(**非重建**)** | 任务 `screw_unscrew_bottle_cap` 命中分件 CAD: object_0=bottle_body(6.5×6.5×19.7cm); object_1=bottle_cap(3.5×3.5×1.7cm)。分配依据: mask 面积中位数降序 <-> 部件最大边降序。触发: VLM 判 part_change=None parts=None |
| 物体尺度 | **资产库给定** | CAD 本身米制，**跳过 sam3d / sam3d_scale** —— 不做尺度估计 |
| 物体位姿 | ****重建**** | FoundationPose 逐帧跟踪 |
| 接触区间 | ****重建**** | mask 重叠检测；已产出 contact_auto.json |
| conf_pos / conf_rot | ****估计**** | 88.0 / 63.0，判定 good |

## 不在本条数据里的东西

* **Retrieval（资产库检索）** —— 本条**已使用**资产库分件 CAD，见上表『物体网格』行。资产库在 `ego_pipeline/Retargeting/assets/retrieval/`，索引 `registry.json`。目前按任务名硬指定；按外观/几何匹配的真检索尚未实现。
* **物体真值** —— EgoDex 只给物体的文字名（`llm_objects`），没有位姿/网格/尺寸。物体那条线的精度只能靠 ARCTIC 验。

任务描述：Unscrew the cap from the metal water bottle placed on a wooden table.
物体（LLM 标注）：['metal water bottle', 'cap']
