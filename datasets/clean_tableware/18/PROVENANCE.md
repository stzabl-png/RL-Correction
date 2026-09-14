# 数据来源 · egodex_auto/clean_tableware__18

帧数 300 · variant `dev` · 生成于本目录的 `world_fused.npz`

**⚠ 本条数据不是纯重建。** 下表逐项说明每个量是我们估出来的，还是数据集直接给的。
拿它做精度结论前请先看这张表。

| 内容 | 来源 | 说明 |
|---|---|---|
| 相机位姿 c2w | **数据集给定** | EgoDex `transforms/camera`（设备 SLAM），非 ViPE 估计 |
| 相机内参 K | **数据集给定** | EgoDex `camera/intrinsic`（设备标定），非 ViPE 估计 |
| 世界系 / 重力 | **数据集给定** | EgoDex 本身重力对齐米制，仅做 Y-up→Z-up 换轴；原点在地面而非首帧相机 |
| 深度 | ****重建**** | ViPE 估计，再按 depth_scale=0.2737502701033877 缩放到真实米制 |
| 双手轨迹 | **数据集给定** | EgoDex ARKit 手部追踪（腕位）；有效帧比例 left 100% right 100%；手指 45 维由 ARKit 25 关节拟合（arkit_to_mano，腕系逐帧拟合），仅当拟合失败才退回零占位（重建日志有醒目警告） |
| 物体 mask | ****重建**** | HOI-DETR 找交互 + SAM2 传播（v17A 全自动） |
| 物体网格 | ****重建**** | SAM3D 单帧重建 |
| 物体尺度 | ****重建**** | sam3d_scale（单帧深度 + 单帧 FoundationPose） |
| 物体位姿 | ****重建**** | FoundationPose 逐帧跟踪 |
| 接触区间 | ****重建**** | mask 重叠检测；已产出 contact_auto.json |
| 材质判定 object_0001 | **VLM 推断** | ? conf=high 建议过滤=False（只记录，未删数据） |
| 材质判定 object_0002 | **VLM 推断** | ? conf=high 建议过滤=False（只记录，未删数据） |
| conf_pos / conf_rot | ****估计**** | 62.0 / 3.0，判定 mixed |

## 不在本条数据里的东西

* **Retrieval（资产库检索）** —— 本条**未使用**资产（物体网格是 SAM3D 重建的）。原因: VLM 判 part_change=none, 不需要分件资产。资产库在 `ego_pipeline/Retargeting/assets/retrieval/`。
* **物体真值** —— EgoDex 只给物体的文字名（`llm_objects`），没有位姿/网格/尺寸。物体那条线的精度只能靠 ARCTIC 验。

任务描述：Clean plates, forks, spoons, and knives placed on a wooden table while sitting against a white background.
物体（LLM 标注）：['plates', 'fork', 'spoon', 'knife', 'table']


## 2026-09-05 手锚尺度复标定
相机中心规范变换(投影不变): object_0 ×1.00, object_1 ×1.00。k 由抓握核心帧接触指节到物面距离最小化求得(手=EgoDex 设备真值)。原件在 _prescale_backup/。


## 2026-09-05 类别先验尺度(回滚手锚后重做)
盘子锚定 object_0=⌀24cm(标准餐盘先验), 整条 ×1.85(相机中心规范变换,投影不变)。⚠ 尺度来源=**类别常识先验非重建**(此任务非抓握,手锚失效;重建内无可靠尺度锚)。原件在 _prescale_backup/。

---

## 2026-09-10 入库到 RL_Correction (擦盘子多母带)

来源 `Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_auto/clean_tableware/18` 原样拷贝
(recon 产物 + poseqa + contact + objects), 未做任何缩放或换网格。

**⚠ 本条只作运动来源, 不提供训练用的物体资产。** 训练时盘与海绵一律用 **take 3** 的资产
(`datasets/clean_tableware/3/objects|retarget|cache`) 与 take 3 的两份 GraspPose 先验
(`tasks/pregrasp/priors/Clean3_{plate_left,sponge_right}.npz`)。原因:
1. 多母带训练要求三条 take 共用同一套物体资产 —— 一个 env 装不下三套网格;
2. 三条 take 的盘都被同一条类别先验归一到 ⌀24cm, 是同一物体的不同重建, take 3 那份已经过
   Dexonomy 合成 + Isaac 抬升验证 + 30M 步训练 (16/16 成功) 的完整检验;
3. 本条自己的 `objects/` 保留在目录里只为溯源与目检, 训练链不读它。

母带 = `tasks/Clean/3/A_Design/L2_Reference/clean18_reference_v1.npz`, 构建参数见台账 §5.12。
