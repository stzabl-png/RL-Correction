# fp_pose 改进备忘(FoundationPose 外围优化)

> 归档笔记。前提:**FoundationPose 是 NVIDIA 的,当黑盒不改源码**,只在外围
> wrapper(`recon_pipeline/fp_pose/fp_common.py`、`kalman_filter_6d.py`)动手,
> 或组合 FP 已暴露的功能。fp_pose 不是项目重心,目标是"尽量调好一些"。
>
> 基线版本:HV2RD `main@39d3ef4`(已同步进 Reconstruct_and_Retarget)。
> 当前特性:anchor 帧 register 一次 → 从 anchor **双向(前+后)全帧** FP++ 追踪
> (mask 质心修正 xy + 6D 卡尔曼),输出相机系逐帧 `ob_in_cam` 4×4;支持多物体。

## 当前已知短板(要治的)
1. **静默漂移**:`track_one` 即使物体全遮挡也返回位姿,**无置信度 / valid 标志**,坏帧原样写盘。
2. **远离 anchor 漂移**:track 模式全程只 register 一次。
3. **anchor 两侧接缝**:正/反向是两段独立 KF,anchor±1 处可能小跳变。
4. **欧拉角 KF**:`mat_to_6d/mat_from_6d` 用 euler xyz,有 gimbal / 大旋转插值差。
5. **尺度依赖**:mesh 尺度来自 sam3d_scale,scale 偏→位姿系统性偏(那步 warnings 只告警)。
6. **内存**:一次性预加载全部帧+深度,长 HOI4D 片段可能 OOM。

---

## A. 低成本 wrapper 改进(不碰 FP 源码)

| # | 改进 | 做法 | 成本 | 收益 |
|---|---|---|---|---|
| A1 | **每帧置信度 + valid 标志** | 复用 FP 的 `ScorePredictor` 给 `track_one` 结果打分;或自己把 mesh 按 `ob_in_cam` 投影算 **silhouette-IoU vs SAM2 mask**。写入 meta,低于阈值→`valid=0` | 低 | ⭐最关键:下游/IsaacSim 知道哪帧不可信 |
| A2 | **自适应重注册** | A1 置信度/IoU 掉到阈值以下时,用该帧 SAM2 mask 调 `est.register` 恢复 | 低 | 显著降远端漂移 |
| A3 | **喂 FP 前清理深度** | 用 `sam3_hands` 手 mask + 深度百分位,剔掉物体区域里的手/离群深度再喂(sam3d_scale 有可复用代码) | 低 | 手遮挡更稳 |
| A4 | **双向接缝处理** | 正/反向在 anchor 附近按置信度取优或轻融合 | 低 | 轨迹连续性 |
| A5 | **后处理平滑** | 最终轨迹做四元数 slerp + 平移高斯平滑(按 valid 门控) | 低 | 降抖,IsaacSim 驱动更平 |
| A6 | **四元数/rotvec KF** | 把 `kalman_filter_6d` 的欧拉角换成四元数/rotvec | 中低 | 消除 gimbal |
| A7 | **自动挑 anchor** | 在有 mask 的帧里挑 mask 面积大/深度有效多/手遮挡少的当注册帧(与 SAM2 点选帧解耦) | 低 | 初始注册更准 |
| A8 | **流式读帧防 OOM** | 长视频改滑窗/按需读,别全量预加载 | 中 | 长片段鲁棒性 |

## B. 可组合的 FP 自带功能(直接调,不改 FP)
- **`ScorePredictor`**:FP 注册时本就用它给位姿假设打分;同一 scorer 可给 track 结果打分(→ A1/A2 置信度来源)。
- **`register(iteration=)` / `track_one(iteration=)`**:现 `register_iters=5`、`track_iters=2`;**提高 track_iters** 可降漂移(代价速度)——纯调参。
- **`register-each` 模式**:已有,作为**精度上界基准**,量化 track 模式漂了多少。
- **FP `debug_dir` / debug level**:开 debug 导出内部渲染叠加图,人工核查坏帧(排查用,不进生产)。

## C. 统一效果探究方案
**关键优势:HOI4D 自带物体位姿 GT**(`3Dseg/output.log`),可定量评。

- **测试集**:5–10 条代表性 HOI4D 序列(含遮挡/快速运动)。
- **指标**:
  - **ADD / ADD-S**(对称物用 ADD-S);
  - 逐帧**旋转误差(°)+ 平移误差(cm)** vs GT;
  - **抖动**:帧间速度/角速度方差;
  - **valid 覆盖率 + IoU**(A1 自评 vs GT 是否一致)。
- **消融维度**(逐个开关):
  1. `track_iters` ∈ {2,3,5};
  2. A2 自适应重注册 on/off(扫 score 阈值);
  3. A3 深度清理 on/off;
  4. A5 后平滑 on/off;
  5. A6 欧拉 vs 四元数 KF;
  6. 基准:`register-each`(上界)、纯 track 无 KF(下界)。
- **产出**:一张表选出默认最优组合;A1(valid)无论结果如何设为默认输出。

---

## 优先级建议
1. **第一梯队(先做)**:A1 valid/置信度 + A2 自适应重注册 + A3 深度清理
   —— 低成本,直接治"静默漂移"(对 IsaacSim 验证最致命)。
2. **第二梯队**:A5 平滑 / A6 四元数 KF / A7 自动 anchor(精度与平滑)。
3. **定参**:用 C 的 HOI4D GT 探究一次性选出默认组合。
4. A8 视实际序列长度/显存决定是否需要。

## 相关文件
- `recon_pipeline/fp_pose/fp_common.py` — 主逻辑(`_run_fp_pp_track_one` / `run_fp_pp_track`)
- `recon_pipeline/fp_pose/kalman_filter_6d.py` — 6D KF(A6 在这里改)
- `recon_pipeline/sam3d_scale/run_sequence.py` — 深度反投影/清理可复用代码(A3)
- 部署变体:STEP_6_pose 分支 `foundationpose_pp_ros`(TensorRT,只跟踪)
