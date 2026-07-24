# DexMate(Vega) + Sharpa 手 工作区记录

> 用途：Phase-1 用 Flying Hand 做 MVP（不加机器人），但**手腕工作区必须约束在 DexMate 大致可达范围内**，
> 这样 Phase-2 换成真机时学到的修正是可达的、不用重训。所有数值来自
> `/home/lyh/Project/V2AP/sim2real/assets_dexmate/vega_1/vega_1.urdf` 的 FK 估算（2026-07-16）。

## 运动学拓扑

- **右臂**：7-DOF，关节 `R_arm_j1 … R_arm_j7`（Sharpa 手装在 `R_arm_l7` 腕部）。
- **躯干**：3-DOF `torso_j1..j3`（可重定位肩部、扩展可达域）。
- **底盘**：轮式 `R/L_wheel`（移动，MVP 视为固定）。
- 完整基座→腕链：`torso_j1 j2 j3 · arm_center_j0 · R_arm_j1…j7`。

## 右臂关节限位（rad）

| 关节 | 下限 | 上限 | 说明 |
|---|---|---|---|
| R_arm_j1 | -3.071 | 3.071 | 肩 roll（近全转） |
| R_arm_j2 | -1.553 | 0.453 | 肩 pitch（**受限**，抬臂方向窄） |
| R_arm_j3 | -3.071 | 3.071 | 上臂 yaw |
| R_arm_j4 | -3.071 | 0.244 | 肘（**几乎只能单向弯**） |
| R_arm_j5 | -3.071 | 3.071 | 前臂 roll |
| R_arm_j6 | -1.396 | 1.396 | 腕 pitch（±80°） |
| R_arm_j7 | -1.117 | 1.378 | 腕 yaw（约 ±70°） |

> **腕部末三轴 (j5,j6,j7)** 决定手的朝向自由度：j5 近全转，但 j6/j7 各约 ±80°/±70° →
> 局部可达朝向是一个**宽锥而非完整 SO(3)**。约束 Flying Hand 朝向时以此为界。

## 尺度与安装

| 量 | 值 | 来源 |
|---|---|---|
| 肩(arm_center)相对底座 | `[-0.246, 0, 0.428]` m | URDF FK 零位 |
| **臂展（肩→腕最大半径）** | **≈ 0.755 m** | 臂段长度和 |
| 腕零位高度（相对底座） | 0.927 m | URDF FK |
| 机器人世界安装 | `DEXMATE_ROBOT_POSITION`，绕 Z 转 90° | `V2AP/sim2real/scene_builder.py:44-47` |
| 桌面高度 / 物体基准 | `TABLE_TOP_Z=0.80`，物体 `[0,0.55,0.80]` | `scene_builder.py:36-37` |

**操作工作区（粗略）**：机器人前方、桌面高度(~0.8 m)附近、距肩 <0.75 m 的区域；配合躯干 3-DOF 可前后/上下扩展。

## MVP 的关键结论：工作区约束 ≈ 限制手腕残差幅度

因为 **残差锚点 = cuRobo 生成的可行轨迹**，而 cuRobo 轨迹**按构造就落在 DexMate 可达集内**，所以：

> **只要把手腕残差 `a_wrist` 的幅度 × 时域限制住，Flying Hand 的腕位就自动待在可达集附近。**
> 不需要显式建可达流形。

Phase-1 具体做法（保守）：
- **位置残差**：每步 `action_scale_wrist_pos` 小（如 ≤3 mm/步），并对累计腕位偏离参考设软边界（如 ≤0.15–0.25 m）。
- **朝向残差**：每步小角增量，累计偏离参考朝向 ≤ ~60°（落在 j6/j7 的锥内）。
- **越界处理**：软惩罚（进 reward）或 clip（进 env），Phase-2 换真机时用 `dexmate_curobo.py` 的 IK 可达性做硬校验。

## Phase-2 换真机时的接口

- IK / 可达性 / 运动规划：`V2AP/sim2real/dexmate_curobo.py`（cuRobo，已集成；OCIR 是 cuRobo v2）。
- 动作层：Flying Hand 的 6-DOF 腕位姿目标 → 通过 cuRobo IK 映射成 `R_arm_j1..j7` 关节目标（reward 不变，只换动作/观测层）。
- URDF/USD 资产：`V2AP/sim2real/assets_dexmate/vega_1/`。
