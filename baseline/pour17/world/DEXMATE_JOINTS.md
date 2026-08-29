# DexMate 关节控制名 (**以运行中的 Articulation 为准**)

> 共 **67** 个可控关节。角度=**度**,平移=**米**。
> 在 `dexmate_ctl` 里直接 `<关节名> <数值>`,或写进 `.env_viewer.json` 的 `dexmate.joints`。

> ⚠️ **6 个轮子关节 `L/R/B_wheel_j1/j2` 不在 Articulation 里** —— URDF 有,但 USD 没把它们做进
> 关节链,所以**控制不了**。要移动整机请用 `dummy_base_*` 或 `base_pos`。


## ★ 底盘平面基座 — 移动整机  (3 个)

| 控制名 | 类型 | 轴 | 范围 |
|---|---|---|---|
| `dummy_base_prismatic_x_joint` | prismatic | +1 +0 +0 | `[-2.0, +2.0] m` |
| `dummy_base_prismatic_y_joint` | prismatic | +0 +1 +0 | `[-2.0, +2.0] m` |
| `dummy_base_revolute_z_joint` | revolute | +0 +0 +1 | `[-90.0, +90.0] °` |

## 躯干 torso  (3 个)

| 控制名 | 类型 | 轴 | 范围 |
|---|---|---|---|
| `torso_j1` | revolute | +0 +1 +0 | `[+0.0, +90.0] °` |
| `torso_j2` | revolute | +0 +1 +0 | `[+0.0, +180.0] °` |
| `torso_j3` | revolute | +0 +1 +0 | `[-90.0, +90.0] °` |

## 左臂 L_arm  (7 个)

| 控制名 | 类型 | 轴 | 范围 |
|---|---|---|---|
| `L_arm_j1` | revolute | +0 +1 +0 | `[-176.0, +176.0] °` |
| `L_arm_j2` | revolute | +0 +0 +1 | `[-26.0, +89.0] °` |
| `L_arm_j3` | revolute | +1 +0 +0 | `[-176.0, +176.0] °` |
| `L_arm_j4` | revolute | +0 +1 +0 | `[-176.0, +14.0] °` |
| `L_arm_j5` | revolute | +1 +0 +0 | `[-176.0, +176.0] °` |
| `L_arm_j6` | revolute | +0 +1 +0 | `[-80.0, +80.0] °` |
| `L_arm_j7` | revolute | +0 +0 +1 | `[-79.0, +64.0] °` |

## 右臂 R_arm  (7 个)

| 控制名 | 类型 | 轴 | 范围 |
|---|---|---|---|
| `R_arm_j1` | revolute | +0 +1 +0 | `[-176.0, +176.0] °` |
| `R_arm_j2` | revolute | +0 +0 +1 | `[-89.0, +26.0] °` |
| `R_arm_j3` | revolute | +1 +0 +0 | `[-176.0, +176.0] °` |
| `R_arm_j4` | revolute | +0 +1 +0 | `[-176.0, +14.0] °` |
| `R_arm_j5` | revolute | +1 +0 +0 | `[-176.0, +176.0] °` |
| `R_arm_j6` | revolute | +0 +1 +0 | `[-80.0, +80.0] °` |
| `R_arm_j7` | revolute | +0 +0 +1 | `[-64.0, +79.0] °` |

## 头 head  (3 个)

| 控制名 | 类型 | 轴 | 范围 |
|---|---|---|---|
| `head_j1` | revolute | +0 +1 +0 | `[-85.0, +85.0] °` |
| `head_j2` | revolute | +0 +0 +1 | `[-160.0, +160.0] °` |
| `head_j3` | revolute | +0 +1 +0 | `[-79.0, +85.0] °` |

## 右手 Sharpa (right_*)  (22 个)

| 控制名 | 类型 | 轴 | 范围 |
|---|---|---|---|
| `right_index_MCP_FE` | revolute | +0 +0 +1 | `[-10.0, +90.0] °` |
| `right_middle_MCP_FE` | revolute | +0 +0 +1 | `[-10.0, +90.0] °` |
| `right_pinky_CMC` | revolute | +0 +0 +1 | `[+0.0, +15.0] °` |
| `right_ring_MCP_FE` | revolute | +0 +0 +1 | `[-10.0, +90.0] °` |
| `right_thumb_CMC_FE` | revolute | +0 +0 +1 | `[-10.0, +110.0] °` |
| `right_index_MCP_AA` | revolute | +0 +0 +1 | `[-20.0, +20.0] °` |
| `right_middle_MCP_AA` | revolute | +0 +0 +1 | `[-20.0, +20.0] °` |
| `right_pinky_MCP_FE` | revolute | +0 +0 +1 | `[-10.0, +90.0] °` |
| `right_ring_MCP_AA` | revolute | +0 +0 +1 | `[-20.0, +20.0] °` |
| `right_thumb_CMC_AA` | revolute | +0 +0 +1 | `[-20.0, +20.0] °` |
| `right_index_PIP` | revolute | +0 +0 +1 | `[+0.0, +100.0] °` |
| `right_middle_PIP` | revolute | +0 +0 +1 | `[+0.0, +100.0] °` |
| `right_pinky_MCP_AA` | revolute | +0 +0 +1 | `[-20.0, +20.0] °` |
| `right_ring_PIP` | revolute | +0 +0 +1 | `[+0.0, +100.0] °` |
| `right_thumb_MCP_FE` | revolute | +0 +0 +1 | `[-30.0, +80.0] °` |
| `right_index_DIP` | revolute | +0 +0 +1 | `[+0.0, +80.0] °` |
| `right_middle_DIP` | revolute | +0 +0 +1 | `[+0.0, +80.0] °` |
| `right_pinky_PIP` | revolute | +0 +0 +1 | `[+0.0, +100.0] °` |
| `right_ring_DIP` | revolute | +0 +0 +1 | `[+0.0, +80.0] °` |
| `right_thumb_MCP_AA` | revolute | +0 +0 +1 | `[-20.0, +20.0] °` |
| `right_pinky_DIP` | revolute | +0 +0 +1 | `[+0.0, +80.0] °` |
| `right_thumb_IP` | revolute | +0 +0 +1 | `[+0.0, +100.0] °` |

## 左手 Sharpa (left_*)  (22 个)

| 控制名 | 类型 | 轴 | 范围 |
|---|---|---|---|
| `left_index_MCP_FE` | revolute | +0 +0 +1 | `[-10.0, +90.0] °` |
| `left_middle_MCP_FE` | revolute | +0 +0 +1 | `[-10.0, +90.0] °` |
| `left_pinky_CMC` | revolute | +0 +0 +1 | `[+0.0, +15.0] °` |
| `left_ring_MCP_FE` | revolute | +0 +0 +1 | `[-10.0, +90.0] °` |
| `left_thumb_CMC_FE` | revolute | +0 +0 +1 | `[-10.0, +110.0] °` |
| `left_index_MCP_AA` | revolute | +0 +0 +1 | `[-20.0, +20.0] °` |
| `left_middle_MCP_AA` | revolute | +0 +0 +1 | `[-20.0, +20.0] °` |
| `left_pinky_MCP_FE` | revolute | +0 +0 +1 | `[-10.0, +90.0] °` |
| `left_ring_MCP_AA` | revolute | +0 +0 +1 | `[-20.0, +20.0] °` |
| `left_thumb_CMC_AA` | revolute | +0 +0 +1 | `[-20.0, +20.0] °` |
| `left_index_PIP` | revolute | +0 +0 +1 | `[+0.0, +100.0] °` |
| `left_middle_PIP` | revolute | +0 +0 +1 | `[+0.0, +100.0] °` |
| `left_pinky_MCP_AA` | revolute | +0 +0 +1 | `[-20.0, +20.0] °` |
| `left_ring_PIP` | revolute | +0 +0 +1 | `[+0.0, +100.0] °` |
| `left_thumb_MCP_FE` | revolute | +0 +0 +1 | `[-30.0, +80.0] °` |
| `left_index_DIP` | revolute | +0 +0 +1 | `[+0.0, +80.0] °` |
| `left_middle_DIP` | revolute | +0 +0 +1 | `[+0.0, +80.0] °` |
| `left_pinky_PIP` | revolute | +0 +0 +1 | `[+0.0, +100.0] °` |
| `left_ring_DIP` | revolute | +0 +0 +1 | `[+0.0, +80.0] °` |
| `left_thumb_MCP_AA` | revolute | +0 +0 +1 | `[-20.0, +20.0] °` |
| `left_pinky_DIP` | revolute | +0 +0 +1 | `[+0.0, +80.0] °` |
| `left_thumb_IP` | revolute | +0 +0 +1 | `[+0.0, +100.0] °` |
