# GraspPose (L1-3)

## 名称口径 (2026-08-27 整理)
**GraspPose = 选定候选的完整 29 维位形**(腕 7 + 手指 22, 含 squeeze/pregrasp/接触点),
不是只有腕位姿。

## 选定候选(即"GraspPose 本体")
| 侧 | 文件 | 派生 |
|---|---|---|
| 右手·瓶 | `Right/bottle_right/grasp_data/final_bottle_sharpa_wave_v2__3_5_grasp.npy` | → thumbfix v2(仅 squeeze 拇指 +25°CMC/+15°MCP)→ `tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz`(RL 消费件) |
| 左手·杯 | `Left/cup_left/grasp_data/1_Large_Diameter__1_11_grasp.npy` | → thumbfix v2 同上 → `tasks/pregrasp/priors/Pour17_cup_thumbfix.npz` |

溯源依据: prior npz 的 `source` 字段(已核验, 逐字指向上面两个 npy)。

## 目录
- `Right/bottle_right/`, `Left/cup_left/` — Dexonomy DELIVER 原件(grasp_data=定稿候选,
  all_candidates=全量候选, render=渲染图, workspace 报告)。
- `view_grasppose.sh` — GUI: 当前场景 + 物体按 prior 摆放 + 双手摆到**完整 GraspPose**
  (腕 IK + 手指 22 关节合拢位形, 默认 --fingers grasp; --fingers open 回旧行为只对齐腕)。
