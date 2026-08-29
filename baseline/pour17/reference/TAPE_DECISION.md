# 母带定版裁定

**本 bundle 的权威母带 = `pour17_reference_v2__2ed81358__653rows.npz`**

2026-08-29 由本项目负责人拍板：**对齐 ours 当前训练**（而非复现历史 `eval_best`）。

| 文件 | md5 | 行数 | 交互段 | 本 bundle 中的地位 |
|---|---|---|---|---|
| `pour17_reference_v2__2ed81358__653rows.npz` | `2ed81358…` | 653 | 273 | ✅ **权威** —— 评测与对齐一律用它 |
| `pour17_reference_v2__b95546e9__543rows.npz` | `b95546e9…` | 543 | 163 | 存档：现有 `logs/` 里 eval_best 那批 run 用的旧版 |
| `pour17_reference_v1__9f2dbeb6__515rows.npz` | `9f2dbeb6…` | 515 | — | 存档：仅作溯源，**不要用来训练** |

## 为什么母带版本要单独拍板

**evaluator 会读母带**：`mouth_local()` 从母带首个交互行的 `obj_quat_*` 取瓶口/杯口方向，
`PourProgress` 从母带取静置位姿与参考轨迹 ⟹ **换母带 = 换判据**（G3 的口距判定、
Placed 的位姿基准都会变）。所以这不是"给哪份数据"的问题，而是"用哪套判据"的问题。

## 新版相对旧版改了什么

修了三处（RL session 说明）：
1. **重建朝向噪声**——倒水段 `conf_rot` 全红却被照单全收，IK 把它翻译成 **72°/帧的关节跳**；
2. 腕部奇异；
3. 时间扩张 —— 交互段 163 → 273 行。

⟹ 用旧版（或 v1）训练会重现关节抽搐这个病。这也是选新版的实质理由，不只是"跟最新"。

## 已生效之处

- `evaluator/run_eval.py` 的 `--tape` 默认值就是 `2ed81358` 那份；
- `world/world_manifest.json` 的 `reference_tape.md5` = `2ed813586745e72d74cc1f0f04ec3b5d`，
  与之一致；
- `run_eval.py` 自检会核对两者，不一致直接打印警告。

## 若要改用旧版（不推荐）

```bash
python run_eval.py --tape ../reference/pour17_reference_v2__b95546e9__543rows.npz
```
自检会警告"与 manifest 记录的不是同一版"。**报告里必须写明用了哪一版**，
否则两边的成功率不可比。
