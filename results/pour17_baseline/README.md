# Pour17 Baseline — IB_gh_oh_s51 @20M (正式 Success 0.5586)

当前最好的倒水策略 ckpt（2026-09-02），供 Baseline 对比与 DP（数据生成/蒸馏）使用。

## 成绩（确定性评测，512 回合，t0 从头口径，见 `eval_512ep.log`）

| ★Success (G3倒水 ∧ 放回) | G3 倒水几何 | 放回原位 | 走完时钟 |
|---|---|---|---|
| **0.5586** | 0.9883 | 0.5586 | 0.9531 |

主判据是绝对几何量（瓶口/杯口圆盘相交 6.75cm ∧ 双物回静置 ≤3cm 立正保持 15 步），不读参考轨迹形状。
`demo_20M.mp4` 是该 ckpt 的单集回放（接近→直立抓握→双提→翻腕对口倒→放回立正）。

## 配方（缺一不可，回放前逐项核对横幅）

- 物理规矩：物体 0.1kg / 物体摩擦 5 / 指垫摩擦 5（代码默认，`correction_env.PHYS_RULE`，不要手填）
- 参考母带：`tasks/Pour/17/A_Design/L2_Reference/pour17_reference_v3.npz`（已在仓，LFS）
- 旗：`POUR_GRIP_SHAPE=1`（对向势+跟腕比）、`POUR_HOLD_POSE=1`（抓稳期姿态守恒 v1）、
  `POUR_TIER_FLOOR_START=auto`（起始黄窗 0~18 行）
- 奖励旗（行为的一部分）：`POUR_PLACE_SHAPE=1 POUR_MOUTH_BONUS=1 POUR_OBJ_CONTACT=1
  POUR_COLLIDE=pen POUR_COLLIDE_PEN=0.5 POUR_BONUS_DIST=1 POUR_BONUS_NOW=1 POUR_SQUEEZE_FF=1`
- 网络：503 obs / 58 act（双臂14+双手44 残差），0.34M 参数；特权 12 维（质量+摩擦+tip力）只进 critic/8维嵌入

物理与档位参数会由 `world_fingerprint.restore_physics_env()` 按本目录 `IB_gh_oh_s51/world.json`
自动还原，世界指纹硬闸不匹配会拒跑（这是保护，不要用 `POUR_IGNORE_WORLD=1` 绕过）。

## 怎么跑

```bash
# 评测（512 回合确定性）
export SHARPA_WANDB=0 PYTHONPATH=. \
  POUR_REF_NPZ=tasks/Pour/17/A_Design/L2_Reference/pour17_reference_v3.npz POUR_VARIANT=HYB \
  POUR_TIER_FLOOR_START=auto POUR_GRIP_SHAPE=1 POUR_HOLD_POSE=1 \
  POUR_PLACE_SHAPE=1 POUR_MOUTH_BONUS=1 POUR_OBJ_CONTACT=1 POUR_COLLIDE=pen POUR_COLLIDE_PEN=0.5 \
  POUR_BONUS_DIST=1 POUR_BONUS_NOW=1 POUR_SQUEEZE_FF=1
$PY -u tasks/Pour/17/C_Wiring/eval_pour.py \
  --checkpoint results/pour17_baseline/IB_gh_oh_s51/stage1_nn/last.pth \
  --num_envs 256 --episodes 512 --headless

# 录像（单集 1280x720）
$PY -u tasks/Pour/17/C_Wiring/record_pour.py \
  --checkpoint results/pour17_baseline/IB_gh_oh_s51/stage1_nn/last.pth \
  --out /tmp/pour_baseline.mp4 --headless --enable_cameras
```

`$PY` = 带 isaacsim+isaaclab 的解释器（本机 `/home/lyh/luhr/MagicSim/.venv/bin/python`；
msc `~/miniconda3/envs/env_isaaclab/bin/python`；多卡机记得 `RL_ISAAC_NO_GUARD=1`）。

## 给 DP 的注意事项

- 确定性执行用 `act_inference`（mu，无探索噪声）；成功率即上表。
- 生成示教数据建议按 t0 口径滚整集，用 `sr_t0/success` 判据筛成功集；判据代码在
  `tasks/Pour/17/A_Design/L3_Learning/progress_batch.py`（G3_pour ∧ placed）。
- 特权 12 维（真实质量/摩擦/tip力）在部署侧不可得——蒸馏时只用 503 维 obs（其中前 12 维特权由
  8 维嵌入进 actor，可一并蒸掉）。
- 已知边界：单条母带（pour17）、单套物体、seed 方差大（同配方 4 seed = 0.56/0.004/0/0，本 ckpt 是最好的那个）；
  撤离段 G4 不在判据内。完整台账见 `tasks/Pour/17/A_Design/DECISIONS.md` L5-33~36 与 `ImproveBase/PLAN.md`。
