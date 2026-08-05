# 冠军存档：Approach + Pick 端到端（seed 42）

第一条把**完整端到端任务**跑通的配方：从双臂对称站姿出发 → 飞到 GraspPose →
抓稳 → 微抬升。数据集 `egodex/part2/basic_pick_place/3`（仓库里叫 `Grasp3`）。

机器：A6000（`yanghong@…`），Isaac Sim 5.1。训练起于 2026-08-03 20:19 PDT。

---

## 成绩（确定性评测，无探索噪声）

| 口径 | 成功率 | 回合数 | 成功中位长度 |
|---|---|---|---|
| **端到端**（含站姿前缀，c0=0 标准起点） | **99.99%** | 14287/14289 | 61 步 |
| 端到端（c0~U(0,0.9) 训练分布对照） | 99.97% | 14275/14279 | 61 步 |
| 纯抓取段（从 GraspPose 起步，无站姿前缀） | 99.24% | 11951/12043 | 34 步 |
| 纯抓取段（训练分布对照） | 99.06% | 13326/13452 | 30 步 |

两次失败都落在物体随机抖动的边角（成功回合抖动均值 2.29mm vs 失败 3.05mm）。

学习速度（`milestones.json`）：

| 里程碑 | agent steps | 墙钟 |
|---|---|---|
| 10% | 0.23M | 2.3 min |
| 50% | 1.34M | 15.8 min |
| 90% | 6.75M | 77.4 min |

先抓后飞门控在 **1.44M 步毕业**（抖动起点分布上抓取慢 EMA ≥ 0.8），之后放行接近分支。

---

## 复现：训练

```bash
SHARPA_WANDB=0 PYTHONPATH=. $PY -u -m tasks.pregrasp.train --headless \
  --clip Grasp3 --num_envs 1024 \
  --prior_npz tasks/pregrasp/priors/Grasp3_candidates/8_5.npz --prior_yaw 215 \
  --approach --stance_prefix 60 --kl_threshold 0.02 \
  --grasp_first --gf_target 0.8 --max_agent_steps 40000000
```

`$PY` = 带 isaacsim + isaaclab 的解释器。数据不用另配，`datasets/` 随仓库走。
多卡机加 `RL_ISAAC_NO_GUARD=1`。**`--headless` 不能漏**（漏了会在建场景阶段被 OOM
killer 杀掉，退出码 137）。

40M 步在 A6000 上约 8 小时。只想看结论的话，6.75M 步（约 1.3 小时）就到 90%。

### 四个组件，缺一即回到历史失败模式

1. `--stance_prefix 60` —— 参考轨迹前拼「对称站姿 → 人手轨迹起点」的插值，
   让任务从**设计的起点**出发而不是人手轨迹的起点
2. `--grasp_first` —— 阶段 A 全部回合纯练抓取，慢 EMA ≥ 0.8 毕业才放行接近分支
3. **抖动起点池**（`--grasp_first` 自带）—— 阶段 A 起点带 ±3cm/±15° 随机偏移，
   正好是课程全程合法到达误差的包络，抓取天生容错，交接不断裂
4. `--kl_threshold 0.02` —— trust region 压小；**必要但不充分**

**不需要**（都试过并证伪）：姿态混合、锥形信任管、`kl 0.01`、课程提速、
组3 PPO 超参、在混训语境下人为抖动直接起步。

---

## 复现：直接用存档评测

```bash
CKPT=results/part2/basic_pick_place/3/Approach_Pick_冠军seed42/stage1_nn/last.pth

# 端到端 (应得 ~99.99%)
SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.eval --headless \
  --clip Grasp3 --checkpoint $CKPT \
  --prior_npz tasks/pregrasp/priors/Grasp3_candidates/8_5.npz --prior_yaw 215 \
  --approach --stance_prefix 60

# 录像
SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.record --headless \
  --clip Grasp3 --checkpoint $CKPT \
  --prior_npz tasks/pregrasp/priors/Grasp3_candidates/8_5.npz --prior_yaw 215 \
  --approach --stance_prefix 60
```

⚠ **评测/录像的 flag 必须与训练一致**。`--approach` / `--stance_prefix` 漏掉就是在
评一个不同的任务——这个坑踩过，曾经 eval 和 record 根本没开 approach。

---

## 目录内容

| 路径 | 是什么 |
|---|---|
| `stage1_nn/last.pth` | **最终权重**（39M 步）。复现评测用这个 |
| `stage1_nn/best.pth` | 训练期最佳 |
| `stage1_nn/ep_*_step_*_reward_*.pth` | 30 个中途存档，看学习轨迹用 |
| `stage1_tb/events.*` | TensorBoard 全程曲线（`tensorboard --logdir stage1_tb`） |
| `train.log` | 训练全日志 |
| `eval/先抓后飞_无混合.eval_e2e.log` | 端到端评测日志（99.99% 那次） |
| `eval/先抓后飞_无混合.eval.log` | 纯抓取段评测日志（99.24% 那次） |
| `video/冠军40M_端到端_从对称站姿.mp4` | 验收录像 |
| `score_map.npz` | 接触分数图（`python -m tasks.pregrasp.view_score_map --npz …` 画热力图） |
| `milestones.json` | 各成功率里程碑的步数与墙钟 |
| `经验总结.md` | **机制层面的复盘**：三个被数据确认的转折 + 六条工程教训 |

---

## 想读懂"为什么是这个配方"

看 `经验总结.md`，以及 `docs/DESIGN_LOOP.md` §2.12~§2.18 的完整台账
（含每条被证伪的路线）。一句话版本：

**训练不稳定的病根是「起步竞速」而不是步长。** 两个起步分支共享一张网，
前 3M 步谁先站稳谁活；9 条混训 run 只活了 2 条，`kl` 调到 0.05/0.02/0.01 三档
都救不了。把竞速改成顺序（先抓后飞），运气就变成了制度。
