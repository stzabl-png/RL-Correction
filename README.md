# RL-Correction

**Optimization-Guided Object-Aware RL Trajectory Correction**

From noisy egocentric video reconstruction to high-quality sim-verified robot trajectories.

![Pipeline overview](RL_Correction.png)

## Pipeline

1. **Egocentric Human Video** — input
2. **Noisy Reconstruction** — MANO pose, partial object point cloud, object 6D pose, hand-object trajectory, affordance / contact prior → `τ_raw`
3. **a. Object Geometry** (mesh, point cloud, SDF, normals, scale/shape feature → `O`) · **b. Optimization Prior** (contact consistency, non-penetration, smoothness, joint limits, grasp/IK feasibility → `τ_opt`)
4. **Object-Conditioned RL Correction** — residual correction on top of the optimization prior: `τ_corr = τ_opt + Δτ_RL`
5. **Isaac Sim Verification** — real contact, non-penetration, grasp stability, collision-free motion, task completion, fidelity to video intent
6. **Sim-Verified Data** — `D_verified = {τ_corr | s_i > δ}`
7. **Data Curation / Deduplication** — geometry, contact pattern, grasp pose, trajectory similarity, outcome diversity
8. **High-Quality Sim-Verified Data** — `D_high-quality`, for robot policy training and dataset construction

A trajectory quality reward (`+ contact + stability + progress + success − penetration − collision − joint-limit − jerk − deviation`) closes the loop, with a fidelity term keeping the corrected trajectory close to the original video intent.

---

## 本分支：Step 4 — RL Correction

在优化先验之上做 object-conditioned 的残差修正（keyframe residual correction）：

```
π_θ(τ_raw, τ_opt, O, C) → Δτ_RL
τ_corr = τ_opt + Δτ_RL
```

策略条件包括 hand-object state、contact / affordance prior、phase（pre-grasp / contact / grasp / lift / place）。

轨迹质量奖励：

```
R = + contact + stability + progress + success
    − penetration − collision − joint-limit − jerk − deviation
```

其中 fidelity 项约束修正后的轨迹不偏离原视频意图。修正结果经 Isaac Sim 验证（Step 5）后进入 `D_verified`，再去重得到 `D_high-quality`。

---

## 实现现状（本分支代码）

本分支已并入 RL correction 的可运行实现（IsaacLab + 自研 PPO）。按「抓姿合成器对该物体是否管用」
分成**两套并存、互不干扰**的训练设定，详见 [`docs/TRAINING_SETUPS_A_B.md`](docs/TRAINING_SETUPS_A_B.md)：

| | 设定 A | 设定 B |
|---|---|---|
| | RL 学习 **GraspPose 能处理**的物体（普遍偏大，可整手包络） | RL 学习 **GraspPose 处理不了**的物体（普遍偏小/偏扁，只能指尖捏取） |
| 骨干 | cuRobo close 轨迹 + 28 维残差修正 | 重建人手腕位姿 + affordance 逐点热图，RL 自学指尖抓取 |
| 状态 | 单物体 `Grasp0` 确定性评测 **99.9%** | 管线已跑通，首个对象仍在调试 |

- 上手与踩坑：[`docs/MANUAL.md`](docs/MANUAL.md)
- 外部依赖路径通过环境变量配置：`RR_ROOT`（重建/retarget 仓）、`AFFORDANCE_ROOT`、`OCIR_ROOT`
  （见 `rl_rebuild/correction/paths.py`）

### 训练环境：飞手 → DexMate 真机械臂

原先的训练环境用一只**飞着的** SharpaWave 右手（浮动根刚体，被凭空的 wrench 推动，且关掉重力）。
它能无视物理上不可行的参考轨迹，训出来的东西没有真机意义。现已换成
**DexMate(Vega) 双臂整机 + Sharpa 手**，在**关节空间**做残差修正 ——
限位 / 力矩上限 / 运动学耦合 / 奇异位形全部由仿真器天然执行。

| | 飞手 `--robot flying` | DexMate `--robot dexmate` |
|---|---|---|
| 腕 | 浮动根刚体，wrench-PD 推 | 7 关节链的末端，电机 PD 驱动 |
| 重力 | 关掉 | 带 |
| 动作 | 28 = 腕Δpos3 + Δ轴角3 + 指Δq22（笛卡尔） | **29 = 臂Δq7 + 指Δq22（关节）** |
| 观测 | 144 | **172** |
| 约束 | 靠 reward 惩罚"劝" | 仿真器强制执行 |

两套并存，`--robot` 切换（`rl_rebuild/correction/env/registry.py`），reward / RSI / 冻结窗 /
成功判定 / 日志全部共用，所以同一条 clip 上的数可以直接对照。

**完整实测数据、标定过程与踩过的坑**（吞吐、可达性、残差界、执行器增益、躯干锁死、
接近段设计）见 [`docs/DEXMATE_TRAINING_FEASIBILITY.md`](docs/DEXMATE_TRAINING_FEASIBILITY.md)。

```bash
# 训练 / 评测 / 看
python -m rl_rebuild.correction.train      --robot dexmate --clip Grasp2 --num_envs 1024 --headless
python -m rl_rebuild.correction.eval_policy --robot dexmate --clip Grasp2 --zero_action
./rl_rebuild/correction/view.sh --speed 0.5        # GUI，跑的就是训练那个 env
```

#### ⚠ 首次使用要先生成派生机器人资产

机器人 USD 需要两处修改才能用于训练（原始资产在 MagicSim 仓，**不修改它**，
在本仓 `assets/` 下生成派生副本；25MB 二进制不入库）：

```bash
python tools/make_fixed_torso_usd.py   # 躯干/底盘/头 -> fixed joint（原版躯干撑不住，
                                       # 手臂基座每回合漂 10.5cm）
python tools/add_arm_collision.py      # 补手臂碰撞几何（原版 52 个碰撞体全在手上，
                                       # 手臂能穿过桌子）—— 必须在上一条之后跑
```

不生成会**静默得到完全不同的物理**，env 启动时会大声报警。

#### 当前状态

- `Grasp2` 零残差基线 **100%**（64 env，抬升均值 8.5cm）；开环在手物错位 2cm 时降到 3.1%、
  4cm 时 0% —— 这就是 RL 要填的空间
- 20 条 clip 里实测 **3/9 可直接跑通**；其余卡在摆放流水线不通用（`hover_gap` 与合拢构型
  仍是两个全局常数，对不同尺寸物体不成立）
- 未解决：左右手自碰撞（打开会把抓取打坏，已记录在案）、搬运/放置（`ref_obj_pos` 仍是常量）、双手任务
