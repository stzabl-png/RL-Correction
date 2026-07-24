# RL_Correction 使用手册

> **新开 terminal / 新会话，先读这份。** 这里是"怎么跑"和"为什么这么设"。
> 当前进度快照见 `docs/HANDOFF_*.md`（会过时）；本手册只写稳定的东西。
> 最后更新: 2026-07-22

---

## 0. 一句话

用 PPO 训练一个 **残差策略**：在给定的参考手部轨迹上叠加小幅修正，让飞手 SharpaWave 在
IsaacSim 里真正抓起桌上的物体。当前 `pp0` 单物体已跑通（冠军 run = `logs/Grasp0`，**带点云**，
确定性评测 **99.9%**）。

**命名约定（2026-07-22 起）**：run = `Grasp<物体号>`（`Grasp`=RL 学的抓取动作，`0`=物体号，对应 clip `pp0`）。
**点云已是默认输入**（`enable_pointcloud=True / n_obj_points=64`），不再用 `pc`/`nopc` 之类 tag 标注。
⚠️ 改名后目录不再是 `correction_<clip>_<tag>` 格式，`eval_policy`/`record` **无法再自动反推 clip，必须显式 `--clip pp0_anchor`**。

---

## 1. 环境

```bash
PY=/home/lyh/luhr/MagicSim/.venv/bin/python     # isaacsim + isaaclab + torch 2.7.0+cu128
cd /home/lyh/Project/RL_Correction
```

**⚠️ 所有命令前面必须加 `SHARPA_WANDB=0`。** 否则在带终端的会话里 wandb 会弹交互式登录提示
把进程卡死（`rl_rebuild/utils/wandb_writer.py:14`）。headless 跑时它会自己降级，但别赌。

**GPU**: 本机 RTX 5090 (32 GB)。远端双 A6000 见 memory `remote-a6000-deploy`（近期未用）。

---

## 2. 最常用的五条命令

```bash
PY=/home/lyh/luhr/MagicSim/.venv/bin/python

# ① 训练
SHARPA_WANDB=0 $PY -m rl_rebuild.correction.train \
    --clip pp0_anchor --num_envs 2048 --max_agent_steps 40000000 \
    --video_every 0 --tag <实验名> --headless

# ② 确定性评测 (关探索噪声, 只跑 mu) —— 唯一能对外说的成功率
#    改名后的 run 必须显式 --clip (无法从 Grasp0 反推); 冠军权重见下
SHARPA_WANDB=0 $PY -m rl_rebuild.correction.eval_policy \
    --checkpoint logs/Grasp0/2026-07-21_22-18-11/stage1_nn/last.pth --clip pp0_anchor \
    --num_envs 2048 --episodes 3 --headless

# ③ 零残差基线 (不载策略, 纯参考轨迹能做到多少)
SHARPA_WANDB=0 $PY -m rl_rebuild.correction.eval_policy \
    --zero_action --clip pp0_anchor --num_envs 2048 --headless

# ④ 录像 (mp4 落在 <run>/videos/)
SHARPA_WANDB=0 $PY -m rl_rebuild.correction.record \
    --checkpoint logs/<run>/stage1_nn/last.pth --num_envs 1 --fps 10 --headless
#   --fps 10 = 半速慢放 (采集 20Hz); --zero_action 录基线; 去掉 --headless 开窗

# ⑤ GUI 交互试玩 (按 Enter 跑一回合)
SHARPA_WANDB=0 $PY -m rl_rebuild.correction.play --clip pp0_anchor
```

### TensorBoard

```bash
PYTHONPATH=<装了 pkg_resources 的目录> $PY -m tensorboard.main \
    --logdir /home/lyh/Project/RL_Correction/logs --port 6006
# 浏览器开 http://localhost:6006
```
⚠️ venv 里 setuptools 是 82 版（81+ 移除了 `pkg_resources`），tensorboard 的 CLI 入口会挂。
正规修法：`pip install "setuptools<81"`（但该 venv 无 pip，且属于 MagicSim，慎改）。
临时办法：从 anaconda 拷一份 `pkg_resources` 目录出来挂 `PYTHONPATH`。

**看曲线时把左下角 Smoothing 拉到 0.9** —— 曲线很抖，等距采点会误判（踩过坑）。

---

## 3. 脚本清单

| 脚本 | 用途 |
|---|---|
| `train.py` | 训练入口 |
| `eval_policy.py` | **确定性大批量评测**（含逐指 elastomer 接触分析）|
| `record.py` | checkpoint → mp4 + 逐项统计 |
| `play.py` | IsaacSim GUI 交互试玩 |
| `phys_ablation.py` | 物理/控制器/噪声消融（`--kinematic --phys_align --action_noise`）|
| `test_pointcloud.py` | 点云→PointNet 通路自检（形状/数值/置换不变性）|
| `m0_replay.py` | 零残差回放 + 分相位跟踪误差 |
| `m1_check.py` | obs 健全性 / reward 方向性检查 |
| `diag_grasp.py` | 抓握失败诊断（`--pin_object` 钉死物体看真实对齐）|
| `audit_physics.py` | 打印运行时真实物理参数 |
| `stage_training_data.py` | 四类源数据 → `TrainingData/` |
| `export_qpos.py` | 离线 retarget 导出 22 关节 qpos（**pinocchio 与 Kit 冲突，必须离线跑**）|
| `clips.py` | clip 注册表 |
| `schema.py` | 数据契约 `DataUnit/RefTrajectory/GraspTarget` |

### 可用的 clip

`pp0_anchor`（cuRobo 骨干，**当前主力**）、`pp0_human`（人手骨干）、`pp55_anchor`、`pp55_human`

`_anchor` = 残差基准是 cuRobo 规划轨迹；`_human` = 基准是重建人手轨迹（噪声大，见 §7）。

---

## 4. 任务结构（`grasp_only` 模式）

```
静置 20 步    手停 PreGrasp(cuRobo close 段起点=帧50), 物体钉桌上沉降
RL 抓取 60    ★策略学的就是这 3 秒★  参考=cuRobo close+squeeze(帧50→109)
硬编码抬升 20  腕目标 z 匀速 +10cm, 策略仍可微调保持握持
hold 40       腕不动, 判成功
= 140 步 @20Hz
```

**成功判据**：hold 段内 抬升 ≥8cm 且 ≥2 指尖接触，占 hold 步数 ≥95%。
（门槛 8cm < 指令 10cm：手指柔顺必然微滑，要求零滑移不合理。）

**动作 28 维**（残差，叠加在参考帧上）：腕 Δpos(3, ±2cm) + 腕 Δ轴角(3, ±0.05rad) + 手指 Δq(22, ±0.1rad)

---

## 5. 模型

```
obs(144) ⊕ priv嵌入 z(8) ⊕ PointNet特征(64) ─► actor MLP[256,128] ─► mu(28)
                                                              sigma(28) 状态无关
obs(144) ⊕ priv(7) ⊕ PointNet特征(64) ────────► critic MLP[256,128] ─► V
点云 (N,64,3) 腕系 ─► PointNet(逐点MLP+max池化, 置换不变) ─► 64 维, actor/critic 共享
```

- **obs 144**：本体44 + 腕13 + 物体13 + 参考37 + 相位4 + 指尖接触5 + 上步动作28
- **priv 7**（critic 专用原始值，actor 只拿 8 维 tanh 嵌入）：质量1 + 摩擦1 + 指尖力5
- 参数约 15 万；开点云后 FPS 约 11,500（关点云 12,600）

### 关键超参（`agents/ppo_correction.yaml`）

| 参数 | 值 | 为什么 |
|---|---|---|
| `learning_rate` / `max_lr` | 3e-4 / **3e-4** | max_lr=lr ⟹ 调度器**只能减速不能加速**，见 §6 |
| `min_lr` | 3e-5 | 别塌到学不动 |
| `kl_threshold` | 0.05 | 补偿 sigma 缩小带来的 KL 放大 |
| `critic_warmup_iters` | 50 | 前 50 epoch 只训 critic |
| `sigma_init` / `sigma_floor` | 0.1 / 0.02 | 旧默认 exp(0)=1.0 会让策略疯狂抖腕 |
| `mu_init_scale` | 0.01 | mu≈0 ⟹ 初始策略 = 零残差 = 可用参考 |

---

## 6. 🔴 踩过的坑（改之前先读）

### 6.1 动作尺度 × 探索噪声 = 抓握杀手
残差界 0.15m × 训练实测 sigma 0.76 = **每步 ±11.4cm 腕抖动** → 接触率 0.1%、物体被打飞 37cm。
`phys_ablation.py` 实测：**零残差回放本身能抓**（53% 时间 ≥2 指接触），但 sigma=0.05 就把接触打到 3.6%。
**容忍度是毫米级。** 想放大残差界，先想清楚 sigma×界 是多少。

### 6.2 策略必须从"可用解"起步
`models.py` 原本只把 bias 清零，`mu` 权重是随机初始化 ⟹ **初始策略是 obs 的随机投影，不是零残差**。
必须 `mu_init_scale`，否则一开始就偏离好解。

### 6.3 `AdaptiveScheduler` 两侧都会失控
对称调度：KL 太大减速、太小加速。
- `kl_threshold=0.02` → lr 塌到 1e-5，策略冻死
- 改 0.05 → KL 长期低于加速线，lr 每轮 ×1.5 冲到 5.1e-3，把已学到的策略打烂

**修法：`max_lr` = 初始 lr（非对称信赖域）。** 残差修正需要慢而稳，不需要自动提速。
另注：warmup 期 actor 不动 → KL≈0 → 调度器会顶到 max_lr，所以 warmup 期**同时跳过 lr 调度**。

### 6.4 显存泄漏（已修，别改回去）
`ppo.py` 的 `a_losses.append(a_loss)` 等**没 detach**，把每个 minibatch 的整张 autograd 图钉住，
`mini_epochs(5) × minibatches(4) = 20` 张图同时驻留。MLP 时每张几 MB 不显眼，
**接上 PointNet 后每张 243MB → 泄漏约 4.9GB，直接 OOM**。已改成 `.detach()`。

调试显存：`MEMDBG=1` 环境变量会在前 2 个 epoch 打印每个 minibatch 的显存增量。

### 6.5 `grasp_only` 的三处静默失效点
改这块时注意，不改会**静默**出错（不报错但成功率永远 0）：
- `_ref_t` 钳在 `grasp_end` 后，原 `in_hold = _ref_t() >= L-1` **永远不成立** → `hold_ok` 恒 0
- `r_imit` 的腕参考要加上抬升量，否则"提上去"本身被模仿项扣分
- `lam_traj=0`（抬升期物体本就该离开桌面参考位）

### 6.6 杂项
- **`ls | tail` 是字典序**（`ep_900` 排在 `ep_3000` 后面）。取最新 ckpt 用 `last.pth`，或数值排序。
- **`pkill -f` 会匹配到自己的命令行**，杀进程用 PID。
- **`kill` 后要复查**：Isaac 进程可能训练循环停了但卡在关闭流程里，继续占着几个 GB 显存，
  需要 `kill -9`。别把 `pgrep` 的输出误读成"已停止"（踩过）。
- **两个 Isaac 进程清理撞车**会触发 `carb::tasking::Mutex "Recursion not allowed"` 崩溃，
  杀完上一个**等 10 秒**再起下一个。
- **`configclass` 里不能放模块对象**（`import os as _os` 会导致 `cannot pickle 'module' object`）。

---

## 7. 数据现状与已知问题

### 数据布局
```
TrainingData/<dataset>/<obj>/
  ocir_sequence/    human_demo.npz, object.obj        ← 重建人手 + 物体 mesh
  grasp_pose/       grasp_pose_*.json, summary.json   ← BODex/cuRobo 优化抓姿
  curobo_traj/      trajectory.npz/.json, isaac_sim/  ← cuRobo 规划轨迹 + 物理验证
  reconstruction/   world_fused.npz 等
  cache/            object.usd, human_fq_20hz.npz     ← 自动生成
```

### ⚠️ 已知问题（当前结果建立在绕过它们之上，不是解决了）

1. **pp0 的 8 个 GraspPose 候选全部 `ok=false`**。策略是自己找到了可行抓法（实测学成
   **拇指+中指+无名指+小指四指抓，食指基本不参与**），不是抓姿变好了。换物体不保证还行。

2. **摩擦模型不匹配**：GraspPose 规划的 11 个接触点（5×DP + 5×PP + palm）**没有一个是
   `*_elastomer`**，而 RL env 只给 5 个 elastomer 高摩擦(9.0)，其余 0.6。
   OCIR 验证器是整手 26 个碰撞体全 3.0 —— 这就是"同一条轨迹在验证器成功、在 RL 失败"的原因之一。

3. **人手重建有硬伤**：物体起飞时腕还在 28–30cm 外，而腕→指尖最大伸展只有 17.3cm。
   **重建的手根本没碰到物体**，上游 HaWoR 腕位置误差约 20cm。
   ⟹ `pp0_human` 骨干当前不可用；要走"修正 noisy 人手"必须先修上游，或走"移动物体去就手"的方案。

4. **OCIR 验证器的 `ok:true` 是运动学可行性证明，不是动力学可执行性证明** ——
   它的手是 `PhysicsFixedJoint` 锚定的运动学搬运（接触零退让、无限有效力），
   RL 侧是浮动根 wrench-PD。拿它当"RL 的成功案例"是范畴错误。

---

## 8. 排查套路

```bash
# 参考轨迹本身行不行? (最该先做的一步)
$PY -m rl_rebuild.correction.eval_policy --zero_action --clip <clip> --num_envs 2048 --headless

# 探索噪声是不是元凶?
$PY -m rl_rebuild.correction.phys_ablation --clip <clip> --num_envs 64 --headless --action_noise 0.1

# 训练出来的 sigma 实际是多少? (不是配置值!)
$PY -c "import torch;d=torch.load('<ckpt>',map_location='cpu',weights_only=False)
sd=d.get('model',d)
[print(k, torch.exp(v).mean()) for k,v in sd.items() if 'sigma' in k]"

# 策略在优化哪一项? 看 TensorBoard 的 ep_rew/* 分解, 各项加起来 × 回合步数 = return
```

**读指标的两个陷阱**：
- 训练期 `success_rate_t0` 的分母是"这一步恰好 reset 的 env 数"，单点极不可靠
  （只有 1 个 env reset 时它非 0 即 1）。**要用 `eval_policy.py`。**
- `env.extras` 不清空，某次 reset 没算的键会把**上次的旧值**重复写进 TensorBoard
  （`success_rate_rsi` 在 rsi_prob=0 后仍显示 1.000 就是这个）。

---

## 9. 输出物

| 位置 | 内容 |
|---|---|
| `logs/correction_<clip>_<tag>/<时间戳>/` | `stage1_nn/`(ckpt) `stage1_tb/`(TensorBoard) `videos/` |
| `videos_compare/` | 对比视频：OCIR验证器 / 零残差 / 失败策略 / 训练后成功 |
| `data/` | 消融曲线 npz |
