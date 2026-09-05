# Sweep2 Main Ablation Record

更新时间：2026-09-03。本文件只记录当前Sweep2主消融的设计、启动方式、运行产物、结果与解释。完整算法以`Codex_tasks.md`为准；新Sweep轨迹迁移方法以`SWEEP_TRAJECTORY_PLAYBOOK.md`为准。

## 1. 研究问题与三组方法

本实验比较object pose/reference confidence与human motion shaping对Sweep residual RL的影响。Gate4（`fully_inside`）是唯一成功定义；Gate3浅进入不计成功。

| Method | Confidence | Human shaping | 含义 |
|---|---|---|---|
| `full` | reconstruction confidence | ON | 完整方法 |
| `wo_human` | reconstruction confidence | OFF，`w_hand=0` | 关闭human shaping |
| `wo_conf` | confidence全部置1 | OFF，`w_hand=0` | 关闭confidence并保持human shaping关闭 |

按当前论文命名，`wo_human`作为human pose/shaping消融，`wo_conf`作为confidence消融。实现上`wo_conf`相对`full`同时关闭human shaping，因此它是复合配置；结果仍按用户指定作为`w/o conf`报告，但在机制解释时不能把二者差异严格归因于confidence单一因素。

Confidence在代码中同时影响：

1. residual action每步增量和累计偏移上界：高confidence收紧探索，低confidence放宽探索；
2. reference tracking cost的容忍区间：低confidence允许更大的位置/姿态偏差；
3. 完整方法中的human shaping权重由confidence档位决定；高confidence时该项为0，中低confidence时才鼓励实际关节速度方向与重建人手运动方向一致。

`wo_conf`把confidence全部置1，因此残差范围和tracking tolerance始终使用最高confidence设置；由于该组同时设置`w_hand=0`，不产生human shaping reward。

## 2. 严格控制的训练协议

三组均从随机Actor和Critic初始化，完全不使用expert transition预热：

```text
Actor offline BC epochs       = 0
Critic offline regression     = 0
transition normalization data = none
warmup_transitions            = []
```

命令行仍传入25 mm、40 mm、full15和canonical failure文件，只为保持训练入口参数兼容；`warmup_role=none`保证它们没有进入Actor、Critic或observation normalization。三个`world.json`均记录`warmup_transitions=[]`，`bc_summary.txt`均记录Actor/Critic samples为0。

共同设置：

- seed 42；1024 parallel environments；24M agent steps；
- Actor observation 191维，Critic privileged state 22维，双臂residual action 14维；
- 前80 control steps为scripted reference prefix，residual强制为0，并从Actor相关训练统计中排除；
- 前10个PPO epochs只进行在线Critic更新，之后pure on-policy PPO；
- 固定Sweep2 reference、cube起点、物理世界、reward、Gate和网络结构；
- 每3M保存checkpoint和训练窗口metrics；每6M生成一次deterministic录像；
- mouth-floor penalty和`-3 mm`硬失败规则保持一致。

因此正式三组之间的配置差别只有`ablation_settings.py`中的method开关，没有离线数据构成或预热轮数混杂。

## 3. 启动方式与运行位置

实际入口统一为：

```text
tasks/Sweep/2/C_Wiring/train_sweep.py
```

核心公共参数：

```text
--num_envs 1024
--seed 42
--max_agent_steps 24000000
--actor_epochs 0
--critic_epochs 0
--record_every_steps 6000000
--headless
```

三组分别只改变`--method`、run名和artifact prefix：

| Method | GPU / tmux | Run directory | Checkpoint prefix |
|---|---|---|---|
| `full` | GPU1 / `sweep2_nooffline_full_gpu1_20260903` | `logs/Sweep2_ablation_nooffline_full_seed42_20260903/` | `Sweep2AblationNoOfflineFull__20260903_policy` |
| `wo_human` | GPU0 / `sweep2_nooffline_wo_human_gpu0_20260903` | `logs/Sweep2_ablation_nooffline_wo_human_seed42_20260903/` | `Sweep2AblationNoOfflineWoHuman__20260903_policy` |
| `wo_conf` | GPU1 queue / `sweep2_nooffline_wo_conf_queue_gpu1_20260903` | `logs/Sweep2_ablation_nooffline_wo_conf_seed42_20260903/` | `Sweep2AblationNoOfflineWoConf__20260903_policy` |

等价启动命令模板：

```bash
CUDA_VISIBLE_DEVICES=<gpu> RL_ISAAC_LOCK_DIR=<task-owned-lock> \
/home/msc-auto/miniconda3/envs/isaac/bin/python -u \
  tasks/Sweep/2/C_Wiring/train_sweep.py \
  --name <run-name> --artifact_prefix <prefix> --method <method> \
  --num_envs 1024 --seed 42 --max_agent_steps 24000000 \
  --actor_epochs 0 --critic_epochs 0 --record_every_steps 6000000 \
  --expert25 <25mm.npz> --expert40 <40mm.npz> \
  --expert_full <full15.npz> --failure <canonical_failure.npz> --headless
```

必须在唯一命名tmux中启动，并在启动前检查GPU利用率、显存、进程所有者和任务锁；不得停止或修改其他用户进程。实际完整命令保存在各run目录的`launch_train.sh`。

## 4. 当前结果：3M网格训练窗口

以下为每个checkpoint保存时对应窗口的Gate4率。用户已取消固定50回合评测要求，因此本轮直接使用现有训练窗口数据。各窗口完成episode数不同，数字适合描述学习曲线和选择论文观察节点，但不是固定测试集或多seed统计。

| Checkpoint | `full` | episodes | `wo_human` | episodes | `wo_conf` | episodes |
|---:|---:|---:|---:|---:|---:|---:|
| 3M | 25.00% | 8 | 17.65% | 17 | 21.57% | 153 |
| 6M | 77.03% | 74 | 0.00% | 23 | 64.10% | 78 |
| 9M | 66.37% | 113 | 3.12% | 32 | 64.37% | 87 |
| 12M | 66.29% | 89 | 37.31% | 67 | 71.82% | 110 |
| 15M | 78.99% | 138 | 67.90% | 81 | 58.33% | 96 |
| 18M | **85.71%** | 140 | 64.38% | 73 | 76.00% | 100 |
| 21M | 74.07% | 162 | 74.65% | 71 | 83.06% | 124 |
| 24M | 80.42% | 143 | 65.79% | 76 | **88.98%** | 127 |

24M最终窗口排序为：

```text
wo_conf 88.98% > full 80.42% > wo_human 65.79%
```

这不是最终策略优劣的充分证据：只有单seed，且各方法窗口episode数不同。它说明`wo_conf`在本次训练后期达到了最高窗口值，而不是证明confidence机制降低最终泛化性能。

## 5. 论文重点节点：18M

18M是本轮最适合展示完整方法优势的预先指定比较节点：此时`full`达到自身全程最高Gate4率，同时对两条消融都保持正领先。

| 18M metric | `full` | `wo_human` | `wo_conf` |
|---|---:|---:|---:|
| Gate1 | 100.00% | 100.00% | 100.00% |
| Gate2 | 100.00% | 95.89% | 99.00% |
| Gate3 | 86.43% | 67.12% | 77.00% |
| Gate4 / fully-inside | **85.71%** | 64.38% | 76.00% |
| completed episodes | 140 | 73 | 100 |
| successful episodes | 120 | 47 | 76 |
| success mouth clearance | **11.53 mm** | 6.99 mm | 7.00 mm |
| right residual usage | 0.174 | 0.260 | **0.115** |
| left residual usage | 0.145 | 0.210 | **0.123** |
| human-shape reward/step | 0.01059 | 0 | 0 |
| tracking reward/step | -0.01216 | **-0.00596** | -0.02165 |

18M的主要差值：

```text
full - wo_human = +21.33 percentage points
full - wo_conf  =  +9.71 percentage points
```

### 5.1 为什么`full`在18M最好

三组Gate1均为100%，Gate2也接近饱和，因此差距主要不是“场景没有ready”或“扫把从未接触方块”，而是从真实推动到完整进入簸箕的Gate2→Gate4转换效率不同。`full`的Gate3/Gate4分别为86.43%/85.71%，浅进入到完整进入只损失0.72个百分点，说明一旦进入入口，动作与簸箕几何更容易继续形成fully-inside终态。

完整方法在中低confidence片段才提供human direction shaping。这不是直接奖励碰到入口，而是鼓励机器人实际关节增量与重建人手运动方向一致，为稀疏的完整进入目标提供时序先验。18M时`full`双臂residual usage明显低于`wo_human`，同时成功episode的mouth clearance由约7 mm提高到11.53 mm。这与如下机制解释一致：human shaping减少了大幅、相互抵消的残差搜索，使左右臂更早形成协调的扫入/承接运动，并让簸箕保持更安全的离桌高度。

这里应使用“与机制解释一致”而不是“已经证明因果”，因为当前只有一个seed，指标也是成功episode条件统计。

### 5.2 为什么`wo_human`较低

`wo_human`仍保留confidence-dependent residual bounds和tracking tolerance，但没有方向性的人手运动奖励。它在18M的Gate2已有95.89%，却只有64.38%达到Gate4，表明主要困难发生在接触后的动作组织，而非接触获取本身。其左右残差使用量是三组最高，支持策略需要更大范围自行搜索交互动作；更大的探索并没有在18M转化为更高fully-inside率，且成功mouth clearance更低。

### 5.3 为什么`wo_conf`介于两者之间

`wo_conf`在18M达到76%，高于`wo_human`但低于`full`。全1 confidence把所有时刻都视为高置信度：残差步长和累计范围始终较小，reference tracking tolerance也最严格。其residual usage最低，说明策略更贴近IK reference。对于当前这条质量较高且与任务相符的Sweep2 reference，强制紧跟reference本身就是有效先验，因此即使没有human shaping也能比`wo_human`更稳定。

但其tracking cost最负（-0.02165/step），说明“全1”并不等于实际跟踪误差最小；它同时把容忍区间收紧，较小误差也更容易被计罚。`wo_conf`低于`full`的合理解释是：统一收紧虽然抑制无效探索，却也取消了低confidence区间需要的自适应自由度和方向性human guidance，导致Gate2→Gate4转换仍弱于完整方法。

### 5.4 三组关系的核心直觉

这组三折结果可以用“动作自由度是否配有方向引导”来统一理解：

```text
full:
低 confidence → 放宽动作空间
同时提供 human direction shaping
→ 有自由度，也知道往哪里修正

wo_human:
低 confidence → 放宽动作空间
但没有 human direction shaping
→ 有自由度，却不知道如何有效使用

wo_conf:
始终按高 confidence 处理
→ 不提供过多动作自由度
→ 主要依靠高质量 IK reference 完成任务
```

因此，在当前这条reference质量较高的Sweep2轨迹上，18M结果呈现出：

```text
有方向引导的大范围修正（full）
> 紧跟高质量 reference（wo_conf）
> 无方向引导的大范围搜索（wo_human）
```

这一排序也解释了为什么`wo_conf`反而优于`wo_human`：`wo_human`保留了低confidence带来的较大residual步长、累计范围和宽松tracking tolerance，却移除了帮助策略使用这些额外自由度的human direction shaping；`wo_conf`虽然缺少human shaping，但它通过始终收紧residual范围，让策略主要停留在已经较可靠的IK reference附近。18M的residual usage与成功率同时支持这一解释：`wo_human`使用了更大的双臂residual，却没有得到更高的Gate4率。

这进一步说明confidence与human shaping在当前设计中是配套机制：前者决定“允许策略偏离reference多少”，后者在被放宽的区间提供“应该朝哪个方向修正”的先验。移除human shaping后，保留自适应的大动作空间未必比直接依赖高质量reference更有利。不过，这仍是基于单seed训练窗口和诊断量得到的机制解释，不能替代专门的单变量因果实验。

## 6. 18M到24M的后期变化

| Method | 18M | 21M | 24M | 18→24M |
|---|---:|---:|---:|---:|
| `full` | 85.71% | 74.07% | 80.42% | -5.29 pp |
| `wo_human` | 64.38% | 74.65% | 65.79% | +1.41 pp |
| `wo_conf` | 76.00% | 83.06% | 88.98% | +12.98 pp |

`full`在18M达到峰值，21M下降11.64个百分点，24M恢复6.35个百分点但未回到峰值。这更像on-policy训练在已经较高成功率区域的策略分布漂移和窗口采样波动，而不是持续单调提升。24M成功mouth clearance为14.50 mm，高于18M，说明后期下降并没有表现为重新依赖压簸箕/贴桌捷径。

`wo_human`从18M到21M短暂升至74.65%，24M又回落到65.79%。它在后期没有形成稳定、持续的提升；24M成功mouth clearance降至2.94 mm，且mouth-floor penalty绝对值增大，提示最终策略更靠近桌面边界，安全裕度弱于完整方法。

`wo_conf`则从76.00%连续升到83.06%和88.98%。一种合理解释是：固定的小残差范围牺牲了早中期灵活性，但使后期优化更稳定；在高质量reference上，策略经过更多on-policy更新后逐渐找到紧邻reference的有效解。另一种不能排除的解释是单seed和窗口构成差异造成的波动。因此不能仅凭24M断言confidence有害。

从论文叙事看，18M回答“在相同训练预算的一个高性能节点，完整机制是否提高学习效率与动作质量”；24M则揭示“消融策略可能通过更长训练追上甚至超过单seed窗口性能”。两者应同时报告，不能只展示18M而隐藏后期反转。

## 7. 结论与使用边界

本轮可支持的结论：

1. 在无任何离线轨迹预热的严格一致设置下，完整方法在18M达到85.71%，比`wo_human`高21.33个百分点、比`wo_conf`高9.71个百分点。
2. Human shaping的主要可见贡献出现在学习速度和接触后Gate2→Gate4转换；`wo_human`在6M和9M尤其困难，之后才逐步追赶。
3. 完整方法18M同时表现出更低的residual usage和更高的成功mouth clearance，说明成功不仅更多，动作也更接近有结构、较安全的交互。
4. `wo_conf`在24M达到本轮最高窗口率88.98%，说明confidence优势不是“任何训练时刻都必然更高”；它更可能影响早中期探索、数据效率和动作先验，而最终上限需要多seed独立评测确认。

本轮不能支持的强结论：

- 不能把`full`与`wo_conf`的差异单独归因于confidence，因为`wo_conf`也关闭了human shaping；
- 不能把训练窗口百分比当作固定50回合或512回合最终测试；
- 不能从单seed判断统计显著性或跨Sweep轨迹泛化；
- 不能仅根据18M挑点宣称完整方法最终性能最高，而不同时披露21M/24M变化。

后续若进入正式论文表格，建议保留18M作为训练效率/机制分析节点，同时增加多seed、固定episode deterministic evaluation；checkpoint选择规则必须在看测试结果前固定。

## 8. 已废弃的非可比实验

2026-09-03较早一轮曾使用不一致的expert warmup：历史`full`与消融组的数据组成不同，出现`wo_human`在固定50回合中显著超过历史`full`的现象。由于预热不受控，该结果不能归因于human shaping，已从正式消融排除；对应消融产物已删除，只保留历史说明。当前第2–7节的无预热实验完全替代该轮作为正式主消融记录。
