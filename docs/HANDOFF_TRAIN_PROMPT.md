# 交接 Prompt — `pick_lift` 训练执行

> 复制下面 `---` 之间的全部内容给接手的 agent。
> **代码已在 GPU 上冒烟通过**（2026-08-02，Grasp3 / 候选 31_7 / yaw 215）：
> 抓取口径 obs 146 ✅、接近口径 obs 156 ✅、三个相位都到过、150 步无 NaN、
> 前馈自检跟踪差 0.17~1.14cm ✅。冒烟过程中修掉了 3 个 bug（见 §2 表末）。
> Step 0 仍建议在新机器上重跑一次（换机器/换 clip 都可能暴露新问题）。

---

你接手一个 IsaacSim + PPO 的机器人抓取项目，任务是执行已经设计好的训练并汇报结果。
**设计已经定稿，你的职责是执行、判读、如实汇报，不是重新设计。**

## 0. 先读这四份（按顺序，别跳）

| 文档 | 读它拿什么 |
|---|---|
| `CLAUDE.md` | 机器红线、必须加的环境变量、会立刻犯的错 |
| `docs/PLAN_PICK_LIFT.md` | 任务定义、8 条锁定决定 D1~D8、全部实测数据（§7）、参数的出处 |
| `docs/DESIGN_LOOP.md` §2.10/§2.11 + §5 首两行 | 每条设计的**证伪信号**——判读时对着这张表读 |
| `docs/GRASPPOSE_SCREENING.md` | 候选筛选的三关判据（H1/H2；H3 已作废） |

**一句话背景**：机器人从固定站姿出发 → 接近桌上物体 → 抓稳 → 抬 1cm 验证。
两个先验：重建的人手轨迹（提供"怎么走"）+ Dexonomy GraspPose（提供"终点长什么样"）。
训练顺带产出逐点"抓握性分数图"，将来喂 affordance 模型。

## 1. 环境

```bash
PY=/home/lyh/luhr/MagicSim/.venv/bin/python   # 唯一带 isaacsim+isaaclab 的解释器
cd <repo>                                      # 所有命令都在仓库根跑
export PYTHONPATH=. SHARPA_WANDB=0             # 不设 SHARPA_WANDB=0 会弹交互登录卡死
```

**红线**（每条都是事故换来的，见 `CLAUDE.md`）：
1. 训练**必须** `--headless`，漏了 1024 env 会被 OOM killer 杀（退出码 137）。
   核验：日志里 grep `experience file`，必须是 `.headless.kit`。
2. **同一张卡同时只能一个 Isaac 进程**（`gpu_guard` 会排队）。杀进程后**等 10 秒**再起下一个，
   否则 carb mutex 崩溃。杀完要复查显存真的释放了（可能要 `kill -9`）。
3. 取最新 ckpt 用 `last.pth`。`ls | tail` 是字典序（`ep_900` 排在 `ep_3000` 后面）。
4. **`best.pth` 无意义**（课程涨价前的高分），一律用 `last.pth`。
5. 成功率只认 `tasks/pregrasp/eval.py` 的确定性评测。TensorBoard 的 `success_rate_ema`
   含随机化脚手架 + 探索噪声，**系统性低估**（实测 TB 70% ↔ 确定性 99.95%）。
6. **一次只改一个参数**。要改设计先读 `docs/DESIGN_LOOP.md` §4 的纪律。

## 2. 这次改了什么代码（你要验证的就是它）

| # | 改动 | 文件 |
|---|---|---|
| ① | 有 prior 时**关掉 `pregrasp_align`** + **钉死物体 yaw** —— 统一走 `apply_grasp_prior()` | `cfg.py`, 各入口 |
| ② | 接近段：**前馈参考增量**（零动作=复现人手速度剖面）+ 残差"远松近紧"缩放 | `env.py:_pre_physics_step` |
| ③ | 两个新奖励：**对齐势差分** `r_align` + **残差用量罚** `r_imit`（衰减到 0） | `env.py:_get_rewards` |
| ④ | **接触分数图**：verify 斜坡到顶取快照、按力占比加权、(s,n) 计数 + unknown 掩码 | `env.py` + `train.py` 钩子 |
| 修 | `_align_err` 无 prior 时不再崩；起步分支改用持久标记（超时回合曾被误分桶） | `env.py` |
| 修 | obs 维度**按需**扩展（146 抓取 / 156 接近），否则历史 ckpt 全部加载失败 | `cfg.py`, `env.py` |
| 修 | `OBS_APPROACH_EXTRA` 移到模块级 —— `@configclass` 会剥掉类里无类型注解的辅助属性 | `cfg.py` |
| 修 | **`d_pos` 改成"腕位偏离 prior 抓握腕位"**。原来用"合拢中心→接触质心"，实测手停在 prior 位姿上该值仍有 **3.42cm**，而 ε_pos=1.04cm ⟹ **相位永远切不过去**，且对齐势的极小点不在 GraspPose 上 | `env.py:_align_err` |
| 修 | 首步 `prev_wrist_pos` 无效会打出 106cm 的假残差读数；`res_step_cm` 只统计接近段 | `env.py` |

设计要点（判读时要理解的三条）：
- **人手轨迹的绝对位置不可信**（实测系统性偏 14~16cm，`PLAN §7.7`），所以它只以**增量**形式进入：
  前馈用差分、模仿罚罚"本步残差用量"。**没有管壁、没有位置模仿罚**。
- **φ = 参考帧号 / gs 是外生时钟**，策略动不了。奖励权重只能挂在它上面，不能挂在"离物体多远"。
- **`w_align` 恒定**。随相位变会破坏势函数的 telescoping，策略能"早退后进"无限刷分。

## 3. Step 0 — 冒烟（**强制**，2 分钟）

```bash
SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.smoke --headless \
    --clip Grasp3 --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/31_7.npz \
    --prior_yaw 215 --num_envs 8            # 先不加 --approach

SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.smoke --headless \
    --clip Grasp3 --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/31_7.npz \
    --prior_yaw 215 --approach --num_envs 8  # 再加接近段
```

必须看到：
- `[prior] 物体 yaw **钉死** 215°`，且该角 **IK err < 1cm**
- **没有** `[replay_grasp] PreGrasp 对齐(...)` 那行（关掉了才对）
- `--approach` 那次要有 `[approach] 开: 参考 gs=32 ...` 和 `[score] 接触分数图已开: 512 个表面点`
- 无 NaN、能走到 verify 阶段、`✅ 管线通过`

**冒烟挂了先修冒烟，不要带着报错开训练。** 常见位置：obs 维度、张量 shape、
`_set_arm_center` 的 mask 广播、`q_ref[t0]` 的越界。

### Step 0b — 接近段自检（**换 clip 必跑**，2 分钟）

```bash
SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.check_ff --headless \
    --clip Grasp3 --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/31_7.npz --prior_yaw 215
```

它让全部 env 从 t=0 起步、发零动作跑完接近窗口，验证**前馈接对了没**。
Grasp3 / 31_7 的基准读数（2026-08-02 实测）：

| 步 | φ | 腕位移 | 参考位移 | 跟踪差 | d_pos | d_rot |
|---|---|---|---|---|---|---|
| 8 | 0.16 | 2.86cm | 4.00cm | 1.14cm | 12.47cm | 139.0° |
| 20 | 0.53 | 18.39cm | 17.97cm | 0.42cm | 13.05cm | 96.6° |
| 32 (=gs) | 0.91 | 13.56cm | 13.75cm | 0.19cm | 13.60cm | 79.4° |
| 52 | 1.00 | 12.33cm | 12.70cm | 0.37cm | 13.29cm | 78.7° |

**读法**：
- **跟踪差 < ~1cm** ⟹ 前馈接对了，零动作复现了人手的形状与速度剖面。
  若跟踪差与参考位移同量级（手在原地不动）⟹ 前馈**没接上**，去看 `dexmate_env.py:384-392`。
- φ=1 之后腕位移冻结 ⟹ 前馈在 gs 之后正确停止。
- **末端 d_pos 13.3cm / d_rot 78.7° 就是策略必须自己走完的缺口**（人手轨迹的系统偏差）。
  预算核算：残差远端 15mm/步 ⟹ 13.3cm 需 ~9 步；窗口 gs+20=52 步，**约 6 倍余量**。
  姿态 78.7° 分摊到 52 步 = 1.5°/步，也在残差权限内。
- 零动作**不会**切到抓取相位（16/16 仍在接近），最后 timeout —— 这是对的，无强制切换。

## 4. Step 1 — 8 条抓取基线（yaw 钉死口径，不带接近段）

**为什么先跑这个**：yaw 钉死（D1）是全新口径，历史上的 100% 是在自由 yaw 下拿到的。
没有这条基线，接近段训崩了你分不清是 D1 改坏了还是接近段设计错了。
同时它产出分数图的首批正负样本，并前瞻检验两条筛选判据。

每条约 1 小时（8 卡可全部并行）。命令模板：

```bash
SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.train --headless \
    --clip Grasp3 --name G3_<TAG> --num_envs 1024 \
    --prior_npz tasks/pregrasp/priors/Grasp3_candidates/<TAG>.npz \
    --prior_yaw <YAW> --max_agent_steps 20000000
```

| # | `<TAG>` | `<YAW>` | Gate2 | 这条回答什么 |
|---|---|---|---|---|
| 1 | `8_5` | 215 | ✅ Q\*=+0.259 pads\*=4.95 (6/6档) | 正样本（稳）|
| 2 | `19_4` | 215 | ✅ +0.254 / 4.41 (6/6) | 正样本（稳）|
| 3 | `31_7` | 215 | ✅ +0.162 / 4.00 (1/6) | **H1 对照·刚过线** |
| 4 | `14_7` | 215 | ✅ +0.111 / 4.00 (1/6) | 正样本（勉强）|
| 5 | `8_2` | **225** | ❌ +0.224 / **3.92** (0/6) | **H1 对照·刚没过线** |
| 6 | `21_3` | 215 | ❌ **−0.093** / 4.63 (1/6) | **H2 对照**：垫数够但质量分为负 |
| 7 | `47_2` | 215 | ❌ −0.122 / 2.00 | 负样本 |
| 8 | `32_4` | 215 | ❌ −0.024 / 2.00 | 负样本 |

**判读**（对着 `DESIGN_LOOP §2.11` 的证伪信号读）：

| 现象 | 说明什么 |
|---|---|
| 1/2 号到不了 ~100% | **D1（yaw 钉死）本身有问题**，先查这个，别往下走 |
| 3 号成 & 5 号败 | `pads* ≥ 4` 这条线**有分辨力**，判据得到前瞻验证 |
| 3 号败 或 5 号成 | 这条线在 3.92↔4.00 之间**没有分辨力**，`GRASPPOSE_SCREENING` 的 H1 要重定 |
| 6 号出现"回避塌缩"（`grasp/got_candidate`→0、`term/timeout`→1.0、其他 term 全 0、`ep_rew/cent_income`≈0） | **H2（`Q*>0`）得到前瞻验证**，这是台账里复发过 4 次的失败模式 |
| 6 号反而训成了 | H2 不成立，回去查 `cent_income` 的机理推导 |

⚠ **诊断时不要读 `grasp/pads_touched_latch`**（它是"曾碰到过"的 latch，回避塌缩时也能有 4+），
看 `grasp/pads_now`（同时接触数）才准。这个坑让 Grasp1 的判读走过一次弯路。

跑完每条都要：
```bash
SHARPA_WANDB=0 PYTHONPATH=. $PY -u -m tasks.pregrasp.eval --headless \
    --clip Grasp3 --checkpoint logs/G3_<TAG>/<ts>/stage1_nn/last.pth \
    --grasp_prior tasks/pregrasp/priors/Grasp3_candidates/<TAG>.npz \
    --prior_yaw <YAW> --steps 300
```
（`-u` 必须，否则输出被缓冲吞掉。正式成绩看**口径 A**。）

分数图自动落在 `logs/G3_<TAG>/<ts>/score_map.npz`（每 100 epoch 一次 + 收尾一次）。

## 5. Step 2 — 接近段首训（Step 1 的 1/2 号确认 ~100% 之后才开）

```bash
SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.train --headless \
    --clip Grasp3 --name G3_approach_8_5 --num_envs 1024 \
    --prior_npz tasks/pregrasp/priors/Grasp3_candidates/8_5.npz \
    --prior_yaw 215 --approach --max_agent_steps 40000000
```

**必看的指标**（都是新加的）：

| 标量 | 健康的样子 | 不健康说明什么 |
|---|---|---|
| `sr/from_grasp` | **必须仍 ~100%** | 掉了 = 改坏了底层，与接近段无关，**先修这个** |
| `sr/from_approach` | 从 0 慢慢涨 | 一直 0：看下面两行归因 |
| `approach/d_pos_cm` / `d_rot_deg` | 单调下降，最终 <1.04cm / <2.18° | 平台不降 = 够不到，查残差权限或候选 |
| `approach/res_used_cm` | 早期小、晚期大 | **一上来就顶满 = 模仿罚太轻**，策略从第一步就无视人手轨迹 |
| `curr/direct_grasp_prob` / `curr/approach_t0_max` | 随成功率平滑退火 | 不动 = `sr_slow` 没涨起来 |
| `term/timeout` 高但 `d_pos` 在降 | 只是慢 | 加 `approach_extra_steps` |
| `term/timeout` 高且 `d_pos` 平台 | 真够不到 | 查 `dyn_arm_far` / 换候选 |

**接近段的容差曲线要复测**（这是验收指标之一）：训完用 `tasks/pregrasp/tol_curve.py`
重跑一遍，与 `data/tol_curve_Grasp5.json` 里的基线比。**曲线应当右移**——因为抓取相位
这次见过真实的到达误差分布。没右移说明联合训练没起到适应作用。

## 6. 红线：不要做的事

- **不要**为了让候选可达去调机器人站姿或转物体 yaw（D1/D2，摆放忠于视频是任务前提）
- **不要**把 `w_align` 改成随相位变化（会被刷分，`DESIGN_LOOP` A3）
- **不要**把 φ 改成基于"离物体距离"（策略能操纵它，A4）
- **不要**给分数图里没接触过的点写 0 分（那是 unknown，必须用 `weight` 掩码，B3）
- **不要**一次改多个参数

## 7. 汇报格式

每条 run 交回：run 名 / 候选 / 步数 / **确定性评测成功率（口径 A）** / 里程碑 /
上表里那几个诊断量的走势 / 你的归因（对应到 `DESIGN_LOOP` 的哪条证伪信号）。

**如实报告**：训崩了就报崩了并附曲线，不要修饰。这个项目的台账里一半的价值来自失败 run
的准确归因。有判据被数据推翻，直接说哪条、被什么推翻。

---
