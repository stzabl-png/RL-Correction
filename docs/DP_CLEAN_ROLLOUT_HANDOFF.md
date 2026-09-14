# 擦盘子 (Clean) 九条 RL ckpt → DP 数据采集 交接 (2026-09-13)

给用我们的 RL ckpt 在 sim 里 rollout 采 DP 数据的同学。
**三条人类视频 (take 3 / 8 / 18) × 三种设定 (Base / A1 / A2) = 九条策略**, 全部 30M 步训完, 有确定性评测、三机位录像、逐步奖惩。

包里有跑通 rollout 所需的 ckpt / 母带 / 先验 / 数据集 / 任务代码 / 复现脚本; 缺的只有 Isaac 栈本身 (见 §1)。

## 0. 先读这一段: 三种设定是什么

同一条擦盘子任务, 三种"参考信号"设定 (对位 Pour 的 Base/NH/NHNC):

| 设定 | 旗 | 含义 |
|---|---|---|
| **Base** | `CLEAN_S2_HAND_REF=1` | 带**人手轨迹**指引 (重建出的人手腕位姿作为额外参考项) |
| **A1** | 全关 | **撤掉人手指引**, 只留物体功能关系 + 掌系相对位姿软罚 |
| **A2** | `CLEAN_S2_CONF_FLAT=1` | A1 再**把重建置信度拍平** (所有帧同权, 不再按 conf 分档) |

三条共用承重项 `CLEAN_S2_SOFT_REL=1 CLEAN_S2_SOFT_W=0.5` (掌系相对位姿软罚) 与最简六项配方 `recipe=min6`。

**★ 结论先说: 人手指引是有害的。** 三条 take 独立坐实:

| 设定 | 合计成功 | take3 | take8 | take18 |
|---|---|---|---|---|
| Base | **9/30** | 0/10 | 3/10 | 6/10 |
| A1 | **30/30** | 10/10 | 10/10 | 10/10 |
| A2 | **30/30** | 10/10 | 10/10 | 10/10 |

Base 的失败全是 `rel` (物体在手里滑出死线), 海绵转角中位 17~49°; A1/A2 都在 5~15°, 零死亡。
**采 DP 数据请用 A1 或 A2, 不要用 Base** —— Base 在 take3 上是 0/10。

## 1. 机器准备 (一次)

- Isaac Sim 5.1 + IsaacLab + conda 环境: 照仓里 `docs/DEPLOY_NEW_MACHINE.md`。**clone 前先装 git-lfs**。
- 环境变量 (每个 shell): `OMNI_KIT_ACCEPT_EULA=YES SHARPA_WANDB=0 PYTHONPATH=<repo>`;
  多卡共享机再加 `RL_ISAAC_NO_GUARD=1` (flock 是按机器不是按卡, 不加会静默排队);
  臂 IK 的 URDF: `VEGA_URDF=<你机器上的 vega_1p_sharpa.urdf>` (不设会去找原作者本机绝对路径)。
- `/tmp/isaaclab/logs` 不可写时要设 `TMPDIR=$HOME/tmp` (复现脚本已内置这个兜底)。

## 2. 把包放到哪

```
<repo>/datasets/clean_tableware/{3,8,18}/   ← 本包 data/clean_tableware/{3,8,18}  (必须放这里)
<repo>/tasks/Clean/3/                        ← 本包 code/tasks_Clean_3/ 覆盖 (或确认仓里版本不旧于它)
任意位置                                      ← 本包 runs/ assets/ common/ 原地不动即可
```
⚠ `datasets` 的位置**不可配置** (`rl_rebuild/correction/clips.py` 里 `_DATASETS` 是相对模块路径硬编码),
必须放进仓的 `datasets/` 下, 否则建场景时找不到 USD/网格。

## 3. 一条命令复现

```bash
bash common/eval10.sh <repo根> <GPU> 2026 Clean18_ablA1_s42
# 九条全跑:
bash common/eval10.sh <repo根> 0 2026 Clean3_ablBase_s42_r2,Clean3_taskS_s42,Clean3_ablCF_s42_r2,\
Clean8_ablBase_s42,Clean8_ablA1_s42,Clean8_ablA2_s42,Clean18_ablBase_s42,Clean18_ablA1_s42,Clean18_ablA2_s42
```
脚本**逐条从各自 `runs/<run>/world.json` 读配方** (clip / 母带 / 先验 / HAND_REF / CONF_FLAT), 不硬编码 ——
曾经硬编码一次, 漏了 Base 臂的 `HAND_REF`, 被世界核对拦下。你要自己写命令就照这个读法。

**已验**: 2026-09-13 在 Denso (RTX 2080Ti) 上用本包复现 take18 两条 (原始训练/评测在另外两台机器):

| run | 原始 (taitan, 09-12) | 本包复现 (Denso, 09-13) |
|---|---|---|
| `Clean18_ablA1_s42` | success 1.000 cov 0.986 travel 195.4cm relp P/S 1.00/1.18cm | success **1.000** cov 0.984 travel 195.5cm relp 0.99/1.18cm |
| `Clean18_ablBase_s42` | success 0.600 | success **0.700** travel 181.9cm relp 2.02/1.41cm |

A1 逐项对上; Base 差 1 个回合 —— 它是边际臂, **±1 回合是正常机器间方差**, 别拿它当对账基准。

## 4. 九条清单

| take | 设定 | run | ckpt sha256(前8) | 母带 | 盘先验 | HAND_REF | CONF_FLAT | 成功 | 覆盖 | 行程 | 盘漂移 | 海绵漂移 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| take3 | Base | `Clean3_ablBase_s42_r2` | `fb0006d1` | clean3_reference_v1h.npz | Clean3_plate_left.npz | 1 | 0 | 0/10 | 0.81 | 79cm | 3.13cm/23.7° | 1.93cm/49.0° |
| take3 | A1 | `Clean3_taskS_s42` | `a844f330` | clean3_reference_v1.npz | Clean3_plate_left.npz | (无此键,=0) | (无此键,=0) | 10/10 | 0.63 | 110cm | 1.82cm/14.3° | 1.21cm/7.7° |
| take3 | A2 | `Clean3_ablCF_s42_r2` | `46691851` | clean3_reference_v1h.npz | Clean3_plate_left.npz | 0 | 1 | 10/10 | 0.55 | 106cm | 1.36cm/8.1° | 1.05cm/9.4° |
| take8 | Base | `Clean8_ablBase_s42` | `37cd518d` | clean8_reference_v1.npz | Clean18_plate_left.npz | 1 | 0 | 3/10 | 0.80 | 110cm | 2.01cm/24.4° | 3.04cm/17.2° |
| take8 | A1 | `Clean8_ablA1_s42` | `f73c237e` | clean8_reference_v1.npz | Clean18_plate_left.npz | 0 | 0 | 10/10 | 0.78 | 190cm | 1.35cm/8.1° | 1.22cm/5.9° |
| take8 | A2 | `Clean8_ablA2_s42` | `d4af8561` | clean8_reference_v1.npz | Clean18_plate_left.npz | 0 | 1 | 10/10 | 0.80 | 204cm | 1.26cm/9.5° | 0.86cm/4.5° |
| take18 | Base | `Clean18_ablBase_s42` | `8ddf4154` | clean18_reference_v1.npz | Clean18_plate_left.npz | 1 | 0 | 6/10 | 0.98 | 184cm | 2.05cm/16.9° | 1.59cm/37.2° |
| take18 | A1 | `Clean18_ablA1_s42` | `6e6012b7` | clean18_reference_v1.npz | Clean18_plate_left.npz | 0 | 0 | 10/10 | 0.99 | 195cm | 1.00cm/8.2° | 1.18cm/6.2° |
| take18 | A2 | `Clean18_ablA2_s42` | `b3f3a62c` | clean18_reference_v1.npz | Clean18_plate_left.npz | 0 | 1 | 10/10 | 0.94 | 211cm | 1.52cm/14.6° | 1.14cm/7.5° |

**两个命名/口径的坑**:
1. `take3` 的 A1 就是我们的现役冠军 `Clean3_taskS_s42` (16/16), 按项目决定**没有重跑**, 所以:
   它的 `world.json` 里**没有 `S2_HAND_REF` 这个键** (那条 run 早于该旗存在), 默认即"关"= A1 语义。
   另外它用的母带是 `clean3_reference_v1.npz` 而 Base/A2 用 `clean3_reference_v1h.npz` ——
   两条带在被读到的键上逐位相同, 但**文件名不同, 别混用**。
2. `take3` 的 A2 目录名是 `Clean3_ablCF_s42_r2` (CF = ConfFlat), 不叫 ablA2。

## 5. 每条 run 目录里有什么

```
runs/<run>/stage1_nn/last.pth        末档策略 (2.9MB, 30M 步)
runs/<run>/world.json                出生世界指纹: 母带路径+sha / 先验 / 物理 / 全部旗 / 判据 / 策略 IO 维
runs/<run>/stage1_tb/events.*        TensorBoard: 训练全程逐 epoch 账本
runs/<run>/eval/*.json               确定性评测 (16 回合的 ev16 + 10 回合的 e10s2026)
runs/<run>/videos/last_r10_{front,side,top}.mp4   三机位回放, 1280x720, 20fps, 1 帧 = 1 控制步
runs/<run>/videos/last_r10_steps.csv/.npz/_summary.json   逐步奖惩 + 逐步状态
runs/<run>/progress_steps.txt        训练步数进度
assets/tapes/*.npz                   四条母带 (clean3 v1 / v1h, clean8 v1, clean18 v1)
assets/priors/*.npz                  GraspPose 先验 (盘 take3/take18 两份 + 洗碗布一份)
data/clean_tableware/{3,8,18}/        数据集 (USD / 网格 / 纹理 / 重建位姿 / 母带源), 与训练时逐字节一致
code/tasks_Clean_3/                   任务代码 (env / train / eval / record / smoke)
launch_logs/                          训练 .log / 评测 _ev16.log / 录制 _rec.log (31 个)
common/eval10.sh                      §3 的复现脚本
common/eval10_taitan_original.sh      当初跑这批评测的原始脚本 (存档对照)
```

看 TB: `tensorboard --logdir runs`

## 6. 逐步奖惩 csv 怎么读

`runs/<run>/videos/last_r10_steps.csv` —— 每行 = 一个 env 的一个控制步 (20Hz):
```
env,step,row,release_row,released,certified,k,gate_ok,can,contact,reward,
r_adv,r_leash,r_bonus,r_act,r_soft,            <- 奖惩分解
e_n_cm,e_xy_cm,                                 <- 海绵在盘规范系的法向/面内误差
dp_plate_cm,dr_plate_deg,dp_sponge_cm,dr_sponge_deg,   <- 物体在**掌系**相对放手瞬间的漂移 (死线判据原料)
within,plate_tilt_deg,gap_mm,coverage,travel_cm,success,died,die_kind,cross_any,dq_re_deg,done,
F_L_thumb..F_L_pinky,F_R_thumb..F_R_pinky       <- 十指垫接触力 (N)
```
`k` 是母带时钟行; `certified` 是"手已接住"的一次性盖章 (连续 10 步掌系漂移 <1cm/5°);
时钟只在盖章后推进。`die_kind`: `rel`=手物相对位姿超线, `tilt`=盘倾覆, `drop`/`table` 见代码。

## 7. rollout 骨架 (照 `code/tasks_Clean_3/C_Wiring/record_task.py`, 每步能读到什么)

```python
cfg = TE.build_cfg(num_envs=N); raw = TE.CleanTaskEnv(cfg)
env = GymStyleEnvWrapper(raw, clip_actions=1.0)
agent = PPO(...); agent.restore_test(ckpt); agent.set_eval()
obs = env.reset()
for t in range(T):
    mu = agent.model.act_inference({"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]})
    obs, rew, dones, infos = env.step(torch.clamp(mu, -1, 1))
    # 可读:
    #   raw.hand.data.joint_pos / joint_vel      全 DOF 实际
    #   raw.q_cmd[:, raw.act_ids]                58 维关节目标 (臂14 + 指44) —— DP 的动作标签候选
    #   raw.object / raw.aux 的 root_pos_w / root_quat_w   盘 / 海绵世界位姿
    #   raw._pad_force_mat("left"/"right")       指垫力
    #   raw._tick                               该步全部奖惩与判据中间量 (record_task.py 落盘的就是它)
```
动作口径: 策略输出是**有界累积残差的速率** (`cum_res += act * step_sz`, 臂 0.004rad/步上限 0.08,
指 0.03rad/步上限 0.6), 真正下发给 PhysX 的是 `q_cmd` = 母带前馈 + 残差。
**做 DP 建议拿 `q_cmd` (绝对关节目标) 当动作标签, 不要拿策略的 58 维残差** —— 后者没有前馈就没有意义。

## 8. 已知限制 (别踩)

- **中途 ckpt 不全**: 本包每条只给末档。完整 epoch 序列我们留了 7 条 (19~47 档),
  `take8` 的 A1/A2 只剩末档, 中途档已不可恢复。要训练曲线上某一点的策略, 先问我们还在不在。
- **世界核对是有意的**: 脚本从 `--checkpoint` 上两级找 `world.json`, 不符就拒跑。
  本包的复现脚本用 `CLEAN_IGNORE_WORLD=1` 跳过路径比对 (因为包内路径与训练时的绝对路径必然不同),
  但**旗与物理仍然逐条从 world.json 还原**, 不要把这两件事混为一谈。
- **机器间方差**: A1/A2 这类 10/10 的臂跨机稳定; Base 这类边际臂 ±1 回合正常。
  如果 A1 在你机器上差出 2 个回合以上, 先怀疑世界 (§1/§2), 不要怀疑 ckpt。
- **盘是倒扣的** (take18/take8): 视频里盘凹面朝下被举着擦, 不是正放。take3 是正放。
- 物理规矩: 物体 0.3kg / 物体摩擦 1.0 / 指垫摩擦 1.0 (`POUR_*` 三个变量, 复现脚本已写死)。
  **漏写会静默回退到默认值, 成绩直接变样**。



## 附: 训练细节在哪看 (2026-09-14 补)
- **Stage-1 抓稳段 (退火课程)**: `exports/clean_stage1_hold_20260914/Clean3_hold_s42/` —— `stage1_tb/` (sr/success, sr/cert, term/drop_*, hold/dev_*, 课程 release_row 的退火轨迹), `train.log`, 31 个 ckpt (0~19M), `videos/`, `README.md`。课程设计与逐日读数见 `tasks/Clean/3/A_Design/DECISIONS.md` §4~§5 (release_row 50→10 按成功率 EMA 单向退火)。
- **Stage-2 九条 (Base/A1/A2 × take3/8/18)**: `exports/clean_3x3_ckpts_20260914/<run>/` —— `world.json`, `stage1_tb/`, `eval/` (eval10 seed2026 + ev16), `videos/`, `stage1_nn/{last.pth, *step_0020M*.pth}` (20M 节点用于统一 20M 口径的对比; take3 Base/A2 无 20M 节点, 见台账 §9.20)。
- 复现命令: `common/eval10.sh <repo根> <GPU> 2026 <run,...>`; 数据集在 `datasets/clean_tableware/{3,8,18}` (LFS)。
