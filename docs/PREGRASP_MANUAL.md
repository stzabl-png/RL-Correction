# PreGrasp 抓取任务 使用手册

> 适用: `tasks/pregrasp/`(当前主线任务)。设计演进与失败判读见 `docs/DESIGN_LOOP.md`;
> 机器/电源等硬约束见 `CLAUDE.md`。本手册只讲**怎么用**: 输入什么、输出什么、怎么跑。

---

## 0. 一句话

给定 **物体网格(重建)** +(可选)**Dexonomy GraspPose**,用 PPO 训出一个策略:
从 PreGrasp 位姿出发,五指软垫对称挤压物体形成稳定抓握,并通过 **1cm 微抬升物理验证**。

已达成基准(确定性评测,全价判据):

| 任务 | 物体 | prior | 训练量 | 成功率 |
|---|---|---|---|---|
| `PreGrasp_0` | Grasp2 甜甜圈 7.7×2.6cm | 无 | ~1 天 / 8 轮设计迭代 | **99.95%** |
| `GraspPose_5` | Grasp5 小环 5.8×2.5cm | 有 | **5.8M 步 / 40 分钟** | **100.00%** |
| `GraspPose_3` | Grasp3 方块 2.6×3.9×3.9cm | 有 | 10.3M 步(@A6000) | **100.00%** |
| `GraspPose_1` | Grasp1 长条 3.7×8.1×11.7cm | 有 | 12.9M 步 | ❌ 回避塌缩,见 §7.2 |

**对照**(同物体 Grasp5、同配方、各 20M 预算):无 prior 峰值 5.8% 后塌缩、从未过 10%;
有 prior 50%@1.4M、90%@4.1M。**GraspPose prior 是新物体冷启动的决定性要素。**

计算资源:本机 **RTX 4080 SUPER 16GB**(2026-07-31 换卡, 原 5090 32GB; 电源红线已放宽,
新瓶颈是显存)+ A6000 ×2(`yanghong@128.32.164.89`,
见 `docs/DEPLOY_A6000.md`)。**两台可并行跑不同实验**,同一台内仍只能一个 Isaac 满载。

---

## 1. 环境

```bash
PY=/home/lyh/luhr/MagicSim/.venv/bin/python   # 唯一带 isaacsim 5.1 + isaaclab 2.x 的解释器
cd /home/lyh/Project/RL_Correction            # 所有命令都在仓库根跑
export PYTHONPATH=.                           # tasks/ 与 rl_rebuild/ 都从这里解析
export SHARPA_WANDB=0                         # 不设会弹 wandb 交互登录卡死
```

> **换机器跑?** 见 `docs/DEPLOY_NEW_MACHINE.md`(通用流程:硬件要求/吞吐参考、Isaac 栈安装、
> 代码与数据传输、验收冒烟、多卡并行、部署故障表)。A6000 的具体实例见 `docs/DEPLOY_A6000.md`。

外部数据根(可用同名环境变量覆盖,见 `rl_rebuild/correction/paths.py`):

| 变量 | 默认 | 用途 |
|---|---|---|
| `RR_ROOT` | `/home/lyh/Project/Reconstruct_and_Retarget` | 重建 + retarget 产物 |
| `AFFORDANCE_ROOT` | `/home/lyh/Project/AffordanceModel` | affordance 热图 |
| `MAGICSIM_ASSETS` | `/home/lyh/luhr/MagicSim/Assets` | 机器人原始 USD(仓库内已有派生的躯干锁死版,通常不用) |
| `VEGA_URDF` | MagicSim/Third_Party/curobo/… | ArmIK 用的 URDF(**必需**) |

---

## 2. 输入格式

### 2.1 clip(一个物体 = 一条 clip)

在 `rl_rebuild/correction/clips.py` 注册。`replay_grasp` 源需要**四个文件**:

```python
CLIPS["Grasp5"] = dict(
    source="replay_grasp",
    npz       = f"{RT}/replay_world.npz",              # retarget: 人手轨迹 + phase_left/right
    mesh      = f"{RC}/object_mesh_scaled_final.obj",  # 重建: 物体网格(**真实尺度**)
    usd       = f"{RT}/object.usd",                    # 视觉网格(物理运行时贴)
    affordance= f"{AFF}/obj_05/affordance.npz",        # 逐点热图(物体系)
    runtime_object_physics=True,
    semantics = ObjectSemantics(label=..., mass_kg=0.1, friction=0.5),
)
```

| 文件 | 关键字段 | 说明 |
|---|---|---|
| `replay_world.npz` | 手关节轨迹、`phase_left/right` | 交互手由 phase 自动判定,**别写死** |
| `object_mesh_scaled_final.obj` | — | 尺度必须真实(用于表面点云、距离信号) |
| `object.usd` | — | 纯视觉,刚体/碰撞由 env 运行时施加 |
| `affordance.npz` | `points_raw (P,3)`, `heatmap (P,)` | 物体规范系;无 prior 时的对齐目标 |

`Grasp0~19` 已按 EgoDex part2/basic_pick_place 批量注册,新物体照抄一行即可。

### 2.2 GraspPose prior(可选,强烈推荐)

Dexonomy 的 `*_grasp.npy` → 本任务格式(**用系统 python3 跑**,Dexonomy npy 是 numpy2 pickle):

```bash
python3 tasks/pregrasp/make_prior.py \
  --grasp_npy  /path/Dexonomy/output/<obj>_sharpa_wave/grasp_data/fingertip_mid/<obj>/tabletop/scale010/<id>_grasp.npy \
  --info_json  /path/Dexonomy/assets/object/custom/processed_data/<obj>/info/simplified.json \
  --out        tasks/pregrasp/priors/<ClipName>.npz
```

产物 `priors/<ClipName>.npz`(全 plain array,**物体输入系**,即与重建 mesh 同系):

| 键 | 形状 | 含义 |
|---|---|---|
| `grasp` | (29,) | `[pos3, quat4(wxyz), 指22]` 抓握位姿 |
| `squeeze` | (29,) | 挤压位姿(暂未使用) |
| `pregrasp` | (6,29) | 预抓路点(加载器挑 IK 可达且离抓握最近的) |
| `contact_pos/normal` | (N,3) | 接触点与法向 |
| `contact_centroid` | (3,) | **prior 模式下的对齐目标** |

| `canon_rot` | (4,) | Dexonomy 生成抓姿时物体的**规范姿态**(wxyz) |

⚠ **三个已踩过的坑**(都已在代码里处理,但换数据源时要重新核对):
1. **手指关节序**:npy 的 22 维是 `GENERIC_JOINT_ORDER` 序,进 env 必须换到 USD 关节序
   (加载器用 `_generic_perm`)。漏换 → 手型全串位、整条 run 报废(2026-07-30 实际发生)。
2. **规范系旋转**:Dexonomy 的物体规范系 = 重建输入系旋转 `canonical_from_input_rot`,
   `make_prior.py` 转回输入系保存。
3. **物体静置姿态必须与 Dexonomy 一致**:抓姿是在"物体按规范姿态平放桌面"时生成的,
   而 ref builder 用"最大支撑面朝下"独立决定 —— 对**非对称物体**可能选到另一个面。
   Grasp3 实测两姿态差 **179.9°**(上下颠倒),抓姿腕落到桌面下 12cm,全 yaw 角 IK 失败。
   prior 加载器现在按 `canon_rot` 重摆物体并重算贴桌高度(xy 仍用相机锚定)。
   ⚠ 环形/对称物体会**掩盖**这个错误,别用它们验证新数据源。

**加载器还会自动做两件事**:① **物体 yaw 搜索**(重建的 yaw 是丢弃姿态后的随机残留,
可能让抓姿背对机械臂;5° 步长扫一圈选 IK 最优,并对"离另一只手 <25cm"加罚);
② 从 6 个 pregrasp 路点里挑 IK 可达且离抓握位姿最近的那个。

---

## 3. 输出格式

```
logs/<RunName>/<YYYY-MM-DD_HH-MM-SS>/
├── stage1_nn/
│   ├── last.pth                 ← **评测/录像一律用它**
│   ├── best.pth                 ← ⚠ 无意义(课程涨价前的高分),别用
│   └── ep_<N>_step_<M>M_reward_<R>.pth
├── stage1_tb/events.out.*       ← TensorBoard
├── milestones.json              ← 首达 1/10/25/50/80/90% 的步数+墙钟
└── videos/last-step-0.mp4       ← record.py 产物
```

### TensorBoard 关键标量

| 标量 | 读法 |
|---|---|
| `success_rate_ema` | **唯一看进度的指标**;含 c₀ 随机化脚手架,会**低估**真实水平 |
| `grasp/got_candidate` | 候选抓取形成率(进验证段的比例) |
| `grasp/pads_now` | **同时**有效接触的垫数(候选判据看的就是它) |
| `grasp/pads_touched_latch` | 回合内"曾碰到过"几个垫 —— ⚠ 是 latch,**别当同时接触读**(踩过) |
| `grasp/gentle` | 笨拙课程价格 0.2→1.0;**未到 1.0 时回合奖励跨窗口不可比** |
| `ep_rew/*` | 逐项奖励账本(诊断失败先看这组) |
| `term/*` | 终止原因占比(fell/thrown/pushed/stuck/table_crash/timeout) |

### 确定性评测输出(正式成绩)

```
[口径A: c0=0 (标准起点, 正式口径)]  成功率 = 100.00%   (6135/6135 回合)
  成功回合中位长度 / 失败构成 / 验证失败次数 / 逐垫接触率 / 抖动偏移分桶
```
口径 A(`c₀=0`)是**对外报的正式数字**;口径 B(`c₀~U(0,0.9)`)是训练分布对照。

---

## 4. 标准工作流

### 4.1 冒烟(改任何东西之后必跑,2 分钟)

```bash
SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.smoke --headless \
    --clip Grasp5 --grasp_prior tasks/pregrasp/priors/Grasp5.npz
```
看:obs 维度对得上、力/向心分符号正确、无 NaN、能走到 verify 阶段。

### 4.2 训练

```bash
SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.train --headless \
    --clip Grasp5 --name GraspPose_5 --num_envs 1024 --grasp_prior
```

| 参数 | 说明 |
|---|---|
| `--headless` | **必须**(漏了会加载带渲染的 kit,1024 env 会被 OOM killer 杀) |
| `--num_envs` | 软上限 1024(⚡电源红线,见 CLAUDE.md);`RL_MAX_ENVS=2048` 可强制覆盖 |
| `--grasp_prior` | 自动读 `tasks/pregrasp/priors/<clip>.npz` |
| `--obj_jitter` | 物体位置扰动课程 0.003→0.006→0.010→0.015 |
| `--load_path X --resume` | 热启动(换课程档、换物体迁移都用它) |
| `--max_agent_steps` | 预算上限;不给则跑到 1 亿步 |

### 4.3 评测(**唯一算数的成绩**)

```bash
SHARPA_WANDB=0 PYTHONPATH=. $PY -u -m tasks.pregrasp.eval --headless \
    --clip Grasp5 --checkpoint logs/GraspPose_5/<ts>/stage1_nn/last.pth \
    --grasp_prior tasks/pregrasp/priors/Grasp5.npz --steps 300
```
(`-u` 必须,否则输出被缓冲吞掉。)

### 4.4 录像 / GUI 观看 / 静态摆姿

```bash
# 离屏录 mp4(--fps 10 半速)
$PY -m tasks.pregrasp.record  --headless --clip Grasp5 --checkpoint <ckpt> --grasp_prior <npz> --fps 10
# GUI 实时看策略(不要 --headless)
$PY -m tasks.pregrasp.play    --clip Grasp5 --checkpoint <ckpt> --grasp_prior <npz> --slow 2
# GUI 静态看 GraspPose 摆位(只渲染不跑物理)
$PY -m tasks.pregrasp.view_prior --clip Grasp5 --grasp_prior <npz>
# 站姿快照(headless 调姿)
$PY -m tasks.pregrasp.snap_pose --headless --set "L_arm_j2=45,R_arm_j2=-45" --out /tmp/pose.png
```

### 4.5 新物体全流程(约 1 小时)

```
① 重建产物就位 (mesh + replay_world.npz + object.usd + affordance.npz)
② Dexonomy 生成 GraspPose → make_prior.py → priors/<Clip>.npz
③ clips.py 注册一行(Grasp0~19 已批量注册, 新数据集才需要)
④ smoke  (2 min)   —— 看 IK 是否可达、力信号是否正常
⑤ train  (~40 min) —— 带 --grasp_prior, 观察 milestones
⑥ eval + record    —— 拿正式成绩 + 目视验收
```

---

### 4.6 新物体接入清单(逐条核对,约 1 小时)

> ⚠ **换新物体 / 有多个 GraspPose 候选可选时, 先读 `docs/GRASPPOSE_SCREENING.md`。**
> 那里定义了三关筛选(数据同一性 / 可达性 / prior 质量)与两条硬判据
> (`pads* ≥ 4`、`Q* > 0`), 能在 ~5 分钟内判死一个候选, 不用训完一小时才知道。
> **不要用 Dexonomy 的 `isaac_succ/` 当候选池** —— 它的验证器对本平台没有预测力,
> `grasp_data_isaac_failed/` 里的候选必须一并纳入筛选。

```bash
OBJ=pp7; ID=12_3; CLIP=Grasp7; N=7      # ← 只改这一行
```

**① 核对网格同一性**(不同就全废,先做这步)
```bash
python3 -c "
import json,trimesh,numpy as np
d=json.load(open(f'/home/lyh/Project/Dexonomy/assets/object/custom/processed_data/$OBJ/info/simplified.json'))
m=trimesh.load(f'/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex/part2/basic_pick_place/$N/object_mesh_scaled_final.obj',force='mesh')
print('Dexonomy obb:',[round(x,4) for x in d['obb']])
print('重建 OBB    :',[round(x,4) for x in sorted(m.bounding_box_oriented.primitive.extents)])"
```
两行数字必须逐位吻合(±0.5mm)。不吻合 = 尺度/网格版本不一致,**停,先解决**。

**② 生成 prior**(用系统 python3)
```bash
python3 tasks/pregrasp/make_prior.py   --grasp_npy /home/lyh/Project/Dexonomy/output/${OBJ}_sharpa_wave/grasp_data/fingertip_mid/$OBJ/tabletop/scale010/${ID}_grasp.npy   --info_json /home/lyh/Project/Dexonomy/assets/object/custom/processed_data/$OBJ/info/simplified.json   --out tasks/pregrasp/priors/$CLIP.npz
```

**③ 冒烟**(2 分钟,是唯一的接入验收)
```bash
$PY -m tasks.pregrasp.smoke --headless --clip $CLIP --grasp_prior tasks/pregrasp/priors/$CLIP.npz
```
必须看到三件事:
- `IK grasp err < 1cm`(>1cm 或直接 assert 失败 → 见 §7.1);
- 零动作合拢后 `pads ≥ 3`(prior 位姿本身够得着);
- `✅ 管线通过`。

**④ 训练 → 评测 → 录像**(见 §4.2~4.4),run 名约定 `GraspPose_<N>`。

---

## 5. 必须知道的坑

1. **`best.pth` 无意义**:笨拙课程会随成功率涨价,早期低价时的高分回合被记为 best。一律用 `last.pth`。
2. **TB 成功率低估**:训练口径含 c₀ 随机化 + 探索噪声;3mm 档 TB 70% ↔ 确定性 99.95%。定论只认 `eval.py`。
3. **`gentle` 未到 1.0 前,回合奖励不可跨窗口比较**(价格在变)。
4. **一次只改一个旋钮**,改动前把假设和证伪条件写进 `docs/DESIGN_LOOP.md` 台账。
5. **⚡ 不要两个 Isaac 同时满载**(整机断电前科 ×2)。`utils/gpu_guard.py` 有独占槽位,但仍要避免手动并发。
6. **杀训练进程后等 10 秒**再起下一个,否则 carb mutex 崩溃。
7. **手↔桌物理碰撞被 `filter_collisions` 过滤**(手能穿桌)、**PhysX 自碰撞不能开**(会臂自锁)——所以撞桌和指间碰撞都是**几何惩罚**,改这块前先读 `env.py` 注释。
8. **物体 yaw**:重建的姿态被丢弃后 yaw 是随机残留;prior 加载器会自动搜索 yaw 让抓姿朝向机械臂并远离另一只手。

---

---

## 7. 故障排查(按现象查)

### 7.1 冒烟报 `GraspPose prior IK 失败` / IK 误差很大

按顺序排查:
1. **物体静置姿态**:看日志是否打印 `[prior] 物体按 Dexonomy 规范姿态摆放`。没有 →
   prior npz 是老版本(缺 `canon_rot`),重跑 `make_prior.py`。
2. **腕位在桌面下**:离线扫一遍 yaw 看腕高;若所有角度腕 z < 0.85,说明抓姿从物体
   下方接近 —— 就是姿态没对齐(§2.2 坑 3)。
3. **真的够不到**:腕高正常但 IK 误差 >5cm 且各 yaw 都差 → 该抓姿在我们的工作空间外,
   **换 Dexonomy 的另一个抓姿候选**(一个物体通常有几十个)。

### 7.2 训练"回避塌缩":候选率归零、全部超时、无任何失败终止

**这是本项目最常见的失败模式**(9 轮迭代里出现过 4 次)。识别特征:
`grasp/got_candidate` → 0、`term/timeout` → 1.0、其他 `term/*` 全 0、
`ep_rew/cent_income` ≈ 0(说明质量分 Q ≤ 0,待在接触状态**不赚钱**)。

本质是**经济学问题**:策略发现"不接触"的期望收益高于"笨拙地尝试"。按优先级处理:

| 情形 | 旋钮 |
|---|---|
| 有 prior,策略从起点漂走了 | `arm_dev_max` 调小(0.12→0.05):锁住 prior 位姿 |
| 无 prior / 学费太贵 | `gentle_init` 0.2→0.1(笨柤期惩罚再打折) |
| 接触本身很难发生 | `closure_init_max` 提高(更多回合从已接触态起步) |
| 上面都试过 | 换抓姿候选;或该物体的 prior 质量本身不行 |

⚠ **诊断时不要读 `pads_touched_latch`**:它是"曾碰到过"的 latch,回避塌缩时也能有 4+,
看 `grasp/pads_now`(同时接触数)才准。这个坑让 Grasp1 的判读走了一次弯路。

### 7.3 成功率涨到某处掉头

- 先看 `grasp/gentle`:若它同时在涨,是**课程涨价**导致的正常阵痛,等它稳在 1.0;
- 若 gentle 不动而成功率单调跌,查 `ep_rew/*` 找哪一项在放大;
- 若曾出现巨额负奖励(单窗 <-1000),是某个惩罚项没有单步上界 —— 这是**铁律级别的 bug**,
  排查所有铰链/平方项的 clamp。

### 7.4 训练指标好但不敢信

一律以 `tasks/pregrasp/eval.py` 的确定性评测为准。TB 的 `success_rate_ema` 含 c₀ 随机化
脚手架和探索噪声,**系统性低估**(实测 TB 70% ↔ 确定性 99.95%)。

---

## 8. 文件地图

```
tasks/pregrasp/
├── cfg.py         GraspTaskCfg —— 所有可调参数(奖励权重/课程/判据/站姿)都在这
├── env.py         GraspTaskEnv —— 阶段机、动作语义、奖励、观测、复位
├── train.py       训练入口(含笨拙课程钩子: 慢速EMA + 限速 + 棘轮)
├── eval.py        确定性评测(两口径 + 失败普查)
├── record.py      离屏录像      play.py  GUI 实时观看
├── smoke.py       三段脚本冒烟   view_prior.py  静态摆姿   snap_pose.py  站姿快照
├── make_prior.py  Dexonomy npy → prior npz(用系统 python3 跑)
├── view_prior.py  GUI 静态看 GraspPose 摆位   snap_pose.py  站姿快照(headless)
├── _calib_cent.py 用已训策略标定 cent 门槛量纲(换 cent 定义/新物体时用)
├── ppo.yaml       PPO 超参      priors/  各 clip 的 GraspPose
rl_rebuild/        引擎(PPO/wrapper/utils)+ 被继承的底层 env、ref builder、IK —— 只 import 不改
```
