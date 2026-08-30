# 想套用我们的训练设计? 从这里开始

> 一句话: **代码和台账全在 `Step4_RL_Correction` 分支**,默认分支 `main` 只有 pipeline 总览图。

## 分支地图

| 分支 | 装的是什么 |
|---|---|
| `main` | pipeline 总览 README + 图。**没有代码** |
| `Step1_DataInput` | 数据入口 |
| `Step2_NoisyRecon` | 视频重建(noisy HOI) |
| `Step3_Dexonomy` | 抓取位姿生成 |
| **`Step4_RL_Correction`** | **← RL 修正。训练代码、设计台账、判据、自检全在这里** |
| `task5/pour-bimanual` | 双手倒水的早期实现(已被 Step4 里的 `tasks/Pour/17/` 取代) |

```bash
git clone git@github.com:stzabl-png/RL-Correction.git
cd RL-Correction && git checkout Step4_RL_Correction
git lfs pull        # 母带/USD/资产走 LFS,不拉就只有指针文件
```

## 先读哪三份

1. **`CLAUDE.md`**(仓库根)—— 会立刻犯的错(解释器路径、`--headless`、`SHARPA_WANDB=0`、显存上限)
2. **`tasks/Pour/17/A_Design/DECISIONS.md`** —— **主台账**。每条设计连同它的可证伪信号、
   以及被数据推翻的过程,按 L5-xx 编号顺序记。想知道"为什么是这样"看这份
3. **`docs/DESIGN_LOOP.md`** —— 方法论:改设计前先把假设和证伪条件写进台账,一次只改一个参数

## 训练代码在哪

```
tasks/Pour/17/
├── A_Design/
│   ├── DECISIONS.md              主台账
│   ├── L2_Reference/*.npz        参考母带 (LFS)
│   └── L3_Learning/
│       ├── progress.py           判据 + 阶段机 (可脱离 Isaac 导入)
│       └── progress_batch.py     判据的批量版
├── B_SmokeTest/                  探针 (probe_*.py) + 自检家族 (selftest_*.py)
└── C_Wiring/
    ├── train_pour.py             训练入口
    ├── pour_env.py               环境 (继承 tasks/pregrasp/env.py)
    ├── world_fingerprint.py      世界指纹
    ├── eval_pour.py              确定性评测 ← 成功率只认这个
    └── record_pour.py            录像
```

## 训练逻辑,五条

**① 残差策略,不是从零控制。** 参考轨迹上叠加小幅修正。残差界 × sigma = 每步扰动量,
这个任务的容忍度是毫米级 —— 放大过就是接触率 0.1%、物体被打飞。

**② 四 Gate 阶段机 + 认证机。** 任务切成 G1(抓住)→ G2(握稳)→ G3(倒)→ G4(放回),
每个 Gate 有独立判据;认证机在 G2 后周期性做"抬升-滑移"测试,证明是真握住不是卡住。
判据全部集中在 `progress.py`,**可以脱离 Isaac 单独 import 和单测**。

**③ 渐进 RSI(出生点课程)。** 初始只从 t0 出生;某个 Gate 的 t0 口径 EMA ≥ 0.30
且连续 20 窗,才解锁"从该 Gate 出生"。
> ★ **读数陷阱**: 解锁后总口径成功率会被非 t0 出生的回合抬高。
> **真实能力只看 `sr_t0/*`**(t0 口径)。我们踩过:总口径 G4 稳定在 0.13 看着在涨,
> 而 `sr_t0/gate4` 从头到尾是 **0** —— 那 0.13 恰好等于一个出生点的采样份额。

**④ 参考按可信度分通道(两个变体的唯一区别)。**
物体轨迹**全程**定义成败与绝对位置底线;人手轨迹只做"怎么动"的形状参考,
权重按母带每行的重建置信度滑动:

```
档位        物体权重  人手权重     说明
绿(高置信)    1.0      0        两个变体此时完全相同
黄(中)       0.5     0.5
红(低)       0.2     0.8       物轨形状不可信, 但绝对位置仍拴宽皮筋
```
- `POUR_VARIANT=HYB` —— 带手(上表)
- `POUR_VARIANT=OBJ` —— 不带手(人手权重钉死 0)

> ★ 人手只给方向不给绝对位姿,因为人手重建的腕位误差在 20cm 量级。
> ★ **剂量提醒**: 我们实测 HYB 的形状项只占实际到手收入的 **1.1%**,
> 而绿档权重是 0 ⟹ 绝大多数步两个变体是同一个目标函数。**这个剂量下测不出差异**。
> 要真做这个消融,先把剂量提到能看见的量级。

**⑤ 世界指纹 = ckpt 的身份证。** ckpt 只能在它出生的世界里回放。
训练开始采集完整指纹写 `<run>/world.json`;回放/评测自动比对,关键项不符**拒跑(退出码 11)**。
关键项:参考母带 md5、机器人 USD md5、受控关节名与顺序、obs/act 维度、控制步、
桌面高度、物体质量与摩擦、判据摘要 `criteria.digest`。
> ★ 为什么必须有: 换了世界,obs **维度一个字节都不变、语义变了**,所有自动检查全绿,
> 而策略照旧输出记忆动作 —— 手伸到错位置且不报错。我们踩过一次 0/64。
> 细节见 `docs/CKPT_HANDOFF.md`。

## 起一条训练

```bash
SHARPA_WANDB=0 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
  POUR_REF_NPZ=tasks/Pour/17/A_Design/L2_Reference/pour17_reference_v3.npz \
  POUR_VARIANT=HYB POUR_OBJ_CONTACT=1 POUR_COLLIDE=pen POUR_COLLIDE_PEN=0.5 \
  <你的python> -u tasks/Pour/17/C_Wiring/train_pour.py \
    --name MyRun --num_envs 512 --seed 11 --headless
```

评测(**成功率只认这个**,训练期 TensorBoard 的口径含课程脚手架):

```bash
<你的python> tasks/Pour/17/C_Wiring/eval_pour.py --checkpoint logs/MyRun/stage1_nn/last.pth
```

## 四条会让你白跑的坑

1. **必须 `--headless`。** 漏了会加载非 headless kit,512 env 在建场景阶段被 OOM killer
   杀掉(退出码 137)。核验:日志里 grep `experience file`,必须是 `.headless.kit`。
2. **多卡机必须 `RL_ISAAC_NO_GUARD=1`。** `gpu_guard` 的 flock 是**按机器**不是按卡的
   (原本给单卡防电源过流用),第二条训练会被**无限期静默阻塞**,日志里只有一行排队提示,
   极容易被当成"启动慢"。
3. **热启动不得直接指向另一条在跑的线的 `last.pth`。** `torch.load` 读一个正在被写的文件,
   表现是"启动 28 分钟不进训练循环",而你会把它归因到别的改动上。先复制一份再 `--load_path`。
4. **`ls | tail` 是字典序**(`ep_900` 排在 `ep_3000` 后)。取最新 ckpt 用 `last.pth`。

## 我们的工作方式(不是代码,但决定了这套东西为什么长这样)

- **改设计前先写下证伪条件。** 每条设计在台账里配一个"什么数据出现就说明它错了"。
  一次只改一个参数,否则判读表失效。
- **验行为,不验旗。** 旗接上了 / 横幅打了 / 数出来了 / 预检绿了 —— 四个都不等于
  "它在按你以为的方式工作"。我们靠这条抓到过"重写钩子不调 `super()`,整个基座奖励面
  在该体制下是死代码"(白跑 3M 步)。
- **说不出"什么情况下会红"的检查 = 恒真断言。** 同一批检查里若出现 `>0` 与 `>=0.9N`
  混用,就说明有一条在装样子。
- **"没数据"必须红,不能当通过。** 短路成通过的检查比没有检查更糟。
- **一天误报三次的闸门,人第二天就开始无视它** —— 比没有闸门更糟,因为它还额外提供
  "我们有闸门"的错觉。所以判据摘要只覆盖决定成败的常量,不含奖励权重。
