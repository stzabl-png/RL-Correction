# 双卡服务器(128.32.164.89, 2×A6000)RL 训练使用指南

面向: 在双卡机上跑 Pour17 RL 训练的同事。环境已在 yanghong 账户下部署完毕并验绿。

## 0. 一句话流程

```
登录 → 挑空卡 → (第一次)跑零动作冒烟验绿 → 发射训练 → 看日志/TB → 评测只认 eval_pour.py
```

## 1. 环境构成(已就位, 不用重装)

| 组件 | 位置 | 版本(三台机对齐) |
|---|---|---|
| conda 环境 | `~/miniconda3/envs/isaac` | isaacsim 5.1.0.0 |
| IsaacLab 源码树 | `~/IsaacLab_DexAssemble`(可编辑安装) | isaaclab 0.54.3 |
| 代码仓 | `~/RL_Correction`(本仓) | 与 GitHub 同步 |
| RR 数据 | `~/data/RR/Output/...` | pour/17 已就位 |

## 2. 挑卡

```bash
nvidia-smi   # 选 util 低、显存空的卡; 训练一条线约占 5GB 显存
```

## 3. ★铁则: 换机器/改代码后, 先跑零动作冒烟

跨机资产漂移出过一次全线报废事故(机器人 USD 旧站姿 → 出生即死)。任何一台机器
在发射训练前, 必须零动作冒烟全绿:

```bash
cd ~/RL_Correction
export RR_ROOT=$HOME/data/RR TMPDIR=$HOME/tmp OMNI_KIT_ACCEPT_EULA=YES \
  VEGA_URDF=$HOME/RL_Correction/datasets/vega_urdf/vega_1p_sharpa/vega_1p_sharpa.urdf \
  RL_ISAAC_NO_GUARD=1 CUDA_VISIBLE_DEVICES=<卡号> SHARPA_WANDB=0 PYTHONPATH=$PWD
~/miniconda3/envs/isaac/bin/python -u tasks/Pour/17/C_Wiring/smoke_zero.py --headless --steps 400
```

必须看到: `机器段死线误触 = 0` + `✅ 预检骨架通过`。不绿不许发射。

## 4. 发射训练

```bash
bash ~/RL_Correction/tasks/Pour/17/C_Wiring/ucby_launch_arm.sh <线名> <卡号> [消融flag...]
# 例: 基线      bash .../ucby_launch_arm.sh A2 1
#     squeeze线 bash .../ucby_launch_arm.sh B2 1 POUR_SQUEEZE_FF=1
```

可用消融 flag(每线只开一个, 单变量纪律):
`POUR_SQUEEZE_FF=1`(squeeze 掺前馈) | `POUR_BONUS_NOW=1 POUR_BONUS_DIST=1`(贴实奖金)
| `POUR_HOLD_K=30`(热身加长) | `POUR_LEASH_ROT_TILT=1`(皮筋倾角口径)

日志: `logs/E2E_Pour17_<线名>.out`; TB: `logs/E2E_Pour17_<线名>/stage1_tb`。

## 5. 看什么指标(判读口径)

- `sr/gate1-4` = 逐关达成率(**只计挣来的, RSI 预置不算**): gate1 双手抓稳 /
  gate2 真倾倒 / gate3 放回 / gate4 全链成功。
- `ep_rew/*` = 逐项奖惩台账(adv 行进 / leash 皮筋 / ms 里程碑 / pen 死线 / bonus 奖金)。
- ★训练期指标带探索噪声, **成功率只认确定性评测**:

```bash
~/miniconda3/envs/isaac/bin/python -u tasks/Pour/17/C_Wiring/eval_pour.py \
  --checkpoint logs/<线名>/stage1_nn/last.pth --num_envs 256 --headless
```

## 6. 每 1M 步自动录像(可选)

```bash
bash ~/RL_Correction/tasks/Pour/17/C_Wiring/msc_autorec_arm.sh <线名> <卡号> [同套flag...]
# 产物: logs/E2E_Pour17_<线名>/videos/<线名>_<N>M.mp4
```

## 7. 已知坑单(全部已在脚本里处理, 手动跑时别漏)

1. `TMPDIR=$HOME/tmp` —— /tmp/isaaclab 可能被别的用户占住(权限拒绝)。
2. `OMNI_KIT_ACCEPT_EULA=YES` —— 否则 EULA 交互卡死无输出。
3. `VEGA_URDF` 指向仓内 datasets 副本 —— 别依赖机器本地散装文件。
4. `RR_ROOT=$HOME/data/RR` —— 缺了会去仓内找 replay_world.npz 报 FileNotFound。
5. `RL_ISAAC_NO_GUARD=1` + `CUDA_VISIBLE_DEVICES` —— 共享机上手动分卡, 不走 gpu_guard 锁。
6. ★改代码/资产后同步别的机器, **rsync 必须带 `assets/`**(躯干锁死 USD 与站姿绑定,
   漏了 = 机器人错位 = 出生即死, 见 DECISIONS.md 事故记录)。
7. 杀训练用 `pkill -9 -f train_pour.py`, 等 10 秒再起新的(Isaac 关闭流程会挂)。

## 8. 设计文档索引

- 判据/奖惩总纲: `tasks/Pour/17/A_Design/L3_Learning/REWARD_DOC.md`
- 全部设计决策与事故台账: `tasks/Pour/17/A_Design/DECISIONS.md`
- 判据单一来源: `A_Design/L3_Learning/progress.py`(标量规格) + `progress_batch.py`
  (torch 镜像)。**改判据必须双版同步 + 重跑六件自检**(selftest_*.py)。
