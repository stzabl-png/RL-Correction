# 在 A6000 机器上跑 RL_Correction

> 这是**一台具体机器的部署记录**。要部署到新机器请看通用指南 `docs/DEPLOY_NEW_MACHINE.md`,
> 本文件只保留 A6000 这台的路径/账号/已知状态。

目标机: `msc-auto@128.32.164.89`(Lambda Vector,**RTX A6000 ×2 / 48GB each**,20 核,548G 空闲)
使用账号: **`yanghong`**(已配公钥免密;`msc-auto` 是共享账号,别在它下面训练)

> 本机(**4080S 16GB**, 2026-07-31 起; 原 5090)与 A6000 可**并行**跑不同的 run —— 两台机器互不影响电源。
> 但**同一台机器上**仍然只能有一个 Isaac 满载(⚡ 见 CLAUDE.md)。

---

## 1. 已完成的部署(2026-07-30)

| 内容 | 远端路径 | 大小 | 说明 |
|---|---|---|---|
| 代码库 | `~/RL_Correction` | 175M | rsync,排除 logs/cache/.git |
| 重建+retarget 数据 | `~/data/RR/Output/...` | 231M | 目前只传了 clip **2**(Grasp2)和 **5**(Grasp5) |
| affordance | `~/data/AffordanceModel/outputs/pred_egodex_part2_all20` | 6M | 全部 20 个物体 |
| Vega URDF(ArmIK 必需) | `~/data/vega_urdf/` | 16M | MagicSim/curobo 里那份 |
| 环境变量脚本 | `~/RL_Correction/env_a6000.sh` | — | `source` 它即可 |
| Isaac 栈 | `~/miniconda3/envs/isaac` | ~15G | python3.11 + torch2.7cu128 + isaacsim5.1 + isaaclab |

机器人 USD **不需要单独传**:仓库 `assets/vega_1p_sharpa_fixedtorso.usd`(躯干锁死派生版,25MB)
已随代码同步,`dexmate_env_cfg` 优先用它。

---

## 2. 日常使用

```bash
ssh yanghong@128.32.164.89
source ~/RL_Correction/env_a6000.sh     # 设 RR_ROOT / AFFORDANCE_ROOT / VEGA_URDF / PY
cd ~/RL_Correction

# 冒烟(先跑这个确认环境)
$PY -m tasks.pregrasp.smoke --headless --clip Grasp5 --grasp_prior tasks/pregrasp/priors/Grasp5.npz

# 训练(建议 tmux/nohup, 断线不中断)
tmux new -s train
$PY -m tasks.pregrasp.train --headless --clip Grasp5 --name GraspPose_5_a6000 \
    --num_envs 1024 --grasp_prior
```

其余命令与本机完全一致,见 `docs/PREGRASP_MANUAL.md`。

### 指定用哪块 GPU

两块 A6000 可以**各跑一个训练**(功耗压力远小于单卡 5090):

```bash
CUDA_VISIBLE_DEVICES=0 $PY -m tasks.pregrasp.train ... --name runA
CUDA_VISIBLE_DEVICES=1 $PY -m tasks.pregrasp.train ... --name runB
```

⚠ 但 `rl_rebuild/utils/gpu_guard.py` 的独占槽位是**按机器**而不是按卡的,双卡并行会被它挡住。
真要双卡并行:`RL_ISAAC_NO_GUARD=1` 关掉守卫(A6000 没有 5090 那台的电源问题,可以关)。

### num_envs

A6000 48GB 显存宽裕,但吞吐比 5090 低(约 6~7 折)。建议:
- 单卡 `--num_envs 1024`(与本机同口径,便于对照);
- 想快就双卡各 1024 跑不同实验,而不是单卡堆 2048。

---

## 3. 加新物体时要传什么

代码改动直接 rsync;**数据**要补传对应 clip 的三件套:

```bash
# 本机执行,把 clip N 的数据推过去
N=7
R=/home/lyh/Project/Reconstruct_and_Retarget/Output
rsync -az $R/RetargetOutput/egodex/part2/basic_pick_place/$N \
  yanghong@128.32.164.89:~/data/RR/Output/RetargetOutput/egodex/part2/basic_pick_place/
rsync -az $R/ReconstructOutput/egodex/part2/basic_pick_place/$N \
  yanghong@128.32.164.89:~/data/RR/Output/ReconstructOutput/egodex/part2/basic_pick_place/
# prior 文件随代码 rsync 走(在 tasks/pregrasp/priors/)
```

同步代码(不覆盖远端 logs):

```bash
cd /home/lyh/Project/RL_Correction
rsync -az --delete --exclude .git --exclude logs --exclude cache \
      --exclude __pycache__ --exclude '*.pyc' --exclude videos_compare \
      ./ yanghong@128.32.164.89:~/RL_Correction/
```

## 4. 取回结果

```bash
# 把远端某个 run 拉回本机看 TensorBoard
rsync -az yanghong@128.32.164.89:~/RL_Correction/logs/<RunName> ./logs/
```

---

## 5. 环境重装(万一坏了)

```bash
ssh yanghong@128.32.164.89
$HOME/miniconda3/bin/conda create -y -n isaac python=3.11
PY=$HOME/miniconda3/envs/isaac/bin/python
$PY -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
$PY -m pip install isaaclab
```

⚠ 机器上原有 `/home/msc-auto/IsaacLab`(1.2.0)**不能用**:那是 `omni.isaac.lab` 老命名空间 +
Isaac Sim 4.x,与本项目的 `isaaclab` 2.x API 不兼容。我们在 `yanghong` 下装的是独立新栈,互不干扰。

首次启动 Isaac 会拉扩展缓存(几分钟),之后就快了。
