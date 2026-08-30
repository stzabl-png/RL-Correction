# 在一台新机器上部署 RL_Correction

> 通用部署指南。已按此流程成功部署过:A6000 ×2(`yanghong@128.32.164.89`,实例见
> `docs/DEPLOY_A6000.md`)。日常使用见 `docs/PREGRASP_MANUAL.md`。

---

## 0. 硬件要求

| 项 | 要求 | 说明 |
|---|---|---|
| GPU | **NVIDIA RTX,显存 ≥12GB** | 1024 env 训练实测占 **4.2GB**,评测 8.3GB |
| 驱动 | ≥ 535(CUDA 12.x/13.x 均可) | Isaac Sim 5.1 要求 |
| 磁盘 | ≥ 40GB 空闲 | Isaac 栈 ~15GB + 代码/数据 ~0.5GB + logs |
| 内存 | ≥ 32GB | 1024 env 建场景阶段是内存峰值 |
| CPU | ≥ 8 核 | 物理仿真吃 CPU |

**已验证/估算的吞吐**(1024 env,env-steps/s):

| GPU | 吞吐 | 10M 步耗时 | 备注 |
|---|---|---|---|
| RTX 5090 | 2600~3600 | ~1 h | ⚡ 600W,配弱电源时有 OCP 红线,见 `CLAUDE.md` |
| RTX A6000 | ~1800~2500 | ~1.3 h | 48GB,可双卡并行 |
| RTX 4080S | ~1800~2400(估) | ~1.5 h | **本机当前用的就是它**(2026-07-31 起);16GB 够用,320W 无电源顾虑 |

显存紧张时把 `--num_envs` 降到 512:样本效率打折,但所有机制照常成立。

---

## 1. 装 Isaac 栈(~15GB,20~40 分钟)

```bash
# 建议用 conda 隔离;python 必须 3.11
# ★ 新版 conda 必须先接受频道服务条款,否则 conda create 直接失败(见下方坑③)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create -y -n isaac python=3.11
PY=$HOME/miniconda3/envs/isaac/bin/python

$PY -m pip install --upgrade pip
$PY -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
$PY -m pip install isaaclab
# 训练侧依赖(冒烟不走 PPO,不装的话训练才报错 —— 一次装齐)
$PY -m pip install gym tensorboardX tensorboard wandb trimesh termcolor
```

验证:
```bash
OMNI_KIT_ACCEPT_EULA=YES $PY -c "import isaacsim, isaaclab, torch; \
  print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

⚠ 三个坑:
- **必须 `OMNI_KIT_ACCEPT_EULA=YES`**,否则首次启动卡在交互式 EULA 询问(非交互环境直接 EOF 崩)。
- 机器上若已有**旧版 IsaacLab(1.x)**:那是 `omni.isaac.lab` 命名空间 + Isaac Sim 4.x,
  与本项目的 `isaaclab` 2.x API **不兼容**。装在自己的 conda env 里,两者互不干扰。
- **③ conda 频道服务条款(2026-08-30 部署 Denso 时踩到)**:新版 conda 在
  `conda create` 前要求显式接受 `pkgs/main` 与 `pkgs/r` 的 ToS,非交互环境下
  **直接报错退出**:

  ```
  CondaToSNonInteractiveError: Terms of Service have not been accepted for the
  following channels. Please accept or remove them before proceeding:
      - https://repo.anaconda.com/pkgs/main
      - https://repo.anaconda.com/pkgs/r
  ```

  按上面 §1 开头那两行 `conda tos accept` 先接受即可。
  ★ 本文档写于 conda 有这个要求之前,所以老机器上装过的人不会遇到 ——
  **新机器一定会撞**。

  ★ 顺带一条部署脚本的纪律:装机脚本**必须写 `set -e`**。Denso 这次就是靠它在
  第 2 步停住的;没有它,后面 pip 会在一个不存在的 env 里继续跑,最后留下一个
  "看起来装完了、其实什么都没装进去"的环境 —— 而那种环境的症状要到跑训练时
  才暴露。

---

## 2. 传代码与数据

有两条路:**A. 从 GitHub clone**(外部协作者)、**B. 从本机 rsync**(自己开新机器)。

### 2A. 从 GitHub clone —— ⚠ **必须先装 Git LFS**

仓库里的 `*.usd *.usda *.npz *.png *.pth` 等二进制**全部走 LFS**(见 `.gitattributes`)。
没装 LFS 就 clone,拿到的是 133 字节的文本指针,Isaac 读进去**不会报致命错**,
只会静默地什么几何都加载不出来 —— 表现就是**机器人散架、双手往外摊开**。

```bash
git lfs install                     # ← 漏了这步后面全白搭
git clone -b Step4_RL_Correction git@github.com:stzabl-png/RL-Correction.git RL_Correction
cd RL_Correction && git lfs pull

# 验收: 两条都必须过
[ "$(git lfs ls-files | wc -l)" -ge 12 ] && echo "LFS ✅" || echo "LFS ❌ 没装/没拉"
file assets/vega_1p_sharpa_fixedtorso.usd    # 必须含 "data"/"PXR-USDC",若是 "ASCII text" 就是指针
```

机器人 USD(`assets/vega_1p_sharpa_fixedtorso.usd`, 25MB)**已随仓库入库**,
不需要自己重建。只有在要改躯干锁定角/碰撞几何时才跑 `tools/` 那两个脚本,见 §2C。

**Grasp3 那条训练的数据也随仓库走**(`datasets/`, 88MB: 人手轨迹 + 物体 mesh/USD +
affordance + URDF),clone 完不用配任何环境变量就能直接开训:

```bash
# 冒烟 (几分钟)
SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.smoke --headless --clip Grasp3 \
    --prior_npz tasks/pregrasp/priors/Grasp3_candidates/8_5.npz --prior_yaw 215
# 正式训练 (复现当前基线)
SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.train --headless --clip Grasp3 \
    --name repro --num_envs 1024 --approach --max_agent_steps 40000000 \
    --prior_npz tasks/pregrasp/priors/Grasp3_candidates/8_5.npz --prior_yaw 215
```

路径解析优先级(`rl_rebuild/correction/paths.py`): 环境变量 > 本机上游仓(存在才用) >
仓库自带 `datasets/`。所以原作者机器上仍读上游仓最新产物,别人机器上自动用这份快照。
详见 `datasets/README.md`。

**要训 Grasp3 以外的 clip**,才需要按 §2B ② 传完整上游产物。

### 2B. 从本机 rsync

在**本机**(`/home/lyh/Project/RL_Correction`)执行,`REMOTE` 换成目标:

```bash
REMOTE=user@host
# ① 代码(排除日志/缓存;机器人 USD 在 assets/ 里随代码走,不用另传)
rsync -az --delete --exclude .git --exclude logs --exclude cache \
      --exclude __pycache__ --exclude '*.pyc' --exclude videos_compare --exclude outputs \
      ./ $REMOTE:~/RL_Correction/

# ② 外部数据:每个要训练的 clip 传两份 + 全局两份
ssh $REMOTE 'mkdir -p ~/data/RR/Output/{RetargetOutput,ReconstructOutput}/egodex/part2/basic_pick_place ~/data/AffordanceModel/outputs ~/data/vega_urdf'
R=/home/lyh/Project/Reconstruct_and_Retarget/Output
for N in 2 5; do        # ← 要训练的 clip 编号
  rsync -az $R/RetargetOutput/egodex/part2/basic_pick_place/$N   $REMOTE:~/data/RR/Output/RetargetOutput/egodex/part2/basic_pick_place/
  rsync -az $R/ReconstructOutput/egodex/part2/basic_pick_place/$N $REMOTE:~/data/RR/Output/ReconstructOutput/egodex/part2/basic_pick_place/
done
rsync -az /home/lyh/Project/AffordanceModel/outputs/pred_egodex_part2_all20 $REMOTE:~/data/AffordanceModel/outputs/
rsync -az /home/lyh/luhr/MagicSim/Third_Party/curobo/curobo/content/assets/robot/vega_1p_sharpa/ $REMOTE:~/data/vega_urdf/
```

**体积参考**:代码 175MB;每个 clip 的重建+retarget 约 100~130MB;affordance 全量 6MB;URDF 16MB。

**GraspPose prior**(`tasks/pregrasp/priors/*.npz`)随代码 rsync,不用单独处理。

### 2C. 重建机器人 USD(只在要改躯干/碰撞时才做)

产物已入库,**正常部署不需要跑这一节**。要改的话必须**按顺序**跑全三步:

```bash
$PY tools/make_fixed_torso_usd.py     # ① 躯干/底盘/头 -> fixed joint (从 MagicSim 展平生成)
$PY tools/add_arm_collision.py        # ② 补手臂碰撞几何 (必须在 ① 之后)
$PY tools/add_collision_filters.py    # ③ 写自碰撞过滤对
```

源资产位置用 `MAGICSIM_ASSETS` 指定(默认 `/home/lyh/luhr/MagicSim/Assets`)。

⚠ **① 用的是 `Usd.Stage.Open(src).Flatten().Export(dst)` 而不是文件拷贝。**
MagicSim 的 DexMate 资产有两种布局:已展平的单文件、以及模块化(顶层引用
`configuration/vega_1p_sharpa_{base,physics,robot,sensor}.usd`)。旧版脚本用
`shutil.copy` 搬单个文件,遇到模块化布局就会把所有 visuals/collisions 引用留在原地
悬空 —— 症状同样是**机器人散架**。2026-08-03 已改成展平,两种布局产出一致。

---

## 3. 环境脚本

在目标机 `~/RL_Correction/env_remote.sh`:

```bash
export RR_ROOT=$HOME/data/RR
export AFFORDANCE_ROOT=$HOME/data/AffordanceModel
export VEGA_URDF=$HOME/data/vega_urdf/vega_1p_sharpa.urdf
export SHARPA_WANDB=0
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH=$HOME/RL_Correction
export PY=$HOME/miniconda3/envs/isaac/bin/python
```

以后每次: `source ~/RL_Correction/env_remote.sh && cd ~/RL_Correction`

---

## 4. 验收(必做)

```bash
source ~/RL_Correction/env_remote.sh && cd ~/RL_Correction
$PY -u -m tasks.pregrasp.smoke --headless --clip Grasp5 --grasp_prior tasks/pregrasp/priors/Grasp5.npz
```

看到这三行就算通过(首次启动会拉扩展缓存,慢几分钟):
```
[prior] 物体按 Dexonomy 规范姿态摆放 ...
[prior]   IK grasp err 0.35cm | pregrasp err 0.35cm ...
[smoke] ✅ 管线通过: 150 步无 NaN
```

然后正式训练:
```bash
tmux new -s train      # 断线不中断
$PY -m tasks.pregrasp.train --headless --clip Grasp5 --name <RunName> --num_envs 1024 --grasp_prior
```

---

## 5. 多卡 / 并行

```bash
CUDA_VISIBLE_DEVICES=0 RL_ISAAC_NO_GUARD=1 $PY -m tasks.pregrasp.train ... --name runA
CUDA_VISIBLE_DEVICES=1 RL_ISAAC_NO_GUARD=1 $PY -m tasks.pregrasp.train ... --name runB
```

- `rl_rebuild/utils/gpu_guard.py` 的独占槽位是**按机器**不是按卡的,多卡并行必须
  `RL_ISAAC_NO_GUARD=1` 关掉;
- ⚡ **只有确认供电充裕的机器才能这么做**。本机的旧卡(5090)+14700K 曾两次被并发满载
  打到电源 OCP 整机瞬断(见 `CLAUDE.md`);换成 4080S 后该风险解除,A6000 同样没有。

---

## 6. 取回结果

```bash
rsync -az $REMOTE:~/RL_Correction/logs/<RunName> ./logs/     # ckpt + TB + 视频
```

TensorBoard 在本机看:`tensorboard --logdir logs/<RunName>`

---

## 7. 常见部署故障

| 现象 | 原因/处理 |
|---|---|
| `CondaToSNonInteractiveError: Terms of Service have not been accepted` | 新版 conda 要求先接受 `pkgs/main` / `pkgs/r` 的 ToS。跑 §1 开头那两行 `conda tos accept`。**老机器装过的人不会遇到,新机器一定撞** |
| 卡在 `Do you accept the EULA?` | 没设 `OMNI_KIT_ACCEPT_EULA=YES` |
| `ModuleNotFoundError: gym / tensorboardX` | 训练侧依赖没装齐(§1 最后一行);冒烟不走 PPO 所以不报 |
| `No module named 'isaaclab'` 或 API 报错 | 装成了旧版 IsaacLab 1.x(`omni.isaac.lab`),按 §1 重装 |
| `找不到派生资产 vega_1p_sharpa_fixedtorso.usd` | 代码没传全;它在仓库 `assets/`(25MB),别被 rsync 排除 |
| **机器人散架 / 双手往外摊开 / link 位置全错** | 三选一:①`git lfs` 没装没拉,USD 是文本指针(§2A 验收);②自己跑了旧版 `make_fixed_torso_usd.py`(用 copy 不用 Flatten)且源是模块化布局;③`MAGICSIM_ASSETS` 指错。日志里 grep `unresolved reference` / `vega_1p_sharpa_physics.usd` 可确认 |
| 日志报 `unresolved reference ...:/visuals/...` | 同上,USD 的外部引用断链。核验:`Sdf.Layer.FindOrOpen(usd)` 遍历 reference,`assetPath` 非空的应为 **0** |
| ArmIK 报错 / URDF 找不到 | `VEGA_URDF` 没设或 URDF 没传(§2) |
| 建场景阶段被 OOM killer 杀(退出码 137) | 忘了 `--headless`,或内存不足;降 `--num_envs` |
| clip 数据缺失 | 该 clip 的重建/retarget 没传(§2 的循环里加编号) |
