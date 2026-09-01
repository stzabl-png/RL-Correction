# Unscrew V5 跨服务器部署

目标是让新服务器从远端 `unscrew_v5` 分支拉取后，不重新生成数据或规划，
即可验证环境并启动 clip32 的 HYB/OBJ 训练。

## 完成契约

正式训练依赖以下已入库产物：

- 17 条自包含重建数据与左手瓶身 affordance；
- clip32 的 `env_rest.json`、cuRobo Approach/Retreat；
- `reference_v2.npz`；
- `acceptance_v2.json`，它绑定 v2 MD5、至少 3/4 环境数值稳定性和完整世界指纹。
  reference 的穿模、IK 偏差、接触/任务成功率只作为 RL correction 基线记录，
  不属于部署阻塞项。

`train_task.py` 会在导入 IsaacLab 前验证母带与凭据，环境创建后再次核对现场
世界。绝对文件路径不参与关键比较，跨服务器用文件 MD5 和物理参数核对。

## 1. 拉取代码和 LFS

```bash
git lfs install
git clone -b unscrew_v5 https://github.com/stzabl-png/RL-Correction.git RL-Correction
cd RL-Correction
git lfs pull
```

不要忽略 LFS：NPZ、USD、PNG 等若仍是约 130 字节的文本指针，Isaac 可能以几何
缺失的形式静默失败。

```bash
git lfs status
file assets/vega_1p_sharpa_fixedtorso.usd
```

USD 必须被识别为数据/PXR-USDC，而不是包含
`version https://git-lfs.github.com/spec/v1` 的文本。

## 2. 安装已验证软件栈

本分支实际验证版本：

- Python 3.11.16
- PyTorch 2.7.0+cu128
- Isaac Sim 5.1.0
- IsaacLab 2.3.2.post1
- NumPy 1.26.0

```bash
conda create -y -n isaac python=3.11
PY=$HOME/miniconda3/envs/isaac/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
$PY -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
$PY -m pip install isaaclab==2.3.2.post1
$PY -m pip install numpy==1.26.0 trimesh==4.5.1 gym==0.26.2 tensorboardX==2.6.5 wandb==0.19.11 termcolor pyyaml
```

训练不需要 cuRobo：正式机器段已经随分支入库。只有准备在新服务器上重新生成
Approach/Retreat 时才安装 cuRobo：

```bash
mkdir -p $HOME/WorkSpace
git clone https://github.com/NVlabs/curobo.git $HOME/WorkSpace/curobo
git -C $HOME/WorkSpace/curobo checkout 8e734f3ced1df898990bcd92de40abce475907db
$PY -m pip install -e $HOME/WorkSpace/curobo --no-deps
$PY -m pip install "cuda-core[cu12]"
```

仓库自带的 cuRobo YAML 使用相对路径；worker 还会在运行时按仓库位置重定位，
不依赖原机器的 `/home/feiyang` 或 MagicSim。

## 3. 部署验收

先跑无 Isaac 的完整预检：

```bash
export PY=$HOME/miniconda3/envs/isaac/bin/python
bash tasks/Unscrew/part4/C_Wiring/verify_deploy.sh 32
```

验证器和启动器会把 Isaac 临时日志隔离到当前用户的专用目录，避免共享
`/tmp/isaaclab` 的属主冲突。若默认位置不可写，可显式设置
`UNSCREW_TMPDIR=/path/to/writable/tmp`。

它会检查 LFS 实体、机器人资产、数据、双机器段、静置产物、正式 v2、验收凭据、
依赖版本和正式母带门禁。

再在目标训练卡上做短 Isaac 加载：

```bash
GPU=0 SMOKE_STEPS=20 bash tasks/Unscrew/part4/C_Wiring/verify_deploy.sh 32 --isaac
```

多卡机器已有其他 Isaac 进程时，可以在确认指定的是不同物理 GPU 后加
`RL_ISAAC_NO_GUARD=1`；单任务部署保持默认锁即可。

## 4. 启动训练

单卡 HYB：

```bash
PY=$HOME/miniconda3/envs/isaac/bin/python NUM_ENVS=512 MAX_AGENT_STEPS=40000000 bash tasks/Unscrew/part4/C_Wiring/launch_remote.sh Unscrew32_HYB_s42 0 42 HYB 32
```

双 A6000 可分别跑 HYB 和 OBJ：

```bash
PY=$HOME/miniconda3/envs/isaac/bin/python NUM_ENVS=512 bash tasks/Unscrew/part4/C_Wiring/launch_remote.sh Unscrew32_HYB_s42 0 42 HYB 32
PY=$HOME/miniconda3/envs/isaac/bin/python NUM_ENVS=512 bash tasks/Unscrew/part4/C_Wiring/launch_remote.sh Unscrew32_OBJ_s43 1 43 OBJ 32
```

启动器默认关闭自动录像，避免训练时额外拉起 Isaac；需要时设置
`AUTO_RECORD=1`。输出与 PID：

```bash
tail -f logs/Unscrew32_HYB_s42.out
cat logs/Unscrew32_HYB_s42.pid
nvidia-smi
```

日志中必须看到 headless experience、正式母带预检通过以及
`[world] v2 验收世界匹配`。任何一项不符都应停止部署，不要使用
`UNSCREW_ALLOW_UNVERIFIED_REF=1` 绕过正式训练门禁。

## 5. 更新与复现

```bash
git fetch origin
git switch unscrew_v5
git pull --ff-only
git lfs pull
bash tasks/Unscrew/part4/C_Wiring/verify_deploy.sh 32
```

训练产物在 `logs/<NAME>/`，至少保留 `world.json`、TensorBoard 和
`stage1_nn/last.pth`；评测/录像会严格核对该 `world.json`。
