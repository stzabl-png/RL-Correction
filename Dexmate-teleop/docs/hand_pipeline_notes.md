# 手部管线的原始详细说明(存档)

## 两种运行模式

```
模式 A（双进程桥接，默认；真机也走这条路）
  [teleop 进程 .venv]  手套(wuji|mock) → retarget → ZMQ PUB(:5556, JSON+CRC)
  [Isaac Lab 进程]     sim/test_env_sharpa.py --motion zmq  ← SUB 跟随
  [真机进程]          scripts/sharpa_real_runner.py → sinks/sharpa_real.py  ← SUB 跟随真机(单/双手)

模式 B（单进程，全部塞进 Isaac Lab python）
  [Isaac Lab 进程]     sim/teleop_isaac_single.py：手套 + retarget + 仿真，无 ZMQ
```

模式 A 的优点：一份 qpos 流同时喂 sim/real（天然 A/B 对照）、崩溃隔离、teleop 侧可用最新依赖。模式 B 的优点：少开一个终端、env 内直接拿 retarget 结果（采数据方便）。
**numpy 说明**：dex-retargeting 声明 `numpy>=2.0`，但源码无 numpy-2 专属 API——已实证在 Isaac 的 numpy 1.26 上跑出**逐位一致**的结果（`--no-deps` 安装即可，脚本已处理；详见 docs/references/03 §6），所以两种模式都成立。

## 目录结构

```
magicdexmate/            # 核心包（teleop venv 与 Isaac python 通用）
  skeleton.py            # MediaPipe 21 点约定、HandFrame、名字→索引
  sources/               # wuji_source.py（真手套，wuji-sdk 2026.6.18 实测；recv 非阻塞已处理）
                         # mock_source.py（合成手：open/fist/wave/pinch/cycle）
  retarget/              # frames.py（逐帧 estimate_frame→MANO，对手套腕系免疫）
                         # builder.py（构建 SeqRetargeting）mapping.py（按名映射 SDK 22 序+限位 clip）
                         # natural_distal.py（dexpilot 远端松弛：非捏合时拇/小指尖跟手套）
  sinks/                 # qpos_publisher.py（ZMQ，hello 带 joint_names+CRC）sapien_viz.py（调试视图）
                         # sharpa_real.py（真机 SDK 驱动：限位/限速/engage/看门狗/碰撞保护）
configs/retargeting/     # sharpa_wave_{right,left}_{vector,dexpilot}.yml（scaling 1.07 实测）
assets/robots/hands/     # Sharpa URDF+mesh（已入库，Apache-2.0 © Sharpa；prepare_assets.py 可重新生成）
third_party/             # dex-retargeting（vendored，MIT © Yuzhe Qin；只留 src+pyproject+LICENSE）
sim/                     # sharpa_scene.py（共享场景，参数照抄 sharpa-rl-lab）
                         # test_env_sharpa.py（模式 A 消费端；sine/home/zmq）
                         # teleop_isaac_single.py（模式 B 单进程，裸手）
                         # teleop_dual_sharpa.py（模式 A 双手消费端，左右两个独立 Articulation）
                         # vega_sharpa_scene.py + teleop_vega_sharpa.py（Vega-1P 整机，
                         #   资产/增益取自 MagicSim，腕部 rpy 姿态 IK）
scripts/                 # setup_env.sh / install_into_isaaclab.sh / prepare_assets.py
                         # check_env.py（自检）/ teleop_retarget.py（模式 A 生产者）
                         # sharpa_real_runner.py（真机接收端，ZMQ→SDK，单/双手）
tests/                   # pytest（11 项）
plans/  docs/references/ # 计划与参考笔记（见文末导航）
```

## 环境安装

### 1. teleop 环境（uv，已建好在 `.venv/`）

```bash
bash scripts/setup_env.sh      # uv venv py3.11 + torch-cpu + dex-retargeting(-e, vendored 在 third_party/) + wuji-sdk + 自检
# 可选预览窗（teleop_retarget.py --viz，非仿真器；仿真是 Isaac）：uv pip install -p .venv/bin/python "sapien==3.0.0b0"
```

### 2. Isaac 环境（uv venv，`.venv-isaac/`）

isaacsim + isaaclab 以 pip wheel 形式装进独立 uv 环境（`isaaclab[isaacsim]==2.3.0` 自动带配套 isaacsim；retarget 栈同环境内、numpy 受约束保护）：

```bash
bash scripts/setup_isaac_env.sh        # 首次约 GB 级下载（pypi.nvidia.com）
```

首次启动需接受 Omniverse EULA：命令前加 `OMNI_KIT_ACCEPT_EULA=YES`。
（备选：若想复用 `~/isaacsim` standalone 安装，可用 `scripts/install_into_isaaclab.sh ~/isaacsim/python.sh` 把 retarget 栈装进它自带的 python——但其老 pip 解析很慢，推荐 uv 路线。）

### 3. Wuji 数据手套（输入端）

- **SDK**：`setup_env.sh` 已 `pip install wuji-sdk`（实测 2026.6.18）。要求 Ubuntu 22+/py3.10+、与手套同网段。手动装：`uv pip install -p .venv/bin/python wuji-sdk`。
- **网络**：主机网卡设 `192.168.1.x/24`；**左手 192.168.1.100、右手 192.168.1.101**（UDP 50000/50001）。`ping` 验证。
- **标定**：用 **Wuji Studio**（每用户/每设备生成手部 URDF；**不标定则 skeleton 置信度低、帧被看门狗丢弃、手不动**）：
  ```bash
  sudo apt install ~/Downloads/wuji-studio_2026.6.2_amd64.deb && wuji-studio
  ```
- 数据接收要点（实测）：`glove.hand_skeleton().subscribe().recv()` **非阻塞**（无帧返回 `None`，循环须判空）；`pose.position` 是 `[x,y,z]` list、`orientation` 是 `Quaternion`。详见 docs/references/01。

### 4. SharpaWave 真机 SDK（输出端，只有驱动真手才需要）

- 装厂商 deb → `/opt/sharpa-wave-sdk/`（v5.0.3，预编译 `sharpa.so` 覆盖 py3.10–3.13）：
  ```bash
  sudo dpkg -i sharpa-wave-sdk_*.deb
  ```
- python 包不进 pip：代码已 `sys.path.insert(0, "/opt/sharpa-wave-sdk/python")`；若动态库不解析，**启动时加 `LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib`**。
- **网络**：手在 `192.168.10.x`（**左手 .10、右手 .20**，host NIC `192.168.10.240`）；设备发现走**广播心跳**（端口 54321，与 IP 无关）。确认在线：
  ```bash
  LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib PYTHONPATH=/opt/sharpa-wave-sdk/python \
    .venv/bin/python -c "import time,sharpa; m=sharpa.SharpaWaveManager.get_instance(); time.sleep(2); print(m.get_all_device_sn())"
  ```
- 关节序 = `sharpa_sdk_joint_names`（与手册 §1.1 一致），单位 **rad**；`set_joint_position(q22, interpolate)`。手册：`~/MagicSharpa/UserManual/`。

## 快速开始

```bash
# 自检（teleop venv）：imports/资产/4配置构建/跟踪精度/mock管线/ZMQ/耗时 —— 10/10
.venv/bin/python scripts/check_env.py
.venv/bin/python -m pytest tests/ -q                          # 11/11

# ---- 模式 A：双进程桥接 ----
.venv/bin/python scripts/teleop_retarget.py --source mock --motion cycle --viz   # 终端1
.venv-isaac/bin/python sim/test_env_sharpa.py --motion zmq                       # 终端2
.venv-isaac/bin/python sim/test_env_sharpa.py --motion sine                      # 只测场景/资产/执行器

# ---- 模式 B：单进程（全在 .venv-isaac）----
.venv-isaac/bin/python sim/teleop_isaac_single.py --source mock --motion cycle
.venv-isaac/bin/python sim/teleop_isaac_single.py --source mock --headless --duration 10  # 无显示冒烟

# ---- Vega-1P + Sharpa 整机（单进程；手指=retarget，腕=手套 rpy 姿态 IK）----
# 资产/增益来自 MagicSim（Assets/Robots/vega_1p_sharpa.usd + vega1psharpa.py 的实测参数）
OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/teleop_vega_sharpa.py --source mock
OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/teleop_vega_sharpa.py --source wuji --hand right
# 说明：Wuji 手套只有腕部姿态（IMU rpy），没有 translation —— 腕控用 DifferentialIK
# 做"姿态跟踪"：位置目标钉在 R_ee 初始位置，姿态=初始姿态⊗手套相对旋转；
# P4 接入 Pico 后只需把位置目标换成 VR 轨迹，控制结构不变。--wrist off 可关腕控。

# ---- 真手套驱动仿真（把 mock 换成 wuji；--source wuji 时 scripts/* 前加 PYTHONPATH= 躲 ROS）----
PYTHONPATH= .venv/bin/python scripts/teleop_retarget.py --source wuji --hand right        # 模式 A 生产者
.venv-isaac/bin/python sim/teleop_isaac_single.py --source wuji --hand right               # 模式 B

# ---- 仿真双手（左右两个独立 Articulation 并排）----
# 推荐：单进程双手生产者（一条命令同时跑左+右，右→:5556 左→:5557）→ 只需 2 个终端
PYTHONPATH= .venv/bin/python scripts/teleop_retarget_dual.py --source wuji                  # 终端1：双手生产者
OMNI_KIT_ACCEPT_EULA=YES .venv-isaac/bin/python sim/teleop_dual_sharpa.py                   # 终端2：双手消费端（--motion sine 可无手套自测）
# 备选：两个单手生产者各占一端口（崩溃隔离/单独控制每只手时用）
PYTHONPATH= .venv/bin/python scripts/teleop_retarget.py --source wuji --hand right --pub tcp://*:5556
PYTHONPATH= .venv/bin/python scripts/teleop_retarget.py --source wuji --hand left  --pub tcp://*:5557

# ---- 真机遥操作（SharpaWave 实手；bench 上、手周围净空）----
# 单手：终端1 生产者 + 终端2 真机接收端
PYTHONPATH= .venv/bin/python scripts/teleop_retarget.py --source wuji --hand right
LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib PYTHONPATH= .venv/bin/python \
    scripts/sharpa_real_runner.py --hand right --sub tcp://127.0.0.1:5556
# 双手：单进程双手生产者 + 一个接收端 --hand both（推荐，2 终端）
PYTHONPATH= .venv/bin/python scripts/teleop_retarget_dual.py --source wuji
LD_LIBRARY_PATH=/opt/sharpa-wave-sdk/lib PYTHONPATH= .venv/bin/python \
    scripts/sharpa_real_runner.py --hand both --sub tcp://127.0.0.1:5556 --sub-left tcp://127.0.0.1:5557
# （备选：两个单手生产者 scripts/teleop_retarget.py --hand right/left --pub :5556/:5557）
```

**真机安全**：接收端启动只 connect+configure（电机上电保持当前姿、**不动**），**默认未 engage**。运行中按键：`e` engage（开始跟手）/ `w` freeze / `q` 回 home / `x` 停止退出。手套帧 stale>150ms 或故障码非零 → 自动 freeze。目标值限位 clip×0.9 + 每步限速（<固件 20° 自动插值阈值）。

常用参数：`--hand right|left`、`--mode vector|dexpilot`（**默认 dexpilot**；dexpilot 下 `--relax-distal` 默认开：非捏合时拇/小指尖跟手套、捏合对位不变，`--no-relax-distal` 关）、`--rate/--control_hz`（默认 60；真机接收端默认 20）、`--sn/--address`（指定手套）、`--duration N`（定时退出，冒烟用）、`--no-pub`。

## 实测指标（2026-06-11，mock 全链路）

| 指标 | 数值 |
|---|---|
| 指尖向量跟踪误差（FK 验证，60Hz 顺序跟踪） | 四指 7-11mm，拇指 ~17mm |
| retarget 单步耗时 | p50 2.7ms / p95 5.0ms（≈200Hz 上限；numpy1.26+pin2.7 组合实测 2.0ms） |
| 端到端（采集→ZMQ 发布） | p50 6.4ms @ 60Hz |
| 自检 / 单测 | check_env 10/10，pytest 11/11 |

## 已知问题与注意

1. **随机极端位姿单步求解差**（~52mm 驻点；梯度已验证正确，nlopt 2.8/2.10 一致）：teleop 连续小步跟踪不受影响；离线批处理勿从任意位姿单步求解。详见 docs/references/03 §5.5。
2. 拇指跟踪偏松：mock 拇指几何粗糙 + 人/机拇指构型差异，真手套数据到位后按计划 01 R5 调 scaling/DexPilot 参数。
3. `sim/*.py` 用 `.venv-isaac`（isaacsim+isaaclab+retarget 栈），`scripts/*.py` 用 `.venv`；两个环境都由 uv 管理。Isaac 首次启动加 `OMNI_KIT_ACCEPT_EULA=YES`。
4. **isaacsim 退出会挂死**（`simulation_app.close()` 不返回）：三个 sim 脚本都已内置 20 秒退出看门狗（数据落盘后 `os._exit(0)` 硬退），无需手工 pkill。另：本机 CPU governor=powersave，仿真显著慢于墙钟（Kit 启动时有警告），跑 headless 冒烟请给大 timeout 或 `sudo cpupower frequency-set -g performance`。
5. **headless 自检拍照**：`sim/teleop_vega_sharpa.py --snap-dir DIR --snap-times "0.5,4,15.5" --cam-eye "1.6,-1.3,1.7" --cam-target "0.0,-0.2,1.15"`（自动启用离屏渲染），可在无显示器环境拍 RGB 截图核对动作；`--debug-joints` 输出分组关节遥测。
6. **vega_1p_sharpa.usd 的雅可比行索引**：该 USD 的 ArticulationRootAPI 在 /root_joint 上，DiffIK 取雅可比应使用 `body_idx` 行（脚本默认 `--jacobi-row body`，实测保持误差 0-2mm；教程式 `body_idx-1` 会 200-600mm 跑飞）。
7. GPU 与其他任务并发压力下，Kit 图形上下文可能 Xid 31 MMU fault 并**静默 exit 0**（无 traceback；`journalctl -k | grep -i xid` 可查）——遇到"无输出正常退出"先查这个和显存占用。
4. **SharpaWave 真机 SDK 已安装**（`/opt/sharpa-wave-sdk` v5.0.3 + 手册 `~/MagicSharpa/UserManual/`），真机驱动代码已写（`sinks/sharpa_real.py` + `scripts/sharpa_real_runner.py`，左右/双手）。**但两只手当前不在线**（ping 192.168.10.10/.20 不通、SDK 发现为空）——上电接网后才能实跑验证；代码侧仅验过 import/SDK 加载/安全默认。
5. docs.dexmate.ai 需客户账号（401）；omniteleop 的 VR 代码面向 Meta Quest，**PICO 兼容性待确认**（P4 阻塞项）。

## 文档导航

| 文档 | 内容 |
|---|---|
| [plans/00_roadmap.md](plans/00_roadmap.md) | 总路线图 P0-P6、架构、决策记录 |
| [plans/01_retarget_plan.md](plans/01_retarget_plan.md) | retarget 计划（含实施状态：R0/R1/R4/R6 已完成）|
| [plans/02_sim_scene_plan.md](plans/02_sim_scene_plan.md) | Isaac 场景：浮动基座、双手、Vega 装配、触觉 |
| [plans/03_sharpa_real_teleop_plan.md](plans/03_sharpa_real_teleop_plan.md) | 真机操控：bring-up、安全链、联调步骤 |
| [docs/references/01_wuji_glove.md](docs/references/01_wuji_glove.md) | Wuji 手套硬件/SDK/数据流/标定 |
| [docs/references/02_sharpa_wave_hand.md](docs/references/02_sharpa_wave_hand.md) | SharpaWave 关节表/SDK API/触觉 180Hz |
| [docs/references/03_dex_retargeting.md](docs/references/03_dex_retargeting.md) | dex-retargeting 用法 + 实测结论/已知问题 |
| [docs/references/04_dexmate_vega.md](docs/references/04_dexmate_vega.md) | Vega/dexcontrol/omniteleop/SLAM |
| [docs/references/05_sharpa_sim_assets.md](docs/references/05_sharpa_sim_assets.md) | URDF/USD/MJCF 资产变体 + sharpa-rl-lab 解析 |
