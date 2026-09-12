# 安装指南

最后更新：2026-08-17。覆盖三种使用场景：

| 场景 | 需要 | 一条命令 |
|---|---|---|
| A. 头部 ZED 内参 / 离线求解任何 session | Python 环境 | `scripts/setup_env.sh --minimal` |
| B. 手眼标定，但 Kinect 不接本机 | A + dexcontrol/pinocchio/URDF | `scripts/setup_env.sh --no-kinect` |
| C. 手眼标定完整流程（Kinect 接本机） | B + Azure Kinect SDK | `sudo scripts/install_kinect_sdk.sh` 然后 `scripts/setup_env.sh` |

所有 Python 依赖都进仓库内唯一的 `.venv`（内参与手眼共用），由 `uv` 按 `uv.lock` 锁定。

## 1. 系统前提

- Linux x86_64（实测 Ubuntu 24.04；22.04 亦可）。macOS 只能做场景 A/B——Azure Kinect SDK 没有 macOS 版本。
- `curl`、`git`、能访问 GitHub / PyPI / packages.microsoft.com。
- 场景 C 额外需要：一个 **USB 3** 口和 **USB 3 数据线**（Kinect 原装 C-to-A 线，或标注 5 Gbps
  以上的 C-C 线；普通 C-C 充电线只有 USB 2.0，depth 打不开）。
- 场景 B/C 连接机器人需要 dexcontrol 的通信配置：环境变量 `ROBOT_NAME`（如 `dm/vgd1262ab823-1p`）
  和 `~/.dexmate/comm/zenoh/` 下的 zenoh 配置，与 V2AP-demo 的 `setup.sh` 相同。`dexcalib` 只在
  真正打开机器人时才需要它们，离线求解不需要。

## 2. Azure Kinect SDK（仅场景 C，需要 sudo）

```bash
sudo scripts/install_kinect_sdk.sh
```

脚本做的事（幂等，可重复运行）：

1. 下载微软 Ubuntu 18.04 源里的 `libk4a1.4` / `libk4a1.4-dev` / `k4a-tools` 1.4.2 和
   jammy 的 `libsoundio1`（24.04 里只有 `libsoundio2`）到 `~/.cache/dexmate-calib/k4a/`；
2. 用 `ACCEPT_EULA=Y` 非交互接受微软 EULA 并安装（先 `dpkg -i libk4a1.4`，再让 apt 补齐依赖，
   避免半安装状态卡住）；
3. 安装 `/etc/udev/rules.d/99-k4a.rules` 并 reload，让普通用户能打开设备。

安装后应看到 `/usr/lib/x86_64-linux-gnu/libk4a.so.1.4.2` 与
`/usr/lib/x86_64-linux-gnu/libk4a1.4/libdepthengine.so.2.0`。卸载：`sudo scripts/install_kinect_sdk.sh --uninstall`。

## 3. Python 环境（不需要 sudo）

```bash
git clone git@github.com:jiaka1chen/dexmate-calib.git ~/Projects/dexmate-calib
cd ~/Projects/dexmate-calib
scripts/setup_env.sh            # 或 --minimal / --no-kinect
source .venv/bin/activate
dexcalib --help
```

脚本做的事：

1. 没有 `uv` 就装到 `~/.local/bin`；
2. `uv python install 3.12`，并让 `uv sync --managed-python` 使用这个带头文件的 CPython
   （系统 Python 通常缺 `python3-dev`，`pyk4a` 的 C 扩展会编译失败）；`.python-version` 固定 3.12；
3. `uv sync` 对应 extras；
4. 逐个 import 验证。

Extras 对应关系（`pyproject.toml`）：

| extra | 包 | 用途 |
|---|---|---|
| （核心） | numpy, opencv-contrib-python, PyYAML | 全部标定算法 |
| `dev` | pytest, ruff | 测试与 lint |
| `robot` | dexcontrol, dexmate-urdf, pin | 读关节 / 遥操作 / URDF FK |
| `kinect` | pyk4a（源码编译，需要第 2 节的头文件） | Kinect 采集与出厂标定 |

手工等价命令：`uv sync --managed-python --extra dev --extra robot --extra kinect`。

## 4. 验证

```bash
pytest -q                                   # 不需要任何硬件
dexcalib extrinsics selftest                # 手眼求解器合成自检
lsusb | grep 045e                           # 场景 C：应出现 045e:097a Generic Superspeed Hub
dexcalib kinect info                        # 场景 C：出厂内参、serial
dexcalib doctor network                     # 连接机器人时：ZED streamer 网络检查
```

## 5. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `lsusb -t` 里 Kinect 设备都是 `480M`，没有 `097a` | 线或口是 USB 2.0。换 USB 3 线/口后重插。 |
| `dexcalib kinect info` 报 `No Azure Kinect`/权限错误 | udev 规则没生效：`ls -l /dev/bus/usb/*/*` 应为 `root plugdev rw-rw-rw-`；重跑 SDK 脚本或重插。 |
| `apt` 抱怨 `libk4a1.4 ... not installed` / 包状态 `iU` | 之前 EULA 未接受导致半安装。直接重跑 `sudo scripts/install_kinect_sdk.sh`（内部用 `dpkg -i` + `apt -f` 修复）。 |
| `pyk4a` 编译报 `Python.h: No such file` | 用了系统 Python。删掉 `.venv` 重跑 `scripts/setup_env.sh`（强制 managed 3.12），或 `sudo apt install python3.12-dev`。 |
| `pyk4a` 报 `No distribution was found` / `Invalid Wheel-Version` | uv 缓存损坏（0 字节文件）。`uv cache clean pyk4a numpy setuptools` 后重跑。 |
| `dexcontrol` 打开机器人超时 | 检查 `ROBOT_NAME`、`~/.dexmate/comm/zenoh/`、与 `192.168.50.20` 的连通性；离线求解不受影响。 |
| `k4aviewer` 字体极小 | 它不支持 HiDPI；本项目不用它，用 `dexcalib kinect snapshot`。 |
