# 新机器部署指南

克隆本仓库后,按本页从零装到能跑。已在 Ubuntu 22.04 + NVIDIA GPU 上验证。

## 一、仓库里自带什么、缺什么

**自带**(克隆即得):全部代码与文档、PICO SDK 的 Python 绑定源码(`external/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/`,含本项目的修改)、头显连接服务安装包(`external/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb`)、Isaac 用的机器人 USD(`assets/isaac/Robots/`)、回归素材与全部基线(`logs/`)。

**不自带,需向厂商获取**(路径是代码里的默认引用位置):

| 部件 | 放到哪 | 用途 |
|---|---|---|
| Dexmate `dexcontrol` SDK(含其自己的 `.venv`) | `~/Dexmate/dexcontrol/` | 真机手臂桥接 |
| Dexmate `dexmate-urdf` | `~/Dexmate/dexmate-urdf/` | IK 用的官方 URDF(关节限位的唯一权威) |
| Sharpa Wave SDK 5.0.3 | `/opt/sharpa-wave-sdk/` | Sharpa 真手 |

只跑仿真不接真机的话,`dexcontrol` 和 Sharpa SDK 可以不装;`dexmate-urdf` 必须有(IK 依赖)。

## 二、安装步骤

```bash
# 1. 系统依赖
sudo apt install avahi-utils                 # 设备面板的 mDNS 探测
sudo apt install ./external/XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb   # 头显连接服务

# 2. 三个 Python 环境,用仓库自带的脚本装(需要 uv:https://docs.astral.sh/uv/)
bash scripts/setup_env.sh          # .venv:控制台/页面/手套/离线分析,装完自动跑自检
bash scripts/setup_pico_env.sh     # .venv-pico:PICO 数据读取(SDK 用仓库 external/ 内的源码)
bash scripts/setup_isaac_env.sh    # .venv-isaac:Isaac 仿真,首次下载 GB 级(pypi.nvidia.com)

# 3. 环境变量(写进 ~/.bashrc)
export MAGICSIM_ASSETS=<本仓库绝对路径>/assets/isaac    # USD 从仓库内取

# 4. 真机网络(每次网口拔插/重启后也要跑一次)
sudo bash scripts/setup_robot_net.sh
```

## 三、验证与入口

```bash
.venv/bin/python scripts/check_all.py        # 19 项免硬件自检,应全过
.venv/bin/python scripts/vega_console.py     # 操作入口,浏览器 :8086
```

预期结果说明:`check_all.py` 应 19/19(`setup_env.sh` 结尾自动跑的 `check_env.py`
目前为 9/10 —— 失败的「sequential tracking」一项在开发机上也是同样数字,原因是
手部重定向后来加入了平滑滤波(`low_pass_alpha 0.8`,真机验收过的配置),该项的
阈值没有随之更新,不代表安装有问题)。

之后按 [SOP_wholebody_teleop.md](SOP_wholebody_teleop.md)(全身)与 [SOP.md](SOP.md)(手部)操作。
头显侧设置见 [docs/PICO_teleop.md](docs/PICO_teleop.md);项目全貌与未解决问题见
[docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md)。

## 四、已知的机器相关事项

- 网口名写死为 `enp3s0`(本台式机;实验室笔记本为 `enp49s0`)(`scripts/vega_console.py` 顶部 `WIRED_IF`、`scripts/setup_robot_net.sh`),换机器要改;
- 机器人 / Sharpa 手 / 手套的网段与 IP 见 `scripts/setup_robot_net.sh` 注释与 SOP 网络表;
- 停止仿真必须走页面「全部停止」或 `tmux kill-session -t vega`,不要 Ctrl-C 杀 Isaac 进程(会损坏 nvidia_uvm,恢复命令见 SOP 第四节)。
