# dexmate-calib

Dexmate Vega 机器人的相机标定工具，一个 CLI（`dexcalib`）、一个 `.venv`、两条流程：

| 流程 | 命令组 | 相机 | 输出 | 详细文档 |
|---|---|---|---|---|
| **头部相机内参** | `dexcalib intrinsics` | 头部 ZED X Mini 左目，经 Nano 上的 `zed_streamer` TCP 流 | `K`（HD1200 / 1920×1200 / rectified LEFT，畸变固定为零） | [docs/head_left_intrinsics.md](docs/head_left_intrinsics.md)、[docs/intrinsics.md](docs/intrinsics.md) |
| **外部相机手眼标定** | `dexcalib extrinsics` | 固定在环境中的 Azure Kinect（eye-to-hand），板贴在左臂 `L_ee` | `T_base_cam`（Kinect color 相机在 URDF `base` 系下的位姿）、`T_base_depth` | [docs/handeye.md](docs/handeye.md) |

两条流程共享标定板 profile（`configs/boards/`）、ChArUco/AprilTag 检测器、session 落盘与
离线可复算的设计。安装见 [docs/install.md](docs/install.md)。

## 安装

```bash
cd ~/Projects/dexmate-calib
sudo scripts/install_kinect_sdk.sh   # 仅 Kinect 接本机时：Azure Kinect SDK 1.4.2 + udev 规则
scripts/setup_env.sh                 # uv + 托管 Python 3.12 + .venv（--minimal / --no-kinect 可选）
source .venv/bin/activate
dexcalib --help
```

手工等价命令：`uv sync --managed-python --extra dev --extra robot --extra kinect`
（`robot` = dexcontrol + dexmate-urdf + pinocchio，`kinect` = pyk4a）。仓库内只有这一个 `.venv`。

## 标定板

```bash
dexcalib board list                                # 全部 profile：<类型>-<行×列>-<边长>
dexcalib board verify --board <name> --image x.png # 用实拍图检查配置（ChArUco 期望 35 markers / 54 corners）
dexcalib board identify --image x.png              # 扫描全部 ArUco/AprilTag 字典，识别未知 tag 的家族与 id
dexcalib board render --board apriltag-4x4-48mm --output grid.png   # 生成可打印 AprilTag 网格
```

| `--board` | 类型 | 用途 |
|---|---|---|
| `charuco-10x7-27mm`（默认，别名 `dexmate-10x7`） | ChArUco 10×7，27 mm 方格 | 内参 + 手眼，精度最好 |
| `apriltag-4x4-37.5mm` / `apriltag-4x4-48mm` | tag36h11 网格 | 手眼（RobotCamCalib 同款板及其缩印） |
| `apriltag-1x1-75mm` | 单个 tag36h11 id 7 | 手眼，精度受限，需更多视图 |

内参流程只接受 ChArUco；手眼流程接受任意 profile。板必须刚性、平整，`tag_size_m`/
`square_length_m` 必须实测。

---

## 一、头部相机内参（ZED X Mini，`dexcalib intrinsics`）

**锁定条件**：Camera Jetson `192.168.50.22:30000`，serial `59595115`，ZS02 协议（兼容 ZS01），
HD1200 / 1920×1200 / `sl::VIEW::LEFT` rectified；输出畸变固定为零；板 `charuco-10x7-27mm`。
求解器拒绝任何不满足这些条件的 session，不会静默猜测或交换板的 X/Y。

**一站式**（本机通过公钥 SSH 保持一个 Nano 会话，启动固定参数 streamer、验证、采集、关闭，
然后离线求解；不装 service、不写 sudoers、不用 `--clean`）：

```bash
dexcalib doctor network                                   # 只读连通性检查
dexcalib intrinsics quickstart --board charuco-10x7-27mm --samples 40
```

分步：`intrinsics capture`（只采集）→ `intrinsics solve <session>`；多 session 联合：
`intrinsics solve-multi <s1> <s2> ... --output <dir>`。手动启动 streamer、网络路径、采集质量门、
求解步骤（清晰度筛选、姿态多样性、robust 剔除、K-fold、even/odd）和 `results/` 文件说明见
[docs/intrinsics.md](docs/intrinsics.md)。

**当前参数基准**（serial `59595115`，HD1200，rectified LEFT）：

```text
SDK factory rectified（默认生产基准）：fx = fy = 746.9691   cx = 959.9904   cy = 585.4913
dexmate-calib 5-session pooled（193 视图）：fx = 747.6196  fy = 748.7749  cx = 959.6954  cy = 584.7266
D = [0, 0, 0, 0, 0]
```

两套参数只适用于 1920×1200 rectified LEFT；下游 resize/crop 必须显式变换 `K`。完整记录与
适用边界见 [docs/head_left_intrinsics.md](docs/head_left_intrinsics.md)。

---

## 二、外部相机手眼标定（Azure Kinect ↔ Vega base，`dexcalib extrinsics`）

**锁定条件**：eye-to-hand；Kinect serial `000299113912`，USB 3，color 1536P（2048×1536）+
depth NFOV，内参用 SDK 出厂值（含 8 参数畸变、`T_color_depth`）；机器人 `vega_1p`
（dexmate-urdf），`base_frame = base`，板贴左臂 `L_ee`；机器人只允许操作者逐关节小步遥操作
移动，`dexcalib` 从不自动规划或执行位姿序列。默认值都在 `configs/handeye.yaml`。

**采集**（单终端；预览窗口里 `w/s` 逐关节小步动手臂、`0-9` 选关节、`t` 切躯干、空格保存、
`q` 结束；每次保存前检查板质量与机器人静止）：

```bash
lsusb | grep 045e && dexcalib kinect info          # 必须出现 045e:097a SuperSpeed hub
dexcalib extrinsics capture --teleop --samples 30                       # ChArUco 板
dexcalib extrinsics capture --teleop --board apriltag-4x4-37.5mm --samples 30
dexcalib extrinsics capture --teleop --board apriltag-1x1-75mm --samples 50
```

不加 `--teleop` 时用 dexcontrol 自带的 `keyboard_joint_control.py` 在另一个终端移动机器人。
采集只存原始图像、depth、关节角和检测统计；FK 在求解时做，型号/link 写错可事后重解。

**求解**：

```bash
dexcalib extrinsics solve calibration_data/handeye_kinect/<session>
dexcalib extrinsics solve <session> --compare        # 同时跑 RobotCamCalib 移植法并打印差异
dexcalib extrinsics selftest                          # 合成场景自检
```

求解 `T_base_link_i · Y = X · T_cam_board_i` 中的 `X = T_base_cam` 与 `Y = T_link_board`。默认
方法：PnP（IPPE 双候选）→ Kronecker 闭式初值 → 翻转消解 → SE(3) 上以重投影误差做 LM refine
（Huber）→ 逐视图剔除 → leave-one-out；`--method robotcamcalib` 是 `RobotCamCalib/extr_calib.py`
求解器的逐行移植（交替 LS 初值 + 位姿残差 GN），只把输入接到本仓库的板/内参/FK。

结果写入 `<session>/results/`：`T_base_kinect_external.yaml`（K、畸变、`T_base_cam`、
`T_base_depth`、RMS、样本数）、`handeye_result.json`、`per_view.csv`、
`reprojection_contact_sheet.jpg`。判定标准（经验值）：RMS < 1 px、LOO 平移离散度 < 3 mm、
旋转轴秩 3、剔除 ≤ 20%。完整说明见 [docs/handeye.md](docs/handeye.md)。

---

## 方法来源与依赖边界

采集质量、稳健剔除、交叉验证的组织方式参考了同级 `RobotCamCalib`
（`intr_calib_charuco.py`、`extr_calib.py`），默认求解器在本仓库独立实现；手眼另提供
`--method robotcamcalib` 逐行移植作为对照。本仓库不 import 或运行 RobotCamCalib，不引入其
USB camera、fisheye、pupil-apriltags 或 raw-image distortion 模型；AprilTag 用 OpenCV 内置字典。
`dexcontrol` 只用于读关节角和 `--teleop` 的逐关节相对运动。

## 测试

```bash
pytest
```

不需要连接机器人或相机：ZS01/ZS02 header、HD1200/serial gate、board schema（ChArUco 与
AprilTag）、质量筛选、内参合成 session、SE(3) 工具、闭式初值约定、含噪/离群/单 tag 合成
场景、RobotCamCalib 移植数值一致性、URDF FK 链、capture→solve 端到端。

## 数据与安全

- `calibration_data/` 默认被 Git 忽略。
- 仓库中不得保存 SSH 密码、PIN、私钥或证书；SSH 私钥只留在本机。
- 工具不会修改 Nano 上已有的 `zed_stream` 源码，quickstart 不会安装服务或写 sudoers。
- 不会在断线重连时静默切换 camera、resolution 或 board profile。
- 手眼标定不会自动规划或执行位姿序列：只能由操作者逐关节小步遥操作（`--teleop` 或
  dexcontrol 键盘示例），每次按键一步、松手即停。
