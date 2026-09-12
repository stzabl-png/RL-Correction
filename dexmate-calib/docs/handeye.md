# 外部相机手眼标定（Azure Kinect ↔ Vega base）

最后更新：2026-08-17（America/Los_Angeles）

本文说明如何用 `dexcalib extrinsics` 标定固定在环境中的 Azure Kinect 相对 Dexmate Vega
机器人 `base` 的外参 `T_base_cam`，并记录当前锁定的硬件与设计决定。

## 场景与约定

- 类型：**eye-to-hand**。相机固定不动，ChArUco 板刚性固定在机器人 link 上，机器人运动。
- 相机：Microsoft Azure Kinect，serial `000299113912`，通过 USB 3 连接运行 `dexcalib`
  的 Linux 主机。Color `1536P`（2048×1536），depth `NFOV_UNBINNED`（640×576），15 fps。
- 内参：直接使用 SDK 出厂标定（color `K` + 8 参数 rational 畸变，depth `K`，
  `T_color_depth`）。捕获时整体导出为 `camera_calibration.json`，离线求解不依赖 SDK。
- 机器人：`vega_1p`（`dexmate-urdf`），`base_frame = base`（URDF 根），板贴在 **`L_ee`**
  （左臂法兰）。`base → L_ee` 链上的关节：`torso_j1..3`、`L_arm_j1..7`。
- 标定目标：默认 `dexmate-10x7` ChArUco 板（同内参标定，27 mm 方格，`DICT_5X5_50`）；也支持
  **AprilTag 网格**（任意 rows×cols，含 1×1 单个 tag），见下文"标定目标"。
- 运动方式：**只允许手动、逐关节、小步长**。两种等价方式：
  (a) `dexcalib extrinsics capture --teleop`——预览窗口同时接收运动键和拍照键，单终端；
  (b) 另开终端跑 dexcontrol 自带的 `examples/advanced_examples/keyboard_joint_control.py`，
  `dexcalib` 只读关节角。任何情况下 `dexcalib` 都不会自动规划或执行位姿序列。

坐标约定：`T_a_b` 是 4×4 齐次矩阵，把 `b` 系下的点变换到 `a` 系，即 `b` 在 `a` 中的位姿。
`T_base_cam` 中的 `cam` 是 Kinect **color** 相机光心系（OpenCV：x 右、y 下、z 前）。
`T_base_depth = T_base_cam @ T_color_depth` 一并写入结果。

## 标定目标

求解器只消费"板坐标系 3D 点 ↔ 图像 2D 点"，目标类型由 `configs/boards/*.yaml` 的
`target_type` 决定（`dexcalib board list` 列出全部）：

| `--board` 名称 | 类型 | 角点/视图 | 说明 |
|---|---|---|---|
| `charuco-10x7-27mm`（默认，别名 `dexmate-10x7`） | `charuco` | 54 | 27 mm 方格 / 20.25 mm marker；精度最好，实物已实测 |
| `apriltag-4x4-37.5mm` | `apriltag_grid` | 64 | tag36h11 id 0–15，marker 边 37.5 mm，间距 ≈40.8 mm（RobotCamCalib 板的缩印版） |
| `apriltag-4x4-48mm` | `apriltag_grid` | 64 | RobotCamCalib 原版 `compact_apriltag_grid_4x4_tag48mm`，间距 52.17 mm |
| `apriltag-1x1-75mm` | `apriltag_grid` 1×1 | 4 | 单个 tag36h11 id 7，边 75 mm；精度受限，见下 |

命名规则：`<类型>-<行×列>-<边长>`；ChArUco 的边长是方格边长，AprilTag 的是黑框外缘边长。

AprilTag profile 字段：`apriltag.family`（`tag36h11`/`tag36h10`/`tag25h9`/`tag16h5`）、
`layout.rows/cols/tag_id_start/tag_size_m/pitch_m`，或显式 `tags: [{id, center_m, size_m}]`
（可混合不同尺寸）。板坐标系与 OpenCV ChArUco 板同手性：原点在第一个 tag 中心，x 右、y 下、
z 指向纸面内；每个 tag 角点按 OpenCV 顺序 TL、TR、BR、BL。检测用 OpenCV 内置的 AprilTag
字典，无额外依赖。`tag_size_m` 是黑色方框外缘边长，必须用卡尺实测——平移结果按它等比缩放。

相关命令：

```bash
dexcalib board identify --image photo.png        # 扫描全部 ArUco/AprilTag 字典，识别未知 tag 的家族与 id
dexcalib board render --board apriltag-4x4-48mm --output grid.png --dpi 300   # 生成可打印 PNG（100% 打印后量尺寸）
dexcalib board verify --board apriltag-1x1-75mm --image photo.png            # 用配置检测一张实拍图
dexcalib extrinsics capture --board apriltag-1x1-75mm ...                    # 采集时指定目标
```

采集/求解的质量门会自动按目标能力收紧（单 tag：最少 4 角点、1×1 网格）。

**单个平面 tag 的边界**：每视图仅 4 个角点，且存在 IPPE 位姿翻转二义性。求解器会为每视图保留两个
PnP 候选，用闭式初值预测的板位姿消解翻转（结果里 `initialisation.pnp_flips_resolved` 记录次数），
再以重投影误差 refine。合成测试（0.3–1 px 噪声）下 4×4 网格与 ChArUco 精度相当，单 tag 在 40 视图、
0.8–1.1 m 时约 0.06°/1.5 mm；实机建议 0.5–0.8 m 工作距离、50+ 视图、旋转多样性充分。

## 数学模型

每个样本 `i` 提供两件事：

- `T_base_link_i`：由记录的关节角经 URDF 正运动学得到；
- 板角点的 2D 观测，配合已知的板几何可得 `T_cam_board_i`（PnP）。

未知量是 `X = T_base_cam`（目标）和 `Y = T_link_board`（板在 link 上的安装位姿，贴板时
无法精确知道，因此一并求解）。对每个样本有

```text
T_base_link_i · Y = X · T_cam_board_i        （AX = YB 形式）
```

求解流程（`src/dexmate_calib/extrinsics/handeye.py`）：

1. 每张图用 IPPE（平面）求 `T_cam_board_i` 的候选并 LM refine（带完整畸变）；平面目标最多两个候选，
   在闭式初值后按链式预测消解翻转。
2. Kronecker 线性闭式解（Li 2010）给出 `X`、`Y` 初值。这里没有使用 OpenCV 的
   `calibrateRobotWorldHandEye`，因为 opencv-contrib-python 5.0 的 wheel 不再暴露它。
3. 在 SE(3) 上做 Levenberg–Marquardt，**直接最小化所有角点的重投影误差**（12 个参数），
   Huber 权重（默认 2 px）；因此结果不继承单帧 PnP 的位姿噪声。
4. 逐视图 RMS 高于阈值（默认 `max(3 px, median + 3·MAD)`）的样本被剔除后重新求解，
   每轮最多剔除 20%。
5. Leave-one-view-out：每次留出一个样本重解，报告 `T_base_cam` 的旋转/平移离散度和
   留出样本的重投影误差。
6. 额外报告运动多样性：link 位姿的最大两两旋转角、旋转轴秩（应为 3）、平移跨度。

### 备选求解方法：`--method robotcamcalib`（RobotCamCalib 原样移植）

`dexcalib extrinsics solve <session> --method robotcamcalib` 运行
`src/dexmate_calib/extrinsics/robotcamcalib.py`——这是 `RobotCamCalib/extr_calib.py` 的
`calibrate_cammount_and_tag_prob`（含 Lie 工具、`_wahba`、交替最小二乘初值、位姿残差
Gauss-Newton、Huber/σ 权重、停止条件）和 `apriltag_board.py` 的 bundle PnP 的**逐行移植，
算法未做任何修改**（文件内有 `BEGIN/END VERBATIM` 标记）。只在输入侧做了适配：

- 角点对应来自本仓库的板 profile/检测器（ChArUco 或 AprilTag）；
- `K`/`D` 来自 session 的相机标定（Kinect 出厂内参；RobotCamCalib 用 `D=None` 是因为它的
  RealSense 流已经去畸变）；
- `X_WorldTagmount = T_base_link`（Dexmate URDF FK），`X_WorldCammount = I`（相机安装在 base 系）；
- 输出映射：`X_CammountCam → T_base_cam`，`X_TagmountTag → T_link_board`。

选择该方法时**不**应用本仓库的 IPPE 翻转消解、逐视图剔除和 Kronecker 初值（这些不属于
RobotCamCalib）；LOO 诊断如启用只是对子集重复运行同一求解器。合成对照：大板上两者相当，
单 tag 上 RobotCamCalib 法明显差（约 5 mm vs 0.4 mm），因为它把每帧 PnP 位姿当观测。

`--compare` 同时跑另一种方法并打印两者 `T_base_cam` 的差异；该方法的结果写入
`results/handeye_result_robotcamcalib.json`，默认方法的文件名不变。默认值在
`configs/handeye.yaml` 的 `solve.method`。

`dexcalib extrinsics selftest` 用合成场景验证整条链路：0.4 px 像素噪声 + 0.03°/0.5 mm
FK 噪声、25 视图时，`T_base_cam` 误差约 0.05° / 1 mm；注入的离群视图会被识别并剔除。

## 操作流程

### 0. 环境

```bash
cd ~/Projects/dexmate-calib
sudo scripts/install_kinect_sdk.sh   # Azure Kinect SDK 1.4.2 + udev 规则（一次性）
scripts/setup_env.sh                 # uv + Python 3.12 + .venv，含 robot/kinect extras
source .venv/bin/activate
```

依赖明细、`--minimal`/`--no-kinect` 变体、dexcontrol 的 `ROBOT_NAME`/zenoh 配置以及常见问题
见 [install.md](install.md)。

### 1. 检查 Kinect

```bash
lsusb | grep 045e            # 必须出现 045e:097a Generic Superspeed Hub
dexcalib kinect info         # 出厂内参 / depth→color 外参 / serial
dexcalib kinect snapshot --output calibration_data/kinect_snapshot.png
```

如果 `lsusb -t` 里 Kinect 只有 480M（USB 2.0），换 USB 3 线或端口；depth 在 USB 2 下无法工作。

### 2. 固定板与相机

- 板刚性固定在左臂法兰（`L_ee`）附近，整个 session 内不得松动。板不需要精确对准 link 坐标系。
- Kinect 固定在三脚架/支架上，整个 session 内不得移动；标定完成后相机与机器人 base 的相对
  位置一旦改变（包括移动底盘）都必须重标。
- 采集期间**底盘和躯干不要动**（躯干关节在 FK 链上，允许动但建议固定；底盘运动会直接使
  `base` 漂移）。

### 3. 采集

推荐单终端模式（运动 + 拍照都在预览窗口里按键）：

```bash
dexcalib extrinsics capture --teleop --samples 30
```

`--teleop` 根据 `--target-link` 自动选择要控制的手臂（`L_*` → `left_arm`，`R_*` →
`right_arm`），并允许用 `t` 切到 `torso`。预览窗口按键：

| 键 | 作用 |
|---|---|
| `0`–`6`（torso 为 `0`–`2`） | 选择当前组件的关节序号 |
| `w` / `s` | 当前关节 +/− 一步（默认 0.5°/次，按住靠系统自动重复连续走） |
| `W` / `S` | 一步 ×4 |
| `-` / `=` | 步长减半 / 加倍（上限：臂 3°，躯干 1.5°） |
| `t` | 在手臂和躯干之间切换 |
| **空格** | 保存样本 |
| `q` / `Esc` | 结束 |

每次按键只发一次 `move_joint_pos(relative=True)`（速度比例默认 0.3，`--velocity-scale`）；
松开键就没有新命令，机器人不会继续走。最后一次运动命令后 1 s 内（`teleop_settle_s`）空格会
被拒绝，避免拍到还在收敛的姿态。`--step-deg` 可改单步大小。

双终端模式（不加 `--teleop`）：终端 A 跑
`python ~/Projects/dexcontrol/examples/advanced_examples/keyboard_joint_control.py --component left_arm`，
终端 B 跑 `dexcalib extrinsics capture --samples 30`，空格发给预览窗口、`w/s` 发给终端 A。

不论哪种模式，预览窗口按 **空格** 保存一个样本，`q`/`Esc` 结束。每次保存时程序会：

1. 检查板检测质量（角点数 ≥ 20、角点行列 ≥ 3×3）；
2. 连读 3 次关节角（间隔 0.15 s），最大变化 ≤ 0.002 rad 才认为机器人静止；
3. 静止确认后再抓一帧图像，用这帧和这组关节角落盘。

建议 25–40 个样本，覆盖：板在画面中心/四角、远近（0.5–1.2 m）、三个轴各 ±30° 以上的旋转。
只做平移或只绕一个轴转会让 `X` 不可观测，`solve` 会在 `motion diversity` 中提示。

选项：`--no-depth` 不采 depth；`--allow-moving` 放宽静止检查（不建议）；`--no-robot`
仅测试相机（生成的 session 不能求解）；`--solve` 采集结束后立即求解。

Session 目录：`calibration_data/handeye_kinect/<时间戳>_handeye_kinect_external_L_ee/`

```text
manifest.yaml              schema dexmate_calib.handeye_session.v1，robot/camera/board 锁定信息
board_profile.yaml         板配置快照（hash 写入 manifest）
camera_calibration.json    Kinect 出厂标定（color/depth K、畸变、T_color_depth）
color/0000.png             原始 BGR PNG（无损）
depth/0000.png             uint16 毫米深度（可选）
samples.jsonl              每行一个样本：图片路径、时间戳、关节角、静止检查、检测统计
```

采集端**不做 FK**，只存关节角；机器人型号或 link 写错时可以用 `solve --robot-model/--target-link`
重解，无需重采。

### 4. 求解

```bash
dexcalib extrinsics solve calibration_data/handeye_kinect/<session>
```

输出写入 `<session>/results/`：

```text
T_base_kinect_external.yaml   下游使用：K、畸变、T_base_cam、T_base_depth、RMS、样本数
handeye_result.json           完整结果：初值、refine 历史、LOO 统计、逐视图报告、剔除原因
T_base_cam.npy                4×4 numpy
per_view.csv                  每视图 RMS/最大误差、与 PnP 的位姿差、板距离、状态
reprojection_contact_sheet.jpg 绿=检测角点，紫=最终模型重投影，黄线=残差；红框=被剔除
```

判定标准（经验值，正式结果应满足）：

- `rms_reprojection_error_px` < 1.0（2048×1536 下）；
- LOO `T_base_cam` 平移离散度 max < 3 mm，旋转 < 0.1°；
- `motion_diversity.rotation_axis_rank == 3`，`max_pairwise_rotation_deg` > 40；
- 剔除样本不超过 20%。

## 代码结构

```text
src/dexmate_calib/geometry/transforms.py   SO(3)/SE(3) exp/log、逆、位姿误差（纯 numpy）
src/dexmate_calib/extrinsics/handeye.py    PnP（IPPE 双候选）、闭式初值、翻转消解、LM refine、剔除、LOO
src/dexmate_calib/extrinsics/robotcamcalib.py  RobotCamCalib 求解器逐行移植（--method robotcamcalib）
src/dexmate_calib/boards/apriltag.py       AprilTag 网格/单 tag profile、检测器、渲染、identify
src/dexmate_calib/extrinsics/synthetic.py  合成场景（测试与 selftest）
src/dexmate_calib/extrinsics/capture.py    手动采集循环与 session 落盘
src/dexmate_calib/extrinsics/solve.py      读 session、FK、求解、写 results/
src/dexmate_calib/extrinsics/config.py     configs/handeye.yaml 加载
src/dexmate_calib/robot/kinematics.py      pinocchio + dexmate-urdf 正运动学
src/dexmate_calib/robot/dexmate.py         dexcontrol 只读关节采样（静止检查）
src/dexmate_calib/cameras/kinect.py        pyk4a 封装：BGR 帧、depth、出厂标定导出
configs/handeye.yaml                       上述所有锁定参数的默认值
```

默认求解方法在思路上参考了同级 `RobotCamCalib/extr_calib.py`（AX=YB 联合估计、Huber），
但在本仓库独立实现；`--method robotcamcalib` 则是其求解器的逐行移植
（`extrinsics/robotcamcalib.py`），只把输入接到我们的板 profile、Kinect 内参和 Dexmate FK 上。
两者都不 import 或运行 RobotCamCalib，也不引入 Viser、xArm、RealSense、pupil-apriltags 依赖。

## 已知边界

- 只实现 eye-to-hand。若以后需要腕部相机（eye-in-hand），把 `T_base_link` 与
  `T_cam_board` 的角色互换即可复用同一求解器。
- Kinect 内参默认使用出厂值；如需自标可用同一块板走 `intrinsics` 流程做交叉验证，但目前
  `intrinsics solve` 锁定 ZED 1920×1200 rectified，需要扩展后才能用于 Kinect。
- 时间同步依赖"静止后再拍"，没有硬件触发；因此静止检查不要关闭。
- `--teleop` 只做逐关节小步相对运动，没有碰撞检查；靠近关节极限、桌面或自身时请减小步长
  或改用 dexcontrol 自己的遥操作。
