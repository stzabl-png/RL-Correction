# Dexmate 头部左相机内参：操作与标定记录

最后更新：2026-08-15（America/Los_Angeles）

本文同时是操作说明和参数记录。这里的所有数值只适用于下面这条成像链路：

- Camera：ZED X Mini，serial `59595115`
- ZED mode：`HD1200`
- 图像尺寸：`1920×1200`
- ZED view：`sl::VIEW::LEFT`
- 图像几何：ZED SDK rectified left image
- 模型：pinhole，rectified 图像的 distortion 固定为零
- 标定板：`dexmate-10x7`，10×7 squares，27.000 mm square，20.250 mm marker，
  `DICT_5X5_50`，legacy pattern false

这些参数不能直接用于 `VIEW::LEFT_UNRECTIFIED`、HD1080、被 crop 的图像或未经同步变换的
resize 图像。

## 建议参数

### ZED SDK factory rectified 参数

对于当前 streamer 输出的 `sl::VIEW::LEFT`，ZED SDK factory rectified 参数是生产基准：

```yaml
camera_serial: 59595115
resolution: [1920, 1200]
zed_mode: HD1200
view: LEFT
image_geometry: rectified
K:
  - [746.9691162, 0.0, 959.9904175]
  - [0.0, 746.9691162, 585.4913330]
  - [0.0, 0.0, 1.0]
distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
```

这套参数与 ZED SDK 的 rectification/depth 几何一致，因此默认推荐下游使用它。

### `dexmate-calib` 自标定建议参数

如果应用明确要求使用本项目实测值，建议采用下面这套 5-session pooled 参数：

```yaml
camera_serial: 59595115
resolution: [1920, 1200]
zed_mode: HD1200
view: LEFT
image_geometry: rectified
source: dexmate-calib pooled joint solve of 5 accepted sessions, 193 views
K:
  - [747.6196, 0.0, 959.6954]
  - [0.0, 748.7749, 584.7266]
  - [0.0, 0.0, 1.0]
distortion_coefficients: [0.0, 0.0, 0.0, 0.0, 0.0]
```

它把 5 个独立、全部通过质量门的 session 中 198 个清晰候选视图放进同一次联合拟合，所有
图片共享一个 K，每张图片仍有独立的 board pose。稳健拟合剔除 5 张重投影离群图，最终使用
193 张。它不是 5 个 K 的平均值。

Pooled K 相对 factory 的差值为：

- `fx`: `+0.6505 px`（`+0.0871%`）
- `fy`: `+1.8058 px`（`+0.2418%`）
- `cx`: `-0.2950 px`
- `cy`: `-0.7648 px`

Factory rectified K 仍是与 ZED SDK rectification/depth pipeline 完全一致的默认生产基准；
上面的 pooled K 是当前最可信的 `dexmate-calib` 自标定建议值和独立交叉检查。

## 标定原理和边界

求解器在每张图中检测同一块 ChArUco 板的已知平面角点，把毫米定义的 board object points
与图像中的 2D corners 配对。不同图片各自的 `T_camera_board` 是求解过程中的未知中间量；
求解器利用多距离、多倾角、多画面位置的观测联合估计 `fx, fy, cx, cy`。因此内参标定不
需要预先知道相机相对机器人、头部或标定板的外参。

当前输入是 ZED SDK 已经 rectified 的 `LEFT`，所以本项目固定 distortion 为零，而不是再用
ChArUco 拟合一次 raw lens distortion。采集质量和稳健验证流程参考 RobotCamCalib，但代码在
本仓库中独立实现，并增加了 Dexmate 的 stream protocol、serial、HD1200 和 board profile
约束。

## 完整操作流程

### 1. 安装和环境检查

```bash
cd ~/Documents/Projects/dexmate/dexmate-calib
uv sync --extra dev
source .venv/bin/activate

which python
which dexcalib
dexcalib --help
```

`python` 和 `dexcalib` 应来自本仓库的 `.venv/bin/`。

### 2. 确认标定板

```bash
dexcalib board list
dexcalib board show dexmate-10x7
dexcalib board validate configs/boards/dexmate_charuco_10x7_27mm.yaml
```

有标定板正视照片时可额外检查：

```bash
dexcalib board verify \
  --board dexmate-10x7 \
  --image /path/to/board_photo.jpg
```

完整、清晰的正视照片应检测到 35 markers 和 54 ChArUco inner corners。正式采集前确认板
没有翘曲、强反光或污损，并保持刚性。板不必放在桌上，可以手持或固定；采集每张图时必须
停稳，且板的真实尺寸不能改变。

### 3. 只读连接检查

接好 Ethernet 后运行：

```bash
dexcalib doctor network
```

如果 streamer 已经运行，再检查实际图像流：

```bash
dexcalib doctor stream --frames 30
```

应确认 protocol、camera serial `59595115`、`1920×1200`、timestamp 和 JPEG decode 都正常。
完整的 Ethernet 静态地址、SSH key 和 direct/proxy 排查方法见
[quickstart.md](quickstart.md)。

### 4. 正式采集并自动求解

推荐命令：

```bash
dexcalib intrinsics quickstart \
  --board dexmate-10x7 \
  --samples 40 \
  --output calibration_data/head_left
```

`quickstart` 会验证连接，必要时通过 SSH 在 Nano 上启动固定的 HD1200 left-only streamer，
采集达到 40 个 accepted samples 后停止本次启动的 streamer，并默认求解新 session。

这条流程只读取相机，不调用 Dexcontrol，不移动任何机械臂 joint，也不转动头部。头部可以
处于任意固定姿态；内参理论上不随头部外部 pose 改变。采集期间不要移动头部，否则可能造成
运动模糊或让视角变化不可控。

自动模式以 10 Hz 判断候选帧，同时持续消费所有 TCP frame。符合角点数量、板内角点分布、
画面覆盖、清晰度、像素尺度、冷却时间和视角新颖性要求的帧才会保存。达到 `--samples` 后
自动结束；也可按 `q` 或 `Esc` 提前结束。

建议获得约 40 张图，且最终至少保留 25 张高质量视图。需要覆盖中心、四角和边缘，包含
近/中/远距离以及不同 roll/pitch/yaw；不要全部正对相机或只覆盖画面中心。

### 5. Smoke test、只采集和重新求解

仅验证生命周期，不立即求解：

```bash
dexcalib intrinsics quickstart \
  --no-solve \
  --manual \
  --board dexmate-10x7 \
  --samples 5
```

独立采集命令只采集，不自动求解：

```bash
dexcalib intrinsics capture \
  --board dexmate-10x7 \
  --samples 40 \
  --output calibration_data/head_left
```

已有 session 可以在完全断开机器人后离线求解：

```bash
dexcalib intrinsics solve \
  calibration_data/head_left/<session-name>
```

多个兼容 session 可以共同求解一个 K。当前建议结果的复现命令是：

```bash
dexcalib intrinsics solve-multi \
  calibration_data/head_left/20260816_004853_head_left_HD1200 \
  calibration_data/head_left/20260816_012412_head_left_HD1200 \
  calibration_data/head_left/20260816_044219_head_left_HD1200 \
  calibration_data/head_left/20260816_044425_head_left_HD1200 \
  calibration_data/head_left/20260816_044602_head_left_HD1200 \
  --output calibration_data/head_left/pooled_20260816_5_sessions_all_views
```

`solve-multi` 会先确认所有 session 的 serial、HD mode、分辨率、view、image geometry 和
board profile hash 完全一致。每个 session 独立做模糊筛选和姿态多样性选择，然后共同求解
一个 K。默认每个 session 最多使用 40 张，可用 `--max-views-per-session` 降低上限。

### 6. 结果检查

单 session 重点检查 `results/intrinsics_head_left_HD1200_1920x1200.json`；pooled 输出对应
`results/intrinsics_head_left_HD1200_1920x1200_pooled.json`。两者都要检查：

- `quality.all_gates_pass` 必须为 `true`
- 求解至少需要 20 个有效 views；`enough_views` 质量门要求 `views_used >= 25`
- 建议 `rms_reprojection_error_px < 0.5`
- 建议 `held_out_median_error_px < 0.5`
- 检查 `split_stability`，尤其 principal point difference

任何质量门失败，都应先补采、改善清晰度或视角分布，而不是直接降低阈值。还应检查：

- `sample_selection.csv`：每张图保留或拒绝的原因
- `capture_contact_sheet.jpg`：采集覆盖情况
- `reprojection_contact_sheet.jpg`：检测角点和模型重投影的可视比较
- `cross_validation.json`：K-fold held-out 误差和 K 稳定性
- `leave_one_session_out.json`：每次完整留出一个 session 的泛化误差和 K 稳定性

## 当前 `dexmate-calib` 实测记录

下表只保留 5 个全部通过质量门、且纳入当前统计的 session。失败的 session 和早期跨 session
离群结果没有进入表格或建议值，但原始数据仍保留在本机，没有删除。

### 5-session pooled joint solve

最终推荐使用 all-view pooled 结果：

| 指标 | 结果 |
|---|---:|
| 清晰候选视图 | 198 |
| 最终使用视图 | 193 |
| RMS reprojection error | 0.2856 px |
| K-fold held-out median / P90 / max | 0.2526 / 0.4012 / 0.4823 px |
| Leave-one-session-out median / P90 / max | 0.2567 / 0.4008 / 0.4836 px |
| Split principal-point difference | 0.3865 px |
| LOSO fx / fy range | 0.4865 / 0.1789 px |
| LOSO principal-point span | 0.7686 px |
| `all_gates_pass` | `true` |

各 session 对 pooled fit 的贡献为：

| Session | 清晰候选 | 最终使用 |
|---|---:|---:|
| `20260816_004853_head_left_HD1200` | 40 | 40 |
| `20260816_012412_head_left_HD1200` | 39 | 38 |
| `20260816_044219_head_left_HD1200` | 40 | 38 |
| `20260816_044425_head_left_HD1200` | 40 | 39 |
| `20260816_044602_head_left_HD1200` | 39 | 38 |
| **Total** | **198** | **193** |

输出位于：

```text
calibration_data/head_left/pooled_20260816_5_sessions_all_views/results/
```

额外做了每 session 最多 24 张的 balanced-120 敏感性检查。该版本最终使用 117 张，得到
`fx=747.5837, fy=748.6506, cx=959.7985, cy=584.6397`；与 all-view pooled K 的最大单参数
差值只有 `0.1243 px`，且全部质量门同样通过。这说明 pooled 结果对当前选样上限不敏感，
文档仍采用数据更多、验证误差略低的 all-view 结果。

### 各 session 独立求解记录

| Session | Views used | fx | fy | cx | cy | RMS (px) | Held-out median (px) | Held-out P90 (px) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `20260816_004853_head_left_HD1200` | 40 | 747.6033 | 748.9611 | 959.0820 | 584.3603 | 0.2437 | 0.2367 | 0.3072 |
| `20260816_012412_head_left_HD1200` | 39 | 746.4488 | 747.9804 | 959.2618 | 586.4348 | 0.3434 | 0.3266 | 0.4749 |
| `20260816_044219_head_left_HD1200` | 39 | 748.9972 | 749.0391 | 960.7433 | 585.2536 | 0.2768 | 0.2437 | 0.3497 |
| `20260816_044425_head_left_HD1200` | 39 | 747.4726 | 749.2063 | 960.0953 | 583.5743 | 0.2974 | 0.2610 | 0.4121 |
| `20260816_044602_head_left_HD1200` | 37 | 747.3040 | 747.1719 | 961.0813 | 585.2003 | 0.2572 | 0.2400 | 0.3344 |
| **5-session mean** | — | **747.5652** | **748.4718** | **960.0527** | **584.9647** | — | — | — |
| **sample std** | — | **0.9186** | **0.8706** | **0.8810** | **1.0722** | — | — | — |

所有表内 session 都使用同一 camera serial、HD1200、1920×1200、rectified `LEFT`、
`dexmate-10x7` board profile 和 zero-distortion 求解。表中的 5-session mean 仅保留用于比较，
不再作为建议参数；pooled K 相对这个旧均值的 `[fx, fy, cx, cy]` 差值为
`[+0.0544, +0.3032, -0.3573, -0.2381] px`。

## Factory 参数的获取来源

Factory 数值不是网上查到的通用型号参数，而是连接当前 Camera Nano 后，使用 Nano 上的
ZED SDK 5.1 对 serial `59595115`、HD1200 做的一次只读查询。查询字段为：

```cpp
camera.getCameraInformation()
      .camera_configuration
      .calibration_parameters
      .left_cam
```

这里的 `calibration_parameters` 是 rectified 参数，与 streamer 的 `sl::VIEW::LEFT` 匹配。
查询时 SDK 报告由于场景遮挡/纹理不足而跳过 self-calibration，因此记录的是 base factory
rectified 参数。若以后允许 ZED 在启动时成功执行 self-calibration，runtime K 可能有小幅变化；
此时应同时记录该次运行返回的 K，或在生产链路中明确固定 self-calibration 策略。

同一次查询也读取了 `calibration_parameters_raw.left_cam`，仅作为溯源记录：

```yaml
raw_K:
  - [734.8319702, 0.0, 967.3499756]
  - [0.0, 734.7780151, 564.9099731]
  - [0.0, 0.0, 1.0]
raw_distortion_in_zed_sdk_order:
  - 0.8607990146
  - 1.4117399450
  - 0.0003346460
  - -0.0000483978
  - 0.1578720063
  - 0.8658350110
  - 1.4373199940
  - 0.2560980022
  - 0.0
  - 0.0
  - 0.0
  - 0.0
```

这套 raw K/distortion 只能用于与它匹配的 unrectified 图像和正确的 ZED distortion 模型，
绝不能与当前 `VIEW::LEFT` JPEG 混用，也不能把 12 个 SDK 系数未经确认直接截成 OpenCV
5 参数模型。

## Resize 和 crop

1920×1200 是这里所有参数的 source of truth。若图像按 `sx`、`sy` resize：

```text
fx' = sx * fx    cx' = sx * cx
fy' = sy * fy    cy' = sy * cy
```

若随后从左上角 `(x0, y0)` crop：

```text
cx'' = cx' - x0
cy'' = cy' - y0
```

应在数据和结果 metadata 中显式记录这些变换。不要把当前参数直接贴到尺寸相同但经过不同
crop/rectification pipeline 的图像上。
