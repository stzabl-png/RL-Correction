# 头部 ZED 内参标定：网络、采集与求解细节

最后更新：2026-08-18。本文是 `dexcalib intrinsics` 的操作细节（从 README 移出），参数记录与
判定标准见 [head_left_intrinsics.md](head_left_intrinsics.md)，首次网络/公钥配置见
[quickstart.md](quickstart.md)。

## 网络与取流路径

SSH 只负责在 Nano 上启动 streamer；图像数据是 Mac 直接连接 Nano：

```text
Mac -- SSH（可经 .20 ProxyCommand） --> 192.168.50.22
Mac -- direct TCP ------------------> 192.168.50.22:30000
```

`.20` 若用于 `ssh -W`，只转发 SSH 的 TCP 字节，不转发相机数据。

连接机器人后先做只读检查：

```bash
dexcalib doctor network
```

离线时该命令失败是正常的，不影响离线测试和求解。

### 一站式启动的行为边界

```bash
dexcalib intrinsics quickstart --board dexmate-10x7 --samples 40
```

它按顺序执行：

1. 用 `BatchMode=yes` 测试直连 `.22`，失败后才 fallback 到经 `.20` 的 ProxyCommand。
2. 若 `.22:30000` 尚未监听，在保持的 SSH PTY 中启动固定的 HD1200 left-only 命令。
3. 强制验证 ZS protocol、serial `59595115`、1920×1200、timestamp 和 JPEG decode。
4. 进入 ChArUco 采集。
5. 仅当 streamer 是本次 quickstart 启动的，采集结束或异常后才关闭它。
6. 默认在本机求解刚创建的 session；`--no-solve` 可停在采集结束。

若端口开始时已经有 streamer，quickstart 会验证并复用，但不会在结束时停止别人的进程。
如果 Nano 已配置当前用户直接访问相机，可以加 `--no-sudo`。默认不使用 `--clean`，不会
自动杀其他 ZED 程序。

## 在 Nano 上启动内参采集 streamer

这是 quickstart 失败时的手动 fallback。先确认没有其他任务需要 ZED，然后登录 Nano：

```bash
ssh dexmate-nano
cd ~/zed_stream
sudo ./build/zed_streamer \
  --jpeg-quality 100 \
  --max-fps 30 \
  --resolution HD1200 \
  --no-right \
  --no-depth \
  --no-pc \
  --no-imu
```

这里关闭 depth，因为内参只需要左目 JPEG。不要用 HD1080；采集端会拒绝任何不是
1920×1200 的 stream。如果确实需要 `--clean`，必须先确认其他 ZED 任务可以被终止，再由
操作者显式添加；quickstart 永远不会自动添加它。

另一个本机终端验证实际数据：

```bash
dexcalib doctor stream --frames 30
```

它会检查协议、serial、分辨率、timestamp、接收 FPS 和 JPEG decode。ZS01 没有
serial 字段；仅在测试旧 server 时可用 `--expected-serial 0` 关闭 serial gate。

## 采集

```bash
dexcalib intrinsics capture \
  --board dexmate-10x7 \
  --samples 40 \
  --output calibration_data/head_left
```

独立的 `intrinsics capture` 只采集，不自动求解；一站式 `intrinsics quickstart` 默认在
采集后求解。

默认自动采集，按 `q` 或 `Esc` 提前结束。使用 `--manual` 后按空格保存符合质量门限的
画面。每个 session 锁定：

- camera serial、ZS protocol、resolution、channel mask
- `HD1200`, `LEFT`, `rectified`
- 完整 board profile snapshot 及 SHA-256
- 原始 server JPEG（不二次编码）和 source/receive timestamps

建议采集 40 张左右，覆盖画面中心、四角和边缘，并改变距离、倾角和旋转。不要只把板
平行地放在画面中央。

默认质量门限可通过命令行调整：

```bash
dexcalib intrinsics capture --help
```

实时窗口会绘制 ArUco marker 和 ChArUco corner，并显示 corner/grid 数量、画面覆盖、
board bbox、清晰度、pixels-per-square、冷却和重复视角状态。质量筛选除角点数量外还会
检查角点在板内的行列分布，避免只检测到集中在一小块区域的角点。`--manual` 依赖预览
窗口接收空格键，因此不能与 `--no-preview` 同时使用。

自动模式默认以 10 Hz 处理 ChArUco（`--detect-fps 10`），但仍持续读取所有 TCP frame，
不会因为降频而让 socket 数据积压。0.8 秒保存 cooldown 内使用轻量质量路径；只有可能
成为候选的帧才执行 homography 展平和 Laplacian/Tenengrad 详细分析。手动模式保持逐帧
检测。调试时可用 `--detect-fps 0` 恢复逐 stream frame 检测。

## 求解

`intrinsics quickstart` 默认已自动求解。使用 `--no-solve`、独立 `intrinsics capture`
或需要用其他求解参数重新计算时，可在断开机器人后运行：

```bash
dexcalib intrinsics solve \
  calibration_data/head_left/20260814_230000_head_left_HD1200
```

求解器会重新检测所有原始 JPEG，固定所有 distortion coefficients 为零，只估计
`fx, fy, cx, cy` 和每个 board pose，并执行：

- 标定板 homography 展平后的尺度感知清晰度检查
- 确定性的姿态多样性选择（样本超过 `--max-views` 时）
- 每视图 reprojection error
- robust outlier rejection
- deterministic K-fold held-out validation
- even/odd split 参数稳定性检查

结果写入 session 的 `results/`：

```text
intrinsics_head_left_HD1200_1920x1200.yaml
intrinsics_head_left_HD1200_1920x1200.json
K.npy
reprojection_errors.csv
sample_selection.csv
cross_validation.json
quality_summary.json
capture_contact_sheet.jpg
reprojection_contact_sheet.jpg
```

绿色点是检测角点，紫色点是最终模型的重投影位置，黄色短线显示两者之间的残差。
`sample_selection.csv` 会为每张图片记录最终保留或拒绝原因，包括 motion blur、pose
redundancy 和 reprojection outlier。

1920×1200 是 source of truth。任何下游 resize/crop 必须显式变换 `K`；不能直接把
HD1200 内参用于 HD1080，也不能把 1920×1200 非等比例拉伸到 640×360 后仍沿用原 K。
