# Dexmate 头部相机内参：首次配置与 Quick Start

本文对应固定生产模式：ZED X Mini 左目、HD1200 1920×1200、rectified `LEFT`、camera
serial `59595115`。

## 1. 使用前必须满足的条件

### 机器人和网络

- Dexmate 已上电并完成启动，通常等待 2–3 分钟。
- Mac 和机器人 torso/internal switch 位于同一个物理 Ethernet switch。
- Mac Ethernet 使用静态 `192.168.50.x/24`；推荐本机使用 `192.168.50.10`。
- subnet mask 是 `255.255.255.0`，无需 gateway；不要把 subnet mask 错填到 DNS。
- 本机地址不能与 `.20` Thor、`.21` controller、`.22` Nano、`.42/.43` lidar 或另一台
  工作站冲突。
- switch 上可以同时有另一台电脑，但每台电脑必须使用不同 IP。相机 server 支持多 TCP
  client；真正会冲突的是 Nano 上另一个占用 ZED 相机的进程。

可在 macOS System Settings → Network → 对应 USB/Ethernet adapter → Details → TCP/IP
中设置静态地址。接口名称因转接器而异，不在脚本中自动改网络配置。

基础检查：

```bash
ping -c 2 192.168.50.20
ping -c 2 192.168.50.22
```

### 标定板

默认 profile `dexmate-10x7` 必须对应以下实物，不能用“看起来类似”的板代替：

- ChArUco 10×7 complete squares
- square 27.000 mm
- marker 20.250 mm
- `DICT_5X5_50`
- 外框 270×189 mm
- 35 markers，ID 0–34
- 54 ChArUco inner corners
- 100 mm check 已经实测正确
- 板已经贴在平整、刚性的背板上

采集前检查板没有翘曲、反光遮住 marker、污损或二次缩放打印。使用新板时必须新建 YAML
profile 并显式 `--board` 选择，不能修改默认板参数来凑检测结果。

### 场景

- 光照充足且均匀，避免强反光和过曝。
- 相机和板在采集中保持机械稳定；移动板时停稳后再保存。
- 有空间让板覆盖中心、四角、边缘，并改变距离、roll/pitch/yaw。
- 采集期间不要运行需要独占同一 ZED 的其他程序。

## 2. 配置 Mac SSH 公钥

SSH key 只用于登录；图像数据仍是 Mac 直接连接 `192.168.50.22:30000`。

### 2.1 创建或确认本机 key

```bash
test -f ~/.ssh/id_ed25519.pub || \
  ssh-keygen -t ed25519 -a 64 -f ~/.ssh/id_ed25519
```

不要把 `~/.ssh/id_ed25519` 私钥复制到任何其他机器。可以传输的只有
`~/.ssh/id_ed25519.pub`。

### 2.2 首次手动登录并接受 host key

```bash
ssh dexmate@192.168.50.20
ssh dexmate-nano@192.168.50.22
```

首次连接确认主机指纹后退出。这里使用机器人管理员提供的登录凭据；仓库不记录密码。

### 2.3 安装 Mac 公钥

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub dexmate@192.168.50.20
ssh-copy-id -i ~/.ssh/id_ed25519.pub dexmate-nano@192.168.50.22
```

如果 macOS 环境没有 `ssh-copy-id`，可用以下等价方式，只追加 `.pub` 公钥：

```bash
cat ~/.ssh/id_ed25519.pub | ssh dexmate-nano@192.168.50.22 '
  umask 077
  mkdir -p ~/.ssh
  cat >> ~/.ssh/authorized_keys
  chmod 700 ~/.ssh
  chmod 600 ~/.ssh/authorized_keys
'
```

`.20` 上的 key 仅用于第一跳 SSH；`.22` 仍然直接认证 Mac 的公钥。ProxyCommand 中 `.20`
只是 TCP 字节转发器，`.20` 不需要拿自己的 key 去登录 `.22`。

如果当时只能经 `.20` 到 `.22`：

```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub \
  -o 'ProxyCommand=ssh -W %h:%p dexmate@192.168.50.20' \
  dexmate-nano@192.168.50.22
```

### 2.4 验证完全免 SSH 密码

```bash
ssh -o BatchMode=yes dexmate@192.168.50.20 hostname
ssh -o BatchMode=yes dexmate-nano@192.168.50.22 hostname

ssh -o BatchMode=yes \
  -o 'ProxyCommand=ssh -W %h:%p dexmate@192.168.50.20' \
  dexmate-nano@192.168.50.22 hostname
```

至少 Nano 直连或 ProxyCommand 路径之一必须成功。由于相机数据不走 SSH tunnel，Mac 仍
必须能直接访问 `.22:30000`。

可选的 `~/.ssh/config`：

```sshconfig
Host dexmate
    HostName 192.168.50.20
    User dexmate
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes

Host dexmate-nano
    HostName 192.168.50.22
    User dexmate-nano
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
    ProxyCommand ssh -W %h:%p dexmate

Host dexmate-nano-direct
    HostName 192.168.50.22
    User dexmate-nano
    IdentityFile ~/.ssh/id_ed25519
    IdentitiesOnly yes
```

quickstart 不依赖这些 alias；它使用明确的 user/IP，并自动先尝试 direct、再尝试 proxy。

## 3. 本机安装

```bash
cd ~/Documents/Projects/dexmate/dexmate-calib
uv sync --extra dev
source .venv/bin/activate

dexcalib board validate configs/boards/dexmate_charuco_10x7_27mm.yaml
dexcalib doctor network
```

`doctor network` 在 streamer 尚未启动时可以仅有 `.22:30000` 检查失败；SSH 项应成功。

如果有一张当前实物板照片：

```bash
dexcalib board verify --board dexmate-10x7 --image /path/to/board_photo.jpg
```

完整正视照片应达到 35 markers、54 corners；采集中的倾斜视图不要求每帧全部可见。

## 4. 一条命令启动、验证、采集、停止和求解

```bash
dexcalib intrinsics quickstart \
  --board dexmate-10x7 \
  --samples 40 \
  --output calibration_data/head_left
```

默认远端命令等价于：

```bash
sudo /home/dexmate-nano/zed_stream/build/zed_streamer \
  --jpeg-quality 100 \
  --max-fps 30 \
  --resolution HD1200 \
  --no-right \
  --no-depth \
  --no-pc \
  --no-imu
```

它不包含 `--clean`。SSH 登录使用 key；`sudo` 如果需要，会直接在当前终端的远端 TTY
提示密码。Python 不读取、不记录密码。

quickstart 保持该 SSH session 和远端前台 streamer 直到采集结束。按 `q`/`Esc`、正常采满
或采集抛出异常时，`finally` 都会关闭本次 SSH session，从而停止本次 streamer。它不会
停止启动前就已经监听 30000 的现有 streamer。streamer 清理完成后，quickstart 默认在
本机求解刚创建的 session，并把结果写入该 session 的 `results/`。

仅采集、不自动求解：

```bash
dexcalib intrinsics quickstart \
  --no-solve \
  --board dexmate-10x7 \
  --samples 40
```

少量生命周期 smoke test 通常不足默认的 20-view 求解门限，必须使用 `--no-solve`，例如：

```bash
dexcalib intrinsics quickstart \
  --no-solve \
  --manual \
  --board dexmate-10x7 \
  --samples 5
```

如果 `dexmate-nano` 用户本身有相机访问权限：

```bash
dexcalib intrinsics quickstart --no-sudo --board dexmate-10x7 --samples 40
```

如需固定 SSH 路径：

```bash
dexcalib intrinsics quickstart --ssh-route direct
dexcalib intrinsics quickstart --ssh-route proxy
```

## 5. Quickstart 会拒绝的情况

- SSH key 未配置或 host key 尚未接受。
- Mac 不能直接访问 `.22:30000`。
- streamer binary 不存在或不可执行。
- sudo 认证失败。
- 另一个进程占用 ZED，导致新 streamer 提前退出。
- 实际 stream 不是 1920×1200。
- camera serial 不是 `59595115`。
- frame 缺少 left JPEG、协议损坏或 timestamp 异常。

quickstart 不会通过 `--clean` 自动解决相机占用，因为那可能终止别人的实验。确认安全后，
可手动 SSH 到 Nano 排查并启动；见 [intrinsics.md](intrinsics.md) 的 manual fallback。

## 6. 采集动作

- 先让板较大地出现在中心，确认检测稳定。
- 覆盖左上、上、右上、左、右、左下、下、右下和中心。
- 包含近、中、远距离。
- 包含绕三个方向的倾角，不要全部正对相机。
- 每次移动后停稳，避免 motion blur。
- 默认自动保存质量合格且与已有样本足够不同的视图。
- 实时 HUD 显示 markers/corners、板内行列覆盖、画面 coverage、清晰度和板的像素尺度。
- 绿色 marker/corner overlay 表示当前检测结果；状态行会说明 cooldown、重复视角或具体
  质量拒绝原因。
- 自动模式默认 10 Hz 做 ChArUco 判断，同时持续消费全部 stream frame；cooldown 内跳过
  homography/Laplacian/Tenengrad 重计算。若要逐帧检测可显式加 `--detect-fps 0`。

目标约 40 张；有效结果建议不少于 25 张。原始 JPEG 和 partial manifest 会实时落盘。

## 7. 自动与离线求解

正式 quickstart 采满后会自动执行下面的求解。若此前使用 `--no-solve`、独立 capture，或
需要重新求解，机器人可以断开后再运行：

```bash
dexcalib intrinsics solve calibration_data/head_left/<session-name>
```

需要把多个独立 session 联合求解为一个共同 K 时使用：

```bash
dexcalib intrinsics solve-multi \
  calibration_data/head_left/<session-1> \
  calibration_data/head_left/<session-2> \
  --output calibration_data/head_left/<pooled-session-name>
```

该命令会拒绝 camera serial、分辨率、图像几何或 board hash 不一致的输入，并额外输出
`leave_one_session_out.json`。完整的当前 5-session 命令和结果见
[head_left_intrinsics.md](head_left_intrinsics.md)。

检查 `results/` 中的 `all_gates_pass`、RMS、held-out error 和 split stability。任何 gate 失败
都应优先补采视图或检查板/成像，而不是直接降低阈值。

同时检查：

- `sample_selection.csv`：每张图的保留/拒绝原因和质量指标。
- `capture_contact_sheet.jpg`：采集视图覆盖概览。
- `reprojection_contact_sheet.jpg`：检测角点与重投影位置的可视比较。
- `cross_validation.json`：确定性 K-fold held-out 误差和各 fold 的 K 稳定性。

当前 ZED SDK factory rectified 参数、筛选后的多 session 实测表、自标定建议参数以及参数的
适用边界见 [head_left_intrinsics.md](head_left_intrinsics.md)。
