# Wuji 手套参考笔记

来源：https://docs.wuji.tech/docs/en/wuji-glove/latest/（introduction / getting-started / sdk-data-reference / coordinate-frames / time-sync / offline-pipeline / troubleshooting），SDK 文档 https://docs.wuji.tech/docs/en/wuji-sdk/latest/，标定 https://docs.wuji.tech/docs/en/wuji-studio/latest/calibration/。整理日期 2026-06-11。

## 1. 硬件与感知原理

**没有逐指弯曲传感器/逐指 IMU**，三套传感：

| 传感 | 规格 |
|---|---|
| 触觉矩阵 | 24×32 压力阵列（768 格，526 个有效点），覆盖整个手掌面，4mm 空间分辨率，0–20N @0.1N，120 FPS，输出归一化 0–1 |
| EMF 指尖跟踪 | 5 个 EMF 接收模块（每指尖一个），各 **6-DoF**，120 Hz，位置精度 ≤2mm RMS，姿态 <5° RMS；发射线圈在手背（`emf_tx` 系） |
| IMU | 手背 1× 六轴，**800 Hz**，±16g / ±2000°/s |

- 全手 IK 由 EMF 位姿解出 → **每手 21 DOF 关节角**（拇指 5：CMC×3 + MCP + IP；其余四指各 4）。
- **手腕没有绝对 6DoF**：`waist → l_wrist/r_wrist` 的动态 TF 由手背 IMU 驱动（仅姿态，800Hz）。手臂 teleop 的腕位置必须另外来源（VR/外骨骼/视觉）。
- **无电池、有线**：USB Type-C 5V/1A 供电 + **RJ45 以太网 100BASE-TX**（经 USB-C 转接器）。延迟 ≤10ms。手套 67.5g（带线 192.6g）。

## 2. 连接 bring-up

1. 采集板绑在小臂，接 USB-C →（以太网+供电）转接器；供电必须用 **USB-A→USB-C 线**（C-to-C 不供电），电源 LED 进入呼吸状态即正常。
2. 主机网卡设为 192.168.1.x/24；默认 **左手 192.168.1.100，右手 192.168.1.101**；`ping` 验证；UDP 端口 50000/50001（ufw 放行）。
3. 无配对过程（无 BLE/WiFi/dongle）。佩戴时手指微屈，指尖 EMF 模块不要交叉。
4. SN 第 4 位区分左右：J=左，K=右。

## 3. SDK（Python only）

`pip install wuji-sdk`；要求 **Ubuntu 22+，Python 3.10+**，与设备同网段。实测版本 **v2026.6.18**。

> ⚠ **真机实测（2026-06-22）**：`Subscription.recv()` 是**同步非阻塞**——无新帧返回 `None`，循环必须判空（否则 busy-spin 或 `None` 崩）；`SkeletonJoint.pose.position` 是 `[x,y,z]` **list**（不是 `.x/.y/.z` 对象），`orientation` 是 `Quaternion`（`.w/.x/.y/.z`）。已并入 `magicdexmate/sources/wuji_source.py`。

```python
from wuji_sdk import SdkManager, Handedness
manager = SdkManager.instance()
devices = manager.scan()                                   # DiscoveredDevice: .sn, .address
glove = manager.auto_connect(device_name="glove")
# 或 manager.connect(sn=...) / connect(address="192.168.1.100:50000") / connect(handedness=Handedness.Right)

sub = glove.hand_skeleton().subscribe()                    # 也有 subscribe_with_callback()
frame = await sub.recv_async()
sub.close()
```

- 录制：`TopicRecorder` → **MCAP 文件（LZ4）**，可暂停/恢复。
- 设备参数：`glove.sn() / version() / ip().set("…")（写 flash）/ reboot()`；日志 `~/.wuji/logs/`。
- 离线管线：`WujiGlove.offline_pipeline(sn=…, hand_side="right", urdf_path=None)` 可由录制的 emf_poses/IMU 重算 `hand_joint_angles / tip_poses / hand_skeleton`。
- 时间同步：连接后每 30s 自动（类 NTP 四时间戳），精度≈网络 RTT（直连数百 µs）；`sync_time()` 手动。帧头 `FrameHeader`：序号 + **UTC 时间戳（µs）** + frame_id。

## 4. 数据流（与 retarget 相关的重点加粗）

| 流 | API | 频率 | 内容 |
|---|---|---|---|
| **手骨架** | `glove.hand_skeleton()` | 120 | **21 个 MediaPipe 关键点**：每点 `name`（如 `"index_finger_mcp"`）、`pose`（position+quaternion）、`confidence`；**表达在 l_wrist/r_wrist 系** |
| **关节角** | `glove.hand_joint_angles()` | 120 | 5× `FingerJointAngles`，每指 `angles` 5 元素数组（**rad**）+ `confidence`；拇指 5 个有效，其余 4 有效+1 补零，共 21 DOF |
| 指尖位姿 | `glove.tip_poses()` | 120 | 5 指 6-DoF |
| EMF 原始 | `glove.emf_poses()` | 120 | 5 指尖 6-DoF（`emf_tx` 系） |
| IMU | `glove.imu_palm()` | **800** | ROS 风格：quaternion、angular_velocity、linear_acceleration + 协方差 |
| 触觉矩阵 | `glove.tactile()` | 120 | 768 元素（24×32 行主序），0–1，无效=-1 |
| 触觉分区 | `glove.tactile_zones()` | 120 | palm 290 / thumb 52 / index 52 / middle 58 / ring 52 / pinky 45 |
| 触觉二值 | `glove.tactile_binary()` | 120 | 需训练过的接触模型 |
| TF | `manager.tf_static()/tf()` | 1 / 800 | waist→wrist 动态 TF 来自 IMU |

## 5. 坐标系

- 遵循 REP-103/120/155，右手系。树：`world → waist → {l_wrist, r_wrist} → {*_hand_emf_tx, *_palm_imu_link}`。
- **腕系：X = 桡侧（拇指方向），Z = 近端（朝肘），Y = 右手定则**（左右手镜像）。⚠ 与 dex-retargeting 期望的 MANO 约定不同，需一个固定旋转（见 plans/01_retarget_plan.md §R2）。

## 6. 标定（Wuji Studio 桌面程序，非 SDK）

安装（Ubuntu 22+）：从 GitHub Releases 下载 deb（**已下载到 `~/Downloads/wuji-studio_2026.6.2_amd64.deb`**，41MB）：

```bash
sudo apt install ~/Downloads/wuji-studio_2026.6.2_amd64.deb   # 然后终端运行 wuji-studio
```

下载源：https://github.com/wuji-technology/wuji-studio/releases （版本与 wuji-sdk 同步，当前 v2026.6.2）。功能：设备管理、数据可视化、固件升级、手部标定、触觉接触标定。

- 整手标定：6 个姿势（拇指尖依次碰食/中/无名/小指尖——"只碰指尖勿压指腹"、四指弯 90° 半握拳、平摊手）；**生成每用户/每设备的手部 URDF**，自动保存。摔碰、闲置 3 个月、数据异常或提示 "Calibration Expired" 时重标。
- 触觉接触标定另做：录 4 段无接触动作 → 训练基线模型 → 按压验证。

## 8.5 腕部 rpy 通道：真机标定步骤（teleop_vega_sharpa 用）

仿真侧 rpy 管线已验证（2026-06-12，`--motion rpytest`：roll/pitch/yaw 轴向 1:1、峰值角误差 <2°、位置误差 1-14mm）。真手套接入时还差**一个常量**：手套腕系→机器人 EE 系的固定旋转 `--wrist-align`（mock 下为 identity）。标定流程（约 5 分钟）：

1. 跑 `teleop_vega_sharpa.py --source wuji --hand right --debug-joints --wrist-max-deg 30`（小钳位保安全）。
2. 操作者依次做三段**纯单轴**腕动作（各 ~5s）：前臂不动只翻腕（解剖学 pronation/supination ≈ 手套 Z 轴）、屈伸（flexion/extension）、桡尺偏（radial/ulnar deviation）。
3. 读 `[dbg]` 里的 `cmd axis`（手套系）与 `EE rot axis`（机器人系）——三对轴向量直接拼出旋转矩阵 R（按列对应），换算成 euler xyz 度数填进 `--wrist-align`。
4. 复跑第 2 步验证：三段动作的 EE 轴应分别≈机器人系的期望轴。
5. **yaw 漂移检查**：静置手套 2 分钟看 `cmd` 角度是否漂移（IMU 无磁力计校正时 yaw 会缓漂）；漂移明显则需加 re-zero 热键（按一下重置 `glove_quat0_inv`）——已留 TODO。

已知特性：相位急转时有 ~0.3-0.5s 跟踪滞后（60Hz 单步 DLS 带宽），人手快速甩腕会被平滑；需要更高带宽时提高 `--control_hz` 或在循环内多迭代 IK。启动后第 1 秒为 engage 瞬态，属正常。

## 7. 官方 retarget 集成（重要）

https://github.com/wuji-technology/wuji-retargeting（MIT）：**"基于 DexRetargeting 算法的高精度重定向系统"**，输入支持 **Wuji Glove 实时**（`python teleop_sim.py --input wuji_glove --hand right --glove-sn <SN>`）、Apple Vision Pro、MediaPipe、RealSense/ZED，支持 pkl 回放，输出控制 Wuji Hand。→ 我们接 Sharpa 时可直接抄它的"手套输入胶水层"。
其他相关 org 仓库：wuji-hand-teleop（ROS2）、wujihandros2（1000Hz 驱动）、wuji-description、wuji-mjlab、wuji-openpi。

## 8. 对本项目的含义

- `hand_skeleton` 已是 MediaPipe 21 点格式 ⇒ 与 dex-retargeting 的 `target_link_human_indices` 直接对应（0=wrist，4/8/12/16/20=五个指尖）。
- 用 `name` 字段建索引映射，**不要假设数组顺序**就是 MediaPipe 编号。
- 120Hz 输入频率高于真机控制需求（20–60Hz），retarget 不需要插帧。
- MCAP 录制 + 离线管线 ⇒ 天然支持"先录后调"的离线回放开发流程。
