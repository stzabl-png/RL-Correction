# Dexmate Vega 参考笔记

> ⚠ **docs.dexmate.ai 被登录墙挡住**（WebFetch 403 / curl 401 "Authentication missing"，需客户账号）。本文全部来自公开 GitHub org（github.com/dexmate-ai）与产品页（dexmate.ai/product/vega）。拿到 docs 账号后需要补：Pico/VR 官方教程、导航 API、上电/网络配置流程。整理日期 2026-06-11。

## 1. 机器人概要

- 移动双臂人形：**全向轮底盘**（URDF：2 个轮模块，各 转向 revolute `*_wheel_j1` + 驱动 continuous `*_wheel_j2`）+ **3-DoF 躯干**（`torso_j1..j3`，可折叠）+ **2× 7-DoF 手臂**（`L/R_arm_j1..j7`）+ **3-DoF 头**（pan-tilt-roll）。
- 原装手：F5D6 五指手（6 主动 DoF）；也有 gripper 变体；**腕部挂载点 `L/R_hand_mount` + `*_ee_j0` 固定关节**，dexcontrol 有 `control_customized_ee.py`、`config_ee_baud_rate.py` → 串行总线 EE 接口，支持自定义末端（装 Sharpa 时关注：机械法兰 + 供电/网线走线，Sharpa 是以太网设备，不走 EE 串行总线）。
- 载荷 4.5 kg/臂；高 ~171cm；电池 10h+；计算 **NVIDIA AGX Thor**；I/O：4× USB3.2、1 Ethernet、1 DP。
- 传感：头部 ZED X Mini 双目、腕部 ZED X One、底盘相机、**2D RPLIDAR + 3D lidar（Livox MID360 / RoboSense Airy/E1R）**、底盘+头部 IMU、超声×4、**腕部 6 轴力/力矩**、手部触摸传感。
- 型号：`vega_1`（全移动）/ `vega_1p` / `vega_1u`（上半身双臂）。

## 2. SDK：dexcontrol

`pip install dexcontrol`（extras `[example]`）；依赖 **dexcomm（Zenoh 通信层）**；版本耦合 dexcontrol≥0.4.0 ↔ dexcomm≥0.4.0 ↔ 固件≥0.4.0。AGPL-3.0+商业双许可。
连接走 **Zenoh over network**；环境变量：`ROBOT_NAME`、`ZENOH_CONFIG`、`DEXCONTROL_DISABLE_HEARTBEAT`、`DEXCONTROL_DISABLE_ESTOP_CHECKING`。

```python
from dexcontrol.robot import Robot
with Robot() as bot:   # left_arm right_arm left_hand right_hand head chassis torso battery estop ...
    bot.left_arm.set_joint_pos(np.zeros(7), wait_time=4.0)   # 插值轨迹模式
    bot.set_joint_pos({"left_arm": [0]*7})                    # 多部件
    bot.left_hand.open_hand(); bot.left_hand.get_joint_pos()
    bot.torso.set_joint_pos_vel(joint_pos=[j1, j2, j3])
```

- **流式控制**：`set_joint_pos(..., wait_time=0)` 需高频循环调用（**~100 Hz，上限 500 Hz**）；轨迹模式默认 `control_hz=100`。
- 手臂附加：`set_mode("position"|"disable")`、PID 配置、刹车释放、**导纳控制**（关节+笛卡尔示例）、F/T 模式；笛卡尔：`examples/advanced/move_arm_to_pose.py`。
- examples 清单：basic（move_arm/chassis/head/torso、open/close_hand）、advanced（admittance、fold_arms、estop、replay_trajectory、keyboard_joint_control）、sensors（lidar 2D/3D、ZED、IMU、超声、F/T、touch）、teleop（**arm_cartesian_teleop / arm_joints_teleop / base_arm_teleop / chassis_head_teleop**）、benchmark（跟踪、延迟）、troubleshooting（check_time_difference、clear_error）。内置 `apps/dualsense_teleop_base.py`（PS5 手柄底盘）。

## 3. 官方 teleop 栈：omniteleop

github.com/dexmate-ai/omniteleop（AGPL/商业）。Leader→Follower 架构，基于 dexcomm/Zenoh pub-sub：命令处理器 + 安全校验（estop、限位）+ 各部件 processor（arm/hand/head/torso/chassis）+ **Ruckig 轨迹平滑** + Web GUI（`app/launch.sh`）+ **MDP recorder（采数据训练用）** + 回放 + 遥测。Python 3.10–3.13。

CLI：`omni-arm`（**Dynamixel 外骨骼** leader）、`omni-joycon`、`omni-paddle`（**VR reader**）、`omni-cmd`、`omni-robot`、`omni-recorder`、`omni-telemetry`。

**VR teleop**：`leader/vr/` —— VRReader 发布 `vr/controllers` 话题 @ **40 Hz**，intervention 式（扣 trigger 才生效）；求解器 cartesian/joint；手柄分工：左 grip=底盘、右 grip=躯干、无 grip=手；优先级 exit > estop > base > torso > hands。**代码明确写的是 Meta Quest（"Quest-specific emergency controls"），未见 PICO 字样**——Pico 兼容性待向 Dexmate 确认（备选：第三方 qrafty-ai/teleop_xr 声称支持 WebXR 头显 + Vega）。
机器人配置覆盖 `vega_1/vega_1p/vega_1u` × `f5d6/gripper/plain`。

## 4. SLAM / 导航

| 仓库 | 内容 |
|---|---|
| **super-odom** | LiDAR-SLAM（SuperOdom 改）+ **TEASER++ 地图重定位**，Vega-1+ 自主栈；3D LiDAR 10Hz + IMU 200Hz；**ROS 2 Jazzy**，GTSAM/Ceres≥2.1；`ros2 launch super_odometry rs_airy.launch.py`；重定位：先 ~5m SLAM 自举再自动配准全局图 |
| **cartographer_ros_vega** | Google Cartographer（ROS2）按 Vega-1 2D RPLIDAR 调参；输入 `/scan_pcd`(PointCloud2)+可选 `/odom`；`vega_2d.launch.py`（在线建图）/ `offline_vega_2d.launch.py`（bag→.pbstream）/ `vega_2d_localization.launch.py`（纯定位） |
| **dexcontrol-rosbridge** | **Zenoh → ROS2（Humble/Jazzy）转发**：头部 RGB/深度、IMU、lidar 点云、轮式里程计（`/odom`）；`python scripts/republish_sensors.py --sensors head|lidar`、`publish_wheel_odometry.py`。是上面两个 SLAM 的数据入口 |

## 5. 仿真资产

- `pip install dexmate_urdf`（Apache-2.0）：`from dexmate_urdf import robots; robots.humanoid.vega_1.vega_1_f5d6.urdf / .srdf`；变体 `vega_1|vega_1p|vega_1u` × `{plain,_f5d6,_gripper}` + 碰撞球 URDF + 独立手 `f5d6_left/right.urdf`、`dexd/dexs_gripper`。
- **USD 在 GitHub releases** 里（不在 pip 包内）。
- 其他 org 仓库：dexstream（数据流）、dexdata、dexmotion-examples、dexbot-utils、vega-firmware、pyrplidarsdk。

## 6. 对本项目的含义

- 手臂 teleop 腕位姿来源（Wuji 手套没有腕位置）：omniteleop VR（40Hz）或外骨骼 omni-arm；手指由我们的 Wuji→Sharpa retarget 流接管。
- Sharpa 替换 F5D6 时：仿真用 `vega_1_plain` + Sharpa `with_flange` 资产组合；真机要解决机械适配与独立以太网/供电走线。
- SLAM 栈就绪度高（rosbridge→cartographer/super-odom），属于独立工作流，依赖 ROS2 Jazzy 环境。
