# Dexmate-teleop

本仓库实现 Dexmate Vega-1 人形机器人的全身遥操作。操作者佩戴 PICO 头显与手腕追踪器,控制机器人的双臂、头部与腰部;佩戴 Wuji 数据手套,控制机器人所装的 Sharpa 灵巧手。动作先在 Isaac Lab 仿真中执行并显示,确认无误后再驱动真机;各路数据可以录制到统一目录,用于后续训练。

## 快速开始

```bash
# 部署(装依赖、放厂商件、配网络):见 SETUP.md
# 启动(唯一入口,零参数):
.venv/bin/python scripts/vega_console.py
# 打开浏览器 http://localhost:8086,勾选要控制的部位,点「启动」
```

操作步骤、判断方法与故障处理见两份手册:[SOP_wholebody_teleop.md](SOP_wholebody_teleop.md)(手臂/头/腰)与 [SOP.md](SOP.md)(手)。不接任何硬件时,可运行 `.venv/bin/python scripts/check_all.py` 做 15 项自动检查,验证软件安装是否完整。

## 系统结构

两条互相独立的管线,进程之间用 ZMQ 通信:

```
手臂/头/腰:
  PICO 头显 + 手腕追踪器
    → scripts/teleop_pico_producer.py     读取 PICO 数据并发布(端口 5581)
    → sim/teleop_vega_pico.py             映射 + 逆运动学 + Isaac Lab 仿真,发布状态(端口 5583)
    → scripts/dexmate_bridge.py           驱动真机(dexcontrol SDK,独立进程)

手:
  Wuji 数据手套
    → scripts/teleop_retarget.py          手指重定向并发布(右手端口 5556,左手 5557)
    → sim/test_env_sharpa.py              仿真中跟随
    → scripts/sharpa_real_runner.py       驱动真手并录制

页面:
  scripts/vega_console.py                 启动与停止上述全部进程,浏览器中显示机器人、
                                          操作者骨架、设备状态,并提供真机连接与录制按钮
```

## 从 PICO 到 Dexmate:映射相关代码

按数据流顺序列出。要理解或修改"人的动作如何变成机器人动作",读这些文件。

### 1. 数据采集与传输

| 文件 | 职责 |
|---|---|
| `external/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/` | PICO 官方 SDK 的 Python 绑定(含本项目的修改),从头显读取数据 |
| `scripts/teleop_pico_producer.py` | 读取头显位姿、24 个人体关节与追踪器数据,以固定频率发布 |
| `magicdexmate/pico/xr_client.py` | SDK 的封装:取位姿、时间戳、人体关节的统一接口 |
| `magicdexmate/pico/xr_pose.py` | PICO 坐标系到机器人坐标系的变换;One Euro 滤波器实现 |
| `magicdexmate/sources/pico_source.py` | 消费端的输入源,三种实现同一接口:实时接收、回放录制文件、合成数据 |
| `magicdexmate/pico/arm_fusion.py` | 追踪器位置短暂丢失时,用仍然有效的朝向数据补全位置 |

### 2. 映射(人体位姿到机器人末端目标)

| 文件 | 职责 |
|---|---|
| `sim/teleop_vega_pico.py` | 主程序,映射规则在此。核心做法:以操作者躯干为参考系计算两手腕的相对位姿,乘位置比例后作为机器人末端的位置目标;掌心朝向按绝对方式映射为末端朝向目标(机器人掌心与操作者掌心指向一致)。随后调用逆运动学并把关节角发给仿真 |
| `magicdexmate/palm_fix.py` | 手腕追踪器与机器人末端之间的固定朝向差。追踪器绑在手背上,这个常量描述佩戴方式,是绝对朝向映射的基准 |
| `magicdexmate/head_waist_map.py` | 操作者头部朝向与上身前倾,到机器人头部三个关节与腰部关节的映射表(由操作者摆出的参考姿势拟合得到) |
| `magicdexmate/swivel.py` | 手臂冗余自由度(肘部绕肩-腕轴的旋转)的计算与传递 |
| `magicdexmate/bimanual_symmetry.py` | 操作者双臂镜像对称时,使机器人双臂也严格对称(默认关闭) |
| `magicdexmate/home_pose.py` | 机器人默认姿势的唯一定义,仿真、逆运动学与真机共用 |

### 3. 逆运动学与保护

| 文件 | 职责 |
|---|---|
| `sim/pink_vega_ik.py` | 微分逆运动学(Pink / Pinocchio):末端位姿目标解算为 14 个手臂关节角。含腕关节限位的专门处理:目标朝向超出腕关节量程时,降低朝向任务权重并把目标向当前朝向插值,使关节停在限位之外、需求回到量程内后立即恢复跟随 |
| `magicdexmate/joint_guard.py` | 最终指令层的保护:双臂自碰撞检测(将要碰撞时保持上一安全指令)、逐关节速度限幅、桌面等环境障碍 |

### 4. 仿真、显示与真机

| 文件 | 职责 |
|---|---|
| `sim/vega_scene.py` | Isaac Lab 场景定义(机器人 USD 与控制器参数;USD 优先从仓库内 `assets/isaac/` 读取) |
| `scripts/viser_isaac_mirror.py` | 浏览器显示:机器人当前姿态、操作者骨架、跟随误差、贴近关节限位的关节等读数,以及驱动真机的开关 |
| `scripts/dexmate_bridge.py` | 订阅关节指令,通过 dexcontrol SDK 驱动真机。先以低速运动到目标姿势再进入跟随;含速度限制、关节限位钳制、数据超时停止与急停检测 |
| `scripts/dexmate_observer.py` | 只读取真机关节角并发布,页面在仿真未运行时用它显示真机的当前姿势;不发送任何指令 |
| `scripts/replay_check.py` | 录制一段遥操作指令,先在浏览器与仿真中反复确认,再让真机重放同一段。首次连接真机时建议使用 |

## 从 Wuji 手套到 Sharpa 手:相关代码

| 文件 | 职责 |
|---|---|
| `magicdexmate/sources/wuji_source.py` | 接收手套数据(21 个手部关键点与手腕姿态) |
| `magicdexmate/retarget/` | 手指重定向:人手关键点解算为 Sharpa 手 22 个关节角(`frames.py` 关键点到标准手模型、`builder.py` 优化器构建、`mapping.py` 关节名映射与限位) |
| `scripts/teleop_retarget.py` | 单手管线入口:手套 → 重定向 → 发布;`teleop_retarget_dual.py` 为双手版本 |
| `sim/test_env_sharpa.py` | 仿真消费端;`sim/teleop_dual_sharpa.py` 为双手版本 |
| `magicdexmate/sinks/sharpa_real.py`,`scripts/sharpa_real_runner.py` | 真手驱动(限位、限速、看门狗)与数据录制(关节 + 三层触觉) |
| `configs/retargeting/` | 重定向参数配置 |
| `third_party/dex-retargeting` | 所依赖的重定向优化库(随仓库携带,MIT 许可) |

手部管线的更多细节见 [magicdexmate/retarget/README.md](magicdexmate/retarget/README.md) 与 [docs/hand_pipeline_notes.md](docs/hand_pipeline_notes.md)。

## 数据录制

页面上点「开始录制」后,手臂指令、真机关节、手套原始数据、手部关节与触觉、相机点云(接入后)各自写入 `data/sessions/<名称>/` 下的同一目录,统一使用主机时钟做时间戳;`scripts/merge_episode.py` 按时间戳把各路数据合并为一个文件。

## 测试

- `scripts/check_all.py`:15 项不需要硬件的检查(单元测试、各管线的机制测试、录制与回放、整体启动),改动任何代码后运行;
- `scripts/regress_teleop.py`:回归测试,用 `logs/` 中的录制素材重跑整条映射管线并输出精度与关节限位指标,与 `logs/regress/` 中保存的基线对比。

## 后续计划(TODO)

1. **IK 求解器更换为 cuRobo。** 当前使用 Pink(Pinocchio)的微分逆运动学,只做局部求解、不含碰撞约束;cuRobo 提供 GPU 并行、带碰撞感知的逆运动学与轨迹优化,预期改善大幅动作下的解算质量。更换时的验收标准:用 `scripts/regress_teleop.py` 在同一批素材上对比,腕部跟踪误差、贴近关节限位的时间占比、逐帧关节跳变三项指标不劣于 `logs/regress/` 中的现有基线。
2. **PICO 腿部动作映射为底盘运动。** PICO 的全身追踪数据包含腿部关节(髋、膝、踝、脚,见 `docs/PICO_teleop.md` 的 24 关节表),目前未使用。计划把操作者的腿部动作(如原地踏步的方向与幅度)映射为 Vega 底盘的平移与旋转速度指令。前置条件:当前仿真使用的 USD 把轮关节固定住了(`vega_1p_weldwheels.usd`,原因是停放状态的轮关节数值不稳定,会破坏全身求解),需要先恢复可动轮或改用单独的底盘运动学模型;真机侧需要接入 dexcontrol 的底盘控制接口,并补充相应的限速与急停保护。

## 文档索引

| 文档 | 内容 |
|---|---|
| [SETUP.md](SETUP.md) | 新机器部署:装什么、哪些需要向厂商获取、如何验证 |
| [SOP_wholebody_teleop.md](SOP_wholebody_teleop.md) | 手臂/头/腰遥操作的操作手册 |
| [SOP.md](SOP.md) | 手部遥操作的操作手册 |
| [docs/PICO_teleop.md](docs/PICO_teleop.md) | PICO 头显与追踪器的设置、校准与故障排查 |
| [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md) | 项目全貌(英文):设计取舍、量化结果、已解决与未解决的问题 |
| [docs/workflow_fj_snapshot_20260810.md](docs/workflow_fj_snapshot_20260810.md) | 开发过程的完整工作记录(快照) |

## 不在本仓库中的依赖

Dexmate `dexcontrol` SDK、Dexmate `dexmate-urdf`、Sharpa Wave SDK、Wuji 手套 SDK 为厂商件,依许可不随仓库分发;获取方式与安装位置见 [SETUP.md](SETUP.md)。只运行仿真时,仅需要 `dexmate-urdf`。
