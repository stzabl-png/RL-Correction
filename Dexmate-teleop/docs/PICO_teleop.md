# PICO 追踪器数据获取 —— 官方机制与操作 SOP

> 依据:本仓库 vendored SDK(`external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/`
> README.md + examples/example_body_tracking.py)与 NVIDIA 官方文档
> (`docs/source/getting_started/vr_teleop_setup.md`、`docs/source/tutorials/vr_wholebody_teleop.md`)。
> 排查动态见 `workflow_fj.md`。

## 官方获取机制(代码层)

- 读取**纯被动**:`xrt.init()` → `while not xrt.is_body_data_available(): sleep` →
  `xrt.get_body_joints_pose()`(24×7,scalar-last qx,qy,qz,qw)。
  **SDK 无任何"启动追踪"指令,PC 端无法主动拉起**;数据来不来完全取决于头显端。
- 两条追踪器通道:
  - **全身 24 关节**(`is_body_data_available` / `get_body_joints_pose`)——PICO 5 追踪器
    的实际输出通道,我们用这条(producer 合成固定名 LWRIST/RWRIST/WAIST);
  - 单发 motion tracker(`num_motion_data_available` / `get_motion_tracker_data`,序列号键)
    ——Full Body 模式下恒空,legacy。

## 官方三前提(SDK README 原文,数据不来时的 checklist)

1. PICO headset is connected;
2. **Body tracking is enabled in the control panel**(= 头显 XRoboToolkit app 勾 Full Body);
3. **At least two Pico Swift devices are connected and calibrated**
   (**connected ≠ calibrated ≠ tracking**,配对只是第一步,校准必须走完)。

另,NVIDIA 教程加粗警告:**必须穿贴身衣物**保证追踪器在头显摄像头视线内,
宽松衣物会让追踪"不可预测地失败"(官方绑脚踝故强调裤子;腕/腰同理)。

## 头显侧 SOP(vr_teleop_setup.md Step 2)

1. 设置 → 开发者(看不到就连点"软件"数次)→ **Safeguard 关掉**;
2. 追踪器管理入口:快捷菜单 **Wi-Fi 图标** → 头显图片上方**小圆形 Motion Tracker 图标**;
   没有该图标则打开应用库里的 **Motion Tracker(体感追踪器)app**;
3. 每个追踪器 "i" 图标 → 先**全部解绑** → 右上角 **Pair** → 长按追踪器顶部按钮 **6 秒**
   (红蓝闪 = 配对模式);
4. 蓝色 **Calibrate** 按钮,两段:①站直不动、手臂自然下垂;②**低头盯着追踪器**直到
   头显摄像头识别(我们的重点是腕+腰:抬腕入视野,卷起衣袖,灯朝外);
5. 校准后头显可架额头但**必须朝前**(摄像头需持续看到追踪器);**关闭/调长自动休眠**;
6. XRoboToolkit app:PC Service 填工作站 IP → WORKING → Tracking 勾
   **Head + Motion Tracker(Full Body)** → Data/Control 选 **Send**,保持前台。

## PC 侧验证

```bash
cd <仓库根目录> && scripts/run_teleop.sh --no-service   # 服务已跑时
# producer 窗格判据:device_ts=fresh 且 trk>=3;硬判据 = 人动数值动
# consumer 自动啮合(trackers 模式默认,无手柄,键盘 R 重零 / SPACE 暂停)
```

独立最小验证(不经我们代码,须先停 producer——SDK 反复 init/close 会不稳):
`.venv_teleop/bin/python external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/examples/example_body_tracking.py`

## 24 关节索引表(SDK README)

0 Pelvis, 1 L_Hip, 2 R_Hip, 3 Spine1, 4 L_Knee, 5 R_Knee, 6 Spine2, 7 L_Ankle,
8 R_Ankle, 9 Spine3, 10 L_Foot, 11 R_Foot, 12 Neck, 13 L_Collar, 14 R_Collar,
15 Head, 16 L_Shoulder, 17 R_Shoulder, 18 L_Elbow, 19 R_Elbow, **20 L_Wrist,
21 R_Wrist, 22 L_Hand, 23 R_Hand**(官方 G1 用 22/23 驱臂;producer 已对齐 22/23)。

## 排查终局(2026-07-26 深夜)—— 已跑通,根因 = 全身校准未完成

- **✅ 无手柄跑通**:用户把 PICO 系统全身校准**彻底走完**后(硬判据 = PICO 系统内
  的人体模型准确随动),`trk=3` 稳定、机器人端到端跟动,**手柄全程未用**。
- **根因修正**:此前"手臂数据源=手柄、无手柄全身估计不跑"的推论**不成立**——
  真正的开关是**全身校准完成且生效**。官方流程那次跑通与手柄同在场是相关非因果。
  三天卡点(connected≠tracking、24 关节冻结/恒 False)全部源于校准未真正生效。
- **成功 SOP 的关键一步(写死)**:校准后必须在 PICO 系统里**亲眼确认人体模型
  准确随动**,再切到 XRoboToolkit app(前台、WORKING、Full Body、Send)。
  模型不动 = 白校,数据必不通。
- 独立/物体追踪模式(追踪器直出 6DoF+序列号,`get_motion_tracker_data` 通道,
  probe/trackers.env 架构保留)降级为**备选**,body 通道即为主路。
- **坐标轴已锤定(离线,录制 pico_20260726_191602.msgpack)**:本通道原始帧 =
  **x右 / y上 / −z前**(与 07-23 快测一致,两次独立实测吻合)→ consumer 的
  `R_HEADSET_TO_WORLD` 正确,官方 Q(+z 前)不适用本通道。
- 已知设备现象:世界系偶发瞬跳(6-8m,追踪丢失重捕)——body-relative 映射免疫;
  头显休眠会冻结全部数据(consumer 有自动脱离保护),**务必关闭自动休眠**。
- 手感问题两根因已修(详见 workflow_fj.md 2026-07-26 深夜条):trackers 模式
  活腰参考(冻结参考=误差累计感)、腕关节默认 20/21(22/23 仅手柄在手时正确)。

## 数据源质量审计(2026-07-26 深夜,47min 全场录制,物理合理性判据)

| 通道 | >10cm/50ms 位置跳变 | >20° 姿态跳变 | 冻结>300ms |
|---|---|---|---|
| 左腕(关节20) | 84 | 324(max 170°) | 32 段 |
| 右腕(关节21) | 92 | 359(max 172°) | 32 段 |
| 腰(Spine3) | **0** | **0** | 32 段 |
| 头(头显) | **0** | **0** | 50 段 |

- **结论:无手柄时全身估计的"手臂"部分不可靠**(几百次物理不可能的跳变),
  腰/头(有真实设备锚定)干净 → **独立追踪模式(每 tracker 直出 6DoF)预期
  达到"腰级"质量,是无手柄形态的首选下一步**。
- **值冻结病**:体估计会整体冻住(数值逐位相同)而 device_ts 照走 → 现有
  停滞检测(只看 device_ts)抓不到;需补数值级冻结检测。恢复手段:用力晃醒
  追踪器 + 低头看腕让摄像头重捕。
- **新防护实战回放验证(2026-07-26 夜,对整场故障录制离线跑)**:
  TrackerFreezeDetector 抓到 7 次冻结事件(43% 帧,吻合已知休眠/冻结时段);
  TrackerGlitchGate 双腕各拦 ~9.7k 坏步 + 225/230 次持续位移干净重锚定,
  **腰仅 205 步(0.06%)** —— 脏通道大量拦截、干净通道零打扰,阈值校准合适。
- 映射几何本身已用四姿势标定验证正确(方向/幅度全对);机器人侧 L_ee≡L_arm_l8
  (URDF 验证,零平移)。"不准"= 数据源脏 + rtf 0.1× 慢放 + 朝向噪声拖位置。

## 独立追踪(Object tracking)模式查证(2026-07-26 夜,资料闭环)

- **XRoboToolkit app 官方就有三种模式**:None / Full body / **Object tracking**
  (官方 Q&A:仅 object 模式下 PC-service 显示追踪器数量;full body 输出 24 关节)。
  → 7-23"只有 Full body"疑为漏看下拉框;**大概率无需升级 apk**,开机 10 秒可验。
- **PC 侧 pybind 单发通道上限 3 个追踪器**(py_bindings.cpp `array<...,3>`)
  = 恰好双腕+腰;**切独立模式时双踝追踪器关机**。
- PICO 独立追踪是正式功能(官方示例 PICOMotionTrackerSample-Unity),
  但与身体模式**不能混用**(独立模式下无人体估计——正合适)。
- **遮挡风险独立模式同样存在**(视野内=摄像头+IMU 完整追踪,出视野=纯 IMU 漂移)
  ——但输出是诚实实测、无模型脑补,预期无 170° 级抽风;待视野内/外 A/B 定量。
- **下次会话步骤(≈5 分钟)**:app Mode 切 Object tracking、TrackerNum=3、双踝关机
  → `scripts/run_teleop.sh --probe` 挥肢体认 3 个序列号 → 填 `scripts/trackers.env`
  → 正常跑 + 视野内/外各 1 分钟采集审计。

## 问题原因假设清单(2026-07-26 夜,尽可能穷举;状态实时更新)

### A. 腕部数据跳变/漂移/"左臂举着"(核心问题)

| # | 假设 | 状态 | 验证/处置 |
|---|---|---|---|
| A1 | 腕追踪器频繁出头显摄像头视野 → 纯 IMU 推算漂移,重捕获时"啪"地跳变 | **最可能**,未定量 | 下次视野内/外 A/B 采集;操作时手保持胸前 |
| A2 | 无手柄时人体模型对手臂的估计本身不可靠(模型脑补) | 可能,与 A1 叠加 | 切独立追踪模式绕开人体模型即可判别 |
| A3 | 腕追踪器中途休眠(静止即睡) | 未排除 | 下次在 PICO 界面确认各追踪器实时状态;静止一段后复查 |
| A4 | 追踪器绑带松动/佩戴位移 | 未排除(物理) | 下次紧固;观察跳变是否伴随大动作 |
| A5 | PICO 4 Ultra 增强身体追踪期望戴**前臂/大腿**,我们戴手腕 → 模型拟合差 | 可能(资料:5 追踪器模式官方说法是 forearm/thigh) | 下次试戴前臂位置 A/B |
| A6 | 磁干扰(机器人/电脑/金属桌)影响 IMU 航向 | 未排除 | 换空旷位置试;观察 yaw 漂移方向性 |
| A7 | 我们的处理链 bug | **已排除**(四姿势静态验证方向幅度全对;wire 5ms 无丢帧;旁路采集与 consumer 同数学) |—|
| A8 | 传输层丢包/乱序 | **已排除**(腕冻结时同一消息里 head 是活的) |—|
| A9 | 官方 22/23 vs 20/21 取点错 | **已修**(20/21 = 追踪器真实位置;22/23 是无手柄时的模型外推) | 已改默认,待实测确认改善 |
| A10 | 参考系冻结导致"累计误差感" | **已修**(trackers 模式改每帧活腰参考) | 待实测确认 |
| A11 | 头显系统/固件版本问题 | 未排除 | 下次检查 PICO OS 与追踪器固件更新 |
| A12 | 反光/红外干扰(阳光、镜面)影响摄像头识别 | 未排除 | 注意环境;拉窗帘试 |

### B. 延迟高

| # | 假设 | 状态 | 处置 |
|---|---|---|---|
| B1 | 仿真 RTF 0.10-0.15×(GUI 渲染开销,8G 笔记本 GPU) | **确认主因**(wire 仅 5ms) | headless 0.31×;`--render-interval 8`;真机无此层 |
| B2 | wall-clock 数据源 × 慢 sim:动作在慢放世界里被抹平 | **确认**(mock/replay 锁 sim 时间所以以前测不出) | 同 B1;评估手感必须太极速度或真机 |
| B3 | one-euro 滤波滞后 | 次要(实测 1m/s 时 26mm) | 数据源干净后可调 `--filter-*` |
| B4 | ZMQ/msgpack 链路 | **已排除**(mean 1.7ms/p95 2.8ms) |—|

### C. "map 不准"的其余构成

| # | 假设 | 状态 | 处置 |
|---|---|---|---|
| C1 | 朝向噪声经 IK 8:2 权衡拖累位置 | 机理确认,量待测 | 新 `--ori-mode hold` 一开即隔离 |
| C2 | 锚定姿势错位 + 工作空间饱和(人的大幅动作超 Vega 可达域) | 确认存在(上举 +1.0m 必超) | 新 `--anchor home` + 啮合时摆中立姿势;必要时 `--pos-scale 0.7` |
| C3 | 期望差:用户要"姿态模仿",增量式只跟腕点 | 确认(用户明说) | `--anchor home` 即模仿式;肘部再不像人 → 后续肘提示任务 |
| C4 | 腕朝向缺 OFFSETS 对齐 | 存在但被 C1 掩盖 | 数据源+C1 解决后再标 |
| C5 | 比例(人臂 vs Vega 臂) | 待标 | 数据干净后按手感调 `--pos-scale` |

### 关于"仿真人体模型做媒介"的评估(用户 2026-07-26 提议)

- PICO 的 24 关节输出**本身就是人体模型媒介**(官方与我们消费的同一等价物)——再加一层
  仿真人体模型**不能修复坏输入**(它会被同样的坏数据驱动)。
- 提议中真正有价值的两个成分已分别落地:①**解剖学/物理合理性过滤** → TrackerGlitchGate
  (人类速率上限判跳变)+ TrackerFreezeDetector(真传感器必有噪声判冻结);
  ②**看得见的媒介** → `scripts/view_skeleton.py` 实时骨架查看器(producer `--body-full`),
  下次现场一眼看出估计冻结/抽风。
- 完整的"约束拟合人体模型"(固定骨长+关节限位+时序先验去拟合 3 个原始 tracker)保留为
  **独立模式也不干净时的后备方案**(原理正确但工程量大,先用便宜的验证阶梯)。

## 明确范围约束(用户 2026-07-26)

- **头部完全不做**:不取头显位姿驱动 Dexmate 头关节(代码现状即如此:head_j1..3
  恒默认位;tracker 模式下头显 pose 仅作诊断,不参与映射)。
- 无手柄(见 workflow_fj.md 置顶硬约束);轮式底盘、手指均不在本期范围。
