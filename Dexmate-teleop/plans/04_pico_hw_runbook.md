# 计划 04:PICO 硬件日 Runbook(头显 + 全身追踪器 → Vega sim 遥操作)

对应路线图 P4 的实机联调日。**最终设备 = 头显 + 追踪器(双腕 + 腰驱动,身上共
戴 5 个、软件用 3 个),无手柄** —— 操作路径上不允许任何依赖 grip/扳机/按键的
环节(2026-07-26 用户定死)。追踪器数据走 PICO **全身 24 关节通道**
(`get_body_joints_pose`),producer 把用到的关节以固定名 LWRIST/RWRIST/WAIST
转发,**不需要序列号**。
前置:无硬件链路已全部验证(fake ZMQ 全链路、录制/回放、Pink IK 防缠绕,见
GR00T 仓库 `workflow_fj.md`)。手柄模式(grip 离合)仅作 bring-up 对比保留,
不再是任何验收项。

## 0. 目标与验收

- T-A:producer `device_ts=fresh`、`trk>=3`(LWRIST/RWRIST/WAIST 在流里),
  100 Hz 发布不掉线 5 分钟。**硬前提:头显全身追踪真正在跑**(人动关节动;
  `is_body_data_available` 持续 True —— 2026-07-23 的卡点)。
- T-B:sim 双臂 + 躯干**自动啮合**(起动 1s 后,无任何手部操作),身体相对
  1:1 跟随;键盘 `R` 重零 / `SPACE` 暂停行为正确。
- T-C:温和操作下跟随主观流畅,summary 无大范围 NEAR LIMIT 缠绕(默认 Pink IK
  已从根上防缠绕;姿态别扭按 `R` recenter)。
- T-D:全程 `--record`,事后 `--source replay` 能复现同样的动作。

## 1. 硬件到手前 checklist(已全部就绪则跳过)

1. PC Service 已装:`ls /opt/apps/roboticsservice/runService.sh`(deb 备份在
   GR00T 仓库根 `XRoboToolkit_PC_Service_1.0.0_ubuntu_22.04_amd64.deb`)。
2. `.venv-pico` 可用:`.venv-pico/bin/python -c "import xrobotoolkit_sdk, zmq, msgpack"`
   (坏了重跑 `bash scripts/setup_pico_env.sh`;零搭建替代 =
   `~/magicsim/GR00T-WholeBodyControl/.venv_teleop/bin/python`)。
3. 单元测试:`env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
   .venv-pico/bin/python -m pytest tests/test_pico_math.py`(27 passed;
   两个 env 变量是防本机 ROS humble 污染 pytest,必带)。
4. 头显侧一次性安装(参考 GR00T `docs/source/getting_started/vr_teleop_setup.md`):
   开发者模式 → 浏览器下 XRoboToolkit-PICO apk(github.com/XR-Robotics)→ 安装,
   app 出现在库的 Unknown 区。
5. **追踪器(必需)**:PICO 系统设置里完成 **Motion Tracker 配对**(5 个都配),
   并**走完全身校准(T-pose 等系统流程),确认全身追踪真正在运行**——判据:
   XRoboToolkit app 勾 Full Body 后,producer 状态行 `trk>=3` 且人动数值动。
   "connected ≠ tracking":2026-07-23 整晚卡在配对好但 24 关节冻结,该问题
   只能在头显侧解决(建议找跑通过官方 G1 遥操的师兄现场看)。

## 2. 硬件日启动顺序

**懒人一键(推荐)**:`scripts/run_teleop.sh` 用 tmux 一条命令拉起 PC Service + producer +
consumer 三个 pane。**不需要序列号、不需要 trackers.env**(producer 从全身 24
关节合成固定名 LWRIST/RWRIST/WAIST,即 consumer 默认值)。
- 直接:`scripts/run_teleop.sh`(tracker 模式)。
- 其它:`--headless`、`--no-service`、`-- <额外consumer参数>`;
  legacy:`--controllers`(手柄对比)、`--probe`(单发 motion-tracker 通道的
  序列号探针——全身模式下该通道恒空,平时用不到)。
- 停:`tmux kill-session -t vega`;切 pane:`Ctrl-b` 后方向键。

**或手动 3 终端**(下方为原始步骤,便于排查):

### 手动启动(3 个终端)

```
[T1] PC Service        bash /opt/apps/roboticsservice/runService.sh
[T2] producer(py3.10) cd <仓库根目录> && mkdir -p logs && \
                       .venv-pico/bin/python scripts/teleop_pico_producer.py \
                           --record logs/pico_$(date +%Y%m%d_%H%M%S).msgpack
[T3] consumer          cd <仓库根目录> && OMNI_KIT_ACCEPT_EULA=YES \
                       .venv-isaac/bin/python sim/teleop_vega_pico.py --source zmq
     (默认即 tracker 模式 + 固定名;legacy 手柄对比加 --mode controllers)
```

1. 工作站与 PICO 连**同一 Wi-Fi**;`ip addr` 记下工作站 IPv4。
2. 起 T1(或者给 producer 加 `--run-service` 让它代起)。
3. 头显里打开 **XRoboToolkit** app → "PC Service:" 填工作站 IP → Status 显示
   **WORKING**;Tracking 区勾 **Head + Motion Tracker(Full Body)**,戴好追踪器
   (双腕 + 腰必须;多余的脚踝忽略)。Data/Control 选 **Send**。
4. 起 T2,确认打印 `device_ts=fresh` 且 **`trk>=3`**(LWRIST/RWRIST/WAIST)。
   `STALE` = 头显 app 没在前台 / 没点 Send / 网络不通;`trk=0` = 全身追踪没在
   跑(见第 1 节第 5 条),先修好再继续。
5. 起 T3(带 GUI 观察;远程/无显示器机器用 `--headless --snap-dir /tmp/snaps`)。
   等打印 `[run] mode=trackers ... engage=auto` 后,起动 1s 自动啮合,无需任何
   手部操作。**IK 默认 Pink**(限位感知、防缠绕);对比旧路径加 `--ik dls`。
6. **首次热身建议**:`--hands right --pos-scale 0.5` 单臂半比例先试,再上双臂全比例。
7. **键盘**(GUI 窗口聚焦时):`R` = 重零/recenter(重新锚定)、`SPACE` = 暂停。
   这是无手柄形态下仅有的两个运行时操作。

## 3. 操作说明(tracker 模式,无手柄)

- **自动啮合**:起动约 1s(`--auto-engage-delay`,等机器人站稳)后,三路
  (双臂 + 躯干)自动锚定当前姿态为零点,开始**身体相对** 1:1 跟随(腕相对腰,
  一次锚定后连续跟踪;走动/转身不影响)。**全程不需要任何手部按键。**
- **重零 = 键盘 `R`**(GUI 窗口聚焦):姿态别扭/想 recenter 时按一下,下一帧
  以当前姿态重新锚定。**暂停 = `SPACE`**(再按恢复,恢复即重新锚定)。
- **数据冻结自动保护**:头显休眠/链路断 → 手臂自动脱离并打印原因;数据恢复后
  自动重新锚定(不会从旧零点跳变)。
- **躯干跟随**:腰 tracker 俯仰 → torso_j1(啮合瞬间为零点)。旋钮:
  `--torso-scale`(默认 1.0)、`--torso-max-deg`(默认 25°);不要躯干加
  `--no-torso`。
- **EE 目标默认过 one-euro 滤波**(静止去抖、快动低滞后):`--filter-mincutoff`
  (默认 1.0 Hz,0=关)、`--filter-beta`(默认 2.0;实测 1 m/s 快扫滞后 26mm,
  静止抖动压到 24%)。手感"肉"就调大 beta 或调大 mincutoff;"抖"就调小 mincutoff。
- **手臂缠绕**:默认 Pink IK 已从根上防缠绕;仍别扭按 `R` 重零。
- 调参旋钮:`--pos-scale`(人手→EE 平移比例,默认 1.0)、`--pos-max`(平移限幅
  0.8 m)、`--wrist-max-deg`(腕转限幅 120°)。`--null-k` 属实验旋钮,默认 0,
  实验时务必保持 `--null-cap`(默认 0.01)。
- (legacy `--mode controllers` 才有 grip 离合:握≥0.6 啮合/松≤0.4 冻结/再握
  重零,头显做参考系。仅 bring-up 对比用,最终形态无手柄。)

## 3.5 数据源防护与新开关(2026-07-26 深夜新增)

- **跳变门(默认开)**:物理上不可能的单步跳变(2026-07-26 审计:腕单步 170°)
  被拒收、沿用上一好值;持续 0.4s 的"新位置"(如摄像头重捕)→ 干净重锚定而非甩臂。
  `--no-glitch-gate` 关闭。
- **数值冻结检测(默认开,500ms)**:全部 tracker 数值逐位相同 = 估计器死了
  (device_ts 照走,旧检测抓不到)→ 大声报警 + 自动脱离,恢复自动重锚定。
  现场处置:**用力晃追踪器 + 低头看腕**让摄像头重捕。`--tracker-freeze-ms 0` 关闭。
- **`--ori-mode hold`**:朝向冻结在啮合姿态、只跟位置——腕朝向数据脏时(当前
  状态)建议开,坏朝向不再经 IK 拖累位置。
- **`--anchor home`**:模仿式锚定——EE 锚点固定为机器人 home 姿态,啮合时人摆
  对应中立姿势(双手胸前),则"你的手相对身体在哪 = 它的手相对身体在哪",
  每次重锚定后映射一致(治"锚定错位+工作空间饱和")。
- **骨架查看器**:producer 加 `--body-full` 后,
  `.venv-pico/bin/python scripts/view_skeleton.py` 实时画 PICO 人体估计
  (冻结变红、消费的三关节橙色高亮);也可 `--file logs/xxx.msgpack` 回放。
  **下次会话建议常开**——数据一抽风当场看见。
- **现行推荐 consumer 参数(2026-07-29,waist-abs 时代)**:
  `scripts/run_teleop.sh --no-service -- --map waist-abs --waist-yaw-fix-deg <本次穿戴反解值> --ori-mode track --physics-hz 120 --render-interval 4`
  (焊轮已默认;yaw 每次穿戴需重标:录一段前平举 → `analyze_pose_capture.py` 反解;
  `--ori-mode hold --anchor home` 为旧 anchored 时代参数,仅对照用)。

## 3.6 双通道混合方案(2026-07-29 定案;本节旧版"独立 Object 模式"已被其取代)

**终局数据源 = raw 双腕(裸序列号通道)+ model 腰/肘(人体模型通道),两通道并发**
(2026-07-29 实测:头显系统级 Motion Tracker 模式切好后,线上同时出现 3 序列号 +
LWRIST 等模型名,trk=8——"互斥"预期被推翻)。理由:裸腕视野内 σ6-17mm、出视野
"诚实冻结"(好检测),模型通道出视野 170° 抽风;而腰追踪器永远不在头显 FOV →
裸腰 100% 冻结不可用,腰只能用模型通道。

1. 头显侧两件事:系统级 Motion Tracker 切 **Object/独立追踪**(保持双通道并发)
   **+ 重做全身校准**(让 model 腰复活;硬判据=PICO 系统内人体模型准确随动)。
2. 序列号已判定并写死 `scripts/trackers.env`(raw 双腕 + model WAIST;
   run_teleop.sh 自动 source):LWRIST=PC2310MLL4151662G、RWRIST=PC2310MLL4151712G、
   腰 80588G=裸通道不可用仅留档。丢失可用 `_raw` 批数据重新自动判定
   (头显朝向 left 轴投票)。
3. 按 §3.5 现行推荐命令跑;判据:producer trk=8(3 序列号+5 模型名)、
   consumer `[map] waist-abs`、腕位置+朝向随动、**躯干动手臂不动**。

## 3.7 映射精度修复与回归基准(2026-07-29 重写:焊轮后全零新基线)

- **四项修复已入默认**(按发现顺序):①臂执行器阻尼按 kp 配比(`_ARM_KD=kp/16`,
  旧统一 150 让远端关节 τ 高达 1.3s);②Pink ori_cost 2.0→**0.5**;③**"tuck" home**
  (手收拢近胸,j1±0.5/j2∓0.3/j4−2.2/j5∓0.4,旧 home 前向余量仅 15cm);
  ④**焊轮(根因修复,2026-07-29)**:停驻轮关节数值爆走持续污染整棵关节树的求解,
  低阻抗腕关节(j6/j7 kp 210/113)被幻影冲量拖着走——爬行响应、静止停错位置、
  路径依赖、肘任务二值效应,**全部是轮子一个根因**(法证实验
  `sim/dev_wrist_forensics.py`:未钉轮时 j6 无人驱动已 −2.63 rad/s 狂飙;焊轮后
  阶跃 0.8 rad 0.1s 到位)。修复载体=`vega_1p_weldwheels.usd`(6 个 wheel joint
  转 FixedJoint),**`VEGA_WELD_WHEELS` 已默认开**(2026-07-29 转正;=0 退回旧 USD)。
- **回归基准命令**:`--headless --mock-demo --duration 48`(自动评分表)。
  **现行基准(焊轮后,2026-07-29)**:前伸 0.1 / 侧举 0.0 / 上举 0.0 mm;
  腕滚 0.0° / 腕俯仰 0.1°;track err mean **0.9mm** / max 5.4mm。
  任何明显偏离(mm 级 → cm 级)= 回归,先查 `VEGA_WELD_WHEELS` 是否被关。
- **旧基准全部作废(轮子时代,仅史料)**:2026-07-27 的 62~76mm track err、
  上举 46~115mm、腕滚 24° 等数字,以及由此派生的一切"执行瞬态/静态地板/
  局部最优分支/肘权重回退"分析,根因都是轮子,焊轮后归零,不再作对照。
- **肘任务现状**:`--elbow-weight` 默认 0 维持——但旧 A/B 的"回退"证据已随焊轮
  作废(肘二值效应=轮子污染的表现)。真人 model 肘数据的重新评估排在腕收官之后
  (见 workflow_fj.md 杂项队列)。
- **physics-hz**:焊轮后 **`--physics-hz 120` 稳定可用**(GUI 提速 2 倍,
  rtf 0.08→0.14-0.16;"120 失稳 610mm"是轮子时代结论,作废)。
  联调推荐:`--physics-hz 120 --render-interval 4`。

## 4. 已知问题与故障排查

| 症状 | 原因 / 处理 |
|---|---|
| producer `import xrobotoolkit_sdk` 失败 | 用了 py3.11 解释器。只能用 `.venv-pico` 或 GR00T `.venv_teleop`(SDK 是 cp310 预编译 so) |
| producer 打印 `device_ts=STALE` | 头显 app 不在前台 / 没勾 Send / Status 不是 WORKING / 不同网段。头显里点 Reconnect |
| consumer `frames=` 不增长 | producer 没起、端口不符(默认 :5581)或跨机器时 `--sub tcp://<producer-ip>:5581` 没指对 |
| consumer `engaged=[]` 不啮合 | tracker 模式:流里没有 LWRIST/RWRIST/WAIST(producer `trk=0` → 全身追踪没在跑,见第 1 节第 5 条)或帧过期(`--stale_ms` 默认 200ms,Wi-Fi 抖动可放宽)。legacy 手柄模式:grip 没读到(app 里 Controller 没勾) |
| 手臂缠绕、顶限位 | **默认 Pink IK 已从根上防缠绕**(2026-07-22 移植,带零空间姿态+限位;实测激进轨迹 0/14 顶限,优于 DLS)。若仍别扭:GUI 里按 `R` 重零 recenter。只有 `--ik dls` 老路径才有 j3/j5 缠绕。j6/j7 顶限是腕滚超关节量程,不是缠绕 |
| 追踪跟不上、感觉"卡顿/肉" | sim 慢于实时,arm PD 在 sim 时间里推进 → 手感发肉。`[stats]` 行直接打印 `rtf=`(实时率)与 `age=`(线上时延)。降 RTF 开销:正式跑**关掉 `--snap-dir`**;GUI 下调大 `--render-interval`(默认 4=60fps 渲染,可到 8);**焊轮后(默认)`--physics-hz 120` 稳定可用,GUI rtf 0.08→0.14-0.16 约翻倍**(旧"120 失稳 610mm"是轮子时代结论,作废;若 `VEGA_WELD_WHEELS=0` 退回旧 USD 则 120 仍不可用) |
| consumer 启动即 `CUDA unknown error`(cuInit 999),nvidia-smi 却正常 | nvidia_uvm 内核模块状态坏(2026-07-21 实测:杀掉长跑一夜的 Isaac 后新进程全部 cuInit 999)。修复:`sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`(uvm 引用数须为 0,即先退出所有 CUDA 进程);不行就重启 |
| `nvidia-smi` 直接报 `Driver/library version mismatch`,Isaac 报 NVML 18 / CUDA 804 | 系统 unattended-upgrades 升了 NVIDIA 用户态驱动而内核模块还是旧版(2026-07-27 实测:06:51 自动升级 580.159→580.173)。修复:**重启最省事**;或停桌面后重载全部 nvidia 模块。防复发:`sudo apt-mark hold nvidia-driver-580` 或关掉 unattended-upgrades 的 nvidia 源 |
| 追踪器不出现(producer `trk=0`) | 全身追踪没真正在跑(connected ≠ tracking,2026-07-23 的卡点):Motion Tracker 没配对、app 没勾 Full Body、或系统全身校准没走完/追踪器长时间出头显摄像头视野。头显侧解决;`is_body_data_available` 持续 True 且人动关节动才算好 |
| Isaac 退出时挂住 | 已知问题,脚本内置 20s 强杀,等它 |
| 纹理告警 `Image_0.png` 找不到 | 纯外观(dexmate-urdf 导出残留引用),忽略 |
| 端口被占(Address already in use) | 上一个 producer 没死:`pkill -INT -f teleop_pico_producer` |

## 5. 录制与复盘

- 每次 session 都带 `--record logs/...msgpack`(producer 发什么录什么,含按键/grip)。
- 复盘回放:`sim/teleop_vega_pico.py --source replay --replay-file logs/xxx.msgpack
  [--replay-loop]`——按 sim 时间相位锁定,真机 1× 时间戳回放时长 = 录制时长。
- ⚠ 若用 `--fake --fake-speed s` 录合成数据,回放会按 1/s 拉长(时间戳是墙钟),
  回放素材务必 1.0 倍速录(2026-07-20 踩过:0.05 倍速录制回放慢 20×)。
- 硬件日结束:把当天最有代表性的一段录制拷进 `assets/pico_logs/`(小文件)
  当回归素材,路径记入 workflow_fj.md。

## 6. 硬件日记录区(当天填)

- 日期/人员:
- GUI 模式实时率(`[stats] sim t` vs 墙钟):
- 实测 track err(温和操作,summary 输出):
- 缠绕出现频率/离合重零是否够用(不够 → 排期换 Pink IK):
- 遗留问题:
