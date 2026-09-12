# 计划 01：Wuji 手套 → SharpaWave Retarget 代码

对应路线图 P1。事实依据：references/01（Wuji）、02（Sharpa）、03（dex-retargeting）。

> **实施状态（2026-06-11）**：R0/R1/R4/R6 已完成并验证（mock 数据，check_env 10/10、pytest 11/11；四指跟踪 7-11mm，单步 2.7ms）。R2 的方案改为**逐帧 estimate_frame**（与库的 MediaPipe 管线同路，对手套腕系约定免疫，旋转不变性有单测），不再需要 48 矩阵枚举校准；wuji-sdk 2026.6.2 已装好并按真实 API 写了 `WujiGloveSource`，待真手套连通做最终验证。R3（MCAP 录制/回放）与 R5（DexPilot 捏合调参、拇指 scaling）待手套到位。环境用 uv（`.venv`），非 conda。

## 0. 目标与验收

**输入**：Wuji `hand_skeleton()` —— 21 个 MediaPipe 关键点（腕系，120Hz，带 per-joint confidence）。
**输出**：SharpaWave 22 维关节角（rad，URDF/SDK 顺序）+ 腕姿态 quat（来自手套 IMU），经 ZMQ 发布，供 SAPIEN/Isaac/真机消费。

验收标准：
- M1 离线：回放录制的 MCAP，SAPIEN 渲染的 Sharpa 手与人手动作语义一致（握拳/张开/逐指捏合可辨认）。
- M2 实时：端到端（手套时间戳→qpos 发布）延迟 < 50ms，retarget 循环 ≥60Hz 不丢帧。
- M3 捏合：dexpilot 模式下拇-食/拇-中捏合指尖能真正闭合（distance < 5mm in FK）。
- M4 左右手都通过 M1–M3。

## 1. 输入输出契约

```
WujiFrame  : {t_us, hand∈{L,R}, kp: float32[21,3] (m, wrist系, MediaPipe序), conf: float32[21], wrist_quat: float32[4]}
QposMsg    : {t_capture_us, t_pub_us, hand, qpos: float32[22] (rad, SDK序), wrist_quat: float32[4], valid: bool}
```

- MediaPipe 序：0=wrist，4/8/12/16/20=拇/食/中/无名/小指尖。**用 skeleton 每点的 `name` 字段建索引，不假设数组顺序**。
- SDK 序（=URDF 序，见 references/02 §2）：thumb[CMC_FE,CMC_AA,MCP_FE,MCP_AA,IP] + index/middle/ring[MCP_FE,MCP_AA,PIP,DIP] + pinky[CMC,MCP_FE,MCP_AA,PIP,DIP]。

## 2. 代码骨架

conda env `teleop`（python 3.11）：`wuji-sdk`、`pip install -e third_party/dex-retargeting`（已 vendored 入库；含 torch/nlopt/pin/numpy>=2）、`pyzmq`、`mcap`；`sapien==3.0.0b0` 仅 `--viz` 预览可选。

```
MagicDexMate/
  magicdexmate/
    sources/wuji_source.py        # WujiSkeletonSource: connect/auto_connect, get_latest()->WujiFrame；MCAP 回放同接口
    retarget/frames.py            # R_wuji2mano 常量 + convert(kp_wuji)->kp_mano；校准脚本输出贴回这里
    retarget/builder.py           # build_sharpa_retargeting(hand, mode)->SeqRetargeting（set_default_urdf_dir+load_from_file）
    retarget/mapping.py           # JointMapper: retargeting.joint_names -> SDK22 序 / Isaac 序 的 index 数组 + clip 限位
    sinks/sapien_viz.py           # 实时可视化（抄 example/vector_retargeting/show_realtime_retargeting.py）
    sinks/qpos_publisher.py       # ZMQ PUB tcp://*:5556, msgpack(QposMsg)
    sinks/sharpa_real.py          # 真机（实现见计划 03）
  configs/retargeting/
    sharpa_wave_right_vector.yml / sharpa_wave_right_dexpilot.yml / left 同款
  assets/robots/hands/sharpa_wave/
    right_sharpa_wave.urdf + meshes/ ；left 同款（自 sharpa-urdf-usd-xml 拷贝+路径修正）
  scripts/
    prepare_assets.py             # R0：拷贝 URDF、sed package://、pinocchio 自检
    record_glove.py               # TopicRecorder 录 MCAP
    calibrate_frame.py            # R2：枚举求 R_wuji2mano
    replay_retarget.py            # R3：MCAP -> qpos -> SAPIEN 渲染
    teleop_viz.py                 # R4：实时 手套->retarget->SAPIEN+ZMQ
    estimate_scaling.py           # R1：FK 算 scaling_factor 初值
```

## 3. 任务分解

### R0 资产准备（0.5 天）

1. `scripts/prepare_assets.py`：从 `~/luhr/magicsim/Sharpa/sharpa-urdf-usd-xml/wave_01/{right,left}_sharpa_wave/` 拷贝 `*.urdf` + `meshes/` 到 `assets/robots/hands/sharpa_wave/`；把 54 处 `package://{side}_sharpa_wave/meshes/` 改写为相对 `meshes/`（pinocchio 不需要，但 SAPIEN 可视化需要）。
2. 自检：pinocchio 载入后 `dof==22`、关节名集合与 references/02 §2 一致；FK 到 5 个 `*_fingertip` link 不报错。
3. 注意用**裸手变体**（root=`*_hand_C_MC`），不要 with_flange/with_wrist。

### R1 Retarget 配置（0.5 天）

`configs/retargeting/sharpa_wave_right_vector.yml` 初稿：

```yaml
retargeting:
  type: vector
  urdf_path: sharpa_wave/right_sharpa_wave.urdf
  target_origin_link_names: [ "right_hand_C_MC", "right_hand_C_MC", "right_hand_C_MC", "right_hand_C_MC", "right_hand_C_MC" ]
  target_task_link_names: [ "right_thumb_fingertip", "right_index_fingertip", "right_middle_fingertip", "right_ring_fingertip", "right_pinky_fingertip" ]
  target_link_human_indices: [ [ 0, 0, 0, 0, 0 ], [ 4, 8, 12, 16, 20 ] ]
  scaling_factor: 1.1      # 占位，由 estimate_scaling.py 确定
  low_pass_alpha: 0.2
```

`sharpa_wave_right_dexpilot.yml`：

```yaml
retargeting:
  type: DexPilot
  urdf_path: sharpa_wave/right_sharpa_wave.urdf
  wrist_link_name: "right_hand_C_MC"
  finger_tip_link_names: [ "right_thumb_fingertip", "right_index_fingertip", "right_middle_fingertip", "right_ring_fingertip", "right_pinky_fingertip" ]
  scaling_factor: 1.1
  low_pass_alpha: 0.2
```

左手把 `right_` 全换 `left_`。`scaling_factor` 估法（`estimate_scaling.py`）：URDF 张开位 FK 算 `C_MC→middle_fingertip` 距离 ÷ 操作者实测同段长度（约 0.17–0.19m）；对照内置手参考值（allegro 1.6、shadow/ability 1.2），预期 Sharpa ≈ 1.0–1.2。每个操作者可在配置里 override。

### R2 Wuji 输入适配 + 帧约定转换（1 天，关键风险点）

1. `WujiSkeletonSource`：`SdkManager.auto_connect(handedness=...)` → `hand_skeleton().subscribe_with_callback()`，回调里按 `name` 重排成 MediaPipe 序 (21,3)，平移到腕原点（减 kp[0]），写入 latest-slot（带锁单槽，丢旧保新）；同时订阅 `imu_palm()` 存 wrist_quat。confidence 门控：任一指尖 conf < 阈值 → 该帧 `valid=False`（消费端保持上一目标）。
2. **帧约定转换**：dex-retargeting 期望 MANO 朝向约定（库内 MediaPipe 管线是 `kp @ wrist_rot @ OPERATOR2MANO`）；Wuji skeleton 已在腕系（X=桡侧，Z=朝肘），即只差固定旋转 `R_wuji2mano`：`kp_mano = kp_wuji @ R_wuji2mano`。
   - 推导：按两边坐标系定义手工写出候选 R；
   - **兜底校准法**（`calibrate_frame.py`）：录一段"张开+握拳+捏合" MCAP，枚举全部 48 个带符号轴排列矩阵，对每个候选跑 retarget 并累计优化残差（`VectorOptimizer` 的 Huber loss），取 argmin；残差应显著低于次优解，否则人工复核。左手单独校准（腕系镜像）。
   - 结果固化为 `frames.py` 里的常量 + 注释推导过程。
3. 单元测试：合成一个理想张开手 kp，经 convert 后指向 +x 半空间（MANO 手指朝向），拇指在正确一侧。

### R3 离线回放管线（1 天）

1. `record_glove.py`：录 4 段标准动作 MCAP：full-open / fist / 拇指逐指捏合 / 1-5 数数，各 ~15s，左右手。
2. `replay_retarget.py`：MCAP → WujiFrame 流 → retarget → qpos 序列；SAPIEN 双窗渲染（人手 21 点散点 + Sharpa 手），逐帧对照。
3. 量化指标：对每帧把 qpos 做 FK，算 5 个指尖向量 vs 人手向量（×scaling）的残差，报告 per-finger RMS；目标 < 1.5cm。
4. 这一步同时回归测试 R1 的 scaling 与 R2 的 R 矩阵。

### R4 实时管线（1 天）

1. `teleop_viz.py` 主循环：latest-slot 取帧 → convert → `retarget.retarget(ref_value)` → `JointMapper`（名字→SDK 序 index 数组，预计算；clip 到限位×0.9）→ SAPIEN 渲染 + ZMQ 发布。
2. 性能预算：retarget 单步 ms 级（库自带 profiling 例程证实），120Hz 输入下目标循环 ≥60Hz；不够就输入降采样到 60Hz，而不是降 alpha。
3. `low_pass_alpha` 调参：0.2 起步；真机阶段倾向更小（更平滑），可在 QposMsg 之外由消费端再加一层滤波，源头保持低延迟。
4. ZMQ schema 版本号 + `joint_names` 哈希放进首条 hello 消息，消费端校验，防止顺序错配。

### R5 DexPilot 与捏合质量（0.5 天）

- 同一段捏合 MCAP 分别跑 vector / dexpilot，比较拇-食指尖 FK 距离曲线；dexpilot 的 `project_dist/escape_dist` 用默认值起步，捏合"吸合/脱开"迟滞不自然时再调。
- 产出推荐配置：日常 teleop 默认 dexpilot，调试/数据采集用 vector（无先验、更忠实）。

### R6 测试与工程化（0.5 天）

- pytest：左右手 4 个 yaml `build()` 通过；joint_names 集合==URDF 22 名；JointMapper 往返恒等；随机 qpos→FK 生成 ref→retarget 残差<1e-2（抄 tests/test_optimizer.py 套路）；回放确定性（同 MCAP 两次输出一致）。
- README 补运行命令。

## 4. 风险与未知项

| 风险 | 影响 | 缓解 |
|---|---|---|
| R_wuji2mano 推错 | 手势完全错乱 | R2 的枚举校准兜底，残差量化判定 |
| skeleton 数组顺序≠MediaPipe 编号 | 指对错位 | 按 `name` 字段建映射 + 单测 |
| scaling 因人而异 | 捏合够不着/过冲 | per-user override + estimate_scaling 流程 |
| 拇指构型差异（人 CMC3 vs Sharpa CMC2+MCP2） | 拇指对掌姿态怪 | vector/dexpilot 本就只匹配指尖向量，由优化器吸收；必要时给拇指向量加独立 scaling（库支持 per-vector scaling 的话；否则调 origin 选点） |
| numpy>=2 与下游冲突 | 环境装不齐 | 已决策分进程（roadmap §4） |
| 手套未到/没录数据 | P1 无法验证 | 先用 dex-retargeting 自带 MediaPipe 摄像头示例喂同一套 Sharpa 配置（只换输入源），管线先跑通 |

## 5. 里程碑

M1 离线回放（R0–R3，约 3 天）→ M2 实时可视化（R4，+1 天）→ M3 捏合达标（R5）→ M4 左手 + 测试齐（R6）。总计约 1 周。
