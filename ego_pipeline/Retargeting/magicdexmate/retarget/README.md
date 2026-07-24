# Retarget：Wuji 手套 → SharpaWave 22-DOF

把 **Wuji 数据手套**采集的人手姿态实时重定向为 **SharpaWave 灵巧手**的 22 个关节角，
基于 [dex-retargeting](https://github.com/dexsuite/dex-retargeting) 的非线性优化。
这是 MagicDexMate teleop 链路的第一段，输出经 ZMQ 喂给 Isaac Lab 仿真或 SharpaWave 真机。

```
Wuji glove (21 MediaPipe 关键点 @120Hz, 腕系)
        │  sources/                每点按 name 路由，不信任数组顺序
        ▼
   HandFrame(kp[21,3], conf[21], wrist_quat)
        │  retarget/frames.to_mano()    逐帧 estimate_frame → MANO 约定（对手套腕系免疫）
        ▼
   (21,3) MANO 关键点
        │  retarget/builder.compute_ref_value()   取 5 个 指尖-腕 向量
        ▼
   ref_value (5,3)
        │  dex-retargeting  vector / DexPilot 优化（NLopt SLSQP + Huber + 限位）
        ▼
   qpos (pinocchio dof 序)
        │  retarget/mapping.JointMapper.to_sdk()   按名重排为 SDK 22 序 + 限位×0.9 clip
        ▼
   qpos_sdk[22]  ──►  sinks/qpos_publisher (ZMQ)  ──►  Isaac Lab / SharpaWave SDK
                 └──►  sinks/sapien_viz (调试可视化)
```

## 为什么这样设计

- **帧约定免疫**：`frames.to_mano()` 复用 dex-retargeting 自己的 MediaPipe 管线写法——
  每帧用 `estimate_frame_from_hand_points`（腕、食指 MCP、中指 MCP 三点 + SVD）重新估计腕系，
  因此对输入处于哪个固定坐标系**完全不敏感**。Wuji 腕系（X=桡侧、Z=朝肘）无需手工求旋转矩阵。
  实测：MANO 约定下张开手四指指向 +z、拇指偏 +y，与 SharpaWave URDF 的 `*_hand_C_MC` 基座系
  天然吻合（零位中指尖 z=0.203），所以 vector 模式开箱即用、无需额外基座旋转。
- **关节序按名映射**：`retarget()` 返回完整 pinocchio dof 序（`retargeting.joint_names`），
  与 SDK 的 `set_joint_position` 顺序（= URDF 顺序）**不同**，必须按名字重排。
  `JointMapper` 预计算索引并把目标 clip 到限位 ×0.9（与真机一致）。
- **进程隔离**：dex-retargeting 声明 `numpy>=2`（实为纯声明，见
  [docs/references/03](../../docs/references/03_dex_retargeting.md)），可独立成进程，经 ZMQ
  把一份 qpos 流同时喂仿真与真机（天然 sim/real A/B 对照）。

## 模块

| 文件 | 作用 |
|---|---|
| [`../skeleton.py`](../skeleton.py) | MediaPipe 21 关键点约定、`HandFrame`、`joint_name_to_index`（容错别名）|
| [`frames.py`](frames.py) | `to_mano(kp, hand)`：逐帧 estimate_frame → MANO；旋转/平移不变 |
| [`builder.py`](builder.py) | `build_sharpa_retargeting(hand, mode)` → `SeqRetargeting`；`compute_ref_value()` |
| [`mapping.py`](mapping.py) | `JointMapper`（pinocchio 序 → SDK 22 序 + 限位 clip）；`sharpa_sdk_joint_names()` |
| [`../sources/`](../sources/) | `MockGloveSource`（合成手）、`WujiGloveSource`（真手套，wuji-sdk 实测 API）、`make_source()` |
| [`../sinks/`](../sinks/) | `QposPublisher`（ZMQ，hello 带 joint_names+CRC 防错序）、`SapienViz` |
| [`../../configs/retargeting/`](../../configs/retargeting/) | `sharpa_wave_{right,left}_{vector,dexpilot}.yml` |

## SharpaWave 关节序（SDK = URDF 顺序）

`JointMapper.to_sdk()` 输出的 22 维即 SDK `set_joint_position` 顺序，单位 rad：

```
thumb : CMC_FE, CMC_AA, MCP_FE, MCP_AA, IP
index : MCP_FE, MCP_AA, PIP, DIP
middle: MCP_FE, MCP_AA, PIP, DIP
ring  : MCP_FE, MCP_AA, PIP, DIP
pinky : CMC, MCP_FE, MCP_AA, PIP, DIP
```

## vector vs DexPilot

| 模式 | 用途 | 特点 |
|---|---|---|
| `vector` | 调试 / 数据采集 | 5 个 指尖-腕 向量匹配，无捏合先验，最忠实 |
| `dexpilot` | 日常 teleop | 额外投影拇指-各指距离，捏合更稳、抓取更自然 |

`scaling_factor: 1.07`（= URDF 腕→中指尖 203.7mm ÷ 人手实测 ~190mm，每操作者可在 yaml override；
用 `scripts/check_env.py` 打印的机器手尺寸重算）。`low_pass_alpha: 0.2`（越小越平滑、延迟越大）。

## 用法

环境（uv，见根 [README](../../README.md)）：`bash scripts/setup_env.sh`。
SharpaWave URDF + mesh 已随库提供（`assets/robots/hands/sharpa_wave/`，Apache-2.0 © Sharpa）；
如需从厂商源重新生成：`.venv/bin/python scripts/prepare_assets.py`。

### 库 API

```python
from magicdexmate.retarget.builder import build_sharpa_retargeting, compute_ref_value
from magicdexmate.retarget.frames import to_mano
from magicdexmate.retarget.mapping import JointMapper

retargeting = build_sharpa_retargeting("right", "vector")   # 或 "dexpilot"
mapper = JointMapper(retargeting, "right")

# frame: HandFrame，kp 为 (21,3) MediaPipe 序、腕系
ref = compute_ref_value(retargeting, to_mano(frame.kp, "right"))
qpos_sdk = mapper.to_sdk(retargeting.retarget(ref))          # (22,) rad, SDK 序
```

### 实时脚本

```bash
# 自检（imports/资产/4 配置/跟踪精度/mock 管线/ZMQ/耗时）—— 期望 10/10
.venv/bin/python scripts/check_env.py
.venv/bin/python -m pytest tests/ -q                         # 11/11

# mock 手套 → retarget → ZMQ(:5556)（+SAPIEN 可视化，需显示器）
.venv/bin/python scripts/teleop_retarget.py --source mock --motion cycle --viz

# 真 Wuji 手套（接好网线、主机在 192.168.1.x 网段后）
.venv/bin/python scripts/teleop_retarget.py --source wuji --hand right --viz
```

常用参数：`--hand right|left`、`--mode vector|dexpilot`、`--rate`（默认 60Hz）、
`--sn/--address`（指定手套）、`--duration N`（定时退出）、`--no-pub`。

下游消费见根 README 的「模式 A 双进程桥接」与 `sim/test_env_sharpa.py --motion zmq`。

## ZMQ 输出契约

`QposPublisher` 发 JSON 行：

- `hello`（每 2s 重发，供晚加入的订阅者）：`{type, hand, joint_names[22], crc}`
- `qpos`：`{type, seq, hand, t_capture_us, t_pub_us, valid, crc, qpos[22], wrist_quat}`

`crc` = crc32(`,`.join(joint_names))。**消费端必须校验 crc**，与建索引映射时用的 hello 一致，
防止两侧关节顺序错配。

## 实测指标（2026-06, mock 全链路）

| 指标 | 数值 |
|---|---|
| 指尖向量跟踪误差（FK 验证，60Hz 顺序跟踪）| 四指 7–11mm，拇指 ~17mm |
| retarget 单步耗时 | p50 2.7ms / p95 5.0ms（≈200Hz 上限，torch-cpu）|
| 端到端（采集→ZMQ 发布）| p50 6.4ms @ 60Hz |
| 自检 / 单测 | check_env 10/10，pytest 11/11 |

## 已知事项

- **随机极端位姿单步求解差**（驻点 ~52mm）：teleop 是连续小步 + 低通 + 限位 clip，不进入该工况；
  离线批处理勿从任意位姿单步求解。详见 [docs/references/03 §5.5](../../docs/references/03_dex_retargeting.md)。
- 拇指跟踪偏松：人/机拇指构型差异，真手套数据到位后按计划 01 R5 调 scaling / DexPilot 参数。
- 帧约定校准已用旋转不变性单测覆盖；真手套接入主要再核对一次腕系镜像（左手）。

参考：[docs/references/01 Wuji](../../docs/references/01_wuji_glove.md) ·
[02 SharpaWave](../../docs/references/02_sharpa_wave_hand.md) ·
[03 dex-retargeting](../../docs/references/03_dex_retargeting.md) ·
[plans/01 retarget 计划](../../plans/01_retarget_plan.md)
