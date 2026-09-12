# 计划 03：SharpaWave 真机操控

对应路线图 P3。SDK 用法依据 references/02（自 sharpa-rl-lab 部署代码反推，PDF 补齐后逐项核对）。

## 0. 目标与验收

手套实时驱动 SharpaWave 真机（先右手、固定安装在台架上，不涉及手臂）。

验收：
- T-A：真机平稳跟随手套，无异响/过流/抖振，连续 10 分钟无故障。
- T-B：遥操作抓起并保持 3 类物体（圆柱/方块/软球）各 ≥5s。
- T-C：端到端延迟（手套时间戳→`set_joint_position` 调用）实测 < 80ms。
- T-D：安全链全部验证：键盘冻结/回 home、看门狗、限位 clip、低速系数启动。

## 1. 前置阻塞与物料 checklist

1. ✅ **SDK 已就位（2026-06-22）**：SharpaWave SDK **5.0.3** 装在 `/opt/sharpa-wave-sdk/`，手册 `~/MagicSharpa/UserManual/`；API 已对样例+手册核实（见 references/02）。控制通路**已实现**：`magicdexmate/sinks/sharpa_real.py`（`SharpaRealHand` 单手驱动）+ `scripts/sharpa_real_runner.py`（ZMQ→真机，单/双手，安全链）。触觉 180Hz docker 配方待需要触觉数据时再接。
2. 硬件：手的供电 + 以太网线；手在 `192.168.10.x`（左 .10 / 右 .20），host NIC 192.168.10.240。⚠ **当前两只手离线**（ping 不通、SDK 发现为空）——上电接网后才能实跑验证。
3. PDF 到手后**核对清单**：SDK 上限控制频率、电气规格、`set_speed_coeff/current_coeff` 的确切语义、触觉通道→手指对应、固件限位。

## 2. Bring-up 步骤

1. 装 SharpaPilot（deb）→ 按厂商流程做**手标定 + 触觉标定**；确认生成 `~/.sharpa-pilot/config/tactile.json`。
2. SDK：把 `python/sharpa/` 加入 `teleop` env 的 sys.path（py3.10–3.13 都有预编译 .so）；`python -c "from sharpa import SharpaWaveManager"`。
3. 连通测试脚本 `scripts/hand_hello.py`：manager → `get_all_device_sn()` → connect → `get_device_info()`（记下 ip/sn）→ 读 `get_states().angles` → 不动任何关节，退出。
4. 首次动作测试 `scripts/hand_home_wave.py`：POSITION 模式、`speed_coeff=0.2`、`current_coeff=0.3` → 回 home（22×0）→ 单指逐个小幅正弦 → stop。**首测手内不要有物体，桌面留空。**

## 3. 控制通路实现（`magicdexmate/sinks/sharpa_real.py`)

初始化序列（照部署范本）：

```python
manager = SharpaWaveManager.get_instance(); time.sleep(1)
hand = manager.connect(sn)
hand.set_control_mode(ControlMode.POSITION)
hand.set_speed_coeff(0.3)          # 联调期 0.3，熟练后 0.5
hand.set_current_coeff(0.3)
hand.set_control_source(ControlSource.SDK)
hand.start()
```

控制循环（独立线程，订阅计划 01 的 ZMQ 流）：

- 频率：**20Hz 起步**（部署范本同款）→ 稳定后提 60Hz → 摸 SDK 上限（PDF 核对）。sleep 控速，记录实际周期分布。
- 每步：取最新 QposMsg → `valid` 检查 → clip(限位×0.9，并与固件限位取交) → **目标限速**（每步每关节 Δ≤ speed_coeff 对应安全步长，初值 0.05rad@20Hz）→ `set_joint_position(q22)`。
- **engage 缓启动**：按下启用键时，从 `get_states().angles` 线性插值到当前手套目标（1.0s），避免跳变。

## 4. 安全机制（全部做完才允许 T-B 抓取）

| 机制 | 实现 |
|---|---|
| 键盘控制 | 抄 `sharpa-rl-lab/rl_isaaclab/utils/keyboard_listener.py`：`e`=engage、`w`=freeze（保持当前角）、`q`=缓慢回 home、`x`=hand.stop() |
| 看门狗 | 手套帧 stale > 150ms 或 ZMQ 断流 → 自动 freeze；恢复需重新 engage |
| 限位 | clip 限位×0.9 + 步进限速（上表）；拇指 CMC_FE 大行程关节单独盯 |
| 电流/速度 | `current_coeff=0.3` 联调红线；任何异响/高温立刻 `x` |
| 物理 | 台架固定牢靠、手周围净空、急停=拔以太网也安全（固件应停在原位——PDF 核对此行为） |

## 5. 联调步骤

1. **T1 脚本动作**（无手套）：home→张开→握拳循环，验证通路与限速。
2. **T2 MCAP 回放**（无人手实时参与）：计划 01 录的"张开/握拳/捏合"以 0.5× 速度回放到真机，旁观对照仿真渲染。
3. **T3 实时低速**：speed_coeff=0.3 实时跟手，空手做全套动作；记录延迟（手套 t_us vs 调用时刻，时钟经 wuji 时间同步对齐）。
4. **T4 抓取矩阵**：圆柱（柱抓）/ 方块（三指）/ 软球（包络），各 5 次；记录成功率与失败模式（够不着→调 scaling；捏不拢→dexpilot 参数；过冲→alpha/限速）。
5. **T5 长跑**：10 分钟连续 teleop，盯温度/丢帧/漂移。
6. **T6 左手**重复 T1–T4。

## 6. 触觉接入（可选增强，不阻塞验收）

- 最简：板载模式 30Hz —— `set_tactile_config_file(tactile.json: fps=30, infer_from_device=true)` + callback 记录 `F6`（前 3 维力向量）做抓握指示灯/日志。
- 180Hz：走 docker 配方（references/02 §4）；teleop 阶段仅在需要触觉数据集时启用。
- 注意 callback 在 SDK 线程触发：只入队，不在回调里做重活。
- 通道→手指对应（右 0–4，部署代码按 `4-ch` 反序读）**上真机用单指按压逐一验证**后写进 references/02。

## 7. 后续：装上 Vega（P4 预研项，本计划不交付）

- 机械：用 `with_flange` 法兰接口对接 Vega `L/R_hand_mount`；需要适配件设计/确认螺栓圆。
- 电气：Sharpa 是以太网设备，不走 Vega 的 EE 串行总线 → 沿臂走网线+供电（Vega 有 4×USB3.2 + 1 Ethernet，确认腕部可用电源）；docs.dexmate.ai 拿到后核对官方自定义 EE 指引。
- 控制：手指通路保持本计划不变（独立网络直达手）；手臂由 dexcontrol 流式 `set_joint_pos`（~100Hz）+ 腕 6DoF 来源（Pico/VR，见 roadmap 阻塞项 3）。
- 仿真先行：计划 02 §S4 的装配场景先验证可达性与自碰撞。

## 8. 风险

| 风险 | 缓解 |
|---|---|
| SDK 文件拷回后版本/固件不匹配 | 优先用 docker 镜像通路验证；`version-map.json` 核对 |
| 反推的 API 语义有偏差（coeff 含义、限位行为） | T1/T2 低速分级验证；PDF 到手逐项核对 §1.3 清单 |
| 20→60Hz 提频后抖振 | 保留 20Hz 配置开关；速度系数与 alpha 联动调 |
| 手套-主机-手三方时钟不齐 | 延迟统计用同主机单调钟；wuji sync_time 对齐手套侧 |
| 真机与仿真行为差异 | T2 回放同时投仿真对照（同一 qpos 流双端消费，天然 A/B） |

## 9. 里程碑

Bring-up（§2，0.5–1 天，卡在资料重拷）→ T1–T3（1 天）→ 安全链齐 + T4 抓取（1 天）→ T5/T6（0.5 天）。总计 ~3 天纯工作量。
