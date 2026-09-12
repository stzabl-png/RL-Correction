# MagicDexMate 总路线图

制定日期 2026-06-11。详细计划见同目录 01/02/03；事实依据见 ../docs/references/。

## 1. 整体架构

```
              ── 人侧 ──                          ── teleop-core 进程 (conda env `teleop`, py3.11) ──
 Wuji 手套(右/左) ─UDP@120Hz→ wuji-sdk ─ hand_skeleton(21,3) ─→ 帧约定转换 ─→ dex-retargeting ─ qpos(22) ─┐
                              └ IMU 腕姿态(quat)@800Hz ────────────────────────────────────────────────┤
 Pico/Quest VR (P4) ─────── 腕 6DoF + 底盘/躯干 ──→ omniteleop / dexcontrol ──────────────────────────┤
                                                                                                      ▼
                                                                          ┌─ ZMQ pub {qpos22, wrist_quat, t}
                  ┌───────────────────────────────────────────────────────┤
                  ▼                              ▼                        ▼
        SAPIEN 可视化 (P1 调试)      Isaac Lab 场景 (P2, env `isaaclab`)   SharpaWave 真机 (P3, sharpa SDK)
                                                                          Vega 手臂 (P4, dexcontrol ~100Hz)

 Vega lidar/IMU ─ dexcontrol-rosbridge → ROS2 Jazzy: cartographer(2D) / super-odom(3D+TEASER++重定位) (P5)
```

核心解耦决策：**retarget 与仿真/真机分进程**。dex-retargeting 需 numpy>=2.0，Isaac Lab 2.2/2.3 不兼容；且同一个 qpos 流要同时喂 sim 与 real。统一用 ZMQ PUB/SUB（后续如需与 Dexmate 栈统一可换 dexcomm/Zenoh，接口先抽象）。

## 2. 阶段划分

| 阶段 | 内容 | 交付 | 依赖/阻塞 | 计划 |
|---|---|---|---|---|
| **P0 环境与资产** | conda env `teleop`；拷贝/修正 Sharpa URDF 资产；手套连通 | `import` 全通过；`ping` 手套；pinocchio 载入 URDF | 手套到位 | 01 §R0 |
| **P1 Retarget + 可视化** | Wuji→Sharpa retarget（vector/dexpilot）+ SAPIEN 实时可视化 + MCAP 离线回放 | 实时跟手 ≥60Hz，捏合自然 | P0 | **01** |
| **P2 Isaac 场景** | Isaac Lab teleop 场景（固定基→浮动基→双手），桌面+物体，接 ZMQ | 手在场景里跟手套，接触稳定 | P1 | **02** |
| **P3 真机手 teleop** | SharpaPilot 标定 + SDK 控制回路 + 安全机制 + 抓取测试 | 真机跟手，抓起 3 类物体 | **Sharpa SDK/PDF 重拷**（全 0 字节） | **03** |
| **P4 Vega 手臂 + Pico** | 腕 6DoF 来源（VR）+ dexcontrol 手臂流控 + Sharpa 装机 | 全身 real teleop：移动+臂+手 | docs.dexmate.ai 账号；**Pico 兼容性确认**（omniteleop 面向 Quest）；法兰机械适配 | 03 §7 起步，届时单独出计划 |
| **P5 SLAM** | rosbridge→cartographer(2D)/super-odom(3D)，建图+重定位 | 室内图 + 定位跑通 | ROS2 Jazzy 环境；P4 不阻塞它（可并行） | 届时单独出计划 |
| **P6 数据录制** | 手套 MCAP + omniteleop MDP recorder + 触觉(F6/RAW) 对齐时间戳 | 可回放的演示数据集 | P3/P4 | 后续 |

## 3. 阻塞项（按优先级）

1. **重新拷贝 Sharpa 资料**：`MagicTactile/UserManual/` 4 份 PDF + `/home/msc/sharpa/` SDK 树（SharpaWaveSDK_4.6.6、sharpa-pilot deb、DockerEnv）全为 0 字节。验证：`find <dir> -type f -size 0 | wc -l` 应为 0。不阻塞 P0–P2（API 已从 sharpa-rl-lab 反推），**阻塞 P3**。
2. **docs.dexmate.ai 登录**：需要账号才能看 Vega 官方 Pico/导航文档。阻塞 P4 设计定稿。
3. **Pico vs Quest**：omniteleop VR 代码是 Quest 专用写法；若必须 Pico，要么向 Dexmate 要支持，要么评估第三方 teleop_xr（WebXR）。

## 4. 决策记录

| 日期 | 决策 | 理由 |
|---|---|---|
| 2026-06-11 | retarget 用 dex-retargeting 的 vector（起步）+ dexpilot（捏合优化），不自写优化器 | 库成熟、Wuji skeleton 即 MediaPipe 21 点、官方 wuji-retargeting 同路线可参考 |
| 2026-06-11 | 不 fork dex-retargeting；Sharpa yaml/urdf 放本仓库，`load_from_file`+`set_default_urdf_dir` 加载 | 升级无负担；constants.py 注册只是便利函数非必需 |
| 2026-06-11 | retarget 进程与 Isaac/真机分离，ZMQ 桥接（模式 A 默认） | 一份 qpos 流多端消费；崩溃隔离 |
| 2026-06-11 | **修正**：dex-retargeting 的 numpy>=2 pin 实测为纯声明（numpy1.26+pin2.7 结果逐位一致）⇒ 另提供**单进程模式 B**（sim/teleop_isaac_single.py，install_into_isaaclab.sh 装栈） | 用户要求单进程；env 内直接拿 qpos 便于采数据 |
| 2026-06-11 | retarget 用裸手 URDF（root=`*_hand_C_MC`），sim 腕驱动用 float_base 变体，装臂用 with_flange | 各变体各司其职，见 references/05 |
| 2026-06-11 | Isaac 手部执行器用 `IdealPDActuatorCfg(stiffness=None, damping=None)` | 用 USD 预整定动力学（厂商+rl-lab 双重背书） |

## 5. 立即可做的下一步

1. P0：建 `teleop` env，`pip install wuji-sdk` + `pip install -e third_party/dex-retargeting`（已 vendored 入库），跑 01 §R0 资产脚本。
2. 催办阻塞项 1（重拷 Sharpa 资料）和 2（docs.dexmate.ai 账号）。
3. 手套到手后：录第一批 MCAP（张开/握拳/逐指捏合/数数），P1 全程靠它离线开发。
