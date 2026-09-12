# 计划 02：Isaac Lab Teleop 场景搭建

对应路线图 P2。参考 sharpa-rl-lab 的现成写法（references/05），输入来自计划 01 的 ZMQ qpos 流。

## 0. 目标与验收

在 Isaac Lab（release/2.2.0 或 2.3.0，独立 conda env `isaaclab`）里搭一个 teleop 场景：SharpaWave 手 + 桌面 + 可抓物体，手指由手套实时驱动，腕部可控，接触物理稳定。

验收：
- S1：固定基座右手在场景中跟随手套，控制 ≥30Hz，视觉无抖动/穿透爆炸。
- S2：浮动基座腕姿态跟随手套 IMU，能用手指把桌上圆柱拨动/扣住。
- S3（可选）：双手场景。
- S4（后续）：Vega 全身 + Sharpa 法兰装配场景（出装配验证截图即可，控制留到 P4）。

## 1. 分层策略（先快后全）

1. **SAPIEN 冒烟**（已含在计划 01 的 M2）：验证 retarget 正确性，不做物理交互。
2. **MuJoCo 旁路（可选）**：`sharpa-urdf-usd-xml` 的 MJCF 带整定接触参数，`mujoco.viewer` 三十行脚本即可做第一次"有物理"的抓握 sanity check，不依赖 Isaac 环境。
3. **Isaac Lab 正式场景**（本计划主体）：用厂商预整定 USD + rl-lab 的参数组合，后续可平滑接 RL/触觉/Vega。

## 2. 直接复用的资产与参数（来源 references/05）

| 项 | 取值/来源 |
|---|---|
| 手 USD（S1 固定基） | `sharpa-rl-lab/assets/SharpaWave/right_sharpa_wave.usda`（**连同旁边的 configuration/ 目录一起拷**）或 urdf 仓库同名 .usda |
| 手 USD（S2 浮动基） | `sharpa-urdf-usd-xml/wave_01/sharpa_wave_float_base_urdf_usd/right_sharpa_wave_with_float_base/right_sharpa_wave_with_float_base.usda`；基座 6 关节 `right_{x,y,z}_joint`(prismatic) + `right_{roll,pitch,yaw}_joint`(continuous) |
| 执行器 | `IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)` —— 用 USD 预整定值，**不要自己拍 PD** |
| 仿真步长 | `dt=1/240`；teleop 控制 60Hz → `decimation=4`（rl-lab 训练用 12→20Hz，teleop 提高） |
| PhysX | solver_type=1、max_position_iteration_count=8、bounce_threshold_velocity=0.2（照抄 rl-lab） |
| 物体 | `sharpa-rl-lab/assets/cylinder/cylinder.usd`（r24mm h60mm 50g）起步，后加方块/球 |
| 限位安全系数 | 关节目标 clip 到限位 ×0.9（与真机一致，行为对齐） |
| 关节序换序 | Isaac 关节序（BFS）≠ SDK/URDF 序：**按名字算 index 映射**，写法抄 `sharpa-rl-lab/rl_isaaclab/utils/misc.py` |

## 3. 场景组成（S1）

```
/World
 ├─ ground (GroundPlaneCfg) + DomeLight
 ├─ Table: 固定 cuboid (0.8×0.8×0.04, 摩擦 0.7)，台面 z=0.40
 ├─ Hand:  right_sharpa_wave.usda，初始位姿掌心朝下、悬于台面上方 ~0.15m
 └─ Object: cylinder.usd 置于台面
```

`sim/teleop_scene_cfg.py` 骨架：

```python
@configclass
class TeleopSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    table = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(size=(0.8, 0.8, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True), ...),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.40)))
    obj = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ASSETS}/cylinder/cylinder.usd"), ...)
    hand = ArticulationCfg(prim_path="{ENV_REGEX_NS}/Hand",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ASSETS}/SharpaWave/right_sharpa_wave.usda",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.55), rot=PALM_DOWN_QUAT),
        actuators={"fingers": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)})
```

## 4. Teleop 桥（standalone 脚本，不走 RL env 框架）

`sim/teleop_isaac.py`，按 Isaac Lab standalone workflow（AppLauncher → InteractiveScene → 手写 while 循环，参照 rl-lab `scripts/play.py` 的启动样板）：

```python
sub = zmq_sub("tcp://teleop-host:5556")          # CONFLATE=1 只留最新
idx = name_map(hand.joint_names, SDK22_NAMES)    # 预计算换序
while simulation_app.is_running():
    msg = sub.poll_latest()                      # 非阻塞；None 则沿用上一目标
    if msg and msg.valid:
        target_sdk = clip(msg.qpos, LIMITS_09)
        hand.set_joint_position_target(torch.as_tensor(target_sdk)[idx])
    scene.write_data_to_sim(); sim.step(); scene.update(sim_dt)
```

- 启动顺序无关：桥起来后没消息就保持初始姿态。
- 消费端二次低通可选（源头 alpha=0.2 已滤）。
- hello 消息校验 joint_names 哈希（计划 01 §R4）。

## 5. S2 腕部驱动（浮动基座）

- 换 float_base USD，`fix_root_link=True`（浮动由 6 个显式关节实现，根仍固定）。
- 基座 6 关节单独一组 actuator；姿态目标 = 手套 IMU quat → 欧拉(rpy)；位置目标先固定/键盘微调（手套无腕位置，P4 接 VR 后替换）。continuous 关节注意角度展开（unwrap）防跳变。
- 备选方案：固定基 + `write_root_pose_to_sim` 每步硬设根位姿（运动学瞬移）。实现最快但接触动力学差（穿透冲击大），只作 fallback。
- PD：基座关节 USD 里若无预整定值则需自己给（位置环刚度大、姿态环适中），这是 S2 主要调参点。

## 6. S3/S4/S5 扩展

- **S3 双手**：用 `dual_sharpa_wave` USD（每手自带 6 浮动 DOF），或两个独立 Articulation；ZMQ 消息已带 hand 字段，桥侧路由即可。
- **S4 Vega 装配**：~~dexmate_urdf 自行装配~~ **已被 MagicSim 现成资产取代并实现**（2026-06-12）：`MagicSim/Assets/Robots/vega_1p_sharpa.usd`（67 关节整机）+ `vega1psharpa.py` 的实测增益 → 本仓库 `sim/vega_sharpa_scene.py` + `sim/teleop_vega_sharpa.py`（手指 retarget + 腕部 rpy 姿态 DiffIK，headless 冒烟通过）。后续：左手/双手套、底盘+躯干通道（MagicSim 有 holonomic action 与 pink/curobo IK 配置可抄）。
- **S5 触觉**：照抄 rl-lab 的双 ContactSensor 配置（elastomer track_contact_points + DP 过滤物体），teleop HUD 显示 5 指接触力；高保真 VBTS 用 sharpa-tacmap，留到需要触觉数据时再接。

## 7. 目录与运行

```
MagicDexMate/sim/
  teleop_scene_cfg.py  teleop_isaac.py  assets/   # USD 拷贝（带 configuration/）
# 运行（env isaaclab）：
python sim/teleop_isaac.py --hand right --sub tcp://localhost:5556 [--float-base] [--headless --livestream 2]
```

## 8. 风险

| 风险 | 缓解 |
|---|---|
| USD 的 configuration/ 相对引用断 | 整目录拷贝；`omni.usd` 打开后检查无 unresolved reference |
| Isaac 关节序错配 | 只按名字映射 + hello 哈希校验（双保险） |
| teleop 60Hz 下预整定 PD 抖动（整定面向 20Hz RL） | 先 60Hz 观察；不行回 decimation=12 的 20Hz（真机也是 20Hz 起步，行为一致） |
| float_base continuous 关节角跳变 | rpy unwrap + 目标限速（每步 Δ≤0.1rad） |
| Isaac Lab 版本漂移 | 锁 release/2.2.0 或 2.3.0（rl-lab 验证过的两个） |

## 9. 里程碑

S1 固定基跟手（2 天，含环境装机）→ S2 浮动基腕驱动（1–2 天）→ S3 双手（0.5 天）→ S4 Vega 装配验证（2 天，可后置到 P4 前）。
