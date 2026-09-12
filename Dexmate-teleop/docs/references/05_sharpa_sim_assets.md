# Sharpa 仿真资产与 sharpa-rl-lab 解析

整理日期 2026-06-11。关节表/SDK 见 [02_sharpa_wave_hand.md](02_sharpa_wave_hand.md)。

## 1. sharpa-urdf-usd-xml（`/home/msc/luhr/magicsim/Sharpa/sharpa-urdf-usd-xml/wave_01/`）

每个模型同时给 **URDF / USD(.usda + configuration/ 模块) / MJCF(.xml)** 三种格式：

| 变体 | 说明 |
|---|---|
| `{left,right}_sharpa_wave` | 手掌根（`*_hand_C_MC`）为 root 的裸手 ← **retarget 用这个** |
| `*_with_wrist` | 含手腕运动学/碰撞 |
| `*_with_flange` | 以法兰为根 link ← **装机械臂末端用** |
| `sharpa_wave_float_base_urdf_usd/*_with_float_base[_with_flange/_with_wrist]` | 前缀 6 个基座关节：`{side}_{x,y,z}_joint`(prismatic) + `{side}_{roll,pitch,yaw}_joint`(continuous) ← **teleop 腕驱动用** |
| `dual_sharpa_wave*` | 双手（每手 +6 浮动 DOF） |
| `*_MITmode.usda` | 硬件 MIT（力控）模式对应动力学；README 建议 MIT 下控制频率 >80Hz |

- **mesh 引用为 `package://right_sharpa_wave/meshes/*.STL`（54 处）**：pinocchio 建运动学模型不受影响；SAPIEN/其他加载器需改写为相对路径或设 ROS 包路径。meshes/ 有 visual+collision STL 全套。
- USD 模块化：`<name>/configuration/{base,physics,robot,sensor}.usd`（几何/动力学/关节/接触传感分文件，主 .usda 引用）→ **拷贝 USD 时要连 configuration/ 目录一起拷**。
- MJCF：`compiler angle=radian, meshdir=meshes/`，含整定的 armature/frictionloss/damping/solref（elastomer 指尖做了软化处理）。
- 指尖三坐标系（每指）：`*_DP`（MDH）/ `*_elastomer`（触觉）/ `*_fingertip`（IK）。
- 还是 ROS 包（package.xml、display.launch.py，RViz2 展示用）。README 联系人 lei.su@sharpa.com。

## 2. sharpa-rl-lab（`/home/msc/luhr/magicsim/Sharpa/sharpa-rl-lab/`）

Isaac Lab（**release/2.2.0 或 2.3.0**，Ubuntu 22.04）上的 in-hand rotate 任务，pipeline：gen_grasp → train(PPO) → distill(ProprioAdapt) → play / deploy。

```
rl_isaaclab/
  tasks/inhand_rotate/
    sharpa_wave_env.py / sharpa_wave_env_cfg.py        # 训练环境（场景写法在这）
    sharpa_wave_grasp_env*.py                          # 抓取缓存生成
    sharpa_wave_deploy_env.py / *_cfg.py               # 真机部署（SDK 用法范本）
    agents/ppo_cfg.yaml
  scripts/{train,play,gen_grasp,deploy}.py
  utils/{keyboard_listener.py, misc.py, modified_events.py, docker-compose.yml}
  wrapper/{vec_env, sharpa_wave_env_wrapper, sharpa_wave_deploy_env_wrapper}.py
assets/{SharpaWave/right_sharpa_wave.usda, cylinder/, tactile_ha4_map/}
```

### 可直接抄的关键配置（sharpa_wave_env_cfg.py）

- **机器人**：`ArticulationCfg(usd_path="assets/SharpaWave/right_sharpa_wave.usda", actuators={"...": IdealPDActuatorCfg(joint_names_expr=[".*"], stiffness=None, damping=None)})` —— **刚度/阻尼留 None，用 USD 内预整定值**（urdf 仓库 README 也这么建议）。
- **仿真**：`dt=1/240`，`decimation=12`（控制 20Hz），`render_interval=2`；PhysX：solver_type=1、max_position_iter=8、`gpu_max_rigid_contact_count=2^23`；训练时重力 curriculum 从 -0.05 渐增。
- **动作**：22 维关节目标，`action_scale=1/24`，训练里走 torque_control=True 路径；`dof_limits_scale=0.9`。
- **接触/触觉**：每指 2 个 ContactSensor —— elastomer（track_contact_points=True，history 3，max 10 contacts/prim）+ DP（只过滤与物体的碰撞）；`enable_tactile=True`、`contact_threshold=0.05`、`contact_latency=0.005s`、噪声 0.01。
- **域随机化套路**：PD 增益 ×[0.5,2]、摩擦 ×[0.5,2]（elastomer 基础摩擦 0.8 / 金属 0.1）、COM ±1cm、质量 [0.01,0.25]kg、随机外力。
- 物体：`assets/cylinder/cylinder.usd`（r=24mm, h=60mm, 50g）。

### 部署侧（deploy_env*.py）——真机范本

- `control_freq=20Hz`（sleep 控速）；键盘 `e`=启动 / `w`=冻结 / `q`=回 home（`utils/keyboard_listener.py`）；warm_up 预热。
- 关节序换序：`utils/misc.py` 的 `_ISAACLAB2SHARPA_IDX / _SHARPA2ISAACLAB_IDX`（Isaac 按 BFS 顺序 ≠ SDK/URDF 顺序，**必须换序**）。
- 触觉模式：HostComputer(180Hz, docker+GPU) / OnBoard(30Hz)，docker compose 在 `utils/`，镜像 `sharpadev/sharpawave-rl-deploy:1.0.2-cu124`。
- 仓库内 **无任何 retarget/teleop/glove 代码**——teleop 桥要我们自己写。

## 3. sharpa-tacmap（`/home/msc/luhr/magicsim/Sharpa/sharpa-tacmap/`）

TacMap：基于"几何一致穿透深度图"的 VBTS 触觉仿真器（Isaac Lab 集成），240×240 形变图（resolution_step=2 时 120×120），含按压测试用例与可视化导出（arXiv:2602.21625）。→ P2 之后做触觉仿真时再接入，teleop 初期不需要。
