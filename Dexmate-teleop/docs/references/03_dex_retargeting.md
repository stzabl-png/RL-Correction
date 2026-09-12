# dex-retargeting 库参考笔记

源码：`/home/msc/luhr/magicdexmate_Yanghong/dex-retargeting`（v0.5.0，MIT，AnyTeleop 衍生，作者 Yuzhe Qin）。整理日期 2026-06-11。

## 1. 概要

把人手姿态（MediaPipe 21 关键点约定）经非线性优化（NLopt SLSQP + Huber loss + 关节限位）映射为机器人手关节角。三种模式：

| type | 用途 | 优化目标 |
|---|---|---|
| `vector` | teleop | 人手"origin→task"3D 向量 vs 机器人对应 link 向量 |
| `dexpilot` | teleop（带捏合先验） | 指间/指-腕向量 + 捏合投影逻辑（`project_dist`/`escape_dist`） |
| `position` | 离线数据处理 | link 位置直接匹配（可加 6D 自由根关节 `add_dummy_free_joint`） |

内置手型：allegro / shadow / svh / leap / ability / inspire / panda（各左右）。**repo 内无 "sharpa" 字样**——需要我们自己加配置。

## 2. 核心 API

```python
from dex_retargeting.retargeting_config import RetargetingConfig

RetargetingConfig.set_default_urdf_dir("<assets/robots/hands 的绝对路径>")  # 必须先调
config = RetargetingConfig.load_from_file("path/to/xxx.yml")               # 或 from_dict
retargeting = config.build()            # -> SeqRetargeting

qpos = retargeting.retarget(ref_value)  # -> (M,) 关节角
names = retargeting.joint_names          # qpos 的关节名顺序（≠ pinocchio 顺序！）
```

- `ref_value`：vector/dexpilot 为 (K,3) 向量数组；position 为 (N,3) 位置数组。
- vector 模式取法：`indices = retargeting.optimizer.target_link_human_indices`（2×K）；`ref_value = joint_pos[indices[1]] - joint_pos[indices[0]]`，其中 `joint_pos` 为 (21,3) 人手关键点。
- **输出关节序必须按名字重映射**到消费端顺序（SAPIEN/Isaac/SDK）：

```python
idx = np.array([retargeting.joint_names.index(n) for n in consumer_joint_names])
consumer_qpos = qpos[idx]
```

- 其他：`warm_start(wrist_pos, wrist_quat, ...)`（position+自由根用）、`set_qpos/get_qpos`；URDF 有 mimic 关节时自动用 `MimicJointKinematicAdaptor`（Sharpa URDF 无 mimic，不涉及）；`low_pass_alpha`∈(0,1] 启用一阶低通（越小越平滑、延迟越大）。

## 3. 人手输入约定

MediaPipe 21 点：0=wrist，1–4=拇指，5–8=食指，9–12=中指，13–16=无名指，17–20=小指（每指 MCP/PIP/DIP/TIP，TIP=4/8/12/16/20）。

库内 MediaPipe 摄像头管线（`example/vector_retargeting/single_hand_detector.py`）：

```python
keypoint_3d_array -= keypoint_3d_array[0:1, :]                       # 平移到腕
wrist_rot = estimate_frame_from_hand_points(keypoint_3d_array)       # 用 0,5,9 点估计腕系
joint_pos = keypoint_3d_array @ wrist_rot @ operator2mano            # 转 MANO 约定
```

`OPERATOR2MANO_RIGHT/LEFT` 在 `constants.py`。**要点：retarget() 期望的人手关键点是"腕原点 + MANO 朝向约定"的 (21,3)**。Wuji 手套的 skeleton 已在腕系，只需再乘一个固定旋转（见 plans/01 §R2）。

## 4. YAML 配置格式

目录：`src/dex_retargeting/configs/{teleop,offline}/`（40 个现成文件可抄）。

vector 示例（allegro_hand_right.yml）：

```yaml
retargeting:
  type: vector
  urdf_path: allegro_hand/allegro_hand_right.urdf      # 相对 default_urdf_dir
  target_origin_link_names: [ "wrist", "wrist", "wrist", "wrist" ]
  target_task_link_names: [ "link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip" ]
  target_link_human_indices: [ [ 0, 0, 0, 0 ], [ 4, 8, 12, 16 ] ]   # [origin 行; task 行]
  scaling_factor: 1.6        # 机器手/人手 尺度比（allegro 大手→1.6；shadow 1.2）
  low_pass_alpha: 0.2
```

dexpilot 示例（shadow_hand_right_dexpilot.yml）：

```yaml
retargeting:
  type: DexPilot
  urdf_path: shadow_hand/shadow_hand_right.urdf
  wrist_link_name: "ee_link"
  finger_tip_link_names: [ "thtip", "fftip", "mftip", "rftip", "lftip" ]   # 拇指在前
  scaling_factor: 1.2
  low_pass_alpha: 0.2
```

dexpilot 的 `target_link_human_indices` 留空自动生成（按 5 指 tip + wrist 组合）。

## 5. 新手型接入步骤（适用于 Sharpa）

1. URDF 放到 `<urdf_dir>/<hand>/<hand>_<side>.urdf`（pinocchio 只建运动学模型，mesh 路径不影响 retarget；但 SAPIEN 可视化需要 mesh 可解析）。
2. 写 vector / dexpilot yaml（可放在我们自己仓库，用 `load_from_file` 绝对路径加载，**无需 fork** 或改 `constants.py`；改 constants 只是为了 `get_default_config_path` 便利函数）。
3. 选 link：origin=腕/根 link，task=各指尖 link；`target_link_human_indices` 对应 MediaPipe 编号。
4. 估 `scaling_factor`：FK 算零位/张开位的 腕→中指尖 距离，除以人手实测（约 0.17–0.19m）。
5. 验证：`config.build()` 成功、`retargeting.joint_names` 数量正确、随机 ref_value 优化残差小。

## 5.5 实测结论与已知问题（2026-06-11，Sharpa 接入时验证）

- **MANO 约定实测**（mock 张开手经 `to_mano` 后）：四指尖沿 **+z**（中指尖 z≈0.195），拇指偏 **+y**（桡侧）。**Sharpa URDF 零位与之高度吻合**（C_MC 系下 `right_middle_fingertip`=(0,0.01,0.203)，拇指尖 y=0.123）——所以 vector retarget 不需要额外的基座旋转，开箱即用。
- **顺序跟踪（teleop 真实工况，warm-start 60Hz）**：四指尖向量误差均值 7-11mm，拇指 ~17mm（mock 拇指几何粗糙所致，真手套数据后在 R5 再调）。单步求解 p50≈2.7ms（i7 CPU，~200Hz 上限）。
- **已知异常（不影响 teleop）**：从随机极端位姿单步大跳收敛差——20 次随机位姿测试 19 次停在 ~52mm 的驻点；解析梯度已用有限差分验证正确；nlopt 2.8.0/2.10.0 结果完全一致；库自带的 allegro/shadow 同方法测试在同一依赖栈全过 ⇒ Sharpa 构型特有的优化景观问题。还观察到 NLopt SLSQP 在初值≈最优时会返回**从未评估过的点**（`last_optimum_value` 与重算值不符）。teleop 是小步连续跟踪 + low_pass filter + 限位 clip，不会进入该工况；离线批处理若要从任意位姿单步求解需注意。
- 验证所用依赖栈：numpy 2.4.6 / pin 4.0.0 / nlopt 2.10.0 / torch 2.12-cpu / python 3.11（uv venv）。

## 6. 环境约束与示例

- 声明依赖：numpy>=2.0、pin>=3.3.1、nlopt>=2.8、pytransform3d、anytree、**torch（import 时强制检查，cpu 版即可）**；Python >=3.7,<3.13。
- **numpy>=2.0 的 pin 是纯声明性的（2026-06-11 实证）**：源码只用 `npt.NDArray` 注解，无 numpy-2 专属 API。在 **numpy 1.26.0 + pin 2.7.0**（pip/uv 在 numpy1.26 约束下解析出的组合；pin 3/4 的 cmeel 依赖才要 numpy>=2）上跑出与 numpy 2.4.6 + pin 4.0 **逐位一致**的跟踪结果（[16.6 8.3 11.1 7.4 10.2] mm），且更快（单步 2.0ms vs 2.7ms）。安装法：依赖用 `pip install -c <numpy==当前版本> nlopt pin pytransform3d anytree pyyaml lxml`，本库 `pip install --no-deps -e .`——已封装为 `scripts/install_into_isaaclab.sh`，可直接装进 Isaac Sim/Lab 的 python（本机 ~/isaacsim：py3.11/numpy1.26/torch2.7-cu128）。
- 因此与 Isaac Lab **并非硬冲突**：双进程桥接（模式 A）是架构选择（一流多端/隔离），单进程内嵌（模式 B，`sim/teleop_isaac_single.py`）同样成立。
- 安装：`pip install -e /home/msc/luhr/magicdexmate_Yanghong/dex-retargeting`（example extras 另含 mediapipe、sapien==3.0.0b0、opencv、tyro、loguru）。
- `assets/` 是 dex-urdf 子模块，**当前未初始化（空）**——我们用自己的 urdf_dir，不依赖它。
- 示例：`example/vector_retargeting/`（`detect_from_video.py` 离线、`show_realtime_retargeting.py` 实时 SAPIEN 可视化——抄它的 detect→retarget→换序→`robot.set_qpos` 循环）、`example/position_retargeting/`、profiling 示例（单次 retarget 量级 ms 级）。
- 测试：`tests/test_optimizer.py` 展示了"随机 qpos→FK 生成 ref→retarget→误差<1e-2"的验证套路，可复用于 Sharpa 配置自检。
