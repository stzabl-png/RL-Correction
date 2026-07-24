"""SharpaCorrectionEnv 配置 — 底座抄冠军 SharpaWaveEnvCfg (已验证的物理参数),
改动: 28 维残差动作 / 桌子 / clip 物体 / 腕 wrench-PD 增益 / 参考数据路径."""
from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from rl_rebuild.correction.load_replay import CLIP11, CLIP11_MESH

_ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../assets")


@configclass
class SharpaCorrectionEnvCfg(DirectRLEnvCfg):
    # ---- env ----
    episode_length_s = 8.0            # 上限; 实际由参考长度+hold 提前截断
    action_space = 28                 # 腕Δpos3+Δ轴角3 + 手指Δq22
    # ---- 物体几何观测: mesh 表面 FPS 点云 (腕系), 展平进 obs. 两 variant 共用 ----
    # 物体几何点云: 走 PointNet 分支 (置换不变 + 逐点 MLP), 不再展平进 obs.
    # 展平的问题: 128x3=384 维占满第一层、对点序敏感、无局部性先验.
    # PointNet 把它压成 pc_feat_dim(128) 维特征再拼进 actor 输入, 为多物体泛化做准备.
    # 点数直接决定 PointNet 逐点激活的显存: minibatch16384 x 128点 x 128feat x 4B = 1.07GB/层,
    # 前向 6 层都要留给反向 -> 128 点时峰值 ~20GB 会 OOM. 64 点对单物体形状表达足够.
    n_obj_points = 64                 # 0=关闭
    enable_pointcloud = True                  # True -> 单独走 obs_dict["pointcloud"] (N,P,3) + PointNet
    pc_in_dim = 3                     # 每点维度 (xyz; 旧默认 5 = xyz+mask)
    pc_feat_dim = 64                  # PointNet 输出特征维 (拼进 actor 输入)
    observation_space = 144           # 本体44+腕13+物体13+参考37+相位4+接触5+动作28 (点云不在此列)
    state_space = 0                   # 特权走 HORA 式 priv_info 通道 (v3net separate_critic 消费)
    priv_info_dim = 7                 # 质量1 + 摩擦1 + 指尖接触力5
    prop_hist_len = 8                 # 本体历史帧数 (PPO 不用; stage-2 ProprioAdapt 用)
    clip_actions = 1.0
    clip_obs = 5.0                    # 观测截断 (冠军同款)
    obs_vel_scale = 0.2               # 速度类观测缩放 (Allegro demo 同款)
    lookahead = 5                     # 参考前瞻帧数 (0.25s @20Hz)
    decimation = 12                   # 1/240 × 12 = 20 Hz 控制, 与参考重采样一致
    # ---- 残差幅度界 ----
    # 2026-07-21 phys_ablation 实测: 零残差回放本身能抓 (53% 时间 >=2 指接触, 抬 6.8cm,
    # return +199), 但探索噪声按残差界缩放后会彻底摧毁抓握 —— sigma=0.05 时接触掉到 3.6%,
    # 训练实测 sigma=0.76 (读 last.pth) 时接触 0.1%、物体被打飞 37.7cm (= term/obj_div=1.0).
    # 容忍度是毫米级, 故按 "sigma=0.1 时每步扰动 ~2mm / <1度" 重新定界.
    wrist_pos_residual_max = 0.02     # m    (旧 0.15: sigma 0.76 -> 每步 ±11.4cm 抖动)
    wrist_rot_residual_max = 0.05     # rad  (旧 0.3)
    finger_residual_max = 0.10        # rad  (旧 0.3; 保留 ±5.7° 握紧权限, 手指不推飞物体)
    # ---- 腕 wrench-PD 增益 (M0 标定对象) ----
    wrist_kp_pos = 3000.0
    wrist_kd_pos = 120.0
    wrist_kp_rot = 12.0
    wrist_kd_rot = 0.6
    # ---- hold 尾巴 (参考播完后保持的控制步数; 成功判据 2s@20Hz) ----
    hold_steps = 40
    # ---- 静置前奏/冻结窗: episode 开头 settle_steps 步内物体钉在参考位(桌上或手里),
    # 手不施加残差, 让抓握/沉降稳定后再释放 (kinematic-freeze) ----
    settle_steps = 20
    # ---- DeepMimic RSI: 按此概率从"接触+抬升"参考帧起步(而非总从t0), 让策略先见到成功态.
    # 起步帧从 [interaction起, L-1] 采样(覆盖 grasp→lift→carry); 物体初始化到该帧参考位.
    # rsi_prob 是初值; 训练中 PPO hook 按步数课程退火到 0 (逼策略最终学会"从头抓").
    # 评测真实从头成功率用 --rsi_prob 0. ----
    rsi_prob = 0.8
    rsi_mode = "linear"               # 退火方式: "linear"=按步数线性 / "success"=成功率门控棘轮
    # -- linear 模式参数 --
    rsi_warmup_steps = 10_000_000     # 前期保持初值, 让策略充分见成功态
    rsi_decay_steps = 60_000_000      # 之后线性退火到 0 (到 warmup+decay 步时 rsi_prob=0)
    # -- success 模式参数: 当前 RSI 下 sr_ema≥gate(基本能抓)就降 step, 只降不升, 到 0 --
    rsi_gate_sr = 0.5                 # 触发降低的成功率门槛
    rsi_step = 0.005                  # 每次(门槛达标的迭代)降低量
    # ---- simulation: 冠军参数 + 正常重力 (桌面任务, 物体要受重力) ----
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 240,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=8,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=5 * 2**18,
        ),
    )
    # ---- 手: 冠军资产, 浮动根 (wrench-PD 直接推根刚体), 手指隐式 PD ----
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(_ASSETS, "SharpaWave/right_sharpa_wave.usda"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,          # 飞手不受重力, wrench 只管跟踪
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=64 / math.pi * 180.0,
                max_depenetration_velocity=1000.0,
                max_contact_impulse=1e32,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=False,           # 冠军 USD 根是固定的; 飞手必须浮动根
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.3, 1.2)),  # 占位, reset 时写参考
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=20.0, damping=2.0,   # retarget_isaacsim 回放验证过的手指增益
            ),
        },
    )
    # ---- 物体: clip 的 USD + 人工语义物理参数 (塑料圆柱 0.2kg) ----
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{CLIP11}/object.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.97)),  # 占位
    )
    # ---- 桌子: 静态碰撞体, 桌面 z=0.85 (与数据对齐约定一致) ----
    table_size = (1.2, 1.2, 0.04)
    table_top_z = 0.85
    # ---- scene ----
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8, env_spacing=2.0, replicate_physics=False)  # 冠军同款: fabric克隆会破坏接触传感器的prim视图
    # ---- 指尖接触传感器 (grip 门控与 contact shaping 的数据源) ----
    fingertip_bodies = ["right_thumb_elastomer", "right_index_elastomer",
                        "right_middle_elastomer", "right_ring_elastomer",
                        "right_pinky_elastomer"]
    contact_sensors = [
        ContactSensorCfg(
            prim_path=f"/World/envs/env_.*/Robot/{name}",
            history_length=1,
            filter_prim_paths_expr=["/World/envs/env_.*/Object"],  # 只统计与物体的接触
        ) for name in fingertip_bodies
    ]
    contact_force_thresh = 0.1        # N; 超过算"接触" (冠军 held 门槛同源: 0.15 量级)
    # ---- 终止阈值 ----
    term_obj_fall = 0.05              # 物体低于桌面此深度 => 掉落终止
    term_obj_div = 0.40               # 物体偏离参考 (m); 零残差基线峰值 0.34, 阈值须在其上
    term_wrist_div = 0.50             # 腕偏离参考 (m); 控制爆炸保险
    success_hold_frac = 0.95          # hold 段内 (抬≥10cm 且 ≥2 指接触) 的步数占比达标线
    # ---- 简化抓取模式 (单物体 MVP) ----
    # episode = [静置 | RL抓取窗口 | 硬编码抬升 | hold], 不再回放整条参考.
    # 抓取窗口 = cuRobo 的 close 起点(PreGrasp) -> squeeze 末帧, 即 GraspPose 合拢那一段.
    # 抬升是确定性的垂直斜坡 -> "抓好没有" 变成非黑即白的测试 (提上去物体跟不跟得上).
    grasp_only = True
    lift_height = 0.10                # m, 硬编码抬升高度 (也是成功判据的高度门槛)
    lift_steps = 20                   # 抬升斜坡步数 (1s @20Hz)
    # 成功高度门槛 < 指令抬升高度: 腕升 10cm 时物体必然因手指柔顺/微滑略低于 10cm,
    # 要求"一点不滑"是不合理的严苛判据 (实测零残差 0/64 达 9cm). 取 80%.
    success_lift_height = 0.08
    # ---- affordance 指尖抓取 (小/扁物体; 仅 clip 挂了 affordance 时生效) ----
    lam_afford = 0.4                  # 指尖落高 affordance 点的奖励权重
    loose_grip_afford = 0.05          # affordance clip 的松握折扣 (压 cage/scoop, 逼指尖抓)
    # ---- 数据先验开关 (消融实验: Exp1 两个都开, Exp2 都关) ----
    use_grasp_prior = False           # 抓握段手指模仿目标: True=GraspPose纯抓姿 / False=重建手指
    use_curobo_guide = False          # cuRobo 预抓取路点引导 (退火, 初值由 train.py 置)
    lam_curobo_init = 0.5             # use_curobo_guide 时 lam_curobo 初值 (PPO 按 sr_ema 退火)
    curobo_wean_sr = 0.25             # sr_ema 达此值时 lam_curobo 退到 0
    curobo_guide_frac = 0.4           # cuRobo 引导时间窗: active 段前此比例步 (初始阶段指引)
    # ---- 参考数据 (clips.py 注册表按 clip_name 解析; configure_cfg 会改 object usd 路径) ----
    clip_name = "clip11"
    target_hz = 20.0                  # = 1 / (dt*decimation)
