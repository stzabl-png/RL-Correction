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
    # ---- 手指残差累积 (rate 与 total 分离) ----
    # 不累积时 finger_residual_max 一个数同时当"每步扰动界"和"总权限界", 而这两个需求
    # 方向相反, 一个数满足不了:
    #   每步要**小**: sigma × 界 = 每步扰动, 大了指尖一巴掌把物体扇飞 (上面那段实测)
    #   总量要**大**: calib_finger_geom 实测指尖要真碰到物体需 28~59° (中位 45°),
    #                 5.7° 是结构性够不着 —— 不是学不会, 是动不了那么多
    # 累积模式把两者拆开: 每步只准动 finger_rate_max, 但残差在步间**累加**, 上限
    # finger_total_max. 与臂的 accumulate_action 同构, 同样保持"零动作 = 精确跟参考".
    # ⚠ 探索噪声在累积下是随机游走 (T 步后 ≈ sigma·rate·√T), total 钳制不是可选项,
    #   是唯一的安全带; finger_res 也必须进观测, 否则策略看不见自己积了多少 (非马尔可夫).
    accumulate_finger = False
    finger_rate_max = 0.10            # rad/步; 累积模式下 res_scale 的手指段用它
                                      # (= 旧 finger_residual_max, 那是标定过的"每步安全量")
    finger_total_max = 1.2            # rad (69°); 累积残差的绝对上限 = 几何下界 45° × 1.5
    finger_res_decay = 1.0            # 漏积分系数: <1 时残差每步自行向参考回落 (1.0 = 纯积分)
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
    # ---- 桌子: 静态碰撞体 (2026-08-26 Pour 新场景: 黑色 6ft x 4ft, 桌面 87cm) ----
    table_size = (1.2192, 1.8288, 0.04)   # 深4ft x 宽6ft: 长边横在机器人面前, 机器人位于长边中点
    table_top_z = 0.87
    # ---- DexMate(Vega) 视觉参考 ----
    # 只为在 GUI 里核对工作区尺度 (飞手 MVP 的腕位必须落在 Vega 够得到的范围内,
    # 见 docs/DEXMATE_WORKSPACE.md). **不参与控制、不参与训练**: 挂在 /World/Dexmate,
    # 在 env 命名空间之外, clone_environments 不会复制它.
    # 默认关闭; 查看器用 SHOW_DEXMATE=1 打开 (训练脚本不受影响).
    show_dexmate = os.environ.get("SHOW_DEXMATE", "0") == "1"
    dexmate_usd = os.path.join(
        os.environ.get("MAGICSIM_ASSETS", "/home/lyh/luhr/MagicSim/Assets"),
        "Robots", "vega_1p_sharpa.usd")
    dexmate_pos = (-0.5, 0.0, 0.0)    # 底座在地面 (z=0); 从 -0.8 往 +X 前进 0.3
    dexmate_quat = (1.0, 0.0, 0.0, 0.0)   # wxyz 单位四元数 = 面朝 +X (Vega 原生朝向)
    # ---- 双手轨迹摆放预览 (查看器专用, SHOW_HUMAN_TRAJ=1 打开) ----
    # 以交互起始帧为基准, 求让机器人左右手最均衡的 XY 平移; 物体落到交互手掌心、Z 贴桌.
    # 细线画出双手腕轨迹. 详见 bimanual_align.py
    show_human_traj = os.environ.get("SHOW_HUMAN_TRAJ", "0") == "1"
    align_frame = 0                   # 双手刚体对齐基准帧 = 轨迹第一帧
    obj_frame = 34                    # 物体摆放基准帧 = 交互起始帧
    interact_hand = "right"           # phase_left 全 0, 只有右手交互
    traj_width = 0.004                # 细线宽度 (m)
    # 抓取帧掌心距物体**中心**的目标距离. Z 平移量由它反算, 所以调它 = 整体升降
    # 人手轨迹/红球/橙球/蓝球(相机), 物体不动(物体独立贴桌).
    grasp_gap = 0.045
    # 头->头 XY 锚定用哪个 link 当"机器人的头".
    #   vega_1p_head_l3  = 头本体 link 原点 (视觉上和头几何重合)
    #   zed_mid          = ZED 双目中点 (光心, 在头前方约 6.1cm)
    # 选 head_l3 会让整体(轨迹+物体)相对 ZED 方案后移约 6cm.
    head_anchor = "vega_1p_head_l3"
    # 默认姿势: 关节名 -> **度** (prismatic 关节则是米). 建场景时就摆好 (InitialStateCfg),
    # 所以查看器重建后姿态不会丢. 全部可控名见 DEXMATE_JOINTS.md (67 个).
    # 2026-07-25 手调的对称站姿:
    dexmate_joints = {
        # 2026-08-26 Pour 站姿: 0.7072/1.2856/0.0068 rad
        "torso_j1": 40.5196,
        "torso_j2": 73.6595,
        "torso_j3": 0.3896,
        "L_arm_j1": 45.0,  "R_arm_j1": -45.0,     # 肩 pitch, 左右对称
        "L_arm_j2": 0.0,   "R_arm_j2": 0.0,
        "L_arm_j3": 0.0,   "R_arm_j3": 0.0,
        "L_arm_j4": -90.0, "R_arm_j4": -90.0,     # 肘同向弯 90°
        "L_arm_j5": 0.0,   "R_arm_j5": 0.0,
        "L_arm_j6": 0.0,   "R_arm_j6": 0.0,
        "L_arm_j7": 0.0,   "R_arm_j7": 0.0,
    }
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
    # ---- approach 阶段 (从固定起点沿 retarget 接近路径够到 PreGrasp, 再抓) ----
    approach_steps = 0                # >0: episode 提前此步数起步, 腕跟随接近路径(env 变量 GRASP_APPROACH)
    approach_res_scale = 3.0          # 接近段腕残差放大倍率(相对抓取段±2cm; 接触前无精细约束)
                                      # ⚠ 按**参考帧号**切的二值放大. dyn_res=True 时被
                                      # 动态残差取代 (按距离连续调, 且臂/手指分别调)
    # ---- 动态残差: 按 掌心->物体表面 距离连续缩放残差界 ----
    # 人手轨迹是**弱参考**, 只用来把手引到物体附近、省掉盲目探索:
    #   远 -> 臂放开去找物体, 手指基本不动
    #   近 -> 臂收紧(毫米级容忍度, 免得把物体捅飞), 手指放开去试怎么握住
    # 臂和手指方向相反. 飞手默认关闭 (保持旧行为), DexMate 默认打开.
    dyn_res = False
    dyn_d_near = 0.02                 # m; 掌心离表面近于它 -> 用 near 档
    dyn_d_far = 0.15                  # m; 远于它 -> 用 far 档; 中间线性过渡
    dyn_arm_far = 3.0                 # 臂: 远处放开 (= 原 approach_res_scale)
    dyn_arm_near = 1.0                # 臂: 近处 = 标定出来的界本身 (末端 ±2cm)
    dyn_fin_far = 0.2                 # 手指: 远处几乎不动 (±5.7° × 0.2 = ±1.1°)
    dyn_fin_near = 1.0                # 手指: 接触时 = finger_residual_max (之前训练验证过的值)
    start_jitter = 0.0                # m, 每回合起点腕位加均匀随机偏移 ±此值 (env 变量 GRASP_START_JITTER)
                                      # 训练鲁棒性: 让策略覆盖一圈起点; 0=固定起点(旧行为)
    # ---- 接触触发合拢 (指尖靠近物体才合, 不按帧数; 形状泛化用) ----
    contact_close = False             # True: 合拢由指尖-物体距离触发 (env 变量 GRASP_CONTACT_CLOSE)
    close_trigger_dist = 0.035        # m, 指尖离物体表面近于此则开始合拢
    close_speed_steps = 15            # 触发后合拢到位的步数
    # ---- "跟随"类奖励 (无 GraspPose 设定下要关; 落到 RewardWeights 同名字段) ----
    # 没有 GraspPose 就没有 cuRobo, 也就没有任何**经过验证可行**的手部参考:
    # 参考手指 = 人手重建 + retarget 的产物, 是全链路噪声最大的量.
    # 拿它当模仿目标 = 奖励"别修正", 与这个项目的前提(修正带噪重建)直接矛盾,
    # 也与手指累积残差的 69° 权限对着干.
    imit_finger_w = 0.5               # r_imit 手指通道权重; 0 = 只锚腕位, 手指姿态自由
    traj_after_contact = 1.0          # 握住后 r_traj 折扣; 0 = 接触后不再逐帧跟物体参考
    # ---- 终点势函数 (替代逐帧跟随, 给"物体该去哪"的方向) ----
    # 权重量纲的推导见 reward.py 的 lam_goal 注释 (势函数总量 vs 年金项, 差一个 episode 长度).
    lam_goal = 0.0                    # 0 = 关; 完整轨迹任务用 20 (与 contact 年金同量级)
    sigma_goal = 0.10                 # m, 势差分的尺度
    # 门控: 连续多少步 ≥min_contacts 指接触才算"形成稳定抓取", 之后 latch 保持整回合.
    # 用 latch 而不是瞬时接触: 一次滑脱不该把方向信号整个剁断.
    goal_gate_steps = 10              # 0.5s @20Hz
    # ---- 抓取稳定性惩罚 (门开后手物相对位姿漂移; 0 = 关) ----
    # 量纲: 漂 6.7cm 时每步扣 w_grip×(0.067−0.02)=0.094(w=2), 门开约 180 步 → 总量 ≈17,
    # 与终点势总量(≈38)同量级但更小 —— 它是约束, 不该盖过方向信号.
    w_grip = 0.0
    grip_deadband = 0.02              # m, 接触瞬间的沉降不罚
    # ---- 数据先验开关 (消融实验: Exp1 两个都开, Exp2 都关) ----
    use_grasp_prior = False           # 抓握段手指模仿目标: True=GraspPose纯抓姿 / False=重建手指
    use_curobo_guide = False          # cuRobo 预抓取路点引导 (退火, 初值由 train.py 置)
    lam_curobo_init = 0.5             # use_curobo_guide 时 lam_curobo 初值 (PPO 按 sr_ema 退火)
    curobo_wean_sr = 0.25             # sr_ema 达此值时 lam_curobo 退到 0
    curobo_guide_frac = 0.4           # cuRobo 引导时间窗: active 段前此比例步 (初始阶段指引)
    # ---- 诊断仪表 (docs/DESIGN_LOOP.md 的数据来源) ----
    # 滚动窗口回合数. 太小 -> 曲线抖, 读不出趋势; 太大 -> 设计改动的效果被旧回合稀释.
    # 256 ≈ 1024env 时 4 步左右的回合产出, 够平滑又跟得上.
    diag_window = 256
    # ---- 参考数据 (clips.py 注册表按 clip_name 解析; configure_cfg 会改 object usd 路径) ----
    clip_name = "Grasp2"   # 2026-08-11: 旧 clip11(bi_v2ap 源, 无 ref_builder) 已删
    target_hz = 20.0                  # = 1 / (dt*decimation)
