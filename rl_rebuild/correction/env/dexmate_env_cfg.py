"""DexMate(Vega)+Sharpa 关节空间残差修正环境的配置.

与飞手环境 (SharpaCorrectionEnvCfg) 的区别只有"机器人怎么被驱动"这一件事:

  飞手      腕 = 浮动根刚体, 关重力, 被凭空的 wrench 推着走   动作 28 = 腕Δpos3+Δaa3+指22
  DexMate   腕 = 7 关节链的末端, 带重力, 电机 PD 驱动         动作 29 = 臂Δq7 + 指Δq22

换过来的意义: 限位 / 自碰撞 / 奇异位形 / 力矩上限 全部由仿真器天然执行,
不再靠 reward 惩罚去"劝"策略别越界 —— 即用户要的"带着约束训练".

详见 docs/DEXMATE_TRAINING_FEASIBILITY.md (吞吐实测 + 可达性实测 + 残差界标定).
"""
from __future__ import annotations

import json
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from rl_rebuild.correction.env.correction_env_cfg import SharpaCorrectionEnvCfg

_MAGICSIM_USD = os.path.join(
    os.environ.get("MAGICSIM_ASSETS", "/home/lyh/luhr/MagicSim/Assets"),
    "Robots", "vega_1p_sharpa.usd")
# 躯干锁死版 (tools/make_fixed_torso_usd.py 生成; 不改 MagicSim 的共享资产).
# 躯干/底盘/头换成 fixed joint —— 它们在这个任务里全程不动, 与其和求解器较劲
# (实测驱动顶不住、关节限位也没被强制执行, 每回合前 60 步手臂基座漂 10.55cm),
# 不如直接从自由度里去掉. 文件不在就回退到原资产.
_FIXED_USD = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../assets", "vega_1p_sharpa_fixedtorso.usd"))
_DEXMATE_USD = _FIXED_USD if os.path.exists(_FIXED_USD) else _MAGICSIM_USD
if _DEXMATE_USD is _MAGICSIM_USD:
    # 静默回退会得到**完全不同的物理**: 躯干会塌(手臂基座每回合漂 10.5cm)、手臂无碰撞
    # (会穿过桌子). 训出来的东西不可用, 所以必须大声报警.
    print("=" * 78 + f"""
[dexmate] ⚠ 找不到派生资产 {_FIXED_USD}
          回退到原始 MagicSim USD —— 躯干会塌、手臂无碰撞几何, **训练结果不可用**.
          先生成 (顺序不能反, 后者在前者产物上就地追加):
            python tools/make_fixed_torso_usd.py
            python tools/add_arm_collision.py
""" + "=" * 78)
# 用锁死版时, 躯干/底盘/头**不再是关节** (articulation 从 67 个自由度降到 58 个),
# 所以针对它们的 InitialStateCfg / 执行器组 / 运行时锁定 全部要跳过.
TORSO_FIXED = _DEXMATE_USD == _FIXED_USD
_BOUND_JSON = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../../data/offline_ik/arm_residual_bound.json"))

# URDF 里的**真实电机**力矩上限 (Nm). USD 里写的是 1500/250, 比真机大 10~60 倍 ——
# 照 USD 训出来的策略会用真机根本出不了的力, sim2real 直接崩. 必须用 URDF 的值.
ARM_EFFORT = {1: 150.0, 2: 150.0, 3: 80.0, 4: 80.0, 5: 25.0, 6: 25.0, 7: 25.0}

# USD 里的 stiffness 是**度制**, IsaacLab 读进来会 ×180/π. 下面是换算后的实际值 (Nm/rad).
# 保留它们的**相对形状**(近端硬、远端软, 这个是合理的), 整体乘 arm_stiffness_scale 用.
ARM_KP_USD = [140312.7, 124609.6, 108032.0, 60512.1, 37899.2, 12066.2, 6497.3]
ARM_KD_USD = [5729.6] * 7


def _load_arm_residual():
    """臂关节残差界 (rad, j1..j7). 由 calib_arm_residual.py 从雅可比标定得到:
    末端位移 95 分位 = 2cm, 与飞手 wrist_pos_residual_max 等价."""
    try:
        with open(_BOUND_JSON) as f:
            return list(json.load(f)["bound_rad"])
    except Exception:
        # 标定文件缺失时的保守回退 (= 最紧的那一档给所有关节)
        print(f"[cfg] ⚠ 找不到 {_BOUND_JSON}, 臂残差界回退到均一 0.88°"
              f" (先跑 calib_arm_residual.py)")
        return [0.0154] * 7


@configclass
class DexmateCorrectionEnvCfg(SharpaCorrectionEnvCfg):
    # ---- 动作 / 观测 ----
    action_space = 29                 # 臂Δq7 + 指Δq22
    # 本体58(臂7+指22 的 pos/vel) + 腕13 + 物体13 + 参考42 + 相位4 + 接触5 + 力矩7
    # + 动态残差距离1 + 动作29
    observation_space = 172
    dyn_res = True                    # DexMate 默认开动态残差 (飞手保持旧行为)
    # 阈值按**实测距离范围**定, 不能拍脑袋: grasp_only + 接近段关闭时, 掌心到物体表面
    # 全程只在 3.45~11.20cm 之间 (腕被 producer 冻结在 PreGrasp, 只合手指).
    # 用基类默认的 2/15cm 会让 u 卡在 0.34~0.57, 两端都够不到, 等于白开.
    # ⚠ 打开接近段(GRASP_APPROACH>0)后范围完全不同, 要重测重设.
    dyn_d_near = 0.005
    dyn_d_far = 0.05

    # ---- 臂关节残差界 (rad, j1..j7) ----
    # 不能用飞手那个 0.02 —— 那是**笛卡尔**量. 关节空间里同样的 Δq 在肩和腕上产生的
    # 末端位移差 5.7 倍 (雅可比杠杆臂 0.586 vs 0.103 m/rad), 给统一界等于给肩过大权限.
    arm_residual_max = _load_arm_residual()

    # ---- 参考摆放 ----
    # "producer"  = 保留 producer(replay_grasp) 已摆好的手-物关系, 只在够不到时整体平移.
    #               replay_grasp 已经做了 affordance 锚定 / hover_gap / 抓取后冻结腕 /
    #               固定合拢斜坡 —— 这些都是为"抓得住"设计的, 不能覆盖.
    # "bimanual"  = 用 bimanual_align 重摆 (为"把双手轨迹放进机器人第一人称坐标系"设计).
    #               实测在 grasp_only 单手抓取上会把 producer 的设计全抹掉, 接触率 17.3%->0.1%.
    place_mode = "producer"
    # 手在 PreGrasp 位相对物体整体抬高多少 (m). 传给 replay_grasp 的 hover_gap.
    # ⚠ 飞手的 0.02 **不能直接用**: 那个参考会把手压进桌子 (合拢完成时 15 个手部 link
    # 在桌面下, 最深 4.72cm). 飞手靠 wrench 硬顶过去(无力矩上限、关重力), 真机械臂顶不动
    # —— 腕关节只有 25Nm, 结果是全关节饱和、末端误差 14.9cm、接触率 0%.
    # 实测 (Grasp2, 64env×3回合, 零残差):
    #   0.02 -> 跟踪 14.92cm  接触  0.0%  抬升 4.03cm   (手卡在桌子里)
    #   0.06 -> 跟踪  0.89cm  接触 45.4%  抬升 4.16cm   ← 选定
    #   0.09 -> 跟踪  0.31cm  接触 60.1%  抬升 0.93cm   (手太高, 碰得到握不住)
    hover_gap = 0.06
    reach_margin = 0.75               # m; 参考最远点离肩超过它就整体平移拉近 (臂展实测 0.809)

    # ---- 机器人 ----
    hand_side = "right"               # 用哪只手抓 (交互手由 bimanual_align 自动判定后覆盖)
    dexmate_pos = (-0.5, 0.0, 0.0)
    dexmate_quat = (1.0, 0.0, 0.0, 0.0)
    # 非交互臂 + 躯干保持的默认角 (度). 交互臂由参考 IK 轨迹驱动, 不受此表约束.
    dexmate_joints = ({"L_arm_j1": 45.0, "R_arm_j1": -45.0,
                       "L_arm_j4": -90.0, "R_arm_j4": -90.0}
                      if TORSO_FIXED else
                      {"torso_j1": 45.0, "torso_j2": 90.0, "torso_j3": 0.0,
                       "L_arm_j1": 45.0, "R_arm_j1": -45.0,
                       "L_arm_j4": -90.0, "R_arm_j4": -90.0})
    # 增益模式:
    #   "effort_matched" (默认) kp_i = τ_max_i / Δq_max_i —— 让**满幅残差恰好是电机能执行的**.
    #     为什么: USD 的 kp 形状是按 1500Nm 上限写的, 真机是 150/80/25. 实测 j5 在抓取窗内
    #     力矩饱和 **82.8%**, 也就是策略的残差有 83% 的时间**出不了力、静默失效**,
    #     信用分配被污染. 这条定则把残差界标定和执行器标定耦合起来, 不再各调各的.
    #   "usd_scaled"  沿用 ARM_KP_USD × arm_stiffness_scale (旧行为, 对照用)
    gain_mode = os.environ.get("DEX_GAIN_MODE", "effort_matched")
    # 臂增益缩放 (乘在 ARM_KP_USD / ARM_KD_USD 上). 由 calib_arm_gain.py 扫出来.
    # ⚠ kp 与 kd **同比例**缩放时, 速度滞后项 kd·v/kp 不变 —— 实测 kp 扫 0.003~0.3
    #   (100 倍) 末端误差几乎不动. 真正的杠杆是**阻尼比**, 必须单独调.
    # 选定档 (kp×0.2, kd×0.02 即阻尼比 0.1): 末端误差中位 0.23cm / 95分位 1.93cm,
    # 力矩饱和 10.1%, 最大关节速度 1.74rad/s (URDF 限 2.4/2.7, 在限内).
    arm_stiffness_scale = 0.2
    arm_damping_scale = 0.02
    # 躯干增益. ⚠ 不能用 USD 的值: 800000 度制 -> 4.58e7 Nm/rad, 这个刚度配 141kg 的
    # 多体 + 8 次位置迭代, PhysX 求解器收敛不了 —— 实测 torso_j2 从 90° 塌到 83.65°,
    # 整个上半身连着肩转 6.35°, 手臂基座跑掉 8.5cm, 于是"臂跟得很好但末端差 8.5cm".
    # 躯干只需要**举住上半身**(量级 100Nm), 1e5 已经给到 0.06° 的静差, 绰绰有余.
    torso_stiffness = 1.0e5
    torso_damping = 1.0e4
    # 建 env 时先空跑多少个**物理**步让机器人在重力下沉降到平衡, 再读锚点位姿.
    # 只跑 1 步读到的还是初始值, 而稳态会偏几度 —— 那几度就是 8cm 级的基座偏移.
    settle_physics_steps = 24 if TORSO_FIXED else 240   # 锁死版不用等沉降
    settle_torso_tol_deg = 0.2        # 躯干偏离默认位小于它才算收敛, 才敢读锚点
    settle_max_rounds = 40            # 收敛循环上限 (40×30 = 1200 物理步 = 5s)
    # 躯干/底盘/头用**关节限位**锁死, 不靠驱动顶住 (驱动顶不住, 见 _place_and_solve_ik 注释)
    lock_torso = not TORSO_FIXED
    lock_tolerance = 0.0017           # rad (0.1°)
    # 自碰撞: USD 默认值. 开着才谈得上"让仿真器执行自碰撞约束", 但若 USD 缺相邻连杆的
    # 碰撞过滤, 开了会让机器人一动就自锁. 先按 USD 默认跑 baseline, 再决定.
    self_collisions: bool | None = (
        None if "DEX_SELFCOL" not in os.environ
        else os.environ["DEX_SELFCOL"] == "1")

    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",      # env 命名空间内 -> 被 clone 复制
        spawn=sim_utils.UsdFileCfg(
            usd_path=_DEXMATE_USD,
            activate_contact_sensors=True,         # 指尖 ContactSensor 依赖它
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(-0.5, 0.0, 0.0)),
        actuators={
            # 臂: 增益在 __post_init__ 里按 ARM_KP_USD × scale 逐关节填;
            #     力矩上限按 URDF 真机值分三档 (不是 USD 的 1500)
            "arm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["[RL]_arm_j[12]"], stiffness=0.0, damping=0.0,
                effort_limit_sim=150.0),
            "arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["[RL]_arm_j[34]"], stiffness=0.0, damping=0.0,
                effort_limit_sim=80.0),
            "arm_wrist": ImplicitActuatorCfg(
                joint_names_expr=["[RL]_arm_j[567]"], stiffness=0.0, damping=0.0,
                effort_limit_sim=25.0),
            # 手: 与飞手完全同一套增益 (USD 里也是 20/2, 已核对) —— 手侧零改动
            "hands": ImplicitActuatorCfg(
                joint_names_expr=["(left|right)_(thumb|index|middle|ring|pinky).*"],
                stiffness=20.0, damping=2.0),
        },
    )

    # ---- 终止 ----
    # ⚠ 不能用"力矩饱和"当卡死判据: IsaacLab 的 applied_torque 是**近似公式**
    # clip(kp·Δq + kd·Δq̇), kp 一大就恒定顶满, 与"是不是真卡住"无关.
    # 改用真正有意义的信号: 臂关节持续追不上参考 (撞到桌子/自碰撞/力矩不够都会这样).
    term_arm_err = 0.35               # rad (20°); 单关节偏离 q_ref 超过它
    term_arm_steps = 20               # 连续多少控制步 (1s@20Hz) 算真卡住

    # ---- 从默认站姿出发的接近段 ----
    # 每一回合都从**你在 GUI 里调好的那个双臂对称站姿**起步, 在关节空间插值走到 PreGrasp,
    # 再进入抓取窗. 这一段是"怎么把手送到物体旁边", 抓取窗是"怎么握住".
    #
    # ⚠ 为什么不直接用人手轨迹的接近段 (GRASP_APPROACH):
    #   人手在接近时**手肘几乎伸直**(参考 j4≈-28°, 默认站姿 j4=-90°), 力矩臂长得多,
    #   DexMate 撑不住 —— 实测末端比参考低 12.6cm, 触发 term_arm_err 每 20 步重置一次,
    #   参考帧号永远推进不了. 人的手臂能这么伸, 这台机器人在这个负载下不能.
    # 关节空间插值的两端都是已验证可达可撑的姿态, 中间是凸组合, 天然平滑.
    home_steps = 40                   # 0 = 关闭 (直接从 PreGrasp 起步, 旧行为)
    # 关节目标在物理子步间线性插值 (20Hz 决策 -> 240Hz 平滑送达).
    # 不改策略决策频率, 只改目标送达方式; 消掉"每 12 个子步跳一次"的台阶.
    substep_interp = True

    # 求解器位置迭代上限. ⚠ 飞手用的 8 对 DexMate 不够:
    # USD 里 vega 自己请求 solverPositionIterationCount=32, 而 SimulationCfg 的这个字段是
    # **clamp**(IsaacLab 文档明写), 8 会把 32 钳掉 —— 141kg 的多体解不动, 表现就是躯干
    # 在重力下塌几度、连关节限位都拉不回来, 进而让手臂基座漂移、末端偏十几厘米.
    solver_position_iterations = int(os.environ.get("DEX_SOLVER_ITERS", 32))

    def __post_init__(self):
        self.sim.physx.max_position_iteration_count = self.solver_position_iterations
        # 机器人**就是**场景里那台, 不需要再挂一个 /World/Dexmate 视觉参考
        self.show_dexmate = False
        self.show_human_traj = False
        # 默认站姿必须写进 InitialStateCfg —— default_joint_pos 是 _reset_robot 里
        # 非交互臂/躯干的取值来源, 不设的话它们全是 0 位 (手臂直伸, 会插进桌子).
        self.robot_cfg.init_state.pos = tuple(self.dexmate_pos)
        self.robot_cfg.init_state.rot = tuple(self.dexmate_quat)
        self.robot_cfg.init_state.joint_pos = {
            n: (float(v) if "prismatic" in n else float(math.radians(v)))
            for n, v in self.dexmate_joints.items()}
        self.apply_gains()
        if self.self_collisions is not None:
            self.robot_cfg.spawn.articulation_props = \
                sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=self.self_collisions)

    def _effort_matched_kp(self):
        """kp_i = τ_max_i / Δq_max_i, kd 按同一阻尼比 (0.1) 折算."""
        kp = [ARM_EFFORT[i + 1] / max(self.arm_residual_max[i], 1e-6) for i in range(7)]
        kd = [k * (ARM_KD_USD[i] * self.arm_damping_scale)
              / max(ARM_KP_USD[i] * self.arm_stiffness_scale, 1e-9) for i, k in enumerate(kp)]
        return kp, kd

    def apply_gains(self):
        """把 arm_stiffness_scale / torso_stiffness 填进 actuator cfg.

        ⚠ 改完这些标量后**必须**再调一次本方法 —— __post_init__ 在 cfg 构造时就跑完了,
        事后改 scale 不会自动生效 (踩过: 增益扫描五档结果完全相同).

        必须走 actuator cfg 而不是运行时 write_joint_stiffness_to_sim ——
        后者明确"不更新执行器模型"(IsaacLab #128), 只改 PhysX 不改 applied_torque
        的计算, 会让力矩读数和实际物理对不上.
        """
        if self.gain_mode == "effort_matched":
            KP, KD = self._effort_matched_kp()
        else:
            KP = [ARM_KP_USD[i] * self.arm_stiffness_scale for i in range(7)]
            KD = [ARM_KD_USD[i] * self.arm_damping_scale for i in range(7)]
        for grp, idx in (("arm_shoulder", [0, 1]), ("arm_elbow", [2, 3]),
                         ("arm_wrist", [4, 5, 6])):
            self.robot_cfg.actuators[grp].stiffness = {
                f"{side}_arm_j{i+1}": KP[i] for i in idx for side in ("R", "L")}
            self.robot_cfg.actuators[grp].damping = {
                f"{side}_arm_j{i+1}": KD[i] for i in idx for side in ("R", "L")}
        if "torso" in self.robot_cfg.actuators:
            self.robot_cfg.actuators["torso"].stiffness = self.torso_stiffness
            self.robot_cfg.actuators["torso"].damping = self.torso_damping
