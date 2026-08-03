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
    # ---- 消融开关 (环境变量, 免改代码) ----
    # 为什么要这个: 旧 run 的 checkpoint 观测维度与当前代码不同 (full2 是 172, 现在 195),
    # 想拿基线做同口径确定性评测就必须能把新增通道关掉. 顺带让消融实验不用改文件.
    #   RL_ACC_FINGER=0     关累积手指残差 (obs −22)
    #   RL_LAM_GOAL=0       关终点势       (obs −1)
    #   RL_W_GRIP=0         关抓取稳定罚   (不影响 obs)
    #   RL_IMIT_FINGER_W    r_imit 手指通道权重 (旧默认 0.5)
    #   RL_TRAJ_AFTER       接触后 r_traj 折扣  (旧默认 1.0)
    accumulate_finger = os.environ.get("RL_ACC_FINGER", "1") == "1"
    # 本体58(臂7+指22 的 pos/vel) + 腕13 + 物体13 + 参考42 + 相位4 + 接触5 + 力矩7
    # + 动态残差距离1 + 动作29 = 172; 再按开关加 累积残差22 / 终点势门控1
    observation_space = (172
                         + (22 if os.environ.get("RL_ACC_FINGER", "1") == "1" else 0)
                         + (1 if float(os.environ.get("RL_LAM_GOAL", "20.0")) > 0 else 0))
    dyn_res = True                    # DexMate 默认开动态残差 (飞手保持旧行为)
    # 阈值按**实测距离范围**定, 不能拍脑袋: grasp_only + 接近段关闭时, 掌心到物体表面
    # 全程只在 3.45~11.20cm 之间 (腕被 ref builder 冻结在 PreGrasp, 只合手指).
    # 用基类默认的 2/15cm 会让 u 卡在 0.34~0.57, 两端都够不到, 等于白开.
    # ⚠ 打开接近段(GRASP_APPROACH>0)后范围完全不同, 要重测重设.
    dyn_d_near = 0.005
    dyn_d_far = 0.05

    # ---- 动作语义: 可累积增量 + 管道约束 ----
    # 旧: arm_tgt = q_ref[t] + Δq  —— 偏差**不累积**, 任何时刻离参考不超过残差界(±2cm).
    #     后果: 策略结构上走不到 2cm 以外, 只能靠"把物体摆得极准"来补偿.
    #     摆放误差必须 <2cm, 这对任何自动摆放流水线都太苛刻 (实测 3/9 通过).
    # 新: 策略维护自己的目标 q_cmd, 每步累积 Δq; 末端偏离参考超过 tube_radius 就拉回管壁.
    #     参考退化成**弱引导**(管道中心 + 观测), 不再逐帧钉住手.
    # 副产物: "像人"变成结构性保证 —— 每帧都在参考 tube_radius 内, 整条轨迹按构造就相似,
    #         不需要再设相似度奖励.
    accumulate_action = True
    tube_radius = 0.05                # m, 末端允许偏离参考多远 (实测: 不可达帧需要的
                                      # 修正量中位 4.2cm, 90分位 11.5cm —— 5cm 覆盖一半多)
    tube_kp = 2.0                     # 拉回力度 (实际末端落后于指令, 需要过阻尼)
    tube_soft_frac = 0.7              # 从 0.7R 就开始往回拉, 留出跟踪滞后的余量

    # ---- 臂关节残差界 (rad, j1..j7) ----
    # 不能用飞手那个 0.02 —— 那是**笛卡尔**量. 关节空间里同样的 Δq 在肩和腕上产生的
    # 末端位移差 5.7 倍 (雅可比杠杆臂 0.586 vs 0.103 m/rad), 给统一界等于给肩过大权限.
    arm_residual_max = _load_arm_residual()

    # ---- 参考摆放 ----
    # "ref_builder" = 保留 ref builder(replay_grasp) 已摆好的手-物关系, 只在够不到时整体平移.
    #               replay_grasp 已经做了 affordance 锚定 / hover_gap / 抓取后冻结腕 /
    #               固定合拢斜坡 —— 这些都是为"抓得住"设计的, 不能覆盖.
    # "bimanual"  = 用 bimanual_align 重摆 (为"把双手轨迹放进机器人第一人称坐标系"设计).
    #               实测在 grasp_only 单手抓取上会把 ref builder 的设计全抹掉, 接触率 17.3%->0.1%.
    place_mode = "ref_builder"
    # ref builder 内部用哪个锚 (place_mode="ref_builder" 时才有意义):
    # "camera" = 相机/手/物体 xy 刚体同步平移, 手物相对关系保留重建原样 (默认).
    # "palm"   = 旧的"把物体搬到掌心正下方", 会破坏相机对齐 -> 跨身抓取. 只为复现历史 run.
    # ⚠ 换锚会改变物体位置和腕轨迹, 旧 checkpoint 的观测分布对不上, 别混用.
    anchor_mode = "camera"
    # PreGrasp 对齐: 把 gs 帧的腕挪到"锚点悬停在 affordance 正上方"的位置.
    # 不开的话手离物体十几厘米 (重建里手根本没碰到物体), grasp 段结构上学不动.
    # 锚点与 hover 在 place_camera.PREGRASP_ANCHOR / PREGRASP_HOVER_GAP.
    # 环境变量 GRASP_PREGRASP_ALIGN=0 可关掉 (冒烟对照组).
    pregrasp_align = True
    # 悬停余量: None = 由几何推出 (让张开的手最低点离桌面留 clearance).
    # ⚠ 旧的 0.06 是我在 Grasp2 上扫出来的魔数, 换成合拢中心锚点后不再需要 ——
    # 之所以曾经要 6cm, 正是因为锚点差了 6.88cm, 手得抬高才不撞桌.
    hover_gap = None
    clearance = 0.015                 # m, 张开的手最低点离桌面留多少
    # 合拢由**指尖到物体表面的距离**触发, 不按帧号. 这样手指跟着物体走而不是跟着时钟走.
    contact_close = True
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

    # ---- 手指自由度 ----
    # 覆盖基类的 0.10 rad (5.7°). 几何下界实测: 要让指尖真的碰到物体, 6 条 clip 需要
    # 28~59° (中位 45°) —— 5.7° 结构上就够不到, 策略再怎么学也抓不住
    # (见 calib_finger_geom.py; 这也解释了为什么固定合拢构型换个物体就失效).
    # 取下界的 1.5 倍留余量 = 1.2 rad (69°).
    #
    # ⚠ 但 69° **不能**当每步界给: sigma × 界 = 每步扰动, 实测 sigma=0.76 时早就把物体
    #   打飞了. 所以这里走**累积**模式 (基类 accumulate_finger), 把两个界拆开:
    #     finger_rate_max  0.10 rad (5.7°/步)  —— 标定过的"每步安全量", 保住毫米级容忍度
    #     finger_total_max 1.2  rad (69°)      —— 攒够了才够得到物体, 这才是几何要求的量
    #   dyn_res 缩放的是**每步增量**(远 ×0.2 = 1.1°/步, 近 ×1.0 = 5.7°/步),
    #   累积上限不随距离缩 (否则手被推开一点就把攒下的握力泄掉).
    # ⚠ accumulate_finger 在本类**开头**由 RL_ACC_FINGER 决定 (observation_space 也依赖它).
    #   这里不要再赋一次 —— 类体里后写的会覆盖先写的, 而 observation_space 那行读的是
    #   环境变量, 两者会不一致 (踩过: 基线评测报 "观测宽度 194 != observation_space 172").
    finger_rate_max = 0.10
    finger_total_max = 1.2
    finger_residual_max = 1.2         # 只在 accumulate_finger=False 时生效 (对照实验用)

    # ---- 跟随类奖励: 这一组 clip 走的是**无 GraspPose** 设定 (B 组) ----
    # 没有 GraspPose -> 没有 cuRobo -> 参考手部轨迹里没有任何一段是验证过可行的.
    #   手指模仿: 关 (置 0). 参考手指是 retarget 出来的噪声, 不是抓法.
    #             抓法改由 contact(接触数) + afford(指尖落在人的接触区) + lift(棘轮) 定义
    #             —— 全是物理/接触信号, 不依赖 GraspPose.
    #   接触后的物体逐帧跟随: 关 (置 0). 成功由终点判据 (place_tol) 定义, 中间路径
    #             是重建出来的带噪曲线, 没有理由逐帧钉住.
    #   接触**前**的 r_traj 保留 —— 那一段实际含义是"别把物体碰跑", 手指有 69° 权限时
    #             这条防扰动约束比以前更需要.
    # 要跑对照 (Exp1 全先验) 时把这两个改回 0.5 / 1.0.
    imit_finger_w = float(os.environ.get("RL_IMIT_FINGER_W", "0.0"))
    traj_after_contact = float(os.environ.get("RL_TRAJ_AFTER", "0.0"))
    # ---- 终点势 (2026-07-28 上线, Run A 的单一改动) ----
    # accum run 的判读: 接触率 1.9%→76.5% 但成功率 2.9%(基线 9.8%) —— reward 的任务核
    # 测"抬多高", 成功判据是"放到位", 两者不是一件事; 而关掉 traj_after_contact 又把
    # 唯一指向目标的稠密信号删了. 终点势同时补这两个洞, 且不钉带噪的中间路径.
    # lam_goal=20 是按"势函数总量 ≈ contact 年金总量"标定的, 推导见 reward.py.
    lam_goal = float(os.environ.get("RL_LAM_GOAL", "20.0"))
    sigma_goal = 0.10
    goal_gate_steps = 10              # 连续 0.5s 双指接触才开门 (防推/撞/抛刷分)
    # ---- 抓取稳定性 (2026-07-28 Run C 的单一改动) ----
    # Run B 判读: 终点势把成功率从 0.029 拉回 0.10 (追平基线), 但 grip_drift_cm=6.7 —— H2 触发,
    # 形成的不是稳定包络而是"边滑边带". 而 I1 未触发 (fast_frac 0.001), 所以选 C2 不选 C1.
    w_grip = float(os.environ.get("RL_W_GRIP", "2.0"))
    grip_deadband = 0.02

    # ---- RSI (从轨迹中间起步) ----
    # ⚠ 完整轨迹模式下必须关掉, 与 home_steps **不兼容**:
    #   RSI 说"这回合从第 100 帧(搬运途中)起步", 而 home 段把手插值到 **PreGrasp 姿态**
    #   —— 手在抓取位, 物体却在搬运途中, 手物完全对不上, 80% 的样本是无效的.
    #   而且 ref_obj_pos 现在是真实轨迹, RSI 会把物体摆到搬运路径上的随机一点
    #   (肉眼可见: 物体一会儿在两手中间, 一会儿跑到左手旁边).
    #   基类默认 0.8 是给"腕冻结 + 物体常量"的 grasp_only 设计的, 那时两者不冲突.
    rsi_prob = 0.0

    # ---- 完整轨迹任务 (靠近 -> 抓住 -> 搬运 -> 放置 -> 归位) ----
    # 分段全部由 phase_* 给出: 交互开始帧=抓住, 交互结束帧=放置.
    grasp_only = False                # False = 跟完整人手轨迹, 不用硬编码抬升
    freeze_wrist = False              # 抓取窗内腕不冻结, 跟着人手搬运
    place_tol = 0.08                  # m, 物体最终位置离目标多近算成功

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
