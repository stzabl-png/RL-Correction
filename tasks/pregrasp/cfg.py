"""PreGrasp 抓取任务配置 v2 — 稳定抓握 + 微抬升验证 (不学抬, 只验抓).

## 与 rl_rebuild 的关系

`tasks/` 是新任务的家; `rl_rebuild` 冻结为"引擎 + 旧 correction 任务", 只 import 不新增.
本 cfg 继承 DexmateCorrectionEnvCfg **只为拿到验证过的那一层**: 机器人资产(躯干锁死)、
相机锚定摆放、PreGrasp 对齐、IK 求解(含预解的 q_lift 抬升轨迹)、增益标定、子步插值.
任务层 (阶段机 / 手型模板 / 奖励 / 复位 / 观测) 全部在 env.GraspTaskEnv 里重写.

## v2 相对 v1 的结构性修订 (2026-07-29, 依据两次失败 run + 外部 review 交叉判定)

  1. 合拢加**默认参考斜坡**: c += ref_rate + rate_max·a — 零动作 = 匀速合拢,
     策略学"何时停/减速/加速", 不再需要从随机游走里发现"持续正 Δc"
     (13.3M 步实测: 没有参考斜坡时策略收敛到完全不动).
  2. 绝对向心奖励 → **抓取质量势差分** (Q = 向心 − λ·不对称 − λ·净力矩):
     绝对值奖励每步 ~0.9 × 上百步 ≫ 成功 +20 且成功终止会断掉收入流 —— 结构性
     反成功激励. 势差分只奖励改善, 原地刷分为零.
  3. **微抬升验证**: 候选抓取保持 0.4s → 硬编码腕升 1cm (用基类预解的 q_lift) →
     物体跟上且接触不丢才算成功. 区分"真抓取"与"桌面支撑下的稳定接触".
  4. 动作 30 → **13 维** (臂7 + c1 + 每指合拢残差5): c 与 22 维关节残差同一手型
     有无穷组合 (action null space); 每指一维保留调落点能力, 22 维留作后续课程.
  5. 臂残差缩小到"抓取微调"尺度 (每步末端 ~5mm, 累积 ±0.12rad): ±3mm 物体抖动的
     任务里 2cm/步是搬运尺度, 给大了策略会学"挥腕撞物体".
  6. PREGRASP 阶段跳过 (起点实测秒过对齐容差, 纯复杂度); 大扰动课程再启用.
  7. PointNet 分支暂关 (单物体+精确状态下是重复信息, 徒增消融变量);
     表面点云保留 —— 逐垫距离/表面信号还靠它算.
"""
from __future__ import annotations

from isaaclab.utils import configclass

from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg


class Phase:
    """任务阶段. v2 用 GRASP + LIFT(=微抬升验证); 其余留接口."""
    PREGRASP = 0                 # v2 跳过 (起点已在对齐容差内); 大扰动课程再启用
    GRASP = 1
    LIFT = 2                     # v2 语义 = 微抬升验证段 (硬编码斜坡, 不是学抬升)
    TRANSPORT = 3
    PLACE = 4
    RETURN = 5
    N = 6
    NAMES = ("pregrasp", "grasp", "verify", "transport", "place", "return")


@configclass
class GraspTaskCfg(DexmateCorrectionEnvCfg):
    # ================= 钉死的继承项 (不要在实验里改) =================
    rsi_prob = 0.0
    freeze_wrist = True
    grasp_only = False
    place_mode = "ref_builder"
    anchor_mode = "camera"
    pregrasp_align = True
    contact_close = False
    home_steps = 0
    settle_steps = 5             # 物体落稳的静置步数
    # 待命站姿 (2026-07-30 用户设计, snap_pose 顶视验收): 在基类手调站姿
    # (j1 ±45°, j4 ∓90°) 上加 **j2 双臂外展 ±45°** —— 双手对称张到桌子两侧边,
    # 桌面中央工作区完全让空; 对称 => 左手 clip 直接镜像可用. 快照:
    # docs/figs/stance/pose_j2_{out45,top}.png. 另有 prior 加载器的 yaw 间隙
    # 软约束 (<25cm 加罚) 兜底. kinematics.DEFAULT_ARM_DEG 的 IK 种子未同步 (只是种子,
    # 接人手轨迹任务上线时一并对齐).
    dexmate_joints = {"L_arm_j1": 45.0,  "L_arm_j2": 45.0,  "L_arm_j4": -90.0,
                      "R_arm_j1": -45.0, "R_arm_j2": -45.0, "R_arm_j4": -90.0}
    episode_length_s = 12.0      # 上限兜底; 实际由 phase_timeout 逐阶段截断
    # PointNet 分支关闭 (v2 修订 7); n_obj_points 保留 —— 逐垫距离/法向信号要用表面点
    enable_pointcloud = False
    # 64 是当年给 PointNet 定的显存上限; PointNet 已关, 而 cent 的局部法向需要足够
    # 分辨率 (64 点铺在 5.8cm 环上, 最近点可能离真实接触点 1cm+, 法向就没意义).
    # 512 点的 cdist 代价 = 1024env×5垫×512 ≈ 2.6M 距离, 可忽略.
    n_obj_points = 512

    # ================= 动作 (13 维) =================
    #   a = [ Δq_arm(7),  a_c(1),  a_δ(5) ]
    # 合拢: c += closure_ref_rate + a_c·closure_rate_max  (a_c=-1 停, 0 匀速, +1 加速)
    # 每指: δ_i += a_δ·delta_rate_max, 指 i 的合拢深度 = clip(c+δ_i)  (调指尖落点)
    action_space = 13
    # 观测 (布局见 env._get_observations, 改布局必须同步 —— _check_obs_dim 会当场报错)
    #
    # 🔴 2026-08-02 重构: actor / critic 按"部署时拿不拿得到"划分 (data engine 的前提是
    #    最终要有一个只吃重建+retarget 能给的东西的策略). 见 DESIGN_LOOP §2.12.
    #
    #   actor (151, **全部可部署**):
    #     本体 7+22+7+22=58 | 腕位姿+速度 13 | 合拢 c 1 + 每指 δ 5 | 离 PreGrasp 7
    #     | 相位 6+1 | 指尖接触 5 | 臂力矩 7 | 上步动作 13
    #     | **逐垫力向量(腕系) 15** | 到抓姿 位置误差 3 + 姿态误差 3
    #     | 参考跟踪误差 7 | 参考前瞻 7
    #   priv (32, critic 全看 / actor 只经 env_mlp 看**前 7**, 见 ppo.yaml actor_priv_dim):
    #     [0:7] 质量 1 + 摩擦 1 + 指尖力 5        ← 原样, 师生蒸馏那条线不动
    #     [7:32] 物体位姿(腕系) 3+4 + 线速度 3 + 角速度 3 + 逐垫距离 5 + 物体高度 1
    #            + 向心/不对称/净力矩 3 + 实时目标相对 3
    #
    # 搬家的理由: 物体**实时位姿**及其派生量 (逐垫距离/表面法向→向心分/物体高度/目标相对)
    # 是仿真真值, 重建的物体 track 不可信、真机要靠感知 —— 给 actor 就等于训了个没法用的
    # 策略. 而原来它们混在 obs 里, models.py 的 adapt_tconv 只替换 priv 不替换 obs,
    # 师生划分形同虚设.
    # 替换而非单纯删除: 原来的 5 个力**模长** + cent(用真值法向) 换成 5 个力**向量**(腕系)
    # —— 真实力传感器给得出, 且信息量比标量 cent 更大.
    # ⚠ 实际值由 apply_grasp_prior() 按下面两个开关算, 这里只是不带任何扩展时的基数.
    observation_space = 151
    # ---- 参考前瞻 (2026-08-02, 借鉴 ConTrack) ----------------------------
    # ConTrack 的 400 维 obs 里最大的一块是"当前+未来 5 帧完整参考状态"(270 维, 占 67%),
    # 而我们只给了 1 帧 delta(7 维) —— 策略基本看不到参考轨迹接下来的**形状**.
    # ref_look_frames=1 就是原行为; =5 给 5 帧.
    # ⚠ 与 ConTrack 的两点不同:
    #   ① 给**增量** q_ref[t+k]−q_ref[t], 不给绝对位姿 —— 人手轨迹带 14~16cm 系统偏差,
    #      绝对量不可信 (PLAN §7.7);
    #   ② **几何间隔**而不是连续 5 帧: 接近段总共才 gs≈32 帧, 连续 5 帧只覆盖 16%,
    #      而 {1,2,4,8,16} 覆盖到半条轨迹, 对"路径形状"的信息量大得多.
    ref_look_frames = 1
    ref_look_ks = (1, 2, 4, 8, 16)   # 取前 ref_look_frames 个
    # ---- 参考接触标签 (ConTrack 有, 我们**默认关**) ------------------------
    # ConTrack 给 20 维"哪个 link group 该接触", 它随参考帧变化所以有信息量.
    # 我们的手指参考是"傻瓜合拢斜坡", 没有逐帧接触序列; 能拿到的只有 prior 的
    # **静态**接触指集 —— 单 prior 的 run 里它是常数, **零信息量**.
    # 它只在**多 prior 混训 / 蒸馏成 prior-conditioned 策略**(计划 P5) 时才有用:
    # 那时它是"这次做的是哪种抓法"的条件输入. 所以先实现、默认关.
    ref_contact_obs = False

    closure_ref_rate = 0.025     # 默认合拢速度/步 (零动作 ~0.5s 到预包围, ~1s 到位)
    # v2.2: 参考斜坡只推进到 c_grasp(1.0), 不再冲到 1.25 —— 零动作的终态是 nominal
    # 抓握而不是深捏 (实测模板 c>1 时拇指压出 35N, "跟参考"变成净亏, 谁还跟).
    # 更深的挤压是**策略的选择** (a_c>0), 不是参考的默认.
    closure_rate_max = 0.025     # 策略对合拢速度的调制幅度 (=ref: a=-1 恰好停住)
    closure_min = 0.0
    closure_max = 1.25           # >1 = 比 nominal grasp 更紧 (薄物体需要)
    delta_rate_max = 0.03        # 每指残差每步增量上限
    delta_max = 0.30             # 每指残差累积上限 (指 i 深度 = clip(c+δ_i, 0, closure_max))

    # 臂: 每步界 = 标定 arm_residual_max × step_scale (末端 ~2cm × 0.25 ≈ 5mm/步);
    # 累积相对 q_pregrasp 钳 ±arm_dev_max. 抓取微调尺度, 不是搬运尺度 (v2 修订 5).
    arm_step_scale = 0.25
    # 累积偏差半径. 有 GraspPose prior 时腕位基本就是答案, 给太大 = 允许策略"漫游走开":
    # Grasp1(11.7cm 长条) 实测从 prior 起点漂走后再没回来 —— 候选率 6M 步恒 0,
    # 而零动作在 prior 位姿上本可拿到 4 垫同时接触. 0.12→0.05 (末端约 ±2cm 量级).
    arm_dev_max = 0.05           # rad

    # ================= Dexonomy GraspPose prior (A/B 的 B 组开关) =================
    # 非空 = 载入 make_prior.py 产出的 npz (物体输入系), 三处替换:
    #   ① 臂复位/参考位 q_pregrasp <- Dexonomy pregrasp 的 IK 解 (取代 affordance 悬停)
    #   ② 模板 c=1 锚点 q_close   <- Dexonomy 抓握手型 (取代通用 GENERIC_CLOSED)
    #   ③ 对齐目标 aff_local      <- Dexonomy 接触点质心 (取代 affordance 热图重心)
    # 奖励/判据/动作空间零改动 —— prior 只改"从哪出发、朝什么手型合、对准哪".
    grasp_prior_npz = ""
    # 物体 yaw (度). **≥0 = 钉死在这个角**, 由 screen_prior 的 Gate 1 给出 (`best_yaw`,
    # = 可达带里离**视频 yaw**最近的那个角). <0 = 退回旧的"按 IK 可达性自搜 yaw".
    #
    # ⚠ 2026-08-01 D1: 自搜 yaw 会把物体转到与视频不一致的姿态 (Grasp5 实测转了 250°),
    # 而人手参考轨迹是相机锚定钉死的 —— 物体一转, "从哪个方向伸过去、落在物体哪一面"
    # 的语义就整个废掉, 接近段任务 (pick_lift) 无法成立. 见 docs/PLAN_PICK_LIFT.md §1 D1
    # 与 docs/GRASPPOSE_SCREENING.md §Gate 1.
    prior_yaw_deg = -1.0

    # ================= 接近段 (pick_lift, 2026-08-02 方案 C) =================
    # 设计与全部实测依据见 docs/PLAN_PICK_LIFT.md §3; 证伪信号见 DESIGN_LOOP §2.10.
    #
    # 核心约束: 人手轨迹被系统性平移了 **14~16cm** (相机<->手标定偏差, 四条 clip 一致),
    # 所以**一切基于绝对位置的约束都是有害的** —— 没有管壁、没有位置模仿罚.
    # 人手轨迹只以**增量**形式进入: ① 前馈 q_ref[t]-q_ref[t-1]; ② 模仿罚罚"本步残差用量".
    approach = False             # 总开关. False = 完全退回"从 prior 位姿起步"的旧任务
    # ---- 组1 可行性课程 (2026-08-03): 先让接近**可行**, 再要求它快且像人 ----
    # 原设计的错: 模仿罚从第 0 步就满额 —— 那是"已经会接近"才成立的约束.
    # 可行性都没有先谈最优性 = 三次塌缩. 现在三条课程都由 **arrive_rate** 驱动
    # (不是 success_rate —— 相关能力是"到得了", 不是"抓得住"), 纪律同 gentle:
    # 慢速 EMA + 每 epoch 限速 + 棘轮(只朝收紧方向走, 不回头).
    # >0: q_ref 前面拼 K 帧 "默认站姿 -> 人手轨迹起点" 的关节插值 (PLAN D2: 出发点
    # 必须是 dexmate_joints 对称站姿; 人手重建第 0 帧已悬在物体附近, 不是合法起点).
    # gs 顺延 K, 预算/课程/RSI/前馈随 gs 自动缩放. 0 = 关 (= 08-03 之前的行为).
    stance_prefix_frames = 0
    # 方案一 (§2.15): 参考姿态通道混合 —— 位置轨迹 100% 保留人手形状, 腕姿态按走过
    # 弧长 smoothstep 从起点姿态 slerp 到 GraspPose 姿态, 0..gs 重解 IK.
    # 让转向与前进同时发生 (单次接近). 评测/录像必须带同一 flag (改 obs 的参考通道).
    orient_blend = False
    # 方案二 (§2.16): 锥形信任管 —— 允许偏离参考的半径随进度 φ 从 cone_r0 加宽到
    # ‖参考终点腕位−GraspPose 腕位‖+cone_margin (每条 clip 建 env 时自标定).
    # 管内零罚 (是预算不是引力); 接替增量模仿罚的相似度职能, 训练时配 --no_imit.
    cone_trust = False
    cone_r0 = 0.03               # m, 起点管半径 (紧贴参考)
    cone_margin = 0.05           # m, 终点余量 (保证管口罩住真实 GraspPose)
    w_cone = 1000.0              # 出管罚系数 (×超出量² m²; 出管 5cm ≈ -2.5/步)
    approach_extra_steps = 20    # 接近段预算的**终点** (gs + 这个)
    approach_extra0 = 120        # 预算的**起点**: 给足时间探索 (gs+120 ≈ 7.6s)
    eps_pos0 = 0.030             # 切换阈值起点 (松): 3cm
    eps_rot0 = 0.262             # 15°
    w_imit_ramp = 0.0            # 模仿罚的训练期系数 0→1 (乘在 w_imit0 上)
    curr_arrive_target = 0.5     # arrive_rate 到这个值时三条课程退火到位
    curr_rate = 0.005            # 每 epoch 最大变化量 (限速)
    r_arrive = 3.0               # **到位里程碑**: 切进抓取相位时一次性发
                                 #   势函数(密集,不可刷) + 到位奖(稀疏,一次性) = "持续接近给奖励"的安全实现
    # -- 残差权限"远松近紧": 手要自己走完那 16cm, 但接触前又必须是毫米级 --
    #    scale = 1 + u·(far-1),  u = clamp((d_pos - near)/(far_d - near), 0, 1)
    #    远端 3× -> 15mm/步 -> 11 步走完 16cm; 最短窗口 (Grasp3) 32 步, 3 倍余量.
    dyn_arm_far = 3.0
    dyn_d_near = 0.02            # m, 到这么近就收回 1× (毫米级)
    dyn_d_far = 0.15             # m, 到这么远给满 far 倍
    # -- 相位切换 (纯几何 + 滞回; 阈值来自 P0.0 容差曲线的 90% 档) --
    eps_pos = 0.0104             # m
    eps_rot = 0.038              # rad (2.18°)
    eps_hyst = 1.6               # 退出阈值 = 进入阈值 × 它 (滞回, 防抖)
    switch_vel_max = 0.05        # m/s, 腕速门槛
    switch_hold = 2              # 连续满足几步才切
    # -- 奖励 (量级推导见 PLAN §4.2; 抄近路不等式 20×0.02 > 6×0.02) --
    w_align = 6.0                # 对齐势差分权重, **必须恒定** (随相位变会破坏 telescoping)
    cap_align = 0.02             # m/步
    lam_rot = 0.174              # m/rad, **P0.0 实测** (5° ≡ 1.52cm), 是纯运动学力臂的 2 倍
    w_imit0 = 20.0               # 分/(m·步), 作用在**本步残差用量**上
    cap_imit = 0.02              # m/步
    imit_decay_p = 2.0           # w_imit(φ) = w_imit0·(1-φ)^p
    # -- 起步分布 (课程; 训练入口按 sr_ema 更新, 与 gentle 同一个钩子) --
    # direct_grasp_prob: 直接从 GRASP 相位 + prior 位姿起步的比例 = **今天的任务**,
    #   它同时是 RSI 的"从目标态起步"和**免费回归桶** (这一桶成功率必须仍 ~100%).
    # approach_t0_max:   接近相位起步时 t0 ~ U(0, 这个 × gs); 退火到 0 = 逼它从头做.
    direct_grasp_prob = 0.5
    approach_t0_max = 0.8

    # ================= 接触分数图 (P1a, 常驻) =================
    # 只在 verify 斜坡到顶那一刻取快照 (微抬升是物理干预 = 廉价因果筛子),
    # 按该垫的力占比加权. 存 (s,n) 两个计数, 读数 (s+1)/(n+2); n<n_min 记 unknown.
    score_map = True
    score_decay = 0.999          # 每次 decay_score() 调用衰减 (训练入口每 epoch 调一次)
    score_n_min = 5.0            # 低于它的点标 unknown, **不能当 0 分**

    # ================= 手型模板 =================
    # c 沿 GENERIC_OPEN -> GENERIC_CLOSED 插值. 三个锚点:
    #   OPEN(0.0) 张开 | PRESHAPE(0.6) 预包围 (指尖离表面 1.4~3.6cm) | GRASP(1.0) 合拢
    # 0.6 是在 ref builder 的 PreGrasp 位姿上实测扫出来的, 换物体/换 PreGrasp 要重扫.
    c_open = 0.0
    c_preshape = 0.6
    c_grasp = 1.0

    # ================= 复位随机化 =================
    # 物体 (不抖手): 课程 3mm -> 6 -> 10 -> 15mm, 一次只动这一个数.
    obj_jitter_xy = 0.003        # m, 均匀 ±
    obj_jitter_yaw = 0.0         # rad
    # 合拢深度 c0 ~ U(0, 上限): 接触态从第 0 步就被访问 (13.3M 步探索塌缩的教训).
    # v2.2: 0.6 -> 0.9 — 相当一部分回合从**已接触**状态起步 (RSI 的本义: 从目标态
    # 附近起步), 配合笨拙课程让"抓住"的收入在第 0 步就可达. 静置段物体钉住,
    # 复位轻穿透由 PD 手指柔顺侧解掉.
    closure_init_max = 0.9

    # ================= 接触力语义 (软垫↔物体) =================
    # 传感器给**力向量**; 量级 smoke 实测: 轻触 0.7~0.9N, 蛮力单垫 35N.
    pad_force_sign = -1.0        # 读数×它 = 垫施于物体的力 (smoke 已校准: 合拢时向心为正)
    squeeze_f_min = 0.2          # N, 低于它 = "贴着不用力", 不算有效接触
    squeeze_f0 = 3.0             # N, 有效力饱和点
    squeeze_f_max = 8.0          # N, 超出部分按牛顿数铰链罚 (防捏爆)
    w_over_force = 0.1
    obj_char_radius = 0.04       # m, 净力矩归一的特征臂长
    # 向心分的方向基准 (2026-07-30 换成局部法向):
    #   "normal" 每个垫最近表面点的**外法向**: Qᵢ = max(0, -F̂ᵢ·nᵢ) —— 力是否压进表面.
    #            形状无关, 对环/异形物体都正确.
    #   "center" 旧定义: 指向物体刚体原点. 环形物体的原点是**孔心**, 从外侧捏住时
    #            该方向与真实压入方向可差几十度 -> 系统性压低分数、甚至奖励"往孔里挤".
    # ⚠ 净力矩 tau 仍以物体原点(≈质心)为矩心 —— 那是"物体会不会转"的正确物理定义, 不改.
    cent_mode = "normal"

    # ================= 候选抓取 -> 微抬升验证 -> 成功 =================
    # 候选 (瞬时判据全部成立): ≥min_pads 垫有效力 + 质量 Q 达标 + 物体不滑不转不被推走
    success_min_pads = 4         # ≥4 垫 (第 5 垫由首触奖励鼓励, 不作硬门槛; 证伪见台账)
    # v2.5: 0.5 -> 0.3, 且候选保持 8->4 步. 向心分是代理量, 实测它筛掉的构型里
    # 有能通过物理验证的 (候选爆发窗的验证通过率 ~5%), 而验证已是免费重试 ——
    # 让物理裁判代替代理门槛筛选. 成功标准不变 (仍须实抬 1cm 物体跟随).
    grasp_centrip_thresh = 0.3   # 向心分达标线 (进验证的门, 不是成功的门)
    grip_slip_vel = 0.03         # m/s, 手物相对速度
    grip_rot_max = 0.5           # rad/s, 物体角速度
    candidate_hold_steps = 4     # 候选保持 0.2s 即进验证 (物理裁判便宜, 多试)
    # 验证: 硬编码腕升斜坡 (基类预解 q_lift, 2.5mm/级), 物体跟上且接触不丢 = 真抓取
    verify_lift_m = 0.010        # m, 抬多高 (q_lift 前 4 级)
    verify_ramp_steps = 8        # 斜坡步数 (0.4s)
    verify_hold_steps = 5        # 斜坡到顶后再保持 0.25s
    verify_min_rise = 0.005      # m, 物体至少升这么多 = 跟上了
    verify_fail_rise = 0.003     # m, 斜坡到顶+3步物体仍低于它 = 没抓住, 终止
    verify_min_pads = 3          # 验证期间至少保持的有效接触数
    # v2.3: 验证失败**不终止**, 退回 GRASP 重试 (小额罚). 失败若终止, "触发候选"就
    # 等于自断收入现金流 (验证期 99% 失败时), 实测策略学会故意压在候选线以下刷收入
    # 到超时 —— 候选率 66%→23% 崩落. 重试把候选的下行风险归零, 且成功样本翻倍.
    r_verify_fail = -0.2         # v2.5: -1 -> -0.2, 尝试要便宜 (一回合可能试 5+ 次)

    # ================= 奖励权重 =================
    # -- 密集: 接近 (有效接触数 < min_pads 时) --
    w_pad_approach = 1.0         # 逐垫→物体表面 距离进度
    # -- 稀疏: 接触 --
    r_pad_touch = 0.5            # 每垫首次**有效**接触 (力≥f_min; 贴着不用力拿不到)
    # -- 密集: 好接触状态的小额持续收入 (v2.1 引入; v2.4 改按质量分 Q 发) --
    # v2.1 教训: 没有持续收入 → 悬停塌缩 (pads 3.1→0.35).
    # v2.4 教训 (run 04-12-55): 收入按**向心分**发是发错了对象 —— 从上方下压的构型
    # 五力全朝下指向中心, cent 很高但合力不对称度≈1, 一抬就掉; "伪向心"既过候选门槛
    # 又赚满收入, 而能抬起来的对握 (力互相抵消) 没有额外优势, 验证通过率钉在 1%.
    # 改: 收入 = w × max(0, Q),  Q = cent − 0.5·imb − 0.5·τ_n —— 下压式现金流归零,
    # 只有对称力封闭赚钱; 且**验证段照发** (进验证不再有机会成本).
    w_cent_income = 0.1
    # -- 密集: 抓取质量**进度** (v2 修订 2: 不给绝对值, 原地保持不刷分) --
    #   Q = cent − λ_imb·r_imb − λ_tau·τ_n;  r = w_quality·clip(ΔQ, ±q_prog_clip)
    w_quality = 0.5
    quality_lam_imb = 0.5        # 合力不对称度折进 Q (原 w_netF, 避免双重计费)
    quality_lam_tau = 0.5        # 净力矩折进 Q (原 w_netT)
    q_prog_clip = 0.2
    # -- 年金: 候选判据完整成立时每步发 (成功终止天然封顶 ~(8+13)×0.3≈6) --
    w_hold = 0.3
    # -- 笨拙课程 (v2.2, run 02-33-50 的修复): 轻柔类惩罚按成功率门控涨价 --
    # 诊断: 尝试抓的过渡期每回合 -22 (超力/物体乱动/被推走终止), 而收入 +0.01/步 ——
    # 过渡期成本淹没稳态收益, 三次塌缩到"悬停"都是它. 解法 = repo 自己的 ConTrack
    # 式退火反着用: gentle = 0.2 + 0.8·min(sr_ema/0.2, 1), 乘在 w_obj_move / w_push /
    # w_over_force 上; push 终止距离从 10cm 随 gentle 收紧到 4cm. 学费打二折,
    # 会抓了 (sr_ema→0.2) 恢复原价 —— 最终策略仍按全额惩罚结算, 不牺牲最终质量.
    gentle_init = 0.2            # 训练入口每迭代按 sr_ema 更新 env.gentle; 评测/冒烟用 1.0
    push_fail_dist_loose = 0.10  # m, gentle=0.2 时的推走终止距离 (随 gentle 线性收紧)
    # -- 惩罚: 把物体弄动 (GRASP 段; 验证段物体本来就该动, 不罚) --
    w_obj_move = 1.0             # 接触后物体线速度 (×gentle)
    w_obj_rot = 0.2              # 接触后物体角速度
    w_push = 5.0                 # 候选形成前把物体推走的位移惩罚 (二次, ×gentle)
    # -- 惩罚: 碰撞 (必须几何式: 手↔桌物理碰撞被过滤; 自碰撞开关会臂自锁) --
    w_table = 20.0
    table_margin = 0.005         # m
    # ---- P0.3 全臂几何碰撞 (2026-08-02): 接近段专用, 抓取任务里臂不动所以不触发 ----
    # ⚠ 权重**先默认 0**: 打开之前必须先用诊断量 diag/self_gap_cm 与 diag/arm_table_gap_cm
    #   量出参考轨迹与抓握位姿本身的真实间隙, 阈值定在它**之下** —— 否则会去罚参考轨迹
    #   自己, 那是自相矛盾的目标 (同 MANUAL §6.5 "r_imit 的腕参考要加上抬升量"那个坑).
    # 2026-08-02 已用 check_ff 标定并开启: 参考轨迹实测最小间隙 臂离桌 6.24cm /
    # 臂↔躯干 12.25cm, 阈值 3cm / 8cm 都在其下, 不会去罚参考轨迹自己.
    w_arm_table = 20.0           # 臂连杆撞桌 (与手的 w_table 同量级)
    arm_table_margin = 0.03      # m, 臂连杆**原点**离桌面的余量 (原点在连杆内部, 比手大)
    w_self = 5.0                 # 臂↔躯干/头/另一条臂 的间隙罚 (单步最多 -1.0)
    self_margin = 0.08           # m, 连杆原点两两距离的下限
    self_pen_cap = 0.2           # 单步上界 (台账铁律: 每个铰链都要有上界)
    table_crash_depth = 0.02     # m, 深穿终止 (防"从桌下托")
    w_finger_cross = 10.0        # 相邻指远端节距离铰链罚 (拇指不参与)
    finger_cross_dist = 0.012    # m
    # -- 里程碑 --
    r_candidate = 3.0            # 候选抓取形成 (进入验证段, 一次性)
    # v2.7: 20 -> 40. 全价 (gentle=1.0) 下实测 sr 34.5%→9%: 尝试期望 p·(9+20) − 全价
    # 成本(~5-7) 在 p<0.25 时为负, 策略理性剪除抓取 —— 慢课程只是推迟了这一天.
    # 提高奖池: p·(9+40) 在 p≈0.12 即转正, 且不碰惩罚 (最终行为标准不变).
    r_success = 40.0             # 通过微抬升验证
    # -- 正则 (量级压在任务奖励的 1~5%) --
    w_act_rate = 0.02
    w_qvel = 0.01
    w_qvel_hard = 0.1
    qd_soft_arm = 1.5            # rad/s
    qd_soft_fin = 4.0            # rad/s
    qvel_hard_cap = 10.0         # 铰链平方和钳制 (物理尖峰会到几百, 不钳淹没任务信号)
    w_torque = 0.005
    w_time = 0.002
    # -- 失败 (终止) --
    r_drop = -10.0               # 掉落/抛飞/深穿桌
    r_pushed_away = -5.0
    push_fail_dist = 0.04        # m
    fall_below = 0.03            # m
    max_obj_height = 0.25        # m (物体高于桌面这么多 = 被抛飞; 验证只抬 1cm, 不冲突)
    lift_height = 0.05           # m; 只用作物体高度观测通道的归一分母

    # ================= episode =================
    # PREGRASP 那一档在 env 里按 clip 的 gs 覆盖成 gs+approach_extra_steps (approach=True 时);
    # approach=False 时保持 0 = 跳过该相位 (旧任务行为不变).
    phase_timeout = (0, 120, 20, 0, 0, 0)   # GRASP 120; 验证 20
    max_phase = Phase.LIFT       # 走完验证段 = 完整任务


def apply_grasp_prior(cfg, npz_path, yaw_deg=None, approach=None):
    """所有入口统一走这里挂 prior —— 三件事必须一起做, 漏一件就是静默错误.

      ① grasp_prior_npz
      ② **pregrasp_align = False** (D7): 有了定义在物体上的 GraspPose, 就不再需要
         "悬停在 affordance 上方 8cm"那个人造 PreGrasp; 而它会额外平移整条腕轨迹
         (Grasp3 实测 13.68cm), **破坏相机↔人手↔物体的 XY 刚体约定**.
      ③ prior_yaw_deg (D1): 钉死在 screen_prior Gate 1 给的 best_yaw, 不许自搜.
    """
    cfg.grasp_prior_npz = npz_path
    cfg.pregrasp_align = False
    if yaw_deg is not None and yaw_deg >= 0:
        cfg.prior_yaw_deg = float(yaw_deg)
    if approach is not None:
        cfg.approach = bool(approach)
    # obs 维度由开关算出来 (参考通道常开, 不随 approach 变):
    #   144 基数 + 7×参考前瞻帧数 + 5×参考接触标签
    cfg.observation_space = (144 + 7 * int(cfg.ref_look_frames)
                             + (5 if cfg.ref_contact_obs else 0))
    return cfg
