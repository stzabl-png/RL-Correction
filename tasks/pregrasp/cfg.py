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

import os

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



def _load_finger_residual():
    """逐关节手指残差界 (rad, GENERIC_JOINT_ORDER 序), 由 calib_finger_residual.py 标定。

    标定口径与臂完全相同(ε=2cm 指垫位移的 95 分位), 所以 ×step_scale 之后
    臂和手指都是每步 5mm 的笛卡尔尺度。缺文件时回退到**最紧的那一档**给所有关节。
    """
    import json as _json
    import os as _os
    p = _os.path.join(_os.path.dirname(__file__), "finger_residual_bound.json")
    try:
        with open(p) as f:
            return list(_json.load(f)["bound_rad"])
    except Exception:
        print(f"[cfg] ⚠ 找不到 {p}, 手指残差界回退到均一 4.30°(最紧档) "
              f"—— 先跑 tasks.pregrasp.calib_finger_residual")
        return [0.0751] * 22



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
    # 13 = [Δq_arm(7), a_c(1), a_δ(5)] (closure 模式)
    # 29 = [Δq_arm(7), Δq_hand(22)]     (joints 模式, 见 hand_action_mode)
    #   ⚠ 单臂口径。双臂同训时每臂 29 ⇒ 58, 但**双臂 env 尚未实现**
    #     (本 env 只控 cfg.hand_side 一只臂, 另一只臂仅作碰撞障碍参与间隙罚)。
    # ★ Approach-only = **7 维(只有臂)**。接近段 `hand_gate=0` 把 a[7] 和 a[8:13] 整段
    #   乘零(手指全程张开是这个任务的定义), 留着就是 6 个**空转维度**: 白吃探索噪声、
    #   占策略容量、还会被动作平滑罚扫到。这个任务的意义就是"判据干净、失败原因唯一",
    #   留死维度与它矛盾。(2026-08-16 用户裁定)
    action_space = (7 if os.environ.get("RL_APPROACH_ONLY") == "1"
                    else (29 if os.environ.get("RL_HAND_JOINTS") == "1" else 13))
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
    # 151 = closure 模式(a=13, ref_look=1)的观测维; joints 模式(a=29)实测 **206**:
    #   动作通道 13->29 (+16) + 手指内部状态 5->44 (+39)
    #     其中 44 = 逐关节累积残差 22 + 逐关节参考跟踪误差 22
    #     (要跟 22 个关节的参考, 就得让策略看见自己每个关节差多少)
    #   实际值以 `_obs_base()` + 前瞻/接触指集为准, 这里只是 closure 的基数。
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

    # ---- 手指自由度的**距离门控** (2026-08-16, 用户裁定) ----
    # 用户原话:"在接近物体 Approach 的时候不应该给太多自由给到手指, 随着与物体之间的
    # 距离越来越近了, 手指才更多的探索学习。"
    #
    # 为什么它比"按相位开关"更根本(实测依据):
    #   相位切换点 gs 取自**接触标注的起点**, 而 pour/17 的标注在 f8 就触发,
    #   **那一刻指尖离瓶还有 7.8cm** —— 于是策略照着参考做, 就是在**空气里把手握起来**,
    #   然后握着拳头往前挪。相位门是二值的, 它完全信任标注; 距离门**不信任标注**,
    #   它信任当下量到的几何, 所以标注错了也不会让手指提前动。
    #
    # 实现: hand_gate 额外乘一个 0→1 的斜坡, 由**五垫到物体表面的最近距离**驱动。
    #   d ≥ far  -> 0   手指冻结(连参考合拢斜坡一起冻, 不只是策略动作)
    #   d ≤ near -> 1   完全放开
    #   中间线性
    # ⚠ 口径: 这个 d 是 `_pad_dists()`, 即 **elastomer 的 link 原点**到表面,
    #   而胶垫实体离该原点中位 2.4cm(见 env._pad_dists docstring)。所以阈值要按
    #   这个口径定, 不能拿"指尖离物体几厘米"直接填。
    #   实测参照: GraspPose 处该值 11~20mm; 标注触发点 f8 处约 10cm。
    # ⚠ 2026-08-16 修:门控的驱动量从"**指垫**到物体距离"改成"**腕**到物体距离"。
    #
    # 原写法是**死锁**:指垫要靠近物体必须先合拢手指, 而手指要能合拢又要求指垫已经靠近。
    # 张开的手在预抓位时指垫离物体 7~10cm ⟹ 门几乎全关 ⟹ 手指冻死 ⟹ 永远靠不近。
    # 实测:`Grasp3_CTRL_13dim`(冠军 clip + 冠军配方 + 13 维)因此**整条跑废** ——
    # `diag/finger_gate` 首 0.095 / 最小 0.007, 门控中位 0.075(冠军 1.44M 就到 0.8),
    # 13.8M 步 4 次确定性评测全 0。倒水那条物体大、指垫本来就近(门 0.84~0.99), 没暴露。
    # **腕位是手臂控制的量, 与手指开合无关**, 所以挂在它上面没有循环。
    #
    # 阈值不写死厘米数, 按该 clip **自己的 GraspPose 抓取距离** d_g 自动标定:
    #   d <= d_g × near_k -> 1 完全放开;  d >= d_g × far_k -> 0 冻结; 中间线性
    finger_gate_near_k = 1.25     # 到抓取距离的 1.25 倍以内 = 完全放开
    finger_gate_far_k = 2.50      # 超过 2.5 倍 = 冻结
    # 逃生阀 RL_FINGER_GATE=0 关掉 = 退回旧的纯相位门(消融对照用)
    finger_gate_on = os.environ.get("RL_FINGER_GATE", "1") != "0"

    # ================= 手部动作空间 (2026-08-15, 用户裁定"手完全放开") =================
    # "closure" (默认, 旧行为不变): a = [Δq_arm(7), a_c(1), a_δ(5)] = 13
    #     手指被压缩成"一个合拢旋钮 + 5 个微调" —— 沿 q_open->q_close **一条射线**走。
    # "joints": a = [Δq_arm(7), Δq_hand(22)] = 29
    #     逐关节残差叠在合拢模板之上。合拢模板仍按**参考速率**自行推进(参考照样会合拢),
    #     策略对手指的一切意图都走那 22 个关节。
    #
    # 为什么必须换(实测, 见 POUR_TRAINING_DESIGN §15):
    #   pour17 瓶: GraspPose 处五垫离瓶面 3.9~9.9mm; 把合拢旋钮**拧到上限 c=1.25**,
    #   四指扎进 3.4~9.5mm 而**拇指只从 9.9mm 走到 3.8mm, 始终碰不到**。
    #   原因是几何的: 拇指要够到瓶子必须**离开那条射线**, 而合拢参数化不允许。
    #   放开 22 关节后, 两只手都能让五垫同时贴上(瓶 −0.1~−0.7mm / 杯 −0.7~−1.0mm),
    #   所需关节改动中位仅 8.6°/2.3° —— GraspPose 本身是好的, 差的就是这点自由度。
    #
    # ⚠ a_c / a_δ 在 joints 模式下**取消**, 不做保留: 留着就与 22 个关节残差重复控制
    #   同一件事, 冗余动作维度会伤训练。"压得更紧"没有丢 —— 那现在是给屈曲关节加正残差,
    #   而且是逐关节的。
    # 逃生阀 RL_HAND_JOINTS=1 打开 22 关节模式(与仓内 RL_* 消融开关同风格)
    hand_action_mode = "joints" if os.environ.get("RL_HAND_JOINTS") == "1" else "closure"
    finger_residual_max = _load_finger_residual()   # 22, 逐关节标定
    # 每步增量 = 标定界 × 它 (与臂的 arm_step_scale 同一个系数, 两者由构造可比)
    finger_step_scale = 0.25
    # 累积上限 = 标定界 × 它. 4 ⇒ 约 4 步走满, 指垫最多偏离模板 2cm
    # (臂那边 arm_dev_max/arm_residual_max = 0.05/0.0154 ≈ 3.25, 同量级)
    finger_dev_scale = 4.0

    # 臂: 每步界 = 标定 arm_residual_max × step_scale (末端 ~2cm × 0.25 ≈ 5mm/步);
    # 累积相对 q_pregrasp 钳 ±arm_dev_max. 抓取微调尺度, 不是搬运尺度 (v2 修订 5).
    arm_step_scale = 0.25
    # 累积偏差半径. 有 GraspPose prior 时腕位基本就是答案, 给太大 = 允许策略"漫游走开":
    # Grasp1(11.7cm 长条) 实测从 prior 起点漂走后再没回来 —— 候选率 6M 步恒 0,
    # 而零动作在 prior 位姿上本可拿到 4 垫同时接触. 0.12→0.05 (末端约 ±2cm 量级).
    arm_dev_max = 0.05           # rad

    # 方案C (2026-08-23 用户裁定): 臂改 **绝对参考 + 有界累加残差**, 与手指同构。
    #   理由: cuRobo 参考已是全局可行解, 残差要干的只是"略微避蹭"和"到位后再压紧"
    #   (95→100 分那一段), 不需要"改走法"的自由度。而差分前馈是个只被关节限位夹的
    #   积分器, 那点自由度买不到什么, 却是左臂漂到 49cm 外的结构性前提。
    #   关掉(False) = 原差分行为, 逐位不变, 供 A/B。
    # ===== 分段探索门控 (2026-08-24 用户编排) =====
    # 探索进入系统的路径 = sigma × 残差步长(逐关节) × 门控。门控置 0 = 该组该段零探索,
    # 比 entropy_coef(全局标量, 给不了逐维)干净且是硬的。八段顺序同参考 seg_names:
    #   stance_to_5cm / hold1 / root_bend / advance / hold2 / close / squeeze / hold3
    # 用户编排: ①臂动指冻(留探索避障) ②只有拇指动 ③臂动指冻(推进避障) ④⑤全开
    seg_gate = False
    # 9 段版 (侧移参考): stance hold1 root_bend advance **insert** hold2 close squeeze hold3
    #   insert = 新增的"侧向移回 0.5cm", 属阶段③推进 ⟹ 臂有探索、指全冻
    seg_arm_scale   = (0.3, 0.3, 0.0, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0)
    seg_thumb_scale = (0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    seg_other_scale = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    # 合拢前"外壳不许碰物体": 手+l7/l8/ee 到物体表面点的最近距离低于该余量就罚
    # (余量必须从参考自身标定, 否则又在罚参考)
    # ⑤ squeeze 段抓力奖励 (2026-08-24 用户第⑤条): pad_near(垫→表面势差) +
    #   grip_pot(quality 势差) 双项, 均 earn-only。squeeze_row0 由段表推导。
    # FC 逐指目标点取在哪个姿态: False=grasp(原), True=squeeze(参考的真实终态)
    fc_target_squeeze = False    # (已被 fc_target_ref_end 取代, 保留供对照)
    # ★ FC 目标点在**参考终态**下测 (臂=retract_path[-1], 指=_fin_ref_path[-1])。
    #   硬验收: 零动作下 _fc_d ≈ 0。见 2026-08-24 台账。
    fc_target_ref_end = False
    fc_settle_steps = 0          # 拍照前让自由物体沉降的帧数
    fc_post_settle = int(os.environ.get("RL_FC_POST_SETTLE", "0"))
    fc_play_ref = int(os.environ.get("RL_FC_PLAY_REF", "1"))
    # 1: 目标点由"逐行播一遍参考"得到, 而非瞬移到末行 (唯一与运行同口径的拍法)
    # 摆手**之后**再静置的帧数: 消除"瞬移捏紧手"的穿透冲量, 让目标点记录平衡态
    squeeze_grip_w = 0.0         # >0 启用
    squeeze_row0 = 0             # squeeze 段起始行 (播放行空间), 由 train.py 推导
    # ---- 握力信任标量 g (2026-08-25 用户裁定 + 与 RL_Pour 联合设计) ----
    #   语义: "对这个抓握的信任度" 0→1。**不给任何握力奖励**(用户: "不要给力了,
    #   就是原本的接触就行") —— 捏紧由参考自身 squeeze + g2 指尖目标点完成。
    #   g 的唯一职责: 驱动前段密集项衰减 (权重 ×(1−g))。交互段的快档与滑移纯罚
    #   在 tasks/pour 侧 (RL_Pour 的 v10 配方: 滑移连续梯度 + carry_squeeze 动作通道)。
    grip_g = False               # 总开关
    grip_g_up = 1.0 / 200.0      # 快档步进 (交互段, RL_Pour 用); 本侧用 1/4 档
    grip_g_slow_frac = 0.25      # ④ 捏紧段慢档系数 —— **弱证据**: 没有外力时不滑
                                 #   本来就不滑, 所以这一档实质是按时间累积, 故只给 1/4
    grip_g_down_mult = 10.0      # 滑移时的下降倍率 (升慢降快: 信任慢建、一滑就掉)
    grip_g_slip_cm = 0.5         # 滑移判据: 腕系下物体位置相对"抓形已成"快照的偏移
    grip_g_decay_terms = ("align", "imit")
    # ---- 相位奖励日程表 (2026-08-26, 三方定稿的 ①②③ 行) ----
    #   病理 (RL_Pour E2EL 2M 实测): arrive 门修好后, arrive_rate 0.279→0.045→0.016,
    #   **策略在学"躲门"** —— 进门即入抓取期, fail/push/obj_disturb/tilt 罚组全开,
    #   到位的净收益为负 ⟹ 避门=避罚。
    #   ★ 关键设计: **不能用开关, 必须用渐入**。任何按相位硬切的奖励改变都会在边界
    #   造一个断崖, 而策略只要发现门那边更差就学着不过门 —— 硬切的阶段表本身就是
    #   新断崖, 加大 arrive 一次性奖金也只是把坑填浅(且深度会变, 靠调参猜)。
    #   F 线免疫纯因 approach_only **整体移除**了这一组, 门后没有段。
    rew_sched = False            # 总开关 (关闭时行为逐字节不变)
    rew_sched_pre = (            # PREGRASP 段置零的项 = 抓取期的**接触质量**罚
        "over_force", "obj_move", "obj_rot", "tilt", "push", "finger_cross",
    )
    #   ★★ 2026-08-26 判死: 原来这张表里有 "fail" —— **致命终局罚绝不能打折**。
    #   RL_Pour r5 实测: PREGRASP 段 fail×0 ⟹ 拍桌终止**零成本**, 而活着每步要付
    #   imit/align 的负期望 ⟹ **最优策略 = 开局自杀缩短回合**
    #   (term/table_crash 0.000→0.160, d_pos 2.3→19cm, res_used 0.09→0.28cm/步
    #    = 在主动推离; 而 Mean −12→−0.4 的"改善"全是回合变短的会计幻觉)。
    #   **规则: 终局/致命罚永不参与奖励日程表 —— 死刑不参与课程。**
    #   更深一层: 自杀通道能成立的前提是"活着是负期望"。见 preflight ⑦ 的
    #   "零动作基线合计 > 0" 硬项 —— 那才是根上的闸。
    rew_sched_ramp = 60          # 进 GRASP 后把上面这组**线性渐入**的步数 (0=硬切)
    rew_sched_grasp_scale = {    # 进 GRASP 后接近组降权 (定稿③行)
        "align": 0.3, "imit": 0.0,
    }
    #   ★ 只衰减**逐步密集**的接近项。one-shot 项 (arrive/里程碑) 不衰减 ——
    #   它们本来就只发一次, 衰减它们等于改判据而不是改塑形。
    #   选 align/imit 的依据: F 线退化验尸里, 任务饱和后仍在涨且与抓握质量对冲的
    #   正是这两类 (align +33%, 而 pad_hold −53% / succ_hold −68%)。
    pre_close_shell_cm = 0.0     # >0 启用; 建议值由预检的参考实测给出
    arm_abs_res = False
    arm_abs_dev = 0.05           # rad, 绝对模式下臂残差逐关节上限 (默认同 arm_dev_max)

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
    # ============ PickAndPlace: 搬运+放置段 (§2.19, 2026-08-04) ============
    # 开 = 抬升验证通过后不结束, 接 TRANSPORT(跟随人手搬运参考) + PLACE(脚本化松手).
    # 成功判据变为: 物体落在人手示范放置位 ±place_tol 且静置. 搬运目标用**相对量**
    # (进搬运时的物体位 + 人手腕位移), 对重建的绝对偏移免疫.
    place_task = False
    place_tol = 0.03             # m, 放置判定半径
    place_settle_steps = 10      # 松手+达标后静置步数 (0.5s) 才算成功
    place_release_rate = 0.05    # 松手斜坡: closure 每步下降 (~0.9s 全开)
    w_carry = 6.0                # 搬运段物体跟踪势差分权重 (与 w_align 同级)
    # 搬运段残差降档: 热启动策略在到达态学的"减速稳住"本能会抵消搬运前馈 (探针实测
    # 腕只走 46% 参考位移), ff 必须主导, 残差只留防滑微调 (§2.19 P1a 干预)
    carry_res_scale = 0.25
    carry_extra_steps = 30       # 搬运预算 = (re-gs) + 这个
    place_budget_steps = 60      # 放置段预算 (松手 + 静置 + 余量)
    approach_extra_steps = 20    # 接近段预算的**终点** (gs + 这个)
    approach_extra0 = 120        # 预算的**起点**: 给足时间探索 (gs+120 ≈ 7.6s)
    eps_pos0 = 0.030             # 切换阈值起点 (松): 3cm
    eps_rot0 = 0.262             # 15°
    w_imit_ramp = 0.0            # 模仿罚的训练期系数 0→1 (乘在 w_imit0 上)
    curr_arrive_target = 0.5     # arrive_rate 到这个值时三条课程退火到位
    # ---- 公差回退闸 (2026-08-16 用户裁定: "真学坏了才退, 不是一有波动就退") ----
    # 原实现放松是**瞬时**的(收紧才限速), 而驱动量 arrive_rate 剧烈抖动(实测 0.07~0.58),
    # 于是噪声在开车: 成功率掉 -> 公差放松 -> 学到"贴桌勉强够角度"的糙解 -> 精度上不去
    # -> 再掉。实测 eps_rot 在 5.6~10.5° 之间甩, 从不收敛。
    curr_retreat_frac = 0.7      # 慢 EMA 掉到历史最好的这个比例以下才算"退化"
    curr_retreat_epochs = 10     # 连续这么多 epoch 都退化才准放松 (≈32万步)
    # 起点课程(ratio/far)每次回退的幅度。实测 far 0.66->1.0 时成功率 0.87->0,
    # 而棘轮让它退不回去 ⟹ 永远卡在最难档 + 0 成功率。0.15 ≈ 退回上一个能学的难度。
    curr_start_retreat_step = 0.15
    # ---- 自生成起点池 (2026-08-16 用户裁定; 取代退避课程) ----
    # 起点 = 策略自己真到过的状态, 按离目标距离分 7 档存池, 每回合从各档抽。
    #   "" = 关 | "uniform" = 各档等权 | "mastery" = 权重∝(1-掌握度), 带地板
    # ---- cuRobo 前馈参考 (2026-08-17 用户裁定: 在最简配方上加两段轨迹指引) ----
    # npz 含 right_q/left_q (T,7): 先右后左的整条规划(含保持段), 由
    # view_curobo_plan --save_plan 在**两道验收都过**后导出。
    curobo_ref_npz = ""
    curobo_ref_stride = 2        # 324 帧 ÷2 = 162 步, 塞进 250 步的回合预算
    # 近端冻结前馈: 该侧腕距目标 < 此值(cm)时 ff 置零, 收尾交给纯残差(=Minimal_dyn 已验证模式)。
    # 动机: cuRobo 左手抓取段贴着杯壁走, ff 每步都把手往杯子里送(L pushed≈1.0),
    # 残差近场 0.3x 打不过它; 冻结后 ref_q_prev 照常跟踪, 退出冻结区不会跳变。0=关。
    curobo_ff_freeze_cm = 0.0
    # 弱回拉锚定强度 (0=关): 每步把 q_cmd 往 retract_path[t] 拉这个比例, 只在冻结区外生效。
    # 治差分前馈的漂移累积(保持段残差漂移会平移整条后续路径, 见 env.py 注释)。
    curobo_ff_pull = 0.0
    # ---- L5: 边靠近边合指 (2026-08-17 用户三裁定, 台账"L5 设计定稿") ----
    # 开=接近段冻结区内解锁手指通道, 合拢参考追踪 c(d)=c_grasp·clamp((fz−d)/fz,0,1)
    l5_couple = False
    l5_couple_w = 2.0            # 软指标罚 |closure − c_ref(d)| 的权重 (刻意小, 裁定②)
    start_pool = ""
    start_pool_cap = 256         # 每档环形缓冲容量
    start_pool_floor = 0.08      # mastery 模式: 每档权重地板 (防灾难性遗忘)
    start_pool_stance_floor = 0.20   # 站姿档地板更高 —— 它就是评测分布, 不能被饿着
    curr_rate = 0.005            # 每 epoch 最大变化量 (限速)
    r_arrive = 3.0               # **到位里程碑**: 切进抓取相位时一次性发
                                 #   势函数(密集,不可刷) + 到位奖(稀疏,一次性) = "持续接近给奖励"的安全实现
    # -- 残差权限"远松近紧": 手要自己走完那 16cm, 但接触前又必须是毫米级 --
    #    scale = 1 + u·(far-1),  u = clamp((d_pos - near)/(far_d - near), 0, 1)
    #    远端 3× -> 15mm/步 -> 11 步走完 16cm; 最短窗口 (Grasp3) 32 步, 3 倍余量.
    # 裁定B3 (2026-08-18): PreGrasp = 抓姿整手沿"接触质心→腕"掌背法向外移这么多 cm
    # (掌心锚点, 两手指尖间隙对称)。改这里即可, env/bimanual/规划目标三处统一消费。
    pregrasp_palm_cm = 1.0   # 2026-08-18晚 用户裁定: 5→1cm, 腕送到门口, 剩下交给手指RL
    # PreGrasp29 任务 (2026-08-18 晚): 靶点=PreGrasp + 22指解锁 + 到位前手指软抑制
    pregrasp29 = False
    w_fin_quiet = 0.02
    # Phase2-RL (2026-08-18 晚, 用户裁定): 名义斜坡退役 —— 闩后 PreGrasp→GraspPose
    # 由 RL 自走, GraspPose 只作指引(离散里程碑+成功奖)。判据/奖励参数:
    pregrasp_phase2 = False
    phase2_steps = 50            # (斜坡遗产, 仅供将来消融; RL 版不用)
    phase2_fin_eps = 0.26        # 成功判据: 22 关节平均偏离抓姿指型 < 15°
    phase2_hold = 5              # 腕+指同时达标保持步数
    # m1 里程碑的腕距门 (cm)。**派生量, 不是自由参数** (简化一, 2026-08-18 用户核准):
    #   m1 门 = pregrasp_palm_cm (预备位→抓姿靶距)
    #         + eps_pos*100     (到位球半径)
    #         + eps_pos*100     (到位闩后漂移余量)   → 现值 1+1+1 = 3cm
    # 语义 = "到位的手一定拿得到的进场券"; 把关交给 m2 与最终判据。
    # 教训: 里程碑阈值必须罩住上一级判据的驻留区 —— 写成派生式后, 改预备位距离
    # 或到位公差时它自动跟随, "改了一处忘了另一处"的零余量事故 (2cm 时代) 结构性绝迹。
    # None = 按上式派生; 显式数值 (或 --m1_cm) 仍可覆盖, 0 = 删除 m1 (简化二)。
    phase2_m1_cm = None
    # m2 派生阶梯步长 (裁定C, 2026-08-19): 从各侧起始指型偏差 d0−step 起每 step 一级
    # 挂一次性小糖 (+bonus/2), 末级=20° 大门 (+bonus)。阶梯本身按 d0 派生, 零魔法数。
    phase2_ladder_step_deg = 5.0
    # ================= Carry (Pour P0+P1 MVP, 2026-08-19 用户批准) =================
    pour_carry = False           # PourCarryEnv 开关 (train --pour_carry)
    carry_npz = ""               # build_carry_ref 产物 (物体目标序列+双臂路径)
    carry_slip_pos = 0.04        # slip 判负: 腕系物体相对位姿漂移 (m)
    carry_slip_rot = 0.5236      # 30° (rad)
    carry_lost_m = 0.05          # 跟丢判负: 物体离目标 (m)。2026-08-23 用户裁定 20→5cm:
                                 # 与行进门(carry_adv_pos=5cm)重合 ⟹ 离轨即死, 不再留
                                 # "停薪拉回"缓冲带; 误伤率待 pour_diag 成功回合峰值分布验证
    carry_succ_pos = 0.05        # 成功: 末帧物体位置误差 (m)
    carry_succ_rot = 0.5236      # 成功: 末帧物体姿态误差 (rad)
    carry_succ_hold = 5          # 成功保持步数
    carry_w_track = 1.0          # 追踪奖励权重 (每侧)
    carry_w_slip = 0.05          # slip 罚权重
    # -------- CARRY3 进度时钟 (2026-08-19 用户批准) --------
    carry_progress = False       # True = 时钟按进度走 (握稳才启动/跟上才前进/按里程发钱)
    carry_adv_pos = 0.05         # 行进粗容差 pos (m) = CARRY2 实测健康误差 p95×1.4
    carry_adv_rot = 0.7854       # 行进粗容差 rot 45° (从属: 必须松于 30° 成功门)
    carry_start_win = 5          # 启动判据: rel 位姿逐步变化连续 K 步很小
    carry_start_deadline = 60    # 启动限期 (步): 超时判负 (防赖在起点)
    carry_ms_tol = 0.025         # 高置信里程碑容差 = 行进容差一半 (派生)
    # ---- CARRY4 (2026-08-19 用户裁定: 实验A=倾角里程碑, 实验B=低置信小糖宽门) ----
    carry_tilt_ms_deg = ()       # 扩展A: 瓶体倾角几何里程碑(度), 置信度无关;
    #                              峰值实测 71° ⟹ 档位必须 <71 (30,60)
    carry_tilt_ms_bonus = 2.0    # 倾角里程碑一次性大糖 (与高置信里程碑同级)
    carry_ms_low_idx = ()        # 扩展B: 沙漠低置信帧里程碑 (行号)
    carry_ms_low_bonus = 0.7     # 低置信=小糖 (~1/3), 门用行进容差(宽) 不卡精度
    carry_squeeze = False        # 承载段通用公约: q_close ← squeeze 模板 (收紧对抗
    #                              倾转扭矩, 收紧幅度=策略的合拢通道, RL 自学何时挤)
    # ---- FC-D (2026-08-20 用户三裁定: Pose0起手学构型/撞桌即Fail/29维全参考) ----
    fin_ref_track = False        # 指参考四段: open→Pose1(步数斜坡)→hold→阶梯(腕距查表)
    #                              →合拢(限速时间斜坡); 全挂物理信号, 无新时钟
    fin_shape_steps = 40         # 构型段步数 (settle 后 open→pregrasp[0])
    fin_close_steps = 30         # row5→grasp 合拢斜坡步数 (慢合拢的参考侧保险)
    fin_dyn_near = 0.3           # 指残差近物缩放 (腕距≤3cm; ≥10cm=1.0 线性)
    fin_dyn_close = 0.2          # 合拢段指残差再压一档 (慢合拢的探索侧保险)
    w_fin_track = 0.05           # 指偏离参考罚 (取代 fin_quiet 语义)
    w_shape_ms = 1.0             # 构型完成一次性糖
    shape_tol_deg = 15.0         # 构型判定: 指距 Pose1 均值角
    w_pad_first = 0.5            # 逐垫首触一次性 (轻触: 物体速度<0.05 才发)
    w_pad_hold = 0.02            # 接触垫持续分/步 (到位后、candidate 前, 防挂机窗口)
    w_cent_shape = 0.05          # 向心塑形/步 (受力方向指向物心)
    w_stable_ms = 5.0            # 稳抓大糖: 现成 candidate 判据 (≥4垫&向心&保持4步)
    w_succ_hold = 0.05           # 单侧过验证后保持稳抓的逐步分 (2026-08-20 用户裁定:
                                 #   succeeded 侧不躺平, 稳稳握住等另一侧; 双侧才终止)
    w_toppled = 0.0              # 拍倒显式罚 (FC-D RSI-B 线 2026-08-20): approach 模式
                                 #   fail 整项被 APPROACH_OFF 关闭 ⟹ 推倒只损失机会成本,
                                 #   candidate 未首发时机会成本≈0 = 推倒很便宜。此为独立
                                 #   小罚 (量级 < candidate +5), 不解封整个 fail 项
    table_touch_fail = False     # 真机红线: 手部 body 原点低于桌面+容差 ⟹ 立即终止
    table_touch_tol = 0.004      # m (原点口径的"接触"容差; 罚带仍在其上引导贴近)
    # ---- C5/Pour (2026-08-20 用户裁定 B4/B5/B6) ----
    carry_pad_reward = False     # A2: 垫接触奖励进 carry (甜甜圈判据; 传感器已镜像)
    carry_w_pad_first = 0.5      # 逐垫首触一次性 (轻触门: 物体速度<0.05)
    carry_w_pad_hold = 0.02      # 接触垫持续分/步 (启动后生效)
    friction_curriculum = False  # B4: 指垫摩擦开局加倍, 随稳抓能力退火回 3.0
    friction_hi = 6.0            # 起始 static=dynamic (合成 6×3=18 或 6×6=36 视对侧)
    friction_lo = 3.0            # 终点 = SuperGrip 现值 (3×3=9)
    pour_succ = False            # B6 改判 (2026-08-20): 几何倒水判据 = 全程最大**里程碑**
                                 #   (非终点); 成功终止 = 走完加权轨迹末帧, C5/Pour 同口径
                                 #   (轨迹后半段语义 = 倒完水把瓶/杯大致还原放回)
    pour_succ_tilt_deg = 71.0    # 达演示峰值倾角
    pour_succ_mouth_r = 0.06     # 瓶口水平投影落杯口半径 (m)
    pour_succ_hold = 10          # 保持步数
    pour_w_ms = 10.0             # 倒水里程碑一次性大糖 (>其余里程碑 2.0, <成功大奖 20)
    # ---- 自由探索倒水段 (2026-08-21 用户裁定: 低置信段不追踪, Task-Specific Success
    #      Tracker 让 RL 自行探索; 根据=重建帧60~82误差30cm/可见率0.2≈carry行93~127) ----
    pour_free = False            # PourS 试验旗 (C5S 对照不开)
    pour_place = False           # v10: 放稳(carry成功)→RELEASE 松手; ★降级回里程碑
    pour_rel_steps = 60          # 松手斜坡步数 (closure 1→0)
    pour_rel_w = 10.0            # 「松手进度×物体不动」乘积棘轮 telescoping 总额
    pour_rel_still_m = 0.02      # 物体"不动"位移半径 (m); 2倍=判负
    pour_rel_still_deg = 10.0    # 姿态同口径 (度); 2倍=判负
    pour_rel_hold = 10           # 全开后保持步数 = 成功
    pour_retreat = False         # E2E 第五段: 松手后臂脚本撤回出生站姿
    pour_ret_steps = 90          # 撤退斜坡步数 (lerp 冻结位姿→站姿)
    pour_choreo = False          # 编舞播放器: 纯脚本走完全部契约行, 门警全免 (只看不判)
    asym_pen_w = 0.0             # 双侧收入不对称罚 (平坦脊→斜坡, r9); 0=关
    fin_open_from_ref = False    # 每侧 q_open ← 指参考首行 (出生即编舞起始手型)
    pour_free_lo = 93            # 入段行: 时钟到此冻结, 前馈停走, 行进门/跟丢判挂起
    pour_free_hi = 140           # 出段行: pour 成功判据达成 → 重拍 rest 锚 → 时钟跳此续追
    pour_free_budget = 150       # 自由段步数预算: 超时截断 (不判负不发糖, 防赖着刷小钱)
    pour_free_w_mouth = 5.0      # 瓶口→杯口水平距离势差分权重 (全段累计≈w×0.3m≈1.5, 只赚缩短增量)
    carry_stable = False         # 甜甜圈稳抓链进 carry (2026-08-20 用户裁定, 补 A2 缺口):
                                 #   全程向心塑形 + 启动前 candidate 大糖 (判据同 FC-D:
                                 #   success_min_pads/grasp_centrip_thresh/candidate_hold_steps)
    carry_w_cent = 0.05          # 向心塑形逐步分 (与 FC-D w_cent_shape 同量级)
    carry_w_cand = 5.0           # 启动前 candidate 一次性大糖 (与 w_stable_ms 同量级)
    # ---- Pour 端到端 (2026-08-20 用户裁定, 本机 4080S): 站姿→接近→真抓稳→追踪→倒→还原 ----
    pour_e2e = False             # 三段拼接正式实现: FC-D 接近/真抓稳机器 + carry 进度时钟,
                                 #   粘合 = 双侧过微抬升验证才允许进度时钟启动;
                                 #   参考 = curobo_pour17_e2e.npz (341,7; pose1_1cm+blend10+carry3)
    carry_ref_off = 0            # carry 帧0 在合并参考里的行号 (e2e.npz 的 seam=121);
                                 #   进度时钟 ref_t = _c_t + off, 侧信号 t = ref_t - off
    success_nonterminal = False  # e2e: 真抓稳(verify 过)是**门**不是终点 —— 不终止回合,
                                 #   成功终止只认 carry 走完末帧 (C5/Pour 同口径)
    step_reward_log = ""         # 非空=逐步奖惩记录目录 (2026-08-21): 全env各项均值+
                                 #   前8探针env全明细(项值/相位/ref_t/垫数/d_pos),
                                 #   4000行(双侧2000步)一片 npz —— 事后逐步回放状态↔奖惩
    near_exempt_m = 0.03         # 近场碰物豁免圈半径 (腕距): 圈内接触=任务素材不判死。
                                 #   AAG 放宽到 0.07 (2026-08-21 验尸: 编舞"弯指前进"段
                                 #   指尖超前腕 ~4cm, 3cm 圈让右手学成"悬停在圈外赚 align
                                 #   绝不越雷池", d_pos min 恰=0.030 实锤)
    fin_ref_npz = ""             # AAG (2026-08-21 用户裁定): 指参考改从该 npz 逐行取
                                 #   ({side}_q29[:,7:29], ref_t 索引) —— "标准成功轨迹当
                                 #   参考", 取代 fcd 四段查表生成; fcd 其余机制(奖励/闸/
                                 #   candidate链)原样。臂前馈同文件 {side}_q 走 curobo loader
    pours_v5 = False             # PourS-v5 (2026-08-21 用户裁定"抓稳优先"): 哲学=全程
                                 #   手物零相对位移。逐步滑移增量罚(从0起梯度,超线性)+
                                 #   握紧反射奖(滑移速度×垫压正差分)+持续抓稳项(垫数≥
                                 #   模板指数×压力量级×向心,启动后全程)+杯直立约束+
                                 #   左右系统互碰罚+特权滑移块进 actor 观测(+8维/侧)。
                                 #   配套(train 旗一并设): slip 死线 4cm/30°→8cm/60°,
                                 #   thrown 25→28cm(参考峰值18cm+10), 摩擦退火弃用,
                                 #   模板换 thumbfix —— 全部从头训 (obs 维度变了)
    pours_v6 = False             # PourS-v6 (2026-08-21 深夜用户裁定, 阶段纯净化):
                                 #   统一哲学 = 物体位姿约束只在**非交互段**(保持稳定);
                                 #   交互段(抓稳→交互结束)只要求手物零相对移动。
                                 #   ①toppled 交互中豁免 ②thrown 删除, 换"偏离自身
                                 #   参考轨迹>30cm 无条件重置" ③cent 只在启动前
                                 #   ④cup_upright 删除 ⑤cross_pen 全程但改纯碰撞口径
                                 #   ⑥倒水成功=交互结束=回合成功终止(本版不做撤离)
                                 #   ⑦成功线=参考瓶倾角平台×0.9(pour17=93°)
    dev_reset_m = 0.30           # 物体偏离自身参考轨迹超此值 ⟹ 无条件重置 (取代 thrown)
    pad_first_arrived_only = False   # v3.3: 碰垫糖只在到位后发 (门前蹭垫工资拐走左手)
    obj_disturb_pre_x = 1.0      # AAG-Local: 到位前物体扰动罚倍率 (8=掌蹭主罚,
                                 #   修补器证伪离线绕行后改 RL 在线学)
    fin_prog_gate = False        # 手指行进门: 到位前指参考钳在 fin_hold_row
    fin_hold_row = 90            # 弯根部末行 (指参考 180 行制, =全行 180)
    fin_track_contact_fade = False  # 接触后淡出指参考拉力 (罚形状修正: 压在物体上
                                 #   贴不回参考 ⟹ 罚它抓东西)
    form_pot_earn_only = False   # 形态势只赚不罚 (参考在动, 跟不上不该扣钱)
    fin_adv_tol_deg = 12.0       # 抓取相位指参考'跟上才前进'容差
    fin_end_row = 175            # 指参考推进上限行 (180 行制: squeeze 末=170~175)
    aag_grasp_row = 0            # >0: 直抓桶(RSI)的指参考起始行 —— 必须是**合拢段
                                 #   起始**(180行制的120=全行240), 不能沿用 t0=gs
                                 #   (那是人手轨迹编号, 在编舞里是"手指全伸直")
    obj_in_actor = False         # 物体特权进 actor (+10/侧): RL=数据生成器新纪律
    dgp_rsi_hi = 0.0             # >0: RSI 课程起始直抓比例, 随 _fcd_g 退火到
                                 #   direct_grasp_prob (0=关; 0.8=先学抓再学走)
    fin_form_pot_w = 0.0         # v3.4: 形态正向势差分权重 (向当前行参考形态推进给
                                 #   小钱; 罚教躺平, 赚教小心前进; candidate 前生效)
    pour_tilt_pot_w = 0.0        # >0: 自由段倾角势差分权重 (v6c: 稀疏里程碑糖之间的
                                 #   连续梯度; telescoping, 只在 pf 窗支付)
    pour_trend_npz = ""          # v7 趋势钟 (2026-08-22 用户裁定): 自由段 1 维倾角趋势
                                 #   参考 npz (bottle_deg/cup_deg, recon 剖面滤波重采样)
                                 #   —— 重建绝对位姿不可靠但倾角趋势可靠, 只借这一维
    pour_prog_w = 0.0            # v6P: 全局进度势权重 —— φ=已复现行数/总行数 (自由段
                                 #   用倾角棘轮折算等效行), telescoping, 0→100% 共发 w
    pour_ff_play = False         # v8: 自由段不冻臂前馈, 时钟改按**倾角剖面**推进
                                 #   (演示倒水是整臂重构, 残差发明不出来)
    pour_ff_tol_deg = 20.0       # 自由段行进门: 瓶倾角与参考剖面之差容差
    pour_prog_align = False      # 窗内进度 = 倾角 × 对齐 (乘积, 与成功判据同构)
    pour_align_d0 = 0.25         # 对齐进度的起算距离 (m)
    pour_tilt_prog = False       # v6T: 自由段"进度"改判为倾角创新高 (grip_hold 的
                                 #   窗内豁免制造了挂机: 握着不动 +79/片 vs 满倾 +5)
    pour_trend_tol_deg = 12.0    # 趋势行进门容差 (双物体倾角跟踪)
    pour_trend_w = 0.15          # 趋势钟每步前进微糖 (站着零收入)
    grip_prog_gate = 0           # >0: grip_hold 进度门 (2026-08-22 用户裁定) —— 近 N 步
                                 #   c_t 无前进则停发, pf 窗豁免(时钟本就冻结)。
                                 #   v6 12M 实锤: c_t 钉死行65(瓶离桌起飞点), 每步
                                 #   抓稳小钱让"停在起飞点挂机"成经济最优
    slip_step_w = 0.5            # 滑移逐步增量罚权重 (单位: /1cm 或 /5°, 超线性×(1+x),
                                 #   x 封顶 2 —— 冒烟实锤: w=2 无顶在随机初策略上
                                 #   -101/回合, 淹掉全部里程碑信号 ⟹ 学"冻住")
    regrip_w = 1.0               # 握紧反射奖权重 (滑移速度 × 垫压正差分/3N)
    grip_hold_w = 0.05           # 持续抓稳每步小额 (×压力量级×向心, 启动后)
    grip_f_ref = 12.0            # N: 垫压总量参考 (thumbfix 实测单垫 5-6N, 4垫≈12-20)
    grip_f_max = 20.0            # N: 压力上限带, 超出反向罚 (防捏爆换分/穿模假力)
    cup_upright_w = 0.5          # 杯身直立罚权重 (B侧倾角超免罚角后线性)
    cup_tilt_free_deg = 15.0     # 杯倾角免罚带 (deg)
    cross_pen_w = 5.0            # 左右互碰罚权重 (物物/手对侧物/手手, 表面点口径)
    cross_margin = 0.01          # m: 互碰安全边距 (口对口 2-5cm 不受扰)
    arm_ff_gate = False          # AAG-v3 (2026-08-21 用户裁定): 接近段臂残差归零=纯前馈
                                 #   (编舞回放已验证 0.3-0.6cm 落点+零碰物, 臂上探索噪声
                                 #   只会拍倒物体+错过最后1cm); 到位后 phase_step 线性
                                 #   ramp 到 arm_ff_post 小带做抓握微调。JSRL 的
                                 #   "guide 走前半程"特例, h 固定在到位行, 永不前移
    arm_ff_post = 0.33           # 到位后臂残差相对全带比例 (0.33×±6°≈±2°)
    arm_ff_ramp = 20             # 到位后 ramp 步数 (残差 pre→post 的过渡)
    arm_ff_pre_left = -1.0       # 左臂 pre 带宽单独覆盖 (<0=同 arm_ff_pre)。左手消融
                                 #   (2026-08-22): 杯宽 8.3cm 指尖先顶住, ±1.2° 不够
                                 #   变换进场几何 —— 左侧放 0.35 给绕障余地
    fin_near_relax = 0.0         # >0: 腕距目标 < fin_near_m 时 fin_track/fin_quiet
                                 #   乘此系数 (最后三厘米允许手指为绕障变形 ——
                                 #   左手 21M 账本: 指形拘束费-26 vs 到位一次性+3,
                                 #   策略理性躺平在 2.6cm 贴单垫刷 align)
    fin_near_m = 0.03            # 近场松绑半径 (m)
    arm_ff_pre = 0.2             # v3.1 (2026-08-22 用户裁定): 到位前臂残差小带
                                 #   (0.2×≈±1.2°) 而非锁零 —— 两个用途: ①补 ArmIK/
                                 #   仿真两把尺 ~3cm 终点缺口 (v3 锁零实锤: 腕永停
                                 #   4-5cm 外, 指对空抓拍倒瓶); ②用户本意: 编舞
                                 #   "进5cm"段会蹭到物体, RL 微调避蹭
    cent_signed = False          # 向心塑形带负梯度 (2026-08-20 用户裁定 cent_fix):
                                 #   奖励 = w×clamp(cent,-0.5,1) —— 压错方向扣分给方向梯度;
                                 #   地板-0.5 防"弃触避罚"。配套 grasp_centrip_thresh=-1
                                 #   (candidate 去向心门, 由 --grasp_cent_fix 一旗双设)。
                                 #   背景: 模板托架力恒非向心 ⟹ 旧 max(cent,0) 恒零=梯度死区,
                                 #   五线 candidate 全零 8M+ 实锤 (RSI 线 cent_shape 收入 0.0000)
    pour_w_mouth_low = 0.0       # B5 糖已删 (2026-08-20 用户裁定: 该几何条件对口在轴顶
    #                              的瓶需倾角>90°, 演示峰值仅71°永远发不出; 倾角糖
    #                              30/60°+成功大奖71°已教足倾斜。代码保留, 权重置零)
    carry_w_frame = 0.1          # 每前进一帧的微糖 (×2 物体)
    ref_start_is_stance = True   # False = 参考首行不是站姿 (carry: 首行=抓姿 IK)
    # ============ FC 实验: 逐指笛卡尔目标 (2026-08-19 用户批准 A/B) ============
    fin_cart = False             # m2/成功的指型判据换成"每指指垫到各自 GraspPose 位置"
    fin_cart_tol = 0.01          # 逐指到位容差 (m); 口径=elastomer link 原点, 目标同口径
    w_fc_crumb = 1.0             # 逐指一次性面包屑 (+w/指)
    fin_pot = 0.0                # 势差分引导系数 (A/B 变量: A=0, B>0; 只奖进步防刷分)
    w_obj_disturb = 0.05         # 罚后果不罚接触: 物体速度罚, **全程在** (用户裁定)
    phase2_m_bonus = 2.0         # 里程碑一次性奖励 (m1 腕进2cm / m2 指型进20°)
    # 指残差**累计**上限放开到全行程 (每步增量仍是标定值 ⟹ 合拢天然缓慢 ~几十步);
    # 旧 4.0(≈指垫2cm)是"模板主驱动"时代的, RL 自走合拢需要全行程
    phase2_fin_dev_scale = 40.0
    dyn_arm_far = 3.0
    dyn_d_near = 0.02            # m, 到这么近就收回 dyn_arm_near 倍
    dyn_d_far = 0.15             # m, 到这么远给满 far 倍
    # ★ 近端尺度 (2026-08-16 实测加的, 原来近端固定 1× = 5mm/步)。
    # 为什么要它: 把参考路径的档位改成"近密远疏"后, 终点附近前馈每步只有 **0.26mm**,
    # 而策略残差上限仍是 5mm/步 —— 大了 20 倍 ⟹ 近目标处前馈没力气, 全靠策略,
    # 而它的分辨率不够细。实测征状: 手到 1.44cm 后**以约 1cm/s 慢慢漂离**
    # (步90 1.44cm -> 步160 4.93cm), 最近只能到 1.19cm 而门槛 1.04cm。
    # 0.3× = 1.5mm/步, 与"还差 1.5mm"的精度需求同量级; 同时把漂移速度压到 1.5mm/步。
    dyn_arm_near = 0.3
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
    # 2026-08-16: 接近任务里 imit 的参考换成 **retract_path**(站姿→q(D)→…→GraspPose,
    # 退避族倒放, 构造上无碰撞且 IK 全通)。它不再是"像不像人手", 而是"偏离这条可行路多少"。
    # 权重按 approach_only 降档(见 w_imit0_approach): 当**提示**不当枷锁 —— 策略仍可偏离,
    # 只是要付点代价; 不然就退化成纯轨迹跟踪, 物体一抖就废。
    w_imit0 = 20.0               # 分/(m·步), 作用在**本步残差用量**上
    w_imit0_approach = 6.0       # approach_only 用这个 (≈ align 权重的 1/1, 量级同阶)
    cap_imit = 0.02              # m/步
    imit_decay_p = 2.0           # w_imit(φ) = w_imit0·(1-φ)^p
    # -- 起步分布 (课程; 训练入口按 sr_ema 更新, 与 gentle 同一个钩子) --
    # direct_grasp_prob: 直接从 GRASP 相位 + prior 位姿起步的比例 = **今天的任务**,
    #   它同时是 RSI 的"从目标态起步"和**免费回归桶** (这一桶成功率必须仍 ~100%).
    # approach_t0_max:   接近相位起步时 t0 ~ U(0, 这个 × gs); 退火到 0 = 逼它从头做.
    direct_grasp_prob = 0.5
    approach_t0_max = 0.8

    # ---- 退避式起点族 (2026-08-16, 见 docs/APPROACH_DESIGN.md §2) ----
    # 用"从 GraspPose 沿 radial 轴往后退"生成起点, 取代"沿人手轨迹采 t0"。
    # 实测依据: 人手轨迹终点离 GraspPose 差 6~10cm/25~43°(够不到 3cm/15° 的切换判据),
    # 中段还倒退 8cm, 且臂外壳刮桌 −1.76cm; 而退避族 IK 全通、间隙单调、任务永远可解。
    retract_start = os.environ.get("RL_RETRACT_START", "0") == "1"
    retract_dmax_k = 1.5         # D = 这个倍数 × d_g(手座->物体中心距离); 自标定不写死厘米
    retract_levels = 32          # 起点族离散级数
    # 档位间距的幂次: d_k = D·(k/(M-1))^p。p=1 等距(原行为), p>1 近处密。
    # 为什么必须 >1: 等距时每档 1.07cm 而到位窗口 1.04cm, 前馈最后一步**跨过**窗口
    # (实测掠过速度 18.84cm/s vs 判据 5cm/s)。p=2、M=32 时最后一档步长 ≈1.5mm,
    # 折合 3cm/s, 落在判据内。
    retract_spacing_p = 2.0
    retract_lambda_max = 2.0     # 准入门抬升上限: û = normalize(r̂ + λ·ẑ), λ 从 0 逐级试
    retract_drop_tol = 0.005     # m, 允许的逐级外壳下降容差 (>这个就算"退避时朝桌面去")

    # ★★ 课程语义 —— 写死在这里, 别再犯 approach_t0_max 那个错 ★★
    #   `approach_t0_max` 的 0 表示"从头做"(最难), 而下面两个的 0 表示"最简单"。
    #   两套方向相反, 混起来就是 2026-08-16 那次: 评测口径被改成"直接空降到抓握帧",
    #   冠军存档评出 0.00%(DESIGN_LOOP §2.24)。所以:
    #     retract_ratio = 0   -> 全部从 d=0 (就在 GraspPose) 起步  = 最简单
    #     retract_ratio = 1   -> d ~ U(0, D) 全程                  = 最难
    #     stance_prob   = 0   -> 不从站姿起步
    #     stance_prob   = 1   -> **全部从对称站姿起步 = 正式任务口径**
    #   ⟹ **评测把 stance_prob 设成 1.0**(而不是把某个 ratio 设成 0), 语义唯一, 无端点陷阱。
    retract_ratio = 1.0
    stance_prob = 1.0

    # ---- Approach-only 任务 (2026-08-16 用户裁定) ----
    # **只学接近**: 站姿 -> GraspPose 的腕位姿。不抓、不抬、不合拢手指。
    # 判据干净、失败原因唯一, 是"抓稳/抬升/搬运"的地基。
    approach_only = os.environ.get("RL_APPROACH_ONLY", "0") == "1"
    # 到位判据 = 腕位置差 < eps_pos 且 朝向差 < eps_rot 且保持 switch_hold 步
    # (复用现有相位切换那套阈值: eps_pos0=3cm / eps_rot0=15° / switch_hold=2)
    # 2026-08-16 用户裁定 40 -> 100。当时的比例是 终端40 : 稠密累计≈2.7 ≈ 15:1
    # (align 的 ep_rew 0.030/步 × 回合均长 90 步)。提到 100 = 37:1。
    # ⚠ 我的判断是这**不是**当前的杠杆(失败模式是超时=走不到, 不是"到了但奖励不够"),
    #   但代价为零, 且与"退避路径当参考"一起改会混judging —— 判读时记得它俩同时变了。
    r_reach = 100.0              # 到位终端奖励
    # ---- 用时塑形 (2026-08-17): 到位奖励按用时衰减, 给"绕路"一个价格 ----
    #   有效奖励 = r_reach × (1-λ + λ·decay^(用时-ref))
    #   λ 由训练入口按成功率放行(见 --time_shape); ref=94 步 = 冠军实测水平
    time_shape_lambda = 0.0      # 0=不打折; 训练钩子按成功率升到 1
    time_shape_decay = 0.99      # 94步→100分 150步→57分 211步→31分
    time_shape_ref_steps = 94.0
    # 回合预算: 用户定 250 步 @20Hz = 12.5s。理论下限来自标定的每步末端位移上界
    #   arm_residual_max(末端 95 分位 2cm) × arm_step_scale(0.25) = **5mm/步**
    #   ⟹ 站姿->GraspPose 30.1cm(Grasp3) 需 ≥60 步; 41.4cm(瓶) 需 ≥83 步。
    #   250 步 = 下限的 3~4 倍, 给转向/减速对准/修正留余量。
    approach_only_steps = 250
    # ★ 硬底线(用户定): 这两件**直接终止**回合, 不只是罚
    #   ① 臂外壳穿桌   ② 接近段手碰到物体 (还没到该碰的时候)
    approach_hit_obj_m = 0.005   # m, 手部连杆到物体表面 < 这个 = 碰上了
    r_hit = -10.0                # 撞桌/撞物体的终止罚

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
    closure_init_min = -1.0      # >=0: 起步合拢度固定为该值 (诊断用)

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
    # ---- 微拧验证 (2026-08-05, screw 27 盖任务): 盖被解析螺旋**单向投影**钉在瓶身上,
    # 抬不动 —— 微抬升对它结构性不可行 (G3_8_5 教训: 判据可达性要建 env 时核验).
    # 换成同哲学的物理裁判: 硬编码腕绕盖轴旋转斜坡, 盖的 screw_angle 跟进 = 真捏住了
    # (握持传扭矩, 与下游拧开任务同一物理量). clip 注册表 verify_mode 字段选择.
    verify_mode = "lift"         # "lift" 微抬升 (默认) / "twist" 微拧
    affordance_npz = ""          # 视频接触带 (物体局部系点集): 只换 pad_approach 塑形
                                 # 的距离目标, 候选门/cent/验证等物理裁判一律不碰
    pad_contact_calib = True     # 垫↔接触零位校准 (均值平移); 对拇指-四指开口
    # 放行上限: 参与指 ≤ 这个数才应用均值平移。均值平移只对**捏取**成立
    # (两垫 FK 偏差同向); 包络抓各指朝向不同, 均值无物理意义, 平移会一侧压进
    # 物体、另一侧推更远。2026-08-16 实测 Grasp3(五指): 开=3.00 垫 / 关=4.75 垫,
    # 判据要 ≥4 ⟹ 开着这条 clip 永远出不了候选 (详见 DESIGN_LOOP §2.24)。
    pad_calib_max_fingers = 2
                                 # 不对称的候选可能帮倒忙 (35_8 实测), 可关
    # ---- 姿态保持 (2026-08-06, 用户要求"保持物体原姿态略微提起"; s21 实测
    #      每回合倾到 17~19° 触发升级) ----
    # 倾角 = 物体当前局部 z 轴 vs 初始局部 z 轴夹角 (对绕轴 yaw 自旋不敏感,
    # 拧盖任务的盖必须能转 yaw). 开关默认关 —— 已盖章旧任务口径不动.
    upright_hold = False
    w_tilt = 1.0                 # 罚量级: 18° 倾斜整回合 ≈ -20 (与候选/成功奖同量级)
    tilt_deadband_deg = 3.0      # 死区: 毫米级接触抖动不罚
    tilt_norm_deg = 15.0         # 归一: (tilt-死区)/norm, 钳 [0,2]
    tilt_succ_max_deg = 5.0      # 判据: 候选/验证瞬时倾角须 < 此值
    verify_twist_deg = 25.0      # 腕绕物体轴总旋转量 (斜坡终点)
    verify_min_twist_deg = 10.0  # screw_angle 至少跟进这么多 = 扭矩传过去了
    verify_fail_twist_deg = 3.0  # 斜坡到顶+3步仍低于它 = 没捏住, 退回重试
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
    # ---- 外壳口径臂罚 (2026-08-05, screw 27 起; 用户要求真机带壳不碰桌) ----
    # 原点+3cm 罩不住真机外壳: l5/l6 截面半径 4~6.7cm, 原点合法时外壳仍可探到桌下.
    # 开关开启后, 臂罚/诊断改用预采样的连杆**外壳表面点** (tasks/pregrasp/
    # arm_shell_points.npz, 每节 48~96 点) 的最低 z, 余量 1cm 即是真实余量;
    # prior 加载器的 yaw 搜索也会加外壳余量评分, 且锚定位姿外壳穿桌直接 assert.
    # 默认关 —— 已盖章的旧任务 (Grasp3 冠军等) 保持原口径不动.
    arm_table_shell = False
    # 2026-08-16 用户裁定 1.0 -> 0.8cm: Grasp3 的 8_5 候选在 GraspPose 处物理稳态
    # 外壳离桌 **0.89cm**(离线 IK 估计 1.3cm), 略低于原 1cm 余量但**为正、未穿桌**。
    # 这个低点在**终点姿态**上, 是该抓法固有的 —— 换退避轴/换路径都改不掉, 零空间抬肘
    # 也已经试过抬不动。硬终止判的是"外壳离桌 < 0"(真撞), margin 只决定罚函数的起罚点,
    # 所以 0.89cm 处仍会吃到一点罚, 反而会推着策略把肘抬高。
    arm_shell_margin = 0.008     # m, 外壳离桌真实余量
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
    # ---- 推走是否**终止**回合 (2026-08-16 加, 逃生阀 RL_PUSH_TERM=0) ----
    # 背景: 对**细高易倒**的物体, "碰倒" 本身就让质心横移 ~8.7cm(瓶高 17.4cm 的一半),
    # **一次倒下就超过 4~10cm 的推走阈** ⟹ 任何接触尝试都有被判死的风险,
    # 于是"永远不碰"成为局部最优。实测 A2(倒水瓶 + grasp_first)整整 23M 步都在往那里跑:
    #   接触指数 1.365 -> 0.024 | 超时终止 0.000 -> 0.875 | 手到目标 2.9 -> 19.4cm
    # 冠军那条不吃这个亏, 因为它的物体高/底径 1.22(碰了不倒), 倒水的瓶是 3.30。
    # 关掉终止 = **保留 -5.0 的惩罚, 但让策略碰倒之后还能继续试**, 而不是一碰就判死。
    push_terminate = os.environ.get("RL_PUSH_TERM", "1") != "0"
    fall_below = 0.03            # m
    max_obj_height = 0.25        # m (物体高于桌面这么多 = 被抛飞; 验证只抬 1cm, 不冲突)
    # 撞倒判负 (2026-08-18 用户裁定: "瓶子被撞倒 这个应该给直接终止"):
    # 物体长轴相对初始姿态倾转超过此角 = toppled, 直接终止判负。
    # 60° 依据: 历史成功抓取的 tilt_max 实测 17~19°, 余量 3 倍; 微抬升/微拧
    # 验证都不改倾角; 倒水任务的 58° 大倾角属于后续任务, 用它自己的配置覆盖。
    fail_tilt_deg = 60.0
    lift_height = 0.05           # m; 只用作物体高度观测通道的归一分母

    # ================= episode =================
    # PREGRASP 那一档在 env 里按 clip 的 gs 覆盖成 gs+approach_extra_steps (approach=True 时);
    # approach=False 时保持 0 = 跳过该相位 (旧任务行为不变).
    phase_timeout = (0, 120, 20, 0, 0, 0)   # GRASP 120; 验证 20
    max_phase = Phase.LIFT       # 走完验证段 = 完整任务


def _obs_base(cfg) -> int:
    """不含参考前瞻/参考接触指集的观测基数。

    closure 模式 144; joints 模式 +55:
      · 动作通道 13 -> 29                       (+16)
      · 手指内部状态 5(每指残差) -> 44           (+39)
            = 逐关节累积残差 22 + 逐关节参考跟踪误差 22
    改 obs 布局必须同步改这里 —— `env._check_obs_dim` 会当场报错。
    """
    if getattr(cfg, "approach_only", False):
        # 7 维臂动作: 相对 closure 模式去掉 12 维 —— closure 标量 1 + 每指残差 5
        # + 动作缓冲 13->7 (6)。这三样都是被 hand_gate 乘零的通道, 留着是死信息。
        if (getattr(cfg, "pregrasp29", False)
                and getattr(cfg, "hand_action_mode", "closure") == "joints"):
            # PreGrasp29 (2026-08-18晚): approach_only 语义 + 22 指解锁(joints) ——
            # 多出逐关节累积残差 22 维 (实测运行时 161 vs 旧公式 139, 差值恰 22)
            return 144 - 12 + 22
        return 144 - 12
    return 144 + (55 if getattr(cfg, "hand_action_mode", "closure") == "joints" else 0)


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
    cfg.observation_space = (_obs_base(cfg) + 7 * int(cfg.ref_look_frames)
                             + (5 if cfg.ref_contact_obs else 0))
    return cfg
