"""Clean/3 任务参数单一来源 (台账 §5.6)。"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
TASK_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
REPO = os.path.abspath(os.path.join(TASK_ROOT, "..", "..", ".."))

TASK = "Clean3"
# clip / 先验 / 母带三件套可整体换 take (2026-09-10 加, 为 take18 倒扣盘变体; 默认恒等于原 take3 口径)。
# take18 线: CLEAN_CLIP=Clean18_plate CLEAN_PRIOR_PLATE=.../Clean18_plate_left.npz CLEAN_REF_NPZ=.../clean18_reference_v1.npz
# (布的先验不换 —— take18 的布资产不用, 见 clips._clean18)。三者都进 world.json 指纹 (clip/reference/priors)。
CLIP = os.environ.get("CLEAN_CLIP") or "Clean3_plate"   # env.object = 盘(左手, 主体物) / env.aux = 海绵(右手)
REF_V1 = os.path.join(TASK_ROOT, "A_Design", "L2_Reference", "clean3_reference_v1.npz")   # 第 0 行 = grasp 位
REFERENCE = os.environ.get("CLEAN_REF_NPZ") or REF_V1
PRIOR_PLATE = os.environ.get("CLEAN_PRIOR_PLATE") or os.path.join(REPO, "tasks/pregrasp/priors/Clean3_plate_left.npz")
PRIOR_SPONGE = os.environ.get("CLEAN_PRIOR_SPONGE") or os.path.join(REPO, "tasks/pregrasp/priors/Clean3_sponge_right.npz")

# 物理 (用户裁定 §5.0⑦): 盘 0.3kg / 海绵 0.05kg (aux USD 烘焙) / μ1 物+指垫。发射脚本显式写环境变量, 训练日志 grep "难度覆写"。
PHYS = {"POUR_OBJ_MASS": "0.3", "POUR_OBJ_FRIC": "1.0", "POUR_PAD_FRIC": "1.0"}

# ---- Stage-1 抓稳段 (用户 2026-09-07 三点: 碰撞全开 / 手直接从 grasp 位起步 / 只求放手后物体不动) ----
CONTROL_HZ = 20
K_CLOSE = 10                 # 指前馈 grasp→squeeze 斜坡行数
RELEASE_MAX = 50             # 钉住行数上限 (课程起点: 合拢后再多 2s)
RELEASE_MIN = K_CLOSE        # 课程终点: 合拢一结束就放手
RELEASE_STEP = 5             # 每次退火减多少行
RELEASE_JITTER = 4           # 每回合 release_row 在 [cur-jitter, cur] 内随机 (避免策略数拍子)
HOLD_ROWS = 60               # 放手后至少持多久 (3s) —— 回合长 = RELEASE_MAX + HOLD_ROWS 固定
T_EP = RELEASE_MAX + HOLD_ROWS
# G0 认证: 连续 10 步内 (世界位姿 vs 钉住位姿)。冒烟实测零动作放手后 盘 0.96cm/8.2°, 海绵 0.27cm/3.5° ——
# 8° 正卡在被动平衡上; 要"比被动更稳"建议 CLEAN_CERT_POS_CM=1.0 CLEAN_CERT_ROT_DEG=5 (发车时定, 台账 §5.6)
# 2026-09-08 用户裁定: 默认收紧到 1.0cm/5° (零动作被动平衡 0.96cm/8.2° 过不了, 策略必须比被动更稳)
CERT_POS = float(os.environ.get("CLEAN_CERT_POS_CM", "1.0")) / 100.0
CERT_ROT_DEG = float(os.environ.get("CLEAN_CERT_ROT_DEG", "5.0"))
CERT_STEPS = 10
DROP_POS, DROP_ROT_DEG = 0.03, 20.0                     # D1 掉落 (终止)
PAD_FTH = 0.5                                           # N, 垫"有力"
PLATE_SUPPORT_MIN = 2                                   # 盘: 拇指 + ≥2 托底指
SPONGE_PADS_MIN = 3
# 残差界 (rad): 每步步长 / 累积上限。臂小 (只做微调), 指大 (要学抓稳)
ARM_STEP, ARM_DEV = 0.004, 0.08
FIN_STEP, FIN_DEV = 0.03, 0.60
# 奖励系数
W_CONTACT, W_HOLD_POS, W_HOLD_ROT, W_WITHIN = 0.1, 0.5, 0.5, 0.2
B_CERT, B_SUCCESS, B_DROP, W_ACT = 5.0, 20.0, -10.0, 0.001
# 手指交叉/重叠罚 (2026-09-08 用户要求): 同手相邻指尖 (食-中, 中-无名, 无名-小) 在掌系侧向 (palm y) 的间距
# 低于 CROSS_GAP_MIN 视为交叉 (正负号以 grasp 姿的顺序为准, 翻转即交叉); 指尖 3D 距离低于 OVERLAP_MIN 视为重叠。
# grasp 姿实测相邻指尖侧向间距 1.7~3.8cm, 3D 距离 ≥2.5cm, 故名义姿罚为 0。
W_CROSS, CROSS_GAP_MIN, OVERLAP_MIN = 1.0, 0.010, 0.015
# 前馈 squeeze 姿自身的交叉 (URDF FK 实测): 右手 squeeze 小指 CMC +22.4° / MCP_AA −10.7° 把小指扫到无名指下面
# (侧向差 −0.67cm 翻转, 3D 距 1.49cm)。这两个关节的合拢剂量置 0 (FE/PIP/DIP 照旧全卷) → 侧向 1.84cm / 3D 2.48cm 干净。
# 左手 squeeze 无交叉 (侧向 2.0/1.9/3.9cm)。
SQUEEZE_JOINT_BETA = {"right": {"right_pinky_CMC": 0.0, "right_pinky_MCP_AA": 0.0}, "left": {}}

# ======================= Stage-2 交互段 (最简六项配方; 台账 §5.7, 用户 2026-09-08 批准, 三项默认已拍板) =======================
# 回合 = 抓稳段 (钉住, release 课程 50→10 唯一课程) → 放手锁存 → 认证 (10 步 1cm/5° 全转角) → 时钟 k 走母带 424 行 → 时钟走完结束。
S2_CERT_POS = float(os.environ.get("CLEAN_S2_CERT_POS_CM", "1.0")) / 100.0      # 相对锁存位姿
S2_CERT_ROT_DEG = float(os.environ.get("CLEAN_S2_CERT_ROT_DEG", "5.0"))         # 全转角 (拍板: 不只看长轴)
S2_CERT_STEPS = 10
S2_CERT_BUDGET = 60                 # 放手后多少行内必须认证, 否则超时结束不另罚
# 死线 (任一物体相对锁存超限即死)。2026-09-10 用户裁定: "只要不是抓不住掉落, 手里有一些转动很正常
# (原视频洗碗确实有旋转和偏移)" ⟹ 转角死线做成可放宽的旋钮。**默认仍是 20°**, 因为 Denso 在跑的
# 两条消融与 take3 冠军都用这把尺子, 改默认就不可比了; 要放宽在发射脚本里显式写 CLEAN_S2_DIE_ROT_DEG。
S2_DIE_POS = float(os.environ.get("CLEAN_S2_DIE_POS_CM", "3.0")) / 100.0
S2_DIE_ROT_DEG = float(os.environ.get("CLEAN_S2_DIE_ROT_DEG", "20.0"))
# 软罚(H-C3)的归一化跨度**与死线解耦**: 死线放宽时不希望罚的斜率跟着变平 —— 那是两处改动同时发生,
# 判读表会失效 (台账"一次只改一个参数")。这两个常量恒为原死线值, 不跟 CLEAN_S2_DIE_* 走。
S2_SOFT_SPAN_POS, S2_SOFT_SPAN_ROT_DEG = 0.03, 20.0
S2_PLATE_TILT_DIE_DEG = float(os.environ.get("CLEAN_S2_PLATE_TILT_DIE_DEG", "30.0"))
S2_PLATE_DEV_DIE = 0.10             # 盘偏离第 k 行参考 (m)
S2_GATE_N, S2_GATE_XY = 0.01, 0.05  # 时钟门: 海绵在盘规范系 法向 / 面内
S2_LEASH_N = 0.01
S2_LEASH_XY = {2: 0.03, 1: 0.05, 0: 0.08}        # 按置信档 (Pour 同款数值)
S2_TIER_HI, S2_TIER_LO = 70.0, 40.0              # 档位阈值 (Pour progress 同款), 置信度 = min(conf_0, conf_1)
S2_R_ADV, S2_B_CERT, S2_B_SUCC, S2_B_DIE, S2_W_ACT = 1.0, 5.0, 20.0, -10.0, 0.001
S2_CLOCK_SLACK = 1.3                # 交互段预算 = 母带行数 × 1.3
S2_ARM_DEV = 0.08                   # 臂累积残差上限 (rad); 每步界 = cfg.arm_residual_max × arm_step_scale (Pour 标定形状)
S2_CLAMP_JOINTS = {"left": ("left_middle_MCP_AA", "left_ring_MCP_AA"),          # 握形合法性: 残差硬钳, 无罚项
                   "right": ("right_pinky_CMC", "right_pinky_MCP_AA")}
S2_LOOKAHEAD = (0, 5, 10)           # 观测里给的参考前瞻行
# ---- Stage-2 候选旗 (2026-09-08 晚 2×2 因子实验; 台账 §5.7 候选清单) ----
S2_CLOCK_NEEDS_CONTACT = os.environ.get("CLEAN_S2_CLOCK_CONTACT", "0") == "1"   # 时钟推进 (=adv) 要求海绵接触盘面 (封"悬停刷时钟")
S2_REANCHOR = os.environ.get("CLEAN_S2_REANCHOR", "0") == "1"                   # 海绵臂前馈按实际盘位姿在线重锚 (PhysX 雅可比 DLS 微分 IK)
# 重锚 = 盘面系里海绵位置误差 e = psc_actual − psc_ref[k] 的闭环伺服 (积分式, 经 PhysX 雅可比 DLS 落到右臂关节前馈), 认证后生效。
#   前两版 (盘位姿偏差前馈 / 锁存) 实测正反馈跑飞: 海绵压盘→盘沉→偏差变大→再压, 1.2~1.5cm 压进盘面。闭环误差伺服是负反馈: 压进去 e_n<0 就抬。
S2_REANCHOR_KP = 0.2             # 每步修正 = KP × e (m→m), 积分累加; 抗饱和: 已接触时法向不再往下积分 (只允许压过头时抬)
S2_REANCHOR_DEADBAND = 0.002     # |e| < 2mm 不动
S2_REANCHOR_DQ_MAX = 0.15        # 累积关节修正上限 (rad)
S2_REANCHOR_DAMP = 0.05          # DLS 阻尼 λ
# ---- H-C3 预注册改动 (用户 2026-09-08 批准, 本机对照): 相对位姿软罚 ----
#   认证后每步: −S2_SOFT_W × mean_物体 clamp( max( (dp−1cm)/(3cm−1cm), (dr−5°)/(20°−5°) ), 0, 1 ); 死线处满值 = −0.5/步 (< adv +1, 不诱导早死)
S2_SOFT_REL = os.environ.get("CLEAN_S2_SOFT_REL", "0") == "1"
S2_SOFT_W = float(os.environ.get("CLEAN_S2_SOFT_W", "0.5"))
# ---- 2026-09-08 晚 自主迭代 (用户授权 Denso 4 卡): 两面新旗, 均为 base + 一项 ----
S2_GATE_HOLD = os.environ.get("CLEAN_S2_GATE_HOLD", "0") == "1"      # 时钟推进 (=adv) 还要求两物体相对锁存位姿在认证窗 (1cm/5°) 内
S2_PUSH = os.environ.get("CLEAN_S2_PUSH", "0") == "1"                # 认证后随机外力推物体 (CartPole 式扰动课程 / 鲁棒性)
S2_PUSH_P = float(os.environ.get("CLEAN_S2_PUSH_P", "0.02"))         # 每步起一次推力的概率 (≈每 50 行一次)
S2_PUSH_STEPS = 8                                                     # 每次持续行数
S2_PUSH_F = (0.5, 1.5)                                                # 力 = U(下,上) × m·g, 方向掌系均匀随机 (盘 0.3kg→1.5~4.4N, 海绵 0.05kg→0.25~0.74N)

# ================= 消融旗 (2026-09-10, 台账 §5.15; 与 Pour Base/NH/NHNC 对位) =================
# Base   = HAND_REF=1            人手指姿指引开
# A1     = 全关 (= 现役冠军 Clean3_taskS_s42 配置, 直接复用不重跑)
# A2     = CONF_FLAT=1           在 A1 基础上再把置信度分档拍平成黄档
S2_HAND_REF = os.environ.get("CLEAN_S2_HAND_REF") == "1"
S2_HAND_W = float(os.environ.get("CLEAN_S2_HAND_W", "1.0"))
#   人手层只进**指列**: f_ff += W × (human_*_f − 母带 squeeze 姿), 只在认证后生效。
#   与 Pour 的差别: Pour 的 W_HAND 是**奖励项**权重且按档位 (绿0/黄0.5/红0.8); Clean 走**前馈**,
#   逐行切档会让指姿抖, 所以用常量权重。这条差异写进台账 §5.15, 判读时别当成同一个机制。
S2_CONF_FLAT = os.environ.get("CLEAN_S2_CONF_FLAT") == "1"
#   拍平 = 母带每行档位一律置 1(黄) ⇒ 皮筋常量 S2_LEASH_XY[1]=0.05, 观测里的档位独热同时变常量。
#   臂累积界 S2_ARM_DEV 本来就不分档 (常量 0.08), 无需再拍 —— 数值与 Pour NHNC 的 LEASH_FLAT/DEV_ARM_FLAT 恰好同口径。

