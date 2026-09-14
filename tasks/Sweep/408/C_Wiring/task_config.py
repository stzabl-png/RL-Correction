"""Sweep408 任务参数单一来源 (抓稳段 Stage-1)。

用户 2026-09-11 裁定:
  · 抓稳段要测的是"**移动时**不发生偏移" —— 回合不是静态持握, 而是**手臂走完整条母带**, 全程保持手物关系。
  · 相对位姿一律在**掌系**度量 (不是世界系)。这一条同时解决了"桌子替手扶着物体、握松了也能过认证"的作弊路径:
    手一动, 桌子扶住物体就会立刻表现为掌系漂移。所以桌面接触**保留** (它正是扫地时的主要扰动);
    用户原本选的"关掉物体↔桌面碰撞"降级成旗 SWEEP408_NO_TABLE_CONTACT=1, 默认不开。
  · 接触条件用初值: 扫把 = 拇指 + ≥2 指; 簸箕 = ≥4 垫。
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
TASK_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
REPO = os.path.abspath(os.path.join(TASK_ROOT, "..", "..", ".."))

TASK = "Sweep408"
CLIP = os.environ.get("SWEEP408_CLIP") or "Sweep408_broom"      # env.object=扫把(右, 主体) / env.aux=簸箕(左)
REFERENCE = os.environ.get("SWEEP408_REF_NPZ") or os.path.join(
    TASK_ROOT, "A_Design", "L2_Reference", "sweep408_reference_v1.npz")
PRIOR_BROOM = os.environ.get("SWEEP408_PRIOR_BROOM") or os.path.join(REPO, "tasks/pregrasp/priors/Sweep408_broom.npz")
PRIOR_PAN = os.environ.get("SWEEP408_PRIOR_PAN") or os.path.join(REPO, "tasks/pregrasp/priors/Sweep408_dustpan.npz")

# 物理 (G-A 规矩; clip 里声明 扫把 0.15kg / 簸箕 0.20kg, μ1 物+指垫)。发射脚本显式写, 日志 grep "难度覆写"。
PHYS = {"POUR_OBJ_MASS": "0.15", "POUR_OBJ_FRIC": "1.0", "POUR_PAD_FRIC": "1.0"}
PAN_MASS_KG = 0.20                                              # 走 aux USD 的 ObjectSemantics, 只进指纹

NO_TABLE_CONTACT = os.environ.get("SWEEP408_NO_TABLE_CONTACT") == "1"

# ---- 回合结构 ----
CONTROL_HZ = 20
K_CLOSE = 10                 # 指前馈 grasp→squeeze 斜坡行数
RELEASE_MAX = 50             # 钉住行数上限 (课程起点)
RELEASE_MIN = K_CLOSE        # 课程终点: 合拢一结束就放手
RELEASE_STEP = 5
RELEASE_JITTER = 4
CERT_BUDGET = 60             # 放手后多少行内必须认证, 否则截断 (不另罚)
CLOCK_SLACK = 1.05           # 时钟预算 = 母带行数 × 该系数

# ---- 认证 / 死线 (全部在**掌系**度量, 相对放手瞬间的锁存位姿) ----
CERT_POS = float(os.environ.get("SWEEP408_CERT_POS_CM", "1.0")) / 100.0
CERT_ROT_DEG = float(os.environ.get("SWEEP408_CERT_ROT_DEG", "5.0"))
CERT_STEPS = 10
DIE_POS = float(os.environ.get("SWEEP408_DIE_POS_CM", "3.0")) / 100.0
DIE_ROT_DEG = float(os.environ.get("SWEEP408_DIE_ROT_DEG", "20.0"))
# 软罚跨度与死线**解耦** (Clean §5.17 的教训: 放宽死线时罚的斜率不该跟着变平, 否则判读表失效)
SOFT_SPAN_POS, SOFT_SPAN_ROT_DEG = 0.03, 20.0
TABLE_MARGIN = 0.005         # 手掌/指垫低于桌面这么多即判死 (母带实测最低余量 +2.0cm, 见 L2 抬柄注释)
OBJ_DROP_Z = 0.03            # 物体低于桌面这么多即判死

# ---- 接触条件: v2 起**只作诊断, 不进认证** (2026-09-11 用户批准 v2 阶梯) ----
#   v1 把 broom_ok & pan_ok 写进认证, 实测 within_end=0.902 (纯位姿判据满足) 而 sr/cert=0.007 ——
#   堵住认证的正是这个接触条件。Clean Stage-2 当初明确裁定过"认证不带接触条件"(台账 §5.7 三项默认),
#   408 v1 是我自作主张加的。v2 回到 Clean 口径; 接触分只进 TB 和 L3 臂的奖励。
# ---- (以下阈值保留: 诊断 + L3 臂的接触奖) ----
#   扫把 [0.64 1.63 1.64 2.25 4.44]cm —— 模板 8_Prismatic_2_Finger 只有 4 指参与, 小指本就不碰
#   簸箕 [0.93 1.56 0.65 0.97 1.43]cm —— 模板 11_Power_Sphere, 9 个规划接触点, 五指都贴
PAD_FTH = 0.5                # N, 垫"有力"
BROOM_SUPPORT_MIN = 2        # 扫把: 拇指 + 其余四指中 >=2
PAN_PADS_MIN = 4             # 簸箕: 五垫中 >=4

# ---- 残差界 (rad): 臂小 (只微调) 指大 (要学抓稳); 同 Clean/3 ----
ARM_STEP, ARM_DEV = 0.004, 0.08
FIN_STEP, FIN_DEV = 0.03, 0.60

# ================= v2 四条阶梯 (2026-09-11 用户批准) =================
# 每条只比前一条**多一项指引**, 这样"哪项指引有用"能直接读出来。SWEEP408_ARM 选臂:
#   L0 净时钟 = adv + 认证/成功/死亡 + 动作罚          (最简)
#   L1 = L0 + 相对位姿软罚 (Clean H-C3 的赢家配方)
#   L2 = L1 + 贴桌奖       (用户要的"扫把簸箕贴着桌面")
#   L3 = L2 + 指垫接触奖   (= v1 那项; 顺带检验它是帮忙还是造局部最优)
ARM = (os.environ.get("SWEEP408_ARM") or "L3").upper()
assert ARM in ("L0", "L1", "L2", "L3"), ARM
USE_SOFT = ARM in ("L1", "L2", "L3")
USE_TABLE = ARM in ("L2", "L3")
USE_CONTACT = ARM == "L3"

# ---- 奖励 ----
# ★铁则: adv (1.0/步) 必须大于任何一项指引 (软罚 ≤0.5 / 贴桌 ≤0.2 / 接触 ≤0.1)。
#   v1 翻车正是因为接触奖 0.1×588≈130 是回合全部收入, 而认证只值 +5 —— 蹲着刷接触最划算。
R_ADV = 1.0                  # 时钟每推进一行 (走完 455 行 ≈ +455, 回合主收入)
B_CERT, B_SUCCESS, B_DIE = 5.0, 20.0, -10.0
W_ACT = 0.001
# L1+: 相对位姿软罚 (认证后每步)。跨度**与死线解耦** —— 放宽死线时罚的斜率不变, 判读表才不失效。
SOFT_W = float(os.environ.get("SWEEP408_SOFT_W", "0.5"))
SOFT_SPAN_POS, SOFT_SPAN_ROT_DEG = 0.03, 20.0
# L2+: 贴桌奖 (认证后每步)。形式照 Sweep2 的 clear_q: 以目标间隙为中心的高斯。
#   认证前不给 —— 那时物体还钉在母带第 0 行、本来就贴桌, 白送钱。
W_TABLE = float(os.environ.get("SWEEP408_W_TABLE", "0.2"))
TABLE_SIGMA = 0.008          # m, 高斯宽度 (Sweep2 clear_q 同款)
TABLE_TARGET = {"right": -0.001, "left": 0.003}   # 扫把头底轻压桌下 1mm / 簸箕斗底离桌 3mm (= 母带的设计值)
TABLE_HULL_N = 256           # 头/斗段凸包抽成多少个极值顶点 (刚体最低点必在凸包顶点上)
# L3: 指垫接触奖
W_CONTACT = 0.1

# ---- Success Checker (用户 2026-09-11 自定义) ----
# v1 的 success = 走完全部 455 行, 全有全无、早期恒 0, 判读上没有信息。
# 改按 L5-27 铁则"参考自己做得到": 母带零动作参考 100% 可达 (双臂 IK <2cm 占比 100%), 取其 80%。
SUCCESS_CLOCK_FRAC = float(os.environ.get("SWEEP408_SUCCESS_CLOCK", "0.80"))
#   success = 时钟 ≥ 80% 母带 ∧ 终局两物体仍在认证窗内

# ---- 课程: **双向** (v1 死在单向退火退到 20 塌了回不去, 白烧 22.5M 步 = 75% 预算) ----
ANNEAL_DOWN_EMA = float(os.environ.get("SWEEP408_ANNEAL_DOWN", "0.70"))   # cert EMA ≥ 此值 → release −STEP
ANNEAL_UP_EMA = float(os.environ.get("SWEEP408_ANNEAL_UP", "0.30"))       # cert EMA ≤ 此值 → release +STEP
ANNEAL_SUSTAIN = int(os.environ.get("SWEEP408_ANNEAL_SUSTAIN", "10"))
