"""任务参数单一来源 —— Unscrew(拧瓶盖) 实例 (对应 CHECKLIST 第 1 步)。

框架身: Pour17 v5 已验收版 (steps/step4_rl/framework, 2026-08-28)。
本实例 2026-08-29 落地, 数据 = datasets/unscrew_bottle (egodex_part4, 18 条重建,
物体逐帧 conf_pos/conf_rot + 人手逐帧置信度)。

★ 数据引擎口径: 一个实例吃全部 17 条可用 clip —— `UNSCREW_CLIP=<n>` 选条
  (默认 32 = README 榜首, 双物体最低分 85), 所有路径/母带/场景由它派生。
  逐条生成演示数据的循环见 C_Wiring/data_engine.sh。

★ 物体编号约定 (与 PourEnv 同构, 但与数据集相反, 别搞混):
  母带/进度机 obj_0 = 瓶身 (env.object, 左手持) | obj_1 = 盖 (env.aux, 右手拧)
  数据集 object_0=瓶 object_1=盖 (17/18 条; 构建器按尺寸认件, 不信编号)
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
TASK_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
REPO = os.path.abspath(os.path.join(TASK_ROOT, "..", "..", ".."))

# ---- 任务身份 ----
CLIP_ID = os.environ.get("UNSCREW_CLIP", "32")   # 数据引擎逐条覆写
TASK = f"Unscrew{CLIP_ID}"
CLIP = f"unscrew{CLIP_ID}_task"                  # clips.configure_cfg 场景名
TAKE_DIR = os.path.join(REPO, "datasets", "unscrew_bottle", CLIP_ID)

# ---- 三 Prior 路径 ----
# 母带: v1=离线构建 (make_reference.py; 交互段腕参考=物体轨迹推导, 上游腕平移
#       是静态填充死数据 —— clip32 实测全程位移 <0.6cm 而盖走 66cm),
#       v2=Isaac 内物体轨迹反解 IK 重铸 (build_reference.py 产物)
_L2 = os.path.join(TASK_ROOT, "A_Design", "L2_Reference", CLIP_ID)
REF_V1 = os.path.join(_L2, "reference_v1.npz")
REF_V2 = os.path.join(_L2, "reference_v2.npz")
REST_JSON = os.path.join(_L2, "env_rest.json")   # probe_rest.py 产物 (env 实测静置)
# GraspPose 模板: 左手瓶 = Screw27_body (CAD 与本批字节相同, md5 已核对);
# 右手盖 = 设定 B (盖太小, GraspPose 处理不了) —— 无 grasp prior, 指尖抓取
# 由人手手指流 (活数据) + affordance 热区引导, PRIOR_MAIN 留空。
PRIOR_MAIN = ""                                              # [TASK] 盖侧: 设定 B
PRIOR_AUX = os.path.join(REPO, "tasks", "pregrasp", "priors", "Screw27_body.npz")
PRIOR_APPROACH_DEG = -1.0        # 不覆写 yaw (canon_rot 原样)
# 数据集自带的盖侧 affordance (60k 点接触频率热区, 逐 clip)
AFFORDANCE_CAP = os.path.join(TAKE_DIR, "contact", "expected_area_object_1_right.npz")

# ---- squeeze 剂量 (CHECKLIST 第 3 步: probe_beta 标定后回填, 现值=初值) ----
BETA_R = 0.0     # [TASK] 右手盖: 无 squeeze prior (三指精捏交给人手指流+残差)
BETA_L = 2.0     # [TASK] 左手瓶: Screw27_body squeeze; 瓶 0.53kg 与 Pour17 同级,
                 #        初值抄 Pour17 重物侧实证值, 必须 probe_beta 复标

# ---- 物体几何 (Success Tracker 判据原料; CAD 全批统一, md5 已核对) ----
BOTTLE_HALF_H = 0.0985           # 瓶身长轴半长 (mesh 实测 19.7cm/2)
CAP_HALF_H = 0.0085              # 盖半高 (1.7cm/2)
UP_LOCAL_BOTTLE = (0.0, 0.0, 1.0)   # 瓶局部竖直轴 = z (螺轴, 与 screw_assembly 一致)
UP_LOCAL_CAP = (0.0, 0.0, 1.0)
CAP_RADIUS = 0.0175              # 盖半径 (贴近奖/判据几何)

# ---- 螺纹口径 (数据引擎逐条可覆写; 默认承旧台账用户裁定) ----
# turns=0.75 (U30b: 演示实测 ~266° 分离; 2.0 圈是标准件假设, 难 2.7×)
# 释放判据 = screw_assembly 的 detach (拧满 turns 即脱开, screw_detach_at_full)
SCREW_TURNS = float(os.environ.get("UNSCREW_TURNS", "0.75"))

# ---- 机器段 (cuRobo 规划产物; plan_machine_segs.py 逐 clip 产出) ----
# 没有规划产物时 make_reference 退回关节 smoothstep 占位 (无碰撞背书, 只够冒烟)。
# ⚠ cuRobo 是 MagicSim 定制 fork (curobo.motion_planner API), 不在 PyPI;
#   跨机需 MAGICSIM_ROOT=<检出路径> 且该 fork pip 装进本解释器。
MOTION_DIR = os.path.join(TASK_ROOT, "A_Design", "L1_Data", "Motion_Planning",
                          CLIP_ID)
APPROACH_NPZ = os.path.join(MOTION_DIR, "Approach.npz")
RETREAT_NPZ = os.path.join(MOTION_DIR, "Retreat.npz")

# ---- 时钟/重采样 ----
FPS_RECON = 15.0                 # 重建帧率 (replay_world.fps; 单时钟=母带行轴)
