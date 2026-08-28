"""任务参数单一来源 —— 新任务实例化第一站 (对应 CHECKLIST 第 1 步)。

框架身: Pour17 v5 已验收版 (2026-08-28, 自检家族全绿+训练冒烟全绿)。
本文件集中"每个任务必改"的参数; 判据级改动看各文件 [TASK] 标记。
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
TASK_ROOT = os.path.abspath(os.path.join(_HERE, ".."))

# ---- 任务身份 ----
TASK = "TEMPLATE"                 # 任务名: 训练名前缀/日志用
CLIP = "Pour17_bottle"            # [TASK] clips.configure_cfg 场景名 (资产/桌面布局)

# ---- 三 Prior 路径 ----
# 母带: v1=人手行版(重建流水线产物), v2=物体轨迹反解IK版(build_reference.py 产物)
REF_V1 = os.path.join(TASK_ROOT, "A_Design", "L2_Reference", "reference_v1.npz")
REF_V2 = os.path.join(TASK_ROOT, "A_Design", "L2_Reference", "reference_v2.npz")
# GraspPose 模板 (含 squeeze 层): 主手侧(右)/辅手侧(左)
PRIOR_MAIN = "tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz"   # [TASK]
PRIOR_AUX = "tasks/pregrasp/priors/Pour17_cup_thumbfix.npz"       # [TASK]
PRIOR_APPROACH_DEG = 19.5         # [TASK] apply_grasp_prior 的 approach 参数

# ---- squeeze 剂量 (CHECKLIST 第 3 步: 零动作探针标定, 勿拍脑袋) ----
BETA_R = 2.0                      # [TASK] 主手侧剂量 (重物深捏)
BETA_L = 1.0                      # [TASK] 辅手侧剂量 (轻薄物勿深, 会挤飞)

# ---- 物体几何 (Success Tracker 判据原料) ----
MOUTH_HALF_MAIN = 0.087           # [TASK] 主物体长轴半长 (口部推导)
MOUTH_HALF_AUX = 0.066            # [TASK] 辅物体长轴半长
UP_LOCAL_MAIN = (0.0, 1.0, 0.0)   # [TASK] 主物体局部"竖直"轴 (倾角口径)
UP_LOCAL_AUX = (0.0, 1.0, 0.0)    # [TASK] 辅物体局部"竖直"轴
