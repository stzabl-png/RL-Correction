"""数据契约 — PIPELINE_FRAMEWORK.md §B 的落地版.

MVP 阶段 (飞手 + 单轨迹 + 无 grasp 标签) 只填得上一部分字段:
- RefTrajectory.anchor_* (③ cuRobo) 与 grasp (② OCIR/BODex) 暂为 None
- human_finger (Sharpa 22 关节 qpos) 需离线 retarget 导出后回填

所有数组遵循 FrameConvention: z-up / wxyz / 米 / "桌面局部系"
(cv2zup 旋转 + 落桌平移之后的坐标; env_origin 偏移由 env 在 reset 时再加).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameConvention:
    up_axis: str = "z"
    quat_order: str = "wxyz"     # scalar-first
    units: str = "m"
    table_height: float = 0.85   # 落桌对齐后桌面上表面的 z (权威支撑平面)


@dataclass
class GraspTarget:               # ② OCIR/BODex, 抓取段"可信目标"
    finger_q: np.ndarray         # (22,) 目标手指构型 (finger_names 序, 弧度)
    wrist_pose: np.ndarray       # (7,)  目标手根位姿
    contact_fingers: np.ndarray  # (5,)  bool 哪几指该接触
    finger_names: list | None = None           # (22,) finger_q 的关节名, env 按名映射 USD 序
    contact_points: np.ndarray | None = None   # (P,3) 可选
    contact_normals: np.ndarray | None = None  # (P,3)
    grasp_phase_frame: int = 0   # 轨迹里"到达抓取"的帧号


@dataclass
class RefTrajectory:
    """对齐后的参考束, 长度 L, 共享同一帧时钟 (源数据 15fps)."""

    # ① 视频重建通道 (Reconstruct_and_Retarget Path B) — MVP 的主参考
    track_object: np.ndarray     # (L,7) 物体参考轨迹 pos+quat(wxyz)
    track_wrist: np.ndarray      # (L,7) SharpaWave 浮动基座位姿 (retarget README 约定:
                                 #       +z指向/+y拇指侧/+x掌法向, 与在线replay同一约定)
    mano_joints: np.ndarray      # (L,21,3) MANO 关节点 (retarget / 指尖参考用)
    valid: np.ndarray            # (L,) bool 手部追踪有效 (无效帧已最近邻填充)
    valid_seg: tuple             # (起, 止) 最长连续有效段, 含端点 — episode 裁剪范围
    interaction_seg: tuple       # (起, 止) 指尖贴近物体表面的帧段, 含端点

    # ③ cuRobo 可行轨迹 = 残差锚点 (MVP=None, 锚点即 track_* 本身)
    anchor_wrist: np.ndarray | None = None    # (L,7)
    anchor_finger: np.ndarray | None = None   # (L,22)
    curobo_pregrasp: np.ndarray | None = None  # (7,) cuRobo 接近段末(close起点)预抓取腕位姿,
                                               #      桌面局部系 — 退火路点引导用
    # ①-手指: Sharpa 22 关节参考 (export_qpos.py 产物)
    human_finger: np.ndarray | None = None    # (L,22) SDK 序
    finger_names: list | None = None          # (22,) SDK 序关节名, env 按名映射到 USD 序
    grasp: GraspTarget | None = None          # ② (MVP=None)
    # ---- 参考质量 (逐帧置信度的原料; 只记录, 当前不参与控制) ----
    # 为什么要留: 参考轨迹的可信度**逐帧不同**, 而现在整条轨迹被当成同等可信
    # (统一的 tube_radius / 统一的残差界). 这些量以前算完就打印丢掉了.
    # 见 docs/DESIGN_LOOP.md §2.4.
    obj_drift: np.ndarray | None = None       # (K,) 物体轨迹被判为离群/漂移的帧号
                                              #      (fix_outliers 修过的帧, 位置是插值出来的)
    # ref builder 对腕轨迹施加的平移 (3,). replay_grasp 的 re-anchor 把腕整体挪了一次
    # (xy 恒 0, z = 掌心降到 affordance 高度 + clearance). 换用相机锚定摆放时这一次
    # 平移**必须撤掉** —— 它是为"物体挪到掌心下方"那个旧锚算的, 换锚之后就是残留.
    builder_wrist_shift: np.ndarray | None = None

    @property
    def L(self) -> int:
        return len(self.track_object)


@dataclass
class ObjectSemantics:
    """物体语义 -> 物理参数. source="human" 为人工标注;
    未来换 VLM ref builder (输入: 重建mesh渲染+视频关键帧) 输出同一结构, 即插即用.
    mass/friction 是中心值, env 里按 ±范围做域随机化, 天然容忍估计误差."""
    label: str = "unknown"
    mass_kg: float = 0.2
    friction: float = 0.5
    mass_range: tuple = (0.1, 0.4)      # 域随机化区间
    friction_range: tuple = (0.4, 0.7)
    source: str = "human"               # human | vlm


@dataclass
class DataUnit:                  # env 每个 case 的完整输入
    clip_id: str                 # 例 "egodex/part2/basic_pick_place/11"
    mesh_path: str               # 原始 .obj (碰撞/AABB 用)
    usd_path: str                # Isaac 资产
    object_init_pose: np.ndarray  # (7,) = ref.track_object[0]
    goal_object_pose: np.ndarray  # (7,) MVP: 终帧位姿即任务目标
    ref: RefTrajectory
    fps: float                   # ref 数组的实际帧率 (重采样后 = 控制频率)
    frame: FrameConvention = field(default_factory=FrameConvention)
    semantics: ObjectSemantics = field(default_factory=ObjectSemantics)
