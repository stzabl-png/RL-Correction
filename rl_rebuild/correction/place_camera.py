"""相机锚定摆放 — 把重建的(相机/手/物体)整体放进机器人工作空间.

## 为什么换掉旧摆放

旧链条有两个锚, **都绑在物体上**:
  align_replay:   物体首帧 xy -> 桌面原点 (0,0);  物体最低顶点 -> 桌面
  replay_grasp:   再把物体挪到 PreGrasp 掌心正下方   <- 这一步把上面的锚破坏掉

后果 (2026-07-28 实测, Grasp0~9 全部如此): 物体落到机器人**左侧** y=+0.141,
而交互手是**右手** —— 每条 clip 都变成跨中线 25cm 的跨身抓取. 重建系里物体本来
在人的右边 (y=-0.082), 是被第二个锚搬过去的.

## 新摆放

  XY: 相机/手/物体**刚体同步**平移, 由"重建相机 xy -> 机器人 ZED 光心 xy"定死.
      三者之间的 xy 相对关系完全保留重建原样.
  Z:  两条**独立**约束 (不是刚体):
        物体 -> 贴桌面 (align_replay 已做)
        手   -> 全程最小抬升, 保证 SharpaHand 最低点 >= 桌面+clearance
  接触: Z 关系被打断了, 所以 PreGrasp 位姿要重新算 —— 让合拢中心悬停在
        affordance 区上方 hover_gap. 没有这一步, 手会全程悬在物体上方够不着
        (place_mode="bimanual" 当年接触率 17.3%->0.1% 就是这个失败模式).

重建世界系: `gravity_z_up_world`, 原点在相机(人头), +X = 中间帧相机前向的水平投影.
"""
from __future__ import annotations

import os

import numpy as np

from rl_rebuild.correction import frames as F

# 机器人 ZED 光心 (桌面局部系) — **标称值**, 从训练 env 实测导出.
# ref builder 不启动 Isaac, 拿不到活的机器人位姿, 所以先用这个把参考摆好;
# env 在躯干沉降收敛之后会重新实测, 把差值当**残差**补上 (_apply_camera_anchor).
# 于是: 常数过期了也只是让 env 多平移一点点, 不会静默摆错 —— 权威始终是活测值.
# ⚠ DEXMATE_BODY_POSES.json 里那份是**飞手 env** 导出的, 与训练 env 差 2.3cm, 别用.
ZED_NOMINAL = np.array([-0.4079, -0.0004, 1.4021])

# ---- PreGrasp 悬停位姿的两个选定值 (2026-07-28, 在 view_placement GUI 里目视选的) ----
# ⚠ 这两个必须一起改: 换锚点就要改 hover. 中指根比合拢中心**离腕近 5.08cm**, 拿它对准
#   同一个目标点会把腕沿手指指向多送 5cm ——  hover 不跟着加, 张开的手就戳桌
#   (实测 middle_base + hover 0.03 -> 手最低点在桌下 1.64cm; 而 grasp_center + 0.03
#   是桌上 +4.47cm). 两者相差的 5.08cm 就是 0.08 与 0.03 的由来.
# 放在这里而不是各个工具的 argparse 默认值里: 摆放的数只能有一份, 这一串问题的根源
# 就是"同一件事在两处各算一次" (见本文件开头 "为什么换掉旧摆放").
PREGRASP_ANCHOR = "middle_base"     # "middle_base"=中指根(掌根) / "grasp_center"=合拢夹持点
PREGRASP_HOVER_GAP = 0.08           # m, 上面那个点悬停在 affordance 区正上方多少


def load_camera_xy(mesh_path) -> np.ndarray | None:
    """重建相机(人头)的水平位置 (2,). 取全程中位数 —— 头会小幅晃动
    (Grasp2 实测范围仅 ~2cm), 中位比首帧稳."""
    wf = os.path.join(os.path.dirname(mesh_path), "world_fused.npz")
    if not os.path.exists(wf):
        return None
    try:
        c2w = np.load(wf, allow_pickle=True)["c2w"]
    except Exception:
        return None
    return np.median(c2w[:, :3, 3], axis=0)[:2].astype(np.float64)


def camera_anchor_shift(mesh_path, robot_cam_xy, obj_first_xy) -> np.ndarray | None:
    """求 xy 平移量: 让重建相机落到机器人相机的 xy 上.

    ⚠ align_replay 已经把整组平移过一次 (物体首帧 xy -> 原点), 所以重建系里的
      相机 xy 在 env 系里其实是 (cam_xy − obj_first_xy). 这里必须减掉那一次,
      否则会平移两倍.
    """
    cam = load_camera_xy(mesh_path)
    if cam is None:
        return None
    cam_in_env = cam - np.asarray(obj_first_xy, dtype=np.float64)
    return np.asarray(robot_cam_xy, dtype=np.float64) - cam_in_env


def min_lift_for_clearance(wrist_pos, wrist_quat, hand, table_top_z,
                           clearance, lowest_fn, stride=1) -> float:
    """全程最小抬升 (m): 让**每一帧**张开的手最低点都 >= 桌面+clearance.

    ⚠ 旧实现只算 PreGrasp **单帧** (replay_grasp: lo_open = hand_lowest_world(wp[gs], wq[gs])),
      所以 clearance 只在抓取那一刻成立 —— 实测 Grasp2 仍有 56/166 帧手低于桌面.
      这里取全程 min.
    """
    lo = min(lowest_fn(wrist_pos[t], wrist_quat[t], hand)
             for t in range(0, len(wrist_pos), stride))
    return max(float((table_top_z + clearance) - lo), 0.0)


def pregrasp_hover_pose(afford_world, wrist_quat_gs, anchor_local,
                        hover_gap) -> np.ndarray:
    """PreGrasp 腕位置 (3,): 让手上的 `anchor_local` 悬停在 affordance 区正上方 hover_gap.

    朝向直接沿用重建在 PreGrasp 帧的腕朝向 —— 腕位姿(位置+朝向)是重建里唯一
    可信的通道 (replay_grasp 的可信度判断), 掌心法向不该我们瞎编.

    `anchor_local` = 手基座系里"拿来对准物体"的那个点, 换它就换了对准的含义:
      `grasp_center_local(hand)`   五指 elastomer 合拢构型质心 —— 对准**夹持点**.
                                   比"腕 + 掌法向 9cm"准 6.88cm, 但它下方还有 2.95cm
                                   的手 (合拢时掌法向鼓到 7.14cm, 中心只在 4.19cm),
                                   对着 2.6cm 厚的扁物体会戳桌.
      `finger_base_local(hand)`    中指根, 掌上固定点 —— 对准**掌根**. 位置更高更靠后,
                                   手整体会抬起来、往后退.
    """
    target = np.asarray(afford_world, dtype=np.float64) + np.array([0.0, 0.0, hover_gap])
    offset = F.rot_apply(np.asarray(wrist_quat_gs, dtype=np.float64)[None],
                         np.asarray(anchor_local, dtype=np.float64)[None])[0]
    return target - offset


def affordance_world(obj_pos, obj_quat, afford_local) -> np.ndarray:
    """affordance 区(热图重心)的世界位置 (3,)."""
    off = F.rot_apply(np.asarray(obj_quat, dtype=np.float64)[None],
                      np.asarray(afford_local, dtype=np.float64)[None])[0]
    return np.asarray(obj_pos, dtype=np.float64) + off
