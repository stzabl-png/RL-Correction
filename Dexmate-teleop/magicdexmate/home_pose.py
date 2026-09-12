"""机器人的默认姿势 —— 一段遥操作的起点,唯一的一份。

为什么要有这个文件:这个常量此前有两份副本(`sim/vega_scene.py` 的 init_state
和 `sim/pink_vega_ik.py` 的 HOME_ARM),两处都写着「MUST stay in sync」——
靠注释维持的一致性,这个仓库为同类问题栽过 4 次。而现在真机那边也要用它
(回合开始时把机器人送回这个姿势),第三份副本不能再出现。

**纯 Python,不 import 任何东西**,所以 Isaac 环境、dexcontrol 环境、分析环境
都能直接 import,不会把各自的依赖拖进来。

姿势本身是 2026-07-27 定的「收拢」位:双手朝胸前收,整个遥操作包络
(前 25cm / 侧 30cm / 上 30cm,朝向保持)都能解到 0mm;更早那个伸展位只剩
约 15cm 的前向余量,每次外伸都要赔 86 到 106mm。
"""
from __future__ import annotations

ARM_HOME: dict[str, float] = {
    "L_arm_j1": -0.5, "L_arm_j2": 0.30, "L_arm_j3": 0.0, "L_arm_j4": -2.2,
    "L_arm_j5": -0.4, "L_arm_j6": 0.0, "L_arm_j7": 0.0,
    "R_arm_j1": 0.5, "R_arm_j2": -0.30, "R_arm_j3": 0.0, "R_arm_j4": -2.2,
    "R_arm_j5": 0.4, "R_arm_j6": 0.0, "R_arm_j7": 0.0,
}

HEAD_HOME: dict[str, float] = {"head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0}


def arm_home_full() -> dict[str, float]:
    """14 个手臂关节的完整默认位(没有列出的关节都是 0)。"""
    return dict(ARM_HOME)


def home_for(components: set[str] | list[str]) -> dict[str, float]:
    """按部件取默认位。躯干不在这里:它由操作者手工摆放,或者由映射驱动,
    两种情况下都不该被「回默认位」这个动作碰到(与桥接 goal_of 的约定一致)。"""
    comps = set(components)
    out: dict[str, float] = {}
    if "left_arm" in comps:
        out.update({n: v for n, v in ARM_HOME.items() if n.startswith("L_")})
    if "right_arm" in comps:
        out.update({n: v for n, v in ARM_HOME.items() if n.startswith("R_")})
    if "head" in comps:
        out.update(HEAD_HOME)
    return out
