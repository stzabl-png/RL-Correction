"""双臂 S1:两只手各抓各的物体 —— 侧状态容器 + 上下文切换。

## 为什么是这个架构(而不是把 env 重构成双臂)

`tasks/pregrasp/env.py` 有 1893 行,`self.object` / `self.q_ref` / `self.task_phase` 等
**单边量被引用 100+ 处**,奖励函数是整块的。把它们全参数化 = 大重构 + 高翻车风险。

本模块走另一条:**保持 env 的单边逻辑一字不动**,外面套一层"侧状态容器"——
每只手一份状态,调用 env 的方法前把该侧的状态**换进** `self.*`,调用完**换回**。
于是 `_pre_physics_step` / `_get_rewards` / `_get_dones` 都可以**原样跑两遍**。

代价:换入换出必须覆盖**全部**单边量,漏一个就是静默串台(A 手的动作写进 B 手的状态)。
所以 `SIDE_ATTRS` 是本模块唯一的真理来源,加字段必须同步加进去 ——
`assert_covered()` 会在 env 构造后扫一遍实例属性,把"看起来像单边量但不在清单里"的报出来。

## 适用范围

**只覆盖 S1(两手各抓各的,互不干涉)**。倒水的会合/倾倒(S3)是**双手耦合**任务,
奖励不可分离(目标是两物体的**相对**位姿),那时这个"跑两遍再相加"的结构就不够用了。
S3 需要在本模块之上再加一层"关系项",不在本轮。
"""
from __future__ import annotations

import contextlib

import torch

# ---------------------------------------------------------------------------
# 单边状态量清单 —— **本模块唯一的真理来源**
#
# 生成方式(可复现): 在 env.py 上跑
#   grep -oE 'self\.(arm_jids|hand_jids|object|q_ref|...)\b' tasks/pregrasp/env.py | sort -u
# 加新单边字段时**必须**同步加到这里, 否则两只手会共用它 = 静默串台。
# ---------------------------------------------------------------------------
SIDE_ATTRS = (
    # -- 机器人侧: 关节索引与目标 --
    "arm_jids", "hand_jids", "arm_tgt", "finger_tgt", "finger_tgt_prev",
    "q_pregrasp", "q_open", "q_close", "q_ref", "_generic_perm",
    # -- 残差与合拢内部状态 --
    "closure", "fin_delta", "fin_res", "finger_res_scale", "finger_dev_max",
    # -- 参考轨迹 --
    "ref_wrist_pos", "ref_wrist_quat", "gs", "grasp_start", "grasp_end",
    # -- 物体 --
    "object", "obj_init_pos", "obj_init_quat", "obj_start_pos", "obj_start_quat",
    # -- 相位与判据 --
    "task_phase", "succeeded", "got_candidate", "cand_run", "verify_k",
    "verify_ok_run", "tilt_max_deg",
    # -- 抓取先验 --
    "_grasp_pos_w", "_grasp_quat_w", "aff_local", "_aff_pts_local",
    "n_active", "finger_active", "fmap",
    # -- 诊断累计 --
    "_diag_contacts", "_diag_actnorm", "_diag_n", "_diag_fgate", "_diag_fgate_n",
    # ===== 以下 40 项由 assert_covered() 在真实 env 上扫出来(我手写的清单漏了近一半)=====
    # 每漏一个就是一次"两只手静默共用它" —— 所以清单必须由扫描器维护, 不能靠人记。
    # -- 臂:索引/限位/残差尺度/误差计数 --
    "arm_bids", "arm_center", "arm_dev_hi", "arm_dev_lo", "arm_effort_limit",
    "arm_err_ctr", "arm_joint_names", "arm_lower", "arm_upper", "arm_res_scale",
    "arm_tgt_prev", "hand_joint_names", "tip_ids", "_joint_hand",
    # -- 手指 --
    "finger_res", "finger_weight", "ref_finger", "pad_touched", "prev_pad_d",
    # -- 物体(逐侧各一个) --
    "obj_fric", "obj_mass", "obj_p0", "obj_points", "obj_rest_z",
    # -- 参考轨迹的派生量 --
    "ref_bad_segs", "ref_contact", "ref_ik_err", "ref_ik_ok", "ref_ik_subst",
    "ref_obj_pos", "ref_obj_quat", "ref_obj_vel", "ref_q_prev", "ref_t",
    # -- 相位/验证/抓取先验 --
    "phase_step", "phase_timeout_t", "started_grasp", "grasped_once",
    "verify_lvl", "verify_oz0", "verify_wz0", "vf_obj_mm", "_prior_q_grasp",
)

# 看起来像单边量、但**故意共用**的(共用是对的, 别加进 SIDE_ATTRS)
SHARED_OK = {
    "hand",            # articulation: 一个机器人两条臂, 是同一个 articulation
    "scene", "sim", "cfg", "device", "num_envs",
    "episode_length_buf", "actions_buf", "prev_actions",   # 动作向量是拼起来的整体
    "table_top_z", "obj_normals",
}


class SideState:
    """一只手的全部单边状态。构造时从 env 抓一份快照。"""

    __slots__ = ("name", "data")

    def __init__(self, name: str, env):
        self.name = name
        self.data = {k: getattr(env, k) for k in SIDE_ATTRS if hasattr(env, k)}

    def keys(self):
        return self.data.keys()


@contextlib.contextmanager
def use_side(env, side: SideState):
    """把 `side` 的状态换进 env, 退出时**把 env 上的当前值写回 side**。

    写回这一步是必须的: env 的方法里大量是**重新绑定**(`self.closure = ...`)而不是
    原地写, 不写回的话那一侧的更新会丢。原地写(`buf[idx] = v`)两种方式都安全。
    """
    saved = {k: getattr(env, k, None) for k in side.data}
    for k, v in side.data.items():
        setattr(env, k, v)
    try:
        yield
    finally:
        for k in side.data:                       # 先回收本侧的最新值
            side.data[k] = getattr(env, k)
        for k, v in saved.items():                # 再恢复调用前的现场
            setattr(env, k, v)


def assert_covered(env, extra_ok=()) -> list[str]:
    """扫 env 实例属性, 报出"像单边量但不在 SIDE_ATTRS 里"的可疑字段。

    判据是保守的启发式(名字里带 arm/hand/finger/obj/grasp/ref/phase 等),
    **不会自动加**, 只打印让人来判。漏一个字段的后果是静默串台, 值得每次构造都扫一遍。
    """
    import re
    pat = re.compile(r"(arm|hand|finger|obj|grasp|ref|phase|closure|tip|pad|cand|verify)",
                     re.I)
    ok = set(SIDE_ATTRS) | SHARED_OK | set(extra_ok)
    sus = []
    for k, v in vars(env).items():
        if k in ok or not pat.search(k):
            continue
        if isinstance(v, (torch.Tensor, int, float, list, tuple)) or hasattr(v, "data"):
            sus.append(k)
    if sus:
        print(f"[bimanual] ⚠ 这些字段名字像单边量但不在 SIDE_ATTRS 里, 请人工确认是否该分侧:\n"
              f"           {sorted(sus)}")
    return sorted(sus)
