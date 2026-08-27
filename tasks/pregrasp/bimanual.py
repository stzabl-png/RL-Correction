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
    # v10.8 松手臂冻结快照 (pour_place 专用, 懒创建 —— 仅 bi_native 底座
    # 的 setattr 路由能承接懒创建, v1 快照底座不支持; pour 线全是 native)
    "_rel_arm_q", "_rel_arm_has",
    # E2E RETREAT 出生站姿快照 (撤退 lerp 终点, 同为懒创建)
    "_stance_q",
    "q_pregrasp", "q_open", "q_close", "q_ref", "_generic_perm",
    "_contact_sensors",
    "_fcd_keys", "_fcd_close_t", "_fcd_shape_done", "_fcd_pad_done", "_fcd_ref_last",
    "_fcd_cand_ema", "_fcd_g",
    "_c5_pad_done", "_cc_run", "_cc_done", "_fin_ref_path", "_seg_rows", "_thumb_mask",
    # 2026-08-24: 外壳-物体距离逐侧 —— 漏进清单时几何审计两侧读到同一份(实测
    # 右/左逐 body 距离逐位相同 1.20, 而 _sig 里的分侧值是 0.55/1.20)
    "_shell_obj_d", "_shell_obj_per", "_fc_meas_op", "_fc_meas_oq",
    # -- 残差与合拢内部状态 --
    "closure", "fin_delta", "fin_res", "arm_res", "finger_res_scale", "finger_dev_max",
    # 握力信任标量 + 其滑移基线快照 (2026-08-25): 逐侧逐 env —— 两只手可能一只滑一只不滑
    "_grip_g", "_grip_ref_p", "_grip_has",
    # -- 参考轨迹 --
    "ref_wrist_pos", "ref_wrist_quat", "gs", "grasp_start", "grasp_end",
    # -- 物体 --
    "object", "obj_init_pos", "obj_init_quat", "obj_start_pos", "obj_start_quat",
    # -- 相位与判据 --
    "task_phase", "succeeded", "got_candidate", "cand_run", "verify_k",
    "arrive_step_abs",   # 到位时刻的绝对回合步 (相位日程表的渐入窗口用)
    "verify_ok_run", "tilt_max_deg",
    # -- L5 审计补录 (2026-08-17): 这三个此前漏了 --
    # _approach_hit: _get_dones 写/_get_rewards 读, 不分侧则 A 的撞击罚错拿 B 的值
    # verify_ang0: twist 验证基线(懒创建 → env.__init__ 已改为预创建, 否则快照抓不到)
    # _fgate_dg: 手指门控标定距离(float), 瓶/杯几何不同必须分侧
    "_approach_hit", "verify_ang0", "_fgate_dg",
    # L5 c(d) 耦合缓存 (pre_physics 写 / reward 读, 与 _approach_hit 同一类跨方法读写)
    "_l5_dp", "_l5_cref",
    # -- 抓取先验 --
    "_grasp_pos_w", "_grasp_quat_w", "_pregrasp_w", "aff_local", "_aff_pts_local",
    "_p2_arm", "_p2_fin", "_p2_t",     # Phase2 斜坡遗产 (2026-08-18 晚)
    # Phase2-RL: 真抓姿靶点 + 第二段判据/里程碑状态 (逐侧)
    "_g2_pos_w", "_g2_quat_w", "_g2_run", "_g2_done", "_m1_done", "_m2_done",
    "_g2_gates", "_g2_a", "_g2_d", "_g2_fe",   # 逐闸诊断也必须分侧, 否则两侧读同一份
    "_g2_d", "_g2_fe", "_m2_ladder", "_m2_lvl",   # 裁定C 派生阶梯 (逐侧)
    "_fc_local", "_fc_done", "_fc_prev", "_fc_d",   # FC 逐指笛卡尔目标 (逐侧)
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
    # ===== 2026-08-17 实测补的: 末端 body =====
    # `_resolve_joint_ids` 设了 6 个字段, 上面只覆盖了 4 个 —— `ee_body`/`ee_id` 漏了。
    # 后果不是崩, 是**静默串台的测量**: 两侧的 d_pos/腕位/臂间隙全用**同一只手腕**算,
    # 双臂冒烟实测 R/L 的 d_pos 逐位相同(59.018 vs 59.018)而 action_norm 不同
    # (0.268 vs 0.000) —— 动作分开了、测量没分开。
    # ⚠ `assert_covered` 的正则不含 "ee", 所以扫不出来 —— 正则已同步补上。
    "ee_body", "ee_id",
    # ===== 2026-08-17 全量列举后补的 (list_uncovered 报了 75 个, 逐条裁) =====
    # 判据: 这个量在两只手上**会不会取不同的值**。会 -> 分侧; 不会(物体/桌面/机器人
    # 全局属性) -> 留在 SHARED_OK。手写清单和关键词扫描各漏一批, 只有全量列举才裁得干净。
    # -- 每回合的状态机/判据 (逐 env, 显然逐侧) --
    "arrived", "arrive_step", "switch_run", "contact_run", "goal_gate", "hold_ok",
    "close_started", "close_prog", "any_contact", "freeze_ctr", "rsi_start", "was_rsi",
    "reset_buf", "reset_terminated", "reset_time_outs",
    # -- 势函数/差分用的"上一步"量 (串了就是奖励串台) --
    "prev_phi", "prev_goal_dist", "prev_palm_dist", "prev_quality", "prev_valid",
    "prev_wrist_pos", "q_base_prev",
    # -- 臂指令与残差界 --
    "q_cmd", "band_lo", "band_hi", "res_scale", "res_step_cm", "_dyn_u", "ee_jac",
    # -- 腕目标/抖动 (逐侧各有各的目标) --
    "wrist_tgt_pos", "wrist_tgt_quat", "wrist_jitter", "anchor_local", "grip_rel0",
    # -- 退避参考族 (逐侧各一条路径) --
    "retract_path", "retract_q", "retract_d", "retract_d0",
    # -- 抬升/验证 --
    "q_lift", "q_lift_delta", "p_lift", "lift_hw", "vf_has", "vf_wrist_mm",
    "tilt_final_deg", "yaw_final_deg", "yaw_max_deg", "wrench_norm",
    # -- 手型模板/关节限位 (跟着 hand_jids 走, 逐侧) --
    "open_pose", "closed_pose", "dof_lower", "dof_upper", "pend_idx", "pend_w",
    # -- 亲和/接触带 (跟着**各自的物体**走) --
    "aff_points", "aff_heat",
    # -- 起点池 (逐侧各一个池: 两只手的可达状态完全不同) --
    "_sp_q", "_sp_n", "_sp_ptr", "_sp_arr", "_sp_seen", "_sp_start_bin",
    "_sp_best_d", "_sp_best_q", "_sp_edges",
    # -- 到位公差快照 --
    "_eps_final",
    # -- 本体感觉历史 (喂给网络的, 逐侧) --
    "proprio_hist",
    # ===== 2026-08-17 最后两个漏网的 (构造时是空 dict, 检测器的 `and v` 跳过了) =====
    # `_sig`   : 每步算好的全部信号(d_pos/d_rot/arm_gap/接触...), 奖励与日志都读它。
    #            不分侧 = **两只手的距离/间隙全是后算那一侧的** —— 冒烟实测 R/L 的
    #            d_pos 逐位相同(86.601), 而 action_norm 不同 ⟹ 动作分开、测量没分开。
    # `_ep_sums`: 奖励分项的逐回合累计, 不分侧 = ep_rew/* 两侧互相污染。
    "_sig", "_ep_sums",
    # ===== 我一开始裁错进 SHARED_OK 的 (2026-08-17 冒烟抓回来) =====
    # `arm_bids` 用本侧前缀、`self_bids` 用**对侧**前缀(要避让的正是另一条臂),
    # `shell_*` 由 arm_bids 派生 ⟹ 全都逐侧。裁成共用的后果: 两只手用**同一条臂**
    # 的外壳算离桌间隙, 冒烟实测 R/L 的 arm_table_gap 逐位相同 7.027。
    # `cross_a/cross_b` 是手指 body 对, 跟着 hand_side 走, 同样逐侧。
    "self_bids", "shell_bids", "shell_pts", "cross_a", "cross_b",
    # `table_bids` 名字像"桌子的索引", 实际是**要与桌/物做碰撞检查的手部连杆**
    # (env.py 注释原文: "只覆盖手 + l7/l8/ee") ⟹ 逐侧。裁成共用的后果: B 侧拿
    # **A 侧的手**去和自己的物体判接触。名字误导是这类错误的常见来源。
    "table_bids",
    # 子步插值计数: 每侧各推进一次, 共用会让 w 双倍速冲到 1.0(插值形同虚设)
    "_substep",
)

# 看起来像单边量、但**故意共用**的(共用是对的, 别加进 SIDE_ATTRS)
SHARED_OK = {
    "hand",            # articulation: 一个机器人两条臂, 是同一个 articulation
    "scene", "sim", "cfg", "device", "num_envs",
    "episode_length_buf",
    # ⚠ actions_buf/prev_actions: 概念上是"拼起来的整体", 但父类 `_pre_physics_step`
    #   会 `copy_` 单侧切片进去 —— 共用直接维度崩。双臂 env 在构造时把它们**逐侧
    #   覆盖成单侧宽度**(见 bimanual_env)。这里留着只是表示"默认不分侧"。
    "actions_buf", "prev_actions",
    "actions",         # 本步动作(拼起来的 14 维), 由双臂 env 自己切
    "table_top_z", "obj_normals",
    # (原来这里放 table_bids —— 裁错了, 见下面 SIDE_ATTRS 里的说明)
    # -- 接触分数图: 按物体格点, 与手无关 --
    "score_n", "score_s",
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
def use_side(env, side):
    """把 `side` 的状态换进 env, 退出时**把 env 上的当前值写回 side**。

    v2 分派 (2026-08-18): 原生双臂底座 (bimanual_native_env) 的侧状态常驻命名空间,
    切侧是指针翻转 —— 对它直接转发 env._use(side), 快照机制只服务 v1。
    工具脚本 (check_ff_bi 等) 因此对两代底座通用。

    写回这一步是必须的: env 的方法里大量是**重新绑定**(`self.closure = ...`)而不是
    原地写, 不写回的话那一侧的更新会丢。

    ⚠ **原地写不是自动安全的** (2026-08-17 更正, 原注释说"两种方式都安全", 错):
    `SideState` 存的是**引用**。若两侧的某个键指向**同一块**缓冲, 原地写
    (`buf[idx] = v`) 会互相覆盖 —— 而且字段明明在 SIDE_ATTRS 里、明明被"换"了,
    所以查起来极其隐蔽。**构造第二侧后必须把张量深拷贝一份**(见 bimanual_env)。
    """
    if getattr(env, "_bi_native", False):
        with env._use(side):
            yield
        return
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


def list_uncovered(env, extra_ok=()) -> list[tuple[str, str]]:
    """列出**所有**没被 SIDE_ATTRS/SHARED_OK 覆盖的张量类字段(不做名字过滤)。

    为什么要这个: `assert_covered` 靠关键词正则挑"看起来像单边量"的, 而**关键词漏一个
    就对那一族完全失明** —— 2026-08-17 实测漏了 `ee_body`/`ee_id`(名字里没有任何
    关键词), 后果是两只手用同一只手腕测距离, 训练照跑、指标照出、结论全错。
    打地鼠不如全量列出来人工裁一遍。
    """
    ok = set(SIDE_ATTRS) | SHARED_OK | set(extra_ok)
    out = []
    for k, v in sorted(vars(env).items()):
        if k in ok:
            continue
        if isinstance(v, torch.Tensor):
            out.append((k, f"Tensor{tuple(v.shape)}"))
        elif isinstance(v, dict):
            # ⚠ **空字典也要报**。第一版写了 `and v` 跳过空的 —— 而 `_sig`/`_ep_sums`
            #   构造时正好是空的、运行时才填满, 于是检测器全绿而它们在串台。
            #   "构造时空" 恰恰是最危险的一类: 静态看不见, 动态才出问题。
            out.append((k, f"dict[{len(v)}]" + (" of Tensor" if v and all(
                isinstance(x, torch.Tensor) for x in list(v.values())[:3]) else " (构造时空)")))
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
            out.append((k, f"{type(v).__name__}[{len(v)}]"))
    return out


def assert_covered(env, extra_ok=()) -> list[str]:
    """扫 env 实例属性, 报出"像单边量但不在 SIDE_ATTRS 里"的可疑字段。

    判据是保守的启发式(名字里带 arm/hand/finger/obj/grasp/ref/phase 等),
    **不会自动加**, 只打印让人来判。漏一个字段的后果是静默串台, 值得每次构造都扫一遍。
    """
    import re
    # ⚠ 关键词漏一个 = 扫描器对那一族字段完全失明。2026-08-17 补 "ee"(末端 body):
    #   ee_body/ee_id 名字里没有上面任何一个词, 于是漏了两个真单边量而扫描器全绿。
    pat = re.compile(r"(arm|hand|finger|obj|grasp|ref|phase|closure|tip|pad|cand|verify|ee)",
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
