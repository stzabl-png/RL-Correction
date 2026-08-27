"""GraspTaskEnv v2 — PreGrasp 出发的稳定抓握 + 微抬升验证 (阶段机, 事件门控).

继承 DexmateCorrectionEnv 只拿**验证过的底层**:
  场景/材质/传感器 (_setup_scene)、ref builder 摆放 + IK (q_ref, 预解 q_lift)、
  增益标定、子步插值下发 (_apply_action)、关节分组、GPU 槽位.
任务层 (动作语义 / 阶段机 / 奖励 / 终止 / 复位 / 观测) 全部在这里重写.

## 时间轴

  [静置5 (物体钉住, 动作不生效)]
  → GRASP (≤120 步): 合拢参考斜坡 + 策略调制, 形成候选抓取 (瞬时判据持续 0.4s)
  → VERIFY (≤20 步): **硬编码**腕升 1cm (q_lift 前 4 级插值), 物体跟上 + 接触不丢
  → 成功 (+20, 终止)

## 动作 13 维

  a = [Δq_arm(7), a_c(1), a_δ(5)] ∈ [-1,1]
  臂:   增量累积 (每步 = 标定界×0.25 ≈ 末端 5mm), 相对 q_pregrasp 钳 ±arm_dev_max.
  合拢: c += ref_rate + a_c·rate_max —— **零动作 = 匀速合拢** (参考动作附近探索,
        与 correction 任务"零残差 = 跟参考"同构); a_c=-1 恰好停住.
  每指: δ_i 累积 ±delta_max, 指 i 的模板深度 = clip(c+δ_i) —— 只调落点, 无 22 维冗余.

## 奖励要点 (完整权重见 cfg)

  质量势差分 Q = 向心 − λ|ΣF|/Σ|F| − λ·净力矩, 只奖 ΔQ —— 原地保持不刷分,
  成功终止不再是"断掉收入流"的坏事. 撞桌/指间交叉是几何罚 (物理层管不了, 见 cfg).
"""
from __future__ import annotations

from collections.abc import Sequence

import os

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from rl_rebuild.correction import clips
from rl_rebuild.correction.env.dexmate_env import DexmateCorrectionEnv
from rl_rebuild.correction.ref_builders.replay_grasp import (
    GENERIC_CLOSED, GENERIC_JOINT_ORDER, GENERIC_OPEN, _affordance_target,
    grasp_center_local)

from tasks.pregrasp.cfg import GraspTaskCfg, Phase

FINGERS = ("thumb", "index", "middle", "ring", "pinky")   # 与 fingertip_bodies 同序


def _slerp_R(R0, R1, t):
    """两个旋转矩阵之间的最短弧插值 (轴角形式, 不引依赖)。"""
    import numpy as _n
    dR = R0.T @ R1
    c = (_n.trace(dR) - 1.0) / 2.0
    th = float(_n.arccos(_n.clip(c, -1.0, 1.0)))
    if th < 1e-8:
        return R1.copy()
    w = _n.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) \
        / (2.0 * _n.sin(th))
    a = th * t
    K = _n.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    return R0 @ (_n.eye(3) + _n.sin(a) * K + (1 - _n.cos(a)) * K @ K)


class GraspTaskEnv(DexmateCorrectionEnv):
    cfg: GraspTaskCfg
    # Approach-only 要关掉的奖励项 (用户 2026-08-16 逐条裁定, 见 _get_rewards 注释)
    #   imit  —— 与 align 对冲(本任务不跟人手轨迹, 没有要模仿的参考)
    #   fail  —— 物体掉落/甩飞判据, 接近段不碰物体所以用不上
    #   其余  —— 抓取段专用, 接近段全部恒 0 (从抓取任务继承的壳)
    # ⚠ `cone` 不在这里: 它由 cfg.cone_trust 控制且默认 False, 本来就没生效。
    # ⚠ `self_gap` **不关**: 恒 0 无代价, 双手任务(臂↔另一臂)会真的用上。
    #   ⚠ `imit` 2026-08-16 **重新启用**: 用户指出退避族自带一条**保证可行**的回程
    #      (退避过程倒放), 可以当参考。之前关掉它的理由是"没有可信参考要跟" ——
    #      那个理由在参考换成 retract_path 之后就不成立了。权重给小(当提示不当枷锁)。
    APPROACH_OFF = (
        "fail",
        "pad_approach", "pad_touch", "cent_income", "quality_prog", "hold",
        "over_force", "obj_move", "obj_rot", "tilt", "push", "finger_cross",
        "carry",
    )
    # 手指距离门控的自标定基准。类属性 = 只在 clip 没有 GraspPose 时兜底;
    # 有 prior 时由 __init__ 里的 `_load_grasp_prior()` 覆盖成实例属性。
    # (放这里而不是 __init__ 里, 是为了不可能再把算好的值覆盖回 None —— 踩过这个坑)
    _fgate_dg = None

    def __init__(self, cfg: GraspTaskCfg, render_mode: str | None = None, **kwargs):
        # ---- 接近任务强制外壳口径 (2026-08-16, docs/APPROACH_DESIGN.md §4) ----
        # `arm_table_shell` 默认关是**有意的**(已盖章的旧任务保持原口径), 这里不动默认值,
        # 只在退避式接近任务里强制打开 —— 因为人手轨迹上**右臂外壳穿桌 −1.76cm 而手部
        # 连杆一直 ≥2cm**, 原点口径**根本看不见臂在刮桌**(l5/l6 截面半径 4~6.7cm)。
        # 底线是"抓稳之前不许撞桌", 用看不见撞击的口径去判就是自欺。
        if getattr(cfg, "approach_only", False):
            # 回合预算 250 步 @20Hz = 12.5s (用户定)。理论下限来自标定的每步末端位移
            # 上界 5mm(= arm_residual_max 的 2cm × arm_step_scale 0.25):
            # 站姿->GraspPose 30.1cm(Grasp3) 需 ≥60 步 / 41.4cm(瓶) 需 ≥83 步。
            # episode_length_s 必须跟着放大, 否则 12.0s 会先把瓶子那条截断。
            cfg.episode_length_s = max(cfg.episode_length_s,
                                       cfg.approach_only_steps * cfg.decimation
                                       * cfg.sim.dt * 1.05)
            cfg.retract_start = True          # 接近任务默认用退避起点族
        if getattr(cfg, "retract_start", False) and not getattr(cfg, "arm_table_shell", False):
            cfg.arm_table_shell = True
            print("[approach] retract_start=1 -> 强制打开 arm_table_shell "
                  "(臂罚/诊断改用连杆外壳采样点; 原点口径看不见上臂刮桌)")
        # ---- 双物体螺旋装配 (screw 27): clip 带 secondary 时自动激活 ----
        from tasks.pregrasp import screw_assembly as SA
        self._SA = SA
        SA.pre_init(self, cfg)
        super().__init__(cfg, render_mode, **kwargs)
        SA.init_state(self)
        dev, N = self.device, self.num_envs
        to = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)

        # ---- 站姿前缀 (2026-08-03, PLAN D2): q_ref 前拼 "默认站姿 -> 人手轨迹起点" ----
        # 端到端出发点应是 cfg.dexmate_joints 对称待命站姿, 而人手重建第 0 帧已悬在
        # 物体附近. 关节空间 smoothstep 插值 K 帧, grasp_start 顺延 —— 下游 gs/预算/
        # 课程/RSI/前馈/φ 全部由 gs 派生, 自动一致. ref_wrist_* 用端点插值近似
        # (它只喂增量式模仿罚和 d_ref 差分, 直线速度剖面正是想要的参考形状).
        _K = int(getattr(cfg, "stance_prefix_frames", 0) or 0)
        if cfg.approach and _K > 0:
            q_stance = self.hand.data.default_joint_pos[0, self.arm_jids].to(dev)
            _s = torch.linspace(0.0, 1.0, _K + 1, device=dev)[:-1]
            _s = _s * _s * (3.0 - 2.0 * _s)                    # smoothstep 软启停
            self.q_ref = torch.cat(
                [q_stance + _s.unsqueeze(1) * (self.q_ref[0] - q_stance), self.q_ref], 0)
            _w0 = (self.wrist_pos_w[0] - self.scene.env_origins[0]).to(dev)
            _wq0 = self.wrist_quat_w[0].to(dev)
            self.ref_wrist_pos = torch.cat(
                [_w0 + _s.unsqueeze(1) * (self.ref_wrist_pos[0] - _w0),
                 self.ref_wrist_pos], 0)
            _q1 = self.ref_wrist_quat[0]
            if float((_wq0 * _q1).sum()) < 0.0:
                _q1 = -_q1                                     # 同半球再 nlerp
            self.ref_wrist_quat = torch.cat(
                [torch.nn.functional.normalize(
                    _wq0 + _s.unsqueeze(1) * (_q1 - _wq0), dim=1),
                 self.ref_wrist_quat], 0)
            self.grasp_start = int(self.grasp_start) + _K
            self.grasp_end = int(self.grasp_end) + _K   # 放置帧标注同步顺延 (place 用)
            self.L = int(self.L) + _K          # 基类 clip 长度与 q_ref 保持一致
            _d = float((self.ref_wrist_pos[_K] - self.ref_wrist_pos[0]).norm()) * 100
            print(f"[stance_prefix] K={_K} 帧 | 站姿->轨迹起点 腕直线 {_d:.1f}cm "
                  f"(峰值 ≈{_d * 15.0 / max(_K, 1):.1f}mm/帧, 上限 15) | "
                  f"gs {self.grasp_start - _K} -> {self.grasp_start}")

        # ---- PreGrasp 姿态 (臂参考钉死在这一帧) ----
        self.q_pregrasp = self.q_ref[self.grasp_start].clone()          # (7,)
        assert getattr(self, "q_lift", None) is not None or cfg.grasp_prior_npz, \
            "微抬升验证依赖基类预解的 q_lift (place_mode=ref_builder 路径; " \
            "prior 模式会在 _load_grasp_prior 里自行重解)"

        # ---- 手型模板 (USD 关节序) + 关节→手指映射 ----
        side = cfg.hand_side
        names = [n.replace("right_", f"{side}_") for n in GENERIC_JOINT_ORDER]
        perm = [names.index(n) for n in self.hand_joint_names]
        self._generic_perm = perm            # GENERIC 序 -> USD 关节序 (prior 加载器也要用)
        self.q_open = to(GENERIC_OPEN[perm])                            # (22,)
        self.q_close = to(GENERIC_CLOSED[perm])
        fmap = []
        for n in self.hand_joint_names:
            fid = [i for i, f in enumerate(FINGERS) if f"_{f}_" in n or n.endswith(f"_{f}")]
            assert len(fid) == 1, f"关节 {n} 无法归属到唯一手指"
            fmap.append(fid[0])
        self.fmap = torch.tensor(fmap, dtype=torch.long, device=dev)    # (22,) ∈ [0,5)
        # ---- 模板参与指集合 (2026-08-05, 用户裁定"判据按模板指数走") ----
        # 默认五指全参与; prior 加载器按接触点归属自动改写 (2 指 Tip_Pinch 模板
        # → {thumb,index}), 同步派生 min_pads / cent 归一 / 非参与指姿态钉死.
        self.finger_active = torch.ones(5, device=dev)                  # (5,) 0/1
        self.n_active = 5
        self._aff_pts_local = None       # 视频接触带 (物体局部系); 只喂 pad_approach 塑形

        # ---- 表面法向 (cent_mode="normal" 用): 与 obj_points 一一对应的外法向 ----
        self.obj_normals = None
        if cfg.cent_mode == "normal" and self.obj_points is not None:
            import trimesh
            _m = trimesh.load(self.du.mesh_path, force="mesh")
            _vi = _m.kdtree.query(self.obj_points.cpu().numpy().astype(np.float64))[1]
            self.obj_normals = to(np.asarray(_m.vertex_normals)[_vi])   # (P,3) 规范系
            print(f"[pregrasp] 表面法向已采样 {tuple(self.obj_normals.shape)} "
                  f"(cent 用局部法向, 形状无关)")

        # ---- 抓取几何: 合拢中心锚点 (手基座系) + affordance 目标 (物体系) ----
        self.anchor_local = to(grasp_center_local(side))                # (3,)
        e = clips.clip_entry(cfg.clip_name)
        if e.get("affordance"):
            self.aff_local = to(_affordance_target(e["affordance"]))    # (3,)
        else:
            # prior 模式对齐目标来自 GraspPose 接触质心 (下面 _load_grasp_prior 覆盖),
            # 不需要热图; 无 prior 又无热图才是配置错误.
            assert cfg.grasp_prior_npz, \
                f"{cfg.clip_name} 没有 affordance 也没有 GraspPose prior —— 缺对齐目标"
            self.aff_local = to(np.zeros(3))

        # ---- 碰撞惩罚的 body 集合 ----
        bn = list(self.hand.body_names)
        P = "R" if side == "right" else "L"
        self.table_bids = [i for i, n in enumerate(bn)
                           if n.startswith(f"{side}_")
                           or n in (f"{P}_arm_l7", f"{P}_arm_l8", f"{P}_ee")]
        self.cross_a = [bn.index(f"{side}_{f}_DP") for f in ("index", "middle", "ring")]
        self.cross_b = [bn.index(f"{side}_{f}_DP") for f in ("middle", "ring", "pinky")]
        # ---- P0.3 (2026-08-02): 接近段要扫过 1.6 秒的空间, 而这条臂 l1~l6 六节
        # **完全没有任何碰撞约束** —— 上面那个 table_bids 只覆盖手 + l7/l8/ee.
        # 而且手↔桌物理碰撞被 filter_collisions 过滤、PhysX 自碰撞不能开(会臂自锁),
        # ⟹ 接近段的一切碰撞约束**只能是几何罚**. 这里把臂补全 + 加躯干/另一臂间隙.
        # ⚠ 命名不统一: l1/l6 带 `vega_1p_` 前缀, l2~l5/l7/l8 不带 —— 用**子串**匹配.
        O = "L" if P == "R" else "R"
        self._build_collide_ids(bn, P, O, to)   # 逐侧: 双臂任务要按侧各建一份

        # ---- Dexonomy GraspPose prior (可选; 替换 q_pregrasp / q_close / aff_local) ----
        if cfg.grasp_prior_npz:
            self._load_grasp_prior(cfg.grasp_prior_npz, to)

        # ---- 方案一: 参考姿态通道混合 (§2.15) ----
        # 位置轨迹不动 (人手形状全保留); 腕姿态按走过弧长 smoothstep 从起点姿态
        # slerp 到 GraspPose 姿态, 0..gs 重解 IK —— 前馈自带"边前进边转向",
        # 第一次贴近时角度已达标, 切换判据满足即脱离参考 (单次接近).
        if cfg.orient_blend:
            assert cfg.grasp_prior_npz and getattr(self, "_grasp_quat_w", None) is not None, \
                "--orient_blend 需要 GraspPose prior (混合终点姿态来自它)"
            _g = int(self.grasp_start)
            _P = self.ref_wrist_pos[:_g + 1].cpu().numpy().astype(np.float64)
            _seg = np.linalg.norm(np.diff(_P, axis=0), axis=1)
            _u = np.concatenate([[0.0], np.cumsum(_seg)]) / max(float(_seg.sum()), 1e-9)
            _s = _u * _u * (3.0 - 2.0 * _u)
            _q0 = self.ref_wrist_quat[0].cpu().numpy().astype(np.float64)
            _q1 = self._grasp_quat_w.cpu().numpy().astype(np.float64)
            if float(np.dot(_q0, _q1)) < 0.0:
                _q1 = -_q1
            _th = float(np.arccos(np.clip(np.dot(_q0, _q1), -1.0, 1.0)))
            if _th < 1e-6:
                _Q = np.tile(_q0, (len(_s), 1))
            else:
                _Q = (np.sin((1.0 - _s)[:, None] * _th) * _q0[None]
                      + np.sin(_s[:, None] * _th) * _q1[None]) / np.sin(_th)
            _Q /= np.linalg.norm(_Q, axis=1, keepdims=True)
            from rl_rebuild.correction.kinematics import ArmIK
            _ik = ArmIK(cfg.hand_side, anchor_link="arm_center", anchor_T=self._anchor_T)
            _sols = _ik.solve_traj(_P, _Q,
                                   q_init=self.q_ref[0].cpu().numpy().astype(np.float64))
            _qn = np.stack([x["q"] for x in _sols])
            _ok = np.array([x["ok"] for x in _sols])
            _pe = np.array([x["pos_err"] for x in _sols])
            assert _ok.mean() >= 0.90, (
                f"[orient_blend] IK 可达率 {_ok.mean()*100:.0f}% < 90%, 判死 "
                "(§2.15 证伪信号①: 该位置路径配抓取姿态在运动学上走不通)")
            _bad = np.flatnonzero(~_ok | ~np.isfinite(_qn).all(1))
            if len(_bad):
                _gd = np.flatnonzero(_ok & np.isfinite(_qn).all(1))
                _qn[_bad] = _qn[_gd[np.abs(_gd[None] - _bad[:, None]).argmin(1)]]
            _qn[0] = self.q_ref[0].cpu().numpy()   # 帧 0 = 站姿关节, 分毫不动
            self.q_ref[:_g + 1] = to(_qn)
            self.ref_wrist_quat[:_g + 1] = to(_Q)
            print(f"[orient_blend] 0..{_g} 帧姿态已混合 (总转角 {np.degrees(_th):.0f}°) | "
                  f"IK 可达 {_ok.mean()*100:.1f}% | 位置误差中位 "
                  f"{np.median(_pe[_ok])*100:.2f}cm | 顶替 {len(_bad)} 帧")

        # ---- 方案二: 锥形信任管半径自标定 (§2.16) ----
        if cfg.cone_trust:
            assert cfg.grasp_prior_npz, "--cone 需要 GraspPose prior (终点缺口自标定)"
            self._ref_end_gap = float((self.ref_wrist_pos[int(self.grasp_start)]
                                       - self._grasp_pos_w).norm())
            print(f"[cone] 信任管: R {cfg.cone_r0*100:.0f}cm --smoothstep(φ)--> "
                  f"{(self._ref_end_gap + cfg.cone_margin)*100:.1f}cm "
                  f"(终点缺口 {self._ref_end_gap*100:.1f}cm + 余量 {cfg.cone_margin*100:.0f}cm)")

        # ---- 残差界 ----
        self.arm_res_scale = to(np.asarray(cfg.arm_residual_max)) * cfg.arm_step_scale
        self.arm_dev_lo = torch.maximum(self.q_pregrasp - cfg.arm_dev_max,
                                        self.arm_lower[0]).unsqueeze(0)  # (1,7)
        self.arm_dev_hi = torch.minimum(self.q_pregrasp + cfg.arm_dev_max,
                                        self.arm_upper[0]).unsqueeze(0)
        if cfg.grasp_prior_npz and getattr(self, "_prior_q_grasp", None) is not None:
            # 偏差带必须同时罩住 起点(pregrasp) 与 目标(grasp) 两个 IK 解
            gq = self._prior_q_grasp
            self.arm_dev_lo = torch.minimum(self.arm_dev_lo, (gq - 0.08).unsqueeze(0))
            self.arm_dev_lo = torch.maximum(self.arm_dev_lo, self.arm_lower[0:1])
            self.arm_dev_hi = torch.maximum(self.arm_dev_hi, (gq + 0.08).unsqueeze(0))
            self.arm_dev_hi = torch.minimum(self.arm_dev_hi, self.arm_upper[0:1])
        # 抓取相位的逐 env 偏差带 (中心 = arm_center, 见下). 接近段不用它, 只钳关节限位.
        self.band_lo = self.arm_dev_lo.expand(N, 7).clone()
        self.band_hi = self.arm_dev_hi.expand(N, 7).clone()

        # ---- 微抬升验证的关节斜坡: q_lift 级差 (2.5mm/级), 取到 verify_lift_m ----
        if cfg.verify_mode == "twist":
            # 微拧: q_lift 已被 prior 加载器替换成绕物体轴的旋转斜坡, 全程用满
            self.verify_lvl = float(self.cfg.lift_steps)
        else:
            lvl = cfg.verify_lift_m / (self.cfg.lift_height / self.cfg.lift_steps)
            self.verify_lvl = float(min(lvl, self.cfg.lift_steps))      # 例 1cm -> 4 级
        self.q_lift_delta = self.q_lift - self.q_lift[0:1]              # (21,7)

        # ---- 任务状态 ----
        self.closure = torch.zeros(N, device=dev)                       # 合拢 c
        # 握力信任标量 g (2026-08-25): 见 cfg.grip_g 的说明。逐 env, 分侧路由。
        self._grip_g = torch.zeros(N, device=dev)
        self._grip_ref_p = torch.zeros(N, 3, device=dev)   # 腕系下物体位置的基线快照
        self._grip_has = torch.zeros(N, dtype=torch.bool, device=dev)
        self.fin_delta = torch.zeros(N, 5, device=dev)                  # 每指残差 δ
        # ---- joints 模式: 逐关节手指残差 (2026-08-15) ----
        # 界按 calib_finger_residual 标定(逐关节, 与臂同口径), 换到 USD 关节序。
        _fr = np.asarray(cfg.finger_residual_max, dtype=np.float64)[self._generic_perm]
        self.finger_res_scale = to(_fr * cfg.finger_step_scale)         # (22,) 每步增量
        # ⚠ 原注释: "Phase2-RL 累计上限放开到全行程 (每步增量不变 ⟹ 单步扰动仍毫米级)"
        # ★ 2026-08-23 判死: 这正是造成臂漂移那个 bug 的**同一套错误推理** ——
        #   小步长 × 几百步 = 无界漂移。40.0 ⟹ 逐关节上限中位 316°, 等于没有界。
        #   实测 AAGE 7M 步: fin_track 单调恶化 右手 4.5x / 左手 124x, 左手合计
        #   +0.047 -> -0.106 (零动作基线 +0.060) —— 策略比什么都不做差三倍。
        #   臂用方案C 的 ±2.86° 硬界治好了, 手指必须同款处理 (--fin_dev_scale)。
        _fdev = (float(getattr(cfg, "phase2_fin_dev_scale", 40.0))
                 if getattr(cfg, "pregrasp29", False) else cfg.finger_dev_scale)
        self.finger_dev_max = to(_fr * _fdev)                           # (22,) 累积上限
        self.fin_res = torch.zeros(N, 22, device=dev)                   # 累积逐关节残差
        self.arm_res = torch.zeros(N, 7, device=dev)                    # 方案C: 臂累积残差
        # 分段探索门控用: 拇指关节掩码 (USD 序), 逐侧建 —— 左右手 USD 序可能不同
        self._thumb_mask = to(np.array(
            [1.0 if "thumb" in n else 0.0 for n in self.hand_joint_names],
            dtype=np.float64))                                          # (22,)
        self._joint_hand = (cfg.hand_action_mode == "joints")
        if self._joint_hand:
            print(f"[action] 手部 = **22 关节全放开** (a=29 维): 每步界 中位 "
                  f"{np.degrees(np.median(_fr * cfg.finger_step_scale)):.2f}°/关节, "
                  f"累积上限 中位 {np.degrees(np.median(_fr * _fdev)):.2f}° "
                  f"| a_c/a_δ 已取消")
        self.task_phase = torch.full((N,), Phase.GRASP, dtype=torch.long, device=dev)
        self.phase_step = torch.zeros(N, dtype=torch.long, device=dev)
        self.cand_run = torch.zeros(N, dtype=torch.long, device=dev)    # 候选判据连续计数
        self._diag_contacts = torch.zeros(N, device=dev)   # 盘面: 累计接触指数
        self._diag_actnorm = torch.zeros(N, device=dev)    # 盘面: 累计动作范数
        self._diag_n = torch.zeros(N, device=dev)          # 盘面: 步数
        # ⚠ 不要在这里写 `self._fgate_dg = None` —— 它由本 __init__ **上面**第 191 行的
        #   `_load_grasp_prior()` 算好(GraspPose 处的腕-物体距离)。2026-08-16 我在这里补了
        #   一行"默认值 None", 结果把算好的 15.1cm 抹成 None, `getattr(..., None)` 恒假 ⟹
        #   **手指距离门控从上线起一次都没执行过**(TB 里 diag/finger_gate 恒 0 就是这个,
        #   不是"门关死", 是"整段代码没跑")。默认值改成在类层面声明, 不覆盖实例值。
        self.retract_d0 = torch.full((N,), float('nan'), device=dev)  # 盘面: 本回合起点 d
        self._diag_fgate = torch.zeros(N, device=dev)      # 盘面: 手指门开度累计
        self._diag_fgate_n = torch.zeros(N, device=dev)
        self.verify_k = torch.zeros(N, dtype=torch.long, device=dev)    # 验证段步数
        self.verify_ok_run = torch.zeros(N, dtype=torch.long, device=dev)
        self.got_candidate = torch.zeros(N, dtype=torch.bool, device=dev)
        self.succeeded = torch.zeros(N, dtype=torch.bool, device=dev)
        self.pad_touched = torch.zeros(N, 5, dtype=torch.bool, device=dev)
        self.any_contact = torch.zeros(N, dtype=torch.bool, device=dev)
        # 姿态保持仪表 (2026-08-06, 用户观察到瓶被歪着抓): 接触后物体长轴相对
        # 初始姿态的倾角, 回合内取 max. 判据/奖励是否吃它由 interaction_role 定.
        self.tilt_max_deg = torch.zeros(N, device=dev)
        self.yaw_max_deg = torch.zeros(N, device=dev)
        # 终值快照: reset 清零前拷贝, 评测在 done 之后读的是这两个
        # (🔴 2026-08-06 踩坑: 评测直接读 tilt_max_deg 拿到的是复位后的 0,
        #  据此得出"倾角<0.05°"的错误结论, 实际 17~19° —— 仪表本身也要被证伪)
        self.tilt_final_deg = torch.zeros(N, device=dev)
        self.yaw_final_deg = torch.zeros(N, device=dev)
        self.obj_start_pos = torch.zeros(N, 3, device=dev)
        self.obj_start_quat = self.obj_init_quat.expand(N, 4).clone()
        self.prev_pad_d = torch.zeros(N, 5, device=dev)                 # 势差分状态
        self.prev_quality = torch.zeros(N, device=dev)
        self.phase_timeout_t = torch.tensor(cfg.phase_timeout, dtype=torch.long,
                                            device=dev)
        self._sig: dict[str, torch.Tensor] = {}

        # ---- 接近段 (方案 C) ----------------------------------------------
        self.gs = int(self.grasp_start)              # 人手参考轨迹的 PreGrasp 帧号
        self.ref_t = torch.zeros(N, dtype=torch.long, device=dev)   # 参考时钟 (外生)
        self.switch_run = torch.zeros(N, dtype=torch.long, device=dev)
        self.prev_wrist_pos = torch.zeros(N, 3, device=dev)
        self.prev_phi = torch.zeros(N, device=dev)                  # 对齐势 Φ 的上一步值
        self.started_grasp = torch.ones(N, dtype=torch.bool, device=dev)  # 起步分支(分桶用)
        self.prev_valid = torch.zeros(N, dtype=torch.bool, device=dev)    # prev_wrist_pos 是否有效
        self.arrived = torch.zeros(N, dtype=torch.bool, device=dev)       # 本回合是否切进抓取相位
        # Phase2-RL 第二段判据/里程碑状态 (无条件预创建, SideState 才能快照到)
        self._g2_run = torch.zeros(N, dtype=torch.long, device=dev)
        self._g2_done = torch.zeros(N, dtype=torch.bool, device=dev)
        self._m1_done = torch.zeros(N, dtype=torch.bool, device=dev)
        self._m2_done = torch.zeros(N, dtype=torch.bool, device=dev)
        self.arrive_step = torch.zeros(N, dtype=torch.long, device=dev)   # 用了多少步到位
        # 到位那一刻的**绝对回合步**(arrive_step 存的是 ref_t, 不是步数) ——
        # 相位奖励日程表的渐入窗口要用它算"进门后过了几步"
        self.arrive_step_abs = torch.zeros(N, dtype=torch.long, device=dev)
        # ---- 验证段诊断 (2026-08-02): 判据是"物体升 ≥5mm"(绝对), 而它的严格程度
        # 其实取决于**腕实际抬了多少** —— 腕抬 6.35mm 时它等于要求 79% 跟随率,
        # 腕抬 10mm 时只要求 50%. 而 q_lift 只是**指令**, 抓着物体带接触力时臂实际
        # 能抬多少从没测过. 先把三个量记下来, 再决定要不要把判据改成相对口径.
        self.verify_wz0 = torch.zeros(N, device=dev)      # 进验证段那一刻的腕高
        self.verify_oz0 = torch.zeros(N, device=dev)      # 进验证段那一刻的物体高
        self.vf_wrist_mm = torch.zeros(N, device=dev)     # 斜坡到顶时腕实际抬了多少
        self.vf_obj_mm = torch.zeros(N, device=dev)       # 同一刻物体实际升了多少
        self.vf_has = torch.zeros(N, dtype=torch.bool, device=dev)
        # twist 验证基线 —— 必须在这里预创建 (原来是 _get_dones 里懒创建):
        # 双臂 SideState 在 __init__ 后立即快照, 懒创建的字段抓不到 = 静默共用
        self.verify_ang0 = torch.zeros(N, device=dev)
        self._approach_hit = torch.zeros(N, dtype=torch.bool, device=dev)
        # L5 c(d) 耦合的逐步缓存 (pre_physics 写 / reward 读, 双臂必须分侧+预创建)
        self._l5_dp = torch.zeros(N, device=dev)
        self._l5_cref = torch.zeros(N, device=dev)
        # 臂偏差带的中心: 抓取相位钳在它 ±arm_dev_max. 直接起步 = q_pregrasp;
        # 接近切过来 = **切换那一刻的 q_cmd** (不这么做, 切换会把目标一把拽回 prior 位形,
        # 位控臂跟不上 -> term/stuck; 语义上"微调"本来就该围绕到达位形).
        self.arm_center = self.q_pregrasp.expand(N, 7).clone()
        self.ref_q_prev = self.q_ref[0].expand(N, 7).clone()        # 前馈增量的上一步参考
        self.res_step_cm = torch.zeros(N, device=dev)               # 诊断: 本步残差用量
        if cfg.approach:
            assert cfg.grasp_prior_npz, "接近段必须配 GraspPose prior (对齐势的终点来自它)"
            # 预算/阈值都从**课程起点**开始 (由训练入口按 arrive_rate 逐步收紧).
            # 终点值先存下来 —— 下一行就把 cfg.eps_* 覆盖成起点了.
            self._eps_final = (float(cfg.eps_pos), float(cfg.eps_rot))
            self.phase_timeout_t[Phase.PREGRASP] = self.gs + cfg.approach_extra0
            self.cfg.eps_pos = cfg.eps_pos0
            self.cfg.eps_rot = cfg.eps_rot0
            # ep_total 按**最宽预算**算死, 中途收紧课程不会让回合被提前截断
            self.ep_total = (cfg.settle_steps + self.gs + cfg.approach_extra0
                             + int(cfg.phase_timeout[Phase.GRASP])
                             + int(cfg.phase_timeout[Phase.LIFT]) + 8)
            print(f"[approach] 开: 参考 gs={self.gs} ({self.gs/20:.1f}s) | 预算 "
                  f"{int(self.phase_timeout_t[Phase.PREGRASP])} 步 | 切换 "
                  f"<{cfg.eps_pos*100:.2f}cm & <{np.degrees(cfg.eps_rot):.1f}° 保持"
                  f"{cfg.switch_hold}步 | 直接抓取起步比例 {cfg.direct_grasp_prob:.2f}")
            # 前馈播放到哪一帧为止: pick_lift 到 gs; place 任务播完搬运段 (下面覆盖)
            self._ff_end = self.gs

        # ---- PickAndPlace: 搬运+放置段 (§2.19) ----
        if cfg.place_task:
            assert cfg.approach, "--place 基于端到端任务 (需 --approach)"
            _re = int(min(self.grasp_end, self.q_ref.shape[0] - 1))
            assert _re > self.gs + 3, f"搬运参考太短: gs={self.gs} re={_re}"
            self.re = _re
            self._ff_end = _re
            # 搬运位移参考 = 人手腕相对抓取帧的位移 (相对量, 对重建绝对偏移免疫);
            # 物体目标(t) = 进搬运时的实测物体位 + carry_delta[t]. 与 ref builder 的
            # track_object 同构 (replay_grasp.py "物体参考=抓取帧物体位+腕相对位移").
            self.carry_delta = (self.ref_wrist_pos[:_re + 1]
                                - self.ref_wrist_pos[self.gs]).clone()
            self.phase_timeout_t[Phase.TRANSPORT] = (_re - self.gs) + cfg.carry_extra_steps
            self.phase_timeout_t[Phase.PLACE] = cfg.place_budget_steps
            self.ep_total += int(self.phase_timeout_t[Phase.TRANSPORT]) \
                + int(cfg.place_budget_steps)
            self.carry_anchor = torch.zeros(N, 3, device=dev)
            self.carry_prev_d = torch.zeros(N, device=dev)
            self.place_target = torch.zeros(N, 3, device=dev)
            self.settle_ctr = torch.zeros(N, dtype=torch.long, device=dev)
            print(f"[place] PickAndPlace 开: 搬运参考 {_re - self.gs} 帧, 末端位移 "
                  f"{float(self.carry_delta[_re].norm()) * 100:.1f}cm | 预算 搬运 "
                  f"{int(self.phase_timeout_t[Phase.TRANSPORT])} + 放置 "
                  f"{cfg.place_budget_steps} 步 | 容差 {cfg.place_tol * 100:.0f}cm | "
                  f"松手斜坡 {cfg.place_release_rate}/步")

        # ---- 参考接触指集 (静态): prior 的 c=1 手型下, 哪几个垫离表面 <8mm ----
        # 单 prior 的 run 里是常数 (零信息量), 留给 P5 的 prior-conditioned 蒸馏.
        self.ref_contact = torch.zeros(N, 5, device=dev)
        if cfg.ref_contact_obs and getattr(self, "_prior_contact_pads", None) is not None:
            self.ref_contact = self._prior_contact_pads.expand(N, 5).clone()

        # ---- 接触分数图 (P1a) ----------------------------------------------
        P = 0 if self.obj_points is None else self.obj_points.shape[0]
        self.score_on = bool(cfg.score_map and P > 0)
        if self.score_on:
            self.score_s = torch.zeros(P, device=dev)      # 成功计数 (力加权)
            self.score_n = torch.zeros(P, device=dev)      # 访问计数 (力加权, 带折扣)
            self.pend_idx = torch.zeros(N, 5, dtype=torch.long, device=dev)
            self.pend_w = torch.zeros(N, 5, device=dev)    # 0 = 该垫没进快照
            print(f"[score] 接触分数图已开: {P} 个表面点; 只在 verify 斜坡到顶取快照, "
                  f"按力占比加权")

        # 复位时臂的起始关节位形池. None = 一律 q_pregrasp (默认行为).
        # 给一个 (K,7) 张量 => 每个 env 复位时从中**随机抽一行**. 两个用途共用这一个钩子:
        #   ① 到位误差容差实验 (tol_curve.py): 池 = q_pregrasp 附近抖动的 IK 解
        #   ② 接近段的 RSI (P3): 池 = 参考轨迹 q_ref[t0] 各帧
        # ⚠ 臂偏差带 (arm_dev_lo/hi) 仍以 q_pregrasp 为中心 —— 走廊由 prior 定义,
        #   不由"这一回合恰好从哪起步"定义.
        self.arm_start_pool: torch.Tensor | None = None
        # 自生成起点池 (见 _sp_init). 放这儿是因为它要 self.hand/arm_jids, 都已就绪。
        self._sp_q = None
        self._sp_best_d = None
        if getattr(cfg, "start_pool", ""):
            self._sp_init()

        # 笨拙课程标量 (训练入口每迭代按 sr_ema 更新; 评测/冒烟脚本应显式置 1.0)
        self.gentle = float(cfg.gentle_init)
        self.ep_total = (cfg.settle_steps + int(sum(cfg.phase_timeout)) + 8)
        if getattr(cfg, "retract_start", False):
            # ⚠ ep_total 不来自 episode_length_s, 来自相位超时之和 —— 只改
            #   episode_length_s 是**无效的**(冒烟活性检查抓到: 预算 250 但 ep_total=153)。
            #   退避起点族的接近距离由 D=1.5×d_g 定, 与人手轨迹的 gs 无关 ⟹ 接近段预算
            #   统一按 approach_only_steps(250) 给, 免得沿用 gs 时长度对不上。
            _ap = int(cfg.approach_only_steps)
            self.phase_timeout_t[Phase.PREGRASP] = _ap
            if getattr(cfg, "approach_only", False):
                self.ep_total = _ap            # 只学接近: 到位即终止, 没有后续相位
            else:
                # 完整任务: 接近 + 抓握 + 微抬升验证
                self.ep_total = (cfg.settle_steps + _ap
                                 + int(cfg.phase_timeout[Phase.GRASP])
                                 + int(cfg.phase_timeout[Phase.LIFT]) + 8)
        print(f"[pregrasp] v2 稳定抓握+微抬升验证: 起点=PreGrasp(帧{self.grasp_start}) | "
              f"合拢参考 {cfg.closure_ref_rate}/步 | 候选 ≥{cfg.success_min_pads}垫 "
              f"向心≥{cfg.grasp_centrip_thresh} 保持{cfg.candidate_hold_steps}步 | "
              f"验证升 {cfg.verify_lift_m*1000:.0f}mm ({self.verify_lvl:.0f}级) | "
              f"episode ≤ {self.ep_total} 步")

    def _reinstall_phase_table(self):
        """按当前 cfg **重新**安装 PREGRASP 相位预算与 ep_total(幂等)。

        ★ 2026-08-25 判死: 构造期这两个量被写**两次** —— `if cfg.approach:`(402 行)
        先写 `gs+approach_extra0`, `if cfg.retract_start:`(481 行)再覆盖成
        `approach_only_steps`。而 `phase_timeout_t` 在 SIDE_ATTRS 里(**分侧**),
        若两次写之间发生过换侧, 就会一侧拿到"先写值"、另一侧拿到"后写值"
        —— e2e 实测 R=145 / L=1050, 短的那侧钟先响, 纯前馈走到半路被重置,
        双侧 arrive 门永不齐 ⟹ 整条链不启动。

        修法不追"哪一跳换了侧"(那是构造顺序的问题, 治标), 而是**两侧构造全部
        完成后各调一次本方法**: 最终值只由 cfg 决定, 与时序无关。这一族病
        (stance_prob 被 approach 块覆盖 / retract_start 落点晚于消费点 / 本条)
        共性都是"构造期分侧写入的最终归属不确定", 后置重装一次治一族。

        调用点: BimanualNativeEnv.__init__ 末尾, 对 _A/_B 各一次。
        单臂环境不需要(只有一侧, 不存在归属问题), 但调了也无害(幂等)。
        """
        cfg = self.cfg
        if getattr(cfg, "approach", False):
            self.phase_timeout_t[Phase.PREGRASP] = self.gs + cfg.approach_extra0
        # 基线 ep_total(与 474 行同式), 下面按 retract_start 可能再覆盖
        self.ep_total = (cfg.settle_steps + int(sum(cfg.phase_timeout)) + 8)
        if getattr(cfg, "retract_start", False):
            _ap = int(cfg.approach_only_steps)
            self.phase_timeout_t[Phase.PREGRASP] = _ap
            self.ep_total = (_ap if getattr(cfg, "approach_only", False) else
                             (cfg.settle_steps + _ap
                              + int(cfg.phase_timeout[Phase.GRASP])
                              + int(cfg.phase_timeout[Phase.LIFT]) + 8))

    # ---- 双物体螺旋装配钩子 (clip 无 secondary 时全部按 screw_spec=None 短路) ----
    def _setup_scene(self):
        super()._setup_scene()
        self._SA.setup_scene(self)

    def _apply_action(self):
        super()._apply_action()
        self._SA.apply_screw(self)

    def _build_collide_ids(self, bn, P, O, to):
        """按侧建碰撞/外壳的 body 索引。

        ⚠ 全是**逐侧**的: `arm_bids` 用本侧前缀 P, `self_bids` 用**对侧** O
        (要避让的正是另一条臂)。双臂任务必须按侧各建一份 —— 否则两只手用
        **同一条臂**的外壳算离桌间隙 (2026-08-17 冒烟实测: R/L 的
        arm_table_gap 逐位相同 7.027, 而它本该差很多)。
        """
        cfg = self.cfg
        self.arm_bids = [i for i, n in enumerate(bn)
                         if f"{P}_arm_l" in n or n == f"{P}_ee"]
        self.self_bids = [i for i, n in enumerate(bn)
                          if "torso_l" in n or "head_l" in n or n == "vega_1p_base"
                          or f"{O}_arm_l" in n or n == f"{O}_ee"]
        print(f"[collide] 臂连杆 {len(self.arm_bids)} 个 "
              f"({[bn[i] for i in self.arm_bids]}) | 需避让 {len(self.self_bids)} 个 "
              f"(躯干/头/另一臂)")
        # ---- 外壳口径臂罚 (cfg.arm_table_shell): 连杆外壳表面采样点 ----
        # 原点+3cm 罩不住真机外壳 (l5/l6 截面半径 4~6.7cm), 用预采样表面点的
        # 最低 z 做罚/诊断, 余量 1cm 即真实余量. 点集: calib 阶段从 URDF 碰撞
        # 网格采样 (48~96 点/节), 见 tasks/pregrasp/arm_shell_points.npz.
        self.shell_bids, self.shell_pts = None, None
        if getattr(cfg, "arm_table_shell", False):
            import os as _os
            _sp = np.load(_os.path.join(_os.path.dirname(__file__),
                                        "arm_shell_points.npz"))
            bids, plist = [], []
            for i in self.arm_bids:
                if bn[i] in _sp.files:
                    bids.append(i)
                    plist.append(np.asarray(_sp[bn[i]], np.float32))
            assert bids, "arm_table_shell=True 但外壳点集一个都没匹配上"
            P = max(len(p) for p in plist)
            padded = np.stack([np.concatenate([p, np.repeat(p[:1], P - len(p), 0)])
                               for p in plist])                     # (L,P,3)
            self.shell_bids = bids
            self.shell_pts = to(padded)
            print(f"[collide] 外壳口径臂罚开启: {len(bids)} 节连杆 × ≤{P} 表面点 "
                  f"(余量 {cfg.arm_shell_margin*100:.0f}cm)")

    def _load_grasp_prior(self, npz_path, to):
        """Dexonomy GraspPose -> ① q_pregrasp ② q_close ③ aff_local (见 cfg 注释).

        抓姿在**物体输入系** (make_prior.py 已从规范系转回, wxyz 约定已自检),
        这里按 env 的物体初始位姿变换到 env 系, 再用基类同款 ArmIK (实测锚) 解臂关节.
        """
        from isaaclab.utils.math import quat_apply as _qa, quat_mul as _qm
        from rl_rebuild.correction.kinematics import ArmIK, quat_to_R

        cfg, dev = self.cfg, self.device
        z = np.load(npz_path)
        # ---- 模板参与指分类 + 垫↔接触零位校准 (物体系, 与场景无关, IK 之前做) ----
        # ① 参与指: 抓握位姿下 FK 五个胶垫中心, 逐指到最近接触点距离 <3.5cm 判参与
        #    (2 指 Tip_Pinch → {thumb,index}; 5 指 fingertip_middle → 全参与).
        #    用户裁定: min_pads / cent 归一 / 非参与指钉死 全部由它派生, 不逐任务手改.
        # ② 零位校准: Dexonomy 的接触标注与我们手模的胶垫面有**系统偏差** (右手
        #    fingertip_mid 模板是穿透方向, 台账 2026-07-31 已记录; 左手 Tip_Pinch
        #    实测是"够不着"方向: 三个盖候选名义位姿两垫一致差 1.7~2.1cm, 模板加深
        #    到 1.6 也合不拢 —— 结构性够不着, RL 学不出来). 把参与指垫质心到对应
        #    最近接触点质心的偏差向量补进腕位 (>8mm 才补): 接触位置仍是 Dexonomy
        #    定的, 只校掉手模 FK 偏差, 不换用户选的候选.
        from rl_rebuild.correction.ref_builders.replay_grasp import _urdf
        zg = np.asarray(z["grasp"], np.float64).copy()      # (29,) 可能被校准平移
        zpre = np.asarray(z["pregrasp"], np.float64).copy()  # (6,29)
        _u = _urdf()
        _side = cfg.hand_side
        _qd = {n.replace("right_", f"{_side}_"): float(v)
               for n, v in zip(GENERIC_JOINT_ORDER, zg[7:29])}
        _Th = np.eye(4)
        _Th[:3, :3] = quat_to_R(zg[3:7])
        _Th[:3, 3] = zg[:3]
        _pads = np.stack([_u.link_pose(f"{_side}_{f}_elastomer", _qd, _Th,
                                       f"{_side}_hand_C_MC")[:3, 3]
                          for f in FINGERS])                    # (5,3) 物体系
        _cts = np.asarray(z["contact_pos"], np.float64)
        _d2 = np.linalg.norm(_pads[:, None, :] - _cts[None], axis=-1)   # (5,C)
        _cd = _d2.min(axis=1)
        if "active_fingers" in z.files:
            # 显式声明优先 (合成模板用)
            _act = np.asarray(z["active_fingers"]).astype(bool)
        else:
            # 归属制分类 (2026-08-06 修): 每个接触点归属**最近的那根手指**,
            # 拥有 ≥1 个接触点的手指 = 参与指. 旧的距离阈值制会把收拢在接触区
            # 附近的非参与指也算进来 (3_1 实测: 2 指抓被判成 4 指, min_pads
            # 被抬到 4, 候选永远差两垫).
            _owner = _d2.argmin(axis=0)               # (C,) 每个接触点最近的手指
            _act = np.zeros(5, dtype=bool)
            for _fi in range(5):
                if (( _owner == _fi) & (_d2[_fi, :] < 0.05)).any():
                    _act[_fi] = True
        assert _act.sum() >= 2, \
            f"prior 接触点只归属到 {int(_act.sum())} 根手指 (<2), 分类阈值或数据有问题"
        self.finger_active = to(_act.astype(np.float32))
        self.n_active = int(_act.sum())
        cfg.success_min_pads = min(cfg.success_min_pads, self.n_active)
        cfg.verify_min_pads = min(cfg.verify_min_pads, self.n_active)
        _near = _cts[_d2.argmin(axis=1)]                        # (5,3) 逐指最近接触点
        _shift = (_near[_act] - _pads[_act]).mean(axis=0)
        # ⚠ 平移只对**同向系统偏差**成立(两指捏取: 两个垫的 FK 偏差同向, 均值就是它)。
        #   五指包络时五个垫从各方向朝内指, 均值没有物理意义 —— 整体平移会把一侧压进
        #   物体、另一侧推得更远。而且这个距离是在**手指尚未合拢的模板位姿**上量的,
        #   本就该由合拢斜坡消掉, 不该用腕位去补。
        #   2026-08-16 实测(Grasp3 五指, 零动作合拢到底): 校准开 → 指垫接触 3.00,
        #   校准关 → 4.75; 候选判据要 ≥4 ⟹ **开着校准这条 clip 永远出不了候选**
        #   (CTRL2 跑 16.4M 步 0% 就是这个)。
        _dist0 = np.linalg.norm(_near[_act] - _pads[_act], axis=1)
        _dist1 = np.linalg.norm(_near[_act] - (_pads[_act] + _shift), axis=1)
        _objp = (self.obj_points.cpu().numpy().astype(np.float64)
                 if getattr(self, "obj_points", None) is not None else None)
        if _objp is not None:
            _s0 = np.linalg.norm(_pads[_act][:, None, :] - _objp[None], axis=-1).min(axis=1)
            _s1 = np.linalg.norm((_pads[_act] + _shift)[:, None, :] - _objp[None],
                                 axis=-1).min(axis=1)
            _TH = float(getattr(cfg, "pad_calib_reach_cm", 1.5)) / 100.0
            print(f"[prior] 零位校准参考数 (静态 FK: 垫->物体表面, 阈值 {_TH*100:.1f}cm): "
                  f"不校准 {[f'{v*100:.1f}' for v in _s0]}cm ({int((_s0 < _TH).sum())} 垫); "
                  f"校准后 {[f'{v*100:.1f}' for v in _s1]}cm ({int((_s1 < _TH).sum())} 垫)"
                  f"  ⚠仅供参考, 不作判据 (见下)")
        # ★ 开不开校准 = **按 clip 实测记死**(clips 注册表的 `pad_contact_calib`),
        #   代码不猜。我连续三次用错代理量, 每次都自洽、每次都和物理相反:
        #     ① "平移后垫↔接触残差变小" —— 张开位姿上的账, 与合拢后碰几个垫无关
        #     ② "参与指 ≤2 才应用"      —— Grasp3 对(关=4.75垫/开=3.00), 瓶子错(关=0.00/开=3.38)
        #     ③ "静态 FK 算合拢后够不够得到" —— 方向仍相反: 它说 Grasp3 该开校准,
        #        实测该关。因为模板 c=1 的关节位形**不是**真正的合拢终点 ——
        #        `close_anchor` 让 PD 顶到"壁内"目标, 比 Dexonomy 抓姿更深。
        #   定法: 用 `tasks/pregrasp/diag_fgate.py` 跑 A/B(零动作合拢, 看 pads_now),
        #   把胜出的那个写进 clips 注册表。新 clip 上线前必须做这一步。
        _helps = True
        if getattr(cfg, "pad_contact_calib", True) and np.linalg.norm(_shift) > 0.008:
            if _helps:
                zg[:3] += _shift
                zpre[:, :3] += _shift
                print(f"[prior] ⚠ 垫↔接触零位校准: 腕位平移 "
                      f"{np.round(_shift * 100, 2).tolist()}cm "
                      f"(|Δ|={np.linalg.norm(_shift)*100:.2f}cm) —— 手模胶垫面与 "
                      f"Dexonomy 接触标注的系统差, 校准后最差指残差 "
                      f"{_dist0.max()*100:.2f} -> {_dist1.max()*100:.2f}cm")
            else:
                pass
        print(f"[prior] 模板参与指 {self.n_active}/5: "
              f"{[f for f, a in zip(FINGERS, _act) if a]} "
              f"(逐指最近接触点 {[f'{d*100:.1f}' for d in _cd]}cm) | "
              f"判据随之: min_pads={cfg.success_min_pads} "
              f"verify_min_pads={cfg.verify_min_pads} cent 归一 /{self.n_active}")
        if getattr(cfg, "affordance_npz", ""):
            # 视频接触带 (2026-08-06, 设定B affordance 经验回归): 模板回放因
            # Dexonomy 极点↔力垫 ~2.5cm 系统差从盖旁掠过, pad_approach 的
            # "到表面最近点"梯度指不对地方; 换成推参与垫去视频实证的接触环带.
            # 裁判几何 (_pad_dist_normal / cent / 候选门) 不碰.
            _az = np.load(cfg.affordance_npz)
            _apts = np.asarray(_az["pts"], np.float64)
            self._aff_pts_local = to(_apts.astype(np.float32))
            _ad = np.linalg.norm(_pads[:, None, :] - _apts[None], axis=-1).min(axis=1)
            print(f"[affordance] 视频接触带 {len(_apts)} 点 (物体局部系, "
                  f"z {_apts[:, 2].min()*1000:.0f}~{_apts[:, 2].max()*1000:.0f}mm, "
                  f"r {np.linalg.norm(_apts[:, :2], axis=1).mean()*1000:.1f}mm) | "
                  f"grasp FK 逐垫距带 {[f'{d*100:.1f}' for d in _ad]}cm | "
                  f"参与指距带 {[f'{d*100:.1f}' for d in _ad[_act]]}cm")
        oq = self.obj_init_quat.cpu().numpy().astype(np.float64)
        op = self.obj_init_pos.cpu().numpy().astype(np.float64)
        _entry = clips.clip_entry(cfg.clip_name)
        # ★ 摆放三步的**顺序**(2026-08-16 用户裁定): ①定姿态(canon_rot ∘ 钉死 yaw)
        #   -> ②用**①旋转之后**的 affordance 偏移做物体听手定 XY -> ③该姿态下最低点贴桌。
        #   这里先把 ref builder 用过的姿态存下来, 等最终姿态定了(yaw 也转完)再回来
        #   按 ② 重解 XY。为什么必须重解:
        #     物体听手的不变量是 `物体原点 + a_off = 交互帧手锚点`, 而 a_off 依赖姿态。
        #     builder 用 rest_q 算 a_off 定了 XY, 随后这里把姿态换成 canon_rot∘yaw
        #     (瓶实测差 113.6°, 杯 149.7°) —— **原点没动, 目标区却转跑了**:
        #     瓶错位 3.83cm / 杯 1.46cm。这正好把"物体听手"要消除的手↔物系统差
        #     又装回去一部分(该机制原本是为消除 10cm+ 的系统差引入的)。
        from rl_rebuild.correction import frames as _F0
        _rest_q0 = oq.copy()                       # builder 摆放时用的姿态
        try:
            from rl_rebuild.correction.ref_builders.replay_grasp import (
                _affordance_target as _aff_t)
            _a_obj = (_aff_t(_entry["affordance"]) if _entry.get("affordance")
                      else _F0.load_obj_verts(self.du.mesh_path).mean(0))
        except Exception as _e:                    # 缺 affordance 就退化到质心 (与 builder 同)
            print(f"[prior] ⚠ 取 affordance 目标失败({_e}), 物体听手改用质心")
            _a_obj = _F0.load_obj_verts(self.du.mesh_path).mean(0)
        # ---- 物体姿态对齐 Dexonomy 规范姿态 ----
        # 抓姿是在"物体按规范姿态平放桌面"时生成的; 而 ref builder 用"最大支撑面朝下"
        # 独立决定姿态, 对非对称物体可能选到**另一个面** (Grasp3 实测两者差 179.9°,
        # 物体上下颠倒 -> 抓姿腕落到桌面下 12cm, 所有 yaw 都 IK 失败).
        # 有 prior 时以 Dexonomy 姿态为准, xy 保留 ref builder 的相机锚定结果.
        if _entry.get("resting_pose_json"):
            # 静置位姿权威已下沉到 ref builder (rest_override, 2026-08-07):
            # XY=交互帧手锚点 / 姿态+Z=resting_pose. 这里不再覆盖, 只留断言.
            _canon = np.asarray(z["canon_rot"], np.float64) if "canon_rot" in z.files \
                else np.array([1.0, 0, 0, 0])
            assert abs(_canon[0]) > 0.999, \
                f"resting_pose 权威假设 canon_rot≈单位阵, 实际 {_canon}"
        elif "canon_rot" in z.files:
            from rl_rebuild.correction import frames as _F
            oq = np.asarray(z["canon_rot"], np.float64)
            _v = _F.load_obj_verts(self.du.mesh_path) @ quat_to_R(oq).T
            op = op.copy()
            op[2] = cfg.table_top_z + 0.002 - float(_v[:, 2].min())   # 最低点贴桌 +2mm
            self.obj_init_quat, self.obj_init_pos = to(oq), to(op)
            print(f"[prior] 物体按 Dexonomy 规范姿态摆放 (原点 z={op[2]:.4f}, "
                  f"高度 {(_v[:,2].max()-_v[:,2].min())*100:.2f}cm)")
            # ★ 与 scene_layout 的独立裁定**交叉核验**(2026-08-15 加)。
            #   Grasp3 的事故正是"两者差 179.9° -> 抓姿腕落到桌面下 12cm, 所有 yaw
            #   IK 全失败"; 那次是靠 IK 全崩才发现。这里在开训前就报出来:
            #   两条来源(Dexonomy 规范姿态 / 物理稳定候选+重建主轴)本应一致, 不一致
            #   说明其中一条错了, 必须人裁, 不许静默。
            _lay = _entry.get("scene_layout_json")
            if _lay and os.path.isfile(_lay):
                import json as _json
                _lo = _json.load(open(_lay))["objects"].get(_entry.get("primary_oid"))
                if _lo:
                    # 2026-08-20 修: 比较轴从"网格最长轴"换成**上轴**。旧代理对立物
                    # 成立(最长轴≈上轴, yaw 不变); 对**躺姿**物体最长轴是水平轴, 被
                    # yaw 支配 ⟹ 必然误报约 180°(sweep2 扫帚实测: 钉死 yaw 前后都
                    # 170.5°, 而双端复核上轴一致 0.0°)。上轴 = 重力相关且 yaw 不变,
                    # Grasp3 型倒置(上轴翻转)仍被同一阈值捕获。
                    _up_in = quat_to_R(oq).T @ np.array([0.0, 0.0, 1.0])
                    _a1 = np.array([0.0, 0.0, 1.0])
                    _a2 = quat_to_R(np.asarray(_lo["quat_wxyz"], np.float64)) @ _up_in
                    _ang = float(np.degrees(np.arccos(np.clip(
                        _a1 @ _a2 / (np.linalg.norm(_a1) * np.linalg.norm(_a2)), -1, 1))))
                    _tag = "✅一致" if _ang < 30 else ("⛔**差约180°(Grasp3 型倒置)**"
                                                    if _ang > 150 else "⚠ 不一致")
                    print(f"[prior] 姿态交叉核验: Dexonomy canon_rot vs scene_layout "
                          f"主轴夹角 {_ang:.1f}° {_tag}")
                    assert _ang < 150, (
                        f"canon_rot 与 scene_layout 的主轴差 {_ang:.1f}° —— 物体上下颠倒, "
                        f"抓姿会落到桌面下(Grasp3 事故复现)。先裁定用哪一个, 不许静默开训。")
        Ro = quat_to_R(oq)

        def to_env(row29):
            p, q = row29[:3], row29[3:7]
            return Ro @ p + op, self._qmul_np(oq, q)

        ik = ArmIK(cfg.hand_side, anchor_link="arm_center", anchor_T=self._anchor_T)
        seed = self.q_pregrasp.cpu().numpy().astype(np.float64)
        # ---- 物体 yaw 搜索: 重建摆放的 yaw 是丢弃姿态后的随机残留, 抓姿的接近方向
        # 可能背对机械臂 (Grasp5 实测原 yaw 下 IK 差 17cm, 转 180-240° 后 0.04cm).
        # 转的是**物体** (抓姿保持物体相对, 对任意物体都正确), 5° 步长扫一圈取 IK 最优.
        # 非交互手位置 (间隙项用): 抓姿 yaw 只按 IK 误差选会贴到左手上
        bn2 = list(self.hand.body_names)
        other = "left" if cfg.hand_side == "right" else "right"
        oh_name = f"{other}_hand_C_MC"
        oh_pos = (self.hand.data.body_pos_w[0, bn2.index(oh_name)]
                  - self.scene.env_origins[0]).cpu().numpy().astype(np.float64)             if oh_name in bn2 else None
        best_yaw, best_err, qwarm = 0.0, np.inf, seed
        if cfg.prior_yaw_deg >= 0.0:
            # D1: yaw 钉死在筛选给出的角 (= 可达带里离视频 yaw 最近的那个). 不再自搜.
            best_yaw = float(np.radians(cfg.prior_yaw_deg))
            gp_t = quat_to_R(self._qmul_np(
                np.array([np.cos(best_yaw / 2), 0.0, 0.0, np.sin(best_yaw / 2)]), oq)
            ) @ zg[:3] + op
            gq_t = self._qmul_np(self._qmul_np(
                np.array([np.cos(best_yaw / 2), 0.0, 0.0, np.sin(best_yaw / 2)]), oq),
                zg[3:7])
            r = ik.solve(gp_t, quat_to_R(gq_t), q0=seed, iters=200)
            qwarm = r["q"] if r["ok"] else seed
            print(f"[prior] 物体 yaw **钉死** {cfg.prior_yaw_deg:.0f}° (来自 Gate 1 的 "
                  f"best_yaw, 与视频一致) | 该角 IK err {r['pos_err']*100:.2f}cm")
            assert r["pos_err"] < 0.02, (
                f"钉死的 yaw {cfg.prior_yaw_deg:.0f}° 上抓姿 IK 误差 {r['pos_err']*100:.1f}cm "
                f">2cm —— 这个候选在视频 yaw 下够不着, 换候选 (别改 yaw)")
        else:
            print("[prior] ⚠ prior_yaw_deg 未设 —— 退回按 IK 可达性**自搜** yaw. "
                  "这会把物体转到与视频不一致的姿态, 接近段任务不可用 (见 cfg 注释 D1)")
            _cands = []
            for deg in range(0, 360, 5):
                a = np.radians(deg)
                yq = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
                oq2 = self._qmul_np(yq, oq)
                gp_t = quat_to_R(oq2) @ zg[:3] + op
                gq_t = self._qmul_np(oq2, zg[3:7])
                r = ik.solve(gp_t, quat_to_R(gq_t), q0=qwarm, iters=80)
                err = r["pos_err"] + 0.3 * r["rot_err"]
                if oh_pos is not None:      # 离非交互手 <25cm 开始加罚 (软约束)
                    gap = float(np.linalg.norm(gp_t - oh_pos))
                    err += 2.0 * max(0.0, 0.25 - gap)
                if r["ok"] and self.shell_pts is not None:
                    # 外壳口径 (2026-08-05): 旋转对称物体的接近方位角是自由维度,
                    # 选一个前臂外壳离桌高的方向 —— 真机带壳不碰桌 (用户要求)
                    _clr = self._shell_clearance_np(ik, r["q"])
                    err += 4.0 * max(0.0, 0.02 - _clr)
                if r["ok"]:
                    qwarm = r["q"]
                _cands.append((err, a))
                if err < best_err:
                    best_yaw, best_err = a, err
            # ---- yaw 定案前预演 (2026-08-05): 评分只看瞬时 IK, 看不见"抬升斜坡
            # 病态"的 yaw (40° 曾 IK 可达但紧公差斜坡只抬得动 6.3mm, 被 G3_8_5
            # 防线拦下). 对前 10 名逐个用**真实紧公差**预演 0→verify_lift 四级
            # 斜坡, 抬得动的第一名才定案; 全军覆没时保底用 argmin (断言仍会拦).
            if cfg.verify_mode != "twist":
                _lvl = max(int(round(cfg.verify_lift_m
                                     / (cfg.lift_height / cfg.lift_steps))), 1)
                _tried = 0
                for _e, a in sorted(_cands):
                    _tried += 1
                    yq = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
                    oq2 = self._qmul_np(yq, oq)
                    gp_t = quat_to_R(oq2) @ zg[:3] + op
                    gq_t = self._qmul_np(oq2, zg[3:7])
                    r0 = ik.solve(gp_t, quat_to_R(gq_t), q0=qwarm, iters=200)
                    if not r0["ok"]:
                        continue
                    qw2 = r0["q"].copy()
                    _qs = [qw2.copy()]
                    for i in range(1, _lvl + 1):
                        tgt = gp_t + np.array(
                            [0.0, 0.0, cfg.lift_height / cfg.lift_steps * i])
                        rr = ik.solve(tgt, quat_to_R(gq_t), q0=qw2,
                                      iters=300, pos_tol=2e-4)
                        if rr["ok"]:
                            qw2 = rr["q"]
                        _qs.append(qw2.copy())
                    _gain = float(ik.fk(qw2)[0][2] - ik.fk(_qs[0])[0][2])
                    if _gain >= 0.9 * cfg.verify_lift_m:
                        if a != best_yaw:
                            print(f"[prior] yaw 预演: 评分第 1 的 "
                                  f"{np.degrees(best_yaw):.0f}° 抬升病态, 第 {_tried} 名 "
                                  f"{np.degrees(a):.0f}° 通过 (预演实抬 {_gain*1000:.1f}mm)")
                        best_yaw = a
                        # 预演通过的逐级解交给正式斜坡复用 —— 预演/重解结果
                        # 可能因求解路径不同而不一致, 直接复用已验证的解
                        self._ramp_rehearsal = (float(a), _qs)
                        break
        if best_yaw != 0.0:
            yq = np.array([np.cos(best_yaw / 2), 0.0, 0.0, np.sin(best_yaw / 2)])
            oq = self._qmul_np(yq, oq)
            Ro = quat_to_R(oq)
            self.obj_init_quat = to(oq)          # 物体复位姿态跟着转
            print(f"[prior] 物体 yaw 旋转 {np.degrees(best_yaw):.0f}° 使抓姿朝向机械臂"
                  f" (绕竖轴旋转, 物体仍平放于桌面)")
        # ---- ② 姿态定案后, 按最终姿态重解物体听手的 XY (见上面 ★) ----
        #   不变量: 物体原点 + a_off(姿态) = 交互帧手锚点。手锚点不动, 于是
        #     op_new[:2] = op_old[:2] + a_off(rest_q0)[:2] - a_off(oq_final)[:2]
        #   这样 afford_world = op + a_off 仍落在手锚点上 ⟹ builder 里 pregrasp_align
        #   的悬停点不受影响, 不用回头改参考轨迹。
        _off0 = quat_to_R(_rest_q0) @ _a_obj
        _off1 = quat_to_R(oq) @ _a_obj
        _dxy = _off0[:2] - _off1[:2]
        if float(np.linalg.norm(_dxy)) > 1e-4:
            op = op.copy()
            op[:2] += _dxy
            self.obj_init_pos = to(op)
            print(f"[prior] 物体听手 XY 按最终姿态重解: 平移 "
                  f"{np.round(_dxy * 100, 2).tolist()}cm "
                  f"(|Δ|={np.linalg.norm(_dxy) * 100:.2f}cm) —— 姿态从 builder 的 rest_q "
                  f"换成 canon_rot∘yaw 后, 抓取目标区绕原点转跑了这么多")
        # 抓握位姿 IK (在选定 yaw 下精解)
        gp, gq = to_env(zg)
        # 裁定B3 (2026-08-18, 取代 Dexonomy pregrasp 档): PreGrasp = GraspPose 整手
        # 沿"接触质心→腕"方向(≈掌背法向)平移 3cm —— 锚点是**掌心/指尖间隙**而非腕,
        # 两手视觉间隙对称(腕定义下实测左指尖贴零/右悬空的病根)。朝向=抓姿, 指伸直(B2)。
        # 与 gp 同一条变换链(含垫↔接触零位校准), 保证规划终点与到位闩靶点一致。
        _cc = np.asarray(z["contact_centroid"], np.float64)
        _u = zg[:3] - _cc
        _u = _u / max(np.linalg.norm(_u), 1e-9)
        _pre_row = zg.copy()
        _pre_row[:3] = zg[:3] + (
            float(getattr(self.cfg, "pregrasp_palm_cm", 5.0)) / 100.0) * _u
        self._pregrasp_w = [to_env(_pre_row)]
        if getattr(self.cfg, "pregrasp29", False):
            # PreGrasp29: 到位/对齐/成功的靶点整体换成掌心 PreGrasp ——
            # 接近任务的终点就是 PreGrasp, GraspPose 归下一阶段
            gp, gq = self._pregrasp_w[0]
        if getattr(self.cfg, "pregrasp_phase2", False):
            # Phase2 名义斜坡 (A 侧): 腕 PreGrasp→GraspPose 直线插入 (朝向不变),
            # 逐帧 IK (链式种子); 指型终点 = 抓姿 22 关节 (USD 序)。
            _K2 = int(getattr(self.cfg, "phase2_steps", 50))
            _gp_g, _gq_g = to_env(zg)                      # 真抓姿 (插入终点)
            _pp0 = np.asarray(self._pregrasp_w[0][0], np.float64)
            _qn = np.asarray(_gq_g, np.float64)
            _qn = _qn / max(np.linalg.norm(_qn), 1e-12)
            _Rq = quat_to_R(_qn)
            _seed = self.q_pregrasp.cpu().numpy().astype(np.float64)
            _rows, _errs = [], []
            for _k in range(_K2):
                _al = _k / max(_K2 - 1, 1)
                _pt = (1 - _al) * _pp0 + _al * np.asarray(_gp_g, np.float64)
                _rk = ik.solve(_pt, _Rq, q0=_seed, iters=200)
                _seed = _rk["q"]
                _rows.append(_rk["q"])
                _errs.append(_rk["pos_err"])
            self._p2_arm = to(np.stack(_rows).astype(np.float32))
            _fg = np.clip(np.asarray(zg[7:29], np.float64)[self._generic_perm],
                          self.dof_lower[0].cpu().numpy(),
                          self.dof_upper[0].cpu().numpy())
            self._p2_fin = to(_fg.astype(np.float32))       # (22,) 抓姿指型
            # m2 派生阶梯 (裁定C): 起始偏差 d0 = |张开指型−抓姿指型| 均值;
            # 从 d0−5° 起每 5° 一级直到 20° 大门 —— 第一块糖永远够得着。
            _d0 = float((self.q_open - self._p2_fin).abs().mean())
            _step = np.radians(float(getattr(cfg, "phase2_ladder_step_deg", 5.0)))
            _gate = 0.35                                    # 20° 大门 (m2 终点)
            _lad = []
            _r = _d0 - _step
            while _r > _gate + 1e-6:
                _lad.append(_r)
                _r -= _step
            _lad.append(_gate)
            self._m2_ladder = to(np.asarray(_lad, np.float32))
            self._m2_lvl = torch.zeros(self.num_envs, dtype=torch.long,
                                       device=self.device)
            print(f"[phase2] m2 阶梯(A): 起始偏差 {np.degrees(_d0):.1f}° -> "
                  f"{[f'{np.degrees(x):.1f}°' for x in _lad]}")
            if getattr(cfg, "fin_cart", False):
                # FC: 逐指目标改为**首次 reset 时物理测量** (把手摆到抓姿读指垫,
                # 与运行时同口径)。FK 方案已废: 左手指尖在 URDF 链里读数错 50cm
                # (URDF/USD 连杆命名不一致), 修数学不如直接量。
                self._fc_local = None                            # 待测
                self._fc_done = torch.zeros(self.num_envs, 5, dtype=torch.bool,
                                            device=self.device)
                self._fc_prev = torch.full((self.num_envs, 5), float("nan"),
                                           device=self.device)
            self._p2_t = torch.zeros(self.num_envs, dtype=torch.long,
                                     device=self.device)
            # Phase2-RL: 真抓姿靶点 (g2 判据/里程碑用; 腕口径与 _align_err 一致)
            self._g2_pos_w = to(np.asarray(_gp_g, np.float64))
            self._g2_quat_w = to(_qn)
            print(f"[phase2] A 侧就绪: 真抓姿靶点+抓姿指型 | (斜坡 {_K2} 帧留档, "
                  f"IK 误差 max {max(_errs)*100:.2f}cm)")
        rg = ik.solve(gp, quat_to_R(gq), q0=qwarm)
        # pregrasp: 逐行试, 取 IK 成功且离抓握解最近的一行 (6 个候选)
        best = None
        for row in zpre:
            pp, pq = to_env(row)
            r = ik.solve(pp, quat_to_R(pq), q0=rg["q"])
            if r["ok"]:
                d = float(np.abs(r["q"] - rg["q"]).max())
                if best is None or d < best[0]:
                    best = (d, r)
        assert rg["ok"] and best is not None, \
            f"GraspPose prior IK 失败 (grasp ok={rg['ok']}): 该抓姿在本场景不可达, 换候选"
        rp = best[1]
        # ---- 零空间抬肘 (2026-08-06): 腕位姿不变, 肘沿零空间抬升外壳余量 ----
        if self.shell_pts is not None:
            _c0 = self._shell_clearance_np(ik, rg["q"])
            _ql, _c1 = self._nullspace_shell_lift(ik, rg["q"], gp, gq)
            if _c1 > _c0 + 1e-3:
                print(f"[prior] 零空间抬肘: 臂外壳余量(指令位形) {_c0*100:+.1f}cm "
                      f"-> {_c1*100:+.1f}cm | 腕位姿不变, 末端误差 ≤0.5mm")
                rg = dict(rg)
                rg["q"] = _ql
                best2 = None            # pregrasp 换到同一肘位形分支重解
                for row in zpre:
                    pp2, pq2 = to_env(row)
                    r2 = ik.solve(pp2, quat_to_R(pq2), q0=_ql, iters=200)
                    if r2["ok"]:
                        d2 = float(np.abs(r2["q"] - _ql).max())
                        if best2 is None or d2 < best2[0]:
                            best2 = (d2, r2)
                if best2 is not None:
                    rp = best2[1]
                # 预演解在旧肘分支上, 作废 —— 斜坡从抬肘解重建
                self._ramp_rehearsal = None

        self._prior_q_grasp = to(rg["q"])
        # 抓握位姿的**世界位姿** —— 接近段的对齐势函数要用它当终点 (物体复位在 obj_init_*,
        # 训练期只有 ±3~15mm 抖动, 所以这里当常量存; 抖动由势函数自己吸收).
        self._grasp_pos_w = to(gp)                          # (3,)
        # 门控自标定基准: GraspPose 处的**腕到物体**距离。手指门按它的倍数开合,
        # 于是换物体/换 clip 不用手调厘米数(小物体自动收紧, 大物体自动放宽)。
        import numpy as _np2
        self._fgate_dg = float(_np2.linalg.norm(
            _np2.asarray(gp, float) - _np2.asarray(self.obj_init_pos.cpu().numpy(), float)))
        print(f"[finger_gate] 自标定基准 d_g={self._fgate_dg*100:.1f}cm (GraspPose 处腕-物体距离) "
              f"| 放开 <{self._fgate_dg*self.cfg.finger_gate_near_k*100:.1f}cm "
              f"| 冻结 >{self._fgate_dg*self.cfg.finger_gate_far_k*100:.1f}cm")
        self._grasp_quat_w = to(gq)                         # (4,) wxyz
        self.q_pregrasp = to(rp["q"])                       # ① 臂复位/参考位
        # ⚠ npy 的 22 维是 GENERIC_JOINT_ORDER 序, 必须换到 USD 关节序再进模板
        # (2026-07-30 踩坑: 漏换序 -> 手型全串位, B 组整条 run 作废)
        # Tip_Pinch 校准候选带 close_anchor (内压合拢锚点, calib_tip_pinch.py 产物):
        # c=1 的模板终点用它 (PD 顶着壁内目标 = 持续内压), 抓握位姿本身仍是 grasp
        _fsrc = (np.asarray(z["close_anchor"], np.float64)
                 if "close_anchor" in z.files else zg[7:29])
        fingers = np.clip(_fsrc[self._generic_perm],
                          self.dof_lower[0].cpu().numpy(), self.dof_upper[0].cpu().numpy())
        self.q_close = to(fingers)                          # ② 模板 c=1 锚点
        self.aff_local = to(z["contact_centroid"])          # ③ 对齐目标
        # 微抬升验证的关节斜坡按抓握位姿重解 (基类那份是给旧 PreGrasp 解的)
        # 🔴 2026-08-02 修一个**静默**失效点 (G3_8_5 那条 20M 步 run 的根因):
        #   原来是 `if r["ok"] or i==0: qw = r["q"]` —— IK 某一级解不出就沿用上一级,
        #   斜坡退化成阶梯且**不报错**. 实测 8_5 上第 1/2 级完全不动、第 4 级(验证用的
        #   那级)只抬 4.85mm 而不是 10mm; 而成功判据要求**物体**升 ≥5mm ⟹ 物理上够不到,
        #   成功率必然恒 0. 手抓得再好也没用 (该 run 五垫接触率 0.83~0.85, 成功率 0.00%).
        # 🔴 真根因: `ArmIK.solve` 的默认 **pos_tol=5e-3 (5mm) 比每级增量 2.5mm 还大** ——
        #   求解器一步不动就已"达标", 每级都返回 ok=True (实测 0/20 级失败) 而末端纹丝不动.
        #   所以光看 ok 标志永远发现不了; 必须**核验实际末端位移**.
        # 修法: ① 抬升斜坡单独收紧 pos_tol 到 0.2mm (要分辨 2.5mm 的增量);
        #       ② 迭代加倍; ③ 仍解不出时用雅可比一步补 (小增量下 J⁺Δp 良态);
        #       ④ 最后核验实际抬升量, 不达标直接 assert —— 宁可开不起来也不要静默失败.
        lo_np = self.arm_lower[0].cpu().numpy().astype(np.float64)
        hi_np = self.arm_upper[0].cpu().numpy().astype(np.float64)
        if cfg.verify_mode == "twist":
            # ---- 微拧斜坡 (2026-08-05): 腕绕物体轴 (过物体原点的竖轴) 旋转 ----
            # 盖被螺旋单向投影钉死, 抬不动 ⟹ 验证换成"腕转 → screw_angle 跟进".
            # 正角 = 俯视逆时针 = 右旋螺纹的**拧开**方向 (与 screw_angle 增向一致).
            # 绕盖轴旋转对捏取几何是对称操作: 指尖与盖的相对构型逐级保持不变.
            total = np.radians(cfg.verify_twist_deg)
            lift, qw, n_ik_fail = [], rg["q"].copy(), 0
            for i in range(cfg.lift_steps + 1):
                phi = total * i / max(cfg.lift_steps, 1)
                c_, s_ = np.cos(phi), np.sin(phi)
                Rz = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
                tgt_p = Rz @ (gp - op) + op
                tgt_R = Rz @ quat_to_R(gq)
                r = ik.solve(tgt_p, tgt_R, q0=qw, iters=300, pos_tol=2e-4)
                if r["ok"] or np.isfinite(r["q"]).all():
                    qw = np.clip(r["q"], lo_np, hi_np)
                    n_ik_fail += 0 if r["ok"] else 1
                lift.append(qw.copy())
            self.q_lift = to(np.stack(lift))
            _R0 = ik.fk(lift[0])[1]
            _R1 = ik.fk(lift[-1])[1]
            _Rr = _R1 @ _R0.T
            _turn = float(np.degrees(np.arctan2(_Rr[1, 0] - _Rr[0, 1],
                                                _Rr[0, 0] + _Rr[1, 1])))
            print(f"[prior]   微拧斜坡: 末级腕绕竖轴实际转 {_turn:.1f}° "
                  f"(要求 {cfg.verify_twist_deg:.0f}°) | IK 失败 "
                  f"{n_ik_fail}/{cfg.lift_steps} 级")
            assert _turn >= 0.9 * cfg.verify_twist_deg, (
                f"微拧斜坡只转得动 {_turn:.1f}° < 要求 {cfg.verify_twist_deg:.0f}° —— "
                f"验证段物理上不可能通过 (G3_8_5 同款静默失效). 该抓握位姿附近的"
                f"绕轴旋转 IK 病态, 换候选或降低 verify_twist_deg")
        else:
            # yaw 预演通过的逐级解优先复用 (ik.solve 带随机重启, 重解不保证复现)
            _reh = getattr(self, "_ramp_rehearsal", None)
            _reh_qs = (_reh[1] if _reh is not None
                       and abs(_reh[0] - best_yaw) < 1e-9 else None)
            lift, qw, n_ik_fail = [], rg["q"].copy(), 0
            for i in range(cfg.lift_steps + 1):
                if _reh_qs is not None and i < len(_reh_qs):
                    qw = _reh_qs[i].copy()
                    lift.append(qw.copy())
                    continue
                tgt = gp + np.array([0.0, 0.0, cfg.lift_height * i / max(cfg.lift_steps, 1)])
                r = ik.solve(tgt, quat_to_R(gq), q0=qw, iters=300, pos_tol=2e-4)
                if r["ok"]:
                    qw = r["q"].copy()
                elif i > 0:
                    n_ik_fail += 1
                    J = ik.jacobian(qw)
                    dp = tgt - ik.fk(qw)[0]
                    dq = np.linalg.lstsq(J[:3], dp, rcond=None)[0]
                    qw = np.clip(qw + dq, lo_np, hi_np)
                else:
                    qw = r["q"].copy()
                lift.append(qw.copy())
            self.q_lift = to(np.stack(lift))
            _lv = int(round(min(cfg.verify_lift_m /
                                (cfg.lift_height / max(cfg.lift_steps, 1)), cfg.lift_steps)))
            _rise = float(ik.fk(lift[_lv])[0][2] - ik.fk(lift[0])[0][2])
            print(f"[prior]   抬升斜坡: 第 {_lv} 级实际抬 {_rise*1000:.2f}mm "
                  f"(要求 {cfg.verify_lift_m*1000:.0f}mm) | IK 直解失败 {n_ik_fail}/{cfg.lift_steps} 级"
                  f"{' (已用雅可比补)' if n_ik_fail else ''}")
            assert _rise >= 0.9 * cfg.verify_lift_m, (
                f"抬升斜坡只抬得动 {_rise*1000:.2f}mm < 要求 {cfg.verify_lift_m*1000:.0f}mm —— "
                f"验证段物理上不可能通过, 成功率会恒 0. 该抓握位姿附近的竖直方向 IK 病态, "
                f"换候选或降低 verify_lift_m")
        self.q_lift_delta = self.q_lift - self.q_lift[0:1]
        dmax = float(np.degrees(np.abs(rp["q"] - rg["q"]).max()))
        print(f"[prior] GraspPose 已接入: {npz_path}")
        print(f"[prior]   IK grasp err {rg['pos_err']*100:.2f}cm | pregrasp err "
              f"{rp['pos_err']*100:.2f}cm | pre->grasp 最大关节差 {dmax:.1f}°")
        if self.shell_pts is not None:
            _cg = self._shell_clearance_np(ik, rg["q"])
            _cp = self._shell_clearance_np(ik, rp["q"])
            print(f"[prior]   臂外壳离桌余量(指令位形估计): 抓握位 {_cg*100:+.1f}cm | "
                  f"预抓位 {_cp*100:+.1f}cm")
            # ⚠ 指令位形 ≠ 物理稳态: 实测 PD 稳态肘角与 IK 解差 8.4°, 外壳差 4cm
            # (指令 -4.0 / 稳态 0.0). 这里只警告; **权威口径 = 运行时
            # diag/arm_table_gap_cm** (外壳采样点+仿真真值, 每次 reset 进 TB),
            # 台账 §2.20 把它登记为证伪信号 (训练中持续为负 ⟹ 停训调摆位).
            if min(_cg, _cp) < 0.0:
                print(f"[prior]   ⚠ 指令位形外壳估计为负 —— 以运行时 "
                      f"diag/arm_table_gap_cm 为准, 冒烟/训练时必须核验其为正")
        if getattr(cfg, "retract_start", False):
            self._build_retract_family(ik, gp, gq, op, to)

    def _build_retract_family(self, ik, gp, gq, op, to):
        """退避式起点族: q(d) = IK(gp + d·û, R_g), d ∈ [0, D]。见 docs/APPROACH_DESIGN.md §2。

        û = normalize(手座抓握位 − 物体中心) 再按 λ 抬升。不变量: 沿 û 退避时到物体的
        距离**严格增加** ⟹ 退得越远越安全是构造保证, 不靠 RL 学, 且把退避过程倒放
        就是可行解 ⟹ **任务永远可解**。

        ⚠ 判据不许用代理量(仰角>0 之类) —— 2026-08-16 在零位校准上被代理量连骗三次
          (DESIGN_LOOP §2.24)。这里直接判它要保证的两件事: 外壳离桌单调不降、手→物体
          单调不增反(即距离单调增)。不满足就抬 λ 重试, 抬到上限仍不满足 ⟹ 报错换候选。
        """
        cfg = self.cfg
        r = np.asarray(gp, np.float64) - np.asarray(op, np.float64)
        d_g = float(np.linalg.norm(r))
        r_hat = r / max(d_g, 1e-9)
        D = float(cfg.retract_dmax_k) * d_g
        M = int(cfg.retract_levels)
        from rl_rebuild.correction.kinematics import quat_to_R
        Rg = quat_to_R(np.asarray(gq, np.float64))
        # ★ 档位间距: **近处密、远处疏**(2026-08-16 实测定位)。
        # 等距切分时每档 D/(M-1) = 24.6/23 ≈ **1.07cm**, 而到位判据的位置窗口只有
        # 1.04cm ⟹ 前馈播到最后一步直接**跨过**窗口而不是落进去。实测: 确定性策略
        # 在离目标最近那一刻 实际 1.62cm / 命令 1.75cm / **腕速 18.84cm/s**
        # (参考段自身速度 21.4cm/s, 对得上) —— 是"高速掠过"不是"停在门口",
        # 位置达标率 0.0% 就是这么来的(判据体检 §2.25)。
        # 幂次分布让最后几档的步长落到毫米级, 腕速自然降到判据的 5cm/s 以内。
        _u = np.linspace(0.0, 1.0, M) ** float(cfg.retract_spacing_p)
        ds = D * _u

        def _try(lam):
            u = r_hat + lam * np.array([0.0, 0.0, 1.0])
            u = u / np.linalg.norm(u)
            qs, clr, ok_all = [], [], True
            qw = np.asarray(self._prior_q_grasp.cpu().numpy(), np.float64)
            for d in ds:
                sol = ik.solve(np.asarray(gp, np.float64) + d * u, Rg, q0=qw, iters=200)
                if sol["ok"]:
                    qw = sol["q"]
                else:
                    ok_all = False
                qs.append(qw.copy())
                clr.append(self._shell_clearance_np(ik, qw)
                           if self.shell_pts is not None else np.nan)
            return u, np.stack(qs), np.asarray(clr), ok_all

        lam, chosen = 0.0, None
        for lam in np.arange(0.0, float(cfg.retract_lambda_max) + 1e-9, 0.25):
            u, qs, clr, ok_all = _try(float(lam))
            # 判据①: IK 全程可达。判据②: 外壳离桌全程 ≥ margin 且不出现明显下降。
            drop = (float(np.min(np.diff(clr))) if np.isfinite(clr).all() and len(clr) > 1
                    else 0.0)
            lo = float(np.nanmin(clr)) if np.isfinite(clr).any() else np.inf
            good = ok_all and (not np.isfinite(clr).any()
                               or (lo >= cfg.arm_shell_margin and drop >= -cfg.retract_drop_tol))
            print(f"[retract] λ={lam:.2f} 仰角 {np.degrees(np.arcsin(u[2])):+5.1f}° | "
                  f"IK {'全通' if ok_all else '**有失败**'} | 外壳离桌 "
                  f"{lo * 100:.1f}~{np.nanmax(clr) * 100:.1f}cm | 逐级最大下降 "
                  f"{-drop * 100:.2f}cm  -> {'✅采用' if good else '不合格'}")
            if good:
                chosen = (float(lam), u, qs, clr)
                break
        assert chosen is not None, (
            f"退避族在 λ≤{cfg.retract_lambda_max} 内都不满足准入判据 —— 这个 GraspPose "
            f"候选不适合接近段(手座可能在物体中心下方, 退避会往桌里钻)。换候选, 别调判据。")
        lam, u, qs, clr = chosen
        self.retract_u = u
        self.retract_d = to(ds)
        self.retract_q = to(qs.astype(np.float32))              # (M,7)
        # ---- 退避参考路径 (2026-08-16 用户裁定: 用退避路径当参考) ----
        # 为什么必须做: 前馈是 `ff = q_ref[t] − q_ref[t-1]`, 一直在播**人手轨迹**的关节
        # 增量。退避分支只换了起点没换 ref_t ⟹ 手从 q(d) 起步、参考时钟停在人手轨迹的
        # **随机帧**上, 前馈放的是与当前位置不相干的增量, 策略得用残差去抵消它。
        # (实测征状: term/timeout≈1 走不到、成功率在 0.07~0.58 之间甩。)
        #
        # 修法: 参考路径改成"站姿 → q(D) → … → q(0)=GraspPose", 于是
        #   · 前馈真的指向 GraspPose
        #   · imit 罚的是"偏离这条**保证可行**的路的用量"(退避族倒放即可行解)
        #   · 评测(全站姿起步)也有完整参考, 不再是训练/评测两套
        _K = int(getattr(cfg, "stance_prefix_frames", 0) or 0) or 60
        _stance = self.hand.data.default_joint_pos[0, self.arm_jids].cpu().numpy()
        _rev = qs[::-1].copy()                       # q(D) … q(0), 终点 = GraspPose
        # 前缀段: **手腕在空间里走直线**, 逐点 IK。
        # ⚠ 不能用关节空间插值 —— 关节角走直线时手在空间里画弧, 而弧可能比两个端点
        #   都低。实测(站姿 15.5cm → 终点 3.3cm)中段掉到 **0.42cm**。
        #   直线的好处是有几何保证: 两端点都在桌面以上 ⟹ 连线段必全程在桌面以上(凸性)。
        #   (保证的是**腕点**; 肘/上臂仍要靠下面的外壳校验兜底。)
        _p0, _R0 = ik.fk(_stance.astype(np.float64))
        _p1, _R1 = ik.fk(_rev[0].astype(np.float64))
        _pre, _qw2, _nbad2 = [], _stance.astype(np.float64), 0
        for _i in range(_K):
            _t = _i / float(_K)                       # [0,1), 终点由 _rev[0] 接上
            _ts = _t * _t * (3.0 - 2.0 * _t)          # smoothstep: 速度剖面软启停
            _sol = ik.solve(_p0 + _ts * (_p1 - _p0), _R0 if _ts < 1e-9 else
                            _slerp_R(_R0, _R1, _ts), q0=_qw2, iters=200)
            if _sol["ok"]:
                _qw2 = _sol["q"]
            else:
                _nbad2 += 1
            _pre.append(_qw2.copy())
        _pre = np.stack(_pre)
        _pc = [self._shell_clearance_np(ik, q) for q in _pre] if self.shell_pts is not None \
            else [float("nan")]
        print(f"[retract] 前缀段(腕走直线) {_K} 帧: IK 失败 {_nbad2}/{_K} | "
              f"臂外壳离桌 {np.nanmin(_pc)*100:.2f}~{np.nanmax(_pc)*100:.2f}cm "
              f"{'⚠ 低于余量' if np.nanmin(_pc) < cfg.arm_shell_margin else '✅'}")
        self.retract_path = to(np.concatenate([_pre, _rev], 0).astype(np.float32))
        self.retract_K = _K                          # 前缀长度; 族内第 k 档 -> path 下标 K+(M-1-k)
        print(f"[retract] 参考路径已建: 站姿前缀 {_K} 帧 + 退避 {M} 档 = "
              f"{self.retract_path.shape[0]} 帧, 终点 = GraspPose")
        print(f"[retract] 起点族已建: d_g={d_g*100:.1f}cm D={D*100:.1f}cm "
              f"({cfg.retract_dmax_k}×d_g) × {M} 级 | λ={lam:.2f} "
              f"仰角 {np.degrees(np.arcsin(u[2])):+.1f}°")

    def _nullspace_shell_lift(self, ik, q0, gp, gq, iters=40):
        """零空间抬肘 (2026-08-06, 用户要求真机带壳不碰桌).

        7 自由度臂对同一腕位姿有一维零空间 (肘高低). IK 默认解肘偏低, 前臂外壳
        贴/穿桌面; 沿零空间方向爬升外壳余量, **腕位姿严格不变** (每步 IK 微修回
        目标, 位置公差 0.5mm) —— 抓姿/手指几何零改动.
        """
        from rl_rebuild.correction.kinematics import quat_to_R
        lo = self.arm_lower[0].cpu().numpy().astype(np.float64)
        hi = self.arm_upper[0].cpu().numpy().astype(np.float64)
        Rt = quat_to_R(gq)

        def _drift_fix(qq):
            # 纯零空间步会有一阶漂移, 用 2~3 步 6D 最小二乘拉回 (不换分支)
            for _ in range(3):
                p, R = ik.fk(qq)
                ep = gp - p
                Re = Rt @ R.T
                er = 0.5 * np.array([Re[2, 1] - Re[1, 2],
                                     Re[0, 2] - Re[2, 0],
                                     Re[1, 0] - Re[0, 1]])
                if np.linalg.norm(ep) < 5e-4 and np.linalg.norm(er) < 2e-3:
                    break
                dq = np.linalg.lstsq(ik.jacobian(qq),
                                     np.concatenate([ep, er]), rcond=None)[0]
                qq = np.clip(qq + dq, lo, hi)
            return qq

        q = q0.copy()
        best = self._shell_clearance_np(ik, q)
        for _ in range(80):
            J = ik.jacobian(q)
            n = np.linalg.svd(J)[2][-1]              # 最小奇异方向 ≈ 零空间
            improved = False
            for step in (0.04, -0.04):
                qq = _drift_fix(np.clip(q + step * n, lo, hi))
                if np.linalg.norm(gp - ik.fk(qq)[0]) > 1e-3:
                    continue
                c = self._shell_clearance_np(ik, qq)
                if c > best + 3e-4:
                    q, best, improved = qq, c, True
                    break
            if not improved:
                break
        return q, best

    def _shell_clearance_np(self, ik, q_arm):
        """锚定链离线 FK: 臂外壳表面点最低 z 离桌余量 (m). arm_table_shell 专用.

        ⚠ 关节名必须用**当前侧**的 arm_joint_names (2026-08-18 实锤): 原来按
        cfg.hand_side 写死 "R" 前缀 —— 给左臂算时左关节角被套在右臂关节名上,
        左连杆全按默认角摆, 三个不同位形读出同一个 -9.87cm 鬼数。
        """
        qd = {n: float(v) for n, v in zip(self.arm_joint_names, q_arm)}
        bn = list(self.hand.body_names)
        pts = self.shell_pts.cpu().numpy().astype(np.float64)
        zmin = np.inf
        for k, bi in enumerate(self.shell_bids):
            T = ik.u.link_pose(bn[bi], qd, self._anchor_T, "arm_center")
            zs = (T[:3, :3] @ pts[k].T + T[:3, 3:4])[2]
            zmin = min(zmin, float(zs.min()))
        return zmin - self.cfg.table_top_z

    @staticmethod
    def _qmul_np(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])

    # ---- 动作 ---------------------------------------------------------
    def _fcd_fin_ref(self):
        """FC-D 指参考四段查表 (2026-08-20 用户裁定①③: Pose0 起手学构型 + 29维全参考).
        段1 构型: settle 后 fin_shape_steps 步 open→Pose1 (拇指对掌在站姿旁完成);
        段2 保持: Pose1 直到腕进 4cm; 段3 阶梯: 腕距 4→1cm 查 r0..r5 (Dexonomy 走廊);
        段4 合拢: 到位且腕距<1.2cm → 限速时间斜坡 r5→grasp (单向计数器)。
        全部挂现有物理信号 (episode_length/arrived/_g2_d), 无新时钟; 逐侧路由。"""
        cfg = self.cfg
        K = self._fcd_keys                                            # (8,22)
        el = self.episode_length_buf.float()
        a1 = ((el - cfg.settle_steps)
              / max(float(cfg.fin_shape_steps), 1.0)).clamp(0.0, 1.0)
        ref = K[0].unsqueeze(0) + a1.unsqueeze(1) * (K[1] - K[0]).unsqueeze(0)
        d = getattr(self, "_g2_d", None)
        if d is not None and getattr(self, "arrived", None) is not None:
            lad = ((0.04 - d) / 0.03).clamp(0.0, 1.0) * 5.0
            i0 = lad.floor().clamp(max=4.0).long()
            af = (lad - i0.float()).unsqueeze(1)
            Kl = K[1:7]                                               # r0..r5
            r_lad = Kl[i0] * (1.0 - af) + Kl[(i0 + 1).clamp(max=5)] * af
            use_lad = self.arrived & (d < 0.04)
            ref = torch.where(use_lad.unsqueeze(1), r_lad, ref)
            close_gate = self.arrived & (d < 0.012)
            self._fcd_close_t = self._fcd_close_t + close_gate.long()
            a4 = (self._fcd_close_t.float()
                  / max(float(cfg.fin_close_steps), 1.0)).clamp(0.0, 1.0)
            r_close = K[6].unsqueeze(0) + a4.unsqueeze(1) * (K[7] - K[6]).unsqueeze(0)
            ref = torch.where((self._fcd_close_t > 0).unsqueeze(1), r_close, ref)
            # 2026-08-20 用户裁定(两次修正后定稿): 第四段(合拢到模板 grasp)**保留**,
            # 是免费的参考脚手架; 但它不是学习的终点 —— 模板 grasp 是"贴而未压"的
            # 四指托架(零策略探针: 4/5垫 7N 静稳、搬运段拇指缺口漏瓶), RL 必须在
            # 参考尽头之外继续探索, 真终点由结果判据裁定(≥4垫+向心+candidate+抬升)。
            # 残差界 ±68.8° 足够越过模板深合拢; 大钱全在结果侧(稳抓+5/成功+20)。
        self._fcd_ref_last = ref
        return ref

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        cfg = self.cfg
        self.prev_actions.copy_(self.actions_buf)
        self.actions_buf.copy_(actions.clamp(-1.0, 1.0))
        frozen = self.freeze_ctr > 0
        if frozen.any():
            idx = torch.nonzero(frozen, as_tuple=False).squeeze(-1)
            pose = torch.cat([self.obj_start_pos[idx] + self.scene.env_origins[idx],
                              self.obj_start_quat[idx]], dim=1)
            self.object.write_root_pose_to_sim(pose, idx)
            self.object.write_root_velocity_to_sim(
                torch.zeros(len(idx), 6, device=self.device), idx)
            self.freeze_ctr[idx] -= 1
        gate = (~frozen).float()
        a = self.actions_buf * gate.unsqueeze(1)                        # (N,13)

        # ---- 臂 ----------------------------------------------------------
        # 接近段 (方案 C): q_cmd += 参考增量(前馈) + 残差·远松近紧缩放, 只钳关节限位
        # 抓取段        : q_cmd += 残差,            钳在 arm_center ± arm_dev_max
        in_app = self.task_phase == Phase.PREGRASP
        in_carry = self.task_phase == Phase.TRANSPORT   # place_task 才会出现
        in_ctrl = in_app | in_carry                     # 前馈+残差、只钳关节限位 的相位集
        res = a[:, 0:7] * self.arm_res_scale
        if self.cfg.approach:
            # 前馈 = 参考轨迹**自己**这一步走了多少 (只出现差分, 绝对位置不进公式).
            # 零动作 ⟹ 逐帧复现人手的速度剖面 —— 这条性质是 repo 两次成功的共同依赖
            # (dexmate_env.py:384-392 的注释: 只累积不给前馈 ⟹ 手臂冻在起点).
            # place: 前馈播到 _ff_end=re, 搬运段零动作 = 复现人手的搬运动作.
            if getattr(self, "retract_path", None) is not None:
                # 退避参考: 前馈来自"站姿→q(D)→…→q(0)=GraspPose"这条**保证可行**的路,
                # 而不是人手轨迹 (见 _build_retract_family 里的说明)。
                _L = self.retract_path.shape[0]
                t = self.ref_t.clamp(max=_L - 1)
                q_base = self.retract_path[t]
                ff = (q_base - self.ref_q_prev) * (in_ctrl & (self.ref_t < _L - 1)
                                                   ).float().unsqueeze(1)
            else:
                t = self.ref_t.clamp(max=self._ff_end)
                q_base = self.q_ref[t]
                ff = (q_base - self.ref_q_prev) * (in_ctrl & (self.ref_t < self._ff_end)
                                                   ).float().unsqueeze(1)
            if getattr(self.cfg, "minimal_no_ff", False):
                ff = torch.zeros_like(ff)      # 最简模式: 无前馈, 手全靠策略自己走
            self.ref_q_prev = q_base
            # Phase2-RL (2026-08-18晚, 用户裁定): 名义斜坡退役 —— 闩后 ff 自然结束,
            # PreGrasp→GraspPose 由 RL **自走**, GraspPose 只作指引(离散里程碑+成功奖,
            # 见 _get_dones 的 g2 判据与 _get_rewards 的 g2_m*)。斜坡构建保留备消融。
            # 远松近紧: 手要自己走完那 ~16cm, 但接触前必须回到毫米级 (只在接近段)
            d_pos = (self._anchor_w() - self._target_w()).norm(dim=1)
            u = ((d_pos - self.cfg.dyn_d_near)
                 / max(self.cfg.dyn_d_far - self.cfg.dyn_d_near, 1e-6)).clamp(0.0, 1.0)
            # 近端尺度 dyn_arm_near(默认 1.0=原行为) -> 远端 dyn_arm_far
            _nr = float(getattr(self.cfg, "dyn_arm_near", 1.0))
            if getattr(self.cfg, "minimal_fixed_res", False):
                s_arm = torch.full_like(u, _nr)   # 最简模式: 恒定步长上限
            else:
                s_arm = _nr + u * (self.cfg.dyn_arm_far - _nr) * in_app.float()
            res = res * s_arm.unsqueeze(1)
            if getattr(cfg, "arm_ff_gate", False):
                # AAG-v3.1: 到位前臂残差**小带**(arm_ff_pre, 微调避蹭+补两把尺缺口;
                # v3 锁零已证伪 —— ff 终点差 ~3cm, 锁零=腕永停 4-5cm 外), 到位后按
                # phase_step 线性 ramp 到 arm_ff_post。arrived 在直抓桶出生即 True;
                # GRASP→LIFT 换相 phase_step 清零会再 ramp 一次, 温和副作用可接受。
                _pre = float(getattr(cfg, "arm_ff_pre", 0.2))
                _pl = float(getattr(cfg, "arm_ff_pre_left", -1.0))
                if _pl >= 0.0 and getattr(getattr(self, "_cur", None),
                                          "name", "") == "left":
                    _pre = _pl        # 左臂单独带宽 (消融1: 绕障余地)
                _post = float(getattr(cfg, "arm_ff_post", 0.33))
                _gr = _pre + (self.phase_step.float()
                              / max(int(getattr(cfg, "arm_ff_ramp", 20)), 1)
                              ).clamp(0.0, 1.0) * max(_post - _pre, 0.0)
                res = res * torch.where(self.arrived, _gr,
                                        torch.full_like(_gr, _pre)).unsqueeze(1)
            if getattr(cfg, "seg_gate", False):
                _sga, _sgf = self._seg_scale()
                res = res * _sga
                self._seg_f = _sgf              # 手指段门, 下面指残差处用
            _fz = float(getattr(self.cfg, "curobo_ff_freeze_cm", 0.0))
            # _fz=0 ⟹ _far 全 1: 不冻结任何 env, 但回拉照常生效
            _far = (d_pos * 100.0 >= _fz).float().unsqueeze(1)
            if _fz > 0.0:
                # 近端冻结: 距目标 < _fz cm 的 env 前馈清零 (ref_q_prev 已在上面更新, 无跳变)
                ff = ff * _far
            # 弱回拉锚定 (2026-08-17 L2 事故): 差分前馈会把保持段里残差攒出的漂移
            # 原样平移进后面整条路径(左手撞障→单关节滞后20°→term/stuck 全灭)。
            # 每步把 q_cmd 往计划位形拉 curobo_ff_pull 比例, 漂移半衰期 ~1/pull 步;
            # 冻结区内不拉 —— 近端绕杯的主动偏离不能被磨掉。
            #
            # ★ 2026-08-23 修 (AAG 全系静默失效): 回拉原本嵌在 `if _fz > 0.0:` 里,
            # 而 AAG 一律用 --ff_freeze_cm 0 ⟹ 锚从未执行, 但启动日志照样打印
            # "回拉 0.08/步"。臂残差是积分器 (q_cmd += ff + res, 只被关节限位夹),
            # 少了锚就无界漂移: v37 实测左腕离靶 39cm→49cm 单调恶化 20M 步,
            # 而同参考纯前馈(res=0)两手都停在 3.1/3.5cm —— 参考没问题, 是锚没上。
            _pl = float(getattr(self.cfg, "curobo_ff_pull", 0.0))
            if _pl > 0.0:
                ff = ff + _pl * (q_base - self.q_cmd) * _far * in_ctrl.float().unsqueeze(1)
            if self.cfg.place_task:
                # 搬运段残差降档 (§2.19 P1a): ff 主导复现人手搬运, 残差只做防滑微调
                res = torch.where(in_carry.unsqueeze(1),
                                  res * self.cfg.carry_res_scale, res)
            self.res_step_cm = (res.abs().mean(dim=1) * 100.0)          # 诊断: 残差用量
            if getattr(cfg, "arm_abs_res", False):
                # ---- 方案C: 绝对参考 + 有界累加残差 (与手指路径同构) ------------
                # q_cmd = q_ref[t] + arm_res, arm_res 每步累加动作、夹在 ±arm_abs_dev。
                # 与差分版的区别只有一条: 参考每步**重新钉死**, 残差再大也只是一个
                # 有界偏移 ⟹ 漂移在结构上不可能, 不再依赖 ff_pull 那种软回归力。
                # 残差**照样能累积**(修正下一步还在), 丢掉的只是"持久形变改走法"——
                # cuRobo 参考已全局可行, 这个自由度用不上 (用户 2026-08-23 裁定)。
                _adev = float(getattr(cfg, "arm_abs_dev", cfg.arm_dev_max))
                self.arm_res = (self.arm_res + res).clamp(-_adev, _adev)
                self.q_cmd = torch.where(
                    in_ctrl.unsqueeze(1),
                    (q_base + self.arm_res).clamp(self.arm_lower, self.arm_upper),
                    (self.q_cmd + res).clamp(self.band_lo, self.band_hi))
            else:
                q_new = self.q_cmd + ff * gate.unsqueeze(1) + res
                self.q_cmd = torch.where(in_ctrl.unsqueeze(1),
                                         q_new.clamp(self.arm_lower, self.arm_upper),
                                         q_new.clamp(self.band_lo, self.band_hi))
        else:
            self.q_cmd = (self.q_cmd + res).clamp(self.band_lo, self.band_hi)
        self.arm_tgt_prev = self.arm_tgt.clone()
        in_verify = self.task_phase == Phase.LIFT
        lvl_f = (self.verify_k.float() / max(cfg.verify_ramp_steps, 1)
                 ).clamp(0.0, 1.0) * self.verify_lvl * in_verify.float()  # (N,)
        i0 = lvl_f.floor().long().clamp(0, self.cfg.lift_steps - 1)
        w = (lvl_f - i0.float()).unsqueeze(1)
        dq_lift = (1 - w) * self.q_lift_delta[i0] + w * self.q_lift_delta[i0 + 1]
        self.arm_tgt = (self.q_cmd + dq_lift).clamp(self.arm_lower, self.arm_upper)
        _ro10 = getattr(self, "_rel_on", None)
        if _ro10 is not None:
            # v10.8 (2026-08-25): 松手期臂目标冻结 —— release 相位不在策略
            # 训练分布里, 臂残差乱挥把已放稳的物体推走 (rel-scope: 全开后
            # still=0.00, 杯被推 12cm/41°)。松手=外生斜坡哲学, 臂一并脚本化:
            # 进松手瞬间拍快照, _rel_on 期间锁死 (episode reset 时 _rel_on
            # 清零 ⟹ has 位自动解锁)。_rel_arm_q/_rel_arm_has 在 SIDE_ATTRS。
            if not hasattr(self, "_rel_arm_q"):
                self._rel_arm_q = self.arm_tgt.clone()
                self._rel_arm_has = torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device)
            if not hasattr(self, "_stance_q"):
                self._stance_q = self.arm_tgt.clone()
            # 出生站姿快照 (E2E RETREAT 的撤退终点): 回合首步实拍
            _st12 = self.episode_length_buf <= 1
            if _st12.any():
                self._stance_q[_st12] = self.arm_tgt[_st12]
            _ns10 = _ro10 & ~self._rel_arm_has
            if _ns10.any():
                self._rel_arm_q[_ns10] = self.arm_tgt[_ns10]
            self._rel_arm_has = (self._rel_arm_has | _ns10) & _ro10
            self.arm_tgt = torch.where(_ro10.unsqueeze(1),
                                       self._rel_arm_q, self.arm_tgt)
            _ra12 = getattr(self, "_ret_alpha", None)
            if _ra12 is not None:
                # E2E RETREAT: 松手完成后臂从冻结位姿 lerp 回出生站姿 (合成
                # 撤退 —— 分段器实证示范 take 被剪, 无完整撤退可播)
                _a12 = _ra12.clamp(0.0, 1.0).unsqueeze(1)
                _ret_tgt = (1.0 - _a12) * self._rel_arm_q \
                    + _a12 * self._stance_q
                self.arm_tgt = torch.where((_ra12 > 0).unsqueeze(1),
                                           _ret_tgt, self.arm_tgt)

        # 合拢: 参考斜坡 + 策略调制 (零动作 = 匀速合到 c_grasp 停; a_c=-1 停住;
        # 比 nominal 更深的挤压只能由 a_c>0 主动选择)
        # ⚠ 接近段整条手指通道**门控关闭** —— 手必须张开着飞过去, 否则是握着拳头去碰物体.
        # place: 搬运段手指冻结在抓握深度 (握稳搬运), 放置段脚本化松手 (下方覆盖).
        in_place = self.task_phase == Phase.PLACE
        _pg = ~(in_app | in_carry | in_place)
        if getattr(cfg, "l5_couple", False):
            # L5 (2026-08-17 用户裁定③): 冻结区(腕距目标 < ff_freeze_cm)内解锁手指
            # 通道 —— 边缓速进门边渐进合指, 不设"先到位再合指"的硬闸。
            _fzm = float(getattr(cfg, "curobo_ff_freeze_cm", 5.0)) / 100.0
            self._l5_dp = (self._anchor_w() - self._target_w()).norm(dim=1)
            _pg = _pg | (in_app & (self._l5_dp < _fzm))
        hand_gate = gate * _pg.float()
        # ★ 距离门控 (2026-08-16 用户裁定): 手指自由度随"离物体多近"渐进放开。
        #   相位门信任标注, 而 pour/17 的接触标注在指尖还差 7.8cm 时就触发 ——
        #   结果是在空气里握拳。距离门只信当下量到的几何, 标注错了也拦得住。
        #   它同时门住**参考合拢斜坡**(ref)与**策略动作**, 因为两者都乘 hand_gate。
        if getattr(cfg, "finger_gate_on", False) and getattr(self, "_fgate_dg", None):
            # ⚠ 用**腕到物体**的距离, 不是指垫到物体 —— 后者与手指开合互为因果, 会死锁
            #   (见 cfg.finger_gate_near_k 的注释与 Grasp3_CTRL 的实测)。
            _d = (self.wrist_pos_w - self.object.data.root_pos_w).norm(dim=1)
            _near = self._fgate_dg * cfg.finger_gate_near_k
            _far = self._fgate_dg * cfg.finger_gate_far_k
            _dg = ((_far - _d) / max(_far - _near, 1e-6)).clamp(0.0, 1.0)
            hand_gate = hand_gate * _dg
            self._diag_fgate += _dg                                   # 盘面: 平均开度
            self._diag_fgate_n += 1
        if getattr(cfg, "l5_couple", False):
            # 合拢参考追踪 c(d) = c_grasp·clamp((fz−d)/fz, 0, 1):
            # d=fz 参考张开, d→0 参考到 c_grasp; 追踪速率仍受 closure_ref_rate 限幅,
            # 策略动作 a_c 照旧叠加 —— 模板只是软指标, 力度细节 RL 自己修(裁定②)。
            _fzm = float(getattr(cfg, "curobo_ff_freeze_cm", 5.0)) / 100.0
            self._l5_cref = cfg.c_grasp * ((_fzm - self._l5_dp)
                                           / max(_fzm, 1e-6)).clamp(0.0, 1.0)
            ref = (self._l5_cref - self.closure).clamp(-cfg.closure_ref_rate,
                                                       cfg.closure_ref_rate)
        else:
            ref = cfg.closure_ref_rate * (self.closure < cfg.c_grasp).float()
        # joints 模式: a_c 取消, 合拢模板只按**参考速率**推进(策略走 22 关节残差)
        # 7 维臂动作(approach_only)/joints 模式: 都没有 a[:,7] 这一维
        _arm_only = a.shape[1] <= 7
        _ac = (torch.zeros_like(ref) if (self._joint_hand or _arm_only)
               else a[:, 7] * cfg.closure_rate_max)
        self.closure = (self.closure + hand_gate * (ref + _ac)
                        ).clamp(cfg.closure_min, cfg.closure_max)
        if cfg.place_task:
            # 脚本化松手斜坡 (与微抬升同哲学: 廉价可靠的物理裁判, 不学释放时序)
            self.closure = torch.where(
                in_place, (self.closure - cfg.place_release_rate).clamp(min=0.0),
                self.closure)
        # 每指残差: 指 i 的模板深度 = clip(c + δ_i)
        if not (self._joint_hand or _arm_only):   # joints/臂-only 模式: a_δ 取消
            self.fin_delta = (self.fin_delta
                              + a[:, 8:13] * cfg.delta_rate_max * hand_gate.unsqueeze(1)
                              ).clamp(-cfg.delta_max, cfg.delta_max)
        c_i = (self.closure.unsqueeze(1) + self.fin_delta
               ).clamp(cfg.closure_min, cfg.closure_max)                # (N,5)
        if self.n_active < 5:
            # 非参与指钉在 c=1 (= prior 姿态): Dexonomy 生成时它们就摆在不碰撞的
            # 收拢位, 跟随 c 张开反而会伸进场景 (捏盖时中/环/小指戳到瓶身)
            c_i = torch.where(self.finger_active.bool().unsqueeze(0).expand_as(c_i),
                              c_i, torch.ones_like(c_i))
        cj = c_i[:, self.fmap]                                          # (N,22)
        self.finger_tgt_prev = self.finger_tgt.clone()
        _rr10 = getattr(self, "_rel_ramp", None)
        if _rr10 is not None:
            cj = cj * _rr10.unsqueeze(1)   # v10 RELEASE: 松手斜坡外生压低合拢系数
        _tmpl = self.q_open + cj * (self.q_close - self.q_open)         # 合拢模板
        if getattr(cfg, "fin_ref_track", False) and \
                getattr(self, "_fcd_keys", None) is not None:
            if getattr(self, "_fin_ref_path", None) is not None:
                # AAG: 指参考=标准成功轨迹逐行 (含 squeeze 深拇段), ref_t 索引
                _fi2 = self.ref_t.clamp(max=self._fin_ref_path.shape[0] - 1)
                if getattr(cfg, "fin_prog_gate", False):
                    # 手指行进门 (2026-08-22 用户采纳"跟上才前进"): 到位前指参考
                    # 钳在弯根部末行 —— 治 v3 对空合拢/左手门外戳 (腕没到, 合拢
                    # 时钟不许走)
                    _fh = int(getattr(cfg, "fin_hold_row", 90))
                    _open = self.arrived
                    _gd = float(getattr(cfg, "fin_gate_dpos_cm", 0.0))
                    if _gd > 0.0:
                        # ★ 2026-08-24 判死"硬闸断崖": 原判据只认 arrived 闩死 ⟹ 腕距
                        # 一旦漂过 eps(1.35cm), 指参考永久钳在 fin_hold_row,
                        # pad_first/pad_hold/fc_pot/g2_m1 **同时归零** —— 下游收入断崖式
                        # 消失, 策略掉进"什么都不做"的局部最优(实测左手 +0.072 -> +0.019,
                        # 而唯一涨项只有 form_pot +0.0016 ⟹ 不是逃逸, 是爬不出来)。
                        # 改软闸: 腕进到 fin_gate_dpos_cm 就放行 —— 原意("别对空合拢/
                        # 门外戳")靠这个距离带保住, 但梯度不再断崖。
                        _dp_now, _, _ = self._align_err()
                        _open = _open | (_dp_now < _gd / 100.0)
                    _fi2 = torch.where(_open, _fi2, _fi2.clamp(max=_fh))
                _tmpl = self._fin_ref_path[_fi2]
                self._fcd_ref_last = _tmpl
                self._dbg_fi2 = _fi2                  # 探针: 指参考实际用的行号
            else:
                _tmpl = self._fcd_fin_ref()   # FC-D: 四段指参考 (查表生成)
        # Phase2-RL: 指型不再走脚本斜坡 —— 模板保持张开, 合拢由 RL 的 22 关节残差
        # 自己完成 (指引 = g2 里程碑/成功判据里的"指型贴近抓姿")。
        if self._joint_hand:
            # 逐关节残差叠在模板之上。门控与合拢同一个(接近/搬运/放置段手指冻结)。
            # PreGrasp29 (2026-08-18晚): 手指**全程解锁**(只随 settle 冻结), 约束改由
            # fin_quiet 软奖励承担 —— "鼓励别动"而非"锁死", 力度细节留给 RL。
            _fgate = (gate if getattr(cfg, "pregrasp29", False)
                      else hand_gate)
            _fscale = self.finger_res_scale
            if getattr(cfg, "fin_ref_track", False) and \
                    getattr(self, "_g2_d", None) is not None:
                # FC-D 慢合拢闸 (2026-08-20 用户裁定"收拢慢慢学"): 指残差(=探索幅度)
                # 随腕距缩放 ≥10cm→1.0x / ≤3cm→fin_dyn_near; 合拢段再压到 fin_dyn_close
                _fd = ((self._g2_d - 0.03) / 0.07).clamp(0.0, 1.0)
                _fdyn = cfg.fin_dyn_near + _fd * (1.0 - cfg.fin_dyn_near)
                _fdyn = torch.where(self._fcd_close_t > 0,
                                    torch.full_like(_fdyn, cfg.fin_dyn_close), _fdyn)
                _fscale = self.finger_res_scale.unsqueeze(0) * _fdyn.unsqueeze(1)
            if getattr(cfg, "seg_gate", False):
                _sgf2 = getattr(self, "_seg_f", None)
                if _sgf2 is None:
                    _, _sgf2 = self._seg_scale()
                _fscale = _fscale * _sgf2       # 逐关节×逐段 (拇指与其余分开)
            self.fin_res = (self.fin_res
                            + a[:, 7:29] * _fscale * _fgate.unsqueeze(1)
                            ).clamp(-self.finger_dev_max, self.finger_dev_max)
            if getattr(cfg, "pregrasp29", False):
                self._fin_act_mag = a[:, 7:29].abs().mean(dim=1)
            _tmpl = _tmpl + self.fin_res
        self.finger_tgt = _tmpl.clamp(self.dof_lower, self.dof_upper)
        self._substep = 0
        # _apply_action 继承 DexMate 的子步插值下发, 零改动

    # ---- 任务几何 -----------------------------------------------------
    def _seg_scale(self):
        """按参考段号给出 (臂 scale (N,1), 指 scale (N,22))。

        2026-08-24 用户编排: 探索预算按阶段+关节组分配, 而不是全局 entropy_coef
        (后者是 loss 里的**一个标量**, 给不了逐维)。门控置 0 = 该组该段**零探索**,
        因为探索进入系统的唯一路径是 `动作 × 残差步长 × 门控`。
          ① stance_to_5cm/hold1: 臂动(留探索避障) 指冻
          ② root_bend:           只有拇指动
          ③ advance/hold2:       臂动(推进避障) 指冻
          ④⑤ close/squeeze/hold3: 全开
        """
        N, dev = self.num_envs, self.device
        segs = getattr(self, "_seg_rows", None)
        a = torch.ones(N, 1, device=dev)
        f = torch.ones(N, 22, device=dev)
        if not segs or not getattr(self.cfg, "seg_gate", False):
            return a, f
        _sa = self.cfg.seg_arm_scale
        _st = self.cfg.seg_thumb_scale
        _so = self.cfg.seg_other_scale
        tm = self._thumb_mask.unsqueeze(0)                       # (1,22)
        t = self.ref_t
        assert len(segs) == len(_sa), (
            f"分段门控表长 {len(_sa)} != 参考段数 {len(segs)} —— "
            f"改了编舞就必须同步改 cfg.seg_*_scale, 静默截断会让后面几段全是 1.0")
        for i, (_nm, lo, hi) in enumerate(segs):
            m = (t >= lo) & (t <= hi)
            if not bool(m.any()):
                continue
            a = torch.where(m.unsqueeze(1), torch.full_like(a, float(_sa[i])), a)
            _fv = tm * float(_st[i]) + (1.0 - tm) * float(_so[i])    # (1,22)
            f = torch.where(m.unsqueeze(1), _fv.expand(N, 22), f)
        return a, f

    def _anchor_w(self) -> torch.Tensor:
        return self.wrist_pos_w + quat_apply(
            self.wrist_quat_w, self.anchor_local.expand(self.num_envs, 3))

    def _target_w(self) -> torch.Tensor:
        return self.object.data.root_pos_w + quat_apply(
            self.object.data.root_quat_w, self.aff_local.expand(self.num_envs, 3))

    def _pad_dists(self) -> torch.Tensor:
        """(N,5) 每个软垫到物体表面采样点的最近距离.

        ⚠ **口径 = elastomer 的 link 原点**(`tip_pos_w = body_pos_w[tip_ids]`), 不是垫面。
        垫体整个偏在 link 原点一侧, STL 顶点离原点中位 **2.35~2.59cm** ⟹ 这个值读到
        ~12~20mm 时, 垫面其实已经接触。**不要把它当"还差多少才碰到"读**。
        判据侧不受影响(接触/候选走的是**接触力**), 塑形侧也不受影响(单调势, 常数偏移
        不改变优化方向) —— 受影响的只有**人的判读**。2026-08-15 我按它量了一整轮,
        所有数偏大约 2.4cm, 并据此作废过一个结论。要量"垫面到表面"请用 STL 顶点。
        """
        return self._pad_dist_normal()[0]

    def _pad_dist_normal(self):
        """(N,5) 最近表面距离 + (N,5,3) 世界外法向 (无法向时 None) + (N,5) 最近点**索引**.

        法向是 cent 的方向基准: 力压进表面 = 好接触. 比"指向物体原点"形状无关 ——
        环形物体的原点是孔心, 从外侧捏住时两者能差几十度 (见 cfg.cent_mode).
        索引原来算完就丢, 现在留给接触分数图 (P1a) —— 零新增计算.
        """
        d = torch.cdist(self.tip_pos_w, self._obj_surface_points())      # (N,5,P)
        dmin, idx = d.min(dim=2)
        if self.obj_normals is None:
            return dmin, None, idx
        nl = self.obj_normals[idx.reshape(-1)]                          # (N*5,3) 规范系
        nq = self.object.data.root_quat_w[:, None, :].expand(-1, 5, -1).reshape(-1, 4)
        nw = quat_apply(nq, nl).reshape(self.num_envs, 5, 3)            # 只旋转, 不平移
        return dmin, nw / nw.norm(dim=-1, keepdim=True).clamp(min=1e-6), idx

    # ---- 接近段几何 ---------------------------------------------------
    def _set_arm_center(self, mask, q):
        """把这些 env 的抓取相位偏差带中心设成 q, 并重算 band (逐 env).

        为什么中心要能变: 接近段切到抓取段时, q_cmd 停在"到达位形"上, 而它不一定
        等于 prior 的 q_pregrasp (IK 分支可能不同). 若把带子钉死在 q_pregrasp 上,
        切换那一刻目标会被一把拽回去 -> 位控臂跟不上 -> term/stuck.
        """
        if not mask.any():
            return
        lo = q - self.cfg.arm_dev_max
        hi = q + self.cfg.arm_dev_max
        if getattr(self, "_prior_q_grasp", None) is not None:
            gq = self._prior_q_grasp.unsqueeze(0)     # 带子始终罩住抓握 IK 解
            lo = torch.minimum(lo, gq - 0.08)
            hi = torch.maximum(hi, gq + 0.08)
        self.arm_center[mask] = q[mask] if q.dim() == 2 else q
        self.band_lo[mask] = lo[mask].clamp(min=self.arm_lower[0]) if lo.dim() == 2 \
            else lo.clamp(min=self.arm_lower[0])
        self.band_hi[mask] = hi[mask].clamp(max=self.arm_upper[0]) if hi.dim() == 2 \
            else hi.clamp(max=self.arm_upper[0])

    def _align_err(self):
        """(N,) 位置误差 m, (N,) 姿态误差 rad, (N,3) 姿态误差轴角(腕系, 给观测).

        ⚠ 位置量的是 **腕位 → GraspPose 要求的腕位**, 不是"合拢中心 → 接触质心".
        两者不是同一个量: 2026-08-02 冒烟实测, 手**停在 prior 抓握位姿上**时
        "合拢中心→接触质心" 仍有 **3.42cm** (通用合拢模板的五垫质心 ≠ Dexonomy 的
        接触质心), 于是 ε_pos=1.04cm 永远达不到, 相位**永远切不过去**;
        而对齐势的极小点也不在 GraspPose 上. 而 P0.0 容差曲线量的正是"腕位姿偏离
        prior 多少", 阈值必须配同一个量.
        """
        org = self.scene.env_origins
        d_pos = ((self.wrist_pos_w - org) - self._grasp_pos_w).norm(dim=1)
        qw = self._qsign(self.wrist_quat_w)
        dq = quat_mul(self._grasp_quat_w.expand(self.num_envs, 4), quat_conjugate(qw))
        dq = self._qsign(dq)
        ang = 2.0 * torch.acos(dq[:, 0].clamp(-1.0, 1.0))               # (N,)
        s = (1.0 - dq[:, 0] * dq[:, 0]).clamp(min=1e-8).sqrt()
        axis = dq[:, 1:] / s.unsqueeze(1)
        return d_pos, ang, quat_apply(quat_conjugate(qw), axis * ang.unsqueeze(1))

    def _phi(self) -> torch.Tensor:
        """接近进度条 φ = t_ref / gs ∈ [0,1]. **外生时钟**, 策略动不了它 (见 DESIGN_LOOP A4)."""
        return (self.ref_t.float() / max(self.gs, 1)).clamp(0.0, 1.0)

    def _pad_signals(self, normals=None):
        """接触力几何 -> (mag, e, cent, r_imb, tau_n, **G(N,5,3) 力向量**). 详见 cfg.

        G 是原始力向量 (世界系). 2026-08-02 起 actor 拿的是它转到腕系的版本 ——
        `cent`/`r_imb`/`tau_n` 都用了**物体真值位姿**(表面法向/质心), 是特权信息,
        只能给 critic; 而力向量本身是真实力传感器就能给的, 且信息量比标量 cent 更大.
        """
        cfg = self.cfg
        G = cfg.pad_force_sign * torch.cat(
            [s.data.force_matrix_w.view(self.num_envs, 1, 3)
             for s in self._contact_sensors], dim=1)                    # (N,5,3)
        G = G.nan_to_num(0.0)
        # 模长钳 50N: 物理瞬态能读出上千牛, 顺着 over_force 打出 -2.5万/窗的尖峰
        # (31M 实测); 合法挤压 8N 已算"捏爆", 50 只挡发散不挡语义.
        mag = G.norm(dim=-1).clamp(max=50.0)
        e = ((mag - cfg.squeeze_f_min)
             / max(cfg.squeeze_f0 - cfg.squeeze_f_min, 1e-6)).clamp(0.0, 1.0)
        # 非参与指 (模板指数 < 5 时) 不进任何接触判据/收入: 2 指捏取里中/环/小指
        # 蹭到物体不该被算成"垫", 也不该贡献向心/不对称/力矩统计
        e = e * self.finger_active.unsqueeze(0)
        tips = self.tip_pos_w
        c = self.object.data.root_pos_w[:, None, :]
        Gh = G / mag.unsqueeze(-1).clamp(min=1e-6)
        if normals is not None:                 # 局部法向版: 力是否压进表面
            u = -normals                        # 外法向取反 = "压入"方向
        else:                                   # 旧版: 指向物体原点
            u = c - tips
            u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cent = (((Gh * u).sum(-1)).clamp(min=0.0) * e).sum(dim=1) / float(self.n_active)
        tot = mag.sum(dim=1)
        r_imb = G.sum(dim=1).norm(dim=1) / (tot + 1e-6)
        tau = torch.cross(tips - c, G, dim=-1).sum(dim=1)
        tau_n = (tau.norm(dim=1) / ((tot + 1e-6) * cfg.obj_char_radius)).clamp(max=2.0)
        return mag, e, cent, r_imb, tau_n, G

    # ---- 阶段机 + 终止 (dones 先于 rewards 被调, 信号缓存给后者) ----
    def _get_dones(self):
        # 最后一个物理子步之后、reward/obs 之前的纯投影, 消掉一个子步的积分滞后
        # (与 BottleReconstructionEnv 同款钩子; 无螺旋装配时短路)
        self._SA.apply_screw(self, integrate_angle=False)
        cfg = self.cfg
        origins = self.scene.env_origins
        active = self.episode_length_buf >= cfg.settle_steps
        obj_pos = self.object.data.root_pos_w - origins
        obj_spd = self.object.data.root_lin_vel_w.norm(dim=1).nan_to_num(1e3)
        obj_rot = self.object.data.root_ang_vel_w.norm(dim=1).nan_to_num(1e3)
        pad_d, pad_n, pad_i = self._pad_dist_normal()
        if self._aff_pts_local is not None:
            # 塑形专用距离: 垫 -> 视频接触带 (随物体位姿旋转平移); 裁判用的 pad_d 不变
            _M = self._aff_pts_local.shape[0]
            _aq = self.object.data.root_quat_w[:, None, :].expand(-1, _M, -1).reshape(-1, 4)
            _aw = quat_apply(_aq, self._aff_pts_local[None].expand(self.num_envs, -1, -1)
                             .reshape(-1, 3)).view(self.num_envs, _M, 3) \
                + self.object.data.root_pos_w[:, None, :]
            pad_d_shape = torch.cdist(self.tip_pos_w, _aw).min(dim=2).values
        else:
            pad_d_shape = pad_d
        mag, e_pad, cent, r_imb, tau_n, _G = self._pad_signals(pad_n)
        pads_on = e_pad > 0
        n_pads = pads_on.sum(dim=1)
        rel_spd = (self.object.data.root_lin_vel_w - self.wrist_linvel_w).norm(dim=1)
        disp = (obj_pos - self.obj_start_pos)[:, :2].norm(dim=1).nan_to_num(1e3)
        rise = obj_pos[:, 2] - self.obj_rest_z
        quality = cent - cfg.quality_lam_imb * r_imb - cfg.quality_lam_tau * tau_n

        newly_touch = pads_on & ~self.pad_touched & active.unsqueeze(1)
        self.pad_touched |= pads_on & active.unsqueeze(1)
        self.any_contact |= pads_on.any(dim=1) & active
        # 物体倾角 (相对初始姿态的轴偏转): 初始局部 z 轴 vs 当前局部 z 轴
        _z0 = quat_apply(self.obj_start_quat,
                         torch.tensor([0.0, 0.0, 1.0], device=self.device
                                      ).expand(self.num_envs, 3))
        _zn = quat_apply(self.object.data.root_quat_w,
                         torch.tensor([0.0, 0.0, 1.0], device=self.device
                                      ).expand(self.num_envs, 3))
        tilt_deg = torch.rad2deg(torch.acos(
            (_z0 * _zn).sum(dim=1).clamp(-1.0, 1.0))).nan_to_num(0.0)
        self.tilt_max_deg = torch.where(self.any_contact,
                                        torch.maximum(self.tilt_max_deg, tilt_deg),
                                        self.tilt_max_deg)
        # yaw 自旋 (绕自身轴, 倾角指标对它盲): 相对四元数的 z-twist 分量
        _qr = quat_mul(quat_conjugate(self.obj_start_quat),
                       self.object.data.root_quat_w)
        yaw_deg = torch.rad2deg(2.0 * torch.atan2(_qr[:, 3].abs(),
                                                  _qr[:, 0].abs())).nan_to_num(0.0)
        self.yaw_max_deg = torch.where(self.any_contact,
                                       torch.maximum(self.yaw_max_deg, yaw_deg),
                                       self.yaw_max_deg)

        # ---- 撞桌 (几何) + 指间交叉 (几何) ----
        # ⚠ 铰链必须钳上限 (台账铁律"惩罚项永远要有单步上界"): 物理发散时 body 掉到
        # 桌下几百米, 深度² × 40 body × 20 权重 = 百万量级, 实测把回合奖励打到 -115万
        # 并毒掉 value 归一化器 (run 03-37-46). 深穿本身由 table_crash 终止兜底.
        z = (self.hand.data.body_pos_w[:, self.table_bids, 2] - origins[:, 2:3]
             ).nan_to_num(nan=cfg.table_top_z)
        table_pen = (cfg.table_top_z + cfg.table_margin - z).clamp(min=0.0, max=0.05)
        table_pen = table_pen.square().sum(dim=1)
        if getattr(cfg, "table_touch_fail", False):
            # 2026-08-20 用户裁定(真机红线): 接触即 Fail —— 手部 body 原点低于
            # 桌面+容差即终止 (原点在连杆内部, 到面即已实际接触); 罚带保留在其上,
            # 实现"允许贴近, 不允许碰撞"。旧口径(深穿 2cm 才判)仅防"从桌下托"。
            table_crash = active & \
                (z.min(dim=1).values < cfg.table_top_z + cfg.table_touch_tol)
        else:
            table_crash = active & \
                (z.min(dim=1).values < cfg.table_top_z - cfg.table_crash_depth)
        bp = self.hand.data.body_pos_w
        cross_d = (bp[:, self.cross_a] - bp[:, self.cross_b]).norm(dim=-1)
        cross_pen = (cfg.finger_cross_dist - cross_d).clamp(min=0.0).sum(dim=1)
        # ---- P0.3: 全臂撞桌 + 臂↔躯干/另一臂 间隙 (诊断常开, 罚由 w_* 控制) ----
        arm_p = bp[:, self.arm_bids]                                    # (N,A,3)
        z_arm = (arm_p[:, :, 2] - origins[:, 2:3]).nan_to_num(nan=cfg.table_top_z)
        if self.shell_pts is not None:
            # 外壳口径: 连杆表面点旋到世界系取最低 z (真机带壳不碰桌, 2026-08-05)
            L, P = self.shell_pts.shape[:2]
            lq = self.hand.data.body_quat_w[:, self.shell_bids]         # (N,L,4)
            lp = bp[:, self.shell_bids]                                 # (N,L,3)
            pw = quat_apply(
                lq.unsqueeze(2).expand(-1, -1, P, -1).reshape(-1, 4),
                self.shell_pts.unsqueeze(0).expand(self.num_envs, -1, -1, -1
                                                   ).reshape(-1, 3),
            ).view(self.num_envs, L, P, 3)
            z_shell = (lp[:, :, 2].unsqueeze(2) + pw[..., 2]
                       - origins[:, 2].view(-1, 1, 1)).nan_to_num(nan=cfg.table_top_z)
            z_sh_min = z_shell.amin(dim=(1, 2))                         # (N,)
            arm_table_pen = (cfg.table_top_z + cfg.arm_shell_margin - z_sh_min
                             ).clamp(min=0.0, max=0.05).square()
            arm_gap = z_sh_min - cfg.table_top_z                        # 诊断: 外壳离桌
        else:
            arm_table_pen = (cfg.table_top_z + cfg.arm_table_margin - z_arm
                             ).clamp(min=0.0, max=0.05).square().sum(dim=1)
            arm_gap = (z_arm - cfg.table_top_z).amin(dim=1)             # 诊断: 原点离桌
        self_d = torch.cdist(arm_p, bp[:, self.self_bids]).amin(dim=2)  # (N,A) 到最近躯干件
        self_gap = self_d.amin(dim=1)                                   # 诊断
        self_pen = (cfg.self_margin - self_d).clamp(min=0.0).square().sum(dim=1) \
            .clamp(max=cfg.self_pen_cap)

        # ---- 接近段: 参考时钟推进 + **纯几何**切换 (无强制切换; 到不了就 timeout) ----
        ph = self.task_phase
        if getattr(self, "_grasp_quat_w", None) is not None:
            d_pos, d_rot, _ = self._align_err()
        else:                       # 无 prior (设定 B): 没有抓姿终点, 位置退化到合拢中心→目标
            d_pos = (self._anchor_w() - self._target_w()).norm(dim=1)
            d_rot = torch.zeros_like(d_pos)
        if cfg.approach:
            in_app = ph == Phase.PREGRASP
            # 参考时钟是**外生**的: 每步 +1, 与机器人在哪、做得好不好无关 (DESIGN_LOOP A4)
            # place: 搬运段时钟同样外生推进 (gs -> re), 播放人手搬运剖面
            _adv = (in_app | (ph == Phase.TRANSPORT)) & active
            self.ref_t = torch.where(_adv, self.ref_t + 1, self.ref_t)
            if getattr(cfg, "fin_prog_gate", False) and \
                    getattr(self, "_fin_ref_path", None) is not None and \
                    getattr(self, "_fcd_ref_last", None) is not None:
                # ★ 2026-08-23: GRASP 相位 ref_t 原本**不推进** ⟹ 直抓桶的指参考被
                # 永久钉在合拢段起始行(弯根/张开), 奖励一直要求"张开手"而任务是
                # "合拢" —— form_pot 系统性为负、fin_track 持续罚 的真根因。
                # 改: 抓取相位按"跟上才前进"推进指参考走完 close→squeeze
                # (跟不上就等, 顺带治 stride2 快播导致的 qvel_hard 爆表)
                _qf3 = self.hand.data.joint_pos[:, self.hand_jids]
                _err3 = (_qf3 - self._fcd_ref_last).abs().mean(dim=1)
                _advg = ((ph == Phase.GRASP) & active
                         & (_err3 < np.radians(float(cfg.fin_adv_tol_deg)))
                         & (self.ref_t < int(cfg.fin_end_row)))
                self.ref_t = torch.where(_advg, self.ref_t + 1, self.ref_t)
                if getattr(self, "_dbg_adv", None) is not None:      # 探针
                    _ing = (ph == Phase.GRASP) & active
                    if bool(_ing.any()):
                        self._dbg_adv[0] += float(_ing.float().sum())
                        self._dbg_adv[1] += float(_advg.float().sum())
                        self._dbg_adv[2] += float(torch.rad2deg(_err3[_ing]).sum())
                        self._dbg_adv[3] = max(self._dbg_adv[3],
                                               float(torch.rad2deg(_err3[_ing]).max()))
                        self._dbg_adv[4] += float(((self.ref_t >= int(cfg.fin_end_row)) & _ing).float().sum())
            ok = in_app & active & (d_pos < cfg.eps_pos) & (d_rot < cfg.eps_rot) & \
                (self.wrist_linvel_w.norm(dim=1) < cfg.switch_vel_max)
            self.switch_run = torch.where(ok, self.switch_run + 1,
                                          torch.zeros_like(self.switch_run))
            to_grasp = in_app & (self.switch_run >= cfg.switch_hold)
            newly_arrive = to_grasp & ~self.arrived
            if getattr(cfg, "approach_only", False):
                # Approach-only: 到位就是**成功并终止**, 不切进 GRASP 相位。
                # (用户 2026-08-16 定: 判据 = 腕位置差<3cm 且 朝向差<15° 且保持 2 步;
                #  手指全程张开 —— 合拢是下一阶段的事, 接近途中合拢会握着拳头撞物体。)
                self.arrived |= to_grasp
                to_grasp = torch.zeros_like(to_grasp)
                if getattr(cfg, "pregrasp_phase2", False) and \
                        getattr(self, "_g2_pos_w", None) is not None:
                    # Phase2-RL 第二段判据: 闩后 RL 自走, 腕对**真抓姿** 1cm/15°
                    # 且 22 关节平均偏离抓姿指型 < phase2_fin_eps, 同时保持
                    # phase2_hold 步 ⟹ g2_done (成功=双侧 g2, 见 bimanual)。
                    _og = self.scene.env_origins
                    _d2 = ((self.wrist_pos_w - _og) - self._g2_pos_w).norm(dim=1)
                    _qw2 = self._qsign(self.wrist_quat_w)
                    _dq2 = quat_mul(self._g2_quat_w.expand(self.num_envs, 4),
                                    quat_conjugate(_qw2))
                    _a2 = 2.0 * torch.acos(
                        self._qsign(_dq2)[:, 0].abs().clamp(max=1.0))
                    if getattr(cfg, "fin_cart", False):
                        # FC: 逐指指垫到"自己的 GraspPose 位置" (物体系目标随物体走)
                        _tp = (self.hand.data.body_pos_w[:, self.tip_ids]
                               - _og.unsqueeze(1))
                        _oq_c = self.object.data.root_quat_w
                        _tw = quat_apply(
                            _oq_c.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4),
                            self._fc_local.unsqueeze(0).expand(self.num_envs, -1, -1
                                                               ).reshape(-1, 3)
                        ).view(self.num_envs, 5, 3) \
                            + (self.object.data.root_pos_w - _og).unsqueeze(1)
                        self._fc_d = (_tp - _tw).norm(dim=2)                # (N,5)
                        _fmask = self.finger_active > 0.5
                        _fe2 = self._fc_d[:, _fmask].max(dim=1).values      # 最大参与指距 (m)
                        _fin_ok = _fe2 < cfg.fin_cart_tol
                    else:
                        _fe2 = (self.finger_q - self._p2_fin.unsqueeze(0)
                                ).abs().mean(dim=1)
                        _fin_ok = _fe2 < cfg.phase2_fin_eps
                    self._g2_d, self._g2_fe = _d2, _fe2          # 奖励/诊断复用
                    self._g2_gates = {                            # 逐闸诊断
                        "arrived": self.arrived,
                        "腕位<eps": (_d2 < cfg.eps_pos),
                        "腕转<eps": (_a2 < cfg.eps_rot),
                        "指<tol": _fin_ok,
                    }
                    self._g2_a = _a2
                    _ok2 = self.arrived & (_d2 < cfg.eps_pos) & \
                        (_a2 < cfg.eps_rot) & _fin_ok
                    self._g2_run = torch.where(
                        _ok2, self._g2_run + 1, torch.zeros_like(self._g2_run))
                    self._g2_done = self._g2_done | \
                        (self._g2_run >= int(cfg.phase2_hold))
            if to_grasp.any():
                # 偏差带改以"到达位形"为中心, 否则切换会把目标拽回 prior 位形
                self._set_arm_center(to_grasp, self.q_cmd)
                ph = torch.where(to_grasp, torch.full_like(ph, Phase.GRASP), ph)
                self.arrive_step = torch.where(newly_arrive, self.ref_t, self.arrive_step)
                self.arrive_step_abs = torch.where(
                    newly_arrive, self.episode_length_buf, self.arrive_step_abs)
                self.arrived |= to_grasp
        else:
            newly_arrive = torch.zeros_like(active)

        # ---- 候选抓取 (GRASP 段瞬时判据) ----
        cand_ok = active & (ph == Phase.GRASP) & (n_pads >= cfg.success_min_pads) & \
            (cent >= cfg.grasp_centrip_thresh) & (rel_spd < cfg.grip_slip_vel) & \
            (obj_rot < cfg.grip_rot_max) & (disp < cfg.push_fail_dist)
        if getattr(cfg, "upright_hold", False):
            # 姿态保持判据 (2026-08-06): 候选必须在物体基本竖直的状态下形成
            cand_ok = cand_ok & (tilt_deg < cfg.tilt_succ_max_deg)
        # 逐闸诊断 (预检用): 哪一条把 candidate 卡死
        self._cand_dbg = {
            "active": active, "相位GRASP": (ph == Phase.GRASP),
            f"垫≥{int(cfg.success_min_pads)}": (n_pads >= cfg.success_min_pads),
            "向心": (cent >= cfg.grasp_centrip_thresh),
            "滑移": (rel_spd < cfg.grip_slip_vel),
            "物转": (obj_rot < cfg.grip_rot_max),
            "位移": (disp < cfg.push_fail_dist),
        }
        if getattr(cfg, "upright_hold", False):
            self._cand_dbg["倾角"] = (tilt_deg < cfg.tilt_succ_max_deg)
        self._cand_dbg["★合闸"] = cand_ok
        # ---- 握力信任标量 g: ④ 捏紧段慢档 (2026-08-25) ----
        #   语义 = "对这个抓握的信任度"。**不发任何握力奖励**(用户裁定: 捏紧由参考
        #   自身 squeeze + g2 指尖目标点完成), g 的唯一职责是驱动 C 衰减。
        #   ⚠ 诚实标注: ④ 段没有外力, 滑移本来就 ≈0 ⟹ 这一档实质是**按时间累积**,
        #   属**弱证据**, 所以只给 grip_g_slow_frac(1/4) 速率。真正的强证据("边搬边倒
        #   还不滑")在交互段, 由 tasks/pour 侧的快档推进 (RL_Pour)。
        #   升慢降快: 信任慢慢建立, 一次打滑就掉回去。
        if getattr(cfg, "grip_g", False) and int(getattr(cfg, "squeeze_row0", 0)) > 0:
            # ★★ 2026-08-26 修: 慢档必须**只在交互开始前**生效。
            #   原来只有下界 (ref_t >= squeeze_row0), 在 e2e 的长参考里过了 squeeze 段
            #   之后会一路跑到 carry/pour 全程 —— 与 tasks/pour 侧的快档**同时**推
            #   _grip_g, 双重计数, 信任度虚高。
            #   分工契约: 慢档(弱证据) = 交互前; 快档(强证据) = 交互后, 由 pour 侧管。
            _st_g = getattr(self, "_c_started", None)
            _pre_g = (~_st_g) if _st_g is not None else torch.ones_like(active)
            _insq_g = active & (self.ref_t >= int(cfg.squeeze_row0)) & _pre_g
            # 腕系下的物体位置 (搬运时物体在世界系本来就动, 必须用腕系)
            _relp = quat_apply(quat_conjugate(self._qsign(self.wrist_quat_w)),
                               self.object.data.root_pos_w - self.wrist_pos_w)
            # 基线快照: "抓形已成"那一刻 (首次进 squeeze 段) 各 env 自己拍
            _new_g = _insq_g & ~self._grip_has
            if bool(_new_g.any()):
                self._grip_ref_p[_new_g] = _relp[_new_g]
                self._grip_has |= _new_g
            _slip = ((_relp - self._grip_ref_p).norm(dim=1) * 100.0
                     > float(cfg.grip_g_slip_cm)) & self._grip_has
            _up = float(cfg.grip_g_up) * float(cfg.grip_g_slow_frac)
            _dn = float(cfg.grip_g_up) * float(cfg.grip_g_down_mult)
            self._grip_g = torch.where(
                _slip, self._grip_g - _dn,
                torch.where(_insq_g & self._grip_has, self._grip_g + _up,
                            self._grip_g)).clamp(0.0, 1.0)

        self.cand_run = torch.where(cand_ok, self.cand_run + 1,
                                    torch.zeros_like(self.cand_run))
        to_verify = self.cand_run >= cfg.candidate_hold_steps
        newly_cand = to_verify & ~self.got_candidate
        self.got_candidate |= to_verify
        ph = torch.where(to_verify & (ph == Phase.GRASP),
                         torch.full_like(ph, Phase.LIFT), ph)

        # ---- 微抬升验证 (硬编码斜坡在 _pre_physics_step 叠加) ----
        in_verify = ph == Phase.LIFT
        self.verify_k = torch.where(in_verify, self.verify_k + 1,
                                    torch.zeros_like(self.verify_k))
        # 验证段诊断采集 (只读, 不参与任何判据)
        _wz = self.wrist_pos_w[:, 2] - origins[:, 2]
        _entering = in_verify & (self.verify_k == 1)
        self.verify_wz0 = torch.where(_entering, _wz, self.verify_wz0)
        self.verify_oz0 = torch.where(_entering, obj_pos[:, 2], self.verify_oz0)
        if cfg.verify_mode == "twist":
            # 微拧验证: 记录进入验证时的螺旋角, 判据看 screw_angle 的**增量**
            # (verify_ang0 已在 __init__ 预创建 —— 双臂快照需要, 别改回懒创建)
            self.verify_ang0 = torch.where(_entering, self.screw_angle,
                                           self.verify_ang0)
        _at_top = in_verify & (self.verify_k == cfg.verify_ramp_steps)
        if _at_top.any():
            self.vf_wrist_mm = torch.where(_at_top, (_wz - self.verify_wz0) * 1000,
                                           self.vf_wrist_mm)
            self.vf_obj_mm = torch.where(_at_top, (obj_pos[:, 2] - self.verify_oz0) * 1000,
                                         self.vf_obj_mm)
            self.vf_has |= _at_top

        ramp_done = self.verify_k >= cfg.verify_ramp_steps
        if cfg.verify_mode == "twist":
            # 盖被螺旋单向投影钉死 ⟹ 抬升判据结构性不可达 (G3_8_5 教训), 改判
            # "腕旋转时 screw_angle 跟进" = 握持传扭矩. rel_spd 判据必须去掉:
            # 微拧时腕在动而盖线速度≈0, 相对速度 ~16cm/s 是**设计内**的.
            twist_prog = self.screw_angle - self.verify_ang0
            vf_ok = in_verify & ramp_done & \
                (twist_prog >= np.radians(cfg.verify_min_twist_deg)) & \
                (n_pads >= cfg.verify_min_pads)
        else:
            vf_ok = in_verify & ramp_done & (rise >= cfg.verify_min_rise) & \
                (n_pads >= cfg.verify_min_pads) & (rel_spd < 2 * cfg.grip_slip_vel)
        if getattr(cfg, "upright_hold", False):
            # 姿态保持判据: 验证通过要求全程竖直 (s21 实测倾 17~19° 是在验证段
            # 累积到峰值的 —— 判据不含姿态就会奖励歪抓)
            vf_ok = vf_ok & (tilt_deg < cfg.tilt_succ_max_deg)
        self.verify_ok_run = torch.where(vf_ok, self.verify_ok_run + 1,
                                         torch.zeros_like(self.verify_ok_run))
        success = self.verify_ok_run >= cfg.verify_hold_steps
        if cfg.place_task:
            # ---- §2.19: 验证通过不再=成功, 而是进搬运; 成功=放置达标 ----
            to_carry = success & (ph == Phase.LIFT)
            if to_carry.any():
                self.ref_t[to_carry] = self.gs          # 搬运时钟从抓取帧起播
                # ⚠ ref_q_prev 必须同步跳到 gs —— 接近分支到达时 ref_t 停在 ~41,
                # 不同步的话下一步前馈 = 几十帧关节差一次打出 (瞬间猛甩)
                self.ref_q_prev[to_carry] = self.q_ref[self.gs]
                self.carry_anchor[to_carry] = obj_pos[to_carry]
                self.place_target[to_carry] = obj_pos[to_carry] + self.carry_delta[self.re]
                self.carry_prev_d[to_carry] = 0.0       # 进入时物体即在目标上 (delta=0)
                self.verify_k[to_carry] = 0
                self.verify_ok_run[to_carry] = 0
                ph = torch.where(to_carry, torch.full_like(ph, Phase.TRANSPORT), ph)
            in_carry_d = ph == Phase.TRANSPORT
            to_place = in_carry_d & (self.ref_t >= self.re)
            if to_place.any():
                # 带子重定心到当前位形 (同切换路径的理由): 放置段回到带内微调模式
                self._set_arm_center(to_place, self.q_cmd)
                ph = torch.where(to_place, torch.full_like(ph, Phase.PLACE), ph)
            in_place_d = ph == Phase.PLACE
            released = in_place_d & (self.closure <= 0.05)
            _near = (obj_pos - self.place_target).norm(dim=1) < cfg.place_tol
            _still = obj_spd < 0.05
            _ok_p = released & _near & _still
            self.settle_ctr = torch.where(_ok_p, self.settle_ctr + 1,
                                          torch.zeros_like(self.settle_ctr))
            success = self.settle_ctr >= cfg.place_settle_steps
        # ★★ 2026-08-24 用户裁定 B: approach_only 下 candidate→Phase.GRASP→LIFT
        #   →微抬升验证 这条链**结构性跑不起来** —— to_grasp 在 approach_only 里被
        #   显式清零, 相位永远到不了 GRASP (零动作实测 相位GRASP=0.0%,
        #   candidate 合闸 0/25600)。于是 fin_ref_track 走 else 分支 ⟹ success 恒 0,
        #   终局奖金一分不发、课程不推进、eval 恒 0.00% (历史多代皆亡于此)。
        #   2026-08-20 的"FCD 真抓稳后才算成功"裁定假设了那条链是通的; 在
        #   approach_only 下不成立。改: phase2 开着时一律走 arrived & g2_done
        #   (g2 = 腕到真抓姿 + 五指到位, 已于同日修好, 零动作 100% 命中)。
        #   分工: g2 管"GraspPose 摆没摆准", squeeze 段奖励管"抓得紧不紧"。
        if getattr(cfg, "approach_only", False) and \
                (not getattr(cfg, "fin_ref_track", False)
                 or getattr(cfg, "pregrasp_phase2", False)):
            # Approach-only: 成功 = 到位 (self.arrived 在上面的相位段里被置位)。
            # ⚠ 必须在这里覆盖 —— newly_success / self.succeeded 紧接着就用它,
            #   放到下面 terminated 那行再改就晚了 (第一版写错在那儿)。
            # ★ FC-D (fin_ref_track) 不走这条覆盖: 终点不预给 (2026-08-20 用户裁定
            #   "FCD是真抓稳后算成功终止"), 成功沿用上面的 candidate→微抬升验证链,
            #   而不是"到位+指尖贴近模板 GraspPose"(那正是被裁定废弃的预给终点)。
            success = self.arrived.clone()
            if getattr(cfg, "pregrasp_phase2", False) and \
                    getattr(self, "_g2_done", None) is not None:
                # Phase2-RL: 成功从"到位"推迟到"g2 达成"(腕到真抓姿+指型贴近, 保持)
                success = success & self._g2_done
        newly_success = success & ~self.succeeded
        self.succeeded |= success
        # 验证失败: ① 斜坡到顶+3 步物体没跟上来 (rise<3mm); ② **尝试预算耗尽**
        # (v2.8): rise 落在 3~5mm 的"半滑"灰区既不成功也不失败, 会挂到 LIFT 超时
        # 把回合截断 —— 实测 86% 候选 × 58 步回合 × 86% 超时全耗在这. 预算 =
        # 斜坡+保持+5 步, 到时未成功一律判负退回重试, 验证段不存在无判决状态.
        # **不终止**: 退回 GRASP 重试 (v2.3). 候选计数清零, 重新攒资格.
        if cfg.verify_mode == "twist":
            _no_follow = ((self.screw_angle - self.verify_ang0)
                          < np.radians(cfg.verify_fail_twist_deg))
        else:
            _no_follow = rise < cfg.verify_fail_rise
        verify_fail = in_verify & (
            ((self.verify_k >= cfg.verify_ramp_steps + 3) & _no_follow)
            | (self.verify_k >= cfg.verify_ramp_steps + cfg.verify_hold_steps + 5))
        verify_fail = verify_fail & ~success
        if verify_fail.any():
            ph = torch.where(verify_fail, torch.full_like(ph, Phase.GRASP), ph)
            self.verify_k[verify_fail] = 0
            self.verify_ok_run[verify_fail] = 0
            self.cand_run[verify_fail] = 0

        self.phase_step = torch.where(ph != self.task_phase,
                                      torch.zeros_like(self.phase_step),
                                      self.phase_step + active.long())
        self.task_phase = ph

        # ---- 失败 ----
        fell = active & (obj_pos[:, 2] < cfg.table_top_z - cfg.fall_below)
        _maxh = 0.35 if cfg.place_task else cfg.max_obj_height   # 搬运会抬高物体
        thrown = active & (obj_pos[:, 2] > cfg.table_top_z + _maxh)
        # 推走终止距离随笨拙课程收紧: gentle 0.2 -> 10cm (学习期宽容), 1.0 -> 4cm
        g01 = (self.gentle - 0.2) / 0.8
        push_dist = cfg.push_fail_dist_loose + \
            (cfg.push_fail_dist - cfg.push_fail_dist_loose) * g01
        pushed = active & ~self.got_candidate & (disp > push_dist)
        # 撞倒判负 (2026-08-18 用户裁定): 倾角超限直接终止 —— fell/pushed 都看不见
        # "躺在桌上的瓶子" (高度没掉够, 位移可能没超), PGA ep200 录像实锤该漏洞:
        # 右手 8 秒内把瓶子拍倒, 回合继续白跑。
        toppled = active & (tilt_deg > float(getattr(cfg, "fail_tilt_deg", 60.0)))
        if (getattr(cfg, "pour_e2e", False) or getattr(cfg, "pours_v6", False)) \
                and getattr(self, "_c_started", None) is not None:
            # e2e/v6: 进度时钟启动后倾倒是任务本身, 拍倒判负只管非交互段。
            # v6 统一哲学 (2026-08-21 用户): 物体位姿约束只在非交互段;
            # 交互段脱手风险由滑移/偏轨/掉落兜底
            toppled = toppled & ~self._c_started
        if getattr(cfg, "pours_v6", False):
            # v6: thrown(绝对高度) 退役 —— 换 carry_env 的"偏离自身参考>30cm
            # 无条件重置"(dev_reset_m), 判据从"离桌多高"变成"离该在的位置多远"
            thrown = thrown & False
        if getattr(cfg, "pregrasp29", False):
            # 1cm PreGrasp (2026-08-18晚): 近场(<3cm 或已闩)轻推不判死 ——
            # 最后一厘米的接近和合拢必然碰物; fell/thrown 照旧判死
            pushed = pushed & ~(self.arrived | (d_pos < float(getattr(cfg, 'near_exempt_m', 0.03))))
        off = (self.arm_q - self.arm_tgt).abs().max(dim=1).values >= cfg.term_arm_err
        self.arm_err_ctr = torch.where(off, self.arm_err_ctr + 1,
                                       torch.zeros_like(self.arm_err_ctr))
        stuck = self.arm_err_ctr >= cfg.term_arm_steps

        # push_terminate=False: 保留 pushed 的**惩罚**(见 terms["fail"]), 但不终止回合
        _pt = pushed if getattr(cfg, "push_terminate", True) else torch.zeros_like(pushed)
        if getattr(cfg, "approach_only", False):
            # ★ 硬底线(用户 2026-08-16 定): 这两件**直接终止**, 不只是罚
            #   ① 臂外壳穿桌 —— 必须外壳口径。人手轨迹上右臂外壳穿桌 −1.76cm 时
            #      手部连杆还有 ≥2cm, 原点口径**根本看不见**(l5/l6 截面半径 4~6.7cm)。
            #   ② 接近段手碰到物体 —— 还没到该碰的时候。
            # ⚠ 用**本函数里的局部** arm_gap: `self._sig` 是在 _get_dones **末尾**
            #   才组装的, 在这里读它第一步就 KeyError(冒烟当场抓到)。
            shell_hit = active & (arm_gap < 0.0)
            _hp = self.hand.data.body_pos_w[:, self.table_bids]
            _M = self.obj_points.shape[0]
            _ow = quat_apply(
                self.object.data.root_quat_w[:, None, :].expand(-1, _M, -1).reshape(-1, 4),
                self.obj_points[None].expand(self.num_envs, -1, -1).reshape(-1, 3)
            ).view(self.num_envs, _M, 3) + self.object.data.root_pos_w[:, None, :]
            _cd = torch.cdist(_hp, _ow)                     # (N, K_body, M_pts)
            self._shell_obj_per = _cd.amin(dim=2)           # (N,K) 逐 body 最近距离
            self._shell_obj_d = self._shell_obj_per.amin(dim=1)
            obj_hit = active & (self._shell_obj_d
                                < cfg.approach_hit_obj_m)
            if getattr(cfg, "pregrasp29", False):
                # 1cm PreGrasp (2026-08-18晚): 近场(<3cm 或已闩)接触物体是任务素材,
                # obj_hit 不判死不罚 —— RL 要学的正是"碰而不倒"; 桌面/外壳 hit 照旧,
                # fell/thrown 任何时候照旧判死
                obj_hit = obj_hit & ~(self.arrived | (d_pos < float(getattr(cfg, 'near_exempt_m', 0.03))))
            self._approach_hit = shell_hit | obj_hit
        else:
            self._approach_hit = torch.zeros_like(active)
        _succ_t = success if not getattr(cfg, "success_nonterminal", False) \
            else torch.zeros_like(success)   # e2e: 真抓稳是门不是终点
        terminated = fell | thrown | toppled | _pt | stuck | table_crash | _succ_t \
            | self._approach_hit
        to_t = self.phase_timeout_t[ph]
        timeout = active & (to_t > 0) & (self.phase_step >= to_t)
        truncated = timeout | (self.episode_length_buf >= self.ep_total - 1)

        # ---- 接触分数图: 在 verify 斜坡到顶那一刻取快照 (物理干预 = 因果筛子) ----
        if self.score_on:
            snap = in_verify & (self.verify_k == cfg.verify_ramp_steps)
            if snap.any():
                share = mag / mag.sum(dim=1, keepdim=True).clamp(min=1e-6)   # 力占比
                w = share * pads_on.float()
                self.pend_idx[snap] = pad_i[snap]
                self.pend_w[snap] = w[snap]

        if getattr(self, "_sp_q", None) is not None:
            # 与到位判据同源的位置口径 (腕锚点 -> GraspPose 靶点)
            self._sp_track((self._anchor_w() - self._target_w()).norm(dim=1))
        self._sig = dict(active=active, obj_pos=obj_pos, obj_spd=obj_spd,
                         tilt_deg=tilt_deg, tilt_max=self.tilt_max_deg,
                         obj_rot=obj_rot, pad_d=pad_d, pad_d_shape=pad_d_shape,
                         mag=mag, e_pad=e_pad,
                         cent=cent, r_imb=r_imb, tau_n=tau_n, quality=quality,
                         pads_on=pads_on, n_pads=n_pads, newly_touch=newly_touch,
                         table_pen=table_pen, cross_pen=cross_pen, disp=disp,
                         rise=rise, cand_ok=cand_ok, newly_cand=newly_cand,
                         newly_success=newly_success, verify_fail=verify_fail,
                         fell=fell, thrown=thrown, toppled=toppled,
                         pushed=pushed, stuck=stuck,
                         succ_t=_succ_t, approach_hit=self._approach_hit,
                         table_crash=table_crash, timeout=timeout,
                         d_pos=d_pos, d_rot=d_rot,
                         arm_table_pen=arm_table_pen, self_pen=self_pen,
                         arm_gap=arm_gap, self_gap=self_gap,
                         shell_obj_d=getattr(self, "_shell_obj_d",
                                             torch.full_like(disp, 9.0)),
                         newly_arrive=newly_arrive)
        return terminated, truncated

    # ---- 奖励 ---------------------------------------------------------
    def apply_reward_schedule(self, terms, ph=None):
        """相位奖励日程表 + C 衰减 —— **抽成公共方法供各体制复用**。

        ★ 2026-08-26 第六犯: `PourCarryEnv._get_rewards` **完全不调 super()**,
        于是写进基座 `_get_rewards` 的一切(日程表 / C 衰减 / _srl 账本)对 carry
        体制**隐形** —— 横幅照打、旗照接、预检照绿, 没有任何信号提示"这段不会执行"。
        铁证: r5 的 TB 里 `ep_rew/*` 标签数 = 0(F 线满屏), E2E 接近段一直在**无薪训练**。
        抽成公共方法后, carry 侧在自己的 `_get_rewards` 末尾调一次即可。

        参数
        ----
        terms : dict[str, Tensor]   奖励项字典(原地修改)
        ph    : Tensor              当前相位 (N,)
        """
        cfg = self.cfg
        # ★ 相位取自 self.task_phase —— 原来这段写在 `_get_rewards` 里直接引用局部
        #   变量 `ph`, 而那个方法里**没有** ph。因为整段被 `if cfg.rew_sched:` 包着、
        #   而我从没带 --rew_sched 跑过, 这个 NameError 藏了整整一轮:
        #   **只要有人真开这面旗就会当场炸**。抽成方法后调用变无条件, 立刻自曝。
        #   教训: "写了但从没在开启状态下跑过"的代码 = 没写。
        if ph is None:
            ph = self.task_phase
        # ---- 相位奖励日程表 (2026-08-26 定稿 ①②③ 行) ----
        #   PREGRASP: 抓取期罚组置零 (F 线靠 approach_only 整体移除, 这里按相位做)
        #   进 GRASP : 该组**线性渐入** rew_sched_ramp 步 —— 渐入而非硬切, 否则
        #             边界断崖会让策略学"躲门"(实测 arrive_rate 0.279→0.016)
        #   GRASP 段 : 接近组按 rew_sched_grasp_scale 降权
        if getattr(cfg, "rew_sched", False):
            _ing = (ph == Phase.GRASP)
            _rmp = int(getattr(cfg, "rew_sched_ramp", 0))
            if _rmp > 0:
                _since = (self.episode_length_buf - self.arrive_step_abs).clamp(min=0)
                _w_pen = (_since.float() / _rmp).clamp(0.0, 1.0) * _ing.float()
            else:
                _w_pen = _ing.float()
            for _kP in getattr(cfg, "rew_sched_pre", ()):
                if _kP in terms:
                    terms[_kP] = terms[_kP] * _w_pen
            for _kG, _sc in (getattr(cfg, "rew_sched_grasp_scale", {}) or {}).items():
                if _kG in terms:
                    terms[_kG] = terms[_kG] * (1.0 - (1.0 - float(_sc)) * _ing.float())

        # ---- C 方案 (2026-08-25 用户裁定): 同一个 g 驱动前段密集项衰减 ----
        #   "一个标量管两件事, 不会出现握力已进保持模式而接近分还在付钱的错位"。
        #   ⚠ 只衰减**逐步密集**项; one-shot(arrive/里程碑) 不碰 —— 衰减它们
        #   等于改判据而不是改塑形。
        if getattr(cfg, "grip_g", False):
            _g1 = (1.0 - self._grip_g).clamp(0.0, 1.0)
            for _kD in getattr(cfg, "grip_g_decay_terms", ()):
                if _kD in terms:
                    terms[_kD] = terms[_kD] * _g1

    def _get_rewards(self, return_terms: bool = False):
        """`return_terms=True` 时**只返回 terms 字典**, 不求和、不记账。

        ★ 2026-08-26 (第六犯的解法): `PourCarryEnv._get_rewards` 完全不调 super,
        于是整个 approach/grasp **收入面**在 carry/E2E 体制下从不存在
        (铁证: r5 的 TB 里 `ep_rew/*` 标签数 = 0) —— 接近段一直在**无薪训练**,
        连 `terms["fail"]` 的 −10 也在死代码里 ⟹ **自杀历来 0 元**,
        r5/r6 的"出口经济学"按"死免费"重算才自洽。

        为什么用"加一个出口"而不是"把 470 行拆成 _build_terms":
        拆分要重排大量局部状态(prev_*/done 位/诊断累加), 风险远高于收益。
        这里只加一个早返回, **调用方拿到的是同一份 terms**, 所有 prev_*/done 位
        照常在本方法内更新 —— 这正是"整块接、不挑子集"的前提。
        """
        cfg, s = self.cfg, self._sig
        active = s["active"]
        af = active.float()

        newly = active & (self.episode_length_buf == cfg.settle_steps)
        if newly.any():
            self.prev_pad_d[newly] = s["pad_d_shape"][newly]
            self.prev_quality[newly] = s["quality"][newly]

        terms = {}
        # ---- 接近段专属两项 (只在 PREGRASP 相位发/罚) ----------------------
        if cfg.approach:
            in_app = (self.task_phase == Phase.PREGRASP).float()
            # ① 对齐势差分. Φ = −(d_pos + λ_rot·d_rot), λ_rot=0.174 是 P0.0 实测
            #    (5° ≡ 1.52cm), 是纯运动学力臂的 2 倍.
            #    ⚠ w_align **必须恒定**: 权重随相位变会破坏 telescoping,
            #      策略可以"早期后退-晚期前进"无限刷分 (DESIGN_LOOP A3).
            phi_pot = -(s["d_pos"] + cfg.lam_rot * s["d_rot"])
            if newly.any():
                self.prev_phi[newly] = phi_pot[newly]
            terms["align"] = cfg.w_align * in_app * \
                (phi_pot - self.prev_phi).clamp(-cfg.cap_align, cfg.cap_align)
            self.prev_phi = torch.where(active, phi_pot, self.prev_phi)
            # ② 模仿罚 —— 罚的是"本步残差用量", 不是"离参考位置多远".
            #    因为 q_cmd += 参考增量 + 残差, 所以 手这步走的 − 参考这步走的 ≡ 残差.
            #    人手轨迹被系统性平移了 14~16cm (PLAN §7.7), 位置口径会逼策略去修正
            #    那 16cm 从而与对齐项打架; 增量口径与该常数偏移完全无关.
            #    量纲: **末端米**, 不是关节弧度 —— 直接量"实际走的 − 参考走的".
            # w_imit_ramp 是**训练期**系数 (0→1, 由 arrive_rate 驱动); (1−φ)^p 是**回合内**形状.
            # 两条轴分开: 先让它到得了(ramp≈0), 再要求像人(ramp→1).
            _w0 = (cfg.w_imit0_approach if getattr(cfg, "approach_only", False)
                   else cfg.w_imit0)
            w_imit = (_w0 * cfg.w_imit_ramp
                      * (1.0 - self._phi()).clamp(min=0.0) ** cfg.imit_decay_p)
            org = self.scene.env_origins
            w_now = self.wrist_pos_w - org
            t = self.ref_t.clamp(max=self.gs)
            d_ref = self.ref_wrist_pos[t] - self.ref_wrist_pos[(t - 1).clamp(min=0)]
            step_dev = ((w_now - self.prev_wrist_pos) - d_ref).norm(dim=1)
            step_dev = torch.where(self.prev_valid, step_dev, torch.zeros_like(step_dev))
            # 诊断量只统计接近段 (抓取段手不动而参考仍在走, 差值无意义)
            self.res_step_cm = step_dev * 100.0 * in_app
            terms["imit"] = -w_imit * in_app * step_dev.clamp(max=cfg.cap_imit)
            if getattr(cfg, "l5_couple", False):
                # L5 软指标(裁定②): 罚 |closure − c_ref(d)|, 引导"边靠近边合指"的
                # 节奏而不锁死力度 —— 权重刻意小, 违背它换取更好抓形是允许的。
                terms["fin_couple"] = -cfg.l5_couple_w * \
                    (self.closure - self._l5_cref).abs()
            if getattr(cfg, "pregrasp29", False) and \
                    getattr(self, "_fin_act_mag", None) is not None:
                # PreGrasp29 (2026-08-18晚): 稳定到位**之前**鼓励手指别动 ——
                # 软抑制(罚动作幅度)而非锁死; 到位(arrived)后不罚。
                _nrx = 1.0
                if float(getattr(cfg, "fin_near_relax", 0.0)) > 0.0 \
                        and "d_pos" in s:
                    # 消融2 (2026-08-22): 最后 fin_near_m 内指拘束松绑 —— 近场
                    # 手指本该为绕障变形, 拘束费压过到位收益=躺平 (左手账本实锤)
                    _nrx = torch.where(
                        s["d_pos"] < float(getattr(cfg, "fin_near_m", 0.03)),
                        torch.full_like(s["d_pos"],
                                        float(cfg.fin_near_relax)),
                        torch.ones_like(s["d_pos"]))
                object.__setattr__(self, "_fin_nrx", _nrx)
                terms["fin_quiet"] = -cfg.w_fin_quiet * self._fin_act_mag * \
                    (~self.arrived).float() * _nrx
            if getattr(cfg, "pregrasp_phase2", False) and \
                    getattr(self, "_g2_d", None) is not None:
                # Phase2-RL 离散里程碑 (每回合各一次): m1 腕进真抓姿 m1 门;
                # m2 指型进 20°。成功大奖走现有 newly_success 通道。
                # m1 门默认**派生** (简化一): palm_cm + 2×eps_pos ⟹ 必然罩住
                # "到位球+闩后漂移", 到位的手一定拿得到 (零余量事故结构性绝迹)。
                _m1_cfg = getattr(cfg, "phase2_m1_cm", None)
                _m1cm = (float(_m1_cfg) / 100.0 if _m1_cfg is not None else
                         float(getattr(cfg, "pregrasp_palm_cm", 1.0)) / 100.0
                         + 2.0 * float(cfg.eps_pos))
                _n1 = self.arrived & (self._g2_d < _m1cm) & ~self._m1_done
                if os.environ.get("RL_DBG_M1"):
                    print(f"[m1-dbg] {self.ee_body}: arrived {int(self.arrived.sum())}"
                          f" | _g2_d min {float(self._g2_d.min())*100:.2f}cm"
                          f" 中位 {float(self._g2_d.median())*100:.2f}cm"
                          f" | m1新发 {int(_n1.sum())} 已发 {int(self._m1_done.sum())}"
                          f" | 门 {_m1cm*100:.1f}cm", flush=True)
                # m2 派生阶梯 (2026-08-19 用户裁定 C): 糖从"起点附近"一路挂到 20° 大门。
                # 病根: 写死 20° 单门没罩住起始指型偏差 (右手/瓶 29.5° 起步差 9.5°,
                # 13M 步探索撞不进门, m2 恒零; 左手 23.7° 起步只差 3.7° 所以能学)。
                # 阶梯在构造时按各侧起始偏差铺 (_m2_ladder, 降序, 末级=20°);
                # 中间级各 +bonus/2, 末级 +bonus, 全部一次性 (防刷分原则不破)。
                if getattr(cfg, "fin_cart", False):
                    # FC 奖励 (2026-08-19): 逐指一次性面包屑 + 新 m2(全参与指到位且
                    # 物体未扰动) + 可选势差分 (fin_pot>0, 只奖进步, 复位后首步无收入)
                    _fd = self._fc_d
                    _fm = (self.finger_active > 0.5).unsqueeze(0)
                    _newf = (self.arrived.unsqueeze(1) & (_fd < cfg.fin_cart_tol)
                             & ~self._fc_done & _fm)
                    self._fc_done = self._fc_done | _newf
                    terms["fc_crumb"] = cfg.w_fc_crumb * _newf.float().sum(dim=1)
                    _alldone = (self._fc_done | ~_fm).all(dim=1)
                    _n2 = _alldone & ~self._m2_done & (s["obj_spd"] < 0.05)
                    self._m2_done = self._m2_done | _n2
                    terms["g2_m2"] = cfg.phase2_m_bonus * _n2.float()
                    if float(getattr(cfg, "fin_pot", 0.0)) > 0:
                        _gain = (self._fc_prev - _fd).clamp(-0.02, 0.02)
                        _gain = torch.where(torch.isfinite(_gain), _gain,
                                            torch.zeros_like(_gain))
                        # FC-D: 势差分(拉向模板点)同样按稳抓能力退火 ×(1−g)
                        _potann = (1.0 - float(getattr(self, "_fcd_g", 0.0))
                                   if getattr(cfg, "fin_ref_track", False) else 1.0)
                        if getattr(cfg, "fc_pot_earn_only", False):
                            # 基线归零 (2026-08-23 用户第1条: 学参考不要搞坏):
                            # 势差分在 squeeze 段天然为负(参考故意压过 grasp 点),
                            # 于是"严格复现参考"反而挨罚 —— 实测左手 -0.046/步。
                            # earn-only ⟹ 只奖进步不罚参考。
                            _gain = _gain.clamp(min=0.0)
                        terms["fc_pot"] = (cfg.fin_pot * _potann
                                           * (_gain * _fm.float()).sum(dim=1)
                                           * self.arrived.float())
                    self._fc_prev = _fd.clone()
                    self._m1_done = self._m1_done | _n1
                    terms["g2_m1"] = cfg.phase2_m_bonus * _n1.float()
                    if getattr(cfg, "fin_ref_track", False) and \
                            getattr(self, "_fcd_keys", None) is not None:
                        # ---- FC-D 奖励组 (2026-08-20 用户裁定: 慢合拢/Pad 多触/稳抓) ----
                        _qf = self.hand.data.joint_pos[:, self.hand_jids]
                        # 指跟参考罚 (取代 fin_quiet 语义: 教"贴着参考走", 构型段起效)
                        # 参考拉力 ×(1−g): 稳抓能力起来后参考罚退火归零 (棘轮)
                        _ann = 1.0 - float(getattr(self, "_fcd_g", 0.0))
                        _ftc = 1.0
                        if getattr(cfg, "fin_track_contact_fade", False):
                            # 2026-08-22 自动巡检发现: 手指压在物体上就**再也贴不回
                            # 参考**(左手 35.7% 步有垫接触, fin_track -71.2 = 右手的
                            # 18 倍), 等于罚它抓东西。接触后按已触垫数淡出参考拉力,
                            # 交给 pad/cent/candidate 判据接管 —— 参考只管自由空间成形
                            _ftc = (1.0 - s["n_pads"].float()
                                    / max(float(self.n_active), 1.0)).clamp(0.0, 1.0)
                        terms["fin_track"] = -cfg.w_fin_track * _ann * \
                            (_qf - self._fcd_ref_last).abs().mean(dim=1) * \
                            getattr(self, "_fin_nrx", 1.0) * _ftc  # 消融2 + 接触淡出
                        if float(getattr(cfg, "fin_form_pot_w", 0.0)) > 0.0:
                            # v3.4 (2026-08-22 用户裁定): 形态**正向**势差分 ——
                            # 罚的最优解是躺平, 赚的最优解才是小心翼翼向 GraspPose
                            # 形态推进; telescoping ±0.02 封顶防刷, candidate 前生效
                            # ⚠ 前后两步必须对**同一行**参考量距离 —— 参考行自己
                            # 每步在动, 拿"距离标量"做差分会把参考的移动记成手指
                            # 的倒退 (2026-08-22 账本实锤: 左手 -9.7 全是伪信号)
                            if not hasattr(self, "_ffq_prev"):
                                object.__setattr__(self, "_ffq_prev",
                                                   _qf.detach().clone())
                            _ffd = (_qf - self._fcd_ref_last).abs().mean(dim=1)
                            _ffd_p = (self._ffq_prev
                                      - self._fcd_ref_last).abs().mean(dim=1)
                            # 只赚不罚 (2026-08-23 用户语义): 参考行在动, 跟不上时
                            # 距离变大会被记成负分 —— 左手 form_pot 恒 −18 的成因
                            _fg = (_ffd_p - _ffd).clamp(
                                0.0 if getattr(cfg, "form_pot_earn_only", False)
                                else -0.02, 0.02)
                            terms["form_pot"] = float(cfg.fin_form_pot_w) * _fg \
                                * (~self.got_candidate).float()
                            self._ffq_prev = _qf.detach().clone()
                        # 构型糖: 构型窗后指距 Pose1 < tol, 一次性 (裁定①: 学会放拇指)
                        _dev1 = (_qf - self._fcd_keys[1].unsqueeze(0)).abs().mean(dim=1)
                        _nsh = ((self.episode_length_buf
                                 >= cfg.settle_steps + cfg.fin_shape_steps)
                                & (_dev1 < np.radians(cfg.shape_tol_deg))
                                & ~self._fcd_shape_done)
                        self._fcd_shape_done = self._fcd_shape_done | _nsh
                        terms["shape_ms"] = cfg.w_shape_ms * _nsh.float()
                        # Pad 接触 (传感器逐侧镜像, filter=各自物体; 轻触才发)
                        _Fp = cfg.pad_force_sign * torch.cat(
                            [s_.data.force_matrix_w.view(self.num_envs, 1, 3)
                             for s_ in self._contact_sensors], dim=1).nan_to_num(0.0)
                        _on = (_Fp.norm(dim=-1) > 0.5) \
                            & (self.finger_active > 0.5).unsqueeze(0)
                        _quiet = s["obj_spd"] < 0.05
                        _newp = _on & _quiet.unsqueeze(1) & ~self._fcd_pad_done
                        if getattr(cfg, "pad_first_arrived_only", False):
                            # v3.3 (2026-08-22 检验裁定): 碰垫糖只在**到位后**发 ——
                            # 它曾是"不进门"的工资: L 停靠点距门 0.9mm, 却被门前
                            # 蹭垫收入拐去 2.6cm (账本 pad_first +11.7 实锤)
                            _newp = _newp & self.arrived.unsqueeze(1)
                        self._fcd_pad_done = self._fcd_pad_done | _newp
                        terms["pad_first"] = cfg.w_pad_first * _newp.float().sum(dim=1)
                        # 持续分+向心塑形: 到位后、candidate 前 (防挂机窗口)
                        _pre = self.arrived & ~self.got_candidate
                        terms["pad_hold"] = cfg.w_pad_hold \
                            * _on.float().sum(dim=1) / max(self.n_active, 1) \
                            * _pre.float() * _quiet.float()
                        _tipp = self.hand.data.body_pos_w[:, self.tip_ids]
                        _dirs = self.object.data.root_pos_w.unsqueeze(1) - _tipp
                        _dirs = _dirs / _dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        _fn = _Fp / _Fp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        _cent = ((-_fn) * _dirs).sum(dim=-1) * _on.float()
                        _centm = _cent.sum(dim=1) / _on.float().sum(dim=1).clamp(min=1.0)
                        _cs = (_centm.clamp(min=-0.5, max=1.0)
                               if getattr(cfg, "cent_signed", False)
                               else _centm.clamp(min=0.0))   # cent_fix: 负梯度
                        if getattr(cfg, "cent_earn_only", False):
                            # ★ 2026-08-23 判死 (AAGE v3 左手塌方的唯一涨项):
                            # ① 本路径的方向基准硬编码"指向物体**原点**", 没走
                            #    cfg.cent_mode="normal" —— 而 env.py:1561 早写明
                            #    环形物体原点是孔心, 从外侧捏差几十度。杯资产原点在
                            #    底部、抓的是上沿 ⟹ 左手 cent 结构性为负。
                            # ② 它是**唯一一个"碰到就每步扣分"的稠密罚**, 而
                            #    arrive/g2_m1/pad_first 都是一次性稀疏奖。梯度里
                            #    稠密负赢过稀疏正 ⟹ 策略学"别碰杯子"。
                            # 实测: 左手 cent_shape -0.0074 -> +0.0000 是塌方期间
                            # 唯一上涨项, 同期 arrive 0.0304 -> 0.0000, 净亏 0.043。
                            _cs = _cs.clamp(min=0.0)
                            self._cent_raw = _centm            # 诊断保留
                        terms["cent_shape"] = cfg.w_cent_shape * _cs * _pre.float()
                        # 稳抓大糖: 复用现成 candidate 判据 (≥4垫&向心≥0.3&保持4步)
                        if "newly_cand" in s:
                            terms["stable_ms"] = cfg.w_stable_ms * s["newly_cand"].float()
                        # succeeded 后保持稳抓 (2026-08-20 用户裁定): 单侧过验证
                        # 不躺平 —— 按垫接触份额逐步发保持分, 直到双侧都成功终止
                        terms["succ_hold"] = float(getattr(cfg, "w_succ_hold", 0.05)) \
                            * _on.float().sum(dim=1) / max(self.n_active, 1) \
                            * self.succeeded.float() * _quiet.float()
                else:
                    _lad = self._m2_ladder                      # (L,) 降序阈值 (rad)
                    _lvl = self._m2_lvl                          # (N,) 下一级台阶号
                    _Ln = _lad.shape[0]
                    _thr = _lad[_lvl.clamp(max=_Ln - 1)]
                    _n2 = self.arrived & (_lvl < _Ln) & (self._g2_fe < _thr)
                    _final = _n2 & (_lvl == _Ln - 1)
                    self._m2_lvl = _lvl + _n2.long()
                    self._m2_done = self._m2_lvl >= _Ln
                    self._m1_done = self._m1_done | _n1
                    terms["g2_m1"] = cfg.phase2_m_bonus * _n1.float()
                    terms["g2_m2"] = (cfg.phase2_m_bonus * _final.float()
                                      + 0.5 * cfg.phase2_m_bonus
                                      * (_n2 & ~_final).float())
            self.prev_wrist_pos = torch.where(active.unsqueeze(1), w_now,
                                              self.prev_wrist_pos)
            self.prev_valid = self.prev_valid | active
            # -- 方案二: 锥形信任管 (§2.16) —— 管内零罚, 出管按超出量²罚 --
            if cfg.cone_trust:
                e_dev = (w_now - self.ref_wrist_pos[t]).norm(dim=1)
                ph_c = self._phi()
                ss_c = ph_c * ph_c * (3.0 - 2.0 * ph_c)
                r_cone = cfg.cone_r0 + (self._ref_end_gap + cfg.cone_margin) * ss_c
                out_c = (e_dev - r_cone).clamp(min=0.0)
                terms["cone"] = -cfg.w_cone * in_app * out_c.square()
                self.cone_out_cm = out_c * 100.0 * in_app
            # -- PickAndPlace: 搬运段物体跟踪势差分 (§2.19) --
            # 目标(t) = 进搬运时的物体位 + 人手腕相对位移; 差分支付, 与 w_align 同数学.
            if cfg.place_task:
                in_cr = (self.task_phase == Phase.TRANSPORT).float()
                tgt_c = self.carry_anchor \
                    + self.carry_delta[self.ref_t.clamp(max=self.re)]
                d_cr = (s["obj_pos"] - tgt_c).norm(dim=1)
                terms["carry"] = cfg.w_carry * (self.carry_prev_d - d_cr) \
                    .clamp(-0.05, 0.05) * in_cr
                self.carry_prev_d = torch.where(in_cr.bool(), d_cr, self.carry_prev_d)

        # -- 密集: 逐垫接近进度 (有效接触数够了就关, 该拿质量分了; 只算参与指) --
        d_pad = ((self.prev_pad_d - s["pad_d_shape"]).clamp(-0.05, 0.05)
                 * self.finger_active.unsqueeze(0)).sum(dim=1) / float(self.n_active)
        not_app = (self.task_phase != Phase.PREGRASP).float()
        terms["pad_approach"] = cfg.w_pad_approach * d_pad * not_app * \
            (s["n_pads"] < cfg.success_min_pads).float()
        # -- 稀疏: 每垫首次有效接触 (接近段不发: 那时碰到物体是坏事, 由 push/obj_move 管) --
        # PG 任务 (2026-08-18晚): 相位永远是接近, 门改"到位闩后发" —— 落位糖:
        # 每垫首次**有效**接触(力≥f_min) +0.5 一次性, "手指落对位置"按物理裁定结算
        _pt_gate = (self.arrived.float()
                    if getattr(cfg, "pregrasp29", False) else not_app)
        terms["pad_touch"] = cfg.r_pad_touch * _pt_gate * \
            s["newly_touch"].float().sum(dim=1)
        # -- 密集: 好接触的小额持续收入, 按质量分 Q 发 (v2.4: 按 cent 发会资助
        #    "下压式伪向心"; 验证段照发, 进验证无机会成本. 见 cfg 注释) --
        terms["cent_income"] = cfg.w_cent_income * s["quality"].clamp(min=0.0)
        # -- 密集: 抓取质量**进度** (原地保持不刷分; 质量掉了扣分) --
        dq = (s["quality"] - self.prev_quality).clamp(-cfg.q_prog_clip, cfg.q_prog_clip)
        terms["quality_prog"] = cfg.w_quality * dq
        # -- 年金: 候选判据完整成立时发 (成功终止天然封顶) --
        terms["hold"] = cfg.w_hold * s["cand_ok"].float()
        # -- 超力罚 (×gentle 笨拙课程: 学习期打折, 会抓了恢复原价) --
        terms["over_force"] = -cfg.w_over_force * self.gentle * \
            (s["mag"] - cfg.squeeze_f_max).clamp(min=0.0).sum(dim=1)
        # -- 惩罚: 把物体弄动 (只在 GRASP 段; 验证段物体本来就该动; ×gentle) --
        in_grasp = (self.task_phase == Phase.GRASP).float()
        touched = self.any_contact.float() * in_grasp
        terms["obj_move"] = -cfg.w_obj_move * self.gentle * \
            s["obj_spd"].clamp(max=1.0) * touched
        terms["obj_rot"] = -cfg.w_obj_rot * s["obj_rot"].clamp(max=3.0) * touched
        if getattr(cfg, "upright_hold", False):
            # 姿态保持罚 (2026-08-06): 接触后物体倾角超死区按线性罚, 单步有界.
            # 量级: 18° 全程 ≈ -1/步 × ~20 步 = -20, 与候选+成功奖同量级 (预注册 Y2)
            terms["tilt"] = -cfg.w_tilt * (
                (s["tilt_deg"] - cfg.tilt_deadband_deg) / cfg.tilt_norm_deg
            ).clamp(min=0.0, max=2.0) * self.any_contact.float()
        terms["push"] = -cfg.w_push * self.gentle * \
            s["disp"].clamp(max=0.3).square() * (~self.got_candidate).float()
        # -- 惩罚: 碰撞 --
        terms["table"] = -cfg.w_table * s["table_pen"]
        terms["finger_cross"] = -cfg.w_finger_cross * s["cross_pen"]
        # P0.3: 权重默认 0 (先量再开, 见 cfg 注释)
        if cfg.w_arm_table > 0:
            terms["arm_table"] = -cfg.w_arm_table * s["arm_table_pen"]
        if cfg.w_self > 0:
            terms["self_gap"] = -cfg.w_self * s["self_pen"]
        # -- 里程碑 --
        if cfg.approach:
            terms["arrive"] = cfg.r_arrive * s["newly_arrive"].float()
        if getattr(cfg, "approach_only", False):
            # Approach-only 的四层奖励 (用户 2026-08-16 定):
            #   主项 = align 势差分 (上面已加, w_align·Δφ) —— **必须是势函数**:
            #     telescoping 使累积奖励只取决于起终点, 中间怎么走不影响总量 ⟹
            #     ① 不改变最优策略(Ng 的 shaping 定理) ② 不能靠"在物体旁来回蹭"刷分。
            #     直接给"越近奖励越高"会让最优解变成"贴着不走了"。
            #   终端 = 到位一次性 r_reach, 然后终止
            #   底线 = 撞桌/碰物体 直接终止 + r_hit
            # ★ 用时折扣 (2026-08-17 用户裁定): 到位奖励按**用了多少步**衰减,
            #   否则奖励对路径长度**完全中性** —— 直着去和先退后进拿一样多的分,
            #   策略没有任何动力走直线 (实测: 自学的路 211 步, 冠军 94 步)。
            #   形式: r × decay^(用时 − 基准)。指数而非硬地板 ⟹ 永远 >0
            #   (到位始终优于不到位), 且快慢**全程有区别**、无饱和区。
            #   ⚠ `w_time_shape` 由训练入口按**成功率**放行(先学会到位, 再学快) ——
            #     从零就打折会把唯一的强信号削没, 可能永远学不会 (用户提的顾虑)。
            _ms = cfg.r_reach * s["newly_success"].float()
            _sh = float(getattr(cfg, "time_shape_lambda", 0.0))
            if _sh > 0.0:
                _used = self.episode_length_buf.float()
                _dec = float(getattr(cfg, "time_shape_decay", 0.99))
                _ref = float(getattr(cfg, "time_shape_ref_steps", 94.0))
                _f = torch.pow(torch.tensor(_dec, device=self.device),
                               (_used - _ref).clamp(min=0.0))
                _ms = _ms * (1.0 - _sh + _sh * _f)      # λ=0 不打折, λ=1 全打折
            terms["milestone"] = _ms
            terms["hit"] = cfg.r_hit * self._approach_hit.float()
        else:
            terms["milestone"] = (cfg.r_candidate * s["newly_cand"].float()
                                  + cfg.r_success * s["newly_success"].float())
        # -- 失败罚 (verify_fail 不终止, 只小额罚 + 退回 GRASP 重试) --
        # 罚后果不罚接触 (2026-08-19 用户裁定): 物体被扰动(速度)全程连续罚,
        # 轻碰不动零罚 —— "碰而不倒"仍是任务素材, 晃了才交钱。
        _odw = float(getattr(cfg, "w_obj_disturb", 0.0))
        _odx = float(getattr(cfg, "obj_disturb_pre_x", 1.0))
        if _odx != 1.0:
            # AAG-Local (2026-08-22 用户裁定): 到位前扰动罚加权 —— 修补器证实掌蹭
            # 是结构性的(≤16mm 侧移绕不开), 改由 RL 在残差带内自己学绕行;
            # 物体动了就罚, 掌根/指背/指垫谁碰的都看得见 (垫力转储实锤盲区)
            _ods = torch.where(self.arrived,
                               torch.ones_like(s["obj_spd"]),
                               torch.full_like(s["obj_spd"], _odx))
            terms["obj_disturb"] = -_odw * _ods * s["obj_spd"].clamp(max=2.0)
        else:
            terms["obj_disturb"] = -_odw * s["obj_spd"].clamp(max=2.0)
        if float(getattr(cfg, "w_toppled", 0.0)) > 0.0:
            # RSI-B 线独立拍倒罚: 不受 APPROACH_OFF 影响 (fail 整项仍关)
            terms["toppled_pen"] = -cfg.w_toppled * \
                s.get("toppled", s["fell"] & False).float()
        terms["fail"] = (cfg.r_drop * (s["fell"] | s["thrown"] | s["table_crash"]
                                       | s.get("toppled", s["fell"] & False)).float()
                         + cfg.r_pushed_away * s["pushed"].float()
                         + cfg.r_verify_fail * s["verify_fail"].float())
        # -- 正则 (速度先去 NaN 再钳: 物理尖峰的平方会淹没任务信号, 见台账) --
        terms["act_rate"] = -cfg.w_act_rate * \
            (self.actions_buf - self.prev_actions).square().mean(dim=1)
        # 盘面用(不进奖励, 只累计给 TB): 指尖接触根数 + 残差动作范数
        self._diag_contacts += self._tip_contacts().sum(dim=1)
        self._diag_actnorm += self.actions_buf.norm(dim=1)
        self._diag_n += 1
        qd_arm = self.arm_qd.nan_to_num(0.0)
        qd_fin = self.finger_qd.nan_to_num(0.0)
        qd = torch.cat([qd_arm / cfg.qd_soft_arm, qd_fin / cfg.qd_soft_fin], dim=1)
        terms["qvel"] = -cfg.w_qvel * qd.square().mean(dim=1).clamp(max=10.0)
        over = torch.cat([(qd_arm.abs() - cfg.qd_soft_arm).clamp(min=0.0),
                          (qd_fin.abs() - cfg.qd_soft_fin).clamp(min=0.0)], dim=1)
        terms["qvel_hard"] = -cfg.w_qvel_hard * \
            over.square().sum(dim=1).clamp(max=cfg.qvel_hard_cap)
        if getattr(cfg, "qvel_settle_exempt", False):
            # ★ 2026-08-23 (预检抓到): 复位后的冻结窗口内 `a = actions*gate` 被**归零**,
            # 策略对关节速度毫无影响力; 而复位是 write_joint_state_to_sim **传送**,
            # 头一两帧的读数是伪值(实测指 1112 rad/s, 直接把 qvel_hard 打到 cap)。
            # 于是"严格复现参考"白挨罚 —— 实测右手 -0.048/步, 与"学参考"直接冲突。
            _nf = (self.freeze_ctr <= 0).float()
            terms["qvel"] = terms["qvel"] * _nf
            terms["qvel_hard"] = terms["qvel_hard"] * _nf
        terms["torque"] = -cfg.w_torque * self.arm_torque_norm.square().mean(dim=1)
        # ---- ⑤ squeeze 段抓力奖励 (2026-08-24 用户第⑤条) --------------------
        # 用户原话: "Squeeze 全部给探索, 鼓励指垫在物体表面接触, 让指垫在 GraspPose
        #   位置鼓励向着物体表面靠近, 奖励最后调整稳定的抓力"
        # 两件现成的量, 都做成 **earn-only 势差分**(只奖进步不罚参考, 与今日全线口径一致):
        #   ① pad_near : 逐垫→物体表面距离的进步 (pad_d_shape 势差, 原点口径的常数
        #                偏移在**差分**里自动抵消, 所以那个 2.4cm 已知偏差无害)
        #   ② grip_pot : quality = cent − λ₁·r_imb − λ₂·τ 的进步
        #                (压入表面 − 合力不平衡 − 净力矩 = "稳定的综合抓力")
        # 窗口 = squeeze 段起始之后 (由段表推导的 fin_end_row 之前的 squeeze 起点)
        _sqw = float(getattr(cfg, "squeeze_grip_w", 0.0))
        if _sqw > 0.0 and int(getattr(cfg, "squeeze_row0", 0)) > 0:
            _insq = (active & (self.ref_t >= int(cfg.squeeze_row0))).float()
            _pn = ((self.prev_pad_d - s["pad_d_shape"]).clamp(min=0.0, max=0.02)
                   * self.finger_active.unsqueeze(0)).sum(dim=1) \
                / max(float(self.n_active), 1.0)
            terms["pad_near"] = _sqw * _pn * 50.0 * _insq
            _gq = (s["quality"] - self.prev_quality).clamp(min=0.0, max=0.05)
            terms["grip_pot"] = _sqw * _gq * _insq

        # ---- 合拢前禁触 (2026-08-24 用户裁定) --------------------------------
        # "从 5cm 进入的时候会碰到物体; 其实在准备开始合拢成 GraspPose 之前
        #  都应该避免碰到物体。"
        # 窗口 = ref_t < aag_grasp_row (=close 段起始行, 180行制 120)。时钟是**外生**的
        # (每步 +1, 策略动不了), 所以窗口长度固定, 不存在"拖时间躲罚"。
        # 死区从**参考自身**标定: 零动作实测窗口内 右手位移上限 0.65cm / 左手 0.00cm,
        # 两手**零指垫接触** ⟹ 死区 1.0cm 时该项在基线上恒为 0 (不罚参考)。
        _pcw = float(getattr(cfg, "pre_close_w", 0.0))
        if _pcw > 0.0 and int(getattr(cfg, "aag_grasp_row", 0)) > 0:
            _win = (active & (self.ref_t < int(cfg.aag_grasp_row))
                    & (self.ref_t > 0))          # ref_t==0 是复位帧, 读数是残留
            _dead = float(getattr(cfg, "pre_close_dead_cm", 1.0)) / 100.0
            # 倾角项 (2026-08-24 用户: "蹭歪了物体, 让物体进入一直倾斜的状态"):
            # disp 只看平移, 手指能把物体**转歪**而位移不大 ⟹ 必须单列。
            # 量纲对齐: 5° 超死区 = 1 个指垫接触 = 1cm 超死区。
            _tdead = float(getattr(cfg, "pre_close_tilt_deg", 5.0))
            # 外壳间隙 (2026-08-24 用户: "机器人全身外壳在摆到 GraspPose 之前都不想
            # 碰到物体和桌子")。手+l7/l8/ee 到物体表面点的最近距离低于余量就罚。
            # ⚠ 余量必须从**参考自身**标定 —— 参考在 advance 段末本来就会贴近。
            _shm = float(getattr(cfg, "pre_close_shell_cm", 0.0)) / 100.0
            _shp = ((_shm - s["shell_obj_d"]).clamp(min=0.0) * 100.0
                    if _shm > 0.0 else torch.zeros_like(s["disp"]))
            terms["pre_close"] = -_pcw * _win.float() * (
                s["n_pads"].float()
                + (s["disp"] - _dead).clamp(min=0.0) * 100.0
                + (s["tilt_deg"] - _tdead).clamp(min=0.0) / 5.0
                + _shp)
        terms["time"] = torch.full_like(af, -cfg.w_time)

        # ---- Approach-only 的奖励最小集 (2026-08-16 用户逐条裁定) ----
        # 实测依据: 接近任务下 24 个奖励项里 **13 个恒 0**(全是从抓取任务继承的壳),
        # 而 `imit` 是**活的且与主项对冲** —— 它罚"每步残差用量", 前提是有条可信参考
        # 要跟; 但本任务的设计前提就是**不跟人手轨迹**(改用退避族+势函数)。
        # 于是它变成纯粹压制动作幅度, 和 `align`(要求主动走向目标)直接打架,
        # 量级还相当(实测 imit −0.014 vs align +0.030)。
        # 用户裁定: imit 关 / 抓取段项全关 / fail 关; self_gap 留(恒 0 无代价, 双手要用)。
        if getattr(cfg, "approach_only", False):
            for _k in self.APPROACH_OFF:
                terms.pop(_k, None)
            if not getattr(self, "_rew_printed", False):
                self._rew_printed = True
                print(f"[reward] Approach-only 生效项: "
                      f"{sorted(k for k, v in terms.items())}")
                print(f"[reward] 已关闭: {sorted(self.APPROACH_OFF)}")
        if return_terms:
            return terms          # ← 调用方自己过日程表并并入自己的标量
        self.apply_reward_schedule(terms)
        total = (sum(terms.values()) * af).nan_to_num(0.0)   # 最后一道 NaN 闸
        self.prev_pad_d = torch.where(active.unsqueeze(1), s["pad_d_shape"], self.prev_pad_d)
        self.prev_quality = torch.where(active, s["quality"], self.prev_quality)
        for k, v in terms.items():
            if k not in self._ep_sums:
                self._ep_sums[k] = torch.zeros(self.num_envs, device=self.device)
            self._ep_sums[k] += v * af
        if getattr(cfg, "step_reward_log", ""):
            # 逐步奖惩记录 (2026-08-21 用户点单): 全 env 均值 + 前8探针env全明细,
            # 每 2000 步一片 npz。事后可逐步回放"状态↔奖惩"。
            import os as _os
            if not hasattr(self, "_srl"):
                object.__setattr__(self, "_srl", {"n": 0, "shard": 0, "rows": []})
                _os.makedirs(cfg.step_reward_log, exist_ok=True)
            _ks = sorted(terms.keys())
            object.__setattr__(self, "_srl_keys", _ks)   # 预检打标签用
            _P = 8
            _row = dict(
                side=getattr(self, "hand_side", "?")
                if not hasattr(self, "_cur") else self._cur.name,
                step=int(self._srl["n"]),
                terms_mean=np.array([float(terms[k].mean()) for k in _ks],
                                    np.float32),
                terms_probe=np.stack(
                    [terms[k][:_P].detach().cpu().numpy() for k in _ks]
                ).astype(np.float32),                       # (T项, 8)
                phase=self.task_phase[:_P].cpu().numpy().astype(np.int8),
                ref_t=self.ref_t[:_P].cpu().numpy().astype(np.int32),
                n_pads=s["n_pads"][:_P].cpu().numpy().astype(np.int8),
                d_pos=s.get("d_pos", torch.zeros(1))[:_P] .cpu().numpy()
                .astype(np.float32) if "d_pos" in s else np.zeros(_P, np.float32),
            )
            self._srl["rows"].append(_row)
            self._srl["n"] += 1
            if len(self._srl["rows"]) >= 1000:              # 双侧各500步一片
                _R = self._srl["rows"]
                np.savez_compressed(
                    _os.path.join(cfg.step_reward_log,
                                  f"shard_{self._srl['shard']:05d}.npz"),
                    term_names=np.array(_ks),
                    side=np.array([r["side"] for r in _R]),
                    step=np.array([r["step"] for r in _R], np.int64),
                    terms_mean=np.stack([r["terms_mean"] for r in _R]),
                    terms_probe=np.stack([r["terms_probe"] for r in _R]),
                    phase=np.stack([r["phase"] for r in _R]),
                    ref_t=np.stack([r["ref_t"] for r in _R]),
                    n_pads=np.stack([r["n_pads"] for r in _R]),
                    d_pos=np.stack([r["d_pos"] for r in _R]))
                self._srl["shard"] += 1
                self._srl["rows"] = []
        return total

    # ---- 观测 ---------------------------------------------------------
    def _get_observations(self) -> dict:
        cfg, origins = self.cfg, self.scene.env_origins
        vs = cfg.obs_vel_scale
        wrist_pos = self.wrist_pos_w - origins
        wrist_quat = self._qsign(self.wrist_quat_w)
        q_inv = quat_conjugate(wrist_quat)
        obj_pos = self.object.data.root_pos_w - origins
        obj_quat = self.object.data.root_quat_w
        arm_q = self.arm_q
        # 接触力通道现算 (不能用 _sig 缓存: 刚 reset 的 env 那份是上一回合的)
        pad_d, pad_n, _ = self._pad_dist_normal()
        mag, _, cent, r_imb, tau_n, G = self._pad_signals(pad_n)
        N = self.num_envs

        phase_oh = torch.nn.functional.one_hot(self.task_phase, Phase.N).float()
        to_t = self.phase_timeout_t[self.task_phase].clamp(min=1).float()
        phase_prog = (self.phase_step.float() / to_t).clamp(max=1.0)
        if cfg.approach:
            # 接近相位的进度条直接用 φ = t_ref/gs —— 它就是奖励权重的调度变量,
            # 策略必须能看到它, 否则"权重随时间变而策略看不到时间" = 非平稳 MDP.
            phase_prog = torch.where(self.task_phase == Phase.PREGRASP,
                                     self._phi(), phase_prog)
        phase_prog = phase_prog.unsqueeze(1)

        # ---- 到 GraspPose 的误差 (腕系): prior 给的**固定**目标, 不含物体真值 ----
        if getattr(self, "_grasp_pos_w", None) is not None:
            g_pos = quat_apply(q_inv, self._grasp_pos_w.expand(N, 3) - wrist_pos)
            _, _, g_rot = self._align_err()
        else:                                   # 无 prior (设定 B) 时置 0
            g_pos = torch.zeros(N, 3, device=self.device)
            g_rot = torch.zeros(N, 3, device=self.device)
        # ---- 参考轨迹通道 (跟踪误差 + 前瞻), 与 approach 开关无关, 常开 ----
        L = self.q_ref.shape[0]
        t_now = self.ref_t.clamp(max=L - 1)
        look = [self.q_ref[(self.ref_t + k).clamp(max=L - 1)] - self.q_ref[t_now]
                for k in cfg.ref_look_ks[:int(cfg.ref_look_frames)]]
        # ---- 逐垫力**向量**(腕系) —— 替换原来的 5 个模长 + cent/imb/tau ----
        Gw = quat_apply(q_inv.unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4),
                        G.reshape(-1, 3)).reshape(N, 15) / cfg.squeeze_f0

        # ============ actor 观测 (151): **只含部署时拿得到的量** ============
        # 判据: 本体感受(编码器/FK/力矩/触觉) + 流水线离线已知(参考轨迹/GraspPose/相位)
        #      + 自身内部状态. **物体的实时位姿及其派生量一律不在这里** —— 它是仿真真值,
        #      真机上要靠感知, 重建/retarget 给不了 (物体 track 本身就不可信).
        # 见 DESIGN_LOOP §2.12 / PLAN_PICK_LIFT §3.6.
        obs = torch.cat([
            arm_q,                                                      # 7  编码器
            self.finger_q,                                              # 22
            self.arm_qd * vs,                                           # 7
            self.finger_qd * vs,                                        # 22
            wrist_pos,                                                  # 3  FK
            wrist_quat,                                                 # 4
            self.wrist_linvel_w,                                        # 3
            self.wrist_angvel_w * vs,                                   # 3
            # approach_only(7 维臂动作): 手指全程张开, closure 与每指残差恒为 0
            # ⟹ **零信息量**, 整段去掉 (obs 基数 144-12, 见 cfg._obs_base)
            *([] if getattr(cfg, "approach_only", False) else [
                (self.closure / cfg.closure_max).unsqueeze(1),          # 1  内部
                # closure 模式 5 维(每指残差); joints 模式 44 维(逐关节累积残差 22
                # + 逐关节参考跟踪误差 22 —— 要跟 22 个关节的参考就得看见自己差多少)
                *([self.fin_res / self.finger_dev_max,
                   (self.finger_q - self.finger_tgt) / self.finger_dev_max]
                  if self._joint_hand else [self.fin_delta / cfg.delta_max])]),
            self.q_pregrasp - arm_q,                                    # 7  prior
            phase_oh,                                                   # 6  内部时钟
            phase_prog,                                                 # 1  (接近段 = φ)
            self._tip_contacts(),                                       # 5  触觉
            self.arm_torque_norm,                                       # 7  电流
            self.actions_buf,                                           # 13 内部
            Gw.clamp(-3.0, 3.0),                                        # 15 触觉力向量(腕系)
            g_pos,                                                      # 3  到抓姿位置误差
            g_rot,                                                      # 3  到抓姿姿态误差
            self.q_ref[t_now] - arm_q,                                  # 7  参考跟踪误差
            *look,                                                      # 7×N 参考前瞻
            *([self.ref_contact] if cfg.ref_contact_obs else []),        # 5  参考接触指集
            # v5 (2026-08-21 用户裁定): 特权滑移块 8 维/侧 —— 腕系滑移量/速度、
            # 垫压总量、垫数、物物最小距、杯倾角。**部署口径的例外** (用户拍板:
            # 抓稳靠特权反射弧, actor 必须"感到滑"; 蒸馏时再换感知源), 台账已录。
            *([] if not getattr(cfg, "pours_v5", False) else
              [getattr(self, "_v5ob", {}).get(
                  getattr(getattr(self, "_cur", None), "name", "right"),
                  torch.zeros(N, 10 if getattr(cfg, "pours_v6", False) else 8,
                              device=self.device))]),        # 8 (v6: +started/pf_on=10)
            # AAG-Local (2026-08-22 用户裁定新纪律): RL=数据生成器不部署 ⟹ 物体
            # 特权进 actor —— 闭环重瞄准/绕蹭要"看着瓶子"做 (部署约束移到 DP 数据
            # 集模态, 见台账 §2.12 改写)
            *([] if not getattr(cfg, "obj_in_actor", False) else [
                quat_apply(q_inv, obj_pos - wrist_pos),                 # 3
                self._qsign(quat_mul(q_inv, obj_quat)),                 # 4
                self.object.data.root_lin_vel_w,                        # 3
                # ★ 2026-08-24 用户点单: "actor 应该知道手指每动一步会让物体发生
                # 怎么样的变化"。角速度原本是 critic 独占, 而"被蹭歪"恰恰是转动 ——
                # actor 看不到转速就无法把"这一步指动"和"物体转了"关联起来。
                self.object.data.root_ang_vel_w * vs,                   # 3
            ]),                                              # =13 物体特权块
        ], dim=1).clamp(-cfg.clip_obs, cfg.clip_obs).nan_to_num(0.0)
        self._check_obs_dim(obs)

        # ============ priv (32): critic 全看; actor 只经 env_mlp 看**前 7** ============
        # 前 7 维保持原样 (质量/摩擦/指尖力), 教师-学生蒸馏那条线不受影响;
        # 后 25 维是新搬进来的特权信息 —— 原来它们混在 obs 里, 学生照样看得到,
        # 师生划分等于形同虚设 (models.py 的 adapt_tconv 只替换 priv, 不替换 obs).
        tip_f = torch.cat([s_.data.force_matrix_w.view(N, 1, 3)
                           for s_ in self._contact_sensors], dim=1).norm(dim=-1)
        priv = torch.cat([
            self.obj_mass, self.obj_fric, tip_f,                        # 7  (原样)
            quat_apply(q_inv, obj_pos - wrist_pos),                     # 3  物体位置(腕系)
            self._qsign(quat_mul(q_inv, obj_quat)),                     # 4  物体姿态(相对腕)
            self.object.data.root_lin_vel_w,                            # 3
            self.object.data.root_ang_vel_w * vs,                       # 3
            (pad_d * 10.0).clamp(max=2.0),                              # 5  逐垫到表面距离
            ((obj_pos[:, 2] - self.obj_rest_z)
             / cfg.lift_height).clamp(-1.0, 1.5).unsqueeze(1),          # 1  物体高度
            cent.unsqueeze(1), r_imb.unsqueeze(1), tau_n.unsqueeze(1),  # 3  力几何(用了真值法向)
            quat_apply(q_inv, (self._target_w() - self._anchor_w())),   # 3  实时目标相对
        ], dim=1).clamp(-cfg.clip_obs, cfg.clip_obs).nan_to_num(0.0)
        assert priv.shape[1] == 32, f"priv 维度 {priv.shape[1]} != 32 (ppo.yaml 的 priv_info_dim)"
        self.proprio_hist[:, :-1].copy_(self.proprio_hist[:, 1:].clone())
        self.proprio_hist[:, -1] = torch.cat(
            [arm_q, self.finger_q, self.arm_qd * vs, self.finger_qd * vs], dim=1)
        return {"policy": obs, "priv_info": priv, "proprio_hist": self.proprio_hist}

    # ---- 复位 ---------------------------------------------------------
    # ================= 自生成起点池 (2026-08-16 用户裁定) =====================
    # 动机: 退避课程的起点来自**手工构造的几何路径** —— 于是有"关节插值下沉 0.42cm"、
    #   "参考自己绕路(先到 14cm 再退回 25cm)"这些几何问题, 换个物体还得重验。
    #   而且它是"移动一个窗口": 难度一上去成绩就掉, 一掉就触发回退 ⟹ 课程被自己的
    #   保护机制锁死在容易区 (实测: 四条 13~15M 步, far 一直在 0~0.36 荡, 评测全 0%)。
    # 改成: 起点 = **策略自己真到过的状态**, 按"离目标多远"分档存池, 每回合从各档抽。
    #   天然可行(它自己走到过 ⟹ 一定可达、一定不穿桌), 不需要任何几何验收;
    #   而且是"始终覆盖全程"而不是"移动窗口" ⟹ 没有可以震荡的东西。
    # 两种权重 (A/B 对照, 只差这一项):
    #   uniform  —— 各档等权。简单、无参数。
    #   mastery  —— 权重 ∝ (1 - 该档到位率), 但**每档有地板**。
    #     用户提的: 前面的档掌握了就少分点资源给后面更新的问题; "不完全砍掉"是关键 ——
    #     砍到 0 会灾难性遗忘, 评测(从站姿)直接崩。站姿档地板更高(它就是考试分布)。
    SP_EDGES_CM = (1.0, 2.0, 4.0, 8.0, 15.0, 25.0)   # 6 条边 -> 7 档, 最后一档 >=25cm

    def _sp_init(self):
        cfg, dev, N = self.cfg, self.device, self.num_envs
        nd = len(self.SP_EDGES_CM) + 1        # 距离档数
        nb = nd + 1                           # +1 = **站姿虚拟档** (下标 nd)
        cap = int(getattr(cfg, "start_pool_cap", 256))
        self._sp_nd, self._sp_nb, self._sp_cap = nd, nb, cap
        self._sp_edges = torch.tensor(self.SP_EDGES_CM, device=dev) / 100.0
        self._sp_q = torch.zeros(nb, cap, 7, device=dev)
        self._sp_n = torch.zeros(nb, dtype=torch.long, device=dev)
        self._sp_ptr = torch.zeros(nb, dtype=torch.long, device=dev)
        # ★ 站姿是**独立的虚拟档**, 不参与距离分档, 也永远不会被环形缓冲挤掉。
        #   第一版把它塞进"最远档"并让该档不收样本 —— 结果 Pour17 站姿离目标 41cm、
        #   超出最高档边界 25cm, 于是**所有样本都落进那个不收的档**, 池全程是空的
        #   (冒烟实测 7 档全 0)。距离档现在**全部开放收集**。
        q_stance = self.hand.data.default_joint_pos[0, self.arm_jids].to(dev)
        self._sp_q[nd, 0] = q_stance
        self._sp_n[nd] = 1
        self._sp_ptr[nd] = 1
        self._sp_arr = torch.zeros(nb, device=dev)      # 各档到位率慢 EMA = "掌握度"
        self._sp_seen = torch.zeros(nb, device=dev)     # 各档样本数(判掌握度可不可信)
        self._sp_start_bin = torch.full((N,), nb - 1, dtype=torch.long, device=dev)
        self._sp_best_d = torch.full((N,), 1e9, device=dev)
        self._sp_best_q = torch.zeros(N, 7, device=dev)
        print(f"[start_pool] 已建: {nb} 档 (边界 {self.SP_EDGES_CM} cm) × 容量 {cap} | "
              f"权重={str(getattr(cfg, 'start_pool', 'uniform') or 'uniform')} | 最远档已用站姿播种")

    def _sp_track(self, d_pos):
        """每步记录: 本回合到过的**最近**位形 (纯位置口径, 与判据同源)。"""
        q_now = self.hand.data.joint_pos[:, self.arm_jids]
        better = d_pos < self._sp_best_d
        self._sp_best_d = torch.where(better, d_pos, self._sp_best_d)
        self._sp_best_q = torch.where(better.unsqueeze(1), q_now, self._sp_best_q)

    def _sp_push(self, env_ids):
        """回合结束: 把"最近点位形"存进对应档; 同时更新该回合起点档的掌握度。"""
        d, q = self._sp_best_d[env_ids], self._sp_best_q[env_ids]
        ok = torch.isfinite(d) & (d < 1e8)
        if not bool(ok.any()):
            return
        d, q = d[ok], q[ok]
        b = torch.bucketize(d, self._sp_edges)          # 0..nb-1, 越大越远
        for i in range(self._sp_nd):                    # 距离档全部收; 站姿虚拟档(nd)不收
            m = b == i
            k = int(m.sum())
            if k == 0:
                continue
            idx = (self._sp_ptr[i] + torch.arange(k, device=d.device)) % self._sp_cap
            self._sp_q[i, idx] = q[m]
            self._sp_ptr[i] = (self._sp_ptr[i] + k) % self._sp_cap
            self._sp_n[i] = torch.clamp(self._sp_n[i] + k, max=self._sp_cap)
        # 掌握度: 按**这一回合从哪一档起步**归账 (不是按落到哪一档)
        sb, arr = self._sp_start_bin[env_ids], self.arrived[env_ids].float()
        for i in range(self._sp_nb):
            m = sb == i
            if not bool(m.any()):
                continue
            a = float(arr[m].mean())
            self._sp_arr[i] = 0.9 * self._sp_arr[i] + 0.1 * a
            self._sp_seen[i] += float(m.sum())

    def _sp_sample(self, n, dev):
        """按权重抽档 -> 档内随机抽一个位形。返回 (q_arm, bin_id)。"""
        cfg = self.cfg
        avail = (self._sp_n > 0).float()
        if str(getattr(cfg, "start_pool", "uniform") or "uniform") == "mastery":
            # 权重 ∝ (1 - 掌握度), 地板防遗忘; 站姿档(最远)地板更高 = 考试分布不能饿着
            floor = float(getattr(cfg, "start_pool_floor", 0.08))
            w = (1.0 - self._sp_arr).clamp(min=floor)
            # 样本太少的档掌握度不可信 -> 当作"没掌握", 给满权重 (今天被 4 样本噪声骗过)
            w = torch.where(self._sp_seen < 200.0, torch.ones_like(w), w)
            w[self._sp_nd] = torch.clamp(w[self._sp_nd],     # 站姿虚拟档
                                             min=float(getattr(cfg, "start_pool_stance_floor", 0.20)))
        else:
            w = torch.ones_like(avail)
        w = w * avail
        if float(w.sum()) <= 0:
            w = avail
        b = torch.multinomial(w / w.sum(), n, replacement=True)
        j = (torch.rand(n, device=dev) * self._sp_n[b].float()).long().clamp(min=0)
        j = torch.minimum(j, (self._sp_n[b] - 1).clamp(min=0))
        return self._sp_q[b, j], b

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # ⚠ pads_touched 是**回合内 latch**("曾经碰到过几个垫"), 不是同时接触数 ——
        #   Grasp1 判读时曾被它误导 (latch 4.45 而实测同时 ≥4 垫的帧数为 0).
        #   pads_now 才是候选判据真正看的量, 两个都记.
        log = {"success_rate": self.succeeded[env_ids].float().mean().item(),
               "grasp/got_candidate": self.got_candidate[env_ids].float().mean().item(),
               "grasp/gentle": float(self.gentle),
               "grasp/pads_touched_latch": self.pad_touched[env_ids].float().sum(dim=1)
               .mean().item(),
               "grasp/pads_now": (self._sig["n_pads"][env_ids].float().mean().item()
                                  if self._sig else 0.0)}
        if self._sig:
            for k in ("fell", "thrown", "toppled", "pushed", "stuck", "table_crash",
                      "verify_fail", "timeout"):
                if k in self._sig:
                    log[f"term/{k}"] = self._sig[k][env_ids].float().mean().item()
            log["diag/arm_table_gap_cm"] = self._sig["arm_gap"][env_ids].min().item() * 100
            log["diag/self_gap_cm"] = self._sig["self_gap"][env_ids].min().item() * 100
            log["diag/obj_tilt_max_deg"] = self.tilt_max_deg[env_ids].mean().item()
        # ---- 用户点名的四个盘面指标 (2026-08-15) ----
        #   grasp/  抓取本身好不好(与"整条任务成没成"分开看)
        #   drop_rate = **形成过候选却没走完** —— 抓到了又丢的比例, 抓取不稳的直接证据
        _n = self._diag_n[env_ids].clamp(min=1)
        log["diag/n_contacts"] = (self._diag_contacts[env_ids] / _n).mean().item()
        log["diag/action_norm"] = (self._diag_actnorm[env_ids] / _n).mean().item()
        _fn = self._diag_fgate_n[env_ids].clamp(min=1)
        log["diag/finger_gate"] = (self._diag_fgate[env_ids] / _fn).mean().item()
        gc = self.got_candidate[env_ids]
        log["grasp/candidate_rate"] = gc.float().mean().item()      # 形成候选(≥N垫+向心+保持)
        if bool(gc.any()):
            log["grasp/drop_rate"] = (~self.succeeded[env_ids][gc]).float().mean().item()
        log["grasp/success_rate"] = self.succeeded[env_ids].float().mean().item()
        if self.cfg.approach:
            # 按起步分支分桶 —— 单 run 内分辨"接近坏了"还是"抓取坏了" (DESIGN_LOOP A7)
            sg = self.started_grasp[env_ids]
            for tag, m in (("from_approach", ~sg), ("from_grasp", sg)):
                if m.any():
                    log[f"sr/{tag}"] = self.succeeded[env_ids][m].float().mean().item()
            # ⚠ 接近段的量**只统计接近起步的回合** —— 混进抓取起步的会把它们稀释到 ≈0,
            #   完全掩盖真实误差 (2026-08-02 判读时踩过这个坑).
            ap = ~sg
            if bool(ap.any()):
                arr = self.arrived[env_ids][ap]
                log["approach/arrive_rate"] = arr.float().mean().item()
                log["approach/res_used_cm"] = self.res_step_cm[env_ids][ap].mean().item()
                if self.cfg.cone_trust and getattr(self, "cone_out_cm", None) is not None:
                    log["approach/cone_out_cm"] = \
                        self.cone_out_cm[env_ids][ap].mean().item()
                if self._sig:
                    log["approach/d_pos_cm"] = \
                        self._sig["d_pos"][env_ids][ap].mean().item() * 100
                    log["approach/d_rot_deg"] = float(np.degrees(
                        self._sig["d_rot"][env_ids][ap].mean().item()))
                if bool(arr.any()):
                    log["approach/steps_to_arrive"] = \
                        self.arrive_step[env_ids][ap][arr].float().mean().item()
            if getattr(self, "_sp_q", None) is not None:
                for i in range(self._sp_nb):
                    if i == self._sp_nd:
                        tag = "stance"
                    else:
                        lo = 0.0 if i == 0 else self.SP_EDGES_CM[i - 1]
                        hi = (self.SP_EDGES_CM[i] if i < len(self.SP_EDGES_CM)
                              else 99.0)
                        tag = f"{lo:g}-{hi:g}cm"
                    log[f"pool/n_{tag}"] = float(self._sp_n[i])
                    log[f"pool/mastery_{tag}"] = float(self._sp_arr[i])
                    log[f"pool/startfrac_{tag}"] = float(
                        (self._sp_start_bin[env_ids] == i).float().mean())
            log["curr/eps_pos_cm"] = self.cfg.eps_pos * 100
            log["curr/eps_rot_deg"] = float(np.degrees(self.cfg.eps_rot))
            log["curr/w_imit_ramp"] = float(self.cfg.w_imit_ramp)
            log["curr/approach_budget"] = float(self.phase_timeout_t[Phase.PREGRASP])
            if getattr(self.cfg, "grip_g", False):
                log["grip/g_mean"] = float(self._grip_g.mean())
                log["grip/g_max"] = float(self._grip_g.max())
                log["grip/snapshot_rate"] = float(self._grip_has.float().mean())
        # ---- 验证段诊断 (腕实际抬了多少 / 物体跟了多少 / 跟随率) ----
        _m = self.vf_has[env_ids]
        if bool(_m.any()):
            _w = self.vf_wrist_mm[env_ids][_m]
            _o = self.vf_obj_mm[env_ids][_m]
            log["verify/wrist_rise_mm"] = _w.mean().item()
            log["verify/obj_rise_mm"] = _o.mean().item()
            log["verify/follow_ratio"] = (_o / _w.clamp(min=0.5)).clamp(-1, 2).mean().item()
        self.vf_has[env_ids] = False
        # ---- 接触分数图落账 (verify 快照 × 本回合成败) ----
        if self.score_on:
            w = self.pend_w[env_ids].reshape(-1)
            i = self.pend_idx[env_ids].reshape(-1)
            sc = self.succeeded[env_ids].float().unsqueeze(1).expand(-1, 5).reshape(-1)
            self.score_n.index_add_(0, i, w)
            self.score_s.index_add_(0, i, w * sc)
            self.pend_w[env_ids] = 0.0
            log["score/n_covered"] = float((self.score_n >= self.cfg.score_n_min).sum())
        # 2026-08-20 铁案4(双侧顺序家族): 双臂复位 A 侧先清零 episode_length_buf,
        # B 侧再到这里时 ep_len=0→clamp(1) ⟹ B 报"回合总额"、A 报"每步均值",
        # 两侧 ep_rew 口径差 240 倍 —— "右手m1判负/左手提速"等历史判读全部因此失实。
        # 修法: 双臂入口在循环前快照真实回合长度 (_bi_ep_len), 两侧共用。
        _epl = getattr(self, "_bi_ep_len", None)
        ep_len = (_epl if _epl is not None
                  else self.episode_length_buf[env_ids]).clamp(min=1).float()
        for k, v in self._ep_sums.items():
            log[f"ep_rew/{k}"] = (v[env_ids] / ep_len).mean().item()
            v[env_ids] = 0.0
        DirectRLEnv._reset_idx(self, env_ids)       # 跳过基类的 RSI/参考时钟复位
        self.extras.update(log)
        self.extras["log"] = log

        n = len(env_ids)
        dev, cfg = self.device, self.cfg
        origins = self.scene.env_origins[env_ids]
        if getattr(self, "_bi_skip_scene", False):
            # ★ 双臂 B 侧复位: 场景已由 A 侧那遍摆好, **不许重摆**。
            #   2026-08-18 实锤: B 侧 self.object 已是 aux, 这里重摆会先把 aux 写到
            #   B 数学位, 再被 reset_aux_free 叠一次相对偏移 —— 杯子被推到 2×offset
            #   (实测 y=0.328 vs 0.164, 差 16.6cm), 左手全系"前馈抓空气/奖励拽真杯"。
            #   B 侧只把 obj_start_* 记成 aux 的**真实摆放**(fell/pushed 判据的基准)。
            _ap = getattr(self, "_aux_last_pos", None)
            if _ap is not None:
                self.obj_start_pos[env_ids] = _ap[env_ids]
                self.obj_start_quat[env_ids] = self._aux_last_quat[env_ids]
            else:   # 兜底: 同帧 data 可能滞后一拍, 但仅毫米级
                self.obj_start_pos[env_ids] = (
                    self.object.data.root_pos_w[env_ids] - origins)
                self.obj_start_quat[env_ids] = self.object.data.root_quat_w[env_ids]
        else:
            # ---- 物体: 稳定初始位 + xy 抖动 ----
            obj_p = self.obj_init_pos.expand(n, -1).clone()
            obj_p[:, :2] += (torch.rand(n, 2, device=dev) * 2 - 1) * cfg.obj_jitter_xy
            obj_q = self.obj_init_quat.expand(n, -1)
            if cfg.obj_jitter_yaw > 0:
                half = (torch.rand(n, device=dev) * 2 - 1) * cfg.obj_jitter_yaw * 0.5
                yaw = torch.stack([torch.cos(half), torch.zeros_like(half),
                                   torch.zeros_like(half), torch.sin(half)], dim=1)
                obj_q = quat_mul(yaw, obj_q)
            obj = torch.zeros(n, 13, device=dev)
            obj[:, 0:3] = obj_p + origins
            obj[:, 3:7] = obj_q
            self.object.write_root_pose_to_sim(obj[:, :7], env_ids)
            self.object.write_root_velocity_to_sim(obj[:, 7:], env_ids)
            self.obj_start_pos[env_ids] = obj_p
            self.obj_start_quat[env_ids] = obj_q
            # 双物体螺旋装配: Aux 按闭合相对位姿跟着 (抖动后的) 主体摆, 状态复位
            self._SA.reset_screw(self, env_ids)

        # ---- 机器人: 起步分支 (RSI) ----
        # ① 直接抓取起步 (prior 位姿, phase=GRASP) = **今天的任务**, 也是免费回归桶
        # ② 接近起步     (q_ref[t0], phase=PREGRASP), t0 ~ U(0, approach_t0_max·gs)
        # 两条课程 (direct_grasp_prob / approach_t0_max) 都由训练入口按 sr_ema 退火.
        c0 = torch.rand(n, device=dev) * cfg.closure_init_max
        _c0m = float(getattr(cfg, "closure_init_min", -1.0))
        if _c0m >= 0.0:
            c0 = torch.full_like(c0, _c0m)   # 诊断口径: 起步合拢度固定
        t0 = torch.full((n,), self.gs, dtype=torch.long, device=dev)
        start_grasp = torch.ones(n, dtype=torch.bool, device=dev)
        if cfg.approach:
            _dgp = float(cfg.direct_grasp_prob)
            _dhi = float(getattr(cfg, "dgp_rsi_hi", 0.0))
            if _dhi > 0.0:
                # RSI 课程 (2026-08-22 AAG-Local 验尸): 从头训时右手 100% 回合在
                # "臂抵达抓姿"瞬间拍倒瓶子(死于参考行 86), 永远体验不到合拢 ⟹
                # 早期大比例直抓起步先学"合拢→接触→稳抓", 随稳抓能力棘轮 _fcd_g
                # 退火回 direct_grasp_prob (外部建议 RSI 三段式的自动版)
                _dgp = _dhi + (_dgp - _dhi) * float(getattr(self, "_fcd_g", 0.0))
            start_grasp = torch.rand(n, device=dev) < _dgp
            # ⚠ 2026-08-16 一改一退, 记在这里免得有人再改一次:
            #
            # 我曾把 t0 的采样区间从 [0, ratio×gs) 改成"相对真接近段" [lo, gs), 动机是:
            # pour17 瓶 gs=71(站姿前缀占 0~59, 真人接近只有 60~70), [0, 0.774×71)=[0,54)
            # **100% 落在合成的站姿插值里**, 最后 3cm 的对准一次都没从那儿起步过。动机成立。
            #
            # **但那个改法把课程和评测的终点改反了, 必须回滚**:
            #   `eval_distribution()` 把 approach_t0_max 设成 0.0 表示"**从头做**"(t0=0),
            #   这正是正式口径"从对称站姿走完全程"的实现方式;
            #   而新写法在 ratio=0 时给出 lo=gs ⟹ **评测变成"直接空降到抓握帧"**。
            # **t0=0 是课程的目标状态, 不是要被退火掉的东西。**
            # ⚠ 更正(同日): 我一度用"冠军存档权重在当前代码上评出 0.00%"当作这条的实证,
            #   那是**误判** —— 真因是场景本身已按设计改过(物体听手把物体挪了 15cm,
            #   垫↔接触零位校准把腕位挪了 2.2cm), 冠军权重抓的是旧位置。
            #   语义论证本身仍成立(eval.py:75 把它设成 0.0 表示"从 q_ref[0] 出发"), 故保留回滚;
            #   但**冠军存档评测已不再是有效对照**, 场景一改它就失效。
            #
            # 原动机(真接近段覆盖不到)仍然成立, 但必须用**不动 ratio=0 端点**的办法解决,
            # 例如"以概率 p 额外从 [K, gs) 采一个 t0", 留待另开。
            hi = max(int(cfg.approach_t0_max * self.gs), 1)
            t0 = torch.where(start_grasp, t0,
                             torch.randint(0, hi, (n,), device=dev))
            c0 = torch.where(start_grasp, c0, torch.zeros_like(c0))   # 接近段手张开
        # ---- 退避式起点族 (取代沿人手轨迹采 t0; 见 docs/APPROACH_DESIGN.md §2) ----
        # ★ 语义(cfg 里写死了, 这里再说一遍): stance_prob=1 就是**正式任务口径**
        #   (全部从对称站姿起步); retract_ratio=0 最简单(全在 GraspPose 上)。
        #   两个都是"越大越难", 与 approach_t0_max 方向相反 —— 别混。
        use_retract = getattr(cfg, "retract_start", False) and \
            getattr(self, "retract_q", None) is not None
        if use_retract:
            # ★ 起点在**整条参考路径**上均匀抽 (2026-08-16 用户裁定)。
            #   之前是 {站姿} ∪ {退避族 24 档}, 中间那 59 帧前缀**一个起点都不落**,
            #   课程难度是断的: 要么"就在物体旁边", 要么"从最远的站姿从头做"。
            #   而评测(全站姿起步)恰恰必须穿过那 59 帧 —— 训练分布里的空白。
            #   现在 j0 ~ U(0, ratio×(L-1)): ratio=0 全在终点(最易), =1 铺满整条路(最难),
            #   站姿那一档自然就是 j0=0, 不再需要单独的 stance_prob 旋钮。
            _L = self.retract_path.shape[0]
            if getattr(self, "_sp_q", None) is not None:
                pass          # 起点由起点池决定 (见下面 use_pool 分支), 这里不再采 t0
            if float(getattr(cfg, "stance_prob", 0.0)) >= 1.0:
                # ★ 正式口径(评测): **全部从站姿出发**, 钉死 j0=0。
                #   不能靠"均匀抽恰好抽到 0"的概率 —— 评测必须是确定的分布。
                t0 = torch.zeros(n, dtype=torch.long, device=dev)
            else:
                # 两段式课程 (2026-08-16 用户裁定: "一开始多在近处, 能力上来了加大远端概率,
                # 直到完全退火")。路径下标: j0=L-1 是终点(最易), j0=0 是站姿(最难)。
                #   ① retract_ratio 0→1 放宽**下界**: 从"只在终点"扩到"铺满整条路"
                #   ② stance_prob   0→1 压低**上界**: 从"铺满"收到"全部在最远端"
                # ⚠ 第②段是我 2026-08-16 合并课程时**删掉的**(原 stance_prob), 结果
                #   拉满后永远是均匀分布, 最远那一档只占 1/84 —— 而评测 100% 考最远档。
                #   TB 0.37~0.55(均匀平均) vs 评测 0.00%(全最远) 就是这么来的。
                _hi = (1.0 - min(max(float(cfg.stance_prob), 0.0), 1.0)) * (_L - 1)
                _lo = min((1.0 - max(float(cfg.retract_ratio), 0.0)) * (_L - 1), _hi)
                j0 = (_lo + torch.rand(n, device=dev) * (_hi - _lo)).round().long()
                t0 = j0.clamp(0, _L - 1)
            # ★★ 2026-08-23 破案: 这里原本无条件 `start_grasp = zeros` ⟹ **直抓桶
            # (RSI) 被静默清零**, direct_grasp_prob / dgp_rsi_hi 全程空转 ——
            # 今晚设的 0.8/1.0 是空转, AAGPROBE(dgp0.95) 的旧结论第二次作废,
            # 也解释了 candidate 为何永不点亮: 从没有过"从抓姿起步"的回合, 策略
            # 必须先精通 100 步接近才可能碰到物体。
            # 修: 直抓桶保留 —— 其退避下标钉在**路径终点(=GraspPose)**, 合拢度不清零
            _dg_keep = start_grasp & (float(getattr(cfg, "dgp_rsi_hi", 0.0)) > 0.0
                                      or float(getattr(cfg, "direct_grasp_prob", 0.0)) > 0.0)
            t0 = torch.where(_dg_keep, torch.full_like(t0, _L - 1), t0)
            q_ret = self.retract_path[t0]                              # (n,7)
            start_grasp = _dg_keep                                      # 其余走接近相位
            c0 = torch.where(_dg_keep, c0, torch.zeros_like(c0))        # 接近段手张开
            # 盘面: 本回合起点离终点还有几帧 (越大越难)
            self.retract_d0[env_ids] = (_L - 1 - t0).float()
        tmpl0 = self.q_open + c0.unsqueeze(1) * (self.q_close - self.q_open)
        if self.arm_start_pool is None:
            q_arm = self.q_pregrasp.expand(n, 7).clone()
        else:                       # 从起始位形池随机抽 (见 __init__ 里 arm_start_pool 注释)
            k = torch.randint(len(self.arm_start_pool), (n,), device=dev)
            q_arm = self.arm_start_pool[k]
        if cfg.approach and not use_retract:
            # ⚠ 只在**非退避**模式下用 t0 索引 q_ref —— 退避模式的 t0 是 **retract_path**
            #   的下标(0~K+M-1), 而 q_ref 是人手轨迹(长度 L)。两者长度不同,
            #   拿前者去索引后者会 CUDA 越界(冒烟当场崩, device-side assert)。
            q_arm = torch.where(start_grasp.unsqueeze(1), q_arm, self.q_ref[t0])
        if use_retract:
            q_arm = q_ret                     # 退避族/站姿 覆盖上面的参考起点
        if getattr(self, "_sp_q", None) is not None:
            # ★ 起点池优先于以上一切 —— 入池用的是**上一回合**的最近点, 所以先 push 再 sample
            self._sp_push(env_ids)
            q_arm, _b = self._sp_sample(n, dev)
            self._sp_start_bin[env_ids] = _b
            start_grasp = torch.zeros(n, dtype=torch.bool, device=dev)
            c0 = torch.zeros_like(c0)
            tmpl0 = self.q_open + c0.unsqueeze(1) * (self.q_close - self.q_open)
        # 本回合的"最近点"重新开始记
        if getattr(self, "_sp_best_d", None) is not None:
            self._sp_best_d[env_ids] = 1e9
        q = self.hand.data.default_joint_pos[env_ids].clone()
        q[:, self.arm_jids] = q_arm
        q[:, self.hand_jids] = tmpl0
        self.hand.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.hand.set_joint_position_target(q, env_ids=env_ids)
        self.q_cmd[env_ids] = q_arm
        self.arm_tgt[env_ids] = q_arm
        self.arm_tgt_prev[env_ids] = q_arm
        self.finger_tgt[env_ids] = tmpl0
        self.finger_tgt_prev[env_ids] = tmpl0

        # ---- 任务状态清零 ----
        self.freeze_ctr[env_ids] = cfg.settle_steps
        self.closure[env_ids] = c0
        self._grip_g[env_ids] = 0.0            # 信任度随回合清零
        # ★ 2026-08-26: 势差分类项的 prev_* 复位时若留 0, 首步会算出一个巨大假差分
        #   (align: prev_phi=0 而实际势 ≈ −0.4 ⟹ 首步吃满 clamp −0.12, 每回合一次)。
        #   复位时把 prev 初始化成**当前值** ⟹ 首步差分 = 0, 从第二步起才有真差分。
        self.prev_phi[env_ids] = -(
            (self._anchor_w() - self._target_w()).norm(dim=1)[env_ids]
            if getattr(self, "_grasp_pos_w", None) is None else
            torch.zeros(len(env_ids), device=self.device))
        self.prev_wrist_pos[env_ids] = (self.wrist_pos_w
                                        - self.scene.env_origins)[env_ids]
        self._grip_has[env_ids] = False
        self.fin_delta[env_ids] = 0.0
        self.fin_res[env_ids] = 0.0
        self.arm_res[env_ids] = 0.0            # 方案C: 臂累积残差随回合清零
        self.task_phase[env_ids] = torch.where(
            start_grasp, torch.full_like(t0, Phase.GRASP),
            torch.full_like(t0, Phase.PREGRASP))
        # 接近段状态
        if getattr(self, "_fin_ref_path", None) is not None and \
                int(getattr(cfg, "aag_grasp_row", 0)) > 0:
            # ★ AAG 直抓桶的参考行对表修正 (2026-08-22 验尸): t0=gs 是**人手原轨迹**
            # 的 PreGrasp 帧号(25), 而 AAG 指参考是另一套 180 行编舞 —— 其第 25 行
            # 是"手指全伸直"。于是直抓起步(手已成抓形)被 fin_track/form_pot 要求
            # **张开手**, 左手 fin_track 一片 -61。dgp 0.3→0.8 后放大成崩盘;
            # 同一 bug 解释 AAGPROBE(dgp0.95, 4M candidate 全零) 的旧悬案。
            # 修正: 直抓桶从**合拢段起始行**开跑 (臂 ff 已被 clamp 在终点, 不受影响)
            # ★ 自动对表 (2026-08-23): 起始参考行 = **与初始手型最接近的那一行**
            # (在 [aag_grasp_row, fin_end_row] 内搜), 而不是猜一个固定行 ——
            # 手型与参考行不匹配 ⟹ "跟上才前进"永不前进 + fin_track 狂罚 (诊断
            # 实测 c0=1.0 起步时左手 fin_track −91.6), 这是同一类 bug 的第三次
            _lo6 = int(cfg.aag_grasp_row); _hi6 = int(getattr(cfg, "fin_end_row", 175))
            _cand = self._fin_ref_path[_lo6:_hi6 + 1]           # (K,22)
            _q_init = tmpl0 if tmpl0.dim() == 2 else tmpl0.unsqueeze(0)
            _dmat = (_q_init.unsqueeze(1) - _cand.unsqueeze(0)).abs().mean(dim=2)
            _best = _lo6 + _dmat.argmin(dim=1)
            t0 = torch.where(start_grasp, _best.to(t0.dtype), t0)
            if not getattr(self, "_am_dbg", False):
                object.__setattr__(self, "_am_dbg", True)
                print(f"[对表] side={getattr(getattr(self,'_cur',None),'name','?')} "
                      f"搜索行[{_lo6},{_hi6}] 命中中位 {int(_best.float().median())} "
                      f"| 直抓比例 {float(start_grasp.float().mean()):.2f} "
                      f"| _dgp={_dgp:.2f} cfg.dgp={float(cfg.direct_grasp_prob):.2f} "
                      f"hi={float(getattr(cfg,'dgp_rsi_hi',0)):.2f} "
                      f"g={float(getattr(self,'_fcd_g',0)):.2f} n={n}", flush=True)
        self.ref_t[env_ids] = t0
        self.started_grasp[env_ids] = start_grasp
        self.arrived[env_ids] = start_grasp   # 抓取起步 = 天然已到位
        self.arrive_step_abs[env_ids] = 0
        if getattr(self, "_p2_t", None) is not None:
            self._p2_t[env_ids] = 0           # Phase2 斜坡进度清零 (遗产)
        self._g2_run[env_ids] = 0             # Phase2-RL 第二段判据/里程碑清零
        self._g2_done[env_ids] = False
        self._m1_done[env_ids] = False
        if getattr(self, "_fcd_close_t", None) is not None:
            # FC-D 参考退火 (2026-08-20 用户裁定, 冠军配方同款纪律): 参考拉力随
            # "稳抓能力"退火 —— 驱动 = candidate 率慢 EMA(0.98), 棘轮只退不返,
            # 逐侧独立(左右各按自己的能力退)。到位目标 cand_ema=0.3 时拉力归零。
            # 采样必须在 got_candidate 清零(下方)之前。
            _frac = float(self.got_candidate[env_ids].float().mean()) \
                if len(env_ids) else 0.0
            self._fcd_cand_ema = 0.98 * float(getattr(self, "_fcd_cand_ema", 0.0)) \
                + 0.02 * _frac
            self._fcd_g = max(float(getattr(self, "_fcd_g", 0.0)),
                              min(self._fcd_cand_ema / 0.3, 1.0))
            self._fcd_close_t[env_ids] = 0
            self._fcd_shape_done[env_ids] = False
            self._fcd_pad_done[env_ids] = False
        if getattr(self, "_c5_pad_done", None) is not None:
            self._c5_pad_done[env_ids] = False
        if getattr(self, "_cc_done", None) is not None:
            self._cc_run[env_ids] = 0
            self._cc_done[env_ids] = False
        self._m2_done[env_ids] = False
        if getattr(self, "_m2_lvl", None) is not None:
            self._m2_lvl[env_ids] = 0         # m2 阶梯回到第一级 (裁定C)
        if getattr(self, "_fc_done", None) is not None:
            self._fc_done[env_ids] = False    # FC 逐指面包屑复位
            self._fc_prev[env_ids] = float("nan")
        self.arrive_step[env_ids] = 0
        # ⚠ 前馈是差分 (q_base − ref_q_prev), 所以复位时 ref_q_prev 必须落在**同一条**
        #   参考路径的 t0 上 —— 落错路径的话第一步前馈是个跳变。
        self.ref_q_prev[env_ids] = (self.retract_path[t0]
                                    if getattr(self, "retract_path", None) is not None
                                    else self.q_ref[t0])
        self.switch_run[env_ids] = 0
        self.prev_phi[env_ids] = 0.0
        self.prev_wrist_pos[env_ids] = 0.0
        self.prev_valid[env_ids] = False            # 首步没有上一帧, 该步不罚
        # 抓取相位的偏差带回到"以 prior 位姿为中心"的默认 (接近段用不到, 切换时再设)
        self.arm_center[env_ids] = self.q_pregrasp
        self.band_lo[env_ids] = self.arm_dev_lo
        self.band_hi[env_ids] = self.arm_dev_hi
        # ⚠ 抖动池起步 (§2.17): 带子必须以**实际起步位形**为中心 —— 池样本可以偏出
        # ±arm_dev_max, 带子钉在 q_pregrasp 会让第一步 clamp 把臂猛甩回带边
        # (与 _set_arm_center 注释里"切换被一把拽回"同一失败模式, reset 路径同罪).
        # _set_arm_center 自带"带子始终罩住抓握 IK 解"的保证, 合拢可达性不受影响.
        if self.arm_start_pool is not None and start_grasp.any():
            _gm = torch.zeros(self.num_envs, dtype=torch.bool, device=q_arm.device)
            _gm[env_ids[start_grasp]] = True
            _gq = self.arm_center.clone()
            _gq[env_ids[start_grasp]] = q_arm[start_grasp]
            self._set_arm_center(_gm, _gq)
        self.phase_step[env_ids] = 0
        self.cand_run[env_ids] = 0
        self.verify_k[env_ids] = 0
        self.verify_ok_run[env_ids] = 0
        self.got_candidate[env_ids] = False
        if cfg.place_task:
            self.settle_ctr[env_ids] = 0
            self.carry_prev_d[env_ids] = 0.0
        self.succeeded[env_ids] = False
        self._diag_contacts[env_ids] = 0.0
        self._diag_actnorm[env_ids] = 0.0
        self._diag_n[env_ids] = 0.0
        self._diag_fgate[env_ids] = 0.0
        self._diag_fgate_n[env_ids] = 0.0
        self.pad_touched[env_ids] = False
        self.any_contact[env_ids] = False
        self.tilt_final_deg[env_ids] = self.tilt_max_deg[env_ids]
        self.yaw_final_deg[env_ids] = self.yaw_max_deg[env_ids]
        self.tilt_max_deg[env_ids] = 0.0
        self.yaw_max_deg[env_ids] = 0.0
        self.arm_err_ctr[env_ids] = 0
        self.actions_buf[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.proprio_hist[env_ids] = 0.0
        self.prev_pad_d[env_ids] = 0.0
        self.prev_quality[env_ids] = 0.0

    # ---- 接触分数图: 对外接口 -----------------------------------------
    def decay_score(self):
        """折扣一次 (训练入口每 epoch 调). 早期弱策略的标签不会永久压住后期数据."""
        if self.score_on:
            self.score_s *= self.cfg.score_decay
            self.score_n *= self.cfg.score_decay

    def save_score_map(self, path, extra_meta=None):
        """落盘成 AffordanceModel 的统一 npz 格式 (DATA_FORMAT.md) + 我们的 (s,n).

        heatmap = (s+1)/(n+2)  Beta 后验均值; weight = 0 表示 **unknown**(没试够),
        **不是 0 分** —— 训练时必须用 weight 做逐点 mask, 否则没接触过的点会被当负样本.
        """
        import json
        import os
        if not self.score_on:
            return None
        n = self.score_n.cpu().numpy()
        sc = self.score_s.cpu().numpy()
        meta = dict(hand=self.cfg.hand_side, task="pick_lift",
                    clip=self.cfg.clip_name, prior=os.path.basename(self.cfg.grasp_prior_npz),
                    template="fingertip_mid", n_min=float(self.cfg.score_n_min),
                    note="support = 候选池接触区 ∪ ±2cm; 其余为外推")
        if extra_meta:
            meta.update(extra_meta)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path,
                 points=self.obj_points.cpu().numpy(),
                 points_raw=self.obj_points.cpu().numpy(),
                 normals=(self.obj_normals.cpu().numpy() if self.obj_normals is not None
                          else np.zeros_like(self.obj_points.cpu().numpy())),
                 heatmap=((sc + 1.0) / (n + 2.0)).astype(np.float32),
                 weight=(n >= self.cfg.score_n_min).astype(np.float32),
                 s=sc.astype(np.float32), n=n.astype(np.float32),
                 meta=np.array(json.dumps(meta, ensure_ascii=False), dtype=object))
        print(f"[score] 已写出 {path} | 有效点 {int((n >= self.cfg.score_n_min).sum())}"
              f"/{len(n)} | 访问总量 {n.sum():.0f}")
        return path
