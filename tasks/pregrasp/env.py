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


class GraspTaskEnv(DexmateCorrectionEnv):
    cfg: GraspTaskCfg

    def __init__(self, cfg: GraspTaskCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
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
            self.L = int(self.L) + _K          # 基类 clip 长度与 q_ref 保持一致
            _d = float((self.ref_wrist_pos[_K] - self.ref_wrist_pos[0]).norm()) * 100
            print(f"[stance_prefix] K={_K} 帧 | 站姿->轨迹起点 腕直线 {_d:.1f}cm "
                  f"(峰值 ≈{_d * 15.0 / max(_K, 1):.1f}mm/帧, 上限 15) | "
                  f"gs {self.grasp_start - _K} -> {self.grasp_start}")

        # ---- PreGrasp 姿态 (臂参考钉死在这一帧) ----
        self.q_pregrasp = self.q_ref[self.grasp_start].clone()          # (7,)
        assert getattr(self, "q_lift", None) is not None, \
            "微抬升验证依赖基类预解的 q_lift (place_mode=ref_builder 路径)"

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
        assert e.get("affordance"), \
            f"{cfg.clip_name} 没有 affordance —— 对齐目标取自 affordance 重心"
        self.aff_local = to(_affordance_target(e["affordance"]))        # (3,)

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
        self.arm_bids = [i for i, n in enumerate(bn)
                         if f"{P}_arm_l" in n or n == f"{P}_ee"]
        self.self_bids = [i for i, n in enumerate(bn)
                          if "torso_l" in n or "head_l" in n or n == "vega_1p_base"
                          or f"{O}_arm_l" in n or n == f"{O}_ee"]
        print(f"[collide] 臂连杆 {len(self.arm_bids)} 个 "
              f"({[bn[i] for i in self.arm_bids]}) | 需避让 {len(self.self_bids)} 个 "
              f"(躯干/头/另一臂)")

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
        lvl = cfg.verify_lift_m / (self.cfg.lift_height / self.cfg.lift_steps)
        self.verify_lvl = float(min(lvl, self.cfg.lift_steps))          # 例 1cm -> 4 级
        self.q_lift_delta = self.q_lift - self.q_lift[0:1]              # (21,7)

        # ---- 任务状态 ----
        self.closure = torch.zeros(N, device=dev)                       # 合拢 c
        self.fin_delta = torch.zeros(N, 5, device=dev)                  # 每指残差 δ
        self.task_phase = torch.full((N,), Phase.GRASP, dtype=torch.long, device=dev)
        self.phase_step = torch.zeros(N, dtype=torch.long, device=dev)
        self.cand_run = torch.zeros(N, dtype=torch.long, device=dev)    # 候选判据连续计数
        self.verify_k = torch.zeros(N, dtype=torch.long, device=dev)    # 验证段步数
        self.verify_ok_run = torch.zeros(N, dtype=torch.long, device=dev)
        self.got_candidate = torch.zeros(N, dtype=torch.bool, device=dev)
        self.succeeded = torch.zeros(N, dtype=torch.bool, device=dev)
        self.pad_touched = torch.zeros(N, 5, dtype=torch.bool, device=dev)
        self.any_contact = torch.zeros(N, dtype=torch.bool, device=dev)
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
        self.arrive_step = torch.zeros(N, dtype=torch.long, device=dev)   # 用了多少步到位
        # ---- 验证段诊断 (2026-08-02): 判据是"物体升 ≥5mm"(绝对), 而它的严格程度
        # 其实取决于**腕实际抬了多少** —— 腕抬 6.35mm 时它等于要求 79% 跟随率,
        # 腕抬 10mm 时只要求 50%. 而 q_lift 只是**指令**, 抓着物体带接触力时臂实际
        # 能抬多少从没测过. 先把三个量记下来, 再决定要不要把判据改成相对口径.
        self.verify_wz0 = torch.zeros(N, device=dev)      # 进验证段那一刻的腕高
        self.verify_oz0 = torch.zeros(N, device=dev)      # 进验证段那一刻的物体高
        self.vf_wrist_mm = torch.zeros(N, device=dev)     # 斜坡到顶时腕实际抬了多少
        self.vf_obj_mm = torch.zeros(N, device=dev)       # 同一刻物体实际升了多少
        self.vf_has = torch.zeros(N, dtype=torch.bool, device=dev)
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

        # 笨拙课程标量 (训练入口每迭代按 sr_ema 更新; 评测/冒烟脚本应显式置 1.0)
        self.gentle = float(cfg.gentle_init)
        self.ep_total = (cfg.settle_steps + int(sum(cfg.phase_timeout)) + 8)
        print(f"[pregrasp] v2 稳定抓握+微抬升验证: 起点=PreGrasp(帧{self.grasp_start}) | "
              f"合拢参考 {cfg.closure_ref_rate}/步 | 候选 ≥{cfg.success_min_pads}垫 "
              f"向心≥{cfg.grasp_centrip_thresh} 保持{cfg.candidate_hold_steps}步 | "
              f"验证升 {cfg.verify_lift_m*1000:.0f}mm ({self.verify_lvl:.0f}级) | "
              f"episode ≤ {self.ep_total} 步")

    def _load_grasp_prior(self, npz_path, to):
        """Dexonomy GraspPose -> ① q_pregrasp ② q_close ③ aff_local (见 cfg 注释).

        抓姿在**物体输入系** (make_prior.py 已从规范系转回, wxyz 约定已自检),
        这里按 env 的物体初始位姿变换到 env 系, 再用基类同款 ArmIK (实测锚) 解臂关节.
        """
        from isaaclab.utils.math import quat_apply as _qa, quat_mul as _qm
        from rl_rebuild.correction.kinematics import ArmIK, quat_to_R

        cfg, dev = self.cfg, self.device
        z = np.load(npz_path)
        oq = self.obj_init_quat.cpu().numpy().astype(np.float64)
        op = self.obj_init_pos.cpu().numpy().astype(np.float64)
        # ---- 物体姿态对齐 Dexonomy 规范姿态 ----
        # 抓姿是在"物体按规范姿态平放桌面"时生成的; 而 ref builder 用"最大支撑面朝下"
        # 独立决定姿态, 对非对称物体可能选到**另一个面** (Grasp3 实测两者差 179.9°,
        # 物体上下颠倒 -> 抓姿腕落到桌面下 12cm, 所有 yaw 都 IK 失败).
        # 有 prior 时以 Dexonomy 姿态为准, xy 保留 ref builder 的相机锚定结果.
        if "canon_rot" in z.files:
            from rl_rebuild.correction import frames as _F
            oq = np.asarray(z["canon_rot"], np.float64)
            _v = _F.load_obj_verts(self.du.mesh_path) @ quat_to_R(oq).T
            op = op.copy()
            op[2] = cfg.table_top_z + 0.002 - float(_v[:, 2].min())   # 最低点贴桌 +2mm
            self.obj_init_quat, self.obj_init_pos = to(oq), to(op)
            print(f"[prior] 物体按 Dexonomy 规范姿态摆放 (原点 z={op[2]:.4f}, "
                  f"高度 {(_v[:,2].max()-_v[:,2].min())*100:.2f}cm)")
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
            ) @ z["grasp"][:3] + op
            gq_t = self._qmul_np(self._qmul_np(
                np.array([np.cos(best_yaw / 2), 0.0, 0.0, np.sin(best_yaw / 2)]), oq),
                z["grasp"][3:7])
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
            for deg in range(0, 360, 5):
                a = np.radians(deg)
                yq = np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])
                oq2 = self._qmul_np(yq, oq)
                gp_t = quat_to_R(oq2) @ z["grasp"][:3] + op
                gq_t = self._qmul_np(oq2, z["grasp"][3:7])
                r = ik.solve(gp_t, quat_to_R(gq_t), q0=qwarm, iters=80)
                err = r["pos_err"] + 0.3 * r["rot_err"]
                if oh_pos is not None:      # 离非交互手 <25cm 开始加罚 (软约束)
                    gap = float(np.linalg.norm(gp_t - oh_pos))
                    err += 2.0 * max(0.0, 0.25 - gap)
                if r["ok"]:
                    qwarm = r["q"]
                if err < best_err:
                    best_yaw, best_err = a, err
        if best_yaw != 0.0:
            yq = np.array([np.cos(best_yaw / 2), 0.0, 0.0, np.sin(best_yaw / 2)])
            oq = self._qmul_np(yq, oq)
            Ro = quat_to_R(oq)
            self.obj_init_quat = to(oq)          # 物体复位姿态跟着转
            print(f"[prior] 物体 yaw 旋转 {np.degrees(best_yaw):.0f}° 使抓姿朝向机械臂"
                  f" (绕竖轴旋转, 物体仍平放于桌面)")
        # 抓握位姿 IK (在选定 yaw 下精解)
        gp, gq = to_env(z["grasp"])
        rg = ik.solve(gp, quat_to_R(gq), q0=qwarm)
        # pregrasp: 逐行试, 取 IK 成功且离抓握解最近的一行 (6 个候选)
        best = None
        for row in z["pregrasp"]:
            pp, pq = to_env(row)
            r = ik.solve(pp, quat_to_R(pq), q0=rg["q"])
            if r["ok"]:
                d = float(np.abs(r["q"] - rg["q"]).max())
                if best is None or d < best[0]:
                    best = (d, r)
        assert rg["ok"] and best is not None, \
            f"GraspPose prior IK 失败 (grasp ok={rg['ok']}): 该抓姿在本场景不可达, 换候选"
        rp = best[1]

        self._prior_q_grasp = to(rg["q"])
        # 抓握位姿的**世界位姿** —— 接近段的对齐势函数要用它当终点 (物体复位在 obj_init_*,
        # 训练期只有 ±3~15mm 抖动, 所以这里当常量存; 抖动由势函数自己吸收).
        self._grasp_pos_w = to(gp)                          # (3,)
        self._grasp_quat_w = to(gq)                         # (4,) wxyz
        self.q_pregrasp = to(rp["q"])                       # ① 臂复位/参考位
        # ⚠ npy 的 22 维是 GENERIC_JOINT_ORDER 序, 必须换到 USD 关节序再进模板
        # (2026-07-30 踩坑: 漏换序 -> 手型全串位, B 组整条 run 作废)
        fingers = np.clip(z["grasp"][7:29][self._generic_perm],
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
        lift, qw, n_ik_fail = [], rg["q"].copy(), 0
        for i in range(cfg.lift_steps + 1):
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
        res = a[:, 0:7] * self.arm_res_scale
        if self.cfg.approach:
            # 前馈 = 参考轨迹**自己**这一步走了多少 (只出现差分, 绝对位置不进公式).
            # 零动作 ⟹ 逐帧复现人手的速度剖面 —— 这条性质是 repo 两次成功的共同依赖
            # (dexmate_env.py:384-392 的注释: 只累积不给前馈 ⟹ 手臂冻在起点).
            t = self.ref_t.clamp(max=self.gs)
            q_base = self.q_ref[t]
            ff = (q_base - self.ref_q_prev) * (in_app & (self.ref_t < self.gs)
                                               ).float().unsqueeze(1)
            self.ref_q_prev = q_base
            # 远松近紧: 手要自己走完那 ~16cm, 但接触前必须回到毫米级
            d_pos = (self._anchor_w() - self._target_w()).norm(dim=1)
            u = ((d_pos - self.cfg.dyn_d_near)
                 / max(self.cfg.dyn_d_far - self.cfg.dyn_d_near, 1e-6)).clamp(0.0, 1.0)
            s_arm = 1.0 + u * (self.cfg.dyn_arm_far - 1.0) * in_app.float()
            res = res * s_arm.unsqueeze(1)
            self.res_step_cm = (res.abs().mean(dim=1) * 100.0)          # 诊断: 残差用量
            q_new = self.q_cmd + ff * gate.unsqueeze(1) + res
            self.q_cmd = torch.where(in_app.unsqueeze(1),
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

        # 合拢: 参考斜坡 + 策略调制 (零动作 = 匀速合到 c_grasp 停; a_c=-1 停住;
        # 比 nominal 更深的挤压只能由 a_c>0 主动选择)
        # ⚠ 接近段整条手指通道**门控关闭** —— 手必须张开着飞过去, 否则是握着拳头去碰物体.
        hand_gate = gate * (~in_app).float()
        ref = cfg.closure_ref_rate * (self.closure < cfg.c_grasp).float()
        self.closure = (self.closure
                        + hand_gate * (ref + a[:, 7] * cfg.closure_rate_max)
                        ).clamp(cfg.closure_min, cfg.closure_max)
        # 每指残差: 指 i 的模板深度 = clip(c + δ_i)
        self.fin_delta = (self.fin_delta
                          + a[:, 8:13] * cfg.delta_rate_max * hand_gate.unsqueeze(1)
                          ).clamp(-cfg.delta_max, cfg.delta_max)
        c_i = (self.closure.unsqueeze(1) + self.fin_delta
               ).clamp(cfg.closure_min, cfg.closure_max)                # (N,5)
        cj = c_i[:, self.fmap]                                          # (N,22)
        self.finger_tgt_prev = self.finger_tgt.clone()
        self.finger_tgt = (self.q_open + cj * (self.q_close - self.q_open)
                           ).clamp(self.dof_lower, self.dof_upper)
        self._substep = 0
        # _apply_action 继承 DexMate 的子步插值下发, 零改动

    # ---- 任务几何 -----------------------------------------------------
    def _anchor_w(self) -> torch.Tensor:
        return self.wrist_pos_w + quat_apply(
            self.wrist_quat_w, self.anchor_local.expand(self.num_envs, 3))

    def _target_w(self) -> torch.Tensor:
        return self.object.data.root_pos_w + quat_apply(
            self.object.data.root_quat_w, self.aff_local.expand(self.num_envs, 3))

    def _pad_dists(self) -> torch.Tensor:
        """(N,5) 每个软垫到物体表面采样点的最近距离."""
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
        tips = self.tip_pos_w
        c = self.object.data.root_pos_w[:, None, :]
        Gh = G / mag.unsqueeze(-1).clamp(min=1e-6)
        if normals is not None:                 # 局部法向版: 力是否压进表面
            u = -normals                        # 外法向取反 = "压入"方向
        else:                                   # 旧版: 指向物体原点
            u = c - tips
            u = u / u.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        cent = (((Gh * u).sum(-1)).clamp(min=0.0) * e).sum(dim=1) / 5.0
        tot = mag.sum(dim=1)
        r_imb = G.sum(dim=1).norm(dim=1) / (tot + 1e-6)
        tau = torch.cross(tips - c, G, dim=-1).sum(dim=1)
        tau_n = (tau.norm(dim=1) / ((tot + 1e-6) * cfg.obj_char_radius)).clamp(max=2.0)
        return mag, e, cent, r_imb, tau_n, G

    # ---- 阶段机 + 终止 (dones 先于 rewards 被调, 信号缓存给后者) ----
    def _get_dones(self):
        cfg = self.cfg
        origins = self.scene.env_origins
        active = self.episode_length_buf >= cfg.settle_steps
        obj_pos = self.object.data.root_pos_w - origins
        obj_spd = self.object.data.root_lin_vel_w.norm(dim=1).nan_to_num(1e3)
        obj_rot = self.object.data.root_ang_vel_w.norm(dim=1).nan_to_num(1e3)
        pad_d, pad_n, pad_i = self._pad_dist_normal()
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

        # ---- 撞桌 (几何) + 指间交叉 (几何) ----
        # ⚠ 铰链必须钳上限 (台账铁律"惩罚项永远要有单步上界"): 物理发散时 body 掉到
        # 桌下几百米, 深度² × 40 body × 20 权重 = 百万量级, 实测把回合奖励打到 -115万
        # 并毒掉 value 归一化器 (run 03-37-46). 深穿本身由 table_crash 终止兜底.
        z = (self.hand.data.body_pos_w[:, self.table_bids, 2] - origins[:, 2:3]
             ).nan_to_num(nan=cfg.table_top_z)
        table_pen = (cfg.table_top_z + cfg.table_margin - z).clamp(min=0.0, max=0.05)
        table_pen = table_pen.square().sum(dim=1)
        table_crash = active & \
            (z.min(dim=1).values < cfg.table_top_z - cfg.table_crash_depth)
        bp = self.hand.data.body_pos_w
        cross_d = (bp[:, self.cross_a] - bp[:, self.cross_b]).norm(dim=-1)
        cross_pen = (cfg.finger_cross_dist - cross_d).clamp(min=0.0).sum(dim=1)
        # ---- P0.3: 全臂撞桌 + 臂↔躯干/另一臂 间隙 (诊断常开, 罚由 w_* 控制) ----
        arm_p = bp[:, self.arm_bids]                                    # (N,A,3)
        z_arm = (arm_p[:, :, 2] - origins[:, 2:3]).nan_to_num(nan=cfg.table_top_z)
        arm_table_pen = (cfg.table_top_z + cfg.arm_table_margin - z_arm
                         ).clamp(min=0.0, max=0.05).square().sum(dim=1)
        arm_gap = (z_arm - cfg.table_top_z).amin(dim=1)                 # 诊断: 最低臂连杆离桌
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
            self.ref_t = torch.where(in_app & active, self.ref_t + 1, self.ref_t)
            ok = in_app & active & (d_pos < cfg.eps_pos) & (d_rot < cfg.eps_rot) & \
                (self.wrist_linvel_w.norm(dim=1) < cfg.switch_vel_max)
            self.switch_run = torch.where(ok, self.switch_run + 1,
                                          torch.zeros_like(self.switch_run))
            to_grasp = in_app & (self.switch_run >= cfg.switch_hold)
            newly_arrive = to_grasp & ~self.arrived
            if to_grasp.any():
                # 偏差带改以"到达位形"为中心, 否则切换会把目标拽回 prior 位形
                self._set_arm_center(to_grasp, self.q_cmd)
                ph = torch.where(to_grasp, torch.full_like(ph, Phase.GRASP), ph)
                self.arrive_step = torch.where(newly_arrive, self.ref_t, self.arrive_step)
                self.arrived |= to_grasp
        else:
            newly_arrive = torch.zeros_like(active)

        # ---- 候选抓取 (GRASP 段瞬时判据) ----
        cand_ok = active & (ph == Phase.GRASP) & (n_pads >= cfg.success_min_pads) & \
            (cent >= cfg.grasp_centrip_thresh) & (rel_spd < cfg.grip_slip_vel) & \
            (obj_rot < cfg.grip_rot_max) & (disp < cfg.push_fail_dist)
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
        _at_top = in_verify & (self.verify_k == cfg.verify_ramp_steps)
        if _at_top.any():
            self.vf_wrist_mm = torch.where(_at_top, (_wz - self.verify_wz0) * 1000,
                                           self.vf_wrist_mm)
            self.vf_obj_mm = torch.where(_at_top, (obj_pos[:, 2] - self.verify_oz0) * 1000,
                                         self.vf_obj_mm)
            self.vf_has |= _at_top

        ramp_done = self.verify_k >= cfg.verify_ramp_steps
        vf_ok = in_verify & ramp_done & (rise >= cfg.verify_min_rise) & \
            (n_pads >= cfg.verify_min_pads) & (rel_spd < 2 * cfg.grip_slip_vel)
        self.verify_ok_run = torch.where(vf_ok, self.verify_ok_run + 1,
                                         torch.zeros_like(self.verify_ok_run))
        success = self.verify_ok_run >= cfg.verify_hold_steps
        newly_success = success & ~self.succeeded
        self.succeeded |= success
        # 验证失败: ① 斜坡到顶+3 步物体没跟上来 (rise<3mm); ② **尝试预算耗尽**
        # (v2.8): rise 落在 3~5mm 的"半滑"灰区既不成功也不失败, 会挂到 LIFT 超时
        # 把回合截断 —— 实测 86% 候选 × 58 步回合 × 86% 超时全耗在这. 预算 =
        # 斜坡+保持+5 步, 到时未成功一律判负退回重试, 验证段不存在无判决状态.
        # **不终止**: 退回 GRASP 重试 (v2.3). 候选计数清零, 重新攒资格.
        verify_fail = in_verify & (
            ((self.verify_k >= cfg.verify_ramp_steps + 3)
             & (rise < cfg.verify_fail_rise))
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
        thrown = active & (obj_pos[:, 2] > cfg.table_top_z + cfg.max_obj_height)
        # 推走终止距离随笨拙课程收紧: gentle 0.2 -> 10cm (学习期宽容), 1.0 -> 4cm
        g01 = (self.gentle - 0.2) / 0.8
        push_dist = cfg.push_fail_dist_loose + \
            (cfg.push_fail_dist - cfg.push_fail_dist_loose) * g01
        pushed = active & ~self.got_candidate & (disp > push_dist)
        off = (self.arm_q - self.arm_tgt).abs().max(dim=1).values >= cfg.term_arm_err
        self.arm_err_ctr = torch.where(off, self.arm_err_ctr + 1,
                                       torch.zeros_like(self.arm_err_ctr))
        stuck = self.arm_err_ctr >= cfg.term_arm_steps

        terminated = fell | thrown | pushed | stuck | table_crash | success
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

        self._sig = dict(active=active, obj_pos=obj_pos, obj_spd=obj_spd,
                         obj_rot=obj_rot, pad_d=pad_d, mag=mag, e_pad=e_pad,
                         cent=cent, r_imb=r_imb, tau_n=tau_n, quality=quality,
                         pads_on=pads_on, n_pads=n_pads, newly_touch=newly_touch,
                         table_pen=table_pen, cross_pen=cross_pen, disp=disp,
                         rise=rise, cand_ok=cand_ok, newly_cand=newly_cand,
                         newly_success=newly_success, verify_fail=verify_fail,
                         fell=fell, thrown=thrown, pushed=pushed, stuck=stuck,
                         table_crash=table_crash, timeout=timeout,
                         d_pos=d_pos, d_rot=d_rot,
                         arm_table_pen=arm_table_pen, self_pen=self_pen,
                         arm_gap=arm_gap, self_gap=self_gap,
                         newly_arrive=newly_arrive)
        return terminated, truncated

    # ---- 奖励 ---------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        cfg, s = self.cfg, self._sig
        active = s["active"]
        af = active.float()

        newly = active & (self.episode_length_buf == cfg.settle_steps)
        if newly.any():
            self.prev_pad_d[newly] = s["pad_d"][newly]
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
            w_imit = (cfg.w_imit0 * cfg.w_imit_ramp
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

        # -- 密集: 逐垫接近进度 (有效接触数够了就关, 该拿质量分了) --
        d_pad = (self.prev_pad_d - s["pad_d"]).clamp(-0.05, 0.05).mean(dim=1)
        not_app = (self.task_phase != Phase.PREGRASP).float()
        terms["pad_approach"] = cfg.w_pad_approach * d_pad * not_app * \
            (s["n_pads"] < cfg.success_min_pads).float()
        # -- 稀疏: 每垫首次有效接触 (接近段不发: 那时碰到物体是坏事, 由 push/obj_move 管) --
        terms["pad_touch"] = cfg.r_pad_touch * not_app * \
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
        terms["milestone"] = (cfg.r_candidate * s["newly_cand"].float()
                              + cfg.r_success * s["newly_success"].float())
        # -- 失败罚 (verify_fail 不终止, 只小额罚 + 退回 GRASP 重试) --
        terms["fail"] = (cfg.r_drop * (s["fell"] | s["thrown"] | s["table_crash"]).float()
                         + cfg.r_pushed_away * s["pushed"].float()
                         + cfg.r_verify_fail * s["verify_fail"].float())
        # -- 正则 (速度先去 NaN 再钳: 物理尖峰的平方会淹没任务信号, 见台账) --
        terms["act_rate"] = -cfg.w_act_rate * \
            (self.actions_buf - self.prev_actions).square().mean(dim=1)
        qd_arm = self.arm_qd.nan_to_num(0.0)
        qd_fin = self.finger_qd.nan_to_num(0.0)
        qd = torch.cat([qd_arm / cfg.qd_soft_arm, qd_fin / cfg.qd_soft_fin], dim=1)
        terms["qvel"] = -cfg.w_qvel * qd.square().mean(dim=1).clamp(max=10.0)
        over = torch.cat([(qd_arm.abs() - cfg.qd_soft_arm).clamp(min=0.0),
                          (qd_fin.abs() - cfg.qd_soft_fin).clamp(min=0.0)], dim=1)
        terms["qvel_hard"] = -cfg.w_qvel_hard * \
            over.square().sum(dim=1).clamp(max=cfg.qvel_hard_cap)
        terms["torque"] = -cfg.w_torque * self.arm_torque_norm.square().mean(dim=1)
        terms["time"] = torch.full_like(af, -cfg.w_time)

        total = (sum(terms.values()) * af).nan_to_num(0.0)   # 最后一道 NaN 闸
        self.prev_pad_d = torch.where(active.unsqueeze(1), s["pad_d"], self.prev_pad_d)
        self.prev_quality = torch.where(active, s["quality"], self.prev_quality)
        for k, v in terms.items():
            if k not in self._ep_sums:
                self._ep_sums[k] = torch.zeros(self.num_envs, device=self.device)
            self._ep_sums[k] += v * af
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
            (self.closure / cfg.closure_max).unsqueeze(1),              # 1  内部
            self.fin_delta / cfg.delta_max,                             # 5  内部
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
            for k in ("fell", "thrown", "pushed", "stuck", "table_crash",
                      "verify_fail", "timeout"):
                log[f"term/{k}"] = self._sig[k][env_ids].float().mean().item()
            log["diag/arm_table_gap_cm"] = self._sig["arm_gap"][env_ids].min().item() * 100
            log["diag/self_gap_cm"] = self._sig["self_gap"][env_ids].min().item() * 100
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
            log["curr/eps_pos_cm"] = self.cfg.eps_pos * 100
            log["curr/eps_rot_deg"] = float(np.degrees(self.cfg.eps_rot))
            log["curr/w_imit_ramp"] = float(self.cfg.w_imit_ramp)
            log["curr/approach_budget"] = float(self.phase_timeout_t[Phase.PREGRASP])
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
        for k, v in self._ep_sums.items():
            ep_len = self.episode_length_buf[env_ids].clamp(min=1).float()
            log[f"ep_rew/{k}"] = (v[env_ids] / ep_len).mean().item()
            v[env_ids] = 0.0
        DirectRLEnv._reset_idx(self, env_ids)       # 跳过基类的 RSI/参考时钟复位
        self.extras.update(log)
        self.extras["log"] = log

        n = len(env_ids)
        dev, cfg = self.device, self.cfg
        origins = self.scene.env_origins[env_ids]
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

        # ---- 机器人: 起步分支 (RSI) ----
        # ① 直接抓取起步 (prior 位姿, phase=GRASP) = **今天的任务**, 也是免费回归桶
        # ② 接近起步     (q_ref[t0], phase=PREGRASP), t0 ~ U(0, approach_t0_max·gs)
        # 两条课程 (direct_grasp_prob / approach_t0_max) 都由训练入口按 sr_ema 退火.
        c0 = torch.rand(n, device=dev) * cfg.closure_init_max
        t0 = torch.full((n,), self.gs, dtype=torch.long, device=dev)
        start_grasp = torch.ones(n, dtype=torch.bool, device=dev)
        if cfg.approach:
            start_grasp = torch.rand(n, device=dev) < cfg.direct_grasp_prob
            hi = max(int(cfg.approach_t0_max * self.gs), 1)
            t0 = torch.where(start_grasp, t0,
                             torch.randint(0, hi, (n,), device=dev))
            c0 = torch.where(start_grasp, c0, torch.zeros_like(c0))   # 接近段手张开
        tmpl0 = self.q_open + c0.unsqueeze(1) * (self.q_close - self.q_open)
        if self.arm_start_pool is None:
            q_arm = self.q_pregrasp.expand(n, 7).clone()
        else:                       # 从起始位形池随机抽 (见 __init__ 里 arm_start_pool 注释)
            k = torch.randint(len(self.arm_start_pool), (n,), device=dev)
            q_arm = self.arm_start_pool[k]
        if cfg.approach:
            q_arm = torch.where(start_grasp.unsqueeze(1), q_arm, self.q_ref[t0])
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
        self.fin_delta[env_ids] = 0.0
        self.task_phase[env_ids] = torch.where(
            start_grasp, torch.full_like(t0, Phase.GRASP),
            torch.full_like(t0, Phase.PREGRASP))
        # 接近段状态
        self.ref_t[env_ids] = t0
        self.started_grasp[env_ids] = start_grasp
        self.arrived[env_ids] = start_grasp   # 抓取起步 = 天然已到位
        self.arrive_step[env_ids] = 0
        self.ref_q_prev[env_ids] = self.q_ref[t0]
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
        self.succeeded[env_ids] = False
        self.pad_touched[env_ids] = False
        self.any_contact[env_ids] = False
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
