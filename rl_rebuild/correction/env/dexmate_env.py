"""DexMate(Vega)+Sharpa 关节空间残差修正环境.

继承 SharpaCorrectionEnv, 只换"机器人怎么被驱动"这一层:
reward / RSI / 冻结窗 / 成功判定 / 日志 全部复用基类, 保证两套环境可比.

覆盖点 (每一处都对应一个物理事实的改变):
  _resolve_joint_ids  整机 67 关节里挑出 22 手指 + 7 交互臂
  _build_res_scale    残差界 28(笛卡尔) -> 29(关节, 逐关节标定)
  __init__ 尾部       用 bimanual_align 重摆参考轨迹, 再 IK 成 q_ref (L,7)
  _pre_physics_step   算臂关节目标, 不再算腕位姿
  _apply_action       set_joint_position_target, 不再 set_external_force_and_torque
  _get_observations   本体加臂 7 维、参考误差换成关节误差、新增关节力矩通道
  effort_norm         凭空 wrench 用量 -> 关节力矩用量
  _reset_robot        写关节状态, 不再写浮动根位姿
  wrist_*_w           腕 = body(hand_C_MC) 的位姿, 不再是根刚体
"""
from __future__ import annotations

import numpy as np
import torch

from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from rl_rebuild.correction import clips
from rl_rebuild.correction import frames as F
from rl_rebuild.correction.env.correction_env import SharpaCorrectionEnv
from rl_rebuild.correction.env.dexmate_env_cfg import DexmateCorrectionEnvCfg


class DexmateCorrectionEnv(SharpaCorrectionEnv):
    cfg: DexmateCorrectionEnvCfg

    # ---- 关节分组 ------------------------------------------------------
    def _resolve_joint_ids(self):
        jn = list(self.hand.joint_names)
        # 交互手跟着数据走 (phase_* 判定), 不用 cfg 里写死的 —— 否则左手 clip 会驱动右臂
        side = clips.interact_hand(self.cfg.clip_name, self.cfg.hand_side)
        if side != self.cfg.hand_side:
            print(f"[dexmate] clip {self.cfg.clip_name} 的交互手是 {side}, "
                  f"覆盖 cfg.hand_side={self.cfg.hand_side}")
        self.cfg.hand_side = side
        P = "R" if side == "right" else "L"
        # 手指: 按飞手 USD 的同名关节挑出来 (已核对: DexMate USD 里 22 个名字完全一致)
        self.hand_joint_names = [n for n in jn if n.startswith(f"{side}_")]
        self.hand_jids = [jn.index(n) for n in self.hand_joint_names]
        self.arm_joint_names = [f"{P}_arm_j{i}" for i in range(1, 8)]
        missing = [n for n in self.arm_joint_names if n not in jn]
        assert not missing, f"DexMate 关节表里找不到 {missing}"
        self.arm_jids = [jn.index(n) for n in self.arm_joint_names]
        self.ee_body = f"{side}_hand_C_MC"
        self.ee_id = list(self.hand.body_names).index(self.ee_body)
        print(f"[dexmate] 交互侧={side} | 臂 {self.arm_joint_names} | "
              f"手 {len(self.hand_jids)} 关节 | 末端 body={self.ee_body}")

    def _build_res_scale(self, to):
        return to(np.concatenate([
            np.asarray(self.cfg.arm_residual_max, dtype=np.float64),   # 7, 逐关节标定
            np.full(22, self.cfg.finger_residual_max)]))

    # ---- 机器人状态访问器 (基类的 reward/终止 通过这几个读) ------------
    @property
    def wrist_pos_w(self) -> torch.Tensor:
        return self.hand.data.body_pos_w[:, self.ee_id]

    @property
    def wrist_quat_w(self) -> torch.Tensor:
        return self.hand.data.body_quat_w[:, self.ee_id]

    @property
    def wrist_linvel_w(self) -> torch.Tensor:
        return self.hand.data.body_lin_vel_w[:, self.ee_id]

    @property
    def wrist_angvel_w(self) -> torch.Tensor:
        return self.hand.data.body_ang_vel_w[:, self.ee_id]

    @property
    def arm_q(self) -> torch.Tensor:
        return self.hand.data.joint_pos[:, self.arm_jids]

    @property
    def arm_qd(self) -> torch.Tensor:
        return self.hand.data.joint_vel[:, self.arm_jids]

    @property
    def arm_torque_norm(self) -> torch.Tensor:
        """(N,7) 每个臂关节的归一化力矩 |τ|/effort_limit ∈ [0,1] 量级.
        分母用 URDF 真机值, 不用 USD 的 1500 —— 否则罚不到任何东西."""
        tau = self.hand.data.applied_torque[:, self.arm_jids]
        return (tau / self.arm_effort_limit).abs()

    @property
    def effort_norm(self) -> torch.Tensor:
        """省力正则的输入. 换成关节力矩用量: 罚的是电机真的要出多少力,
        而不是飞手那个凭空的 wrench."""
        return self.arm_torque_norm.mean(dim=1)

    # ------------------------------------------------------------------
    def __init__(self, cfg: DexmateCorrectionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        dev = self.device

        assert self.start_jitter == 0.0, \
            "start_jitter 在关节空间下要重新定义 (腕位抖动需重解 IK), 暂不支持"

        # 力矩归一化的分母 (URDF 真机值)
        from rl_rebuild.correction.env.dexmate_env_cfg import ARM_EFFORT
        self.arm_effort_limit = torch.tensor(
            [ARM_EFFORT[i] for i in range(1, 8)], dtype=torch.float32, device=dev)

        # 臂关节限位 (残差钳制用)
        lim = self.hand.root_physx_view.get_dof_limits().to(dev)
        self.arm_lower = lim[..., 0][:, self.arm_jids]
        self.arm_upper = lim[..., 1][:, self.arm_jids]

        # 目标缓存: 关节空间
        self.arm_tgt = torch.zeros(self.num_envs, 7, device=dev)
        self.arm_tgt_prev = torch.zeros(self.num_envs, 7, device=dev)
        self.finger_tgt_prev = torch.zeros(self.num_envs, 22, device=dev)
        self._substep = 0
        self.arm_err_ctr = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        self.q_cmd = torch.zeros(self.num_envs, 7, device=dev)   # 策略自己的累积目标
        self.q_base_prev = torch.zeros(self.num_envs, 7, device=dev)  # 上一步的参考(算前馈增量)
        self.ee_jac = self.ee_id - 1        # 固定基座: jacobian 的 body 索引要 -1
        # 动态残差的归一化距离; reset 后第一次取观测时 _pre_physics_step 还没跑过, 先占位
        self._dyn_u = torch.ones(self.num_envs, device=dev)

        # ---- 关键: 重摆参考轨迹 + 解 IK ----
        self._place_and_solve_ik()

        # ---- 从默认站姿出发的接近段: 把时间轴整体后移 home_steps ----
        # Static reconstruction uses the video hand only as an object-placement
        # marker. Keep both robot arms in their configured default pose.
        self.home_steps = (0 if cfg.place_mode == "object_only"
                           else int(cfg.home_steps))
        # grasp_only=False 时基类不定义 lift_step0/hold_step0 (那是硬编码抬升的产物).
        # 完整轨迹模式下用不到它们, 但相位显示/日志会读, 给个等价定义:
        #   参考播完 = settle + home + (L - t0), 之后是 hold.
        if not cfg.grasp_only:
            self.lift_step0 = self.hold_step0 = cfg.settle_steps + (self.L - self.t0)
        if self.home_steps > 0:
            self.q_home = self.hand.data.default_joint_pos[:, self.arm_jids].clone()  # (N,7)
            self.q_pregrasp = self.q_ref[self.grasp_start]                            # (7,)
            self.lift_step0 += self.home_steps
            self.hold_step0 += self.home_steps
            self.ep_total += self.home_steps
            _d = torch.norm(self.q_home[0] - self.q_pregrasp).item()
            print(f"[dexmate] 从默认站姿出发: {self.home_steps} 步插值到 PreGrasp "
                  f"(关节空间距离 {np.degrees(_d):.1f}°) | episode {self.ep_total} 步")
        else:
            self.q_home = self.q_pregrasp = None

    # ------------------------------------------------------------------
    def _place_and_solve_ik(self):
        """把参考轨迹搬到"机器人够得到"的位置, 再解成臂关节轨迹.

        为什么必须重摆: 训练 env 的 load_replay 参考是给**飞手**摆的 (物体放桌心,
        手在物体周围), 和 bimanual_align 给机器人摆的那一套**不是同一个放置**
        (实测 Grasp2 两者腕轨迹差 10cm 量级, 且不是简单平移). 飞手能凭空飞过去,
        DexMate 不能 —— 必须用为"机器人够得到"设计的那套摆放.
        """
        import numpy as _np
        from rl_rebuild.correction import bimanual_align as BA
        from rl_rebuild.correction.kinematics import ArmIK

        dev, cfg = self.device, self.cfg
        bn = list(self.hand.body_names)
        # 机器人侧锚点: 从 Articulation **实测** (URDF 零位偏移会差 87cm, 躯干姿态会抬肩).
        # ⚠ 必须先让它在重力下**沉降到平衡**再读: 躯干不会精确停在配置角度, 一步之后读到的
        # 还是初始值, 而稳态会偏几度 —— 那几度会让手臂基座跑掉 8cm 级别.
        _q0 = self.hand.data.default_joint_pos
        self.hand.write_joint_state_to_sim(_q0, torch.zeros_like(_q0))
        self.hand.set_joint_position_target(_q0)
        # 躯干/底盘/头不参与任务, 用**关节限位**把它们钉死, 不靠驱动去"顶住".
        # 为什么: 位置驱动顶不住 —— 实测 kp=1e5 时躯干仍偏 4.69°, 反推需要 8190Nm,
        # 而上半身重力矩只有 ~20Nm 量级. 这是 141kg 多体 + 8 次位置迭代下**求解器
        # 不收敛**的伪平衡, 不是真实载荷. 限位是硬约束, 条件数比刚性驱动好得多,
        # 而且真机的躯干在位置控制下本来也不会塌 5°.
        if cfg.lock_torso and any(
                n.startswith(("torso_j", "dummy_base", "head_j"))
                for n in self.hand.joint_names):
            _lo = self.hand.data.joint_pos_limits.clone()
            _fz = [i for i, n in enumerate(self.hand.joint_names)
                   if n.startswith(("torso_j", "dummy_base", "head_j"))]
            _fz = _fz or None
            _eps = cfg.lock_tolerance
            _lo[:, _fz, 0] = _q0[:, _fz] - _eps
            _lo[:, _fz, 1] = _q0[:, _fz] + _eps
            self.hand.write_joint_position_limit_to_sim(_lo)
            # 限位写完再 snap 一次关节状态: 否则起点可能落在新限位之外, PhysX 要花很多步
            # 才把它推回来 —— 而锚点如果在这期间读走就**过期**了 (下面的收敛循环兜底).
            self.hand.write_joint_state_to_sim(_q0, torch.zeros_like(_q0))
            print(f"[dexmate] 已锁定 {len(_fz)} 个非任务关节 (躯干/底盘/头) 在默认位 ±"
                  f"{np.degrees(_eps):.2f}°")
        self.hand.write_data_to_sim()
        # ⚠ 必须 settle 到躯干**真的收敛**才能读锚点.
        # 踩过: 固定跑 240 步就读, 那时躯干还停在 81.7°(没到锁定的 90°); 等 reset() 之后
        # 被限位拉回 90°, 锚点就过期了 —— 8.3° 躯干转角 × 0.95m 力臂 = 末端差 9.5cm,
        # 表现成"臂关节误差 0.00° 但末端离参考 9cm", 而且手指会插进桌面 12cm.
        _jn = list(self.hand.joint_names)
        _tj = [_jn.index(f"torso_j{i}") for i in (1, 2, 3) if f"torso_j{i}" in _jn]
        if not _tj:                       # 锁死版 USD: 躯干已不是关节, 无需等它沉降
            for _ in range(cfg.settle_physics_steps):
                self.sim.step(render=False)
            self.hand.update(self.sim.get_physics_dt())
            print("[dexmate] 躯干已在 USD 里锁死 (fixed joint), 无沉降过程")
            _tj = None
        _td = np.degrees(_q0[0, _tj].cpu().numpy()) if _tj else None
        _chunk = max(cfg.settle_physics_steps // 8, 1)
        _ta = _td
        for _it in range(cfg.settle_max_rounds if _tj else 0):
            for _ in range(_chunk):
                self.sim.step(render=False)
            self.hand.update(self.sim.get_physics_dt())
            _ta = np.degrees(self.hand.data.joint_pos[0, _tj].cpu().numpy())
            if np.abs(_ta - _td).max() < cfg.settle_torso_tol_deg:
                break
        _dev = float(np.abs(_ta - _td).max()) if _tj else 0.0
        if _tj:
            print(f"[dexmate] 躯干收敛 {np.round(_ta,2).tolist()}° (目标 {np.round(_td,2).tolist()}°, "
                  f"偏差 {_dev:.2f}°, {(_it+1)*_chunk} 物理步)")
        if _tj and _dev >= cfg.settle_torso_tol_deg:
            print(f"[dexmate] ⚠ 躯干没收敛到 {cfg.settle_torso_tol_deg}° 以内, "
                  f"锚点会带这个误差 -> 末端约偏 {_dev/57.3*0.95*100:.1f}cm")
        org = self.scene.env_origins[0].cpu().numpy()
        bp = (self.hand.data.body_pos_w[0].cpu().numpy() - org)      # env-local
        rh = {"left": bp[bn.index("left_hand_C_MC")].tolist(),
              "right": bp[bn.index("right_hand_C_MC")].tolist()}
        _ha = getattr(cfg, "head_anchor", "vega_1p_head_l3")
        rhead = bp[bn.index(_ha)].tolist() if _ha in bn else None

        # ---- place_mode="ref_builder": 保留 ref builder 已经摆好的手-物关系, 只做刚体平移 ----
        # replay_grasp 的摆放**已经为"抓得住"精心设计过**, 有四件 bimanual_align 不做的事:
        #   ① 物体锚在 affordance 加权重心(甜甜圈的"环"), 不是质心
        #   ② hover_gap=2cm: 复位时张开的手不戳进物体 (否则 PhysX 退穿透把物体弹飞 -> NaN)
        #   ③ **抓取起点后腕冻结**, 只合手指 —— grasp_only 把物体钉桌上, 腕跟着人手轨迹
        #      飘走就是抓空气 (实测覆盖掉之后接触率从 17.3% 掉到 0.1%)
        #   ④ 手指用固定合拢斜坡, 重建手指本就被丢弃
        # bimanual_align 解决的是另一个问题("把双手轨迹放进机器人的第一人称坐标系"),
        # 用它整体覆盖会把上面四条一起抹掉. 所以默认只在**需要时**做整体平移.
        if cfg.place_mode == "object_only":
            self._use_object_only_reference(bn, org)
            return
        if cfg.place_mode == "ref_builder":
            if getattr(cfg, "anchor_mode", "camera") == "camera":
                self._recheck_camera_anchor(bn, org)
            self._solve_ik_only(bn, org)
            return

        e = clips.clip_entry(cfg.clip_name)
        res = BA.compute_placement(e["npz"], e["mesh"], rh, robot_head=rhead,
                                   table_top_z=cfg.table_top_z, obj_gap=0.002,
                                   grasp_gap=cfg.grasp_gap)
        prim = res["primary"]
        if prim != cfg.hand_side:
            print(f"[dexmate] ⚠ bimanual_align 判定交互手={prim}, 但 cfg.hand_side="
                  f"{cfg.hand_side}. 以 cfg 为准 (改 hand_side 可切换).")
            prim = cfg.hand_side
        J = res["joints"][prim]                            # (T,21,3) 已摆放, **源帧率**
        # ⚠ compute_placement 读的是原始 npz (15fps), 而 env 的参考被 load_replay
        # 处理过: NaN 最近邻填充 -> 高斯平滑 -> 重采样到 target_hz(20Hz).
        # 实测 Grasp2 是 125 帧 vs 166 帧. 摆放后的轨迹必须走**同一套处理**,
        # 否则时间轴对不上 (而且会把重建自带的 NaN 帧直接带进 q_ref).
        T = len(J)
        vld = _np.isfinite(J).all(axis=(1, 2))
        assert vld.any(), f"{cfg.clip_name}: 摆放后轨迹全是 NaN"
        iv = _np.flatnonzero(vld)
        J = J[iv[_np.abs(_np.arange(T)[:, None] - iv[None, :]).argmin(1)]]
        J = F.smooth_channels(J.reshape(T, -1), sigma=1.5).reshape(T, 21, 3)
        src_T = T
        if T != self.L:
            J = F.resample(J, self.L)
        _sc = self.L / max(src_T - 1, 1)                   # 源帧号 -> env 帧号
        print(f"[dexmate] 参考重采样 {src_T} -> {self.L} 帧 (NaN 填充 {int((~vld).sum())} 帧)")

        # ---- 覆盖参考: 腕 + 物体 (手指 qpos 与放置无关, 不动) ----
        to = lambda x: torch.tensor(_np.asarray(x), dtype=torch.float32, device=dev)
        self.ref_wrist_pos = to(J[:, 0])
        self.ref_wrist_quat = to(F.sharpa_base_quat_from_joints(J, prim))
        op, oq = to(res["obj_pos"]), to(res["obj_quat"])
        self.obj_init_pos, self.obj_init_quat = op, oq
        self.ref_obj_pos[:] = op                           # 常量 (见 doc "未解决")
        self.ref_obj_quat[:] = oq
        self.ref_obj_vel[:] = 0.0
        self.obj_rest_z = float(res["obj_pos"][2])
        print(f"[dexmate] 参考已重摆: 交互手={prim} 抓取帧={res['grasp_frame']} "
              f"锚定={res['anchor_mode']} | 物体 {res['obj_pos'].round(4).tolist()}")

        # ---- IK: 腕位姿 -> 臂关节轨迹 ----
        # 锚在**实测的 arm_center** 上, 不用"躯干在配置角度"这个假设 ——
        # 否则躯干沉降几度就让整条臂的基座偏掉, q_ref 会是个到不了的目标.
        from rl_rebuild.correction.kinematics import quat_to_R
        _ac = bn.index("arm_center")
        _ap = self.hand.data.body_pos_w[0, _ac].cpu().numpy() - org
        _aq = self.hand.data.body_quat_w[0, _ac].cpu().numpy()
        anchor_T = _np.eye(4)
        anchor_T[:3, :3], anchor_T[:3, 3] = quat_to_R(_aq), _ap
        ik = ArmIK(prim, anchor_link="arm_center", anchor_T=anchor_T)
        self._anchor_T = anchor_T
        sols = ik.solve_traj(J[:, 0], F.sharpa_base_quat_from_joints(J, prim))
        q = _np.stack([s["q"] for s in sols])
        ok = _np.array([s["ok"] for s in sols])
        pe = _np.array([s["pos_err"] for s in sols])
        # 解不出的帧 (重建 NaN / 够不到): 用最近的可达帧顶上, 保证 q_ref 无 NaN 且连续.
        # 不能留 NaN —— 它会顺着 obs 和目标值污染整个训练.
        bad = _np.flatnonzero(~ok | ~_np.isfinite(q).all(1))
        if len(bad):
            good = _np.flatnonzero(ok & _np.isfinite(q).all(1))
            assert len(good), "整条 clip 都解不出 IK, 这条数据不能用"
            q[bad] = q[good[_np.abs(good[None] - bad[:, None]).argmin(1)]]
        self.q_ref = to(q)                                 # (L,7)
        self._record_ik_quality(ok, pe, bad)
        a, b = res["inter"]["per_hand"][prim]              # 源帧号 -> env 帧号
        a, b = int(a * _sc), min(int(b * _sc), self.L - 1)
        print(f"[dexmate] IK: 全程可达 {ok.mean()*100:.1f}% | 接触段[{a},{b}] "
              f"{ok[a:b+1].mean()*100:.1f}% | 位置误差中位 {_np.median(pe[ok])*100:.2f}cm | "
              f"顶替 {len(bad)} 帧")

    def _ref_t(self) -> torch.Tensor:
        """参考帧下标. home 段内时钟不走 —— 那一段是"把手送过去", 参考轨迹还没开始."""
        if getattr(self, "home_steps", 0) <= 0:
            return super()._ref_t()
        played = (self.episode_length_buf - self.cfg.settle_steps
                  - self.home_steps).clamp(min=0)
        cap = self.grasp_end if self.cfg.grasp_only else self.L - 1
        return (self.rsi_start + played).clamp(max=cap)

    @staticmethod
    def _smoothstep(u: torch.Tensor) -> torch.Tensor:
        """3u²-2u³: 两端导数为 0 的 S 曲线.

        为什么要它: 线性斜坡在相位边界上**速度是跳变的**(0 -> 常速 -> 0),
        表现就是机器人"一卡一卡"。smoothstep 让起停都是平滑加减速。
        """
        return u * u * (3.0 - 2.0 * u)

    def _home_blend(self) -> torch.Tensor | None:
        """(N,) home 段进度 s∈[0,1] (已过 smoothstep); 不在 home 段返回 None."""
        if getattr(self, "home_steps", 0) <= 0:
            return None
        ep = self.episode_length_buf
        if not bool((ep < self.cfg.settle_steps + self.home_steps).any()):
            return None
        u = ((ep - self.cfg.settle_steps).float() / self.home_steps).clamp(0.0, 1.0)
        return self._smoothstep(u)

    # ---- 动作 ---------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions_buf)
        self.actions_buf.copy_(actions.clamp(-1.0, 1.0))
        t = self._ref_t()
        # 冻结窗: 与飞手同构 (把物体钉在参考位, 动作不生效)
        frozen = self.freeze_ctr > 0
        if frozen.any():
            idx = torch.nonzero(frozen, as_tuple=False).squeeze(-1)
            pose = torch.cat([self.ref_obj_pos[t[idx]] + self.scene.env_origins[idx],
                              self.ref_obj_quat[t[idx]]], dim=1)
            vel = torch.zeros(len(idx), 6, device=self.device)
            vel[:, :3] = self.ref_obj_vel[t[idx]]
            self.object.write_root_pose_to_sim(pose, idx)
            self.object.write_root_velocity_to_sim(vel, idx)
            self.freeze_ctr[idx] -= 1
        a = self.actions_buf * (~frozen).float().unsqueeze(1) * self.res_scale   # (N,29)

        arm_res, fin_res = a[:, 0:7], a[:, 7:29]
        if self.cfg.dyn_res:
            # 按 掌心->物体表面 距离连续缩放: 臂"远松近紧", 手指"远紧近松"(方向相反)
            s_arm, s_fin, self._dyn_u = self._dyn_res_scale()
            arm_res = arm_res * s_arm.unsqueeze(1)
            fin_res = fin_res * s_fin.unsqueeze(1)
        elif self.approach_steps > 0:              # 旧行为: 按参考帧号二值放大
            in_app = (t < self.grasp_start).float().unsqueeze(1)
            arm_res = arm_res * (1 + in_app * (self.cfg.approach_res_scale - 1))
        # home 段: 关节空间从默认站姿线性插值到 PreGrasp; 之后才跟参考轨迹
        q_base = self.q_ref[t]
        # 抬升段: 换成预解的抬升关节轨迹 (飞手是在腕目标 z 上加斜坡, 关节空间加不了)
        if getattr(self, "q_lift", None) is not None:
            pr = self._smoothstep(self._lift_prog())                # (N,) ∈[0,1], 平滑起停
            f = (pr * self.cfg.lift_steps).clamp(0, self.cfg.lift_steps)
            i0 = f.floor().long().clamp(0, self.cfg.lift_steps - 1)
            w = (f - i0.float()).unsqueeze(1)
            q_lift = (1 - w) * self.q_lift[i0] + w * self.q_lift[i0 + 1]   # 线性插值, 不取整
            q_base = torch.where((pr > 0).unsqueeze(1), q_lift, q_base)
        sh = self._home_blend()
        if sh is not None:
            u_ = sh.unsqueeze(1)
            q_base = torch.where(u_ < 1.0, (1 - u_) * self.q_home + u_ * self.q_pregrasp,
                                 q_base)
        self.arm_tgt_prev = self.arm_tgt.clone()
        if self.cfg.accumulate_action:
            # 参考提供**增量**(前馈), 策略的修正在其上**累积** —— 两者缺一不可:
            #   只累积不给前馈 -> 零动作时手臂冻在起点, 参考完全不起作用 (实测抬升 0.15cm)
            #   只给前馈不累积 -> 退回旧行为, 偏差每步归零, 走不出 2cm
            # 现在: 零动作 = 精确跟参考; 有动作 = 偏差**持续累加**, 直到管壁.
            # ⚠ 前馈增量必须取自 **q_base 本身**, 不能取 q_ref[t]:
            #   抬升段 q_base 来自预解的 q_lift, 而参考时钟 t 已被钳在 grasp_end,
            #   q_ref[t]-q_ref[t-1] 恒为 0 —— 抬升增量会整个丢掉 (实测抬升 0.16cm).
            q_cmd = self.q_cmd + (q_base - self.q_base_prev) + arm_res
            # home 段仍走插值 (那一段是"把手送过去", 不是任务)
            if sh is not None:
                q_cmd = torch.where((sh < 1.0).unsqueeze(1), q_base, q_cmd)
            self.q_cmd = q_cmd.clamp(self.arm_lower, self.arm_upper)
            if sh is None:                # home 段不受管道约束 (那是转移, 不是任务)
                self._tube_project(t)
            self.arm_tgt = self.q_cmd
        else:
            self.arm_tgt = (q_base + arm_res).clamp(self.arm_lower, self.arm_upper)
        self.q_base_prev = q_base

        finger_ref = self._contact_close_ref() if self.contact_close else self.ref_finger[t]
        self.finger_tgt_prev = self.finger_tgt.clone()
        # 手指与臂同构: 参考给基线, 策略的修正在其上**累积** (dyn_res 缩的是每步增量).
        self.finger_tgt = (finger_ref + self._accum_finger(fin_res)
                           ).clamp(self.dof_lower, self.dof_upper)
        self._substep = 0

    def _tube_project(self, t):
        """末端偏离参考超过 tube_radius 就拉回管壁 (雅可比一步修正).

        用**实际末端**而不是 FK(q_cmd) 算偏差: IsaacLab 直接给实际末端位姿和雅可比,
        不用自己再算一遍 FK; 20Hz 下这点反馈滞后可以忽略.
        管道的意义: 给策略自由(可累积), 同时保证"不会走得离人手太远" ——
        相似度是**结构性**保证, 不是奖励项.
        """
        # ⚠ 用**实际末端**算偏差, 而实际落后于指令 —— 等它到 R 才拉已经晚了.
        # 提前量: 从 soft_frac·R 就开始往回拉, 拉力随超出量线性增长.
        R = self.cfg.tube_radius
        Rs = R * self.cfg.tube_soft_frac
        ee = self.wrist_pos_w - self.scene.env_origins          # (N,3)
        center = self.ref_wrist_pos[t]
        if getattr(self, "p_lift", None) is not None:
            pr = self._smoothstep(self._lift_prog())
            f = (pr * self.cfg.lift_steps).clamp(0, self.cfg.lift_steps)
            i0 = f.floor().long().clamp(0, self.cfg.lift_steps - 1)
            w = (f - i0.float()).unsqueeze(1)
            center = torch.where((pr > 0).unsqueeze(1),
                                 (1 - w) * self.p_lift[i0] + w * self.p_lift[i0 + 1],
                                 center)
        d = ee - center
        n = d.norm(dim=1, keepdim=True)
        over = (n - Rs).clamp(min=0.0)
        if not bool((over > 0).any()):
            return
        # 世界系修正量: 沿偏离方向把末端拉回管壁
        corr = -(d / n.clamp(min=1e-6)) * over * self.cfg.tube_kp
        # ⚠ 分两步切: [:, int, :, list] 会触发 advanced-index 重排, 静默给出错的雅可比
        jf = self.hand.root_physx_view.get_jacobians()
        jac = jf[:, self.ee_jac, :3, :][:, :, self.arm_jids]     # (N,3,7) 位置部分
        # 阻尼最小二乘伪逆, 奇异处不炸
        JT = jac.transpose(1, 2)
        A = jac @ JT + 1e-4 * torch.eye(3, device=self.device)
        dq = (JT @ torch.linalg.solve(A, corr.unsqueeze(-1))).squeeze(-1)
        self.q_cmd = (self.q_cmd + dq).clamp(self.arm_lower, self.arm_upper)

    def _apply_action(self) -> None:
        # 关节位置目标 —— 飞手那一整套 wrench-PD (增益/力钳制/坐标系转换/四元数归正)
        # 在这里全部不需要了: 约束由电机 + PhysX 自己执行.
        #
        # 子步插值: 策略在 20Hz 决策, 但物理跑 240Hz. 直接把目标**保持 12 个子步再跳**,
        # 关节每跳一次动 ~1.7°, 看起来就是"一卡一卡". 这里把目标在子步间线性铺开,
        # 台阶变成 240Hz 的斜坡. 真机的伺服控制器本来也是这么做的.
        # ⚠ 不改变策略的决策频率, 只改变**目标怎么送达** —— 对训练是纯增益:
        #   接触冲量更小、jerk 更低, 而策略看到的观测/动作接口完全不变.
        if self.cfg.substep_interp:
            w = min((self._substep + 1) / self.cfg.decimation, 1.0)
            arm = self.arm_tgt_prev + (self.arm_tgt - self.arm_tgt_prev) * w
            fin = self.finger_tgt_prev + (self.finger_tgt - self.finger_tgt_prev) * w
            self._substep += 1
        else:
            arm, fin = self.arm_tgt, self.finger_tgt
        self.hand.set_joint_position_target(arm, joint_ids=self.arm_jids)
        self.hand.set_joint_position_target(fin, joint_ids=self.hand_jids)

    # ---- 观测 ---------------------------------------------------------
    def _get_observations(self) -> dict:
        cfg, origins = self.cfg, self.scene.env_origins
        t = self._ref_t()
        t5 = (t + cfg.lookahead).clamp(max=self.L - 1)
        vs = cfg.obs_vel_scale

        wrist_pos = self.wrist_pos_w - origins
        wrist_quat = self._qsign(self.wrist_quat_w)
        q_inv = quat_conjugate(wrist_quat)
        obj_pos = self.object.data.root_pos_w - origins
        obj_quat = self.object.data.root_quat_w
        arm_q = self.arm_q

        # 物体表面点云 (腕系) — 与飞手同构
        pc_wrist = None
        if self.obj_points is not None:
            P = self.obj_points.shape[0]
            qp = obj_quat[:, None, :].expand(-1, P, -1).reshape(-1, 4)
            pc = self.obj_points[None].expand(self.num_envs, -1, -1).reshape(-1, 3)
            pw = quat_apply(qp, pc) + obj_pos[:, None, :].expand(-1, P, -1).reshape(-1, 3)
            qi = q_inv[:, None, :].expand(-1, P, -1).reshape(-1, 4)
            wp = wrist_pos[:, None, :].expand(-1, P, -1).reshape(-1, 3)
            pc_wrist = quat_apply(qi, pw - wp).reshape(self.num_envs, P, 3)
        if pc_wrist is not None and not cfg.enable_pointcloud:
            obj_geom = pc_wrist.reshape(self.num_envs, -1)
        else:
            obj_geom = torch.zeros(self.num_envs, 0, device=self.device)

        # 相位 (与飞手同构)
        in_settle = (self.episode_length_buf < cfg.settle_steps).float()
        if cfg.grasp_only:
            progress = (self.episode_length_buf.float() / max(self.ep_total, 1)
                        ).clamp(max=1.0).unsqueeze(1)
            in_lift = (self.episode_length_buf >= self.lift_step0).float()
            phase = torch.stack([in_settle, 1.0 - in_settle - in_lift, in_lift], dim=1)
        else:
            played = (self.episode_length_buf - cfg.settle_steps).clamp(min=0).float()
            progress = (played / max(self.L - self.t0, 1)).clamp(max=1.0).unsqueeze(1)
            in_hold = (self.episode_length_buf >= cfg.settle_steps + self.L - self.t0).float()
            phase = torch.stack([in_settle, 1.0 - in_settle - in_hold, in_hold], dim=1)

        obs = torch.cat([
            arm_q,                                                      # 7  臂关节角
            self.finger_q,                                              # 22
            self.arm_qd * vs,                                           # 7
            self.finger_qd * vs,                                        # 22
            wrist_pos,                                                  # 3  末端(手基座)位姿
            wrist_quat,                                                 # 4   —— 现在是 FK 的结果
            self.wrist_linvel_w,                                        # 3   而不是被控量
            self.wrist_angvel_w * vs,                                   # 3
            quat_apply(q_inv, obj_pos - wrist_pos),                     # 3  物体位置(腕系)
            self._qsign(quat_mul(q_inv, obj_quat)),                     # 4  物体姿态(相对腕)
            self.object.data.root_lin_vel_w,                            # 3
            self.object.data.root_ang_vel_w * vs,                       # 3
            self.ref_finger[t],                                         # 22 参考手指
            self.q_ref[t] - arm_q,                                      # 7  臂关节参考误差
            self.ref_obj_pos[t] - obj_pos,                              # 3  物体参考误差
            self.q_ref[t5] - self.q_ref[t],                             # 7  前瞻: 臂将怎么动
            self.ref_obj_pos[t5] - self.ref_obj_pos[t],                 # 3  前瞻: 物体该去哪
            progress,                                                   # 1
            phase,                                                      # 3
            self._tip_contacts(),                                       # 5
            self.arm_torque_norm,                                       # 7  电机在多用力
            # 动态残差的归一化距离 u: **必须给**, 否则残差界随状态变而策略看不见,
            # 同一个网络输出在不同距离下含义不同 -> 策略在盲猜
            self._dyn_u.unsqueeze(1) if cfg.dyn_res
            else torch.zeros(self.num_envs, 0, device=self.device),     # 1 (dyn_res 时)
            # 累积手指残差: 同理必须给 —— 动作是增量, 策略得知道积分器现在在哪
            self.finger_res / cfg.finger_total_max if cfg.accumulate_finger
            else torch.zeros(self.num_envs, 0, device=self.device),     # 22 (累积模式)
            # 终点势门控: 奖励依赖它, 策略必须看得见 (否则信用分配靠猜)
            self.goal_gate.float().unsqueeze(1) if self.use_goal
            else torch.zeros(self.num_envs, 0, device=self.device),     # 1 (终点势时)
            self.actions_buf,                                           # 29
            obj_geom,
        ], dim=1).clamp(-cfg.clip_obs, cfg.clip_obs).nan_to_num(0.0)
        self._check_obs_dim(obs)

        tip_f = torch.cat([s.data.force_matrix_w.view(self.num_envs, 1, 3)
                           for s in self._contact_sensors], dim=1).norm(dim=-1)
        priv = torch.cat([self.obj_mass, self.obj_fric, tip_f], dim=1
                         ).clamp(-cfg.clip_obs, cfg.clip_obs).nan_to_num(0.0)
        self.proprio_hist[:, :-1].copy_(self.proprio_hist[:, 1:].clone())
        self.proprio_hist[:, -1] = torch.cat(
            [arm_q, self.finger_q, self.arm_qd * vs, self.finger_qd * vs], dim=1)
        out = {"policy": obs, "priv_info": priv, "proprio_hist": self.proprio_hist}
        if cfg.enable_pointcloud and pc_wrist is not None:
            out["pointcloud"] = pc_wrist.clamp(-cfg.clip_obs, cfg.clip_obs).nan_to_num(0.0)
        return out

    def _success_step(self, obj_pos, contacts) -> torch.Tensor:
        """完整轨迹任务的成功 = **物体最终落在人手结束交互时的位置附近**.

        不看抬多高、不看怎么抓 —— 抓法交给 RL 自己找. 这才对应
        "输入一条视频 -> 输出一条相似的、物理可行的轨迹".
        (grasp_only 模式仍走基类的"抬够高+握住"判据.)
        """
        if self.cfg.grasp_only:
            return super()._success_step(obj_pos, contacts)
        in_hold = self._ref_t() >= (self.L - 1)
        tgt = self.ref_obj_pos[self.L - 1]
        near = (obj_pos - tgt).norm(dim=1) <= self.cfg.place_tol
        return in_hold & near

    # ---- 终止 ---------------------------------------------------------
    def _get_dones(self):
        terminated, truncated = super()._get_dones()
        # 卡死判据 = 臂**持续追不上下发的目标** (撞桌/自碰撞/力矩不够都会表现成这个).
        # ⚠ 比的是 arm_tgt 而不是 q_ref[t]: home 段(从默认站姿插值过去)时臂本来就不在参考上,
        #   拿参考比会直接判死 —— 踩过, 表现是每 20 步重置一次、参考时钟永远推进不了.
        # 也不能用力矩饱和: applied_torque 只是 clip(kp·Δq+kd·Δq̇) 的近似, kp 一大就恒顶满.
        off = (self.arm_q - self.arm_tgt).abs().max(dim=1).values >= self.cfg.term_arm_err
        self.arm_err_ctr = torch.where(off, self.arm_err_ctr + 1,
                                       torch.zeros_like(self.arm_err_ctr))
        stuck = self.arm_err_ctr >= self.cfg.term_arm_steps
        self._last_stuck = stuck
        return terminated | stuck, truncated

    # ---- reset --------------------------------------------------------
    def _reset_robot(self, env_ids, starts, origins):
        """关节空间摆位: 写臂 + 手指的关节状态, 没有浮动根要写."""
        q = self.hand.data.default_joint_pos[env_ids].clone()
        # home 段: 臂从**默认站姿**起步 (= 你在 GUI 里调好的那个姿态), 不是参考首帧
        # ⚠ 不要把冻结窗延长到覆盖 home 段. 冻结窗每个控制步都把物体**写回参考位姿**,
        #   延长之后手在接近途中碰到物体, 物体会被瞬移回原位 —— 物理上是假的, 而且
        #   会把"手撞到物体"这个策略本该学会避免的事件整个抹掉.
        #   物体只在 reset 时定位一次, 之后纯物理. 冻结窗只保留静置段(让它在桌上停稳).
        if getattr(self, "home_steps", 0) <= 0:
            q[:, self.arm_jids] = self.q_ref[starts]
        q[:, self.hand_jids] = self.ref_finger[starts]
        self.hand.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        self.hand.set_joint_position_target(q, env_ids=env_ids)
        self.arm_tgt[env_ids] = (q[:, self.arm_jids] if getattr(self, "home_steps", 0) > 0
                                 else self.q_ref[starts])
        self.q_cmd[env_ids] = self.arm_tgt[env_ids]      # 累积目标从参考起点开始
        self.q_base_prev[env_ids] = self.arm_tgt[env_ids]
        self.finger_tgt[env_ids] = self.ref_finger[starts]
        self.arm_err_ctr[env_ids] = 0

    def _record_ik_quality(self, ok, pe, bad) -> None:
        """把 IK 的逐帧质量存下来 (以前算完打印就丢). **只记录, 不改任何控制行为.**

        为什么需要: "不可达"有好几种, 而现在它们被同一句 `q[bad] = q[最近可达帧]` 抹平:
          ① 姿态本身机械臂摆不出来 (关节限位/奇异)   -> ok=False
          ② 重建轨迹本身有偏差 (比如插进桌子里)      -> ok 可能为 True 但 pos_err 大
          ③ 够不到 (超臂展)                          -> 已由整体平移处理
        顶替之后那一段 q_ref 是**编出来的**: 在 accumulate 模式下前馈增量恒为 0、
        段边界跳变, 而 tube 中心仍用没被顶替的 ref_wrist_pos —— 两个约束互相打架,
        日志上看不见. 存下来是做逐帧置信度的第一步 (docs/DESIGN_LOOP.md §2.4).
        """
        import numpy as _np
        dev = self.device
        L = len(ok)
        subst = _np.zeros(L, dtype=bool)
        subst[bad] = True
        self.ref_ik_ok = torch.tensor(ok.astype(_np.float32), device=dev)      # (L,)
        self.ref_ik_err = torch.tensor(_np.nan_to_num(pe, nan=_np.inf, posinf=1e3
                                                      ).astype(_np.float32), device=dev)
        self.ref_ik_subst = torch.tensor(subst, device=dev)                    # (L,) bool
        # 连续坏段比孤立坏帧危险得多 (孤立帧被邻帧插值掩盖, 连续段是整段编造), 单独报出来
        segs, s = [], None
        for i in range(L + 1):
            b = i < L and subst[i]
            if b and s is None:
                s = i
            elif not b and s is not None:
                segs.append((s, i - 1)); s = None
        self.ref_bad_segs = segs
        if segs:
            longest = max(e - s + 1 for s, e in segs)
            print(f"[ref质量] IK 顶替段 {len(segs)} 段 (最长 {longest} 帧): "
                  f"{segs[:6]}{' ...' if len(segs) > 6 else ''}")
        _f = _np.isfinite(pe) & ok
        if _f.any():
            print(f"[ref质量] 可达帧位置误差: 中位 {_np.median(pe[_f])*100:.2f}cm "
                  f"95分位 {_np.percentile(pe[_f], 95)*100:.2f}cm "
                  f"最大 {pe[_f].max()*100:.2f}cm")
        d = getattr(self.du.ref, "obj_drift", None)
        if d is not None and len(d):
            print(f"[ref质量] 物体漂移帧 {len(d)} 个 (位置是插值出来的): {list(d[:12])}"
                  f"{' ...' if len(d) > 12 else ''}")

    def _recheck_camera_anchor(self, bn, org):
        """把 ref builder 用**标称** ZED 摆好的参考, 按**实测** ZED 补一次残差平移.

        为什么分两步: ref builder 跑在 `correction_env.__init__` 里 (clips.load_data_unit),
        那时躯干还没在重力下沉降收敛 —— 这个方法所在的 `_place_and_solve_ik` 才等到收敛
        (上面那段 settle 循环, 注释里记着"躯干偏 8.3° -> 末端差 9.5cm"). 所以 ref builder
        只能用常数 `place_camera.ZED_NOMINAL`, 由这里用活值补差.
        常数是对的 -> 这一步是 0; 常数漂了 -> 补上并打印, 不会静默摆错.

        只补 xy: 相机锚定本来就只定 xy (z 由"物体贴桌" + "手全程最小抬升"两条独立约束
        各自定死), 而 xy 平移不改变手离桌面的高度, 所以 clearance 不用重算.
        """
        import numpy as _np
        from rl_rebuild.correction import place_camera as PC

        dev = self.device
        need = ("zed_left_camera", "zed_right_camera")
        if not all(n in bn for n in need):
            print(f"[dexmate] ⚠ 资产里没有 {need}, 跳过相机锚定残差修正 "
                  f"(参考仍是 ref builder 用标称 ZED 摆的)")
            return
        bp = self.hand.data.body_pos_w[0].cpu().numpy() - org
        zed = (bp[bn.index(need[0])] + bp[bn.index(need[1])]) / 2.0
        d = zed[:2] - PC.ZED_NOMINAL[:2]
        if float(_np.linalg.norm(d)) < 1e-3:
            print(f"[dexmate] 相机锚定: 实测 ZED xy={_np.round(zed[:2], 4).tolist()} "
                  f"与标称一致 (差 {float(_np.linalg.norm(d))*1000:.1f}mm), 无需修正")
            return
        v = torch.tensor([d[0], d[1], 0.0], dtype=torch.float32, device=dev)
        self.ref_wrist_pos += v
        self.ref_obj_pos += v
        self.obj_init_pos = self.obj_init_pos + v
        print(f"[dexmate] 相机锚定残差: 实测 ZED xy={_np.round(zed[:2], 4).tolist()} vs "
              f"标称 {_np.round(PC.ZED_NOMINAL[:2], 4).tolist()} -> 参考整体补移 "
              f"{_np.round(d*100, 2).tolist()}cm  "
              f"(标称过期了就改 place_camera.ZED_NOMINAL)")

    def _use_object_only_reference(self, body_names, env_origin):
        """Keep the robot at its configured pose and validate object reach.

        The reconstructed human wrist is used only by the static ref builder
        to place the object. It is never replayed as a robot command, no IK is
        solved, and the placed object is never shifted to make it reachable.
        """
        cfg, dev = self.cfg, self.device
        origin = torch.as_tensor(env_origin, dtype=torch.float32, device=dev)
        default_q = self.hand.data.default_joint_pos[0]

        arm = default_q[self.arm_jids].clone()
        finger = default_q[self.hand_jids].clone()
        wrist_pos = self.hand.data.body_pos_w[0, self.ee_id] - origin
        wrist_quat = self.hand.data.body_quat_w[0, self.ee_id].clone()

        self.q_ref = arm.unsqueeze(0).repeat(self.L, 1)
        self.ref_finger = finger.unsqueeze(0).repeat(self.L, 1)
        self.ref_wrist_pos = wrist_pos.unsqueeze(0).repeat(self.L, 1)
        self.ref_wrist_quat = wrist_quat.unsqueeze(0).repeat(self.L, 1)
        self.ref_obj_pos = self.obj_init_pos.unsqueeze(0).repeat(self.L, 1)
        self.ref_obj_quat = self.obj_init_quat.unsqueeze(0).repeat(self.L, 1)
        self.ref_obj_vel.zero_()
        self.open_pose = finger.clone()
        self.closed_pose = finger.clone()
        self.q_lift = None

        side = "R" if cfg.hand_side == "right" else "L"
        shoulder_candidates = (
            f"vega_1p_{side}_arm_l1",
            f"{side}_arm_l1",
        )
        shoulder_name = next(
            (name for name in shoulder_candidates if name in body_names), None)
        if shoulder_name is None:
            raise KeyError(
                f"DexMate asset has none of the shoulder bodies "
                f"{shoulder_candidates}"
            )
        shoulder = (
            self.hand.data.body_pos_w[0, body_names.index(shoulder_name)] - origin
        ).cpu().numpy()

        vertices = F.load_obj_verts(self.du.mesh_path)
        center_local = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
        object_quat = self.obj_init_quat.cpu().numpy()
        object_center = (
            F.rot_apply(object_quat[None], center_local[None])[0]
            + self.obj_init_pos.cpu().numpy()
        )
        distance = float(np.linalg.norm(object_center - shoulder))
        if distance > cfg.reach_margin:
            raise ValueError(
                f"{cfg.clip_name}: placed object centre is {distance:.3f}m from "
                f"the {cfg.hand_side} shoulder; limit is "
                f"{cfg.reach_margin:.3f}m. object_only placement is not shifted"
            )
        print(
            f"[dexmate] place_mode=object_only | robot reference=default pose "
            f"(no human replay / no IK) | object-centre reach {distance:.3f}m "
            f"<= {cfg.reach_margin:.3f}m"
        )

    def _solve_ik_only(self, bn, org):
        """用 ref builder 已摆好的参考直接解 IK; 够不到时**整体刚体平移**进工作空间.

        只平移、不旋转、不改手-物相对关系 —— 与用户约束一致
        (物体只能平移不能旋转; 手↔物相对关系由 ref builder 定, 这里不动).
        """
        import numpy as _np
        from rl_rebuild.correction.kinematics import ArmIK, quat_to_R

        cfg, dev = self.cfg, self.device
        side = cfg.hand_side
        _ac = bn.index("arm_center")
        anchor_T = _np.eye(4)
        anchor_T[:3, :3] = quat_to_R(self.hand.data.body_quat_w[0, _ac].cpu().numpy())
        anchor_T[:3, 3] = self.hand.data.body_pos_w[0, _ac].cpu().numpy() - org
        ik = ArmIK(side, anchor_link="arm_center", anchor_T=anchor_T)
        self._anchor_T = anchor_T

        P = self.ref_wrist_pos.cpu().numpy().astype(_np.float64)
        Q = self.ref_wrist_quat.cpu().numpy().astype(_np.float64)

        # 够不到就整体平移: 沿"肩->轨迹中心"方向把轨迹拉近, 二分找最小平移量
        sh = ik.link_pose_world(f"vega_1p_{'R' if side == 'right' else 'L'}_arm_l1",
                                ik.q_default)[:3, 3]
        shift = _np.zeros(3)
        d = _np.linalg.norm(P - sh, axis=1).max()
        if d > cfg.reach_margin:
            u = (P.mean(0) - sh); u /= _np.linalg.norm(u) + 1e-9
            shift = -u * (d - cfg.reach_margin)
            print(f"[dexmate] 参考最远点离肩 {d:.3f}m > {cfg.reach_margin}m, "
                  f"整体平移 {_np.round(shift, 4).tolist()} 拉进工作空间")
            P = P + shift
            for t, v in ((self.ref_wrist_pos, shift), (self.ref_obj_pos, shift)):
                t += torch.tensor(v, dtype=torch.float32, device=dev)
            self.obj_init_pos = self.obj_init_pos + torch.tensor(
                shift, dtype=torch.float32, device=dev)

        sols = ik.solve_traj(P, Q)
        q = _np.stack([s["q"] for s in sols])
        ok = _np.array([s["ok"] for s in sols])
        pe = _np.array([s["pos_err"] for s in sols])
        bad = _np.flatnonzero(~ok | ~_np.isfinite(q).all(1))
        if len(bad):
            good = _np.flatnonzero(ok & _np.isfinite(q).all(1))
            assert len(good), "整条 clip 都解不出 IK"
            q[bad] = q[good[_np.abs(good[None] - bad[:, None]).argmin(1)]]
        self.q_ref = torch.tensor(q, dtype=torch.float32, device=dev)
        self._record_ik_quality(ok, pe, bad)
        # ---- 硬编码抬升的关节轨迹 ----
        # grasp_only 的成功判据就是"腕垂直升 lift_height, 物体跟不跟得上". 飞手直接在腕
        # 目标 z 上加斜坡; 关节空间里加不了, 必须**把抬升后的腕位姿逐级 IK 出来**.
        # 漏了这一段的后果: 臂停在 PreGrasp 不动, 物体永远抬不起来 (实测抬升 0.00cm).
        ge = min(self.grasp_end, self.L - 1)
        p_top, q_top = P[ge].copy(), Q[ge].copy()
        lift = []
        qw = q[ge].copy()
        for i in range(cfg.lift_steps + 1):
            tgt = p_top + _np.array([0.0, 0.0, cfg.lift_height * i / max(cfg.lift_steps, 1)])
            r = ik.solve(tgt, quat_to_R(q_top), q0=qw)
            if r["ok"] or i == 0:
                qw = r["q"].copy()
            lift.append(qw.copy())
        self.q_lift = torch.tensor(_np.stack(lift), dtype=torch.float32, device=dev)
        # 抬升段的末端位置 —— 管道中心要用它, 不能用 ref_wrist_pos[t]:
        # 抬升不在参考轨迹里(参考时钟被钳在 grasp_end), 手一升 10cm 管道就以为偏了 10cm,
        # 会把抬升整个拽回去 (实测抬升只剩 2.7cm).
        self.p_lift = torch.tensor(_np.stack([ik.fk(q)[0] for q in lift]),
                                   dtype=torch.float32, device=dev)
        _dz = float(_np.linalg.norm(ik.fk(lift[-1])[0] - ik.fk(lift[0])[0]))
        print(f"[dexmate] 抬升轨迹: {cfg.lift_steps} 级 x {cfg.lift_height*100:.0f}cm, "
              f"末端实际升 {_dz*100:.1f}cm")
        a = self.grasp_start
        b = min(self.grasp_end, self.L - 1)
        print(f"[dexmate] place_mode=ref_builder (保留 ref builder 摆放, 平移 "
              f"{_np.round(shift, 4).tolist()}) | IK 全程可达 {ok.mean()*100:.1f}% | "
              f"抓取窗[{a},{b}] {ok[a:b+1].mean()*100:.1f}% | "
              f"位置误差中位 {_np.median(pe[ok])*100:.2f}cm | 顶替 {len(bad)} 帧")
