"""Carry 环境: GraspPose 起步 → 学稳定抓握 → 带物体追踪重建轨迹 (Pour P0+P1 MVP)。

2026-08-19 用户批准的计划, 外科式复用 v2 双臂底座 (PGB 同款机制), 只换四件事:
  ① 起步: retract_path = 携带参考 (build_carry_ref 产物), 首行=抓姿 IK ⟹ 现有
     起点族机制天然把双臂摆在 GraspPose; 手指参考逐侧换成抓形模板 (q_open←_p2_fin)。
  ② 时钟: 复用 ref_t 外生推进 (热身 40 行原地, 之后逐帧走轨迹)。
  ③ 判据: 成功 = 时钟到末帧 & 双物体贴住末帧目标(5cm/30°) & 保持 5 步;
     判负 = slip(4cm/30°) / 跟丢(20cm) / fell / 超时。基座的 arrive/g2 永不触发
     (eps 调 1mm), toppled/pushed/obj_hit 由配置豁免 (携带必然碰物、瓶必然翻)。
  ④ 奖励: 全部自算 (追踪 exp 衰减 + slip 罚 + 1/3·2/3·末帧一次性里程碑 + 成功大奖
     + 微动作正则), 不吃基座奖励。
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from isaaclab.utils.math import quat_conjugate, quat_mul

from tasks.pregrasp import bimanual as BM
from tasks.pregrasp.bimanual_native_env import BimanualNativeEnv
from tasks.pregrasp.cfg import Phase


class PourCarryEnv(BimanualNativeEnv):

    def __init__(self, cfg, render_mode=None, **kw):
        super().__init__(cfg, render_mode, **kw)
        z = np.load(cfg.carry_npz, allow_pickle=True)
        to = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32,
                                    device=self.device)
        # 物体目标序列: npz 序 0=瓶(A/右), 1=杯(B/左)
        # 目标存**相对首帧位移算子** (§5 公约; 冒烟实锤: 绝对姿态目标与本回合实际
        # 摆放差常量 30°/156°)。运行时: 目标_k = dq_k ⊗ q_rest, p_rest + dp_k
        self._c_obj_pos, self._c_obj_quat = {}, {}
        for _tag, _i in (("A", 0), ("B", 1)):
            _P = to(z[f"obj_pos_{_i}"])
            _Q = to(z[f"obj_quat_{_i}"])
            _Q = _Q / _Q.norm(dim=-1, keepdim=True).clamp(min=1e-9)   # ★builder 的
            # m2q 在近 180° 帧输出未归一 (|q| 实测低至 0.207), 角度计算不除模长
            # ⟹ 杯 e_rot 假读 156° (2026-08-19 冒烟五连环的最后一环)。载入即归一。
            _q0 = _Q[0] / _Q[0].norm()
            _q0c = _q0 * torch.tensor([1.0, -1.0, -1.0, -1.0], device=self.device)
            self._c_obj_pos[_tag] = _P - _P[0:1]
            self._c_obj_quat[_tag] = quat_mul(_Q, _q0c.unsqueeze(0).expand_as(_Q))
        _N0 = self.num_envs
        self._c_rest_p = {t: torch.zeros(_N0, 3, device=self.device) for t in ("A", "B")}
        self._c_rest_q = {t: torch.zeros(_N0, 4, device=self.device) for t in ("A", "B")}
        self._c_T = int(len(z["obj_pos_0"]))
        self._c_grip = int(z["grip_frames"])   # 热身帧数 (slip 基线在此刻拍)
        # T_rel* 基线 = **每回合 reset 时运行时快照** (POUR_DESIGN §3 原文语义)。
        # 冒烟实锤: 用先验链推算的规格在杯侧差恒 70° (场景摆放"不用重建旋转",
        # 与 canon+yaw 链是两套姿态) —— 快照物理真值, 零约定风险。
        N0 = self.num_envs
        self._c_rel_q = {t: torch.zeros(N0, 4, device=self.device) for t in ("A", "B")}
        self._c_rel_p = {t: torch.zeros(N0, 3, device=self.device) for t in ("A", "B")}
        N = self.num_envs
        _ms = [self._c_T // 3, 2 * self._c_T // 3, self._c_T - 1]
        self._c_ms_idx = torch.tensor(_ms, dtype=torch.long, device=self.device)
        self._c_ms_done = torch.zeros(N, 3, dtype=torch.bool, device=self.device)
        self._c_hold = torch.zeros(N, dtype=torch.long, device=self.device)
        self._c_succ = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._c_snap_pending = torch.ones(N, dtype=torch.bool, device=self.device)
        # CARRY3 进度时钟 (cfg.carry_progress)
        self._c_prog = bool(getattr(cfg, "carry_progress", False))
        self._c_t = torch.zeros(N, dtype=torch.long, device=self.device)
        self._c_started = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._c_stab = torch.zeros(N, dtype=torch.long, device=self.device)
        self._c_step = torch.zeros(N, dtype=torch.long, device=self.device)
        self._c_relprev = {t: (torch.full((N, 4), float("nan"), device=self.device),
                               torch.full((N, 3), float("nan"), device=self.device))
                           for t in ("A", "B")}
        self._c_msidx = (to(z["ms_idx"]).long() if "ms_idx" in z.files
                         else torch.zeros(0, dtype=torch.long, device=self.device))
        self._c_msdone = torch.zeros(N, max(1, len(self._c_msidx)),
                                     dtype=torch.bool, device=self.device)
        if self._c_prog:
            print(f"[carry3] 进度时钟 ON: 行进容差 {cfg.carry_adv_pos*100:.0f}cm/"
                  f"{np.degrees(cfg.carry_adv_rot):.0f}° | 启动限期 "
                  f"{cfg.carry_start_deadline} 步 | 里程碑 {len(self._c_msidx)} 个 "
                  f"@{[int(x) for x in self._c_msidx]}")
        # 手指参考逐侧换成抓形模板 (零残差 = 保持抓形; reset 的 tmpl0 也吃它)。
        # carry_squeeze (CARRY4): q_close ← squeeze 模板 ⟹ 合拢通道 = grasp→squeeze
        # 收紧轴, 策略自学何时挤 (对抗倾转扭矩); 关掉则保持旧行为 (q_close=抓形)。
        for ns, _pp in ((self._A, getattr(cfg, "grasp_prior_npz", "")),
                        (self._B, getattr(cfg, "prior_b_npz", ""))):
            with self._use(ns):
                self.q_open = self._p2_fin.clone()
                if getattr(cfg, "carry_squeeze", False) and _pp:
                    _zs = np.load(_pp, allow_pickle=True)
                    _sq = np.asarray(_zs["squeeze"], np.float64)[7:29][self._generic_perm]
                    _sq = np.clip(_sq, self.dof_lower[0].cpu().numpy(),
                                  self.dof_upper[0].cpu().numpy())
                    self.q_close = torch.tensor(_sq, dtype=torch.float32,
                                                device=self.device)
                    _dsq = float(torch.rad2deg(
                        (self.q_close - self._p2_fin).abs().mean()))
                    print(f"[carry4] {ns.name} 侧 q_close ← squeeze (距抓形均值 {_dsq:.1f}°)")
                else:
                    self.q_close = self._p2_fin.clone()
        # 扩展A: 瓶体倾角几何里程碑 (置信度无关); 上轴=首帧标定 R(q0)^T ẑ (§5 公约)
        _tms = [np.radians(float(x))
                for x in (getattr(cfg, "carry_tilt_ms_deg", ()) or ())]
        self._c_tilt_ms = torch.tensor(_tms, dtype=torch.float32, device=self.device)
        self._c_tiltdone = torch.zeros(N, max(1, len(_tms)), dtype=torch.bool,
                                       device=self.device)
        self._c_tiltnew = torch.zeros_like(self._c_tiltdone)
        with self._use(self._A):
            _q0 = self.obj_init_quat.to(torch.float32)
        _q0c = quat_conjugate(_q0.unsqueeze(0))
        self._c_up_body = self._qrot(_q0c, torch.tensor(
            [[0.0, 0.0, 1.0]], device=self.device))[0]     # 瓶身体系上轴
        if getattr(cfg, "pour_ff_play", False):
            # v8: 参考瓶倾角剖面 (逐 carry 行, 世界系 vs 重力) —— 自由段行进门用
            _qs = self._c_obj_quat["A"].to(torch.float32)
            _upw2 = self._qrot(quat_mul(_qs, _q0.unsqueeze(0).expand(
                _qs.shape[0], 4).to(torch.float32)),
                self._c_up_body.unsqueeze(0).expand(_qs.shape[0], 3))
            object.__setattr__(self, "_c_tiltref",
                               torch.acos(_upw2[:, 2].clamp(-1.0, 1.0)))
            print(f"[v8] 自由段臂前馈续播 ON: 参考倾角剖面 "
                  f"峰 {float(torch.rad2deg(self._c_tiltref.max())):.0f}° | "
                  f"行进容差 {cfg.pour_ff_tol_deg}°")
        # 扩展B: 低置信小糖宽门里程碑 (门=行进容差, 糖=carry_ms_low_bonus)
        _lms = [int(x) for x in (getattr(cfg, "carry_ms_low_idx", ()) or ())]
        self._c_low_idx = torch.tensor(_lms, dtype=torch.long, device=self.device)
        self._c_lowdone = torch.zeros(N, max(1, len(_lms)), dtype=torch.bool,
                                      device=self.device)
        self._c_lownew = torch.zeros_like(self._c_lowdone)
        if _tms or _lms:
            print(f"[carry4] 倾角里程碑 {[f'{np.degrees(x):.0f}°' for x in _tms]} "
                  f"(大糖 {getattr(cfg, 'carry_tilt_ms_bonus', 2.0)}) | 低置信里程碑 "
                  f"{_lms} (小糖 {getattr(cfg, 'carry_ms_low_bonus', 0.7)}, 宽门)")
        # C5 (2026-08-20 A2 裁定): 垫接触奖励进 carry —— 逐侧首触锁存缓冲
        if getattr(cfg, "carry_pad_reward", False):
            for ns in (self._A, self._B):
                with self._use(ns):
                    self._c5_pad_done = torch.zeros(N, 5, dtype=torch.bool,
                                                    device=self.device)
            print(f"[c5] 垫接触奖励 ON: 首触 {cfg.carry_w_pad_first}/垫 (轻触门) "
                  f"+ 持续 {cfg.carry_w_pad_hold}/步 (启动后)")
        # 甜甜圈稳抓链 (2026-08-20 用户裁定, 补 A2 缺口): 向心塑形 + candidate
        if getattr(cfg, "carry_stable", False):
            assert getattr(cfg, "carry_pad_reward", False), \
                "--carry_stable 依赖逐侧垫接触传感器 (--carry_pad_reward)"
            for ns in (self._A, self._B):
                with self._use(ns):
                    self._cc_run = torch.zeros(N, dtype=torch.long,
                                               device=self.device)
                    self._cc_done = torch.zeros(N, dtype=torch.bool,
                                                device=self.device)
            print(f"[c5] 稳抓链 ON: 向心塑形 {getattr(cfg, 'carry_w_cent', 0.05)}/步"
                  f"(全程, 倾斜段=传扭矩) + candidate ≥{cfg.success_min_pads}垫"
                  f"&向心≥{cfg.grasp_centrip_thresh}&静&{cfg.candidate_hold_steps}步 "
                  f"一次性 +{getattr(cfg, 'carry_w_cand', 5.0)} (启动前窗口=免滑移账)")
        # Pour (B5/B6 裁定): 瓶口/杯口几何点 (物体局部系, 由表面点云顶部簇心估计)
        if getattr(cfg, "pour_succ", False):
            def _top_center(_ns):
                with self._use(_ns):
                    _pts = self.obj_points.cpu().numpy()
                _zt = _pts[:, 2].max()
                return torch.tensor(_pts[_pts[:, 2] > _zt - 0.01].mean(0),
                                    dtype=torch.float32, device=self.device)
            object.__setattr__(self, "_pour_mouth_l", _top_center(self._A))
            object.__setattr__(self, "_pour_cuptop_l", _top_center(self._B))
            object.__setattr__(self, "_pour_mlow_done", torch.zeros(
                N, dtype=torch.bool, device=self.device))
            object.__setattr__(self, "_pour_hold", torch.zeros(
                N, dtype=torch.long, device=self.device))
            object.__setattr__(self, "_pour_ms_done", torch.zeros(
                N, dtype=torch.bool, device=self.device))
        if getattr(cfg, "pour_free", False):
            assert getattr(cfg, "pour_succ", False) and self._c_prog, \
                "--pour_free 依赖 --pour_succ (Success Tracker 判据) 与 --carry_progress"
            object.__setattr__(self, "_pf_on", torch.zeros(
                N, dtype=torch.bool, device=self.device))
            object.__setattr__(self, "_pf_steps", torch.zeros(
                N, dtype=torch.long, device=self.device))
            object.__setattr__(self, "_pf_prevd", torch.full(
                (N,), float("nan"), device=self.device))
            print(f"[pour_free] 自由探索段 ON: 行[{cfg.pour_free_lo},"
                  f"{cfg.pour_free_hi}) 参考停用/行进门·跟丢判挂起 | 预算 "
                  f"{cfg.pour_free_budget} 步 | 口对口势差分 w="
                  f"{cfg.pour_free_w_mouth} | 出段=pour成功→重拍rest锚→跳行"
                  f"{cfg.pour_free_hi} 续追 (根据: 重建帧60~82 误差30cm/可见率0.2)")
            print(f"[pour] 倒水里程碑 ON (B6 改判 2026-08-20: 非终点, 全程最大里程碑"
                  f" +{float(getattr(cfg, 'pour_w_ms', 10.0)):.0f}; 成功终止=走完轨迹末帧"
                  f" C5 同口径): 倾角≥{cfg.pour_succ_tilt_deg}° & 瓶口投影落"
                  f"杯口 {cfg.pour_succ_mouth_r*100:.0f}cm & 保持 {cfg.pour_succ_hold} 步 "
                  f"| 瓶口局部 {self._pour_mouth_l.cpu().numpy().round(3).tolist()} "
                  f"杯口局部 {self._pour_cuptop_l.cpu().numpy().round(3).tolist()}")
        # B4 裁定: 摩擦课程 —— 指垫 SuperGrip 开局 friction_hi, 随"稳抓交互存活率"
        # 慢EMA 棘轮退火回 friction_lo (帮助先学会保持抓稳, 再回真实摩擦)
        object.__setattr__(self, "_c5_mu_ema", 0.0)
        object.__setattr__(self, "_c5_mu_g", 0.0)
        if getattr(cfg, "friction_curriculum", False):
            self._c5_set_grip_mu(float(cfg.friction_hi))
            print(f"[c5] 摩擦课程 ON: {cfg.friction_hi} → {cfg.friction_lo} "
                  f"(驱动=稳抓存活慢EMA, 棘轮)")
        print(f"[carry-env] 路径 {self._c_T} 行 | 里程碑帧 {_ms} | "
              f"T_rel* 就位 | 手指参考="
              f"{'grasp→squeeze 收紧轴' if getattr(cfg, 'carry_squeeze', False) else '抓形模板'} (双侧)")

    def _c5_set_grip_mu(self, mu: float):
        """运行时改 SuperGrip 摩擦 (USD MaterialAPI 属性, physx 会跟随更新)。"""
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            p = stage.GetPrimAtPath("/World/Materials/SuperGrip")
            if p.IsValid():
                p.GetAttribute("physics:staticFriction").Set(float(mu))
                p.GetAttribute("physics:dynamicFriction").Set(float(mu))
                object.__setattr__(self, "_c5_mu_now", float(mu))
        except Exception as _e:
            print(f"[c5] ⚠ 摩擦课程写入失败: {_e}")

    @staticmethod
    def _qrot(q, v):
        w = q[:, 0:1]; u = q[:, 1:]
        return v + 2.0 * torch.cross(u, torch.cross(u, v, dim=1) + w * v, dim=1)

    @staticmethod
    def _qang(q1, q2):
        d = quat_mul(quat_conjugate(q1), q2)
        return 2.0 * torch.acos(d[:, 0].abs().clamp(max=1.0))

    def _carry_side_sig(self, ns, tag):
        """逐侧信号: 物体追踪误差 + slip (相对 T_rel* 规格)。"""
        with self._use(ns):
            _offs = int(getattr(self.cfg, "carry_ref_off", 0))
            t = (self.ref_t - _offs).clamp(min=0, max=self._c_T - 1)
            og = self.scene.env_origins
            op = self.object.data.root_pos_w - og
            oq = self.object.data.root_quat_w
            wp = self.wrist_pos_w - og
            wq = self._qsign(self.wrist_quat_w)
        tp = self._c_rest_p[tag] + self._c_obj_pos[tag][t]
        tq = quat_mul(self._c_obj_quat[tag][t], self._c_rest_q[tag])
        e_pos = (op - tp).norm(dim=1)
        e_rot = self._qang(oq, tq)
        # slip: 当前腕系下的物体相对位姿 vs 规格
        rel_q = quat_mul(quat_conjugate(wq), oq)
        rel_p = self._qrot(quat_conjugate(wq), op - wp)
        s_pos = (rel_p - self._c_rel_p[tag]).norm(dim=1)
        s_rot = self._qang(rel_q, self._c_rel_q[tag])
        _pend = self._c_snap_pending          # 基线/静置锚未立 (热身期): slip 与
        s_pos = torch.where(_pend, torch.zeros_like(s_pos), s_pos)   # 追踪误差都不计
        s_rot = torch.where(_pend, torch.zeros_like(s_rot), s_rot)   # (锚=启动时实拍,
        e_pos = torch.where(_pend, torch.zeros_like(e_pos), e_pos)   #  拍前目标未定义)
        e_rot = torch.where(_pend, torch.zeros_like(e_rot), e_rot)
        return dict(e_pos=e_pos, e_rot=e_rot, s_pos=s_pos, s_rot=s_rot, t=t,
                    rel_q=rel_q, rel_p=rel_p)

    # ---------------------------------------------------------------- 判据
    def _get_dones(self):
        term, trunc = super()._get_dones()          # 机制照跑 (ref_t/复位簿记)
        cfg = self.cfg
        if self._c_prog:
            self._c_step = self._c_step + 1
        if self._c_prog:
            _ripe = torch.zeros_like(self._c_snap_pending)   # 进度模式: 启动事件时拍照
        else:
            with self._use(self._A):
                _rt_now = self.ref_t.clone()
            _ripe = self._c_snap_pending & (_rt_now >= self._c_grip)
        if _ripe.any():
            pm = _ripe
            for tag, ns in (("A", self._A), ("B", self._B)):
                with self._use(ns):
                    og = self.scene.env_origins
                    op = self.object.data.root_pos_w - og
                    oq = self.object.data.root_quat_w
                    wp = self.wrist_pos_w - og
                    wq = self._qsign(self.wrist_quat_w)
                rq = quat_mul(quat_conjugate(wq), oq)
                rp = self._qrot(quat_conjugate(wq), op - wp)
                self._c_rel_q[tag][pm] = rq[pm]
                self._c_rel_p[tag][pm] = rp[pm]
                self._c_rest_p[tag][pm] = op[pm]     # 静置锚同刻实拍 (CARRY2 兼容路径)
                self._c_rest_q[tag][pm] = self._qsign(oq)[pm]
            self._c_snap_pending = self._c_snap_pending & ~pm
        sa = self._carry_side_sig(self._A, "A")
        sb = self._carry_side_sig(self._B, "B")
        self._c_sa, self._c_sb = sa, sb             # 奖励复用
        _start_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._c_adv = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._c_msnew = torch.zeros_like(self._c_msdone)
        if self._c_prog:
            # -- 启动按条件: rel 位姿逐步变化连续 K 步很小 → 拍基线+启动 --
            _stable = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            for tag, sg in (("A", sa), ("B", sb)):
                pq, pp = self._c_relprev[tag]
                dq = self._qang(torch.nan_to_num(pq, nan=1.0), sg["rel_q"])
                dp = (sg["rel_p"] - torch.nan_to_num(pp, nan=1e3)).norm(dim=1)
                _stable = _stable & (dp < 0.002) & (dq < np.radians(1.0))
                self._c_relprev[tag] = (sg["rel_q"].clone(), sg["rel_p"].clone())
            self._c_stab = torch.where(_stable & ~self._c_started,
                                       self._c_stab + 1, torch.zeros_like(self._c_stab))
            _go = (~self._c_started) & (self._c_stab >= cfg.carry_start_win)
            if getattr(cfg, "pour_e2e", False):
                # e2e 粘合 (2026-08-20 用户裁定): 双侧过微抬升验证 (真抓稳) 才允许
                # 进度时钟启动 —— 把"接近→抓稳"和"抓稳→追踪"焊在同一根轴上
                _su_a = self._A.data.get("succeeded")
                _su_b = self._B.data.get("succeeded")
                if _su_a is not None and _su_b is not None:
                    _go = _go & _su_a & _su_b
            if _go.any():
                for tag, sg, ns in (("A", sa, self._A), ("B", sb, self._B)):
                    self._c_rel_q[tag][_go] = sg["rel_q"][_go]
                    self._c_rel_p[tag][_go] = sg["rel_p"][_go]
                    with self._use(ns):     # 静置位姿同刻实拍 (位移算子的锚)
                        _og2 = self.scene.env_origins
                        self._c_rest_p[tag][_go] = (self.object.data.root_pos_w
                                                    - _og2)[_go]
                        self._c_rest_q[tag][_go] = self._qsign(
                            self.object.data.root_quat_w)[_go]
                self._c_started = self._c_started | _go
                self._c_snap_pending = self._c_snap_pending & ~_go
                if getattr(cfg, "pour_e2e", False):
                    # e2e 相位交接: LIFT→TRANSPORT (退出验证斜坡, 进 ff+残差纯控制态);
                    # 前馈接到缝行防猛甩 (place §2.19 同款教训: ref_q_prev 必须同步)
                    _off0 = int(getattr(cfg, "carry_ref_off", 0))
                    for ns in (self._A, self._B):
                        with self._use(ns):
                            _ph = self.task_phase
                            self.task_phase = torch.where(
                                _go, torch.full_like(_ph, Phase.TRANSPORT), _ph)
                            self.verify_k[_go] = 0
                            self.verify_ok_run[_go] = 0
                            self.ref_q_prev[_go] = self.retract_path[_off0]
            _start_fail = (~self._c_started) & (self._c_step > cfg.carry_start_deadline)
            if getattr(cfg, "pour_free", False):
                # 入段: 时钟触及低置信区起点 → 冻结时钟/前馈, 进自由探索
                _pf_enter = (self._c_started & ~self._pf_on & ~self._pour_ms_done
                             & (self._c_t >= int(cfg.pour_free_lo))
                             & (self._c_t < int(cfg.pour_free_hi)))
                self._pf_on = self._pf_on | _pf_enter
            # -- 行进按进度: 双物体进当前帧粗容差 → +1 --
            self._c_adv = (self._c_started
                           & (sa["e_pos"] < cfg.carry_adv_pos)
                           & (sb["e_pos"] < cfg.carry_adv_pos)
                           & (sa["e_rot"] < cfg.carry_adv_rot)
                           & (sb["e_rot"] < cfg.carry_adv_rot)
                           & (self._c_t < self._c_T - 1))
            if getattr(cfg, "pour_free", False):
                self._c_adv = self._c_adv & ~self._pf_on   # 自由段时钟冻结
                if getattr(cfg, "pour_ff_play", False) and \
                        getattr(self, "_c_tiltref", None) is not None and \
                        getattr(self, "_c_tilt_now", None) is not None:
                    # ★ v8 (2026-08-22 dump 破案): 冻结时钟 = 冻结臂前馈, 而演示的
                    # 倒水是**整臂重构**(肩 j1 摆 124°/上臂 j3 摆 94°), 残差 2°/步
                    # 在 7 维里发明不出来 —— 四条线倾角同卡 ~50°(纯手腕硬倒) 的
                    # 共同根因。改: 自由段时钟按**倾角剖面**推进(跟上才前进, 用的是
                    # 重建里可信的面内量), 让臂前馈继续播倒水动作; 位置门/跟丢判
                    # 仍挂起 (那才是低置信的部分)
                    _tr = self._c_tiltref[self._c_t.clamp(
                        max=self._c_tiltref.shape[0] - 1)]
                    _okt = (self._c_tilt_now - _tr).abs() < np.radians(
                        float(cfg.pour_ff_tol_deg))
                    self._c_adv = self._c_adv | (
                        self._pf_on & _okt & (self._c_t < self._c_T - 1))
            self._c_t = self._c_t + self._c_adv.long()
            # -- 高置信里程碑: 时钟已过且双物体贴紧 (2.5cm), 一次性 --
            if len(self._c_msidx):
                _near = ((sa["e_pos"] < cfg.carry_ms_tol)
                         & (sb["e_pos"] < cfg.carry_ms_tol)).unsqueeze(1)
                _passed = self._c_t.unsqueeze(1) >= self._c_msidx.unsqueeze(0)
                self._c_msnew = (_passed & _near & ~self._c_msdone
                                 & self._c_started.unsqueeze(1))
                self._c_msdone = self._c_msdone | self._c_msnew
            # -- 扩展A: 倾角里程碑 -- 实际瓶体倾角过档 + 双物在行进容差内(宽门,
            #    防"掀翻白拿糖": 摔在桌上的瓶离空中参考远, 过不了门), 一次性
            if self._c_tilt_ms.numel() or getattr(cfg, "pour_succ", False):
                with self._use(self._A):
                    _qA = self.object.data.root_quat_w
                _upw = self._qrot(_qA, self._c_up_body.unsqueeze(0).expand(
                    self.num_envs, 3))
                _tilt = torch.acos(_upw[:, 2].clamp(-1.0, 1.0))
                object.__setattr__(self, "_c_tilt_now", _tilt)  # pour 成功判据复用
                if self._c_tilt_ms.numel():
                    _nearw = ((sa["e_pos"] < cfg.carry_adv_pos)
                              & (sb["e_pos"] < cfg.carry_adv_pos)).unsqueeze(1)
                    if getattr(cfg, "pour_free", False):
                        # 自由段: 参考容差门失义, 改由滑移/thrown 守卫兜底
                        _nearw = _nearw | self._pf_on.unsqueeze(1)
                    self._c_tiltnew = ((_tilt.unsqueeze(1)
                                        > self._c_tilt_ms.unsqueeze(0))
                                       & _nearw & self._c_started.unsqueeze(1)
                                       & ~self._c_tiltdone)
                    self._c_tiltdone = self._c_tiltdone | self._c_tiltnew
            # -- 扩展B: 低置信小糖宽门 -- 时钟过行号 + 宽门(行进容差), 一次性
            if self._c_low_idx.numel():
                _nearw2 = ((sa["e_pos"] < cfg.carry_adv_pos)
                           & (sb["e_pos"] < cfg.carry_adv_pos)).unsqueeze(1)
                _pass2 = self._c_t.unsqueeze(1) >= self._c_low_idx.unsqueeze(0)
                self._c_lownew = (_pass2 & _nearw2 & ~self._c_lowdone
                                  & self._c_started.unsqueeze(1))
                self._c_lowdone = self._c_lowdone | self._c_lownew
            # -- 时钟写回两侧 (下一步 ff 用) --
            _offc = int(getattr(cfg, "carry_ref_off", 0))
            for ns in (self._A, self._B):
                with self._use(ns):
                    if getattr(cfg, "pour_e2e", False):
                        # e2e: 已启动 = carry 帧 (全局行号 +off); 未启动 = 接近段,
                        # 钳在缝行 —— 外生时钟不许把前馈拖进 carry 段
                        self.ref_t = torch.where(self._c_started,
                                                 self._c_t + _offc,
                                                 self.ref_t.clamp(max=_offc))
                    else:
                        self.ref_t = self._c_t.clone()
        slip = ((sa["s_pos"] > cfg.carry_slip_pos) | (sb["s_pos"] > cfg.carry_slip_pos)
                | (sa["s_rot"] > cfg.carry_slip_rot) | (sb["s_rot"] > cfg.carry_slip_rot))
        lost = (sa["e_pos"] > cfg.carry_lost_m) | (sb["e_pos"] > cfg.carry_lost_m)
        _dev = None
        if getattr(cfg, "pours_v6", False):
            # v6: 偏轨硬闸 (取代 thrown) —— 物体偏离自身参考 >dev_reset_m 无条件重置,
            # **不吃 pf_on 挂起** (用户: "无论任何条件都应该重置, 因为物体已经偏离了"),
            # 故在 pf 挂起之后再 OR 回来 (见下)
            _dev = ((sa["e_pos"] > float(cfg.dev_reset_m))
                    | (sb["e_pos"] > float(cfg.dev_reset_m)))
        _pf_trunc = None
        if getattr(cfg, "pour_free", False):
            lost = lost & ~self._pf_on          # 自由段跟丢判失义, 挂起
            self._pf_steps = torch.where(self._pf_on, self._pf_steps + 1,
                                         torch.zeros_like(self._pf_steps))
            _pf_trunc = self._pf_steps > int(cfg.pour_free_budget)
        at_end = sa["t"] >= self._c_T - 1
        if getattr(cfg, "pour_succ", False):
            # B6 改判 (2026-08-20 用户裁定): "倾角≥峰值71° & 瓶口对杯口 & 稳住
            # 10 步"**不是终点**, 是全程最大的关键里程碑 (一次性 pour_w_ms 大糖)。
            # 倒完水还要把瓶/杯大致还原放回 —— 加权物体轨迹的后半段就是还原过程,
            # 成功终止 = 走完轨迹末帧 (与 C5 同口径, 见下方共用 ok)。
            N = self.num_envs
            with self._use(self._A):
                _pA2 = self.object.data.root_pos_w
                _qA2 = self.object.data.root_quat_w
            with self._use(self._B):
                _pB2 = self.object.data.root_pos_w
                _qB2 = self.object.data.root_quat_w
            _mouth = _pA2 + self._qrot(_qA2, self._pour_mouth_l.unsqueeze(0)
                                       .expand(N, 3))
            _ctop = _pB2 + self._qrot(_qB2, self._pour_cuptop_l.unsqueeze(0)
                                      .expand(N, 3))
            _dmc = (_mouth[:, :2] - _ctop[:, :2]).norm(dim=1)
            object.__setattr__(self, "_dmc_now", _dmc)   # 供逐步日志/诊断
            _pok = (self._c_started
                    & (self._c_tilt_now >= np.radians(float(cfg.pour_succ_tilt_deg)))
                    & (_dmc < cfg.pour_succ_mouth_r))
            self._pour_hold = torch.where(_pok, self._pour_hold + 1,
                                          torch.zeros_like(self._pour_hold))
            _pnew = ((self._pour_hold >= int(cfg.pour_succ_hold))
                     & ~self._pour_ms_done)
            self._pour_ms_done = self._pour_ms_done | _pnew
            object.__setattr__(self, "_pour_ms_new", _pnew)
            if getattr(cfg, "pours_v6", False):
                # v6: 倒水完成 = 交互结束 = 回合**成功终止** (本版不做撤离恢复)
                self._c_succ = self._c_succ | _pnew
            if getattr(cfg, "pour_free", False):
                # 口对口势差分 (只在自由段计, 只赚缩短增量)
                _dn = torch.where(self._pf_on & torch.isfinite(self._pf_prevd),
                                  self._pf_prevd - _dmc, torch.zeros_like(_dmc))
                object.__setattr__(self, "_pf_dnew", _dn.clamp(-0.05, 0.05))
                self._pf_prevd = torch.where(
                    self._pf_on, _dmc, torch.full_like(_dmc, float("nan")))
                _ex = _pnew & self._pf_on
                if _ex.any():
                    # 出段: pour 成功 → 重拍 rest 锚(以行 hi 为参考基点) → 时钟跳 hi
                    _hi = int(cfg.pour_free_hi)
                    for tag, ns in (("A", self._A), ("B", self._B)):
                        with self._use(ns):
                            _og3 = self.scene.env_origins
                            _cp = self.object.data.root_pos_w - _og3
                            _cq = self._qsign(self.object.data.root_quat_w)
                        self._c_rest_p[tag][_ex] = (
                            _cp - self._c_obj_pos[tag][_hi].unsqueeze(0))[_ex]
                        self._c_rest_q[tag][_ex] = quat_mul(
                            quat_conjugate(self._c_obj_quat[tag][_hi]
                                           .unsqueeze(0).expand_as(_cq)), _cq)[_ex]
                    self._c_t[_ex] = _hi
                    if len(self._c_msidx):
                        _skip = ((self._c_msidx >= int(cfg.pour_free_lo))
                                 & (self._c_msidx < _hi)).unsqueeze(0)
                        self._c_msdone[_ex] = self._c_msdone[_ex] | _skip
                    self._pf_on[_ex] = False
                    print(f"[pour_free] ★ {int(_ex.sum())} env 倒水成功出段 → "
                          f"重锚续追行{_hi}", flush=True)
            # B5 裁定: 瓶口低于瓶质心水平面 = 倒得出的几何条件, 一次性糖
            _mlow = ((_mouth[:, 2] < _pA2[:, 2]) & self._c_started
                     & ~self._pour_mlow_done)
            self._pour_mlow_done |= _mlow
            object.__setattr__(self, "_pour_mlow_new", _mlow)
        # 成功终止 (C5/Pour 同口径): 完整走完加权物体轨迹 + 双侧误差达标 + 保持
        ok = (at_end & (sa["e_pos"] < cfg.carry_succ_pos)
              & (sb["e_pos"] < cfg.carry_succ_pos)
              & (sa["e_rot"] < cfg.carry_succ_rot)
              & (sb["e_rot"] < cfg.carry_succ_rot))
        _hold_req = int(cfg.carry_succ_hold)
        self._c_hold = torch.where(ok, self._c_hold + 1, torch.zeros_like(self._c_hold))
        newly = (self._c_hold >= _hold_req) & ~self._c_succ
        self._c_succ = self._c_succ | newly
        m = dict(self._sig_merged or {})
        m["newly_success"] = newly
        m["carry_slip"] = slip
        m["carry_lost"] = lost
        object.__setattr__(self, "_sig_merged", m)
        _k = getattr(self, "_bi_dbg", 0)
        if _k < 1200 and _k % 40 == 0:
            print(f"[carry] 步{_k:4d} t={int(sa['t'][0])}/{self._c_T} | "
                  f"瓶 e={float(sa['e_pos'].mean())*100:.1f}cm/"
                  f"{np.degrees(float(sa['e_rot'].mean())):.0f}° s={float(sa['s_pos'].mean())*100:.1f}cm/"
                  f"{np.degrees(float(sa['s_rot'].mean())):.0f}° | "
                  f"杯 e={float(sb['e_pos'].mean())*100:.1f}cm/"
                  f"{np.degrees(float(sb['e_rot'].mean())):.0f}° s={float(sb['s_pos'].mean())*100:.1f}cm/"
                  f"{np.degrees(float(sb['s_rot'].mean())):.0f}° | "
                  f"滑判 {int(slip.sum())} 丢判 {int(lost.sum())} 成 {int(self._c_succ.sum())}"
                  + (f" | 已启动 {int(self._c_started.sum())} 进度中位 "
                     f"{int(self._c_t.median())} 里程碑 {int(self._c_msdone.sum())}"
                     if self._c_prog else ""),
                  flush=True)
        if _pf_trunc is not None:
            trunc = trunc | _pf_trunc           # 自由段超预算: 截断 (不判负)
        if _dev is not None:
            lost = lost | _dev                  # v6 偏轨硬闸不吃 pf 挂起
        # 死因实名 (2026-08-23): 终止组件快照, 供 record 侧定罪 (回放批量早夭排查)
        object.__setattr__(self, "_kill_dbg", dict(
            base_term=term, slip=slip, lost=lost,
            dev=(_dev if _dev is not None else torch.zeros_like(term)),
            start_fail=_start_fail, c_succ=self._c_succ,
            pf_trunc=(_pf_trunc if _pf_trunc is not None
                      else torch.zeros_like(term))))
        return term | slip | lost | self._c_succ | _start_fail, trunc

    # ---------------------------------------------------------------- 奖励
    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        sa, sb = self._c_sa, self._c_sb
        r = torch.zeros(self.num_envs, device=self.device)
        # 逐步奖惩记账 (2026-08-21): 快照差分法 —— 组件 = 相邻检查点的 r 差,
        # 不改任何计算; cfg.step_reward_log 非空才生效
        _cksnap = [] if getattr(cfg, "step_reward_log", "") else None
        def _ck(name):
            if _cksnap is not None:
                _cksnap.append((name, r.clone()))
        _ck("_zero")
        for s in (sa, sb):
            r = r - cfg.carry_w_slip * (s["s_pos"] / 0.04
                                        + s["s_rot"] / np.radians(30.0))
        _ck("slip_pen")
        if getattr(cfg, "pours_v5", False):
            # ---- v5 (2026-08-21 用户裁定): 手物相对滑移的逐步增量罚 ----
            # 哲学: 本任务理想 = 全程手物零相对位移 (人抓水瓶不容一点滑)。
            # 每个 action 落地都用特权信息算"这一步新增了多少滑移", 从 0 起有
            # 梯度 (不等 4cm 带); 超线性 —— 滑得越快罚得越重 (快速止损信号)。
            if not hasattr(self, "_v5_sp"):
                _z = lambda: {t: torch.zeros(self.num_envs, device=self.device)
                              for t in ("right", "left")}
                for _nm in ("_v5_sp", "_v5_sr", "_v5_dvp", "_v5_dvr",
                            "_v5_f", "_v5_np"):
                    object.__setattr__(self, _nm, _z())
                object.__setattr__(self, "_v5ob", {})
            for _ns2, s in ((self._A, sa), (self._B, sb)):
                _sd = getattr(_ns2, "name", "right")
                dvp = (s["s_pos"] - self._v5_sp[_sd]).clamp(min=0.0)
                dvr = (s["s_rot"] - self._v5_sr[_sd]).clamp(min=0.0)
                self._v5_dvp[_sd], self._v5_dvr[_sd] = dvp, dvr
                self._v5_sp[_sd] = s["s_pos"].clone()
                self._v5_sr[_sd] = s["s_rot"].clone()
                _x = (dvp / 0.01 + dvr / np.radians(5.0)).clamp(max=2.0)
                r = r - float(cfg.slip_step_w) * _x * (1.0 + _x)
            _ck("slip_step")
            if int(getattr(cfg, "grip_prog_gate", 0)) > 0:
                # 进度门年龄: 时钟前进或身处 pf 窗 ⟹ 清零, 否则 +1 (每步一次)
                if not hasattr(self, "_v6_age"):
                    object.__setattr__(self, "_v6_age", torch.zeros(
                        self.num_envs, dtype=torch.long, device=self.device))
                _adv_now = self._c_adv.clone()
                if getattr(cfg, "pour_free", False):
                    if getattr(cfg, "pour_tilt_prog", False) and \
                            getattr(self, "_c_tilt_now", None) is not None:
                        # v6T (2026-08-22): 自由段时钟冻结, "进度"改判为**倾角创新高**
                        # —— 原先整窗豁免 ⟹ 窝在窗里每步领 grip_hold(每片+79) 比倾到
                        # 底(满 93° 才 +5) 划算 16 倍, 挂机模式在窗内复活 (账本实锤)
                        if not hasattr(self, "_v6_tmax"):
                            object.__setattr__(self, "_v6_tmax", torch.zeros(
                                self.num_envs, device=self.device))
                        _hi_new = self._c_tilt_now > (self._v6_tmax
                                                      + np.radians(1.0))
                        self._v6_tmax = torch.maximum(self._v6_tmax,
                                                      self._c_tilt_now)
                        _adv_now = _adv_now | (self._pf_on & _hi_new)
                    else:
                        _adv_now = _adv_now | self._pf_on
                self._v6_age = torch.where(_adv_now,
                                           torch.zeros_like(self._v6_age),
                                           self._v6_age + 1)
        if self._c_prog:
            # CARRY3: 按里程发钱 —— 前进一帧发微糖 (站着零收入, 防刷分钥匙);
            # 高置信里程碑一次性大糖; 无按时计费的追踪收入。
            r = r + 2.0 * cfg.carry_w_frame * self._c_adv.float()
            _ck("frame_adv")
            r = r + 2.0 * self._c_msnew.float().sum(dim=1)
            _ck("ms_pos")
            if self._c_tilt_ms.numel():        # 扩展A: 倾角大糖
                r = r + float(getattr(cfg, "carry_tilt_ms_bonus", 2.0)) \
                    * self._c_tiltnew.float().sum(dim=1)
            _ck("ms_tilt")
            if self._c_low_idx.numel():        # 扩展B: 低置信小糖
                r = r + float(getattr(cfg, "carry_ms_low_bonus", 0.7)) \
                    * self._c_lownew.float().sum(dim=1)
            if getattr(cfg, "pour_succ", False) and \
                    getattr(self, "_pour_mlow_new", None) is not None:
                r = r + cfg.pour_w_mouth_low * self._pour_mlow_new.float()
            if getattr(cfg, "pour_succ", False) and \
                    getattr(self, "_pour_ms_new", None) is not None:
                # B6 改判: 倒水姿态稳住 = 全程最大里程碑 (一次性), 非终点
                r = r + float(getattr(cfg, "pour_w_ms", 10.0)) \
                    * self._pour_ms_new.float()
            _ck("pour_ms")
            if getattr(cfg, "pour_free", False) and \
                    getattr(self, "_pf_dnew", None) is not None:
                # 自由段口对口势差分 (telescoping: 只赚净缩短)
                r = r + float(cfg.pour_free_w_mouth) * self._pf_dnew
            _ck("mouth_pot")
            if float(getattr(cfg, "pour_tilt_pot_w", 0.0)) > 0.0 and \
                    getattr(cfg, "pour_free", False) and \
                    getattr(self, "_c_tilt_now", None) is not None:
                # v6c (2026-08-22 值守裁定): 自由段倾角**势差分** —— 30/60/90 一次性
                # 糖太稀疏, 策略学成"进窗坐着"(窗倾角 90 分位 40°→18° 回退实锤);
                # 连续梯度补空档, telescoping 防振荡刷分, 只在 pf 窗支付
                _pt = (self._c_tilt_now
                       / np.radians(float(cfg.pour_succ_tilt_deg))).clamp(0.0, 1.2)
                if not hasattr(self, "_v6_tphi"):
                    object.__setattr__(self, "_v6_tphi", _pt.clone())
                _dpt = (_pt - self._v6_tphi).clamp(-0.1, 0.1)
                r = r + float(cfg.pour_tilt_pot_w) * _dpt * self._pf_on.float()
                self._v6_tphi = _pt.clone()      # 每步更新 (入窗/复位无跳变支付)
            _ck("tilt_pot")
            if float(getattr(cfg, "pour_prog_w", 0.0)) > 0.0:
                # v6P (2026-08-22 用户裁定): **全局进度势** —— 重建轨迹的绝对位姿
                # 不可信, 但"复现到了百分之几"可信。φ = 已复现行数/总行数, 自由段
                # 时钟冻结处用**倾角棘轮**换算成等效行进度 (面内量可信)。
                # telescoping ⟹ 0%→100% 总收入恰为 w, 不可刷; 与"握着不动"直接
                # 竞争 (账本: 挂机 +79/片 vs 旧 tilt_pot 满倾 +5)
                _lo6 = float(getattr(cfg, "pour_free_lo", 93))
                _hi6 = float(getattr(cfg, "pour_free_hi", 140))
                if not hasattr(self, "_pp_tmax"):
                    object.__setattr__(self, "_pp_tmax", torch.zeros(
                        self.num_envs, device=self.device))
                    object.__setattr__(self, "_pp_prev", torch.zeros(
                        self.num_envs, device=self.device))
                if getattr(self, "_c_tilt_now", None) is not None:
                    self._pp_tmax = torch.maximum(self._pp_tmax,
                                                  self._c_tilt_now)
                _inw = (self._pp_tmax
                        / np.radians(float(cfg.pour_succ_tilt_deg))).clamp(0, 1)
                if getattr(cfg, "pour_prog_align", False) and \
                        getattr(self, "_dmc_now", None) is not None:
                    # 2026-08-23: 倾角奖励只看"倾多少度"不看"往哪倾" ⟹ 策略朝任意
                    # 方向倾拿分, 瓶口反被甩远 (dmc 中位 23cm, 比参考 12.7 还差)。
                    # 进度改判 = 倾角进度 × **对齐进度** (棘轮), 两者都到位才算推进,
                    # 与成功判据(倾角≥93° 且 口距<6cm)同构
                    if not hasattr(self, "_pp_dmin"):
                        object.__setattr__(self, "_pp_dmin", torch.full(
                            (self.num_envs,), 1.0, device=self.device))
                    self._pp_dmin = torch.minimum(self._pp_dmin, self._dmc_now)
                    _d0 = float(getattr(cfg, "pour_align_d0", 0.25))
                    _dr = float(cfg.pour_succ_mouth_r)
                    _al = ((_d0 - self._pp_dmin) / max(_d0 - _dr, 1e-6)).clamp(0, 1)
                    _inw = _inw * _al
                if getattr(cfg, "pour_prog_joint", False) and \
                        getattr(self, "_c_tilt_now", None) is not None and \
                        getattr(self, "_dmc_now", None) is not None:
                    # v9J (2026-08-24 用户裁定): 联合乘积棘轮 —— 进度改记
                    # 「每步 倾角分×对齐分 乘积」的历史最大值。两棘轮各记各的
                    # 可被"先倾后凑"分时刷穿 (12M 实测: 圈内占时 10% 而同框步=0),
                    # 判据要的是同一时刻 ⟹ 把"同框"写进进度定义 (与判据逐点同构;
                    # 连续无悬崖, 45°×半对齐也给约一半率, 自带课程)
                    _d0j = float(getattr(cfg, "pour_align_d0", 0.25))
                    _drj = float(cfg.pour_succ_mouth_r)
                    _ftj = (self._c_tilt_now
                            / np.radians(float(cfg.pour_succ_tilt_deg))).clamp(0.0, 1.0)
                    _faj = ((_d0j - self._dmc_now)
                            / max(_d0j - _drj, 1e-6)).clamp(0.0, 1.0)
                    if not hasattr(self, "_pp_q"):
                        object.__setattr__(self, "_pp_q", torch.zeros(
                            self.num_envs, device=self.device))
                    self._pp_q = torch.maximum(self._pp_q, _ftj * _faj)
                    _inw = self._pp_q
                _rows = self._c_t.float() + torch.where(
                    self._pf_on, _inw * (_hi6 - _lo6),
                    torch.zeros_like(_inw))
                _phi6 = (_rows / _hi6).clamp(0.0, 1.0)
                _dphi = (_phi6 - self._pp_prev).clamp(min=0.0, max=0.05)
                r = r + float(cfg.pour_prog_w) * _dphi \
                    * self._c_started.float()
                self._pp_prev = _phi6.detach().clone()
            _ck("prog_pot")
            if getattr(cfg, "pour_trend_npz", "") and not hasattr(self, "_ptr_b"):
                _zt = np.load(cfg.pour_trend_npz)
                object.__setattr__(self, "_ptr_b", torch.tensor(
                    np.radians(_zt["bottle_deg"]), dtype=torch.float32,
                    device=self.device))
                object.__setattr__(self, "_ptr_c", torch.tensor(
                    np.radians(_zt["cup_deg"]), dtype=torch.float32,
                    device=self.device))
                object.__setattr__(self, "_pt_t", torch.zeros(
                    self.num_envs, dtype=torch.long, device=self.device))
                print(f"[v7] 趋势钟 ON: {cfg.pour_trend_npz} "
                      f"({len(_zt['bottle_deg'])}行, 瓶峰 {float(_zt['bottle_deg'].max()):.0f}°"
                      f"/杯峰 {float(_zt['cup_deg'].max()):.0f}°) | "
                      f"容差 {cfg.pour_trend_tol_deg}° 微糖 {cfg.pour_trend_w}/步")
            if getattr(self, "_ptr_b", None) is not None and \
                    getattr(cfg, "pour_free", False):
                # v7 趋势钟 (2026-08-22 用户裁定): 自由段 1 维倾角趋势参考 ——
                # 重建绝对位姿不可靠但倾角剖面(面内量)可靠, 只借这一维当路标。
                # 行进门(双物体倾角跟踪在容差内才走钟) + 每步前进微糖 (站着零收入,
                # 本周三次"小额常流收入养懒汉"手术的教训直接编译进来)
                with self._use(self._B):
                    _qB4 = self.object.data.root_quat_w
                _upB4 = self._qrot(_qB4, self._v5_upB.unsqueeze(0).expand(
                    self.num_envs, 3)) if hasattr(self, "_v5_upB") else None
                if _upB4 is not None:
                    _tB4 = torch.acos(_upB4[:, 2].clamp(-1.0, 1.0))
                    _T = self._ptr_b.shape[0]
                    _idx = self._pt_t.clamp(max=_T - 1)
                    _eA = (self._c_tilt_now - self._ptr_b[_idx]).abs()
                    _eB = (_tB4 - self._ptr_c[_idx]).abs()
                    _tol = np.radians(float(cfg.pour_trend_tol_deg))
                    _adv = (self._pf_on & (_eA < _tol) & (_eB < _tol)
                            & (self._pt_t < _T - 1))
                    self._pt_t = self._pt_t + _adv.long()
                    r = r + float(cfg.pour_trend_w) * _adv.float()
            _ck("trend_adv")
            if getattr(cfg, "carry_pad_reward", False):
                # A2 裁定: 垫接触奖励 (甜甜圈判据) —— 逐侧传感器已镜像, 各测各物
                for _ns in (self._A, self._B):
                    with self._use(_ns):
                        _Fp = torch.cat(
                            [s_.data.force_matrix_w.view(self.num_envs, 1, 3)
                             for s_ in self._contact_sensors], dim=1).nan_to_num(0.0)
                        _on = _Fp.norm(dim=-1) > 0.5
                        _spd = self.object.data.root_lin_vel_w.norm(dim=1)
                        _newp = (_on & (_spd < 0.05).unsqueeze(1)
                                 & ~self._c5_pad_done)
                        self._c5_pad_done = self._c5_pad_done | _newp
                        r = r + cfg.carry_w_pad_first * _newp.float().sum(dim=1)
                        r = r + cfg.carry_w_pad_hold * _on.float().mean(dim=1) \
                            * self._c_started.float()
                        if getattr(cfg, "carry_stable", False):
                            # 甜甜圈稳抓链: ①全程向心塑形 (法向压紧=可传扭矩,
                            # 倾斜段的真杠杆; δ 通道 17° 够得着); ②启动前
                            # candidate 大糖 —— T_rel* 启动时才实拍, 启动前
                            # 压紧/微调不吃滑移账, 教 RL 先抓稳再出发
                            _Fs = cfg.pad_force_sign * _Fp
                            _tipp = self.hand.data.body_pos_w[:, self.tip_ids]
                            _dirs = (self.object.data.root_pos_w.unsqueeze(1)
                                     - _tipp)
                            _dirs = _dirs / _dirs.norm(dim=-1, keepdim=True
                                                       ).clamp(min=1e-6)
                            _fn = _Fs / _Fs.norm(dim=-1, keepdim=True
                                                 ).clamp(min=1e-6)
                            _cent = ((-_fn) * _dirs).sum(dim=-1) * _on.float()
                            _cm = _cent.sum(dim=1) \
                                / _on.float().sum(dim=1).clamp(min=1.0)
                            _cw = (_cm.clamp(min=-0.5, max=1.0)
                                   if getattr(cfg, "cent_signed", False)
                                   else _cm.clamp(min=0.0))   # cent_fix: 负梯度
                            if getattr(cfg, "pours_v6", False):
                                # v6: 向心塑形只在交互开始前 (鼓励抓稳); 启动后
                                # 只要求手物零相对移动 (滑移账管), 大倾角下垫力
                                # 方向合理地偏离物心, 不再按向心扣分
                                _cw = _cw * (1.0 - self._c_started.float())
                            r = r + float(getattr(cfg, "carry_w_cent", 0.05)) * _cw
                            _cok = ((_on.float().sum(dim=1)
                                     >= cfg.success_min_pads)
                                    & (_cm >= cfg.grasp_centrip_thresh)
                                    & (_spd < 0.05) & ~self._c_started)
                            self._cc_run = torch.where(
                                _cok, self._cc_run + 1,
                                torch.zeros_like(self._cc_run))
                            _ccn = (self._cc_run
                                    >= int(cfg.candidate_hold_steps)) \
                                & ~self._cc_done
                            self._cc_done = self._cc_done | _ccn
                            r = r + float(getattr(cfg, "carry_w_cand", 5.0)) \
                                * _ccn.float()
                            if getattr(cfg, "pours_v5", False):
                                # ---- v5: 握紧反射 + 持续抓稳 (启动后全程) ----
                                _sd = getattr(_ns, "name", "right")
                                _fsum = _Fs.norm(dim=-1).sum(dim=1)
                                _npd = _on.sum(dim=1)
                                # ① 反射: 检测到滑移速度时, 垫压总量正差分给奖,
                                #   滑得越快奖越大 (×min(dv/1cm,1)); 不滑=0
                                _dv = (self._v5_dvp[_sd] / 0.01
                                       + self._v5_dvr[_sd]
                                       / np.radians(5.0)).clamp(max=1.0)
                                _df = (_fsum - self._v5_f[_sd]).clamp(0.0, 3.0)
                                r = r + float(cfg.regrip_w) * (_df / 3.0) \
                                    * _dv * self._c_started.float()
                                # ② 持续抓稳: 垫数≥模板指数 × 压力量级 × 向心,
                                #   每步小额; 压力超上限带反向罚 (防捏爆换分)
                                _hold = ((_npd >= cfg.success_min_pads).float()
                                         * (_fsum / float(cfg.grip_f_ref)
                                            ).clamp(0.0, 1.0)
                                         * (torch.ones_like(_cw)
                                            if getattr(cfg, "pours_v6", False)
                                            else _cw.clamp(min=0.0)))
                                # v6: 摘方向因子 —— cent 只属交互前; 持续抓稳只看
                                # 垫数×压力 (维持"手物零相对移动"的物质基础)
                                _pg = 1.0
                                _gpn = int(getattr(cfg, "grip_prog_gate", 0))
                                if _gpn > 0:
                                    # 进度门: 近 N 步时钟未前进 ⟹ 抓稳粮停发
                                    # (行军粮不是养老金; pf 窗豁免)
                                    if not hasattr(self, "_v6_age"):
                                        object.__setattr__(
                                            self, "_v6_age",
                                            torch.zeros(self.num_envs,
                                                        dtype=torch.long,
                                                        device=self.device))
                                    _pg = ((self._v6_age < _gpn)
                                           | self._pf_on).float()
                                r = r + float(cfg.grip_hold_w) * _hold \
                                    * self._c_started.float() * _pg
                                r = r - float(cfg.grip_hold_w) * (
                                    (_fsum - float(cfg.grip_f_max))
                                    / float(cfg.grip_f_ref)).clamp(0.0, 1.0)
                                self._v5_f[_sd] = _fsum.detach().clone()
                                self._v5_np[_sd] = _npd.float()
        else:
            for s in (sa, sb):
                r = r + cfg.carry_w_track * torch.exp(-s["e_pos"] / 0.05) \
                    * torch.exp(-s["e_rot"] / np.radians(30.0))
            t = sa["t"].unsqueeze(1)
            near = ((sa["e_pos"] < 0.08) & (sb["e_pos"] < 0.08)).unsqueeze(1)
            fire = (t >= self._c_ms_idx.unsqueeze(0)) & near & ~self._c_ms_done
            self._c_ms_done = self._c_ms_done | fire
            r = r + 2.0 * fire.float().sum(dim=1)
        _ck("pads_cent_cand")
        if getattr(cfg, "pours_v5", False):
            N = self.num_envs
            # ---- v5: 杯身直立约束 (堵"倾杯凑口"作弊 —— 倾角糖本就只发瓶侧,
            #   作弊赚的是 mouth_pot; 真实倒水里接水杯不动) ----
            if not hasattr(self, "_v5_upB"):
                with self._use(self._B):
                    _q0b = self.obj_init_quat.to(torch.float32)
                object.__setattr__(self, "_v5_upB", self._qrot(
                    quat_conjugate(_q0b.unsqueeze(0)),
                    torch.tensor([[0.0, 0.0, 1.0]], device=self.device))[0])
                # 物体表面点 (物体局部系, ≤48 点/物) —— 借臂外壳罚的表面点惯用法
                _pts = {}
                for _ns3 in (self._A, self._B):
                    with self._use(_ns3):
                        _P = self.obj_points.to(self.device, torch.float32)
                    # v6 纯碰撞口径要小边距 ⟹ 点要密 (48 点在 17cm 物上间距 ~2cm,
                    # 配 5mm 阈会漏检)
                    _k = min(96 if getattr(cfg, "pours_v6", False) else 48,
                             _P.shape[0])
                    _pts[getattr(_ns3, "name", "?")] = \
                        _P[torch.randperm(_P.shape[0], device=_P.device)[:_k]]
                object.__setattr__(self, "_v5_pts", _pts)
            with self._use(self._B):
                _qB = self.object.data.root_quat_w
            _upBw = self._qrot(_qB, self._v5_upB.unsqueeze(0).expand(N, 3))
            _tB = torch.acos(_upBw[:, 2].clamp(-1.0, 1.0))
            if not getattr(cfg, "pours_v6", False):
                # v6 删除杯直立罚 (演示杯对接窗倾到 ~57°; 统一哲学: 交互中物体
                # 位姿无约束, 非交互段稳定由 toppled 管) —— _tB 仅留观测用
                r = r - float(cfg.cup_upright_w) * (
                    (_tB - np.radians(float(cfg.cup_tilt_free_deg)))
                    / np.radians(30.0)).clamp(min=0.0, max=2.0)
            _ck("cup_upright")
            # ---- v5: 左右系统互不碰撞 (特权): 物↔物 / 手↔对侧物 / 手↔手 ----
            _pw, _tip = {}, {}
            for _ns3 in (self._A, self._B):
                _sd3 = getattr(_ns3, "name", "?")
                with self._use(_ns3):
                    _op = self.object.data.root_pos_w
                    _oq = self.object.data.root_quat_w
                    _tip[_sd3] = self.hand.data.body_pos_w[:, self.tip_ids]
                _P = self._v5_pts[_sd3]
                _K = _P.shape[0]
                _pw[_sd3] = _op.unsqueeze(1) + self._qrot(
                    _oq.unsqueeze(1).expand(-1, _K, -1).reshape(-1, 4),
                    _P.unsqueeze(0).expand(N, -1, -1).reshape(-1, 3)
                ).view(N, _K, 3)
            _sa3 = getattr(self._A, "name", "right")
            _sb3 = getattr(self._B, "name", "left")
            _doo = torch.cdist(_pw[_sa3], _pw[_sb3]).amin(dim=(1, 2))
            _dho = torch.minimum(
                torch.cdist(_tip[_sa3], _pw[_sb3]).amin(dim=(1, 2)),
                torch.cdist(_tip[_sb3], _pw[_sa3]).amin(dim=(1, 2)))
            _dhh = torch.cdist(_tip[_sa3], _tip[_sb3]).amin(dim=(1, 2))
            _mg = (0.005 if getattr(cfg, "pours_v6", False)
                   else float(cfg.cross_margin))
            # v6 (用户裁定): 全程开启但**纯碰撞口径** —— 不设安全圈, 只罚真挨上
            # (5mm≈点采样分辨率下的接触); 手手仍留 2 倍小余量
            _cp = (((_mg - _doo).clamp(min=0.0) / _mg).square()
                   + ((_mg - _dho).clamp(min=0.0) / _mg).square()
                   + (((2 * _mg) - _dhh).clamp(min=0.0) / (2 * _mg)).square())
            r = r - float(cfg.cross_pen_w) * _cp
            _ck("cross_pen")
            # ---- v5: 特权观测块 (8 维/侧, env._get_observations 末尾拼接) ----
            for _sd3, s3 in ((_sa3, sa), (_sb3, sb)):
                _cols = [
                    (s3["s_pos"] * 20.0).clamp(max=2.0),
                    (s3["s_rot"] / np.radians(30.0)).clamp(max=2.0),
                    (self._v5_dvp[_sd3] * 100.0).clamp(max=2.0),
                    (self._v5_dvr[_sd3] / np.radians(5.0)).clamp(max=2.0),
                    (self._v5_f[_sd3] / float(cfg.grip_f_ref)).clamp(max=2.0),
                    self._v5_np[_sd3] / 5.0,
                    (_doo * 20.0).clamp(max=2.0),
                    _tB / np.radians(45.0),
                ]
                if getattr(cfg, "pours_v6", False):
                    # v6: 阶段信号进观测 (用户裁定⑥: 不加"豁免位", 补阶段本身 ——
                    # 豁免/奖惩切换全是阶段的确定性函数, 策略看得见阶段即可自推)
                    _cols += [self._c_started.float(),
                              (self._pf_on.float()
                               if getattr(cfg, "pour_free", False)
                               else torch.zeros(N, device=self.device))]
                self._v5ob[_sd3] = torch.stack(_cols, dim=1)
        m = self._sig_merged or {}
        if "newly_success" in m:
            r = r + 20.0 * m["newly_success"].float()
        _ck("success")
        for ns in (self._A, self._B):
            ab = ns.data.get("actions_buf")
            if ab is not None:
                r = r - 0.001 * ab.square().mean(dim=1)
        if _cksnap is not None:
            _ck("act_reg")
            import os as _os
            if not hasattr(self, "_csrl"):
                object.__setattr__(self, "_csrl", {"n": 0, "shard": 0, "rows": []})
                _os.makedirs(cfg.step_reward_log, exist_ok=True)
            _P = 8
            _names = [nm for nm, _ in _cksnap[1:]]
            _prev = _cksnap[0][1]
            _mean, _probe = [], []
            for nm, snap in _cksnap[1:]:
                _dv = snap - _prev
                _prev = snap
                _mean.append(float(_dv.mean()))
                _probe.append(_dv[:_P].detach().cpu().numpy())
            self._csrl["rows"].append(dict(
                step=int(self._csrl["n"]),
                terms_mean=np.array(_mean, np.float32),
                terms_probe=np.stack(_probe).astype(np.float32),
                c_t=self._c_t[:_P].cpu().numpy().astype(np.int32),
                started=self._c_started[:_P].cpu().numpy(),
                pf_on=(self._pf_on[:_P].cpu().numpy()
                       if getattr(cfg, "pour_free", False)
                       else np.zeros(_P, bool)),
                tilt=(np.degrees(self._c_tilt_now[:_P].cpu().numpy())
                      .astype(np.float32)
                      if getattr(self, "_c_tilt_now", None) is not None
                      else np.zeros(_P, np.float32)),
                dmc=(self._dmc_now[:_P].cpu().numpy().astype(np.float32)
                     if getattr(self, "_dmc_now", None) is not None
                     else np.zeros(_P, np.float32))))
            self._csrl["n"] += 1
            if len(self._csrl["rows"]) >= 500:
                _R = self._csrl["rows"]
                np.savez_compressed(
                    _os.path.join(cfg.step_reward_log,
                                  f"carry_shard_{self._csrl['shard']:05d}.npz"),
                    term_names=np.array(_names),
                    step=np.array([x["step"] for x in _R], np.int64),
                    terms_mean=np.stack([x["terms_mean"] for x in _R]),
                    terms_probe=np.stack([x["terms_probe"] for x in _R]),
                    c_t=np.stack([x["c_t"] for x in _R]),
                    started=np.stack([x["started"] for x in _R]),
                    pf_on=np.stack([x["pf_on"] for x in _R]),
                    tilt=np.stack([x["tilt"] for x in _R]),
                    dmc=np.stack([x["dmc"] for x in _R]))
                self._csrl["shard"] += 1
                self._csrl["rows"] = []
        return r

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        ids = (torch.arange(self.num_envs, device=self.device)
               if env_ids is None else torch.as_tensor(env_ids, device=self.device))
        # ★起步强制 = 携带路径首行 (抓姿 IK) —— 基座的退避起点族终点虽是 GraspPose,
        #   但 stance_prob=1 会让它从**站姿**起步 (PGB 里站姿恰好=路径首行所以无感;
        #   carry 首行=抓姿, 不强制的话右臂从 46cm 外起步, slip 秒判负, 冒烟实锤)。
        if not getattr(self.cfg, "pour_e2e", False):
            # (e2e 不强制: 出生走底座 —— 站姿桶 t0=0 沿接近参考, 直抓桶 dgp 出生
            #  在 GraspPose; 参考首行=站姿, 无 46cm 跳变问题)
            q = self.hand.data.joint_pos[ids].clone()
            v = torch.zeros_like(self.hand.data.joint_vel[ids])
            for ns in (self._A, self._B):
                with self._use(ns):
                    row0 = self.retract_path[0]
                    q[:, self.arm_jids] = row0.unsqueeze(0)
                    self.q_cmd[ids] = row0
                    self.arm_tgt[ids] = row0
                    self.arm_tgt_prev[ids] = row0
                    self.ref_q_prev[ids] = row0
                    self.ref_t[ids] = 0
            self.hand.write_joint_state_to_sim(q, v, env_ids=ids)
            self.hand.set_joint_position_target(q, env_ids=ids)
        self._c_ms_done[ids] = False
        self._c_hold[ids] = 0
        self._c_succ[ids] = False
        # slip 基线在**复位后的第一个物理步**打快照 (reset 里 body 缓冲还是旧的)
        self._c_snap_pending[ids] = True
        if getattr(self.cfg, "pours_v5", False) and hasattr(self, "_v5_sp"):
            for _d in (self._v5_sp, self._v5_sr, self._v5_dvp, self._v5_dvr,
                       self._v5_f, self._v5_np):
                for _k in _d:
                    _d[_k][ids] = 0.0
        if hasattr(self, "_v6_age"):
            self._v6_age[ids] = 0
        if hasattr(self, "_v6_tmax"):
            self._v6_tmax[ids] = 0.0
        if hasattr(self, "_pp_tmax"):
            self._pp_tmax[ids] = 0.0
            self._pp_prev[ids] = 0.0
        if hasattr(self, "_pp_dmin"):
            self._pp_dmin[ids] = 1.0
        if hasattr(self, "_pp_q"):
            self._pp_q[ids] = 0.0
        if hasattr(self, "_pt_t"):
            self._pt_t[ids] = 0
        self._c_t[ids] = 0
        self._c_started[ids] = False
        self._c_stab[ids] = 0
        # C5 摩擦课程: 驱动=稳抓交互存活率(启动后再活 40 步)慢 EMA, 棘轮退火
        if getattr(self.cfg, "friction_curriculum", False) and len(ids):
            if getattr(self.cfg, "pour_e2e", False):
                # e2e 摩擦锚 (2026-08-20): 存活 = 进度过倾斜峰值帧 140 —— 旧口径
                # (启动+40步) 对倾斜段全盲, μ 退到底时瓶还在倾斜段死 (当日诊断)
                _sv = float((self._c_t[ids] >= 140).float().mean())
            else:
                _sv = float((self._c_step[ids]
                             > self.cfg.carry_start_deadline + 40).float().mean())
            object.__setattr__(self, "_c5_mu_ema",
                               0.98 * self._c5_mu_ema + 0.02 * _sv)
            _g = max(self._c5_mu_g, min(self._c5_mu_ema / 0.6, 1.0))
            if _g > self._c5_mu_g + 1e-6:
                object.__setattr__(self, "_c5_mu_g", _g)
                _mu = (self.cfg.friction_hi
                       + (self.cfg.friction_lo - self.cfg.friction_hi) * _g)
                if abs(_mu - getattr(self, "_c5_mu_now", 1e9)) > 0.05:
                    self._c5_set_grip_mu(_mu)
                    print(f"[c5] 摩擦退火: μ→{_mu:.2f} "
                          f"(稳抓存活EMA {self._c5_mu_ema:.3f})")
        if getattr(self.cfg, "pour_succ", False):
            self._pour_mlow_done[ids] = False
            self._pour_hold[ids] = 0
            self._pour_ms_done[ids] = False
        if getattr(self.cfg, "pour_free", False):
            self._pf_on[ids] = False
            self._pf_steps[ids] = 0
            self._pf_prevd[ids] = float("nan")
        self._c_step[ids] = 0
        self._c_msdone[ids] = False
        self._c_tiltdone[ids] = False
        self._c_lowdone[ids] = False
        for tag in ("A", "B"):
            self._c_relprev[tag][0][ids] = float("nan")
            self._c_relprev[tag][1][ids] = float("nan")
