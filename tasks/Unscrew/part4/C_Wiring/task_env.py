"""Unscrew 任务训练环境 (框架身=Pour17 v5; [TASK] 块=扭盖换核)。

体制总纲 (判据单一来源 = A_Design/L3_Learning/progress*.py, 本文件只做接线):
  前馈 = 母带行 (时钟行), 残差 = 绝对锚有界累积, 零动作 = 放音。
  动作 58 = [R臂7, L臂7, R指22, L指22]。
  行指针 self.row: 机器行开环 1/步; 交互行 = IA0 + 进度机时钟(棘轮)。
  奖励 = adv + leash + ms + screw(拧转势) + 反射/斜坡/贴实 + 形状指引 - 死线罚。

与 Pour17 接线的差异 (逐条有据, 详见 REWARD_DOC):
  物体角色   env.object=瓶(左手, obj_0) env.aux=盖(右手, obj_1); 垫序 前5左/后5右
             (configure_cfg 按交互手=left 重建了基类传感器, 与 Pour 相反)
  螺旋       tasks/pregrasp/screw_assembly 既有通路 + 三个训练钩子:
             screw_drive_gain (U34 三指分级慢拧: 拇/食/中 1指1/3速 3指全速,
             无接触 0 —— 无摩擦解析螺旋会被亚阈值轻擦免费空转, 旧台账实测
             41% 假 release), screw_omega_damping=0.9 (螺纹粘滞),
             screw_detach_at_full (确定性策略拧满停手不该永卡)
  拧转势     r_screw = K_SCREW×Δθ (双向计, 回拧扣分; 总额≈9 与 G3 的 10 同量级)
  形状指引   [TASK] 换到手指通道 (V5 文档预留的"拧瓶盖三指扭动 prior 通道"):
             +0.2×W_HAND(物conf档)×W_HCONF(人手conf档)×cos(Δq右指实测,Δq人手指流)
             —— 臂通道弃用: 本批上游腕平移是静态填充死数据, 臂行是推导产物不是
             人手观测; 手指流是活的 (clip32 实测 44°/49° 幅度) 且有逐指置信度
  观测 507   = 框架 503 (物体块换 [盖-座差/瓶倾角] 语义, 维数不变) + 螺旋块 4
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from rl_rebuild.correction import clips
from tasks.pregrasp.cfg import GraspTaskCfg
from tasks.pregrasp.env import GraspTaskEnv

_HERE = os.path.dirname(os.path.abspath(__file__))
_L3 = os.path.abspath(os.path.join(_HERE, "..", "A_Design", "L3_Learning"))
sys.path.insert(0, _L3)
from progress import (  # noqa: E402
    CERT_RAMP, CERT_RET, D1_DROP, D2_TILT, D3_DEV, D4_SLIP,
    D5_BELOW_TABLE, G1_HOLD, PAD_FTH, PADS_MIN, SCREW_CONTACT_FTH,
    SCREW_TRIAD, UnscrewProgress, _axis_tilt)
from progress_batch import (  # noqa: E402
    M2_HOLD, M3_HOLD, M4_HOLD, TABLE_Z, UnscrewProgressBatch)

import task_config as TC  # noqa: E402

MASTER = os.environ.get("POUR_REF_NPZ") or (
    TC.REF_V2 if os.path.isfile(TC.REF_V2) else TC.REF_V1)
OBS_DIM = 507   # 框架 503 + 螺旋块 4 (frac/released/n_triad/gain)
ACT_DIM = 58
LOOK_KS = (1, 2, 4, 8, 16)
DEV_ARM_MACHINE, DEV_ARM_HUMAN = 0.05, 0.08
FAIL_PEN, D6_PEN, D6_CAP = -10.0, -0.5, -10.0
K_SCREW = 2.0                       # [TASK] 拧转势: 270°×2.0 ≈ 9.4 总额
_RPADS = ["right_thumb_elastomer", "right_index_elastomer",
          "right_middle_elastomer", "right_ring_elastomer",
          "right_pinky_elastomer"]


def build_cfg(num_envs=1):
    """框架配方: clip 场景 + 双手垫传感器 + D6 跨侧互撞."""
    from isaaclab.sensors import ContactSensorCfg
    cfg = GraspTaskCfg()
    clips.configure_cfg(cfg, TC.CLIP)     # hand_side=left, 左垫×/Object(瓶) 重建
    assert cfg.hand_side == "left", f"unscrew clip 应判左手, got {cfg.hand_side}"
    # 右垫 × /Aux(盖) —— 右手是拧盖手 (与 Pour 的左右互换同构)
    cfg.contact_sensors = list(cfg.contact_sensors) + [
        ContactSensorCfg(prim_path=f"/World/envs/env_.*/Robot/{n}",
                         history_length=1,
                         filter_prim_paths_expr=["/World/envs/env_.*/Aux"])
        for n in _RPADS]
    # U33 同款: 材质绑定是"整手 LowGrip → fingertip_bodies 覆盖回 SuperGrip",
    # fingertip_bodies 已被 configure_cfg 换成左侧 —— 右垫要显式补绑
    cfg.extra_supergrip_bodies = list(_RPADS)
    # D6 跨侧互撞 (判力不判距; POUR_NO_D6=1 回放规避 PhysX 断言)
    if os.environ.get("POUR_NO_D6") != "1":
        # One source body per sensor: PhysX requires filter expansions to
        # match the source count (Pour17 L5-25).
        _rd = ["R_arm_l5", "R_arm_l7", "R_arm_l8", "right_hand_C_MC"]
        _lf = [f"/World/envs/env_.*/Robot/{n}" for n in
               ("L_arm_l5", "L_arm_l7", "L_arm_l8", "left_hand_C_MC")]
        cfg.contact_sensors = list(cfg.contact_sensors) + [
            ContactSensorCfg(
                prim_path=f"/World/envs/env_.*/Robot/{n}",
                history_length=1,
                filter_prim_paths_expr=_lf,
            )
            for n in _rd]
    # 基类接近段机制**整个不接** (V5 接线全覆写; 接近由母带 Approach 行承载)。
    # approach=True 会一路索要 GraspPose prior (对齐势/退避族/canon 重摆 ——
    # 三者都与本任务冲突, 见 task_config PRIOR_MAIN 注释), 全关走最小脚手架。
    cfg.approach_only = False
    cfg.approach = False
    cfg.pregrasp_align = False
    cfg.screw_turns_override = TC.SCREW_TURNS
    # 父类初始化仍会构造 verify 斜坡；本任务把该 RL 面全部覆写，显式请求同设备
    # 零占位，而不是在 DexMate 共享层偷偷保留子类预设状态。
    cfg.bypass_lift_scaffold = True
    # [TASK] 盖侧无 GraspPose prior (设定 B); 左手瓶 squeeze 层由 env 前馈掺入
    if TC.PRIOR_MAIN:
        from tasks.pregrasp.cfg import apply_grasp_prior
        apply_grasp_prior(cfg, TC.PRIOR_MAIN, TC.PRIOR_APPROACH_DEG,
                          approach=True)
    cfg.scene.num_envs = num_envs
    cfg.obj_jitter_xy = 0.0
    cfg.action_space = ACT_DIM
    cfg.observation_space = OBS_DIM
    return cfg


class UnscrewEnv(GraspTaskEnv):
    """接线 env: 父类只借场景/资产/传感器/螺旋组件, RL 面全覆写."""

    def _setup_scene(self):
        super()._setup_scene()
        self._all_sensors = list(self._contact_sensors)
        self._contact_sensors = self._all_sensors[:5]   # 父类内部形状 5 (左垫)

    def __init__(self, cfg, **kw):
        super().__init__(cfg, **kw)
        dev, N = self.device, self.num_envs
        assert self.screw_spec is not None, "clip 无螺旋装配 —— 选错场景"
        assert abs(float(self.screw_spec.turns) - TC.SCREW_TURNS) < 1e-9, (
            self.screw_spec.turns, TC.SCREW_TURNS)
        self.pad_force_threshold_N = PAD_FTH
        self.pads_min_per_hand = PADS_MIN
        self._master_path = MASTER                 # 完整世界指纹的母带来源
        assert self._screw_primary == "body", self._screw_primary
        z = np.load(MASTER, allow_pickle=True)
        rows_h = np.where(np.asarray(z["source"]) == 1)[0]
        # ---- 母带 -> 58 维布局 + 关节映射 ----
        jn = list(self.hand.joint_names)
        fin = [str(n) for n in z["fin_names"]]
        ids = [jn.index(f"{P}_arm_j{i}") for P in ("R", "L") for i in range(1, 8)]
        for s in ("right", "left"):
            ids += [jn.index(n.replace("right_", f"{s}_")) for n in fin]
        assert len(set(ids)) == ACT_DIM
        self.map_ids = ids
        self.map_ids_t = torch.tensor(ids, dtype=torch.long, device=dev)
        T = len(np.asarray(z["source"]))
        ref = np.zeros((T, ACT_DIM))
        ref[:, 0:7] = np.asarray(z["right_q"], np.float64)
        ref[:, 7:14] = np.asarray(z["left_q"], np.float64)
        ref[:, 14:36] = np.asarray(z["right_f"], np.float64)
        ref[:, 36:58] = np.asarray(z["left_f"], np.float64)
        self.ref58 = torch.tensor(ref, dtype=torch.float32, device=dev)
        self.SRC = torch.tensor(np.asarray(z["source"], np.int64), device=dev)
        self.T_ROW = T
        self.IA0, self.IA1 = int(rows_h[0]), int(rows_h[-1])
        segl = [int(v) for v in z["seg_lens"]]
        self.EXIT0 = self.IA1 + 1                        # seam2 首行 = 换基捕获点
        self.D7 = int(self.T_ROW * 1.2) + 120
        self.RETREAT0 = segl[0] + segl[1] + segl[2] + segl[3]
        # 物体静置位 (母带交互首行; obj_0=瓶 obj_1=盖) + 末行目标
        self.rest_pose = {oi: torch.tensor(np.concatenate([
            np.asarray(z[f"obj_pos_{oi}"], np.float64)[rows_h][0],
            np.asarray(z[f"obj_quat_{oi}"], np.float64)[rows_h][0]]),
            dtype=torch.float32, device=dev) for oi in (0, 1)}
        # ---- 认证行 (提升测试臂参考, v2 母带才有) ----
        if "cert_arm7_right" in z:
            _ca = np.concatenate([np.asarray(z["cert_arm7_right"], np.float64),
                                  np.asarray(z["cert_arm7_left"], np.float64)])
            self.cert_arm14 = torch.tensor(_ca, dtype=torch.float32, device=dev)
        else:
            self.cert_arm14 = None
        # ---- [TASK] 人手手指形状先验 (P-HYB): 右指人手流的行间差分 ----
        self._variant = os.environ.get("POUR_VARIANT", "HYB").upper()
        assert self._variant in ("HYB", "OBJ"), self._variant
        _hkey = "human_right_f" if "human_right_f" in z else "right_f"
        if self._variant == "HYB":
            _hf = np.asarray(z[_hkey], np.float64)[rows_h]
            _dh = np.diff(_hf, axis=0)
            self.fin_dh = torch.tensor(np.vstack([_dh, np.zeros((1, 22))]),
                                       dtype=torch.float32, device=dev)
        else:
            self.fin_dh = None
        self._prev_finq = torch.zeros(N, 22, device=dev)
        # ---- 进度机 (判据单一来源) ----
        self.KCAP = int(os.environ.get("POUR_KCAP", "0"))
        self.PB = UnscrewProgressBatch(
            MASTER, num_envs=N, device=str(dev),
            kcap=self.KCAP if self.KCAP > 0 else None,
            no_hand_ref=(self._variant == "OBJ"))
        self._ps = UnscrewProgress(MASTER,
                                   no_hand_ref=(self._variant == "OBJ"))
        print(f"[UnscrewEnv] 变体={self._variant} clip={TC.CLIP_ID} "
              f"({'纯物轨消融' if self._variant == 'OBJ' else '置信门控+指形指引'})")
        self.unlocked = set(int(x) for x in
                            os.environ.get("POUR_UNLOCK", "").split(",") if x)
        self._rebuild_entries()
        self.p_t0 = 0.2
        self.phase_b = os.environ.get("POUR_BONUS_NOW") == "1"
        self.pad_pot_max = torch.zeros(N, 2, device=dev)
        self.force_entry = None
        # ---- 残差机械 ----
        to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
        arm_step = to(self.cfg.arm_residual_max) * float(self.cfg.arm_step_scale)
        fin_step = to(self.cfg.finger_residual_max) * float(self.cfg.finger_step_scale)
        self.step_bound = torch.cat([arm_step, arm_step, fin_step, fin_step])
        fin_dev = to(self.cfg.finger_residual_max) * float(self.cfg.finger_dev_scale)
        self.dev_fin = torch.cat([fin_dev, fin_dev])
        self.cum_res = torch.zeros(N, ACT_DIM, device=dev)
        self.rebase_off = torch.zeros(N, ACT_DIM, device=dev)
        self.rebase_armed = torch.zeros(N, dtype=torch.bool, device=dev)
        # ---- 逐env状态 ----
        self.row = torch.zeros(N, dtype=torch.long, device=dev)
        self.grasp_d0 = torch.full((N, 2), float("nan"), device=dev)
        self.d6_acc = torch.zeros(N, device=dev)
        self.last_act = torch.zeros(N, ACT_DIM, device=dev)
        self._prev_d = torch.zeros(N, 2, device=dev)
        self._prev_pf = torch.zeros(N, 2, device=dev)
        self._slip_obs = torch.zeros(N, 8, device=dev)
        self._tick_out = None
        self.prev_screw = torch.zeros(N, device=dev)     # [TASK] 拧转势差分
        self.screw_omega_damping = 0.9                   # [TASK] 螺纹粘滞 (v3)
        self.screw_detach_at_full = True                 # [TASK] 拧满即脱开
        self.screw_drive_gain = torch.zeros(N, device=dev)
        # 身体索引
        bn = list(self.hand.body_names)
        self.wid = {"R": bn.index("right_hand_C_MC"), "L": bn.index("left_hand_C_MC")}
        _pads = ["thumb", "index", "middle", "ring", "pinky"]
        self._pad_bids = [bn.index(f"left_{n}_elastomer") for n in _pads] + \
            [bn.index(f"right_{n}_elastomer") for n in _pads]
        self.hand_bids = [i for i, n in enumerate(bn)
                          if ("elastomer" in n or "hand" in n)]
        self.arm_jids_t = self.map_ids_t[:14]
        # ---- 参考系 z 对齐 (接线口径#3): 瓶母带静置 z vs 物理推导合法静置 z ----
        _orz = getattr(self, "obj_rest_z", None)
        if _orz is not None:
            dz0 = float(_orz) - float(self.rest_pose[0][2])
            print(f"[UnscrewEnv] 瓶 z 换基 Δz={dz0*100:+.2f}cm (obj_rest_z="
                  f"{float(_orz):.4f} vs 母带 {float(self.rest_pose[0][2]):.4f})")
            assert abs(dz0) < 0.05, f"瓶 z 差 {dz0*100:.1f}cm 异常大, 先查资产/母带"
            for oi in (0, 1):        # 盖合拢在瓶上, 随瓶同移
                self.PB.ref_obj[oi][:, 2] += dz0
                self.PB.rest[oi] = self.PB.ref_obj[oi][0].clone()
                self.PB.end[oi] = self.PB.ref_obj[oi][-1].clone()
                self.rest_pose[oi][2] += dz0
        # ---- squeeze 掺前馈 (山丘剖面; [TASK] 只有左手瓶侧有 prior) ----
        self.beta_l = float(os.environ.get("POUR_BETA_L", str(TC.BETA_L)))
        self.beta_r = float(os.environ.get("POUR_BETA_R", str(TC.BETA_R)))
        self.squeeze_ff_enabled = (
            os.environ.get("POUR_SQUEEZE_FF") == "1"
            and os.path.isfile(TC.PRIOR_AUX))
        self.sq_add = None
        self._sq_delta = None
        if self.squeeze_ff_enabled:
            _sql = np.asarray(np.load(TC.PRIOR_AUX)["squeeze"],
                              np.float64).reshape(-1)[7:29]
            _dsq = np.concatenate([
                self.beta_r * np.zeros(22),             # 右手盖: 无 squeeze prior
                self.beta_l * (_sql - ref[self.IA0, 36:58])])
            print(f"[UnscrewEnv] squeeze 剂量: βL={self.beta_l} "
                  f"(Screw27_body, 待 probe_beta 复标) βR={self.beta_r} "
                  "(盖侧无 prior)")
            self._sq_delta = torch.tensor(_dsq, dtype=torch.float32, device=dev)
        # ---- 手指/臂门 (Approach 冻结照谱, 缝1 起全开, Retreat 指冻臂开) ----
        self.APP_END = segl[0]
        gate_rows = torch.zeros(self.T_ROW, device=dev)
        gate_rows[self.APP_END:self.RETREAT0] = 1.0
        self.fin_gate_rows = gate_rows
        arm_gate = torch.ones(self.T_ROW, device=dev)
        arm_gate[:self.APP_END] = 0.0
        self.arm_gate_rows = arm_gate
        print(f"[UnscrewEnv] 手指门: 冻结 [0,{self.APP_END}) 与 "
              f"[{self.RETREAT0},末] | 全链 {T} 行 IA=[{self.IA0},{self.IA1}]")
        if self._sq_delta is not None:
            prof = torch.zeros(self.T_ROW, device=dev)
            r1 = torch.arange(self.APP_END, self.IA0, device=dev)
            prof[r1] = (r1 - self.APP_END).float() / max(self.IA0 - self.APP_END, 1)
            prof[self.IA0:self.IA1 + 1] = 1.0
            r2 = torch.arange(self.IA1 + 1, self.RETREAT0, device=dev)
            prof[r2] = 1.0 - (r2 - self.IA1).float() / max(
                self.RETREAT0 - self.IA1, 1)
            self.sq_add = prof.unsqueeze(1) * self._sq_delta.unsqueeze(0)
        # ---- RSI 焊接热身 ----
        self.HOLD_K = int(os.environ.get("POUR_HOLD_K", "15"))
        self.hold_left = torch.zeros(N, dtype=torch.long, device=dev)
        self.hold_pose = {oi: torch.zeros(N, 7, device=dev) for oi in (0, 1)}
        self.hold_released = torch.zeros(N, dtype=torch.bool, device=dev)
        # TB 计数
        self.racc = {"adv": 0.0, "leash": 0.0, "ms": 0.0, "pen": 0.0,
                     "pen6": 0.0, "bonus": 0.0, "regrip": 0.0, "slope": 0.0,
                     "wage": 0.0, "fshape": 0.0, "screw": 0.0, "n": 0}
        self.tb = {"term/M4_success": 0, "d6_pen_sum": 0.0, "ep": 0}
        self.diag_acc = {"screw_deg": 0.0, "released": 0, "n_triad": 0.0,
                         "gain": 0.0, "cap_any": 0.0, "escort_fail": 0.0,
                         "n": 0, "ep": 0}

    def _rebuild_entries(self):
        """按已解锁 Gate 重建出生表 (渐进RSI)."""
        self.entries = []
        for row_i, ms, label in self._ps.entry_table(self.unlocked):
            if label == "t0":
                self.entries.append((0, -1, frozenset(), "rest", "t0"))
            elif label == "ret":
                self.entries.append((self.RETREAT0, row_i, frozenset(ms),
                                     "ref", label))
            else:
                self.entries.append((self.IA0 + row_i, row_i, frozenset(ms),
                                     "ref" if row_i > 0 else "rest", label))
        print(f"[UnscrewEnv] RSI出生表: {[e[4] for e in self.entries]}")

    # ================= 动作 =================
    def _update_screw_drive_gain(self):
        """Refresh the contact-gated screw drive used by RL and manual probes."""
        fcap = self._pads_f().norm(dim=-1)[:, 5:]        # 右垫×盖
        n_any = (fcap > SCREW_CONTACT_FTH).sum(dim=1)
        n_triad = (fcap[:, list(SCREW_TRIAD)] > SCREW_CONTACT_FTH) \
            .sum(dim=1).float()
        self.screw_drive_gain = torch.where(
            n_any >= 1, (n_triad / 3.0).clamp(min=1.0 / 3.0, max=1.0),
            torch.zeros_like(n_triad))
        self._n_triad, self._n_cap_any = n_triad, n_any

    def _pre_physics_step(self, actions):
        a = actions.clamp(-1.0, 1.0)
        self.last_act = a.clone()
        r0 = self.row.clamp(max=self.T_ROW - 1)
        delta = a * self.step_bound
        delta[:, :14] *= self.arm_gate_rows[r0].unsqueeze(1)
        delta[:, 14:] *= self.fin_gate_rows[r0].unsqueeze(1)
        self.cum_res = self.cum_res + delta
        r = self.row.clamp(max=self.T_ROW - 1)
        src = self.SRC[r]
        dev_arm = (DEV_ARM_MACHINE + (DEV_ARM_HUMAN - DEV_ARM_MACHINE)
                   * (src == 1).float()).unsqueeze(1)
        self.cum_res[:, :14] = torch.maximum(
            torch.minimum(self.cum_res[:, :14], dev_arm), -dev_arm)
        self.cum_res[:, 14:] = torch.maximum(
            torch.minimum(self.cum_res[:, 14:], self.dev_fin), -self.dev_fin)
        cap = self.rebase_armed & (r >= self.EXIT0)
        if cap.any():
            qnow = self.hand.data.joint_pos[:, self.map_ids_t]
            self.rebase_off[cap] = (qnow - self.ref58[self.EXIT0].unsqueeze(0))[cap]
            self.rebase_armed[cap] = False
        self._update_screw_drive_gain()
        self._ff = self._ff_row(r)
        self.q_tgt = self._ff + self.cum_res

    def _ff_row(self, r):
        ff = self.ref58[r]
        if self.sq_add is not None:
            ff = ff.clone()
            ff[:, 14:] += self.sq_add[r]
        # 认证窗: 站位行臂参考向认证行插值 (v2 母带才有认证行)
        if self.cert_arm14 is not None and hasattr(self, "PB"):
            ph, ct = self.PB.cert_phase, self.PB.cert_t.float()
            a = torch.where(
                ph == 1, (ct + 1) / CERT_RAMP,
                torch.where(ph == 2, torch.ones_like(ct),
                            torch.where(ph == 3, 1.0 - (ct + 1) / CERT_RET,
                                        torch.zeros_like(ct)))).clamp(0, 1)
            a = a * (r == self.IA0).float()
            if bool((a > 0).any()):
                ff = ff.clone()
                ff[:, :14] = (ff[:, :14] * (1 - a).unsqueeze(1)
                              + self.cert_arm14.unsqueeze(0) * a.unsqueeze(1))
        span = max(self.T_ROW - 1 - self.EXIT0, 1)
        fade = ((self.T_ROW - 1 - r).float() / span).clamp(0.0, 1.0)
        post = (r >= self.EXIT0).float()
        return ff + self.rebase_off * (fade * post).unsqueeze(1)

    def _apply_action(self):
        self.hand.set_joint_position_target(self.q_tgt, joint_ids=self.map_ids)
        self._SA.apply_screw(self)          # [TASK] 螺旋投影 (每物理子步)

    # ================= 每步记账 =================
    def _get_dones(self):
        self._tick()
        o = self._tick_out
        return o["terminated"], o["timeout"]

    def _read_objs(self):
        org = self.scene.env_origins
        bot = torch.cat([self.object.data.root_pos_w - org,
                         self.object.data.root_quat_w], dim=1)
        cap = torch.cat([self.aux.data.root_pos_w - org,
                         self.aux.data.root_quat_w], dim=1)
        return bot, cap                                  # obj0=瓶, obj1=盖

    def _pads_f(self):
        F = torch.cat([s.data.force_matrix_w.view(self.num_envs, 1, 3)
                       for s in self._all_sensors[:10]], dim=1).nan_to_num(0.0)
        return F                                     # (N,10,3) 前5左vs瓶 后5右vs盖

    def _d6_hit(self):
        if len(self._all_sensors) <= 10:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        fm = torch.cat([s.data.force_matrix_w.reshape(self.num_envs, -1, 3)
                        for s in self._all_sensors[10:]], dim=1).nan_to_num(0.0)
        return (fm.norm(dim=-1) > 1.0).any(dim=1)

    def _tick(self):
        N, dev = self.num_envs, self.device
        self._SA.apply_screw(self, integrate_angle=False)   # 末子步后纯投影
        holding = self.hold_left > 0
        if holding.any():
            hids = holding.nonzero().squeeze(1)
            org = self.scene.env_origins[hids]
            for oi, art in ((0, self.object), (1, self.aux)):
                pose = self.hold_pose[oi][hids].clone()
                pose[:, :3] += org
                art.write_root_pose_to_sim(pose, env_ids=hids)
                art.write_root_velocity_to_sim(
                    torch.zeros(len(hids), 6, device=dev), env_ids=hids)
            self.hold_left[holding] -= 1
        bot, cap = self._read_objs()
        armq_r = self.hand.data.joint_pos[:, self.map_ids_t[:7]]
        armq_l = self.hand.data.joint_pos[:, self.map_ids_t[7:14]]
        finq_r = self.hand.data.joint_pos[:, self.map_ids_t[14:36]]
        # ---- G1 垫数原料: [TASK] 左手 >=3/5 垫 (右手时序上后进场) ----
        f = self._pads_f().norm(dim=-1)                  # (N,10) 前5左 后5右
        pads3 = (f[:, :5] > PAD_FTH).sum(dim=1) >= PADS_MIN
        # [TASK] T2-3 护送原料: 右手任一垫与盖有力 = 右手还拿着盖
        pads_r_cap = (f[:, 5:] > PAD_FTH).sum(dim=1) >= 1
        org_w = self.scene.env_origins
        wr_pos = self.hand.data.body_pos_w[:, self.wid["R"]] - org_w
        wl_pos = self.hand.data.body_pos_w[:, self.wid["L"]] - org_w
        # ---- [TASK] 螺旋状态 (释放锁存喂进度机) ----
        released = self.screw_has_depth & ~self.screw_engaged
        # ---- 进度机 (仅交互行起管辖) ----
        run_mask = (self.row >= self.IA0) & ~holding
        out = self.PB.step(bot, cap, armq_r, armq_l, pads3, wr_pos, wl_pos,
                           screw_released=released, run_mask=run_mask,
                           pads_r_cap=pads_r_cap)
        # ---- [TASK] 拧转势: K_SCREW×Δθ (双向计, 回拧扣分; prev 在 reset 时对齐) ----
        dtheta = self.screw_angle - self.prev_screw
        self.prev_screw = self.screw_angle.clone()
        r_screw = K_SCREW * dtheta * (~holding).float() * self.PB.g2.float()
        # ---- [TASK] 手指形状指引 (P-HYB): 物conf档 × 人手conf档 × cos ----
        r_fshape = torch.zeros(N, device=dev)
        if self.fin_dh is not None:
            ki_s = self.PB.k.clamp(max=self.PB.N_ROW - 1)
            dq_act = finq_r - self._prev_finq
            cs = torch.nn.functional.cosine_similarity(
                dq_act, self.fin_dh[ki_s], dim=1)
            r_fshape = 0.2 * out["w_hand"] * out["w_hconf_r"] \
                * cs.clamp(min=0.0) * (self.PB.g2 & run_mask).float()
        self._prev_finq = finq_r.detach().clone()
        # ---- env 侧死线 (机器段 D1/D2/D3 + D4 滑移 + D5 撞桌 + D7 超时) ----
        fail_env = torch.zeros(N, dtype=torch.bool, device=dev)
        pre = self.row < self.IA0
        for oi, _o in ((0, bot), (1, cap)):
            fail_env |= pre & (_o[:, 2] < TABLE_Z - D1_DROP)          # D1 机器段
            qn = _o[:, 3:7] / _o[:, 3:7].norm(dim=1, keepdim=True).clamp(min=1e-9)
            upw = quat_apply(qn, self.PB.up[oi].unsqueeze(0).expand(len(qn), 3))
            tilt = torch.acos((upw[:, 2] / upw.norm(dim=1).clamp(min=1e-9))
                              .clamp(-1, 1))
            fail_env |= pre & (tilt > D2_TILT)                         # D2 机器段
            rest = self.rest_pose[oi]
            fail_env |= pre & ((_o[:, :3] - rest[:3]).norm(dim=1) > D3_DEV)  # D3
        # D4 滑移 (交互行, G2 后, 相对基线): 左腕-瓶 / 右腕-盖
        d_l = (self.hand.data.body_pos_w[:, self.wid["L"]]
               - self.object.data.root_pos_w).norm(dim=1)
        d_r = (self.hand.data.body_pos_w[:, self.wid["R"]]
               - self.aux.data.root_pos_w).norm(dim=1)
        need0 = self.PB.g2 & torch.isnan(self.grasp_d0[:, 0])
        self.grasp_d0[:, 0] = torch.where(need0, d_l, self.grasp_d0[:, 0])
        self.grasp_d0[:, 1] = torch.where(need0, d_r, self.grasp_d0[:, 1])
        in_ia = (self.row >= self.IA0) & (self.row <= self.IA1)
        # [TASK] 右腕-盖滑移只在**未释放**时判 —— 释放后盖在右手里被携带,
        # 换把/递送是正业; 左腕-瓶全程判 (瓶不该离手)
        slip = in_ia & (~torch.isnan(self.grasp_d0[:, 0])) & (
            ((d_l - self.grasp_d0[:, 0]).abs() > D4_SLIP)
            | ((~released) & ((d_r - self.grasp_d0[:, 1]).abs() > D4_SLIP)))
        fail_env |= slip
        # ---- 滑移量/滑速/垫压 → 反射奖+斜坡罚+观测块 (框架 v5 移植) ----
        d_now = torch.stack([d_l, d_r], dim=1)
        pf_now = torch.stack([f[:, :5].sum(dim=1), f[:, 5:].sum(dim=1)], dim=1)
        d0 = torch.nan_to_num(self.grasp_d0, nan=0.0)
        has0 = ~torch.isnan(self.grasp_d0[:, 0])
        slip_amt = (d_now - d0) * has0.float().unsqueeze(1)
        slip_v = (d_now - self._prev_d).clamp(min=0.0)
        gate_g = (self.PB.g2 & has0).float()
        dv = (slip_v / 0.01).clamp(max=1.0)
        dfp = (pf_now - self._prev_pf).clamp(0.0, 3.0)
        r_reflex = 1.0 * ((dfp / 3.0) * dv).sum(dim=1) * gate_g
        pen_slope = -0.5 * ((slip_amt.clamp(min=0.0) / D4_SLIP) ** 2) \
            .clamp(max=1.0)[:, 0] * gate_g * in_ia.float()   # 斜坡只罚左侧(瓶)
        SQF = float(self.cfg.squeeze_f0)
        npads = torch.stack([(f[:, :5] > PAD_FTH).sum(dim=1),
                             (f[:, 5:] > PAD_FTH).sum(dim=1)], dim=1).float()
        self._slip_obs = torch.cat([
            (slip_amt / 0.05).clamp(-3, 3), (slip_v / 0.01).clamp(0, 3),
            (pf_now / SQF).clamp(0, 3), npads / 5.0], dim=1)
        self._prev_d = d_now.detach().clone()
        self._prev_pf = pf_now.detach().clone()
        # D5 撞桌
        hz = self.hand.data.body_pos_w[:, self.hand_bids, 2]
        fail_env |= (hz < TABLE_Z - D5_BELOW_TABLE).any(dim=1)
        # D7 超时
        timeout = self.episode_length_buf >= self.D7
        fail_env &= ~holding
        fail = out["fail"] | fail_env
        terminated = fail | out["done"]
        # ---- 行指针推进 ----
        self.row = torch.where(pre & ~holding, self.row + 1, self.row)
        ia = run_mask & (self.PB.k < self.PB.N_ROW - 1)
        self.row = torch.where(ia, self.IA0 + self.PB.k, self.row)
        post = run_mask & (self.PB.k >= self.PB.N_ROW - 1) \
            & (self.row >= self.IA1 - 1) & ~holding
        self.row = torch.where(post, (self.row + 1).clamp(max=self.T_ROW - 1),
                               self.row)
        # ---- 奖励合成 ----
        pen = torch.where(fail, torch.full((N,), FAIL_PEN, device=dev),
                          torch.zeros(N, device=dev))
        d6 = self._d6_hit() & ~holding
        pen6 = torch.where(d6 & (self.d6_acc > D6_CAP),
                           torch.full((N,), D6_PEN, device=dev),
                           torch.zeros(N, device=dev))
        self.d6_acc += pen6
        self.tb["d6_pen_sum"] += float(-pen6.sum())
        rew = (out["adv"] + out["leash"] + out["ms"] + out["wage"] + pen
               + r_reflex + pen_slope + r_fshape + r_screw) \
            * (~holding).float() + pen6
        bonus = torch.zeros(N, device=dev)
        # 贴实奖金 (框架 C 线: 距离势+力势各半, earn-only 棘轮, G1 后发放)
        if self.phase_b:
            r_now = self.row.clamp(max=self.T_ROW - 1)
            in_win = (r_now >= self.APP_END) & (r_now < self.RETREAT0) \
                & ~holding & self.PB.g1
            pot = torch.stack([(f[:, :5].clamp(0, 3) / 3).mean(dim=1),
                               (f[:, 5:].clamp(0, 3) / 3).mean(dim=1)], dim=1)
            if os.environ.get("POUR_BONUS_DIST") == "1":
                bn_pos = self.hand.data.body_pos_w
                pads_l = torch.stack([bn_pos[:, self._pad_bids[i]]
                                      for i in range(5)], dim=1)
                pads_r = torch.stack([bn_pos[:, self._pad_bids[i + 5]]
                                      for i in range(5)], dim=1)
                obj_xy = {"L": self.object.data.root_pos_w[:, :2],
                          "R": self.aux.data.root_pos_w[:, :2]}
                dpot = []
                # [TASK] 半径按物件: 瓶 3.25cm / 盖 1.75cm
                for pads, side, rr in ((pads_l, "L", 0.0325),
                                       (pads_r, "R", TC.CAP_RADIUS)):
                    d = ((pads[:, :, :2] - obj_xy[side].unsqueeze(1))
                         .norm(dim=-1) - rr).clamp(min=0)
                    dpot.append((1 - d / 0.05).clamp(0, 1).mean(dim=1))
                pot = 0.5 * torch.stack(dpot, dim=1) + 0.5 * pot
            inc = (pot - self.pad_pot_max).clamp(min=0) * in_win.float().unsqueeze(1)
            self.pad_pot_max = torch.maximum(self.pad_pot_max, pot)
            bonus = 0.5 * inc.sum(dim=1)
            rew = rew + bonus
        nh = (~holding).float()
        for kk, vv in (("adv", out["adv"]), ("leash", out["leash"]),
                       ("ms", out["ms"]), ("pen", pen), ("wage", out["wage"]),
                       ("regrip", r_reflex), ("slope", pen_slope),
                       ("fshape", r_fshape), ("screw", r_screw)):
            self.racc[kk] += float((vv * nh).sum())
        self.racc["pen6"] += float(pen6.sum())
        self.racc["bonus"] += float(bonus.sum())
        self.racc["n"] += N
        # [TASK] 螺旋诊断
        da = self.diag_acc
        da["screw_deg"] += float(torch.rad2deg(self.screw_angle).sum())
        da["released"] += int(released.sum())
        da["n_triad"] += float(self._n_triad.sum())
        da["gain"] += float(self.screw_drive_gain.sum())
        da["cap_any"] += float((self._n_cap_any >= 1).float().sum())
        da["escort_fail"] += float(self.PB.escort_fail.float().sum())
        da["n"] += N
        succ = self.PB.g4
        self._tick_out = {"terminated": terminated, "timeout": timeout & ~terminated,
                          "rew": rew, "out": out, "bot": bot, "cap": cap,
                          "armq_r": armq_r, "armq_l": armq_l,
                          "fail_env": fail_env, "succ": succ,
                          "released": released}

    def _get_rewards(self):
        return self._tick_out["rew"]

    def pop_racc(self):
        n = max(self.racc["n"], 1)
        out = {f"ep_rew/{k}": v / n for k, v in self.racc.items() if k != "n"}
        self.racc = {k: 0.0 for k in self.racc}
        self.racc["n"] = 0
        nd = max(self.diag_acc["n"], 1)
        out.update({f"diag/{k}": v / nd for k, v in self.diag_acc.items()
                    if k not in ("n", "ep")})
        self.diag_acc = {"screw_deg": 0.0, "released": 0, "n_triad": 0.0,
                         "gain": 0.0, "cap_any": 0.0, "escort_fail": 0.0,
                         "n": 0, "ep": 0}
        return out

    # ================= 观测 (507 = 框架503 + 螺旋块4) =================
    def _get_observations(self):
        N, dev = self.num_envs, self.device
        vs = float(self.cfg.obs_vel_scale)
        org = self.scene.env_origins
        q = self.hand.data.joint_pos[:, self.map_ids_t]
        qd = self.hand.data.joint_vel[:, self.map_ids_t]
        r = self.row.clamp(max=self.T_ROW - 1)
        bot, cap = self._read_objs()
        wp = {s: self.hand.data.body_pos_w[:, self.wid[s]] - org for s in ("R", "L")}
        wq = {s: self._qsign(self.hand.data.body_quat_w[:, self.wid[s]])
              for s in ("R", "L")}
        wlv = {s: self.hand.data.body_lin_vel_w[:, self.wid[s]] for s in ("R", "L")}
        wav = {s: self.hand.data.body_ang_vel_w[:, self.wid[s]] for s in ("R", "L")}
        ff = self._ff_row(r)
        src = self.SRC[r]
        dev_arm = (DEV_ARM_MACHINE + (DEV_ARM_HUMAN - DEV_ARM_MACHINE)
                   * (src == 1).float()).unsqueeze(1)
        res_n = torch.cat([self.cum_res[:, :14] / dev_arm,
                           self.cum_res[:, 14:] / self.dev_fin], dim=1)
        look = []
        for kk in LOOK_KS:
            rk = (r + kk).clamp(max=self.T_ROW - 1)
            look.append(self.ref58[rk][:, :14] - self.ref58[r][:, :14])
            in_ia_k = (rk >= self.IA0) & (rk <= self.IA1)
            wk = torch.where(in_ia_k,
                             self.PB.WO[self.PB.tmix[(rk - self.IA0).clamp(
                                 0, self.PB.N_ROW - 1)]],
                             torch.ones(N, device=dev))
            look.append(wk.unsqueeze(1))
        F = self._pads_f()
        qin_l = quat_conjugate(wq["L"]).unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
        qin_r = quat_conjugate(wq["R"]).unsqueeze(1).expand(-1, 5, -1).reshape(-1, 4)
        Gw = torch.cat([
            quat_apply(qin_l, F[:, :5].reshape(-1, 3)).reshape(N, 15),
            quat_apply(qin_r, F[:, 5:].reshape(-1, 3)).reshape(N, 15)],
            dim=1) / float(self.cfg.squeeze_f0)
        tipc = (F.norm(dim=-1) > PAD_FTH).float()
        tq = (self.hand.data.applied_torque[:, self.arm_jids_t] * 0.1).clamp(-3, 3)
        # 制度量纲 5
        ki = (r - self.IA0).clamp(0, self.PB.N_ROW - 1)
        in_ia = ((r >= self.IA0) & (r <= self.IA1)).float().unsqueeze(1)
        w_obj = self.PB.WO[self.PB.tmix[ki]].unsqueeze(1)
        lp = torch.minimum(self.PB.LP[self.PB.tp[0][ki]],
                           self.PB.LP[self.PB.tp[1][ki]]).unsqueeze(1)
        rot_ban = self.PB.rot_ban[torch.minimum(
            self.PB.tr[0][ki], self.PB.tr[1][ki])].float().unsqueeze(1)
        lr = torch.minimum(self.PB.LR[self.PB.tr[0][ki]],
                           self.PB.LR[self.PB.tr[1][ki]]).clamp(max=3.15).unsqueeze(1)
        gate_hand = (self.PB.tmix[ki] == 0).float().unsqueeze(1)
        regime = torch.cat([
            w_obj * in_ia + (1 - in_ia) * 1.0,
            lp * in_ia, lr * in_ia * (1 - rot_ban),
            rot_ban * in_ia + (1 - in_ia) * 1.0,
            gate_hand * in_ia], dim=1)
        # 物体块 38: 瓶-左腕系 / 盖-右腕系 + [TASK] 盖-座差/瓶倾角 + 偏差
        blocks = []
        for _o, s, art in ((bot, "L", self.object), (cap, "R", self.aux)):
            qi = quat_conjugate(wq[s])
            blocks += [quat_apply(qi, _o[:, :3] - wp[s]),
                       self._qsign(quat_mul(qi, _o[:, 3:7])),
                       art.data.root_lin_vel_w,
                       art.data.root_ang_vel_w * vs]
        qb = bot[:, 3:7] / bot[:, 3:7].norm(dim=1, keepdim=True).clamp(min=1e-9)
        seat_w = bot[:, :3] + quat_apply(
            qb, torch.tensor([0.0, 0.0, float(self.screw_spec.closed_offset_m)],
                             device=dev).unsqueeze(0).expand(N, 3))
        upv = quat_apply(qb, self.PB.up[0].unsqueeze(0).expand(N, 3))
        tiltcos = (upv[:, 2] / upv.norm(dim=1).clamp(min=1e-9)).unsqueeze(1)
        devs = []
        for oi, _o in ((0, bot), (1, cap)):
            refp = torch.where(in_ia.bool(),
                               self.PB.ref_obj[oi][ki][:, :3],
                               self.rest_pose[oi][:3].unsqueeze(0).expand(N, 3))
            dq = self.PB.ref_obj[oi][ki][:, 3:7]
            qo = _o[:, 3:7] / _o[:, 3:7].norm(dim=1, keepdim=True).clamp(min=1e-9)
            dang = 2 * torch.acos((qo * dq).sum(1).abs().clamp(max=1.0))
            devs += [_o[:, :3] - refp, dang.unsqueeze(1)]
        # 进度状态 7
        hold = torch.where(
            ~self.PB.g1, self.PB.g1_run.float() / G1_HOLD,
            torch.where(
                ~self.PB.g2,
                (self.PB.cert_phase.float() * 5 + self.PB.cert_t.float()) / 15.0,
                torch.where(~self.PB.g3,
                            (self.screw_angle
                             / (2 * np.pi * float(self.screw_spec.turns))
                             ).clamp(0, 1),
                            torch.where(~self.PB.placed,
                                        self.PB.m3_run.float() / M3_HOLD,
                                        self.PB.m4_run.float() / M4_HOLD)))
        ).clamp(0, 1)
        prog = torch.cat([
            (self.row.float() / (self.T_ROW - 1)).unsqueeze(1),
            self.PB.g1.float().unsqueeze(1), self.PB.g2.float().unsqueeze(1),
            self.PB.g3.float().unsqueeze(1), self.PB.g4.float().unsqueeze(1),
            hold.unsqueeze(1), src.float().unsqueeze(1)], dim=1)
        # [TASK] 螺旋块 4
        max_ang = 2 * np.pi * float(self.screw_spec.turns)
        released = (self.screw_has_depth & ~self.screw_engaged).float()
        screw_blk = torch.cat([
            (self.screw_angle / max_ang).clamp(0, 1).unsqueeze(1),
            released.unsqueeze(1),
            (getattr(self, "_n_triad", torch.zeros(N, device=dev)) / 3.0
             ).unsqueeze(1),
            self.screw_drive_gain.unsqueeze(1)], dim=1)
        obs = torch.cat([
            q[:, :14], q[:, 14:],                              # 58
            qd[:, :14] * vs, qd[:, 14:] * vs,                  # 58
            wp["R"], wq["R"], wlv["R"], wav["R"] * vs,         # 13
            wp["L"], wq["L"], wlv["L"], wav["L"] * vs,         # 13
            ff - q,                                            # 58
            res_n,                                             # 58
            *look,                                             # 75
            Gw.clamp(-3, 3), tipc,                             # 40
            tq,                                                # 14
            self.last_act,                                     # 58
            regime,                                            # 5
            *blocks, cap[:, :3] - seat_w, tiltcos, *devs,      # 38
            prog,                                              # 7
            self._slip_obs,                                    # 8
            screw_blk,                                         # 4 [TASK]
        ], dim=1).float().clamp(-float(self.cfg.clip_obs),
                                float(self.cfg.clip_obs)).nan_to_num(0.0)
        assert obs.shape[1] == OBS_DIM, f"obs {obs.shape[1]} != {OBS_DIM}"
        tip_f = F.norm(dim=-1)
        priv = torch.cat([self.obj_mass.float(), self.obj_fric.float(),
                          tip_f.float()], dim=1) \
            .clamp(-float(self.cfg.clip_obs), float(self.cfg.clip_obs)) \
            .nan_to_num(0.0)
        return {"policy": obs, "priv_info": priv}

    # ================= 重置 (RSI) =================
    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if self._tick_out is not None:
            o = self._tick_out
            self.tb["ep"] += len(env_ids)
            self.tb["term/M4_success"] += int(o["succ"][env_ids].sum())
            self.diag_acc["ep"] += len(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)
        n = len(env_ids)
        # ---- 采样进入点 ----
        if self.force_entry is not None:
            pick = [self.force_entry[int(i) % len(self.force_entry)]
                    for i in env_ids.cpu().tolist()]
        else:
            pick = []
            u = torch.rand(n)
            for i in range(n):
                if float(u[i]) < self.p_t0 or len(self.entries) <= 1:
                    pick.append(0)
                else:
                    pick.append(1 + int(torch.randint(len(self.entries) - 1,
                                                      (1,))))
        rows_env, rows_ia, g1, g2, g3, plc, objsrc = [], [], [], [], [], [], []
        for pi in pick:
            er, ir, ms, osrc, _ = self.entries[pi]
            rows_env.append(er); rows_ia.append(max(ir, 0))
            g1.append(1 in ms); g2.append(2 in ms); g3.append(3 in ms)
            plc.append("placed" in ms)
            objsrc.append(osrc)
        dev = self.device
        rows_env_t = torch.tensor(rows_env, dtype=torch.long, device=dev)
        self.row[env_ids] = rows_env_t
        t0_mask = torch.tensor([p == 0 for p in pick], device=dev)
        ent_ids = env_ids[~t0_mask]
        if len(ent_ids):
            sel = (~t0_mask).nonzero().squeeze(1).cpu().tolist()
            self.PB.enter(ent_ids,
                          torch.tensor([rows_ia[i] for i in sel], device=dev),
                          torch.tensor([g1[i] for i in sel], device=dev),
                          torch.tensor([g2[i] for i in sel], device=dev),
                          torch.tensor([g3[i] for i in sel], device=dev),
                          torch.tensor([plc[i] for i in sel], device=dev))
        if t0_mask.any():
            self.PB.reset_idx(env_ids[t0_mask])
        # ---- 写机器人状态 ----
        qfull = self.hand.data.default_joint_pos[env_ids].clone()
        qfull[:, self.map_ids_t] = self.ref58[rows_env_t]
        self.hand.write_joint_state_to_sim(qfull, torch.zeros_like(qfull),
                                           env_ids=env_ids)
        self.hand.set_joint_position_target(qfull, env_ids=env_ids)
        # ---- 写物体状态 + 螺旋复位 ----
        org = self.scene.env_origins[env_ids]
        # 瓶 (主体): 按出生行/静置; 同时更新父类 obj_start_* (reset_screw 依赖)
        pose_b = torch.zeros(n, 7, device=dev)
        for i, osrc in enumerate(objsrc):
            pose_b[i] = self.rest_pose[0] if osrc == "rest" \
                else self.PB.ref_obj[0][rows_ia[i]]
        self.obj_start_pos[env_ids] = pose_b[:, :3]
        self.obj_start_quat[env_ids] = pose_b[:, 3:7]
        pose_bw = pose_b.clone()
        pose_bw[:, :3] += org
        self.object.write_root_pose_to_sim(pose_bw, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(
            torch.zeros(n, 6, device=dev), env_ids=env_ids)
        # 螺旋: 先按合拢复位 (盖 seated + engaged), G3/ret 出生再改成已释放
        self._SA.reset_screw(self, env_ids)
        rel_born = torch.tensor(g3, dtype=torch.bool, device=dev)
        if rel_born.any():
            rids = env_ids[rel_born]
            sel = rel_born.nonzero().squeeze(1).cpu().tolist()
            pose_c = torch.stack([self.PB.ref_obj[1][rows_ia[i]] for i in sel])
            pose_cw = pose_c.clone()
            pose_cw[:, :3] += self.scene.env_origins[rids]
            self.aux.write_root_pose_to_sim(pose_cw, env_ids=rids)
            self.aux.write_root_velocity_to_sim(
                torch.zeros(len(rids), 6, device=dev), env_ids=rids)
            self.screw_engaged[rids] = False
            self.screw_has_depth[rids] = True
            self.screw_angle[rids] = 2 * np.pi * float(self.screw_spec.turns)
        self.prev_screw[env_ids] = self.screw_angle[env_ids]
        # ---- 热身钳位姿 (瓶=出生行; 盖=seated 或出生行) ----
        axis0 = quat_apply(pose_b[:, 3:7],
                           torch.tensor([0.0, 0.0, 1.0], device=dev)
                           .expand(n, 3))
        pose_c_hold = pose_b.clone()
        pose_c_hold[:, :3] += float(self.screw_spec.closed_offset_m) * axis0
        if rel_born.any():
            for j, i in enumerate(rel_born.nonzero().squeeze(1).cpu().tolist()):
                pose_c_hold[i] = self.PB.ref_obj[1][rows_ia[i]]
        self.hold_pose[0][env_ids] = pose_b
        self.hold_pose[1][env_ids] = pose_c_hold
        self.hold_left[env_ids] = self.HOLD_K
        # ---- 残差/记账状态 ----
        self.cum_res[env_ids] = 0.0
        self.rebase_off[env_ids] = 0.0
        armed = torch.tensor([self.entries[p][0] < self.EXIT0 for p in pick],
                             device=dev)
        self.rebase_armed[env_ids] = armed
        self.grasp_d0[env_ids] = float("nan")
        self._prev_finq[env_ids] = self.ref58[rows_env_t][:, 14:36]
        self.d6_acc[env_ids] = 0.0
        self.pad_pot_max[env_ids] = 0.0
        self._prev_d[env_ids] = 0.0
        self._prev_pf[env_ids] = 0.0
        self._slip_obs[env_ids] = 0.0
        self.last_act[env_ids] = 0.0
        self.screw_drive_gain[env_ids] = 0.0
