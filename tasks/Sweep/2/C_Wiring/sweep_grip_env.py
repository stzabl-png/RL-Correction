"""Sweep2 体制 + 手指动作 + 可释放的附着 (Track B, 2026-09-13, 台账 Sweep/408 §9).

在 ``sweep_env.SweepEnv`` 上加三件事, 其余 (方块任务/四道门/奖励/母带时钟) 原样继承:
  ① 动作 14 -> 58: [R臂7, L臂7, R指22, L指22]; 指残差叠在先验 grasp 姿上 (408 同款界 0.03rad/步, ±0.60rad)
  ② 附着方式 SWEEP_ATTACH=joint|pin_release
       joint       : 原 USD FixedJoint 焊死 (B1: 只加手指, 工具仍焊着 —— 测 44 维手指本身有没有害)
       pin_release : 不建关节; 前 SWEEP_RELEASE_STEP 步每个物理子步把工具写到 手∘inv(grasp)
                     (408 钉住期同款, 碰撞全开、指力照常建立); 之后放手, 纯摩擦握
  ③ 放手后的 408 抓稳尺子 (SWEEP_SOFT_REL=1 才开软罚; 死线永远开):
       放手瞬间锁存掌系相对位姿; within = 1cm/5° (SWEEP_CERT_*), 死线 6cm/45° (SWEEP_DIE_*),
       软罚 −SOFT_W·ramp(认证窗→死线) 每步; 越死线 = 终止 −10
观测 = 原 191 + 指 q(44) + 指 qd(44) + 指残差(44) + 指 last_act(44) + 掌系漂移(12) + [released, certified](2) = 381
"""
from __future__ import annotations

import os

import numpy as np
import torch
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_inv, quat_mul

import sweep_env as _SE
from sweep_env import *  # noqa: F401,F403  (build_cfg/SPEC/REFERENCE/CUBE_VARIANTS/… 全部透传)
from sweep_env import SweepEnv as _Base, _qangle, ACT_DIM as _ARM_DIM, OBS_DIM as _BASE_OBS, PRIV_DIM

ATTACH = os.environ.get("SWEEP_ATTACH", "joint")
assert ATTACH in ("joint", "pin_release"), ATTACH
if ATTACH == "pin_release":
    # 钉住式沉降与关节沉降的静差不同 (sweep2 实测右 3.5mm vs 关节 1.3mm); 审计放到 ≥6mm
    # 解穿钳到 1 m/s 后 64 env 实测 右 13.1mm/1.7° (未钳时 1.18m/175°); 钉住工具压在指上的臂下垂就是这个量级,
    # 任务几何在簸箕系、母带跟踪容差 3cm, 2cm 审计足够 —— 这是复位一致性诊断, 不是任务判据
    # Denso 2080Ti/另一 IsaacLab 版本上 512 env 实测 max 21.6~25.9mm (本机 14.9mm), 中位/p90 仍是 3.7/6.9mm ⇒ 放到 40mm
    _SE.AUDIT_MAX_M = float(os.environ.get("SWEEP_AUDIT_MAX_M", max(float(_SE.SPEC.attach_audit_max_m), 0.040)))
RELEASE_STEP = int(os.environ.get("SWEEP_RELEASE_STEP", str(_SE.SCRIPTED_PRELUDE_STEPS)))   # 默认 = 前奏结束 = 策略接管那一步
# 为什么要和前奏对齐: 放手后零动作的扫把 30~50 步就滑过死线 (probe_pin_term 实测), 放手必须与策略接管同步;
# 又因钉住的工具是运动学体, 前奏里撞方块会把方块打飞 (实测掉到地上), 所以 408 上把前奏缩到 contact_row=20
SOFT_REL = os.environ.get("SWEEP_SOFT_REL", "0") == "1"
SOFT_W = float(os.environ.get("SWEEP_SOFT_W", "0.5"))
CERT_POS = float(os.environ.get("SWEEP_CERT_POS_CM", "1.0")) / 100.0
CERT_ROT = np.radians(float(os.environ.get("SWEEP_CERT_ROT_DEG", "5.0")))
DIE_POS = float(os.environ.get("SWEEP_DIE_POS_CM", "6.0")) / 100.0
DIE_ROT = np.radians(float(os.environ.get("SWEEP_DIE_ROT_DEG", "45.0")))
CERT_STEPS, B_CERT, B_DIE = 10, 5.0, -10.0
FIN_STEP, FIN_DEV = 0.03, 0.60
FIN_DIM = 44
ACT_DIM = _ARM_DIM + FIN_DIM                       # 58
OBS_DIM = _BASE_OBS + FIN_DIM * 4 + 12 + 2         # 381


def build_cfg(num_envs=1, reference=_SE.REFERENCE, ablation_method="full"):
    cfg = _SE.build_cfg(num_envs, reference, ablation_method)
    cfg.action_space = ACT_DIM
    cfg.observation_space = OBS_DIM
    return cfg


class SweepEnv(_Base):
    """名字保持 SweepEnv, 让 train/eval/smoke 通过 `SE.SweepEnv` 无差别使用。"""

    def __init__(self, cfg, **kw):
        self._attach = ATTACH
        self._pinned_ids = None
        self.grasp_T = None
        super().__init__(cfg, **kw)
        dev, N = self.device, self.num_envs
        to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
        # 指: fixed_finger_ids 顺序 = [右22, 左22] (基类按 right, left 装)
        self.fin_ids_t = torch.tensor(self.fixed_finger_ids, dtype=torch.long, device=dev)
        self.fin_cum = torch.zeros(N, FIN_DIM, device=dev)
        self.fin_step = to([FIN_STEP] * FIN_DIM)
        self.fin_dev = to([FIN_DEV] * FIN_DIM)
        self.last_act_full = torch.zeros(N, ACT_DIM, device=dev)
        self._ensure_grasp_T()
        self.released = torch.zeros(N, dtype=torch.bool, device=dev)
        self.certified = torch.zeros(N, dtype=torch.bool, device=dev)
        self.cert_run = torch.zeros(N, dtype=torch.long, device=dev)
        self.died = torch.zeros(N, dtype=torch.bool, device=dev)
        self.rel_p0 = {s: torch.zeros(N, 3, device=dev) for s in ("right", "left")}
        self.rel_q0 = {s: torch.zeros(N, 4, device=dev) for s in ("right", "left")}
        self.rel_q0["right"][:, 0] = 1; self.rel_q0["left"][:, 0] = 1
        self._grip_stats = {"episodes": 0, "cert": 0, "die": 0, "rel_sum": torch.zeros(4, device=dev), "rel_n": 0}
        for name in ("soft", "grip_bonus"):
            self._reward_names = tuple(self._reward_names) + (name,)
            self._reward_sums[name] = torch.zeros((), device=dev)
        print(f"[SweepGripEnv] attach={self._attach} release_step={RELEASE_STEP} soft_rel={SOFT_REL} (W={SOFT_W}) "
              f"cert={CERT_POS*100:.1f}cm/{np.degrees(CERT_ROT):.0f}° die={DIE_POS*100:.1f}cm/{np.degrees(DIE_ROT):.0f}° "
              f"act={ACT_DIM} obs={OBS_DIM}", flush=True)

    # ------------------------------------------------------------ 附着 ----
    def _create_tool_joints(self):
        if self._attach == "joint":
            return super()._create_tool_joints()
        print("[SweepGripEnv] pin_release: 不建 FixedJoint, 手↔工具碰撞全开 (408 钉住期同款)", flush=True)

    def _setup_scene(self):
        super()._setup_scene()
        if self._attach == "pin_release":
            self._tame_depenetration(v_max=float(os.environ.get("SWEEP_PIN_DEPEN_V", "0.2")))

    def _tame_depenetration(self, v_max=1.0):
        """pin_release 不加 own-hand 碰撞过滤 (放手后要靠指垫摩擦握), 钉住的工具与手指互穿时 MeshConverter 默认
        maxDepenetrationVelocity=1000 m/s 会把臂弹飞 (Denso 512 env 实测: 右工具审计 1.18m/175°, 8 env 时没暴露)。
        408 grip_env 同款: 压到 v_max, 工具 + 机器人刚体一起压。"""
        import omni.usd
        from pxr import PhysxSchema, Usd
        stage = omni.usd.get_context().get_stage()
        n = 0
        for ei in range(self.cfg.scene.num_envs):
            for nm in ("Object", "Aux", "Robot"):
                root = stage.GetPrimAtPath(f"/World/envs/env_{ei}/{nm}")
                if not root.IsValid():
                    continue
                for pr in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                    if pr.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                        PhysxSchema.PhysxRigidBodyAPI(pr).CreateMaxDepenetrationVelocityAttr().Set(float(v_max))
                        n += 1
        print(f"[SweepGripEnv] maxDepenetrationVelocity -> {v_max} m/s ×{n} 刚体 (Object/Aux/Robot)", flush=True)

    def _ensure_grasp_T(self):
        """工具 = 手 ∘ inv(grasp): grasp = 工具局部系里的手位姿 (FixedJoint 的 LocalPos1/Rot1)。"""
        if self.grasp_T is not None:
            return
        N, dev = self.num_envs, self.device
        to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
        self.grasp_T = {}
        for side, prior in (("right", np.load(_SE.PRIOR_BROOM)), ("left", self._pan_prior_npz)):
            g = np.asarray(prior["grasp"], np.float64)
            self.grasp_T[side] = (to(g[:3]).expand(N, 3).clone(), to(g[3:7]).expand(N, 4).clone())

    @property
    def tool_of(self):
        return {"right": self.object, "left": self.aux}

    def _hand_pose_w(self, side):
        b = self.hand_bid[side]
        return self.hand.data.body_pos_w[:, b], self.hand.data.body_quat_w[:, b]

    def _pin_tools(self, env_ids=None):
        """钉住期: 工具写到**母带当前行**的位姿 (408 grip_env 同款), 不跟手。

        首版 (工具 = 手∘inv(grasp), 逐子步跟手) 在 512 env 上有反馈环: 工具插在指上→推手→工具跟过去→再推,
        表现为簸箕口沿抖到 -10mm 触发 failed, 零动作回合 15 步就结束 (probe_pin_term.py 实测)。
        钉到母带位姿则手被推也不影响工具, 指的 PD 吸收互穿力 —— 408 Stage-1 30M 步验证过的机制。
        母带的手与工具由 hand=tool∘T_oh 构造, 两者只差 IK 残差 (0.02cm) + 臂 PD 下垂 (3~7mm)。
        """
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if len(ids) == 0:
            return
        row = self.row.clamp(max=self.T - 1)[ids]
        org = self.scene.env_origins[ids]
        for oi, art in ((0, self.aux), (1, self.object)):
            pose = torch.cat([self.ref_pos[oi][row] + org, self.ref_quat[oi][row]], 1)
            art.write_root_pose_to_sim(pose, env_ids=ids)
            art.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)

    def _settle_attachment_reset(self, physics_steps=24):
        if self._attach == "joint":
            return super()._settle_attachment_reset(physics_steps)
        # 无关节: 同一套沉降流程, 但每个子步把工具钉回手上
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        org = self.scene.env_origins
        qfull = self.hand.data.default_joint_pos.clone()
        qfull[:, self.map_ids_t] = self.ref_arm[0]
        qfull[:, self.fixed_finger_ids] = self.fixed_finger_q
        self.hand.write_joint_state_to_sim(qfull, torch.zeros_like(qfull))
        self.hand.set_joint_position_target(qfull)
        for oi, art in ((0, self.aux), (1, self.object)):
            pose = torch.cat([self.ref_pos[oi][0].expand(self.num_envs, 3),
                              self.ref_quat[oi][0].expand(self.num_envs, 4)], 1).clone()
            pose[:, :3] += org
            art.write_root_pose_to_sim(pose, env_ids=env_ids)
            art.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, device=self.device), env_ids=env_ids)
        dt = self.sim.get_physics_dt()
        for _ in range(int(physics_steps)):
            self.hand.set_joint_position_target(qfull); self.hand.write_data_to_sim()
            self._pin_tools(); self.sim.step(render=False); self.hand.update(dt)
        ref0 = self.ref_arm[0].expand(self.num_envs, -1)
        for _ in range(int(_SE.SPEC.attach_refine_rounds)):
            arm_err = ref0 - self.hand.data.joint_pos[:, self.map_ids_t]
            qfull[:, self.map_ids_t] += arm_err.clamp(-0.02, 0.02)
            for _ in range(8):
                self.hand.set_joint_position_target(qfull); self.hand.write_data_to_sim()
                self._pin_tools(); self.sim.step(render=False); self.hand.update(dt)
        self._pin_tools()
        self.object.update(dt); self.aux.update(dt)

    # ------------------------------------------------------------ 动作 ----
    def _pre_physics_step(self, actions):
        a = actions.clamp(-1.0, 1.0)
        policy_active = (self.episode_length_buf >= _SE.SCRIPTED_PRELUDE_STEPS).unsqueeze(1)
        a = a * policy_active
        self.last_act_full = a
        super()._pre_physics_step(a[:, :_ARM_DIM])          # 臂: 基类原样 (含 last_act/prev_act/q_tgt)
        self.fin_cum = torch.maximum(torch.minimum(self.fin_cum + a[:, _ARM_DIM:] * self.fin_step, self.fin_dev), -self.fin_dev)
        self._pinned_ids = torch.nonzero(self.episode_length_buf < RELEASE_STEP).flatten() \
            if self._attach == "pin_release" else None

    def _apply_action(self):
        self.hand.set_joint_position_target(self.q_tgt, joint_ids=self.map_ids)
        self.hand.set_joint_position_target(self.fixed_finger_q + self.fin_cum, joint_ids=self.fixed_finger_ids)
        if self._pinned_ids is not None:
            self._pin_tools(self._pinned_ids)

    # ------------------------------------------------------------ 抓稳尺子 ----
    def _rel(self, side):
        hp, hq = self._hand_pose_w(side)
        op, oq = self.tool_of[side].data.root_pos_w, self.tool_of[side].data.root_quat_w
        return quat_apply(quat_conjugate(hq), op - hp), quat_mul(quat_conjugate(hq), oq)

    def _grip_step(self):
        """-> (dp_r, dr_r, dp_l, dr_l, within, new_cert, new_die, r_soft)"""
        N, dev = self.num_envs, self.device
        if self._attach == "joint":
            z = torch.zeros(N, device=dev)
            return z, z, z, z, torch.ones(N, dtype=torch.bool, device=dev), torch.zeros(N, dtype=torch.bool, device=dev), \
                torch.zeros(N, dtype=torch.bool, device=dev), z
        rel_now = self.episode_length_buf >= RELEASE_STEP
        newrel = rel_now & ~self.released
        dp, dr = {}, {}
        for s in ("right", "left"):
            rp, rq = self._rel(s)
            if newrel.any():
                self.rel_p0[s][newrel] = rp[newrel]; self.rel_q0[s][newrel] = rq[newrel]
            self.released |= newrel
            dp[s] = torch.linalg.vector_norm(rp - self.rel_p0[s], dim=1) * self.released.float()
            dr[s] = _qangle(rq, self.rel_q0[s]) * self.released.float()
        within = torch.stack([(dp[s] < CERT_POS) & (dr[s] < CERT_ROT) for s in ("right", "left")], 1).all(1)
        cert_now = self.released & within
        self.cert_run = torch.where(cert_now, self.cert_run + 1, torch.zeros_like(self.cert_run))
        new_cert = (self.cert_run >= CERT_STEPS) & ~self.certified
        self.certified |= new_cert
        over = torch.stack([(dp[s] > DIE_POS) | (dr[s] > DIE_ROT) for s in ("right", "left")], 1).any(1)
        new_die = self.released & over & ~self.died
        self.died |= new_die
        r_soft = torch.zeros(N, device=dev)
        if SOFT_REL:
            terms = []
            for s in ("right", "left"):
                tp = (dp[s] - CERT_POS) / max(DIE_POS - CERT_POS, 1e-6)
                tr = (dr[s] - CERT_ROT) / max(DIE_ROT - CERT_ROT, 1e-6)
                terms.append(torch.maximum(tp, tr).clamp(0.0, 1.0))
            r_soft = -SOFT_W * torch.stack(terms, 1).mean(1) * (self.released & ~self.died).float()
        return dp["right"], dr["right"], dp["left"], dr["left"], within, new_cert, new_die, r_soft

    def _get_dones(self):
        terminated, timeout = super()._get_dones()
        dp_r, dr_r, dp_l, dr_l, within, new_cert, new_die, r_soft = self._grip_step()
        bonus = B_CERT * new_cert.float() + B_DIE * new_die.float()
        self._tick_out["reward"] = self._tick_out["reward"] + r_soft + bonus
        self._tick_out["reward_terms"]["soft"] = r_soft
        self._tick_out["reward_terms"]["grip_bonus"] = bonus
        self._reward_sums["soft"] += r_soft.detach().sum(); self._reward_sums["grip_bonus"] += bonus.detach().sum()
        self._tick_out["grip"] = dict(dp_r=dp_r, dr_r=dr_r, dp_l=dp_l, dr_l=dr_l, within=within,
                                      released=self.released.clone(), certified=self.certified.clone(), died=self.died.clone())
        if self._attach == "pin_release":
            # pin_release 全程不用 Sweep2 的"口沿低于桌面 -3mm = 失败"与 mouth_floor 罚: 那条规则假设簸箕焊在手上永不落桌;
            # 放手后簸箕靠摩擦握着, 自然要搁在桌面上 (零动作实测口沿 -0.5mm 中位, 偶发 -3.x), 真人扫地也是把簸箕放地上。
            # 8M 实测 mouth_floor 是仅次于 die 的最大罚项 (-0.12~-0.24/步), B4 因此塌成零动作。只保留"方块掉桌"失败。
            # ★ 奖励抵消不能受 suppress_terminal_reset 门控 (2026-09-13 录像器 record_sweep 开着这个旗, 曾把 B2/B4 录成 -156 的假账);
            #   只有终止改写才在录像模式下让位给基类。
            mf = self._tick_out["reward_terms"].get("mouth_floor")
            if mf is not None:
                self._tick_out["reward"] = self._tick_out["reward"] - mf          # 抵消基类的 mouth_floor 罚
                self._reward_sums["mouth_floor"] -= mf.detach().sum()
                self._tick_out["reward_terms"]["mouth_floor"] = torch.zeros_like(mf)
        if self._attach == "pin_release" and not bool(getattr(self, "suppress_terminal_reset", False)):
            cube_z = self.cube.data.root_pos_w[:, 2]
            failed = (cube_z < self.cfg.table_top_z - 0.03)
            terminated = self._tick_out["success"] | (failed & self.released) | new_die
            self._tick_out["terminated"] = terminated
            self._tick_out["timeout"] = self._tick_out["timeout"] & ~terminated
        if self.released.any():
            m = self.released
            self._grip_stats["rel_sum"] += torch.stack([dp_r[m].sum(), dr_r[m].sum(), dp_l[m].sum(), dr_l[m].sum()])
            self._grip_stats["rel_n"] += int(m.sum())
        return terminated, self._tick_out["timeout"]

    def _get_observations(self):
        base = super()._get_observations()
        q = self.hand.data.joint_pos[:, self.fin_ids_t]
        qd = self.hand.data.joint_vel[:, self.fin_ids_t]
        g = self._tick_out.get("grip") if self._tick_out is not None else None
        if g is None:
            z = torch.zeros(self.num_envs, device=self.device)
            g = dict(dp_r=z, dr_r=z, dp_l=z, dr_l=z, released=self.released, certified=self.certified)
        drift = []
        for s in ("right", "left"):
            rp, rq = self._rel(s)
            dq = quat_mul(quat_conjugate(self.rel_q0[s]), rq)
            drift.append(torch.cat([rp - self.rel_p0[s], 2.0 * dq[:, 1:] * torch.sign(dq[:, :1])], 1) * self.released.float().unsqueeze(1))
        obs = torch.cat([base["policy"], q, qd * 0.1, self.fin_cum / self.fin_dev, self.last_act_full[:, _ARM_DIM:],
                         *drift, self.released.float().unsqueeze(1), self.certified.float().unsqueeze(1)], 1)
        assert obs.shape[1] == OBS_DIM, obs.shape
        base["policy"] = obs.float().clamp(-10, 10).nan_to_num(0.0)
        return base

    def _reset_idx(self, env_ids):
        if len(env_ids) and self._tick_out is not None:
            ids = env_ids if torch.is_tensor(env_ids) else torch.tensor(env_ids, dtype=torch.long, device=self.device)
            self._grip_stats["episodes"] += len(ids)
            self._grip_stats["cert"] += int(self.certified[ids].sum()); self._grip_stats["die"] += int(self.died[ids].sum())
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        ids = env_ids if torch.is_tensor(env_ids) else torch.tensor(env_ids, dtype=torch.long, device=self.device)
        self.fin_cum[ids] = 0; self.last_act_full[ids] = 0
        self.released[ids] = False; self.certified[ids] = False; self.cert_run[ids] = 0; self.died[ids] = False
        if self._attach == "pin_release":
            self._pin_tools(ids)

    def pop_rates(self):
        out = super().pop_rates()
        n = max(self._grip_stats["episodes"], 1)
        out["grip/cert"] = self._grip_stats["cert"] / n
        out["grip/die"] = self._grip_stats["die"] / n
        if self._grip_stats["rel_n"]:
            v = (self._grip_stats["rel_sum"] / self._grip_stats["rel_n"]).cpu().numpy()
            out["grip/dp_broom_cm"], out["grip/dr_broom_deg"] = float(v[0] * 100), float(np.degrees(v[1]))
            out["grip/dp_pan_cm"], out["grip/dr_pan_deg"] = float(v[2] * 100), float(np.degrees(v[3]))
        out["residual/finger_usage"] = float((self.fin_cum.abs() / self.fin_dev).mean())
        self._grip_stats = {"episodes": 0, "cert": 0, "die": 0, "rel_sum": torch.zeros(4, device=self.device), "rel_n": 0}
        return out
