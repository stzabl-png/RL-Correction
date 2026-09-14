"""Sweep408 Stage-1 抓稳段 env —— "**移动时**不发生偏移" (用户 2026-09-11 裁定)。

回合 (20Hz):
  行 0..release-1 : 臂 = 母带第 0 行 (+一次性 PD 下垂补偿), 指 grasp→squeeze 斜坡 K_CLOSE 行,
                    两物体**钉住**在母带第 0 行世界位姿 (每物理子步写回, 碰撞全开, 指力照常建立)。
  行 release      : 放手, **锁存**两物体在各自掌系的位姿 (rel_p0 / rel_q0)。
  认证窗          : 连续 CERT_STEPS 步 掌系相对位姿 vs 锁存 <CERT_POS/CERT_ROT ∧ 接触条件 ⇒ 认证, 时钟 k 从 0 开走。
                    放手后 CERT_BUDGET 行内没认证 ⇒ 截断 (不另罚)。
  走母带 (k)      : 臂前馈 = 母带第 k 行 + 残差; 每步 k+1。走完 = 成功。
  死线 (终止)     : 掌系漂移 >DIE_POS/DIE_ROT | 物体落桌 | 手掌/指垫低于桌面。

为什么在**掌系**而不是世界系度量 (与 Clean/3 Stage-1 的关键差别):
  Clean 的抓稳段物体悬在空中、手不动, 世界系"不动"就等价于"握得住"。408 的物体贴着桌面且手要走母带 ——
  世界系度量下, 桌子替手扶着物体也能算"没动", 握得再松都能过认证。掌系度量把这条路堵死: 手一动,
  物体被桌子扶住就直接表现为漂移。于是桌面接触可以**保留**(它正是扫地时的主要扰动),
  不必像原方案那样关掉物体↔桌面碰撞 —— 那条降级成旗 SWEEP408_NO_TABLE_CONTACT=1, 默认不开。

动作 58 = [R臂7, L臂7, R指22, L指22] 有界累积残差 (Sweep2/Clean 同体制)。
角色: env.object = 扫把 (右手, 主体物) | env.aux = 簸箕 (左手)。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from rl_rebuild.correction import clips
from rl_rebuild.correction.kinematics import ArmIK
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior
from tasks.pregrasp.env import GraspTaskEnv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_config as TC  # noqa: E402

ACT_DIM, OBS_DIM, PRIV_DIM = 58, 368, 22
SIDES = ("right", "left")                 # right=扫把(主体/Object) left=簸箕(Aux)
OI = {"right": 0, "left": 1}
_LPADS = [f"left_{f}_elastomer" for f in ("thumb", "index", "middle", "ring", "pinky")]


def _safe_arm_reference(z, margin=1.0e-6):
    out = {}
    for side, key in (("right", "right_q"), ("left", "left_q")):
        ik = ArmIK(side)
        q = np.asarray(z[key], dtype=np.float64)
        out[key] = np.clip(q, ik.lower + margin, ik.upper - margin).astype(np.float32)
    return out


def _qangle(a, b):
    d = (a * b).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d)


def _quat_to_mat(q):
    """(N,4) wxyz -> (N,3,3)。"""
    q = torch.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)


def build_cfg(num_envs=1, reference=None):
    reference = reference or TC.REFERENCE
    assert os.path.isfile(reference), f"缺母带 {reference}; 先跑 A_Design/L2_Reference/build_reference.py"
    z = np.load(reference, allow_pickle=True)
    cfg = GraspTaskCfg()
    clips.configure_cfg(cfg, TC.CLIP)                  # hand_side=right; 右垫 × /Object(扫把)
    assert cfg.hand_side == "right", cfg.hand_side
    cfg.extra_supergrip_bodies = list(_LPADS)          # 左垫也要 SuperGrip (簸箕在左手)
    from isaaclab.sensors import ContactSensorCfg
    cfg.contact_sensors = list(cfg.contact_sensors) + [
        ContactSensorCfg(prim_path=f"/World/envs/env_.*/Robot/{n}", history_length=1,
                         filter_prim_paths_expr=["/World/envs/env_.*/Aux"]) for n in _LPADS]
    # 语义: 物体从第 0 行就在手里, 基类不得重摆/退避
    cfg.fixed_attached_tools = True
    cfg.approach_only = True
    # 扫把的"最大支撑面" ≠ Dexonomy 静置面 (上轴差 94.3°), 必须按后者摆, 否则抓姿整个转掉
    cfg.canon_rest_override = True
    # prior_yaw = -1 (让基类自搜可达 yaw)。这个闸检查的是"脚手架把物体静置在桌上时抓姿够不够得着",
    # 而本任务**起手就握住**: 每次 reset 都把两物体钉到母带第 0 行 (见 _reset_idx -> _pin),
    # 脚手架那个摆放当场被覆盖, 从不参与。钉死 yaw=0 时闸报 10.5cm 纯属对空气开枪。
    # 真正的可达性证据是母带自己的 ik_report: 双臂 <2cm 占比 100% (右中位 0.09cm / 左 0.25cm)。
    apply_grasp_prior(cfg, TC.PRIOR_BROOM, -1.0, approach=True)
    cfg.scene.num_envs = int(num_envs)
    cfg.obj_jitter_xy = 0.0
    cfg.action_space = ACT_DIM
    cfg.observation_space = OBS_DIM
    T = len(z["right_q"])
    cfg.sweep408_T_EP = int(TC.RELEASE_MAX + TC.CERT_BUDGET + round(T * TC.CLOCK_SLACK))
    cfg.episode_length_s = float((cfg.sweep408_T_EP + 2) / TC.CONTROL_HZ)
    cfg.sweep408_reference = os.path.abspath(reference)
    jp = dict(cfg.robot_cfg.init_state.joint_pos)
    safe = _safe_arm_reference(z)
    for side, P, key, prior in (("right", "R", "right_q", TC.PRIOR_BROOM),
                                ("left", "L", "left_q", TC.PRIOR_PAN)):
        for i, v in enumerate(safe[key][0], 1):
            jp[f"{P}_arm_j{i}"] = float(v)
        pr = np.load(prior)
        for n, v in zip(GENERIC_JOINT_ORDER, pr["grasp"][7:29]):
            jp[n.replace("right_", f"{side}_")] = float(v)
    cfg.robot_cfg.init_state.joint_pos = jp
    cfg.object_cfg.init_state.pos = tuple(float(v) for v in z["obj_pos_0"][0])
    cfg.object_cfg.init_state.rot = tuple(float(v) for v in z["obj_quat_0"][0])
    return cfg


class Sweep408GripEnv(GraspTaskEnv):
    def __init__(self, cfg, **kwargs):
        self._z = np.load(cfg.sweep408_reference, allow_pickle=True)
        super().__init__(cfg, **kwargs)
        dev, N = self.device, self.num_envs
        to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
        self.to = to
        safe = _safe_arm_reference(self._z)
        self.ref_arm = torch.cat([to(safe["right_q"]), to(safe["left_q"])], 1)   # (T,14) R 在前
        self.ref_pos = {i: to(self._z[f"obj_pos_{i}"]) for i in (0, 1)}
        self.ref_quat = {i: to(self._z[f"obj_quat_{i}"]) for i in (0, 1)}
        self.T_REF = self.ref_arm.shape[0]
        self.T_EP = int(cfg.sweep408_T_EP)
        self.map_ids = [self.hand.joint_names.index(f"{P}_arm_j{i}") for P in ("R", "L") for i in range(1, 8)]
        self.map_ids_t = torch.tensor(self.map_ids, dtype=torch.long, device=dev)
        self.fid, self.f_grasp, self.f_squeeze = {}, {}, {}
        for side, prior in (("right", np.load(TC.PRIOR_BROOM)), ("left", np.load(TC.PRIOR_PAN))):
            self.fid[side] = [self.hand.joint_names.index(n.replace("right_", f"{side}_")) for n in GENERIC_JOINT_ORDER]
            self.f_grasp[side] = to(prior["grasp"][7:29])
            self.f_squeeze[side] = to(prior["squeeze"][7:29])
        bn = list(self.hand.body_names)
        self.hand_bid = {s: bn.index(f"{s}_hand_C_MC") for s in SIDES}
        self.pad_ids = {s: [bn.index(f"{s}_{f}_elastomer")
                            for f in ("thumb", "index", "middle", "ring", "pinky")] for s in SIDES}
        self.art = {"right": self.object, "left": self.aux}
        self.dt = float(self.sim.get_physics_dt())
        self.n_sub = int(round(1.0 / (float(self._z["control_hz"]) * self.dt)))
        self.act_ids = torch.tensor(self.map_ids + self.fid["right"] + self.fid["left"], dtype=torch.long, device=dev)
        self.step_sz = to([TC.ARM_STEP] * 14 + [TC.FIN_STEP] * 44)
        self.dev_hi = to([TC.ARM_DEV] * 14 + [TC.FIN_DEV] * 44)
        self.cum_res = torch.zeros(N, ACT_DIM, device=dev)
        self.last_act = torch.zeros(N, ACT_DIM, device=dev)
        self.q_cmd = self.hand.data.default_joint_pos.clone()
        self.nominal = {s: (self.ref_pos[OI[s]][0].clone(), self.ref_quat[OI[s]][0].clone()) for s in SIDES}
        self.release_row_cur = int(os.environ.get("SWEEP408_RELEASE_START") or TC.RELEASE_MAX)
        self.release_row = torch.full((N,), int(TC.RELEASE_MAX), dtype=torch.long, device=dev)
        self.latched = torch.zeros(N, dtype=torch.bool, device=dev)
        self.rel_p0 = {s: torch.zeros(N, 3, device=dev) for s in SIDES}
        self.rel_q0 = {s: torch.zeros(N, 4, device=dev) for s in SIDES}
        self.cert = torch.zeros(N, dtype=torch.bool, device=dev)
        self.cert_run = torch.zeros(N, dtype=torch.long, device=dev)
        self.k = torch.zeros(N, dtype=torch.long, device=dev)
        self.died = torch.zeros(N, dtype=torch.bool, device=dev)
        self.die_kind = torch.zeros(N, dtype=torch.long, device=dev)
        self.succ = torch.zeros(N, dtype=torch.bool, device=dev)
        self.rel_max = torch.zeros(N, 4, device=dev)                # [dp_R, dp_L, dr_R, dr_L] 回合内最大
        self._low_ids = torch.tensor([self.hand_bid[s] for s in SIDES]
                                     + self.pad_ids["right"] + self.pad_ids["left"], device=dev)
        self._tick, self._acc = None, {}
        self._ep = {k: torch.zeros(N, device=dev) for k in
                    ("r_adv", "r_soft", "r_table", "r_contact", "r_bonus", "r_act",
                     "c_broom", "c_pan", "cert_timeout", "gap_broom_mm", "gap_pan_mm")}
        self.hull = self._table_hulls() if TC.USE_TABLE else None
        self.arm_sag = self._calibrate_sag()
        print(f"[Sweep408Grip] N={N} obs={OBS_DIM} act={ACT_DIM} 母带 {self.T_REF} 行 T_EP={self.T_EP} "
              f"release={self.release_row_cur} | 认证 {TC.CERT_POS*100:.1f}cm/{TC.CERT_ROT_DEG:.0f}°×{TC.CERT_STEPS} "
              f"死线 {TC.DIE_POS*100:.0f}cm/{TC.DIE_ROT_DEG:.0f}° | 接触 扫把拇指+≥{TC.BROOM_SUPPORT_MIN} 簸箕≥{TC.PAN_PADS_MIN} "
              f"| 桌面接触 {'关(旗)' if TC.NO_TABLE_CONTACT else '开'} | sag(deg)="
              f"{np.round(np.degrees(self.arm_sag.cpu().numpy()), 2).tolist()}", flush=True)

    # ------------------------------------------------------------ 场景 ----
    def _setup_scene(self):
        self.aux_init_pose_np = np.r_[np.asarray(self._z["obj_pos_1"])[0], np.asarray(self._z["obj_quat_1"])[0]]
        super()._setup_scene()
        self._all_sensors = list(self._contact_sensors)
        self._contact_sensors = self._all_sensors[:5]          # 基类内部形状只认 5 (hand_side=右垫)
        self._tame_depenetration()
        if TC.NO_TABLE_CONTACT:
            self._filter_object_table()

    def _tame_depenetration(self, v_max=1.0):
        """把两物体的 maxDepenetrationVelocity 从 MeshConverter 默认的 1000 m/s 压到 v_max。

        408 的母带第 0 行按设计把扫把头压在桌下 1mm (press_depth), 配 1000 m/s 的允许解穿速度,
        PhysX 一个子步 (1/240s) 就能把它弹出 3cm —— 实测: 钉住期物体位移 29.8mm、零动作放手后
        掌系漂移 22cm (Clean 的同类探针只有 0.96cm)。这是数值伪影, 不是我们想要的物理扰动。
        """
        import omni.usd
        from pxr import PhysxSchema, Usd
        stage = omni.usd.get_context().get_stage()
        n = 0
        for ei in range(self.cfg.scene.num_envs):
            for nm in ("Object", "Aux"):
                root = stage.GetPrimAtPath(f"/World/envs/env_{ei}/{nm}")
                if not root.IsValid():
                    continue
                for pr in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                    if pr.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                        PhysxSchema.PhysxRigidBodyAPI(pr).CreateMaxDepenetrationVelocityAttr().Set(float(v_max))
                        n += 1
        print(f"[Sweep408Grip] maxDepenetrationVelocity -> {v_max} m/s ×{n} 刚体", flush=True)

    def _filter_object_table(self):
        """旗 SWEEP408_NO_TABLE_CONTACT=1: 物体↔桌面不发生物理接触 (Sweep2 ramp×Table 同款手法)。"""
        import omni.usd
        from pxr import Sdf, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        n = 0
        for ei in range(self.cfg.scene.num_envs):
            tbl = Sdf.Path(f"/World/envs/env_{ei}/Table")
            if not stage.GetPrimAtPath(tbl).IsValid():
                continue
            for nm in ("Object", "Aux"):
                pr = stage.GetPrimAtPath(f"/World/envs/env_{ei}/{nm}")
                if pr.IsValid():
                    UsdPhysics.FilteredPairsAPI.Apply(pr).CreateFilteredPairsRel().AddTarget(tbl); n += 1
        print(f"[Sweep408Grip] 物体↔桌面碰撞已过滤 ×{n}", flush=True)

    # ------------------------------------------------------------ 工具 ----
    def _org(self):
        return self.scene.env_origins

    def _phys(self, qfull, n):
        for _ in range(int(n)):
            self.hand.set_joint_position_target(qfull)
            self.hand.write_data_to_sim()
            self.sim.step(render=False)
            self.hand.update(self.dt); self.object.update(self.dt); self.aux.update(self.dt)
            for s_ in getattr(self, "_all_sensors", []):
                s_.update(self.dt)

    def _hand_pose(self, side):
        b = self.hand_bid[side]
        return self.hand.data.body_pos_w[:, b] - self._org(), self.hand.data.body_quat_w[:, b]

    def _obj_pose(self, side):
        a = self.art[side]
        return a.data.root_pos_w - self._org(), a.data.root_quat_w

    def _rel(self, side):
        """物体在**掌系**的位姿 (p, q)。"""
        hp, hq = self._hand_pose(side); op, oq = self._obj_pose(side)
        return quat_apply(quat_conjugate(hq), op - hp), quat_mul(quat_conjugate(hq), oq)

    def _pad_force_mat(self, side):
        sens = self._all_sensors[:5] if side == "right" else self._all_sensors[5:10]
        cols = []
        for s_ in sens:
            fw = s_.data.net_forces_w
            cols.append(torch.linalg.vector_norm(fw.reshape(self.num_envs, -1, 3), dim=2).amax(dim=1))
        return torch.stack(cols, 1)

    def _ff_fingers(self, row):
        a = (row.float() / float(TC.K_CLOSE)).clamp(0.0, 1.0).unsqueeze(1)
        return torch.cat([(1 - a) * self.f_grasp[s] + a * self.f_squeeze[s] for s in ("right", "left")], 1)

    def _pin(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if len(ids) == 0:
            return
        org = self._org()[ids]
        for s in SIDES:
            p, q = self.nominal[s]
            pose = torch.cat([p.expand(len(ids), 3) + org, q.expand(len(ids), 4)], 1)
            self.art[s].write_root_pose_to_sim(pose, env_ids=ids)
            self.art[s].write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)

    def _calibrate_sag(self, steps=8, rounds=3, refine=4):
        """一次性 PD 下垂补偿 (Sweep2/Clean 同款): 钉住物体、指在 grasp, 臂目标微调到实测=参考。"""
        N = self.num_envs
        qfull = self.hand.data.default_joint_pos.clone()
        qfull[:, self.map_ids_t] = self.ref_arm[0]
        for s in SIDES:
            qfull[:, self.fid[s]] = self.f_grasp[s]
        self.hand.write_joint_state_to_sim(qfull, torch.zeros_like(qfull))
        for _ in range(rounds):
            self._pin(); self._phys(qfull, steps)
        ref0 = self.ref_arm[0].expand(N, -1)
        for _ in range(refine):
            err = ref0 - self.hand.data.joint_pos[:, self.map_ids_t]
            qfull[:, self.map_ids_t] += err.clamp(-0.02, 0.02)
            self._pin(); self._phys(qfull, steps)
        return (qfull[:, self.map_ids_t] - ref0).mean(dim=0)

    def _table_hulls(self):
        """头/斗段的**凸包顶点** (物体局部系), 用来算最低点离桌高度。

        刚体在任意姿态下的最低点必落在凸包顶点上, 所以取凸包即精确。
        ⚠ 不要再抽样: 第一版按 256 个球面方向取支撑点抽到 152 个, 实测贴桌量偏 5.6mm ——
        扫把头底是一大片近乎水平的平面, 抽样只留下**角点**, 而角点在平面略微倾斜时不是最低点,
        于是系统性高估。全量凸包也不贵: 最低点只需要 z 分量, 用 R[:,2,:] @ hullᵀ 得 (N,K),
        而不是 (N,K,3) —— 512 env × 7814 点也就 16MB, 每控制步一次。
        """
        import trimesh
        out = {}
        for side in SIDES:
            e = clips.clip_entry(TC.CLIP)
            mp = e["mesh"] if OI[side] == 0 else e["secondary"]["mesh"]
            V = np.asarray(trimesh.load(mp, process=False).vertices, np.float64)
            seg = V[V[:, 2] > V[:, 2].min() + 0.101]            # 头/斗段 (柄长 10.1cm, 见 L2 构建器注释)
            H = np.asarray(trimesh.convex.convex_hull(seg).vertices, np.float64)
            out[side] = self.to(H)
            print(f"[Sweep408Grip] {side} 贴桌凸包 {len(H)} 点 (目标间隙 {TC.TABLE_TARGET[side]*1000:+.0f}mm)", flush=True)
        return out

    def _table_gap(self, side):
        """(N,) 头/斗段最低点离桌面的高度 (m); 负 = 压进桌面。"""
        p, q = self._obj_pose(side)
        R = _quat_to_mat(q)                                      # (N,3,3)
        z = (R[:, 2, :] @ self.hull[side].T).amin(dim=1)         # 只要 z 分量 -> (N,K)
        return z + p[:, 2] - float(self.cfg.table_top_z)

    # ------------------------------------------------------------ RL 面 ----
    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if self._tick is not None:
            self._book(env_ids)
        DirectRLEnv._reset_idx(self, env_ids)
        n = len(env_ids)
        qfull = self.hand.data.default_joint_pos[env_ids].clone()
        qfull[:, self.map_ids_t] = self.ref_arm[0]
        for s in SIDES:
            qfull[:, self.fid[s]] = self.f_grasp[s]
        self.hand.write_joint_state_to_sim(qfull, torch.zeros_like(qfull), env_ids=env_ids)
        qcmd = qfull.clone(); qcmd[:, self.map_ids_t] += self.arm_sag
        self.hand.set_joint_position_target(qcmd, env_ids=env_ids)
        self.q_cmd[env_ids] = qcmd
        self._pin(env_ids)
        self.cum_res[env_ids] = 0; self.last_act[env_ids] = 0
        self.latched[env_ids] = False; self.cert[env_ids] = False; self.cert_run[env_ids] = 0
        self.k[env_ids] = 0; self.died[env_ids] = False; self.die_kind[env_ids] = 0
        self.succ[env_ids] = False; self.rel_max[env_ids] = 0
        lo = max(int(TC.RELEASE_MIN), self.release_row_cur - int(TC.RELEASE_JITTER))
        self.release_row[env_ids] = torch.randint(lo, self.release_row_cur + 1, (n,), device=self.device)
        for kk in self._ep:
            self._ep[kk][env_ids] = 0

    def _pre_physics_step(self, actions):
        self.last_act = actions.clamp(-1.0, 1.0)
        self.cum_res = torch.maximum(torch.minimum(self.cum_res + self.last_act * self.step_sz, self.dev_hi), -self.dev_hi)
        row = self.episode_length_buf
        # 臂前馈 = 母带第 k 行 (认证前 k=0, 即定在第 0 行等认证)
        arm = self.ref_arm[self.k.clamp(max=self.T_REF - 1)] + self.arm_sag + self.cum_res[:, :14]
        fin = self._ff_fingers(row) + self.cum_res[:, 14:]
        self.q_cmd[:, self.act_ids] = torch.cat([arm, fin], 1)
        self._pinned_ids = torch.nonzero(row < self.release_row).flatten()

    def _apply_action(self):
        self.hand.set_joint_position_target(self.q_cmd)
        self._pin(self._pinned_ids)             # 钉住期每物理子步写回 (碰撞全开, 指力照常建立)

    def _get_dones(self):
        N, dev = self.num_envs, self.device
        row = self.episode_length_buf                       # 已 +1
        released = row > self.release_row

        # ---- 掌系相对位姿 + 放手锁存 ----
        newrel = released & ~self.latched
        rel, dp, dr = {}, {}, {}
        for s in SIDES:
            rp, rq = self._rel(s)
            rel[s] = (rp, rq)
            if newrel.any():
                self.rel_p0[s][newrel] = rp[newrel]
                self.rel_q0[s][newrel] = rq[newrel]
        self.latched |= newrel
        for i, s in enumerate(SIDES):
            rp, rq = rel[s]
            dp[s] = torch.linalg.vector_norm(rp - self.rel_p0[s], dim=1) * self.latched.float()
            dr[s] = _qangle(rq, self.rel_q0[s]) * self.latched.float()
            self.rel_max[:, i] = torch.maximum(self.rel_max[:, i], dp[s])
            self.rel_max[:, 2 + i] = torch.maximum(self.rel_max[:, 2 + i], dr[s])

        # ---- 接触 ----
        F = {s: self._pad_force_mat(s) for s in SIDES}
        on = {s: F[s] > TC.PAD_FTH for s in SIDES}
        broom_ok = on["right"][:, 0] & (on["right"][:, 1:].sum(1) >= TC.BROOM_SUPPORT_MIN)
        pan_ok = on["left"].sum(1) >= TC.PAN_PADS_MIN
        c_broom = on["right"][:, 0].float() + on["right"][:, 1:].sum(1).clamp(max=3).float() / 3.0   # [0,2]
        c_pan = on["left"].sum(1).float() / 5.0                                                      # [0,1]

        # ---- 认证 (掌系, 相对锁存) ----
        within = torch.stack([(dp[s] < TC.CERT_POS) & (dr[s] < np.radians(TC.CERT_ROT_DEG))
                              for s in SIDES], 1).all(1)
        # v2: 认证**只看位姿** —— 接触条件降为诊断 (见 task_config 注释; v1 就死在这里)
        cert_now = self.latched & within
        self.cert_run = torch.where(cert_now, self.cert_run + 1, torch.zeros_like(self.cert_run))
        new_cert = (self.cert_run >= TC.CERT_STEPS) & ~self.cert
        self.cert |= new_cert
        cert_timeout = self.latched & ~self.cert & ((row - self.release_row) > TC.CERT_BUDGET)

        # ---- 时钟: 认证后每步走一行母带 ----
        can = self.cert & ~self.died & (self.k < self.T_REF - 1)
        self.k = torch.where(can, self.k + 1, self.k)
        clock_done = self.cert & (self.k >= self.T_REF - 1)
        # Success Checker (用户自定义): 时钟 ≥ SUCCESS_CLOCK_FRAC 母带 ∧ 当下仍在认证窗内。
        # v1 的"走完全部 455 行"全有全无、早期恒 0, 判读上没有信息; 这里按 L5-27"参考自己做得到"取 80%。
        _need = int(round((self.T_REF - 1) * TC.SUCCESS_CLOCK_FRAC))
        new_succ = self.cert & (self.k >= _need) & within & ~self.died & ~self.succ
        self.succ |= new_succ

        # ---- 死线 ----
        die_rel = self.latched & ((torch.stack([dp[s] for s in SIDES], 1) > TC.DIE_POS).any(1)
                                  | (torch.stack([dr[s] for s in SIDES], 1) > np.radians(TC.DIE_ROT_DEG)).any(1))
        z_low = torch.stack([self._obj_pose(s)[0][:, 2] < self.cfg.table_top_z - TC.OBJ_DROP_Z
                             for s in SIDES], 1).any(1)
        die_drop = released & z_low
        die_table = (self.hand.data.body_pos_w[:, self._low_ids, 2]
                     < self.cfg.table_top_z - TC.TABLE_MARGIN).any(1)
        new_die = torch.zeros(N, dtype=torch.bool, device=dev)
        for i, dk in enumerate((die_rel, die_drop, die_table)):
            nd = dk & ~self.died & ~new_die
            self.die_kind[nd] = i + 1; new_die |= nd
        self.died |= new_die

        # ---- 奖励 (v2 四条阶梯; 铁则: adv 1.0/步 > 任何一项指引) ----
        r_adv = TC.R_ADV * can.float()
        r_bonus = TC.B_CERT * new_cert.float() + TC.B_SUCCESS * new_succ.float() + TC.B_DIE * new_die.float()
        r_act = -TC.W_ACT * (self.last_act ** 2).mean(1)
        alive = (self.cert & ~self.died).float()          # 三项指引都只在**认证后**生效
        # L1+: 相对位姿软罚 (认证后), 1cm/5° 起渐进到 3cm/20° 满值 −SOFT_W
        r_soft = torch.zeros(N, device=dev)
        if TC.USE_SOFT:
            terms = []
            for s_ in SIDES:
                tp = (dp[s_] - TC.CERT_POS) / max(TC.SOFT_SPAN_POS - TC.CERT_POS, 1e-6)
                tr = (dr[s_] - np.radians(TC.CERT_ROT_DEG)) / max(
                    np.radians(TC.SOFT_SPAN_ROT_DEG - TC.CERT_ROT_DEG), 1e-6)
                terms.append(torch.maximum(tp, tr).clamp(0.0, 1.0))
            r_soft = -TC.SOFT_W * torch.stack(terms, 1).mean(1) * alive
        # L2+: 贴桌奖 (认证后), 以目标间隙为中心的高斯 (Sweep2 clear_q 同款)
        r_table = torch.zeros(N, device=dev)
        gaps = {}
        if TC.USE_TABLE:
            qs = []
            for s_ in SIDES:
                g = self._table_gap(s_)
                gaps[s_] = g
                qs.append(torch.exp(-(((g - TC.TABLE_TARGET[s_]) / TC.TABLE_SIGMA) ** 2)))
            r_table = TC.W_TABLE * torch.stack(qs, 1).mean(1) * alive
        # L3: 指垫接触奖
        r_contact = TC.W_CONTACT * (c_broom + c_pan) if TC.USE_CONTACT else torch.zeros(N, device=dev)
        reward = r_adv + r_soft + r_table + r_contact + r_bonus + r_act

        timeout = row >= self.T_EP
        terminated = self.died.clone()
        truncated = (clock_done | cert_timeout | timeout) & ~terminated
        for kk, v in (("r_adv", r_adv), ("r_soft", r_soft), ("r_table", r_table), ("r_contact", r_contact),
                      ("r_bonus", r_bonus), ("r_act", r_act), ("c_broom", c_broom), ("c_pan", c_pan),
                      ("cert_timeout", cert_timeout.float())):
            self._ep[kk] += v
        if gaps:
            self._ep["gap_broom_mm"] += gaps["right"] * 1000
            self._ep["gap_pan_mm"] += gaps["left"] * 1000
        # 逐步账目快照 (record_grip.py 落盘用; 只读诊断, 不参与任何计算)
        self._tick = dict(reward=reward, dp=dp, dr=dr, F=F, within=within, released=released,
                          broom_ok=broom_ok, pan_ok=pan_ok, terminated=terminated, truncated=truncated,
                          r_adv=r_adv, r_soft=r_soft, r_table=r_table, r_contact=r_contact,
                          r_bonus=r_bonus, r_act=r_act,
                          c_broom=c_broom, c_pan=c_pan, latched=self.latched.clone(),
                          cert=self.cert.clone(), cert_run=self.cert_run.clone(), new_cert=new_cert,
                          k=self.k.clone(), can=can, clock_done=clock_done, new_succ=new_succ,
                          died=self.died.clone(), die_kind=self.die_kind.clone(), new_die=new_die,
                          cert_timeout=cert_timeout, row=row.clone(), gaps=gaps,
                          release_row=self.release_row.clone())
        return terminated, truncated

    def _get_rewards(self):
        return self._tick["reward"]

    def _get_observations(self):
        row = self.episode_length_buf
        q, qd = self.hand.data.joint_pos, self.hand.data.joint_vel
        fin_ids = self.act_ids[14:]
        blocks = [q[:, self.map_ids_t], qd[:, self.map_ids_t] * 0.1,
                  q[:, fin_ids], qd[:, fin_ids] * 0.1, self.cum_res / self.dev_hi]
        devs, forces = [], []
        for s in SIDES:
            hp, hq = self._hand_pose(s); op, oq = self._obj_pose(s)
            rp, rq = self._rel(s)
            dq = quat_mul(quat_conjugate(self.rel_q0[s]), rq)
            dvec = torch.cat([rp - self.rel_p0[s], 2.0 * dq[:, 1:] * torch.sign(dq[:, :1])], 1) \
                * self.latched.float().unsqueeze(1)
            blocks += [hp, hq, op, oq, rp, rq, dvec,
                       self.art[s].data.root_lin_vel_w, self.art[s].data.root_ang_vel_w * 0.1]
            devs.append(dvec)
            forces.append((self._pad_force_mat(s) / 10.0).clamp(max=3.0))
        F = torch.cat(forces, 1)
        blocks += [F, (F > TC.PAD_FTH / 10.0).float()]
        pinned = (row < self.release_row).float().unsqueeze(1)
        t_to = ((self.release_row - row).float() / 60.0).clamp(-2, 2).unsqueeze(1)
        t_since = ((row - self.release_row).float() / 60.0).clamp(-2, 2).unsqueeze(1)
        blocks += [pinned, t_to, t_since, self.cert.float().unsqueeze(1),
                   (self.cert_run.float() / TC.CERT_STEPS).clamp(max=1).unsqueeze(1),
                   (self.k.float() / max(self.T_REF - 1, 1)).unsqueeze(1),
                   self._ff_fingers(row), self.last_act]
        obs = torch.cat(blocks, 1)
        assert obs.shape[1] == OBS_DIM, (obs.shape, OBS_DIM)
        priv = torch.cat(devs + [F], 1)
        assert priv.shape[1] == PRIV_DIM, priv.shape
        return {"policy": obs.float().clamp(-10, 10).nan_to_num(0.0),
                "priv_info": priv.float().clamp(-10, 10).nan_to_num(0.0)}

    # ------------------------------------------------------------ 台账 ----
    def _book(self, env_ids):
        t = self._tick
        rows = self.episode_length_buf[env_ids].float().clamp(min=1)
        add = {
            "sr/success": self.succ[env_ids].float(),
            "sr/cert": self.cert[env_ids].float(),
            "sr/clock_frac": (self.k[env_ids].float() / max(self.T_REF - 1, 1)),
            "term/die": self.died[env_ids].float(),
            "term/die_rel": (self.die_kind[env_ids] == 1).float(),
            "term/die_drop": (self.die_kind[env_ids] == 2).float(),
            "term/die_table": (self.die_kind[env_ids] == 3).float(),
            "term/cert_timeout": (self._ep["cert_timeout"][env_ids] > 0).float(),
            "hold/relp_max_broom_cm": self.rel_max[env_ids, 0] * 100,
            "hold/relp_max_pan_cm": self.rel_max[env_ids, 1] * 100,
            "hold/relrot_max_broom_deg": torch.rad2deg(self.rel_max[env_ids, 2]),
            "hold/relrot_max_pan_deg": torch.rad2deg(self.rel_max[env_ids, 3]),
            "hold/within_end": t["within"][env_ids].float(),
            "contact/broom_score": self._ep["c_broom"][env_ids] / rows,
            "contact/pan_score": self._ep["c_pan"][env_ids] / rows,
            "ep_rew/adv": self._ep["r_adv"][env_ids], "ep_rew/soft": self._ep["r_soft"][env_ids],
            "ep_rew/table": self._ep["r_table"][env_ids], "ep_rew/contact": self._ep["r_contact"][env_ids],
            "ep_rew/bonus": self._ep["r_bonus"][env_ids], "ep_rew/action": self._ep["r_act"][env_ids],
            "hold/gap_broom_mm": self._ep["gap_broom_mm"][env_ids] / rows,
            "hold/gap_pan_mm": self._ep["gap_pan_mm"][env_ids] / rows,
            "curr/release_row": self.release_row[env_ids].float(),
            "ep/len": self.episode_length_buf[env_ids].float(),
        }
        for kk, v in add.items():
            s, n = self._acc.get(kk, (0.0, 0))
            self._acc[kk] = (s + float(v.sum()), n + int(v.numel()))

    def pop_rates(self):
        out = {kk: (s / n if n else float("nan")) for kk, (s, n) in self._acc.items()}
        out["curr/release_row_cur"] = float(self.release_row_cur)
        self._acc = {}
        return out
