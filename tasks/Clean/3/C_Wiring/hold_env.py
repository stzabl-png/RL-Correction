"""Clean/3 A0 握持探针环境 (台账 §5.2 A0): 双手持物起步 + **真摩擦握**, 无策略。

只做三件事, 全部直接驱动物理 (不走 env.step):
  ① 手内复位 _settle_inhand: 臂 = 母带 0 行 IK; 物体按 T_obj = T_hand × inv(T_obj_hand) 放进手里
     (GraspPose prior, 输入系); 指 grasp_qpos → squeeze_qpos 斜坡合拢; 审计 手-物相对位姿 vs prior。
  ② 静持 hold: 目标不变, 持续 hold_s, 记录漂移。
  ③ 放音 replay: 逐行送母带臂目标 (指 = squeeze 常量), 记录 漂移 / 盘倾角 / 擦盘面贴合 / 覆盖率 / 行程。
物理 (用户裁定): 盘 0.3kg (POUR_OBJ_MASS, 主体物) / 海绵 0.05kg (aux USD 烘焙 semantics) / μ1 (POUR_OBJ_FRIC,
POUR_PAD_FRIC; 右垫经 extra_supergrip_bodies 补绑 SuperGrip)。入口脚本必须在 import 本模块前设这些环境变量。
角色: env.object = 盘 (左手, 主体物, 左垫传感器) | env.aux = 海绵 (右手)。
"""
from __future__ import annotations

import os

import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from rl_rebuild.correction import clips
from rl_rebuild.correction.kinematics import ArmIK
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior
from tasks.pregrasp.env import GraspTaskEnv

_HERE = os.path.dirname(os.path.abspath(__file__))
_TASK = os.path.abspath(os.path.join(_HERE, ".."))
_REPO = os.path.abspath(os.path.join(_TASK, "..", "..", ".."))
REFERENCE = os.environ.get("CLEAN_REF_NPZ") or os.path.join(
    _TASK, "A_Design", "L2_Reference", "clean3_reference_v1.npz")
# clip/先验 与 task_config 同源可换 take (2026-09-10, take18 倒扣盘变体; 不传环境变量 = 原 take3 口径)。
# ⚠ 场景是在**本文件**建的 (hold_env 是 Stage-1/2 共同的场景来源), task_config 里同名三项只是给
#   world.json 指纹和外部脚本读的 —— 两处必须用同一组环境变量, 否则会出现"指纹写着 take18、
#   场景装的却是 take3"。
PRIOR_PLATE = os.environ.get("CLEAN_PRIOR_PLATE") or os.path.join(_REPO, "tasks/pregrasp/priors/Clean3_plate_left.npz")
PRIOR_SPONGE = os.environ.get("CLEAN_PRIOR_SPONGE") or os.path.join(_REPO, "tasks/pregrasp/priors/Clean3_sponge_right.npz")
CLIP = os.environ.get("CLEAN_CLIP") or "Clean3_plate"
_RPADS = [f"right_{f}_elastomer" for f in ("thumb", "index", "middle", "ring", "pinky")]
ACT_DIM, OBS_DIM = 14, 191
PLATE_RIM_R = 0.085
CONTACT_TOL = 0.005          # 擦盘面离盘面 < 5mm 计"擦到"


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


def build_cfg(num_envs=1, reference=REFERENCE, plate_hulls=128):
    assert os.path.isfile(reference), f"missing {reference}; run A_Design/L2_Reference/build_reference.py"
    z = np.load(reference, allow_pickle=True)
    safe = _safe_arm_reference(z)
    cfg = GraspTaskCfg()
    clips.configure_cfg(cfg, CLIP)                 # hand_side=left; 左垫 × /Object(盘)
    assert cfg.hand_side == "left", cfg.hand_side
    cfg.extra_supergrip_bodies = list(_RPADS)      # 右垫也要 SuperGrip (海绵在右手)
    # 右垫 × /Aux(海绵) 接触传感器 (Unscrew part4 同款; 基类内部形状只认前 5 个左垫, _setup_scene 里切片)
    from isaaclab.sensors import ContactSensorCfg
    cfg.contact_sensors = list(cfg.contact_sensors) + [
        ContactSensorCfg(prim_path=f"/World/envs/env_.*/Robot/{n}", history_length=1,
                         filter_prim_paths_expr=["/World/envs/env_.*/Aux"]) for n in _RPADS]
    # 语义: 物体从第 0 行就在手里, 基类不得重摆/退避 (task_sweep 三道门; 名字虽叫 attached, 门的内容与焊接无关)
    cfg.fixed_attached_tools = True
    cfg.approach_only = True
    apply_grasp_prior(cfg, PRIOR_PLATE, 0.0, approach=True)
    cfg.scene.num_envs = int(num_envs)
    cfg.obj_jitter_xy = 0.0
    cfg.action_space = ACT_DIM
    cfg.observation_space = OBS_DIM
    cfg.episode_length_s = float(len(z["right_q"]) / float(z["control_hz"]) + 2.0)
    cfg.clean_reference = os.path.abspath(reference)
    cfg.clean_plate_hulls = int(plate_hulls)
    jp = dict(cfg.robot_cfg.init_state.joint_pos)
    for side, P, key, prior in (("right", "R", "right_q", PRIOR_SPONGE),
                                ("left", "L", "left_q", PRIOR_PLATE)):
        for i, value in enumerate(safe[key][0], 1):
            jp[f"{P}_arm_j{i}"] = float(value)
        pr = np.load(prior)
        for name, value in zip(GENERIC_JOINT_ORDER, pr["grasp"][7:29]):
            jp[name.replace("right_", f"{side}_")] = float(value)
    cfg.robot_cfg.init_state.joint_pos = jp
    cfg.object_cfg.init_state.pos = tuple(float(v) for v in z["obj_pos_0"][0])
    cfg.object_cfg.init_state.rot = tuple(float(v) for v in z["obj_quat_0"][0])
    return cfg


class HoldProbeEnv(GraspTaskEnv):
    def __init__(self, cfg, **kwargs):
        self._z = np.load(cfg.clean_reference, allow_pickle=True)
        super().__init__(cfg, **kwargs)
        dev, N = self.device, self.num_envs
        to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
        self.to = to
        safe = _safe_arm_reference(self._z)
        self.ref_arm = torch.cat([to(safe["right_q"]), to(safe["left_q"])], 1)
        self.ref_pos = {i: to(self._z[f"obj_pos_{i}"]) for i in (0, 1)}
        self.ref_quat = {i: to(self._z[f"obj_quat_{i}"]) for i in (0, 1)}
        self.T = self.ref_arm.shape[0]
        self.map_ids = [self.hand.joint_names.index(f"{P}_arm_j{i}")
                        for P in ("R", "L") for i in range(1, 8)]
        self.map_ids_t = torch.tensor(self.map_ids, dtype=torch.long, device=dev)
        self.fid, self.f_grasp, self.f_squeeze, self.T_oh = {}, {}, {}, {}
        for side, prior in (("right", np.load(PRIOR_SPONGE)), ("left", np.load(PRIOR_PLATE))):
            self.fid[side] = [self.hand.joint_names.index(n.replace("right_", f"{side}_"))
                              for n in GENERIC_JOINT_ORDER]
            self.f_grasp[side] = to(prior["grasp"][7:29])
            self.f_squeeze[side] = to(prior["squeeze"][7:29])
            trim = np.asarray(getattr(cfg, f"clean_trim_{side}", (0.0, 0.0, 0.0)), np.float32)
            if f"T_oh_{side}" in self._z.files:        # 母带已按物理沉降位姿重锚 (v2): 摆物用同一相对位姿
                oh = np.asarray(self._z[f"T_oh_{side}"], np.float64)
                print(f"[HoldProbe] {side}: 用母带 T_oh 覆写 (沉降锚) pos={np.round(oh[:3],4).tolist()}")
            else:
                oh = np.asarray(prior["grasp"][:7], np.float64)
            self.T_oh[side] = (to(oh[:3] + trim), to(oh[3:7]))   # 腕微调 (物体输入系, m)
        bn = list(self.hand.body_names)
        self.hand_bid = {"right": bn.index("right_hand_C_MC"), "left": bn.index("left_hand_C_MC")}
        self.ac_bid = bn.index("arm_center")
        self.art = {"left": self.object, "right": self.aux}
        self.oi = {"left": 0, "right": 1}
        self.q_ci = to(np.load(PRIOR_PLATE)["canon_rot"]).expand(N, 4)
        self.prof_r = np.asarray(self._z["plate_top_profile_r"], np.float64)
        self.prof_y = np.asarray(self._z["plate_top_profile_y"], np.float64)
        self.face_off = float(self._z["sponge_face_offset"])
        self.dt = float(self.sim.get_physics_dt())
        self.n_sub = int(round(1.0 / (float(self._z["control_hz"]) * self.dt)))
        # 覆盖栅格: 盘规范系 1cm 格, 盘面圆 r<=8.5cm; 海绵足迹 = 规范系 7.5×13.3cm 矩形采样
        gx = np.arange(-0.09, 0.09, 0.01) + 0.005
        cx, cy = np.meshgrid(gx, gx, indexing="ij")
        self.disk = torch.tensor((cx**2 + cy**2) <= PLATE_RIM_R**2, device=dev)
        fx = np.arange(-0.03, 0.031, 0.01); fy = np.arange(-0.06, 0.061, 0.01)
        fxx, fyy = np.meshgrid(fx, fy, indexing="ij")
        self.foot = to(np.stack([fxx.ravel(), fyy.ravel(), np.zeros(fxx.size)], 1))   # (F,3)
        self.cover = torch.zeros(N, 18, 18, dtype=torch.bool, device=dev)
        self.q_hold = None
        print(f"[HoldProbe] rows={self.T} dt={self.dt:.5f} n_sub={self.n_sub} "
              f"prof_r={np.round(self.prof_r*100,1).tolist()} face_off={self.face_off*100:.2f}cm")

    # ------------------------------------------------------------ 场景 ----
    def _setup_scene(self):
        self.aux_init_pose_np = np.r_[np.asarray(self._z["obj_pos_1"])[0],
                                     np.asarray(self._z["obj_quat_1"])[0]]
        super()._setup_scene()
        self._all_sensors = list(self._contact_sensors)
        self._contact_sensors = self._all_sensors[:5]      # 基类内部形状 5 (左垫)
        self._set_plate_hulls(int(getattr(self.cfg, "clean_plate_hulls", 128)))
        if getattr(self.cfg, "clean_kinematic", False):
            self._disable_object_collisions()

    def _disable_object_collisions(self):
        """纯运动学目检: 两物体关碰撞 (钉住后物理子步会按穿插把物体踢开, 实测海绵被踢 0.9cm/9°)。"""
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        n = 0
        for ei in range(self.cfg.scene.num_envs):
            for name in ("Object", "Aux"):
                root = stage.GetPrimAtPath(f"/World/envs/env_{ei}/{name}")
                for prim in Usd.PrimRange(root):
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False); n += 1
        print(f"[HoldProbe] kinematic: 物体碰撞已关 ×{n} collider")

    def _set_plate_hulls(self, hulls):
        """盘是浅碟: 默认 VHACD 预算会把碟心桥平 (海绵悬空), 显式给 hull 预算 + shrink wrap。"""
        import omni.usd
        from pxr import PhysxSchema, Usd, UsdGeom
        stage = omni.usd.get_context().get_stage()
        n = 0
        for ei in range(self.cfg.scene.num_envs):
            root = stage.GetPrimAtPath(f"/World/envs/env_{ei}/Object")
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdGeom.Mesh):
                    cd = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                    cd.CreateMaxConvexHullsAttr(int(hulls))
                    cd.CreateShrinkWrapAttr(True)
                    cd.CreateHullVertexLimitAttr(64)
                    n += 1
        print(f"[HoldProbe] plate convex decomposition: maxConvexHulls={hulls} shrinkWrap ×{n} mesh")

    # ------------------------------------------------------------ 工具 ----
    def _org(self):
        return self.scene.env_origins

    def _phys(self, qfull, n):
        render = bool(getattr(self.cfg, "clean_render", False))     # GUI 目检: 每物理步渲染
        for _ in range(int(n)):
            self.hand.set_joint_position_target(qfull)
            self.hand.write_data_to_sim()
            self.sim.step(render=render)
            self.hand.update(self.dt); self.object.update(self.dt); self.aux.update(self.dt)
            for s_ in getattr(self, "_all_sensors", []):
                s_.update(self.dt)

    def pad_forces(self, side):
        """逐指垫接触力 (N, env0): 左垫对 /Object(盘), 右垫对 /Aux(海绵)."""
        sens = self._all_sensors[:5] if side == "left" else self._all_sensors[5:10]
        out = {}
        for f, s_ in zip(("thumb", "index", "middle", "ring", "pinky"), sens):
            fw = s_.data.net_forces_w
            out[f] = round(float(torch.linalg.vector_norm(fw[0].reshape(-1, 3), dim=1).max()), 2)
        return out

    def _hand_pose(self, side):
        b = self.hand_bid[side]
        return self.hand.data.body_pos_w[:, b] - self._org(), self.hand.data.body_quat_w[:, b]

    def _obj_pose(self, side):
        a = self.art[side]
        return a.data.root_pos_w - self._org(), a.data.root_quat_w

    def _place_world(self, side, row):
        """物体钉在母带 row 行的**世界**位姿 (前奏期: 手在接近, 物体原地不动)。"""
        N = self.num_envs; oi = self.oi[side]
        pose = torch.cat([self.ref_pos[oi][row].expand(N, 3) + self._org(), self.ref_quat[oi][row].expand(N, 4)], 1)
        ids = torch.arange(N, device=self.device)
        self.art[side].write_root_pose_to_sim(pose, env_ids=ids)
        self.art[side].write_root_velocity_to_sim(torch.zeros(N, 6, device=self.device), env_ids=ids)

    def _place_in_hand(self, side):
        """T_obj = T_hand × inv(T_obj_hand): 物体按实测手位姿摆进手里 (零速度)。"""
        N = self.num_envs
        hp, hq = self._hand_pose(side)
        p_oh, q_oh = self.T_oh[side]
        q_obj = quat_mul(hq, quat_conjugate(q_oh.expand(N, 4)))
        p_obj = hp - quat_apply(q_obj, p_oh.expand(N, 3))
        pose = torch.cat([p_obj + self._org(), q_obj], 1)
        ids = torch.arange(N, device=self.device)
        self.art[side].write_root_pose_to_sim(pose, env_ids=ids)
        self.art[side].write_root_velocity_to_sim(torch.zeros(N, 6, device=self.device), env_ids=ids)

    def _hand_in_obj_err(self, side):
        hp, hq = self._hand_pose(side); op, oq = self._obj_pose(side)
        p = quat_apply(quat_conjugate(oq), hp - op); q = quat_mul(quat_conjugate(oq), hq)
        p_oh, q_oh = self.T_oh[side]
        return torch.linalg.vector_norm(p - p_oh, dim=1), _qangle(q, q_oh.expand_as(q))

    def _plate_can(self):
        """盘规范系 (z-up) 的世界位姿。"""
        pp, pq = self._obj_pose("left")
        return pp, quat_mul(pq, quat_conjugate(self.q_ci))

    def _plate_tilt(self):
        _, qc = self._plate_can()
        up = quat_apply(qc, torch.tensor([0., 0., 1.], device=self.device).expand(self.num_envs, 3))
        return torch.rad2deg(torch.acos(up[:, 2].clamp(-1, 1)))

    def _top_at(self, r):
        rr = np.clip(r.detach().cpu().numpy(), self.prof_r[0], self.prof_r[-1])
        return torch.tensor(np.interp(rr, self.prof_r, self.prof_y), dtype=torch.float32, device=self.device)

    def _sponge_on_plate(self):
        """海绵中心在盘规范系的 xy / 半径 / 擦盘面离盘面高度 (m) / 足迹格."""
        pp, pqc = self._plate_can()
        sp, sq = self._obj_pose("right")
        sqc = quat_mul(sq, quat_conjugate(self.q_ci))
        p_sc = quat_apply(quat_conjugate(pqc), sp - pp)                 # 海绵中心 in 盘规范系
        q_rel = quat_mul(quat_conjugate(pqc), sqc)                       # 海绵规范系 in 盘规范系
        r = torch.linalg.vector_norm(p_sc[:, :2], dim=1)
        # 擦盘面最低点: 足迹点沿海绵 -z 偏 face_off
        F = self.foot.shape[0]
        pts = quat_apply(q_rel[:, None, :].expand(-1, F, -1).reshape(-1, 4),
                         (self.foot - torch.tensor([0., 0., self.face_off], device=self.device))[None].expand(self.num_envs, -1, -1).reshape(-1, 3))
        pts = pts.reshape(self.num_envs, F, 3) + p_sc[:, None, :]
        rp = torch.linalg.vector_norm(pts[:, :, :2], dim=2)
        gap = pts[:, :, 2] - self._top_at(rp.reshape(-1)).reshape(self.num_envs, F)   # 各足迹点离盘面
        gap_min = gap.amin(dim=1)
        touch = (gap < CONTACT_TOL) & (rp <= PLATE_RIM_R)                 # 逐足迹点是否擦到
        return p_sc, r, gap_min, touch, pts

    def _mark_cover(self, touch, pts):
        idx = torch.floor((pts[:, :, :2] + 0.09) / 0.01).long().clamp(0, 17)
        for e in range(self.num_envs):
            m = touch[e]
            if m.any():
                self.cover[e, idx[e, m, 0], idx[e, m, 1]] = True

    def coverage(self):
        return (self.cover & self.disk[None]).sum(dim=(1, 2)).float() / self.disk.sum().float()

    # -------------------------------------------------------- ① 手内复位 ----
    def pad_distances(self, side):
        """各指垫 (elastomer body 原点) 到该手物体网格表面的有符号距离 (cm, 负=在网格内); 用输入系网格."""
        import trimesh
        if not hasattr(self, "_tm"):
            self._tm = {}
            for s_, oid in (("left", "object_0"), ("right", "object_1")):
                m = trimesh.load(os.path.join(_REPO, "datasets/clean_tableware/3/objects", oid,
                                              "object_mesh_scaled_final.obj"), force="mesh")
                self._tm[s_] = m          # 5 个查询点, 全分辨率 proximity 足够快
        bn = list(self.hand.body_names)
        pads = [f"{side}_{f}_elastomer" for f in ("thumb", "index", "middle", "ring", "pinky")]
        ids = [bn.index(n) for n in pads]
        op, oq = self._obj_pose(side)
        pw = self.hand.data.body_pos_w[:, ids] - self._org()[:, None, :]
        rel = quat_apply(quat_conjugate(oq)[:, None, :].expand(-1, 5, -1).reshape(-1, 4),
                         (pw - op[:, None, :]).reshape(-1, 3)).reshape(self.num_envs, 5, 3)
        pts = rel[0].detach().cpu().numpy().astype(np.float64)
        try:
            d = trimesh.proximity.signed_distance(self._tm[side], pts)     # 正=内
            d = -d
        except Exception:
            d = trimesh.proximity.ProximityQuery(self._tm[side]).signed_distance(pts); d = -d
        return {f: round(float(v) * 100, 2) for f, v in zip(("thumb", "index", "middle", "ring", "pinky"), d)}

    def settle_inhand(self, arm_steps=8, rounds=3, close_steps=24, post_steps=24, beta=1.0,
                      beta_left=None, beta_right=None):
        N, dev = self.num_envs, self.device
        self.beta = float(beta)
        self.beta_side = {"left": float(beta if beta_left is None else beta_left),
                          "right": float(beta if beta_right is None else beta_right)}
        qfull = self.hand.data.default_joint_pos.clone()
        qfull[:, self.map_ids_t] = self.ref_arm[0]
        for side in ("left", "right"):
            qfull[:, self.fid[side]] = self.f_grasp[side]
        self.hand.write_joint_state_to_sim(qfull, torch.zeros_like(qfull))
        self.hand.set_joint_position_target(qfull)
        for _ in range(rounds):
            for side in ("left", "right"):
                self._place_in_hand(side)
            self._phys(qfull, arm_steps)
        ref0 = self.ref_arm[0].expand(N, -1)
        for _ in range(4):                       # Sweep2 同款: 抵消臂 PD 稳态下垂 (只改目标)
            arm_err = ref0 - self.hand.data.joint_pos[:, self.map_ids_t]
            qfull[:, self.map_ids_t] += arm_err.clamp(-0.02, 0.02)
            for side in ("left", "right"):
                self._place_in_hand(side)
            self._phys(qfull, 8)
        pre = {s: [float(v.max()) for v in self._hand_in_obj_err(s)] for s in ("left", "right")}
        pads_pre = {s: self.pad_distances(s) for s in ("left", "right")}
        print(f"[HoldProbe] 指垫离物面 (cm, 合拢前, grasp_qpos): {pads_pre}")
        # 合拢期间物体**钉在手相对位姿** (逐步重摆, 零速度): 等价于 Dexonomy Isaac 抬升里"物体先靠桌托着,
        # 指合拢后再抬"; 否则悬空物体在 0.1s 斜坡里先自由落体 5cm (v1 探针的假阴性根因)。
        for k in range(1, close_steps + 1):
            for side in ("left", "right"):
                a = k / close_steps * self.beta_side[side]      # β>1 = 沿 squeeze 方向外推 (Unscrew probe_beta 同款)
                qfull[:, self.fid[side]] = (1 - a) * self.f_grasp[side] + a * self.f_squeeze[side]
                self._place_in_hand(side)
            self._phys(qfull, 1)
        for _ in range(post_steps):               # 指力建立 (仍钉住)
            for side in ("left", "right"):
                self._place_in_hand(side)
            self._phys(qfull, 1)
        f_pinned = {s: self.pad_forces(s) for s in ("left", "right")}
        print(f"[HoldProbe] 指垫接触力 (N, 钉住状态, 松开前): {f_pinned}")
        self._phys(qfull, post_steps)               # 松开: 物体只靠指力
        pads_post = {s: self.pad_distances(s) for s in ("left", "right")}
        print(f"[HoldProbe] 指垫离物面 (cm, 合拢后, β={self.beta_side}): {pads_post}")
        forces = {s: self.pad_forces(s) for s in ("left", "right")}
        print(f"[HoldProbe] 指垫接触力 (N, 松开 {post_steps} 步后): {forces}")
        pads_q = {s: torch.rad2deg(self.hand.data.joint_pos[0, self.fid[s]] - qfull[0, self.fid[s]]).abs().max().item()
                  for s in ("left", "right")}
        print(f"[HoldProbe] 指关节 实测-目标 最大差 (deg, 被物体挡住的量): {pads_q}")
        self.q_hold = qfull.clone()
        post = {s: [float(v.max()) for v in self._hand_in_obj_err(s)] for s in ("left", "right")}
        # 锚点/手位姿核对
        org = self._org()
        ac_p = (self.hand.data.body_pos_w[0, self.ac_bid] - org[0]).cpu().numpy()
        anchor = np.asarray(self._z["anchor_T"], np.float64)
        hand_delta = {}
        for side in ("left", "right"):
            hp, hq = self._hand_pose(side)
            oi = self.oi[side]
            p_oh, q_oh = self.T_oh[side]
            tgt_p = self.ref_pos[oi][0] + quat_apply(self.ref_quat[oi][0].expand(N, 4), p_oh.expand(N, 3))
            tgt_q = quat_mul(self.ref_quat[oi][0].expand(N, 4), q_oh.expand(N, 4))
            hand_delta[side] = [float(torch.linalg.vector_norm(hp - tgt_p, dim=1).max()),
                                float(torch.rad2deg(_qangle(hq, tgt_q)).max())]
        arm_q_err = float(torch.rad2deg((self.hand.data.joint_pos[:, self.map_ids_t] - ref0).abs()).max())
        masses = {s: float(self.art[s].root_physx_view.get_masses()[0].sum()) for s in ("left", "right")}
        mats = {s: self.art[s].root_physx_view.get_material_properties()[0][0][:2].tolist() for s in ("left", "right")}
        audit = dict(hand_in_obj_err_pre_close=pre, hand_in_obj_err_post_close=post,
                     pads_pre_cm=pads_pre, pads_post_cm=pads_post, pad_forces_pinned_N=f_pinned, pad_forces_N=forces,
                     finger_block_deg=pads_q, beta=self.beta, beta_side=self.beta_side,
                     hand_vs_ref_target=hand_delta, arm_q_err_deg=arm_q_err,
                     anchor_pos_delta_cm=(100 * (ac_p - anchor[:3, 3])).round(3).tolist(),
                     masses_kg=masses, obj_friction=mats,
                     plate_tilt_deg=float(self._plate_tilt().max()))
        print(f"[HoldProbe] settle: {audit}")
        return audit

    # -------------------------------------------------------- ② 静持 ----
    def hold(self, hold_s=3.0, every=24):
        n = int(round(hold_s / self.dt)); log = []
        for k in range(0, n, every):
            self._phys(self.q_hold, min(every, n - k))
            row = {"t": (k + every) * self.dt}
            for s in ("left", "right"):
                pe, re = self._hand_in_obj_err(s)
                row[f"{s}_pos_cm"] = float(pe.max() * 100); row[f"{s}_rot_deg"] = float(torch.rad2deg(re).max())
            row["plate_tilt_deg"] = float(self._plate_tilt().max())
            _, _, gap, _, _ = self._sponge_on_plate(); row["sponge_gap_mm"] = float(gap.mean() * 1000)
            row["forces"] = {s: self.pad_forces(s) for s in ("left", "right")}
            log.append(row)
        summ = {k: max(r[k] for r in log) for k in log[0] if k not in ("t", "forces")}
        summ["forces_end"] = log[-1]["forces"]
        summ["settled_oh"] = self.settled_oh()
        summ["dropped"] = self._dropped()
        print(f"[HoldProbe] hold {hold_s:.1f}s: max {summ}")
        return dict(summary=summ, log=log)

    def settled_oh(self):
        """沉降后 手-物相对位姿 (物体输入系: hand pos/quat wxyz), env 平均位置 + env0 四元数."""
        out = {}
        for s_ in ("left", "right"):
            hp, hq = self._hand_pose(s_); op, oq = self._obj_pose(s_)
            p = quat_apply(quat_conjugate(oq), hp - op); q = quat_mul(quat_conjugate(oq), hq)
            out[s_] = dict(pos=p.mean(dim=0).cpu().numpy().round(5).tolist(),
                           quat_wxyz=q[0].cpu().numpy().round(6).tolist(),
                           pos_spread_cm=float((p - p.mean(dim=0)).norm(dim=1).max() * 100))
        return out

    def _dropped(self):
        out = {}
        for s in ("left", "right"):
            p, _ = self._obj_pose(s)
            pe, _ = self._hand_in_obj_err(s)
            out[s] = int(((p[:, 2] < self.cfg.table_top_z + 0.03) | (pe > 0.05)).sum())
        return out

    # -------------------------------------------------------- ③ 放音 ----
    def replay(self, rows=None):
        rows = self.T if rows is None or rows <= 0 else min(int(rows), self.T)
        N = self.num_envs
        qfull = self.q_hold.clone()
        sag = qfull[:, self.map_ids_t] - self.ref_arm[0].expand(N, -1)     # 复位时的稳态修正, 全程沿用
        self.cover.zero_()
        rec = {k: np.zeros((rows, N), np.float32) for k in
               ("left_pos_cm", "left_rot_deg", "right_pos_cm", "right_rot_deg", "plate_tilt_deg",
                "sponge_gap_mm", "sponge_r_cm", "contact", "plate_z", "sponge_z")}
        travel = torch.zeros(N, device=self.device); prev_xy = None
        first_drop = {"left": None, "right": None}
        for r in range(rows):
            qfull[:, self.map_ids_t] = self.ref_arm[r].expand(N, -1) + sag
            self._phys(qfull, self.n_sub)
            for s in ("left", "right"):
                pe, re = self._hand_in_obj_err(s)
                rec[f"{s}_pos_cm"][r] = (pe * 100).cpu().numpy(); rec[f"{s}_rot_deg"][r] = torch.rad2deg(re).cpu().numpy()
                if first_drop[s] is None and self._dropped()[s] > 0:
                    first_drop[s] = r
            rec["plate_tilt_deg"][r] = self._plate_tilt().cpu().numpy()
            p_sc, rr, gap, touch, pts = self._sponge_on_plate()
            rec["sponge_gap_mm"][r] = (gap * 1000).cpu().numpy(); rec["sponge_r_cm"][r] = (rr * 100).cpu().numpy()
            contact = touch.any(dim=1)
            rec["contact"][r] = contact.float().cpu().numpy()
            rec["plate_z"][r] = self._obj_pose("left")[0][:, 2].cpu().numpy()
            rec["sponge_z"][r] = self._obj_pose("right")[0][:, 2].cpu().numpy()
            self._mark_cover(touch, pts)
            if prev_xy is not None:
                travel += torch.linalg.vector_norm(p_sc[:, :2] - prev_xy, dim=1) * contact.float()
            prev_xy = p_sc[:, :2].clone()
            if r % 100 == 0 or r == rows - 1:
                print(f"[HoldProbe] row {r}/{rows}: L {rec['left_pos_cm'][r].max():.2f}cm/{rec['left_rot_deg'][r].max():.1f}° "
                      f"R {rec['right_pos_cm'][r].max():.2f}cm/{rec['right_rot_deg'][r].max():.1f}° "
                      f"tilt {rec['plate_tilt_deg'][r].max():.1f}° gap {rec['sponge_gap_mm'][r].mean():.1f}mm "
                      f"r {rec['sponge_r_cm'][r].mean():.1f}cm contact {rec['contact'][r].mean():.2f}")
        cov = self.coverage().cpu().numpy()
        summ = dict(rows=rows, first_drop_row=first_drop,
                    left_pos_cm_max=float(rec["left_pos_cm"].max()), left_rot_deg_max=float(rec["left_rot_deg"].max()),
                    right_pos_cm_max=float(rec["right_pos_cm"].max()), right_rot_deg_max=float(rec["right_rot_deg"].max()),
                    plate_tilt_deg_max=float(rec["plate_tilt_deg"].max()),
                    plate_tilt_deg_p95=float(np.percentile(rec["plate_tilt_deg"], 95)),
                    contact_row_frac=float(rec["contact"].mean()),
                    coverage_frac=cov.tolist(), travel_cm=(travel * 100).cpu().numpy().tolist(),
                    sponge_gap_mm_p50=float(np.percentile(rec["sponge_gap_mm"], 50)),
                    dropped=self._dropped())
        print(f"[HoldProbe] replay summary: {summ}")
        return dict(summary=summ, rec=rec)

    def set_object_gravity(self, on: bool):
        # PhysX tensors 这个接口只收 CPU 张量 (expected device -1)
        ids = torch.arange(self.num_envs, dtype=torch.int32)
        for art in (self.object, self.aux):
            flag = torch.full((self.num_envs,), (not on), dtype=torch.bool)
            art.root_physx_view.set_disable_gravities(flag, ids)
        print(f"[HoldProbe] 物体重力 {'开' if on else '关'}")

    def kinematic_place0(self):
        """把机器人写到母带 0 行 (指=grasp_qpos), 两物体钉到 GraspPose 相对位姿, 不步进物理 (暂停观察用)."""
        N = self.num_envs
        qfull = self.hand.data.default_joint_pos.clone()
        qfull[:, self.map_ids_t] = self.ref_arm[0].expand(N, -1)
        for side in ("left", "right"):
            qfull[:, self.fid[side]] = self.f_grasp[side]
        self.hand.write_joint_state_to_sim(qfull, torch.zeros_like(qfull))
        self.hand.set_joint_position_target(qfull)
        self.hand.write_data_to_sim()
        self.hand.update(self.dt)
        for side in ("left", "right"):
            self._place_in_hand(side)
        self.object.update(self.dt); self.aux.update(self.dt)

    # -------------------------------------------------------- 运动学回放 (不开物理) ----
    def _human_arm_rows(self):
        """人手重定向轨迹 (ref_qpos_*.npz) -> 逐行臂 q + 指 q (20Hz).
        腕: 第 0 行焊接在母带 0 行手位姿上, 只借人手腕的增量 (Sweep2/Pour 首帧焊接口径); 指: 人手 22 维指流按名映射。"""
        from rl_rebuild.correction.kinematics import ArmIK, quat_to_R
        N = self.num_envs
        hz = float(self._z["control_hz"]); out_q, out_f, stats = {}, {}, {}
        for side in ("left", "right"):
            zh = np.load(os.path.join(_REPO, "datasets/clean_tableware/3", f"ref_qpos_{side}.npz"), allow_pickle=True)
            fps = float(zh["fps"]); nf = len(zh["wrist_pos"])
            times = np.arange(0.0, (nf - 1) / fps + 1e-9, 1.0 / hz) * fps          # 源帧下标 (连续)
            i0 = np.clip(np.floor(times).astype(int), 0, nf - 2); a = (times - i0)[:, None]
            wp = (1 - a) * zh["wrist_pos"][i0] + a * zh["wrist_pos"][i0 + 1]
            wq = (1 - a) * zh["wrist_quat_wxyz"][i0] + a * zh["wrist_quat_wxyz"][i0 + 1]
            wq /= np.linalg.norm(wq, axis=1, keepdims=True)
            fq = (1 - a) * zh["finger_qpos"][i0] + a * zh["finger_qpos"][i0 + 1]
            names = [str(n) for n in zh["joint_names"]]
            order = [names.index(n.replace("right_", f"{side}_")) for n in GENERIC_JOINT_ORDER]
            out_f[side] = fq[:, order].astype(np.float32)
            # 焊接: 母带 0 行手位姿 (FK) 为锚
            ik = ArmIK(side, anchor_link="arm_center", anchor_T=self._anchor_T)
            q0 = self.ref_arm[0, (0 if side == "right" else 7):(7 if side == "right" else 14)].cpu().numpy().astype(np.float64)
            p_anchor, R_anchor = ik.fk(q0)
            R_w0 = quat_to_R(wq[0]); qs, ok = [], 0; q_prev = q0
            for t in range(len(times)):
                p_t = p_anchor + (wp[t] - wp[0])
                R_t = (quat_to_R(wq[t]) @ R_w0.T) @ R_anchor
                r = ik.solve(p_t, R_t, q0=q_prev, iters=200, pos_tol=0.005, rot_tol=0.05)
                if r["ok"]:
                    q_prev = np.asarray(r["q"], np.float64); ok += 1
                qs.append(q_prev.copy())
            out_q[side] = np.asarray(qs, np.float32)
            stats[side] = dict(rows=len(times), ik_ok=ok / len(times),
                               wrist_range_cm=np.round((wp.max(0) - wp.min(0)) * 100, 2).tolist())
        print(f"[HoldProbe] 人手轨迹 -> 臂/指行: {stats}")
        T = min(len(out_q["left"]), len(out_q["right"]))
        arm = np.concatenate([out_q["right"][:T], out_q["left"][:T]], 1)
        return (torch.tensor(arm, device=self.device), {s_: torch.tensor(out_f[s_][:T], device=self.device) for s_ in out_f})

    def kinematic_replay(self, rows=None, hold_s=2.0, traj="ref"):
        """纯运动学目检: 每个物理子步直接写机器人关节状态 (臂=母带行, 指=GraspPose grasp_qpos),
        两物体按 T_obj = T_hand × inv(T_obj_hand) 逐步钉在手里 —— 手和物体始终保持 GraspPose 相对位姿, 无重力/接触效应。"""
        N = self.num_envs
        self.set_object_gravity(False)
        if traj == "human":
            arm_rows, fin_rows = self._human_arm_rows()
        else:
            arm_rows = self.ref_arm
            fin_rows = ({s_: self.to(self._z[f"{s_}_f"]) for s_ in ("left", "right")}
                        if "left_f" in self._z.files else None)      # 母带逐行指参考 (含前奏的接近/合拢)
            if fin_rows is not None:
                print(f"[HoldProbe] kinematic: 指按母带逐行参考 (prelude_rows={int(self._z['prelude_rows']) if 'prelude_rows' in self._z.files else 0})")
        T = arm_rows.shape[0]
        rows = T if rows is None or rows <= 0 else min(int(rows), T)
        qfull = self.hand.data.default_joint_pos.clone()
        for side in ("left", "right"):
            qfull[:, self.fid[side]] = self.f_grasp[side]
        zero = torch.zeros_like(qfull)

        pre = int(self._z["prelude_rows"]) if ("prelude_rows" in self._z.files and traj == "ref") else 0

        def _kin_step(r):
            qfull[:, self.map_ids_t] = arm_rows[r].expand(N, -1)
            if fin_rows is not None:                      # 逐行指参考 (母带前奏的接近/合拢, 或人手指流)
                for side in ("left", "right"):
                    qfull[:, self.fid[side]] = fin_rows[side][r].expand(N, -1)
            self.hand.write_joint_state_to_sim(qfull, zero)
            self.hand.set_joint_position_target(qfull)
            self.hand.write_data_to_sim()
            for side in ("left", "right"):
                if r < pre:
                    self._place_world(side, r)           # 前奏: 物体钉在原位, 手去接近
                else:
                    self._place_in_hand(side)            # 擦拭: 物体锁在手上跟手走
            self.sim.step(render=bool(getattr(self.cfg, "clean_render", False)))
            self.hand.update(self.dt); self.object.update(self.dt); self.aux.update(self.dt)

        for _ in range(int(round(hold_s / self.dt))):     # 第 0 行静止观察
            _kin_step(0)
        print(f"[HoldProbe] kinematic: rows={rows} prelude(world-pinned)={pre}", flush=True)
        lim = self.hand.data.joint_pos_limits[0, self.map_ids_t].cpu().numpy()
        print(f"[HoldProbe] Isaac 臂限位(deg) lower={np.round(np.degrees(lim[:,0]),0).tolist()} upper={np.round(np.degrees(lim[:,1]),0).tolist()}")
        for r_ in (0, pre, min(pre + 9, rows - 1)):
            _kin_step(r_)
            act = self.hand.data.joint_pos[0, self.map_ids_t].cpu().numpy(); cmd = arm_rows[r_].cpu().numpy()
            print(f"[HoldProbe] row {r_}: 实测-命令 |Δq| (deg) = {np.round(np.degrees(np.abs(act - cmd)), 2).tolist()}")
            for side in ("left", "right"):
                hp, hq = self._hand_pose(side); p_oh, q_oh = self.T_oh[side]; oi = self.oi[side]
                tq = quat_mul(self.ref_quat[oi][r_:r_+1], q_oh[None])
                print(f"[HoldProbe]   {side} hand rot vs target: {float(torch.rad2deg(_qangle(hq[:1], tq))[0]):.2f}°")
        for r in range(rows):
            for _ in range(self.n_sub):
                _kin_step(r)
            if r % 100 == 0 or r == rows - 1 or r in (pre, pre + 9):
                p_sc, rr, gap, touch, _ = self._sponge_on_plate()
                dev = {}
                for side in ("left", "right"):
                    op, oq = self._obj_pose(side); oi = self.oi[side]
                    dev[side] = (float(torch.linalg.vector_norm(op[0] - self.ref_pos[oi][r]) * 100),
                                 float(torch.rad2deg(_qangle(oq[:1], self.ref_quat[oi][r:r+1]))[0]))
                    hp, hq = self._hand_pose(side); p_oh, q_oh = self.T_oh[side]
                    tgt_p = self.ref_pos[oi][r] + quat_apply(self.ref_quat[oi][r:r+1], p_oh[None])[0]
                    dev[side + "_hand"] = float(torch.linalg.vector_norm(hp[0] - tgt_p) * 100)
                print(f"[HoldProbe] kinematic row {r}/{rows}: sponge r={float(rr.mean())*100:.1f}cm "
                      f"gap={float(gap.mean())*1000:.1f}mm touch_pts={int(touch[0].sum())}/91 "
                      f"plate_tilt={float(self._plate_tilt().max()):.1f}° | obj-vs-ref (cm,deg) "
                      f"L={dev['left'][0]:.2f}/{dev['left'][1]:.1f} R={dev['right'][0]:.2f}/{dev['right'][1]:.1f} "
                      f"| hand-vs-target cm L={dev['left_hand']:.2f} R={dev['right_hand']:.2f}", flush=True)

    # DirectRLEnv 抽象面: 探针不走 step/reset, 给最小占位
    def _reset_idx(self, env_ids):
        DirectRLEnv._reset_idx(self, env_ids)

    def _get_observations(self):
        return {"obs": torch.zeros(self.num_envs, OBS_DIM, device=self.device)}

    def _get_rewards(self):
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self):
        z = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return z, z

    def _pre_physics_step(self, actions):
        pass

    def _apply_action(self):
        if self.q_hold is not None:
            self.hand.set_joint_position_target(self.q_hold)
