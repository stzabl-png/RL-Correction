"""Sweep2 bimanual 14-D residual environment.

The robot begins at reference row zero with broom and dustpan attached by physical
USD FixedJoints.  Fingers never receive policy actions.  The reconstructed tool
trajectory is the feed-forward reference; PPO only controls bounded arm residuals.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

from rl_rebuild.correction import clips
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior
from tasks.pregrasp.env import GraspTaskEnv

_HERE = os.path.dirname(os.path.abspath(__file__))
_TASK = os.path.abspath(os.path.join(_HERE, ".."))
_L3 = os.path.join(_TASK, "A_Design", "L3_Learning")
sys.path.insert(0, _L3)
from progress_batch import SweepGeometry, SweepProgressBatch, sweep_signals  # noqa: E402

REFERENCE = os.environ.get("SWEEP_REF_NPZ") or os.path.join(
    _TASK, "A_Design", "L2_Reference", "sweep2_reference_v1.npz")
ACT_DIM = 14
OBS_DIM = 176
PRIOR_BROOM = "tasks/pregrasp/priors/Sweep2_broom.npz"
PRIOR_PAN = "tasks/pregrasp/priors/Sweep2_dustpan.npz"


def _qangle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = (a * b).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d)


def _pan_cube_start(pan_pos, pan_quat, table_z, geometry=SweepGeometry()):
    """Fixed easy start: centred, 3 cm outside the pan lip, resting on table."""
    local = torch.tensor([0.0, 0.0, geometry.pan_mouth_z + geometry.start_outside],
                         dtype=pan_pos.dtype, device=pan_pos.device).expand_as(pan_pos)
    p = pan_pos + quat_apply(pan_quat, local)
    p[:, 2] = float(table_z) + geometry.cube_half + 0.0005
    return p


def build_cfg(num_envs=1, reference=REFERENCE):
    assert os.path.isfile(reference), (
        f"missing {reference}; run A_Design/L2_Reference/build_reference.py first")
    z = np.load(reference, allow_pickle=True)
    cfg = GraspTaskCfg()
    clips.configure_cfg(cfg, "Sweep2_broom")
    cfg.approach_only = True
    apply_grasp_prior(cfg, PRIOR_BROOM, 30.0, approach=True)
    cfg.scene.num_envs = int(num_envs)
    cfg.obj_jitter_xy = 0.0
    cfg.action_space = ACT_DIM
    cfg.observation_space = OBS_DIM
    cfg.episode_length_s = 14.0
    cfg.sweep_reference = os.path.abspath(reference)
    cfg.sweep_pan_prior = os.path.abspath(PRIOR_PAN)
    # Physics starts with a constraint-consistent state.  Exact-name entries are
    # installed after generic defaults and therefore win regex resolution.
    jp = dict(cfg.robot_cfg.init_state.joint_pos)
    for side, P, key in (("right", "R", "right_q"), ("left", "L", "left_q")):
        for i, value in enumerate(np.asarray(z[key])[0], 1):
            jp[f"{P}_arm_j{i}"] = float(value)
        prior = np.load(PRIOR_BROOM if side == "right" else PRIOR_PAN)
        for name, value in zip(GENERIC_JOINT_ORDER, prior["grasp"][7:29]):
            jp[name.replace("right_", f"{side}_")] = float(value)
    cfg.robot_cfg.init_state.joint_pos = jp
    cfg.object_cfg.init_state.pos = tuple(float(v) for v in z["obj_pos_1"][0])
    cfg.object_cfg.init_state.rot = tuple(float(v) for v in z["obj_quat_1"][0])
    return cfg


class SweepEnv(GraspTaskEnv):
    def __init__(self, cfg, **kwargs):
        self._z = np.load(cfg.sweep_reference, allow_pickle=True)
        self._pan_prior_npz = np.load(cfg.sweep_pan_prior)
        super().__init__(cfg, **kwargs)
        dev, N = self.device, self.num_envs
        to = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
        self.ref_arm = torch.cat([to(self._z["right_q"]), to(self._z["left_q"])], 1)
        self.ref_pos = {i: to(self._z[f"obj_pos_{i}"]) for i in (0, 1)}
        self.ref_quat = {i: to(self._z[f"obj_quat_{i}"]) for i in (0, 1)}
        self.conf = torch.stack([to(self._z["confidence_1"]),
                                 to(self._z["confidence_0"])], 1) / 100.0
        self.T = self.ref_arm.shape[0]
        self.contact_row = int(self._z["contact_row"]) if "contact_row" in self._z else max(1, self.T // 4)
        self.cube_start_ref = to(self._z["cube_start_w"]) if "cube_start_w" in self._z else None
        self.row = torch.zeros(N, dtype=torch.long, device=dev)
        self.map_ids = [self.hand.joint_names.index(f"{P}_arm_j{i}")
                        for P in ("R", "L") for i in range(1, 8)]
        self.map_ids_t = torch.tensor(self.map_ids, dtype=torch.long, device=dev)
        self.fixed_finger_ids, self.fixed_finger_q = [], []
        for side, prior in (("right", np.load(PRIOR_BROOM)),
                            ("left", self._pan_prior_npz)):
            for name, value in zip(GENERIC_JOINT_ORDER, prior["grasp"][7:29]):
                self.fixed_finger_ids.append(self.hand.joint_names.index(
                    name.replace("right_", f"{side}_")))
                self.fixed_finger_q.append(float(value))
        self.fixed_finger_q = to(self.fixed_finger_q).expand(N, -1).clone()
        self.cum_res = torch.zeros(N, ACT_DIM, device=dev)
        self.last_act = torch.zeros_like(self.cum_res)
        # Right broom arm has twice the exploration envelope of the left pan arm.
        self.step_hi = to([0.020] * 7 + [0.010] * 7)
        self.dev_hi = to([0.25] * 7 + [0.12] * 7)
        self.step_lo = to([0.008] * 7 + [0.005] * 7)
        self.dev_lo = to([0.10] * 7 + [0.05] * 7)
        self.q_tgt = self.ref_arm[0].expand(N, -1).clone()
        self.geometry = SweepGeometry()
        self.progress = SweepProgressBatch(N, dev, self.geometry)
        self.cube_start = torch.zeros(N, 3, device=dev)
        self._tick_out = None
        self._stats = {f"gate{i}": 0 for i in range(1, 5)} | {"episodes": 0}

        bn = list(self.hand.body_names)
        self.hand_bid = {"right": bn.index("right_hand_C_MC"),
                         "left": bn.index("left_hand_C_MC")}
        # Brush-head surface proxy (broad +z half of mesh), deterministic sample.
        import trimesh
        mesh = trimesh.load(clips.clip_entry("Sweep2_broom")["mesh"], force="mesh")
        pts = np.asarray(mesh.vertices, np.float32)
        pts = pts[pts[:, 2] > 0.04]
        rng = np.random.RandomState(0)
        pts = pts[rng.choice(len(pts), min(192, len(pts)), replace=False)]
        self.broom_points = to(pts)

        self._validate_attachment_reset()

    def _setup_scene(self):
        # The aux helper normally uses scene_layout.  Override it with the exact
        # reference row-zero pan pose before it spawns the RigidObject.
        self.aux_init_pose_np = np.r_[np.asarray(self._z["obj_pos_0"])[0],
                                     np.asarray(self._z["obj_quat_0"])[0]]
        super()._setup_scene()
        cube_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/SweepCube",
            spawn=sim_utils.CuboidCfg(
                size=(0.010, 0.010, 0.010),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=0.0015, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.02,
                    friction_combine_mode="average", restitution_combine_mode="multiply"),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.08))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.table_top_z + 0.006)))
        self.cube = RigidObject(cube_cfg)
        self.scene.rigid_objects["sweep_cube"] = self.cube
        self._create_tool_joints()
        self.scene.filter_collisions()

    def _create_tool_joints(self):
        import omni.usd
        from pxr import Gf, Sdf, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        specs = (("right", "Object", np.load(PRIOR_BROOM)["grasp"]),
                 ("left", "Aux", self._pan_prior_npz["grasp"]))
        # Collect relative collider paths once. Traversing the full cloned stage for
        # every environment made the old 512-env setup needlessly quadratic.
        suffixes = {"right": [], "left": []}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            for side in suffixes:
                prefix = f"/World/envs/env_0/Robot/{side}_"
                if path.startswith(prefix) and prim.HasAPI(UsdPhysics.CollisionAPI):
                    suffixes[side].append(path[len("/World/envs/env_0"):])
        filters = 0
        for ei in range(self.cfg.scene.num_envs):
            env = f"/World/envs/env_{ei}"
            for side, tool, grasp in specs:
                hand = f"{env}/Robot/{side}_hand_C_MC"
                tool_path = f"{env}/{tool}"
                j = UsdPhysics.FixedJoint.Define(stage, f"{env}/Joints/{side}_tool_fixed")
                j.CreateBody0Rel().SetTargets([Sdf.Path(hand)])
                j.CreateBody1Rel().SetTargets([Sdf.Path(tool_path)])
                j.CreateLocalPos0Attr().Set(Gf.Vec3f(0, 0, 0))
                j.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
                j.CreateLocalPos1Attr().Set(Gf.Vec3f(*[float(v) for v in grasp[:3]]))
                q = grasp[3:7]
                j.CreateLocalRot1Attr().Set(Gf.Quatf(float(q[0]), float(q[1]),
                                                     float(q[2]), float(q[3])))
                rel = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(tool_path)) \
                    .CreateFilteredPairsRel()
                targets = [Sdf.Path(env + suffix) for suffix in suffixes[side]]
                if targets:
                    rel.AddTarget(Sdf.Path(hand))
                    for target in targets: rel.AddTarget(target)
                    filters += len(targets) + 1
        print(f"[SweepEnv] physical FixedJoints={2*self.cfg.scene.num_envs}; "
              f"own-hand collision filters={filters}")

    def _validate_attachment_reset(self):
        org = self.scene.env_origins
        checks = []
        for oi, art, side, prior in (
                (1, self.object, "right", np.load(PRIOR_BROOM)),
                (0, self.aux, "left", self._pan_prior_npz)):
            p = art.data.root_pos_w - org
            q = art.data.root_quat_w
            target_p = self.ref_pos[oi][0].expand_as(p)
            target_q = self.ref_quat[oi][0].expand_as(q)
            pose_err = torch.linalg.vector_norm(p - target_p, dim=1).max()
            rot_err = _qangle(q, target_q).max()
            hp = self.hand.data.body_pos_w[:, self.hand_bid[side]]
            hq = self.hand.data.body_quat_w[:, self.hand_bid[side]]
            gp = torch.tensor(prior["grasp"][:3], dtype=torch.float32,
                              device=self.device).expand(self.num_envs, 3)
            gq = torch.tensor(prior["grasp"][3:7], dtype=torch.float32,
                              device=self.device).expand(self.num_envs, 4)
            expected_hp = p + org + quat_apply(q, gp)
            expected_hq = quat_mul(q, gq)
            joint_pos = torch.linalg.vector_norm(hp - expected_hp, dim=1).max()
            joint_rot = _qangle(hq, expected_hq).max()
            checks.append((side, float(pose_err), float(rot_err),
                           float(joint_pos), float(joint_rot)))
        print(f"[SweepEnv] attachment reset audit={checks}")
        assert max(x[1] for x in checks) < 0.003, checks
        assert max(x[3] for x in checks) < 0.003, checks
        assert max(x[4] for x in checks) < np.radians(2.0), checks

    def _pre_physics_step(self, actions):
        self.last_act = actions.clamp(-1.0, 1.0)
        r = self.row.clamp(max=self.T - 1)
        confidence = self.conf[r]                     # right, left
        c14 = torch.cat([confidence[:, :1].expand(-1, 7),
                         confidence[:, 1:].expand(-1, 7)], 1)
        # High confidence narrows exploration; low confidence opens it.
        step = self.step_hi + c14 * (self.step_lo - self.step_hi)
        dev = self.dev_hi + c14 * (self.dev_lo - self.dev_hi)
        self.cum_res = torch.maximum(torch.minimum(
            self.cum_res + self.last_act * step, dev), -dev)
        self.q_tgt = self.ref_arm[r] + self.cum_res

    def _apply_action(self):
        self.hand.set_joint_position_target(self.q_tgt, joint_ids=self.map_ids)
        self.hand.set_joint_position_target(self.fixed_finger_q,
                                            joint_ids=self.fixed_finger_ids)

    def _tool_pose(self, art):
        return art.data.root_pos_w - self.scene.env_origins, art.data.root_quat_w

    def _signals(self):
        pan_p, pan_q = self._tool_pose(self.aux)
        broom_p, broom_q = self._tool_pose(self.object)
        sig = sweep_signals(self.cube.data.root_pos_w - self.scene.env_origins,
                            self.cube.data.root_lin_vel_w, pan_p, pan_q,
                            self.aux.data.root_lin_vel_w, self.cube_start, self.geometry)
        pts = quat_apply(broom_q[:, None, :].expand(-1, len(self.broom_points), -1).reshape(-1, 4),
                         self.broom_points[None].expand(self.num_envs, -1, -1).reshape(-1, 3))
        pts = pts.reshape(self.num_envs, -1, 3) + broom_p[:, None, :]
        cube_p = self.cube.data.root_pos_w - self.scene.env_origins
        broom_dist = torch.linalg.vector_norm(pts - cube_p[:, None, :], dim=-1).amin(dim=1)
        pan_up = quat_apply(pan_q, torch.tensor([0., 1., 0.], device=self.device).expand(self.num_envs, 3))
        ready = (pan_up[:, 2] > np.cos(np.radians(25.0))) & (sig["corridor"] > 0.5)
        sig["broom_dist"] = broom_dist
        return sig, ready, broom_dist < 0.018

    def _get_dones(self):
        sig, ready, broom_near = self._signals()
        out = self.progress.step(sig, ready, broom_near)
        r = self.row.clamp(max=self.T - 1)
        # Reference tracking is a confidence-weighted cost, never a success proxy.
        pp = {0: self.aux.data.root_pos_w - self.scene.env_origins,
              1: self.object.data.root_pos_w - self.scene.env_origins}
        qq = {0: self.aux.data.root_quat_w, 1: self.object.data.root_quat_w}
        track = torch.zeros(self.num_envs, device=self.device)
        for oi, ci in ((1, 0), (0, 1)):
            ep = torch.linalg.vector_norm(pp[oi] - self.ref_pos[oi][r], dim=1)
            er = _qangle(qq[oi], self.ref_quat[oi][r])
            track -= self.conf[r, ci] * (2.0 * ep + 0.05 * er)
        reward = out["task_reward"] + track - 0.002 * self.last_act.square().sum(1)
        # Open-loop only until nominal contact; afterwards require real proximity or
        # cube displacement, preventing the task clock from outrunning the object.
        advance = (self.row < self.contact_row) | broom_near | (out["moved"] >= self.geometry.moved_gate)
        self.row = torch.where(advance, (self.row + 1).clamp(max=self.T - 1), self.row)
        cube_z = self.cube.data.root_pos_w[:, 2]
        failed = cube_z < self.cfg.table_top_z - 0.03
        timeout = self.episode_length_buf >= int(self.max_episode_length - 1)
        self._tick_out = {**out, "reward": reward, "track": track,
                          "terminated": out["success"] | failed,
                          "timeout": timeout & ~out["success"] & ~failed}
        return self._tick_out["terminated"], self._tick_out["timeout"]

    def _get_rewards(self):
        return self._tick_out["reward"]

    def _get_observations(self):
        r = self.row.clamp(max=self.T - 1)
        q = self.hand.data.joint_pos[:, self.map_ids_t]
        qd = self.hand.data.joint_vel[:, self.map_ids_t]
        look = [self.ref_arm[(r+k).clamp(max=self.T-1)] - self.ref_arm[r]
                for k in (1, 4, 8, 16)]
        pan_p, pan_q = self._tool_pose(self.aux)
        broom_p, broom_q = self._tool_pose(self.object)
        cube_p = self.cube.data.root_pos_w - self.scene.env_origins
        sig, _, _ = self._signals()
        gates = self.progress.gates.float()
        obs = torch.cat([
            q, qd * 0.1, self.ref_arm[r] - q, self.cum_res / self.dev_hi,
            *look,
            pan_p, pan_q, self.aux.data.root_lin_vel_w, self.aux.data.root_ang_vel_w * 0.1,
            broom_p, broom_q, self.object.data.root_lin_vel_w, self.object.data.root_ang_vel_w * 0.1,
            cube_p, self.cube.data.root_lin_vel_w,
            sig["cube_pan"], broom_p - cube_p, self.conf[r],
            sig["progress"].unsqueeze(1), sig["corridor"].unsqueeze(1),
            sig["rel_speed"].unsqueeze(1), sig["moved"].unsqueeze(1),
            gates, (self.progress.stable_run.float()/self.geometry.stable_steps).unsqueeze(1),
            (r.float()/max(self.T-1, 1)).unsqueeze(1), self.last_act], 1)
        assert obs.shape[1] == OBS_DIM, obs.shape
        obs = obs.float().clamp(-10, 10).nan_to_num(0.0)
        # Both actor and critic already receive exact cube/tool state in policy obs.
        priv = torch.cat([sig["cube_pan"], self.cube.data.root_lin_vel_w,
                          sig["broom_dist"].unsqueeze(1), sig["rel_speed"].unsqueeze(1)], 1)
        return {"policy": obs, "priv_info": priv}

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0: return
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if self._tick_out is not None:
            self._stats["episodes"] += len(env_ids)
            for i in range(4):
                self._stats[f"gate{i+1}"] += int(self.progress.gates[env_ids, i].sum())
        DirectRLEnv._reset_idx(self, env_ids)
        n = len(env_ids); org = self.scene.env_origins[env_ids]
        qfull = self.hand.data.default_joint_pos[env_ids].clone()
        qfull[:, self.map_ids_t] = self.ref_arm[0]
        qfull[:, self.fixed_finger_ids] = self.fixed_finger_q[env_ids]
        self.hand.write_joint_state_to_sim(qfull, torch.zeros_like(qfull), env_ids=env_ids)
        self.hand.set_joint_position_target(qfull, env_ids=env_ids)
        for oi, art in ((0, self.aux), (1, self.object)):
            pose = torch.cat([self.ref_pos[oi][0].expand(n, 3),
                              self.ref_quat[oi][0].expand(n, 4)], 1).clone()
            pose[:, :3] += org
            art.write_root_pose_to_sim(pose, env_ids=env_ids)
            art.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)
        pan_p = self.ref_pos[0][0].expand(n, 3)
        pan_q = self.ref_quat[0][0].expand(n, 4)
        cube = (self.cube_start_ref.expand(n, 3).clone() if self.cube_start_ref is not None
                else _pan_cube_start(pan_p, pan_q, self.cfg.table_top_z, self.geometry))
        self.cube_start[env_ids] = cube
        pose = torch.cat([cube + org, torch.tensor([1., 0., 0., 0.], device=self.device).expand(n, 4)], 1)
        self.cube.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.cube.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), env_ids=env_ids)
        self.row[env_ids] = 0
        self.cum_res[env_ids] = 0
        self.last_act[env_ids] = 0
        self.progress.reset(env_ids)

    def pop_rates(self):
        n = self._stats["episodes"]
        out = {f"sr/gate{i}": (self._stats[f"gate{i}"]/n if n else float("nan"))
               for i in range(1, 5)}
        out["n/episodes"] = n
        self._stats = {f"gate{i}": 0 for i in range(1, 5)} | {"episodes": 0}
        return out
