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
from rl_rebuild.correction.kinematics import ArmIK
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
OBS_DIM = 191
PRIV_DIM = 22
SCRIPTED_PRELUDE_STEPS = 80
SWEEP2_FIXED_CUBE_START = (-0.0259767957, -0.1788897067, 0.8830000162)
PRIOR_BROOM = "tasks/pregrasp/priors/Sweep2_broom.npz"
PRIOR_PAN = "tasks/pregrasp/priors/Sweep2_dustpan.npz"


def _safe_arm_reference(z, margin=1.0e-6):
    """Keep float32 reference rows strictly inside the URDF/Isaac joint limits."""
    out = {}
    for side, key in (("right", "right_q"), ("left", "left_q")):
        ik = ArmIK(side)
        q = np.asarray(z[key], dtype=np.float64)
        out[key] = np.clip(q, ik.lower + margin, ik.upper - margin).astype(np.float32)
    return out


def _qangle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    d = (a * b).sum(dim=-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(d)


def _pan_cube_start(pan_pos, pan_quat, table_z, geometry=SweepGeometry()):
    """Fixed easy start with exact pan-local x/z and table-resting world height."""
    local = torch.tensor([0.0, 0.0, geometry.pan_mouth_z + geometry.start_outside],
                         dtype=pan_pos.dtype, device=pan_pos.device).expand_as(
                             pan_pos).clone()
    base = pan_pos + quat_apply(pan_quat, local)
    up_axis = quat_apply(
        pan_quat, torch.tensor([0.0, 1.0, 0.0], dtype=pan_pos.dtype,
                               device=pan_pos.device).expand_as(pan_pos))
    assert bool((up_axis[:, 2].abs() > 0.5).all()), up_axis
    target_z = float(table_z) + geometry.cube_half + 0.0005
    local[:, 1] = (target_z - base[:, 2]) / up_axis[:, 2]
    return pan_pos + quat_apply(pan_quat, local)


def build_cfg(num_envs=1, reference=REFERENCE):
    assert os.path.isfile(reference), (
        f"missing {reference}; run A_Design/L2_Reference/build_reference.py first")
    z = np.load(reference, allow_pickle=True)
    safe_arm = _safe_arm_reference(z)
    cfg = GraspTaskCfg()
    clips.configure_cfg(cfg, "Sweep2_broom")
    cfg.fixed_attached_tools = True
    cfg.approach_only = True
    # The actual object pose comes from the shared reconstruction registration below.
    # Keep the inherited single-object prior path neutral; never inject a task yaw.
    apply_grasp_prior(cfg, PRIOR_BROOM, 0.0, approach=True)
    cfg.scene.num_envs = int(num_envs)
    cfg.obj_jitter_xy = 0.0
    cfg.action_space = ACT_DIM
    cfg.observation_space = OBS_DIM
    cfg.episode_length_s = float(len(z["right_q"]) / float(z["control_hz"]) + 2.0)
    cfg.sweep_reference = os.path.abspath(reference)
    cfg.sweep_pan_prior = os.path.abspath(PRIOR_PAN)
    # Physics starts with a constraint-consistent state.  Exact-name entries are
    # installed after generic defaults and therefore win regex resolution.
    jp = dict(cfg.robot_cfg.init_state.joint_pos)
    for side, P, key in (("right", "R", "right_q"), ("left", "L", "left_q")):
        for i, value in enumerate(safe_arm[key][0], 1):
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
        safe_arm = _safe_arm_reference(self._z)
        self.ref_arm = torch.cat([to(safe_arm["right_q"]), to(safe_arm["left_q"])], 1)
        self.human_arm = torch.cat([to(self._z["human_right_q"]),
                                    to(self._z["human_left_q"])], 1)
        self.human_dh = torch.cat([self.human_arm[1:] - self.human_arm[:-1],
                                   torch.zeros(1, ACT_DIM, device=dev)], 0)
        self.ref_pos = {i: to(self._z[f"obj_pos_{i}"]) for i in (0, 1)}
        self.ref_quat = {i: to(self._z[f"obj_quat_{i}"]) for i in (0, 1)}
        self.conf = torch.stack([to(self._z["confidence_1"]),
                                 to(self._z["confidence_0"])], 1) / 100.0
        confidence_floor = self.conf.amin(dim=1)
        self.w_hand = torch.where(
            confidence_floor >= 0.70, torch.zeros_like(confidence_floor),
            torch.where(confidence_floor >= 0.40,
                        torch.full_like(confidence_floor, 0.50),
                        torch.full_like(confidence_floor, 0.80)))
        self.T = self.ref_arm.shape[0]
        # Reference construction already searches the full source-faithful tool
        # track and stores the measured closest-contact row and easy cube start.
        # Never replace that contract with a hand-authored late row.
        self.contact_row = min(int(self._z["contact_row"]), self.T - 1)
        self.cube_start_ref = to(self._z["cube_start_w"]).clone()
        self.cube_start_ref[2] = (float(cfg.table_top_z)
                                  + SweepGeometry().cube_half + 0.0005)
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
        self.prev_act = torch.zeros_like(self.cum_res)
        self._prev_arm_q = self.ref_arm[0].expand(N, -1).clone()
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
        self._stats = ({f"gate{i}": 0 for i in range(1, 5)}
                       | {"episodes": 0, "fully_inside": 0})
        self._reward_names = (
            "task", "acquire", "push", "pan_quality", "success_quality",
            "track", "shape", "action", "left_anchor")
        self._reward_sums = {name: torch.zeros((), device=dev)
                             for name in self._reward_names}
        self._reward_n = 0
        self._quality_sums = {name: torch.zeros((), device=dev) for name in (
            "pan_tilt_deg", "pan_clearance_mm", "pan_lin_speed",
            "pan_ang_speed", "broom_dist_mm", "cube_z_pan_mm",
            "deep_margin_mm", "broom_assisted_fraction")}
        self._quality_n = torch.zeros((), device=dev)
        self.best_contact = torch.zeros(N, device=dev)
        self.best_inward_progress = torch.zeros(N, device=dev)
        self.broom_assisted_progress = torch.zeros(N, device=dev)
        self.ever_fully_inside = torch.zeros(N, dtype=torch.bool, device=dev)
        self.best_pan_quality = torch.zeros(N, device=dev)

        bn = list(self.hand.body_names)
        self.hand_bid = {"right": bn.index("right_hand_C_MC"),
                         "left": bn.index("left_hand_C_MC")}
        # MeshConverter inserts an internal transform below the rigid root.  Load
        # the visible bristle face through the live USD hierarchy so all runtime
        # distances and expert targets share the Object rigid-root coordinates.
        pts, contact = self._load_broom_working_points()
        self.broom_points = to(pts)
        self.broom_contact_local = to(contact)

        self._settle_attachment_reset()
        # Calibrate the reconstructed pan track to the physical left-hand chain.
        # The analytic left-arm FK and Isaac articulation differ by a small,
        # deterministic translation.  Apply it once to the whole pan trajectory
        # (and cube start) so all 6DoF increments remain source-identical.
        pan_delta = (self.aux.data.root_pos_w - self.scene.env_origins
                     - self.ref_pos[0][0]).mean(dim=0)
        pan_shift = float(torch.linalg.vector_norm(pan_delta))
        if pan_shift >= 0.003:
            assert pan_shift < 0.010, (
                f"left tool runtime registration {pan_shift*1000:.2f}mm exceeds "
                "the approved 1cm initial-pose adjustment")
            self.ref_pos[0] = self.ref_pos[0] + pan_delta
            if self.cube_start_ref is not None:
                self.cube_start_ref[:2] = self.cube_start_ref[:2] + pan_delta[:2]
            print(f"[SweepEnv] left tool runtime registration mm="
                  f"{(pan_delta*1000).detach().cpu().numpy().round(4).tolist()}")
            self._settle_attachment_reset()
        self._validate_attachment_reset()
        # Freeze the accepted v1 physical task.  The reference NPZ's easy-start
        # field was edited after the canonical replay and experts were produced.
        self.cube_start_ref[:] = to(SWEEP2_FIXED_CUBE_START)

    def _setup_scene(self):
        # The aux helper normally uses scene_layout.  Override it with the exact
        # reference row-zero pan pose before it spawns the RigidObject.
        self.aux_init_pose_np = np.r_[np.asarray(self._z["obj_pos_0"])[0],
                                     np.asarray(self._z["obj_quat_0"])[0]]
        super()._setup_scene()
        self._replace_pan_collision()
        cube_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/SweepCube",
            spawn=sim_utils.CuboidCfg(
                size=(0.025, 0.025, 0.025),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False, solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, contact_offset=0.0012, rest_offset=0.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.005),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.6, dynamic_friction=0.5, restitution=0.02,
                    friction_combine_mode="average", restitution_combine_mode="multiply"),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.12, 0.08))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.table_top_z + 0.006)))
        self.cube = RigidObject(cube_cfg)
        self.scene.rigid_objects["sweep_cube"] = self.cube
        self._create_tool_joints()
        self.scene.filter_collisions()

    def _load_broom_working_points(self):
        """Return deterministic bristle points in the broom rigid-root frame."""
        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        root_path = "/World/envs/env_0/Object"
        root = stage.GetPrimAtPath(root_path)
        assert root.IsValid(), root_path
        cache = UsdGeom.XformCache()
        root_inv = cache.GetLocalToWorldTransform(root).GetInverse()
        chunks = []
        contact_candidates = []
        source_contact = np.asarray(self._z["brush_contact_local"], np.float64)
        for prim in stage.Traverse():
            if not (str(prim.GetPath()).startswith(root_path)
                    and prim.IsA(UsdGeom.Mesh)):
                continue
            mesh = UsdGeom.Mesh(prim)
            local = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            mask = ((local[:, 1] < -0.050) & (local[:, 2] > 0.020)
                    & (local[:, 2] < 0.090))
            if not mask.any():
                continue
            mesh_world = cache.GetLocalToWorldTransform(prim)
            root_points = np.asarray([
                root_inv.Transform(mesh_world.Transform(Gf.Vec3d(*point)))
                for point in local[mask]], dtype=np.float32)
            chunks.append(root_points)
            contact_root = root_inv.Transform(
                mesh_world.Transform(Gf.Vec3d(*source_contact)))
            contact_candidates.append((len(root_points), np.asarray(
                contact_root, dtype=np.float32)))
        assert chunks, "no live USD bristle working face found"
        points = np.concatenate(chunks, axis=0)
        assert len(points) >= 2048, len(points)
        rng = np.random.RandomState(0)
        points = points[rng.choice(len(points), 2048, replace=False)]
        contact = max(contact_candidates, key=lambda item: item[0])[1]
        print("[SweepEnv] live USD bristle proxy: "
              f"points={len(points)} bbox_mm="
              f"{np.round(np.c_[points.min(0), points.max(0)]*1000, 2).tolist()} "
              f"contact_root_mm={np.round(contact*1000, 2).tolist()}")
        return points, contact

    def _replace_pan_collision(self):
        """Use an open compound collider instead of VHACD's sealed pan mouth.

        The reconstructed mesh remains the visual source of truth.  Dynamic
        triangle meshes are not supported by PhysX, while convex decomposition
        bridges the concave mouth with an invisible wall.  Five thin boxes model
        the load-bearing floor, smooth entry ramp, side walls, and back wall.
        """
        import omni.usd
        from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        disabled = 0
        for ei in range(self.cfg.scene.num_envs):
            root_path = f"/World/envs/env_{ei}/Aux"
            root = stage.GetPrimAtPath(root_path)
            # Traverse only this environment subtree.  A full-stage traversal
            # inside the environment loop is quadratic and makes 1024-env
            # startup impractically slow.
            for prim in Usd.PrimRange(root):
                if (prim.IsA(UsdGeom.Mesh)
                        and prim.HasAPI(UsdPhysics.CollisionAPI)):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                    disabled += 1
            # Local axes: x=width, +y=up, +z=handle->open mouth.  The repaired
            # asset's central work surface is 8.5 mm high and falls to 6 mm only
            # over z=80--108 mm.  Keep the collision surface coincident with that
            # profile; in particular, never extend the ramp toward the initial
            # cube/broom geometry as the superseded 80 mm proxy did.
            boxes = {
                "floor": ((0.0, 0.0065, 0.0475), (0.120, 0.004, 0.065), 0.0),
                # A 28 mm, 5.102165-degree wedge.  Its top surface is exactly
                # y=8.5 mm at z=80 mm and y=6 mm at z=108 mm.
                "ramp": ((0.0, 0.00625396, 0.09391105),
                         (0.120, 0.002, 0.02811138), 5.102165),
                "side_l": ((-0.062, 0.020, 0.055), (0.004, 0.028, 0.080), 0.0),
                "side_r": ((+0.062, 0.020, 0.055), (0.004, 0.028, 0.080), 0.0),
                "back": ((0.0, 0.020, 0.015), (0.124, 0.028, 0.004), 0.0),
            }
            for name, (center, size, rotate_x_deg) in boxes.items():
                path = f"{root_path}/SweepOpenCollision/{name}"
                cube = UsdGeom.Cube.Define(stage, path)
                cube.CreateSizeAttr(1.0)
                xf = UsdGeom.Xformable(cube)
                # IsaacLab clones env_0 after its task-local children exist, so
                # env_1+ may already inherit authored transform operations.
                # Reuse those ops instead of attempting to author duplicates.
                ops = {op.GetOpType(): op for op in xf.GetOrderedXformOps()}
                translate = ops.get(UsdGeom.XformOp.TypeTranslate)
                if translate is None:
                    translate = xf.AddTranslateOp()
                translate.Set(Gf.Vec3d(*center))
                if rotate_x_deg:
                    rotate = ops.get(UsdGeom.XformOp.TypeRotateX)
                    if rotate is None:
                        rotate = xf.AddRotateXOp()
                    rotate.Set(float(rotate_x_deg))
                scale = ops.get(UsdGeom.XformOp.TypeScale)
                if scale is None:
                    scale = xf.AddScaleOp()
                scale.Set(Gf.Vec3d(*size))
                cube.CreateVisibilityAttr().Set("invisible")
                collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                if name == "ramp":
                    pc = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
                    pc.CreateContactOffsetAttr(0.0002)
                    pc.CreateRestOffsetAttr(0.0)
                    table_path = Sdf.Path(f"/World/envs/env_{ei}/Table")
                    assert stage.GetPrimAtPath(table_path).IsValid(), table_path
                    UsdPhysics.FilteredPairsAPI.Apply(cube.GetPrim()) \
                        .CreateFilteredPairsRel().AddTarget(table_path)
        print(f"[SweepEnv] open pan compound collision: disabled_meshes={disabled}; "
              f"boxes={5*self.cfg.scene.num_envs} (includes ramp)")

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

    def _settle_attachment_reset(self, physics_steps=24):
        """Audit the attached-tool reset under continuously applied arm control.

        Dexmate's anchor calibration intentionally advances physics after a single
        actuator write. That is sufficient for the fixed torso, but lets arm joints
        sag before this task gets its first control step. Recreate the actual Sweep
        reset here and keep sending its target while the FixedJoints settle.
        """
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
            art.write_root_velocity_to_sim(
                torch.zeros(self.num_envs, 6, device=self.device), env_ids=env_ids)
        dt = self.sim.get_physics_dt()
        for _ in range(int(physics_steps)):
            self.hand.set_joint_position_target(qfull)
            self.hand.write_data_to_sim()
            self.sim.step(render=False)
            self.hand.update(dt)
        # The two attached tools load both arms asymmetrically.  Cancel the small
        # steady-state PD sag at reset without teleporting either tool: refine only
        # the commanded arm target, then let the physical FixedJoints settle again.
        ref0 = self.ref_arm[0].expand(self.num_envs, -1)
        for _ in range(4):
            arm_err = ref0 - self.hand.data.joint_pos[:, self.map_ids_t]
            qfull[:, self.map_ids_t] += arm_err.clamp(-0.02, 0.02)
            for _ in range(8):
                self.hand.set_joint_position_target(qfull)
                self.hand.write_data_to_sim()
                self.sim.step(render=False)
                self.hand.update(dt)
        self.object.update(dt)
        self.aux.update(dt)

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
            print(f"[SweepEnv] {side} tool pose delta mm="
                  f"{((p - target_p)[0] * 1000).detach().cpu().numpy().round(4).tolist()}")
            checks.append((side, float(pose_err), float(rot_err),
                           float(joint_pos), float(joint_rot)))
        actual_q = self.hand.data.joint_pos[:, self.map_ids_t]
        target_q = self.ref_arm[0].expand_as(actual_q)
        q_err_deg = torch.rad2deg((actual_q - target_q).abs()).amax(dim=1)
        default_q = self.hand.data.default_joint_pos[:, self.map_ids_t]
        default_err_deg = torch.rad2deg((default_q - target_q).abs()).amax(dim=1)
        commanded_q = self.hand.data.joint_pos_target[:, self.map_ids_t]
        commanded_err_deg = torch.rad2deg((commanded_q - target_q).abs()).amax(dim=1)
        q_error_vector = torch.rad2deg(actual_q[0] - target_q[0]).detach().cpu().numpy()
        torque_vector = self.hand.data.applied_torque[0, self.map_ids_t].detach().cpu().numpy()
        print("[SweepEnv] reset arm audit: "
              f"actual-vs-reference max={float(q_err_deg.max()):.6f}deg; "
              f"default-vs-reference max={float(default_err_deg.max()):.6f}deg; "
              f"commanded-vs-reference max={float(commanded_err_deg.max()):.6f}deg; "
              f"q_error_deg={np.round(q_error_vector, 5).tolist()}; "
              f"applied_torque={np.round(torque_vector, 3).tolist()}; "
              f"runtime_anchor_T={np.asarray(self._anchor_T).round(8).tolist()}")
        print(f"[SweepEnv] attachment reset audit={checks}")
        assert max(x[1] for x in checks) < 0.003, checks
        assert max(x[3] for x in checks) < 0.003, checks
        assert max(x[4] for x in checks) < np.radians(2.0), checks

    def _pre_physics_step(self, actions):
        self.prev_act = self.last_act.clone()
        policy_active = self.episode_length_buf >= SCRIPTED_PRELUDE_STEPS
        self.last_act = actions.clamp(-1.0, 1.0) * policy_active.unsqueeze(1)
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
        point_dist = torch.linalg.vector_norm(pts - cube_p[:, None, :], dim=-1)
        broom_dist, nearest_id = point_dist.min(dim=1)
        nearest = pts[torch.arange(self.num_envs, device=self.device), nearest_id]
        pan_inv = quat_conjugate(pan_q)
        nearest_rel_pan = quat_apply(pan_inv, nearest - cube_p)
        cube_vel_pan = quat_apply(
            pan_inv, self.cube.data.root_lin_vel_w - self.aux.data.root_lin_vel_w)
        pan_up = quat_apply(pan_q, torch.tensor([0., 1., 0.], device=self.device).expand(self.num_envs, 3))
        pan_tilt = torch.acos(pan_up[:, 2].clamp(-1.0, 1.0))
        # Lowest edge of the physical entry wedge (local y=4mm, z=108mm).
        lip_local = torch.tensor(
            [[-0.060, 0.004, 0.108], [0.060, 0.004, 0.108]],
            device=self.device).expand(self.num_envs, -1, -1)
        lip_q = pan_q[:, None, :].expand(-1, 2, -1).reshape(-1, 4)
        lip_world = (quat_apply(lip_q, lip_local.reshape(-1, 3)).reshape(
            self.num_envs, 2, 3) + pan_p[:, None, :])
        mouth_clearance = lip_world[:, :, 2].amin(dim=1) - float(self.cfg.table_top_z)
        c = sig["cube_pan"]
        half = self.geometry.cube_half
        margins = torch.stack([
            self.geometry.pan_half_width - half - c[:, 0].abs(),
            self.geometry.pan_mouth_z - c[:, 2],
            c[:, 1] - self.geometry.pan_center_y_min,
            self.geometry.pan_center_y_max - c[:, 1]], dim=1)
        ready = (pan_up[:, 2] > np.cos(np.radians(25.0))) & (sig["corridor"] > 0.5)
        sig["broom_dist"] = broom_dist
        sig["nearest_rel_pan"] = nearest_rel_pan
        sig["cube_vel_pan"] = cube_vel_pan
        sig["pan_up"] = pan_up
        sig["pan_tilt"] = pan_tilt
        sig["mouth_clearance"] = mouth_clearance
        sig["margins"] = margins
        sig["pan_lin_speed"] = torch.linalg.vector_norm(
            self.aux.data.root_lin_vel_w, dim=1)
        sig["pan_ang_speed"] = torch.linalg.vector_norm(
            self.aux.data.root_ang_vel_w, dim=1)
        return sig, ready, broom_dist < (self.geometry.cube_half + 0.006)

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
            conf = self.conf[r, ci]
            pos_tol = 0.03 + (1.0 - conf) * 0.05
            rot_tol = np.radians(15.0) + (1.0 - conf) * np.radians(30.0)
            rot_excess = torch.where(conf < 0.40, torch.zeros_like(er),
                                     (er - rot_tol).clamp_min(0.0))
            track -= 2.0 * (ep - pos_tol).clamp_min(0.0) + 0.05 * rot_excess
        actual_arm_q = self.hand.data.joint_pos[:, self.map_ids_t]
        dq_actual = actual_arm_q - self._prev_arm_q
        shape_cos = torch.nn.functional.cosine_similarity(
            dq_actual, self.human_dh[r], dim=1).clamp(min=0.0)
        shape = 0.2 * self.w_hand[r] * shape_cos
        self._prev_arm_q = actual_arm_q.detach()
        contact_pot = torch.exp(-((sig["broom_dist"] / 0.025) ** 2))
        acquire = (contact_pot - self.best_contact).clamp_min(0.0)
        self.best_contact = torch.maximum(self.best_contact, contact_pot)
        height_ok = ((sig["margins"][:, 2] >= 0.0)
                     & (sig["margins"][:, 3] >= 0.0)).float()
        # Attribute broom quality through complete-footprint entry, but not Deep20.
        inward_progress = 0.5 * (sig["progress"] + sig["full_progress"])
        inward_delta = (inward_progress - self.best_inward_progress).clamp_min(0.0)
        self.best_inward_progress = torch.maximum(
            self.best_inward_progress, inward_progress)
        rel = sig["nearest_rel_pan"]
        contact_q = torch.exp(-((sig["broom_dist"] / 0.040) ** 2))
        behind_q = ((rel[:, 2] + 0.005) / 0.030).clamp(0.0, 1.0)
        lateral_q = torch.exp(-((rel[:, 0] / 0.030) ** 2))
        vertical_q = torch.exp(-((rel[:, 1] / 0.025) ** 2))
        broom_q = contact_q * behind_q * lateral_q * vertical_q
        assisted_delta = inward_delta * broom_q * sig["corridor"] * height_ok
        self.broom_assisted_progress += assisted_delta
        push = 4.0 * assisted_delta
        push_pot = broom_q * inward_progress
        self.ever_fully_inside |= sig["fully_inside"]
        level_q = torch.exp(-((sig["pan_tilt"] / np.radians(10.0)) ** 2))
        clear_q = torch.exp(-(((sig["mouth_clearance"] - 0.003) / 0.008) ** 2))
        still_q = torch.exp(-((sig["pan_lin_speed"] / 0.05) ** 2)
                            - ((sig["pan_ang_speed"] / 0.5) ** 2))
        pan_pot = level_q * clear_q * still_q
        pan_quality = (pan_pot - self.best_pan_quality).clamp_min(0.0)
        self.best_pan_quality = torch.maximum(self.best_pan_quality, pan_pot)
        success_quality = 2.0 * pan_pot * out["new_gate"][:, 3].float()
        action_pen = (-0.002 * self.last_act.square().sum(1)
                      - 0.001 * (self.last_act - self.prev_act).square().sum(1))
        left_anchor = -0.001 * (
            self.cum_res[:, 7:] / self.dev_hi[7:]).square().sum(1)
        reward_terms = {
            "task": out["task_reward"], "acquire": acquire, "push": push,
            "pan_quality": pan_quality, "success_quality": success_quality,
            "track": track, "shape": shape, "action": action_pen,
            "left_anchor": left_anchor,
        }
        reward = sum(reward_terms.values())
        # Open-loop only until nominal contact; afterwards require real proximity or
        # cube displacement, preventing the task clock from outrunning the object.
        replay_only = bool(getattr(self, "force_replay", False))
        advance = (torch.ones_like(self.row, dtype=torch.bool) if replay_only else
                   ((self.row < self.contact_row) | broom_near |
                    (out["moved"] >= self.geometry.moved_gate)))
        self.row = torch.where(advance, (self.row + 1).clamp(max=self.T - 1), self.row)
        cube_z = self.cube.data.root_pos_w[:, 2]
        failed = ((cube_z < self.cfg.table_top_z - 0.03)
                  | (sig["mouth_clearance"] < -0.001))
        timeout = self.episode_length_buf >= int(self.max_episode_length - 1)
        terminated = (torch.zeros_like(out["success"]) if replay_only else
                      (out["success"] | failed))
        self._tick_out = {**out, "reward": reward, "track": track, "shape": shape,
                          "reward_terms": reward_terms,
                          "signals": sig, "push_potential": push_pot,
                          "pan_potential": pan_pot,
                          "terminated": terminated,
                          "timeout": timeout & ~out["success"] & ~failed}
        for name, value in reward_terms.items():
            self._reward_sums[name] += value.detach().sum()
        self._reward_n += self.num_envs
        success_now = out["new_gate"][:, 3]
        self._quality_n += success_now.sum()
        self._quality_sums["pan_tilt_deg"] += torch.rad2deg(
            sig["pan_tilt"][success_now]).sum()
        self._quality_sums["pan_clearance_mm"] += (
            1000.0 * sig["mouth_clearance"][success_now]).sum()
        self._quality_sums["pan_lin_speed"] += sig["pan_lin_speed"][success_now].sum()
        self._quality_sums["pan_ang_speed"] += sig["pan_ang_speed"][success_now].sum()
        self._quality_sums["broom_dist_mm"] += (
            1000.0 * sig["broom_dist"][success_now]).sum()
        self._quality_sums["cube_z_pan_mm"] += (
            1000.0 * sig["cube_pan"][success_now, 2]).sum()
        self._quality_sums["deep_margin_mm"] += (
            1000.0 * sig["deep_margin"][success_now]).sum()
        self._quality_sums["broom_assisted_fraction"] += (
            self.broom_assisted_progress[success_now]).sum()
        if bool(getattr(self, "suppress_terminal_reset", False)):
            # Recorder needs to render the real terminal physics state.  DirectRLEnv
            # otherwise resets before env.step returns.
            zeros = torch.zeros_like(self._tick_out["terminated"])
            return zeros, zeros
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
            (r.float()/max(self.T-1, 1)).unsqueeze(1), self.last_act,
            sig["nearest_rel_pan"], sig["cube_vel_pan"], sig["pan_up"],
            sig["mouth_clearance"].unsqueeze(1), sig["margins"],
            (self.episode_length_buf >= SCRIPTED_PRELUDE_STEPS).float().unsqueeze(1)], 1)
        assert obs.shape[1] == OBS_DIM, obs.shape
        obs = obs.float().clamp(-10, 10).nan_to_num(0.0)
        # Both actor and critic already receive exact cube/tool state in policy obs.
        priv = torch.cat([sig["cube_pan"], self.cube.data.root_lin_vel_w,
                          sig["broom_dist"].unsqueeze(1), sig["rel_speed"].unsqueeze(1),
                          sig["nearest_rel_pan"], sig["cube_vel_pan"],
                          sig["mouth_clearance"].unsqueeze(1),
                          sig["pan_tilt"].unsqueeze(1),
                          sig["pan_lin_speed"].unsqueeze(1),
                          sig["pan_ang_speed"].unsqueeze(1), sig["margins"]], 1)
        assert priv.shape[1] == PRIV_DIM, priv.shape
        active = (self.episode_length_buf >= SCRIPTED_PRELUDE_STEPS).float().unsqueeze(1)
        return {"policy": obs, "priv_info": priv, "actor_mask": active}

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0: return
        if not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        if self._tick_out is not None:
            self._stats["episodes"] += len(env_ids)
            for i in range(4):
                self._stats[f"gate{i+1}"] += int(self.progress.gates[env_ids, i].sum())
            self._stats["fully_inside"] += int(self.ever_fully_inside[env_ids].sum())
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
        self.prev_act[env_ids] = 0
        self._prev_arm_q[env_ids] = self.ref_arm[0]
        self.progress.reset(env_ids)
        self.best_contact[env_ids] = 0
        self.best_inward_progress[env_ids] = 0
        self.broom_assisted_progress[env_ids] = 0
        self.ever_fully_inside[env_ids] = False
        self.best_pan_quality[env_ids] = 0

    def pop_rates(self):
        n = self._stats["episodes"]
        out = {f"sr/gate{i}": (self._stats[f"gate{i}"]/n if n else 0.0)
               for i in range(1, 5)}
        out["sr/fully_inside"] = (self._stats["fully_inside"] / n if n else 0.0)
        out["n/episodes"] = n
        denom = max(self._reward_n, 1)
        out.update({f"ep_rew/{name}": float(value / denom)
                    for name, value in self._reward_sums.items()})
        qn = float(self._quality_n)
        out.update({f"quality/{name}_success": (float(value / qn)
                    if qn > 0 else 0.0)
                    for name, value in self._quality_sums.items()})
        out["quality/n_success"] = qn
        out["residual/right_usage"] = float((
            self.cum_res[:, :7].abs() / self.dev_hi[:7]).mean())
        out["residual/left_usage"] = float((
            self.cum_res[:, 7:].abs() / self.dev_hi[7:]).mean())
        self._stats = ({f"gate{i}": 0 for i in range(1, 5)}
                       | {"episodes": 0, "fully_inside": 0})
        for value in self._reward_sums.values(): value.zero_()
        for value in self._quality_sums.values(): value.zero_()
        self._quality_n.zero_()
        self._reward_n = 0
        return out
