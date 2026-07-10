#!/usr/bin/env python3
"""Play a generated grasp trajectory back in Isaac Sim with real PhysX physics.

The hand USD is a PhysX **articulation** (ArticulationRootAPI baked into its
physics layer) whose palm is anchored to the world by a zero-offset
``root_joint``. That joint is deactivated at load, turning the hand into a
floating-base articulation, and the whole hand is then driven through the
PhysX tensor API (``isaacsim.core.prims.SingleArticulation``): root pose +
finite-difference root velocities set every trajectory step, finger joints
tracked by their PD position drives (radians, via ``ArticulationAction``).
Driving through the articulation view keeps the joints solved in reduced
coordinates -- links physically cannot separate -- unlike teleporting authored
USD transforms, which this PhysX version does not reliably honor for
articulations (and which destabilizes the solver when combined with a
per-link kinematic flag). The object is a dynamic rigid body resting on a
static table, gravity on. Renders a video.

Reuses ``ocir.isaac.replay_dexycb``'s DexYCB camera-frame-to-Isaac-world
mapping and video/camera helpers. Physics-material/collision/drive-gain
recipe modeled on (not copied from) proven references:
``~/OCIR/third_party/Articulation_Bodex/visualize_sharpa_grasp.py`` and the
MagicSim floating Sharpa hand config
(``MagicSim/src/magicsim/Env/Robot/Cfg/Dexterous/SharpaWaveFloating.py``,
source of the articulation solver iteration counts).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ocir.grasp_synthesis.assets import DEFAULT_SHARPA_WAVE_RIGHT_CONFIG, load_sharpa_wave_right
from ocir.grasp_traj.trajectory_schema import SEGMENT_CARRY, GraspTrajectory
from ocir.isaac.replay_dexycb import (
    DexYCBFrameMapper,
    capture_camera_png,
    load_dexycb_frame_mapper,
    make_record_camera,
    matrix_to_quat_wxyz,
    write_video,
)
from ocir.isaac.sim_cli import apply_mode_defaults, run_sim_cli_main
from ocir.sim.control_client import request_json
from ocir.sim.isaac_server import DEFAULT_OCIR_DATA_ROOT

DEFAULT_ISAACSIM_MODE = os.environ.get("OCIR_ISAACSIM_MODE", "webrtc").lower()
DEFAULT_MANIFEST = DEFAULT_OCIR_DATA_ROOT / "processed_data/dex_ycb/manifests/selected_5_sequences.json"
DEFAULT_OUT_DIR = DEFAULT_OCIR_DATA_ROOT / "testing/grasp_traj/isaac_sim"
OCIR_ARTIFACT_DIRS = {"raw_data", "processed_data", "testing"}

PHYSICS_SCENE_PATH = "/World/PhysicsScene"
OBJECT_WRAP, OBJECT_REF = "/World/Object", "/World/Object/ref"
HAND_WRAP, HAND_REF = "/World/Hand", "/World/Hand/ref"

#: Articulation solver iteration counts, from MagicSim's floating Sharpa
#: hand config (SharpaWaveFloating.py) -- the per-rigid-body iteration
#: attributes are ignored for articulation links; these are the ones PhysX
#: actually reads.
ARTICULATION_SOLVER_POSITION_ITERATIONS = 20
ARTICULATION_SOLVER_VELOCITY_ITERATIONS = 10

CARRY_MODE_FRICTION = "friction"
CARRY_MODE_KINEMATIC = "kinematic"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] OCIR_GRASP_TRAJ_SIM {message}", flush=True)


def resolve_artifact_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] in OCIR_ARTIFACT_DIRS:
        return DEFAULT_OCIR_DATA_ROOT / path
    return path


# ---------------------------------------------------------------------------
# Camera-frame -> Isaac-world pose mapping (batched wrapper around
# DexYCBFrameMapper, which itself expects a 3x4 [R|t] "pose_y"-style matrix).
# ---------------------------------------------------------------------------


def camera_pos_quat_to_isaac(
    pos: np.ndarray, quat_wxyz: np.ndarray, frame_mapper: DexYCBFrameMapper, z_offset: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """(...,3), (...,4) wxyz, camera frame -> (pos (...,3), quat (...,4) wxyz), Isaac world."""

    from ocir.grasp_traj.trajectory_schema import quat_wxyz_to_matrix

    pos = np.asarray(pos, dtype=np.float64)
    rot_cam = quat_wxyz_to_matrix(np.asarray(quat_wxyz, dtype=np.float64))
    rot = frame_mapper.rotation[None, ...] @ rot_cam if rot_cam.ndim > 2 else frame_mapper.rotation @ rot_cam
    if pos.ndim > 1:
        out_pos = (frame_mapper.rotation[None] @ pos[..., None]).squeeze(-1) + frame_mapper.translation[None]
    else:
        out_pos = frame_mapper.rotation @ pos + frame_mapper.translation
    out_pos = np.array(out_pos, dtype=np.float64)
    out_pos[..., 2] += float(z_offset)
    if rot.ndim > 2:
        quats = np.stack([matrix_to_quat_wxyz(rot[i]) for i in range(rot.shape[0])], axis=0)
    else:
        quats = matrix_to_quat_wxyz(rot)
    return out_pos, quats


# ---------------------------------------------------------------------------
# Physics scene / material setup
# ---------------------------------------------------------------------------


def setup_physics_scene(stage, *, time_steps_per_second: float) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics

    prim = stage.GetPrimAtPath(PHYSICS_SCENE_PATH)
    if not prim.IsValid():
        prim = stage.DefinePrim(PHYSICS_SCENE_PATH, "PhysicsScene")
    scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    if not prim.HasAPI(PhysxSchema.PhysxSceneAPI):
        PhysxSchema.PhysxSceneAPI.Apply(prim)
    physx_scene = PhysxSchema.PhysxSceneAPI(prim)
    physx_scene.CreateSolverTypeAttr().Set("TGS")
    # Substep granularity only: each app.update() always advances sim time by
    # 1/60 s; PhysX covers it in time_steps_per_second-sized substeps. Keep
    # this a multiple of 60 so every update gets a whole number of substeps.
    physx_scene.CreateTimeStepsPerSecondAttr().Set(float(time_steps_per_second))
    aggregate_pairs_capacity = 2**20
    physx_scene.CreateGpuFoundLostAggregatePairsCapacityAttr().Set(aggregate_pairs_capacity)
    physx_scene.CreateGpuTotalAggregatePairsCapacityAttr().Set(aggregate_pairs_capacity)


def deactivate_hand_root_joint(stage, ref_path: str) -> list[str]:
    """Find and deactivate the joint(s) that anchor the hand to the world
    (an empty ``physics:body0`` relationship -- our asset's URDF-import
    convention names this ``root_joint``, a zero-offset PhysicsFixedJoint from
    "nothing" to the palm). With it gone the hand becomes a floating-base
    articulation whose root pose can be set through the PhysX tensor API
    every step (see ``HandArticulationDriver``). Nothing is marked kinematic:
    a kinematic link inside a live articulation destabilizes the solver
    (links visibly separate, fingers jitter).

    Logs (and returns) whatever it finds; never assumes silently."""

    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(ref_path)
    deactivated: list[str] = []
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        if len(joint.GetBody0Rel().GetTargets()) == 0:
            prim.SetActive(False)
            deactivated.append(str(prim.GetPath()))
    if deactivated:
        log(f"deactivated world-anchoring joint(s) under {ref_path}: {deactivated}")
    else:
        log(f"WARNING: no world-anchoring joint found under {ref_path} (hand root may still be fixed to the world)")
    return deactivated


def setup_hand_articulation_root(stage, ref_path: str) -> str:
    """Ensure the hand has an ``ArticulationRootAPI`` and set the
    articulation-level solver iteration counts there. Uses the asset's own
    API if one survives layer composition; otherwise applies it to the hand's
    reference root (the asset's authoring ``.usda`` applies it to the same
    top-level prim, but that layer is not in the loaded USD's stack). Returns
    the prim path (needed to build the ``SingleArticulation`` view after the
    timeline starts)."""

    from pxr import PhysxSchema, Usd, UsdPhysics

    root = stage.GetPrimAtPath(ref_path)
    root_prim = None
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            root_prim = prim
            log(f"articulation root (from asset): {prim.GetPath()}")
            break
    if root_prim is None:
        root_prim = root
        UsdPhysics.ArticulationRootAPI.Apply(root_prim)
        log(f"articulation root (applied at runtime): {root_prim.GetPath()}")
    if not root_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
        PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    articulation = PhysxSchema.PhysxArticulationAPI(root_prim)
    articulation.CreateSolverPositionIterationCountAttr().Set(ARTICULATION_SOLVER_POSITION_ITERATIONS)
    articulation.CreateSolverVelocityIterationCountAttr().Set(ARTICULATION_SOLVER_VELOCITY_ITERATIONS)
    # Finger links deliberately interpenetrate their neighbors at grasp/
    # squeeze postures; resolving those as contacts destabilizes the solver.
    articulation.CreateEnabledSelfCollisionsAttr().Set(False)
    return str(root_prim.GetPath())


def high_friction_material(stage, root_path: str, mat_path: str, *, static_friction: float, dynamic_friction: float, restitution: float = 0.0, combine_mode: str = "multiply") -> int:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    if not stage.GetPrimAtPath(mat_path).IsValid():
        UsdShade.Material.Define(stage, mat_path)
    mat_prim = stage.GetPrimAtPath(mat_path)
    if not mat_prim.HasAPI(UsdPhysics.MaterialAPI):
        UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_mat = UsdPhysics.MaterialAPI(mat_prim)
    phys_mat.CreateStaticFrictionAttr().Set(float(static_friction))
    phys_mat.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    phys_mat.CreateRestitutionAttr().Set(float(restitution))
    if not mat_prim.HasAPI(PhysxSchema.PhysxMaterialAPI):
        PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    PhysxSchema.PhysxMaterialAPI(mat_prim).CreateFrictionCombineModeAttr().Set(combine_mode)

    root = stage.GetPrimAtPath(root_path)
    bound = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI(prim).Bind(
                UsdShade.Material(mat_prim), bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
            )
            bound += 1
    return bound


def set_world_pose(stage, prim_path: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    translate = orient = None
    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
            orient = op
    if translate is None:
        translate = xform.AddTranslateOp()
    if orient is None:
        orient = xform.AddOrientOp()
    p = np.asarray(pos, dtype=float)
    w, x, y, z = [float(v) for v in quat_wxyz]
    translate.Set(Gf.Vec3d(p[0], p[1], p[2]))
    if orient.GetPrecision() == UsdGeom.XformOp.PrecisionDouble:
        orient.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    else:
        orient.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


# ---------------------------------------------------------------------------
# Hand: collision, drives, joint targets
# ---------------------------------------------------------------------------


def setup_hand_collision(stage, ref_path: str, *, min_thickness: float, hull_vertex_limit: int, max_convex_hulls: int, contact_offset: float, rest_offset: float) -> int:
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(ref_path)
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            log(f"{ref_path}: hand ships baked collision; using it as-is")
            return 0

    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.Joint):
            continue
        for api in (UsdPhysics.CollisionAPI, UsdPhysics.MeshCollisionAPI, PhysxSchema.PhysxCollisionAPI, PhysxSchema.PhysxConvexDecompositionCollisionAPI):
            if prim.HasAPI(api):
                prim.RemoveAPI(api)

    count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        cur = prim
        in_rigid_body = False
        for _ in range(5):
            if cur.HasAPI(UsdPhysics.RigidBodyAPI):
                in_rigid_body = True
                break
            cur = cur.GetParent()
            if not cur or not cur.IsValid():
                break
        if not in_rigid_body or "visual" in prim.GetName().lower():
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexDecomposition")
        decomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        decomp.CreateMinThicknessAttr().Set(float(min_thickness))
        decomp.CreateHullVertexLimitAttr().Set(int(hull_vertex_limit))
        decomp.CreateMaxConvexHullsAttr().Set(int(max_convex_hulls))
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr().Set(float(contact_offset))
        collision.CreateRestOffsetAttr().Set(float(rest_offset))
        count += 1
    log(f"{ref_path}: rebuilt convexDecomposition collision on {count} hand meshes")
    return count


def setup_hand_drives(stage, ref_path: str, *, stiffness: float, damping: float, max_force: float, armature: float, joint_friction: float) -> int:
    from pxr import PhysxSchema, Usd, UsdPhysics

    root = stage.GetPrimAtPath(ref_path)
    n = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            # Stability caps (reference values): without them, finger drives
            # squeezing into a heavy or kinematic (infinite-mass) object build
            # up unbounded depenetration/joint velocities and the articulation
            # explodes.
            rb_api = PhysxSchema.PhysxRigidBodyAPI(prim)
            rb_api.CreateMaxDepenetrationVelocityAttr().Set(2.0)
            rb_api.CreateMaxLinearVelocityAttr().Set(4.0)
            rb_api.CreateMaxAngularVelocityAttr().Set(4.0 * 57.29577951308232)  # attr is in deg/s
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            (drive.GetStiffnessAttr() or drive.CreateStiffnessAttr()).Set(float(stiffness))
            (drive.GetDampingAttr() or drive.CreateDampingAttr()).Set(float(damping))
            (drive.GetMaxForceAttr() or drive.CreateMaxForceAttr()).Set(float(max_force))
            (drive.GetTargetPositionAttr() or drive.CreateTargetPositionAttr()).Set(0.0)
            if not prim.HasAPI(PhysxSchema.PhysxJointAPI):
                PhysxSchema.PhysxJointAPI.Apply(prim)
            physx_joint = PhysxSchema.PhysxJointAPI(prim)
            physx_joint.CreateJointFrictionAttr().Set(float(joint_friction))
            physx_joint.CreateArmatureAttr().Set(float(armature))
            physx_joint.CreateMaxJointVelocityAttr().Set(570.0)  # deg/s, ~10 rad/s
            n += 1
    log(f"{ref_path}: configured {n} finger joint drives")
    return n


def quats_to_angular_velocities(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """(T,4) wxyz -> (T,3) world-frame angular velocities by finite
    differences (angle-axis of the relative rotation over dt); last row 0."""

    from ocir.grasp_traj.trajectory_schema import quat_wxyz_to_matrix

    quats = np.asarray(quats_wxyz, dtype=np.float64)
    omega = np.zeros((quats.shape[0], 3), dtype=np.float64)
    for t in range(quats.shape[0] - 1):
        r0 = quat_wxyz_to_matrix(quats[t])
        r1 = quat_wxyz_to_matrix(quats[t + 1])
        rel = r1 @ r0.T
        cos_angle = np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)
        angle = float(np.arccos(cos_angle))
        if angle < 1e-8:
            continue
        axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
        axis = axis / (2.0 * np.sin(angle))
        omega[t] = axis * (angle / float(dt))
    return omega


class HandArticulationDriver:
    """Drives the floating-base hand articulation through the PhysX tensor
    API: root pose + root velocities + finger PD position targets (radians).
    Must be constructed after the timeline is playing (the physics simulation
    view does not exist before that)."""

    def __init__(self, articulation_root_path: str, joint_order: tuple[str, ...]):
        from isaacsim.core.prims import SingleArticulation

        self.articulation = SingleArticulation(articulation_root_path)
        self.articulation.initialize()
        dof_names = list(self.articulation.dof_names)
        missing = [name for name in joint_order if name not in dof_names]
        if missing:
            raise RuntimeError(f"trajectory joints not present in articulation DOFs: {missing} (DOFs: {dof_names})")
        self.dof_indices = np.asarray([dof_names.index(name) for name in joint_order], dtype=np.int32)
        self.num_dofs = len(dof_names)
        log(f"articulation view ready: {self.num_dofs} DOFs, driving {len(joint_order)} of them")

    def reset_joints(self, joint_positions_rad: np.ndarray) -> None:
        """Teleport joints to the given positions (start of playback, so the
        drives don't have to swing from the asset's zero pose first)."""

        self.articulation.set_joint_positions(
            np.asarray(joint_positions_rad, dtype=np.float32), joint_indices=self.dof_indices
        )

    def set_drive_strength(self, stiffness: float, damping: float, max_force: float) -> None:
        """Set PD gains + effort cap on all driven finger joints at once
        (used to soften the fingers during the close segment so an early-
        touching finger stalls against the object instead of shoving it)."""

        controller = self.articulation.get_articulation_controller()
        kps = np.full(self.num_dofs, float(stiffness), dtype=np.float32)
        kds = np.full(self.num_dofs, float(damping), dtype=np.float32)
        controller.set_gains(kps=kps, kds=kds)
        controller.set_max_efforts(np.full(self.num_dofs, float(max_force), dtype=np.float32))

    def drive(
        self,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
        joint_targets_rad: np.ndarray,
        *,
        linear_velocity: np.ndarray | None = None,
        angular_velocity: np.ndarray | None = None,
    ) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        self.articulation.set_world_pose(position=np.asarray(pos, dtype=np.float32), orientation=np.asarray(quat_wxyz, dtype=np.float32))
        self.articulation.set_linear_velocity(
            np.zeros(3, dtype=np.float32) if linear_velocity is None else np.asarray(linear_velocity, dtype=np.float32)
        )
        self.articulation.set_angular_velocity(
            np.zeros(3, dtype=np.float32) if angular_velocity is None else np.asarray(angular_velocity, dtype=np.float32)
        )
        self.articulation.apply_action(
            ArticulationAction(
                joint_positions=np.asarray(joint_targets_rad, dtype=np.float32),
                joint_indices=self.dof_indices,
            )
        )

    def max_joint_tracking_error(self, joint_targets_rad: np.ndarray) -> float:
        actual = self.articulation.get_joint_positions(joint_indices=self.dof_indices)
        return float(np.abs(np.asarray(actual, dtype=np.float64) - np.asarray(joint_targets_rad, dtype=np.float64)).max())


# ---------------------------------------------------------------------------
# Object / table
# ---------------------------------------------------------------------------


def estimate_object_mass(mesh_path: Path, density: float, *, lo: float = 0.05, hi: float = 2.0) -> float:
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    try:
        volume = float(mesh.convex_hull.volume) if not mesh.is_watertight else float(mesh.volume)
    except Exception:
        volume = float(mesh.convex_hull.volume)
    mass = abs(volume) * float(density)
    return float(np.clip(mass, lo, hi))


def build_object(stage, mesh_path: Path, *, mass_kg: float, kinematic: bool, args: argparse.Namespace) -> dict:
    import trimesh
    from pxr import UsdGeom, UsdPhysics

    from ocir.isaac.visualize_grasp import add_material, define_mesh

    loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
    vertices = np.asarray(loaded.vertices, dtype=float)
    faces = np.asarray(loaded.faces, dtype=np.int64)

    UsdGeom.Xform.Define(stage, OBJECT_WRAP)
    material = add_material(stage, "/World/Materials/Object", (0.95, 0.72, 0.16))
    mesh_report = define_mesh(stage, f"{OBJECT_REF}", vertices, faces, material)
    prim = stage.GetPrimAtPath(OBJECT_REF)

    # --carry-mode kinematic: the object is a pure visual prim teleported
    # along the recorded reference trajectory every step -- no collision, no
    # rigid body. Fingers squeezing into an immovable, per-frame-teleporting
    # collider build unbounded contact energy and detonate the articulation,
    # and this mode only exists to validate trajectory/frame-mapping geometry
    # anyway. --carry-mode friction (default): a normal dynamic body, moved
    # only by gravity + hand contact.
    if not kinematic:
        from pxr import PhysxSchema

        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(float(mass_kg))
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("convexDecomposition")
        decomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
        decomp.CreateMinThicknessAttr().Set(0.002)
        decomp.CreateHullVertexLimitAttr().Set(64)
        decomp.CreateMaxConvexHullsAttr().Set(int(args.convex_decomp_max_hulls))
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr().Set(0.004)
        collision.CreateRestOffsetAttr().Set(0.001)
        # Same stability caps as the hand links: a squeeze pinch between
        # finger colliders otherwise ejects the object at unbounded
        # depenetration velocity (watermelon-seed style).
        rb_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb_api.CreateMaxDepenetrationVelocityAttr().Set(2.0)
        # Nothing in this scene legitimately moves faster than the capped
        # wrist (0.25 m/s) plus a drop -- a tight cap keeps a slipped grasp's
        # pinch-ejection local instead of firing the object across the room.
        rb_api.CreateMaxLinearVelocityAttr().Set(1.5)
        rb_api.CreateMaxAngularVelocityAttr().Set(4.0 * 57.29577951308232)
        rb_api.CreateSolverPositionIterationCountAttr().Set(16)
        rb_api.CreateSolverVelocityIterationCountAttr().Set(2)

    mesh_report["mass_kg"] = float(mass_kg)
    mesh_report["vertices_local"] = vertices
    return mesh_report


def build_table(stage, *, center_xy: np.ndarray, size_xy: np.ndarray, top_z: float, margin: float) -> dict:
    from pxr import Gf, UsdGeom, UsdPhysics

    from ocir.isaac.visualize_grasp import add_material, bind_material

    material = add_material(stage, "/World/Materials/Table", (0.45, 0.42, 0.36))
    thickness = 0.04
    cube = UsdGeom.Cube.Define(stage, "/World/Table")
    size_xy = np.maximum(size_xy + 2.0 * margin, np.asarray([0.35, 0.35]))
    cube.AddTranslateOp().Set(Gf.Vec3d(float(center_xy[0]), float(center_xy[1]), float(top_z - thickness / 2.0)))
    cube.AddScaleOp().Set(Gf.Vec3f(float(size_xy[0]) / 2.0, float(size_xy[1]) / 2.0, thickness / 2.0))
    bind_material(cube.GetPrim(), material)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return {"path": "/World/Table", "top_z": float(top_z), "size_xy": size_xy.tolist()}


def build_hand(stage, hand_usd_path: Path, args: argparse.Namespace) -> dict:
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Usd, UsdGeom

    UsdGeom.Xform.Define(stage, HAND_WRAP)
    add_reference_to_stage(str(hand_usd_path), HAND_REF)
    if not hand_usd_path.name.endswith("instanceable.usd"):
        root = stage.GetPrimAtPath(HAND_REF)
        for prim in Usd.PrimRange(root):
            if prim.IsInstanceable():
                prim.SetInstanceable(False)

    deactivated_joints = deactivate_hand_root_joint(stage, HAND_REF)
    articulation_root = setup_hand_articulation_root(stage, HAND_REF)
    collider_count = setup_hand_collision(
        stage, HAND_REF,
        min_thickness=0.002, hull_vertex_limit=64, max_convex_hulls=args.convex_decomp_max_hulls,
        contact_offset=0.004, rest_offset=0.001,
    )
    drive_count = setup_hand_drives(
        stage, HAND_REF,
        stiffness=args.joint_stiffness, damping=args.joint_damping, max_force=args.joint_max_force,
        armature=args.joint_armature, joint_friction=args.joint_friction,
    )
    return {
        "deactivated_joints": deactivated_joints,
        "articulation_root": articulation_root,
        "collider_count": collider_count,
        "drive_count": drive_count,
    }


# ---------------------------------------------------------------------------
# Object pose readback (for physics metrics)
# ---------------------------------------------------------------------------


def read_object_world_pose(stage) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(OBJECT_REF)
    xform_cache = UsdGeom.XformCache()
    matrix = xform_cache.GetLocalToWorldTransform(prim)
    pos = np.array([matrix[3][0], matrix[3][1], matrix[3][2]], dtype=float)
    rot = np.array([[matrix[i][j] for j in range(3)] for i in range(3)], dtype=float)
    quat = matrix_to_quat_wxyz(rot)
    return pos, quat


def compute_physics_metrics(
    object_track_pos: list[np.ndarray],
    resting_z: float,
    carry_mask: np.ndarray,
    reference_final_pos: np.ndarray,
    reference_final_quat: np.ndarray,
    final_actual_pos: np.ndarray,
    final_actual_quat: np.ndarray,
    *,
    lift_threshold: float,
    drop_threshold: float,
) -> dict:
    positions = np.stack(object_track_pos, axis=0)
    carry_z = positions[: len(carry_mask)][carry_mask][:, 2] if carry_mask.any() else np.asarray([])
    lifted = bool(carry_z.size and (carry_z.max() - resting_z) >= lift_threshold)
    dropped = False
    if lifted and carry_z.size:
        peak_idx = int(np.argmax(carry_z))
        if peak_idx < carry_z.size - 1:
            dropped = bool((carry_z[-1] - resting_z) <= drop_threshold)
    pos_error = float(np.linalg.norm(final_actual_pos - reference_final_pos))
    return {
        "lifted": lifted,
        "object_dropped": dropped,
        "resting_z": float(resting_z),
        "max_carry_z": float(carry_z.max()) if carry_z.size else None,
        "final_object_position_error_m": pos_error,
    }


# ---------------------------------------------------------------------------
# Main sim loop
# ---------------------------------------------------------------------------


def simulate_grasp_traj(app, args: argparse.Namespace, progress=None) -> dict:
    import omni.timeline
    import omni.usd
    from pxr import UsdGeom, UsdLux

    def phase(message: str) -> None:
        log(message)
        if progress is not None:
            progress(message, phase="run")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    phase("loading trajectory")
    traj = GraspTrajectory.load(args.trajectory_dir)
    object_mesh_path = Path(traj.extra_metadata.get("object_mesh") or "")
    if not object_mesh_path.exists():
        raise FileNotFoundError(f"object mesh referenced by trajectory not found: {object_mesh_path}")

    sequence_id = args.sequence_id or Path(traj.sequence_dir).name
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    sequence = next((s for s in manifest["sequences"] if s["sequence_id"] == sequence_id), None)
    if sequence is None:
        raise KeyError(f"sequence_id {sequence_id!r} not found in manifest {args.manifest}")
    frame_mapper = load_dexycb_frame_mapper(manifest, sequence)

    phase("computing scene alignment (z-offset)")
    init_pos, init_quat = camera_pos_quat_to_isaac(traj.object_pos_camera[0], traj.object_quat_camera[0], frame_mapper, z_offset=0.0)
    import trimesh

    mesh_for_bounds = trimesh.load(str(object_mesh_path), force="mesh", process=False)
    from ocir.grasp_traj.trajectory_schema import quat_wxyz_to_matrix

    init_rot = quat_wxyz_to_matrix(init_quat)
    world_verts_init = np.asarray(mesh_for_bounds.vertices, dtype=float) @ init_rot.T + init_pos
    z_offset = float(-world_verts_init[:, 2].min() + 0.001)
    tabletop_z = float(args.tabletop_z)
    z_offset += tabletop_z

    phase("setting up stage")
    omni.usd.get_context().new_stage()
    app.update()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Materials")
    UsdLux.DistantLight.Define(stage, "/World/KeyLight").CreateIntensityAttr(450.0)
    UsdLux.DomeLight.Define(stage, "/World/FillLight").CreateIntensityAttr(180.0)

    time_steps_per_second = float(args.time_steps_per_second)
    setup_physics_scene(stage, time_steps_per_second=time_steps_per_second)

    all_pos_isaac, all_quat_isaac = camera_pos_quat_to_isaac(traj.hand_pos_camera, traj.hand_quat_camera, frame_mapper, z_offset=z_offset)
    obj_pos_isaac_all, obj_quat_isaac_all = camera_pos_quat_to_isaac(traj.object_pos_camera, traj.object_quat_camera, frame_mapper, z_offset=z_offset)
    # world_verts_init was computed with z_offset=0.0 above (needed to derive
    # z_offset itself); shift it by the full offset to align with the
    # already-offset hand/object trajectories for the camera bbox below.
    world_verts_init_final = world_verts_init + np.array([0.0, 0.0, z_offset])
    world_verts_all = np.concatenate([world_verts_init_final, all_pos_isaac], axis=0)
    bbox_min = np.minimum(world_verts_all.min(axis=0), all_pos_isaac.min(axis=0))
    bbox_max = np.maximum(world_verts_all.max(axis=0), all_pos_isaac.max(axis=0))
    table_center_xy = (bbox_min[:2] + bbox_max[:2]) / 2.0
    table_size_xy = bbox_max[:2] - bbox_min[:2]

    phase("building table")
    table_report = build_table(stage, center_xy=table_center_xy, size_xy=table_size_xy, top_z=tabletop_z, margin=args.table_margin)

    phase("building object")
    mass_kg = float(args.object_mass) if args.object_mass is not None else estimate_object_mass(object_mesh_path, args.object_density)
    object_report = build_object(
        stage, object_mesh_path, mass_kg=mass_kg, kinematic=(args.carry_mode == CARRY_MODE_KINEMATIC), args=args
    )
    set_world_pose(stage, OBJECT_WRAP, obj_pos_isaac_all[0], obj_quat_isaac_all[0])

    phase("building hand")
    hand_usd_path = Path(args.hand_usd) if args.hand_usd else load_sharpa_wave_right(args.asset_config).usd_path
    hand_report = build_hand(stage, hand_usd_path, args)
    hand_friction_bound = high_friction_material(
        stage, HAND_REF, "/World/Materials/HandFriction", static_friction=args.friction, dynamic_friction=args.friction
    )
    object_friction_bound = high_friction_material(
        stage, OBJECT_REF, "/World/Materials/ObjectFriction", static_friction=args.friction, dynamic_friction=args.friction
    )
    set_world_pose(stage, HAND_WRAP, all_pos_isaac[0], all_quat_isaac[0])

    phase("framing camera")
    camera, camera_report = make_record_camera(stage, args, center=(bbox_min + bbox_max) / 2.0, bbox_min=bbox_min, bbox_max=bbox_max)

    phase("starting timeline")
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        app.update()

    phase("initializing hand articulation view")
    hand_driver = HandArticulationDriver(hand_report["articulation_root"], traj.joint_order)
    hand_driver.reset_joints(traj.finger_targets[0])

    # Finite-difference root velocities: without them every set_world_pose is
    # a zero-velocity teleport and finger-object contacts never see the
    # wrist's true motion (friction can't carry the object). Each app.update()
    # advances sim time by exactly 1/60 s, so the dt these velocities must be
    # consistent with is the SIMULATED time per trajectory frame -- not
    # traj.dt -- or the root overshoots its own teleports every frame and
    # pumps energy into the articulation.
    sim_dt_per_frame = int(args.sim_steps_per_frame) / 60.0
    if abs(sim_dt_per_frame - float(traj.dt)) > 1e-6:
        log(
            f"WARNING: simulated dt per frame ({sim_dt_per_frame:.4f}s) != trajectory dt "
            f"({traj.dt:.4f}s); playback speed scales by {float(traj.dt) / sim_dt_per_frame:.2f}x"
        )
    lin_vel = np.zeros_like(all_pos_isaac)
    lin_vel[:-1] = np.diff(all_pos_isaac, axis=0) / sim_dt_per_frame
    ang_vel = quats_to_angular_velocities(all_quat_isaac, sim_dt_per_frame)

    carry_mask = traj.segment == SEGMENT_CARRY
    resting_z = float(obj_pos_isaac_all[0, 2])
    object_track: list[np.ndarray] = []
    frame_paths: list[Path] = []
    joint_error_per_step: list[float] = []
    blowup_logged = False

    # Softened finger drives during the close segment: the close targets are
    # contact-projected but sphere-vs-hull geometry differences still leave a
    # few mm of commanded overlap, and full-strength drives turn that into a
    # shove. Full gains are restored for squeeze/carry (the squeeze IS a
    # force request).
    from ocir.grasp_traj.trajectory_schema import SEGMENT_CLOSE

    compliant = None  # None until the first close step; then True/False
    phase(f"simulating {traj.num_steps} steps")
    for t in range(traj.num_steps):
        in_close = bool(traj.segment[t] == SEGMENT_CLOSE)
        if in_close and compliant is not True:
            hand_driver.set_drive_strength(
                float(args.close_joint_stiffness), float(args.joint_damping), float(args.close_joint_max_force)
            )
            compliant = True
        elif not in_close and compliant is True:
            hand_driver.set_drive_strength(
                float(args.joint_stiffness), float(args.joint_damping), float(args.joint_max_force)
            )
            compliant = False
        hand_driver.drive(
            all_pos_isaac[t], all_quat_isaac[t], traj.finger_targets[t],
            linear_velocity=lin_vel[t], angular_velocity=ang_vel[t],
        )
        if args.carry_mode == CARRY_MODE_KINEMATIC:
            set_world_pose(stage, OBJECT_WRAP, obj_pos_isaac_all[t], obj_quat_isaac_all[t])
        for _ in range(int(args.sim_steps_per_frame)):
            app.update()
        step_error = hand_driver.max_joint_tracking_error(traj.finger_targets[t])
        joint_error_per_step.append(step_error)
        if step_error > 1.0 and not blowup_logged:
            blowup_logged = True
            log(f"WARNING: joint tracking error {step_error:.2f} rad at step {t} (segment {int(traj.segment[t])}) -- articulation destabilizing")
        pos, _ = read_object_world_pose(stage)
        object_track.append(pos)
        if t % int(args.capture_every) == 0:
            frame_path = frames_dir / f"frame_{t:05d}.png"
            capture_camera_png(app, camera, frame_path)
            frame_paths.append(frame_path)

    phase(f"settling ({args.settle_steps} steps)")
    for _ in range(int(args.settle_steps)):
        # Keep holding the final commanded pose: the hand is a floating-base
        # articulation, so it would free-fall under gravity the moment root
        # driving stops.
        hand_driver.drive(all_pos_isaac[-1], all_quat_isaac[-1], traj.finger_targets[-1])
        if args.carry_mode == CARRY_MODE_KINEMATIC:
            set_world_pose(stage, OBJECT_WRAP, obj_pos_isaac_all[-1], obj_quat_isaac_all[-1])
        app.update()
        pos, _ = read_object_world_pose(stage)
        object_track.append(pos)

    final_pos, final_quat = read_object_world_pose(stage)
    metrics = compute_physics_metrics(
        object_track,
        resting_z,
        carry_mask,
        reference_final_pos=obj_pos_isaac_all[-1],
        reference_final_quat=obj_quat_isaac_all[-1],
        final_actual_pos=final_pos,
        final_actual_quat=final_quat,
        lift_threshold=float(args.lift_threshold),
        drop_threshold=float(args.drop_threshold),
    )

    phase("capturing final screenshot")
    screenshot_path = out_dir / "screenshot.png"
    capture_camera_png(app, camera, screenshot_path)

    phase("writing video")
    video_path = out_dir / "video.mp4"
    video_fps = float(args.video_fps) if args.video_fps else traj.fps
    video_ok = write_video(frame_paths, video_path, fps=int(round(video_fps)))

    phase("exporting stage")
    stage_path = out_dir / "scene.usd"
    stage.GetRootLayer().Export(str(stage_path.resolve()))

    if bool(args.hold_open):
        deadline = time.monotonic() + float(args.hold_open_seconds)
        while time.monotonic() < deadline:
            hand_driver.drive(all_pos_isaac[-1], all_quat_isaac[-1], traj.finger_targets[-1])
            if args.carry_mode == CARRY_MODE_KINEMATIC:
                set_world_pose(stage, OBJECT_WRAP, obj_pos_isaac_all[-1], obj_quat_isaac_all[-1])
            app.update()
            time.sleep(1.0 / 60.0)

    report = {
        "ok": True,
        "task": "grasp_traj_simulation",
        "trajectory_dir": str(args.trajectory_dir),
        "out_dir": str(out_dir),
        "sequence_id": sequence_id,
        "carry_mode": args.carry_mode,
        "z_offset": z_offset,
        "tabletop_z": tabletop_z,
        "time_steps_per_second": time_steps_per_second,
        "max_joint_tracking_error_rad": max(joint_error_per_step) if joint_error_per_step else None,
        "joint_tracking_error_per_step": [round(v, 4) for v in joint_error_per_step],
        "object": {k: v for k, v in object_report.items() if k != "vertices_local"},
        "table": table_report,
        "hand": hand_report,
        "hand_friction_colliders_bound": hand_friction_bound,
        "object_friction_colliders_bound": object_friction_bound,
        "camera": camera_report,
        "video": str(video_path) if video_ok else None,
        "screenshot": str(screenshot_path),
        "stage": str(stage_path),
        "num_frames_captured": len(frame_paths),
        "metrics": metrics,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"done: {json.dumps(metrics)}")
    return report


# ---------------------------------------------------------------------------
# CLI / task registration
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["webrtc", "local"], default=DEFAULT_ISAACSIM_MODE)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stream-ui", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trajectory-dir", type=Path, default=None, help="Directory containing trajectory.npz/.json (Stage A output).")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sequence-id", default=None, help="Defaults to the trajectory's own sequence_dir basename.")
    parser.add_argument("--asset-config", type=Path, default=DEFAULT_SHARPA_WAVE_RIGHT_CONFIG)
    parser.add_argument("--hand-usd", type=Path, default=None, help="Override the hand USD (defaults to the asset config's usd_path).")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    parser.add_argument("--object-mass", type=float, default=None)
    parser.add_argument("--object-density", type=float, default=700.0)
    parser.add_argument("--friction", type=float, default=2.0)
    parser.add_argument("--joint-stiffness", type=float, default=80.0)
    parser.add_argument("--joint-damping", type=float, default=20.0)
    parser.add_argument("--joint-max-force", type=float, default=300.0)
    parser.add_argument("--joint-armature", type=float, default=0.01)
    parser.add_argument("--joint-friction", type=float, default=0.05)
    parser.add_argument("--close-joint-stiffness", type=float, default=20.0, help="Softened finger drive stiffness during the close segment.")
    parser.add_argument("--close-joint-max-force", type=float, default=60.0, help="Softened finger drive effort cap during the close segment.")
    parser.add_argument("--convex-decomp-max-hulls", type=int, default=32)
    parser.add_argument("--sim-steps-per-frame", type=int, default=2, help="app.update() calls per trajectory frame; each advances sim time 1/60s, so 2 matches a 30fps trajectory in real time.")
    parser.add_argument("--time-steps-per-second", type=float, default=120.0, help="PhysX substep rate; keep a multiple of 60.")
    parser.add_argument("--capture-every", type=int, default=1)
    parser.add_argument("--settle-steps", type=int, default=60)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--carry-mode", choices=[CARRY_MODE_FRICTION, CARRY_MODE_KINEMATIC], default=CARRY_MODE_FRICTION)
    parser.add_argument("--lift-threshold", type=float, default=0.02)
    parser.add_argument("--drop-threshold", type=float, default=0.005)
    parser.add_argument("--tabletop-z", type=float, default=0.0)
    parser.add_argument("--table-margin", type=float, default=0.18)

    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--camera-eye", type=float, nargs=3, default=[0.85, -1.85, 0.85])
    parser.add_argument("--camera-target", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    parser.add_argument("--auto-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--camera-focal-length", type=float, default=45.0)
    parser.add_argument("--camera-horizontal-aperture", type=float, default=38.0)

    parser.add_argument("--hold-open", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--hold-open-seconds", type=float, default=10.0)
    parser.add_argument("--livestream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--reuse-instance", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8765)
    parser.add_argument("--request-timeout", type=float, default=0.0)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--shutdown-server", action="store_true")
    return parser


def normalize_paths(args: argparse.Namespace) -> None:
    args.trajectory_dir = resolve_artifact_path(args.trajectory_dir)
    args.manifest = resolve_artifact_path(args.manifest)
    args.out_dir = resolve_artifact_path(args.out_dir)
    args.asset_config = resolve_artifact_path(args.asset_config)
    args.hand_usd = resolve_artifact_path(args.hand_usd)
    if args.status or args.shutdown_server:
        return
    if args.trajectory_dir is None:
        raise ValueError("--trajectory-dir is required")
    if int(args.width) <= 0 or int(args.height) <= 0:
        raise ValueError("--width and --height must be positive")


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args()
    apply_mode_defaults(args, parser)
    normalize_paths(args)
    return args


def normalize_task_args(params: dict) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in params.items():
        if hasattr(args, key):
            setattr(args, key, value)
    apply_mode_defaults(args, parser)
    normalize_paths(args)
    return args


def run_simulation_task(app, args: argparse.Namespace, progress) -> dict:
    global log
    old_log = log

    def task_log(message: str) -> None:
        old_log(message)
        progress(message)

    log = task_log
    try:
        return simulate_grasp_traj(app, args, progress=progress)
    finally:
        log = old_log


def register_sim_tasks(registry) -> None:
    registry.register(
        "grasp_traj_simulation",
        run=run_simulation_task,
        normalize=normalize_task_args,
        description="Play a generated grasp trajectory back in Isaac Sim with real PhysX physics and render a video.",
    )


def main() -> int:
    args = parse_args()
    if args.status:
        try:
            print(json.dumps(request_json(args.control_host, args.control_port, "GET", "/status", timeout=1.0), indent=2), flush=True)
            return 0
        except Exception as exc:
            print(f"OCIR_GRASP_TRAJ_SIM no control server at {args.control_host}:{args.control_port}: {exc}", flush=True)
            return 1
    if args.shutdown_server:
        try:
            print(json.dumps(request_json(args.control_host, args.control_port, "POST", "/shutdown", {}, timeout=2.0), indent=2), flush=True)
            return 0
        except Exception as exc:
            print(f"OCIR_GRASP_TRAJ_SIM no control server at {args.control_host}:{args.control_port}: {exc}", flush=True)
            return 1
    return run_sim_cli_main(
        args,
        log=log,
        run_batch=simulate_grasp_traj,
        task="grasp_traj_simulation",
        log_prefix="OCIR_GRASP_TRAJ_SIM",
        restart_hint=(
            "Restart it with this task module, for example:\n"
            "  scripts/run_isaacsim_conda.sh scripts/sim/start_isaacsim_server.py "
            "--task-module scripts/isaac/simulate_grasp_traj.py"
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
