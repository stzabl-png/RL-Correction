#!/usr/bin/env python3
"""Play a generated grasp trajectory back in Isaac Sim with real PhysX physics.

The hand USD (the Articulation_Bodex tuned Sharpa asset, see
``assets/robots/hands/sharpa_wave/usd/right/bodex_reference/``) is a pure
22-DOF **floating-base articulation**: baked convexDecomposition colliders,
4mm/1mm contact/rest offsets, and BODex-tuned per-joint drive effort limits.
It is driven through the PhysX tensor API
(``isaacsim.core.prims.SingleArticulation``): every physics substep sets the
root pose (interpolated along the trajectory between frames) plus
finite-difference root velocities, while the finger joints track PD position
targets in radians (``ArticulationAction``). Driving through the articulation
view keeps the joints solved in reduced coordinates -- links physically
cannot separate -- unlike teleporting authored USD transforms, which this
PhysX version does not reliably honor for articulations. The object is a
dynamic rigid body resting on a static table (hand-table collision filtered
out by default, as in the reference validator), gravity on. Renders a video
and writes lift/grasp metrics to ``report.json``.

Reuses ``ocir.isaac.replay_dexycb``'s DexYCB camera-frame-to-Isaac-world
mapping and video/camera helpers. The physics recipe (friction 3.0/3.0
multiply on hand AND object, contact slop 0.2, convex-decomposition object
collision, hand-table collision groups) follows the proven reference
validator ``ref/sharpa_tabletop.py`` (Articulation_Bodex); the articulation
solver iteration counts come from MagicSim's floating Sharpa hand config
(``SharpaWaveFloating.py``).
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
from ocir.grasp_traj.trajectory_schema import (
    SEGMENT_CARRY,
    SEGMENT_CLOSE,
    SEGMENT_SQUEEZE,
    GraspTrajectory,
    quat_wxyz_to_matrix,
)
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

RAD_TO_DEG = 180.0 / np.pi

CARRY_MODE_FRICTION = "friction"
CARRY_MODE_KINEMATIC = "kinematic"

# Published YCB object masses.  Mesh-volume*density is a poor estimate for
# hollow containers and also overestimates the wood block in our test set.
# Unknown/non-YCB objects still use the density fallback below.
YCB_OBJECT_MASS_KG = {
    "002_master_chef_can": 0.414,
    "025_mug": 0.118,
    "036_wood_block": 0.729,
}


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


def setup_physics_scene(stage, *, time_steps_per_second: float, gravity: float = 9.81) -> None:
    from pxr import Gf, PhysxSchema, UsdPhysics

    prim = stage.GetPrimAtPath(PHYSICS_SCENE_PATH)
    if not prim.IsValid():
        prim = stage.DefinePrim(PHYSICS_SCENE_PATH, "PhysicsScene")
    scene = UsdPhysics.Scene.Define(stage, PHYSICS_SCENE_PATH)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(float(gravity))
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
    """Find and deactivate any joint that anchors the hand to the world (an
    empty ``physics:body0`` relationship). The default asset ships
    floating-base (its palm anchor was stripped from the file), so this is a
    no-op there; it remains as a safety net for ``--hand-usd`` overrides
    that still carry a world anchor. With no anchor the hand is a
    floating-base articulation whose root pose can be set through the PhysX
    tensor API every substep (see ``HandArticulationDriver``). Nothing is
    marked kinematic: a kinematic link inside a live articulation
    destabilizes the solver (links visibly separate, fingers jitter).

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
        log(f"{ref_path}: no world-anchoring joint found (asset ships floating-base)")
    return deactivated


def setup_hand_articulation_root(stage, ref_path: str, *, self_collisions: bool = True) -> str:
    """Ensure the hand has an ``ArticulationRootAPI`` and set the
    articulation-level solver iteration counts + self-collision flag there.
    Uses the asset's own API where it ships one (the tuned asset does, on
    its top-level prim); otherwise applies it to the hand's reference root.
    Returns the prim path (needed to build the ``SingleArticulation`` view
    after the timeline starts)."""

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
    articulation.CreateEnabledSelfCollisionsAttr().Set(bool(self_collisions))
    return str(root_prim.GetPath())


def high_friction_material(stage, root_path: str, mat_path: str, *, static_friction: float, dynamic_friction: float, restitution: float = 0.0, combine_mode: str = "multiply") -> dict:
    """Define (or reuse) a physics material at ``mat_path`` and bind it to
    every collision mesh under ``root_path`` (physics purpose, stronger than
    descendants). Returns a report of what it authored and bound."""

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
    bound_paths: list[str] = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI(prim).Bind(
                UsdShade.Material(mat_prim), bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
            )
            bound_paths.append(str(prim.GetPath()))
    return {
        "material_path": mat_path,
        "static_friction": float(static_friction),
        "dynamic_friction": float(dynamic_friction),
        "restitution": float(restitution),
        "friction_combine_mode": combine_mode,
        "binding_strength": "strongerThanDescendants",
        "bound_collider_count": len(bound_paths),
        "bound_collider_paths": bound_paths,
    }


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


def apply_hand_collision_offsets(stage, ref_path: str, *, contact_offset: float, rest_offset: float) -> int:
    """Override contact/rest offsets on every hand collider. The rest offset
    effectively INFLATES the collision surface: contacts come to rest with
    the surfaces separated by the sum of both bodies' rest offsets, so a 1mm
    hand rest offset acts as the '+1mm on the hand collision mesh' the
    physical gripper calibration calls for -- and keeps the fingers from
    ever entering the deep-penetration regime where the solver's
    depenetration pushes look like the object being sucked into the hand or
    fired out of it."""

    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(ref_path)
    n = 0
    for prim in Usd.PrimRange(root):
        if not (prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI)):
            continue
        if not prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
            PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision = PhysxSchema.PhysxCollisionAPI(prim)
        collision.CreateContactOffsetAttr().Set(float(contact_offset))
        collision.CreateRestOffsetAttr().Set(float(rest_offset))
        n += 1
    log(f"{ref_path}: set contactOffset={contact_offset} restOffset={rest_offset} on {n} hand colliders")
    return n


def count_hand_colliders(stage, ref_path: str) -> int:
    """The hand asset must ship its collision model baked in (the tuned
    asset: 26 collider meshes, ``convexDecomposition`` with minThickness 2mm
    / hullVertexLimit 64 / maxConvexHulls 16, and 4mm/1mm contact/rest
    offsets). Count them, failing loudly on an asset without any."""

    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(ref_path)
    count = 0
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
            count += 1
    if count == 0:
        raise RuntimeError(
            f"{ref_path}: hand USD ships no baked collision meshes; the tuned asset "
            "(bodex_reference/sharpa_right_tuned_instanceable.usd) is required"
        )
    log(f"{ref_path}: hand ships {count} baked collider meshes; using them as-is")
    return count


def setup_hand_drives(
    stage, ref_path: str, *, stiffness: float, damping: float, max_force: float, armature: float,
    joint_friction: float, effort_profile: str = "baked",
) -> tuple[int, dict[str, float]]:
    """Configure finger drives. Under the ``baked`` effort profile the
    asset's own per-joint ``maxForce`` values (the BODex-tuned limits shipped
    in the USD) are READ and left in place; ``uniform`` overwrites them all
    with the ``max_force`` scalar (the pre-tuned behavior). Returns the drive
    count and the resolved ``{joint_name: maxForce}`` map."""

    from pxr import PhysxSchema, Usd, UsdPhysics

    root = stage.GetPrimAtPath(ref_path)
    n = 0
    efforts: dict[str, float] = {}
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
            rb_api.CreateMaxAngularVelocityAttr().Set(4.0 * RAD_TO_DEG)  # attr is in deg/s
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
            (drive.GetStiffnessAttr() or drive.CreateStiffnessAttr()).Set(float(stiffness))
            (drive.GetDampingAttr() or drive.CreateDampingAttr()).Set(float(damping))
            if effort_profile == "baked":
                baked = drive.GetMaxForceAttr().Get() if drive.GetMaxForceAttr() else None
                if baked is None:
                    raise RuntimeError(
                        f"joint {prim.GetName()} has no baked drive maxForce, but --joint-effort-profile=baked; "
                        "use an asset with tuned limits (bodex_reference) or pass --joint-effort-profile uniform"
                    )
                efforts[prim.GetName()] = float(baked)
                # The tuned caps bound CONTACT forces; free-space tracking
                # (retarget/approach, where fingers swing switch -> pregrasp
                # fast) needs full authority or lagging fingers sweep through
                # space the planner assumed clear. Author the scalar here;
                # the tuned per-joint vector takes over at the first close
                # step and stays for squeeze/carry.
                (drive.GetMaxForceAttr() or drive.CreateMaxForceAttr()).Set(float(max_force))
            else:
                (drive.GetMaxForceAttr() or drive.CreateMaxForceAttr()).Set(float(max_force))
                efforts[prim.GetName()] = float(max_force)
            (drive.GetTargetPositionAttr() or drive.CreateTargetPositionAttr()).Set(0.0)
            if not prim.HasAPI(PhysxSchema.PhysxJointAPI):
                PhysxSchema.PhysxJointAPI.Apply(prim)
            physx_joint = PhysxSchema.PhysxJointAPI(prim)
            physx_joint.CreateJointFrictionAttr().Set(float(joint_friction))
            physx_joint.CreateArmatureAttr().Set(float(armature))
            physx_joint.CreateMaxJointVelocityAttr().Set(570.0)  # deg/s, ~10 rad/s
            n += 1
    log(f"{ref_path}: configured {n} finger joint drives (effort profile: {effort_profile})")
    return n, efforts


def quats_to_angular_velocities(quats_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """(T,4) wxyz -> (T,3) world-frame angular velocities by finite
    differences (angle-axis of the relative rotation over dt); last row 0."""

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


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, frac: float) -> np.ndarray:
    """Spherical linear interpolation between two wxyz quaternions along the
    shorter arc. Falls back to normalized lerp when the two are nearly
    parallel (sin(theta) -> 0)."""

    a = np.asarray(q0, dtype=np.float64)
    b = np.asarray(q1, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:  # shorter arc
        b = -b
        dot = -dot
    if dot > 0.9995:  # nearly parallel
        out = a + float(frac) * (b - a)
        return out / np.linalg.norm(out)
    theta0 = np.arccos(dot)
    sin0 = np.sin(theta0)
    s0 = np.sin((1.0 - float(frac)) * theta0) / sin0
    s1 = np.sin(float(frac) * theta0) / sin0
    return s0 * a + s1 * b


class HandArticulationDriver:
    """Drives the floating-base hand articulation through the PhysX tensor
    API: root pose + root velocities + finger PD position targets (radians).
    Must be constructed after the timeline is playing (the physics simulation
    view does not exist before that)."""

    def __init__(
        self,
        articulation_root_path: str,
        joint_order: tuple[str, ...],
        efforts_by_name: dict[str, float] | None = None,
        fallback_max_force: float = 300.0,
    ):
        from isaacsim.core.prims import SingleArticulation

        self.articulation = SingleArticulation(articulation_root_path)
        self.articulation.initialize()
        dof_names = list(self.articulation.dof_names)
        missing = [name for name in joint_order if name not in dof_names]
        if missing:
            raise RuntimeError(f"trajectory joints not present in articulation DOFs: {missing} (DOFs: {dof_names})")
        self.dof_indices = np.asarray([dof_names.index(name) for name in joint_order], dtype=np.int32)
        self.num_dofs = len(dof_names)
        # Per-DOF effort vector resolved from the drive-setup map (baked
        # profile: the asset's own tuned per-joint limits; uniform: scalars).
        efforts_by_name = efforts_by_name or {}
        unmapped = [name for name in dof_names if name not in efforts_by_name]
        if efforts_by_name and unmapped:
            log(f"WARNING: DOFs without a resolved effort limit, using fallback {fallback_max_force}: {unmapped}")
        self.effort_vector = np.asarray(
            [float(efforts_by_name.get(name, fallback_max_force)) for name in dof_names], dtype=np.float32
        )
        log(f"articulation view ready: {self.num_dofs} DOFs, driving {len(joint_order)} of them")

    def reset_joints(self, joint_positions_rad: np.ndarray) -> None:
        """Teleport joints to the given positions (start of playback, so the
        drives don't have to swing from the asset's zero pose first)."""

        self.articulation.set_joint_positions(
            np.asarray(joint_positions_rad, dtype=np.float32), joint_indices=self.dof_indices
        )

    def set_drive_strength(self, stiffness: float, damping: float, max_force: float | np.ndarray) -> None:
        """Set PD gains + effort cap on all driven finger joints at once
        (used to soften the fingers during the close segment so an early-
        touching finger stalls against the object instead of shoving it).
        ``max_force`` may be a per-DOF vector (baked tuned limits) or a
        scalar (uniform profile)."""

        controller = self.articulation.get_articulation_controller()
        kps = np.full(self.num_dofs, float(stiffness), dtype=np.float32)
        kds = np.full(self.num_dofs, float(damping), dtype=np.float32)
        controller.set_gains(kps=kps, kds=kds)
        if np.isscalar(max_force):
            efforts = np.full(self.num_dofs, float(max_force), dtype=np.float32)
        else:
            efforts = np.asarray(max_force, dtype=np.float32)
            if efforts.shape != (self.num_dofs,):
                raise ValueError(f"max_force vector shape {efforts.shape} != ({self.num_dofs},)")
        controller.set_max_efforts(efforts)

    def joint_positions(self) -> np.ndarray:
        """Return the driven finger-joint positions in trajectory order."""

        return np.asarray(
            self.articulation.get_joint_positions(joint_indices=self.dof_indices), dtype=np.float64
        )

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
        return float(np.abs(self.joint_positions() - np.asarray(joint_targets_rad, dtype=np.float64)).max())


def govern_contact_targets(
    desired_rad: np.ndarray, actual_rad: np.ndarray, max_target_lead_rad: float
) -> tuple[np.ndarray, int]:
    """Bound each finger drive target around its current physical position.

    A synthesized squeeze pose is a force direction, not a configuration the
    physics solver must reach through a rigid object.  Limiting the virtual
    target's angular lead bounds the proportional spring load independently
    for every joint: a blocked finger maintains preload while unblocked
    fingers can continue closing on later frames.
    """

    desired = np.asarray(desired_rad, dtype=np.float64)
    actual = np.asarray(actual_rad, dtype=np.float64)
    lead = float(max_target_lead_rad)
    if lead <= 0.0:
        raise ValueError(f"max_target_lead_rad must be positive, got {lead}")
    error = desired - actual
    limited = np.abs(error) > lead
    return actual + np.clip(error, -lead, lead), int(limited.sum())


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
        # convex (default): convex decomposition, the reference validator's
        # object collision (hull count capped by --convex-decomp-max-hulls;
        # deep concavities may be bridged by hulls). sdf: exact SDF
        # triangle-mesh collision -- concavities like a mug's opening/handle
        # stay hollow, at the cost of softer penetration handling.
        if args.object_collision == "sdf":
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set("sdf")
            sdf = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
            sdf.CreateSdfResolutionAttr().Set(int(args.sdf_resolution))
        else:
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
        # Contact slop: penetration shallower than this (scaled) tolerance is
        # not corrected, so the solver stops chasing sub-mm contact noise at
        # every resting finger contact (jitter and creep). Matches the
        # reference validator's 0.2. 0 restores exact-correction behavior.
        if float(args.contact_slop) > 0.0:
            rb_api.CreateContactSlopCoefficientAttr().Set(float(args.contact_slop))
        # Nothing in this scene legitimately moves faster than the capped
        # wrist (0.25 m/s) plus a drop -- a tight cap keeps a slipped grasp's
        # pinch-ejection local instead of firing the object across the room.
        rb_api.CreateMaxLinearVelocityAttr().Set(1.5)
        rb_api.CreateMaxAngularVelocityAttr().Set(4.0 * RAD_TO_DEG)
        rb_api.CreateSolverPositionIterationCountAttr().Set(16)
        rb_api.CreateSolverVelocityIterationCountAttr().Set(2)

    mesh_report["mass_kg"] = float(mass_kg)
    mesh_report["rigid_body_dynamic"] = not bool(kinematic)
    mesh_report["collider_type"] = None if kinematic else str(args.object_collision)
    mesh_report["sdf_resolution"] = int(args.sdf_resolution) if (not kinematic and args.object_collision == "sdf") else None
    mesh_report["contact_offset_m"] = 0.004 if not kinematic else None
    mesh_report["rest_offset_m"] = 0.001 if not kinematic else None
    mesh_report["contact_slop_coefficient"] = float(args.contact_slop) if (not kinematic and float(args.contact_slop) > 0.0) else None
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


def disable_hand_table_collision(stage) -> None:
    """Filter out every hand-table contact pair via UsdPhysics collision
    groups (the ref/sharpa_tabletop.py mechanism): the trajectory skims the
    tabletop during retarget/approach, and palm/finger scraping against the
    table injects contact noise without validating anything about the grasp.
    The object still collides with both."""

    from pxr import UsdGeom, UsdPhysics

    groups_root = "/World/CollisionGroups"
    if not stage.GetPrimAtPath(groups_root).IsValid():
        UsdGeom.Scope.Define(stage, groups_root)
    table_group = UsdPhysics.CollisionGroup.Define(stage, f"{groups_root}/tableGroup")
    hand_group = UsdPhysics.CollisionGroup.Define(stage, f"{groups_root}/handGroup")
    table_group.CreateFilteredGroupsRel().AddTarget(f"{groups_root}/handGroup")
    table_group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/World/Table")
    hand_group.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(HAND_WRAP)
    log("hand-table collision disabled (collision groups)")


def build_hand(stage, hand_usd_path: Path, args: argparse.Namespace) -> dict:
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Usd, UsdGeom

    UsdGeom.Xform.Define(stage, HAND_WRAP)
    add_reference_to_stage(str(hand_usd_path), HAND_REF)
    # Always de-instance: per-collider authoring (friction binding, offset
    # overrides) cannot target instance proxies, and a single hand gains
    # nothing from instancing.
    root = stage.GetPrimAtPath(HAND_REF)
    for prim in Usd.PrimRange(root):
        if prim.IsInstanceable():
            prim.SetInstanceable(False)

    deactivated_joints = deactivate_hand_root_joint(stage, HAND_REF)
    articulation_root = setup_hand_articulation_root(
        stage, HAND_REF, self_collisions=args.hand_self_collisions
    )
    collider_count = count_hand_colliders(stage, HAND_REF)
    if args.hand_rest_offset is not None:
        # Explicit override; None (the default) respects the offsets baked
        # into the asset (4mm contact / 1mm rest on the tuned hand).
        offset_count = apply_hand_collision_offsets(
            stage, HAND_REF,
            contact_offset=float(args.hand_rest_offset) + 0.004,
            rest_offset=float(args.hand_rest_offset),
        )
    else:
        offset_count = 0
        log(f"{HAND_REF}: keeping the asset's baked collider contact/rest offsets")
    drive_count, resolved_efforts = setup_hand_drives(
        stage, HAND_REF,
        stiffness=args.joint_stiffness, damping=args.joint_damping, max_force=args.joint_max_force,
        armature=args.joint_armature, joint_friction=args.joint_friction,
        effort_profile=str(args.joint_effort_profile),
    )
    return {
        "deactivated_joints": deactivated_joints,
        "articulation_root": articulation_root,
        "self_collisions": bool(args.hand_self_collisions),
        "collider_count": collider_count,
        "offset_collider_count": offset_count,
        "hand_rest_offset_m": float(args.hand_rest_offset) if args.hand_rest_offset is not None else None,
        "drive_count": drive_count,
        "joint_effort_profile": str(args.joint_effort_profile),
        "resolved_max_efforts": {name: round(value, 6) for name, value in sorted(resolved_efforts.items())},
    }


# ---------------------------------------------------------------------------
# Object pose readback (for physics metrics)
# ---------------------------------------------------------------------------


def read_object_world_pose(stage) -> tuple[np.ndarray, np.ndarray]:
    from pxr import UsdGeom

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
    final_actual_pos: np.ndarray,
    *,
    lift_threshold: float,
    drop_threshold: float,
) -> dict:
    positions = np.stack(object_track_pos, axis=0)
    carry_z = positions[: len(carry_mask)][carry_mask][:, 2] if carry_mask.any() else np.asarray([])
    lift = carry_z - float(resting_z)
    lifted_mask = lift >= float(lift_threshold)
    lifted = bool(lifted_mask.any())
    max_consecutive = 0
    current_consecutive = 0
    for value in lifted_mask:
        current_consecutive = current_consecutive + 1 if bool(value) else 0
        max_consecutive = max(max_consecutive, current_consecutive)
    sustained_lift = bool(max_consecutive >= 5)
    dropped = False
    if lifted and carry_z.size:
        peak_idx = int(np.argmax(carry_z))
        if peak_idx < carry_z.size - 1:
            dropped = bool((carry_z[-1] - resting_z) <= drop_threshold)
    pos_error = float(np.linalg.norm(final_actual_pos - reference_final_pos))
    return {
        "lifted": lifted,
        "sustained_lift": sustained_lift,
        "grasp_success": bool(sustained_lift and not dropped),
        "object_dropped": dropped,
        "resting_z": float(resting_z),
        "max_carry_z": float(carry_z.max()) if carry_z.size else None,
        "max_lift_m": float(lift.max()) if lift.size else None,
        "final_carry_lift_m": float(lift[-1]) if lift.size else None,
        "num_lifted_carry_steps": int(lifted_mask.sum()),
        "max_consecutive_lifted_steps": int(max_consecutive),
        "lifted_carry_fraction": float(lifted_mask.mean()) if lifted_mask.size else 0.0,
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
    setup_physics_scene(stage, time_steps_per_second=time_steps_per_second, gravity=float(args.gravity))

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
    if not bool(args.hand_table_collision):
        disable_hand_table_collision(stage)
    table_report["hand_table_collision"] = bool(args.hand_table_collision)

    phase("building object")
    object_name = str(traj.extra_metadata.get("object_name") or "")
    if args.object_mass is not None:
        mass_kg = float(args.object_mass)
        mass_source = "explicit_cli"
    elif object_name in YCB_OBJECT_MASS_KG:
        mass_kg = float(YCB_OBJECT_MASS_KG[object_name])
        mass_source = "published_ycb"
    else:
        mass_kg = estimate_object_mass(object_mesh_path, args.object_density)
        mass_source = "mesh_volume_times_density"
    object_report = build_object(
        stage, object_mesh_path, mass_kg=mass_kg, kinematic=(args.carry_mode == CARRY_MODE_KINEMATIC), args=args
    )
    object_report.update(
        object_name=object_name or None,
        mass_source=mass_source,
        density_fallback_kg_m3=float(args.object_density),
        mesh_watertight=bool(mesh_for_bounds.is_watertight),
        mesh_volume_m3=float(abs(mesh_for_bounds.volume)) if mesh_for_bounds.is_watertight else float(abs(mesh_for_bounds.convex_hull.volume)),
    )
    set_world_pose(stage, OBJECT_WRAP, obj_pos_isaac_all[0], obj_quat_isaac_all[0])

    phase("building hand")
    hand_usd_path = Path(args.hand_usd) if args.hand_usd else load_sharpa_wave_right(args.asset_config).usd_path
    hand_report = build_hand(stage, hand_usd_path, args)
    # Friction targets model where high friction lives. "both" (default, the
    # ref/sharpa_tabletop.py setup): ONE material at --friction bound to
    # every hand collider AND the object, combine mode "multiply", so the
    # hand-object pair sees friction*friction (3.0 -> an effective 9.0
    # supergrip) while object-table sees friction*0.5. "object":
    # Articulation_Bodex open_by_handle-style -- the material goes on the
    # OBJECT only and every pair the object touches (table included) sees
    # --friction.
    if str(args.friction_target) == "both":
        hand_friction_report = high_friction_material(
            stage, HAND_REF, "/World/Materials/SuperGrip",
            static_friction=args.friction, dynamic_friction=args.friction,
            combine_mode=str(args.friction_combine_mode),
        )
        object_friction_report = high_friction_material(
            stage, OBJECT_REF, "/World/Materials/SuperGrip",
            static_friction=args.friction, dynamic_friction=args.friction,
            combine_mode=str(args.friction_combine_mode),
        )
        if hand_friction_report["bound_collider_count"] != 26:
            log(
                f"WARNING: expected 26 hand colliders, bound "
                f"{hand_friction_report['bound_collider_count']}"
            )
    else:
        hand_friction_report = {
            "material_path": None,
            "bound_collider_count": 0,
            "note": "hand keeps its ordinary asset material; object material combine mode sets the pair friction",
        }
        object_friction_report = high_friction_material(
            stage, OBJECT_REF, "/World/Materials/ObjectFriction",
            static_friction=args.friction, dynamic_friction=args.friction,
            combine_mode=str(args.friction_combine_mode),
        )
    log(
        f"friction target={args.friction_target}: hand colliders bound={hand_friction_report['bound_collider_count']}, "
        f"object colliders bound={object_friction_report['bound_collider_count']}"
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
    hand_driver = HandArticulationDriver(
        hand_report["articulation_root"], traj.joint_order,
        efforts_by_name=hand_report["resolved_max_efforts"],
        fallback_max_force=float(args.joint_max_force),
    )
    hand_driver.reset_joints(traj.finger_targets[0])
    # Segment-dependent effort caps: under the baked profile the asset's
    # tuned per-joint limits apply from the first CLOSE step onward (close,
    # squeeze, carry -- every segment where finger-object contact is
    # intended), never expanded back to the uniform scalars. The free-space
    # retarget/approach segments keep the scalar --joint-max-force authority
    # (authored in setup_hand_drives): with the tuned caps active there, the
    # fast switch->pregrasp finger swing lags by >1 rad and the still-closed
    # fingers sweep into the object (mug sequence: 6.6m ejection during
    # approach).
    baked_profile = str(args.joint_effort_profile) == "baked"
    close_max_force = hand_driver.effort_vector if baked_profile else float(args.close_joint_max_force)
    normal_max_force = hand_driver.effort_vector if baked_profile else float(args.joint_max_force)

    # Finite-difference root velocities: without them every set_world_pose is
    # a zero-velocity teleport and finger-object contacts never see the
    # wrist's true motion (friction can't carry the object). Each app.update()
    # advances sim time by exactly 1/60 s, so the dt these velocities must be
    # consistent with is the SIMULATED time per trajectory frame -- not
    # traj.dt -- or the root overshoots its own teleports every frame and
    # pumps energy into the articulation. The per-substep pose is INTERPOLATED
    # from pos[t] toward pos[t+1] (see the inner loop) so the teleport advances
    # along the path in step with this velocity instead of snapping back to a
    # fixed pos[t] each substep (which showed up as a ~60Hz vertical shake in
    # the live view and injected a velocity kick into every live contact).
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
    drive_target_error_per_step: list[float] = []
    governed_joint_count_per_step: list[int] = []
    actual_joint_positions_per_step: list[np.ndarray] = []
    driven_targets_per_step: list[np.ndarray] = []
    blowup_logged = False

    # Softened finger drives during the close segment: the close targets are
    # contact-projected but sphere-vs-hull geometry differences still leave a
    # few mm of commanded overlap, and full-strength drives turn that into a
    # shove. Full gains are restored for squeeze/carry.
    close_gains_active = None  # None until the first close step; then True/False
    # Close must retain its full (soft-gain) trajectory target so the fingers
    # can traverse free space and actually reach the contact posture. The
    # bounded virtual spring begins only once the trajectory enters squeeze,
    # then remains active through carry.
    governed_segments = {SEGMENT_SQUEEZE, SEGMENT_CARRY}
    target_lead_rad = float(args.contact_target_lead_rad)
    if bool(args.contact_aware_finger_targets) and target_lead_rad <= 0.0:
        raise ValueError(f"--contact-target-lead-rad must be positive, got {target_lead_rad}")
    phase(f"simulating {traj.num_steps} steps")
    for t in range(traj.num_steps):
        in_close = bool(traj.segment[t] == SEGMENT_CLOSE)
        if in_close and close_gains_active is not True:
            hand_driver.set_drive_strength(
                float(args.close_joint_stiffness), float(args.joint_damping), close_max_force
            )
            close_gains_active = True
        elif not in_close and close_gains_active is True:
            hand_driver.set_drive_strength(
                float(args.joint_stiffness), float(args.joint_damping), normal_max_force
            )
            close_gains_active = False
        desired_targets = traj.finger_targets[t]
        driven_targets = desired_targets
        governed_count = 0
        if args.carry_mode == CARRY_MODE_KINEMATIC:
            set_world_pose(stage, OBJECT_WRAP, obj_pos_isaac_all[t], obj_quat_isaac_all[t])
        # Advance the commanded root pose ACROSS the substeps (pos[t] ->
        # pos[t+1]) rather than re-teleporting to pos[t] every substep. The
        # substep displacement (Delta/N) then matches what lin_vel[t] would
        # carry over one substep, so the teleport re-anchors against gravity
        # sag without ever snapping backwards. Last frame holds its own pose.
        t_next = min(t + 1, traj.num_steps - 1)
        pos0, pos1 = all_pos_isaac[t], all_pos_isaac[t_next]
        quat0, quat1 = all_quat_isaac[t], all_quat_isaac[t_next]
        n_sub = int(args.sim_steps_per_frame)
        for k in range(n_sub):
            frac = (k + 1) / n_sub
            sub_pos = pos0 + (pos1 - pos0) * frac
            sub_quat = slerp_wxyz(quat0, quat1, frac)
            # Recompute the bounded virtual target at physics cadence. A
            # target limited only once per trajectory frame can become far
            # from a fast-moving joint after the first substep and inject the
            # same large spring impulse the governor is intended to prevent.
            if bool(args.contact_aware_finger_targets) and int(traj.segment[t]) in governed_segments:
                driven_targets, substep_governed_count = govern_contact_targets(
                    desired_targets, hand_driver.joint_positions(), target_lead_rad
                )
                governed_count = max(governed_count, substep_governed_count)
            hand_driver.drive(
                sub_pos, sub_quat, driven_targets,
                linear_velocity=lin_vel[t], angular_velocity=ang_vel[t],
            )
            app.update()
        step_error = hand_driver.max_joint_tracking_error(desired_targets)
        drive_step_error = hand_driver.max_joint_tracking_error(driven_targets)
        joint_error_per_step.append(step_error)
        drive_target_error_per_step.append(drive_step_error)
        governed_joint_count_per_step.append(governed_count)
        actual_joint_positions_per_step.append(hand_driver.joint_positions())
        driven_targets_per_step.append(np.asarray(driven_targets, dtype=np.float64).copy())
        if drive_step_error > 1.0 and not blowup_logged:
            blowup_logged = True
            log(f"WARNING: drive-target error {drive_step_error:.2f} rad at step {t} (segment {int(traj.segment[t])}) -- articulation destabilizing")
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
        settle_targets = traj.finger_targets[-1]
        if bool(args.contact_aware_finger_targets):
            settle_targets, _ = govern_contact_targets(
                settle_targets, hand_driver.joint_positions(), target_lead_rad
            )
        hand_driver.drive(all_pos_isaac[-1], all_quat_isaac[-1], settle_targets)
        if args.carry_mode == CARRY_MODE_KINEMATIC:
            set_world_pose(stage, OBJECT_WRAP, obj_pos_isaac_all[-1], obj_quat_isaac_all[-1])
        app.update()
        pos, _ = read_object_world_pose(stage)
        object_track.append(pos)

    final_pos, _ = read_object_world_pose(stage)
    metrics = compute_physics_metrics(
        object_track,
        resting_z,
        carry_mask,
        reference_final_pos=obj_pos_isaac_all[-1],
        final_actual_pos=final_pos,
        lift_threshold=float(args.lift_threshold),
        drop_threshold=float(args.drop_threshold),
    )
    np.savez_compressed(
        out_dir / "object_track.npz",
        position_world=np.asarray(object_track, dtype=np.float64),
        reference_position_world=np.asarray(obj_pos_isaac_all, dtype=np.float64),
        segment=np.asarray(traj.segment, dtype=np.int8),
        carry_mask=np.asarray(carry_mask, dtype=bool),
    )
    np.savez_compressed(
        out_dir / "finger_track.npz",
        desired_position_rad=np.asarray(traj.finger_targets, dtype=np.float64),
        driven_target_rad=np.asarray(driven_targets_per_step, dtype=np.float64),
        actual_position_rad=np.asarray(actual_joint_positions_per_step, dtype=np.float64),
        joint_order=np.asarray(traj.joint_order),
        segment=np.asarray(traj.segment, dtype=np.int8),
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
            hold_targets = traj.finger_targets[-1]
            if bool(args.contact_aware_finger_targets):
                hold_targets, _ = govern_contact_targets(
                    hold_targets, hand_driver.joint_positions(), target_lead_rad
                )
            hand_driver.drive(all_pos_isaac[-1], all_quat_isaac[-1], hold_targets)
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
        "contact_aware_finger_targets": bool(args.contact_aware_finger_targets),
        "contact_target_lead_rad": target_lead_rad,
        "max_drive_target_error_rad": max(drive_target_error_per_step) if drive_target_error_per_step else None,
        "max_contact_drive_target_error_rad": max(
            (error for error, segment in zip(drive_target_error_per_step, traj.segment) if int(segment) in governed_segments),
            default=None,
        ),
        "drive_target_error_per_step": [round(v, 4) for v in drive_target_error_per_step],
        "governed_joint_count_per_step": governed_joint_count_per_step,
        "num_governed_steps": int(sum(count > 0 for count in governed_joint_count_per_step)),
        "object": {k: v for k, v in object_report.items() if k != "vertices_local"},
        "table": table_report,
        "hand": hand_report,
        "hand_friction": hand_friction_report,
        "object_friction": object_friction_report,
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
    parser.add_argument("--friction-target", choices=["both", "object"], default="both", help="both (ref/sharpa_tabletop.py setup): bind ONE material at --friction to all 26 hand colliders AND the object, so the hand-object pair multiplies to friction^2. object: the object only, at --friction (Articulation_Bodex open_by_handle-style; every pair the object touches, table included, sees it).")
    parser.add_argument("--friction", type=float, default=3.0, help="Static+dynamic friction of the bound material.")
    parser.add_argument("--friction-combine-mode", choices=["max", "multiply", "average", "min"], default="multiply", help="PhysX friction combine mode of the bound material; multiply/max both outrank the default material's 'average', so the bound side wins any pair against an unbound collider.")
    parser.add_argument("--joint-stiffness", type=float, default=80.0)
    parser.add_argument("--joint-damping", type=float, default=20.0)
    parser.add_argument("--joint-effort-profile", choices=["baked", "uniform"], default="baked", help="baked: use the asset's own per-joint drive maxForce limits (the BODex-tuned values shipped in the hand USD) in every segment; uniform: overwrite all joints with the --joint-max-force / --close-joint-max-force scalars (pre-tuned behavior).")
    parser.add_argument("--joint-max-force", type=float, default=300.0, help="Uniform-profile effort cap outside close (also the fallback for DOFs missing a baked limit).")
    parser.add_argument("--joint-armature", type=float, default=0.01)
    parser.add_argument("--joint-friction", type=float, default=0.05)
    parser.add_argument("--close-joint-stiffness", type=float, default=20.0, help="Softened finger drive stiffness during the close segment.")
    parser.add_argument("--close-joint-max-force", type=float, default=60.0, help="Softened finger drive effort cap during the close segment (uniform profile only; baked keeps the tuned per-joint limits).")
    parser.add_argument("--contact-aware-finger-targets", action=argparse.BooleanOptionalAction, default=False, help="Bound finger position targets around the actual joints during squeeze and carry so blocked fingers apply finite impedance instead of holding the synthesized angle through the object. Default off: the baked per-joint effort caps already bound contact forces, and direct targeting of the synthesized squeeze pose lets every joint hold its full tuned authority.")
    parser.add_argument("--contact-target-lead-rad", type=float, default=0.03, help="Maximum per-joint angular lead of a contact-phase drive target beyond the current physical joint position.")
    parser.add_argument("--convex-decomp-max-hulls", type=int, default=32, help="Hull ceiling for the object's convex decomposition (the hand's colliders are baked into its asset).")
    parser.add_argument("--object-collision", choices=["sdf", "convex"], default="convex", help="Object collider type: convex decomposition (default, matches the reference validators) or exact SDF triangle mesh (concavities stay hollow).")
    parser.add_argument("--sdf-resolution", type=int, default=256)
    parser.add_argument("--hand-rest-offset", type=float, default=None, help="Explicit rest offset (m) override for every hand collider (contact offset becomes this + 4mm). Default: keep the offsets baked into the asset (4mm/1mm on the tuned hand).")
    parser.add_argument("--hand-self-collisions", action=argparse.BooleanOptionalAction, default=True, help="PhysX self-collision between the hand's own links (PhysxArticulationAPI enabledSelfCollisions). Default on. Disable if squeeze/carry postures cause solver instability from expected finger-finger interpenetration at the closed grasp.")
    parser.add_argument("--hand-table-collision", action=argparse.BooleanOptionalAction, default=False, help="Hand-table contact pairs. Default off (collision-group filtered, as in ref/sharpa_tabletop.py): the trajectory may skim the tabletop and hand-table scraping only injects contact noise. The object always collides with both.")
    parser.add_argument("--sim-steps-per-frame", type=int, default=2, help="app.update() calls per trajectory frame; each advances sim time 1/60s, so 2 matches a 30fps trajectory in real time.")
    parser.add_argument("--time-steps-per-second", type=float, default=120.0, help="PhysX substep rate; keep a multiple of 60.")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity magnitude (m/s^2), -z. Default 9.81 (realistic weight); ref/sharpa_tabletop.py uses 30 as a ~3g stress load.")
    parser.add_argument("--contact-slop", type=float, default=0.2, help="Object PhysX contactSlopCoefficient: penetration below this (scaled) tolerance is left uncorrected, damping resting-contact jitter/creep. Default 0.2 matches ref/sharpa_tabletop.py; 0 disables.")
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
