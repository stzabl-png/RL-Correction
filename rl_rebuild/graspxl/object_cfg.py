# GraspXL object spawning for the rebuild — ports the custom env's bare-mesh rigid-body authoring.
# The SHARPA object.usd files are visual meshes with NO RigidBodyAPI; this applies rigid+collision+mass
# at spawn (so a GraspXL object can be used as the in-hand object in the SharpaWave rebuild env).
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from .sharpa_dataset import DEFAULT_SHARPA_DATASET_ROOT, load_sharpa_grasp

OBJECT_MASS = 0.10  # kg


@configclass
class _GraspXLUsdFileCfg(sim_utils.UsdFileCfg):
    """UsdFileCfg + an OPTIONAL physics material (UsdFileCfg itself has no physics_material field).

    Default None -> the custom spawner binds NO material (baseline: object uses the sim default material).
    Supply a RigidBodyMaterialCfg (e.g. friction_combine_mode="max") to scope a material to one task.
    It is a declared configclass field, so it survives the Hydra to_dict/from_dict round-trip (same as the
    visual_material field)."""
    physics_material: sim_utils.RigidBodyMaterialCfg | None = None


def object_usd_for(object_id: str, dataset_root=DEFAULT_SHARPA_DATASET_ROOT, pose_index: int = 0) -> str:
    grasp = load_sharpa_grasp(dataset_root=dataset_root, object_id=object_id, pose_index=pose_index)
    return grasp.object_usd_path


def _iter_mesh_prims(root_prim):
    from pxr import Usd, UsdGeom
    return [p for p in Usd.PrimRange(root_prim) if p.IsA(UsdGeom.Mesh)]


@clone
def spawn_graspxl_object_from_usd(prim_path, cfg, translation=None, orientation=None,
                                  collision_approximation: str = "convexDecomposition", **kwargs):
    from pxr import UsdPhysics
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        # `isaaclab.sim.create_prim` was added in IsaacLab 2.3.1; on 2.3.0 it is absent. Prefer it
        # (2.3.2 is the tested version) and fall back to the isaacsim core util, which authors into
        # the current USD stage (== `stage` here). It has no `stage=` kwarg, hence the branch.
        if hasattr(sim_utils, "create_prim"):
            prim = sim_utils.create_prim(prim_path, usd_path=cfg.usd_path, translation=translation,
                                         orientation=orientation, scale=cfg.scale, stage=stage)
        else:
            from isaacsim.core.utils.prims import create_prim as _isaacsim_create_prim
            prim = _isaacsim_create_prim(prim_path, usd_path=cfg.usd_path, translation=translation,
                                         orientation=orientation, scale=cfg.scale)
    if cfg.rigid_props is not None:
        sim_utils.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)
    if cfg.collision_props is not None:
        mesh_prims = _iter_mesh_prims(prim)
        for mp in mesh_prims:
            api = UsdPhysics.MeshCollisionAPI.Apply(mp)
            api.GetApproximationAttr().Set(collision_approximation)
            sim_utils.define_collision_properties(str(mp.GetPath()), cfg.collision_props, stage=stage)
        if not mesh_prims:
            sim_utils.define_collision_properties(prim_path, cfg.collision_props, stage=stage)
    if cfg.mass_props is not None:
        sim_utils.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    # OPTIONAL physics material (friction/restitution + COMBINE MODE). Default None -> bind nothing
    # (baseline behavior: object uses the sim default material). When a RigidBodyMaterialCfg is supplied
    # (e.g. friction_combine_mode="max" for the 3B adjust-rotate task), author it and bind it to the object
    # so object<->finger contact uses the higher-priority combine mode (PhysX: max > multiply > min > avg).
    # The runtime set_friction DR overwrites static/dynamic coefficients but NOT the combine mode, so it
    # persists. bind_physics_material is apply_nested -> binds to the collision child meshes.
    phys_mat = getattr(cfg, "physics_material", None)
    if phys_mat is not None:
        mat_path = f"{prim_path}/physicsMaterial"
        phys_mat.func(mat_path, phys_mat)
        sim_utils.bind_physics_material(prim_path, mat_path, stage=stage)
        print(f"[GraspXL] physics material bound at {mat_path}: "
              f"friction_combine={getattr(phys_mat, 'friction_combine_mode', '?')} "
              f"restitution_combine={getattr(phys_mat, 'restitution_combine_mode', '?')} "
              f"mu_s={getattr(phys_mat, 'static_friction', '?')} mu_d={getattr(phys_mat, 'dynamic_friction', '?')} "
              f"(object<->finger uses max(mu))")
    if getattr(cfg, "activate_contact_sensors", False):
        from isaaclab.sim.schemas import activate_contact_sensors as _activate_contact
        _activate_contact(prim_path, 0.0, stage=stage)
    if cfg.visual_material is not None:
        mat_path = (cfg.visual_material_path if cfg.visual_material_path.startswith("/")
                    else f"{prim_path}/{cfg.visual_material_path}")
        cfg.visual_material.func(mat_path, cfg.visual_material)
        sim_utils.bind_visual_material(prim_path, mat_path, stage=stage)
    return prim


def _object_spawn_cfg(object_usd: str, mass: float = OBJECT_MASS, activate_contact_sensors: bool = True,
                      physics_material=None):
    return _GraspXLUsdFileCfg(
        func=spawn_graspxl_object_from_usd,
        usd_path=object_usd,
        scale=(1.0, 1.0, 1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.42, 0.12), roughness=0.6),
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, kinematic_enabled=False,
            linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        # default None -> custom spawner binds NO material (baseline). UsdFileCfg's own default is a
        # non-None RigidBodyMaterialCfg(), so we force None here to keep current behavior unless requested.
        physics_material=physics_material,
    )


def make_graspxl_multi_object_cfg(object_usds: list, prim_path: str = "/World/envs/env_.*/object",
                                  mass: float = OBJECT_MASS) -> RigidObjectCfg:
    """Per-env different GraspXL objects (MultiAssetSpawnerCfg, random_choice=False -> env_i gets
    object i%len). REQUIRES scene.replicate_physics=False."""
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=[_object_spawn_cfg(u, mass) for u in object_usds],
                                             random_choice=False),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_graspxl_object_cfg(object_usd: str, prim_path: str = "/World/envs/env_.*/object",
                            mass: float = OBJECT_MASS, activate_contact_sensors: bool = True,
                            physics_material=None) -> RigidObjectCfg:
    spawn = _GraspXLUsdFileCfg(
        func=spawn_graspxl_object_from_usd,
        usd_path=object_usd,
        scale=(1.0, 1.0, 1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.90, 0.42, 0.12), roughness=0.6),
        activate_contact_sensors=activate_contact_sensors,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, kinematic_enabled=False,
            linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=mass),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        # default None -> bind NO material (baseline). Pass a RigidBodyMaterialCfg (e.g. combine="max") to scope it.
        physics_material=physics_material,
    )
    return RigidObjectCfg(
        prim_path=prim_path, spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
