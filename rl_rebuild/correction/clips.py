"""Clip 注册表 — 训练/评估脚本用 --clip <名字> 选数据源 (B0 配置泛化).

新增 clip: 在 CLIPS 里加一条; env/train/record/m0_replay 全部自动可用.
"""
from __future__ import annotations

import os

from rl_rebuild.correction.load_replay import (CLIP11, CLIP11_MESH,
                                               CLIP11_SEMANTICS, load)
from rl_rebuild.correction import paths
from rl_rebuild.correction.producers.ocir import load_ocir
from rl_rebuild.correction.schema import DataUnit, ObjectSemantics

_OCIR = os.path.join(paths.OCIR_ROOT, "data", "testing")
_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/ocir_cache"))
# 自包含训练数据根 (stage_training_data.py 归置的 数据集/物体 布局)
_TD = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../TrainingData"))


def _td_ocir(dataset, obj, grasp_file="grasp_pose_6.json", lift_target=0.08):
    """TrainingData/<dataset>/<obj>/ 下的 OCIR 五源 -> clip 条目.
    seq/grasp/traj 读 TrainingData; usd 与 human retarget 缓存落在该物体 cache/.
    grasp_file: 抓姿合成成功时用 grasp_pose_N.json; 失败时退到 failed_grasp_000.json
    (anchor 骨干只用 cuRobo 轨迹, grasp_json 仅供接触角色/信息)."""
    base = f"{_TD}/{dataset}/{obj}"
    return dict(
        source="ocir", variant="human",
        seq_dir=f"{base}/ocir_sequence",
        grasp_json=f"{base}/grasp_pose/{grasp_file}",
        traj_dir=f"{base}/curobo_traj",
        mesh=f"{base}/ocir_sequence/object.obj",
        usd=f"{base}/cache/object.usd",          # MeshConverter 产物 (物理已烘焙)
        cache_dir=f"{base}/cache",               # human retarget 缓存 (按物体隔离)
        # 摩擦=验证器 SuperGrip 真值 (3.0, combine=multiply 在 env 里绑定)
        semantics=ObjectSemantics(label=f"{dataset}/{obj}", mass_kg=0.2, friction=3.0),
        runtime_object_physics=False,
        lift_target=lift_target,   # pp0 演示抬升 9.1cm, 10cm 判据过严
    )


# =============================================================================
# 设定 A — RL 学习 **GraspPose 能处理**的物体 (普遍偏大, 可整手包络)
#   骨干 = cuRobo 规划的 close 轨迹, RL 只做残差修正. 详见 docs/TRAINING_SETUPS_A_B.md
# =============================================================================
CLIPS = {
    "clip11": dict(
        source="bi_v2ap",
        npz=f"{CLIP11}/replay_world.npz", mesh=CLIP11_MESH,
        usd=f"{CLIP11}/object.usd", semantics=CLIP11_SEMANTICS,
        runtime_object_physics=True,      # object.usd 无物理, 运行时贴
    ),
    "pp0_human": _td_ocir("egodex", "pp0"),
    "pp55_human": _td_ocir("egodex", "pp55", grasp_file="failed_grasp_002.json"),
}
CLIPS["pp0_anchor"] = dict(CLIPS["pp0_human"], variant="anchor")
CLIPS["pp55_anchor"] = dict(CLIPS["pp55_human"], variant="anchor")

# =============================================================================
# 设定 B — RL 学习 **GraspPose 处理不了**的物体 (普遍偏小/偏扁, 只能指尖捏取)
#   BODex 包络与桌面几何互斥 -> 抓不出可用抓姿. 不给 GraspPose / 不给 cuRobo,
#   只给可信的 PreGrasp(重建腕位姿) + affordance 逐点热图, 让 RL 自己学指尖抓取.
#   详见 docs/TRAINING_SETUPS_A_B.md
# =============================================================================
_BI = paths.RR_OUTPUT


_AFF = os.path.join(paths.AFFORDANCE_ROOT, "outputs", "pred_egodex_part2_all20")


def _replay_grasp(part, name, aff_obj, mass_kg=0.1, friction=0.5):
    rt = f"{_BI}/RetargetOutput/egodex/{part}/{name}"
    rc = f"{_BI}/ReconstructOutput/egodex/{part}/{name}"
    return dict(
        source="replay_grasp",
        npz=f"{rt}/replay_world.npz",
        mesh=f"{rc}/object_mesh_scaled_final.obj",
        usd=f"{rt}/object.usd",                 # retarget 视觉网格, 运行时贴物理
        affordance=f"{_AFF}/{aff_obj}/affordance.npz",   # 逐点热图 (物体系)
        runtime_object_physics=True,
        semantics=ObjectSemantics(label=f"{part}/{name}", mass_kg=mass_kg, friction=friction),
    )


# 物体2 = basic_pick_place/2 (7.7x2.6cm 甜甜圈, BODex 抓不出)
CLIPS["Grasp2"] = _replay_grasp("part2", "basic_pick_place/2", "obj_02")


def clip_entry(name: str) -> dict:
    if name not in CLIPS:
        raise KeyError(f"未知 clip '{name}', 可选: {list(CLIPS)}")
    return CLIPS[name]


def configure_cfg(cfg, name: str):
    """把 clip 的资产路径写进 env cfg (在 env 构建之前调用)."""
    e = clip_entry(name)
    cfg.clip_name = name
    cfg.object_cfg.spawn.usd_path = e["usd"]
    return cfg


def ensure_object_usd(name: str):
    """ocir 源: object.obj -> 物理烘焙 USD (缓存). 需 Kit 已启动 (env _setup_scene 内调)."""
    e = clip_entry(name)
    if e["source"] != "ocir":
        return e["usd"]
    usd = e["usd"]
    if os.path.exists(usd) and os.path.getmtime(usd) >= os.path.getmtime(e["mesh"]):
        return usd
    import isaaclab.sim as sim_utils
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas.schemas_cfg import ConvexDecompositionPropertiesCfg
    os.makedirs(os.path.dirname(usd), exist_ok=True)
    sem = e["semantics"]
    MeshConverter(MeshConverterCfg(
        asset_path=e["mesh"],
        usd_dir=os.path.dirname(usd),
        usd_file_name=os.path.basename(usd),
        force_usd_conversion=True,
        mesh_collision_props=ConvexDecompositionPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=sem.mass_kg),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            max_depenetration_velocity=1000.0,
            sleep_threshold=0.005, stabilization_threshold=0.0025),
    ))
    # MeshConverter 把碰撞 mesh 包成 instanceable prim, 运行时 bind_physics_material
    # 会命中 'instanced prim' 而绑不进 SuperGrip. 关掉 instanceable, 使 _setup_scene
    # 的逐-env 材质绑定生效 (物体小, 非实例化的显存代价可忽略).
    from pxr import Usd
    stage = Usd.Stage.Open(usd)
    n_off = 0
    for prim in stage.Traverse():
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
            n_off += 1
    stage.GetRootLayer().Save()
    print(f"[clips] object.obj -> {usd} (convexDecomposition, mass={sem.mass_kg}kg, "
          f"instanceable off ×{n_off})")
    return usd


def load_data_unit(cfg) -> DataUnit:
    e = clip_entry(cfg.clip_name)
    if e["source"] == "replay_grasp":
        from rl_rebuild.correction.producers.replay_grasp import load_replay_grasp
        return load_replay_grasp(e["npz"], e["mesh"], usd_path=e["usd"],
                                 clip_id=cfg.clip_name, target_hz=cfg.target_hz,
                                 table_height=cfg.table_top_z, affordance_npz=e.get("affordance"),
                                 semantics=e["semantics"], verbose=True)
    if e["source"] == "bi_v2ap":
        return load(e["npz"], e["mesh"], usd_path=e["usd"],
                    clip_id=cfg.clip_name, target_hz=cfg.target_hz,
                    semantics=e["semantics"], obj_gap=0.002)
    return load_ocir(e["seq_dir"], e["grasp_json"], e["traj_dir"],
                     usd_path=e["usd"], variant=e["variant"],
                     target_hz=cfg.target_hz, table_height=cfg.table_top_z,
                     cache_dir=e.get("cache_dir"),
                     semantics=e["semantics"], verbose=True)
