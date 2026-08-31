"""Clip 注册表 — 训练/评估脚本用 --clip <名字> 选数据源 (B0 配置泛化).

新增 clip: 在 CLIPS 里加一条; env/train/record/m0_replay 全部自动可用.
"""
from __future__ import annotations

import json
import os

from rl_rebuild.correction.load_replay import (CLIP11, CLIP11_MESH,
                                               CLIP11_SEMANTICS, load)
from rl_rebuild.correction import paths
from rl_rebuild.correction.ref_builders.ocir import load_ocir
from rl_rebuild.correction.schema import DataUnit, ObjectSemantics

_OCIR = os.path.join(paths.OCIR_ROOT, "data", "testing")
_CACHE = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/ocir_cache"))
# 自包含训练数据根 (stage_training_data.py 归置的 数据集/物体 布局)
_TD = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../TrainingData"))
_DATASETS = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets"))


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


def _td_static(dataset, obj, mass_kg=None, friction=None):
    """TrainingData entry for a static SAM3D/FoundationPose reconstruction."""

    base = f"{_TD}/{dataset}/{obj}"
    meta_path = os.path.join(base, "meta.json")
    metadata = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    mass_kg = float(metadata.get("mass_kg", 0.2) if mass_kg is None else mass_kg)
    friction = float(metadata.get("friction", 0.5) if friction is None else friction)
    return dict(
        source="static_reconstruction",
        npz=f"{base}/retarget/replay_world.npz",
        mesh=f"{base}/reconstruction/object_mesh_scaled_final.obj",
        usd=f"{base}/cache/object.usd",
        runtime_object_physics=False,
        place_mode="object_only",
        semantics=ObjectSemantics(
            label=f"{dataset}/{obj}", mass_kg=mass_kg, friction=friction),
    )


def _water_bottle_static(screw_mode: str | None = None):
    """Bundled two-part PCO-1810 bottle scene used for reconstruction QA."""

    base = os.path.join(_DATASETS, "recon_kailang", "water_bottle_twist_static")
    entry = dict(
        source="static_reconstruction",
        npz=os.path.join(base, "retarget", "replay_world.npz"),
        mesh=os.path.join(base, "reconstruction", "bottle_body.obj"),
        usd=os.path.join(base, "cache", "bottle_body.usd"),
        runtime_object_physics=False,
        override_cfg_mass=True,
        flatten_converted_usd=True,
        place_mode="object_only",
        hand="left",
        placement_frame=31,
        semantics=ObjectSemantics(
            label="PCO-1810 filled bottle body", mass_kg=0.53, friction=0.5),
        secondary=dict(
            label="PCO-1810 cap",
            mesh=os.path.join(base, "reconstruction", "bottle_cap.obj"),
            usd=os.path.join(base, "cache", "bottle_cap.usd"),
            hand="right",
            placement_frame=13,
            semantics=ObjectSemantics(
                label="PCO-1810 cap", mass_kg=0.003, friction=0.4),
        ),
    )
    if screw_mode is not None:
        entry["secondary"]["assembly"] = dict(
            pitch_m=0.00318,
            turns=2.0,
            closed_offset_m=0.180,
            direction=1,
            mode=screw_mode,
            capture_radial_m=0.003,
            capture_axial_m=0.003,
            capture_tilt_deg=10.0,
            capture_yaw_deg=30.0,
            max_angular_velocity_rad_s=20.0,
        )
    return entry


def _screw_unscrew_cap1(screw_mode: str | None = None):
    """EgoDex screw_unscrew_bottle_cap/1 快照 (左手持瓶身, 右手拧盖放桌).

    CAD 与 water_bottle_twist_static 的两个 mesh **字节相同** (sha256 已核对),
    故螺纹参数 (pitch/turns/closed_offset/捕获阈值) 与质量/摩擦假设整段复用
    (本快照 meta.json 未提供实测质量). 摆放帧依据 replay 相位:
      body: 左手从帧 0 起持续接触 (phase_left[0-67]=1) 且 valid → 0;
      cap : 帧 35 起被右手拧, 帧 81 是留在桌面的最后接触帧 (phase_right[78-81]=1)
            → 取 81 (视频里盖子**终点**在桌上, 与水瓶 clip 的"起点在桌上"相反).
            实测右腕 [35..82] XY 几乎不动, 与 body 摆放点距 29.4cm, 无重叠.
    ⚠ replay fps 元数据=15 而源视频 30 帧率 (meta.json retarget_clock_status
      =unresolved); 静态摆放/螺纹审计不受影响, 但接时间相关奖励前必须先定口径.
    ⚠ 盖子绕螺轴的视觉转角不可作圈数真值 (README §limitations 1)."""
    base = os.path.join(_DATASETS, "recon_kailang", "screw_unscrew_bottle_cap_1")
    entry = dict(
        source="static_reconstruction",
        npz=os.path.join(base, "retarget", "replay_world.npz"),
        mesh=os.path.join(base, "reconstruction", "objects", "object_0",
                          "object_mesh_scaled_final.obj"),
        usd=os.path.join(base, "cache", "bottle_body.usd"),
        runtime_object_physics=False,
        override_cfg_mass=True,
        flatten_converted_usd=True,
        place_mode="object_only",
        hand="left",
        placement_frame=0,
        # 视频里瓶身全程在左手里, 没有任何桌面静置帧: FoundationPose 倾角 ≥19.9°,
        # 超过平底圆柱的倾倒阈值 (~18.3°), 保留原样会在 settle 阶段倒瓶 (实测倒下
        # 滚出 10cm). 只保留 yaw, 竖直落桌.
        upright=True,
        semantics=ObjectSemantics(
            label="PCO-1810 filled bottle body", mass_kg=0.53, friction=0.5),
        secondary=dict(
            label="PCO-1810 cap",
            mesh=os.path.join(base, "reconstruction", "objects", "object_1",
                              "object_mesh_scaled_final.obj"),
            usd=os.path.join(base, "cache", "bottle_cap.usd"),
            hand="right",
            placement_frame=81,
            semantics=ObjectSemantics(
                label="PCO-1810 cap", mass_kg=0.003, friction=0.4),
        ),
    )
    if screw_mode is not None:
        entry["secondary"]["assembly"] = dict(
            pitch_m=0.00318,
            # U30b (2026-08-27 用户裁定+实测): 不硬性要求 720°. 演示实测盖相对
            # 瓶的在瓶拧转 ≈ -266° 后分离 (f0-54 单调累积, f54-60 轴向 21→6cm
            # 脱离). 原 2.0 转是 PCO-1810 标准件假设, 比演示难 2.7 倍.
            turns=0.75,
            closed_offset_m=0.180,
            direction=1,
            mode=screw_mode,
            capture_radial_m=0.003,
            capture_axial_m=0.003,
            capture_tilt_deg=10.0,
            capture_yaw_deg=30.0,
            # U34 (2026-08-27 用户裁定 "不要太快的拧"): 20 rad/s = 1146°/s,
            # 270° 五六个控制步就拧完 —— 既不像人手, 也让"单指戳一下"足以
            # 拧完全程 (拇指戳局部最优的制度性根源, 见 pk22 取证).
            # 2.0 rad/s 下拧满 270° 需持续驱动 ~2.6s, 接近人手实际速度.
            max_angular_velocity_rad_s=2.0,
        )
    return entry


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
    # Staged from the selected Test-video reconstruction.
    "task1_static_smoke": _td_static("egodex", "task1_static_smoke"),
    "water_bottle_twist_static": _water_bottle_static(),
    "water_bottle_twist_assembled": _water_bottle_static("preengaged"),
    "water_bottle_twist_screw_on": _water_bottle_static("capture"),
    # EgoDex screw_unscrew_bottle_cap/1 (真实视频重建; 见 _screw_unscrew_cap1 注释)
    "screw_unscrew_cap1_static": _screw_unscrew_cap1(),
    "screw_unscrew_cap1_assembled": _screw_unscrew_cap1("preengaged"),
}
# 扭盖 RL 任务入口: 场景 = assembled, 但**机器人侧**用右手 (瓶身摆放仍按左手相位).
# robot_hand 只影响 interact_hand 的判定和 loader 给 du 的手侧, 不影响物体摆放.
CLIPS["screw_unscrew_cap1_task"] = dict(
    _screw_unscrew_cap1("preengaged"), robot_hand="right")
# 任务口径的螺旋角速度上限: 指尖驱动的真实量级 (~5-10 rad/s), 不是审计用的 20.
# 20 rad/s 下 0.63s 就能转完 2 圈, 轻弹即成 —— 训练学到的是"搓帽"不是"拧帽".
CLIPS["screw_unscrew_cap1_task"]["secondary"]["assembly"]["max_angular_velocity_rad_s"] = 6.0
# U40 (2026-08-30 用户裁定 "最接近真实情况建模"): 真实螺纹副任务口径.
# 与 _task 的区别: 接触门/triad 分级/ω 阻尼退役, 换成咬合期重惯量螺旋自由度
# (指尖可传扭矩受 PhysX 摩擦锥真实限制) + 静锁 breakaway + 库仑 + 粘滞.
# 参数量级 = 已破封的松盖 (全新盖 breakaway ~0.4-1 N·m, 超出机器手能力);
# τ=0.075 N·m 时稳态 2 rad/s ≈ 人手拧速. 旧 _task 条目原样保留 (Dyn23 回放).
CLIPS["screw_unscrew_cap1_task_real"] = dict(
    _screw_unscrew_cap1("preengaged"), robot_hand="right")
CLIPS["screw_unscrew_cap1_task_real"]["secondary"]["assembly"].update(
    max_angular_velocity_rad_s=4.0,     # 仅安全夹, 摩擦模型才是整形器
    breakaway_torque_nm=0.04,
    kinetic_torque_nm=0.015,
    viscous_nms=0.03,
    inertia_eff_kgm2=5e-3,
    torque_ema_s=0.025,
    lock_omega_eps=0.05,
)
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


# EgoDex part2/basic_pick_place 的 20 个物体, 资产已全齐 (npz/mesh/usd/affordance).
# Grasp2 = 7.7x2.6cm 甜甜圈 (BODex 抓不出), 是首个跑通的.
for _i in range(20):
    CLIPS[f"Grasp{_i}"] = _replay_grasp("part2", f"basic_pick_place/{_i}", f"obj_{_i:02d}")
del _i


def clip_entry(name: str) -> dict:
    if name not in CLIPS:
        raise KeyError(f"未知 clip '{name}', 可选: {list(CLIPS)}")
    return CLIPS[name]


def configure_cfg(cfg, name: str):
    """把 clip 的资产路径写进 env cfg (在 env 构建之前调用)."""
    e = clip_entry(name)
    cfg.clip_name = name
    cfg.object_cfg.spawn.usd_path = e["usd"]
    if e.get("override_cfg_mass", False):
        # The base cfg carries a 0.2 kg placeholder. This task must use the
        # bottle semantics instead of silently overriding the converted USD.
        cfg.object_cfg.spawn.mass_props.mass = float(e["semantics"].mass_kg)
    if "place_mode" in e:
        cfg.place_mode = e["place_mode"]
    return cfg


def _flatten_usd_file(usd: str) -> None:
    """Make a converted mesh USD self-contained before another conversion.

    IsaacLab's MeshConverter reuses ``<usd_dir>/Props/instanceable_meshes.usd``.
    Two assets in one cache directory would otherwise overwrite each other's
    referenced geometry.
    """

    from pxr import Usd
    stage = Usd.Stage.Open(usd)
    if stage is None:
        raise RuntimeError(f"failed to open converted USD: {usd}")
    for prim in stage.Traverse():
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
    flattened = stage.Flatten()
    temporary = usd + ".flattening.tmp.usd"
    if not flattened.Export(temporary):
        raise RuntimeError(f"failed to flatten converted USD: {usd}")
    os.replace(temporary, usd)


def ensure_object_usd(name: str):
    """ocir 源: object.obj -> 物理烘焙 USD (缓存). 需 Kit 已启动 (env _setup_scene 内调)."""
    e = clip_entry(name)
    if e["source"] not in ("ocir", "static_reconstruction"):
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
    if e.get("flatten_converted_usd", False):
        _flatten_usd_file(usd)
    print(f"[clips] object.obj -> {usd} (convexDecomposition, mass={sem.mass_kg}kg, "
          f"instanceable off ×{n_off})")
    return usd


def ensure_mesh_usd(mesh: str, usd: str, semantics: ObjectSemantics):
    """Convert one auxiliary mesh to a cached rigid-body USD.

    This mirrors ``ensure_object_usd`` for task-specific secondary assets that
    are deliberately not registered as the environment's primary object.
    Kit must already be running.
    """

    if os.path.exists(usd) and os.path.getmtime(usd) >= os.path.getmtime(mesh):
        return usd
    import isaaclab.sim as sim_utils
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
    from isaaclab.sim.schemas.schemas_cfg import ConvexDecompositionPropertiesCfg
    os.makedirs(os.path.dirname(usd), exist_ok=True)
    MeshConverter(MeshConverterCfg(
        asset_path=mesh,
        usd_dir=os.path.dirname(usd),
        usd_file_name=os.path.basename(usd),
        force_usd_conversion=True,
        mesh_collision_props=ConvexDecompositionPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        mass_props=sim_utils.MassPropertiesCfg(mass=semantics.mass_kg),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            max_depenetration_velocity=1000.0,
            sleep_threshold=0.005, stabilization_threshold=0.0025),
    ))
    from pxr import Usd
    stage = Usd.Stage.Open(usd)
    n_off = 0
    for prim in stage.Traverse():
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
            n_off += 1
    stage.GetRootLayer().Save()
    _flatten_usd_file(usd)
    print(
        f"[clips] auxiliary mesh -> {usd} "
        f"(convexDecomposition, mass={semantics.mass_kg}kg, "
        f"instanceable off x{n_off})"
    )
    return usd


def interact_hand(clip_name: str, default: str = "right") -> str:
    """从 phase_left/phase_right 判定这条 clip 是哪只手在交互."""
    e = CLIPS.get(clip_name, {})
    # 双物体任务可显式指定机器人侧 (如扭盖: 瓶身按左手相位摆放, 但任务手是右手).
    if e.get("robot_hand"):
        return e["robot_hand"]
    if not e.get("npz"):
        return default
    if e.get("source") == "static_reconstruction":
        try:
            import numpy as _np
            from rl_rebuild.correction.recon_kailang.static_reconstruction import (
                first_interaction,
            )
            with _np.load(e["npz"], allow_pickle=True) as data:
                return first_interaction(data, e.get("hand"))[0]
        except Exception:
            return default
    try:
        import numpy as _np
        d = _np.load(e["npz"], allow_pickle=True)
        n = {h: int((d[f"phase_{h}"].astype(int) == 1).sum()) for h in ("left", "right")}
    except Exception:
        return default
    if max(n.values()) == 0:
        return default
    return max(n, key=n.get)


def load_data_unit(cfg) -> DataUnit:
    e = clip_entry(cfg.clip_name)
    if e["source"] == "static_reconstruction":
        from rl_rebuild.correction.recon_kailang.static_reconstruction import (
            load_static_reconstruction,
        )
        return load_static_reconstruction(
            e["npz"], e["mesh"], usd_path=e["usd"], clip_id=cfg.clip_name,
            hand=e.get("hand"), placement_frame=e.get("placement_frame"),
            upright=e.get("upright", False),
            robot_hand=e.get("robot_hand"),
            target_hz=cfg.target_hz,
            table_height=cfg.table_top_z,
            table_half=min(cfg.table_size[0], cfg.table_size[1]) / 2.0,
            semantics=e["semantics"], verbose=True)
    if e["source"] == "replay_grasp":
        from rl_rebuild.correction.ref_builders.replay_grasp import load_replay_grasp
        return load_replay_grasp(e["npz"], e["mesh"], usd_path=e["usd"],
                                 # 交互手从 phase_* 自动判定, 不能写死 "right":
                                 # Grasp10/12 在重建里是**左手**交互, 写死右手 = 拿垃圾数据
                                 # (实测 Grasp12 的物体被摆到 x=-0.58, 在机器人底座后面)
                                 hand=interact_hand(cfg.clip_name),
                                 clearance=getattr(cfg, "clearance", None),
                                 freeze_wrist=getattr(cfg, "freeze_wrist", True),
                                 anchor_mode=getattr(cfg, "anchor_mode", None),
                                 pregrasp_align=getattr(cfg, "pregrasp_align", None),
                                 clip_id=cfg.clip_name, target_hz=cfg.target_hz,
                                 table_height=cfg.table_top_z, affordance_npz=e.get("affordance"),
                                 # 手离物体的悬停高度: DexMate 需要比飞手大得多 (飞手能把手
                                 # 硬顶进桌子, 真机械臂顶不动). cfg 没这项时沿用旧默认.
                                 hover_gap=getattr(cfg, "hover_gap", None),
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
