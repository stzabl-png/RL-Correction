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
    """Bundled two-part PCO-1810 bottle scene used for reconstruction QA.

    ⚠ **重建里的左右手命名是反的** (2026-08-05 核实, 依据源视频
    ``V2AP/data/egocentric/egodex/test/screw_unscrew_bottle_cap/27.mp4``):
    视频里**左手拧瓶盖、右手扶瓶身**, 而 `meta.json` / `VALIDATION.md` 记的正相反。

    但**物体↔轨迹的关联是正确的**, 只有轨迹的名字错了:

        track_left  (低, 均低 4cm)  -> 扶瓶身  -> 物理**右**手
        track_right (高)            -> 拧瓶盖  -> 物理**左**手

    所以这里分成两个字段, 不要混:
      * ``hand``       选**哪条轨迹**去定物体的桌面 XY —— 保持 meta.json 的原值,
                       改了会把物体摆到另一只手的位置上 (踩过)。
      * ``robot_hand`` 机器人实际该用哪只手 (决定用左/右手的 GraspPose 模板库)。
    """

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
        hand="left",              # 轨迹名 (实为物理右手) —— 勿改, 见 docstring
        robot_hand="right",       # 机器人用右手扶瓶身
        placement_frame=31,
        semantics=ObjectSemantics(
            label="PCO-1810 filled bottle body", mass_kg=0.53, friction=0.5),
        secondary=dict(
            label="PCO-1810 cap",
            mesh=os.path.join(base, "reconstruction", "bottle_cap.obj"),
            usd=os.path.join(base, "cache", "bottle_cap.usd"),
            hand="right",         # 轨迹名 (实为物理左手) —— 勿改
            robot_hand="left",    # 机器人用左手拧瓶盖
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


# =============================================================================
# 设定 A — RL 学习 **GraspPose 能处理**的物体 (普遍偏大, 可整手包络)
#   骨干 = cuRobo 规划的 close 轨迹, RL 只做残差修正. 详见 docs/TRAINING_SETUPS_A_B.md
# =============================================================================
CLIPS = {
    "pp0_human": _td_ocir("egodex", "pp0"),
    "pp55_human": _td_ocir("egodex", "pp55", grasp_file="failed_grasp_002.json"),
    # Staged from the selected Test-video reconstruction.
    "task1_static_smoke": _td_static("egodex", "task1_static_smoke"),
    "water_bottle_twist_static": _water_bottle_static(),
    "water_bottle_twist_assembled": _water_bottle_static("preengaged"),
    "water_bottle_twist_screw_on": _water_bottle_static("capture"),
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


# EgoDex part2/basic_pick_place 的 20 个物体, 资产已全齐 (npz/mesh/usd/affordance).
# Grasp2 = 7.7x2.6cm 甜甜圈 (BODex 抓不出), 是首个跑通的.
for _i in range(20):
    CLIPS[f"Grasp{_i}"] = _replay_grasp("part2", f"basic_pick_place/{_i}", f"obj_{_i:02d}")
del _i
# 垫↔接触零位校准: **Grasp3 实测必须关**。2026-08-16 用 diag_fgate 做零动作合拢 A/B
# (8 env, 候选判据要 ≥4 垫): 开 = 3.00 个指垫接触(永远出不了候选) / 关 = **4.75**。
# 开着它时 Grasp3_CTRL2(冠军 clip + 冠军配方)跑 16.4M 步确定性成功率恒 0;
# 关掉后 Grasp3_CTRL3 在 9.8 万步到 1%、29.5 万步到 10%(冠军存档 6.5万/22.9万)。
# ⚠ 这个开关**没有通用规则, 只能逐 clip 实测** —— Pour17_* 恰恰相反(开 3.38 / 关 0.00)。
# 详见 DESIGN_LOOP §2.24; 新 clip 上线前先跑 diag_fgate 的 A/B。
CLIPS["Grasp3"]["pad_contact_calib"] = False


# =============================================================================
# 双手拧瓶盖 egodex/test/screw_unscrew_bottle_cap/27 (2026-08-05 入库)
#   人手/物体轨迹 = 27_scene 重建 (188帧@15fps, 双手全有效, phase 通道由
#   contact_auto.json 自动标注烘入; ⚠ 与旧 water_bottle_twist_static 相反,
#   27_scene 的左右手命名**物理正确**: left=拧盖手, right=扶瓶手).
#   物体网格 = BOSL2 CAD 替身 (SAM3D 分不开瓶身瓶盖, 用户裁定另行建模).
#   摆放 = recon_pose.infer_resting_pose 推断的静置位姿 (稳定支撑姿态自动否决
#   前 42 帧横躺跟踪失败段), 见 resting_pose.json; 盖按螺旋全闭合预咬合在瓶口.
#   两条 clip 共享同一个双物体场景, 只是"哪个物体是抓取主体"不同:
#     Screw27_body: 右手 5 指抓瓶身 (11_5 模板), 微抬升验证
#     Screw27_cap : 左手 2 指捏瓶盖 (Tip_Pinch 模板), 微拧验证 (盖被螺旋钉死,
#                   抬不动 —— 判据换成"腕旋转时 screw_angle 跟进" = 握持传扭矩)
# =============================================================================


def _screw27(primary: str):
    rt = os.path.join(_DATASETS, "RR", "Output", "RetargetOutput",
                      "egodex", "screw_unscrew_bottle_cap", "27_scene")
    rc = os.path.join(_DATASETS, "RR", "Output", "ReconstructOutput",
                      "egodex", "screw_unscrew_bottle_cap", "27_scene")
    wb = os.path.join(_DATASETS, "recon_kailang", "water_bottle_twist_static")
    body = dict(
        label="PCO-1810 filled bottle body",
        mesh=os.path.join(wb, "reconstruction", "bottle_body.obj"),
        usd=os.path.join(wb, "cache", "bottle_body.usd"),
        semantics=ObjectSemantics(
            label="PCO-1810 filled bottle body", mass_kg=0.53, friction=0.5),
    )
    cap = dict(
        label="PCO-1810 cap",
        mesh=os.path.join(wb, "reconstruction", "bottle_cap.obj"),
        usd=os.path.join(wb, "cache", "bottle_cap.usd"),
        semantics=ObjectSemantics(label="PCO-1810 cap", mass_kg=0.003, friction=0.4),
    )
    pri, sec = (body, cap) if primary == "body" else (cap, body)
    hand = "right" if primary == "body" else "left"   # 27_scene 命名物理正确, 轨迹=机器人同侧
    entry = dict(
        source="replay_grasp",
        npz=os.path.join(rt, "replay_world.npz"),
        mesh=pri["mesh"],
        usd=pri["usd"],
        runtime_object_physics=True,
        override_cfg_mass=True,
        flatten_converted_usd=True,     # 同 cache 目录两个 USD, 不 flatten 会互相覆盖几何
        hand=hand,
        robot_hand=hand,
        semantics=pri["semantics"],
        # 抓取主体之外的另一个物体 (场景第二刚体), 由任务层建体 + 螺旋约束钉接
        secondary=dict(
            label=sec["label"], mesh=sec["mesh"], usd=sec["usd"],
            semantics=sec["semantics"],
            assembly=dict(
                pitch_m=0.00318, turns=2.0, closed_offset_m=0.180, direction=1,
                mode="preengaged", capture_radial_m=0.003, capture_axial_m=0.003,
                capture_tilt_deg=10.0, capture_yaw_deg=30.0,
                max_angular_velocity_rad_s=20.0,
            ),
        ),
        screw_primary=primary,          # 抓取主体是螺旋的哪一端 ("body"/"cap")
        resting_pose_json=os.path.join(rc, "resting_pose.json"),
        verify_mode=("lift" if primary == "body" else "twist"),
        arm_table_shell=True,           # 臂罚按连杆外壳口径 (真机带壳不碰桌, 用户要求)
        upright_hold=True,              # 姿态保持 (用户要求物体原姿态略微提起; 盖任务
                                        # 同样需要瓶身/盖轴竖直 —— 倾角指标对 yaw 不敏感)
        # 无 affordance: prior 模式对齐目标来自 GraspPose 接触质心, 不需要热图
    )
    if primary == "body":
        # 用户裁定模板 35_8 (2026-08-06, 顶替 11_5): 11_5 的腕朝向使前臂外壳
        # 必然贴/穿桌面 (全池垫底 -3.8cm, 压桌的 l5~l7 段被腕位姿钉死, 上移/
        # 换 yaw/零空间抬肘三招全部无效); 35_8 腕位高 3cm, 外壳稳态 +0.8~1.8cm.
        # 工程适配 (全部烘进 35_8_mid_p12.npz): ① 接触带沿圆柱对称上移到质心
        # 9.2cm; ② 沿"四指→拇指"轴平移 12mm 使开口居中 (原位姿拇指早接触
        # 33.7N/四指够不着); 适配后 c=1.3 全五指接触, 深挤 Q 转正 (+0.17~0.35,
        # 过筛选硬判据 H2). closure_max 放宽给挤压行程, 零动作仍停 c_grasp=1.0.
        entry["closure_max"] = 1.6
        entry["pad_contact_calib"] = False   # 定向平移已烘进 npz, 均值校准会双重平移
    if primary == "cap":
        # 模板 = Dexonomy 15_2 (用户 2026-08-06 定, 原样加载): 六候选合拢扫描中
        # 唯一"候选门可达"的 (零学习 cand_ok 7.8%, 拇 5.85N+中 1.03N 同时受力).
        # 死因复盘 (47_6/4_6 训至全零): Dexonomy 接触标在指尖极点, 我们力垫在
        # 指腹, 系统差 ~2.5cm, 模板回放合拢轨迹从盖旁掠过 -> 接触梯度为零.
        # 补救 = 视频接触点 affordance (甜甜圈设定B经验): 左手 frame47 热图蒸馏
        # 成盖侧壁环带 (72 点), pad_approach 塑形改推参与垫去环带; 候选门/cent/
        # 微拧验证等物理裁判不动. 蒸馏脚本见台账 §2.22.
        entry["pad_contact_calib"] = False   # 只用 affordance 塑形, 不平移腕
        entry["affordance_npz"] = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            "../../tasks/pregrasp/priors/Screw27_cap_affordance.npz"))
    return entry


CLIPS["Screw27_body"] = _screw27("body")
CLIPS["Screw27_cap"] = _screw27("cap")


# =============================================================================
# 倒水 egodex/test/pour/17 (2026-08-14 入库) —— **a 组**(简化手指动作空间)
#   数据 = `datasets/pour17/`(自包含 stage), 上游 take:
#     recon  R&R/Output/ReconstructOutput/egodex_auto/pour/17
#     retarget .../RetargetOutput/egodex_auto/pour/17
#   两个自由物体, 各听各的手(逐物体接触自动配对): 杯 object_0×左, 瓶 object_1×右。
#   **不是装配任务**: 两物体之间没有约束 —— 所以 secondary 不带 assembly(与 screw27 的
#   螺旋钉接相反), 第二个物体只是场景里另一个自由刚体。
#
#   摆放/关键帧**不写死在这里**, 由两份自动产物给(它们是唯一来源):
#     scene_layout.json  逐物体听手 + 姿态(稳定候选->筛直立->筛开口朝上, **不用重建旋转**)
#     keyframes.json     里程碑链 = 训练阶段顺序/目标/容差(B 项裁定: 顺序不许人工设定)
#
#   ⚠ 本 clip 含**人为改动**, 全部记在 datasets/pour17/PROVENANCE.md:
#     杯等比缩放 0.4516(超出手抓握包络)、两网格减面到 80k、抓握模板手写 fingertip_middle。
#     上游 Dexonomy / 重建尺度成熟后应逐条还掉。
#
#   分两条注册, 对应关键帧链的前两阶段 —— 它们都是**单臂**任务, 现有 env 直接可跑,
#   不必等双臂改造(S3 倒水才需要双臂):
#     Pour17_bottle : 右手抓瓶 (链上 S1, f19)
#     Pour17_cup    : 左手抓杯 (链上 S2, f28)
# =============================================================================


def _pour17_aff(primary: str) -> str:
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "../../tasks/pregrasp/priors",
        f"Pour17_{'bottle' if primary == 'bottle' else 'cup'}_affordance.npz"))


def _pour17(primary: str):
    base = os.path.join(_DATASETS, "pour17")
    rt = os.path.join(paths.RR_OUTPUT, "RetargetOutput", "egodex_auto", "pour", "17")
    cup = dict(
        oid="object_0", hand="left",
        label="graycup (scaled 0.4516 -> 7.0x10.5cm)",
        mesh=os.path.join(base, "objects", "object_0", "object_mesh_scaled_final.obj"),
        usd=os.path.join(base, "objects", "object_0.usd"),          # 视觉(主体时运行时贴物理)
        usd_physics=os.path.join(base, "cache", "object_0.usd"),    # 烘焙物理(当 aux 时)
        semantics=ObjectSemantics(label="graycup", mass_kg=0.15, friction=0.5,
                                  mass_range=(0.08, 0.30)),
    )
    bottle = dict(
        oid="object_1", hand="right",
        label="Proud Source bottle (capped)",
        mesh=os.path.join(base, "objects", "object_1", "object_mesh_scaled_final.obj"),
        usd=os.path.join(base, "objects", "object_1.usd"),
        usd_physics=os.path.join(base, "cache", "object_1.usd"),
        # 与 screw27 同一只瓶 -> 沿用已验证的质量/摩擦
        semantics=ObjectSemantics(label="Proud Source bottle", mass_kg=0.53, friction=0.5),
    )
    pri, sec = (bottle, cup) if primary == "bottle" else (cup, bottle)
    return dict(
        source="replay_grasp",
        npz=os.path.join(rt, "replay_world.npz"),
        mesh=pri["mesh"], usd=pri["usd"],
        # ★ 必须锁定 ref_builder 摆放: 默认路径会让 `bimanual_align` 整体覆盖, 而它是
        #   **单物体**逻辑 —— 对本 clip 它把交互帧取成两手的并集起点 f0(左手从 f0 就
        #   贴着杯子), 于是物体被摆到两手之间。2026-08-15 实测: 参考腕轨迹到物体最近
        #   **41.5cm**, 五指全程零接触(与 screw27 端到端 0% 同一个病)。
        place_mode="ref_builder",
        runtime_object_physics=True,
        override_cfg_mass=True,
        flatten_converted_usd=True,
        hand=pri["hand"], robot_hand=pri["hand"],
        semantics=pri["semantics"],
        primary_oid=pri["oid"],
        # aux 用**烘焙物理**的 USD(ensure_mesh_usd 生成); 容器类必须 128 hulls +
        # shrink_wrap, 否则 VHACD 把"肩→颈"/杯内腔的凹陷桥接成幻影壳(screw27 实测
        # 手指在离盖 1.5cm 处被看不见的壳挡住, 整瓶被推走)。
        secondary=dict(label=sec["label"], mesh=sec["mesh"], usd=sec["usd_physics"],
                       semantics=sec["semantics"], oid=sec["oid"],
                       usd_convex_hulls=128, usd_shrink_wrap=True),
        # ★ 摆放与关键帧的唯一来源(env 读这两份, 不在 clips 里写死任何帧号/位姿)
        scene_layout_json=os.path.join(base, "scene_layout.json"),
        keyframes_json=os.path.join(base, "keyframes.json"),
        # GraspPose(2026-08-15 到货, Dexonomy 自动选型): **两手都选了 1_Large_Diameter**
        # (力抓环抱), 而不是先前人工指定的 fingertip_middle —— 自动选型接管, 欠账已还。
        #   杯×左 1_11(17 接触点) / 瓶×右 3_5(8 接触点), ho_c 贴合 0.8~2.0mm
        # 姿态交叉核验: canon_rot vs scene_layout 主轴夹角 0.0°/0.1° ✅
        grasp_prior_npz_default=os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../../tasks/pregrasp/priors",
            f"Pour17_{'bottle' if primary == 'bottle' else 'cup'}.npz")),
        grasp_template="1_Large_Diameter",
        # 视频接触带(自动蒸馏, 还了台账 §2.22 "自动蒸馏链路待建"的欠账)
        # ⚠ 键名必须是 affordance_npz(蒸馏接触带); `affordance` 是 B 组 AffordanceModel
        #   的另一种格式(需要 points_raw), 用错会在 env 里报 KeyError。
        affordance_npz=_pour17_aff(primary),      # pad_approach 塑形(要 pts)
        affordance=_pour17_aff(primary),          # 对齐目标(要 points_raw+heatmap)
        verify_mode="lift",                  # S1/S2 = 接近+抓稳+微抬升
        arm_table_shell=True,
        upright_hold=True,
        # 垫↔接触零位校准: **本 clip 实测必须开**。2026-08-16 用 diag_fgate 做零动作
        # 合拢 A/B(瓶, 8 env): 开 = 3.38 个指垫接触 / 关 = **0.00** —— 关掉时合拢到底
        # 指垫离物体表面还有 1.5~3.5cm, 一个都碰不到。
        # ⚠ 这个开关**没有通用规则, 只能逐 clip 实测**(冠军 clip Grasp3 恰恰相反:
        #   开 3.00 / 关 4.75, 所以那边关)。新 clip 上线前先跑:
        #   $PY -m tasks.pregrasp.diag_fgate --headless --clip <名> --grasp_prior <npz> \
        #       --prior_yaw <角> --stance_prefix 60 [--no_calib]
        #   比较两次的 pads_now 平台值, 胜出的写进这里。详见 DESIGN_LOOP §2.24。
        pad_contact_calib=True,
    )


CLIPS["Pour17_bottle"] = _pour17("bottle")
CLIPS["Pour17_cup"] = _pour17("cup")


# =============================================================================
#   sweep_2 (扫地, 2026-08-19 深夜首注册): 左手簸箕 / 右手扫帚。
#   ⚠ 最小可行注册 (GUI/规划先行), 三个临时项待还:
#     ① 无 affordance (env 全条件分支, 缺省可跑; 接触带在
#        results/sweep_2_better/contact/region_object_{0,1}.npz, 蒸馏待做)
#     ② pad_contact_calib=False 未做逐 clip A/B (上线训练前必须跑 diag_fgate)
#     ③ semantics 质量为估计值 (簸箕/扫帚未称重)
#   GraspPose = 用户钦点 (Dexonomy 功能池, demo_angle 口径):
#     簸箕×左 fingertip_small__8_34 (64.2°, 4 接触; ⚠中指力仅 3%, Gate2 边界)
#     扫帚×右 8_Prismatic_2_Finger__37_16 (30.0°, 5 接触, 力分配 29/28/23/20)
#   scene_layout 红旗: object_0(簸箕) 稳定姿态吸附判"倒置"(轴夹角 157.6°),
#   GUI 首验必看摆放。
# =============================================================================


def _sweep2(primary: str):
    base = os.path.join(_DATASETS, "sweep2")
    rr = "/home/lyh/Project/Reconstruct_and_Retarget/results/sweep_2_better"
    dustpan = dict(
        oid="object_0", hand="left",
        label="dustpan (15.8x2.9x21.6cm, handle near ground)",
        mesh=os.path.join(base, "objects", "object_0", "object_mesh_scaled_final.obj"),
        usd=os.path.join(rr, "retarget", "object_0.usd"),
        usd_physics=os.path.join(base, "cache", "object_0.usd"),
        semantics=ObjectSemantics(label="dustpan", mass_kg=0.15, friction=0.6,
                                  mass_range=(0.08, 0.30)),
    )
    broom = dict(
        oid="object_1", hand="right",
        label="hand broom (7.8x11.6x29.2cm)",
        mesh=os.path.join(base, "objects", "object_1", "object_mesh_scaled_final.obj"),
        usd=os.path.join(rr, "retarget", "object_1.usd"),
        usd_physics=os.path.join(base, "cache", "object_1.usd"),
        semantics=ObjectSemantics(label="hand broom", mass_kg=0.25, friction=0.6,
                                  mass_range=(0.12, 0.45)),
    )
    pri, sec = (broom, dustpan) if primary == "broom" else (dustpan, broom)
    return dict(
        source="replay_grasp",
        npz=os.path.join(rr, "retarget", "replay_world.npz"),
        mesh=pri["mesh"], usd=pri["usd"],
        place_mode="ref_builder",          # 同 pour17 病历: bimanual_align 是单物体逻辑
        runtime_object_physics=True,
        override_cfg_mass=True,
        flatten_converted_usd=True,
        hand=pri["hand"], robot_hand=pri["hand"],
        semantics=pri["semantics"],
        primary_oid=pri["oid"],
        secondary=dict(label=sec["label"], mesh=sec["mesh"], usd=sec["usd_physics"],
                       semantics=sec["semantics"], oid=sec["oid"],
                       usd_convex_hulls=128, usd_shrink_wrap=True),
        scene_layout_json=os.path.join(base, "scene_layout.json"),
        keyframes_json=os.path.join(base, "keyframes.json"),
        grasp_prior_npz_default=os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../../tasks/pregrasp/priors",
            f"Sweep2_{'broom' if primary == 'broom' else 'dustpan'}.npz")),
        grasp_template=("8_Prismatic_2_Finger" if primary == "broom"
                        else "fingertip_small"),
        verify_mode="lift",
        arm_table_shell=True,
        upright_hold=True,
        pad_contact_calib=False,           # ⚠ 未做 A/B, 训练前必测 (见上)
    )


CLIPS["Sweep2_broom"] = _sweep2("broom")
CLIPS["Sweep2_dustpan"] = _sweep2("dustpan")


def clip_entry(name: str) -> dict:
    if name not in CLIPS:
        raise KeyError(f"未知 clip '{name}', 可选: {list(CLIPS)}")
    return CLIPS[name]


def _screw_from_layout(recon_dir: str, *, screw_primary: str = "cap",
                       robot_hand: str | None = None):
    """重建产物 + `scene_layout.json` -> 走**既有** `secondary` 双物体螺旋通路的 clip。

    ★ 不要再另造多物体机制: `tasks/pregrasp/screw_assembly.py` 已经实现了双物体螺旋
      装配, 只要注册表里有 `secondary` 就自动激活(train/eval/record/play 全部免改)。
      本函数只负责把 `scene_layout.py` 算出来的**逐物体听手摆放**翻译成它的字段。

    摆放规则(在 scene_layout.py, 与本文件无关):
        每个物体听自己的那只手 —— XY = 该手在**该物体**接触起始帧的抓取锚点, Z 贴桌。
        clip0 实测: 瓶身听左手@f3(该手 0.5mm/另一手 216.6mm),
                    瓶盖听右手@f10(该手 0.2mm/另一手 175.1mm)。

    `screw_primary` 决定谁是 env.object(机器人抓的那个), 另一个是 env.aux ——
    因为 **RL env 是单手的**(robot_cfg 写死一只 SharpaWave)。
    """
    import json as _j
    import os as _os
    rd = _os.path.abspath(recon_dir)
    lay = _j.load(open(_os.path.join(rd, "scene_layout.json")))
    objs = lay["objects"]
    # 按尺寸认部件: 大的是瓶身, 小的是盖 —— 不靠 object_id 顺序(实测 29 条里有 6 条
    # 的 object_0 其实是盖)
    by_size = sorted(objs, key=lambda o: -max(objs[o]["extent_cm"]))
    body_id, cap_id = by_size[0], by_size[-1]
    A = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(
        _os.path.dirname(_os.path.abspath(__file__))))))   # 仓根
    rt = rd.replace("ReconstructOutput", "RetargetOutput")
    prim_id = cap_id if screw_primary == "cap" else body_id
    rh = robot_hand or objs[prim_id]["anchor_hand"]
    # ⚠ USD 必须指向**待转换的 cache**, 不能用 RetargetOutput 里的 —— 那是纯视觉网格,
    #   没有 RigidBodyAPI, 直接用会报 "Failed to find a rigid body"。
    #   指向不存在的 cache 路径, `ensure_object_usd` 会用 MeshConverter 从 .obj 烘焙物理。
    _cache = _os.path.join(rd, "cache")
    def _part(oid, mass, fric, label):
        return dict(label=label, mesh=objs[oid]["mesh"],
                    usd=_os.path.join(_cache, f"{label}.usd"),
                    hand=objs[oid]["anchor_hand"],
                    robot_hand=objs[oid]["anchor_hand"],
                    placement_frame=objs[oid]["onset_frame"],
                    semantics=ObjectSemantics(label=label, mass_kg=mass, friction=fric))
    # ★ 2026-09-01 用户裁定 (Unscrew/17 台账 U4 修订): 盖 = kailang 质量 3g, 摩擦 **5.0**
    #   (原 0.4)。瓶身的 0.53/0.5 只是占位 —— 主体物在 correction_env 基类里被 G-A
    #   PHYS_RULE 覆写成 0.1kg/μ5; 盖是 aux, 材质走 screw_assembly 的 ScrewAux,
    #   **只有这里**能改它的摩擦 (指垫 multiply 优先级高于 average ⟹ 手↔盖合成 25)。
    prim, sec_id = _part(prim_id, 0.53 if prim_id == body_id else 0.003,
                         0.5 if prim_id == body_id else 5.0,
                         "bottle_body" if prim_id == body_id else "bottle_cap"), \
        (body_id if prim_id == cap_id else cap_id)
    sec = _part(sec_id, 0.53 if sec_id == body_id else 0.003,
                0.5 if sec_id == body_id else 5.0,
                "bottle_body" if sec_id == body_id else "bottle_cap")
    # 螺纹参数直接沿用 screw27 —— **我们用的就是它那套 CAD**(同出 27_cad2), 不是近似。
    # `mode` 由 VLM 的分件判定推出, 不用人填:
    #     combines(把盖拧回去) -> 盖起始是**分开**的, 要被捕获 -> capture
    #     separates(把盖拧下来) -> 盖起始是**装好**的            -> preengaged
    _pc = (lay.get("part_change") or {}).get("part_change") if isinstance(
        lay.get("part_change"), dict) else None
    if _pc is None:
        # interim 的目录名是 <task>__<n>, 与最终目录的 <task>/<n> 不同 —— 要换算
        try:
            _n = _os.path.basename(rd)
            _t = _os.path.basename(_os.path.dirname(rd))
            _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(rd)))
            _ds = _os.path.basename(_os.path.dirname(_os.path.dirname(rd)))
            _gp = _os.path.join(_root, "interim", _ds, f"{_t}__{_n}", "vlm_gate.json")
            _pc = (_j.load(open(_gp)).get("part_change") or {}).get("part_change")
        except Exception:
            _pc = None
    _mode = "capture" if _pc in ("combines", "both", None) else "preengaged"
    sec["assembly"] = dict(
        pitch_m=0.00318, turns=2.0, closed_offset_m=0.180, direction=1,
        mode=_mode, capture_radial_m=0.003, capture_axial_m=0.003,
        capture_tilt_deg=10.0, capture_yaw_deg=30.0, max_angular_velocity_rad_s=20.0)
    sec["assembly_mode_source"] = f"VLM part_change={_pc}"
    return dict(
        source="static_reconstruction", npz=_os.path.join(rt, "replay_world.npz"),
        mesh=prim["mesh"], usd=prim["usd"],
        runtime_object_physics=False, override_cfg_mass=True,
        flatten_converted_usd=True, place_mode="object_only",
        hand=prim["hand"], robot_hand=rh,
        placement_frame=prim["placement_frame"],
        semantics=prim["semantics"], secondary=sec,
        screw_primary=screw_primary,
        scene_layout=lay, recon_dir=rd,
    )


# ── 由 scene_layout 驱动的双物体螺旋 clip ──
_RR_OUT = "/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput"
for _n, _d, _pm in (("screw0_cap", f"{_RR_OUT}/egodex_auto/screw_unscrew_bottle_cap/0", "cap"),
                    ("screw0_body", f"{_RR_OUT}/egodex_auto/screw_unscrew_bottle_cap/0", "body")):
    try:
        CLIPS[_n] = _screw_from_layout(_d, screw_primary=_pm)
    except Exception:      # 缺 scene_layout.json 不该让模块导入失败
        pass


# ── egodex_part4 拧瓶盖 18 条 (datasets/unscrew_bottle/<n>, 2026-08-29 入库) ──
# take 目录是**自包含**的 (replay_world.npz 与重建产物同目录, 不走 RR Output 双树),
# 所以不能直接用 _screw_from_layout 的 ReconstructOutput→RetargetOutput 路径置换,
# 包一层把 npz 指回 take 目录自己。其余全部走既有 secondary 双物体螺旋通路。
#
# CAD 已核对 (md5): 17/18 条的瓶身+瓶盖与 water_bottle_twist_static **字节相同**
# (bottle_body c9d18519 / bottle_cap c755e07c) —— 螺纹参数/质量/摩擦整段沿用
# 已验证配方, 不重标。78 号只注册到 1 个物体, 不注册 (README 已知缺口)。
#
# 任务口径覆写 (承旧台账 tasks/recon_kailang LEDGER_unscrew, 用户裁定):
#   turns=0.75            U30b: 演示实测拧 ~266° 即分离, 2.0 圈是标准件假设 (难 2.7×)
#   mode=preengaged       拧开任务: 盖起始装在瓶上 (VLM part_change 不用猜)
#   U40 真实螺纹副 (2026-08-30 用户裁定 "最接近真实情况建模"; 老方法线 U40/b/c/d):
#     旧口径的"接触门 + ω 阻尼"是假摩擦替身 —— 有接触就白给转动, 于是**碰一下
#     瓶盖就自己转开/脱落**。换成: 咬合期盖用球形重惯量 (指尖→盖可传扭矩由
#     PhysX 摩擦锥真实裁决, 捏得紧才传得多) + 解析螺纹阻力 (静锁 breakaway
#     0.04N·m / 库仑 0.015 / 粘滞 0.03 → τ=0.075 时稳态 2rad/s ≈ 人手拧速);
#     max_ang_vel 退化成 4rad/s 安全夹, 整形交给摩擦模型。
# 角色拍板 (2026-08-29): screw_primary="body" —— 瓶身=env.object (置于桌面, 听
# 左手相位摆放 + upright 投影: 重建静置帧带 ~21° FoundationPose 噪声 > 平底圆柱
# 18.3° 倾倒极限, 旧台账 U24 的总根因, 必须投直); 盖=env.aux, 由 reset_screw 按
# closed_offset 合拢在瓶顶 —— preengaged 的物理正确开局. 若反过来 primary=cap,
# 主体摆放机制会把盖摆到它自己的桌面锚点 (f32 时盖还在倾斜的瓶顶 25cm 空中),
# 瓶身再被反推到斜下方 —— 开局即错.
def _unscrew_take(take_dir: str):
    entry = _screw_from_layout(take_dir, screw_primary="body",
                               robot_hand="left")
    entry["npz"] = os.path.join(os.path.abspath(take_dir), "replay_world.npz")
    entry["upright"] = True
    # 左手×瓶身 affordance (设定 B 对齐目标; 由 expected_area_*_left.npz 转换,
    # points/weight -> points_raw/heatmap, 见 tasks/Unscrew/part4 台账 T2-1)
    _aff = os.path.join(os.path.abspath(take_dir), "contact",
                        "affordance_bottle_left.npz")
    if os.path.isfile(_aff):
        entry["affordance"] = _aff
    entry["resting_pose_json"] = os.path.join(
        os.path.abspath(take_dir), "resting_pose.json")
    asm = entry["secondary"]["assembly"]
    asm["mode"] = "preengaged"
    # 2026-09-01 用户裁定: 瓶盖转 **30°** 即可拧下 (先说 50, 复核后定 30;
    # "0.75 圈=270° 太多了")。原值来自 U30b 的演示实测分离角, 现按实物改。
    asm["turns"] = 30.0 / 360.0
    asm.update(
        # U45: 准静态螺纹 (ω=(|τ|−kinetic)⁺/b). 安全夹 4.0→2.5: 人手拧盖约
        # 2 rad/s, 准静态下顶到 2.5 需持续 τ≈0.09N·m, 是真安全栏非整形器.
        max_angular_velocity_rad_s=2.5,
        breakaway_torque_nm=0.04,         # 已破封的松盖量级 (全新盖 0.4-1N·m)
        kinetic_torque_nm=0.015,
        viscous_nms=0.03,
        # U45: 5e-3→5e-4. (a) 更接近真实盖 (≈7e-7, 5g/r1.7cm); (b) 准静态下
        # I_eff 过大会与"手指用静摩擦强制盖面速度"形成正反馈: τ_ema 每子步
        # 增益 α·I_eff/(dt·b), 5e-3 约 6.7 (周期2振荡), 5e-4 约 0.67 (稳定).
        # 观测器不受影响: τ=I·dw/dt 对 I 不变 (dw ∝ 1/I).
        inertia_eff_kgm2=5e-4,
        torque_ema_s=0.025,
        unlock_dwell_s=0.033,
        lock_omega_eps=0.05,
        react_on_bottle=True,             # 反作用扭矩回瓶身: 左手须抗扭
    )
    # ★ Unscrew/17 拔盖变体 (2026-09-01 用户裁定 "不拧只拔"): UNSCREW_DETACH=pull 时
    #   旋转脱扣关闭, 唯一脱扣通路 = 轴向拉力 ≥ UNSCREW_PULL_N (探针标定量, 初值 3N);
    #   默认 twist = 同事 clip32 原口径不变。台账 tasks/Unscrew/17/A_Design/DECISIONS.md §2。
    if os.environ.get("UNSCREW_DETACH", "twist") == "pull":
        asm.update(detach_mode="pull",
                   breakaway_pull_n=float(os.environ.get("UNSCREW_PULL_N", "3.0")),
                   mass_eff_kg=float(os.environ.get("UNSCREW_PULL_MEFF", "0.2")))
    return entry


_UNSCREW_DIR = os.path.join(_DATASETS, "unscrew_bottle")
if os.path.isdir(_UNSCREW_DIR):
    for _c in sorted(os.listdir(_UNSCREW_DIR)):
        _d = os.path.join(_UNSCREW_DIR, _c)
        if not os.path.isfile(os.path.join(_d, "scene_layout.json")):
            continue               # 78 号单物体 / 未跑 layout 的目录: 静默跳过
        try:
            CLIPS[f"unscrew{_c}_task"] = _unscrew_take(_d)
        except Exception as _e:    # 单条坏数据不该让模块导入失败
            print(f"[clips] ⚠ unscrew{_c}_task 注册失败: {_e}")


def configure_cfg(cfg, name: str):
    """把 clip 的资产路径写进 env cfg (在 env 构建之前调用)."""
    e = clip_entry(name)
    cfg.clip_name = name
    cfg.object_cfg.spawn.usd_path = e["usd"]
    # ---- 交互手 → 接触传感器 (2026-08-05 修) ----
    # cfg 里 fingertip_bodies/contact_sensors 写死 right_*_elastomer, 而场景在
    # _resolve_joint_ids 之前就建好了 —— 左手 clip 会**静默**读右手的垫: pads 恒 0、
    # 向心分恒 0、成功率恒 0 且不报错 (2026-08-05 左手冒烟日志里 pads=0 就是它).
    # 修法: 在 env 构建之前按交互手重建传感器表. 传感器 prim 与 _resolve_joint_ids
    # 用同一个 interact_hand() 判定, 二者必然一致.
    side = interact_hand(name, getattr(cfg, "hand_side", "right"))
    ovr = e.get("sensor_link_override", {})
    if hasattr(cfg, "fingertip_bodies"):
        names = [f"{side}_{f}_{ovr.get(f, 'elastomer')}"
                 for f in ("thumb", "index", "middle", "ring", "pinky")]
        if list(cfg.fingertip_bodies) != names:
            from isaaclab.sensors import ContactSensorCfg
            cfg.hand_side = side
            cfg.fingertip_bodies = names
            cfg.contact_sensors = [
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{n}",
                    history_length=1,
                    filter_prim_paths_expr=["/World/envs/env_.*/Object"],
                ) for n in names]
            print(f"[clips] 交互手={side}: 接触传感器已重建 ({names})")
    if e.get("override_cfg_mass", False):
        # The base cfg carries a 0.2 kg placeholder. This task must use the
        # bottle semantics instead of silently overriding the converted USD.
        cfg.object_cfg.spawn.mass_props.mass = float(e["semantics"].mass_kg)
    if "place_mode" in e:
        cfg.place_mode = e["place_mode"]
    if "verify_mode" in e and hasattr(cfg, "verify_mode"):
        cfg.verify_mode = e["verify_mode"]   # screw 27 盖 = "twist" (微拧验证)
    if "closure_max" in e and hasattr(cfg, "closure_max"):
        cfg.closure_max = float(e["closure_max"])
    if "arm_table_shell" in e and hasattr(cfg, "arm_table_shell"):
        cfg.arm_table_shell = bool(e["arm_table_shell"])
    if "pad_contact_calib" in e and hasattr(cfg, "pad_contact_calib"):
        cfg.pad_contact_calib = bool(e["pad_contact_calib"])
    if "affordance_npz" in e and hasattr(cfg, "affordance_npz"):
        cfg.affordance_npz = e["affordance_npz"]   # 视频接触带 (pad_approach 塑形)
    if "upright_hold" in e and hasattr(cfg, "upright_hold"):
        cfg.upright_hold = bool(e["upright_hold"])
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


def ensure_mesh_usd(mesh: str, usd: str, semantics: ObjectSemantics,
                    max_convex_hulls: int | None = None,
                    shrink_wrap: bool | None = None):
    """Convert one auxiliary mesh to a cached rigid-body USD.

    This mirrors ``ensure_object_usd`` for task-specific secondary assets that
    are deliberately not registered as the environment's primary object.
    Kit must already be running.

    ``max_convex_hulls``/``shrink_wrap``: VHACD 预算覆盖. 默认 32 hulls 会把
    瓶身"肩→颈"的凹陷桥接成幻影锥壳 (screw 27 实测: 手指在离盖 1.5cm 处被
    看不见的壳挡住, 把整瓶推走), 瓶类资产要开到 128 + shrink_wrap.
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
        mesh_collision_props=ConvexDecompositionPropertiesCfg(
            max_convex_hulls=max_convex_hulls, shrink_wrap=shrink_wrap),
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
    """从 phase_left/phase_right 判定这条 clip 是哪只手在交互.

    注册条目带 ``robot_hand`` 时它是权威答案 —— 双手 clip (screw 27) 的 npz 里
    两只手都有 phase 段, "帧数多者胜"会把两条 clip 判成同一只手; 以及旧
    water_bottle_twist_static 的轨迹名 left/right 是反的 (见 _water_bottle_static
    docstring), 都不能靠 phase 通道自动判.
    """
    e = CLIPS.get(clip_name, {})
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
        # 权威静置姿态 (screw 27 等跟踪坏帧 clip): 稳定支撑约束推断的 竖直姿态+贴桌 z
        # 下沉给 builder —— 摆放的 XY 听手 (约定第③步), 姿态/z 听 resting_pose.
        _ro = None
        if e.get("resting_pose_json"):
            import json as _j
            with open(e["resting_pose_json"]) as _f:
                _rj = _j.load(_f)
            _k = "body" if e.get("screw_primary", "body") == "body" else "cap"
            _ro = (tuple(float(v) for v in _rj[f"{_k}_quat_wxyz"]),
                   float(_rj[f"{_k}_pos"][2]))
        return load_replay_grasp(e["npz"], e["mesh"], usd_path=e["usd"],
                                 rest_override=_ro,
                                 # 交互手从 phase_* 自动判定, 不能写死 "right":
                                 # Grasp10/12 在重建里是**左手**交互, 写死右手 = 拿垃圾数据
                                 # (实测 Grasp12 的物体被摆到 x=-0.58, 在机器人底座后面)
                                 hand=interact_hand(cfg.clip_name),
                                 clearance=getattr(cfg, "clearance", None),
                                 freeze_wrist=getattr(cfg, "freeze_wrist", True),
                                 pregrasp_align=getattr(cfg, "pregrasp_align", None),
                                 scene_layout_json=e.get("scene_layout_json"),
                                 clip_id=cfg.clip_name, target_hz=cfg.target_hz,
                                 table_height=cfg.table_top_z, affordance_npz=e.get("affordance"),
                                 # 手离物体的悬停高度: DexMate 需要比飞手大得多 (飞手能把手
                                 # 硬顶进桌子, 真机械臂顶不动). cfg 没这项时沿用旧默认.
                                 hover_gap=getattr(cfg, "hover_gap", None),
                                 semantics=e["semantics"], verbose=True)
    return load_ocir(e["seq_dir"], e["grasp_json"], e["traj_dir"],
                     usd_path=e["usd"], variant=e["variant"],
                     target_hz=cfg.target_hz, table_height=cfg.table_top_z,
                     cache_dir=e.get("cache_dir"),
                     semantics=e["semantics"], verbose=True)
