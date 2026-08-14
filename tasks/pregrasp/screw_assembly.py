"""双物体螺旋场景组件 (2026-08-05, screw_unscrew_bottle_cap/27 任务).

从 tasks/recon_kailang/bottle_reconstruction 移植 (那是 QA 场景, 与 GraspTaskEnv
是**兄弟类**不能继承), 以纯函数挂在 GraspTaskEnv 上, clip 注册表有 ``secondary``
时自动激活 —— 所有入口 (train/eval/record/play/smoke/诊断脚本) 免改动获得支持.

角色参数化: 抓取主体 (env.object, prim "/Object") 可以是瓶身也可以是瓶盖,
由 entry["screw_primary"] 决定; 另一个物体 = env.aux (prim "/Aux").
螺旋约束的语义永远是 "读瓶身位姿 → 写瓶盖位姿" (单向投影, GPU 张量写),
谁是主体只决定张量从哪个 handle 取:

    Screw27_body: body=env.object, cap=env.aux   (右手抓瓶身, 盖跟着瓶走)
    Screw27_cap : body=env.aux,   cap=env.object (左手捏盖; 盖被钉死 => 微拧验证)

⚠ 已知语义边界 (与源实现相同):
  - 咬合期约束是**单向**的: 对盖施力不会传到瓶身 (盖的位姿每步被投影覆盖).
    所以"捏住盖抬"物理上抬不动 —— 这正是盖任务的验证要换微拧的原因.
  - 盖↔瓶身三角面碰撞被 FilteredPairs 豁免 (可见螺纹是渲染几何, 会跟解析
    螺旋打架); 机器人/桌面对两者的接触都保留.
"""
from __future__ import annotations

import os

import numpy as np
import torch

from rl_rebuild.correction import clips
from rl_rebuild.correction import frames as F
from tasks.recon_kailang.bottle_reconstruction.screw_joint import (
    ScrewSpec,
    assembled_cap_pose,
)

__all__ = ["ScrewSpec", "assembled_cap_pose", "pre_init", "setup_scene",
           "reset_screw", "apply_screw", "screw_entry"]


def screw_entry(clip_name: str):
    """clip 有双物体**螺旋装配**时返回 (entry, secondary), 否则 (entry, None)。"""
    e = clips.clip_entry(clip_name)
    sec = e.get("secondary")
    if sec and sec.get("assembly"):
        return e, sec
    return e, None


def aux_entry(clip_name: str):
    """clip 有**第二个物体**时返回 (entry, secondary) —— 不要求是装配。

    2026-08-14 解耦: 原先"建第二刚体"与"加螺旋约束"绑在一起, 于是倒水这类
    **两个自由物体**的任务(secondary 无 assembly)第二个物体根本不会被生成。
    现在: 有 secondary 就建刚体; 有 assembly 才加螺旋约束与碰撞豁免。
    """
    e = clips.clip_entry(clip_name)
    return e, (e.get("secondary") or None)


def _yaw_of(q) -> float:
    """wxyz -> 绕 z 的偏航角(弧度)。"""
    w, x, y, z = [float(v) for v in q]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _free_aux_from_layout(env, cfg, entry, sec) -> bool:
    """无装配的第二物体: 从 `scene_layout.json` 取**相对主体**的摆放。

    只用**相对量**(sec - pri), 因为 layout 在重建世界系、env 在机器人场景系 ——
    两者都 z-up 且已对齐重力, 差的是平移与一个 yaw; 相对量对平移免疫, yaw 在
    reset 时按"主体实际姿态 vs layout 主体姿态"的偏航差补上。
    """
    lay_p = entry.get("scene_layout_json")
    if not lay_p or not os.path.isfile(lay_p):
        print(f"[aux] ⚠ clip 有 secondary 但没有 scene_layout_json, 不建第二物体")
        return False
    import json
    lay = json.load(open(lay_p))["objects"]
    pri_oid, sec_oid = entry.get("primary_oid"), sec.get("oid")
    if pri_oid not in lay or sec_oid not in lay:
        print(f"[aux] ⚠ scene_layout 缺 {pri_oid}/{sec_oid}, 不建第二物体")
        return False
    p_pri = np.asarray(lay[pri_oid]["pos"], float)
    p_sec = np.asarray(lay[sec_oid]["pos"], float)
    env.aux_rel_offset_np = p_sec - p_pri                    # 相对主体的偏移
    env.aux_quat_lay_np = np.asarray(lay[sec_oid]["quat_wxyz"], float)
    env.aux_yaw_lay = _yaw_of(lay[pri_oid]["quat_wxyz"])     # layout 里主体的偏航
    # 生成占位位姿(真实位姿 reset 时重算): 用 layout 绝对位姿, z 换到 env 桌面
    z_lay_table = float(json.load(open(lay_p)).get("scene_table_z", p_sec[2]))
    env.aux_init_pose_np = np.concatenate([
        [p_sec[0], p_sec[1], p_sec[2] - z_lay_table + float(cfg.table_top_z)],
        env.aux_quat_lay_np])
    print(f"[aux] 第二物体 {sec_oid}({sec.get('label')}): 相对主体偏移 "
          f"{np.round(env.aux_rel_offset_np * 100, 1).tolist()}cm (来自 scene_layout)")
    return True


def pre_init(env, cfg) -> bool:
    """super().__init__ **之前**调用: 解析 spec/角色/几何常量 + 初始摆放.

    返回是否激活了螺旋装配 (无 secondary 的普通 clip 返回 False, 全部后续
    钩子按 env.screw_spec is None 自动短路).
    """
    env.screw_spec = None
    env.aux = None
    env.aux_entry = None
    env.aux_rel_offset_np = None
    entry, any_sec = aux_entry(cfg.clip_name)
    if any_sec is None:
        return False
    if not any_sec.get("assembly"):
        # ---- 两个自由物体(倒水等): 只建第二刚体, 不加任何约束 ----
        env.aux_entry = any_sec
        if not _free_aux_from_layout(env, cfg, entry, any_sec):
            env.aux_entry = None
        return env.aux_entry is not None
    entry, sec = screw_entry(cfg.clip_name)
    env._screw_primary = entry.get("screw_primary", "body")
    env.aux_entry = sec
    env.screw_spec = ScrewSpec.from_mapping(sec["assembly"])
    assert env.screw_spec.mode == "preengaged", \
        "pregrasp 任务系目前只接了拧开(preengaged)模式"
    body_mesh = entry["mesh"] if env._screw_primary == "body" else sec["mesh"]
    cap_mesh = sec["mesh"] if env._screw_primary == "body" else entry["mesh"]
    bv = F.load_obj_verts(body_mesh)
    cv = F.load_obj_verts(cap_mesh)
    env.body_top_offset_m = float(bv[:, 2].max())
    env.free_collision_radius_m = float(
        np.linalg.norm(bv[:, :2], axis=1).max()
        + np.linalg.norm(cv[:, :2], axis=1).max())
    # 初始摆放占位 (真实位姿 reset 时按 obj_init_* 重算): 用 resting_pose.json,
    # 保证建场景时三者已处于闭合相对位姿, 开局无冲量.
    import json
    with open(entry["resting_pose_json"]) as f:
        rp = json.load(f)
    key = "cap" if env._screw_primary == "body" else "body"
    env.aux_init_pose_np = np.array(
        rp[f"{key}_pos"] + rp[f"{key}_quat_wxyz"], dtype=np.float64)
    cfg.rsi_prob = 0.0
    return True


def setup_scene(env):
    """GraspTaskEnv._setup_scene 里 super() 之后调用: 建 Aux 刚体 + 碰撞豁免.

    2026-08-14: 建刚体的条件从 `screw_spec` 放宽到 `aux_entry` —— 两个**自由**物体
    (倒水)也要有第二刚体, 只是不加螺旋约束/碰撞豁免。
    """
    if getattr(env, "aux_entry", None) is None:
        return
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    sec = env.aux_entry
    clips.ensure_mesh_usd(sec["mesh"], sec["usd"], sec["semantics"],
                          max_convex_hulls=sec.get("usd_convex_hulls"),
                          shrink_wrap=sec.get("usd_shrink_wrap"))
    pose = env.aux_init_pose_np
    aux_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Aux",
        spawn=sim_utils.UsdFileCfg(usd_path=sec["usd"]),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=tuple(float(v) for v in pose[:3]),
            rot=tuple(float(v) for v in pose[3:7]),
        ),
    )
    env.aux = RigidObject(aux_cfg)
    env.scene.rigid_objects["aux"] = env.aux

    # Aux 摩擦材质 —— **两条路都要绑**(自由第二物体同样需要正确摩擦)
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=float(sec["semantics"].friction),
        dynamic_friction=float(sec["semantics"].friction),
        restitution=0.0,
        friction_combine_mode="average",
        restitution_combine_mode="multiply",
    )
    material.func("/World/Materials/ScrewAux", material)
    for prim_path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Aux"):
        sim_utils.bind_physics_material(prim_path, "/World/Materials/ScrewAux")

    if env.screw_spec is None:
        print(f"[aux] 自由第二物体就绪: {sec.get('label')} "
              f"mass={sec['semantics'].mass_kg}kg friction={sec['semantics'].friction} "
              f"(无约束/不豁免碰撞)")
        return

    # 螺旋元数据 + 盖↔瓶身碰撞豁免 (对每个 env; prim 名与源实现不同, 这里
    # 直接写 Object↔Aux 对 —— FilteredPairs 是对称的, 谁挂 API 无所谓)
    import omni.usd
    from pxr import Sdf, UsdGeom, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    n = 0
    for obj_path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Object"):
        env_path = str(obj_path).rsplit("/", 1)[0]
        meta = UsdGeom.Scope.Define(stage, f"{env_path}/BottleScrew").GetPrim()
        meta.CreateAttribute("bottleScrew:pitchM", Sdf.ValueTypeNames.Double
                             ).Set(env.screw_spec.pitch_m)
        meta.CreateAttribute("bottleScrew:turns", Sdf.ValueTypeNames.Double
                             ).Set(env.screw_spec.turns)
        meta.CreateAttribute("bottleScrew:mode", Sdf.ValueTypeNames.String
                             ).Set(env.screw_spec.mode)
        meta.CreateAttribute("bottleScrew:primary", Sdf.ValueTypeNames.String
                             ).Set(env._screw_primary)
        flt = UsdPhysics.FilteredPairsAPI.Apply(
            stage.GetPrimAtPath(f"{env_path}/Aux"))
        flt.CreateFilteredPairsRel().AddTarget(Sdf.Path(f"{env_path}/Object"))
        n += 1
    env.scene.filter_collisions()
    print(f"[screw] 双物体装配就绪: primary={env._screw_primary} | aux×{n} "
          f"mass={sec['semantics'].mass_kg}kg friction={sec['semantics'].friction} | "
          f"pitch={env.screw_spec.pitch_m*1000:.2f}mm turns={env.screw_spec.turns:.1f} "
          f"travel={env.screw_spec.travel_m*1000:.2f}mm")


def init_state(env):
    """super().__init__ 之后 (device 就绪) 调用: 螺旋状态张量."""
    if env.screw_spec is None:
        return
    env.screw_angle = torch.zeros(env.num_envs, dtype=torch.float32,
                                  device=env.device)
    env.screw_engaged = torch.ones(env.num_envs, dtype=torch.bool,
                                   device=env.device)   # preengaged
    env.screw_has_depth = env.screw_engaged.clone()


def reset_screw(env, env_ids):
    """GraspTaskEnv._reset_idx 里主体物体写完之后调用: Aux 跟着主体摆 + 状态复位.

    主体的 xy/yaw 抖动 (obj_jitter) 已包含在 obj_start_pos/quat 里, Aux 按
    闭合相对位姿从**抖动后**的主体位姿推出, 装配永远一致.
    """
    if env.screw_spec is None:
        return reset_aux_free(env, env_ids)
    n = len(env_ids)
    origins = env.scene.env_origins[env_ids]
    p = env.obj_start_pos[env_ids]                     # (n,3) origin 相对
    q = env.obj_start_quat[env_ids]                    # (n,4) wxyz
    axis = _quat_apply_t(q, torch.tensor([0.0, 0.0, 1.0], device=q.device
                                         ).expand(n, 3))
    off = float(env.screw_spec.closed_offset_m)
    sign = 1.0 if env._screw_primary == "body" else -1.0
    aux_p = p + sign * off * axis                      # body→cap 加, cap→body 减
    pose = torch.cat([aux_p + origins, q], dim=1)
    vel = torch.zeros(n, 6, dtype=torch.float32, device=q.device)
    env.aux.write_root_pose_to_sim(pose, env_ids)
    env.aux.write_root_velocity_to_sim(vel, env_ids)
    env.screw_angle[env_ids] = 0.0
    env.screw_engaged[env_ids] = True
    env.screw_has_depth[env_ids] = True


def _quat_apply_t(q, v):
    from isaaclab.utils.math import quat_apply
    return quat_apply(q, v)


def apply_screw(env, *, integrate_angle: bool = True):
    """每个物理子步 (integrate=True) 与 _get_dones 前 (False) 各调一次.

    移植自 BottleReconstructionEnv.apply_screw_constraint, 仅替换 handle 解析
    (body/cap ← primary 角色) 与去掉外部演示力矩口. 语义未动.
    """
    if env.screw_spec is None:
        return
    from isaaclab.utils.math import quat_apply, quat_conjugate, quat_mul

    spec = env.screw_spec
    body = env.object if env._screw_primary == "body" else env.aux
    cap = env.aux if env._screw_primary == "body" else env.object

    body_q = body.data.root_quat_w
    cap_q = cap.data.root_quat_w
    axis_local = torch.zeros(env.num_envs, 3, device=env.device)
    axis_local[:, 2] = 1.0
    axis_w = quat_apply(body_q, axis_local)

    relative_pos = cap.data.root_pos_w - body.data.root_pos_w
    absolute_axis_offset = (relative_pos * axis_w).sum(dim=1)
    axial = absolute_axis_offset - spec.closed_offset_m
    radial_vector = relative_pos - absolute_axis_offset[:, None] * axis_w
    radial = radial_vector.norm(dim=1)

    # 拧开(preengaged)模式: 脱扣后的自由盖没有"进螺纹"的正当理由, 不给对准豁免
    # —— 一律被瓶身包络挡住, 落在瓶口/瓶肩上 (2026-08-05 修的穿模雷, 保持一致).
    blocked = (
        ~env.screw_engaged
        & (radial <= env.free_collision_radius_m)
        & (absolute_axis_offset < env.body_top_offset_m)
    )
    if blocked.any():
        corrected_pos = cap.data.root_pos_w + (
            env.body_top_offset_m - absolute_axis_offset
        )[:, None] * axis_w
        relative_lin = cap.data.root_lin_vel_w - body.data.root_lin_vel_w
        inward_speed = (relative_lin * axis_w).sum(dim=1).clamp(max=0.0)
        corrected_lin = cap.data.root_lin_vel_w - inward_speed[:, None] * axis_w
        blocked_pose = torch.cat([corrected_pos, cap_q], dim=1)
        blocked_vel = torch.cat([corrected_lin, cap.data.root_ang_vel_w], dim=1)
        ids = blocked.nonzero(as_tuple=False).squeeze(1)
        cap.write_root_pose_to_sim(blocked_pose[ids], ids)
        cap.write_root_velocity_to_sim(blocked_vel[ids], ids)

    relative_ang = cap.data.root_ang_vel_w - body.data.root_ang_vel_w
    angular_velocity = (relative_ang * axis_w).sum(dim=1).clamp(
        -spec.max_angular_velocity_rad_s, spec.max_angular_velocity_rad_s)
    # 死区 (2026-08-05, 与源实现的差异①): 噪声级相对角速度不积分. 无死区时
    # 瓶身在桌上的求解器慢旋 → 盖测出持续微小相对角速度 → screw_angle 零接触
    # 自漂 ~0.6°/步, 会污染微拧验证 (11 步漂过 3° 失败线 = 假通过). 真实拧转
    # ~3 rad/s ≫ 0.1, 不受影响.
    angular_velocity = torch.where(angular_velocity.abs() < 0.1,
                                   torch.zeros_like(angular_velocity),
                                   angular_velocity)
    active = env.screw_engaged
    angle_step = (angular_velocity * float(env.cfg.sim.dt)
                  if integrate_angle else torch.zeros_like(angular_velocity))
    proposed = env.screw_angle + angle_step
    max_angle = 2.0 * np.pi * spec.turns
    env.screw_angle[active] = proposed[active].clamp(0.0, max_angle)
    env.screw_has_depth |= active & (env.screw_angle < max_angle - 0.25 * np.pi)

    detach = (active & env.screw_has_depth
              & (proposed >= max_angle) & (angular_velocity > 0.0))
    env.screw_engaged[detach] = False
    active = env.screw_engaged
    if active.any():
        lead = spec.direction * spec.pitch_m / (2.0 * np.pi)
        angle = env.screw_angle
        target_offset = spec.closed_offset_m + lead * angle
        target_pos = body.data.root_pos_w + target_offset[:, None] * axis_w
        half = 0.5 * angle
        twist = torch.zeros(env.num_envs, 4, device=env.device)
        twist[:, 0] = torch.cos(half)
        twist[:, 3] = torch.sin(half)
        target_q = quat_mul(body_q, twist)
        pose = torch.cat([target_pos, target_q], dim=1)

        # 与源实现的差异② (2026-08-05): 盖速度**只跟瓶身走**, 不把 allowed_omega
        # 回注 —— 回注会让"测量→回写"闭环把任何一次性扰动永久保持 (无衰减),
        # 是上面自漂的根源. 去掉后, 角度积分只吃**当个子步物理新注入**的相对
        # 角速度 (手指真实施加的扭矩), 等效于"螺纹高阻尼/不溜转" —— 真瓶盖
        # 本来就有旋合摩擦, 语义更贴近实物, 且验证判据不再被惯性滑行污染.
        cap_lin = body.data.root_lin_vel_w
        cap_ang = body.data.root_ang_vel_w
        velocity = torch.cat([cap_lin, cap_ang], dim=1)
        ids = active.nonzero(as_tuple=False).squeeze(1)
        cap.write_root_pose_to_sim(pose[ids], ids)
        cap.write_root_velocity_to_sim(velocity[ids], ids)


def reset_aux_free(env, env_ids):
    """无约束的第二物体复位: 跟着**抖动后的主体**摆, 保持 layout 给的相对关系。

    yaw 处理: layout 在重建世界系、env 在场景系, 两者可能差一个偏航。用
    "主体实际偏航 − layout 里主体偏航" 作为补偿角, 同时旋转相对偏移与第二物体姿态,
    于是两物体的相对布局在任何 yaw 下都一致。
    """
    if getattr(env, "aux", None) is None or getattr(env, "aux_rel_offset_np", None) is None:
        return
    n = len(env_ids)
    dev = env.device
    origins = env.scene.env_origins[env_ids]
    p = env.obj_start_pos[env_ids]                       # (n,3) origin 相对
    q = env.obj_start_quat[env_ids]                      # (n,4) wxyz
    # 逐 env 的偏航补偿
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw_env = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    dyaw = yaw_env - float(env.aux_yaw_lay)
    c, s = torch.cos(dyaw), torch.sin(dyaw)
    off = torch.tensor(env.aux_rel_offset_np, dtype=torch.float32, device=dev)
    ox = c * off[0] - s * off[1]
    oy = s * off[0] + c * off[1]
    aux_p = p + torch.stack([ox, oy, off[2].expand(n)], dim=1)
    # 姿态 = Rz(dyaw) ∘ layout 姿态
    ql = torch.tensor(env.aux_quat_lay_np, dtype=torch.float32, device=dev).expand(n, 4)
    h = dyaw * 0.5
    qz = torch.stack([torch.cos(h), torch.zeros_like(h), torch.zeros_like(h),
                      torch.sin(h)], dim=1)
    aux_q = torch.stack([
        qz[:, 0] * ql[:, 0] - qz[:, 3] * ql[:, 3],
        qz[:, 0] * ql[:, 1] - qz[:, 3] * ql[:, 2],
        qz[:, 0] * ql[:, 2] + qz[:, 3] * ql[:, 1],
        qz[:, 0] * ql[:, 3] + qz[:, 3] * ql[:, 0]], dim=1)
    aux_q = aux_q / aux_q.norm(dim=1, keepdim=True)
    env.aux.write_root_pose_to_sim(torch.cat([aux_p + origins, aux_q], dim=1), env_ids)
    env.aux.write_root_velocity_to_sim(
        torch.zeros(n, 6, dtype=torch.float32, device=dev), env_ids)
