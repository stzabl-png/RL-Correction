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


def _spec_from_cfg(cfg, assembly) -> ScrewSpec:
    """Build a runtime spec with task-local overrides, without mutating CLIPS."""
    values = dict(assembly)
    turns = getattr(cfg, "screw_turns_override", None)
    if turns is not None:
        values["turns"] = float(turns)
    return ScrewSpec.from_mapping(values)


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
    `place_mode="ref_builder"` 的摆放是 `camera_anchor_shift` 给的**纯 XY 平移**加一个
    z 抬升(见 replay_grasp "相机锚定 (唯一摆放模式)"), **不含任何 yaw**, 所以相对量
    可以原样搬过去。

    ⚠ 2026-08-15 修掉的坑: 原实现拿"主体实际偏航 − layout 里主体偏航"当场景偏航补偿。
    这把**两件不同的事**混为一谈 —— 物体自身的朝向 ≠ 两个坐标系之间的偏航。pour17 的
    主体是瓶子(旋转体), 它的 yaw 在两边都是任意值: layout 那边来自稳定位姿候选(−180°),
    env 那边来自 Dexonomy `canon_rot`(+93.6°, 编码的是**抓取接近方向**不是物体朝向)。
    两个任意值相减得 −86.4°, 于是相对偏移 (−0.5, +24.2)cm(左右并排)被转成
    (+24.0, +3.0)cm(一前一后) —— GUI 里双物体几乎排在同一条 X 线上, 就是这么来的。
    env.py 的 canon_rot↔scene_layout 交叉核验查的是**主轴**夹角, 对绕主轴的 yaw 无感,
    所以核验照过不误。
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
    # 生成占位位姿(真实位姿 reset 时重算): 用 layout 绝对位姿, z 换到 env 桌面
    z_lay_table = float(json.load(open(lay_p)).get("scene_table_z", p_sec[2]))
    env.aux_init_pose_np = np.concatenate([
        [p_sec[0], p_sec[1], p_sec[2] - z_lay_table + float(cfg.table_top_z)],
        env.aux_quat_lay_np])
    print(f"[aux] 第二物体 {sec_oid}({sec.get('label')}): 相对主体偏移 "
          f"{np.round(env.aux_rel_offset_np * 100, 1).tolist()}cm (来自 scene_layout)")
    # 把"曾经被当成场景偏航"的那个量打出来, 便于一眼看出它有多没意义(见 docstring)。
    _dy = np.degrees(_yaw_of(env.aux_quat_lay_np) - _yaw_of(lay[pri_oid]["quat_wxyz"]))
    print(f"[aux]   (诊断) layout 内两物体偏航差 {(_dy + 180) % 360 - 180:.1f}° —— "
          f"**不**用它做任何补偿: 摆放是纯平移, 且旋转体的 yaw 是任意值")
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
    env.screw_spec = _spec_from_cfg(cfg, sec["assembly"])
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
    # aux_contact_sensors: 只有显式置旗的任务(Pour17 的 POUR_OBJ_CONTACT=1)才开
    # 接触上报; 默认 False, 其余任务行为一字不变。
    _auxc = bool(getattr(env.cfg, "aux_contact_sensors", False))
    aux_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Aux",
        spawn=sim_utils.UsdFileCfg(usd_path=sec["usd"],
                                   activate_contact_sensors=_auxc),
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
    # aux_grip_parity (2026-08-20 用户裁定"左右数值一致"): 副物体材质与主物同策略
    # (SuperGrip 3.0/multiply), 否则 B 垫×Aux 合成 1.5 ≠ A 垫×主物 9.0。
    # screw 类任务不受影响 (它们不设此旗)。
    _aux_mat = ("/World/Materials/SuperGrip"
                if getattr(env.cfg, "aux_grip_parity", False)
                else "/World/Materials/ScrewAux")
    for prim_path in sim_utils.find_matching_prim_paths("/World/envs/env_.*/Aux"):
        sim_utils.bind_physics_material(prim_path, _aux_mat)
    if _aux_mat.endswith("SuperGrip"):
        print("[aux] aux_grip_parity: 副物体绑 SuperGrip (与主物一致, 物↔桌 1.5)")

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
    init_thread_state(env)


def init_thread_state(env):
    """U40 真实螺纹副的逐 env 状态 (spec 无 breakaway 时什么都不建)."""
    spec = env.screw_spec
    if spec is None or spec.breakaway_torque_nm is None:
        return
    N, dev = env.num_envs, env.device
    env.screw_omega = torch.zeros(N, device=dev)      # 螺旋自由度角速度
    env.screw_tau_ema = torch.zeros(N, device=dev)    # 指尖传入轴向力矩 (EMA)
    env.screw_locked = torch.ones(N, dtype=torch.bool, device=dev)
    env._screw_capw_vec = torch.zeros(N, 3, device=dev)   # 上一子步实写给盖的 ω 矢量
    env._screw_unlock_dwell = torch.zeros(N, device=dev)
    env._bottle_react_f = torch.zeros(N, 1, 3, device=dev)
    env._bottle_react_t = torch.zeros(N, 1, 3, device=dev)
    env._cap_heavy_state = None                       # 惯量缓冲惰性初始化
    env._react_zeroed = False


def reset_thread_state(env, env_ids):
    """U40 状态复位 —— 与 screw_angle/engaged 同一时刻 (含 RSI 后段出生)."""
    spec = env.screw_spec
    if spec is None or spec.breakaway_torque_nm is None:
        return
    env.screw_omega[env_ids] = 0.0
    env.screw_tau_ema[env_ids] = 0.0
    env.screw_locked[env_ids] = True
    env._screw_capw_vec[env_ids] = 0.0
    env._screw_unlock_dwell[env_ids] = 0.0


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
    reset_thread_state(env, env_ids)


def _quat_apply_t(q, v):
    from isaaclab.utils.math import quat_apply
    return quat_apply(q, v)


def _thread_friction_step(env, relative_ang, axis_w, integrate_angle):
    """U40 真实螺纹副: 重惯量螺旋自由度上的静锁/库仑/粘滞 (老方法线移植)。

    咬合期盖的惯量被换成 ``inertia_eff_kgm2`` (球形, 见 _thread_friction_finish),
    指尖→盖的可传扭矩因此受 PhysX 摩擦锥真实限制 (≤ μ·N·r —— 捏得紧才传得多,
    打滑/粘着由求解器裁决)。本函数只负责**螺纹阻力**:

    - 静锁: 上一子步写入 ω 之后, 物理子步里接触实际注入的角冲量给出净轴向力矩
      估计 τ = I_eff·Δω/dt (EMA 抗单子步碰撞尖峰); |τ| 持续超过 breakaway 才解锁
      —— 轻拍/轻擦永远不解锁 (旧口径"有接触就白给转动"= 盖自己脱落的根因)。
    - 动阶段: 物理携带 ω, 每子步扣除 (τ_k + b·|ω|)·dt/I_eff 的阻力; 停止驱动 →
      ω 衰减, |ω| 低于 lock_omega_eps 且 τ 低于阈值时回锁 (换把重捏 = 重新破静摩擦)。

    力矩估计的三条防幻影 (老台账 U40b/c/d 三次尸检, 逐条都是实测教训):
    ① 只看**盖侧** (相对 Δω 会把瓶身加速误读成拧盖扭矩);
    ② 按**矢量**差分再投影 (存轴向标量时, 左手晃瓶使螺轴变向即漏出 I_eff·Ω⊥²);
    ③ 指尖对盖零接触 ⇒ 物理上没有外力矩, 估计强制归零 (真值门, 封死残余泄漏)。
    """
    spec = env.screw_spec
    if not integrate_angle:
        return env.screw_omega          # 子步末投影调用: 不重复推进状态
    cap = env.aux if env._screw_primary == "body" else env.object
    dt = float(env.cfg.sim.dt)
    # U44 (2026-08-31, 老方法线 5f35f844 同步): 上面三条防幻影只保护了**力矩
    # 估计**, **转动积分**当年原样用着有毒的 relative_ang (盖ω−瓶ω), 且真值门
    # 管不到它。盖是自由刚体、螺旋约束靠每子步"写回"事后施加, 子步内并不随瓶身
    # 加速 => 读到的相对 ω 混入 −Δ瓶ω, 被逐步累加成螺纹转速。瓶身抖动
    # 即燃料。**量级更正 (U44b)**: 瓶身抖动 ±0.5 rad/s 是每**控制步**(12 子步)
    # 的幅度, 摊到子步约 ±0.04 rad/s, 与接触注入的 dw 同量级而非压倒性 ——
    # 本 bug 是真实泄漏, 但不是"盖自己猛转"的主因 (主因见下), 修前修后行为
    # 相近, 已有训练成绩不因本条失效。
    # 修法: 转动与力矩同源, 都用"接触注入的角速度增量" dw ——
    #   dw = (盖ω − 上一子步写入矢量)·当前轴   (写入矢量含写时瓶ω, 故瓶身加速
    #                                          自动抵消: 盖同样没跟随它)
    #   ω_phys = 上一子步写入的螺纹转速 + dw
    cap_tau = 10.0 * spec.breakaway_torque_nm
    dw_max = cap_tau * dt / spec.inertia_eff_kgm2
    dw = ((cap.data.root_ang_vel_w - env._screw_capw_vec) * axis_w
          ).sum(dim=1).clamp(-dw_max, dw_max)
    _gate = getattr(env, "_screw_cap_contact_n", None)
    if _gate is not None and getattr(env, "_thread_tau_contact_gate", True):
        dw = dw * (_gate() > 0).float()
    tau_in = spec.inertia_eff_kgm2 * dw / dt
    omega_phys = env.screw_omega + dw
    alpha = dt / max(spec.torque_ema_s, dt)
    env.screw_tau_ema += alpha * (tau_in - env.screw_tau_ema)
    above = env.screw_tau_ema.abs() > spec.breakaway_torque_nm
    env._screw_unlock_dwell = (env._screw_unlock_dwell + dt) * above.float()
    unlock = env.screw_locked & (env._screw_unlock_dwell >= spec.unlock_dwell_s)
    env.screw_locked = env.screw_locked & ~unlock
    drag = (spec.kinetic_torque_nm + spec.viscous_nms * omega_phys.abs()) \
        * dt / spec.inertia_eff_kgm2
    omega = torch.where(
        env.screw_locked, torch.zeros_like(omega_phys),
        torch.sign(omega_phys) * (omega_phys.abs() - drag).clamp(min=0.0))
    relock = (~env.screw_locked & (omega.abs() < spec.lock_omega_eps)
              & (env.screw_tau_ema.abs() < spec.breakaway_torque_nm))
    env.screw_locked = env.screw_locked | relock
    omega = torch.where(env.screw_locked, torch.zeros_like(omega), omega)
    env.screw_omega = omega.clamp(-spec.max_angular_velocity_rad_s,
                                  spec.max_angular_velocity_rad_s)
    return env.screw_omega


def _thread_friction_finish(env, axis_w, written, body, cap):
    """写回之后的收尾: 记实写 ω 矢量 / 咬合边界切惯量 / 反作用扭矩回瓶身。"""
    spec = env.screw_spec
    env.screw_omega = written
    # U40c: 记录实际写给盖的角速度**矢量** (瓶 ω 矢量 + 螺旋 DOF ω·轴)。
    # 脱扣 env 没有写入, 记当前实测矢量使下一子步差分 ≈0。
    written_vec = body.data.root_ang_vel_w + written[:, None] * axis_w
    env._screw_capw_vec = torch.where(
        env.screw_engaged[:, None], written_vec, cap.data.root_ang_vel_w)

    # 咬合 ↔ 脱扣边界: 盖惯量在 I_eff (球形) 与实物之间切换 —— 脱扣后的自由盖
    # 必须还原实物惯量, 否则抓放手感全错。U40d: 三主轴全部加重, 只加重 Izz 时
    # Izz/Ixx~1e4, 瓶晃引入横向 ω 后欧拉陀螺项让 ω 矢量真实变化, 差分读出幻影。
    if env._cap_heavy_state is None:
        orig = cap.root_physx_view.get_inertias().clone()
        heavy = orig.clone()
        heavy[:, 0] = spec.inertia_eff_kgm2
        heavy[:, 4] = spec.inertia_eff_kgm2
        heavy[:, 8] = spec.inertia_eff_kgm2
        env._cap_inertia_orig, env._cap_inertia_heavy = orig, heavy
        env._cap_heavy_state = torch.zeros(orig.shape[0], dtype=torch.bool,
                                           device=orig.device)
    want = env.screw_engaged.to(env._cap_heavy_state.device)
    flip = want != env._cap_heavy_state
    if flip.any():
        data = torch.where(want.unsqueeze(1), env._cap_inertia_heavy,
                           env._cap_inertia_orig)
        cap.root_physx_view.set_inertias(
            data, flip.nonzero(as_tuple=False).squeeze(1))
        env._cap_heavy_state = want

    if spec.react_on_bottle:
        # 反作用扭矩回瓶身: 锁定 = 指尖扭矩经锁死螺纹透传 (≤breakaway),
        # 转动 = 库仑+粘滞阻力矩的反作用。左手持瓶必须抗住这份扭。
        tau = torch.where(
            env.screw_locked,
            env.screw_tau_ema.clamp(-spec.breakaway_torque_nm,
                                    spec.breakaway_torque_nm),
            torch.sign(env.screw_omega) * (spec.kinetic_torque_nm
                                           + spec.viscous_nms
                                           * env.screw_omega.abs()))
        tau = torch.where(env.screw_engaged, tau, torch.zeros_like(tau))
        env._bottle_react_t[:, 0, :] = tau.unsqueeze(1) * axis_w
        # 全零扭矩且上一子步也已清零 -> 不必再调 (外力是持久量, 写零一次就够)。
        # 省掉的是无接触期每子步一次的 API 调用 (以及它每 5s 刷一行的弃用告警)。
        nz = bool((tau != 0).any())
        if nz or not getattr(env, "_react_zeroed", False):
            body.set_external_force_and_torque(
                env._bottle_react_f, env._bottle_react_t, body_ids=[0],
                is_global=True)
            env._react_zeroed = not nz


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
    real_thread = spec.breakaway_torque_nm is not None
    if real_thread:
        # U40: 接触门/ω 阻尼是假摩擦替身, real 模式下全部退役 (见函数 docstring)。
        # drive_gain 仍然逐步计算 —— 它是观测里的接触特征, 只是不再乘进角速度。
        angular_velocity = _thread_friction_step(env, relative_ang, axis_w,
                                                 integrate_angle)
        active = env.screw_engaged
        angle_step = (angular_velocity * float(env.cfg.sim.dt)
                      if integrate_angle
                      else torch.zeros_like(angular_velocity))
        _skip_legacy = True
    else:
        _skip_legacy = False
    angular_velocity_legacy = (relative_ang * axis_w).sum(dim=1).clamp(
        -spec.max_angular_velocity_rad_s, spec.max_angular_velocity_rad_s)
    # 死区 (2026-08-05, 与源实现的差异①): 噪声级相对角速度不积分. 无死区时
    # 瓶身在桌上的求解器慢旋 → 盖测出持续微小相对角速度 → screw_angle 零接触
    # 自漂 ~0.6°/步, 会污染微拧验证 (11 步漂过 3° 失败线 = 假通过). 真实拧转
    # ~3 rad/s ≫ 0.1, 不受影响.
    angular_velocity_legacy = torch.where(
        angular_velocity_legacy.abs() < 0.1,
        torch.zeros_like(angular_velocity_legacy), angular_velocity_legacy)
    # ---- 训练任务可选钩子 (2026-08-29, 自 recon_kailang 扭盖任务移植; 默认 None
    #      = 行为与既有任务逐位相同, 审计入口不受影响) ----
    # screw_drive_gain (N,) float∈[0,1] 或 bool: 螺纹静摩擦/分级驱动抽象.
    #   0 = 指尖无接触时相对角速度不积分也不存留 (解析螺旋无摩擦, 亚阈值轻擦
    #   会免费空转到释放 —— 旧台账实测近随机策略 41% 假 release);
    #   分数 = U34 三指分级慢拧 (拇/食/中 1 指 1/3 速, 3 指全速, 无悬崖).
    # screw_omega_damping float: 螺纹粘滞摩擦, 相对角速度逐子步衰减 —— 轻弹的
    #   惯性立刻消散, 只有持续接触驱动才维持转动.
    if not _skip_legacy:
        _gain = getattr(env, "screw_drive_gain", None)
        if _gain is not None:
            angular_velocity_legacy = angular_velocity_legacy * _gain.float()
        _damp = getattr(env, "screw_omega_damping", None)
        if _damp is not None:
            angular_velocity_legacy = angular_velocity_legacy * float(_damp)
        angular_velocity = angular_velocity_legacy
        active = env.screw_engaged
        angle_step = (angular_velocity * float(env.cfg.sim.dt)
                      if integrate_angle
                      else torch.zeros_like(angular_velocity))
    proposed = env.screw_angle + angle_step
    max_angle = 2.0 * np.pi * spec.turns
    env.screw_angle[active] = proposed[active].clamp(0.0, max_angle)
    env.screw_has_depth |= active & (env.screw_angle < max_angle - 0.25 * np.pi)

    # 可选钩子: 拧满即脱开 (默认 False = 原行为). 原条件要求"顶满时仍有 ω>0"
    # (最后一推), 但确定性策略拧到顶就停手, drive_gain 下 ω 归零 → 永卡在
    # 释放门前 (旧台账 UnscrewRef1 实锤). 训练任务置 env.screw_detach_at_full=True.
    if getattr(env, "screw_detach_at_full", False):
        detach = active & env.screw_has_depth & (env.screw_angle >= max_angle - 1e-4)
    else:
        detach = (active & env.screw_has_depth
                  & (proposed >= max_angle) & (angular_velocity > 0.0))
    env.screw_engaged[detach] = False
    active = env.screw_engaged
    outward_w = (env.screw_angle >= max_angle) & (angular_velocity > 0.0)
    inward_w = (env.screw_angle <= 0.0) & (angular_velocity < 0.0)
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
        if real_thread:
            # U40: 螺旋 ω 是**状态积分量** (由力矩推进), 必须随位姿一起写回盖 ——
            # 差异②"只跟瓶身走"是为旧口径 (ω 来自测量反馈) 防自漂的, 在这里会让
            # 力矩估计的差分基线对不上, 直接造出第四条幻影通道。
            allowed = torch.where(outward_w | inward_w,
                                  torch.zeros_like(angular_velocity),
                                  angular_velocity)
            cap_lin = body.data.root_lin_vel_w + (lead * allowed)[:, None] * axis_w
            cap_ang = body.data.root_ang_vel_w + allowed[:, None] * axis_w
        else:
            cap_lin = body.data.root_lin_vel_w
            cap_ang = body.data.root_ang_vel_w
        velocity = torch.cat([cap_lin, cap_ang], dim=1)
        ids = active.nonzero(as_tuple=False).squeeze(1)
        cap.write_root_pose_to_sim(pose[ids], ids)
        cap.write_root_velocity_to_sim(velocity[ids], ids)

    if real_thread:
        written = torch.where(active & ~(outward_w | inward_w), angular_velocity,
                              torch.zeros_like(angular_velocity))
        _thread_friction_finish(env, axis_w, written, body, cap)


def reset_aux_free(env, env_ids):
    """无约束的第二物体复位: 跟着**抖动后的主体位置**摆, 保持 layout 给的相对关系。

    **不做任何偏航旋转**, 两条独立理由(任一条成立都足够):
      1. layout 系 → env 系是**纯平移**(`camera_anchor_shift` 的 XY + z 抬升),
         没有偏航可补 —— 详见 `_free_aux_from_layout` 的 docstring 与那个坑。
      2. 主体的 yaw 抖动是世界系预乘(`quat_mul(yaw, obj_init_quat)`), 即**原地自旋**;
         第二个物体是桌上独立的一件东西, 不该绕着主体公转。

    位置抖动照跟(保持两物体相对布局恒定), 姿态直接用 layout 的。
    与 `reset_screw`(瓶盖拧在瓶上, 刚性连体, 必须跟自旋)是**两条不同的路**, 别混。
    """
    if getattr(env, "aux", None) is None or getattr(env, "aux_rel_offset_np", None) is None:
        return
    n = len(env_ids)
    dev = env.device
    origins = env.scene.env_origins[env_ids]
    p = env.obj_start_pos[env_ids]                       # (n,3) origin 相对, 含抖动
    off = torch.tensor(env.aux_rel_offset_np, dtype=torch.float32, device=dev)
    aux_p = p + off.expand(n, 3)
    aux_q = torch.tensor(env.aux_quat_lay_np, dtype=torch.float32,
                         device=dev).expand(n, 4).contiguous()
    env.aux.write_root_pose_to_sim(torch.cat([aux_p + origins, aux_q], dim=1), env_ids)
    env.aux.write_root_velocity_to_sim(
        torch.zeros(n, 6, dtype=torch.float32, device=dev), env_ids)
    # 记下 aux 的真实摆放(origin 相对): 双臂 B 侧复位要拿它当 obj_start 基准
    # (B 侧不重摆场景, 同帧读 data.root_pos_w 有脏读风险, 用这份权威值)
    if not hasattr(env, "_aux_last_pos"):
        env._aux_last_pos = torch.zeros(env.num_envs, 3, device=dev)
        env._aux_last_quat = torch.zeros(env.num_envs, 4, device=dev)
    env._aux_last_pos[env_ids] = aux_p
    env._aux_last_quat[env_ids] = aux_q
