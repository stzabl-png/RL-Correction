"""世界指纹: 采集 / 落盘 / 比对。

为什么存在
----------
策略 ckpt 是一张"输入数字 → 输出数字"的查表, 它学到的一切都锚在一套固定几何上
(肩膀在哪 / 桌面多高 / 物体多重多滑 / 参考轨迹长什么样)。换了世界, 同一个物体在
策略眼里的数字就变了, 它照旧记忆输出 → 全盘失败。

★而且不会报错: 观测维度一个不差, 所有形状检查全绿。我们叫它"同维异义"。
实测事故: 站姿 USD 换版 → 某 ckpt 回放 0/64, obs 348 维一字未变。

所以规矩是: **ckpt 只能在它出生的世界里回放; 换世界必须带上世界的身份证。**
本模块把"身份证"做成机制 —— 训练时自动落盘, 回放/评测时自动核对, 对不上就拒跑。

用法
----
    import world_fingerprint as WF
    WF.write(env, os.path.join(log_dir, "world.json"))          # 训练开始时
    WF.assert_match(env, os.path.join(run_dir, "world.json"))   # 回放/评测前
"""
from __future__ import annotations

import hashlib
import json
import os
import re

# 关键项: 对不上 = ckpt 在这台机器上无效, 直接拒跑
CRITICAL = (
    "robot.usd_md5", "robot.controlled_joint_names_in_order",
    "reference.md5", "policy_io.obs_dim", "policy_io.act_dim",
    "criteria.digest", "criteria.schema",
    "assembly.pitch_m", "assembly.turns", "assembly.closed_offset_m",
    "assembly.direction", "assembly.max_angular_velocity_rad_s",
    "assembly.omega_damping", "assembly.detach_at_full",
    # U40 真实螺纹副 (不适用时一律 0.0/legacy —— 不能留 None, 那会被判"未验")
    "assembly.thread_model", "assembly.breakaway_torque_nm",
    "assembly.kinetic_torque_nm", "assembly.viscous_nms",
    "assembly.inertia_eff_kgm2", "assembly.torque_ema_s",
    "assembly.unlock_dwell_s", "assembly.lock_omega_eps",
    "assembly.react_on_bottle",
    "sensors.pad_force_threshold_N", "sensors.pads_min_per_hand",
    "method.clip", "method.variant", "method.squeeze_ff_enabled",
    "method.beta_r", "method.beta_l",
    "time.control_dt_s", "time.decimation",
    "table.table_top_z_m",
    "objects.object_1.mass_kg", "objects.object_1.static_friction",
    "objects.object_0.mass_kg", "objects.object_0.static_friction",
)
# 提示项: 对不上通常不致命, 但值得知道
WARN = (
    "time.physics_dt_s", "physx.solver_type", "scene.env_spacing_m",
    "switches.friction_curriculum", "switches.obj_jitter_xy",
    "robot.self_collision", "sensors.count",
)


def _md5(path):
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None


def _f(x):
    try:
        return round(float(x), 9)
    except Exception:
        return None


def _self_collision(env, spawn_cfg):
    """Read the effective self-collision value from USD, falling back to cfg."""
    path = getattr(getattr(env, "hand", None), "cfg", None)
    path = getattr(path, "prim_path", None)
    why = "no prim_path"
    if path:
        path0 = re.sub(r"env_[^/]*", "env_0", path)
        try:
            import omni.usd
            from pxr import Usd

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(path0)
            if prim and prim.IsValid():
                for child in Usd.PrimRange(prim):
                    attr = child.GetAttribute(
                        "physxArticulation:enabledSelfCollisions")
                    if (attr and attr.IsValid()
                            and attr.HasAuthoredValue()):
                        return bool(attr.Get()), f"USD:{child.GetPath()}"
                why = f"USD subtree has no authored value@{path0}"
            else:
                why = f"USD missing prim@{path0}"
        except Exception as exc:
            why = f"USD unreadable({type(exc).__name__})"
    value = getattr(getattr(spawn_cfg, "articulation_props", None),
                    "enabled_self_collisions", None)
    if value is not None:
        return bool(value), f"cfg fallback ({why})"
    return None, f"unreadable ({why}; cfg missing)"


def _criteria():
    try:
        import progress as task_progress
        schema, digest = task_progress.criteria_digest()
        return {"schema": schema, "digest": digest,
                "items": task_progress.criteria_items()}
    except Exception as exc:
        return {"schema": None, "digest": None, "items": None,
                "error": type(exc).__name__}


def collect(env) -> dict:
    """从活的 env 采集世界指纹 (只采"世界", 不采方法)。"""
    sim, cfg = env.sim, env.cfg
    pcfg = getattr(cfg, "sim", None)
    jn = list(env.hand.joint_names)
    ctrl = [jn[i] for i in env.map_ids_t.cpu().tolist()]
    sp = getattr(env.hand.cfg, "spawn", None)
    usd = getattr(sp, "usd_path", None)

    objs = {}
    for oid, art in (("object_0", env.object), ("object_1", env.aux)):
        d = {}
        try:
            d["mass_kg"] = _f(art.root_physx_view.get_masses()[0].sum())
        except Exception:
            d["mass_kg"] = None
        try:
            m = art.root_physx_view.get_material_properties()[0]
            d["static_friction"] = _f(m[0][0])
            d["dynamic_friction"] = _f(m[0][1])
            d["restitution"] = _f(m[0][2])
        except Exception:
            pass
        d["usd"] = getattr(getattr(art.cfg, "spawn", None), "usd_path", None)
        objs[oid] = d

    self_collision, self_collision_source = _self_collision(env, sp)
    screw = getattr(env, "screw_spec", None)
    assembly = {
        "pitch_m": _f(getattr(screw, "pitch_m", None)),
        "turns": _f(getattr(screw, "turns", None)),
        "closed_offset_m": _f(getattr(screw, "closed_offset_m", None)),
        "direction": getattr(screw, "direction", None),
        "max_angular_velocity_rad_s": _f(
            getattr(screw, "max_angular_velocity_rad_s", None)),
        # real 模式下 ω 阻尼不再施加 —— 记 0.0 (事实), 不留 None (会被判未验)
        "omega_damping": _f(getattr(env, "screw_omega_damping", None) or 0.0),
        "detach_at_full": getattr(env, "screw_detach_at_full", None),
    }
    _brk = getattr(screw, "breakaway_torque_nm", None)
    assembly.update(
        thread_model=("real_friction" if _brk is not None
                      else "legacy_gain_damping"),
        breakaway_torque_nm=_f(_brk or 0.0),
        kinetic_torque_nm=_f(getattr(screw, "kinetic_torque_nm", 0.0)
                             if _brk is not None else 0.0),
        viscous_nms=_f(getattr(screw, "viscous_nms", 0.0)
                       if _brk is not None else 0.0),
        inertia_eff_kgm2=_f(getattr(screw, "inertia_eff_kgm2", 0.0)
                            if _brk is not None else 0.0),
        torque_ema_s=_f(getattr(screw, "torque_ema_s", 0.0)
                        if _brk is not None else 0.0),
        unlock_dwell_s=_f(getattr(screw, "unlock_dwell_s", 0.0)
                          if _brk is not None else 0.0),
        lock_omega_eps=_f(getattr(screw, "lock_omega_eps", 0.0)
                          if _brk is not None else 0.0),
        react_on_bottle=bool(getattr(screw, "react_on_bottle", False)
                             if _brk is not None else False),
    )
    method = {
        "clip": getattr(cfg, "clip_name", None),
        "variant": getattr(env, "_variant", None),
        "squeeze_ff_enabled": getattr(env, "squeeze_ff_enabled", None),
        "beta_r": _f(getattr(env, "beta_r", None)),
        "beta_l": _f(getattr(env, "beta_l", None)),
    }
    ref = getattr(env, "_master_path", None) or os.environ.get("POUR_REF_NPZ")
    fp = {
        "reference": {"path": ref, "md5": _md5(ref) if ref else None},
        "policy_io": {"obs_dim": int(getattr(cfg, "observation_space", 0)) or None,
                      "act_dim": int(getattr(cfg, "action_space", 0)) or None},
        "schema": "world_fingerprint_v1",
        "criteria": _criteria(),
        "assembly": assembly,
        "method": method,
        "time": {"physics_dt_s": _f(sim.get_physics_dt()),
                 "control_dt_s": _f(sim.get_physics_dt()
                                    * int(getattr(cfg, "decimation", 1))),
                 "decimation": int(getattr(cfg, "decimation", 1)),
                 "render_interval": int(getattr(pcfg, "render_interval", -1))
                 if pcfg else None},
        "physx": {"solver_type": getattr(getattr(pcfg, "physx", None),
                                         "solver_type", None) if pcfg else None,
                  "gravity": list(getattr(pcfg, "gravity", ())) if pcfg else None},
        "table": {"table_top_z_m": _f(getattr(cfg, "table_top_z", None)),
                  "table_size_m": list(getattr(cfg, "table_size", ()))},
        "objects": objs,
        "robot": {"usd": usd, "usd_md5": _md5(usd) if usd else None,
                  "num_joints_articulation": len(jn),
                  "controlled_joint_names_in_order": ctrl,
                  "self_collision": self_collision,
                  "self_collision_source": self_collision_source},
        "sensors": {"count": len(getattr(env, "_all_sensors", [])),
                    "pad_force_threshold_N": _f(
                        getattr(env, "pad_force_threshold_N", None)),
                    "pads_min_per_hand": getattr(env, "pads_min_per_hand", None)},
        "scene": {"env_spacing_m": _f(getattr(cfg.scene, "env_spacing", None)),
                  "replicate_physics": bool(
                      getattr(cfg.scene, "replicate_physics", False))},
        "switches": {"approach_only": bool(getattr(cfg, "approach_only", False)),
                     "obj_jitter_xy": _f(getattr(cfg, "obj_jitter_xy", None)),
                     "friction_curriculum": bool(
                         getattr(cfg, "friction_curriculum", False))},
    }
    return fp


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def compare(recorded: dict, current: dict):
    """返回 (不符, 提示, 无法核对); 每项 = (键, 记录值, 当前值)."""
    a, b = _flat(recorded), _flat(current)
    crit, warn = [], []
    def _ne(key):
        x, y = a[key], b[key]
        if key.endswith("md5") and isinstance(x, str) and isinstance(y, str):
            n = min(len(x), len(y))          # 旧记录只存 8 位前缀, 按短的一方比
            return x[:n] != y[:n]
        return x != y

    # ★L5-16 自审修复: 原来写的是 `if key in a and key in b and _ne(key)` ——
    # **键缺失或值为 None 时直接跳过 = 缺数据即通过**, 正是我们刚立规矩要防的形状。
    # 现在三分类: 一致 / 不符 / 无法核对; "无法核对"必须显式报出, 不得冒充通过。
    unver = []
    for key in CRITICAL:
        x, y = a.get(key, KeyError), b.get(key, KeyError)
        if x is KeyError or y is KeyError or x is None or y is None:
            unver.append((key, None if x is KeyError else x,
                          None if y is KeyError else y))
        elif _ne(key):
            crit.append((key, a[key], b[key]))
    for key in WARN:
        if key in a and key in b and a[key] is not None and b[key] is not None \
                and a[key] != b[key]:
            warn.append((key, a[key], b[key]))
    return crit, warn, unver


def write(env, path, extra=None):
    fp = collect(env)
    if extra:
        for k, v in extra.items():
            if isinstance(v, dict) and isinstance(fp.get(k), dict):
                fp[k].update(v)
            else:
                fp[k] = v
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(fp, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    r = fp["robot"]
    print(f"[world] USD md5={str(r['usd_md5'])[:8]} | 母带 md5="
          f"{str(fp.get('reference', {}).get('md5'))[:8]} | 受控关节 "
          f"{len(r['controlled_joint_names_in_order'])} | 控制步 "
          f"{fp['time']['control_dt_s']}s | 桌面 {fp['table']['table_top_z_m']}",
          flush=True)
    return fp


def _abort_unverified_world():
    import sys
    sys.stdout.flush()
    os._exit(11)


def assert_recorded(env, recorded, strict=True, label="recorded world", ignore=()):
    """Compare an in-memory current-schema fingerprint against a live env."""
    crit, warn, unver = compare(recorded, collect(env))
    ignored = set(ignore)
    crit = [item for item in crit if item[0] not in ignored]
    unver = [item for item in unver if item[0] not in ignored]
    for key, old, new in warn:
        print(f"[world] {label} 提示 {key}: 记录={old} 当前={new}", flush=True)
    if crit:
        print(f"[world] ★{label} 与当前环境不匹配:", flush=True)
        for key, old, new in crit:
            print(f"[world]   {key}: 记录={old} 当前={new}", flush=True)
    if unver:
        print(f"[world] ★{label} 有 {len(unver)} 项关键值无法核对:", flush=True)
        for key, old, new in unver:
            print(f"[world]   {key}: 记录={old} 当前={new}", flush=True)
    if crit or unver:
        if strict:
            _abort_unverified_world()
        return False
    print(f"[world] ✅ {label} 匹配", flush=True)
    return True


def assert_match(env, path, strict=True):
    """回放/评测前核对。critical 不符时: strict=True 直接退出, 否则只告警。"""
    if not os.path.exists(path):
        print(f"[world] ⚠ 找不到世界指纹 {path} —— 无法核对, 这个 ckpt 的出生世界未知",
              flush=True)
        if strict:
            _abort_unverified_world()
        return False
    with open(path) as f:
        rec = json.load(f)
    # ★旧格式兼容 (L5-12 实测漏洞): 早期 world.json 只有 9 个平铺键, 键名与 v1 不同
    # (ref_md5 / obs_dim / usd="default"), 比对时一个关键项都匹配不上 -> 静默放行,
    # 而那恰恰是最需要核对的一批 ckpt。这里显式转译并对无法核对的项报警。
    if rec.get("schema") != "world_fingerprint_v1":
        legacy = {"reference": {"md5": rec.get("ref_md5"),
                                "path": rec.get("ref_npz")},
                  "policy_io": {"obs_dim": rec.get("obs_dim"),
                                "act_dim": rec.get("act_dim")}}
        u = rec.get("usd")
        if u and u != "default":
            legacy["robot"] = {"usd_md5": _md5(u), "usd": u}
        print(f"[world] ⚠ 旧格式世界指纹 (schema={rec.get('schema')}): 只能核对 "
              f"母带md5/obs/act" + ("" if u and u != "default"
                                     else " —— usd 记的是 'default', 无法确认"
                                          "实际加载的机器人文件") +
              "; 桌高/物体质量摩擦/关节表/控制步 **全部无法核对**", flush=True)
        rec = legacy
    crit, warn, unver = compare(rec, collect(env))
    if unver:
        print(f"[world] ⚠ {len(unver)}/{len(CRITICAL)} 项关键项**无法核对**"
              f"(缺失或值为 None) —— 这不是通过, 是未验:", flush=True)
        for k, x, y in unver:
            print(f"[world]   {k}: 记录={x} 当前={y}", flush=True)
    for k, x, y in warn:
        print(f"[world] 提示 {k}: 记录={x} 当前={y}", flush=True)
    if crit:
        print("[world] ★世界不匹配 —— 该 ckpt 不是在这个世界里训练的:", flush=True)
        for k, x, y in crit:
            sx = str(x)[:60] + ("…" if len(str(x)) > 60 else "")
            sy = str(y)[:60] + ("…" if len(str(y)) > 60 else "")
            print(f"[world]   {k}\n[world]     出生世界: {sx}\n"
                  f"[world]     当前世界: {sy}", flush=True)
        print("[world] 回放结果无意义(实测同类事故: 站姿USD换版 -> 0/64, "
              "而 obs 维度一字未变)。要强跑请显式设 POUR_IGNORE_WORLD=1。", flush=True)
        if strict:
            # 用 os._exit 而非 raise: Isaac 的 app 会吞 SystemExit, 实测退出码变 0
            # —— 闸门"报了警却照样跑完"比不设闸更危险。
            import sys as _sys
            _sys.stdout.flush()
            os._exit(11)
        return False
    if unver:
        print("[world] 关键项存在未验值；严格模式拒绝回放。"
              "要强跑请显式设 POUR_IGNORE_WORLD=1。", flush=True)
        if strict:
            _abort_unverified_world()
        return False

    print("[world] ✅ 世界匹配", flush=True)
    return True
