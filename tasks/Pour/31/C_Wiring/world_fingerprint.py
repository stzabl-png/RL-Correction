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
    # ★L5-34: 指垫摩擦是手↔物合成摩擦的一半(multiply), 换了它就是换了抓握物理。
    #   老 world.json 没这一项 -> 走"未验"报警, 不拦。
    "robot.pad_friction",
    "robot.supergrip_bodies",
    # ★L5-36: 通用设计 G-B 起始黄窗 + 打乱/反向对照 —— 都改档位表 = 改残差界/皮筋/时钟门/
    #   regime 观测, ckpt 跨表回放无意义。老 world.json 缺键 -> 未验不拦。
    "switches.tier_floor_start", "switches.tier_shuffle", "switches.tier_reverse",
    # ★L5-36.2 抓握建立整形旗: 改 G1→G2 奖励地形, ckpt 跨旗回放无意义。
    "switches.grip_shape",
    # ★L5-36.3 抓稳期姿态守恒旗: 改认证判据 (旗开时常量也进 criteria.digest) + G1→G2 罚。
    "switches.hold_pose",
    # ★POUR_MIN 最简配方旗: 改认证判据 (常量进 digest, schema 3) + D4 死线口径 + 关 5 项整形。
    "switches.min_recipe",
    # ★L5-36.7 place 双棘轮旗: 改放回段奖励地形。
    "switches.place_both",
    # ★L5-36.9 placed=成功终点旗: 改判据(倾角5°→digest已变)+终止+奖励+RSI出生表。
    "switches.place_terminal",
    "switches.place_grace",
    "switches.place_curr",
    "reference.md5", "policy_io.obs_dim", "policy_io.act_dim",
    # ★L5-26: 判据摘要 —— 改阈值=改成败判定, 属关键项。与整文件哈希不同,
    # 加计数器不会动它(见 progress.criteria_items 的说明)。
    "criteria.digest", "criteria.schema",
    # ★L5-31 消融旗: conf_flat 把置信度整条链路拍平 —— 时钟门从红档 8cm 变 5cm、
    # 朝向从"红档禁判"变"照判", **直接改成败判定**(主判据 clock_done 就靠时钟门)。
    # 但 criteria.digest 抓不到它: 常量一个没变, 变的是"每一行用哪一档"。
    # 不列关键项的具体风险: 拿 flat 训的 ckpt 在 base 环境评测, 世界闸会放行,
    # 而判据其实不一样 —— 正是本项目反复踩的"静默匹配"。
    "switches.conf_flat",
    # ★L5-31 straight 臂: 冻结臂前馈 + 放大残差界。它不改判据, 但**彻底改变动作
    # 的含义** —— 大界训出的 ckpt 拿到小界环境里回放, 每一步都会被钳住。
    "switches.arm_ref_free", "switches.arm_free_scale",
    "switches.goal_only",
    "switches.rel_ref",
    "switches.place_shape", "switches.mouth_bonus",
    # ★L5-32 (2026-08-31): VARIANT (HYB=物轨+人手 / OBJ=纯物轨) —— 本轮消融的
    # **主变量**, 而在此之前世界门里一项都看不见它(它只被写进 world.json 的
    # method.variant, 没人拿它比对)。
    # 排查过的具体后果链: variant → no_hand_ref → W_OBJ/W_HAND → 时钟门 `ok` →
    # 时钟 `k` 推进快慢 → `ff` 取母带哪一行。**前馈的时间对齐是变的**, 所以
    # HYB 的 ckpt 放进 OBJ 环境跑, 它每一步拿到的参考行都和训练时不一样。
    # (顺带查清: hand_dh 只进奖励 r_shape, **不进观测** —— obs_dim 两边相同,
    #  所以维度检查这一关也拦不住, 必须显式列关键项。)
    "method.variant",
    "time.control_dt_s", "time.decimation",
    "table.table_top_z_m",
    "objects.object_1.mass_kg", "objects.object_1.static_friction",
    "objects.object_0.mass_kg", "objects.object_0.static_friction",
)
# 提示项: 对不上通常不致命, 但值得知道
WARN = (
    "time.physics_dt_s", "physx.solver_type", "scene.env_spacing_m",
    "switches.friction_curriculum", "switches.obj_jitter_xy",
    "robot.self_collision", "sensors.pad_force_threshold_N",
    "sensors.pads_min_per_hand",
    # ★L5-25: 加/减接触传感器**会改世界**, 而此前指纹里没有任何一项能看见它 ——
    # 换了传感器配置的 ckpt 会静默"匹配"。至少让它在提示层可见。
    "sensors.count",
    # ★L5-31: 轴对称假设。`_axis_only_R`(参考生成) 与 `_tilt`(G3/placed/G4/皮筋/死线)
    # 全链都把"绕长轴的自转"当作不可观测而丢弃 —— 对瓶/杯成立, 换非轴对称物体
    # (带把手的杯、勺、盒、壶嘴瓶) 则整条判据链静默失效: 策略把物体绕长轴转任意
    # 角度, 所有 Gate 照样全绿。此前指纹里**没有任何一项能看见这个前提**。
    # 现列为提示项而非关键项: 关键项会让四条在跑的线的 ckpt 立刻失效; 下次冻结
    # 判据(改终止条件)时应提升为 CRITICAL。
    "geometry.up_local_bot", "geometry.up_local_cup",
    "geometry.axis_spin_ignored",
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


def _self_collision(env, sp):
    """自碰撞开关的**生效值**, 返回 (值, 来源)。

    ★USD 优先, 不是 cfg 优先: cfg 里写的值不一定被写进 stage
    (2026-08-29 实测 cfg=True 而 USD authored=False, PhysX 认的是后者)。
    ★读不到时返回 (None, ...) 而非 False —— "读不到"和"关着"是两回事,
    把默认值当事实记下来就是"假值", 比记"未知"危险。
    只查 env_0 一个 prim, 不做全 stage 遍历 —— 1024 env 下遍历会拖慢启动。
    """
    path = getattr(getattr(env, "hand", None), "cfg", None)
    path = getattr(path, "prim_path", None)
    why = "无 prim_path"
    if path:
        p0 = re.sub(r"env_[^/]*", "env_0", path)
        try:
            import omni.usd
            stg = omni.usd.get_context().get_stage()
            prim = stg.GetPrimAtPath(p0)
            if prim and prim.IsValid():
                a = prim.GetAttribute("physxArticulation:enabledSelfCollisions")
                if a and a.IsValid() and a.HasAuthoredValue():
                    return bool(a.Get()), f"USD:{p0}"
                # ★属性常常 authored 在子 prim(articulation root 未必是顶层)。
                # 早先只查顶层, 在 msc 上报"无 authored 值", 而独立 pxr 复核到的是
                # False —— 少查一层就把"已知"降级成了"未知"。只遍历 env_0 这一棵
                # 子树, 与 1024 env 无关, 开销可忽略。
                from pxr import Usd
                for c in Usd.PrimRange(prim):
                    a = c.GetAttribute("physxArticulation:enabledSelfCollisions")
                    if a and a.IsValid() and a.HasAuthoredValue():
                        return bool(a.Get()), f"USD:{c.GetPath().pathString}"
                why = f"USD子树无authored值@{p0}"
            else:
                why = f"USD无此prim@{p0}"
        except Exception as e:
            # ★不能在这里 return: USD 读不到不代表 cfg 也没有。早先写成直接 return,
            # 等于"有兜底也不用", 把已知信息丢掉退回未知 —— 是"记未知"矫枉过正的另一面。
            why = f"USD不可读({type(e).__name__})"
    v = getattr(getattr(sp, "articulation_props", None),
                "enabled_self_collisions", None)
    if v is not None:
        return bool(v), f"cfg.spawn.articulation_props(★未必生效; {why})"
    return None, f"unreadable({why}; cfg亦无)"


def collect(env) -> dict:
    """从活的 env 采集世界指纹 (只采"世界", 不采方法)。"""
    sim, cfg = env.sim, env.cfg
    pcfg = getattr(cfg, "sim", None)
    jn = list(env.hand.joint_names)
    ctrl = [jn[i] for i in env.map_ids_t.cpu().tolist()]
    sp = getattr(env.hand.cfg, "spawn", None)
    usd = getattr(sp, "usd_path", None)

    objs = {}
    for oid, art in (("object_1", env.object), ("object_0", env.aux)):
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

    _sc, _sc_src = _self_collision(env, sp)
    try:
        import progress as _PG
        _cs, _cd = _PG.criteria_digest()
        _crit = {"schema": _cs, "digest": _cd, "items": _PG.criteria_items()}
    except Exception as e:                       # 读不到就记未知, 不记假值
        _crit = {"schema": None, "digest": None,
                 "items": None, "error": type(e).__name__}
    # ★L5-31 轴对称假设: 从**活的 PB 实例**读实际在用的长轴, 不从常量抄 ——
    # 抄常量只能证明"常量是多少", 证明不了"判据用的是哪个"。
    try:
        _pb = env.PB
        _geom = {"up_local_bot": [_f(x) for x in _pb.up.tolist()],
                 "up_local_cup": [_f(x) for x in _pb.upc.tolist()],
                 "axis_spin_ignored": True,
                 "note": "绕长轴自转在 参考IK/皮筋/G3/placed/G4/死线 全链不判"}
    except Exception as e:                       # 读不到记 None, 不填假值
        _geom = {"up_local_bot": None, "up_local_cup": None,
                 "axis_spin_ignored": None, "error": type(e).__name__}
    ref = getattr(env, "_master_path", None) or os.environ.get("POUR_REF_NPZ")
    fp = {
        "reference": {"path": ref, "md5": _md5(ref) if ref else None},
        "policy_io": {"obs_dim": int(getattr(cfg, "observation_space", 0)) or None,
                      "act_dim": int(getattr(cfg, "action_space", 0)) or None},
        "schema": "world_fingerprint_v1",
        "criteria": _crit,
        "geometry": _geom,
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
                  # 记生效值: correction_env 建 SuperGrip 材质时读的就是这个变量
                  # (pour_env 导入期已填默认, 见 PHYS_RULE), 不设时旧默认 3.0。
                  "pad_friction": _f(float(os.environ.get("POUR_PAD_FRIC", "3.0"))),
                  # ★2026-09-03 CRITICAL: 绑到 SuperGrip(高摩擦=pad_friction)的 body 全集
                  #   = fingertip_bodies(交互侧, 也走接触传感器) + extra_supergrip_bodies
                  #   (另一手补绑)。pour 曾漏设后者 => 左手垫落 LowGrip 0.2, 双手不对称。
                  #   进指纹避免"digest 相同但左右垫摩擦不同的世界"混判。
                  "supergrip_bodies": sorted(
                      list(getattr(cfg, "fingertip_bodies", []))
                      + list(getattr(cfg, "extra_supergrip_bodies", []) or [])),
                  # ★记生效值 + 来源, 见 _self_collision() 注释。
                  "self_collision": _sc, "self_collision_source": _sc_src},
        "sensors": {"count": len(getattr(env, "_all_sensors", [])),
                    "pad_force_threshold_N": None, "pads_min_per_hand": None},
        "scene": {"env_spacing_m": _f(getattr(cfg.scene, "env_spacing", None)),
                  "replicate_physics": bool(
                      getattr(cfg.scene, "replicate_physics", False))},
        "switches": {"approach_only": bool(getattr(cfg, "approach_only", False)),
                     "obj_jitter_xy": _f(getattr(cfg, "obj_jitter_xy", None)),
                     "conf_flat": bool(getattr(env.PB, "conf_flat", False))
                     if hasattr(env, "PB") else None,
                     "arm_ref_free": bool(getattr(env, "arm_free", False)),
                     # ★L5-32 `goal` 臂: 去掉 adv/leash/r_shape/D3 并**冻结时钟**。
                     #   冻时钟 = 前馈恒取 IA0 行 —— 环境动力学与其它臂完全不同,
                     #   ckpt 互放就是胡来。criteria.digest 抓不到(常量没变)。
                     "goal_only": bool(getattr(env, "goal_only", False)),
                     # ★L5-33 两个 earn-only 整形旗: 都改奖励地形(placed 从零梯度
                     #   变有梯度 / 对位从无梯度变有梯度), ckpt 跨旗回放没有意义。
                     #   digest 抓不到(判据常量没变 —— 它们是奖励, 该静里正该静)。
                     "place_shape": bool(getattr(env.PB, "place_shape", False))
                     if hasattr(env, "PB") else None,
                     "mouth_bonus": bool(getattr(env, "mouth_bonus", False)),
                     "min_recipe": bool(getattr(env, "min_recipe", False)),
                     # ★关旗时写 1.0 而不是 None: None 在闸里意味着"**读不到 /
                     # 无法核对**", 用它表示"不适用"会让每一份指纹都变成不可核对。
                     # 关旗时实际生效的缩放就是 1.0, 如实写。
                     "arm_free_scale": _f(float(os.environ.get(
                         "POUR_ARM_FREE_SCALE", "20.0"))
                         if getattr(env, "arm_free", False) else 1.0),
                     "friction_curriculum": bool(
                         getattr(cfg, "friction_curriculum", False)),
                     # ★L5-36 G-B / 对照变换: 记生效值 (无进度机的任务写 0/False = 不适用,
                     #   不写 None —— None 在闸里是"读不到")。
                     "tier_floor_start": int(getattr(env.PB, "tier_info", {}).get(
                         "tier_floor_start", 0)) if hasattr(env, "PB") else 0,
                     "tier_shuffle": int(getattr(env.PB, "tier_info", {}).get(
                         "tier_shuffle", 0)) if hasattr(env, "PB") else 0,
                     "tier_reverse": bool(getattr(env.PB, "tier_info", {}).get(
                         "tier_reverse", False)) if hasattr(env, "PB") else False,
                     # ★L5-36.2: 关旗写 False/0 (不适用), 不写 None
                     "grip_shape": bool(getattr(env, "grip_shape", False)),
                     "grip_opp_k": _f(getattr(env, "K_OPP", 0.0)) if getattr(env, "grip_shape", False) else 0.0,
                     "grip_follow_k": _f(getattr(env, "K_FOLLOW", 0.0)) if getattr(env, "grip_shape", False) else 0.0,
                     # ★L5-36.3: 关旗写 False (不适用)
                     "hold_pose": bool(getattr(env.PB, "hold_pose", False)) if hasattr(env, "PB") else False,
                     "place_both": bool(getattr(env.PB, "place_both", False)) if hasattr(env, "PB") else False,
                     "place_terminal": bool(getattr(env.PB, "place_terminal", False)) if hasattr(env, "PB") else False,
                     # ★2026-09-03 PLACE-FIX ①③④: ①改终止(D3_dev宽限)、④改判据(placed容差退火)
                     #   => CRITICAL; ③改奖励(放置期脱握罚) 记生效值。digest 按最终紧容差算不受④影响。
                     "grip_hold": bool(getattr(env.PB, "grip_hold", False)) if hasattr(env, "PB") else False,
                     "grip_hold_k": _f(getattr(env.PB, "grip_hold_k", 0.0)) if getattr(getattr(env, "PB", None), "grip_hold", False) else 0.0,
                     "place_grace": int(getattr(env.PB, "place_grace", 0)) if hasattr(env, "PB") else 0,
                     "place_curr": bool(getattr(env.PB, "place_curr", False)) if hasattr(env, "PB") else False,
                     # v1/v2 语义不同 (v2 量绝对倾斜/漂移, 罚窗含合拢期); 关旗 0
                     "hold_mode": (2 if getattr(env.PB, "hold_v2", False) else
                                   (1 if getattr(env.PB, "hold_pose", False) else 0)) if hasattr(env, "PB") else 0,
                     # ★R (2026-09-04): 关系表征 (leash+时钟门 世界绝对→关系向量)。绝对 vs 关系是不同世界。
                     "rel_ref": bool(getattr(env.PB, "rel_ref", False)) if hasattr(env, "PB") else False},
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
    with open(path, "w") as f:
        json.dump(fp, f, indent=1, ensure_ascii=False)
    r = fp["robot"]
    print(f"[world] USD md5={str(r['usd_md5'])[:8]} | 母带 md5="
          f"{str(fp.get('reference', {}).get('md5'))[:8]} | 受控关节 "
          f"{len(r['controlled_joint_names_in_order'])} | 控制步 "
          f"{fp['time']['control_dt_s']}s | 桌面 {fp['table']['table_top_z_m']}",
          flush=True)
    return fp


def assert_match(env, path, strict=True):
    """回放/评测前核对。critical 不符时: strict=True 直接退出, 否则只告警。"""
    if not os.path.exists(path):
        print(f"[world] ⚠ 找不到世界指纹 {path} —— 无法核对, 这个 ckpt 的出生世界未知",
              flush=True)
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
    print("[world] ✅ 世界匹配", flush=True)
    return True


def world_json_of(ckpt_path):
    """ckpt 的出生世界 = 它同 run 目录下的 world.json (stage1_nn/x.pth -> ../world.json)。"""
    return os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(ckpt_path))), "world.json")


def restore_physics_env(path):
    """回放/探针专用, 必须在 `import pour_env` **之前**调用。

    把 ckpt 出生世界记录的物体质量/摩擦/指垫摩擦还原成 POUR_OBJ_MASS / POUR_OBJ_FRIC /
    POUR_PAD_FRIC。背景 (L5-34 规矩): pour_env 导入期把三者默认填成 0.1/5.0/5.0, 而 L5-33
    及更早的 ckpt 生在硬物理里 (瓶0.53/杯0.15, 摩擦3.0/0.5, 指垫3.0), 不还原就会在
    assert_match 硬闸处被拦。两物体同值 ⟹ 当年就是覆写出来的, 原样覆写; 异值 ⟹ 母带原生
    物理, 用空串关掉覆写。找不到 world.json 时不动环境变量, 交给硬闸去报。
    只是"还原可复现的输入", 不替代 assert_match —— 回放前照样要核对。"""
    if not os.path.exists(path):
        print(f"[world] ⚠ 无 world.json, 物理参数按当前默认 (硬闸稍后会报未知): {path}",
              flush=True)
        return None
    with open(path) as f:
        w = json.load(f)
    objs = w.get("objects") or {}
    ms = [o.get("mass_kg") for o in objs.values()]
    fr = [o.get("static_friction") for o in objs.values()]
    same = (ms and None not in ms and None not in fr
            and max(ms) - min(ms) < 1e-4 and max(fr) - min(fr) < 1e-3)
    if same:
        os.environ["POUR_OBJ_MASS"] = f"{ms[0]:.6g}"
        os.environ["POUR_OBJ_FRIC"] = f"{fr[0]:.6g}"
    else:
        os.environ["POUR_OBJ_MASS"] = ""
        os.environ["POUR_OBJ_FRIC"] = ""
    pad = (w.get("robot") or {}).get("pad_friction")
    os.environ["POUR_PAD_FRIC"] = f"{pad:.6g}" if pad is not None else "3.0"
    print(f"[world] 物理参数按出生世界还原: 物体质量={os.environ['POUR_OBJ_MASS'] or '母带原生'} "
          f"物体摩擦={os.environ['POUR_OBJ_FRIC'] or '母带原生'} "
          f"指垫摩擦={os.environ['POUR_PAD_FRIC']}"
          + ("" if pad is not None else " (world.json 未记指垫, 按旧默认 3.0)"), flush=True)
    return w
