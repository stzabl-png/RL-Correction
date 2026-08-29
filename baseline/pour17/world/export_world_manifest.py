"""导出 Pour17 世界指纹 (baseline bundle 用)。

只导出"世界"层面的可复现事实: 时间步/物理材质/质量/碰撞/传感器/关节表/桌面。
**不导出** ours 的方法(reward 权重、obs/action 维度语义、GraspPose、confidence 口径)。

用法:
  cd <repo>
  SHARPA_WANDB=0 RL_ISAAC_NO_GUARD=1 PYTHONPATH=. \\
    $PY baseline/pour17/world/export_world_manifest.py --headless \\
        --out baseline/pour17/world/world_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--out", default="baseline/pour17/world/world_manifest.json")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
os.environ.setdefault("POUR_NO_D6", "0")

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("world_manifest")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

_CW = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "..",
                                   "tasks", "Pour", "17", "C_Wiring"))
sys.path.insert(0, _CW)
import pour_env as PE  # noqa: E402


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


cfg = PE.build_cfg(num_envs=2)
E = PE.PourEnv(cfg)
sim = E.sim
scn = E.scene
M = {}

# ---- 1. 时间步 ----
pcfg = getattr(E.cfg, "sim", None)
M["time"] = {
    "physics_dt_s": _f(sim.get_physics_dt()),
    "control_dt_s": _f(sim.get_physics_dt() * int(getattr(E.cfg, "decimation", 1))),
    "decimation": int(getattr(E.cfg, "decimation", 1)),
    "render_interval": int(getattr(pcfg, "render_interval", -1)) if pcfg else None,
    "episode_length_s": _f(getattr(E.cfg, "episode_length_s", None)),
}

# ---- 2. PhysX 求解器 ----
px = getattr(pcfg, "physx", None) if pcfg else None
M["physx"] = {k: (_f(getattr(px, k)) if isinstance(getattr(px, k, None), float)
                  else getattr(px, k, None))
              for k in ("solver_type", "min_position_iteration_count",
                        "max_position_iteration_count",
                        "min_velocity_iteration_count",
                        "max_velocity_iteration_count", "bounce_threshold_velocity",
                        "friction_offset_threshold", "friction_correlation_distance",
                        "gpu_max_rigid_contact_count", "enable_ccd",
                        "enable_stabilization")} if px else {}
M["physx"]["gravity"] = list(getattr(pcfg, "gravity", (0, 0, -9.81))) if pcfg else None

# ---- 3. 桌面 ----
M["table"] = {
    "table_top_z_m": _f(getattr(E.cfg, "table_top_z", None)),
    "table_size_m": list(getattr(E.cfg, "table_size", ())),
    "static_friction": 0.5, "dynamic_friction": 0.5,
    "note": "桌面板 = 静态 Cuboid, 中心 z = table_top_z - size_z/2; 上表面 = table_top_z",
}

# ---- 4. 物体(运行时读真值) ----
objs = {}
for key, art, oid in (("object_1_bottle_primary", E.object, "object_1"),
                      ("object_0_cup_aux", E.aux, "object_0")):
    d = {}
    try:
        d["mass_kg"] = _f(art.root_physx_view.get_masses()[0].sum())
    except Exception:
        d["mass_kg"] = None
    try:
        mat = art.root_physx_view.get_material_properties()[0]
        d["static_friction"] = _f(mat[0][0])
        d["dynamic_friction"] = _f(mat[0][1])
        d["restitution"] = _f(mat[0][2])
    except Exception:
        pass
    try:
        d["rest_pose_env_frame_xyz_wxyz"] = [
            _f(v) for v in E.rest_pose[1 if oid == "object_1" else 0].cpu().numpy()]
    except Exception:
        pass
    d["prim_path"] = getattr(art.cfg, "prim_path", None)
    sp = getattr(art.cfg, "spawn", None)
    d["usd_loaded"] = getattr(sp, "usd_path", None)
    objs[key] = d
M["objects"] = objs

# ---- 5. 机器人 ----
jn = list(E.hand.joint_names)
ctrl = [jn[i] for i in E.map_ids_t.cpu().tolist()]
M["robot"] = {
    "prim_path": getattr(E.hand.cfg, "prim_path", None),
    "usd": getattr(getattr(E.hand.cfg, "spawn", None), "usd_path", None),
    "num_joints_articulation": len(jn),
    "num_controlled_joints": len(ctrl),
    "controlled_joint_names_in_order": ctrl,
    "controlled_layout": "[R_arm 7, L_arm 7, right_fingers 22, left_fingers 22]",
    # ★不可读时记 None 而非 False: "读不到"与"关着"是两回事 (2026-08-29 实测,
    # cfg 写 True 而此处记了 False, 差点把错值交出去)
    "self_collision": getattr(getattr(getattr(E.hand.cfg, "spawn", None),
                                      "articulation_props", None),
                              "enabled_self_collisions", None),

    "body_names": list(E.hand.body_names),
}
try:
    lim = E.hand.data.soft_joint_pos_limits[0][E.map_ids_t].cpu().numpy()
    M["robot"]["controlled_joint_limits_rad"] = [
        [_f(a), _f(b)] for a, b in lim]
except Exception:
    pass

# ---- 6. 传感器 ----
sens = []
for i, s in enumerate(E._all_sensors):
    c = s.cfg
    sens.append({"idx": i, "prim_path": getattr(c, "prim_path", None),
                 "history_length": getattr(c, "history_length", None),
                 "filter_prim_paths_expr": list(
                     getattr(c, "filter_prim_paths_expr", []) or []),
                 "update_period": _f(getattr(c, "update_period", None))})
M["sensors"] = {
    "count": len(sens),
    "pad_force_threshold_N": PE.PAD_FTH,
    "pads_min_per_hand": PE.PADS_MIN,
    "detail": sens,
}

# ---- 7. 场景与开关 ----
M["scene"] = {
    "num_envs_in_export": int(scn.num_envs),
    "env_spacing_m": _f(getattr(E.cfg.scene, "env_spacing", None)),
    "replicate_physics": bool(getattr(E.cfg.scene, "replicate_physics", False)),
}
M["switches"] = {
    "approach_only": bool(getattr(E.cfg, "approach_only", False)),
    "obj_jitter_xy": _f(getattr(E.cfg, "obj_jitter_xy", None)),
    "friction_curriculum": bool(getattr(E.cfg, "friction_curriculum", False)),
    "friction_hi": _f(getattr(E.cfg, "friction_hi", None)),
    "friction_lo": _f(getattr(E.cfg, "friction_lo", None)),
    "POUR_NO_D6_effective": os.environ.get("POUR_NO_D6") == "1",
    "note": "POUR_NO_D6=1 时不装 D6 跨侧互撞传感器(1-env 回放规避 PhysX 断言); "
            "训练默认不设该变量, 即装 D6。",
}

# ---- 8. 参考母带(只记指纹, 不解释内容) ----
import hashlib  # noqa: E402
try:
    with open(PE.MASTER, "rb") as fh:
        b = fh.read()
    M["reference_tape"] = {
        "path": PE.MASTER, "md5": hashlib.md5(b).hexdigest(),
        "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
except Exception:
    pass

out = os.path.abspath(args.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    json.dump(M, fh, indent=2, ensure_ascii=False)
print(f"[manifest] 已写 {out}", flush=True)
for k in ("time", "table", "switches"):
    print(f"[manifest] {k}: {json.dumps(M.get(k, {}), ensure_ascii=False)}", flush=True)
print(f"[manifest] 受控关节 {len(ctrl)} 个, 传感器 {len(sens)} 个, "
      f"物体 {list(objs)}", flush=True)
print("[manifest] 完毕", flush=True)
try:
    _slot.release()
except Exception:
    pass
sys.stdout.flush()
app.close()
os._exit(0)
