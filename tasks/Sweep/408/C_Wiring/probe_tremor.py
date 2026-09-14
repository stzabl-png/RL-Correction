"""Sweep408 手抖诊断探针 (2026-09-12)。

为什么要新写一个: record_grip 只在**控制步** (20Hz) 采样, 抖动在 5~25Hz, 会严重别频。
本探针给 `_apply_action` 挂钩子, 在**每个物理子步** (240Hz) 记一次关节状态,
这样才能分辨 "指令本来就抖" / "指令平滑但执行器振铃" / "接触求解器噪声"。

旋钮 (一次只动一个, 跑对照):
  --zero_action        不用策略, 残差恒 0 (纯母带前馈) —— 隔离"抖是策略造的吗"
  --damp_scale X       臂阻尼 arm_damping_scale = X (默认 0.02, 即阻尼比 0.1)
  --vel_iters N        solver_velocity_iteration_count (默认 0)
  --obj_mass M         物体质量 kg (默认 0.15)
  --fin_dev R          手指累积残差上限 rad (默认 0.60)

落盘 <out>.npz: qcmd/qact/qvel (T_sub, N, 58) + 控制步标记 + 掌系漂移 + 指垫力。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default=None)
p.add_argument("--out", required=True)
p.add_argument("--num_envs", type=int, default=2)
p.add_argument("--release_row", type=int, default=10)
p.add_argument("--steps", type=int, default=200, help="最多记多少个控制步")
p.add_argument("--zero_action", action="store_true")
p.add_argument("--damp_scale", type=float, default=None)
p.add_argument("--vel_iters", type=int, default=None)
p.add_argument("--obj_mass", type=float, default=None)
p.add_argument("--fin_dev", type=float, default=None)
p.add_argument("--tag", default="")
p.add_argument("--ff_interp", action="store_true",
               help="臂前馈在 12 个物理子步内从上一行**线性插值**到本行 (把 1° 台阶摊成 0.09°/子步), 不改奖励不改策略")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import task_config as TC  # noqa: E402
if args.obj_mass is not None:
    TC.PHYS["POUR_OBJ_MASS"] = f"{args.obj_mass}"
for k, v in TC.PHYS.items():
    os.environ[k] = v
os.environ.setdefault("SHARPA_WANDB", "0")
if args.fin_dev is not None:
    TC.FIN_DEV = float(args.fin_dev)

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("sweep408_tremor")
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import grip_env as GE  # noqa: E402

cfg = GE.build_cfg(args.num_envs)
# ---- 旋钮 ----
if args.damp_scale is not None:
    cfg.arm_damping_scale = float(args.damp_scale)
    cfg.apply_gains()                      # ⚠ 必须重调, __post_init__ 已经跑过 (cfg 注释)
if args.vel_iters is not None:
    n = int(args.vel_iters)
    cfg.sim.physx.max_velocity_iteration_count = max(n, cfg.sim.physx.max_velocity_iteration_count)
    ap = cfg.robot_cfg.spawn.articulation_props
    if ap is not None:
        ap.solver_velocity_iteration_count = n
        ap.solver_position_iteration_count = max(getattr(ap, "solver_position_iteration_count", None) or 8, 8)
    for oc in (cfg.object_cfg, getattr(cfg, "aux_cfg", None)):
        if oc is not None and getattr(oc.spawn, "rigid_props", None) is not None:
            oc.spawn.rigid_props.solver_velocity_iteration_count = n
def _vi():
    ap = cfg.robot_cfg.spawn.articulation_props
    v = getattr(ap, "solver_velocity_iteration_count", None) if ap is not None else None
    return v if v is not None else f"sim默认{cfg.sim.physx.max_velocity_iteration_count}"


raw = GE.Sweep408GripEnv(cfg)
raw.release_row_cur = int(args.release_row)
TC.RELEASE_JITTER = 0

KP = {g: dict(cfg.robot_cfg.actuators[g].stiffness) for g in ("arm_shoulder", "arm_elbow", "arm_wrist")}
KD = {g: dict(cfg.robot_cfg.actuators[g].damping) for g in ("arm_shoulder", "arm_elbow", "arm_wrist")}
print(f"[tremor] tag={args.tag} zero_action={args.zero_action} damp_scale={getattr(cfg,'arm_damping_scale',None)} "
      f"vel_iters={_vi()} "
      f"obj_mass={os.environ.get('POUR_OBJ_MASS')} fin_dev={TC.FIN_DEV}", flush=True)
print(f"[tremor] 臂增益 kp={ {k: round(list(v.values())[0],1) for k,v in KP.items()} } "
      f"kd={ {k: round(list(v.values())[0],3) for k,v in KD.items()} }", flush=True)

agent = None
if not args.zero_action:
    import yaml
    from rl_rebuild.algo.ppo.ppo import PPO
    from rl_rebuild.wrapper.config_wrapper import ConfigWrapper
    from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
    _HERE = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(_HERE, "ppo_grip.yaml")) as f:
        acfg = yaml.safe_load(f)
    acfg["algorithm"]["num_actors"] = args.num_envs
    env = GymStyleEnvWrapper(raw, clip_actions=1.0)
    agent = PPO(env, output_dir=os.path.join(os.environ.get("TMPDIR", "/tmp"), "s408_tremor"),
                full_config=ConfigWrapper(acfg, {}, test=True), create_output_dir=False)
    agent.restore_test(os.path.abspath(args.checkpoint))
    agent.set_eval()
else:
    from rl_rebuild.wrapper.sharpa_wave_env_wrapper import GymStyleEnvWrapper
    env = GymStyleEnvWrapper(raw, clip_actions=1.0)

# ---- 每物理子步挂钩 ----
AID = raw.act_ids
SUB = {"qcmd": [], "qact": [], "qvel": [], "ctl": [], "palm": [], "obj": []}
_ctl = {"i": -1}
_orig_apply = raw._apply_action


_ARM = AID[:14]
_ff = {"prev": None, "sub": 0, "ctl": -1}


def _apply_hooked():
    if args.ff_interp:
        if _ctl["i"] != _ff["ctl"]:
            _ff["ctl"] = _ctl["i"]; _ff["sub"] = 0
            _ff["tgt"] = raw.q_cmd[:, _ARM].detach().clone()       # 本控制步的目标 (含残差)
            if _ff["prev"] is None:
                _ff["prev"] = _ff["tgt"].clone()
        _ff["sub"] += 1
        w = min(_ff["sub"] / float(cfg.decimation), 1.0)
        raw.q_cmd[:, _ARM] = (1.0 - w) * _ff["prev"] + w * _ff["tgt"]
        if w >= 1.0:
            _ff["prev"] = _ff["tgt"].clone()
    _orig_apply()
    SUB["qcmd"].append(raw.q_cmd[:, AID].detach().clone())
    SUB["qact"].append(raw.hand.data.joint_pos[:, AID].detach().clone())
    SUB["qvel"].append(raw.hand.data.joint_vel[:, AID].detach().clone())
    SUB["ctl"].append(_ctl["i"])
    # 可见的"手抖"是掌心刚体在世界里的运动, 不是关节角 —— 一起记
    SUB["palm"].append(torch.cat([torch.cat(raw._hand_pose(s_), 1) for s_ in GE.SIDES], 1).detach().clone())
    SUB["obj"].append(torch.cat([torch.cat(raw._obj_pose(s_), 1) for s_ in GE.SIDES], 1).detach().clone())


raw._apply_action = _apply_hooked

obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]
CTL = {k: [] for k in ("dp_r", "dr_r", "dp_l", "dr_l", "cert", "k", "row", "F")}
_done_at = np.full(args.num_envs, -1, dtype=int)
with torch.no_grad():
    for t in range(args.steps):
        _ctl["i"] = t
        if agent is None:
            act = torch.zeros(args.num_envs, GE.ACT_DIM, device=raw.device)
        else:
            inp = {"obs": agent.running_mean_std(obs["obs"]), "priv_info": obs["priv_info"]}
            act = torch.clamp(agent.model.act_inference(inp), -1.0, 1.0)
        obs, rew, dones, infos = env.step(act)
        tk = raw._tick
        f = lambda x: x.detach().float().cpu().numpy()
        CTL["dp_r"].append(f(tk["dp"]["right"])); CTL["dr_r"].append(f(tk["dr"]["right"]))
        CTL["dp_l"].append(f(tk["dp"]["left"])); CTL["dr_l"].append(f(tk["dr"]["left"]))
        CTL["cert"].append(f(tk["cert"])); CTL["k"].append(f(tk["k"])); CTL["row"].append(f(tk["row"]))
        CTL["F"].append(np.concatenate([f(tk["F"]["right"]), f(tk["F"]["left"])], 1))
        dn = dones.reshape(-1).bool().cpu().numpy()
        _done_at[:] = np.where(dn & (_done_at < 0), t, _done_at)
        if (_done_at >= 0).all():
            print(f"[tremor] 全部 env 结束于控制步 {_done_at.tolist()}", flush=True)
            break

out = os.path.abspath(args.out)
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez_compressed(
    out + ".npz",
    qcmd=torch.stack(SUB["qcmd"]).cpu().numpy().astype(np.float32),
    qact=torch.stack(SUB["qact"]).cpu().numpy().astype(np.float32),
    qvel=torch.stack(SUB["qvel"]).cpu().numpy().astype(np.float32),
    ctl=np.asarray(SUB["ctl"], dtype=np.int32),
    palm=torch.stack(SUB["palm"]).cpu().numpy().astype(np.float32),
    obj=torch.stack(SUB["obj"]).cpu().numpy().astype(np.float32),
    **{k: np.asarray(v, dtype=np.float32) for k, v in CTL.items()},
    done_at=_done_at,
    meta=json.dumps(dict(tag=args.tag, zero_action=args.zero_action,
                         damp_scale=getattr(cfg, "arm_damping_scale", None),
                         vel_iters=_vi(),
                         obj_mass=os.environ.get("POUR_OBJ_MASS"), fin_dev=TC.FIN_DEV,
                         dt=raw.dt, decimation=cfg.decimation, release_row=args.release_row,
                         kp={k: list(v.values())[0] for k, v in KP.items()},
                         kd={k: list(v.values())[0] for k, v in KD.items()})),
)
nsub = len(SUB["qcmd"])
print(f"[tremor] → {out}.npz  子步 {nsub} (控制步 {len(CTL['k'])}, 每控制步 {nsub/max(len(CTL['k']),1):.1f} 子步)", flush=True)
sys.stdout.flush()
os._exit(0)
