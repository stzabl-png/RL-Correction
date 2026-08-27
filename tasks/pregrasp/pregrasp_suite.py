"""PreGrasp 三合一体检 (单次 Isaac): 静态摆位 → cuRobo 规划 → 零动作回放 → 对比。

只启动一次 Isaac(省两次冷启动 ~5 分钟), 全程无 GUI, 末尾输出:
  [1静态] 双侧 PreGrasp 腕位(IK) + 静态指尖离桌高度
  [2规划] worker 验收摘要 (终点误差/腕最低/行程)
  [3回放] 零动作全程: 腕最近位 / 指尖全程最低 / 终止统计
  [对比]  三方左右腕坐标并排 + 两两偏差(cm)

    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.pregrasp_suite --headless \
        --act_dist 0.015 --table_pad 0.0
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--clip", default="Pour17_bottle")
p.add_argument("--grasp_prior", default="tasks/pregrasp/priors/Pour17_bottle.npz")
p.add_argument("--prior_yaw", type=float, default=19.5)
p.add_argument("--prior_b", default="tasks/pregrasp/priors/Pour17_cup.npz")
p.add_argument("--prior_b_yaw", type=float, default=90.0)
p.add_argument("--out_npz", default="tasks/pregrasp/priors/curobo_pour17_pregrasp0_ref.npz")
p.add_argument("--act_dist", type=float, default=0.015)
p.add_argument("--table_pad", type=float, default=0.0)
p.add_argument("--ff_freeze_cm", type=float, default=0.0)
p.add_argument("--ref_stride", type=int, default=2)
p.add_argument("--num_envs", type=int, default=2)
p.add_argument("--finger_lock", choices=["open", "pose1"], default="open",
               help="规划时手指锁在什么构型: open=伸直(旧行为) / pose1=各侧先验"
                    " pregrasp[0] (拇指对掌, 碰撞球包按真实手型)")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()

from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("pregrasp_suite")
app = AppLauncher(args).app

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

import numpy as _np  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl_rebuild.correction import clips  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402
from tasks.pregrasp import bimanual as BM  # noqa: E402
from tasks.pregrasp.bimanual_env import BimanualApproachEnv  # noqa: E402
from tasks.pregrasp.cfg import GraspTaskCfg, apply_grasp_prior  # noqa: E402
from tasks.pregrasp.env import GENERIC_JOINT_ORDER, GENERIC_OPEN  # noqa: E402

BAR = "=" * 70

# ---------------- 场景组装 (与 check_ff_bi/train 逐项一致, approach_only) ----------------
env_cfg = GraspTaskCfg()
clips.configure_cfg(env_cfg, args.clip)
env_cfg.approach_only = True
env_cfg.action_space = 7
apply_grasp_prior(env_cfg, args.grasp_prior, args.prior_yaw, approach=True)
env_cfg.direct_grasp_prob = 0.0
env_cfg.approach_t0_max = 0.0
env_cfg.minimal_no_ff = True
env_cfg.minimal_fixed_res = False
env_cfg.eps_pos, env_cfg.eps_rot = 0.01, _np.radians(15.0)
env_cfg.eps_pos0, env_cfg.eps_rot0 = env_cfg.eps_pos, env_cfg.eps_rot
env_cfg.w_imit0_approach = 0.0
env_cfg.w_imit_ramp = 0.0
env_cfg.stance_prob, env_cfg.retract_ratio = 1.0, 1.0
env_cfg.curobo_ref_npz = os.path.abspath(args.out_npz)   # 先用旧的建 env, 规划后热切换
env_cfg.minimal_no_ff = False
env_cfg.curobo_ff_freeze_cm = args.ff_freeze_cm
env_cfg.curobo_ff_pull = 0.0
env_cfg.curobo_ref_stride = int(args.ref_stride)
env_cfg.prior_b_npz = os.path.abspath(args.prior_b)
env_cfg.prior_b_yaw_deg = float(args.prior_b_yaw)
_a1, _o1 = env_cfg.action_space, env_cfg.observation_space
env_cfg.action_space, env_cfg.observation_space = 2 * _a1, 2 * _o1
env_cfg._obs_single = _o1
env_cfg.scene.num_envs = args.num_envs
env_cfg.obj_jitter_xy = 0.0

env = BimanualApproachEnv(env_cfg)
env.gentle = 1.0
env.reset()
hand = env.hand
jn = list(hand.joint_names)
bn = list(hand.body_names)
W = env.scene.env_origins[0].cpu().numpy().astype(np.float64)
TZ = float(getattr(env_cfg, "table_top_z", 0.85))
_tips = {s: [i for i, n in enumerate(bn)
             if n.startswith(f"{s}_") and n.endswith("_DP")]
         for s in ("right", "left")}
REC = {}      # 三方坐标记录: REC[("静态"|"规划"|"回放", "R"|"L")] = np.array(3)

# ================= 1) 静态摆位: IK 到 PreGrasp, 记录腕位 + 指尖高度 =================
print(f"\n{BAR}\n[1静态] PreGrasp 摆位 (掌心锚点, 指伸直)\n{BAR}")
q = hand.data.default_joint_pos[0].clone()
for side_name, side in ((env._A_name, env._A), (env._B_name, env._B)):
    with BM.use_side(env, side):
        pw = env._pregrasp_w[0]
        anchor_T = env._anchor_T
    pos = np.asarray(pw[0], np.float64)
    quat = np.asarray(pw[1], np.float64)
    quat = quat / max(np.linalg.norm(quat), 1e-12)
    tag = side_name[0].upper()
    REC[("静态", tag)] = pos.copy()
    ik = ArmIK(side_name, anchor_link="arm_center", anchor_T=anchor_T)
    pfx = "R" if side_name == "right" else "L"
    aid = [jn.index(f"{pfx}_arm_j{i}") for i in range(1, 8)]
    r = ik.solve(pos, quat_to_R(quat), q0=q[aid].cpu().numpy().astype(np.float64),
                 iters=300)
    print(f"[1静态] {tag} 腕目标 {np.round(pos, 4).tolist()} m | "
          f"IK {'✅' if r['ok'] else '❌'} 残差 {r['pos_err']*100:.2f}cm")
    for i, jid in enumerate(aid):
        q[jid] = float(r["q"][i])
    for nname, val in zip(GENERIC_JOINT_ORDER, GENERIC_OPEN):
        q[jn.index(nname.replace("right_", f"{side_name}_"))] = float(val)
qb = q.unsqueeze(0).expand(args.num_envs, -1).contiguous()
hand.write_joint_state_to_sim(qb, torch.zeros_like(qb))
hand.set_joint_position_target(qb)
hand.write_data_to_sim()
env.sim.step(render=False)
env.scene.update(env.sim.get_physics_dt())
for s in ("right", "left"):
    z0 = float(hand.data.body_pos_w[0, _tips[s], 2].min())
    print(f"[1静态] {s[0].upper()} 指尖最低 {z0:.4f}m = 桌面上方 {(z0-TZ)*100:+.1f}cm")

# ================= 2) 规划: 组 targets json -> worker 子进程 =================
print(f"\n{BAR}\n[2规划] cuRobo 联合规划 (act_dist={args.act_dist} "
      f"table_pad={args.table_pad})\n{BAR}")
A_name, B_name = env._A_name, env._B_name
_e = clips.clip_entry(args.clip)
_sec = _e.get("secondary") or {}
_sx, _sy, _sz = env_cfg.table_size
goals, pregoals, objs = {}, {}, []
with BM.use_side(env, env._A):
    A_obj = env.obj_init_pos.cpu().numpy().astype(np.float64)
    A_oq = env.obj_init_quat.cpu().numpy().astype(np.float64)
    goals[f"{A_name}_hand_C_MC"] = (env._grasp_pos_w.cpu().numpy().astype(np.float64),
                                    env._grasp_quat_w.cpu().numpy().astype(np.float64))
    pregoals[f"{A_name}_hand_C_MC"] = (np.asarray(env._pregrasp_w[0][0], np.float64),
                                       np.asarray(env._pregrasp_w[0][1], np.float64))
with BM.use_side(env, env._B):
    B_obj = env.obj_init_pos.cpu().numpy().astype(np.float64)
    B_oq = env.object.data.root_quat_w[0].cpu().numpy().astype(np.float64)
    goals[f"{B_name}_hand_C_MC"] = (env._grasp_pos_w.cpu().numpy().astype(np.float64),
                                    env._grasp_quat_w.cpu().numpy().astype(np.float64))
    pregoals[f"{B_name}_hand_C_MC"] = (np.asarray(env._pregrasp_w[0][0], np.float64),
                                       np.asarray(env._pregrasp_w[0][1], np.float64))
objs = [{"name": "obj_primary", "mesh": _e["mesh"],
         "pos": A_obj.tolist(), "quat": A_oq.tolist()}]
if _sec.get("mesh"):
    objs.append({"name": "obj_secondary", "mesh": _sec["mesh"],
                 "pos": B_obj.tolist(), "quat": B_oq.tolist()})
# ⚠ 坐标一律用**env 系**(origin 相对), 转 cuRobo 底座系只减 _root_p, 不许再 +W:
#   旧公式 (p_env + W − root_p) 在 W=0(单环境)下碰巧对, 多环境时 W 被双计
#   (2026-08-18 实测: 套件 2 env 目标整体 +1.0m, 规划必失败)
_root_p = (hand.data.root_pos_w[0].cpu().numpy().astype(np.float64) - W)
# 手指锁值 (2026-08-19): open=伸直(旧行为); pose1=各侧先验 pregrasp[0] (Dexonomy
# 模板起点, 拇指对掌) —— 规划器按真实手型的碰撞球留净空。病根: 伸直锁值规划出的
# 净空对 Pose1 手型失效, 成形后的手在 transit 段撞物 (GUI --full 演示实证)。
if getattr(args, "finger_lock", "open") == "pose1":
    _lock_fin_R = np.asarray(np.load(args.grasp_prior)["pregrasp"][0][7:29], float)
    _lock_fin_L = np.asarray(np.load(args.prior_b)["pregrasp"][0][7:29], float)
    print("[suite] 手指锁值 = Pose1 (双侧先验 pregrasp[0], 拇指对掌构型)")
else:
    _lock_fin_R = _lock_fin_L = np.asarray(GENERIC_OPEN, float)
_tmp = tempfile.mkdtemp(prefix="pregrasp_suite_")
_tgt, _out = os.path.join(_tmp, "targets.json"), os.path.join(_tmp, "plan.npz")
with open(_tgt, "w") as f:
    json.dump({
        "table_pose": [float(-_root_p[0]), float(-_root_p[1]),
                       float(env_cfg.table_top_z - _sz / 2 - _root_p[2])],
        "table_dims": [float(_sx), float(_sy), float(_sz)],
        "objects": [dict(o, pos=(np.asarray(o["pos"]) - _root_p).tolist())
                    for o in objs],
        "start_joints": dict(
            {n: float(v) for n, v in zip(
                jn, hand.data.default_joint_pos[0].cpu().numpy().astype(float))},
            **{"torso_j1": float(np.radians(45.0)),
               "torso_j2": float(np.radians(90.0)), "torso_j3": 0.0,
               "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0},
            **{n.replace("right_", f"{_sd}_"): float(v)
               for _sd, _fv in (("right", _lock_fin_R), ("left", _lock_fin_L))
               for n, v in zip(GENERIC_JOINT_ORDER, _fv)}),
        "lock_joints": ["torso_j1", "torso_j2", "torso_j3",
                        "head_j1", "head_j2", "head_j3"]
                       + [n for n in jn if n.startswith(("right_", "left_"))],
        "tool_frames": [f"{A_name}_hand_C_MC", f"{B_name}_hand_C_MC"],
        "hand_targets": {f"{A_name}_hand_C_MC": "obj_primary",
                         f"{B_name}_hand_C_MC": "obj_secondary"},
        "goals": {k: {"pos": (v[0] - _root_p).tolist(), "quat": v[1].tolist()}
                  for k, v in goals.items()},
        "pregrasp_goals": {k: {"pos": (v[0] - _root_p).tolist(),
                               "quat": v[1].tolist()}
                           for k, v in pregoals.items()},
    }, f)
subprocess.run([sys.executable, "-u", "-m", "tasks.pregrasp.curobo_plan_worker",
                "--targets", _tgt, "--out", _out,
                "--act_dist", str(args.act_dist), "--table_pad", str(args.table_pad),
                "--joint", "1"],
               cwd=os.getcwd(), env=dict(os.environ, PYTHONPATH=os.getcwd()),
               timeout=2400)
_z = np.load(_out, allow_pickle=True)
if not bool(_z["ok"]):
    print(f"[2规划] ❌ 失败: {_z.get('failed_frame')} —— 不再回放, 退出")
    try:
        _slot.release()
    except Exception:
        pass
    import os as _os_exit  # noqa: E402
    _os_exit._exit(1)      # 硬退且带真实错误码 (2026-08-20: 假成功吞码案同修)
# 规划终点 (worker FK, env 系) 记入对比
for k, v in pregoals.items():
    tag = "R" if k.startswith("right") else "L"
    _ee = np.asarray(_z[f"end_expect_{k}"], np.float64) + _root_p
    REC[("规划", tag)] = _ee
# 保存 + 热切换 retract_path
import shutil  # noqa: E402
_save = {"right_q": None, "left_q": None}
traj = np.asarray(_z["traj"], np.float32)
names = [str(x) for x in _z["joint_names"]]
for sd in ("right", "left"):
    pfx = "R" if sd == "right" else "L"
    idx = [names.index(f"{pfx}_arm_j{i}") for i in range(1, 8)]
    _save[f"{sd}_q"] = traj[:, idx]
np.savez(args.out_npz, right_q=_save["right_q"], left_q=_save["left_q"],
         seg_frames=_z["seg_frames"], seg_lens=_z["seg_lens"],
         joint_mode=1, source="pregrasp_suite")
print(f"[2规划] ✅ 已存 {args.out_npz} (右 {_save['right_q'].shape} "
      f"左 {_save['left_q'].shape})")
_st = int(args.ref_stride)
for sd_name, side in ((A_name, env._A), (B_name, env._B)):
    _qp = torch.tensor(np.asarray(_save[f"{sd_name}_q"], np.float32)[::_st],
                       device=env.device)
    _q_st = hand.data.default_joint_pos[0, side.data["arm_jids"]]
    _d0 = float((_qp[0] - _q_st).abs().max())
    assert _d0 < 0.02, f"{sd_name} 新参考首帧偏站姿 {np.degrees(_d0):.2f}°"
    side.data["retract_path"] = _qp
env.retract_path = env._A.data["retract_path"]

# ================= 3) 零动作回放 (新参考) =================
print(f"\n{BAR}\n[3回放] 零动作 x ep_total (stride {_st})\n{BAR}")
env.reset()
z = torch.zeros((args.num_envs, env_cfg.action_space), device=env.device)
_best, _tipmin = {}, {}
_obj0 = {}
for tag, side in (("瓶", env._A), ("杯", env._B)):
    with BM.use_side(env, side):
        _obj0[tag] = (env.object.data.root_pos_w[0]
                      - env.scene.env_origins[0]).cpu().numpy().copy()
for tag, side in (("R", env._A), ("L", env._B)):
    with BM.use_side(env, side):
        _jq = env.hand.data.joint_pos[0, side.data["arm_jids"]].cpu().numpy()
        _r0 = side.data["retract_path"][0].cpu().numpy()
        _st0 = hand.data.default_joint_pos[0, side.data["arm_jids"]].cpu().numpy()
    print(f"[步0] {tag} 实测vs参考首帧 逐关节(°): "
          f"{np.round(np.degrees(_jq - _r0), 1).tolist()}")
    print(f"[步0] {tag} 参考首帧vs站姿 逐关节(°): "
          f"{np.round(np.degrees(_r0 - _st0), 1).tolist()}")
n = env.ep_total
for i in range(n):
    env.step(z)
    if i in (10, 30):
        # 终极分叉判据: 实测臂关节 vs 参考航点 —— 吻合=坐标口径问题, 不吻合=前馈层bug
        for tag, side, sname in (("R", env._A, "right"), ("L", env._B, "left")):
            with BM.use_side(env, side):
                _jq = env.hand.data.joint_pos[0, side.data["arm_jids"]].cpu().numpy()
                _rp = side.data["retract_path"]
                _wi = min(i, _rp.shape[0] - 1)
                _ref = _rp[_wi].cpu().numpy()
            _dd = np.degrees(np.abs(_jq - _ref)).max()
            print(f"[分叉] 步{i} {tag}: 关节vs航点{_wi} 最大偏差 {_dd:.1f}° "
                  f"{'✓跟踪中' if _dd < 5 else '❌没在跟这条参考'} | 逐关节 "
                  f"{np.round(np.degrees(_jq - _ref), 1).tolist()}")
    for tag, side, sname in (("R", env._A, "right"), ("L", env._B, "left")):
        with BM.use_side(env, side):
            d = float((env._anchor_w() - env._target_w()).norm(dim=1)[0])
            wp = (env._anchor_w()[0] - env.scene.env_origins[0]).cpu().numpy().copy()
        if tag not in _best or d < _best[tag][0]:
            _best[tag] = (d, wp)
        z0 = float(hand.data.body_pos_w[0, _tips[sname], 2].min())
        if tag not in _tipmin or z0 < _tipmin[tag][0]:
            _tipmin[tag] = (z0, i)
for tag, (d, wp) in _best.items():
    REC[("回放", tag)] = np.asarray(wp, np.float64)
    print(f"[3回放] {tag} 腕最近位 {np.round(wp, 4).tolist()} m | 距抓姿目标 {d*100:.2f}cm")
for tag, (z0, st) in _tipmin.items():
    print(f"[3回放] {tag} 指尖全程最低 {z0:.4f}m = 桌面上方 {(z0-TZ)*100:+.1f}cm "
          f"(@步 {st}) {'⚠ 已触桌' if z0 <= TZ else ''}")
# 物体漂移探针 (用户 2026-08-18 疑问: 回放中杯子的实际位姿 ≠ 静态/靶点假设的位姿?)
for tag, side in (("瓶", env._A), ("杯", env._B)):
    with BM.use_side(env, side):
        _pe = (env.object.data.root_pos_w[0]
               - env.scene.env_origins[0]).cpu().numpy()
    _d = (_pe - _obj0[tag]) * 100
    print(f"[3回放] {tag} 位姿漂移: 复位时 {np.round(_obj0[tag], 4).tolist()} -> "
          f"回放末 {np.round(_pe, 4).tolist()} | Δ=({_d[0]:+.1f},{_d[1]:+.1f},"
          f"{_d[2]:+.1f})cm {'⚠ 被动过' if np.linalg.norm(_d) > 0.5 else '✓ 没动'}")

# ================= 对比 =================
print(f"\n{BAR}\n[对比] 三方腕坐标 (m, env 世界系)\n{BAR}")
for tag in ("R", "L"):
    print(f"  {tag}:")
    for src in ("静态", "规划", "回放"):
        v = REC.get((src, tag))
        print(f"    {src}: {np.round(v, 4).tolist() if v is not None else '缺'}")
    a, b, c = (REC.get((s, tag)) for s in ("静态", "规划", "回放"))
    if a is not None and b is not None:
        print(f"    静态↔规划 {np.linalg.norm(a-b)*100:.2f}cm", end="")
    if b is not None and c is not None:
        print(f" | 规划↔回放 {np.linalg.norm(b-c)*100:.2f}cm", end="")
    if a is not None and c is not None:
        print(f" | 静态↔回放 {np.linalg.norm(a-c)*100:.2f}cm")
try:
    _slot.release()
    print("[suite] GPU 槽位已主动释放")
except Exception:
    pass
import os as _os_exit  # noqa: E402
_os_exit._exit(0)      # 硬退 (2026-08-20 三入口同修: app.close 挂死/假成功案)
