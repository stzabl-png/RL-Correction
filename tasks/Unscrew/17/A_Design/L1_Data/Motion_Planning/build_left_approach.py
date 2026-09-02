"""左臂 初始站姿 -> GraspPose 路径构建 (U-P1, 照 Pour L1-4 用户定稿设计):
  站姿 --cuRobo cspace 满障碍(桌+立瓶+盖)--> Dexonomy 原生 pregrasp 梯级0
  --PCHIP 单调样条六级梯 (腕指同步, 单次缓入缓出)--> 完整 GraspPose (母带站位行)
右臂全程站姿 (U11)。梯级位姿 = 站位腕 FK ∘ (prior grasp)⁻¹ ∘ prior pregrasp_k (刚性相对变换, 免场景换基)。
产物: LeftApproach_LD227.npz (left_q/left_f/right_q/right_f, fin_names=母带列序, seg 表)。
"""
import argparse, json, os, subprocess, sys, tempfile, time
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--form_frames", type=int, default=12, help="成形段每级梯插值帧数")
p.add_argument("--act_dist", type=float, default=0.015)
p.add_argument("--attempts", type=int, default=20)
p.add_argument("--obj_inflate", type=float, default=0.01)
p.add_argument("--out", default="tasks/Unscrew/17/A_Design/L1_Data/Motion_Planning/LeftApproach_LD227.npz")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
import numpy as np
from scipy.interpolate import PchipInterpolator
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "part4", "C_Wiring"))
os.environ.setdefault("POUR_NO_D6", "1")
import task_config as TC
import task_env as PE
from rl_rebuild.correction import clips
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER

def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])
def qconj(q): return q * np.array([1.0, -1.0, -1.0, -1.0])
def R2q(R):
    w = np.sqrt(max(0.0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2
    if w < 1e-8:
        i = int(np.argmax(np.diag(R))); j, k = (i+1)%3, (i+2)%3
        s = np.sqrt(max(1e-12, 1 + R[i,i] - R[j,j] - R[k,k])) * 2
        q = np.zeros(4); q[1+i] = s/4; q[1+j] = (R[j,i]+R[i,j])/s; q[1+k] = (R[k,i]+R[i,k])/s; q[0] = (R[k,j]-R[j,k])/s
        return q/np.linalg.norm(q)
    return np.array([w, (R[2,1]-R[1,2])/(4*w), (R[0,2]-R[2,0])/(4*w), (R[1,0]-R[0,1])/(4*w)])

cfg = PE.build_cfg(num_envs=1)
E = PE.UnscrewEnv(cfg); E.force_entry = [0]; E.reset()
for _ in range(40):
    E._SA.apply_screw(E, integrate_angle=False)
    E.scene.write_data_to_sim(); E.sim.step(render=False); E.scene.update(E.sim.get_physics_dt())
E._SA.apply_screw(E, integrate_angle=False)
W = E.scene.env_origins[0].cpu().numpy().astype(np.float64)
jn = list(E.hand.joint_names)
dq = E.hand.data.default_joint_pos[0].cpu().numpy().astype(float)
_root_p = E.hand.data.root_pos_w[0].cpu().numpy().astype(np.float64) - W

# ---- 母带: 站位左臂/左指 (fin_names 列序) ----
z2 = np.load(TC.REF_V2 if hasattr(TC, "REF_V2") else
             os.path.join(os.path.dirname(TC.REF_V1), "reference_v2.npz"), allow_pickle=True)
rows_ia = np.flatnonzero(np.asarray(z2["source"], np.int8) == 1)
IA0 = int(rows_ia[0])
q_station = np.asarray(z2["left_q"], np.float64)[IA0]
f_station = np.asarray(z2["left_f"], np.float64)[IA0]
fin_names = [str(n) for n in z2["fin_names"]]
perm = [GENERIC_JOINT_ORDER.index(n) for n in fin_names]   # GENERIC 值 -> 母带列序

# ---- prior 梯级 (刚性相对变换搬到站位系) ----
zp = np.load(TC.PRIOR_AUX)
g = np.asarray(zp["grasp"], np.float64)
pre = np.asarray(zp["pregrasp"], np.float64)
order = np.argsort(-np.linalg.norm(pre[:, :3] - g[:3], axis=1))   # 远->近
pre = pre[order]
import json as _json
_rest = _json.load(open(TC.REST_JSON))
def _mk_ik_left():
    if _rest.get("anchor_T_left"):
        return ArmIK("left", anchor_link="arm_center", anchor_T=np.asarray(_rest["anchor_T_left"], float))
    return ArmIK("left", torso_deg={"torso_j1": 40.5196, "torso_j2": 73.6595, "torso_j3": 0.3896})
ik = _mk_ik_left()
Tst = ik.u.link_pose("left_hand_C_MC", ik._qdict(q_station), ik.base_T, ik.anchor_link)
Rst, pst = Tst[:3, :3], Tst[:3, 3]
Rg, pg = quat_to_R(g[3:7]), g[:3]
lad_pose, lad_fin = [], []
for k in range(len(pre)):
    Rp, pp = quat_to_R(pre[k, 3:7]), pre[k, :3]
    R_rel = Rg.T @ Rp; p_rel = Rg.T @ (pp - pg)          # 腕系相对变换
    Rw = Rst @ R_rel; pw = pst + Rst @ p_rel
    lad_pose.append((pw, Rw))
    lad_fin.append(pre[k, 7:29][perm])
print(f"[approach] 梯级 {len(pre)} 级, 退距 " +
      " ".join(f"{np.linalg.norm(pre[k,:3]-g[:3])*100:.1f}" for k in range(len(pre))) + " cm", flush=True)

# ---- 逐级 IK (限位收缩 3°, 从站位解回溯热启: 近->远) ----
ikc = _mk_ik_left()
ikc.lower = ikc.lower + np.radians(3.0); ikc.upper = ikc.upper - np.radians(3.0)
qs = [None] * len(lad_pose); seed = q_station.copy()
for k in range(len(lad_pose) - 1, -1, -1):
    pw, Rw = lad_pose[k]
    r = ikc.solve(pw, Rw, q0=seed, iters=300, w_rot=0.25)
    assert r["pos_err"] < 0.02, f"梯级{k} IK 误差 {r['pos_err']*100:.1f}cm"
    qs[k] = np.asarray(r["q"], np.float64); seed = qs[k]
    print(f"[approach] 梯级{k}: IK {r['pos_err']*100:.2f}cm/{np.degrees(r['rot_err']):.1f}° 距站位 {np.degrees(np.abs(qs[k]-q_station).max()):.0f}°", flush=True)

# ---- cuRobo cspace: 站姿 -> 梯级0 (满障碍; 右臂目标=站姿=U11) ----
_e = clips.clip_entry(TC.CLIP); _sec = _e["secondary"]
_sx, _sy, _sz = cfg.table_size
bot_p = (E.object.data.root_pos_w[0].cpu().numpy() - W).astype(np.float64)
bot_q = E.object.data.root_quat_w[0].cpu().numpy().astype(np.float64)
cap_p = (E.aux.data.root_pos_w[0].cpu().numpy() - W).astype(np.float64)
cap_q = E.aux.data.root_quat_w[0].cpu().numpy().astype(np.float64)
start = {n: float(v) for n, v in zip(jn, dq)}
start.update({"torso_j1": float(np.radians(40.5196)), "torso_j2": float(np.radians(73.6595)),
              "torso_j3": float(np.radians(0.3896)), "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0})
lock = (["torso_j1", "torso_j2", "torso_j3", "head_j1", "head_j2", "head_j3"]
        + [n for n in jn if n.startswith(("right_", "left_"))])
T = {"table_pose": [float(-_root_p[0]), float(-_root_p[1]), float(cfg.table_top_z - _sz/2 - _root_p[2])],
     "table_dims": [float(_sx), float(_sy), float(_sz)],
     "objects": [{"name": "obj_primary", "mesh": _e["mesh"], "pos": (bot_p - _root_p).tolist(), "quat": bot_q.tolist()},
                 {"name": "obj_secondary", "mesh": _sec["mesh"], "pos": (cap_p - _root_p).tolist(), "quat": cap_q.tolist()}],
     "start_joints": start, "lock_joints": lock,
     "tool_frames": ["right_hand_C_MC", "left_hand_C_MC"],
     "cspace_goal": {}}
for i in range(1, 8):
    T["cspace_goal"][f"R_arm_j{i}"] = float(dq[jn.index(f"R_arm_j{i}")])   # 右臂在家
# 目标候选梯: 梯级0 原样 -> 沿退让方向再外推 4/8cm (目标构型判碰的兜底)
back_dir = (lad_pose[0][0] - pst); back_dir = back_dir / max(np.linalg.norm(back_dir), 1e-9)
goal_alts = [qs[0]]
for extra in (0.04, 0.08):
    pw = lad_pose[0][0] + extra * back_dir
    r = ikc.solve(pw, lad_pose[0][1], q0=qs[0], iters=300, w_rot=0.25)
    if r["pos_err"] < 0.02:
        goal_alts.append(np.asarray(r["q"], np.float64))
plan_rows = None
for gi, qg in enumerate(goal_alts):
    for i in range(1, 8):
        T["cspace_goal"][f"L_arm_j{i}"] = float(qg[i - 1])
    _tmp = tempfile.mkdtemp(prefix="u17_lapp_"); _tgt = os.path.join(_tmp, "targets.json")
    _out = os.path.join(_tmp, "plan.npz")
    json.dump(T, open(_tgt, "w"))
    cmd = [sys.executable, "-u", "-m", "tasks.pregrasp.curobo_plan_worker",
           "--targets", _tgt, "--out", _out, "--act_dist", str(args.act_dist),
           "--attempts", str(args.attempts), "--obj_inflate", str(args.obj_inflate)]
    print(f"[approach] cuRobo 目标候选 #{gi} …", flush=True)
    st_ns = time.time_ns()
    run = subprocess.run(cmd, cwd=os.getcwd(), env=dict(os.environ, PYTHONPATH=os.getcwd()), timeout=2400)
    if run.returncode == 0 and os.path.isfile(_out) and os.stat(_out).st_mtime_ns >= st_ns:
        with np.load(_out, allow_pickle=True) as _z:
            if bool(_z["ok"]):
                _names = [str(n) for n in _z["joint_names"]]
                _tr = np.asarray(_z["traj"], np.float64)
                _ir = [_names.index(f"R_arm_j{i}") for i in range(1, 8)]
                _il = [_names.index(f"L_arm_j{i}") for i in range(1, 8)]
                plan_rows = {"right": _tr[:, _ir], "left": _tr[:, _il]}
                goal_used = gi
                break
assert plan_rows is not None, "cuRobo 三档目标全灭"
Tp = len(plan_rows["left"])
print(f"[approach] cuRobo 通过 (候选 #{goal_used}, {Tp} 行)", flush=True)

# ---- PCHIP 成形段: [cuRobo末行] + 梯级(远->近, 若外推档从外推点进) + [站位] ----
way_a = [plan_rows["left"][-1]] + qs + [q_station]
f_open = np.array([dq[jn.index(n.replace("right_", "left_"))] for n in fin_names])
way_f = [f_open] + lad_fin + [f_station]
Tf = max(2, int(args.form_frames)) * (len(way_a) - 1)
sway = np.arange(len(way_a), dtype=np.float64)
tt = np.linspace(0.0, 1.0, Tf)
u = (tt * tt * (3 - 2 * tt)) * (len(way_a) - 1)
form_a = PchipInterpolator(sway, np.stack(way_a), axis=0)(u)
form_f = PchipInterpolator(sway, np.stack(way_f), axis=0)(u)
jmp = np.degrees(np.abs(np.diff(np.concatenate([plan_rows["left"], form_a]), axis=0)).max())
print(f"[approach] 成形段 {Tf} 行 (路点 {len(way_a)}) | 全程峰值行跳 {jmp:.1f}°", flush=True)

st_fin_r = np.array([dq[jn.index(n)] for n in fin_names])
out = {"left_q": np.concatenate([plan_rows["left"], form_a]).astype(np.float32),
       "left_f": np.concatenate([np.tile(f_open, (Tp, 1)), form_f]).astype(np.float32),
       "right_q": np.concatenate([plan_rows["right"], np.tile(plan_rows["right"][-1], (Tf, 1))]).astype(np.float32),
       "right_f": np.tile(st_fin_r, (Tp + Tf, 1)).astype(np.float32),
       "fin_names": np.array(fin_names, dtype=object),
       "seg_names": np.array(["curobo_stance2pre0", "form_pre2grasp"], dtype=object),
       "seg_lens": np.array([Tp, Tf], np.int64),
       "goal_alt": goal_used,
       "meta": f"LeftApproach_LD227: cuRobo({Tp}) + PCHIP 六级梯({Tf}); 右臂在家; U-P1 用户定稿设计"}
os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez(args.out, **out)
# 终点核验
Tend = ik.u.link_pose("left_hand_C_MC", ik._qdict(form_a[-1].astype(np.float64)), ik.base_T, ik.anchor_link)
print(f"[approach] ★已写 {args.out} ({Tp}+{Tf} 行) | 终点腕误差 {np.linalg.norm(Tend[:3,3]-pst)*100:.2f}cm", flush=True)
app.close(); os._exit(0)
