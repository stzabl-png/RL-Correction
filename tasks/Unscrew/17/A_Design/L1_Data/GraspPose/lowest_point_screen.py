"""左手候选 "装臂后最低点离桌" 离线筛 (§8 漏的判据): 立瓶 yaw 扫描 → 左臂 IK (3 种子取余量最大) → 全臂+手 FK →
腕链 (l6/l7/l8/ee/C_MC) 与手指链 (候选抓形角) 的链接原点最低 z − 桌高. 原点不含网格半径, 用 Isaac 沙盒实测标定 (P30/LDup6)."""
import json, sys, numpy as np
from scipy.spatial.transform import Rotation as Rt
sys.path.insert(0, "/home/lyh/Project/RL_Correction")
from rl_rebuild.correction.kinematics import ArmIK, Urdf
from rl_rebuild.correction.ref_builders.replay_grasp import GENERIC_JOINT_ORDER
C = json.load(open("tasks/Unscrew/17/A_Design/L1_Data/GraspPose/grasp_cands.json"))
REST = np.array([-0.123, 0.089, 0.871]); TABLE = 0.87; HUMAN_AZ = 140.0
def T_of(R, p): M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M
def qR(q): return Rt.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
u = Urdf(); ik = ArmIK("left", urdf=u)
lo = hi = None
for a, b in (("lo", "hi"), ("q_lo", "q_hi"), ("lower", "upper"), ("qmin", "qmax")):
    if hasattr(ik, a): lo, hi = np.asarray(getattr(ik, a)), np.asarray(getattr(ik, b)); break
def margin(q): return float(np.degrees(np.minimum(q - lo, hi - q).min())) if lo is not None else float("nan")
CHAIN = ["vega_1p_L_arm_l6", "L_arm_l7", "L_arm_l8", "L_ee", "left_hand_C_MC"]
FING = [j["child"] for n, j in u.joints.items() if n.startswith("left_")]
LEFT22 = [n.replace("right_", "left_") for n in GENERIC_JOINT_ORDER]
def lowest(q_arm, fin22):
    d = ik._qdict(q_arm); d.update({n: float(v) for n, v in zip(LEFT22, fin22)})
    def z(link):
        try: return float(u.link_pose(link, d, ik.base_T, ik.anchor_link)[2, 3])
        except Exception: return np.nan
    zc = {l: z(l) for l in CHAIN}; zf = {l: z(l) for l in FING}
    lc = min(zc, key=lambda k: zc[k] if zc[k] == zc[k] else 9); lf = min(zf, key=lambda k: zf[k] if zf[k] == zf[k] else 9)
    return zc[lc] - TABLE, lc, zf[lf] - TABLE, lf
def azim(p): d = p[:2] - REST[:2]; return np.degrees(np.arctan2(d[1], d[0]))
def angd(a, b): return abs((a - b + 180) % 360 - 180)
def screen(name, Tcw, fin22):
    rows = []; q = None
    for deg in range(0, 360, 10):
        M = T_of(Rz(np.radians(deg)), REST) @ Tcw; best = None
        for s in range(3):
            q0 = q if (s == 0 and q is not None) else np.random.uniform(-0.5, 0.5, 7)
            r = ik.solve(M[:3, 3], M[:3, :3], q0=q0, iters=150, pos_tol=0.008, rot_tol=0.12)
            if r["ok"] and (best is None or margin(r["q"]) > best[0]): best = (margin(r["q"]), r["q"])
        if best:
            q = best[1]; cz, cl, fz, fl = lowest(q, fin22)
            rows.append(dict(yaw=deg, az=azim(M[:3, 3]), marg=best[0], chain=cz, chain_link=cl, fing=fz, fing_link=fl, wrist_z=M[2, 3] - TABLE))
    if not rows: print(f"{name:36s} 不可达"); return
    bc = max(rows, key=lambda r: min(r["chain"], r["fing"]))
    near = [r for r in rows if angd(r["az"], HUMAN_AZ) <= 30]
    bn = max(near, key=lambda r: min(r["chain"], r["fing"])) if near else None
    print(f"{name:36s} 可达 {len(rows):2d}/36 | 腕高 {bc['wrist_z']*100:4.1f}cm | 最低点最好: yaw{bc['yaw']:3d} 链 {bc['chain']*100:+5.1f}cm({bc['chain_link']}) 指 {bc['fing']*100:+5.1f}cm({bc['fing_link'].replace('left_','')}) 余量 {bc['marg']:.0f}°"
          + (f" | 人手方位±30°: yaw{bn['yaw']} 链 {bn['chain']*100:+5.1f} 指 {bn['fing']*100:+5.1f} 余量 {bn['marg']:.0f}°" if bn else " | 人手方位±30° 不可达"))
np.random.seed(0)
comB = np.asarray(C["bottle_left"]["canonical_frame"]["com_offset"])
print("== Dexonomy 12 候选 (grasp 抓形角) ==")
for c in C["bottle_left"]["cands"]:
    g = np.asarray(c["grasp"]); Tcw = T_of(np.eye(3), comB) @ T_of(qR(g[3:7]), g[:3])
    screen(c["file"].replace("_grasp.npy", ""), Tcw, g[7:29])
print("== 先验变体 (标定用; npz 已含 com 平移) ==")
for nm in ("Screw17_bottle_left", "Screw17_bottle_left_LDup6", "Screw17_bottle_left_P30", "Screw17_bottle_left_P15u3"):
    z = np.load(f"tasks/pregrasp/priors/{nm}.npz", allow_pickle=True); g = z["grasp"]
    screen(nm, T_of(qR(g[3:7]), g[:3]), g[7:29])
