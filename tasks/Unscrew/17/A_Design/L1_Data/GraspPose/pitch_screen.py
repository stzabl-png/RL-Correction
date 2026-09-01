"""LD 先验 "腕上翘" 变体离线筛: 绕接触质心、绕切向轴把整只手俯仰 θ (腕升高, 接触带不动), 立瓶 yaw 扫描左臂 IK 可达 + 限位余量."""
import sys, numpy as np
from scipy.spatial.transform import Rotation as Rt
sys.path.insert(0, "/home/lyh/Project/RL_Correction")
from rl_rebuild.correction.kinematics import ArmIK, Urdf
P = np.load("tasks/pregrasp/priors/Screw17_bottle_left.npz", allow_pickle=True)
REST = np.array([-0.123, 0.089, 0.871]); TABLE = 0.87
def T_of(R, p): M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M
def qR(q): return Rt.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
def Rq(R): x, y, z, w = Rt.from_matrix(R).as_quat(); return np.array([w, x, y, z])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
u = Urdf(); ik = ArmIK("left", urdf=u)
lo = hi = None
for a, b in (("lo", "hi"), ("q_lo", "q_hi"), ("lower", "upper"), ("qmin", "qmax")):
    if hasattr(ik, a): lo, hi = np.asarray(getattr(ik, a)), np.asarray(getattr(ik, b)); break
def margin(q): return float(np.degrees(np.minimum(q - lo, hi - q).min())) if lo is not None else float("nan")
def variant(theta_deg, dz):
    g = P["grasp"].copy(); c = P["contact_centroid"].copy()
    w = g[:3]; r = np.array([w[0], w[1], 0.0]); r /= np.linalg.norm(r); t = np.cross([0, 0, 1.0], r)
    R = Rt.from_rotvec(t * np.radians(theta_deg)).as_matrix()
    def xf(pose): p = c + R @ (pose[:3] - c) + [0, 0, dz]; return np.concatenate([p, Rq(R @ qR(pose[3:7])), pose[7:]])
    out = dict(grasp=xf(g), squeeze=xf(P["squeeze"]), pregrasp=np.stack([xf(pp) for pp in P["pregrasp"]]),
               contact_pos=(c + (P["contact_pos"] - c) @ R.T) + [0, 0, dz], contact_normal=P["contact_normal"] @ R.T)
    return out
def scan(theta, dz):
    v = variant(theta, dz); g = v["grasp"]; Tcw = T_of(qR(g[3:7]), g[:3]); band = []; q = None
    for deg in range(0, 360, 10):
        M = T_of(Rz(np.radians(deg)), REST) @ Tcw
        best = None
        for s in range(3):
            q0 = q if (s == 0 and q is not None) else np.random.uniform(-0.5, 0.5, 7)
            r = ik.solve(M[:3, 3], M[:3, :3], q0=q0, iters=150, pos_tol=0.008, rot_tol=0.12)
            if r["ok"] and (best is None or margin(r["q"]) > best[0]): best = (margin(r["q"]), r["q"])
        if best: q = best[1]; band.append((deg, best[0]))
    cz = v["contact_pos"][:, 2]; wz = g[2]
    return dict(theta=theta, dz=dz, wrist_z=wz, contact_lo=cz.min(), contact_hi=cz.max(), band=band)
np.random.seed(0)
print(f"{'θ':>4s} {'dz':>5s} | 腕高(瓶底上)  接触带  | 可达 yaw 数/36 | 余量最佳 yaw(余量°) | 人手方位 140° 附近可达 yaw")
for theta in (0, -15, -30, -45, -60):
    for dz in (0.0, 0.02):
        s = scan(theta, dz); b = s["band"]
        best = max(b, key=lambda x: x[1]) if b else None
        near = [d for d, m in b if abs(((d + 60) - 140 + 180) % 360 - 180) <= 30]  # 腕方位 ≈ yaw+? 仅粗略
        print(f"{theta:4d} {dz*100:4.0f}cm | {s['wrist_z']*100:5.1f}cm  {s['contact_lo']*100:4.1f}~{s['contact_hi']*100:4.1f}cm | {len(b):2d}/36 | {best[0] if best else '-'}({best[1]:.0f}°)" if best else f"{theta:4d} {dz*100:4.0f}cm | {s['wrist_z']*100:5.1f}cm  {s['contact_lo']*100:4.1f}~{s['contact_hi']*100:4.1f}cm | 0/36 | -", 
              f"| {[d for d,_ in b][:12]}" if b else "")
