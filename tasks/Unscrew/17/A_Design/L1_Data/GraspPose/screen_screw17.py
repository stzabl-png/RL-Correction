"""screw17 GraspPose 选型离线筛 (纯 numpy, 不开 Isaac).

左手×瓶: 立瓶 yaw 扫描可达带 + 放倒轨迹 (f23→f45) 全程可达 (手↔瓶刚性, 滚转可选)
右手×盖: 放倒后 (f40) 盖系候选 → 世界腕位姿 → 右臂可达 + PreGrasp 可达 + 朝向合理
输出: 逐候选表 + 联合可行 (左候选, 抓握 yaw, 放倒滚转, 右候选)。
"""
import json, sys, os, numpy as np
from scipy.spatial.transform import Rotation as Rt
sys.path.insert(0, "/home/lyh/Project/RL_Correction")
from rl_rebuild.correction.kinematics import ArmIK, Urdf
from rl_rebuild.correction import place_camera as PC

SC = os.path.dirname(os.path.abspath(__file__))
T = "/home/lyh/Project/Reconstruct_and_Retarget/Output/ReconstructOutput/egodex_part4/screw_unscrew_bottle_cap/17"
C = json.load(open(f"{SC}/grasp_cands.json"))
w = np.load(f"{T}/world_fused.npz", allow_pickle=True)
P = w["object_ob_in_world_all"]; B = P[0]
TABLE = 0.87; TABLE_RECON = 1.005; GAP = 0.005

def T_of(R, p):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M
def qR(q):  # wxyz
    return Rt.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
def Rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
def R_from_z_to(v):
    v = v / np.linalg.norm(v); z = np.array([0, 0, 1.0]); c = float(np.clip(z @ v, -1, 1))
    if c > 1 - 1e-9: return np.eye(3)
    ax = np.cross(z, v); s = np.linalg.norm(ax); ax /= s
    return Rt.from_rotvec(ax * np.arctan2(s, c)).as_matrix()

# ---- 场景换基: 相机锚定 (xy 刚体平移) + z 贴桌 ----
mesh0 = f"{T}/object_mesh_scaled_final.obj"   # 只用 dirname 找 world_fused.npz
obj_first = B[0, :2, 3].copy()
sxy = PC.camera_anchor_shift(mesh0, PC.ZED_NOMINAL[:2], obj_first)
shift_xy = sxy - obj_first            # env_xy = recon_xy + shift_xy
def env_base(t):
    p = B[t, :3, 3].copy(); p[:2] += shift_xy; p[2] = p[2] - TABLE_RECON + TABLE + GAP; return p
def bottle_axis(t):
    ax = B[t, :3, 2].copy(); tilt = np.degrees(np.arccos(np.clip(ax[2], -1, 1)))
    return (np.array([0, 0, 1.0]) if tilt < 22 else ax / np.linalg.norm(ax)), tilt
print(f"[scene] shift_xy={np.round(shift_xy,3)}  bottle base env f0={np.round(env_base(0),3)}  f40={np.round(env_base(40),3)}  axis f40={np.round(bottle_axis(40)[0],2)}")

def T_env_bottle(t, psi):
    ax, _ = bottle_axis(t)
    return T_of(R_from_z_to(ax) @ Rz(psi), env_base(t))

# ---- 候选 → 物体 CAD 系腕位姿 ----
comB = np.asarray(C["bottle_left"]["canonical_frame"]["com_offset"]); comC = np.asarray(C["cap_right"]["canonical_frame"]["com_offset"])
def cand_T(c, com):            # T_CAD_wrist
    g = np.asarray(c["grasp"]); return T_of(np.eye(3), com) @ T_of(qR(g[3:7]), g[:3])
def cand_pre_T(c, com):        # 最远的 PreGrasp 级
    g = np.asarray(c["grasp"]); pg = np.asarray(c["pregrasp"])
    k = int(np.argmax(np.linalg.norm(pg[:, :3] - g[:3], axis=1)))
    return T_of(np.eye(3), com) @ T_of(qR(pg[k, 3:7]), pg[k, :3]), float(np.linalg.norm(pg[k, :3] - g[:3]))
T_bottle_cap = T_of(np.eye(3), [0, 0, 0.18])

u = Urdf(); ikL = ArmIK("left", urdf=u); ikR = ArmIK("right", urdf=u)
def reach(ik, M, q0=None):
    r = ik.solve(M[:3, 3], M[:3, :3], q0=q0, iters=120, pos_tol=0.008, rot_tol=0.12)
    return r
HUMAN_AZ = 144.0
def azim(p, base): d = p[:2] - base[:2]; return np.degrees(np.arctan2(d[1], d[0]))
def angd(a, b): return abs((a - b + 180) % 360 - 180)

# ================= 左手 =================
print("\n================ 左手×瓶 (立瓶 yaw 扫描, 步 10°) ================")
LEFT = []
for c in C["bottle_left"]["cands"]:
    Tcw = cand_T(c, comB); band = []; q = None
    for deg in range(0, 360, 10):
        M = T_env_bottle(0, np.radians(deg)) @ Tcw
        r = reach(ikL, M, q0=q)
        if r["ok"]: q = r["q"]; band.append((deg, azim(M[:3, 3], env_base(0)), r["rot_err"]))
    # 放倒可达: 对可达带内每个 yaw, 滚转增量 Δ ∈ {0,+32,-32,+60,-60} (f23→f40 线性), 逐行 f23..f45
    lay = {}
    for deg, az, _ in band:
        for droll in (0, 32, -32, 60, -60):
            ok = 0; q = None; worst = 0.0
            for t in range(23, 46):
                ramp = min(max((t - 23) / 17.0, 0.0), 1.0)
                M = T_env_bottle(t, np.radians(deg + droll * ramp)) @ Tcw
                r = reach(ikL, M, q0=q); q = r["q"] if r["ok"] else q
                ok += int(r["ok"]); worst = max(worst, r["pos_err"])
            lay[(deg, droll)] = (ok, worst)
    good = [(k, v) for k, v in lay.items() if v[0] == 23]
    az_near = min(band, key=lambda b: angd(b[1], HUMAN_AZ)) if band else None
    LEFT.append(dict(file=c["file"], tmpl=c["tmpl"], demo=c["demo_angle"], elev=c["approach_elev"],
                     isaac=(c["isaac"] or {}).get("grasp_success"), band=[b[0] for b in band],
                     az_at_band={b[0]: round(b[1]) for b in band}, nearest_human=az_near,
                     lay_ok=[k for k, _ in good], n_lay_ok=len(good)))
    print(f"{c['file']:38s} demo{c['demo_angle']:5.1f}° elev{c['approach_elev']:5.1f}° isaac={LEFT[-1]['isaac']} | 立瓶可达 yaw {len(band)}/36"
          f" | 离人手方位最近: {('yaw%d→az%d(Δ%d)' % (az_near[0], az_near[1], angd(az_near[1], HUMAN_AZ))) if az_near else '-'}"
          f" | 放倒全程可达 (yaw,Δroll) {len(good)} 组: {[k for k,_ in good][:8]}")

# ================= 右手 (放倒后 f40, 滚转 psi40 扫描) =================
print("\n================ 右手×盖 (f40 放倒态; 瓶滚转 psi40 扫描, 步 10°) ================")
RIGHT = {}
for c in C["cap_right"]["cands"]:
    Tcw = cand_T(c, comC); Tpw, dpre = cand_pre_T(c, comC)
    rows = {}; q = None
    for deg in range(0, 360, 10):
        Tb = T_env_bottle(40, np.radians(deg)); Tcap = Tb @ T_bottle_cap
        Mg = Tcap @ Tcw; Mp = Tcap @ Tpw
        rg = reach(ikR, Mg, q0=q); rp = reach(ikR, Mp, q0=rg["q"] if rg["ok"] else q)
        if rg["ok"]: q = rg["q"]
        capc = Tcap[:3, 3]; wz = Mg[2, 3]; appr = Mg[:3, 3] - capc
        sane = (wz > TABLE + 0.03) and (appr[2] > -0.03)          # 腕离桌 ≥3cm, 不从桌下够
        rows[deg] = dict(ok=bool(rg["ok"] and rp["ok"] and sane), g=rg["ok"], p=rp["ok"], sane=sane,
                         wz=round(float(wz), 3), appr=np.round(appr, 3).tolist(), rot=round(float(rg["rot_err"]), 2))
    okd = [d for d, r in rows.items() if r["ok"]]
    RIGHT[c["file"]] = dict(tmpl=c["tmpl"], demo=c["demo_angle"], elev=c["approach_elev"], dpre=dpre, rows=rows, ok_psi=okd,
                            isaac=(c["isaac"] or {}).get("grasp_success") if c["isaac"] else None)
for f, r in sorted(RIGHT.items(), key=lambda kv: kv[1]["demo"]):
    print(f"{f:38s} demo{r['demo']:5.1f}° elev{r['elev']:6.1f}° pre{r['dpre']*100:4.1f}cm isaac={r['isaac']} | 可用 psi40: {len(r['ok_psi'])}/36 {r['ok_psi'][:12]}")

# ================= 联合 =================
print("\n================ 联合可行 (左候选, yaw, Δroll → psi40 = yaw+Δroll; 右候选在该 psi40 可用) ================")
joint = []
for L in LEFT:
    for (deg, droll) in L["lay_ok"]:
        psi40 = (deg + droll) % 360; psi_bin = int(round(psi40 / 10.0) * 10) % 360
        for f, r in RIGHT.items():
            if psi_bin in r["ok_psi"]:
                joint.append((L["file"], deg, droll, psi40, f, r["demo"], r["tmpl"], angd(L["az_at_band"].get(deg, 0), HUMAN_AZ)))
print(f"联合可行组合 {len(joint)}; 按 (左离人手方位差, 右 demo 角) 排序前 25:")
for j in sorted(joint, key=lambda x: (x[7], x[5]))[:25]:
    print(f"  L {j[0]:34s} yaw{j[1]:4d} Δroll{j[2]:4d} → psi40 {j[3]:4.0f} | R {j[4]:38s} demo{j[5]:5.1f}° | 左方位Δ{j[7]:.0f}°")
json.dump(dict(shift_xy=shift_xy.tolist(), left=LEFT, right={k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in RIGHT.items()},
               joint=joint), open(f"{SC}/screen_result.json", "w"), default=float)
