"""v5 母带生成 (L5-1 拍板): 交互段(190..324)由物体轨迹反解IK重铸。
第一幕 Approach照谱+缝1剂量渐入(βR=2.0/βL=1.0) → 站稳30步 → 捕获增量空间锚 →
逐行IK(135行×双臂) + 5mm认证行IK → 缝2重融接 → 写 pour17_reference_v2.npz。
Approach/Retreat/物体轨迹/conf 原样; 人手行存 human_* 供 P-HYB 红档参考。"""
import argparse, hashlib, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("build_v5")
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ.pop("POUR_SQUEEZE_FF", None)
import pour_env as PE
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R

BR, BL = 2.0, 1.0
V1 = "tasks/Pour/17/A_Design/L2_Reference/pour17_reference_v1.npz"
OUT = "tasks/Pour/17/A_Design/L2_Reference/pour17_reference_v2.npz"
cfg = PE.build_cfg(num_envs=1)
E = PE.PourEnv(cfg)
E.force_entry = [0]
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))
sqr = np.asarray(np.load("tasks/pregrasp/priors/Pour17_bottle_thumbfix.npz")["squeeze"],
                 np.float64).reshape(-1)[7:29]
sql = np.asarray(np.load("tasks/pregrasp/priors/Pour17_cup_thumbfix.npz")["squeeze"],
                 np.float64).reshape(-1)[7:29]
ref = E.ref58.cpu().numpy()
dsq_r = torch.tensor(sqr - ref[E.IA0, 14:36], dtype=torch.float32, device=dev)
dsq_l = torch.tensor(sql - ref[E.IA0, 36:58], dtype=torch.float32, device=dev)

def drive(row, sR, sL):
    r = min(row, E.T_ROW - 1)
    tgt = E.ref58[r].clone()
    tgt[14:36] += sR * dsq_r
    tgt[36:58] += sL * dsq_l
    full = E.hand.data.joint_pos.clone()
    full[0, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    for _ in range(DECI):
        E.scene.write_data_to_sim(); E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())

print("[v5] 第一幕: Approach+缝1 (βR=2.0/βL=1.0)", flush=True)
for r in range(0, 166):
    drive(r, 0.0, 0.0)
for i, r in enumerate(range(166, 190)):
    a = (i + 1) / 24.0
    drive(r, BR * a, BL * a)
for _ in range(30):
    drive(190, BR, BL)
f = E._pads_f().norm(dim=-1)[0]
print(f"[v5] 站位垫: R{int((f[:5]>0.5).sum())}/L{int((f[5:]>0.5).sum())}", flush=True)

ik = {"right": ArmIK("right", anchor_link="arm_center", anchor_T=E._anchor_T),
      "left": ArmIK("left", anchor_link="arm_center", anchor_T=E._anchor_T)}
ref_obj_raw = {oi: E.PB.ref_obj[oi].cpu().numpy() for oi in (0, 1)}
side_obj = {"right": 1, "left": 0}
Nrow_raw = E.PB.N_ROW
# ---- 放回窗时间扩张 (L5-3): 交互行 72..100 (=全链262..290) 二倍细分 ----
# 母带放回 = 回正旋转55° + 10cm/s 下降复合, v2硬闸实证 3垫握被拧脱; 扩张后速率减半
W0, W1 = 72, 100
frac_rows = []
for k in range(Nrow_raw):
    frac_rows.append(float(k))
    if W0 <= k < W1:
        frac_rows.append(k + 0.5)
frac_rows = np.array(frac_rows)
Nrow = len(frac_rows)
def _lerp_track(tr):
    lo = np.floor(frac_rows).astype(int)
    hi = np.minimum(lo + 1, Nrow_raw - 1)
    a = (frac_rows - lo)[:, None]
    pos = tr[lo, :3] * (1 - a) + tr[hi, :3] * a
    q0, q1 = tr[lo, 3:7], tr[hi, 3:7]
    sgn = np.sign((q0 * q1).sum(axis=1, keepdims=True)); sgn[sgn == 0] = 1
    q = q0 * (1 - a) + q1 * sgn * a
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-9)
    return np.concatenate([pos, q], axis=1)
ref_obj = {oi: _lerp_track(ref_obj_raw[oi]) for oi in (0, 1)}
floor_map = np.floor(frac_rows).astype(int)          # 新交互行 -> 原交互行
print(f"[v5] 放回窗扩张: 交互 {Nrow_raw} -> {Nrow} 行 (窗{W0}..{W1} 二倍)", flush=True)
q_ik = {s: np.zeros((Nrow, 7)) for s in ("right", "left")}
cert = {}
q_seed, w0 = {}, {}
for s in ("right", "left"):
    sl = slice(0, 7) if s == "right" else slice(7, 14)
    q_seed[s] = E.hand.data.joint_pos[0, E.map_ids_t[sl]].cpu().numpy().astype(np.float64)
    w0[s] = ik[s].fk(q_seed[s])
fail_ik, err_max = 0, 0.0
rng = np.random.default_rng(17)
for s in ("right", "left"):
    best = None
    for trial in range(12):
        q0t = q_seed[s] if trial == 0 else q_seed[s] + rng.normal(0, 0.03, 7)
        r5 = ik[s].solve(w0[s][0] + np.array([0, 0, 0.015]), w0[s][1], q0=q0t, iters=300)
        if best is None or r5["pos_err"] < best["pos_err"]:
            best = r5
        if best["pos_err"] < 0.001:
            break
    cert[s] = np.asarray(best["q"], np.float64)
    print(f"[v5] 认证行IK {s}: pos_err={best['pos_err']*1000:.2f}mm", flush=True)
for k in range(Nrow):
    for s in ("right", "left"):
        oi = side_obj[s]
        p0, q0_ = ref_obj[oi][0][:3], ref_obj[oi][0][3:7]
        pk, qk_ = ref_obj[oi][k][:3], ref_obj[oi][k][3:7]
        Rk = quat_to_R(qk_) @ quat_to_R(q0_).T
        tp = pk - Rk @ p0
        r = ik[s].solve(Rk @ w0[s][0] + tp, Rk @ w0[s][1], q0=q_seed[s], iters=60)
        if r["pos_err"] > 0.01:
            fail_ik += 1
        err_max = max(err_max, float(r["pos_err"]))
        q_ik[s][k] = r["q"]
        q_seed[s] = np.asarray(r["q"], np.float64)
print(f"[v5] 交互IK: >1cm 失败 {fail_ik}/{Nrow*2} 最大误差={err_max*100:.2f}cm", flush=True)

d1 = dict(np.load(V1, allow_pickle=True))
IA0, IA1, RET0 = 190, 324, 350          # v1 锚点
SEAM2, NRET = RET0 - IA1 - 1, len(d1["right_q"]) - RET0
Tn = IA0 + Nrow + SEAM2 + NRET          # 新全链长
ia_rows = IA0 + np.arange(Nrow)
def _splice(colv, ia_val, tail0=RET0 - SEAM2):
    pre = np.asarray(colv)[:IA0]
    tail = np.asarray(colv)[IA1 + 1:]
    return np.concatenate([pre, ia_val, tail], axis=0)
v2r = _splice(d1["right_q"], q_ik["right"])
v2l = _splice(d1["left_q"], q_ik["left"])
v2rf = _splice(d1["right_f"], np.asarray(d1["right_f"])[IA0 + floor_map])
v2lf = _splice(d1["left_f"], np.asarray(d1["left_f"])[IA0 + floor_map])
# 缝2 重融接 (新交互末行 → retreat 首行)
RET0n = IA0 + Nrow + SEAM2
for i in range(SEAM2):
    a = (i + 1) / (SEAM2 + 1)
    rr = IA0 + Nrow + i
    v2r[rr] = (1-a) * v2r[IA0 + Nrow - 1] + a * v2r[RET0n]
    v2l[rr] = (1-a) * v2l[IA0 + Nrow - 1] + a * v2l[RET0n]
    v2rf[rr] = (1-a) * v2rf[IA0 + Nrow - 1] + a * v2rf[RET0n]
    v2lf[rr] = (1-a) * v2lf[IA0 + Nrow - 1] + a * v2lf[RET0n]
# 物体轨迹/conf/source/frame_of_row 列重拼 (交互段=扩张后; 物体列用母带系原值插值)
obj_cols = {}
for oi in (0, 1):
    raw = np.concatenate([np.asarray(d1[f"obj_pos_{oi}"], np.float64),
                          np.asarray(d1[f"obj_quat_{oi}"], np.float64)], axis=1)
    ia_raw = raw[IA0:IA1 + 1]
    lo = floor_map; hi = np.minimum(lo + 1, Nrow_raw - 1)
    a = (frac_rows - lo)[:, None]
    pos = ia_raw[lo, :3] * (1 - a) + ia_raw[hi, :3] * a
    q0, q1 = ia_raw[lo, 3:7], ia_raw[hi, 3:7]
    sgn = np.sign((q0 * q1).sum(axis=1, keepdims=True)); sgn[sgn == 0] = 1
    q = q0 * (1 - a) + q1 * sgn * a
    q = q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-9)
    obj_cols[f"obj_pos_{oi}"] = _splice(d1[f"obj_pos_{oi}"], pos)
    obj_cols[f"obj_quat_{oi}"] = _splice(d1[f"obj_quat_{oi}"], q)
    for cn in (f"conf_pos_{oi}", f"conf_rot_{oi}"):
        obj_cols[cn] = _splice(d1[cn], np.asarray(d1[cn])[IA0 + floor_map])
src_new = _splice(d1["source"], np.ones(Nrow, np.int8))
fr_new = _splice(d1["frame_of_row"],
                 np.asarray(d1["frame_of_row"])[IA0 + floor_map])
segl1 = [int(v) for v in d1["seg_lens"]]
segl1[2] = Nrow
assert sum(segl1) == Tn, (segl1, Tn)
with open(V1, "rb") as fh:
    parent_md5 = hashlib.md5(fh.read()).hexdigest()[:8]
hum_r = _splice(d1["right_q"], np.asarray(d1["right_q"])[IA0 + floor_map])
hum_l = _splice(d1["left_q"], np.asarray(d1["left_q"])[IA0 + floor_map])
hum_rf = _splice(d1["right_f"], np.asarray(d1["right_f"])[IA0 + floor_map])
hum_lf = _splice(d1["left_f"], np.asarray(d1["left_f"])[IA0 + floor_map])
out = dict(d1)
out.update(obj_cols, source=src_new, frame_of_row=fr_new,
           seg_lens=np.asarray(segl1, np.int64),
           right_q=v2r, left_q=v2l, right_f=v2rf, left_f=v2lf,
           human_right_q=hum_r, human_left_q=hum_l,
           human_right_f=hum_rf, human_left_f=hum_lf,
           cert_arm7_right=cert["right"], cert_arm7_left=cert["left"],
           meta_v5=np.array([f"gen=L5-1;parent_v1_md5={parent_md5};betaR={BR};betaL={BL};"
                             f"obj_rest_z=0.959;ik=delta_space;date=2026-08-27"]))
np.savez(OUT, **out)
with open(OUT, "rb") as fh:
    print(f"[v5] 已写 {OUT} md5={hashlib.md5(fh.read()).hexdigest()[:8]}", flush=True)
print("[v5] 完毕", flush=True)
try: _slot.release()
except Exception: pass
app.close(); os._exit(0)
