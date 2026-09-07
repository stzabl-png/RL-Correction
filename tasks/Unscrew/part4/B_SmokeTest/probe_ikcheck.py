"""IK 行运动学诊断：FK(v2 臂行) vs v1 人腕增量重锚目标。

1cm/10° 坏行作为 RL correction 基线记录；只有 NaN/Inf 才阻塞训练。
"""
import argparse, os, sys
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot
_slot = isaac_slot("ikchk")
app = AppLauncher(args).app
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ.pop("POUR_SQUEEZE_FF", None)
import task_config as TC
import task_env as PE
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R

cfg = PE.build_cfg(num_envs=1)
E = PE.UnscrewEnv(cfg)
E.force_entry = [0]; E.reset()
ik = {side: ArmIK(side, anchor_link="arm_center",
                  anchor_T=TC.rest_anchor_T(side)) for side in ("right", "left")}
z = np.load(PE.MASTER, allow_pickle=True)
rows = np.where(np.asarray(z["source"]) == 1)[0]
arm = {"right": np.asarray(z["right_q"], np.float64)[rows],
       "left": np.asarray(z["left_q"], np.float64)[rows]}
N = E.PB.N_ROW
# 与 build_reference 完全同口径：把 v1 人腕的逐行平移/姿态增量重锚到
# Isaac 实测 P1 腕位。不能再用“物体增量∘抓变换”：右腕并非与盖刚性固连。
wt = {"right": np.asarray(z["wrist_tgt_r"], np.float64),
      "left": np.asarray(z["wrist_tgt_l"], np.float64)}
assert len(wt["right"]) == len(wt["left"]) == N
w0 = {s: ik[s].fk(arm[s][0]) for s in ("right", "left")}
bad = {"right": [], "left": []}
for s in ("right", "left"):
    p0, q0_ = wt[s][0][:3], wt[s][0][3:7]
    for k in range(N):
        pk, qk_ = wt[s][k][:3], wt[s][k][3:7]
        Rk = quat_to_R(qk_) @ quat_to_R(q0_).T
        tgt_p = w0[s][0] + (pk - p0)
        tgt_R = Rk @ w0[s][1]
        fp, fR = ik[s].fk(arm[s][k])
        e = float(np.linalg.norm(fp - tgt_p))
        er = float(np.arccos(np.clip((np.trace(fR.T @ tgt_R) - 1) * 0.5, -1, 1)))
        if e > 0.01 or er > np.radians(10):
            bad[s].append((k, round(e * 100, 2), round(np.degrees(er), 1)))
critical_bad = {}
for s in ("right", "left"):
    print(f"[ikchk] {s}: 坏行 {len(bad[s])}/{N} | 明细(行,cm,deg): {bad[s][:20]}",
          flush=True)
    ks = E.PB.k_sep
    win = [b for b in bad[s] if max(ks - 40, 0) <= b[0] <= ks]
    critical_bad[s] = win
    print(f"[ikchk] {s}: 拧盖窗({max(ks-40,0)}..{ks})内坏行: {win}", flush=True)
finite = all(np.isfinite(arm[s]).all() for s in ("right", "left"))
ok = finite
print(f"[ikchk] 关键窗坏行仅作 correction 基线，不阻塞训练", flush=True)
print(f"[ikchk] {'✅ 轨迹数值有限' if ok else '❌ 轨迹含 NaN/Inf'}", flush=True)
try: _slot.release()
except Exception: pass
app.close(); os._exit(0 if ok else 1)
