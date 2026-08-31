"""IK 行运动学核查：FK(v2臂行) vs 物体轨迹∘抓变换；逐行检查位置 1cm
与姿态 10° 双阈值，关键拧盖窗不得有坏行。"""
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
ik = {"right": ArmIK("right", anchor_link="arm_center", anchor_T=E._anchor_T),
      "left": ArmIK("left", anchor_link="arm_center", anchor_T=E._anchor_T)}
z = np.load(PE.MASTER, allow_pickle=True)
rows = np.where(np.asarray(z["source"]) == 1)[0]
arm = {"right": np.asarray(z["right_q"], np.float64)[rows],
       "left": np.asarray(z["left_q"], np.float64)[rows]}
ref_obj = {oi: E.PB.ref_obj[oi].cpu().numpy() for oi in (0, 1)}
side_obj = {"right": 1, "left": 0}
N = E.PB.N_ROW
# 抓变换锚: 用行0的 FK 腕与行0物体 (与builder同法: w0 = FK(行0臂))
w0 = {s: ik[s].fk(arm[s][0]) for s in ("right", "left")}
bad = {"right": [], "left": []}
for s in ("right", "left"):
    oi = side_obj[s]
    p0, q0_ = ref_obj[oi][0][:3], ref_obj[oi][0][3:7]
    for k in range(N):
        pk, qk_ = ref_obj[oi][k][:3], ref_obj[oi][k][3:7]
        Rk = quat_to_R(qk_) @ quat_to_R(q0_).T
        tgt_p = Rk @ w0[s][0] + (pk - Rk @ p0)
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
ok = not critical_bad["right"] and not critical_bad["left"]
print(f"[ikchk] {'✅ 关键窗通过' if ok else '❌ 关键窗存在位置/姿态坏行'}", flush=True)
try: _slot.release()
except Exception: pass
app.close(); os._exit(0 if ok else 1)
