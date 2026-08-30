"""Unscrew v2 母带重铸 (框架 L5-1 拍板的移植, 动态行号版):
交互段臂行由**物体轨迹反解 IK** 重铸 —— 第一幕 Approach 照 v1 谱 + 缝1 β 渐入
(βL=左手瓶 squeeze; 盖侧设定B无 prior) → 站稳 30 步 → 捕获增量空间锚
(腕目标[k] = ΔT_obj[k] ∘ 腕FK(站位); 右手跟盖 obj_1, 左手跟瓶 obj_0)
→ 逐行 IK + 15mm 认证行 IK → 写 reference_v2.npz。
物体轨迹/conf/人手指行原样; v1 臂行存 human_* 供追溯 (⚠ 本批人手腕平移是
静态填充死数据, v1 臂行已是物体推导 —— human_* 只是"重铸前"的意思)。

  UNSCREW_CLIP=32 SHARPA_WANDB=0 PYTHONPATH=. $PY \
      tasks/Unscrew/part4/B_SmokeTest/build_reference.py --headless
"""
import argparse
import hashlib
import os
import sys

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
from rl_rebuild.utils.gpu_guard import isaac_slot  # noqa: E402
_slot = isaac_slot("unscrew_build_v2")
app = AppLauncher(args).app
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "C_Wiring"))
os.environ["POUR_NO_D6"] = "1"
os.environ.pop("POUR_SQUEEZE_FF", None)
import task_config as TC  # noqa: E402

os.environ["POUR_REF_NPZ"] = TC.REF_V1        # 重铸的输入永远是 v1
import task_env as PE  # noqa: E402
from rl_rebuild.correction.kinematics import ArmIK, quat_to_R  # noqa: E402

BR, BL = TC.BETA_R, TC.BETA_L
V1, OUT = TC.REF_V1, TC.REF_V2
cfg = PE.build_cfg(num_envs=1)
E = PE.UnscrewEnv(cfg)
E.force_entry = [0]
E.reset()
dev = E.device
DECI = int(getattr(E.cfg, "decimation", 12))
sql = np.asarray(np.load(TC.PRIOR_AUX)["squeeze"], np.float64).reshape(-1)[7:29]
ref = E.ref58.cpu().numpy()
dsq_l = torch.tensor(BL * (sql - ref[E.IA0, 36:58]), dtype=torch.float32,
                     device=dev)
APP, IA0, IA1 = E.APP_END, E.IA0, E.IA1
Nrow = E.PB.N_ROW


def drive(row, sL):
    r = min(row, E.T_ROW - 1)
    tgt = E.ref58[r].clone()
    tgt[36:58] += sL * dsq_l
    full = E.hand.data.joint_pos.clone()
    full[0, E.map_ids_t] = tgt
    E.hand.set_joint_position_target(full)
    for _ in range(DECI):
        E.scene.write_data_to_sim()
        E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())
        E._SA.apply_screw(E)


print(f"[v2] 第一幕: Approach 照谱 + 缝1 βL={BL} 渐入", flush=True)
for r in range(0, APP):
    drive(r, 0.0)
seam1 = IA0 - APP
for i, r in enumerate(range(APP, IA0)):
    drive(r, (i + 1) / max(seam1, 1))
for _ in range(30):
    drive(IA0, 1.0)
f = E._pads_f().norm(dim=-1)[0]
print(f"[v2] 站位垫: L{int((f[:5] > 0.5).sum())}/R{int((f[5:] > 0.5).sum())}",
      flush=True)

ik = {"right": ArmIK("right", anchor_link="arm_center", anchor_T=E._anchor_T),
      "left": ArmIK("left", anchor_link="arm_center", anchor_T=E._anchor_T)}
ref_obj = {oi: E.PB.ref_obj[oi].cpu().numpy() for oi in (0, 1)}
side_obj = {"right": 1, "left": 0}      # [TASK] 右手跟盖, 左手跟瓶
q_ik = {s: np.zeros((Nrow, 7)) for s in ("right", "left")}
cert, q_seed, w0 = {}, {}, {}
for s in ("right", "left"):
    sl = slice(0, 7) if s == "right" else slice(7, 14)
    q_seed[s] = E.hand.data.joint_pos[0, E.map_ids_t[sl]].cpu().numpy() \
        .astype(np.float64)
    w0[s] = ik[s].fk(q_seed[s])
rng = np.random.default_rng(17)
for s in ("right", "left"):
    best = None
    for trial in range(12):
        q0t = q_seed[s] if trial == 0 else q_seed[s] + rng.normal(0, 0.03, 7)
        r5 = ik[s].solve(w0[s][0] + np.array([0, 0, 0.015]), w0[s][1],
                         q0=q0t, iters=300)
        if best is None or r5["pos_err"] < best["pos_err"]:
            best = r5
        if best["pos_err"] < 0.001:
            break
    cert[s] = np.asarray(best["q"], np.float64)
    print(f"[v2] 认证行IK {s}: pos_err={best['pos_err'] * 1000:.2f}mm", flush=True)
fail_ik, err_max = 0, 0.0
for k in range(Nrow):
    for s in ("right", "left"):
        oi = side_obj[s]
        p0, q0_ = ref_obj[oi][0][:3], ref_obj[oi][0][3:7]
        pk, qk_ = ref_obj[oi][k][:3], ref_obj[oi][k][3:7]
        Rk = quat_to_R(qk_) @ quat_to_R(q0_).T
        tp = pk - Rk @ p0
        r = ik[s].solve(Rk @ w0[s][0] + tp, Rk @ w0[s][1], q0=q_seed[s],
                        iters=60)
        if r["pos_err"] > 0.01:
            fail_ik += 1
        err_max = max(err_max, float(r["pos_err"]))
        q_ik[s][k] = r["q"]
        q_seed[s] = np.asarray(r["q"], np.float64)
print(f"[v2] 交互IK: >1cm 失败 {fail_ik}/{Nrow * 2} 最大误差={err_max * 100:.2f}cm",
      flush=True)

d1 = dict(np.load(V1, allow_pickle=True))
v2r = np.asarray(d1["right_q"], np.float64).copy()
v2l = np.asarray(d1["left_q"], np.float64).copy()
hum_r, hum_l = v2r.copy(), v2l.copy()
v2r[IA0:IA1 + 1] = q_ik["right"]
v2l[IA0:IA1 + 1] = q_ik["left"]
# 缝2 重融接 (新交互末行 → retreat 首行)
RET0 = E.RETREAT0
SEAM2 = RET0 - IA1 - 1
for i in range(SEAM2):
    a = (i + 1) / (SEAM2 + 1)
    rr = IA1 + 1 + i
    v2r[rr] = (1 - a) * v2r[IA1] + a * v2r[RET0]
    v2l[rr] = (1 - a) * v2l[IA1] + a * v2l[RET0]
with open(V1, "rb") as fh:
    parent_md5 = hashlib.md5(fh.read()).hexdigest()[:8]
out = dict(d1)
out.update(right_q=v2r, left_q=v2l,
           human_right_q=hum_r, human_left_q=hum_l,
           human_right_f=np.asarray(d1["right_f"], np.float64).copy(),
           human_left_f=np.asarray(d1["left_f"], np.float64).copy(),
           cert_arm7_right=cert["right"], cert_arm7_left=cert["left"],
           meta_v2=np.array(f"gen=unscrew_v2;parent_v1_md5={parent_md5};"
                            f"betaL={BL};betaR={BR};ik=delta_space;"
                            f"clip={TC.CLIP_ID}"))
np.savez(OUT, **out)
with open(OUT, "rb") as fh:
    print(f"[v2] 已写 {OUT} md5={hashlib.md5(fh.read()).hexdigest()[:8]}",
          flush=True)
print("[v2] 完毕", flush=True)
try:
    _slot.release()
except Exception:
    pass
app.close()
os._exit(0)
