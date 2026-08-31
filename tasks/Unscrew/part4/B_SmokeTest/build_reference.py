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
dsq_l = torch.tensor(np.clip(BL * (sql - ref[E.IA0, 36:58]), -TC.SQUEEZE_DELTA_CAP,
                          TC.SQUEEZE_DELTA_CAP), dtype=torch.float32,
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
    E._update_screw_drive_gain()
    for _ in range(DECI):
        E._SA.apply_screw(E)
        E.scene.write_data_to_sim()
        E.sim.step(render=False)
        E.scene.update(E.sim.get_physics_dt())


print(f"[v2] 第一幕: Approach 照谱 + 缝1 βL={BL} 渐入", flush=True)
for r in range(0, APP):
    drive(r, 0.0)
seam1 = IA0 - APP
for i, r in enumerate(range(APP, IA0)):
    drive(r, float(TC.seam_squeeze_profile((i + 1) / max(seam1, 1))))
for _ in range(30):
    drive(IA0, 1.0)
f = E._pads_f().norm(dim=-1)[0]
print(f"[v2] 站位垫: L{int((f[:5] > 0.5).sum())}/R{int((f[5:] > 0.5).sum())}",
      flush=True)

ik = {side: ArmIK(side, anchor_link="arm_center",
                  anchor_T=TC.rest_anchor_T(side)) for side in ("right", "left")}
ref_obj = {oi: E.PB.ref_obj[oi].cpu().numpy() for oi in (0, 1)}
side_obj = {"right": 1, "left": 0}      # [TASK] 右手跟盖, 左手跟瓶
# 增量来源 (2026-08-31 改): 用 **v1 腕轨迹自己的增量**, 不再用物体增量。
# v2 的本意是"把腕参考重锚到 Isaac 实测站位"(消掉离线锚的系统差), 增量应当
# 忠实复现 v1 的腕几何。物体增量只有在"腕与物体刚性固连"时才等价, 而 v1 的
# 右腕口径是"手在盖正上方 reach·螺轴"(不随盖自转走) —— 盖的自转规范漂移
# (±140°) 会被物体增量当成刚体旋转, 把腕目标甩上一个 23.5cm 半径的大圆弧,
# 实测右臂 95/103 行解不出来 (v1 同一条轨迹是 77% 达标)。
_z1 = np.load(TC.REF_V1, allow_pickle=True)
wt_v1 = {"right": np.asarray(_z1["wrist_tgt_r"], np.float64),
         "left": np.asarray(_z1["wrist_tgt_l"], np.float64)}
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
        score = r5["pos_err"] ** 2 + (0.35 * r5["rot_err"]) ** 2
        best_score = (float("inf") if best is None else
                      best["pos_err"] ** 2 + (0.35 * best["rot_err"]) ** 2)
        if score < best_score:
            best = r5
        if best["pos_err"] < 0.001 and best["rot_err"] < 0.05:
            break
    cert[s] = np.asarray(best["q"], np.float64)
    print(f"[v2] 认证行IK {s}: pos={best['pos_err'] * 1000:.2f}mm "
          f"rot={np.degrees(best['rot_err']):.2f}°", flush=True)
# 逐行 IK: 热启保连续 / 坏行冻结上一行 / 逐行跳变上限 30° (2026-08-30 修)。
# 旧版把**失败解**直接当下一行的种子, 一次解崩顺着热启链把整段拖进限位角落
# —— 实测右臂对自己的腕目标中位差 44.7cm、100/103 行贴限, 整段搬运冻死。
fail_pos, fail_rot, critical_bad, pos_max, rot_max = 0, 0, 0, 0.0, 0.0
frozen = {"right": 0, "left": 0}
frozen_run = {"right": 0, "left": 0}
_rngb = np.random.default_rng(23)
for k in range(Nrow):
    for s in ("right", "left"):
        p0, q0_ = wt_v1[s][0][:3], wt_v1[s][0][3:7]
        pk, qk_ = wt_v1[s][k][:3], wt_v1[s][k][3:7]
        Rk = quat_to_R(qk_) @ quat_to_R(q0_).T
        tgt_p = w0[s][0] + (pk - p0)          # 平移增量原样搬到实测站位
        tgt_R = Rk @ w0[s][1]                 # 姿态增量左乘 (世界系)
        seeds = [q_seed[s], None]
        if k % 12 == 0:
            seeds += [_rngb.uniform(ik[s].lower, ik[s].upper) for _ in range(2)]
        best = None
        for q0s in seeds:
            r = ik[s].solve(tgt_p, tgt_R, q0=q0s, iters=200)
            sc = r["pos_err"] + 0.25 * r["rot_err"]
            if best is None or sc < best[0]:
                best = (sc, r)
        r = best[1]
        # 判据与 make_reference 的 solve_side 对齐 (2cm/10°/跳变60°/连冻2行放行):
        # v2 原来用 1cm 且没有"连冻放行", 一处解不出来就顺着热启链把整段冻死
        # (实测右臂冻结 98/103, 而同一条轨迹 v1 是 79% 达标)。
        good = (np.isfinite(r["q"]).all() and r["pos_err"] < 0.02
                and r["rot_err"] < np.radians(10)
                and (frozen_run[s] >= 2
                     or np.abs(np.asarray(r["q"], np.float64)
                               - q_seed[s]).max() < np.radians(60)))
        if good or k == 0:
            q_ik[s][k] = r["q"]
            q_seed[s] = np.asarray(r["q"], np.float64)
            pe_k, re_k = float(r["pos_err"]), float(r["rot_err"])
            frozen_run[s] = 0
        else:                       # 冻结上一行: 不跳分支, 误差如实入账
            q_ik[s][k] = q_seed[s]
            frozen[s] += 1
            frozen_run[s] += 1
            fp, fR = ik[s].fk(q_seed[s])
            pe_k = float(np.linalg.norm(fp - tgt_p))
            re_k = float(np.arccos(np.clip(
                (np.trace(fR.T @ tgt_R) - 1) * 0.5, -1, 1)))
        if pe_k > 0.01:
            fail_pos += 1
        if re_k > np.radians(10):
            fail_rot += 1
        if (pe_k > 0.01 or re_k > np.radians(10)) and \
                max(E.PB.k_sep - 40, 0) <= k <= E.PB.k_sep:
            critical_bad += 1
        pos_max = max(pos_max, pe_k)
        rot_max = max(rot_max, re_k)
_mg = {s: np.degrees(np.minimum(q_ik[s] - ik[s].lower,
                                ik[s].upper - q_ik[s]).min(axis=1))
       for s in ("right", "left")}
print(f"[v2] 交互IK: pos>1cm {fail_pos}/{Nrow * 2} | rot>10° "
      f"{fail_rot}/{Nrow * 2} | 关键窗坏行 {critical_bad} | "
      f"最大={pos_max * 100:.2f}cm/{np.degrees(rot_max):.1f}° | 冻结 "
      f"R{frozen['right']}/L{frozen['left']} 行 | 限位余量中位 "
      f"R{np.median(_mg['right']):.1f}°/L{np.median(_mg['left']):.1f}°",
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
assert np.isfinite(v2r).all() and np.isfinite(v2l).all(), \
    "v2 arm trajectories contain NaN/Inf"
out = dict(d1)
out.update(right_q=v2r, left_q=v2l,
           human_right_q=hum_r, human_left_q=hum_l,
           human_right_f=np.asarray(d1["right_f"], np.float64).copy(),
           human_left_f=np.asarray(d1["left_f"], np.float64).copy(),
           cert_arm7_right=cert["right"], cert_arm7_left=cert["left"],
           meta_v2=np.array(f"gen=unscrew_v2;parent_v1_md5={parent_md5};"
                            f"betaL={BL};betaR={BR};ik=wrist_delta;"
                            f"fail_pos_gt_1cm={fail_pos};fail_rot_gt_10deg={fail_rot};"
                            f"critical_bad={critical_bad};pos_max_cm={pos_max*100:.3f};"
                            f"rot_max_deg={np.degrees(rot_max):.3f};"
                            f"frozen_r={frozen['right']};frozen_l={frozen['left']};"
                            f"marg_r_deg={np.median(_mg['right']):.2f};"
                            f"marg_l_deg={np.median(_mg['left']):.2f};"
                            f"clip={TC.CLIP_ID}"))
tmp_out = OUT + ".tmp"
with open(tmp_out, "wb") as fh:
    np.savez(fh, **out)
os.replace(tmp_out, OUT)
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
