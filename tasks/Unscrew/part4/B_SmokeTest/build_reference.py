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
from rl_rebuild.correction.kinematics import ArmIK, _so3_log, quat_to_R  # noqa: E402

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
# ---- T2-39 臂外壳离桌约束 (2026-09-07) ----
# 旧 v2 母带右前臂 (R_arm_l5) 在行 212~228 穿桌 -1.8~-3.0cm: PhysX 把整臂顶起
# ~3cm, 手被带离盖 (c 出生 写入 5.2cm → 稳态 8.8cm)。基类的臂外壳撞桌罚是逐侧
# 的, Unscrew 交互手=left, 右臂从未被检查。这里用同一份外壳点云 + URDF FK 直接
# 在 IK 里约束: **手位姿不动**, 沿 7 自由度零空间抬肘, 直到外壳离桌 >= SHELL_CLEAR
# (基类罚的余量是 0.8cm, 这里多留 4mm 给残差)。离线口径比 in-sim 一子步读数更
# 深 ~1cm (PhysX 一子步已部分顶出), 作为约束是保守的。
SHELL_CLEAR = 0.012
# 只约束右臂: 左臂 IA 段外壳最深 -0.2cm (in-sim 腕漂移 0.2cm, 无害), 且左手正握着
# 瓶 —— 抬左腕会让手相对物体轨迹 (leash/clock 的真值) 抬高, 破坏抓握几何。
# 首次全侧约束实测把左臂 33 行抬了 ≤22mm, 撤回 (2026-09-07)。
SHELL_SIDES = ("right",)
_SHELL = np.load(os.path.join(TC.REPO, "tasks", "pregrasp", "arm_shell_points.npz"))
_TABLE_Z = float(E.cfg.table_top_z)
_shell_links = {s: [(k, np.asarray(_SHELL[k], np.float64)) for k in _SHELL.files
                    if f"{'R' if s == 'right' else 'L'}_arm" in k]
                for s in ("right", "left")}
assert all(_shell_links[s] for s in _shell_links), "外壳点云缺臂侧连杆"


def shell_gap(s, q):
    """该侧臂连杆外壳最低点离桌面 (m, 负=穿桌)."""
    g = np.inf
    for lk, pts in _shell_links[s]:
        T = ik[s].link_pose_world(lk, q)
        g = min(g, float((T[2, 3] + (T[:3, :3] @ pts.T)[2]).min()) - _TABLE_Z)
    return g


def wrist_fix(s, q, p_tgt, R_tgt, n=6):
    """行空间 Gauss-Newton 腕修正 (最小范数, 姿态贴原带, 不游走)."""
    q = np.asarray(q, np.float64).copy()
    for _ in range(n):
        pc, Rc = ik[s].fk(q)
        e = np.concatenate([p_tgt - pc, _so3_log(R_tgt @ Rc.T)])
        if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-3:
            break
        J = ik[s].jacobian(q)
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e),
                    ik[s].lower, ik[s].upper)
    return q


def lift_clear(s, q, p_tgt, R_tgt, dz_max=0.06):
    """最小抬腕: 腕目标只沿 +z 逐毫米抬高, 直到外壳离桌 >= SHELL_CLEAR.
    返回 (q, gap, 抬高量 dz)。

    为什么不是零空间旋肘: 2026-09-07 离线实测, 手位姿钉死时旋肘圆上的离桌极大只有
    -1.3cm (行 213~226), 无解; "先旋肘到极大再抬腕"会落到关节改变 82° 的另一支
    (拧盖前 0.4s 内 80° 甩臂)。只抬腕 = 最小范数改动: 关节改变 <=10°, 行间跳变
    <=8°, 最大抬高 44mm (行 216~225) —— 物理本来就把手顶高 ~32mm (带接触力),
    这里只是把同一个抬高做成无接触力、带 1.2cm 余量的**有意**参考。
    """
    q = np.asarray(q, np.float64).copy()
    g = shell_gap(s, q)
    dz = 0.0
    while g < SHELL_CLEAR and dz < dz_max:
        dz += 0.001
        q = wrist_fix(s, q, p_tgt + np.array([0.0, 0.0, dz]), R_tgt)
        g = shell_gap(s, q)
    return q, g, dz
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
_rows_v1 = np.flatnonzero(np.asarray(_z1["source"], np.int8) == 1)
arm_v1 = {"right": np.asarray(_z1["right_q"], np.float64)[_rows_v1],
          "left": np.asarray(_z1["left_q"], np.float64)[_rows_v1]}
assert len(arm_v1["right"]) == len(arm_v1["left"]) == Nrow
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
# 逐行 IK: 只沿站位行所在的**局部冗余分支**走。允许几毫米连续误差交给 residual，
# 不允许为了单帧纸面精度在不同肘解之间来回切换。旧版把 q_seed/v1/default 三种
# 初值的结果按几何误差直接取最小，clip32 左臂在平滑物轨上因此出现 32 个 >15°
# 单步跳变；后面的 20° 限速只会把支路切换摊成多行甩臂，物理抓持必然丢瓶。
fail_pos, fail_rot, critical_bad, pos_max, rot_max = 0, 0, 0, 0.0, 0.0
frozen = {"right": 0, "left": 0}
held_count = {"right": 0, "left": 0}
lifted = {"right": 0, "left": 0}          # T2-39: 逐行抬腕次数 / 抬不到位次数
lift_fail = {"right": 0, "left": 0}
rise = {s: np.zeros(Nrow) for s in ("right", "left")}   # 逐行腕抬高量 (m)
gap_final = {}
_rngb = np.random.default_rng(23)
_BRANCH = np.radians(25.0)
tgtP = {s: np.zeros((Nrow, 3)) for s in ("right", "left")}      # 逐行目标 (限速后重投影用)
tgtR = {s: np.zeros((Nrow, 3, 3)) for s in ("right", "left")}
for k in range(Nrow):
    for s in ("right", "left"):
        p0, q0_ = wt_v1[s][0][:3], wt_v1[s][0][3:7]
        pk, qk_ = wt_v1[s][k][:3], wt_v1[s][k][3:7]
        Rk = quat_to_R(qk_) @ quat_to_R(q0_).T
        tgt_p = w0[s][0] + (pk - p0)          # 平移增量原样搬到实测站位
        tgt_R = Rk @ w0[s][1]                 # 姿态增量左乘 (世界系)
        tgtP[s][k], tgtR[s][k] = tgt_p, tgt_R
        # “原地保持”也是合法候选；若当前目标暂时难解，就宁可留一点连续误差，
        # 也不接纳离上一行超过 25° 的远端解。v1 当前行仍可作初值，但它求出的
        # 结果必须落回同一个局部分支才有资格参与比较。
        hp, hR = ik[s].fk(q_seed[s])
        hpe = float(np.linalg.norm(hp - tgt_p))
        hre = float(np.arccos(np.clip(
            (np.trace(hR.T @ tgt_R) - 1) * 0.5, -1, 1)))
        candidates = [(hpe + 0.25 * hre, hpe, hre,
                       q_seed[s].copy(), True)]
        local = np.clip(q_seed[s] + _rngb.normal(0.0, 0.04, 7),
                        ik[s].lower, ik[s].upper)
        seeds = [q_seed[s], local, arm_v1[s][k]]
        if k % 12 == 0:
            seeds += [None] + [
                _rngb.uniform(ik[s].lower, ik[s].upper) for _ in range(2)]
        for q0s in seeds:
            r = ik[s].solve(tgt_p, tgt_R, q0=q0s, iters=200)
            qr = np.asarray(r["q"], np.float64)
            if not np.isfinite(qr).all():
                continue
            dq = float(np.abs(qr - q_seed[s]).max())
            if dq > _BRANCH:
                continue
            base = float(r["pos_err"] + 0.25 * r["rot_err"])
            candidates.append((base + 0.02 * dq,
                               float(r["pos_err"]), float(r["rot_err"]),
                               qr, False))
        _, pe_k, re_k, q_new, held = min(candidates, key=lambda x: x[0])
        # T2-39: 外壳穿桌/贴桌的解先抬腕再落盘; 抬高写回 tgtP (下游重投影/统计
        # 以抬高后的目标为准); 抬后的解作为下一行种子, 连续传播
        if s in SHELL_SIDES and shell_gap(s, q_new) < SHELL_CLEAR:
            q_new, g_row, dz = lift_clear(s, q_new, tgt_p, tgt_R)
            lifted[s] += 1
            lift_fail[s] += int(g_row < SHELL_CLEAR)
            rise[s][k] += dz
            tgtP[s][k] = tgt_p = tgt_p + np.array([0.0, 0.0, dz])
            _fp, _fR = ik[s].fk(q_new)
            pe_k = float(np.linalg.norm(_fp - tgt_p))
            re_k = float(np.arccos(np.clip(
                (np.trace(_fR.T @ tgt_R) - 1) * 0.5, -1, 1)))
        q_ik[s][k] = q_new
        q_seed[s] = q_new.copy()
        if held:
            held_count[s] += 1
            # 静止/极慢目标下保持上一行本来就是最优连续解，不算坏冻结。
            # frozen_* 只统计保持后仍超 correction 基线的行，供验收判断可救性。
            frozen[s] += int(pe_k > 0.01 or re_k > np.radians(10))
        if pe_k > 0.01:
            fail_pos += 1
        if re_k > np.radians(10):
            fail_rot += 1
        if (pe_k > 0.01 or re_k > np.radians(10)) and \
                max(E.PB.k_sep - 40, 0) <= k <= E.PB.k_sep:
            critical_bad += 1
        pos_max = max(pos_max, pe_k)
        rot_max = max(rot_max, re_k)
# ---- 逐行限速 + 轻平滑 + 重投影 (2026-09-01, 与 make_reference.solve_side 同法) ----
# 即使局部分支求解已约束连续，数值重投影仍可能放大单行改变量；保留最终限速作为
# 独立安全栏。旧版曾出现右臂 96°、左臂 46° 双向尖刺，PD 臂一步就会把瓶打飞。
_RATE = np.radians(20.0)
from scipy.ndimage import gaussian_filter1d  # noqa: E402
for s in ("right", "left"):
    q = q_ik[s]
    for k in range(1, Nrow):
        d = q[k] - q[k - 1]
        m = np.abs(d).max()
        if m > _RATE:
            q[k] = q[k - 1] + d * (_RATE / m)
    qs = gaussian_filter1d(q, sigma=0.5, axis=0, mode="nearest")
    n_reproj = 0
    for k in range(Nrow):
        fp0, fR0 = ik[s].fk(q[k])
        e0 = float(np.linalg.norm(fp0 - tgtP[s][k])) + 0.25 * float(np.arccos(np.clip(
            (np.trace(fR0.T @ tgtR[s][k]) - 1) * 0.5, -1, 1)))
        r = ik[s].solve(tgtP[s][k], tgtR[s][k], q0=qs[k], iters=80, w_rot=0.25)
        near = np.abs(np.asarray(r["q"], float) - q[k]).max() < np.radians(20)
        if near and (r["pos_err"] + 0.25 * r["rot_err"]) <= e0:
            q[k] = r["q"]
            n_reproj += 1
    # 二次限速: 重投影允许相邻两行各自在 20° 内反向挪动, 净跳变可到 ~40°
    # (实测右臂 38.1°/左臂 21.7°)。再压一遍, 让 20° 上限真正成立。
    for k in range(1, Nrow):
        d = q[k] - q[k - 1]
        m = np.abs(d).max()
        if m > _RATE:
            q[k] = q[k - 1] + d * (_RATE / m)
    # T2-39: 限速/平滑/重投影可能把前臂带回桌下 —— 终检二次抬腕, 再压一遍限速
    n_relift = 0
    for k in range(Nrow):
        if s in SHELL_SIDES and shell_gap(s, q[k]) < SHELL_CLEAR:
            q[k], _, dz2 = lift_clear(s, q[k], tgtP[s][k], tgtR[s][k])
            rise[s][k] += dz2
            tgtP[s][k] = tgtP[s][k] + np.array([0.0, 0.0, dz2])
            n_relift += 1
    for k in range(1, Nrow):
        d = q[k] - q[k - 1]
        m = np.abs(d).max()
        if m > _RATE:
            q[k] = q[k - 1] + d * (_RATE / m)
    gaps = np.array([shell_gap(s, q[k]) for k in range(Nrow)])
    gap_final[s] = float(gaps.min())
    print(f"[v2] {s} 外壳离桌 (T2-39): 逐行抬腕 {lifted[s]} 行 (抬不到位 {lift_fail[s]}) "
          f"| 终检二次抬腕 {n_relift} 行 | 最大腕抬高 {rise[s].max() * 1000:.0f}mm "
          f"| 最终最小离桌 {gaps.min() * 100:+.2f}cm (要求 >= {SHELL_CLEAR * 100:.1f}cm) "
          f"| 仍低于要求 {int((gaps < SHELL_CLEAR).sum())} 行", flush=True)
    jump = np.degrees(np.abs(np.diff(q, axis=0)).max()) if Nrow > 1 else 0.0
    print(f"[v2] {s} 限速/重投影: 最大逐行跳变 {jump:.1f}° (上限 20°) | 重投影采纳 {n_reproj}/{Nrow}",
          flush=True)
# 最终诊断必须针对“限速+平滑+重投影”后的实际写盘轨迹重算。旧代码沿用
# 限速前的计数，meta_v2 与 probe_ikcheck 会对不上。
fail_pos, fail_rot, critical_bad, pos_max, rot_max = 0, 0, 0, 0.0, 0.0
for s in ("right", "left"):
    for k in range(Nrow):
        fp, fR = ik[s].fk(q_ik[s][k])
        pe_k = float(np.linalg.norm(fp - tgtP[s][k]))
        re_k = float(np.arccos(np.clip(
            (np.trace(fR.T @ tgtR[s][k]) - 1) * 0.5, -1, 1)))
        fail_pos += int(pe_k > 0.01)
        fail_rot += int(re_k > np.radians(10))
        if ((pe_k > 0.01 or re_k > np.radians(10))
                and max(E.PB.k_sep - 40, 0) <= k <= E.PB.k_sep):
            critical_bad += 1
        pos_max = max(pos_max, pe_k)
        rot_max = max(rot_max, re_k)
_mg = {s: np.degrees(np.minimum(q_ik[s] - ik[s].lower,
                                ik[s].upper - q_ik[s]).min(axis=1))
       for s in ("right", "left")}
print(f"[v2] 交互IK: pos>1cm {fail_pos}/{Nrow * 2} | rot>10° "
      f"{fail_rot}/{Nrow * 2} | 关键窗坏行 {critical_bad} | "
      f"最大={pos_max * 100:.2f}cm/{np.degrees(rot_max):.1f}° | 冻结 "
      f"R{frozen['right']}/L{frozen['left']} 坏行 "
      f"(局部保持 R{held_count['right']}/L{held_count['left']}) | 限位余量中位 "
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


def _rise_full(s):
    """T2-39 逐行腕抬高量 (m), 全链长度, IA 外为 0 —— 供探针/录像对照."""
    full = np.zeros(len(v2r))
    full[IA0:IA1 + 1] = rise[s]
    return full
assert np.isfinite(v2r).all() and np.isfinite(v2l).all(), \
    "v2 arm trajectories contain NaN/Inf"
out = dict(d1)
out.update(right_q=v2r, left_q=v2l,
           human_right_q=hum_r, human_left_q=hum_l,
           human_right_f=np.asarray(d1.get("human_right_f", d1["right_f"]),
                                    np.float64).copy(),
           human_left_f=np.asarray(d1["left_f"], np.float64).copy(),
           cert_arm7_right=cert["right"], cert_arm7_left=cert["left"],
           wrist_rise_r=_rise_full("right"), wrist_rise_l=_rise_full("left"),
           meta_v2=np.array(f"gen=unscrew_v2;parent_v1_md5={parent_md5};"
                            f"betaL={BL};betaR={BR};ik=wrist_delta;"
                            f"shell_clear_cm={SHELL_CLEAR * 100:.1f};"
                            f"shell_min_r_cm={gap_final['right'] * 100:.2f};"
                            f"shell_min_l_cm={gap_final['left'] * 100:.2f};"
                            f"lift_r={lifted['right']};lift_l={lifted['left']};"
                            f"rise_max_r_mm={rise['right'].max() * 1000:.0f};"
                            f"rise_max_l_mm={rise['left'].max() * 1000:.0f};"
                            f"fail_pos_gt_1cm={fail_pos};fail_rot_gt_10deg={fail_rot};"
                            f"critical_bad={critical_bad};pos_max_cm={pos_max*100:.3f};"
                            f"rot_max_deg={np.degrees(rot_max):.3f};"
                            f"frozen_r={frozen['right']};frozen_l={frozen['left']};"
                            f"held_r={held_count['right']};held_l={held_count['left']};"
                            "struct_r=0;"
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
