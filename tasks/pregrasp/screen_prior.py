"""GraspPose 候选筛选 —— 三关一次跑完。判据与依据见 `docs/GRASPPOSE_SCREENING.md`。

    Gate 0  数据同一性   秒级   离线      mesh OBB 对不上 = 数据问题, 不是候选问题
    Gate 1  可达性       秒级   离线      yaw 扫描: 可达带是否覆盖**视频 yaw**
    Gate 2  prior 质量   ~2min  Isaac     pads* >= 4 且 Q* > 0

Gate 0/1 不需要 GPU, 所以一批几十个候选先秒级砍掉大半, 只有幸存者才付 Isaac 的钱。

用法
----
    PY=/home/lyh/luhr/MagicSim/.venv/bin/python
    SHARPA_WANDB=0 PYTHONPATH=. $PY -m tasks.pregrasp.screen_prior \\
        --clip Grasp1 \\
        --grasp_dir /home/lyh/Project/Dexonomy/output/pp1_sharpa_wave \\
        --info_json /home/lyh/Project/Dexonomy/assets/object/custom/processed_data/pp1/info/simplified.json

`--grasp_dir` 会**递归**收集 `*_grasp.npy` —— 故意同时吃 `grasp_data/` 和
`grasp_data_isaac_failed/`: Dexonomy 的 Isaac 验证器对本平台没有预测力,
不能拿它当候选池 (见规范 §0)。

  --offline_only   只跑 Gate 0/1 (不开 Isaac, 几秒出结果)
  --tol_deg        Gate 1 的 Δψ 容忍度, 默认 10 (严格档, 2026-08-01 定)
  --prior_dir      跳过 npy 转换, 直接吃已有的 prior npz

⚠ Dexonomy 的 npy 是 numpy2 pickle, **必须用系统 python3 转换** —— 本脚本会自动
  subprocess 调 `python3 make_prior.py`, 不要改成用 Isaac venv 的解释器。
⚠ Gate 2 每个候选起一个独立进程 (一个 Isaac session 建多个 env 不安全)。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PAD_NAMES = ("thumb", "index", "middle", "ring", "pinky")


# ---------------------------------------------------------------- 小工具
def quat_to_R(q):
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def Rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def geo_deg(A, B):
    """两个旋转矩阵的测地角距 (度)."""
    c = (np.trace(A @ B.T) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def ang_diff(a, b):
    """角度差 (度), 取环上最短."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


# ---------------------------------------------------------------- 视频 yaw
def video_yaw_deg(clip_cfg, canon_rot):
    """重建视频里物体静置段的 yaw (度), 以 canon_rot 为基准姿态.

    找 θ 使 Rz(θ)·R_canon 最接近视频姿态。返回 (θ, 残差角, 静置段稳定性)。
    残差角 = 非绕 z 的部分 —— 大 (>15°) 说明"哪个面朝下"和视频都不一致,
    此时 θ 没有意义 (Grasp3 曾出现 179.9° 翻面)。
    """
    from rl_rebuild.correction import frames as F
    z = np.load(clip_cfg["npz"], allow_pickle=True)
    op = z["obj_pose"]
    # 静置段 = 任一只手开始接触之前
    gs = len(op)
    for k in ("phase_right", "phase_left"):
        if k in z.files:
            idx = np.flatnonzero(np.asarray(z[k]).astype(int) == 1)
            if len(idx):
                gs = min(gs, int(idx[0]))
    gs = max(gs, 2)
    R_vid = [quat_to_R(op[i, 3:7]) for i in range(gs)]
    R_ref = R_vid[len(R_vid) // 2]
    stab = float(np.median([geo_deg(R, R_ref) for R in R_vid]))

    R_canon = quat_to_R(canon_rot)
    best = (1e9, 0.0)
    for deg in np.arange(0, 360, 0.5):
        e = geo_deg(Rz(np.radians(deg)) @ R_canon, R_ref)
        if e < best[0]:
            best = (e, float(deg))
    resid, theta = best
    _ = F  # 保持 import 语义 (mesh 顶点由调用方处理)
    return theta, resid, stab


def object_placement(clip_cfg, canon_rot, table_top_z):
    """物体在 env 世界系的摆放 (xy 相机锚定, z 贴桌). 纯离线, 与 env 实测吻合 <0.05mm."""
    from rl_rebuild.correction import frames as F
    from rl_rebuild.correction import place_camera as PC
    raw = np.load(clip_cfg["npz"], allow_pickle=True)
    # c2w 的三处候选统一走 place_camera.find_cam_src(与 replay_grasp 同一份逻辑)。
    # 之前这里只看网格旁, pour17 的软链在上一层, 会误报"缺 c2w 过不了 Gate 0"。
    _cam = PC.find_cam_src(clip_cfg["mesh"], clip_cfg["npz"])
    sxy = (PC.camera_anchor_shift(_cam, PC.ZED_NOMINAL[:2], raw["obj_pose"][0, :2])
           if _cam is not None else None)
    if sxy is None:
        raise RuntimeError("world_fused.npz 缺 c2w, 无法相机锚定 —— 这条 clip 过不了 Gate 0")
    v = F.load_obj_verts(clip_cfg["mesh"]) @ quat_to_R(canon_rot).T
    return np.array([sxy[0], sxy[1], table_top_z + 0.002 - float(v[:, 2].min())])


# ---------------------------------------------------------------- Gate 0
def gate0(clip_cfg, info_json):
    """网格同一性: 重建 mesh 的 OBB 必须与 Dexonomy 的逐位吻合 (±0.5mm)."""
    import trimesh
    d = json.load(open(info_json))
    dex = sorted(float(x) for x in d["obb"])
    m = trimesh.load(clip_cfg["mesh"], force="mesh")
    rec = sorted(float(x) for x in m.bounding_box_oriented.primitive.extents)
    dmax = max(abs(a - b) for a, b in zip(dex, rec))
    return dmax <= 0.0005, dex, rec, dmax


# ---------------------------------------------------------------- Gate 1
def gate1(prior_npz, obj_pos, video_yaw, tol_deg, step=5, hand="right"):
    """yaw 扫描: 可达带 Ψ 是否够到视频 yaw.

    ⚠ `hand` 必须跟 clip 的 `robot_hand` 走 —— 之前写死 "right", 拿右臂去筛左手候选
    (pour17 的杯是左手)会给出完全无意义的可达带。2026-08-15 修。
    """
    from rl_rebuild.correction.kinematics import ArmIK
    z = np.load(prior_npz)
    if "canon_rot" not in z.files:      # 老 prior(Screw27 等)没有这一项, 跳过而不是崩
        return dict(ok=False, n_reach=0, dpsi=None, best_yaw=None, best_err=float("nan"),
                    band="", skipped="无 canon_rot(老 prior)")
    canon = np.asarray(z["canon_rot"], np.float64)
    grasp = np.asarray(z["grasp"], np.float64)
    ik = ArmIK(hand)
    q = np.zeros(len(ik.arm_joints))
    q[0] = np.radians(-45.0)
    if len(q) > 3:
        q[3] = np.radians(-90.0)
    rows = []
    for deg in range(0, 360, step):
        a = np.radians(deg)
        oq2 = qmul(np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)]), canon)
        gp = quat_to_R(oq2) @ grasp[:3] + obj_pos
        gq = qmul(oq2, grasp[3:7])
        r = ik.solve(gp, quat_to_R(gq), q0=q, iters=80)
        if r["ok"]:
            q = r["q"]
        rows.append((float(deg), float(r["pos_err"])))
    reach = [d for d, e in rows if e < 0.01]
    if not reach:
        return dict(ok=False, n_reach=0, dpsi=None, best_yaw=None,
                    best_err=min(e for _, e in rows), band="")
    dpsi = min(ang_diff(d, video_yaw) for d in reach)
    # 可达带里离视频 yaw 最近的那个角 (这才是我们要用的 yaw, 不是 IK 最优的那个)
    best_yaw = min(reach, key=lambda d: ang_diff(d, video_yaw))
    best_err = dict(rows)[best_yaw]
    segs, s, p = [], reach[0], reach[0]
    for v in reach[1:]:
        if v - p > step:
            segs.append((s, p))
            s = v
        p = v
    segs.append((s, p))
    return dict(ok=dpsi <= tol_deg, n_reach=len(reach), dpsi=dpsi,
                best_yaw=best_yaw, best_err=best_err,
                band=", ".join(f"{a:.0f}-{b:.0f}" for a, b in segs))


# ---------------------------------------------------------------- Gate 1b (H3)
# 派生常数, 与 tasks/pregrasp 的管壁参数一致 (见 docs/PLAN_PICK_LIFT.md §4.1)
TUBE_SOFT_FRAC = 0.7      # 管壁从 0.7R 起就往回拉 -> 无阻力半径只有 0.7R
TUBE_JITTER = 0.015       # 物体位置课程最大抖动
TUBE_SLACK = 0.02         # IK 残差 + 控制跟踪滞后
TUBE_R_CAP = 0.20         # H3 判死线


def human_pregrasp_wrist(clip_cfg, clip, table_top_z=0.85):
    """人手参考轨迹在 **PreGrasp 帧** 的腕位姿 (env 系) + 该帧号.

    这是接近段的终点参考; H3 量的就是"它离 GraspPose 有多远".
    纯离线 (numpy), 与 env 里 ref builder 走同一条码路.
    """
    from rl_rebuild.correction.ref_builders.replay_grasp import load_replay_grasp
    du = load_replay_grasp(
        clip_cfg["npz"], clip_cfg["mesh"], usd_path=clip_cfg.get("usd", ""),
        clip_id=clip, hand="right", table_height=table_top_z,
        affordance_npz=clip_cfg.get("affordance"), semantics=clip_cfg.get("semantics"),
        # ⚠ 必须 False —— 2026-08-02 D7 之后训练侧已关掉 PreGrasp 对齐, 这里若还开着,
        # 算出来的 G 会比训练时**小一半以上** (Grasp3/8_5: 8.9cm vs 实测 17.8cm),
        # 因为对齐会把整条腕轨迹往物体方向挪 13.68cm.
        anchor_mode="camera", pregrasp_align=False)
    r = du.ref
    # grasp_phase_frame 是**合拢终点** ge; PreGrasp 帧 gs = ge - close_steps(30)
    gs = int(r.grasp.grasp_phase_frame) - 30
    W = np.asarray(r.track_wrist)
    return W[gs, :3].astype(np.float64), W[gs, 3:7].astype(np.float64), gs


def gate1b(prior_npz, obj_pos, yaw_deg, wrist_p, wrist_q):
    """H3 人手一致性: 缺口 G -> 派生管壁 R_hi, 超过 cap 判死. 见 GRASPPOSE_SCREENING Gate 1b."""
    z = np.load(prior_npz)
    canon = np.asarray(z["canon_rot"], np.float64)
    a = np.radians(yaw_deg)
    oq2 = qmul(np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)]), canon)
    gp = quat_to_R(oq2) @ np.asarray(z["grasp"][:3], np.float64) + obj_pos
    gq = qmul(oq2, np.asarray(z["grasp"][3:7], np.float64))
    G = float(np.linalg.norm(gp - wrist_p))
    dth = geo_deg(quat_to_R(gq), quat_to_R(wrist_q))
    R_hi = G / TUBE_SOFT_FRAC + TUBE_JITTER + TUBE_SLACK
    return dict(G=G, d_rot_deg=dth, R_hi=R_hi, ok=bool(R_hi <= TUBE_R_CAP))


# ---------------------------------------------------------------- Gate 2 (子进程)
def gate2_child(clip, prior_npz, yaw_deg=-1.0):
    """在**本进程**里开 Isaac 跑 Gate 2, 结果以 JSON 打到 stdout (由父进程解析)."""
    from isaaclab.app import AppLauncher
    import argparse as _ap
    _p = _ap.ArgumentParser()
    AppLauncher.add_app_launcher_args(_p)
    _a = _p.parse_args(["--headless"])

    from rl_rebuild.utils.gpu_guard import isaac_slot
    _slot = isaac_slot("screen_prior")
    app = AppLauncher(_a).app

    import torch
    from rl_rebuild.correction import clips as _clips
    from tasks.pregrasp.cfg import GraspTaskCfg
    from tasks.pregrasp.env import GraspTaskEnv

    cfg = GraspTaskCfg()
    _clips.configure_cfg(cfg, clip)
    cfg.grasp_prior_npz = prior_npz
    cfg.prior_yaw_deg = float(yaw_deg)      # D1: 钉死在 Gate 1 的 best_yaw
    cfg.scene.num_envs = 16
    cfg.obj_jitter_xy = 0.0
    cfg.closure_init_max = 0.0
    E = GraspTaskEnv(cfg)
    E.gentle = 1.0                       # 全价看信号量级
    N, A = cfg.scene.num_envs, cfg.action_space

    def sweep(thumb_a, steps, settle):
        """压到最深保持, 返回该设定下的稳态读数."""
        E.reset()
        act = torch.zeros((N, A), device=E.device)
        act[:, 7] = 1.0                  # c 全速到 closure_max
        act[:, 8] = thumb_a              # 拇指残差
        act[:, 9:] = 1.0                 # 四指压深
        for _ in range(settle):
            E.step(act)
        acc, nrec = None, 0
        for _ in range(steps):
            E.step(act)
            s = E._sig
            G = cfg.pad_force_sign * torch.cat(
                [c.data.force_matrix_w.view(N, 1, 3) for c in E._contact_sensors], dim=1)
            mag = G.nan_to_num(0.0).norm(dim=-1).clamp(max=50.0)
            over = (mag - cfg.squeeze_f_max).clamp(min=0.0).sum(dim=1)
            row = np.array([
                s["n_pads"].float().mean().item(), s["cent"].mean().item(),
                s["r_imb"].mean().item(), s["tau_n"].mean().item(),
                s["quality"].mean().item(), over.mean().item(),
                mag.max().item(), s["cand_ok"].float().mean().item(),
            ] + mag.mean(dim=0).tolist())
            acc = row if acc is None else acc + row
            nrec += 1
        return acc / max(nrec, 1)

    # ---- 深度轴 (五指全开) + 拇指轴 ----
    settings = [1.0, 0.0, -0.25, -0.5, -0.75, -1.0]
    recs = {}
    for t in settings:
        recs[t] = sweep(t, steps=8, settle=48)

    # 标称位姿 (c=1.0 附近) 的穿透量: 用较浅的 settle 采一次
    nominal = sweep(1.0, steps=6, settle=24)

    Qs = {t: r[4] for t, r in recs.items()}
    Ps = {t: r[0] for t, r in recs.items()}
    t_best = max(Qs, key=lambda k: Qs[k])
    r = recs[t_best]
    pads_star = max(Ps.values())
    n_pass_pads = sum(1 for v in Ps.values() if v >= cfg.success_min_pads)
    pad_f = r[8:13]
    share = float(max(pad_f) / max(sum(pad_f), 1e-6))

    out = dict(
        clip=clip, prior=os.path.basename(prior_npz),
        Q_star=float(max(Qs.values())), pads_star=float(pads_star),
        thumb_a_best=float(t_best),
        cent=float(r[1]), imb=float(r[2]), tau_n=float(r[3]),
        over_N=float(r[5]), peak_pad_N=float(r[6]), cand_ok=float(r[7]),
        pad_forces={n: float(v) for n, v in zip(PAD_NAMES, pad_f)},
        max_pad_share=share,
        n_settings_pads_ok=int(n_pass_pads),      # 0 = 悬崖 (无任何档能拿到 >=4 垫)
        nominal_peak_pad_N=float(nominal[6]),
        min_pads_required=int(cfg.success_min_pads),
    )
    out["H1_pads"] = bool(out["pads_star"] >= cfg.success_min_pads)
    out["H2_Q"] = bool(out["Q_star"] > 0.0)
    out["pass"] = bool(out["H1_pads"] and out["H2_Q"])
    print("###GATE2_JSON###" + json.dumps(out), flush=True)

    E.close()
    app.close()


def gate2(clip, prior_npz, yaw_deg=-1.0, timeout=1800):
    """起独立进程跑 Gate 2 (一个 Isaac session 建多个 env 不安全)."""
    env = dict(os.environ, PYTHONPATH=REPO, SHARPA_WANDB="0")
    cmd = [sys.executable, "-u", "-m", "tasks.pregrasp.screen_prior",
           "--_gate2_child", "--clip", clip, "--prior", prior_npz,
           "--prior_yaw", str(yaw_deg)]
    try:
        p = subprocess.run(cmd, cwd=REPO, env=env, timeout=timeout,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return dict(error="timeout")
    for line in p.stdout.splitlines():
        if line.startswith("###GATE2_JSON###"):
            return json.loads(line[len("###GATE2_JSON###"):])
    tail = "\n".join((p.stdout + p.stderr).splitlines()[-15:])
    return dict(error="no result", tail=tail)


# ---------------------------------------------------------------- 转换
def convert(npy, info_json, out_npz):
    """Dexonomy npy -> prior npz. **必须用系统 python3** (npy 是 numpy2 pickle)."""
    cmd = ["python3", os.path.join(REPO, "tasks/pregrasp/make_prior.py"),
           "--grasp_npy", npy, "--info_json", info_json, "--out", out_npz]
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    return p.returncode == 0, (p.stdout + p.stderr).strip().splitlines()[-1:] or [""]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--grasp_dir", help="递归收集 *_grasp.npy (含 isaac_failed)")
    ap.add_argument("--prior_dir", help="直接吃已转换的 prior npz")
    ap.add_argument("--info_json", help="Dexonomy simplified.json (Gate 0 + 转换需要)")
    ap.add_argument("--out_dir", default=None, help="转换产物落盘处")
    ap.add_argument("--tol_deg", type=float, default=10.0,
                    help="Gate 1 的 Δψ 容忍度 (默认 10 = 严格档)")
    ap.add_argument("--offline_only", action="store_true", help="只跑 Gate 0/1/1b")
    ap.add_argument("--no_h3", action="store_true",
                    help="跳过 Gate 1b —— 只在**没有接近段**的任务上才该用")
    ap.add_argument("--table_top_z", type=float, default=0.85)
    ap.add_argument("--_gate2_child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--prior", help=argparse.SUPPRESS)
    ap.add_argument("--prior_yaw", type=float, default=-1.0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._gate2_child:
        return gate2_child(args.clip, args.prior, args.prior_yaw)

    from rl_rebuild.correction import clips as _clips
    clip_cfg = _clips.CLIPS[args.clip]
    # 用 --prior_dir 时报告落在同一个目录里, 别新建一个 <clip>_candidates 让人找不到
    out_dir = (args.out_dir or args.prior_dir
               or os.path.join(REPO, f"tasks/pregrasp/priors/{args.clip}_candidates"))
    os.makedirs(out_dir, exist_ok=True)

    # ---- 收集候选 ----
    if args.prior_dir:
        priors = sorted(glob.glob(os.path.join(args.prior_dir, "*.npz")))
    else:
        if not (args.grasp_dir and args.info_json):
            ap.error("--grasp_dir 需要配 --info_json (或改用 --prior_dir)")
        npys = sorted(glob.glob(os.path.join(args.grasp_dir, "**", "*_grasp.npy"),
                                recursive=True))
        print(f"[collect] {len(npys)} 个 Dexonomy 候选 (含 isaac_failed)", flush=True)
        priors = []
        for f in npys:
            tag = os.path.basename(f).replace("_grasp.npy", "")
            o = os.path.join(out_dir, f"{tag}.npz")
            ok, msg = convert(f, args.info_json, o)
            if ok:
                priors.append(o)
            else:
                print(f"  ✗ {tag} 转换失败: {msg}", flush=True)
    if not priors:
        print("没有可筛的候选"); return 1

    # ---- Gate 0 ----
    print("\n" + "=" * 78)
    if args.info_json:
        ok, dex, rec, dmax = gate0(clip_cfg, args.info_json)
        print(f"[Gate 0] 网格同一性: Dexonomy OBB {[round(x,4) for x in dex]}")
        print(f"         重建 OBB          {[round(x,4) for x in rec]}   最大差 {dmax*1000:.2f}mm")
        if not ok:
            print("         ❌ 不吻合 —— 尺度/网格版本不一致, 先解决这个, 筛选无意义")
            return 1
        print("         ✅ 通过")
    else:
        print("[Gate 0] 跳过 (未给 --info_json)")

    canon = np.asarray(np.load(priors[0])["canon_rot"], np.float64)
    obj_pos = object_placement(clip_cfg, canon, args.table_top_z)
    vy, resid, stab = video_yaw_deg(clip_cfg, canon)
    print(f"\n[基准] 物体摆放 xy=({obj_pos[0]:+.4f}, {obj_pos[1]:+.4f}) z={obj_pos[2]:.4f}")
    print(f"       视频 yaw = {vy:.1f}°  (静置段稳定性 {stab:.1f}°, 翻面残差 {resid:.1f}°)")
    if resid > 15:
        print(f"       ⚠ 翻面残差 {resid:.1f}° 偏大 —— canon 姿态与视频**朝下的面**不一致, "
              f"视频 yaw 无意义, Gate 1 结果不可信")

    # ---- Gate 1 ----
    print(f"\n[Gate 1] 可达性 (Δψ ≤ {args.tol_deg:.0f}° 才算与视频一致)")
    print(f"  {'候选':>10s} {'可达档':>7s} {'可达带':>22s} {'Δψ':>7s} {'用哪个yaw':>10s} {'IK':>8s}  判定")
    survivors = []
    g1 = {}
    for p in priors:
        tag = os.path.basename(p)[:-4]
        r = gate1(p, obj_pos, vy, args.tol_deg,
                  hand=clip_cfg.get("robot_hand", clip_cfg.get("hand", "right")))
        g1[tag] = r
        if r.get("skipped"):
            print(f"  {tag:>10s} {'—':>7s} {'—':>22s} {'—':>7s} {'—':>10s} {'—':>8s}  ⏭ {r['skipped']}")
            continue
        if r["n_reach"] == 0:
            print(f"  {tag:>10s} {'0/72':>7s} {'—':>22s} {'—':>7s} {'—':>10s} "
                  f"{r['best_err']*100:7.2f}cm  ❌ 全域不可达")
            continue
        mark = "✅" if r["ok"] else "❌"
        print(f"  {tag:>10s} {r['n_reach']:>5d}/72 {r['band']:>22s} {r['dpsi']:>6.0f}° "
              f"{r['best_yaw']:>9.0f}° {r['best_err']*100:7.2f}cm  {mark}")
        if r["ok"]:
            survivors.append(p)
    print(f"  -> {len(survivors)}/{len(priors)} 过 Gate 1")

    # ---- Gate 1b: H3 人手一致性 (只对带接近段的任务是硬判据) ----
    g1b = {}
    if survivors and not args.no_h3:
        wp, wq, gs_f = human_pregrasp_wrist(clip_cfg, args.clip, args.table_top_z)
        print(f"\n[Gate 1b] H3 人手一致性 (PreGrasp 帧 {gs_f}, 腕位 "
              f"{np.round(wp, 3)}; R_hi ≤ {TUBE_R_CAP*100:.0f}cm 才过)")
        print(f"  {'候选':>10s} {'缺口G':>9s} {'姿态差':>8s} {'派生管壁R_hi':>13s}  判定")
        keep = []
        for p in survivors:
            tag = os.path.basename(p)[:-4]
            r = gate1b(p, obj_pos, g1[tag]["best_yaw"], wp, wq)
            g1b[tag] = r
            print(f"  {tag:>10s} {r['G']*100:8.1f}cm {r['d_rot_deg']:7.0f}° "
                  f"{r['R_hi']*100:12.1f}cm  {'✅' if r['ok'] else '❌ 抓法与人差太远'}")
            if r["ok"]:
                keep.append(p)
        print(f"  -> {len(keep)}/{len(survivors)} 过 H3")
        survivors = keep

    if args.offline_only or not survivors:
        # 离线模式也要落报告 —— 下游 (选训练候选 / 台账) 读的是它, 不是日志
        json.dump(dict(clip=args.clip, tol_deg=args.tol_deg, video_yaw=vy,
                       obj_pos=obj_pos.tolist(), n_candidates=len(priors),
                       gate1=g1, gate1b=g1b, gate2={}, passed=[]),
                  open(os.path.join(out_dir, "screen_report.json"), "w"),
                  indent=1, ensure_ascii=False, default=float)
        print(f"\n  报告: {os.path.join(out_dir, 'screen_report.json')}")
        if not survivors:
            print("\n【结论】本批全部不过 Gate 1/1b —— 按规则整批丢弃, 重新生成一批。")
        return 0

    # ---- Gate 2 ----
    print(f"\n[Gate 2] prior 质量 (需 pads* ≥ 4 **且** Q* > 0), {len(survivors)} 个候选")
    passed, results = [], {}
    for p in survivors:
        tag = os.path.basename(p)[:-4]
        print(f"  · {tag} ...", end=" ", flush=True)
        r = gate2(args.clip, p, g1[tag]["best_yaw"])
        results[tag] = r
        if "error" in r:
            print(f"❌ {r['error']}")
            continue
        mark = "✅" if r["pass"] else "❌"
        n_ok = r["n_settings_pads_ok"]
        grad = "悬崖(0档)" if n_ok == 0 else f"{n_ok}/6档≥4垫"
        print(f"{mark}  Q*={r['Q_star']:+.3f}  pads*={r['pads_star']:.2f}  "
              f"imb={r['imb']:.3f}  τ={r['tau_n']:.3f}  峰值={r['peak_pad_N']:.1f}N  "
              f"独吞={r['max_pad_share']*100:.0f}%  {grad}")
        if r["pass"]:
            passed.append(tag)

    # ---- 汇总 ----
    rep = dict(clip=args.clip, tol_deg=args.tol_deg, video_yaw=vy,
               obj_pos=obj_pos.tolist(), n_candidates=len(priors),
               gate1={k: v for k, v in g1.items()}, gate1b=g1b,
               gate2=results, passed=passed)
    rp = os.path.join(out_dir, "screen_report.json")
    json.dump(rep, open(rp, "w"), indent=1, ensure_ascii=False, default=float)
    print("\n" + "=" * 78)
    if passed:
        print(f"【结论】{len(passed)}/{len(priors)} 个候选通过全部三关: {', '.join(passed)}")
        print(f"  用法: --grasp_prior {out_dir}/<tag>.npz")
        print(f"  ⚠ 训练前把该候选的 Gate 1 ' 用哪个yaw ' 钉进摆放, 否则加载器会自己搜 IK 最优 yaw")
    else:
        print("【结论】本批全部不通过 —— 按规则整批丢弃, 重新生成一批。")
    print(f"  详细报告: {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
