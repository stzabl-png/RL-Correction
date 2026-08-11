#!/usr/bin/env python
"""RTS 平滑层 —— 用逐帧可信度驱动测量噪声, 融合出平滑轨迹 + 逐帧不确定度。

## 定位(在 Noisy World 框架里)

这是可信度的**消费者和精加工器**, 不是可信度的替代品:
  我们的判据层  →  逐帧测量噪声 R (conf 低 → R 大; conf<5 → 该帧测量剔除)
  本层          →  融合轨迹 + 逐帧协方差(σ_pos/σ_rot) + innovation
协方差本身就是有原则的可信度, 正好填 object_confidence 通道;
"两端锚定、中间放开"(P4) 在这里成为数学事实: 遮挡段测量被降权,
滤波器靠运动模型滑过去, 不确定度在中段自动鼓起、在两端收紧。

## 能与不能(写死, 防止误用)

  能   抹无偏噪声/抖动; 低可信段按运动模型+两端信息插值; innovation 当在线失效信号
  不能 纠系统性偏差 —— 一段测量若"一直错得很稳"(幽灵旋转平台/62°漂移),
       滤波器会心安理得跟着错。所以喂它的 conf 必须先把这些段压低(已由判据层负责)。

## 实现

  位置  线性 KF (状态 [p,v], 常速模型) + 精确 RTS 后向平滑
  旋转  SO(3) 乘法误差状态滤波(MEKF, 状态 [δθ,δω]) 前向+后向, 切空间协方差加权融合
        (不能拿普通 Kalman 直接滤四元数)
  门控  NIS > χ²(3,99.9%)=16.27 的测量当离群拒收, 记 innovation 尖峰

## 验证协议(跑完自动打印)

  A 静止段: 平滑后逐帧步长应显著小于原始(有真值: 无接触=必静止)
  B 图像一致性: 平滑轨迹重投影的 explained/d_cent 不应差于原始
    —— 尤其检查"好段没被平滑弄坏"(平滑器的经典副作用)
  C innovation 尖峰应对准已知瞬移帧
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

import object_select as objsel
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pose_audit import decimate, load_obj_mask, silhouette  # noqa: E402
import trimesh  # noqa: E402

CHI2_GATE = 16.27          # χ²(3dof, 99.9%)
RECOVER_AFTER = 5          # ★发散恢复: 连续 N 帧被门拒收 => 错的是滤波器不是测量
                           #   (2_scene 门控锁死案例: 坏帧初始化后正确测量被永久拒收,
                           #    绿轮廓横躺在直立瓶子上还自报 σ_rot 2.4°)
VEL_DECAY = 0.8            # 无测量/被拒帧的速度衰减: 桌面物体不会自己持续转/滑
                           #   (bpp/2 案例: 常速外推让物体在证据真空里以 8.6°/帧脑补自转)


# ★ v2 R 映射(v1 验收不及格后的修正):
#   v1 把 conf<5 的测量整段剔除, 混淆了三种"低分" —— 违反了框架原则 P3:
#     ① 不可观测(瓶子末段 conf_rot=0): 测量其实是好的 → v1 剔除后 MEKF 73帧自由漂,
#        好段 explained 0.883→0.489, 触发"高可信段不能变差"的下线标准
#     ② 保守封顶(bpp/11 低分帧 explained 0.995): 测量是好的只是没证据 → v1 丢掉换外推,
#        快速搬运段跟不上, dc 0.085→0.339
#     ③ 被反驳(CT 出面): 唯一该剔除的 —— 需要 audit 提供 refuted 标志, 待接
#   v2: 永不剔除, 只放大 σ; "没证据"="弱测量", 不是"无测量"。
def sigma_pos(conf):       # conf 100 -> 5mm, conf 0 -> 10cm
    return 0.005 + 0.095 * (1.0 - conf / 100.0) ** 2


def sigma_rot(conf):       # conf 100 -> 2°, conf 0 -> 12°
    # v3: 30°→12°。"没证据"意味着没理由偏离测量, σ 只需容纳高频抖动;
    # 30° 弱到让前后向融合把 27 末段本来正确的直立姿态往中段倾斜姿态拽
    # (explained 0.883→0.701)。v3 之后参数冻结 —— 已在同四条上调了三轮,
    # 再调即过拟合; 后续验证必须换未调参的 take。
    return np.radians(2.0 + 10.0 * (1.0 - conf / 100.0) ** 2)


# ------------------------------------------------------------- 位置: KF + RTS
def kf_rts_position(t_meas, conf, dt, sigma_a=2.0):
    n = len(t_meas)
    F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
    q1, q2, q3 = dt**4 / 4, dt**3 / 2, dt**2
    Q = sigma_a**2 * np.block([[q1 * np.eye(3), q2 * np.eye(3)],
                               [q2 * np.eye(3), q3 * np.eye(3)]])
    xs, Ps, xps, Pps = [], [], [], []
    first = 0
    x = np.zeros(6); x[:3] = t_meas[first]
    P = np.diag([0.05**2] * 3 + [0.5**2] * 3)
    innov = np.zeros(n); used = np.zeros(n, bool)
    consec = 0
    for i in range(n):
        if i > 0:
            x = F @ x
            P = F @ P @ F.T + Q
        xps.append(x.copy()); Pps.append(P.copy())
        R = np.eye(3) * sigma_pos(conf[i]) ** 2
        y = t_meas[i] - x[:3]
        S = P[:3, :3] + R
        nis = float(y @ np.linalg.solve(S, y))
        innov[i] = nis
        if nis > CHI2_GATE and consec >= RECOVER_AFTER:
            P[:3, :3] += np.eye(3) * (10 * sigma_pos(conf[i])) ** 2   # 膨胀后强制收敛回测量
            S = P[:3, :3] + R
            nis = CHI2_GATE
        if nis <= CHI2_GATE:
            K = P[:, :3] @ np.linalg.inv(S)
            x = x + K @ y
            P = P - K @ P[:3, :]
            used[i] = True
            consec = 0
        else:
            consec += 1
            x[3:] *= VEL_DECAY
        xs.append(x.copy()); Ps.append(P.copy())
    # RTS 后向
    xs_s = [None] * n; Ps_s = [None] * n
    xs_s[-1], Ps_s[-1] = xs[-1], Ps[-1]
    for i in range(n - 2, -1, -1):
        G = Ps[i] @ F.T @ np.linalg.inv(Pps[i + 1])
        xs_s[i] = xs[i] + G @ (xs_s[i + 1] - xps[i + 1])
        Ps_s[i] = Ps[i] + G @ (Ps_s[i + 1] - Pps[i + 1]) @ G.T
    p = np.array([x[:3] for x in xs_s])
    sig = np.array([np.sqrt(np.trace(P[:3, :3]) / 3) for P in Ps_s])
    p_filt = np.array([x[:3] for x in xs])                    # 前向=因果 filter, 零预判
    sig_filt = np.array([np.sqrt(np.trace(P[:3, :3]) / 3) for P in Ps])
    return p, sig, innov, used, p_filt, sig_filt


# --------------------------------------------------- 旋转: MEKF 前向(可反向跑)
def mekf_pass(R_meas, conf, refuted, dt, sigma_alpha=8.0):
    n = len(R_meas)
    # 初始化必须避开被反驳的帧 —— 27 号第 0 帧即 refuted, 用它初始化=带病起步
    first = next((i for i in range(n) if not refuted[i]), 0)
    q = Rot.from_matrix(R_meas[first])
    w = np.zeros(3)
    # 初始姿态先验放宽到 45°: 第 0 帧可能落在坏段(2_scene), 10° 的自信先验会把门锁死
    P = np.diag([np.radians(45)**2] * 3 + [np.radians(60)**2] * 3)
    F = np.eye(6); F[:3, 3:] = np.eye(3) * dt
    qa = np.radians(sigma_alpha)
    Q = np.block([[np.eye(3) * (qa * dt**2 / 2)**2, np.zeros((3, 3))],
                  [np.zeros((3, 3)), np.eye(3) * (qa * dt)**2]])
    qs, Ps, innov = [], [], np.zeros(n)
    consec = 0
    for i in range(n):
        if i > 0:
            q = q * Rot.from_rotvec(w * dt)
            P = F @ P @ F.T + Q
        did = False
        if not refuted[i]:      # 被CT反驳的测量剔除(唯一剔除理由); "没证据"只放大σ
            y = (q.inv() * Rot.from_matrix(R_meas[i])).as_rotvec()
            Rm = np.eye(3) * sigma_rot(conf[i]) ** 2
            S = P[:3, :3] + Rm
            nis = float(y @ np.linalg.solve(S, y))
            innov[i] = nis
            if nis > CHI2_GATE and consec >= RECOVER_AFTER:
                P[:3, :3] += np.eye(3) * np.radians(45) ** 2
                S = P[:3, :3] + Rm
                nis = CHI2_GATE
            if nis <= CHI2_GATE:
                K = P[:, :3] @ np.linalg.inv(S)
                dx = K @ y
                q = q * Rot.from_rotvec(dx[:3])
                w = w + dx[3:]
                P = P - K @ P[:3, :]
                consec = 0
                did = True
            else:
                consec += 1
        if not did:
            w = w * VEL_DECAY   # 证据真空里不脑补持续旋转
        qs.append(q); Ps.append(P[:3, :3].copy())
    return qs, Ps, innov


def rot_smooth(R_meas, conf, refuted, dt):
    """前向 + 后向 MEKF, 切空间协方差加权融合。"""
    qf, Pf, innov = mekf_pass(R_meas, conf, refuted, dt)
    qb_r, Pb_r, _ = mekf_pass(R_meas[::-1], conf[::-1], refuted[::-1], dt)
    qb, Pb = qb_r[::-1], Pb_r[::-1]
    n = len(R_meas)
    out_R, out_sig = [], []
    for i in range(n):
        d = (qf[i].inv() * qb[i]).as_rotvec()
        W = Pf[i] @ np.linalg.inv(Pf[i] + Pb[i])
        q = qf[i] * Rot.from_rotvec(W @ d)
        Pi = np.linalg.inv(np.linalg.inv(Pf[i]) + np.linalg.inv(Pb[i]))
        out_R.append(q.as_matrix())
        out_sig.append(np.degrees(np.sqrt(np.trace(Pi) / 3)))
    R_filt = np.array([q.as_matrix() for q in qf])            # 前向=因果, 零预判
    sig_filt = np.array([np.degrees(np.sqrt(np.trace(P) / 3)) for P in Pf])
    return np.array(out_R), np.array(out_sig), innov, R_filt, sig_filt


# -------------------------------------------------------------------- 主流程
def run(scene: Path, audit_json: Path, out: Path, viz_video: Path | None = None,
        obj_idx: int = 0):
    z = np.load(scene / "world_fused.npz", allow_pickle=True)
    oid = objsel.object_ids(z)[obj_idx]
    _Tc0, Tw = objsel.poses(z, obj_idx)
    c2w = np.asarray(z["c2w"], float)
    K = np.asarray(z["K"], float)
    n = len(Tw)
    dt = 1.0 / 30.0

    doc = json.loads(audit_json.read_text())
    # (take, object): 老记录没有 "object" 字段, 按 object_0 匹配 —— 那正是它们评的东西
    rec = next(t for t in doc["takes"] if t.get("take") == str(scene)
               and t.get("object", objsel.DEFAULT_OBJECT_ID) == oid)
    cp = np.zeros(n); cr = np.zeros(n); refuted = np.zeros(n, bool)
    for r in rec["per_frame"]:
        cp[r["frame"]] = r.get("conf_pos", 0)
        cr[r["frame"]] = r.get("conf_rot", 0)
        refuted[r["frame"]] = r.get("rot_refuted", False)

    p_s, sig_p, innov_p, used, p_f, sig_pf = kf_rts_position(Tw[:, :3, 3], cp, dt)
    R_s, sig_r, innov_r, R_f, sig_rf = rot_smooth(Tw[:, :3, :3], cr, refuted, dt)

    Tw_s = np.tile(np.eye(4), (n, 1, 1))
    Tw_s[:, :3, :3] = R_s
    Tw_s[:, :3, 3] = p_s
    Tc_s = np.array([np.linalg.inv(c2w[i]) @ Tw_s[i] for i in range(n)])
    Tc_r, _ = objsel.poses(z, obj_idx)

    # ---- 验证 A: 静止段步长(无接触 = 必静止, 有真值) ----
    ct = scene / "contact_auto.json"
    touch = np.zeros(n, bool)
    if ct.is_file():
        for hh in json.loads(ct.read_text())["annotations"].values():
            for a, b in hh:
                touch[max(0, a):min(n, b + 1)] = True
    free = ~touch
    def steps(T, sel):
        idx = [i for i in range(1, n) if sel[i] and sel[i - 1]]
        dp = [np.linalg.norm(T[i][:3, 3] - T[i - 1][:3, 3]) * 1000 for i in idx]
        dr = [np.degrees(np.arccos(np.clip((np.trace(T[i][:3, :3] @ T[i - 1][:3, :3].T) - 1) / 2, -1, 1)))
              for i in idx]
        return (np.median(dp) if dp else None), (np.median(dr) if dr else None)
    raw_p, raw_r = steps(Tw, free)
    smo_p, smo_r = steps(Tw_s, free)

    # ---- 验证 B: 图像一致性(平滑不能把好段弄坏) ----
    meshes = (sorted(glob.glob(str(scene / "objects" / "*" / "*.obj")))
              or sorted(glob.glob(str(scene / "*.obj"))))
    mesh = trimesh.load(meshes[0], force="mesh")
    V0, F0 = decimate(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, int))
    s = rec["mesh_scale_fitted"]
    C = V0.mean(0); V = (V0 - C) * s + C
    def img_stats(Tc, frames):
        ex, dc = [], []
        for i in frames:
            mk = load_obj_mask(scene, i)
            if mk is None or not mk.any():
                continue
            pj = silhouette(V, F0, K, Tc[i], mk.shape, 2)
            mk2 = cv2.resize(mk.astype(np.uint8), (pj.shape[1], pj.shape[0]),
                             interpolation=cv2.INTER_NEAREST) > 0
            am = mk2.sum()
            ex.append(float((mk2 & pj).sum() / am))
            ys, xs = np.nonzero(mk2); yp, xp = np.nonzero(pj)
            if len(xp):
                dc.append(float(np.hypot(xs.mean() - xp.mean(), ys.mean() - yp.mean())
                                / max(np.sqrt(am), 1)))
        return (np.median(ex) if ex else None), (np.median(dc) if dc else None)
    hi = [i for i in range(0, n, 2) if cp[i] >= 60]
    lo = [i for i in range(0, n, 2) if 5 <= cp[i] < 60]
    res_B = {}
    for tag, fr in (("high_conf", hi), ("low_conf", lo)):
        e_r, d_r = img_stats(Tc_r, fr)
        e_s, d_s = img_stats(Tc_s, fr)
        res_B[tag] = dict(n=len(fr), explained_raw=e_r, explained_smooth=e_s,
                          dcent_raw=d_r, dcent_smooth=d_s)

    out.mkdir(parents=True, exist_ok=True)
    # 单物体 take 保持旧文件名, 存量 rts_*.npz 不失效; 多物体才加物体后缀
    tag = f"{scene.parent.name}_{scene.name}"
    if objsel.count(z) > 1:
        tag = f"{tag}_{oid}"
    Tw_f = np.tile(np.eye(4), (n, 1, 1))
    Tw_f[:, :3, :3] = R_f; Tw_f[:, :3, 3] = p_f
    # σ 报告封底: 滤波器内部 σ 会在偏差/剔除段撒谎(bpp/2: 错着转还自报低σ),
    # 对外报告不许低于"该 conf 单帧测量噪声的一半"
    sig_p_rep = np.maximum(sig_p, np.array([sigma_pos(c) for c in cp]) / 2)
    sig_r_rep = np.maximum(sig_r, np.degrees([sigma_rot(c) for c in cr]) / 2)
    np.savez(out / f"rts_{tag}.npz",
             object_ob_in_world_smooth=Tw_s, object_ob_in_cam_smooth=Tc_s,
             object_ob_in_world_filtered=Tw_f,
             sigma_pos_m=sig_p, sigma_rot_deg=sig_r,
             sigma_pos_filtered_m=sig_pf, sigma_rot_filtered_deg=sig_rf,
             sigma_pos_reported_m=sig_p_rep, sigma_rot_reported_deg=sig_r_rep,
             innov_pos_nis=innov_p, innov_rot_nis=innov_r,
             conf_pos=cp, conf_rot=cr, measurement_used=used)
    summ = {
        "scene": str(scene), "n": n,
        "A_static_step": {"raw_mm": raw_p, "smooth_mm": smo_p,
                          "raw_deg": raw_r, "smooth_deg": smo_r,
                          "n_free": int(free.sum())},
        "B_image": res_B,
        "C_innov_spikes_pos": [int(i) for i in np.where(innov_p > CHI2_GATE)[0]][:20],
        "n_rot_refuted": int(refuted.sum()),
        "sigma_pos_mm": {"min": round(float(sig_p.min() * 1000), 1),
                         "median": round(float(np.median(sig_p) * 1000), 1),
                         "max": round(float(sig_p.max() * 1000), 1)},
        "sigma_rot_deg": {"min": round(float(sig_r.min()), 1),
                          "median": round(float(np.median(sig_r)), 1),
                          "max": round(float(sig_r.max()), 1)},
    }
    (out / f"rts_{tag}.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1,
                                                    default=float), encoding="utf-8")

    # ---- 可视化: 灰=原始 彩=平滑(按σ_pos上色) + σ 带 ----
    if viz_video is not None:
        cap = cv2.VideoCapture(str(viz_video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5) // 2 * 2
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5) // 2 * 2
        SH = 90
        strip = np.full((SH, W, 3), 24, np.uint8)
        x0, x1, y0, y1 = 70, W - 10, 8, SH - 22
        smax = max(float(sig_p.max() * 1000), 1.0)
        cv2.rectangle(strip, (x0, y0), (x1, y1), (44, 44, 44), -1)
        pts = [(int(x0 + i / max(1, n - 1) * (x1 - x0)),
                int(y1 - min(sig_p[i] * 1000, smax) / smax * (y1 - y0))) for i in range(n)]
        cv2.polylines(strip, [np.array(pts, np.int32)], False, (120, 255, 120), 1, cv2.LINE_AA)
        cv2.putText(strip, f"sigma_pos (max {smax:.0f}mm)", (x0, SH - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 255, 120), 1, cv2.LINE_AA)
        vp = out / f"rts_{tag}.mp4"
        vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H + SH))
        for i in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            small = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            for T, col, th in ((Tc_r[i], (140, 140, 140), 1),
                               (Tc_s[i], (90, 230, 90), 2)):
                sil = silhouette(V, F0, K, T, (1080, 1920), 2)
                s2 = cv2.resize(sil.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
                cs, _ = cv2.findContours(s2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(small, cs, -1, col, th, cv2.LINE_AA)
            txt = (f"f{i:03d}  sigma_pos {sig_p[i]*1000:5.1f}mm  sigma_rot {sig_r[i]:5.1f}deg"
                   f"  conf(pos {cp[i]:.0f}/rot {cr[i]:.0f})"
                   f"{'  INNOV!' if innov_p[i] > CHI2_GATE else ''}")
            cv2.rectangle(small, (6, 6), (640, 30), (20, 20, 20), -1)
            cv2.putText(small, txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(small, "gray=raw FP   green=RTS smoothed", (12, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            sp = strip.copy()
            px = int(x0 + i / max(1, n - 1) * (x1 - x0))
            cv2.line(sp, (px, y0), (px, y1), (255, 255, 255), 1)
            vw.write(np.vstack([small, sp]))
        cap.release(); vw.release()
        summ["video"] = str(vp)
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--video", type=Path, default=None)
    ap.add_argument("--object", default="0", help="物体 id 或序号")
    a = ap.parse_args(argv)
    _z = np.load(a.scene / 'world_fused.npz', allow_pickle=True)
    s = run(a.scene, a.audit, a.out, a.video, objsel.resolve(a.scene, _z, a.object)[0])
    print(json.dumps(s, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
