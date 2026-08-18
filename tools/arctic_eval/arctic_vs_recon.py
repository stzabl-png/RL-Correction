#!/usr/bin/env python3
"""Our reconstruction vs ARCTIC ground truth -> is confidence predictive of real error?

THE ALIGNMENT PROBLEM.  Two unknown constant transforms sit between the two pose streams:

    T_gt(t)  =  X @ T_ours(t) @ M

  X : our gravity-z-up world -> ARCTIC's world.  Includes SCALE, because our object scale
      is estimated (sam3d_scale) while ARCTIC's is metrically true.
  M : our SAM3D mesh's canonical frame -> ARCTIC's template frame.  Our pose refers to the
      origin of OUR mesh, ARCTIC's to the origin of THEIR template -- different points on
      the same physical object.

Fitting only X (the usual Umeyama trajectory alignment) is WRONG here: M's translation
lives in the object frame, so it rotates with the object and cannot be absorbed by any
world-frame offset.  Ignoring it would charge us for a constant mounting offset as if it
were tracking error, and the charge would grow with how much the object turns.  So both
are solved together, 13 parameters, one nonlinear least-squares.

WHAT IS AND IS NOT MEASURED.  A global fit deliberately absorbs constant bias: after this,
a residual means "the trajectory SHAPE is wrong", not "the pose is offset".  The absorbed
scale is reported separately -- it is itself a result (how far off our size estimate is).
Absolute orientation is not comparable at all (our mesh has its own canonical axes), so
rotation error is measured on the CHANGE in orientation, which is frame-convention free.

NO CONFIDENCE IS USED IN THE FIT.  Frames are never selected or weighted by conf, or the
"does conf predict error" question would be answered with its own answer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ARCTIC = Path("/media/lyh/DATA2/arctic/repo/data/arctic_data/data")


def gt_object_world(subject: str, seq: str):
    """ARCTIC object pose in ITS world frame, plus world->ego-camera, per annotation frame."""
    o = np.load(ARCTIC / "raw_seqs" / subject / f"{seq}.object.npy", allow_pickle=True)
    ego = np.load(ARCTIC / "raw_seqs" / subject / f"{seq}.egocam.dist.npy",
                  allow_pickle=True).item()
    T = len(o)
    P = np.tile(np.eye(4), (T, 1, 1))
    P[:, :3, :3] = Rotation.from_rotvec(o[:, 1:4]).as_matrix()
    P[:, :3, 3] = o[:, 4:7] / 1000.0                 # object.npy translation is MILLIMETRES
    w2e = np.tile(np.eye(4), (T, 1, 1))
    w2e[:, :3, :3] = np.asarray(ego["R_k_cam_np"], float)
    w2e[:, :3, 3] = np.asarray(ego["T_k_cam_np"], float).reshape(T, 3)
    return P, w2e, np.degrees(o[:, 0])               # + articulation angle, for the record


def pack(p):
    s = np.exp(p[0])                                  # log-scale keeps s > 0
    RX = Rotation.from_rotvec(p[1:4]).as_matrix()
    tX = p[4:7]
    RM = Rotation.from_rotvec(p[7:10]).as_matrix()
    tM = p[10:13]
    return s, RX, tX, RM, tM


def predict(p, R_o, t_o):
    s, RX, tX, RM, tM = pack(p)
    pos = s * (R_o @ tM + t_o) @ RX.T + tX
    rot = np.einsum("ij,tjk,kl->til", RX, R_o, RM)
    return pos, rot


def world_sim3_from_camera(c2w, gt_c2w):
    """X (our world -> ARCTIC world) from the CAMERA tracks, not the object.

    X is a property of the two world frames; it does not depend on which object we look
    at.  Solving it jointly with the mesh transform M made a 13-parameter non-convex
    problem that landed in a local minimum -- it reported scale 0.224 while the camera
    tracks say 1.1995, and 142 deg of rotation error, which cannot both be true of one
    world.  The camera pair is a pure point-to-point correspondence, so Umeyama solves it
    in closed form with no local minima (measured residual: 12 mm median).
    """
    P, Q = c2w[:, :3, 3], gt_c2w[:, :3, 3]
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    U, S, Vt = np.linalg.svd(X.T @ Y)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = float((S * np.diag(D)).sum() / (X ** 2).sum())
    t = mq - s * R @ mp
    res = np.linalg.norm((s * (R @ P.T).T + t) - Q, axis=1) * 1000
    return s, R, t, res


def fit_mesh_transform(R_o, t_o, R_g, t_g, s, RX, tX, w_rot=0.3):
    """Only M is unknown once X is pinned by the camera: 6 parameters, well conditioned."""
    def resid(p):
        RM = Rotation.from_rotvec(p[0:3]).as_matrix()
        tM = p[3:6]
        pos = s * (R_o @ tM + t_o) @ RX.T + tX
        rot = np.einsum("ij,tjk,kl->til", RX, R_o, RM)
        dR = np.einsum("tij,tjk->tik", rot.transpose(0, 2, 1), R_g)
        return np.concatenate([(pos - t_g).ravel(),
                               w_rot * Rotation.from_matrix(dR).as_rotvec().ravel()])

    best = None
    for seed in range(8):
        p0 = np.zeros(6)
        if seed:
            rng = np.random.default_rng(seed)
            p0[0:3] = rng.normal(0, 1.8, 3)
        r = least_squares(resid, p0, loss="huber", f_scale=0.02, max_nfev=600)
        if best is None or r.cost < best.cost:
            best = r
    return np.concatenate([[np.log(s)], Rotation.from_matrix(RX).as_rotvec(), tX, best.x])


def fit(R_o, t_o, R_g, t_g, w_rot=0.3):
    def resid(p):
        pos, rot = predict(p, R_o, t_o)
        e_pos = (pos - t_g).ravel()
        dR = np.einsum("tij,tjk->tik", rot.transpose(0, 2, 1), R_g)
        e_rot = Rotation.from_matrix(dR).as_rotvec().ravel()
        return np.concatenate([e_pos, w_rot * e_rot])

    best = None
    # the rotation part is non-convex; seed R_X from a few starts so we do not report a
    # local minimum as "our reconstruction is bad"
    for seed in range(6):
        p0 = np.zeros(13)
        if seed:
            rng = np.random.default_rng(seed)
            p0[1:4] = rng.normal(0, 1.6, 3)
            p0[7:10] = rng.normal(0, 1.6, 3)
        p0[4:7] = t_g.mean(0) - t_o.mean(0)
        r = least_squares(resid, p0, loss="huber", f_scale=0.02, max_nfev=400)
        if best is None or r.cost < best.cost:
            best = r
    return best.x


def errors(p, R_o, t_o, R_g, t_g):
    pos, rot = predict(p, R_o, t_o)
    e_pos = np.linalg.norm(pos - t_g, axis=1) * 1000.0            # mm
    # orientation CHANGE relative to frame 0: cancels the mesh-frame convention entirely
    dR_o = np.einsum("tij,jk->tik", rot, rot[0].T)
    dR_g = np.einsum("tij,jk->tik", R_g, R_g[0].T)
    d = np.einsum("tij,tjk->tik", dR_o.transpose(0, 2, 1), dR_g)
    e_rot = np.degrees(np.linalg.norm(Rotation.from_matrix(d).as_rotvec(), axis=1))
    return e_pos, e_rot


def step_rot_error(p, R_o, t_o, R_g):
    """**逐帧**旋转误差: 相邻帧之间的朝向增量, 我们 vs 真值。

    为什么需要它: `errors()` 量的是"相对第 0 帧的累计偏差" —— 一旦序列中途翻了面,
    之后每一帧的误差都巨大, 与那些帧各自的分数无关。**那个量天生是 take 级的**,
    拿它去验"逐帧判据有没有分辨力"会系统性地判它无效(2026-08-17 实测: take 内
    逐帧相关中位 -0.051, 方向对的只占 55%, 约等于抛硬币)。

    增量误差没有这个问题: 翻面只污染发生的那一帧, 其余帧照常可比。
    第 0 帧无前驱, 置 NaN。
    """
    _, rot = predict(p, R_o, t_o)
    do = np.einsum("tij,tkj->tik", rot[1:], rot[:-1])        # 我们的帧间增量
    dg = np.einsum("tij,tkj->tik", R_g[1:], R_g[:-1])        # 真值的帧间增量
    d = np.einsum("tij,tjk->tik", do.transpose(0, 2, 1), dg)
    e = np.degrees(np.linalg.norm(Rotation.from_matrix(d).as_rotvec(), axis=1))
    return np.concatenate([[np.nan], e])


def errors_sym(p, R_o, t_o, R_g, group):
    """对称感知的旋转误差 + **用了哪个对称元**。

    为什么需要: 位姿在图像上无法区分 (R,t) 与 (R·S, t+R·S_t)。裸测地角会把一次
    180° 翻面记成 180° 误差, 于是"扁平物体"整条 take 的误差堆在 150~170°, 判据看起来
    毫无预测力。(2026-08-11 单条实测: 中位 102°→34.9°, 相关 +0.302→-0.543。)

    ★ 群元作用在**物体frame**, 不能直接乘在残差上。我们的朝向变化
      dR_o = rot[t]·rot[0]ᵀ; 逐帧各自可以选对称元 s_t, 于是
          admissible dR_o = rot[t] · (s_t·s_0ᵀ) · rot[0]ᵀ = rot[t] · u · rot[0]ᵀ,  u ∈ S
      误差取这一族的最小角。

    ★ 同时返回每帧选中的元索引(0 = 恒等)。GraspPose 2026-08-17 指出的关键点:
      只报"最小误差"会把**真实的翻面**静默吸收成"误差小"。一条 take 若持续选用同一个
      非恒等元, 要么它真对称、要么位姿真翻了, 这两种情况处理方式完全不同。

    ⚠ 群从 ARCTIC **真值网格**测(独立于被评估的重建), 但 11 个物体的翻转残差连续分布在
      0.7%~3.1%, **没有干净阈值**能判"对称/不对称"。所以这里不做二元判定, 把三条主轴翻转
      全部当候选 —— 因此 sym 误差是**乐观下界**, 必须与裸误差并列看。
    """
    _, rot = predict(p, R_o, t_o)
    dR_g = np.einsum("tij,jk->tik", R_g, R_g[0].T)
    best = None
    pick = None
    for k, S in enumerate([np.eye(3)] + list(group)):
        dR = np.einsum("tij,jk,lk->til", rot, S, rot[0])      # rot[t]·S·rot[0]ᵀ
        d = np.einsum("tij,tjk->tik", dR.transpose(0, 2, 1), dR_g)
        e = np.degrees(np.linalg.norm(Rotation.from_matrix(d).as_rotvec(), axis=1))
        if best is None:
            best, pick = e, np.zeros(len(e), int)
        else:
            m = e < best
            best = np.where(m, e, best)
            pick = np.where(m, k, pick)
    return best, pick


def bin_report(conf, err, name, unit, edges=(0, 20, 40, 60, 80, 100.001)):
    print(f"\n  {name} 分箱 vs 真实误差({unit})")
    print(f"    {'区间':>10}{'帧数':>7}{'中位':>9}{'p90':>9}{'最大':>9}")
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if not m.any():
            print(f"    {f'{lo:.0f}-{hi:.0f}':>10}{0:>7}{'-':>9}{'-':>9}{'-':>9}")
            continue
        e = err[m]
        med, p90 = float(np.median(e)), float(np.percentile(e, 90))
        rows.append((lo, med))
        print(f"    {f'{lo:.0f}-{hi:.0f}':>10}{int(m.sum()):>7}{med:9.1f}{p90:9.1f}{e.max():9.1f}")
    if len(rows) >= 2:
        meds = [m for _, m in rows]
        mono = all(a >= b for a, b in zip(meds, meds[1:]))
        r = np.corrcoef(conf, err)[0, 1]
        print(f"    单调(分数升->误差降): {'是' if mono else '否 <-- 判据在这条数据上无预测力'}"
              f"   相关系数 {r:+.3f}")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", type=Path, required=True, help="ReconstructOutput take dir")
    ap.add_argument("--meta", type=Path, required=True, help="视频构造时落的 *.meta.json")
    ap.add_argument("--audit", type=Path, required=True, help="poseqa/pose_audit.json")
    ap.add_argument("--object", default="object_0")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--symmetry", type=Path,
                    default=Path(__file__).with_name("arctic_symmetry.json"),
                    help="逐物体对称群(由 measure_symmetry.py 从 ARCTIC 真值网格测出)")
    a = ap.parse_args()

    meta = json.loads(a.meta.read_text())
    v2a = {int(r["video_frame"]): int(r["arctic_vidx"]) for r in meta["index"]}
    z = np.load(a.take / "world_fused.npz", allow_pickle=True)
    ids = [str(x) for x in np.asarray(z["object_ids"]).tolist()] if "object_ids" in z.files \
        else ["object_0"]
    oi = ids.index(a.object)
    W = (np.asarray(z["object_ob_in_world_all"], float)[oi]
         if "object_ob_in_world_all" in z.files
         else np.asarray(z["object_ob_in_world"], float))
    C = (np.asarray(z["object_ob_in_cam_all"], float)[oi]
         if "object_ob_in_cam_all" in z.files
         else np.asarray(z["object_ob_in_cam"], float))

    P_gt, w2e, art = gt_object_world(meta["subject"], meta["seq"])
    f_ours = np.array([t for t in range(len(W)) if t in v2a])
    f_gt = np.array([v2a[t] for t in f_ours])
    ok = f_gt < len(P_gt)
    f_ours, f_gt = f_ours[ok], f_gt[ok]
    print(f"[data] take {a.take.name} / {a.object}: 重建 {len(W)} 帧, 可比对 {len(f_ours)} 帧; "
          f"铰接幅度 {art.ptp():.1f}° (grab 序列, 近似刚体)")

    R_o, t_o = W[f_ours, :3, :3], W[f_ours, :3, 3]
    R_g, t_g = P_gt[f_gt, :3, :3], P_gt[f_gt, :3, 3]
    # X from the camera tracks (closed form, no local minima), then only M is fitted
    c2w = np.asarray(z["c2w"], float)[f_ours]
    gt_c2w = np.linalg.inv(w2e)[f_gt]
    s, RX, tX, cam_res = world_sim3_from_camera(c2w, gt_c2w)
    print(f"[X]     世界系变换取自相机轨迹: s={s:.4f}, 相机残差 中位 {np.median(cam_res):.1f}mm "
          f"p90 {np.percentile(cam_res,90):.1f}mm  ({len(c2w)} 帧, 闭式解)")
    p = fit_mesh_transform(R_o, t_o, R_g, t_g, s, RX, tX)
    e_pos, e_rot = errors(p, R_o, t_o, R_g, t_g)
    e_rot_step = step_rot_error(p, R_o, t_o, R_g)
    # 对称感知误差: 群从 ARCTIC 真值网格测(见 tools/arctic_eval/arctic_symmetry.json)
    sym = json.loads(a.symmetry.read_text()) if a.symmetry and a.symmetry.is_file() else {}
    obj_key = next((k for k in sym if k in meta["seq"]), None)
    group = [np.asarray(c["R"], float) for c in sym.get(obj_key, {}).get("candidates", [])]
    if group:
        e_rot_sym, sym_pick = errors_sym(p, R_o, t_o, R_g, group)
        used = np.bincount(sym_pick, minlength=len(group) + 1)
        print(f"[sym]   对称感知旋转误差 中位 {np.median(e_rot_sym):.2f}° "
              f"(裸 {np.median(e_rot):.2f}°)  物体={obj_key}  "
              f"选中元分布 恒等{used[0]}/" + "/".join(f"flip{i+1}:{used[i+1]}" for i in range(len(group))))
    else:
        e_rot_sym, sym_pick = e_rot.copy(), np.zeros(len(e_rot), int)
        print(f"[sym]   X 没有 {meta['seq']} 的对称群, 退回裸误差")
    print(f"\n[world] 拟合尺度 s={s:.3f}  (>1 = 我们估小了, <1 = 估大了; 这是尺度误差本身)")
    print(f"[world] 位置误差 中位 {np.median(e_pos):.1f}mm  p90 {np.percentile(e_pos,90):.1f}mm  "
          f"最大 {e_pos.max():.1f}mm")
    print(f"[world] 旋转变化误差 中位 {np.median(e_rot):.2f}°  p90 {np.percentile(e_rot,90):.2f}°  "
          f"最大 {e_rot.max():.2f}°")

    # camera frame: same fit, but on ob_in_cam vs world2ego @ gt. Only a diagnostic --
    # if world error >> camera error, the residual is the camera track, not the object pose.
    G_cam = np.einsum("tij,tjk->tik", w2e[f_gt], P_gt[f_gt])
    p_c = fit(C[f_ours, :3, :3], C[f_ours, :3, 3], G_cam[:, :3, :3], G_cam[:, :3, 3])
    e_pos_c, e_rot_c = errors(p_c, C[f_ours, :3, :3], C[f_ours, :3, 3],
                              G_cam[:, :3, :3], G_cam[:, :3, 3])
    print(f"[cam]   位置误差 中位 {np.median(e_pos_c):.1f}mm  p90 {np.percentile(e_pos_c,90):.1f}mm"
          f"   (仅诊断: 世界系远大于相机系 => 残差来自相机轨迹而非物体位姿)")

    doc = json.loads(a.audit.read_text())
    # audit 里存的是**产出机器**上的绝对路径(远程跑、本地分析), 按尾部两级匹配
    tail = "/".join(a.take.resolve().parts[-2:])
    rec = next((t for t in doc["takes"]
                if str(t.get("take", "")).endswith(tail)
                and t.get("object", "object_0") == a.object), None)
    if rec is None or "per_frame" not in rec:
        print("\n[conf] X pose_audit.json 里没有这条 take/物体的逐帧记录")
        return 1
    cp = np.full(len(W), np.nan)
    cr = np.full(len(W), np.nan)
    for r in rec["per_frame"]:
        if r["frame"] < len(W):
            cp[r["frame"]] = r.get("conf_pos", np.nan)
            cr[r["frame"]] = r.get("conf_rot", np.nan)
    m = ~np.isnan(cp[f_ours])
    print(f"\n[conf] 有分数且可比对的帧: {int(m.sum())}/{len(f_ours)}")
    bin_report(cp[f_ours][m], e_pos[m], "conf_pos", "mm")
    bin_report(cr[f_ours][m], e_rot[m], "conf_rot", "度")
    bin_report(cr[f_ours][m], e_rot_sym[m], "conf_rot(对称感知)", "度")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out, video_frame=f_ours, arctic_vidx=f_gt,
                 err_pos_mm=e_pos, err_rot_deg=e_rot,
                 err_rot_deg_sym=e_rot_sym, sym_pick=sym_pick,
                 err_rot_deg_step=e_rot_step,
                 err_pos_mm_cam=e_pos_c, err_rot_deg_cam=e_rot_c,
                 conf_pos=cp[f_ours], conf_rot=cr[f_ours],
                 fitted_scale=s, articulation_deg=art[f_gt])
        print(f"\n[out] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
