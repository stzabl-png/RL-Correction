#!/usr/bin/env python
"""用 CoTracker 的表面点轨迹独立估计物体旋转 —— 补上轮廓判据的结构性盲区。

## 为什么需要它

轮廓判据对**近似对称物体的旋转完全失明**: bpp/13 的苹果被估计成转了 98°, 而
explained 0.95 / iou_occ 0.92 / 质心偏移 0.46px 全是绿的(轴长比 1.07, 转多少度轮廓都一样)。
人眼是靠**果柄**(纹理)看出来的 —— CoTracker 跟的正是纹理点, 盲区互补。

而且它是几何方法, 可以当真值用; VLM 不行, 因为 VLM 自己就是被测对象。

## 怎么做

  1. 在参考帧的物体 mask 内撒点(避开手), CoTracker 跟踪全序列
  2. 参考帧: 每个点的视线与"按 3D 位姿摆好的网格"求交 -> 该点在**物体坐标系**的 3D 坐标
  3. 后续帧: 用 (3D 物体点, 2D 跟踪点) 解 PnP -> 一个**完全独立于轮廓**的位姿
  4. 与 world_fused 的 3D 位姿比对转角

## 自举标定(零人工)

  A. 可观测 take 对拍: rot_observability 高的 take(手机 0.281 / 4_cap 0.254),
     其 3D 位姿转角本身可信 -> 拿来验 CoTracker。
  B. 静止段健全性: 无人接触的帧段物体必然静止, CoTracker 必须报"零旋转"。
     ★ 这条对**任何物体**成立(含低纹理), 用来挡住 A 的选择偏差 ——
       光滑苹果既无纹理又旋转对称, 在有纹理的手机上验出的准确率不能外推给它。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh


def load_masks(scene: Path, n: int, kind="objects", oid: str | None = None):
    """oid 给定时只取该物体的 mask。默认(None)保持并集 —— 手部本来就该并,
    物体并集会让 CoTracker 把种子播到别的物体上再拿本物体的位姿去解释(见 object_select.py)。"""
    out = {}
    for i in range(n):
        d = scene / "masks" / kind / "frames" / f"frame_{i:06d}_masks"
        if not d.is_dir():
            continue
        pat = (f"{oid}.png" if oid else "object_*.png") if kind == "objects" else "*_hand_0.png"
        acc = None
        for p in sorted(d.glob(pat)):
            a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if a is not None:
                b = a > 127
                acc = b if acc is None else (acc | b)
        if acc is not None and acc.any():
            out[i] = acc
    return out


def seed_points(mask, hand, n_pts=120, margin=9):
    """在物体 mask 内均匀撒点, 腐蚀掉边缘(边缘点容易跟到背景), 避开手。"""
    m = cv2.erode(mask.astype(np.uint8), np.ones((margin, margin), np.uint8)) > 0
    if hand is not None:
        m &= ~cv2.dilate(hand.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    ys, xs = np.nonzero(m)
    if len(xs) < 10:
        return None
    idx = np.linspace(0, len(xs) - 1, min(n_pts, len(xs))).astype(int)
    return np.stack([xs[idx], ys[idx]], 1).astype(np.float32)


def decimate_mesh(mesh, voxel=0.003):
    """体素聚类简化 —— 光线求交只需近似表面, 34 万面直接跑太慢。"""
    V = np.asarray(mesh.vertices, float); F = np.asarray(mesh.faces, int)
    key = np.floor(V / voxel).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    m = int(inv.max()) + 1
    Vd = np.zeros((m, 3)); np.add.at(Vd, inv, V)
    Vd /= np.bincount(inv, minlength=m).reshape(-1, 1)
    Fd = inv[F]
    ok = (Fd[:, 0] != Fd[:, 1]) & (Fd[:, 1] != Fd[:, 2]) & (Fd[:, 0] != Fd[:, 2])
    return trimesh.Trimesh(vertices=Vd, faces=Fd[ok], process=False)


def backproject_to_mesh(pts2d, K, T_oc, mesh):
    """参考帧: 视线与(按位姿摆好的)网格求交 -> 点在物体坐标系的 3D 坐标。"""
    Kinv = np.linalg.inv(K)
    R, t = T_oc[:3, :3], T_oc[:3, 3]
    mesh = decimate_mesh(mesh)
    Vc = (R @ np.asarray(mesh.vertices).T).T + t          # 网格搬到相机系
    mc = trimesh.Trimesh(vertices=Vc, faces=mesh.faces, process=False)
    inter = trimesh.ray.ray_triangle.RayMeshIntersector(mc)
    dirs = (Kinv @ np.concatenate([pts2d, np.ones((len(pts2d), 1))], 1).T).T
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = np.zeros_like(dirs)
    loc, ray_idx, _ = inter.intersects_location(origins, dirs, multiple_hits=False)
    if len(loc) == 0:
        return None, None
    obj = (R.T @ (loc - t).T).T                           # 相机系 -> 物体系
    return obj.astype(np.float64), ray_idx


def rel_angle(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra @ Rb.T) - 1) / 2, -1, 1))))


def run(scene: Path, video: Path, out: Path, seed_frame=None, n_pts=150,
        n_seed_frames=3, max_reproj=5.0, viz=True, max_frames=None,
        audit=None, device="cuda"):
    z = np.load(scene / "world_fused.npz", allow_pickle=True)
    K = np.asarray(z["K"], float)
    Tc = np.asarray(z["object_ob_in_cam"], float)
    c2w = np.asarray(z["c2w"], float)
    n = len(Tc) if max_frames is None else min(len(Tc), max_frames)
    meshes = (sorted(glob.glob(str(scene / "objects" / "*" / "*.obj")))
              or sorted(glob.glob(str(scene / "*.obj"))))
    mesh = trimesh.load(meshes[0], force="mesh")

    om = load_masks(scene, n, "objects")
    hm = load_masks(scene, n, "hands")
    if not om:
        return {"scene": str(scene), "error": "no object masks"}

    # 参考帧选取: **conf_pos 优先, 遮挡次之**
    # 反投影是靠参考帧的 3D 位姿把 2D 点打到网格上的 —— 参考帧位姿错, 整套种子点全错。
    # v1 用"零遮挡 + mask 最大", 只优化了"看得清", 没管"位姿对不对":
    #   27_scene 因此选中 frame 4, 而 0~41 正是已知的跟踪失败段(倾角 72°)。
    # v2 改成 score = conf_pos × (1 - 遮挡率):
    #   高可信但有中度遮挡的帧, 优于零遮挡但位姿不可信的帧。
    #   27_scene 的无遮挡帧几乎全在坏段, 硬卡"零遮挡"会无解可选。
    conf = {}
    if audit and Path(audit).is_file():
        doc = json.loads(Path(audit).read_text())
        for t in doc.get("takes", []):
            if t.get("take") == str(scene):
                conf = {r["frame"]: r.get("conf_pos", 0) for r in t.get("per_frame", [])}
                break
    occl = {}
    for i in om:
        if i >= n:
            continue
        h = hm.get(i)
        occl[i] = (float((om[i] & h).sum()) / max(1, int(om[i].sum()))) if h is not None else 0.0
    score = {i: (conf.get(i, 50) if conf else 50) * (1.0 - occl[i]) for i in occl}
    ranked = sorted(score, key=lambda i: -score[i])
    if not ranked:
        return {"scene": str(scene), "error": "no candidate frames"}
    # ★ 质量 + 时间分散 二者缺一不可 (v1 只有分散, v2 只有质量, 都不对):
    #   只按分数取 top-N 会让候选池在时间上挤成一团(实测 bpp/11 选出 [1,9,28]),
    #   视角不分散 -> 3D 种子点共面 -> PnP 回到歧义构型, 静止段漂移从 2.2° 涨到 31.5°。
    #   所以先用**质量门**筛出合格池(而非 top-N), 再在池内**按时间均匀取**。
    best = ranked[0]
    bar = max(25.0, 0.6 * score[best])
    pool = sorted([i for i in ranked if score[i] >= bar])
    while len(pool) < 3 and bar > 5.0:               # 池太小(如 27_scene)才逐步放宽
        bar *= 0.7
        pool = sorted([i for i in ranked if score[i] >= bar])
    cand = pool
    if seed_frame is None:
        seed_frame = best
    print(f"[ct] 合格池 {len(pool)} 帧(质量门 {bar:.0f}, 跨帧 {pool[0]}~{pool[-1]})  "
          f"最高分帧 {best} (conf_pos={conf.get(best,'NA')} 遮挡={occl[best]:.2f})")

    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok or len(frames) >= n:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) < n:
        n = len(frames)
    H0, W0 = frames[0].shape[:2]

    # ★ 单个参考帧撒点必然落在朝向相机的那一面 => 3D 点近似共面 => PnP 双解歧义。
    #   实测 bpp/6(手机): 共面度 0.0024, EPNP 直接解崩(重投影 37.9px)。
    #   改成从若干个视角分散的参考帧各撒一批, 合并后才有深度散布。
    picks = [cand[j] for j in np.linspace(0, len(cand) - 1, n_seed_frames).astype(int)]
    seeds = sorted(set(picks + [seed_frame]))
    all_pts, all_obj, all_t = [], [], []
    for sfi in seeds:
        if sfi not in om:
            continue
        p2 = seed_points(om[sfi], hm.get(sfi), max(30, n_pts // len(seeds)))
        if p2 is None:
            continue
        o3, ri = backproject_to_mesh(p2, K, Tc[sfi], mesh)
        if o3 is None:
            continue
        all_pts.append(p2[ri]); all_obj.append(o3); all_t.append(np.full(len(ri), sfi))
    if not all_pts:
        return {"scene": str(scene), "error": "backprojection missed mesh"}
    pts = np.concatenate(all_pts); obj3d = np.concatenate(all_obj)
    tq = np.concatenate(all_t)
    ev = np.linalg.eigvalsh(np.cov((obj3d - obj3d.mean(0)).T))
    planarity = float(np.sqrt(abs(ev[0] / ev[2])))
    print(f"[ct] 参考帧 {seeds}  撒点 {len(pts)} 个  共面度 {planarity:.4f} "
          f"({'共面, PnP 有歧义' if planarity < 0.05 else '有深度散布'})")

    vid = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float().to(device)
    q = np.concatenate([tq[:, None], pts], 1)
    queries = torch.from_numpy(q).float()[None].to(device)
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline",
                           source="github", trust_repo=True).to(device)
    with torch.no_grad():
        tracks, vis = model(vid, queries=queries)
    tr = tracks[0].cpu().numpy()               # (T, N, 2) 已是原分辨率坐标
    vs = vis[0].cpu().numpy() > 0.5

    # 逐帧 PnP -> 独立位姿
    res = []
    R_seed_ct = None
    for i in range(n):
        ok_pts = vs[i]
        if hm.get(i) is not None:                       # 落在手上的点不可信
            uv = np.round(tr[i]).astype(int)
            inb = ((uv[:, 0] >= 0) & (uv[:, 0] < W0) & (uv[:, 1] >= 0) & (uv[:, 1] < H0))
            onhand = np.zeros(len(uv), bool)
            onhand[inb] = hm[i][uv[inb, 1], uv[inb, 0]]
            ok_pts = ok_pts & ~onhand
        rec = {"frame": i, "n_pts": int(ok_pts.sum())}
        if ok_pts.sum() >= 8:
            # 共面构型用 IPPE(专为平面设计); EPNP 在平面上不适用, 实测会给出重投影 37px 的垃圾解
            flag = cv2.SOLVEPNP_IPPE if planarity < 0.05 else cv2.SOLVEPNP_EPNP
            try:
                okp, rvec, tvec, inl = cv2.solvePnPRansac(
                    obj3d[ok_pts], tr[i][ok_pts].astype(np.float64), K, None,
                    flags=flag, reprojectionError=2.5, iterationsCount=400)
            except cv2.error:
                okp, inl = False, None
            if okp and inl is not None and len(inl) >= 6:
                pr, _ = cv2.projectPoints(obj3d[ok_pts][inl[:, 0]], rvec, tvec, K, None)
                err = float(np.linalg.norm(pr[:, 0] - tr[i][ok_pts][inl[:, 0]], axis=1).mean())
                rec["reproj_px"] = round(err, 2)
                if err <= max_reproj:          # 质量门: 解不好就判无解, 不给错位姿
                    Rct = cv2.Rodrigues(rvec)[0]
                    if R_seed_ct is None and i == seeds[0]:
                        R_seed_ct = Rct
                    rec.update(n_inlier=int(len(inl)), R=Rct.tolist())
        res.append(rec)

    if R_seed_ct is None:
        ok_seed = [r for r in res if "R" in r]
        R_seed_ct = np.array(ok_seed[0]["R"]) if ok_seed else None
    R_seed_3d = Tc[seeds[0]][:3, :3]

    rows = []
    for r in res:
        if "R" not in r or R_seed_ct is None:
            rows.append({**{k: r[k] for k in ("frame", "n_pts")},
                         "ct_rot_deg": None, "pose_rot_deg": None, "diff_deg": None})
            continue
        ct = rel_angle(np.array(r["R"]), R_seed_ct)
        p3 = rel_angle(Tc[r["frame"]][:3, :3], R_seed_3d)
        rows.append({"frame": r["frame"], "n_pts": r["n_pts"], "n_inlier": r.get("n_inlier"),
                     "ct_rot_deg": round(ct, 1), "pose_rot_deg": round(p3, 1),
                     "diff_deg": round(abs(ct - p3), 1)})

    # 轨迹可视化 —— 先看跟踪本身对不对, 再谈下游 PnP
    if viz:
        # 自己写, 不用官方 Visualizer —— 它经 imageio 写 mp4, 本机版本报
        # TiffWriter.write() got an unexpected keyword argument 'fps'
        vp = out / f"tracks_{scene.parent.name}_{scene.name}.mp4"
        sc2 = 0.5
        W2, H2 = int(W0 * sc2) // 2 * 2, int(H0 * sc2) // 2 * 2
        vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W2, H2))
        trail = 10
        for i in range(n):
            img = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), (W2, H2))
            for k in range(len(pts)):
                for j in range(max(0, i - trail), i):        # 拖尾
                    if vs[j, k] and vs[j + 1, k]:
                        cv2.line(img, tuple((tr[j, k] * sc2).astype(int)),
                                 tuple((tr[j + 1, k] * sc2).astype(int)), (0, 210, 255), 1,
                                 cv2.LINE_AA)
                c = tuple((tr[i, k] * sc2).astype(int))
                cv2.circle(img, c, 3, (60, 255, 60) if vs[i, k] else (90, 90, 200),
                           -1 if vs[i, k] else 1, cv2.LINE_AA)
            r = rows_by_frame.get(i) if "rows_by_frame" in dir() else None
            cv2.rectangle(img, (6, 6), (330, 58), (20, 20, 20), -1)
            cv2.putText(img, f"frame {i:04d}  visible {int(vs[i].sum())}/{len(pts)}",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(img, "green=visible  red=occluded  trail=track",
                        (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
            vw.write(img)
        vw.release()
        print(f"[ct] 轨迹视频: {vp}")

    # ---- 静止段健全性检验(验证 B) ----
    # 无人接触 => 物体在**世界系**必然静止。相机是动的, 所以必须把位姿转到世界系再比,
    # 否则相机一动就误判成物体转了。这条对低纹理物体同样成立 ——
    # 用来挡住"在有纹理的手机上验、却把结论用到光滑苹果上"的选择偏差。
    def to_world(Rc, i):
        return c2w[i][:3, :3] @ Rc
    free = []
    for i in range(n):
        if i not in om:
            continue
        h = hm.get(i)
        touching = h is not None and (cv2.dilate(om[i].astype(np.uint8),
                                                 np.ones((15, 15), np.uint8)).astype(bool) & h).any()
        if not touching:
            free.append(i)
    spans, cur = [], []
    for i in free:
        if cur and i == cur[-1] + 1:
            cur.append(i)
        else:
            if len(cur) >= 5:
                spans.append(cur)
            cur = [i]
    if len(cur) >= 5:
        spans.append(cur)
    Rmap = {r["frame"]: np.array(r["R"]) for r in res if "R" in r}
    static_ct, static_3d = [], []
    for sp in spans:
        fs = [i for i in sp if i in Rmap]
        if len(fs) < 3:
            continue
        base_ct = to_world(Rmap[fs[0]], fs[0])
        base_3d = c2w[fs[0]][:3, :3] @ Tc[fs[0]][:3, :3]
        for i in fs[1:]:
            static_ct.append(rel_angle(to_world(Rmap[i], i), base_ct))
            static_3d.append(rel_angle(c2w[i][:3, :3] @ Tc[i][:3, :3], base_3d))
    static = {
        "n_free_spans": len(spans), "n_free_frames": int(sum(len(x) for x in spans)),
        "ct_drift_median_deg": (round(float(np.median(static_ct)), 1) if static_ct else None),
        "ct_drift_p90_deg": (round(float(np.percentile(static_ct, 90)), 1) if static_ct else None),
        "pose3d_drift_median_deg": (round(float(np.median(static_3d)), 1) if static_3d else None),
    }

    valid = [r for r in rows if r["diff_deg"] is not None]
    summ = {
        "scene": str(scene), "seed_frames": [int(x) for x in seeds],
        "planarity": round(planarity, 4), "n_seed_pts": int(len(pts)),
        "n_frames": n, "n_solved": len(valid),
        "diff_median_deg": (round(float(np.median([r["diff_deg"] for r in valid])), 1)
                            if valid else None),
        "diff_p90_deg": (round(float(np.percentile([r["diff_deg"] for r in valid], 90)), 1)
                         if valid else None),
        "static_check": static,
    }
    out.mkdir(parents=True, exist_ok=True)
    tag = scene.parent.name + "_" + scene.name
    (out / f"ct_{tag}.json").write_text(
        json.dumps({"summary": summ, "per_frame": rows}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[ct] 解出 {len(valid)}/{n} 帧  转角差 中位 {summ['diff_median_deg']}° "
          f"p90 {summ['diff_p90_deg']}°  | 静止段 {static['n_free_frames']}帧 "
          f"CT漂移 {static['ct_drift_median_deg']}° (3D位姿 {static['pose3d_drift_median_deg']}°)")
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed-frame", type=int, default=None)
    ap.add_argument("--n-pts", type=int, default=150)
    ap.add_argument("--n-seed-frames", type=int, default=3)
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--audit", type=Path, default=None,
                    help="pose_audit.json —— 用逐帧 conf_pos 选可信参考帧")
    a = ap.parse_args(argv)
    r = run(a.scene, a.video, a.out, a.seed_frame, a.n_pts,
            n_seed_frames=a.n_seed_frames, viz=not a.no_viz,
            max_frames=a.max_frames, audit=a.audit)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
