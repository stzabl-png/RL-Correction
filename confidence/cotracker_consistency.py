#!/usr/bin/env python
"""CoTracker 2D 一致性 —— 第二观察员参与评分标准 1-4, 不解绝对位姿。

## 设计(2026-08-09 定稿)

前一版(cotracker_rotation)走 反投影→逐帧 PnP→绝对位姿, 实测对参考帧选取过度敏感
(三版策略互有胜负), 且共面构型有双解歧义。本版**把 PnP 整个甩掉**: 只做 2D 比较 ——

  预测: 参考帧反投影得到的 3D 物体点, 按 FP 位姿投影到当前帧  (FP 说点该在哪)
  观测: CoTracker 实际跟到的位置                              (画面里点真在哪)

逐帧输出四个信号, 对应评分标准 1-4:

  ct_err_norm    #1 逐点重投影误差中位 / 物体图像尺寸。
                    ★比 mask 质心强: 点阵随 FP 旋转而旋转, 观测点不转 → 能抓住
                    "苹果转 116° 但轮廓不变"这种质心/轮廓全瞎的失效。
  inside_frac    #2 可见观测点落在 FP 投影轮廓内的比例。不依赖 mask(绕开杂块污染)。
  vis_frac       #3 可见点比例 = 逐帧证据量。比"投影∩手mask"多抓: 自遮挡/模糊/出画。
  flow_agree     #4 帧间运动场一致性: FP 预测位移 vs 观测位移。
                    观测在动+预测不动 → FP 跟丢; 预测跳变+观测没跳 → FP 瞬移。

依赖与不依赖:
  仍依赖  参考帧位姿可信(反投影用, 从 audit 的 conf_pos 选) + 网格
  不依赖  逐帧 PnP / RANSAC / 平面性 —— 参考帧敏感性问题随 PnP 一起消失
  注意    网格尺度错会让 3D 点沿视线偏移, 对相对一致性是二阶影响(docstring 备案)

可视化: 绿=CoTracker 观测点, 品红=FP 预测点, 红线=误差向量。
苹果案例里能直接看到品红点阵在旋转、绿点不动。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cotracker_rotation import (backproject_to_mesh, load_masks,  # noqa: E402
                                seed_points)
import trimesh  # noqa: E402

import object_select as objsel  # noqa: E402


def project(K, T, X):
    """物体系 3D 点 -> 像素。X:(N,3) -> (N,2), 以及深度是否为正。"""
    Xc = (T[:3, :3] @ X.T).T + T[:3, 3]
    z = Xc[:, 2:3]
    uv = (K @ Xc.T).T
    return uv[:, :2] / np.where(np.abs(z) < 1e-9, 1e-9, z), (Xc[:, 2] > 1e-6)


def silhouette_mask(V, F, K, T, hw, ss=2):
    H, W = hw
    Kd = K.copy(); Kd[:2] /= ss
    uv, _ = project(Kd, T, V)
    tri = uv[F]
    front = tri[np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) < 0]
    img = np.zeros((H // ss, W // ss), np.uint8)
    if len(front):
        cv2.fillPoly(img, [t.astype(np.int32) for t in front], 255)
    return img > 0


def pick_seed_frames(scene, om, hm, n, audit, k=3):
    """质量门(conf_pos) + 时间均匀 —— 与 rotation 版同思路, 但这里只影响反投影质量。"""
    conf = {}
    if audit and Path(audit).is_file():
        doc = json.loads(Path(audit).read_text())
        for t in doc.get("takes", []):
            if t.get("take") == str(scene):
                conf = {r["frame"]: r.get("conf_pos", 0) for r in t.get("per_frame", [])}
    occl = {}
    for i in om:
        if i >= n:
            continue
        h = hm.get(i)
        occl[i] = (float((om[i] & h).sum()) / max(1, int(om[i].sum()))) if h is not None else 0.0
    score = {i: (conf.get(i, 50) if conf else 50) * (1.0 - occl[i]) for i in occl}
    ranked = sorted(score, key=lambda i: -score[i])
    bar = max(25.0, 0.6 * score[ranked[0]])
    pool = sorted([i for i in ranked if score[i] >= bar])
    while len(pool) < 3 and bar > 5.0:
        bar *= 0.7
        pool = sorted([i for i in ranked if score[i] >= bar])
    return [pool[j] for j in np.linspace(0, len(pool) - 1, min(k, len(pool))).astype(int)]


def run(scene: Path, video: Path, out: Path, audit=None, n_pts=150, obj_idx: int = 0,
        max_frames=115, viz=True, device="cuda"):
    z = np.load(scene / "world_fused.npz", allow_pickle=True)
    K = np.asarray(z["K"], float)
    oid = objsel.object_ids(z)[obj_idx]
    # 单物体 take 保持旧文件名, 存量 cc_*.json 不失效
    sfx = f'_{oid}' if objsel.count(z) > 1 else ''
    Tc, _ = objsel.poses(z, obj_idx)
    n = min(len(Tc), max_frames)
    mp = objsel.mesh_path(scene, oid)
    if mp is None:
        raise SystemExit(f"{scene}: no mesh for {oid}")
    mesh = trimesh.load(mp, force="mesh")
    from cotracker_rotation import decimate_mesh
    md = decimate_mesh(mesh)
    Vm, Fm = np.asarray(md.vertices, float), np.asarray(md.faces, int)

    om = load_masks(scene, n, "objects", oid)
    hm = load_masks(scene, n, "hands")
    if not om:
        return {"scene": str(scene), "error": "no object masks"}

    seeds = pick_seed_frames(scene, om, hm, n, audit)
    all_pts, all_obj, all_t = [], [], []
    for sf in seeds:
        p2 = seed_points(om[sf], hm.get(sf), max(40, n_pts // len(seeds)))
        if p2 is None:
            continue
        o3, ri = backproject_to_mesh(p2, K, Tc[sf], mesh)
        if o3 is None:
            continue
        all_pts.append(p2[ri]); all_obj.append(o3); all_t.append(np.full(len(ri), sf))
    if not all_pts:
        return {"scene": str(scene), "error": "backprojection failed"}
    pts = np.concatenate(all_pts); X = np.concatenate(all_obj)
    tq = np.concatenate(all_t)
    print(f"[cc] 参考帧 {seeds}  种子点 {len(pts)}")

    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok or len(frames) >= n:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    n = min(n, len(frames))
    H0, W0 = frames[0].shape[:2]

    vid = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float().to(device)
    q = np.concatenate([tq[:, None], pts], 1)
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline",
                           source="github", trust_repo=True).to(device)
    with torch.no_grad():
        tracks, vis = model(vid, queries=torch.from_numpy(q).float()[None].to(device))
    tr = tracks[0].cpu().numpy()
    vs = vis[0].cpu().numpy() > 0.5
    del vid
    torch.cuda.empty_cache()

    per = []
    prev_pred = prev_tr = prev_ok = None
    for i in range(n):
        pred, zok = project(K, Tc[i], X)
        okp = vs[i] & zok
        # 落在手上的观测点剔除(手 mask 比 vis 标志多一重保险)
        h = hm.get(i)
        if h is not None:
            uv = np.round(tr[i]).astype(int)
            inb = (uv[:, 0] >= 0) & (uv[:, 0] < W0) & (uv[:, 1] >= 0) & (uv[:, 1] < H0)
            onh = np.zeros(len(uv), bool)
            onh[inb] = h[uv[inb, 1], uv[inb, 0]]
            okp &= ~onh
        rec = {"frame": i, "n_seed": int(len(pts)), "n_vis": int(okp.sum()),
               "vis_frac": round(float(okp.mean()), 3)}
        # 物体图像尺寸: 用 FP 投影 bbox 对角线(mask 可能缺/带杂块)
        sil = silhouette_mask(Vm, Fm, K, Tc[i], (H0, W0), ss=2)
        ys, xs = np.nonzero(sil)
        diag = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()) * 2) if len(xs) else None
        if okp.sum() >= 6 and diag:
            err = np.linalg.norm(pred[okp] - tr[i][okp], axis=1)
            rec["ct_err_norm"] = round(float(np.median(err)) / diag, 4)          # 标准1(池化, 保留)
            # ★ 分组统计 —— 三个参考帧的种子不混池。
            #   苹果案例教训: 帧40/81 的反投影用了 FP 幽灵旋转后的位姿, 那两组点先天带错;
            #   混池后误差被抹成"全程偏高", 定位不到坏帧。分组后各组曲线互相打架,
            #   打架本身(spread)就是"轨迹内部不自洽"的检测量。
            # ★ 误差再拆两个分量:
            #   平移分量 = 误差向量的均值模长   (整体挪 = 位置错)
            #   旋转分量 = 去掉均值后的残差     (点阵形变/旋转 = 旋转错)
            #   苹果位置本来是对的 —— 这样拆才能把"位置分高、旋转分零"归因到位。
            gts, grs = [], []
            for sf in seeds:
                gok = okp & (tq == sf)
                if gok.sum() < 6:
                    continue
                d = pred[gok] - tr[i][gok]
                tvec = d.mean(0)
                gts.append(float(np.linalg.norm(tvec)) / diag)
                grs.append(float(np.median(np.linalg.norm(d - tvec, axis=1))) / diag)
            if gts:
                rec["ct_t_err"] = round(float(np.median(gts)), 4)
                rec["ct_r_err"] = round(float(np.median(grs)), 4)
                if len(grs) >= 2:
                    rec["ct_r_spread"] = round(float(max(grs) - min(grs)), 4)
            uv2 = np.round(tr[i][okp] / 2).astype(int)
            inb2 = ((uv2[:, 0] >= 0) & (uv2[:, 0] < sil.shape[1])
                    & (uv2[:, 1] >= 0) & (uv2[:, 1] < sil.shape[0]))
            ins = np.zeros(len(uv2), bool)
            ins[inb2] = sil[uv2[inb2, 1], uv2[inb2, 0]]
            rec["inside_frac"] = round(float(ins.mean()), 3)                     # 标准2
            if prev_pred is not None:
                both = okp & prev_ok
                if both.sum() >= 6:
                    dp = pred[both] - prev_pred[both]
                    dt = tr[i][both] - prev_tr[both]
                    rec["flow_pred_px"] = round(float(np.median(np.linalg.norm(dp, axis=1))), 2)
                    rec["flow_obs_px"] = round(float(np.median(np.linalg.norm(dt, axis=1))), 2)
                    rec["flow_err_px"] = round(float(np.median(
                        np.linalg.norm(dp - dt, axis=1))), 2)                    # 标准4
        per.append(rec)
        prev_pred, prev_tr, prev_ok = pred, tr[i], okp

    # ---- 可视化: 绿=观测 品红=FP预测 红线=误差 ----
    if viz:
        out.mkdir(parents=True, exist_ok=True)
        vp = out / f"cc_{scene.parent.name}_{scene.name}{sfx}.mp4"
        sc = 0.5
        W2, H2 = int(W0 * sc) // 2 * 2, int(H0 * sc) // 2 * 2
        vw = cv2.VideoWriter(str(vp), cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (W2, H2))
        for i in range(n):
            img = cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR), (W2, H2))
            pred, zok = project(K, Tc[i], X)
            for k in range(len(pts)):
                if not (vs[i, k] and zok[k]):
                    continue
                a = tuple((tr[i, k] * sc).astype(int))
                b = tuple((pred[k] * sc).astype(int))
                cv2.line(img, a, b, (60, 60, 230), 1, cv2.LINE_AA)
                cv2.circle(img, a, 2, (60, 230, 60), -1, cv2.LINE_AA)
                cv2.circle(img, b, 2, (230, 60, 230), -1, cv2.LINE_AA)
            r = per[i]
            txt = (f"f{i:03d}  vis {r['vis_frac']:.2f}  "
                   f"err {r.get('ct_err_norm', float('nan')):.3f}  "
                   f"inside {r.get('inside_frac', float('nan')):.2f}  "
                   f"flow_err {r.get('flow_err_px', float('nan'))}")
            cv2.rectangle(img, (6, 6), (560, 30), (20, 20, 20), -1)
            cv2.putText(img, txt, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (235, 235, 235), 1, cv2.LINE_AA)
            cv2.putText(img, "green=tracked  magenta=FP-predicted  red=error",
                        (12, H2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (180, 180, 180), 1, cv2.LINE_AA)
            vw.write(img)
        vw.release()
        print(f"[cc] 视频: {vp}")

    val = [r for r in per if "ct_err_norm" in r]
    summ = {
        "scene": str(scene), "seed_frames": [int(s) for s in seeds],
        "n_frames": n, "n_scored": len(val),
        "ct_err_norm_median": (round(float(np.median([r["ct_err_norm"] for r in val])), 4)
                               if val else None),
        "ct_err_norm_p90": (round(float(np.percentile([r["ct_err_norm"] for r in val], 90)), 4)
                            if val else None),
        "ct_t_err_median": (round(float(np.median([r["ct_t_err"] for r in val if "ct_t_err" in r])), 4)
                            if any("ct_t_err" in r for r in val) else None),
        "ct_r_err_median": (round(float(np.median([r["ct_r_err"] for r in val if "ct_r_err" in r])), 4)
                            if any("ct_r_err" in r for r in val) else None),
        "vis_frac_median": round(float(np.median([r["vis_frac"] for r in per])), 3),
        "inside_frac_median": (round(float(np.median([r["inside_frac"] for r in val])), 3)
                               if val else None),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"cc_{scene.parent.name}_{scene.name}{sfx}.json").write_text(
        json.dumps({"summary": summ, "per_frame": per}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[cc] 打分 {len(val)}/{n} 帧  err_norm 中位 {summ['ct_err_norm_median']} "
          f"p90 {summ['ct_err_norm_p90']}  vis 中位 {summ['vis_frac_median']}")
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audit", type=Path, default=None)
    ap.add_argument("--object", default="0", help="物体 id 或序号")
    ap.add_argument("--max-frames", type=int, default=115)
    ap.add_argument("--no-viz", action="store_true")
    a = ap.parse_args(argv)
    import numpy as _np
    _z = _np.load(a.scene / 'world_fused.npz', allow_pickle=True)
    _i = objsel.resolve(a.scene, _z, a.object)[0]
    r = run(a.scene, a.video, a.out, audit=a.audit, max_frames=a.max_frames,
            viz=not a.no_viz, obj_idx=_i)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
