#!/usr/bin/env python3
"""一条 ARCTIC take: 真实误差 + 现有 confidence + 候选新判据信号 -> 一行记录。

设计要点(每条都是被实测逼出来的, 不是偏好):

* 误差用**零拟合的质心距**: 两边网格顶点各按自己的位姿变换后取质心, 那是同一个物理点。
  不用 Umeyama / 13参数联合拟合 —— 后者在 laptop 上落进局部极小, 报出"尺度0.224、旋转142°",
  与相机轨迹独立测得的 1.1995 直接矛盾。
* 误差**分深度/横向**: laptop 实测深度 236mm、横向 38mm、投影质心仅差 33px。
  只报总误差会掩盖"图像里看不见"这一核心事实。
* 旋转必须**对称感知**({I,Rx180,Ry180,Rz180} 上取最小): 不处理时 conf_rot 相关 +0.302
  ("无预测力"), 处理后 -0.543 且单调。投影贴合 ≠ 朝向正确。
* 候选信号只用**我们自己的产物**(真值仅用于事后验证), 否则判据无法上线。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ARCTIC = Path("/media/lyh/DATA2/arctic/repo/data/arctic_data/data")
SYM = [np.eye(3)] + [Rotation.from_rotvec(np.pi * np.eye(3)[i]).as_matrix() for i in range(3)]
TIPS = [4, 8, 12, 16, 20]          # MediaPipe 21 关节的五个指尖


def gt_streams(subject: str, seq: str):
    o = np.load(ARCTIC / "raw_seqs" / subject / f"{seq}.object.npy", allow_pickle=True)
    ego = np.load(ARCTIC / "raw_seqs" / subject / f"{seq}.egocam.dist.npy",
                  allow_pickle=True).item()
    mano = np.load(ARCTIC / "raw_seqs" / subject / f"{seq}.mano.npy", allow_pickle=True).item()
    T = len(o)
    w2e = np.tile(np.eye(4), (T, 1, 1))
    w2e[:, :3, :3] = np.asarray(ego["R_k_cam_np"], float)
    w2e[:, :3, 3] = np.asarray(ego["T_k_cam_np"], float).reshape(T, 3)
    P = np.tile(np.eye(4), (T, 1, 1))
    P[:, :3, :3] = Rotation.from_rotvec(o[:, 1:4]).as_matrix()
    P[:, :3, 3] = o[:, 4:7] / 1000.0                      # object.npy 平移是毫米
    return P, w2e, mano, np.degrees(o[:, 0])


def sample(V, n, seed=0):
    if len(V) <= n:
        return V
    return V[np.random.default_rng(seed).choice(len(V), n, replace=False)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--take", type=Path, required=True)
    ap.add_argument("--meta", type=Path, required=True)
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="逐帧 npz 输出目录")
    a = ap.parse_args()

    meta = json.loads(a.meta.read_text())
    subject, seq = meta["subject"], meta["seq"]
    v2a = {int(r["video_frame"]): int(r["arctic_vidx"]) for r in meta["index"]}
    z = np.load(a.take / "world_fused.npz", allow_pickle=True)
    C = np.asarray(z["object_ob_in_cam"], float)
    W = np.asarray(z["object_ob_in_world"], float)
    c2w = np.asarray(z["c2w"], float)
    n = len(C)

    Vo_full = np.asarray(trimesh.load(a.take / "objects/object_0/object_mesh_scaled_final.obj",
                                      force="mesh").vertices, float)
    obj_name = seq.split("_")[0]
    Vg_full = np.asarray(trimesh.load(ARCTIC / "meta/object_vtemplates" / obj_name / "mesh.obj",
                                      force="mesh").vertices, float) / 1000.0
    Vo, Vg = sample(Vo_full, 6000), sample(Vg_full, 6000)

    P_gt, w2e, mano, art = gt_streams(subject, seq)
    f = np.array([t for t in range(n) if t in v2a and v2a[t] < len(P_gt)])
    g = np.array([v2a[t] for t in f])
    G = np.einsum("tij,tjk->tik", w2e[g], P_gt[g])              # 真值物体在 ego 相机系

    # ---- 真实误差(零拟合) ----
    co = np.einsum("tij,j->ti", C[f, :3, :3], Vo_full.mean(0)) + C[f, :3, 3]
    cg = np.einsum("tij,j->ti", G[:, :3, :3], Vg_full.mean(0)) + G[:, :3, 3]
    d = co - cg
    err_pos = np.linalg.norm(d, axis=1) * 1000
    err_dep = np.abs(d[:, 2]) * 1000
    err_lat = np.linalg.norm(d[:, :2], axis=1) * 1000
    depth_ratio = co[:, 2] / cg[:, 2]

    Dl = np.einsum("tij,tjk->tik", np.linalg.inv(C[f, :3, :3]), G[:, :3, :3])
    Rm = Rotation.from_matrix(Dl).mean().as_matrix()
    cand = np.stack([np.degrees(np.linalg.norm(
        Rotation.from_matrix(np.einsum("ij,tjk,kl->til", Rm.T, Dl, Sm)).as_rotvec(), axis=1))
        for Sm in SYM])
    err_rot = cand.min(0)

    # ---- 现有 confidence ----
    doc = json.loads(a.audit.read_text())
    # ⚠ 必须带上数据集名（尾部**三**级）：arctic 与 arctic15 的 <subject>/<seq> 完全同名，
    # 只取两级会静默匹配到另一个数据集的记录 —— 实测 15fps 的 laptop 取到了 30fps 的
    # 84/77，而它真实是 77/71。这类串台不会报错，只会让整批分析用错分数。
    tail = "/".join(a.take.resolve().parts[-3:])
    rec = next((t for t in doc["takes"] if str(t.get("take", "")).endswith(tail)
                and t.get("object", "object_0") == "object_0"), None)
    cp = np.full(n, np.nan)
    cr = np.full(n, np.nan)
    take_lvl = {}
    if rec and "per_frame" in rec:
        for r in rec["per_frame"]:
            if r["frame"] < n:
                cp[r["frame"]] = r.get("conf_pos", np.nan)
                cr[r["frame"]] = r.get("conf_rot", np.nan)
        take_lvl = {k: rec.get(k) for k in
                    ("conf_pos_median", "conf_rot_median", "rot_observability", "rot_factor",
                     "mesh_scale_fitted", "scale_reliable", "median_explained", "median_occl",
                     "n_scored", "mode_counts")}

    # ---- 候选信号(只用我们自己的产物) ----
    sig = {}
    rp_p = a.take / "replay_world.npz"
    ca_p = a.take / "contact_auto.json"
    if rp_p.is_file():
        rp = np.load(rp_p, allow_pickle=True)
        # ① 手的米制尺寸: 人手是每条第一人称视频里天然自带的尺子
        palms, tots = [], []
        for side in ("left", "right"):
            if f"joints_{side}" not in rp.files:
                continue
            J = np.asarray(rp[f"joints_{side}"], float)
            ok = ~np.isnan(J).any(axis=(1, 2))
            if ok.sum() < 10:
                continue
            palms.append(np.median(np.linalg.norm(J[ok, 0] - J[ok, 9], axis=1)) * 100)
            tots.append(np.median(np.linalg.norm(J[ok, 0] - J[ok, 12], axis=1)) * 100)
        if palms:
            sig["hand_palm_cm"] = float(np.mean(palms))
            sig["hand_total_cm"] = float(np.mean(tots))
        # ② 接触时的手-物 3D 间隙: 2D 判接触(与深度无关), 3D 该贴合。差多少 = 深度不自洽
        if ca_p.is_file():
            ca = json.loads(ca_p.read_text())
            per_side = {}
            for side in ("left", "right"):
                if f"joints_{side}" not in rp.files:
                    continue
                J = np.asarray(rp[f"joints_{side}"], float)
                m = min(len(J), n)
                ok = ~np.isnan(J[:m]).any(axis=(1, 2))
                inc = np.zeros(m, bool)
                for s, e in ca.get("annotations", {}).get(side, []):
                    inc[max(0, s):min(m, e + 1)] = True
                idx = np.where(inc & ok)[0]
                if len(idx) < 10:
                    continue
                gaps = []
                for t in idx[:: max(1, len(idx) // 120)]:      # 抽样, 逐帧建树太慢
                    Xw = (W[t, :3, :3] @ Vo.T).T + W[t, :3, 3]
                    gaps.append(cKDTree(Xw).query(J[t][TIPS], k=1)[0].min() * 1000)
                per_side[side] = float(np.median(gaps))
            if per_side:
                sig["contact_gap_mm"] = float(np.median(list(per_side.values())))
                sig["contact_gap_min_mm"] = float(min(per_side.values()))
        # ③ 手 vs 物体的深度比: 同一次重建内部两个来源, 不自洽即可疑
        w2c = np.linalg.inv(c2w)
        hd = []
        for side in ("left", "right"):
            if f"joints_{side}" not in rp.files:
                continue
            J = np.asarray(rp[f"joints_{side}"], float)
            m = min(len(J), n)
            ok = ~np.isnan(J[:m]).any(axis=(1, 2))
            if ok.sum() < 10:
                continue
            Jc = np.einsum("tij,tkj->tki", w2c[:m][ok, :3, :3], J[:m][ok]) + w2c[:m][ok, :3, 3][:, None]
            hd.append(np.median(Jc[..., 2]))
        if hd:
            sig["hand_depth_m"] = float(np.mean(hd))
            sig["obj_depth_m"] = float(np.median(C[f, 2, 3]))
            sig["obj_over_hand_depth"] = sig["obj_depth_m"] / max(sig["hand_depth_m"], 1e-6)

    # 形状误差(PCA 对齐+按最长轴归一, 与尺度朝向无关)。加这一项是因为两条 take 都显示
    # 误差**各向异性**: 长宽 <3%, 最短轴 +100% 以上 —— 任何单标量尺度都修不了,
    # 而只报总误差/深度比会把这个事实盖住。
    def _axis_ratio(V):
        V = V - V.mean(0)
        _, _, Vt = np.linalg.svd(V, full_matrices=False)
        e = np.sort((V @ Vt.T).ptp(0))[::-1]
        return e / e[0]
    shape = {}
    try:
        ro, rg = _axis_ratio(Vo_full), _axis_ratio(Vg_full)
        shape = {"axis_ratio_ours": [round(float(v), 4) for v in ro],
                 "axis_ratio_gt": [round(float(v), 4) for v in rg],
                 "shape_err_mid_pct": round(float((ro[1] / rg[1] - 1) * 100), 1),
                 "shape_err_short_pct": round(float((ro[2] / rg[2] - 1) * 100), 1)}
    except Exception as e:  # noqa: BLE001 - 形状指标缺失不该挡住其余评测
        shape = {"shape_error": f"{type(e).__name__}: {e}"}

    # final take 目录不含 sam2_object（那在 interim 里），两处都找
    lp = a.take / "sam2_object" / "label_prompt.json"
    if not lp.is_file():
        cand_lp = (a.take.parents[2] / "interim" / a.take.parents[1].name
                   / f"{a.take.parent.name}__{a.take.name}" / "sam2_object" / "label_prompt.json")
        if cand_lp.is_file():
            lp = cand_lp
    label_frame = None
    if lp.is_file():
        try:
            label_frame = int(json.loads(lp.read_text())["objects"][0]["frame_idx"])
        except Exception:
            pass

    row = {
        "take": str(a.take), "subject": subject, "seq": seq, "object": obj_name,
        "label_frame": label_frame, **shape,
        "n_frames": int(n), "n_cmp": int(len(f)), "articulation_deg": float(art.ptp()),
        "err_pos_med": float(np.median(err_pos)), "err_pos_p90": float(np.percentile(err_pos, 90)),
        "err_dep_med": float(np.median(err_dep)), "err_lat_med": float(np.median(err_lat)),
        "err_rot_med": float(np.median(err_rot)), "err_rot_p90": float(np.percentile(err_rot, 90)),
        "depth_ratio_med": float(np.median(depth_ratio)),
        "conf_frames": int((~np.isnan(cp[f])).sum()),
        **{f"take_{k}": v for k, v in take_lvl.items()},
        **sig,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / f"{subject}__{seq}.npz", frame=f, arctic_vidx=g,
             err_pos=err_pos, err_dep=err_dep, err_lat=err_lat, err_rot=err_rot,
             depth_ratio=depth_ratio, conf_pos=cp[f], conf_rot=cr[f])
    (a.out / f"{subject}__{seq}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1))
    print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
