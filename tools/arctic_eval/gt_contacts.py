#!/usr/bin/env python
"""ARCTIC GT 接触提取 —— 从官方 MANO+物体真值算逐帧逐指接触, 作为 contact 链路的真值基准。

原理: MANO 前向得手网格(778 顶点, 按蒙皮权重分到 5 指+手掌) → 把手顶点变换进物体
**各部件的规范系**(bottom: 逆全局位姿; top: 再逆铰接角) → 对静态规范部件的表面采样点
建一次 KDTree → 830 帧×双手全批查询逐指最小距离。

反作弊纪律(Phase A 约定):
  * 接触阈值 τ 不挑数: 同时输出 {3,5,8,10}mm 全套敏感性, 下游分析必须看曲线;
  * 本脚本只产真值, 不进任何提取方法;
  * 坐标约定用"投影叠加到视频帧"视觉验收(--viz), 不靠想当然。

输出(写到 --out/<subject>__<seq>/):
  gt_contact.npz   dists_m (T,2,6): [left,right]×[thumb,index,middle,ring,pinky,palm] 最小距离
                   contact_<τ>mm (T,2,6) bool; n_fingers_<τ>mm (T,2)
  gt_summary.json  每手: 接触时间线段、稳定抓握段、接触方式(指数中位/指集合/掌参与率/
                   深度代理=接触顶点占比)
  viz_*.jpg        (--viz) 真值投影叠加帧, 人工目检坐标约定

用法(hawor env):
  python gt_contacts.py --take s05__laptop_grab_01 [--viz] \
      [--video-dir <arctic15目录: mp4+meta.json, 用于叠加>]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARCTIC = Path("/media/lyh/DATA2/arctic/repo/data/arctic_data/data")
HAWOR = Path("/home/lyh/Project/Reconstruct_and_Retarget/third_party/hawor/_DATA")
MANO_PKL = {"right": HAWOR / "data/mano/MANO_RIGHT.pkl",
            "left": HAWOR / "data_left/mano_left/MANO_LEFT.pkl"}
FINGERS = ("thumb", "index", "middle", "ring", "pinky", "palm")
# MANO 关节序: 0=腕, 1-3 index, 4-6 middle, 7-9 pinky, 10-12 ring, 13-15 thumb
JOINT2FINGER = {0: "palm", 1: "index", 2: "index", 3: "index", 4: "middle", 5: "middle",
                6: "middle", 7: "pinky", 8: "pinky", 9: "pinky", 10: "ring", 11: "ring",
                12: "ring", 13: "thumb", 14: "thumb", 15: "thumb"}
TAUS_MM = (3, 5, 8, 10)


def mano_layer(side: str):
    import smplx
    return smplx.create(str(MANO_PKL[side]), model_type="mano", is_rhand=(side == "right"),
                        use_pca=False, flat_hand_mean=False, batch_size=1)


def mano_verts(params: dict, side: str) -> np.ndarray:
    """(T,778,3) 世界系(米)。ARCTIC 口径: use_pca=False, flat_hand_mean=False。"""
    import torch
    lay = mano_layer(side)
    T = len(params["rot"])
    lay.batch_size = T
    with torch.no_grad():
        out = lay(global_orient=torch.from_numpy(np.asarray(params["rot"], np.float32)),
                  hand_pose=torch.from_numpy(np.asarray(params["pose"], np.float32)),
                  betas=torch.from_numpy(np.tile(np.asarray(params["shape"], np.float32), (T, 1))),
                  transl=torch.from_numpy(np.asarray(params["trans"], np.float32)))
    return out.vertices.numpy()


def vert_fingers(side: str) -> np.ndarray:
    """(778,) 每顶点属于哪根指(FINGERS 索引), 由蒙皮权重 argmax 导出。"""
    import torch
    lay = mano_layer(side)
    w = lay.lbs_weights.detach().numpy()          # (778,16)
    jid = w.argmax(1)
    return np.array([FINGERS.index(JOINT2FINGER[int(j)]) for j in jid])


def axangle_mat(v: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_rotvec(v).as_matrix()


def load_object(name: str):
    """规范系两部件的表面采样点 KDTree(米)。top 的铰接在查询时处理。"""
    import trimesh
    from scipy.spatial import cKDTree
    d = ARCTIC / "meta" / "object_vtemplates" / name
    parts = {}
    for p in ("top", "bottom"):
        m = trimesh.load(d / f"{p}.obj", force="mesh", process=False)
        pts, _ = trimesh.sample.sample_surface(m, 60000)
        parts[p] = {"tree": cKDTree(np.asarray(pts, np.float64) / 1000.0),   # 模板单位=毫米
                    "verts": np.asarray(m.vertices, np.float64) / 1000.0}
    return parts


def per_finger_dists(hand_w: np.ndarray, fseg: np.ndarray, obj: dict,
                     opose: np.ndarray) -> np.ndarray:
    """(T,6) 每指最小距离(米)。opose (T,7)=[angle, rotvec3, trans_mm3]。"""
    T = len(hand_w)
    out = np.full((T, len(FINGERS)), np.inf)
    ang, rv, tr = opose[:, 0], opose[:, 1:4], opose[:, 4:7] / 1000.0
    Rw = axangle_mat(rv)                                  # (T,3,3) 物体全局旋转
    ca, sa = np.cos(-ang), np.sin(-ang)                   # top 逆铰接角(绕规范 z 轴)
    for t in range(T):
        h_can = (hand_w[t] - tr[t]) @ Rw[t]               # 世界 → 物体规范系(bottom)
        d_bottom, _ = obj["bottom"]["tree"].query(h_can)
        Rz = np.array([[ca[t], -sa[t], 0], [sa[t], ca[t], 0], [0, 0, 1]])
        d_top, _ = obj["top"]["tree"].query(h_can @ Rz.T)  # 再逆铰接 → top 规范系
        d = np.minimum(d_bottom, d_top)
        for fi in range(len(FINGERS)):
            sel = fseg == fi
            if sel.any():
                out[t, fi] = d[sel].min()
    return out


def contact_depth_frac(hand_w, fseg, obj, opose, tau_m: float) -> np.ndarray:
    """(T,5) 每指"接触顶点占该指顶点比例" —— 深度代理(指尖点触≈小, 整指包握≈大)。"""
    T = len(hand_w)
    out = np.zeros((T, 5))
    ang, rv, tr = opose[:, 0], opose[:, 1:4], opose[:, 4:7] / 1000.0
    Rw = axangle_mat(rv)
    ca, sa = np.cos(-ang), np.sin(-ang)
    for t in range(T):
        h_can = (hand_w[t] - tr[t]) @ Rw[t]
        d_b, _ = obj["bottom"]["tree"].query(h_can)
        Rz = np.array([[ca[t], -sa[t], 0], [sa[t], ca[t], 0], [0, 0, 1]])
        d_t, _ = obj["top"]["tree"].query(h_can @ Rz.T)
        d = np.minimum(d_b, d_t)
        for fi in range(5):
            sel = fseg == fi
            out[t, fi] = float((d[sel] < tau_m).mean())
    return out


def segments(mask: np.ndarray, min_len: int = 5) -> list[list[int]]:
    segs, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        elif not v and s is not None:
            if i - s >= min_len:
                segs.append([s, i - 1])
            s = None
    if s is not None and len(mask) - s >= min_len:
        segs.append([s, len(mask) - 1])
    return segs


def summarize(dists: np.ndarray, depth5: np.ndarray, tau_m: float) -> dict:
    """单手: 接触段 + 稳定抓握段的接触方式。dists (T,6), depth5 (T,5)。"""
    contact = dists < tau_m                                # (T,6)
    n_fing = contact[:, :5].sum(1)                         # 不含掌
    any_c = n_fing >= 1
    segs = segments(any_c)
    grasp = max(segs, key=lambda ab: ab[1] - ab[0]) if segs else None
    mode = None
    if grasp is not None:
        s, e = grasp
        nf = n_fing[s:e + 1]
        fset = [FINGERS[i] for i in range(5) if contact[s:e + 1, i].mean() > 0.5]
        dfrac = depth5[s:e + 1].max(1)
        mode = {
            "grasp_segment": [int(s), int(e)],
            "n_fingers_median": int(np.median(nf)),
            "fingers": fset,
            "palm_contact_frac": round(float(contact[s:e + 1, 5].mean()), 3),
            "depth_frac_median": round(float(np.median(dfrac[dfrac > 0])) if (dfrac > 0).any() else 0.0, 3),
        }
    return {"contact_segments": segs,
            "frames_in_contact": int(any_c.sum()),
            "grasp_mode": mode}


def viz_overlay(take_dir: Path, meta: dict, hands_w: dict, obj_parts: dict,
                opose: np.ndarray, contact5: np.ndarray, out_dir: Path, n_frames: int = 3):
    """真值投影到我们的视频帧 —— 坐标约定的视觉验收。接触指画红, 非接触绿, 物体点蓝。"""
    import cv2
    ego = np.load(ARCTIC / "raw_seqs" / meta["subject"] / f"{meta['seq']}.egocam.dist.npy",
                  allow_pickle=True).item()
    Rc = np.asarray(ego["R_k_cam_np"], np.float64)         # (T,3,3) world→ego
    Tc = np.asarray(ego["T_k_cam_np"], np.float64).reshape(-1, 3, 1)   # ★已是米(实测1.58级), 别除1000
    K = np.asarray(meta["K_video"], np.float64)
    idx = meta["index"]
    cap = cv2.VideoCapture(str(take_dir / meta["video"]))
    picks = np.linspace(0, len(idx) - 1, n_frames + 2)[1:-1].astype(int)
    ang, rv, tr = opose[:, 0], opose[:, 1:4], opose[:, 4:7] / 1000.0
    Rw = axangle_mat(rv)
    for pi in picks:
        vf, vidx = idx[pi]["video_frame"], idx[pi]["arctic_vidx"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
        ok, img = cap.read()
        if not ok:
            continue

        def proj(pts_w):
            pc = (Rc[vidx] @ pts_w.T + Tc[vidx]).T
            uv = (K @ pc.T).T
            return uv[:, :2] / np.clip(uv[:, 2:3], 1e-6, None), pc[:, 2]

        Rz = np.array([[np.cos(opose[vidx, 0]), -np.sin(opose[vidx, 0]), 0],
                       [np.sin(opose[vidx, 0]), np.cos(opose[vidx, 0]), 0], [0, 0, 1]])
        for part, extra in (("bottom", np.eye(3)), ("top", Rz)):
            vw = (extra @ obj_parts[part]["verts"].T).T @ Rw[vidx].T + tr[vidx]
            uv, z = proj(vw[::12])
            for (u, v), zz in zip(uv, z):
                if zz > 0 and 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    cv2.circle(img, (int(u), int(v)), 1, (255, 160, 0), -1)
        for si, side in enumerate(("left", "right")):
            uv, z = proj(hands_w[side][vidx][::6])
            hot = contact5[vidx, si, :5].any()
            col = (0, 0, 255) if hot else (0, 200, 0)
            for (u, v), zz in zip(uv, z):
                if zz > 0 and 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    cv2.circle(img, (int(u), int(v)), 2, col, -1)
        cv2.imwrite(str(out_dir / f"viz_v{vidx:04d}.jpg"), img)
    cap.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--take", required=True, help="如 s05__laptop_grab_01")
    ap.add_argument("--video-dir", type=Path,
                    default=Path("/home/lyh/Project/Reconstruct_and_Retarget/Data/arctic15_local"))
    ap.add_argument("--out", type=Path,
                    default=Path("/home/lyh/Project/Reconstruct_and_Retarget/Data/arctic_gt_contacts"))
    ap.add_argument("--viz", action="store_true")
    a = ap.parse_args(argv)
    subject, seq = a.take.split("__", 1)
    obj_name = seq.split("_")[0]

    mano = np.load(ARCTIC / "raw_seqs" / subject / f"{seq}.mano.npy", allow_pickle=True).item()
    opose = np.asarray(np.load(ARCTIC / "raw_seqs" / subject / f"{seq}.object.npy",
                               allow_pickle=True), np.float64)
    T = len(opose)
    print(f"[gt] {a.take}: {T} 帧, 物体 {obj_name}")

    obj = load_object(obj_name)
    hands_w, dists = {}, np.full((T, 2, 6), np.inf)
    depth5 = np.zeros((T, 2, 5))
    for si, side in enumerate(("left", "right")):
        hv = mano_verts({k: np.asarray(v) for k, v in mano[side].items()}, side)
        hands_w[side] = hv
        fseg = vert_fingers(side)
        dists[:, si, :] = per_finger_dists(hv, fseg, obj, opose)
        depth5[:, si, :] = contact_depth_frac(hv, fseg, obj, opose, 0.005)
        print(f"  {side}: 距离最小值 {dists[:, si, :5].min()*1000:.1f}mm")

    out_dir = a.out / a.take
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"dists_m": dists.astype(np.float32), "depth_frac_5mm": depth5.astype(np.float32),
               "fingers": np.array(FINGERS)}
    summary = {"take": a.take, "object": obj_name, "num_frames": T, "taus_mm": list(TAUS_MM),
               "hands": {}}
    for tau in TAUS_MM:
        c = dists < tau / 1000.0
        payload[f"contact_{tau}mm"] = c
        payload[f"n_fingers_{tau}mm"] = c[:, :, :5].sum(2).astype(np.int8)
    for si, side in enumerate(("left", "right")):
        summary["hands"][side] = {f"tau_{tau}mm": summarize(dists[:, si, :], depth5[:, si, :],
                                                            tau / 1000.0) for tau in TAUS_MM}
    np.savez_compressed(out_dir / "gt_contact.npz", **payload)
    (out_dir / "gt_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    for side in ("left", "right"):
        m = summary["hands"][side]["tau_5mm"]["grasp_mode"]
        print(f"  {side}@5mm: {m}")

    if a.viz:
        meta_p = sorted(a.video_dir.glob(f"{a.take}.meta.json")) or \
                 sorted(a.video_dir.glob(f"*/{seq}.meta.json"))
        if meta_p:
            meta = json.loads(meta_p[0].read_text())
            meta.setdefault("subject", subject); meta.setdefault("seq", seq)
            viz_overlay(meta_p[0].parent, meta, hands_w, obj,
                        opose, payload["contact_5mm"], out_dir)
            print(f"  viz -> {out_dir}/viz_*.jpg")
        else:
            print(f"  ⚠ 找不到 meta.json, 跳过叠加验收({a.video_dir})")
    print(f"[gt] -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
