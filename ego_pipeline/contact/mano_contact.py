#!/usr/bin/env python
"""C1: 用重建 MANO 全手网格测接触 —— 替换"SharpaWave 指尖垫"探头。

Phase B 账单(arctic15 vs GT)的结论: 指垫探头报的指头 94% 是对的但只报出一半
(R 0.46), 病根是 power 抓握的接触界面在中节指骨+手掌, 指垫结构性测不到。
本模块直接用重建管线已经算好的 MANO 778 顶点(replay_world.npz, 世界系):

  逐帧: 手顶点 → 物体规范系 → 最近点贴合(补偿重建的系统性手物深度偏移)
        → KDTree 表面测距 → 按 6 区(5指+掌)取最小距离
  逐段: 接触区间内多帧投票(抗单帧闪烁/重定向抖动/接近相的假贴合)

  贴合依据: 重建的手物绝对距离被深度误差污染(pour 实测: 明明握着, 全手悬空
  34~60mm), 但手相对物体的"哪个部位最近"的形状信息保留着。2D 邻接区间(detect)
  说明该帧手物在接触, 于是把"全手最近顶点到表面"的向量当作该帧系统偏移整手
  平移——贴合幅度(offset_mm)全程记录, 也是重建质量的免费诊断量。
  输出: contact_fingers.json   每手每物体 {指集合, n, 掌, 深度档}   ← 模板选择器的几何证人
        contact_mano_<oid>.npz 逐帧逐区距离时间线                    ← 诊断/对拍
        expected_area_<oid>.npz 物体表面接触热区(顶点权重)           ← Dexonomy 区域条件合成

顶点分区标签是手模型常数(data/mano_vert_regions_*.npy, lbs argmax 离线生成),
管线运行时不依赖 smplx。τ 与投票阈值只允许在 dev 物体上标定(反作弊纪律)。

用法: python -m contact.mano_contact <take_dir> [--tau 0.035] [--vote 0.3]
默认参数在 arctic15 dev 物体(box/laptop/ketchup/mixer)上网格标定(mano_vs_gt.py),
held-out 7 物体终评 P0.91/R0.83/F1 0.87(旧指垫探头 R0.46), 未用 held-out 调参。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
REGIONS = ("thumb", "index", "middle", "ring", "pinky", "palm")
SIDES = ("left", "right")
N_SURF = 60000                    # 物体表面采样数, 与 gt_contacts.py 同口径


def _longest_run(flags, max_gap: int = 5):
    """最长的 True 连续段(容忍 ≤max_gap 帧缺口), 返回 (起, 止) 下标或 None。"""
    idx = np.where(flags)[0]
    if len(idx) == 0:
        return None
    best = cur = [idx[0], idx[0]]
    for i in idx[1:]:
        if i - cur[1] <= max_gap + 1:
            cur[1] = i
        else:
            cur = [i, i]
        if cur[1] - cur[0] > best[1] - best[0]:
            best = list(cur)
    return tuple(best)


def _load_intervals(take: Path, oid: str, side: str) -> list:
    for name in (f"contact_auto_{oid}.json", "contact_auto.json"):
        p = take / name
        if p.is_file():
            iv = json.loads(p.read_text())["annotations"].get(side) or []
            if iv:
                return iv
    return []


def _object_poses(w, n_obj: int):
    """(oid, pose(T,4,4), valid(T)) 列表; 单物体旧契约回退到非 _all 键。"""
    if "object_ob_in_world_all" in w:
        ids = [str(x) for x in w["object_ids"]]
        return [(ids[i], w["object_ob_in_world_all"][i], w["object_pose_valid_by_frame"][i])
                for i in range(n_obj)]
    T = w["object_ob_in_world"]
    return [("object_0", T, np.isfinite(T).all((1, 2)))]


def measure(take: Path, tau: float, vote: float, min_frames: int, align: bool = True) -> dict:
    rw = np.load(take / "replay_world.npz", allow_pickle=True)
    w = np.load(take / "world_fused.npz", allow_pickle=True)
    labels = {s: np.load(HERE / "data" / f"mano_vert_regions_{s}.npy") for s in SIDES}

    meshes = {}
    mesh_names = [str(x) for x in np.atleast_1d(w.get("object_mesh_filenames", w["mesh_filename"]))]
    objs = _object_poses(w, len(mesh_names))
    for (oid, _, _), mn in zip(objs, mesh_names):
        mp = take / mn if (take / mn).is_file() else take / "objects" / Path(mn).name
        m = trimesh.load(mp, force="mesh", process=False)
        pts, _ = trimesh.sample.sample_surface(m, N_SURF)
        meshes[oid] = (np.asarray(pts, dtype=np.float64), cKDTree(pts))

    out = {"schema_version": "contact_fingers_v1", "tau_m": tau, "vote": vote,
           "align": align, "hands": {}}
    for side in SIDES:
        verts_t = np.asarray(rw[f"mano_verts_{side}"], dtype=np.float64)   # (T,778,3)
        hvalid = np.asarray(rw[f"valid_{side}"]) > 0.5
        lab = labels[side]
        reg_idx = [np.where(lab == i)[0] for i in range(6)]
        per_obj = {}
        for oid, pose, ovalid in objs:
            ivs = _load_intervals(take, oid, side)
            frames = [f for s, e in ivs for f in range(s, e + 1)
                      if f < len(verts_t) and hvalid[f] and bool(ovalid[f])]
            if len(frames) < min_frames:
                per_obj[oid] = {"status": "no_interval" if not ivs else "too_few_valid_frames",
                                "n_frames": len(frames)}
                continue
            pts, tree = meshes[oid]
            dists = np.empty((len(frames), 6), dtype=np.float32)
            frame_hits = []
            offsets = np.empty(len(frames), dtype=np.float32)
            depth = np.empty(len(frames), dtype=np.float32)
            for k, f in enumerate(frames):
                R, t = pose[f, :3, :3], pose[f, :3, 3]
                v_obj = (verts_t[f] - t) @ R                       # world -> 物体规范系
                d, nn = tree.query(v_obj, workers=-1)
                if align:                                          # 最近点贴合: 整手平移补偿
                    j = int(np.argmin(d))
                    shift = pts[nn[j]] - v_obj[j]
                    offsets[k] = np.linalg.norm(shift)
                    v_obj = v_obj + shift
                    d, _ = tree.query(v_obj, workers=-1)
                else:
                    offsets[k] = 0.0
                for i in range(6):
                    dists[k, i] = d[reg_idx[i]].min()
                near = d < tau
                depth[k] = float(near.mean())
                if near.any():                                     # 物体侧(核心段过滤在下方)
                    hi = tree.query_ball_point(v_obj[near], tau, workers=-1)
                    frame_hits.append(np.unique(np.concatenate(
                        [np.asarray(h, dtype=np.int64) for h in hi if h]
                        or [np.empty(0, np.int64)])))
                else:
                    frame_hits.append(np.empty(0, np.int64))
            # 抓握核心段: 贴合让最近区每帧免费"接触", 单区证据不算数;
            # ≥2 区同时在 τ 内才算抓握成立, 取允许小缺口(≤5帧)的最长连续段。
            hit = dists < tau                                      # (F,6)
            established = hit.sum(axis=1) >= 2
            core = _longest_run(established, max_gap=5)
            if core is None:
                per_obj[oid] = {"status": "no_grasp_core", "n_frames": len(frames),
                                "max_regions": int(hit.sum(axis=1).max())}
                continue
            c0, c1 = core
            contact_frac = hit[c0:c1 + 1].mean(axis=0)             # 只在核心段内投票
            fingers = [REGIONS[i] for i in range(5) if contact_frac[i] >= vote]
            depth_frac = float(np.median(depth[c0:c1 + 1]))
            per_obj[oid] = {
                "status": "ok", "n_frames": len(frames),
                "fingers": fingers, "n_fingers": len(fingers),
                "palm": bool(contact_frac[5] >= vote),
                "contact_frac": {REGIONS[i]: round(float(contact_frac[i]), 3) for i in range(6)},
                "median_dist_mm": {REGIONS[i]: round(float(np.median(dists[:, i])) * 1000, 1)
                                   for i in range(6)},
                "grasp_core_frames": [int(frames[c0]), int(frames[c1])],
                "offset_mm": {"median": round(float(np.median(offsets)) * 1000, 1),
                              "p90": round(float(np.percentile(offsets, 90)) * 1000, 1)},
                "depth_frac": round(depth_frac, 3),
                # 深度档界: GT 实测 捏取≈0.2 / 包握≈0.45, 中点 0.32 分界
                "depth_class": "full" if depth_frac >= 0.32 else ("pad" if depth_frac >= 0.12 else "tip"),
            }
            np.savez_compressed(take / "contact" / f"contact_mano_{oid}_{side}.npz",
                                frames=np.array(frames), dists_m=dists, regions=np.array(REGIONS),
                                tau_m=tau, depth_frac_per_frame=depth,
                                offset_m=offsets)
            area_hits = np.zeros(len(pts), dtype=np.float32)
            for k in range(c0, c1 + 1):
                area_hits[frame_hits[k]] += 1
            ap = take / "contact" / f"expected_area_{oid}_{side}.npz"
            np.savez_compressed(ap, points=pts.astype(np.float32),
                                weight=area_hits / (c1 - c0 + 1),
                                tau_m=tau, n_frames=c1 - c0 + 1,
                                core_frames=np.array([frames[c0], frames[c1]]))
        ok = {k: v for k, v in per_obj.items() if v.get("status") == "ok"}
        out["hands"][side] = {"objects": per_obj,
                              "primary": max(ok, key=lambda k: ok[k]["n_frames"]) if ok else None}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("take", type=Path)
    ap.add_argument("--tau", type=float, default=0.035, help="接触距离阈值(米); dev 标定 2026-08-12")
    ap.add_argument("--vote", type=float, default=0.3, help="核心段内接触帧占比阈值; dev 标定")
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--no-align", action="store_true", help="关掉最近点贴合(诊断用)")
    a = ap.parse_args(argv)
    (a.take / "contact").mkdir(exist_ok=True)
    out = measure(a.take, a.tau, a.vote, a.min_frames, align=not a.no_align)
    p = a.take / "contact" / "contact_fingers.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for side, h in out["hands"].items():
        for oid, r in h["objects"].items():
            if r.get("status") == "ok":
                star = "★" if oid == h["primary"] else " "
                print(f"[mano_contact] {side:5s} {oid}{star} {r['n_fingers']}指{r['fingers']} "
                      f"palm={r['palm']} depth={r['depth_class']}({r['depth_frac']}) "
                      f"帧数{r['n_frames']}")
            else:
                print(f"[mano_contact] {side:5s} {oid}  [{r['status']}]")
    print(f"[mano_contact] -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
