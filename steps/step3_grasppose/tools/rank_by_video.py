"""按"像不像视频里那次抓取"给 GraspPose 排序。

生成放开、筛选收紧 —— 不再强求种子逐毫米复刻重建结果(重建本身有误差, 而且 SharpaWave
比人手大一圈, 逐毫米对齐没有意义), 改成让生成器自由产出**不穿桌**的候选, 再用两把尺子排序:

  ① 接近方向偏差 `approach_dev_deg`
     接近向量 = 腕点 → 接触点质心(直接从 npy 里的 grasp_qpos/hand_cpn_w 算, 不依赖手模型)。
     只比它**与物体竖轴的夹角**(仰角) —— 绕物体转一圈不算偏差, 因为交接件写明
     `azimuth_is_lower_bound: True`(自遮挡门只留相机可见的一面), 方位角本就不可观测;
     而"从上往下扣" vs "从侧面握"会被这把尺子分开, 那才是人手轨迹里真正可信的信息。

  ② 接触区落点 `in_region_frac`
     由 `filter_region.py --rank-only` 预先写进 npy。没有该字段时该项记 None 并跳过。

综合分 = 归一化后两项加权(默认各半)。视频参考值现算: 用未扰动的视频姿态走同一套变换。

用法:
  python tools/rank_by_video.py --exp-dir output/X_sharpa_wave_left --recon <take> \\
      --object object_0 --hand left --oid X [--top 10] [--copy-top-to <dir>]
"""

import argparse
import json
import os
from glob import glob

import numpy as np
import trimesh


def quat_wxyz_to_R(q):
    w, x, y, z = np.asarray(q, float)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def elevation_deg(v):
    """向量与规范系 +z(物体竖轴)的夹角, 度。0=正上方, 90=水平, 180=正下方。"""
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(v[2] / n, -1, 1))))


def approach_elev(d):
    """一个 grasp npy 的接近仰角。腕点 = grasp_qpos[:3]; 目标 = 接触点质心。"""
    q = np.asarray(d["grasp_qpos"], float).reshape(-1)
    wrist = q[:3]
    cpn = np.asarray(d["hand_cpn_w"], float)[:, :3]
    return elevation_deg(cpn.mean(0) - wrist)


def video_reference(recon, object_id, hand, oid):
    """视频抓握窗里人手的接近仰角(规范系)。与 seed_init_from_video 用同一套变换。"""
    w = np.load(f"{recon}/world_fused.npz", allow_pickle=True)
    rq = np.load(f"{recon}/ref_qpos_{hand}.npz", allow_pickle=True)
    ids = [str(x) for x in np.atleast_1d(w["object_ids"])] if "object_ids" in w.files else ["object_0"]
    oi = ids.index(object_id) if object_id in ids else 0
    OT = w["object_ob_in_world_all"][oi] if "object_ob_in_world_all" in w.files else w["object_ob_in_world"]

    win = None
    gp = f"{recon}/contact/grasp_prompt.json"
    if os.path.isfile(gp):
        for g in json.load(open(gp)).get("grasps", []):
            if g["object_id"] == object_id and g["hand"] == hand:
                win = g.get("grasp_window_frames")
    if win is None:
        win = [0, len(OT) - 1]

    info = json.load(open(f"assets/object/custom/processed_data/{oid}/info/simplified.json"))
    com = np.asarray(info["com_offset"], float)
    R_c2i = quat_wxyz_to_R(info["canonical_from_input_rot_wxyz"])
    mesh = trimesh.load(f"assets/object/custom/processed_data/{oid}/mesh/simplified.obj",
                        force="mesh", process=False)

    wp, ok = rq["wrist_pos"], rq["valid"]
    elevs = []
    for t in range(win[0], min(win[1] + 1, len(wp))):
        if not ok[t]:
            continue
        Rw, tw = OT[t][:3, :3], OT[t][:3, 3]
        p_can = R_c2i @ ((Rw.T @ (wp[t].astype(float) - tw)) - com)
        # 目标点取物体质心 —— 视频侧没有"计划接触点", 用质心代表"手朝物体去"的方向
        elevs.append(elevation_deg(mesh.centroid - p_can))
    if not elevs:
        raise SystemExit("视频窗内没有有效帧")
    return float(np.median(elevs)), len(elevs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--data", default="grasp_data")
    ap.add_argument("--recon", required=True)
    ap.add_argument("--object", default="object_0")
    ap.add_argument("--hand", default="left", choices=("left", "right"))
    ap.add_argument("--oid", required=True)
    ap.add_argument("--w-approach", type=float, default=0.5, help="接近方向在综合分里的权重")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--copy-top-to", default=None, help="把前 N 名拷到该目录(渲染/交付用)")
    a = ap.parse_args()

    ref_elev, nref = video_reference(a.recon, a.object, a.hand, a.oid)
    files = sorted(glob(f"{a.exp_dir}/{a.data}/**/*_grasp.npy", recursive=True))
    if not files:
        raise SystemExit(f"{a.exp_dir}/{a.data} 里没有候选")

    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        dev = abs(approach_elev(d) - ref_elev)
        reg = d.get("in_region_frac")
        reg = None if reg is None else float(np.asarray(reg).reshape(-1)[0])
        rows.append({"file": f, "approach_dev_deg": dev, "in_region_frac": reg,
                     "climb_cm": float(np.asarray(d.get("climb_m", 0.0)).reshape(-1)[0]) * 100,
                     "tmpl": str(d.get("tmpl_name", "?"))})

    devs = np.array([r["approach_dev_deg"] for r in rows], float)
    d_lo, d_hi = float(np.nanmin(devs)), float(np.nanmax(devs))
    has_reg = any(r["in_region_frac"] is not None for r in rows)
    for r in rows:
        s_app = 1.0 - (r["approach_dev_deg"] - d_lo) / max(d_hi - d_lo, 1e-9)
        if has_reg and r["in_region_frac"] is not None:
            r["score"] = a.w_approach * s_app + (1 - a.w_approach) * r["in_region_frac"]
        else:
            r["score"] = s_app
    rows.sort(key=lambda r: -r["score"])

    print(f"视频参考接近仰角 {ref_elev:.1f}° (窗内 {nref} 帧中位; 0=正上方扣, 90=水平握)")
    print(f"候选 {len(rows)} 个" + ("" if has_reg else "  ⚠ npy 里没有 in_region_frac, "
                                    "只按接近方向排(先跑 filter_region --rank-only)"))
    print(f"{'名次':>4} {'分':>6} {'接近偏差':>8} {'落区':>6} {'爬升cm':>7}  模板 / 文件")
    for i, r in enumerate(rows[:a.top], 1):
        reg = "  n/a" if r["in_region_frac"] is None else f"{r['in_region_frac']:5.2f}"
        print(f"{i:>4} {r['score']:6.3f} {r['approach_dev_deg']:7.1f}° {reg} {r['climb_cm']:7.1f}  "
              f"{r['tmpl']} / {os.path.basename(r['file'])}")

    if a.copy_top_to:
        import shutil
        os.makedirs(a.copy_top_to, exist_ok=True)
        for r in rows[:a.top]:
            shutil.copy(r["file"], os.path.join(a.copy_top_to, os.path.basename(r["file"])))
        print(f"前 {min(a.top, len(rows))} 名 -> {a.copy_top_to}")

    out = os.path.join(a.exp_dir, "rank_by_video.json")
    json.dump({"ref_elev_deg": ref_elev, "w_approach": a.w_approach, "rows": rows},
              open(out, "w"), ensure_ascii=False, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
