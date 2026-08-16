"""Filter grasps whose contacts fall outside the video-derived expected area.

区域条件合成的"后门保证": 前门(obj_loader 撒点过滤)只是偏置, 局部优化仍可能把
接触点带出区域。这里对合成产物逐个验收: 把 obj_cpn_w(世界系接触点)变回物体
规范系, 数落在 region.npz(import_object --region 生成, 已在规范系)邻域内的比例,
低于 --min-frac 的搬进 <data>_out_of_region/ (同 filter_hand_orientation 套路)。

用法:
  python tools/filter_region.py --exp-dir output/<名>_sharpa_wave --data grasp_data \
      [--region <region.npz>] [--min-frac 0.6] [--radius 0.02]
--region 缺省时从 npy 的 scene_cfg task.region 里自取。
"""

import argparse
import glob
import os
import shutil

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def in_region_frac(d: dict, tree: cKDTree, radius: float) -> float:
    pose, scale = np.asarray(d["obj_pose"], np.float64), np.asarray(d["obj_scale"], np.float64)
    Rm = R.from_quat(np.roll(pose[3:7], -1)).as_matrix()      # npy 四元数是 wxyz
    p_can = ((np.asarray(d["obj_cpn_w"], np.float64)[:, :3] - pose[:3]) @ Rm) / scale
    dist, _ = tree.query(p_can)
    return float((dist <= radius).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--data", default="grasp_data")
    ap.add_argument("--region", default=None, help="region.npz; 缺省从 npy scene_cfg 自取")
    ap.add_argument("--min-frac", type=float, default=0.6)
    ap.add_argument("--radius", type=float, default=None, help="缺省用 region.npz 里存的")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rank-only", action="store_true",
                    help="★不做硬过滤, 只把落区比例写进 npy 当排序分。用户 2026-08-14 拍板: "
                         "通过力学检验(力封闭+不穿模+不压桌)的候选全部保留, 落区只作偏好。"
                         "原因: 自遮挡门下区域只覆盖手能看见的一面, 整手环握的接触绕一圈, "
                         "硬过滤会把合法抓取全否掉(瓶子 15 个通过力学检验的只剩 1 个)。")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.exp_dir, a.data, "**", "*.npy"), recursive=True))
    files = [f for f in files if not os.path.basename(f).startswith(("_", "."))]
    if not files:
        print(f"[region] {a.exp_dir}/{a.data}: 无 npy")
        return
    tree = radius = None
    if a.region:
        rz = np.load(a.region)
        pts = np.asarray(rz["points"])[np.asarray(rz["weight"]) >= float(rz["min_weight"])]
        tree, radius = cKDTree(pts), a.radius or float(rz["radius"])

    rej_dir = os.path.join(a.exp_dir, a.data + "_out_of_region")
    kept = moved = 0
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        if tree is None:
            rcfg = ((d.get("scene_cfg") or {}).get("task") or {}).get("region")
            if not rcfg and d.get("scene_path"):        # 旧契约 npy 只存 scene_path(相对仓根)
                sp = str(d["scene_path"])
                for cand in (sp, os.path.join(os.path.dirname(os.path.dirname(
                        os.path.abspath(__file__))), sp)):
                    if os.path.isfile(cand):
                        sc = np.load(cand, allow_pickle=True).item()
                        rcfg = (sc.get("task") or {}).get("region")
                        break
            if not rcfg:
                print(f"[region] {os.path.basename(f)}: 无 region 配置, 跳过")
                continue
            rz = np.load(rcfg["path"])
            pts = np.asarray(rz["points"])[np.asarray(rz["weight"]) >= rcfg["min_weight"]]
            tree, radius = cKDTree(pts), a.radius or rcfg["radius"]
        frac = in_region_frac(d, tree, radius)
        if a.rank_only:                       # 只打分不淘汰
            d["in_region_frac"] = float(frac)
            np.save(f, d, allow_pickle=True)
            kept += 1
            continue
        if frac >= a.min_frac:
            d["in_region_frac"] = float(frac)
            np.save(f, d, allow_pickle=True)
            kept += 1
            continue
        moved += 1
        if not a.dry_run:
            os.makedirs(rej_dir, exist_ok=True)
            base = os.path.splitext(f)[0]
            for sib in glob.glob(base + "*"):
                shutil.move(sib, os.path.join(rej_dir, os.path.basename(sib)))
    if a.rank_only:
        print(f"[region] 排序模式: {kept} 个候选已写入 in_region_frac(不淘汰), r={radius}")
    else:
        print(f"[region] 保留 {kept} / 移出 {moved} (阈值 in_region≥{a.min_frac}, r={radius})")


if __name__ == "__main__":
    main()
