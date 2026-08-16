"""按**手对视频接触点的覆盖率**给 GraspPose 排序 —— 也可跨模板汇总, 用来选模板。

判据: 视频提取的接触热点里, 有多大比例落在**手表面** r 以内。
即"手罩住了人抓过的那片区域的多少"。

为什么用它(2026-08-15 用户挑图验证):
  * `thumb_z`(虎口朝下)只管手正不正, 不管抓在哪
  * `in_region_frac` 只统计**手的 5 个计划接触点**落没落进区域, 太稀疏
  * 覆盖率是**反过来算**的: 从 1338 个热点出发看有多少被手罩住, 统计量稳定得多
  实测: 用户从 60 个候选里凭图挑出的最满意那个(5_38), 覆盖率排**第 1**。

★ 阈值用 2cm。3cm 会选出不同的第一名 —— 2cm 更能区分"真贴上去"和"悬在附近"。

用法:
  MUJOCO_GL=egl python tools/rank_by_coverage.py --glob 'output/scan_*_sharpa_wave_v2_left' \\
      --hand assets/hand/sharpa_wave_v2_left/left.xml --region <region.npz> \\
      [--r 0.02] [--min-thumb-z -0.25] [--per-tmpl] [--top 15]
"""

import argparse
import glob
import os
import re

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True, help="实验目录通配")
    ap.add_argument("--hand", required=True, help="手 xml")
    ap.add_argument("--region", required=True, help="processed_data/<oid>/region.npz")
    ap.add_argument("--r", type=float, default=0.02, help="覆盖判定半径(m)")
    ap.add_argument("--min-thumb-z", type=float, default=-0.25,
                    help="先过虎口闸; 设 -2 可关掉")
    ap.add_argument("--mesh", default=None,
                    help="物体细网格(processed_data/<oid>/mesh/simplified.obj)。给了就启用"
                         "**腔内闸**: 手部顶点落在物体凸包内但不在实体内 = 伸进了空腔。"
                         "★ 这是 MuJoCo 碰撞和覆盖率**同时**漏掉的情形 —— 杯腔是空的, 凸分解只包"
                         "杯壁, 手指进腔不算碰撞(实测报 +0.54mm 间隙); 而覆盖率算欧氏距离, "
                         "隔着一层杯壁也算'覆盖'。两个判据同时失效, 于是这种抓取能排进前四。"
                         "对实心物体无害(凸包内=实体内, 恒为 0)。")
    ap.add_argument("--cavity-margin", type=float, default=0.005,
                    help="判'腔内'的余量(m): 离所有凸块都超过这个距离才算。见上面的注释。")
    ap.add_argument("--max-cavity", type=float, default=0.01,
                    help="允许的腔内顶点占比上限。1%% 是为了不误伤边界噪声(实测有个候选 3/10000)")
    ap.add_argument("--min-clearance", type=float, default=0.015,
                    help="手**完整网格**最低点离桌面的最小余量(m)。★ 现有 filter_plane_clearance "
                         "查的是骨架胶囊(skeleton.yaml), 网格会漏出去 —— 实测瓶子 5_35 通过了那道闸, "
                         "手最低点却在桌面以下 0.8cm。接上机械臂必撞。")
    ap.add_argument("--w-cov", type=float, default=0.5, help="覆盖率权重")
    ap.add_argument("--w-clr", type=float, default=0.3, help="离桌余量权重(越远越安全)")
    ap.add_argument("--ref-height", type=float, default=None,
                    help="视频接触带高度中位数(%%物高)。不给则高度项不计分。")
    ap.add_argument("--w-hgt", type=float, default=0.3,
                    help="接触高度与视频接触带的匹配度权重。★ 这是挡 top-down 最有效的量 —— "
                         "覆盖率在小接触区上会饱和(瓶子 5 个候选 @2cm 全 100%), 分辨不出; "
                         "而 top-down 的接触点跑到物体 88% 高度, 视频带中位才 61%。"
                         "不用'小臂仰角对比视频'是因为视频里瓶子已被举离桌面 12.7cm, "
                         "那个手臂姿态不代表桌面抓取(按它排序会把最贴桌的候选排第一)。")
    ap.add_argument("--hgt-tol", type=float, default=25.0,
                    help="高度匹配的容差(%%物高): 偏离 tol 以上得 0 分")
    ap.add_argument("--per-tmpl", action="store_true", help="按模板汇总(选模板用)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--copy-top-to", default=None)
    a = ap.parse_args()

    # ★腔内判定必须用**精确半空间测试**, 不能用 trimesh.contains ——
    #   后者靠随机方向射线投射, 边界点会随机翻转; 实测同样输入两次运行拒绝数 65 vs 42,
    #   冠军 18_Extensior_Type 在其中一次被误拒。凸包和凸分解块都是凸体, 逐面判是确定的。
    hull_pl = piece_pl = None
    if a.mesh:
        import trimesh

        def planes(m):
            n = np.asarray(m.face_normals, float)
            return n, (n * np.asarray(m.triangles, float).mean(1)).sum(1)
        om = trimesh.load(a.mesh, force="mesh", process=False)
        table_z = float(np.asarray(om.vertices)[:, 2].min())   # 规范系里桌面=物体最低点
        obj_top = float(np.asarray(om.vertices)[:, 2].max())
        hull_pl = planes(om.convex_hull)
        pdir = os.path.join(os.path.dirname(os.path.dirname(a.mesh)), "urdf", "meshes")
        piece_pl = [planes(trimesh.load(f, force="mesh", process=False).convex_hull)
                    for f in sorted(glob.glob(f"{pdir}/*.obj"))]
        if not piece_pl:
            raise SystemExit(f"找不到凸分解块: {pdir}")
    rz = np.load(a.region)
    hot = np.asarray(rz["points"], float)[np.asarray(rz["weight"], float) >= float(rz["min_weight"])]
    mj = mujoco.MjModel.from_xml_path(a.hand)
    dat = mujoco.MjData(mj)
    rng = np.random.default_rng(0)
    gv = []
    for g in range(mj.ngeom):
        if mj.geom_type[g] == 7:
            did = mj.geom_dataid[g]
            i0, n = mj.mesh_vertadr[did], mj.mesh_vertnum[did]
            v = mj.mesh_vert[i0:i0 + n].reshape(-1, 3)
            if len(v) > 500:                       # 抽样: 500/几何体足够, 全量慢 10 倍
                v = v[rng.choice(len(v), 500, replace=False)]
            gv.append((g, v))
    side_left = "left" in os.path.basename(a.hand)
    sgn = -1.0 if side_left else 1.0

    rows = []
    n_rej_cav = [0]; n_rej_clr = [0]
    for exp in sorted(glob.glob(a.glob)):
        m = re.search(r"scan_(.+?)_sharpa_wave", os.path.basename(exp))
        tmpl = m.group(1) if m else os.path.basename(exp)
        for f in sorted(glob.glob(f"{exp}/grasp_data/**/*_grasp.npy", recursive=True)):
            d = np.load(f, allow_pickle=True).item()
            q = np.asarray(d["grasp_qpos"], float).reshape(-1)
            Rm = R.from_quat(np.roll(q[3:7], -1)).as_matrix()
            tz = sgn * float(Rm[2, 1])
            if tz < a.min_thumb_z:
                continue
            dat.qpos[:] = 0
            dat.qpos[:mj.nq] = q[7:7 + mj.nq] if len(q) >= 7 + mj.nq else 0
            mujoco.mj_forward(mj, dat)
            P = np.vstack([(v @ dat.geom_xmat[g].reshape(3, 3).T + dat.geom_xpos[g]) @ Rm.T + q[:3]
                           for g, v in gv])
            P0 = P                                        # 未剔腔的全量点, 用于离桌判定
            cav_frac = 0.0
            if hull_pl is not None:
                n, dd_ = hull_pl
                inh = (P @ n.T - dd_).max(1) < 0          # 在凸包内(精确)
                cav = inh.copy()
                if inh.any():
                    # ⚠ 必须留余量: 凸分解块之间有离散化缝隙, 贴着物体表面的点会掉进缝里
                    #   被误判成"腔内"。自检(拿物体自己的顶点测)误判率 1.0%, 而阈值正好 1%,
                    #   于是紧贴杯壁的好抓取被误拒。要求"离所有凸块都超过 margin"才算腔内。
                    for n2, d2 in piece_pl:
                        cav[inh] &= (P[inh] @ n2.T - d2).max(1) >= a.cavity_margin
                    cav_frac = float(cav.mean())
                    if cav_frac > a.max_cavity:
                        n_rej_cav[0] += 1
                        continue
                    P = P[~cav]            # 腔内的点不参与覆盖率计算(隔着杯壁不算覆盖)
            dd, _ = cKDTree(P).query(hot)
            C = np.asarray(d["hand_cpn_w"], float)[:, :3]
            cov = float((dd < a.r).mean())
            clr = np.nan
            if table_z is not None:
                # 手最低点用**完整网格顶点**, 不是骨架也不是包围球 —— 见 --min-clearance 注释
                clr = float(P0[:, 2].min() - table_z)
                if clr < a.min_clearance:
                    n_rej_clr[0] += 1
                    continue
            fa = Rm @ np.array([0.0, 0.0, -1.0])          # 小臂方向(手根系 -z), 仅作诊断输出
            elev = float(np.degrees(np.arcsin(np.clip(fa[2], -1, 1))))
            ch_pct = ((np.median(C[:, 2]) - table_z) / (obj_top - table_z) * 100
                      if table_z is not None else np.nan)
            s_cov = cov
            s_clr = min(max(clr, 0.0) / 0.05, 1.0) if table_z is not None else 0.0
            s_hgt = (max(0.0, 1 - abs(ch_pct - a.ref_height) / a.hgt_tol)
                     if (a.ref_height is not None and table_z is not None) else 0.0)
            use_clr = table_z is not None
            use_hgt = a.ref_height is not None and table_z is not None
            wsum = a.w_cov + (a.w_clr if use_clr else 0) + (a.w_hgt if use_hgt else 0)
            score = (a.w_cov * s_cov + (a.w_clr * s_clr if use_clr else 0)
                     + (a.w_hgt * s_hgt if use_hgt else 0)) / max(wsum, 1e-9)
            rows.append(dict(tmpl=tmpl, f=f, cov=cov, tz=tz, cav=cav_frac,
                             clr=clr, elev=elev, score=score, ch_pct=ch_pct,
                             ch=float(np.median(C[:, 2]))))

    if not rows:
        raise SystemExit("没有候选(可能都被虎口闸筛掉了)")
    print(f"热点 {len(hot)} 个, 覆盖半径 {a.r*100:.0f}cm, 虎口闸 >= {a.min_thumb_z}"
          + (f", 腔内闸 <= {a.max_cavity:.0%} (拒绝 {n_rej_cav[0]})" if hull_pl is not None else "")
          + (f", 离桌 >= {a.min_clearance*100:.1f}cm (拒绝 {n_rej_clr[0]})" if table_z is not None else "")
          + (f", 视频接触带 {a.ref_height:.0f}%±{a.hgt_tol:.0f}" if a.ref_height is not None else ""))
    if a.per_tmpl:
        agg = {}
        for r in rows:
            agg.setdefault(r["tmpl"], []).append(r)
        out = [dict(tmpl=k, n=len(v), cov_max=max(x["cov"] for x in v),
                    cov_med=float(np.median([x["cov"] for x in v])),
                    ch_best=[x for x in v if x["cov"] == max(y["cov"] for y in v)][0]["ch"])
               for k, v in agg.items()]
        out.sort(key=lambda r: -r["cov_max"])
        print(f"\n{'模板':<26}{'过闸数':>6}{'最高覆盖':>9}{'中位覆盖':>9}{'最佳者接触高度':>14}")
        for r in out[:a.top]:
            print(f"{r['tmpl']:<26}{r['n']:>6}{r['cov_max']:>9.1%}{r['cov_med']:>9.1%}"
                  f"{r['ch_best']*100:>+13.1f}cm")
    else:
        rows.sort(key=lambda r: -r["score"])
        print(f"\n{'名次':>4}{'总分':>7}{'覆盖率':>8}{'离桌':>8}{'接触高度':>9}{'小臂仰角':>9}"
              f"  模板 / 文件")
        for i, r in enumerate(rows[:a.top], 1):
            print(f"{i:>4}{r['score']:>7.3f}{r['cov']:>8.0%}"
                  f"{(r['clr']*100 if r['clr'] == r['clr'] else float('nan')):>7.1f}cm"
                  f"{r['ch_pct']:>8.0f}%{r['elev']:>+8.0f}°  {r['tmpl']} / {os.path.basename(r['f'])}")
        if a.copy_top_to:
            import shutil
            os.makedirs(a.copy_top_to, exist_ok=True)
            for r in rows[:a.top]:
                shutil.copy(r["f"], os.path.join(a.copy_top_to,
                                                 f"{r['tmpl']}__{os.path.basename(r['f'])}"))
            print(f"前 {min(a.top, len(rows))} 名 -> {a.copy_top_to}")


if __name__ == "__main__":
    main()
