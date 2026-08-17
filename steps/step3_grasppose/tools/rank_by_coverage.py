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
    ap.add_argument("--w-elev", type=float, default=0.3,
                    help="小臂仰角接近水平的权重。★ 挡'从上方搭在物体口沿'那类假抓取的唯一有效量 —— "
                         "它们指尖勾住外壁、手腕近乎竖直, 而覆盖率/离桌/高度三项都拦不住: "
                         "pour/17 杯子实测 30_Palmar 一批 +69° 的钩沿姿势总分 0.533, 压过了"
                         "四指环握杯身的 18_Extensior_Type__2_3(−7°, 0.509)。"
                         "仰角在数据上是双峰的: 正常环握 −26°~+20°, 钩沿 +68°~+77°, 分得很开。"
                         "★做成**打分**而不是硬闸: 有些物体本来就该 top-down 抓, 硬闸会把它们全毙掉。")
    ap.add_argument("--elev-free", type=float, default=30.0, help="仰角在此以内不扣分(度)")
    ap.add_argument("--elev-span", type=float, default=45.0, help="超出 elev-free 多少度扣到 0 分")
    ap.add_argument("--per-tmpl", action="store_true", help="按模板汇总(选模板用)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--copy-top-to", default=None)
    ap.add_argument("--json", default=None, help="把前 --top 名写成 json(给 run_take.py 汇总)")
    ap.add_argument("--reach-json", default=None,
                    help="tools/reach_filter.py 的产物。默认按**软信号**计分(Δψ 越小越好), "
                         "不淘汰 —— 见 --reach-hard。")
    ap.add_argument("--w-reach", type=float, default=0.3,
                    help="可达性权重: Δψ(可达带里离视频 yaw 最近的偏差)越小越好")
    ap.add_argument("--reach-dpsi-tol", type=float, default=30.0,
                    help="Δψ 超过它记 0 分")
    ap.add_argument("--reach-hard", action="store_true",
                    help="把可达性当**硬闸**(够不到直接出局)。"
                         "⚠ 2026-08-17 **暂不要开**: RL 侧查出 Gate 1 与 env 解的不是同一个"
                         "末端坐标系 —— Gate 1 用 `ArmIK(hand)` 的 URDF **名义基座**, 而 env 用 "
                         "`ArmIK(hand, anchor_link='arm_center', anchor_T=...)`, 后者从活着的 "
                         "articulation 读**实测**臂基座位姿。`dexmate_env.py:290` 的注释写明为什么"
                         "不能用名义值: '躯干沉降几度就让整条臂的基座偏掉, q_ref 会是个到不了的目标'。"
                         "实测已打架: Gate 1 判可达(Δψ 5.5°)的杯候选, env 里 IK 够不着、连摆都摆不上。"
                         "⇒ 当排序键(软信号)可以, 当判死线不行。等 anchor_T 接进 Gate 1 再开。")
    ap.add_argument("--no-lift-ref", action="store_true",
                    help="不把过低的目标接触高度抬到安全高度(诊断用)。默认**抬** —— 见下方注释")
    a = ap.parse_args()

    # ★腔内判定必须用**精确半空间测试**, 不能用 trimesh.contains ——
    #   后者靠随机方向射线投射, 边界点会随机翻转; 实测同样输入两次运行拒绝数 65 vs 42,
    #   冠军 18_Extensior_Type 在其中一次被误拒。凸包和凸分解块都是凸体, 逐面判是确定的。
    hull_pl = piece_pl = None
    if a.mesh:
        import trimesh

        def planes(m):
            """凸体 -> 半空间 (法向, 偏移, AABB)。

            ★AABB 必须一起带出来。重建网格的数值噪声让 trimesh 的凸包碎成海量小面:
              pour/17 杯子实测凸包 **23692 个面**(去重后仍有 23347 个 —— 它们确实是
              不同的平面, 不是重复), 18 块凸分解合计 27150 面。而半空间测试要算
              全部手点 × 全部面: 15883 × 23692 = 3.76 亿次乘加/候选, 348 个候选
              合计 1310 亿次 —— 实测评分一步跑了 17 分钟还没出结果。
              用 AABB 先筛掉盒外的点是**精确的**(盒外必在凸体外), 语义一字不改,
              而手上绝大多数点本来就离物体很远。
            """
            n = np.asarray(m.face_normals, float)
            d = (n * np.asarray(m.triangles, float).mean(1)).sum(1)
            v = np.asarray(m.vertices, float)
            return n, d, (v.min(0), v.max(0))

        def inside(P, pl, margin=0.0):
            """P 中哪些点在凸体内(margin>0 时: 离每个面都超过 margin)。AABB 预筛后精确判。"""
            n, d, (lo, hi) = pl
            out = np.zeros(len(P), bool)
            near = ((P >= lo - margin) & (P <= hi + margin)).all(1)
            if near.any():
                out[near] = (P[near] @ n.T - d).max(1) < -margin if margin else \
                            (P[near] @ n.T - d).max(1) < 0
            return out
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
        # ★模板名有两个来源, 缺一不可:
        #   扫描阶段  每个模板一个实验目录 output/scan_<模板>_<手>/
        #   精生成阶段 全部模板汇总进一个目录, 模板名在**文件名前缀** <模板>__<i>_<j>_grasp.npy
        # 只认前者会让精生成的结果全部退化成目录名 —— 控制台看不出来(它打的是文件名),
        # 但 --json 里的 tmpl 字段会整列变成垃圾, 先验回写也就跟着记错模板。
        m = re.search(r"scan_(.+?)_sharpa_wave", os.path.basename(exp))
        exp_tmpl = m.group(1) if m else None
        for f in sorted(glob.glob(f"{exp}/grasp_data/**/*_grasp.npy", recursive=True)):
            b = os.path.basename(f)
            tmpl = exp_tmpl or (b.split("__")[0] if "__" in b else os.path.basename(exp))
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
                inh = inside(P, hull_pl)                  # 在凸包内(AABB 预筛 + 精确判)
                cav = inh.copy()
                if inh.any():
                    # 凸块测试只跑 inh 那几个点(好抓取通常是 0 个), 不必再优化
                    # ⚠ 必须留余量: 凸分解块之间有离散化缝隙, 贴着物体表面的点会掉进缝里
                    #   被误判成"腔内"。自检(拿物体自己的顶点测)误判率 1.0%, 而阈值正好 1%,
                    #   于是紧贴杯壁的好抓取被误拒。要求"离所有凸块都超过 margin"才算腔内。
                    for n2, d2, _bb in piece_pl:
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
            rows.append(dict(tmpl=tmpl, f=f, cov=cov, tz=tz, cav=cav_frac,
                             clr=clr, elev=elev, ch_pct=ch_pct,
                             ch=float(np.median(C[:, 2]))))

    if not rows:
        raise SystemExit("没有候选(可能都被虎口闸筛掉了)")

    # ── 目标接触高度: 视频给的太低时抬到"安全高度" ────────────────────────────
    # 为什么要抬(pour/17 杯子实测): 杯高 13.2cm, 视频接触带只在离桌 2.5~4.6cm。手要盖住
    # 那一圈, 最低点必然压到 2cm 以内 —— 覆盖率与离桌余量是**几何上互斥**的:
    #     覆盖 54% -> 离桌 2.0cm      覆盖 30% -> 离桌 2.4cm
    #     覆盖 51% -> 离桌 1.8cm      覆盖 0~7% -> 离桌 4.3~7.1cm
    # 此时死守视频高度, 等于奖励"贴着桌子硬凑接触点": 实测排第一的是手腕近乎垂直、从上方
    # 钩杯沿的姿势(离桌 2.0cm、仰角 +77°), 而姿态正常的(覆盖 51%、高度 66%)高度分被判 0。
    # 抬多少: 由**实测候选**给出 —— drop = 接触点高度 − 手最低点高度, 是"这只手 + 这批模板
    # 在接触点以下还挂多长"。安全高度 = min_clearance + drop。这是自校准的, 不是拍脑袋常数。
    # ★对不缺高度的物体自动不生效: 瓶子实测 drop≈5.0cm -> 安全高度 37% < 视频的 61%, 不抬。
    ref_eff, ref_note = a.ref_height, ""
    if a.ref_height is not None and table_z is not None and not a.no_lift_ref:
        H = obj_top - table_z
        drop = float(np.median([(r["ch"] - table_z) - r["clr"] for r in rows]))
        safe_pct = (a.min_clearance + drop) / H * 100
        if a.ref_height < safe_pct:
            ref_eff = safe_pct
            ref_note = (f" → 抬到 {ref_eff:.0f}% (该高度处手必撞桌: 手在接触点下还挂 "
                        f"{drop*100:.1f}cm, 加 {a.min_clearance*100:.1f}cm 离桌余量)")

    # ── 可达性(可选): Δψ = 可达带里离视频 yaw 最近的偏差
    def _rkey(p):
        """匹配键。★不能只用路径: `--copy-top-to` 会把 npy 复制一份并加上 `{模板}__` 前缀,
        于是同一个候选在 grasp_data/ 里叫 `A__1_8_grasp.npy`、在 top/ 里叫
        `A__A__1_8_grasp.npy`。这里把重复的前缀折掉, 两边就对得上了。"""
        b = os.path.basename(p)
        q = b.split("__")
        if len(q) >= 3 and q[0] == q[1]:
            b = "__".join(q[1:])
        return b

    reach = {}
    if a.reach_json:
        import json as _j
        for x in _j.load(open(a.reach_json))["rows"]:
            reach[os.path.realpath(x["npy"])] = x
            reach[_rkey(x["npy"])] = x
    n_rej_reach = 0
    if reach and a.reach_hard:
        keep = []
        for r in rows:
            x = reach.get(os.path.realpath(r["f"])) or reach.get(_rkey(r["f"]))
            if x and x.get("ok"):
                keep.append(r)
            else:
                n_rej_reach += 1        # ★未测过的也出局: 未测 ≠ 可达
        rows = keep
        if not rows:
            raise SystemExit("可达性硬闸把候选全筛光了")

    use_clr = table_z is not None
    use_hgt = ref_eff is not None and table_z is not None
    use_rch = bool(reach)
    wsum = (a.w_cov + (a.w_clr if use_clr else 0) + (a.w_hgt if use_hgt else 0)
            + a.w_elev + (a.w_reach if use_rch else 0))
    for r in rows:
        s_clr = min(max(r["clr"], 0.0) / 0.05, 1.0) if use_clr else 0.0
        s_hgt = max(0.0, 1 - abs(r["ch_pct"] - ref_eff) / a.hgt_tol) if use_hgt else 0.0
        s_elev = 1 - min(max(abs(r["elev"]) - a.elev_free, 0.0) / a.elev_span, 1.0)
        x = (reach.get(os.path.realpath(r["f"])) or reach.get(_rkey(r["f"]))) if use_rch else None
        r["dpsi"] = x.get("dpsi") if x else None
        r["best_yaw"] = x.get("best_yaw") if x else None
        s_rch = (max(0.0, 1 - r["dpsi"] / a.reach_dpsi_tol)
                 if (x and x.get("ok") and r["dpsi"] is not None) else 0.0)
        r["score"] = (a.w_cov * r["cov"] + (a.w_clr * s_clr if use_clr else 0)
                      + (a.w_hgt * s_hgt if use_hgt else 0)
                      + a.w_elev * s_elev
                      + (a.w_reach * s_rch if use_rch else 0)) / max(wsum, 1e-9)

    print(f"热点 {len(hot)} 个, 覆盖半径 {a.r*100:.0f}cm, 虎口闸 >= {a.min_thumb_z}"
          + (f", 腔内闸 <= {a.max_cavity:.0%} (拒绝 {n_rej_cav[0]})" if hull_pl is not None else "")
          + (f", 离桌 >= {a.min_clearance*100:.1f}cm (拒绝 {n_rej_clr[0]})" if table_z is not None else "")
          + (f", 视频接触带 {a.ref_height:.0f}%±{a.hgt_tol:.0f}{ref_note}"
             if a.ref_height is not None else "")
          + (f", 可达性{'硬闸(拒绝 ' + str(n_rej_reach) + ')' if a.reach_hard else '软信号'}"
             f" Δψ≤{a.reach_dpsi_tol:.0f}°" if use_rch else ""))
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
        hdr = (f"\n{'名次':>4}{'总分':>7}{'覆盖率':>8}{'离桌':>8}{'接触高度':>9}{'小臂仰角':>9}")
        print(hdr + (f"{'Δψ':>7}" if use_rch else "") + "  模板 / 文件")
        for i, r in enumerate(rows[:a.top], 1):
            dps = ("" if not use_rch else
                   (f"{r['dpsi']:>6.1f}°" if r.get("dpsi") is not None else f"{'-':>7}"))
            print(f"{i:>4}{r['score']:>7.3f}{r['cov']:>8.0%}"
                  f"{(r['clr']*100 if r['clr'] == r['clr'] else float('nan')):>7.1f}cm"
                  f"{r['ch_pct']:>8.0f}%{r['elev']:>+8.0f}°" + dps
                  + f"  {r['tmpl']} / {os.path.basename(r['f'])}")
        if a.copy_top_to:
            import shutil
            os.makedirs(a.copy_top_to, exist_ok=True)
            for r in rows[:a.top]:
                shutil.copy(r["f"], os.path.join(a.copy_top_to,
                                                 f"{r['tmpl']}__{os.path.basename(r['f'])}"))
            print(f"前 {min(a.top, len(rows))} 名 -> {a.copy_top_to}")
        if a.json:
            import json
            keep = ("score", "cov", "clr", "ch_pct", "elev", "dpsi", "best_yaw", "tmpl")
            json.dump({"ref_height_video": a.ref_height, "ref_height_used": ref_eff,
                       "ref_lifted": bool(ref_note),
                       "rows": [{**{k: (None if r[k] != r[k] else round(float(r[k]), 4))
                                    for k in keep if k != "tmpl"},
                                 "tmpl": r["tmpl"], "npy": os.path.abspath(r["f"])}
                                for r in rows[:a.top]]},
                      open(a.json, "w"), ensure_ascii=False, indent=1)
            print(f"排名 json -> {a.json}")


if __name__ == "__main__":
    main()
