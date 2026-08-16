"""Shadow 抓型模板 -> SharpaWave 模板 的转换器 (源侧提取 + 回写).

拟合本身在 **ocir-grasp-synthesis** 侧做 (那边有 cuRobo 可微 FK、`right_*_fingertip`
零偏移帧, 以及成熟的 `anchored_bodex/retarget.py`)。本文件负责它的两头:

    extract  36 个 Shadow 模板 -> 骨架关键点 npz  (喂给 ocir 的拟合脚本)
    emit     ocir 拟合出的 qpos -> raw_anno/<型>.yaml  (再 dexrun op=tmpl)

用法
----
    PY=/home/lyh/anaconda3/envs/dexonomy/bin/python

    # ① 源侧提取
    MUJOCO_GL=egl $PY tools/retarget_shadow_templates.py extract --out /tmp/shadow_src.npz

    # ② (在 ocir 仓库跑拟合, 见 scripts/grasp_synthesis/retarget_shadow_to_sharpa.py)

    # ③ 回写
    MUJOCO_GL=egl $PY tools/retarget_shadow_templates.py emit --fit /tmp/sharpa_fit.npz \\
        --src /tmp/shadow_src.npz [--suffix _sd]

设计要点
--------
* **两层**: 固定的骨架关键点负责传递**手型**; 接触点在拿到姿态后按最近邻**就近分配**
  到 SharpaWave 自己的 keypoint.yaml 索引。这样每个模板接触体数量不同也不影响拟合。
* **骨架点 = 关节中心 + 指尖**。只用关节中心的话远节绕 DIP 的转角不受约束, 必须补指尖。
  Shadow 没有指尖 site, 由远节网格顶点沿指轴取最远点导出。
* **缩放**由 ocir 侧脚本的 --scale 控制 (本文件只导出未缩放的原始量)。
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

HAND_S = "assets/hand/shadow"
HAND_W = "assets/hand/sharpa_wave"

#: 骨架对应表: (Shadow body, SharpaWave link, 权重, 是否取指尖)
#: 权重递减 = 越靠远端越重要 (远端决定接触, 近端只提供姿态先验)
SKELETON = [
    # ---- 四指: PP/MP/DP 关节中心 + 指尖 ----
    ("rh_ffproximal", "right_index_PP", 0.15, False),
    ("rh_ffmiddle",   "right_index_MP", 0.30, False),
    ("rh_ffdistal",   "right_index_DP", 0.60, False),
    ("rh_ffdistal",   "right_index_fingertip", 1.00, True),
    ("rh_mfproximal", "right_middle_PP", 0.15, False),
    ("rh_mfmiddle",   "right_middle_MP", 0.30, False),
    ("rh_mfdistal",   "right_middle_DP", 0.60, False),
    ("rh_mfdistal",   "right_middle_fingertip", 1.00, True),
    ("rh_rfproximal", "right_ring_PP", 0.15, False),
    ("rh_rfmiddle",   "right_ring_MP", 0.30, False),
    ("rh_rfdistal",   "right_ring_DP", 0.60, False),
    ("rh_rfdistal",   "right_ring_fingertip", 1.00, True),
    ("rh_lfproximal", "right_pinky_PP", 0.15, False),
    ("rh_lfmiddle",   "right_pinky_MP", 0.30, False),
    ("rh_lfdistal",   "right_pinky_DP", 0.60, False),
    ("rh_lfdistal",   "right_pinky_fingertip", 1.00, True),
    # ---- 拇指: Shadow 5 节 vs SharpaWave 3 节, 按链上位置对应 ----
    ("rh_thproximal", "right_thumb_MC", 0.15, False),
    ("rh_thmiddle",   "right_thumb_PP", 0.30, False),
    ("rh_thdistal",   "right_thumb_DP", 0.60, False),
    ("rh_thdistal",   "right_thumb_fingertip", 1.00, True),
]

#: 接触体映射 (Shadow -> SharpaWave), 用于 emit 阶段的就近分配
CONTACT_BODY = {
    "rh_ffdistal": "right_index_DP", "rh_ffmiddle": "right_index_MP",
    "rh_ffproximal": "right_index_PP",
    "rh_mfdistal": "right_middle_DP", "rh_mfmiddle": "right_middle_MP",
    "rh_mfproximal": "right_middle_PP",
    "rh_rfdistal": "right_ring_DP", "rh_rfmiddle": "right_ring_MP",
    "rh_rfproximal": "right_ring_PP",
    "rh_lfdistal": "right_pinky_DP", "rh_lfmiddle": "right_pinky_MP",
    "rh_lfproximal": "right_pinky_PP", "rh_lfmetacarpal": "right_pinky_MC",
    "rh_thdistal": "right_thumb_DP", "rh_thmiddle": "right_thumb_PP",
    "rh_thproximal": "right_thumb_MC",
    "rh_palm": "right_hand_C_MC",
}


# ------------------------------------------------------------------ 工具
def _model(xml):
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    return m, mujoco.MjData(m), [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
                                 for i in range(m.nbody)]


def _tip_offset(m, bid):
    """远节 body 局部系里的指尖点: 该 body 所有网格顶点中离原点最远的那个.

    Shadow 没有指尖 site, 只能从几何导出; 取最远顶点对胶囊状指节是稳的。
    """
    best, bestd = np.zeros(3), -1.0
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid:
            continue
        did = int(m.geom_dataid[g])
        if did < 0:                       # 非网格几何, 退化用 geom_pos + size
            p = m.geom_pos[g] + np.array([0, 0, float(m.geom_size[g].max())])
            if np.linalg.norm(p) > bestd:
                best, bestd = p, float(np.linalg.norm(p))
            continue
        a, n = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
        v = m.mesh_vert[a:a + n].reshape(-1, 3)
        # 顶点在 geom 局部系 -> body 系
        import mujoco
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        v = v @ R.reshape(3, 3).T + m.geom_pos[g]
        d = np.linalg.norm(v, axis=1)
        k = int(np.argmax(d))
        if d[k] > bestd:
            best, bestd = v[k], float(d[k])
    return best


def _palm_frame(d, names, roots):
    """解剖掌系: 原点=四指根中心, x=食指根->小指根, y=指向中指尖 (正交化)."""
    r = [d.xpos[names.index(b)] for b in roots]
    o = np.mean(r, axis=0)
    x = r[3] - r[0]
    x = x / np.linalg.norm(x)
    v = r[4] - r[1]                        # 中指: 根 -> 尖
    y = v - np.dot(v, x) * x
    y = y / np.linalg.norm(y)
    return np.stack([x, y, np.cross(x, y)], axis=1), o


# ------------------------------------------------------------------ extract
def extract(args):
    import mujoco
    m, d, N = _model(f"{HAND_S}/right.xml")
    roots = ["rh_ffproximal", "rh_mfproximal", "rh_rfproximal", "rh_lfproximal", "rh_mfdistal"]

    tips = {b: _tip_offset(m, N.index(b)) for b in {s for s, _, _, t in SKELETON if t}}
    print("[extract] Shadow 指尖偏移 (远节局部系, mm):")
    for b, v in sorted(tips.items()):
        print(f"    {b:16s} {np.round(v * 1000, 1).tolist()}  |{np.linalg.norm(v)*1000:.1f}|")

    def skeleton_at(q):
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        R, o = _palm_frame(d, N, roots)
        pts = []
        for sb, _, _, is_tip in SKELETON:
            bid = N.index(sb)
            p = d.xpos[bid] + (d.xmat[bid].reshape(3, 3) @ tips[sb] if is_tip else 0.0)
            pts.append(R.T @ (p - o))
        return np.asarray(pts), R, o

    flat, _, _ = skeleton_at(np.zeros(m.nq))

    files = sorted(glob.glob(f"{HAND_S}/init_tmpl/*.npy"))
    names, skel, cps, cbs, cns = [], [], [], [], []
    for f in files:
        rec = np.load(f, allow_pickle=True).item()
        q = np.asarray(rec["grasp_qpos"][0][7:], float)[:m.nq]
        s, R, o = skeleton_at(q)
        cpn = np.asarray(rec["hand_cpn_w"], float)
        names.append(os.path.basename(f)[:-4])
        skel.append(s)
        cps.append((R.T @ (cpn[:, :3] - o).T).T)          # 接触点 -> 掌系
        cns.append((R.T @ cpn[:, 3:6].T).T)               # 法向 -> 掌系
        cbs.append(list(rec["hand_cbody"]))

    np.savez(args.out,
             names=np.array(names),
             skeleton=np.stack(skel),                      # (T, K, 3) 掌系
             skeleton_flat=flat,                           # (K, 3)
             sharpa_links=np.array([w for _, w, _, _ in SKELETON]),
             weights=np.array([w for _, _, w, _ in SKELETON]),
             is_tip=np.array([t for _, _, _, t in SKELETON]),
             contact_pos=np.array(cps, dtype=object),
             contact_normal=np.array(cns, dtype=object),
             contact_body=np.array(cbs, dtype=object),
             allow_pickle=True)
    print(f"\n[extract] {len(names)} 个模板 x {len(SKELETON)} 个骨架点 -> {args.out}")
    sp = np.stack(skel)
    print(f"[extract] 骨架点掌系范围 mm: x[{sp[...,0].min()*1000:.0f},{sp[...,0].max()*1000:.0f}] "
          f"y[{sp[...,1].min()*1000:.0f},{sp[...,1].max()*1000:.0f}] "
          f"z[{sp[...,2].min()*1000:.0f},{sp[...,2].max()*1000:.0f}]")
    return 0


# ------------------------------------------------------------------ emit
def emit(args):
    import mujoco
    import yaml
    src = np.load(args.src, allow_pickle=True)
    fit = np.load(args.fit, allow_pickle=True)
    qs = fit["qpos"]                                       # (T, 22)
    names = [str(x) for x in src["names"]]
    assert len(qs) == len(names), f"拟合 {len(qs)} 条 vs 源 {len(names)} 条"

    m, d, N = _model(f"{HAND_W}/right.xml")
    KP = yaml.safe_load(open(f"{HAND_W}/keypoint.yaml"))
    roots = ["right_index_PP", "right_middle_PP", "right_ring_PP",
             "right_pinky_PP", "right_middle_DP"]
    os.makedirs(f"{HAND_S}/../sharpa_wave/raw_anno", exist_ok=True)

    # 按指尖残差筛选 (权重 1.0 的点 = 指尖)
    keep = list(range(len(names)))
    if args.max_tip_mm is not None:
        w = np.asarray(src["weights"], float)
        tip = fit["residual_mm"][:, w > 0.9].max(axis=1)
        keep = [i for i in keep if tip[i] <= args.max_tip_mm]
        print(f"[emit] 残差筛选 <= {args.max_tip_mm}mm: 保留 {len(keep)}/{len(names)}")

    n_ok = 0
    for i, nm in enumerate(names):
        if i not in keep:
            continue
        d.qpos[:] = qs[i]
        mujoco.mj_kinematics(m, d)
        R, o = _palm_frame(d, N, roots)
        tgt_p = src["contact_pos"][i]
        tgt_b = src["contact_body"][i]
        assign, misses = {}, []
        for p, sb in zip(tgt_p, tgt_b):
            wb = CONTACT_BODY.get(sb)
            if wb is None or wb not in N or wb not in KP:
                misses.append(sb)
                continue
            bid = N.index(wb)
            Rb, tb = d.xmat[bid].reshape(3, 3), d.xpos[bid]
            kps = np.asarray(KP[wb], float)[:, :3]
            world = (Rb @ kps.T).T + tb
            local = (R.T @ (world - o).T).T                # 掌系
            k = int(np.argmin(np.linalg.norm(local - p, axis=1)))
            assign.setdefault(wb, set()).add(k)
        out = f"{HAND_W}/raw_anno/{nm}{args.suffix}.yaml"
        with open(out, "w") as fh:
            fh.write(f"# 由 Shadow 模板 `{nm}` 自动重定向而来 "
                     f"(tools/retarget_shadow_templates.py)\n")
            if misses:
                fh.write(f"# ⚠ 无法映射的接触体: {sorted(set(misses))}\n")
            fh.write("qpos: [" + ", ".join(f"{v:.4f}" for v in qs[i]) + "]\n\ncontact:\n")
            for b in sorted(assign):
                ks = sorted(assign[b])
                if len(ks) == 1:
                    fh.write(f"  {b}: {ks[0]}\n")
                else:
                    fh.write(f"  {b}:\n" + "".join(f"    - {k}\n" for k in ks))
        n_ok += 1
        print(f"  ✓ {nm}{args.suffix}.yaml  接触体 {len(assign)}"
              + (f"  ⚠漏 {len(misses)}" if misses else ""))
    # ---- 侧车文件: 每个模板在 **Shadow 原设定** 下用了几根手指 ----
    #      分组用它, 不用我们这边的 pads 判据 (那是 task-specific 的)
    import json, re as _re
    meta = {}
    for i, nm in enumerate(names):
        if i not in keep:
            continue
        cb = list(src["contact_body"][i])
        anyf, dist = set(), set()
        for b in set(cb):
            mm = _re.match(r"rh_(ff|mf|rf|lf|th)(distal|middle|proximal|metacarpal)", b)
            if mm:
                anyf.add(mm.group(1))
                if mm.group(2) == "distal":
                    dist.add(mm.group(1))
        meta[nm] = {"shadow_fingers": len(anyf), "shadow_distal": len(dist),
                    "shadow_bodies": len(set(cb)),
                    "shadow_finger_names": sorted(anyf)}
    mp = os.path.join(HAND_W, "raw_anno", "_shadow_origin.json")
    json.dump(meta, open(mp, "w"), indent=1, ensure_ascii=False)
    print(f"[emit] Shadow 原始接触统计 -> {mp}")

    print(f"\n[emit] {n_ok} 个 -> {HAND_W}/raw_anno/")
    print("       下一步: MUJOCO_GL=egl dexrun op=tmpl hand=sharpa_wave")
    return 0


# ------------------------------------------------------------------ mirror
def mirror(args):
    """右手 raw_anno -> 左手。

    左右手是**精确镜像** (build_assets.py --side left 生成的 left.xml 与 right.xml
    在同一组关节角下 link 位置差 <0.002mm), 而且 keypoint.yaml 的每个 link 索引
    一一对应。所以镜像只需两件事:
        ① qpos 原样复制 (关节序、限位左右完全一致, 实测)
        ② contact 里的 body 名 right_ -> left_ (keypoint 索引不变)
    ⚠ 不要对 qpos 做任何符号翻转 —— 实测 "AA 取反" 会产生 17mm 中位误差。
    """
    src = os.path.join(HAND_W, "raw_anno")
    dst = os.path.join(HAND_W + "_left", "raw_anno")
    os.makedirs(dst, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(src, "*.yaml"))):
        txt = open(f, encoding="utf-8").read()
        out = txt.replace("right_", "left_")
        nm = os.path.basename(f)
        head = (f"# 由右手模板 `{nm}` 镜像而来 (tools/retarget_shadow_templates.py mirror)\n"
                f"# 左右手精确镜像: qpos 原样, 仅 body 名换前缀; 勿翻转任何关节符号。\n")
        open(os.path.join(dst, nm), "w", encoding="utf-8").write(head + out)
        n += 1
    # 侧车元数据一并复制
    meta = os.path.join(src, "_shadow_origin.json")
    if os.path.exists(meta):
        import shutil
        shutil.copy(meta, os.path.join(dst, "_shadow_origin.json"))
    print(f"[mirror] {n} 个 -> {dst}")
    print("       下一步: MUJOCO_GL=egl dexrun op=tmpl hand=sharpa_wave_left")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mirror")
    e = sub.add_parser("extract"); e.add_argument("--out", default="/tmp/shadow_src.npz")
    w = sub.add_parser("emit")
    w.add_argument("--fit", required=True); w.add_argument("--src", required=True)
    w.add_argument("--suffix", default="", help="模板名后缀; 空 = 用分类学原名(会覆盖同名)")
    w.add_argument("--max-tip-mm", type=float, default=None,
                   help="只导出指尖残差 <= 该值的模板 (A组=10, A+B=20)")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return {"extract": extract, "emit": emit, "mirror": mirror}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
