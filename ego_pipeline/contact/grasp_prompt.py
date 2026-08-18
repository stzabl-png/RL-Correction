"""把接触提取结果整理成**给 GraspPose Agent 的交接件**。

GraspPose Agent 要的是"手该抓在物体的哪里"。它拿不到我们这边的中间产物, 也不该去读
npz —— 所以这里产出两份:

    grasp_prompt.json   机器读: 结构化, 带文件路径, 供程序直接消费
    grasp_prompt.md     人/LLM 读: 自然语言描述接触区域 + **可信度与已知缺陷**

============================ 为什么要写"不可信的部分" ============================

接触点是从重建里测出来的, 而重建有已知误差。只给坐标不给误差, 下游会把它当真值用:

  * 物体位姿误差会**整体平移**接触区 —— 实测物体投影 vs 实测 mask 的 IoU 只有 0.45~0.69
  * 接触区常常只覆盖物体的一侧圆弧而不是整圈, 因为手相对物体有偏移, 只有近的那半边登记上
  * 对生度低的结果**不是抓握**(可能是推/扶/蹭), 拿去做 GraspPose 会得到无法夹持的姿势

所以每条记录都带 `trust` 段, 且 md 里用明确的话写出来。**宁可让下游知道不确定, 也不要
给一个看起来干净的数字。**

============================ 几何怎么描述 ============================

Agent 不方便直接吃点云, 所以把接触区在**物体自身坐标系**里换算成三个人能读的量:

    height_pct   沿物体最长轴的高度百分比 (0=最低端)
    azimuth      绕该轴的方位角区间 + 跨度 (跨度接近 360° = 环抱一圈)
    radius_pct   离轴距离 / 物体最大半径 (接近 1 = 在最外表面)

点云本身仍然给路径(ply/npz), 需要精确几何时直接读。
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np


def _provenance() -> dict:
    """产物自报"我是谁生成的" —— 让"我手上这份是不是旧的"变成**本地可自答**。

    为什么需要 (2026-08-17, 同一形状连踩两次):
      ① manifest 裁决同步回 confidence_complete.json 只在 UCB 跑了, 本机那份仍是旧快照
      ② hand_without_target 字段全库重生成也只在 UCB 跑了, 本机 5 条全无此字段
    两次**都不报错**, 都是消费方主动去扫才发现。消费方读到一个没有某字段的文件时,
    分不开"这条 take 真不命中"和"我手上这份是旧版生成器出的" —— 这两种含义完全相反。

    ⇒ 带上生成时间 + 生成器 commit + 主机名后, 读的瞬间就能判断, 不必等人去扫。
      这是 synced_from/synced_at 的更一般形式: 不只回答"同步过没有", 而是
      "这份东西是哪一版代码、什么时候、在哪台机器上出的"。

    ⚠ **`generator_commit` 会说谎, `generator_sha1` 不会。** 加上这个字段后第一次跑就撞上了:
      UCB 报 commit 0415290, 但那台机器的这个文件是我 scp 过去的、比它的 checkout 新。
      只要有人 scp/rsync 单个文件(我们经常这么干), commit 号就与实际执行的代码脱节。
      ⇒ 同时记**本文件内容的 sha1**, 内容寻址, 骗不了。**比对时以 sha1 为准。**
    """
    import hashlib
    import os
    import socket
    import subprocess
    from datetime import datetime, timezone
    here = Path(__file__).resolve()
    commit = None
    try:
        commit = subprocess.run(
            ["git", "-C", str(here.parent), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        pass
    try:
        sha1 = hashlib.sha1(here.read_bytes()).hexdigest()[:12]
    except Exception:
        sha1 = None
    return {
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "generator_sha1": sha1,          # ★ 以此为准: 内容寻址, scp 也骗不了
        "generator_commit": commit,      # 仅供参考: 文件被 scp 覆盖时会与实际代码脱节
        "generator_host": socket.gethostname(),
        "generator_user": os.environ.get("USER"),
    }


def _describe(V: np.ndarray, w: np.ndarray) -> dict:
    """接触热点在物体自身坐标系里的位置 → 可读的几何描述。"""
    hot = w >= 0.5
    if not hot.any():
        hot = w > 0
    ax = int(np.argmax(np.ptp(V, axis=0)))               # 最长轴 = 物体主轴
    lo, hi = float(V[:, ax].min()), float(V[:, ax].max())
    rad_ax = [i for i in range(3) if i != ax]
    c = V[:, rad_ax].mean(0)
    r_all = np.linalg.norm(V[:, rad_ax] - c, axis=1)
    r_hot = np.linalg.norm(V[hot][:, rad_ax] - c, axis=1)
    ang = np.degrees(np.arctan2(*(V[hot][:, rad_ax] - c).T[::-1])) % 360
    occ = np.histogram(ang, bins=12, range=(0, 360))[0] > 0
    return {
        "principal_axis": "xyz"[ax],
        "object_size_m": np.ptp(V, axis=0).round(4).tolist(),
        "height_pct": [round(float((V[hot][:, ax].min() - lo) / max(hi - lo, 1e-9) * 100)),
                       round(float((V[hot][:, ax].max() - lo) / max(hi - lo, 1e-9) * 100))],
        "height_pct_median": round(float((np.median(V[hot][:, ax]) - lo)
                                         / max(hi - lo, 1e-9) * 100)),
        "azimuth_span_deg": int(occ.sum() * 30),
        "azimuth_sectors_30deg": [i * 30 for i, v in enumerate(occ) if v],
        "radius_pct": round(float(np.median(r_hot) / max(r_all.max(), 1e-9) * 100)),
        "n_hot": int(hot.sum()),
        "hot_frac_of_surface": round(float(hot.mean()), 4),
    }


def build(recon_dir: Path) -> tuple[dict, str]:
    """→ (json 结构, markdown 文本)。只读已有产物, 不重算。"""
    D = Path(recon_dir)
    cd = D / "contact"
    conf = {}
    cp = D / "confidence_complete.json"
    if cp.is_file():
        conf = (json.loads(cp.read_text()).get("objects") or {})

    grasps, rejected = [], []
    for f in sorted(glob.glob(str(cd / "contact_v2_*.npz"))):
        z = np.load(f, allow_pickle=True)
        m = json.loads(str(z["meta"]))
        oid, side = m["object_id"], m["side"]
        sw = m["stable_window"]
        g = _describe(z["probe_local"].astype(float), z["weight"])
        oc = conf.get(oid) or {}
        grasps.append({
            "object_id": oid, "hand": side,
            "mesh": str(D / "objects" / oid / "object_mesh_scaled_final.obj"),
            "grasp_window_frames": sw["span"], "n_frames": sw["n"],
            "contact_region": g,
            "contact_cloud_npz": f,
            "contact_cloud_ply": f.replace(".npz", ".ply"),
            "trust": {
                "opposition": round(m["opposition_tips"], 2),
                "opposition_note": "0=接触点全朝同一边(夹不住) 1=完全对生; <0.40 判非抓握",
                "hand_mask_agreement": (None if m.get("hand_mask_agreement") is None
                                        else round(m["hand_mask_agreement"], 3)),
                "hand_mask_agreement_note": "重建手落在实测手mask里的比例; 正常 0.55~0.63",
                # 物体那半边的对称体检: 低 -> 接触区**整体偏移**(形状还像样但位置错了)
                "object_mask_agreement": (None if m.get("object_mask_agreement") is None
                                          else round(m["object_mask_agreement"], 3)),
                "object_pose_suspect": m.get("object_pose_suspect"),
                "object_mask_agreement_note": "重建物体投影 vs 实测物体 mask 的 IoU; "
                                              "实测健康 0.45~0.69, <0.30 判可疑(门槛暂定)",
                "hand_surface_gap_mm": round(m["min_dist_mm_median"], 2),
                "object_conf_pos": oc.get("conf_pos_median"),
                "object_conf_rot": oc.get("conf_rot_median"),
                # 可见性: 自遮挡门只保留相机可见的一面。occluded_frac 高 + 旋转对称物体
                # -> 下游应按**高度带**展开区域(高度可信、方位不可信), 否则整手环握的
                #    抓取候选会被落区约束全否掉(GraspPose 侧实测: 16 个候选最好只 3/14 点落区)。
                "occluded_frac": m.get("occluded_frac"),
                "visibility_note": m.get("visibility_note"),
                "azimuth_span_deg": m.get("azimuth_span_deg"),
                "azimuth_is_lower_bound": m.get("azimuth_is_lower_bound"),
                "n_alternative_windows": sw.get("n_candidates"),
                "alternative_windows": sw.get("candidates", [])[1:4],
            },
        })

    # ★ 被拒列表必须读**本提取器自己**的 summary。早先读 contact_complete.json(旧步骤的
    #   产物)时漏掉了 object_0×right —— 下游会把"试过但不可用"误当成"没这条数据"。
    cs = cd / "contact_v2_summary.json"
    if cs.is_file():
        for v in json.loads(cs.read_text()).get("attempts") or []:
            if v.get("status") != "ok":
                rejected.append({"object_id": v["object_id"], "hand": v["side"],
                                 "status": v.get("status"), "why": v.get("why")})

    # ★ "有手在动, 但它没有可归属的物体" —— 记录**观测**, 不下结论 (2026-08-17)
    #
    # 背景: 拧瓶盖 29 条 take 里 25 条只重建出 1 个物体 —— 盖没被分出来。而且不是分割传播
    # 时丢的: 查 label prompt, v17A **从一开始就只注册了一个实例**, 管线全程没见过那个盖。
    # ⇒ 下游的工作清单唯一来源是 grasps[], 没有那个 object_id 就没有那一条。
    #   **"这个物体压根不存在"是一个缺席, 不是一个拒绝** —— 它不出现在 rejected[] 里,
    #   不留任何痕迹。GraspPose 侧只会安静地少抓一半, 连被驳回的机会都没有。
    #
    # 全库形状(62 条 take): grasps[] 里出现 2 手的 20 条 / 1 手 29 条 / **0 手 13 条**。
    # 典型: 左手抓到瓶身 object_0, 右手 rejected("没有既稳又贴的帧段") —— 它确实在操作盖,
    # 但盖不是物体, 所以找不到任何接触可判。
    #
    # ★ 字段名描述**观测**而不是推断(GraspPose 2026-08-17 提的, 我认): 叫
    #   suspected_missing_part 会被当成事实转述, 而 confidence:"heuristic" 会被跳过 ——
    #   **下游读的是字段名, 不是可信度的值**。所以叫 hand_without_target。
    # ★ 同时吐出原始分量(n_objects_in_take / rejected_reason), 下游不必继承我的阈值:
    #   我哪天把判据从"只有 1 个物体"改成"物体数 < 手数", 消费方不用跟着改。
    # ⚠ 局限: "没有既稳又贴的帧段"也可能就是这只手真的没抓东西(只是扶一下/在画面外)。
    #   本信号**分不开这两种**。要分开得再算一层"该手贴近物体区域的帧占比", 未做。
    n_obj = len({g["object_id"] for g in grasps} | {r.get("object_id") for r in rejected if r.get("object_id")})
    hands_with_target = {g["hand"] for g in grasps}
    unexplained = [
        {"hand": r.get("hand"),
         "n_objects_in_take": n_obj,
         "n_hands_with_target": len(hands_with_target),
         "rejected_reason": r.get("why"),
         "note": "该手无可归属物体。若本 take 是双手任务而只重建出 1 个物体, "
                 "大概率它在操作一个没被分出来的部件 —— 但也可能这只手真的没抓东西。",
         "confidence": "heuristic"}
        for r in rejected
        if r.get("hand") and r.get("hand") not in hands_with_target and n_obj <= 1
    ]

    doc = {
        "schema_version": "grasp_prompt_v2",   # v2: 加 hand_without_target + provenance
        "provenance": _provenance(),
        "take": str(D),
        "consumer": "GraspPose Agent",
        "n_grasps": len(grasps),
        "grasps": grasps,
        "rejected": rejected,
        "hand_without_target": unexplained,
        "how_contact_was_obtained":
            "单目 egocentric 视频重建 -> 逐帧手(MANO)与物体(6DoF)在世界系对齐 -> "
            "在'手物相对位姿稳定 且 贴合 且 对生'的窗口内, 统计物体表面每个采样点被手"
            "碰到的帧数比例。判贴合时**丢弃沿相机视线方向的距离分量**(单目深度近乎不可"
            "观测), 只用图像平面内的分量。",
    }
    return doc, _md(doc)


def _md(d: dict) -> str:
    L = [f"# 抓取接触先验 · 给 GraspPose Agent", "",
         f"来源 take: `{d['take']}`", "",
         "## 这份东西是什么", "",
         d["how_contact_was_obtained"], "",
         "## ⚠ 使用前必读 · 已知误差", "",
         "1. **接触区可能整体偏移。** 物体位姿是重建出来的, 实测物体投影与实测 mask 的 "
         "IoU 只有 0.45~0.69。接触区的**相对形状**(在物体上的高度/方位)比**绝对坐标**可信。",
         "2. **接触区常常只覆盖一侧圆弧而非整圈。** 手相对物体有偏移时只有近的那半边登记上, "
         "所以 `azimuth_span_deg` 是**下界**, 真实包裹范围只会更大不会更小。",
         "3. **`opposition` < 0.40 的不是抓握**(可能是推/扶/蹭), 已被过滤不会出现在下面。",
         "4. 每条都给了 `alternative_windows` —— 同一段视频里手常有多次抓握, "
         "选窗只取了最长的一次。若最长那次的接触区不合理, 换一个候选窗重跑即可。", ""]
    if not d["grasps"]:
        L += ["## 结果", "", "**本 take 没有可用的抓握。** 下面是被拒的原因。", ""]
    for i, g in enumerate(d["grasps"], 1):
        c, t = g["contact_region"], g["trust"]
        L += [f"## 抓握 {i}: {g['object_id']} × {g['hand']}手", "",
              f"- 物体网格: `{g['mesh']}`",
              f"- 尺寸: {c['object_size_m']} m, 主轴 {c['principal_axis']}",
              f"- 抓握发生在第 {g['grasp_window_frames'][0]}~{g['grasp_window_frames'][1]} 帧"
              f"({g['n_frames']} 帧)", "",
              "**接触区在物体上的位置**", "",
              f"- 沿主轴高度: **{c['height_pct'][0]}% ~ {c['height_pct'][1]}%**"
              f"(中位 {c['height_pct_median']}%, 0=最低端)",
              f"- 绕主轴方位跨度: **{c['azimuth_span_deg']}°**"
              f"{'(接近环抱一圈)' if c['azimuth_span_deg'] >= 300 else ''}",
              f"- 离轴距离: 物体最大半径的 {c['radius_pct']}%",
              f"- 热点数 {c['n_hot']}(占表面 {c['hot_frac_of_surface']*100:.1f}%)", "",
              "**可信度**", "",
              f"- 对生度 {t['opposition']}(≥0.40 才算抓握; 0.5≈环抱180°)",
              f"- 手部体检 {t['hand_mask_agreement']}(正常 0.55~0.63)",
              f"- 物体体检 {t.get('object_mask_agreement')}(健康 0.45~0.69)"
              + ("  ⚠**位姿可疑**: 接触区可能整体偏移, 相对形状比绝对坐标可信"
                 if t.get("object_pose_suspect") else ""),
              f"- 手到物体表面 {t['hand_surface_gap_mm']} mm",
              f"- 物体位姿可信度 conf_pos={t['object_conf_pos']} conf_rot={t['object_conf_rot']}"
              # ⚠ 不能写 `t[...] or 99` —— conf_rot 恰好是 0.0 时 falsy, 会被换成 99,
              #   于是"朝向完全不可信"这条最该报警的反而不报(实测瓶盖 conf_rot=0.0 漏报)
              + ("  ⚠ conf_rot 低, 朝向不可信 -> 只用高度/半径, 不要用方位角"
                 if (99 if t["object_conf_rot"] is None
                     else t["object_conf_rot"]) < 20 else ""),
              f"- **只覆盖相机可见的一面**: 自遮挡门删掉 {100*(t.get('occluded_frac') or 0):.0f}% "
              f"的背面点; 方位跨度 {t.get('azimuth_span_deg')}° 是**下界**, 真实包裹只多不少",
              f"- 同一段视频里还有 {t['n_alternative_windows']} 个不同的稳定抓握",
              f"- 精确点云: `{g['contact_cloud_ply']}`(可直接拖进 MeshLab)", ""]
    if d.get("hand_without_target"):
        L += ["## ⚠ 有手没有可归属的物体", "",
              "下面这些手被判无接触, **而本 take 只重建出 1 个物体**。若这是双手任务, "
              "大概率有个部件没被分出来(实测拧瓶盖 29 条里 25 条缺盖, 且是 v17A 从一开始"
              "就没注册, 不是传播丢的)。**这不是拒绝, 是缺席 —— 它不会出现在下面的被拒列表里。**", "",
              "⚠ 也可能这只手真的没抓东西(只是扶一下/在画面外)。本信号分不开这两种。", ""]
        for u in d["hand_without_target"]:
            L += [f"- **{u['hand']}手** — take 内物体数 {u['n_objects_in_take']}, "
                  f"有目标的手 {u['n_hands_with_target']} 只, 被拒原因: {u['rejected_reason']}"]
        L += [""]
    if d["rejected"]:
        L += ["## 被拒的(物体×手)组合", "",
              "这些**试过了但不可用**, 不是没试 —— 别当成缺数据。", ""]
        for r in d["rejected"]:
            L += [f"- `{r['object_id']} × {r['hand']}`: **{r['status']}**"
                  + (f" —— {r['why']}" if r.get("why") else "")]
        L += [""]
    L += ["## 语义对照", "",
          "| 状态 | 含义 |", "|---|---|",
          "| `hand_unreliable` | 手被重建到了错的位置, 该 take 的接触全部不可信 |",
          "| `no_contact_interval` | 这只手压根没碰过这个物体 |",
          "| `no_stable_contact` | 碰了但不构成抓握(不稳/不贴/不对生) |",
          "| `no_contact_verts` | 有窗口但没有点通过 2D 否证, 可疑 |", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recon_dir", type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    doc, md = build(a.recon_dir)
    out = a.out_dir or (a.recon_dir / "contact")
    out.mkdir(parents=True, exist_ok=True)
    (out / "grasp_prompt.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    (out / "grasp_prompt.md").write_text(md, encoding="utf-8")
    print(f"  GraspPose 交接件 -> {out/'grasp_prompt.md'}  "
          f"({doc['n_grasps']} 个抓握, {len(doc['rejected'])} 个被拒)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
