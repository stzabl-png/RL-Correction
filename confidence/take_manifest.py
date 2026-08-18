#!/usr/bin/env python
"""Take 清单 —— 把逐帧打分汇总成 take 级裁决, RL/下游从这里取数据, 不要自己扫目录。

裁决四层:
  0. 旧输出根取代 (superseded): 同一条视频在两代流水线下各建过一次时, 老根那条不参与
     后续任何裁决。见 ROOT_RANK。
  1. 硬排除(人工名单): "空透明"物体等 FP 系统性失灵的存量。
     ★规则 v2(2026-08-10 pour/11 实证): 只过滤**空透明**; 透明容器装深色液体不过滤
     (茶瓶 conf 87/46 可用), 裁决交给 confidence。新数据由重建前 VLM 三分类路由,
     根本不进流水线; EXCLUDE 名单只保留历史遗留(video 3 家族)。
  2. 同视频同物体多次重建 → 自动择优 (deselected):
     - 分组: 同 task 目录 + take 名的源视频前缀(如 2_bottle/2_cap/2_scene 同属视频 2)
     - 同视频 ≠ 同物体: 2_cap 重建的是瓶盖, 2_bottle/2_scene 是瓶身, 不能互相顶替。
       用拟合尺度后的 mesh 三轴尺寸判同物体(各维之比 < 1.35)
     - 同物体组内按 (conf_pos_median, conf_rot_median) 择优, 其余落选。
       实证: 2_scene 75/38 胜 2_bottle 56.5/9, 与人眼判断一致(2026-08-10 校准)。
     - DESELECTED 人工名单仍是最高优先(override), 自动裁决只加不减。
  3. 分数分级 (active take 内部):
     position: conf_pos_median ≥70 good / ≥40 mixed / else poor
     rotation_usable = conf_rot_median ≥ 30 且 refuted 占比 ≤ 20%   (宁缺毋假:
       不满足的 take 旋转通道整条标 unusable, RL 对该维度自由探索, 不给平滑曲线当参考)
     rotation_grade: ≥70 good / ≥50 mixed / else poor  (ARCTIC 真值标定, 2026-08-17)
       ★与 rotation_usable **并存**: 后者是存量口径不动它。真值显示那条 30 分线站不住
       (两侧误差 p90 89.9° vs 91.4°), 三档才单调(20.4/34.9/103.0°)。详见 ROT_GRADE_* 注释,
       **误报率与适用范围随 manifest 的 rotation_grade_caveats 字段一起下发**。
     free_axes: rot_observability < 0.08 的轴永久自由 (近对称轴, 图像上看不出转没转)

并发安全: 写 TAKE_MANIFEST.json 走 poseqa_lock (批量队列多 worker 同时收尾时不互相覆盖)。

用法:
  python take_manifest.py --audit <poseqa>/pose_audit.json --out <poseqa>/TAKE_MANIFEST.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pose_audit import poseqa_lock

# ── 人工裁定 (2026-08-09/10), 改动需重新过用户 ────────────────────────────────
EXCLUDE_TRANSPARENT = {          # 存量"空透明"物体: FP 系统性失灵, 整条不进库(v2:装深色液体的不算)
    "egodex/screw_unscrew_bottle_cap/3_scene",
    "egodex/screw_unscrew_bottle_cap/3_scene_frame20",
}
DESELECTED = {                   # 人工 override; 自动择优结果与此冲突时以此为准
    "egodex/screw_unscrew_bottle_cap/2_bottle": "egodex/screw_unscrew_bottle_cap/2_scene",
    "egodex/screw_unscrew_bottle_cap/12_scene_v17aframes": "egodex/screw_unscrew_bottle_cap/12_scene",
}
TUNED = {                        # 判据阈值在这 4 条上标定过 → 它们的成绩是"背过答案的"
    "egodex/part2/basic_pick_place/13", "egodex/part2/basic_pick_place/6",
    "egodex/screw_unscrew_bottle_cap/27_scene", "egodex/part2/basic_pick_place/11",
}

# ★ 同一条视频在**不同输出根**下重建过两次(流水线换代), audit 会各留一条记录, 而 rel_take
#   带着根名所以两条不撞键 → manifest 同时出现"pour/17 好"和"pour/17 烂"两行 active,
#   RL 按 take 名取数会拿到哪条全看运气。2026-08-17 实测 6 组冲突(EgoDex 3 / ARCTIC 3),
#   最极端的 pour/17 object_0 是 crot 2.0 vs 66.5。⇒ 老根记录判 superseded, 不参与择优。
#   数字越大越新; 未列出的根按 0(与老根并列 → 不判定, 宁可两条都留也不错杀)。
ROOT_RANK = {"egodex": 1, "egodex_auto": 2, "arctic": 1, "arctic15": 2}

# ── 旋转分档 (2026-08-17, ARCTIC 真值标定) ──────────────────────────────────
# 位置早就是三档(position_grade), 旋转却只有一个二元闸 —— 设计不对称, 且那条 30 分线
# 经真值检验站不住: 判"可用"与判"弃用"两侧的真实旋转误差 p90 几乎相同(89.9° vs 91.4°),
# 中位只差 12°; 而 30~50 这一段的误差比 <30 段**还大**(非单调)。
#
# 标定: ARCTIC arctic15 51 条 take / 16889 帧, 误差 = 我们与真值的"朝向变化量"之差
#       (frame-convention free), 见 tools/arctic_eval/arctic_vs_recon.py。
#
#   档          take数   真实旋转误差中位   p75      占比
#   good ≥70      7        20.4°        23.6°    22%
#   mixed 50~70  14        34.9°        86.2°    23%
#   poor  <50    30       103.0°       128.0°    56%
#
# ★★ 这个标签是**排序**不是**合格证**, 必须连同下面两条一起被下游知道:
#   ① good 档 7 条里有 1 条真实误差 113° —— **约 1/7 误报**。别当"这条一定对"。
#      反向也有: poor 档 30 条里 4 条其实只有 9~12°(被冤枉)。
#   ② 三档本身是 take 级判定: 留一验证仅 37% 的 take 内部三档单调 —— 它排的是
#      "哪条视频整体更靠谱", 不是"这条视频里哪几帧更靠谱"。
#
# ★ 逐帧规则(GUIDE 的 "conf_rot≥30 且 σ_rot≤5° → 该帧可用")**是成立的**, 但必须用
#   **增量**误差去验, 不能用累计误差:
#     用累计误差(相对第0帧)  相关中位 -0.051, 方向对 55%, 规则反向 58%  → 看着像"无效"
#     用增量误差(相邻帧)     相关中位 -0.242, 方向对 93%, 规则反向 17%  → 有效
#   累计量在序列中途翻面后会让**之后每一帧**都爆表, 与那些帧各自的分数无关, 天生是
#   take 级的量 —— 拿它验逐帧判据会系统性地判它无效。我 2026-08-17 先用累计量得出
#   "逐帧规则没证据支持"并差点写进本文件, 换增量量后结论完全翻转。
#   效果量: 低分帧每帧多错 +2.0°, 基线 3.69°/帧(即约 1.5 倍), 45 条 take。
#
# ⚠ 标定物体全是 ARCTIC 大件(微波炉/笔记本/盒子), 我们要用的是瓶杯。**跨物体类别是否可搬,
#   未验证**, 且 ARCTIC 里没有瓶杯 —— 重算 ARCTIC 也解决不了这个。
ROT_GRADE_GOOD = 70
ROT_GRADE_MIXED = 50
ROT_GRADE_CALIB = ("arctic15 51 takes/16889 frames, 2026-08-17; "
                   "good档误报率约 1/7; 三档为 take 级判定(仅 37% 的 take 内部单调); "
                   "逐帧规则另行验证成立, 见 per_frame_rule")
ROT_PER_FRAME_OK = True   # 逐帧用法已获真值支持(需用增量误差验, 见上); 效果 ~1.5x

ROT_OBS_FREE = 0.08              # 观测度低于此的轴 → RL 永久自由
ROT_CONF_MIN = 30                # conf_rot 中位低于此 → 旋转整条 unusable
ROT_REFUTED_MAX = 0.20           # 被 CT 反驳帧占比超此 → 同上 (情形 C 高发段)
SAME_OBJ_RATIO = 1.35            # 拟合后 mesh 各轴尺寸之比小于此 → 视为同一物体


def rel_take(rec: dict) -> str:
    t = rec["take"]
    return t.split("/ReconstructOutput/")[-1] if "/ReconstructOutput/" in t else t


def obj_of(rec: dict) -> str:
    """记录评的是哪个物体。改为逐物体打分之前写的记录没有这个字段, 它们评的就是 object_0。"""
    return rec.get("object", "object_0")


def rec_key(rec: dict) -> tuple[str, str]:
    """判定的粒度是 (take, 物体) 而不是 take —— 一条 take 里瓶身落选不该把瓶盖也带下水。"""
    return (rel_take(rec), obj_of(rec))


def art_suffix(rec: dict) -> str:
    """单物体 take 的 rts/cc 文件名不带物体后缀(存量文件如此), 多物体才带。"""
    return f"_{obj_of(rec)}" if int(rec.get("n_objects", 1)) > 1 else ""


def scaled_extents(rec: dict):
    e = rec.get("mesh_extents_m")
    if not e:
        return None
    s = rec.get("mesh_scale_fitted") or 1.0
    return sorted(v * s for v in e)


def same_object(a: dict, b: dict) -> bool:
    ea, eb = scaled_extents(a), scaled_extents(b)
    if ea is None or eb is None:
        return False               # 尺寸未知不敢合并 → 各自保留, 宁可漏合不错杀
    if not (a.get("scale_reliable") and b.get("scale_reliable")):
        # 任一侧尺度拟合不可靠 → 绝对尺寸没意义, 退回比形状(长轴归一, 尺度不变量)。
        # 实证: 12_scene_v17aframes 尺度顶上限 1.6, 但归一后 [0.27,0.29,1.0] 与
        # 12_scene [0.26,0.27,1.0] 同形 → 同一瓶子的两次重建。
        ea = [v / ea[-1] for v in ea]
        eb = [v / eb[-1] for v in eb]
    return all(max(x, y) / max(min(x, y), 1e-6) < SAME_OBJ_RATIO for x, y in zip(ea, eb))


def root_of(rec: dict) -> str:
    return rel_take(rec).strip("/").split("/")[0]


def take_tail(rec: dict) -> str:
    """去掉输出根的 take 标识, 用来认出"同一条视频在两代流水线下各建了一次"。"""
    return "/".join(rel_take(rec).strip("/").split("/")[-2:])


def superseded(recs: list[dict]) -> dict[tuple[str, str], str]:
    """老输出根的记录被新根同名 take 取代。返回 {被取代的 rec_key: 取代它的 take}。

    只在 ROOT_RANK 里有明确先后时判定; 同级或未知一律不判(宁可留两行也不误杀真数据)。
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in recs:
        groups.setdefault((take_tail(r), obj_of(r)), []).append(r)
    out: dict[tuple[str, str], str] = {}
    for grp in groups.values():
        if len(grp) < 2:
            continue
        best = max(ROOT_RANK.get(root_of(r), 0) for r in grp)
        for r in grp:
            if ROOT_RANK.get(root_of(r), 0) < best:
                out[rec_key(r)] = next(x for x in grp
                                       if ROOT_RANK.get(root_of(x), 0) == best)["take"]
    return out


def auto_deselect(recs: list[dict]) -> dict[tuple[str, str], str]:
    """同视频同物体组内 conf 择优。返回 {落选take: 胜者take}。"""
    groups: dict[tuple, list[dict]] = {}
    for r in recs:
        rel = rel_take(r)
        parent = rel.rsplit("/", 1)[0]
        name = rel.rsplit("/", 1)[-1]
        prefix = name.split("_")[0]
        groups.setdefault((parent, prefix), []).append(r)
    out: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        pool = list(members)
        while pool:                       # 贪心聚同物体簇
            seedr = pool.pop(0)
            # ★ 同一条 take 里的两个物体**永远不合并** (2026-08-17 修)。
            #   本分组键是 (父目录, take名前缀), 为**旧布局**设计 —— 那时一条视频产出
            #   2_bottle / 2_cap / 2_scene 三个 take, 同组=同一视频的多次重建, 该择优。
            #   新的逐物体布局下, egodex_auto/pour/5 一条 take 内就装着 object_0/object_1,
            #   两条 audit 记录 rel_take 相同 → 落进同一组 → 尺寸相近就被判"同一物体"
            #   丢掉一个。而 pour 恰恰是双物体任务(瓶+杯), 尺寸本来就接近。
            #   实测 8 条被误 deselect 的, 两物体世界轨迹中位间距 160~406mm,
            #   远大于物体自身尺寸(7~25cm) —— 物理上就是两个东西。
            #   ⚠ 若上游真把同一物体注册成两个实例, 那是标注侧的 bug, 要在那儿修,
            #     不该在这里靠丢数据掩盖。
            cluster = [seedr] + [r for r in pool
                                 if rel_take(r) != rel_take(seedr) and same_object(seedr, r)]
            pool = [r for r in pool if r not in cluster]
            if len(cluster) < 2:
                continue
            cluster.sort(key=lambda r: (r["conf_pos_median"], r["conf_rot_median"]),
                         reverse=True)
            winner = rel_take(cluster[0])
            for loser in cluster[1:]:
                out[rec_key(loser)] = winner
    return out


def verdict(rec: dict, auto_losers: dict[tuple[str, str], str],
            old_roots: dict[tuple[str, str], str] | None = None) -> dict:
    rel = rel_take(rec)
    key = rec_key(rec)
    sfx = art_suffix(rec)
    n = max(rec["n_scored"], 1)
    refuted = sum(1 for r in rec["per_frame"] if r.get("rot_refuted"))
    obs = rec.get("rot_observability") or []
    free_axes = [i for i, v in enumerate(obs) if v < ROT_OBS_FREE]
    cp, cr = rec["conf_pos_median"], rec["conf_rot_median"]

    if (old_roots or {}).get(key):
        status, why = "superseded", f"旧输出根, 已被同名 take 取代: {old_roots[key]}"
    elif rel in EXCLUDE_TRANSPARENT:
        status, why = "excluded", "空透明物体 (FP 系统性失灵, 存量; v2:装深色液体的透明容器不过滤)"
    elif rel in DESELECTED:
        status, why = "deselected", f"同视频有更优重建(人工): {DESELECTED[rel].split('/')[-1]}"
    elif key in auto_losers:
        status, why = "deselected", f"同视频有更优重建(自动): {auto_losers[key].split('/')[-1]}"
    else:
        status, why = "active", ""

    rot_ok = status == "active" and cr >= ROT_CONF_MIN and refuted / n <= ROT_REFUTED_MAX
    return dict(
        take=rel, object=obj_of(rec), status=status, reason=why, tuned=rel in TUNED,
        n_frames=rec["n_frames"], conf_pos_median=cp, conf_rot_median=cr,
        refuted_frames=refuted, scale_reliable=rec.get("scale_reliable"),
        position_grade=("good" if cp >= 70 else "mixed" if cp >= 40 else "poor"),
        # ★ 三档(ARCTIC 真值标定)。与 rotation_usable 并存: 后者是存量口径, 不动它,
        #   下游可自行选用。★ 是排序不是合格证 —— good 档约 1/7 误报, 且仅 take 级有效。
        rotation_grade=("unusable" if status != "active" else
                        "good" if (cr or 0) >= ROT_GRADE_GOOD else
                        "mixed" if (cr or 0) >= ROT_GRADE_MIXED else "poor"),
        rotation_usable=bool(rot_ok),
        rotation_free_axes=free_axes,     # 索引对应 mesh 主轴 (rot_observability 顺序)
        rot_observability=[round(float(v), 3) for v in obs],
        rts_npz=f"rts/rts_{'_'.join(rel.split('/')[-2:])}{sfx}.npz",
        cc_json=f"cc/cc_{'_'.join(rel.split('/')[-2:])}{sfx}.json",
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    audit = json.loads(a.audit.read_text())
    recs = [r for r in audit["takes"] if "error" not in r]
    skipped = len(audit["takes"]) - len(recs)
    old_roots = superseded(recs)
    losers = auto_deselect([r for r in recs if rel_take(r) not in EXCLUDE_TRANSPARENT
                            and rec_key(r) not in old_roots])
    rows = [verdict(r, losers, old_roots) for r in recs]
    rows.sort(key=lambda r: (r["status"] != "active", -(r["conf_pos_median"] or 0)))
    n_act = sum(1 for r in rows if r["status"] == "active")
    summ = dict(
        n_takes=len(rows), n_active=n_act, n_audit_errors=skipped,
        n_excluded=sum(1 for r in rows if r["status"] == "excluded"),
        n_deselected=sum(1 for r in rows if r["status"] == "deselected"),
        n_superseded=sum(1 for r in rows if r["status"] == "superseded"),
        n_rot_good=sum(1 for r in rows if r["rotation_grade"] == "good"),
        n_rot_mixed=sum(1 for r in rows if r["rotation_grade"] == "mixed"),
        n_rotation_usable=sum(1 for r in rows if r["rotation_usable"]),
        rules=dict(rot_obs_free=ROT_OBS_FREE, rot_conf_min=ROT_CONF_MIN,
                   rot_refuted_max=ROT_REFUTED_MAX, same_obj_ratio=SAME_OBJ_RATIO,
                   rot_grade_good=ROT_GRADE_GOOD, rot_grade_mixed=ROT_GRADE_MIXED),
        # ★ 标签的可信度必须跟标签一起走, 否则下游只会看见 "good" 三个字母
        rotation_grade_caveats=dict(
            calibration=ROT_GRADE_CALIB,
            gt_median_deg={"good": 20.4, "mixed": 34.9, "poor": 103.0},
            gt_p75_deg={"good": 23.6, "mixed": 86.2, "poor": 128.0},
            false_pass_rate="good 档 7 条里 1 条真实误差 113° (~1/7)",
            false_fail_rate="poor 档 30 条里 4 条真实仅 9~12°",
            take_level_only=("仅 take 级有效; 留一验证仅 37% 的 take 内部单调, "
                             "take 内逐帧相关中位 -0.051(方向对仅 55%)"),
            per_frame_rule=("已用 ARCTIC 真值验证成立(45 takes): 低分帧每帧多错 +2.0°, 基线 3.69°/帧; ★必须用**增量**误差验, 用累计误差会误判为无效"),
            object_class_transfer="标定物体全为 ARCTIC 大件, 瓶杯类是否可搬**未验证**"),
        takes=rows)
    with poseqa_lock(a.out.parent):
        a.out.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[manifest] {len(rows)} takes: active {n_act} "
          f"(旋转可用 {summ['n_rotation_usable']}), excluded {summ['n_excluded']}, "
          f"deselected {summ['n_deselected']}, superseded {summ['n_superseded']}"
          f"{f', audit失败 {skipped}' if skipped else ''}")
    for r in rows:
        mark = {"active": " ", "excluded": "✗", "deselected": "−", "superseded": "↑"}[r["status"]]
        print(f" {mark} {r['take']:50s} pos {str(r['conf_pos_median']):>5s}({r['position_grade']:5s}) "
              f"rot {str(r['conf_rot_median']):>5s}({'可用' if r['rotation_usable'] else '弃用'}) "
              f"自由轴{r['rotation_free_axes']}"
              f"{'  [调参take]' if r['tuned'] else ''}  {r['reason']}")
    print(f"[manifest] 写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
