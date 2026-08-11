#!/usr/bin/env python
"""Take 清单 —— 把逐帧打分汇总成 take 级裁决, RL/下游从这里取数据, 不要自己扫目录。

裁决三层:
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
            cluster = [seedr] + [r for r in pool if same_object(seedr, r)]
            pool = [r for r in pool if r not in cluster]
            if len(cluster) < 2:
                continue
            cluster.sort(key=lambda r: (r["conf_pos_median"], r["conf_rot_median"]),
                         reverse=True)
            winner = rel_take(cluster[0])
            for loser in cluster[1:]:
                out[rec_key(loser)] = winner
    return out


def verdict(rec: dict, auto_losers: dict[tuple[str, str], str]) -> dict:
    rel = rel_take(rec)
    key = rec_key(rec)
    sfx = art_suffix(rec)
    n = max(rec["n_scored"], 1)
    refuted = sum(1 for r in rec["per_frame"] if r.get("rot_refuted"))
    obs = rec.get("rot_observability") or []
    free_axes = [i for i, v in enumerate(obs) if v < ROT_OBS_FREE]
    cp, cr = rec["conf_pos_median"], rec["conf_rot_median"]

    if rel in EXCLUDE_TRANSPARENT:
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
    losers = auto_deselect([r for r in recs if rel_take(r) not in EXCLUDE_TRANSPARENT])
    rows = [verdict(r, losers) for r in recs]
    rows.sort(key=lambda r: (r["status"] != "active", -(r["conf_pos_median"] or 0)))
    n_act = sum(1 for r in rows if r["status"] == "active")
    summ = dict(
        n_takes=len(rows), n_active=n_act, n_audit_errors=skipped,
        n_excluded=sum(1 for r in rows if r["status"] == "excluded"),
        n_deselected=sum(1 for r in rows if r["status"] == "deselected"),
        n_rotation_usable=sum(1 for r in rows if r["rotation_usable"]),
        rules=dict(rot_obs_free=ROT_OBS_FREE, rot_conf_min=ROT_CONF_MIN,
                   rot_refuted_max=ROT_REFUTED_MAX, same_obj_ratio=SAME_OBJ_RATIO),
        takes=rows)
    with poseqa_lock(a.out.parent):
        a.out.write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[manifest] {len(rows)} takes: active {n_act} "
          f"(旋转可用 {summ['n_rotation_usable']}), excluded {summ['n_excluded']}, "
          f"deselected {summ['n_deselected']}"
          f"{f', audit失败 {skipped}' if skipped else ''}")
    for r in rows:
        mark = {"active": " ", "excluded": "✗", "deselected": "−"}[r["status"]]
        print(f" {mark} {r['take']:50s} pos {str(r['conf_pos_median']):>5s}({r['position_grade']:5s}) "
              f"rot {str(r['conf_rot_median']):>5s}({'可用' if r['rotation_usable'] else '弃用'}) "
              f"自由轴{r['rotation_free_axes']}"
              f"{'  [调参take]' if r['tuned'] else ''}  {r['reason']}")
    print(f"[manifest] 写入 {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
