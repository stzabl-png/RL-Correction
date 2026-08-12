#!/usr/bin/env python
"""GraspPose 模板自动选择 —— VLM 语义 × 接触几何 双证人交叉, 输出逐手候选模板集。

哲学与 confidence 同源: 不猜, 让两个独立观察互证。
  证人A VLM(schema v3.2, 92.6%): 每手 {角色, 指数, 掌部, 接触深度, 对掌, 物形, 物尺}
  证人B 接触热图(第10步): 逐指垫热顶点数 → 真实接触的指头数(几何,不受语言歧义)
  一致(|Δ指数|≤1) → 高置信, 硬过滤收窄; 不一致 → 低置信, 放宽±2 并把分歧写进输出

匹配规则(v2, 2026-08-11 按"抓握体制"分派权威 —— pour 实战教训):
  ★两个证人测的不是同一个量: VLM 数"看起来几根手指参与", 几何数"**指尖**弹性垫压上几个"。
    power 抓握(整指包握/贴掌)的接触界面是中节指骨+手掌, 指尖垫结构性低计 ——
    "VLM=5+掌 vs 指垫=2~3" 是 power 抓法的标准签名, 不是矛盾。
  体制判定(VLM 的 depth+palm 字段, 它答得稳):
    power     (whole_finger 或 palm=true): 指数以 VLM 为准; 几何当下界校验+拇指硬证据
    precision (fingertip/pad 且 palm=false): 指数以几何为准(指垫=真实界面); VLM ±1
  硬过滤: |digits−目标| ≤ 容差; palm 一致; depth 兼容
  兜底: 空集 → 按"容差+1 → 放开掌部"逐级降级, 每级标 degraded 并记原因
  软排序: 对掌 +2, 物形 +1, 物尺 +1, approx −0.5

输出: <take>/grasp_template_plan.json —— 每手: 候选模板排序 + 分数 + 两证人原始值 + 分歧标记。
下游(Dexonomy 合成 / RL): 高置信取 top1-2, 低置信把候选集全带上。

用法:
  python select_grasp_template.py --take <take目录> [--table template_table.json]
前置: take 里已有 vlm_grasp.json(vlm_grasp_schema.py) 与 contact/(第10步)。
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAD_MIN_VERTS = 150          # 指垫热顶点数低于此视为未接触(pour实测: 真接触 305~4000, 假 0)
DEPTH_COMPAT = {"fingertip": {"tip", "pad"}, "pad": {"pad", "tip", "full"},
                "whole_finger": {"full"}, "none": {"tip", "pad", "full"}}


def geometric_fingers(take: Path, side: str) -> dict | None:
    """几何证人。优先 contact_fingers.json(C1 MANO 全手探头: held-out P0.91/R0.83,
    且能报掌与深度——旧指垫探头结构性测不到); 回退旧指垫热图 {n, pads}。"""
    cf = take / "contact" / "contact_fingers.json"
    if cf.is_file():
        h = json.loads(cf.read_text())["hands"].get(side) or {}
        oid = h.get("primary")
        r = (h.get("objects") or {}).get(oid) if oid else None
        if r and r.get("status") == "ok":
            q = r.get("quality") or {}
            if q.get("reliable") is False:      # 几何证人自报不可靠(烂重建) → 让权威给 VLM
                return None
            return {"n": r["n_fingers"], "pads": r["fingers"], "palm": r["palm"],
                    "depth": r["depth_class"], "object": oid, "source": "mano_full_hand",
                    "grasp_core_frames": r.get("grasp_core_frames")}
    hits = sorted(glob.glob(str(take / "contact" / f"contact_heatmap_frame*_{side}*.npz")))
    if not hits:
        return None
    z = np.load(hits[0], allow_pickle=True)
    w, near = z["vertex_weight"], z["nearest_pad_link"]
    names = [str(x) for x in z["pad_link_names"]]
    hot = w > 0.5
    per_pad = {names[i]: int(((near == i) & hot).sum()) for i in range(len(names))}
    pads = [k.split("_")[1] for k, v in per_pad.items() if v >= PAD_MIN_VERTS]
    return {"n": len(pads), "pads": pads, "per_pad": per_pad, "npz": Path(hits[0]).name}


def _filter(table, target, tol, palm, depth, vlm):
    cands = []
    for name, t in table.items():
        if abs(t["digits"] - target) > tol:
            continue
        if palm is not None and bool(t["palm"]) != palm:
            continue
        if t["depth"] not in DEPTH_COMPAT.get(depth, DEPTH_COMPAT["none"]):
            continue
        score = -abs(t["digits"] - target) * 1.0
        if vlm["opposition"] == t.get("opp"):
            score += 2
        if vlm["object_shape"] in t.get("shapes", []):
            score += 1
        if vlm["object_scale"] in t.get("scales", []):
            score += 1
        if t.get("approx"):
            score -= 0.5
        cands.append({"template": name, "score": round(score, 2),
                      "digits": t["digits"], "palm": t["palm"], "depth": t["depth"]})
    cands.sort(key=lambda c: -c["score"])
    return cands


def rank(table: dict, vlm: dict, geo: dict | None) -> dict:
    v_n = int(vlm["n_contact_fingers"])
    power = vlm["contact_depth"] == "whole_finger" or bool(vlm["palm_contact"])
    regime = "power" if power else "precision"
    notes = []
    if power:
        # power: 接触界面=中节指骨+掌, 指尖垫结构性低计 → 指数以 VLM 为准
        target, tol = v_n, 1
        conf = "high"
        if geo is not None:
            if geo["n"] > v_n + 1:
                conf, tol = "low", 2
                notes.append(f"几何热垫数 {geo['n']} 反超 VLM {v_n} —— power 体制下异常, 放宽")
            elif "thumb" in geo["pads"] and vlm["opposition"] == "fingers_vs_palm":
                notes.append("拇指垫有接触(几何硬证据), 与 fingers_vs_palm 对掌可共存")
        else:
            conf = "medium"
    else:
        # precision: 指尖垫就是真实接触界面 → 指数以几何为准
        agree = geo is not None and abs(geo["n"] - v_n) <= 1
        target = geo["n"] if geo is not None else v_n
        tol = 1 if agree else 2
        conf = "high" if agree else ("medium" if geo is None else "low")
        if geo is not None and not agree:
            notes.append(f"VLM 指数 {v_n} vs 几何 {geo['n']} (差>1) —— 候选放宽到 ±2")

    palm = bool(vlm["palm_contact"])
    cands = _filter(table, target, tol, palm, vlm["contact_depth"], vlm)
    degraded = []
    if not cands:                                    # 兜底: 逐级降级, 全程记账
        degraded.append(f"tol {tol}->{tol + 1}")
        cands = _filter(table, target, tol + 1, palm, vlm["contact_depth"], vlm)
    if not cands:
        degraded.append("放开掌部约束")
        cands = _filter(table, target, tol + 1, None, vlm["contact_depth"], vlm)
    if degraded:
        conf = "low"
    return {"confidence": conf, "regime": regime,
            "target_digits": target, "tolerance": tol, "degraded": degraded or None,
            "vlm": {k: vlm[k] for k in ("role", "target_object", "target_part",
                                        "n_contact_fingers", "palm_contact", "contact_depth",
                                        "opposition", "object_shape", "object_scale")},
            "geometric": geo,
            "notes": notes or None,
            "candidates": cands}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--take", type=Path, required=True)
    ap.add_argument("--table", type=Path, default=HERE / "template_table.json")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    table = json.loads(a.table.read_text())["templates"]
    vg = json.loads((a.take / "vlm_grasp.json").read_text())["answer"]

    plan = {"schema_version": "grasp_template_plan_v1", "take": str(a.take),
            "task_summary": vg.get("task_summary"), "hands": {}}
    for side in ("left", "right"):
        h = vg[side]
        if h["role"] == "idle" or h["n_contact_fingers"] == 0:
            plan["hands"][side] = {"confidence": "n/a", "candidates": [],
                                   "note": "idle / 无接触, 不需要模板"}
            continue
        geo = geometric_fingers(a.take, side)
        plan["hands"][side] = rank(table, h, geo)

    out = a.out or (a.take / "grasp_template_plan.json")
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=1), encoding="utf-8")
    for side, p in plan["hands"].items():
        top = ", ".join(c["template"] for c in p.get("candidates", [])[:3]) or "-"
        extra = "".join(f"  ⚠{n}" for n in (p.get("notes") or []))
        if p.get("degraded"):
            extra += f"  ↓降级:{p['degraded']}"
        print(f"[select] {side}: {p.get('regime','-')}体制 conf={p.get('confidence')} "
              f"目标指数={p.get('target_digits','-')} top3=[{top}]" + extra)
    print(f"[select] -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
