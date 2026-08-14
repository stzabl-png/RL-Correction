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
# 模板手指跨距(mm) = 模板接触点两两最大距离, 由 assets/hand/*/init_tmpl/*.npy 量出。
# 用来判"这个物体这只手根本抓不抓得住" —— pour/17 实测: 杯子接触带直径 137~153mm,
# 而选择器给了跨距 58mm 的捏取模板, 合成器只能去勾杯沿(47 个候选全是这个模式)。
# ⚠ 上限按族给, 不能取全库最大值: 跨距最大的 17_Index_Finger_Extension(147mm) 是
#   伸展/钩状, 接触点铺开但**不构成包围**, 不该算进"能抓多大"。
# 各模板实测跨距(mm), 由 Dexonomy/assets/hand/sharpa_wave/init_tmpl/*.npy 量出并落盘
try:
    TEMPLATE_SPAN_MM = json.loads((HERE / "template_spans.json").read_text())
except Exception:
    TEMPLATE_SPAN_MM = {}

WRAP_MAX_MM = 126.0        # 环握族上限: 30_Palmar 126 / 1_Large_Diameter 114 / 3_Medium_Wrap 105
PINCH_MAX_MM = 76.0        # 捏取族上限: 6_Prismatic_4_Finger 76 / 7_Prismatic_3 58 / 8_Prismatic_2 48

DEPTH_COMPAT = {"fingertip": {"tip", "pad"}, "pad": {"pad", "tip", "full"},
                "whole_finger": {"full"}, "none": {"tip", "pad", "full"}}


def load_v2_prompt(take: Path) -> dict:
    """contact v2 的交接件 grasp_prompt.json -> {accepted:{(oid,side):rec}, rejected:{(oid,side):why}}。

    v2 判"是不是抓握"用的是 稳定窗口 + 贴合 + 对生度 三条件, 正是 ARCTIC 标定不出来的
    那一维(实测: ARCTIC 110 只手 GT 全部有抓握, 零负样本 -> τ 网格必然选最大值)。
    所以这里只借它的**否决权**与**对生度数值**, 指数仍由 VLM/几何双证人定。"""
    p = take / "contact" / "grasp_prompt.json"
    if not p.is_file():
        return {"accepted": {}, "rejected": {}, "present": False}
    d = json.loads(p.read_text())
    acc = {(g["object_id"], g["hand"]): g for g in d.get("grasps", [])}
    rej = {(r["object_id"], r["hand"]): r.get("why") or r.get("status")
           for r in d.get("rejected", [])}
    return {"accepted": acc, "rejected": rej, "present": True}


def graspable_size(take: Path, side: str) -> dict | None:
    """接触带处的横截面直径(mm) —— 判物体抓不抓得住用的尺寸。

    ★用**接触带处**的截面, 不用整体包围盒: 剪刀整体 20cm 但人抓的指环只有 2cm,
      按整体尺寸判会把所有模板都毙掉。接触带来自 contact v2 的热点高度分布。
    """
    try:
        import trimesh
    except ImportError:
        return None
    cf = take / "contact" / "contact_fingers.json"
    prompt = take / "contact" / "grasp_prompt.json"
    oid = None
    if prompt.is_file():
        for g in json.loads(prompt.read_text()).get("grasps", []):
            if g["hand"] == side:
                oid = g["object_id"]
                break
    if oid is None and cf.is_file():
        oid = (json.loads(cf.read_text())["hands"].get(side) or {}).get("primary")
    if oid is None:
        return None
    mesh_p = next((q for q in (take / "objects" / oid / "object_mesh_scaled_final.obj",
                               take / "object_mesh_scaled_final.obj") if q.is_file()), None)
    hot_p = next((q for q in take.glob(f"contact/contact_v2_{oid}_{side}.npz")), None)
    if mesh_p is None:
        return None
    m = trimesh.load(mesh_p, force="mesh", process=False)
    V = np.asarray(m.vertices)
    ext = m.bounds[1] - m.bounds[0]
    ax = int(np.argmax(ext))
    band = None
    if hot_p is not None:
        z = np.load(hot_p, allow_pickle=True)
        hot = z["probe_local"][z["weight"] >= 0.5]
        if len(hot) > 20:
            band = (float(hot[:, ax].min()), float(hot[:, ax].max()))
    if band is None:                       # 没有接触带就退回整体(标明来源)
        band, src = (float(V[:, ax].min()), float(V[:, ax].max())), "whole_object"
    else:
        src = "contact_band"
    sl = V[(V[:, ax] >= band[0] - 0.004) & (V[:, ax] <= band[1] + 0.004)]
    if len(sl) < 20:
        sl = V
    o = [i for i in range(3) if i != ax]
    dia = max(float(np.ptp(sl[:, o[0]])), float(np.ptp(sl[:, o[1]]))) * 1000
    return {"object": oid, "diameter_mm": round(dia, 1), "source": src,
            "band_frac": round((band[1] - band[0]) / float(ext[ax]), 2)}


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


def rank(table: dict, vlm: dict, geo: dict | None, v2: dict | None = None,
         size: dict | None = None) -> dict:
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
        # precision: 55条真值实测(2026-08-12) VLM 指数 100% vs 几何 88%,
        # 指数目标也交 VLM; 几何降为核验人 —— 一致加严, 分歧放宽并记录(screw27 式
        # VLM 翻车仍会被分歧标记暴露, 不会静默吞掉)
        agree = geo is not None and abs(geo["n"] - v_n) <= 1
        target = v_n
        tol = 1 if agree else 2
        conf = "high" if agree else ("medium" if geo is None else "low")
        if geo is not None and not agree:
            notes.append(f"VLM 指数 {v_n} vs 几何 {geo['n']} (差>1) —— VLM 为准, 候选放宽到 ±2")

    # 第三证人: v2 的对生度(几何数值, 0=接触点全朝同一边夹不住, 1=完全对生)
    v2info = None
    if v2 is not None:
        opp = float((v2.get("trust") or {}).get("opposition", 0))
        v2info = {"opposition": opp,
                  "grasp_window_frames": v2.get("grasp_window_frames"),
                  "contact_region": v2.get("contact_region"),
                  "contact_cloud_npz": v2.get("contact_cloud_npz"),
                  "hand_mask_agreement": (v2.get("trust") or {}).get("hand_mask_agreement"),
                  "object_conf_rot": (v2.get("trust") or {}).get("object_conf_rot")}
        if opp < 0.5:                       # 0.40 以下 v2 已判非抓握; 0.40~0.5 是勉强
            conf = "low"
            notes.append(f"v2 对生度仅 {opp:.2f}(勉强够抓握线) —— 候选放宽, 下游建议多带几个")
            tol = max(tol, 2)
        cr = v2.get("contact_region") or {}
        if (v2.get("trust") or {}).get("object_conf_rot", 99) < 10:
            notes.append("v2: 物体朝向不可信 -> 区域只用高度/半径, 别用方位角")

    # ★尺寸门: 物体在接触带处有多粗, 决定这只手够不够得着 —— 与 VLM 的语义判断无关,
    #   是纯物理约束。pour/17 杯子(接触带直径 137~153mm)配了跨距 58mm 的捏取模板,
    #   合成器只能去勾杯沿, 47 个候选全废。
    size_note = None
    if size is not None:
        dia = size["diameter_mm"]
        if dia > WRAP_MAX_MM:
            size_note = (f"接触带直径 {dia:.0f}mm > 环握族上限 {WRAP_MAX_MM:.0f}mm —— "
                         f"**全库无模板能抓住它**, 只能扶/勾边; 下游别指望闭合抓取")
            conf = "low"
            notes.append(size_note)
        elif regime == "precision" and dia > PINCH_MAX_MM:
            size_note = (f"接触带直径 {dia:.0f}mm > 捏取族上限 {PINCH_MAX_MM:.0f}mm —— "
                         f"VLM 判 precision 但物体捏不住, 改按环握族出候选")
            notes.append(size_note)
            regime = "power_by_size"          # 体制被物理约束改写
            palm_override = True

    palm = bool(vlm["palm_contact"])
    if size_note and "捏取族上限" in size_note:
        palm = None                            # 放开掌部约束, 让环握族进来
    cands = _filter(table, target, tol, palm, vlm["contact_depth"], vlm)
    if size is not None:                       # 硬过滤: 跨距装不下的模板直接剔除
        cands = [c for c in cands
                 if TEMPLATE_SPAN_MM.get(c["template"], 0) >= size["diameter_mm"] * 0.55]
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
            "size": size,
            "v2": v2info,
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

    v2p = load_v2_prompt(a.take)
    plan = {"schema_version": "grasp_template_plan_v2", "take": str(a.take),
            "task_summary": vg.get("task_summary"),
            "v2_prompt_present": v2p["present"], "hands": {}}
    for side in ("left", "right"):
        h = vg[side]
        if h["role"] == "idle" or h["n_contact_fingers"] == 0:
            plan["hands"][side] = {"confidence": "n/a", "candidates": [],
                                   "note": "idle / 无接触, 不需要模板"}
            continue
        # v2 否决: 这只手被判"不是抓握"(没有既稳又贴又对生的窗口) -> 不编模板,
        # 但要写明原因, 让下游能区分"试过不可用"与"没有这条数据"
        rej = [(oid, why) for (oid, sd), why in v2p["rejected"].items() if sd == side]
        acc = {oid: g for (oid, sd), g in v2p["accepted"].items() if sd == side}
        # ★安全阀: v2 的对生度门槛(0.40)在单条 EgoDex 视频上标定, 跨数据集不通用 ——
        #   ARCTIC 实测 102 只手**全部**被该门拒绝(把门开到 0 则 63 只手有稳定窗)。
        #   所以只在"这条 take 里 v2 至少认可过一只手"时才承认它的否决权;
        #   一条都不认可 = 门槛不适用于该数据, 记警告而非否决。
        v2_applicable = bool(v2p["accepted"])
        if v2p["present"] and v2_applicable and not acc and rej:
            plan["hands"][side] = {"confidence": "n/a", "candidates": [],
                                   "rejected_by_v2": [{"object_id": o, "why": w} for o, w in rej],
                                   "note": "contact v2 判定非抓握, 不编模板"}
            continue
        geo = geometric_fingers(a.take, side)
        v2_note = None
        if v2p["present"] and not v2_applicable and rej:
            v2_note = [{"object_id": o, "why": w, "honored": False,
                        "reason": "v2 在本 take 零认可, 判为门槛不适用, 仅告警"} for o, w in rej]
        v2rec = None
        if acc:
            oid = (geo or {}).get("object") or sorted(acc)[0]
            v2rec = acc.get(oid) or acc[sorted(acc)[0]]
        entry = rank(table, h, geo, v2rec, graspable_size(a.take, side))
        if v2_note:
            entry["v2_warnings"] = v2_note
        plan["hands"][side] = entry

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
