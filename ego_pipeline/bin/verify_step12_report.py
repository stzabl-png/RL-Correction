#!/usr/bin/env python3
"""前两步(HOI 检测 / VLM 门)的验收对照表。只读产物, 不跑任何模型。

一行一条视频, 让人一眼看出**哪条能进重建、哪条要走资产库、哪条根本没判过**:

    take                检测  实例  评审视频  材质(剔除)      分件          结论
    ...__0              1832  2     ✓        opaque/opaque   combines      走资产库
    ...__3              1104  1     ✓        transp_empty(1) none          剔除
    ...__7               980  1     ✓        未判定           未判定         ⚠ 未判定

★ "未判定"必须和"判过了没问题"分开显示。VLM 服务不可达时如果只是静默放行, 透明物体会混进
  训练集而没人知道 —— 这条区分是本报告存在的主要理由。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))


def probe_dir(dataset: str, vid: str) -> Path:
    return RR / "experimental/hoi_detr_v17a/data/interim" / dataset / vid / "hoi_detr_probe"


def row(dataset: str, vid: str) -> dict:
    from vlm_transparency_gate import gate_path                       # noqa: E402
    pd = probe_dir(dataset, vid)
    r = {"take": vid, "n_det": None, "n_inst": None, "vis": pd / "vis" / f"{vid}.mp4",
         "materials": [], "n_filter": 0, "part": None, "gate_status": None}
    pc = pd / "hoi_detr_probe_complete.json"
    if pc.is_file():
        d = json.loads(pc.read_text())
        r["n_det"] = d.get("num_detections")
        r["n_hf"] = d.get("num_hf_links")
    gp = gate_path(dataset, vid)
    if gp.is_file():
        g = json.loads(gp.read_text())
        r["gate_status"] = g.get("status")
        objs = g.get("objects") or {}
        r["n_inst"] = len(objs)
        r["materials"] = [v.get("material", v.get("error", "?")) for v in objs.values()]
        r["n_filter"] = sum(1 for v in objs.values() if v.get("filter"))
        r["part"] = (g.get("part_change") or {}).get("part_change")
        r["needs_retrieval"] = (g.get("part_change") or {}).get("needs_retrieval")
    return r


def verdict(r: dict) -> str:
    """结论只讲**管线会怎么处理这条**。第①步缺产物是另一个维度, 不该把第②步的结论盖掉 ——
    实测 take0 的 hoi_detr_probe 被清理过但 vlm_gate 判得好好的, 早先版本把它报成
    "没产物", 等于把一条已经判明"走资产库"的数据显示成完全没跑。"""
    if r["n_det"] is None and r["gate_status"] != "ok":
        return "X 两步都没产物"
    if r["gate_status"] != "ok":
        return "⚠ 未判定(VLM 没跑成)"          # ← 决不能和"判过了没事"混为一谈
    if r["n_inst"] and r["n_filter"] >= r["n_inst"]:
        return "剔除(全部实例透明)"
    pre = "部分剔除 + " if r["n_filter"] else ""
    tail = "  (①产物已清理)" if r["n_det"] is None else ""
    return pre + ("走资产库" if r.get("needs_retrieval") else "正常重建") + tail


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("videos", nargs="+", type=Path)
    a = ap.parse_args(argv)

    from auto_label_v17a import video_id_for                          # noqa: E402
    rows = [row(a.dataset, video_id_for(v.resolve(), a.root)) for v in a.videos if v.is_file()]
    if not rows:
        print("[verify12] 没有可报告的视频")
        return 1

    w = max(len(r["take"]) for r in rows)
    print(f"\n{'take':<{w}}  {'检测':>6} {'实例':>4}  {'评审视频':<8} "
          f"{'材质(剔除)':<26} {'分件':<12} 结论")
    print("-" * (w + 78))
    for r in rows:
        mats = ",".join(r["materials"])[:22] or "未判定"
        if r["n_filter"]:
            mats += f"({r['n_filter']})"
        print(f"{r['take']:<{w}}  {str(r['n_det'] or '-'):>6} {str(r['n_inst'] or '-'):>4}  "
              f"{'✓' if Path(r['vis']).is_file() else '-':<8} {mats:<26} "
              f"{str(r['part'] or '未判定'):<12} {verdict(r)}")

    n_undecided = sum(1 for r in rows if r["gate_status"] != "ok")
    n_drop = sum(1 for r in rows if r["n_inst"] and r["n_filter"] >= r["n_inst"])
    n_ret = sum(1 for r in rows if r.get("needs_retrieval"))
    print(f"\n共 {len(rows)} 条: 正常重建 {len(rows)-n_drop-n_undecided-n_ret}, "
          f"走资产库 {n_ret}, 剔除 {n_drop}, **未判定 {n_undecided}**")
    if n_undecided:
        print("⚠ 未判定的不要当成'没问题'放进训练 —— 先把 VLM 服务开起来重跑:")
        print("    ./tools/vlm_tunnel.sh  然后 ./ego_pipeline/verify_step12.sh ... --force")
    if rows and Path(rows[0]["vis"]).is_file():
        print(f"\n评审视频(逐帧手/物框 + HF/FS 连线): {rows[0]['vis'].parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
