#!/usr/bin/env python3
"""在每条重建的输出目录里写一份 `PROVENANCE.md` —— **哪些是重建出来的, 哪些是数据集直接给的**。

为什么必须有：EgoDex 变体会用设备数据顶替我们本来要估的量（相机/内参/重力/手），
产出的 `world_fused.npz` 与主 pipeline **格式完全一样但性质不同**。
没有这份说明，几周后没人能分清一条数据是"全自动重建"还是"部分开挂"，
拿它得出的任何结论（尤其是精度）都会被质疑。

判定方式：**读实际落盘的产物**（`egodex_source.json` / `run_meta.json` / `vlm_gate.json` /
`traj_sigma.json` 等），不靠命令行参数猜 —— 参数可能与实际跑的不一致。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def jload(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _retrieval(itm):
    return jload(itm / "retrieval.json") or {}


def _mesh_row(itm, row):
    """★ 物体网格可能来自资产库检索而非重建 —— 写错会让人以为 CAD 是我们重建的。
    判据是 interim/retrieval.json(由 retrieve_assets.py 落盘), 不靠猜。"""
    r = _retrieval(itm)
    if r.get("status") != "ok":
        return row("物体网格", "**重建**", "SAM3D 单帧重建")
    asg = r.get("assignment") or []
    d = "; ".join(f"{a.get('object_id')}={a.get('part')}"
                  f"({'×'.join(str(x) for x in a.get('extent_cm', []))}cm)" for a in asg)
    return row("物体网格", "资产库检索(**非重建**)",
               f"任务 `{r.get('task')}` 命中分件 CAD: {d}。分配依据: {r.get('assign_method')}。"
               f"触发: VLM 判 part_change={r.get('vlm_part_change')} parts={r.get('vlm_parts')}")


def _scale_row(itm, row):
    r = _retrieval(itm)
    if r.get("status") != "ok":
        return row("物体尺度", "**重建**", "sam3d_scale（单帧深度 + 单帧 FoundationPose）")
    return row("物体尺度", "资产库给定",
               "CAD 本身米制，**跳过 sam3d / sam3d_scale** —— 不做尺度估计")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-dir", type=Path, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--variant", default="dev")
    ap.add_argument("--interim-root", type=Path, required=True)
    a = ap.parse_args()

    fin, itm = a.final_dir, a.interim_root / a.dataset / a.video_id
    src = jload(itm / "egodex_source.json") or {}
    hawor_meta = next((jload(p) for p in (itm / "hawor").glob("*/run_meta.json")), None) or {}
    vlm = jload(itm / "vlm_gate.json") or {}
    sig = jload(fin / "traj_sigma.json") or {}
    conf = jload(fin / "confidence_complete.json") or {}
    ca = jload(fin / "contact_auto.json") or {}

    injected = bool(src)
    hand_from_device = hawor_meta.get("hand_source") == "egodex_arkit"

    n = "?"
    wf = fin / "world_fused.npz"
    if wf.is_file():
        try:
            with np.load(wf, allow_pickle=True) as z:
                n = int(len(z["c2w"]))
        except Exception:
            pass

    def row(item, source, detail):
        return f"| {item} | **{source}** | {detail} |"

    lines = [
        f"# 数据来源 · {a.dataset}/{a.video_id}",
        "",
        f"帧数 {n} · variant `{a.variant}` · 生成于本目录的 `world_fused.npz`",
        "",
        "**⚠ 本条数据不是纯重建。** 下表逐项说明每个量是我们估出来的，还是数据集直接给的。",
        "拿它做精度结论前请先看这张表。",
        "",
        "| 内容 | 来源 | 说明 |",
        "|---|---|---|",
    ]

    if injected:
        ds = src.get("depth_scale")
        res = src.get("depth_scale_residual_m")
        lines += [
            row("相机位姿 c2w", "数据集给定", "EgoDex `transforms/camera`（设备 SLAM），非 ViPE 估计"),
            row("相机内参 K", "数据集给定", "EgoDex `camera/intrinsic`（设备标定），非 ViPE 估计"),
            row("世界系 / 重力", "数据集给定",
                "EgoDex 本身重力对齐米制，仅做 Y-up→Z-up 换轴；原点在地面而非首帧相机"),
            row("深度", "**重建**",
                f"ViPE 估计，再按 depth_scale={ds if ds else '?'} 缩放到真实米制"
                + (f"（相机对齐残差 {res*1000:.1f}mm）" if isinstance(res, (int, float)) else "")),
        ]
    else:
        lines += [
            row("相机位姿 c2w", "**重建**", "ViPE 估计"),
            row("相机内参 K", "**重建**", "ViPE 估计"),
            row("世界系 / 重力", "**重建**", "ViPE 估重力 + fuse 做 xy 对齐"),
            row("深度", "**重建**", "ViPE 估计"),
        ]

    if hand_from_device:
        vf = hawor_meta.get("valid_fraction") or {}
        lines.append(row("双手轨迹", "数据集给定",
                         "EgoDex ARKit 手部追踪（腕位）；有效帧比例 "
                         + " ".join(f"{k} {v:.0%}" for k, v in vf.items())
                         + " ⚠ 手指 45 维为**零占位**，ARKit 25 关节→MANO 映射尚未实现"))
    else:
        lines.append(row("双手轨迹", "**重建**",
                         "HaWoR 估计（ARCTIC mocap 实测腕位锚后误差 74mm）"))

    lines += [
        row("物体 mask", "**重建**", "HOI-DETR 找交互 + SAM2 传播（v17A 全自动）"),
        _mesh_row(itm, row),
        _scale_row(itm, row),
        row("物体位姿", "**重建**", "FoundationPose 逐帧跟踪"),
        row("接触区间", "**重建**",
            "mask 重叠检测" + (f"；已产出 contact_auto.json" if ca else "；未产出")),
    ]

    if vlm.get("status") == "ok":
        for oid, v in (vlm.get("objects") or {}).items():
            lines.append(row(f"材质判定 {oid}", "VLM 推断",
                             f"{v.get('verdict','?')} conf={v.get('confidence','?')} "
                             f"建议过滤={v.get('filter')}（只记录，未删数据）"))
    elif vlm:
        lines.append(row("材质判定", "缺失", f"跳过：{vlm.get('reason','?')}"))

    o0 = next(iter((sig.get("objects") or {}).values()), {})
    if o0:
        lines.append(row("轨迹可信度 σ", "**估计**",
                         f"锚点 f{sig.get('anchor_frame')}（{sig.get('anchor_source')}）；"
                         f"σ 中位 {o0.get('sigma_fused_mm_median')}mm p90 {o0.get('sigma_fused_mm_p90')}mm；"
                         f"定标 {sig.get('calibration','?')}"))
    if conf:
        lines.append(row("conf_pos / conf_rot", "**估计**",
                         f"{conf.get('conf_pos_median','?')} / {conf.get('conf_rot_median','?')}"
                         f"，判定 {conf.get('position_grade','?')}"))

    lines += [
        "",
        "## 不在本条数据里的东西",
        "",
        (f"* **Retrieval（资产库检索）** —— 本条**已使用**资产库分件 CAD，见上表『物体网格』行。"
         f"资产库在 `ego_pipeline/Retargeting/assets/retrieval/`，索引 `registry.json`。"
         f"目前按任务名硬指定；按外观/几何匹配的真检索尚未实现。"
         if _retrieval(itm).get("status") == "ok" else
         f"* **Retrieval（资产库检索）** —— 本条**未使用**资产（物体网格是 SAM3D 重建的）。"
         f"原因: {_retrieval(itm).get('reason', '未运行 retrieve_assets')}。"
         f"资产库在 `ego_pipeline/Retargeting/assets/retrieval/`。"),
        "* **物体真值** —— EgoDex 只给物体的文字名（`llm_objects`），没有位姿/网格/尺寸。"
        "物体那条线的精度只能靠 ARCTIC 验。",
        "",
        f"任务描述：{(src.get('task_attrs') or {}).get('llm_description', '(无)')}",
        f"物体（LLM 标注）：{(src.get('task_attrs') or {}).get('llm_objects', '(无)')}",
    ]

    (fin / "PROVENANCE.md").write_text("\n".join(lines) + "\n")
    print(f"[provenance] -> {fin / 'PROVENANCE.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
