#!/usr/bin/env python3
"""三方尺度对比:原版 sam3d_scale vs 手锚点护栏修正版 vs ARCTIC GT。

GT = ARCTIC object_vtemplates/<obj>/mesh.obj(毫米,÷1000 转米),口径 = 包围盒最长边。
原版 = sam3d_scale 输出 mesh(优先 final,缺则 stage1——二者几何相同)的最长边。
护栏版 = scale_guardrail/<object_id>.json 的 L_final_m。
用法: python compare_scale_gt.py   (无参,固定三条视频)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SD = Path(__file__).resolve().parent
RECON_INTERIM = SD.parent / "ego_pipeline/Reconstruction/data/interim/arctic"
GT_ROOT = Path("/media/msc-auto/HDD/dataset/arctic/arctic_repo/unpack/raw/meta/object_vtemplates")
VIDEOS = ["ketchup", "laptop", "microwave"]


def longest_extent(obj_path: Path, unit_scale: float = 1.0) -> float:
    vs = []
    with open(obj_path) as fh:
        for line in fh:
            if line.startswith("v "):
                vs.append([float(x) for x in line.split()[1:4]])
    v = np.asarray(vs) * unit_scale
    return float((v.max(0) - v.min(0)).max())


def main() -> None:
    rows = []
    for s in VIDEOS:
        vid = f"s01_{s}_grab_01"
        gt = longest_extent(GT_ROOT / s / "mesh.obj", 0.001)
        row = {"video": s, "GT_m": round(gt, 3)}

        objdir = RECON_INTERIM / vid / "sam3d_scale/objects/object_0"
        est_mesh = next((p for p in [objdir / "object_mesh_scaled_final.obj",
                                     objdir / "object_mesh_scaled_stage1.obj"] if p.exists()), None)
        if est_mesh:
            L0 = longest_extent(est_mesh)
            row.update(orig_m=round(L0, 3), orig_ratio=round(L0 / gt, 2),
                       orig_src=est_mesh.name)
        guard = list((SD / "runs" / vid / "scale_guardrail").glob("object_*.json")) \
            if (SD / "runs" / vid / "scale_guardrail").exists() else []
        if guard:
            g = json.loads(guard[0].read_text())
            row.update(guard_m=g["L_final_m"], guard_ratio=round(g["L_final_m"] / gt, 2),
                       guard_verdict=g["verdict"],
                       anchor_m=g["L_hand_anchor_m"], band=g["guard_band_m"])
        fuse = list((SD / "runs" / vid / "scale_fuse").glob("object_*.json")) \
            if (SD / "runs" / vid / "scale_fuse").exists() else []
        if fuse:
            f = json.loads(fuse[0].read_text())
            row.update(fuse_m=f["L_final_m"], fuse_ratio=round(f["L_final_m"] / gt, 2),
                       fuse_verdict=f["verdict"])
        rows.append(row)

    print(f"{'video':<12}{'GT(m)':>7}{'原版(m)':>9}{'原/GT':>7}{'护栏(m)':>9}{'护栏/GT':>8}"
          f"{'融合(m)':>9}{'融合/GT':>8}  verdict")
    for r in rows:
        print(f"{r['video']:<12}{r['GT_m']:>7}"
              f"{r.get('orig_m', '—'):>9}{r.get('orig_ratio', '—'):>7}"
              f"{r.get('guard_m', '—'):>9}{r.get('guard_ratio', '—'):>8}"
              f"{r.get('fuse_m', '—'):>9}{r.get('fuse_ratio', '—'):>8}"
              f"  {r.get('fuse_verdict', r.get('guard_verdict', '—'))}")
    out = SD / "runs" / "scale_comparison.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print("->", out)


if __name__ == "__main__":
    main()
