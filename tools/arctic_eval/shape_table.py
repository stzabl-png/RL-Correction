#!/usr/bin/env python3
"""汇总表：重建形状 vs ARCTIC 真值 + 现有 confidence + 真实误差。

形状用 **PCA 对齐 + 按最长轴归一化** 的轴比 —— SAM3D 输出的朝向是任意的，
轴对齐 bbox 会随朝向变化，比出来没有意义；归一化后剩下的才是纯形状。

重点看**最短轴**：laptop / ketchup 上实测误差都集中在那一轴（长宽 <3%，短轴 +100% 以上），
即误差是**各向异性**的 —— 任何单标量尺度都修不了。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

RR = Path(__file__).resolve().parents[2]
GT = Path("/media/lyh/DATA2/arctic/repo/data/arctic_data/data/meta/object_vtemplates")
ROWS = RR / "Output" / "arctic_eval" / "rows15"


def axis_ratio(p: Path) -> np.ndarray | None:
    if not p.is_file():
        return None
    V = np.asarray(trimesh.load(p, force="mesh").vertices, float)
    V = V - V.mean(0)
    _, _, Vt = np.linalg.svd(V, full_matrices=False)
    e = np.sort((V @ Vt.T).ptp(0))[::-1]
    return e / e[0]


def main() -> int:
    takes = sorted((RR / "Output/ReconstructOutput/arctic15").glob("*/*"))
    if not takes:
        print("(还没有 take)")
        return 0
    print(f"\n{'物体':16}{'标注帧':>7}{'轴比(我们)':>22}{'轴比(真值)':>22}"
          f"{'短轴偏差':>10}{'confP':>7}{'confR':>7}{'grade':>8}")
    for t in takes:
        obj = t.name.split("_")[0]
        ours = axis_ratio(t / "objects/object_0/object_mesh_scaled_final.obj")
        gt = axis_ratio(GT / obj / "mesh.obj")
        if ours is None or gt is None:
            continue
        lp = t / "sam2_object" / "label_prompt.json"
        fi = "-"
        row = ROWS / f"{t.parent.name}__{t.name}.json"
        conf = json.loads((t / "confidence_complete.json").read_text()) \
            if (t / "confidence_complete.json").is_file() else {}
        if row.is_file():
            try:
                fi = json.loads(row.read_text()).get("label_frame", "-")
            except Exception:
                pass
        dev = (ours[2] / gt[2] - 1) * 100
        print(f"{obj:16}{str(fi):>7}"
              f"{'  '.join(f'{v:.3f}' for v in ours):>22}"
              f"{'  '.join(f'{v:.3f}' for v in gt):>22}"
              f"{dev:+9.0f}%"
              f"{conf.get('conf_pos_median','-'):>7}{conf.get('conf_rot_median','-'):>7}"
              f"{str(conf.get('position_grade','-')):>8}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
