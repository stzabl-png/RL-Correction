"""Build the Sweep2 dustpan with its reconstructed raised mouth lip removed."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import trimesh


p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--out", required=True)
p.add_argument("--report", required=True)
args = p.parse_args()

source = os.path.abspath(args.source)
out = os.path.abspath(args.out)
report_path = os.path.abspath(args.report)
mesh = trimesh.load(source, force="mesh", process=False)
v0 = np.asarray(mesh.vertices, dtype=np.float64).copy()
faces0 = np.asarray(mesh.faces).copy()
components0 = len(mesh.split(only_watertight=False))
watertight0 = bool(mesh.is_watertight)

# Asset axes from the reconstruction: x=pan width, +y=work-plane normal,
# +z=basin-to-open-mouth.  The basin top is ~8.5 mm.  The scan rises again to
# ~14 mm over the final 20 mm, which is the false lip visible in v1.
z_start, z_end = 0.080, 0.108
core_half, blend_half = 0.050, 0.060
top_start, top_end = 0.0085, 0.0060

v = v0.copy()
z_alpha = np.clip((v[:, 2] - z_start) / (z_end - z_start), 0.0, 1.0)
top_profile = top_start + z_alpha * (top_end - top_start)
abs_x = np.abs(v[:, 0])
lateral_weight = np.clip((blend_half - abs_x) / (blend_half - core_half), 0.0, 1.0)
eligible = ((v[:, 2] >= z_start) & (abs_x < blend_half)
            & (v[:, 1] > top_profile))
# Core vertices land on the smooth profile; the last 10 mm blends continuously
# back into the unchanged side walls, avoiding a new lateral ridge.
v[eligible, 1] -= (v[eligible, 1] - top_profile[eligible]) * lateral_weight[eligible]
mesh.vertices = v
mesh.remove_unreferenced_vertices()

assert len(mesh.vertices) == len(v0), (len(mesh.vertices), len(v0))
assert np.array_equal(np.asarray(mesh.faces), faces0)
assert bool(mesh.is_watertight) == watertight0
assert len(mesh.split(only_watertight=False)) == components0
changed = np.linalg.norm(v - v0, axis=1) > 1.0e-10
assert np.all(v[~changed] == v0[~changed])
assert np.all((v0[changed, 2] >= z_start) & (np.abs(v0[changed, 0]) < blend_half))

def central_max_y(vertices: np.ndarray, zlo: float, zhi: float) -> float:
    q = vertices[(np.abs(vertices[:, 0]) <= core_half)
                 & (vertices[:, 2] >= zlo) & (vertices[:, 2] <= zhi), 1]
    assert len(q)
    return float(q.max())


before_mouth_max = central_max_y(v0, 0.098, 0.1085)
after_mouth_max = central_max_y(v, 0.098, 0.1085)
assert before_mouth_max > 0.012
assert after_mouth_max <= 0.0070, after_mouth_max

report = {
    "source": source,
    "output": out,
    "vertices": int(len(v)),
    "faces": int(len(mesh.faces)),
    "components": int(components0),
    "watertight": bool(mesh.is_watertight),
    "changed_vertices": int(changed.sum()),
    "max_vertex_displacement_mm": float(np.linalg.norm(v-v0, axis=1).max()*1000),
    "central_mouth_max_y_before_mm": before_mouth_max*1000,
    "central_mouth_max_y_after_mm": after_mouth_max*1000,
    "z_start_mm": z_start*1000,
    "z_end_mm": z_end*1000,
    "core_half_width_mm": core_half*1000,
    "blend_half_width_mm": blend_half*1000,
    "top_profile_start_mm": top_start*1000,
    "top_profile_end_mm": top_end*1000,
}

os.makedirs(os.path.dirname(out), exist_ok=True)
os.makedirs(os.path.dirname(report_path), exist_ok=True)
mesh.export(out, file_type="obj", include_normals=True)
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
