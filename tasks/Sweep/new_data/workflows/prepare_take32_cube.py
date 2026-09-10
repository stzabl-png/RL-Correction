"""Select a take9 cube start clear of the entire broom at reset and before contact.

CPU-only preparation. Keeps the accepted tool/joint tracks unchanged. A separating
face of the full broom convex hull certifies conservative cube clearance; brush
surface samples identify the first working-face contact rather than a late minimum.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path('/home/msc-auto/RL_sweep')
RUN = ROOT / 'logs/task1_take32_level_20260910/cube15'
RUN.mkdir(exist_ok=True, parents=True)
src_cfg = ROOT / 'tasks/Sweep/new_data/configs/take32_fixed_level_cube15_base.json'
cfg = json.loads(src_cfg.read_text())
src_ref = ROOT / cfg['reference']
with np.load(src_ref, allow_pickle=True) as data:
    z = {k: data[k].copy() for k in data.files}

def rotations(q):
    return Rotation.from_quat(np.asarray(q)[:, [1, 2, 3, 0]]).as_matrix()

pan_R = rotations(z['obj_quat_0'])
broom_R = rotations(z['obj_quat_1'])
pan_p, broom_p = z['obj_pos_0'], z['obj_pos_1']
mesh = trimesh.load(ROOT / cfg['assets']['broom_mesh'], force='mesh', process=False)
v = np.asarray(mesh.vertices, float)
hull = mesh.convex_hull
hv = np.asarray(hull.vertices)
normals = np.asarray(hull.face_normals)
offsets = -np.einsum('ij,ij->i', normals, hull.triangles[:, 0])
work = v[(v[:, 1] < -.050) & (v[:, 2] > .020) & (v[:, 2] < .090)]
assert len(work) > 2048
half = .0075
# A world-axis cube projects onto the tilted pan axes with these half extents.
projected_half = half * np.abs(pan_R[0]).sum(axis=0)
width = cfg['training_geometry']['pan_half_width']
mouth = cfg['training_geometry']['pan_mouth_z']
local = np.array([[x, 0., depth]
                  for x in np.linspace(-width + projected_half[0] + .004,
                                       width - projected_half[0] - .004, 25)
                  for depth in np.linspace(mouth + projected_half[2] + .006, .245, 45)])
cube_height = .870 + half + .0005
local[:, 1] = (cube_height - (local @ pan_R[0].T + pan_p[0])[:, 2]) / pan_R[0, 2, 1]
candidates = local @ pan_R[0].T + pan_p[0]
N, T = len(candidates), len(broom_p)
cache_path = RUN/'take9_cube_search.npz'
cached = np.load(cache_path) if cache_path.exists() else None
reuse = cached is not None and np.array_equal(cached['candidates'], candidates)
gaps = cached['hull_separation'].copy() if reuse else np.empty((T,N))
brush_box_dist = cached['brush_box_dist'].copy() if reuse else np.empty((T,N))
brush_indices = np.empty((T,N), dtype=int)
for row in ([] if reuse else range(T)):
    nr = normals @ broom_R[row].T
    # Positive value proves separation from the entire convex hull (metres).
    face_gap = (candidates - broom_p[row]) @ nr.T + offsets - half * np.abs(nr).sum(1)
    world_hull = hv @ broom_R[row].T + broom_p[row]
    box_gap = np.maximum(world_hull.min(0) - candidates - half,
                         candidates - half - world_hull.max(0)).max(1)
    gaps[row] = np.maximum(face_gap.max(1), box_gap)
    ww = work @ broom_R[row].T + broom_p[row]
    dist, index = cKDTree(ww).query(candidates, p=np.inf, workers=1)
    brush_box_dist[row] = dist - half
    brush_indices[row] = index
    if row % 50 == 0:
        print(f'geometry row={row}/{T}', flush=True)

choices = []
work_center_world = np.einsum('tij,j->ti', broom_R, work.mean(0)) + broom_p
work_velocity = np.gradient(work_center_world, .05, axis=0)
center_inward = -np.einsum('ti,ti->t', work_velocity, pan_R[:, :, 2])
for ci in range(N):
    if gaps[0, ci] < .010 or gaps[:15, ci].min() < .005:
        continue
    # Nominal replay need not push the cube: a small reachable gap is permitted.
    hits = np.flatnonzero((brush_box_dist[:, ci] <= .015) & (center_inward > .005) &
                         (np.arange(T) >= 20) & (np.arange(T) <= T-80))
    if not len(hits):
        continue
    row = int(hits[np.argmin(brush_box_dist[hits,ci])])
    if not 20 <= row <= T - 80:
        continue
    # Allow the final approach/contact interval, but reject earlier hull overlap.
    if gaps[:max(1, row - 3), ci].min() < .001:
        continue
    _, wi = cKDTree(work @ broom_R[row].T + broom_p[row]).query(candidates[ci], p=np.inf)
    a, b = max(0, row-2), min(T-1, row+2)
    vel = ((broom_R[b] @ work[wi] + broom_p[b]) -
           (broom_R[a] @ work[wi] + broom_p[a])) / ((b-a)*.05)
    inward = float(-vel @ pan_R[row, :, 2])
    if inward <= .001:
        continue
    # Prefer centrally located, inward contacts with useful time after contact.
    score = abs(local[ci, 0]) + .15 * (local[ci, 2]-mouth) + .00004*row - .1*inward + 2*brush_box_dist[row,ci]
    choices.append(dict(index=ci, row=row, score=score, inward_mps=inward, working_index=int(wi),
                        initial_hull_clearance_m=float(gaps[0, ci]),
                        precontact_clearance_m=float(gaps[:max(1,row-3), ci].min())))
choices.sort(key=lambda a: a['score'])
np.savez_compressed(RUN/'take9_cube_search.npz', candidates=candidates,
                    local=local, hull_separation=gaps, brush_box_dist=brush_box_dist)
(RUN/'take9_cube_candidates.json').write_text(json.dumps(choices[:30], indent=2)+'\n')
assert choices, 'No safe first-contact candidate; inspect search NPZ without changing the trajectory.'
best = choices[0]
ci, row = best['index'], best['row']
cube = candidates[ci]
assert np.allclose(cube[2], cube_height)
assert abs(local[ci,0]) + projected_half[0] < width
assert local[ci,2] - projected_half[2] > mouth
old_cube = z['cube_start_w'].copy()
z['cube_start_w'] = cube.astype(np.float32)
z['contact_row'] = np.int32(row)
z['brush_contact_local'] = work[best['working_index']].astype(np.float32)
z['nominal_brush_cube_distance_m'] = np.float32(np.linalg.norm(
    broom_R[row] @ z['brush_contact_local'] + broom_p[row] - cube))
z['acceptance'] = np.array('cube_v2_static_clearance_passed_physics_pending')
z['cube_status'] = np.array('training_geometry_calibrated')
ref = ROOT/'tasks/Sweep/new_data/references/take32_fixed_level_cube15_v1.npz'
conf = ROOT/'tasks/Sweep/new_data/configs/take32_fixed_level_cube15_v1.json'
assert not ref.exists() and not conf.exists(), 'Never overwrite an existing prepared version.'
np.savez(ref, **z)
cfg['reference'] = str(ref.relative_to(ROOT))
cfg['task_name'] = 'Task3Take32FixedLevelCube15'
cfg['training_geometry']['cube_half'] = half
# Base configuration already uses 15 mm geometry.
cfg['training_input']['cube_size_m'] = 2*half
cfg['training_input'].update(contact_row=row, contact_source_frame=float(z['source_frame'][row]),
    cube_placement_audit='logs/task1_take32_level_20260910/cube15/take9_cube_v2_report.json')
cfg['acceptance'] = 'cube_v2_static_clearance_passed_physics_pending'
conf.write_text(json.dumps(cfg, indent=2)+'\n')
report = dict(status='static_passed_physics_pending', source_config=str(src_cfg.relative_to(ROOT)),
    source_reference_sha256=hashlib.sha256(src_ref.read_bytes()).hexdigest(),
    config=str(conf.relative_to(ROOT)), reference=str(ref.relative_to(ROOT)),
    cube_start_world_m=cube.tolist(), cube_start_pan_m=local[ci].tolist(),
    projected_cube_half_in_pan_m=projected_half.tolist(), old_cube_start_world_m=old_cube.tolist(),
    selected=best, surviving_candidates=len(choices), total_candidates=N,
    inward_approach_seconds=row*.05, remaining_rows=T-row-1,
    nominal_brush_to_cube_box_gap_m=float(brush_box_dist[row,ci]),
    certification='full broom convex hull face/box separation; last 3 approach rows exempt',
    cube_side_m=2*half, physics_pending=True, tool_joint_tracks_unchanged=True)
(RUN/'take9_cube_v2_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report, indent=2), flush=True)
