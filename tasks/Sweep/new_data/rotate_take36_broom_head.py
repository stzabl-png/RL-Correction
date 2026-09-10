"""Correct take36 bristle direction only; preserve handle, grasp and tool root track."""
from pathlib import Path
import json,glob,sys,hashlib
import numpy as np,trimesh
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[3]
src_cfg=ROOT/'tasks/Sweep/new_data/configs/take_36_v3.json'
cfg=json.loads(src_cfg.read_text())
m=trimesh.load(ROOT/cfg['assets']['broom_mesh'],force='mesh',process=False)
old=np.array(m.vertices);v=old.copy()
head=old[old[:,2]>-.01]
pivot=(head[:,:2].min(0)+head[:,:2].max(0))/2
a=np.clip((old[:,2]+.03)/.02,0,1)
theta=a*a*(3-2*a)*np.pi/2
xy=old[:,:2]-pivot
v[:,0]=np.cos(theta)*xy[:,0]-np.sin(theta)*xy[:,1]+pivot[0]
v[:,1]=np.sin(theta)*xy[:,0]+np.cos(theta)*xy[:,1]+pivot[1]
keep=old[:,2]<=-.03
v[keep]=old[keep]
assert np.array_equal(v[keep],old[keep])
prior=np.load(ROOT/cfg['grasppose']['broom']['prior'])
assert prior['contact_pos'][:,2].max()<-.03
m.vertices=v
assert m.is_watertight and m.is_winding_consistent and m.volume>0
assert m.area_faces.min()>1e-14
out=ROOT/'tasks/Sweep/new_data/assets/take_36/broom_head_v2'
out.mkdir(parents=True,exist_ok=True)
mesh_path=out/'object_mesh_scaled_final.obj'
m.export(mesh_path,include_normals=True)
sys.path.insert(0,glob.glob('/home/msc-auto/rlcorr-venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*')[0])
from pxr import Usd,UsdGeom,Vt,Gf
stage=Usd.Stage.Open(str(ROOT/cfg['assets']['broom_usd']))
stage=Usd.Stage.Open(stage.Flatten())
for prim in stage.Traverse():
 if prim.IsA(UsdGeom.Mesh):
  mesh=UsdGeom.Mesh(prim);pts=np.asarray(mesh.GetPointsAttr().Get())
  dist,ids=cKDTree(old).query(pts);assert dist.max()<1e-6
  mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v[ids].astype(np.float32)))
  mesh.GetNormalsAttr().Clear()
  mesh.CreateExtentAttr().Set([Gf.Vec3f(*v.min(0)),Gf.Vec3f(*v.max(0))])
usd_path=out/'broom.usd';stage.GetRootLayer().Export(str(usd_path))
report={'source_config':str(src_cfg.relative_to(ROOT)),'angle_deg':90,
 'axis_input':[0,0,1],'pivot_xy_m':pivot.tolist(),'protected_handle_z_max_m':-.03,
 'transition_z_m':[-.03,-.01],'handle_vertices_exact':True,
 'faces_unchanged':True,'watertight':True,'bristles_input_before':[0,-1,0],
 'bristles_input_after':[1,0,0],'prior_unchanged':True,'reference_unchanged':True,
 'reference_sha256':hashlib.sha256((ROOT/cfg['reference']).read_bytes()).hexdigest(),
 'status':'pending_physical_and_visual_review'}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
cfg['assets']['broom_mesh']=str(mesh_path.relative_to(ROOT));cfg['assets']['broom_usd']=str(usd_path.relative_to(ROOT))
cfg['head_orientation_correction']=report
cfg['acceptance']='head_orientation_corrected_pending_user_review'
(ROOT/'tasks/Sweep/new_data/configs/take_36_v4.json').write_text(json.dumps(cfg,indent=2)+'\n')
print(json.dumps(report,indent=2))
