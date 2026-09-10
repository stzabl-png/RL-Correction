"""Preserve take80 v3 handle/grasp/motion; replace only the brush head."""
from pathlib import Path
import json,sys,glob
import numpy as np,trimesh
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,glob.glob('/home/msc-auto/rlcorr-venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*')[0])
from pxr import Usd,UsdGeom,UsdPhysics,Sdf,Vt
OUT=ROOT/'tasks/Sweep/new_data/assets/take_80/broom_head_v6';OUT.mkdir(parents=True,exist_ok=True)
pieces=[];reports=[]
def clipped(source,outname,keep_above,cut,translation):
 st=Usd.Stage.Open(str(ROOT/source));st=Usd.Stage.Open(st.Flatten());root=st.GetDefaultPrim();xf=UsdGeom.XformCache()
 for p in st.Traverse():
  if p.HasAPI(UsdPhysics.RigidBodyAPI):
   p.RemoveAPI(UsdPhysics.RigidBodyAPI);p.RemoveAPI(UsdPhysics.MassAPI);p.RemoveAppliedSchema('PhysxRigidBodyAPI')
  if not p.IsA(UsdGeom.Mesh):continue
  m=UsdGeom.Mesh(p);pts=np.asarray(m.GetPointsAttr().Get(),dtype=np.float64)
  assert np.allclose(np.asarray(xf.GetLocalToWorldTransform(p)*xf.GetLocalToWorldTransform(root).GetInverse()),np.eye(4))
  counts=np.asarray(m.GetFaceVertexCountsAttr().Get());assert np.all(counts==3)
  f=np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
  mask=(pts[f,2].min(1)>=cut) if keep_above else (pts[f,2].max(1)<=cut)
  old_ids=np.flatnonzero(mask);f=f[mask];used=np.unique(f)
  remap=np.full(len(pts),-1,dtype=int);remap[used]=np.arange(len(used));f=remap[f];v=pts[used]
  welded=trimesh.Trimesh(v.copy(),f.copy(),process=True)
  edges=np.concatenate([welded.faces[:,[0,1]],welded.faces[:,[1,2]],welded.faces[:,[2,0]]])
  _,inv,count=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
  boundary=edges[count[inv]==1];successor={int(a):int(b) for a,b in boundary};assert len(successor)==len(boundary)
  loops=[]
  while successor:
   a=next(iter(successor));loop=[a];b=successor.pop(a)
   while b!=a:loop.append(b);b=successor.pop(b)
   loops.append(loop)
  assert 1<=len(loops)<=4,len(loops)
  to_original=cKDTree(v).query(welded.vertices)[1]
  centers=[];cap=[]
  for loop in loops:
   centers.append(welded.vertices[loop].mean(0));ci=len(v)+len(centers)-1
   for a,b in zip(loop,loop[1:]+loop[:1]):cap.append([int(to_original[b]),int(to_original[a]),ci])
  centers=np.array(centers);cap=np.array(cap)
  newv=np.concatenate([v,centers])+translation;newf=np.concatenate([f,cap])
  closed=trimesh.Trimesh(newv.copy(),newf.copy(),process=True)
  assert closed.is_watertight and closed.is_winding_consistent and closed.volume>0
  for pv in UsdGeom.PrimvarsAPI(p).GetPrimvars():
   values=pv.Get()
   if values is None:continue
   assert not pv.IsIndexed()
   interp=pv.GetInterpolation();arr=np.asarray(values)
   if interp=='vertex':
    base=arr[used];extra=np.repeat(base.mean(0,keepdims=True),len(centers),axis=0);arr=np.concatenate([base,extra])
   elif interp=='faceVarying':
    base=arr.reshape((len(counts),3)+arr.shape[1:])[mask].reshape((-1,)+arr.shape[1:])
    extra=np.repeat(base.mean(0,keepdims=True),3*len(cap),axis=0);arr=np.concatenate([base,extra])
   elif interp=='uniform':
    base=arr[mask];arr=np.concatenate([base,np.repeat(base.mean(0,keepdims=True),len(cap),axis=0)])
   else:continue
   pv.Set(type(values).FromNumpy(arr))
  m.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(newv.astype(np.float32)))
  m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(newf),3,np.int32)))
  m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(newf.astype(np.int32).ravel()))
  m.GetNormalsAttr().Clear()
  m.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.stack([newv.min(0),newv.max(0)]).astype(np.float32)))
  UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
  UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr('convexDecomposition')
  pieces.append(closed)
  reports.append(dict(source=source,part=outname,cut_z=cut,keep_above=keep_above,translation=translation.tolist(),cap_loops=len(loops),kept_faces=len(f),added_cap_faces=len(cap),watertight=True,source_vertex_delta_max_m=float(abs((newv[:len(v)]-translation)-pts[used]).max())))
 target=OUT/outname;st.GetRootLayer().Export(str(target));return target
handle=clipped('tasks/Sweep/new_data/assets/take_80/broom_v2/broom.usd','handle.usd',False,.012,np.zeros(3))
head=clipped('tasks/Sweep/new_data/assets/take_32/broom_v2/broom.usd','head.usd',True,-.025,np.array([0,-.008,.028]))
stage=Usd.Stage.CreateNew(str(OUT/'broom.usd'));root=UsdGeom.Xform.Define(stage,'/Broom').GetPrim();stage.SetDefaultPrim(root);UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
for name,path in [('Handle',handle),('Head',head)]:
 child=UsdGeom.Xform.Define(stage,'/Broom/'+name).GetPrim();child.GetReferences().AddReference(str(path))
UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True);UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.1)
root.AddAppliedSchema('PhysxRigidBodyAPI');root.CreateAttribute('physxRigidBody:solverPositionIterationCount',Sdf.ValueTypeNames.Int).Set(8);root.CreateAttribute('physxRigidBody:solverVelocityIterationCount',Sdf.ValueTypeNames.Int).Set(0)
stage.GetRootLayer().Save()
combined=trimesh.util.concatenate(pieces);combined.export(OUT/'object_mesh_scaled_final.obj')
cfg=json.loads((ROOT/'tasks/Sweep/new_data/configs/take_80_v3.json').read_text())
cfg['assets']['broom_mesh']=str((OUT/'object_mesh_scaled_final.obj').relative_to(ROOT));cfg['assets']['broom_usd']=str((OUT/'broom.usd').relative_to(ROOT))
cfg['acceptance']='head_only_repair_pending_physical_visual_review'
cfg['head_replacement']=dict(donor='take32 accepted head',handle='take80 v3 unchanged below z=12mm',bristles_input=[0,-1,0],prior_unchanged=True,reference_unchanged=True)
(ROOT/'tasks/Sweep/new_data/configs/take_80_v6.json').write_text(json.dumps(cfg,indent=2)+'\n')
g=np.load(ROOT/cfg['grasppose']['broom']['prior'])['contact_pos']
assert g[:,2].max()<.012
(OUT/'graft_audit.json').write_text(json.dumps(dict(parts=reports,contact_max_z_m=float(g[:,2].max()),all_contact_region_preserved=True,head_and_handle_closed_components=True,reference=cfg['reference'],prior=cfg['grasppose']['broom']['prior']),indent=2)+'\n')
print(json.dumps(reports,indent=2))
