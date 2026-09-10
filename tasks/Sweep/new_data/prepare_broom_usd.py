"""Bake a Task3-owned textured broom USD with source-identity coordinates."""
import argparse,json,sys,glob
from pathlib import Path
import numpy as np,trimesh
from scipy.spatial import cKDTree
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[3];cfg=json.loads((ROOT/a.config).read_text())
assert not cfg.get('asset_substitution'), 'Use substitute_broom_asset.py for donor-registered USD'
sys.path.insert(0,glob.glob('/home/msc-auto/rlcorr-venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*')[0])
from pxr import Usd,UsdGeom,UsdPhysics,Sdf,Vt
source=ROOT/cfg['source_textured_usd']['broom'] if 'source_textured_usd' in cfg else ROOT/cfg['data_dir']/'retarget/object_1_textured.usd'
stage=Usd.Stage.Open(source.as_posix());stage=Usd.Stage.Open(stage.Flatten())
root=stage.GetDefaultPrim();xf=UsdGeom.XformCache()
v=np.asarray(trimesh.load(ROOT/cfg['assets']['broom_mesh'],process=False).vertices)
report=[]
for prim in stage.Traverse():
 if prim.IsA(UsdGeom.Mesh):
  pts=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
  transform=np.array(xf.GetLocalToWorldTransform(prim)*xf.GetLocalToWorldTransform(root).GetInverse())
  assert np.allclose(transform,np.eye(4),atol=1e-7)
  if cfg.get('broom_head_repair'):
   original=np.asarray(trimesh.load(ROOT/cfg['data_dir']/'objects/object_1/object_mesh_scaled_final.obj',process=False).vertices)
   ds,idx=cKDTree(original).query(pts);assert ds.max()<1e-6
   pts=v[idx];UsdGeom.Mesh(prim).GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
   UsdGeom.Mesh(prim).GetNormalsAttr().Clear()
  distance=cKDTree(v).query(pts)[0];assert distance.max()<1e-6
  UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
  UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
  prim.AddAppliedSchema('PhysxCollisionAPI');prim.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(.002);prim.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(0)
  report.append(dict(mesh=str(prim.GetPath()),input_error_m=float(distance.max()),mesh_to_root=transform.tolist()))
UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
UsdPhysics.MassAPI.Apply(root).CreateMassAttr(.1)
root.AddAppliedSchema('PhysxRigidBodyAPI');root.CreateAttribute('physxRigidBody:solverPositionIterationCount',Sdf.ValueTypeNames.Int).Set(8);root.CreateAttribute('physxRigidBody:solverVelocityIterationCount',Sdf.ValueTypeNames.Int).Set(0)
out=ROOT/cfg['assets']['broom_usd'];out.parent.mkdir(parents=True,exist_ok=True);stage.GetRootLayer().Export(str(out))
out.with_suffix('.audit.json').write_text(json.dumps(report,indent=2)+'\n');print(out)
