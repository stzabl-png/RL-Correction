"""Create task-owned USD geometry from an OBJ that has no delivered USD."""
import argparse,json,sys,glob
from pathlib import Path
import numpy as np,trimesh
p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
ROOT=Path(__file__).resolve().parents[3];cfg=json.loads((ROOT/a.config).read_text())
sys.path.insert(0,glob.glob('/home/msc-auto/rlcorr-venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*')[0])
from pxr import Usd,UsdGeom,UsdPhysics,Sdf,Vt,Gf
for role in ['dustpan','broom']:
 path=ROOT/cfg['source_textured_usd'][role];path.parent.mkdir(parents=True,exist_ok=True)
 mesh=trimesh.load(ROOT/cfg['assets'][role+'_mesh'],force='mesh',process=False)
 assert mesh.visual.kind in (None,'vertex','face'), 'Textured OBJ requires material-preserving conversion'
 stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageUpAxis(stage,'Z');UsdGeom.SetStageMetersPerUnit(stage,1)
 root=UsdGeom.Xform.Define(stage,'/Object').GetPrim();stage.SetDefaultPrim(root)
 m=UsdGeom.Mesh.Define(stage,'/Object/mesh')
 m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices,np.float32)))
 m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces),3,np.int32)))
 m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces,np.int32).ravel()))
 m.CreateSubdivisionSchemeAttr('none');m.CreateDisplayColorAttr([Gf.Vec3f(.72,.74,.77)])
 m.CreateExtentAttr([Gf.Vec3f(*mesh.bounds[0]),Gf.Vec3f(*mesh.bounds[1])])
 stage.GetRootLayer().Save();print(path,flush=True)
