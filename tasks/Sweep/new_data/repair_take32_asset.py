"""Take32 mesh-body repair; preserve handle vertices, winding and connectivity."""
from pathlib import Path
import json,hashlib,sys,glob,argparse
p=argparse.ArgumentParser();p.add_argument("--neck_ellipse",action="store_true");args=p.parse_args()
import numpy as np,trimesh
ROOT=Path(__file__).resolve().parents[3]
source=ROOT/'datasets/sweep_new_data/sweep_dustpan/32/objects/object_0/object_mesh_scaled_final.obj'
out=ROOT/('tasks/Sweep/new_data/assets/take_32/dustpan_v3' if args.neck_ellipse else 'tasks/Sweep/new_data/assets/take_32/dustpan_v2')
out.mkdir(parents=True,exist_ok=True)
m=trimesh.load(source,force='mesh',process=False)
v0=np.array(m.vertices);faces=np.array(m.faces)
v=v0.copy()
# Source body is upside down relative to ego. Reorient body about the longitudinal
# axis, smoothly through the narrow neck; grasped handle (z<=-30mm) is exact.
a=np.clip((v[:,2]+.030)/.025,0,1);a=a*a*(3-2*a)
theta=np.pi*a
x,y=v[:,0].copy(),v[:,1].copy()
v[:,0]=np.cos(theta)*x-np.sin(theta)*y
v[:,1]=np.sin(theta)*x+np.cos(theta)*y
if args.neck_ellipse:
    # Rotate material coordinates within each elliptical neck section, keeping its
    # narrow thickness axis aligned. Interpolate the section centre directly:
    # avoids sweeping the offset neck centre around the global longitudinal axis.
    zs=np.arange(-.031,-.0039,.001)
    centers=[];radii=[]
    for z in zs:
        q=v0[abs(v0[:,2]-z)<.00075,:2]
        low=q.min(0);high=q.max(0)
        centers.append((low+high)/2);radii.append((high-low)/2)
    centers=np.asarray(centers);radii=np.asarray(radii)
    ids=(v0[:,2]>-.03)&(v0[:,2]<-.005)
    z=v0[ids,2]
    center=np.stack([np.interp(z,zs,centers[:,i]) for i in range(2)],1)
    radius=np.stack([np.interp(z,zs,radii[:,i]) for i in range(2)],1)
    local=(v0[ids,:2]-center)/radius
    cs=np.cos(theta[ids]);sn=np.sin(theta[ids])
    rotated=np.stack([cs*local[:,0]-sn*local[:,1],sn*local[:,0]+cs*local[:,1]],1)
    v[ids,:2]=rotated*radius+(1-2*a[ids,None])*center

# Correct body up is -input Y. Repair the whole lip thickness, not only its top:
# map source cross-section to a 3.5--6.5mm supported wedge, preserving orientation.
sem=v*np.array([-1,-1,1])
centres=np.arange(.070,.110,.002)
lo=[];hi=[]
for z in centres:
    q=sem[(abs(sem[:,0])<.070)&(abs(sem[:,2]-z)<.0015),1]
    if len(q):lo.append(q.min());hi.append(q.max())
    else:lo.append(lo[-1]);hi.append(hi[-1])
low=np.interp(sem[:,2],centres,lo);high=np.interp(sem[:,2],centres,hi)
target_top=-.008-.003*np.clip((sem[:,2]-.078)/.029,0,1)
target_bottom=np.full(len(v),-.0145)
mapped=target_bottom+(sem[:,1]-low)/np.maximum(high-low,1e-6)*(target_top-target_bottom)
wz=np.clip((sem[:,2]-.070)/.010,0,1);wz=wz*wz*(3-2*wz)
wx=np.clip((.098-abs(sem[:,0]))/.028,0,1);wx=wx*wx*(3-2*wx)
weight=wz*wx
sem[:,1]+=weight*(mapped-sem[:,1])
v=sem*np.array([-1,-1,1])
m.vertices=v
assert np.array_equal(m.faces,faces)
assert np.array_equal(v[v0[:,2]<=-.03],v0[v0[:,2]<=-.03])
assert m.is_watertight and m.is_winding_consistent
assert np.min(m.area_faces)>1e-14
assert m.volume>0
target=out/'object_mesh_scaled_final.obj'
m.export(target,include_normals=True)
# Keep the existing input OBJ -> USD identity contract and author new geometry only.
sys.path.insert(0,glob.glob('/home/msc-auto/rlcorr-venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*')[0])
from pxr import Usd,UsdGeom,Vt,Gf
s=Usd.Stage.Open(str(ROOT/'datasets/sweep_new_data/sweep_dustpan/32/cache/task3_dustpan.usd'))
s=Usd.Stage.Open(s.Flatten())
meshes=[p for p in s.Traverse() if p.IsA(UsdGeom.Mesh)]
assert len(meshes)==1
mesh=UsdGeom.Mesh(meshes[0])
assert np.max(np.abs(np.asarray(mesh.GetPointsAttr().Get())-v0))<1e-6
mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
mesh.GetNormalsAttr().Clear()
mesh.CreateExtentAttr().Set([Gf.Vec3f(*v.min(0)),Gf.Vec3f(*v.max(0))])
usd=out/'dustpan.usd';s.GetRootLayer().Export(str(usd))
report=dict(source=str(source.relative_to(ROOT)),output=str(target.relative_to(ROOT)),
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    output_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    vertices=len(v),faces=len(faces),face_indices_identical=True,handle_vertices_identical=True,
    watertight=m.is_watertight,winding_consistent=m.is_winding_consistent,
    source_volume=float(trimesh.load(source,force='mesh',process=False).volume),
    repaired_volume=float(m.volume),neck_twist_z_m=[-.03,-.005],neck_ellipse=bool(args.neck_ellipse),
    opening_direction_input=[0,-1,0],floor_top_semantic_y_m=-.008,
    lip_top_end_semantic_y_m=-.011,lip_bottom_semantic_y_m=-.0145,
    acceptance='diagnostic_asset_pending_physical_visual_review')
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
