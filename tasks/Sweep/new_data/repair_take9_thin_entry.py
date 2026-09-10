from pathlib import Path
import numpy as np,trimesh,json,hashlib,sys,glob
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
root=Path.cwd(); base=Path('tasks/Sweep/new_data/assets/take9_powerdisk/dustpan_thin_entry_20260909');base.mkdir(parents=True,exist_ok=True)
src=Path('tasks/Sweep/new_data/prepared/take9_powerdisk/objects/object_0/object_mesh_scaled_final.obj')
m=trimesh.load(src,force='mesh',process=False);v0=np.array(m.vertices);v=v0.copy()
# Canonical x=width, +y=up, +z=toward mouth. Map BOTH surfaces with positive
# thickness. Keep all vertices behind z=45mm, including handle, exactly fixed.
zs=np.linspace(.04,.109,277);lo=[];hi=[]
for z in zs:
 pts=v0[abs(v0[:,2]-z)<.001,1]
 lo.append(pts.min() if len(pts) else lo[-1]);hi.append(pts.max() if len(pts) else hi[-1])
lo=gaussian_filter1d(lo,2);hi=gaussian_filter1d(hi,2)
z=v0[:,2]; low=np.interp(z,zs,lo); high=np.interp(z,zs,hi)
s=lambda t: np.clip(t,0,1)**2*(3-2*np.clip(t,0,1))
w=s((z-.045)/.018)
# Lower surface stays on the existing underside plane. Side walls gradually
# descend toward a 0.5 mm full-thickness mouth, not a zero-area collapsed plane.
thickness=.0285*(1-s((z-.055)/.048))+.0005
mapped=-.01435+(v0[:,1]-low)/(high-low)*thickness
v[:,1]=(1-w)*v0[:,1]+w*mapped
m.vertices=v
assert np.array_equal(v[z<=.045],v0[z<=.045])
assert m.is_watertight and m.is_winding_consistent and m.volume>0
assert m.area_faces.min()>1e-15
m.export(base/'object_mesh_scaled_final.obj',include_normals=True)
# Authored OBJ and USD share canonical coordinates. Verify mapping before write.
sys.path.insert(0,glob.glob('/home/msc-auto/rlcorr-venv/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-*')[0])
from pxr import Usd,UsdGeom,Vt,Gf
st=Usd.Stage.Open('tasks/Sweep/new_data/assets/take9_powerdisk/dustpan.usd');st=Usd.Stage.Open(st.Flatten())
ms=[UsdGeom.Mesh(p) for p in st.Traverse() if p.IsA(UsdGeom.Mesh)];assert len(ms)==1
mesh=ms[0];d,ix=cKDTree(v0).query(np.asarray(mesh.GetPointsAttr().Get()));assert d.max()<1e-6
mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(v[ix].astype(np.float32)));mesh.GetNormalsAttr().Clear();mesh.GetExtentAttr().Set([Gf.Vec3f(*v.min(0)),Gf.Vec3f(*v.max(0))]);st.GetRootLayer().Export(str(base/'dustpan.usd'))
report={'source':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'vertices':len(v),'faces':len(m.faces),'watertight':bool(m.is_watertight),'winding_consistent':bool(m.is_winding_consistent),'min_face_area_m2':float(m.area_faces.min()),'volume_m3':float(m.volume),'handle_and_rear_unchanged':True,'mouth_nominal_thickness_mm':.5,'status':'geometry_checked; runtime_collision_and_table_contact_pending'}
(base/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
