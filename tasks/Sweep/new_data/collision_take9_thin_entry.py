from pathlib import Path
import json,numpy as np,trimesh
base=Path('tasks/Sweep/new_data/assets/take9_powerdisk/dustpan_thin_entry_20260909');m=trimesh.load(base/'object_mesh_scaled_final.obj',process=False)
parts=[]
# Small convex pieces follow real visible shell; no hull across the open basin.
for label,x0,x1 in [('floor',-.056,.056),('left',-.081,-.056),('right',.056,.081)]:
 for k,(z0,z1) in enumerate(zip(np.linspace(-.012,.109,19)[:-1],np.linspace(-.012,.109,19)[1:])):
  v,f=np.asarray(m.vertices),np.asarray(m.faces)
  for origin,normal in [([0,0,z0],[0,0,1]),([0,0,z1],[0,0,-1]),([x0,0,0],[1,0,0]),([x1,0,0],[-1,0,0])]:
   v,f,_=trimesh.intersections.slice_faces_plane(v,f,normal,origin)
  a=trimesh.Trimesh(vertices=v,faces=f,process=False)
  if len(a.vertices)<4:continue
  h=trimesh.convex.convex_hull(a.vertices)
  parts.append({'name':label+str(k),'vertices':h.vertices.tolist(),'faces':h.faces.tolist()})
(base/'collision.json').write_text(json.dumps(parts));print('parts',len(parts))
